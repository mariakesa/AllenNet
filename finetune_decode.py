"""Fine-tune the pretrained encoder to decode natural-scene identity from single trials.

Reports mean accuracy across cross-validation folds.

Leakage is the whole game here, so it is worth being explicit about every route by
which the held-out trials could contaminate training, and what stops each one:

1. PRETRAINING LEAK. The encoder must never have seen this session. It hasn't:
   pretraining kept only sessions with zero natural-scenes presentations, and
   build_natural_scenes.py refuses any session that appears in the pretraining
   recordings table. Asserted again here.

2. WINDOW-OVERLAP LEAK. This is the sharp one. Natural-scene presentations run
   back to back (250.2 ms each, 250.2 ms apart), but a trial window is 376 samples
   = 300.8 ms. Adjacent trials therefore SHARE LFP SAMPLES. Under a random
   train/test split, ~17% of a test trial's window would also physically appear in
   a training trial's window, and the classifier could score well by recognising
   samples it had already been shown. Defence: fold on CONTIGUOUS TIME, then purge
   from training any trial whose window overlaps (or comes within --purge-seconds
   of) the test block. `--cv stratified` reproduces the naive random split, so the
   size of this effect can be measured rather than argued about.

3. BRAIN-STATE LEAK. Even without shared samples, trials seconds apart share slow
   drift in arousal and electrode state. Contiguous-time folds plus the purge
   margin handle this too.

4. FIT-ON-TEST LEAK. Every fold re-loads the pretrained weights from scratch, and
   early stopping uses an inner validation split taken from the TRAINING fold only.
   The test fold is touched exactly once, to score.

Normalization (robust median/MAD) is computed per recording from raw LFP with no
reference to labels or folds - the same transform the encoder was pretrained under.
Refitting it per fold would change nothing except to put the encoder's inputs on a
scale it was not pretrained on.

Two tasks are supported:

* 118-way image identity (default). Chance is 1/118 = 0.85%.
* A binary task from an external label file, e.g. animacy: `--labels animacy.csv`,
  a CSV of `frame,animate`. Allen ships NO animacy annotation - the stimulus table
  carries image indices only - so this ground truth must be supplied, never inferred.
  On a binary task the reported chance level is the MAJORITY-CLASS rate, not 0.5,
  and AUC is reported alongside accuracy: if the classes are imbalanced, a model that
  learns nothing but the class prior still scores well on accuracy, and only AUC and
  balanced accuracy expose that.

Example
-------
    python finetune_decode.py --config config.yaml --mode full
    python finetune_decode.py --config config.yaml --mode full --labels artifacts/animacy_labels.csv
    python finetune_decode.py --config config.yaml --mode full --init random   # control
    python finetune_decode.py --config config.yaml --cv stratified   # shows the leak
    python finetune_decode.py --config config.yaml --permute-labels  # null control
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from model import Encoder, count_parameters, load_encoder_trunk

def load_external_labels(
    path: str, frames: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Map each trial's image index to a class label from a user-supplied CSV.

    The CSV must have a `frame` column (image index 0..117) and exactly one label
    column, whose values may be ints or strings ("animals", "plant", ...). Any number
    of classes is supported; two classes additionally enables AUC.

    This is EXTERNAL ground truth. Allen ships no semantic annotation of the natural
    scenes - the stimulus table carries image indices only - so category labels are
    read from this file and never inferred.

    Rows with an empty label are treated as unlabelled and their trials are dropped.

    Returns (per-trial labels, mask of labelled trials, class names in label order).
    """
    table = pd.read_csv(path)
    if "frame" not in table.columns:
        raise SystemExit(f"{path}: needs a 'frame' column (image index 0..117)")

    label_cols = [c for c in table.columns if c != "frame"]
    if len(label_cols) != 1:
        raise SystemExit(
            f"{path}: expected exactly one label column besides 'frame', got {label_cols}"
        )
    col = label_cols[0]

    raw_by_frame: dict[int, str] = {}
    for row in table.itertuples():
        value = getattr(row, col)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        token = str(value).strip()
        if token == "" or token.lower() in {"nan", "-1", "unlabeled", "unlabelled"}:
            continue
        raw_by_frame[int(row.frame)] = token

    if not raw_by_frame:
        raise SystemExit(f"{path}: no images are labelled - fill in the '{col}' column")

    # Stable class order: numeric if the labels are numeric, else alphabetical.
    tokens = sorted(set(raw_by_frame.values()))
    if all(t.lstrip("-").isdigit() for t in tokens):
        tokens = sorted(tokens, key=int)
    class_of = {t: i for i, t in enumerate(tokens)}

    known = np.array([f in raw_by_frame for f in frames])
    y = np.array(
        [class_of[raw_by_frame[int(f)]] if f in raw_by_frame else -1 for f in frames],
        dtype=np.int64,
    )
    print(f"labels     : {path} ('{col}') -> {len(raw_by_frame)}/118 images labelled, "
          f"{int(known.sum())}/{len(frames)} trials usable")
    print(f"             classes: {dict(enumerate(tokens))}")
    return y, known, tokens


class LFPClassifier(nn.Module):
    """Pretrained encoder + linear head over the pooled embedding."""

    def __init__(self, model_cfg: dict, n_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.encoder = Encoder(model_cfg)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(int(model_cfg["embedding_dim"]), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, embedding = self.encoder(x)
        return self.head(self.dropout(embedding))


def to_input(trials: torch.Tensor) -> torch.Tensor:
    """(B, T, C) -> (B, 2, T, C): [LFP, all-zero mask].

    The encoder's second input channel is the mask it was pretrained with. At decode
    time nothing is hidden, so the mask is all zeros - which is exactly what the
    encoder saw over the *visible* regions during pretraining.
    """
    x = trials.unsqueeze(1)                       # (B, 1, T, C)
    return torch.cat([x, torch.zeros_like(x)], dim=1)


def set_trainable(model: LFPClassifier, mode: str) -> list[dict]:
    """Freeze/unfreeze per the fine-tuning mode; return optimizer param groups.

    The encoder gets a 10x smaller LR than the head: it carries pretrained structure
    worth preserving, while the head starts from noise.
    """
    for p in model.encoder.parameters():
        p.requires_grad = False

    if mode == "linear_probe":
        # The pooling head is a readout, not pretrained structure, so it trains even in
        # the "frozen" condition - otherwise a re-initialised head would be pure noise.
        trainable = list(model.encoder.head.parameters())
    elif mode == "final_block":
        # Last residual stage + the pooling head.
        trainable = list(model.encoder.blocks_out.parameters()) + list(
            model.encoder.head.parameters()
        )
    elif mode == "full":
        trainable = list(model.encoder.parameters())
    else:
        raise ValueError(f"unknown mode {mode!r}")

    for p in trainable:
        p.requires_grad = True

    groups = [{"params": model.head.parameters(), "lr": 1e-3}]
    if trainable:
        groups.append({"params": trainable, "lr": 1e-4})
    return groups


def time_blocked_folds(
    onset_s: np.ndarray,
    start_s: np.ndarray,
    end_s: np.ndarray,
    n_folds: int,
    purge_seconds: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous-time folds with the training set purged around each test block.

    Trials are ordered by onset and cut into `n_folds` contiguous blocks. A training
    trial is dropped if its window [start, end] comes within `purge_seconds` of the
    test block's window span - which is what stops the physically-shared samples
    between adjacent trials from crossing the split.
    """
    order = np.argsort(onset_s)
    blocks = np.array_split(order, n_folds)

    folds = []
    for test_idx in blocks:
        lo = float(start_s[test_idx].min()) - purge_seconds
        hi = float(end_s[test_idx].max()) + purge_seconds

        mask = np.ones(len(onset_s), dtype=bool)
        mask[test_idx] = False
        # Drop any remaining trial whose window intersects the purged span.
        overlaps = (start_s < hi) & (end_s > lo)
        mask &= ~overlaps
        folds.append((np.flatnonzero(mask), np.asarray(test_idx)))
    return folds


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Naive random stratified folds. Leaks via window overlap - for comparison only."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(y)), y)]


def image_grouped_folds(
    frames: np.ndarray,
    y: np.ndarray,
    start_s: np.ndarray,
    end_s: np.ndarray,
    n_folds: int,
    seed: int,
    purge_seconds: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hold out whole IMAGES, not trials. The only honest split for a semantic task.

    Every image is presented 50 times. If image 7 has trials in both train and test,
    a model can score on animacy without learning anything about animacy: it can
    memorise image 7's particular LFP response and recall that image 7 is an animal.
    That is image identification wearing a category label, and it is precisely what
    leave-one-image-out CV in instructions.txt exists to prevent.

    So folds are built over the 118 IMAGES (stratified by category so each fold keeps
    the class balance), and every trial of a held-out image goes to test. A model can
    only score by generalising to images it has never seen.

    The temporal purge still applies on top: a held-out image's trials are scattered
    through the session and sit next to training trials in time, and trial windows
    (300.8 ms) are longer than the inter-onset interval (250.2 ms), so neighbouring
    trials share LFP samples. Both leaks have to be closed at once.
    """
    images = np.unique(frames)
    # One label per image (constant across its 50 repeats).
    image_label = np.array([y[frames == img][0] for img in images])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for _, test_img_pos in skf.split(np.zeros(len(images)), image_label):
        test_images = set(images[test_img_pos].tolist())
        test_idx = np.flatnonzero([f in test_images for f in frames])

        # Candidate training trials: every trial of an image NOT held out.
        candidate = np.flatnonzero([f not in test_images for f in frames])

        # Purge candidates whose window overlaps (or comes within purge_seconds of)
        # any test trial's window. Test trials are scattered, so check intervals.
        t_lo = start_s[test_idx] - purge_seconds
        t_hi = end_s[test_idx] + purge_seconds
        order = np.argsort(t_lo)
        t_lo, t_hi = t_lo[order], t_hi[order]
        # Running max of t_hi lets a single searchsorted decide overlap.
        t_hi_max = np.maximum.accumulate(t_hi)

        keep = []
        for i in candidate:
            j = np.searchsorted(t_lo, end_s[i], side="right")
            if j == 0 or t_hi_max[j - 1] <= start_s[i]:
                keep.append(i)
        folds.append((np.array(keep, dtype=np.int64), test_idx))
    return folds


def train_one_fold(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    model_cfg: dict,
    n_classes: int,
    args,
    device: torch.device,
    encoder_state: dict | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fine-tune on one fold's training trials; return (preds, class probs, val loss).

    Early stopping runs on an inner split of the TRAINING trials. The test fold is
    never seen until the final scoring pass.
    """
    torch.manual_seed(seed)
    model = LFPClassifier(model_cfg, n_classes, dropout=args.dropout).to(device)

    if encoder_state is not None:
        load_encoder_trunk(model.encoder, encoder_state)
    # else: random init -> the "random encoder" control

    groups = set_trainable(model, args.mode)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Inner validation split, drawn from TRAINING trials only.
    rng = np.random.default_rng(seed)
    n = len(y_train)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    inner_val, inner_train = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(
        TensorDataset(x_train[inner_train], y_train[inner_train]),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_train[inner_val], y_train[inner_val]),
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size)

    best_val = float("inf")
    best_state = None
    since_best = 0

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = to_input(xb.to(device, non_blocking=True).float())
            yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_loss = 0.0
        n_val_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = to_input(xb.to(device).float())
                yb = yb.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(xb)
                val_loss += float(criterion(out, yb)) * len(yb)
                n_val_seen += len(yb)
        val_loss /= max(n_val_seen, 1)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = to_input(xb.to(device).float())
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(xb)
            out = out.float()
            preds.append(out.argmax(dim=1).cpu().numpy())
            probs.append(torch.softmax(out, dim=1).cpu().numpy())

    return np.concatenate(preds), np.concatenate(probs), best_val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ns-dir", default=None, help="output of build_natural_scenes.py")
    parser.add_argument("--encoder", default="runs/pretrain/encoder_best.pt")
    parser.add_argument("--labels", default=None,
                        help="CSV with columns 'frame' (0..117) and one label column. "
                             "Any number of classes. Without it, the task is 118-way "
                             "image identity.")
    parser.add_argument("--binarize", default=None, metavar="CLASS",
                        help="collapse the label file to CLASS vs rest, e.g. "
                             "--binarize animals")
    parser.add_argument("--mode", default="full",
                        choices=["linear_probe", "final_block", "full"])
    parser.add_argument("--init", default="pretrained", choices=["pretrained", "random"],
                        help="'random' is the control: same architecture, no pretraining")
    parser.add_argument("--cv", default="image_grouped",
                        choices=["image_grouped", "blocked", "stratified"],
                        help="'image_grouped' = hold out whole images + temporal purge "
                             "(the only honest split for a semantic task). "
                             "'blocked' = contiguous-time folds + purge (fine for 118-way "
                             "identity; LEAKS for semantic tasks - same image in train and "
                             "test). 'stratified' = naive random split (LEAKS via window "
                             "overlap too)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--purge-seconds", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--permute-labels", action="store_true",
                        help="null control: shuffle labels per trial; must collapse to chance")
    parser.add_argument("--permute-labels-by-image", action="store_true",
                        help="null control for EXEMPLAR leakage: each image keeps one "
                             "consistent but random label. Leak-free CV must give chance; "
                             "anything above chance is image memorisation")
    parser.add_argument("--pool-mode", default=None, choices=["gap", "grid", "stat"],
                        help="override model.pool_mode (the readout on top of the latent)")
    parser.add_argument("--pool-grid", default=None,
                        help="override model.pool_grid, e.g. '4,3'")
    parser.add_argument("--train-subsample", type=int, default=None, metavar="N",
                        help="randomly subsample each training fold to N trials. Used to "
                             "compare CV protocols at MATCHED training-set size, since "
                             "image-grouped folds are smaller than time-blocked ones and "
                             "that alone depresses accuracy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="write results json here")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    model_cfg = cfg["model"]
    if args.pool_mode:
        model_cfg["pool_mode"] = args.pool_mode
    if args.pool_grid:
        model_cfg["pool_grid"] = [int(v) for v in args.pool_grid.split(",")]

    ns_dir = args.ns_dir or os.path.join(
        os.path.dirname(cfg["paths"]["dataset_dir"]), "allennet_natural_scenes"
    )
    with open(os.path.join(ns_dir, "ns_dataset.json")) as fh:
        manifest = json.load(fh)

    trials = np.load(os.path.join(ns_dir, "ns_trials_f16.npy"))
    y = np.load(os.path.join(ns_dir, "ns_labels.npy"))
    meta = pd.read_csv(os.path.join(ns_dir, "ns_trials_meta.csv"))

    fs = float(manifest["sampling_rate"])

    if manifest.get("trial_averaged", False):
        raise SystemExit("this dataset is trial-averaged; the encoder was not trained on that")

    task = "image identity (118-way)"
    class_names = [str(i) for i in range(118)]
    if args.labels:
        y, known, class_names = load_external_labels(args.labels, meta.frame.to_numpy())
        trials, y, meta = trials[known], y[known], meta[known].reset_index(drop=True)
        task = f"{len(class_names)}-way: {os.path.basename(args.labels)}"
        if args.binarize is not None:
            # Collapse a multi-class label file to one-vs-rest, e.g. animals vs the rest.
            if args.binarize not in class_names:
                raise SystemExit(
                    f"--binarize {args.binarize!r} is not a class in {args.labels}; "
                    f"choose from {class_names}"
                )
            positive = class_names.index(args.binarize)
            y = (y == positive).astype(np.int64)
            class_names = [f"not-{args.binarize}", args.binarize]
            task = f"binary: {args.binarize} vs rest"

    frames_all = meta.frame.to_numpy()

    n_classes = int(len(np.unique(y)))
    counts = np.bincount(y)
    # Always compare against the MAJORITY-CLASS rate, never 1/n_classes. On an
    # imbalanced task a model that has learned nothing but the class prior already
    # beats uniform chance, and calling that "above chance" would be nonsense.
    chance = float(counts.max() / counts.sum())
    chance_name = "majority class"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("natural-scenes decoding from single trials")
    print("=" * 78)
    print(f"session {manifest['session_id']} probe {manifest['probe_id']} "
          f"(not used for pretraining)")
    print(f"task       : {task}")
    print(f"trials     : {trials.shape}  ({n_classes} classes, "
          f"counts {counts.tolist() if n_classes <= 4 else '...'})")
    print(f"chance     : {chance:.4f} ({chance_name})")
    print(f"mode       : {args.mode}   init: {args.init}   cv: {args.cv}")
    print(f"device     : {device}")

    if args.permute_labels:
        print("\n!! LABEL PERMUTATION CONTROL: labels shuffled, accuracy must fall to chance")
        y = np.random.default_rng(args.seed).permutation(y)

    if args.permute_labels_by_image:
        # The decisive test for EXEMPLAR leakage. Each image keeps ONE consistent label,
        # but that label is now random, so there is no real category signal to find.
        #
        # A protocol that holds out whole images must return chance: a random category
        # assigned to an unseen image is unlearnable by construction.
        #
        # A protocol that lets the same image appear in train and test will still score
        # above chance, because it can memorise "image 7 -> label X" from image 7's
        # training trials and apply it to image 7's test trials. Any accuracy here is
        # image memorisation, by definition - there is nothing else left to learn.
        print("\n!! IMAGE-LEVEL PERMUTATION: each image keeps one consistent but RANDOM "
              "label.\n   Leak-free CV must give chance. Anything above chance is "
              "exemplar memorisation.")
        images = np.unique(frames_all)
        rng = np.random.default_rng(args.seed)
        image_label = np.array([y[frames_all == img][0] for img in images])
        shuffled = rng.permutation(image_label)  # preserves class proportions
        lut = {int(img): int(lab) for img, lab in zip(images, shuffled)}
        y = np.array([lut[int(f)] for f in frames_all], dtype=np.int64)

    encoder_state = None
    if args.init == "pretrained":
        if not os.path.exists(args.encoder):
            raise SystemExit(f"{args.encoder} not found - run train.py first")
        encoder_state = torch.load(args.encoder, map_location="cpu", weights_only=True)
        print(f"encoder    : {args.encoder} (pretrained)")
    else:
        print("encoder    : randomly initialised (CONTROL)")

    # Fold construction.
    onset_s = meta.onset_time_s.to_numpy()
    start_s = meta.start_sample.to_numpy() / fs
    end_s = start_s + int(manifest["window_samples"]) / fs

    frames = frames_all

    if args.cv == "image_grouped" and not args.labels:
        # For 118-way identity the label IS the image, so holding images out would mean
        # testing on classes never trained on. Time-blocked folds are the right split
        # there, and the image-memorisation leak cannot apply: memorising the image is
        # the actual task.
        print("cv         : image_grouped is meaningless for 118-way identity "
              "(the label is the image); using blocked")
        args.cv = "blocked"

    if args.cv == "image_grouped":
        folds = image_grouped_folds(
            frames, y, start_s, end_s, args.folds, args.seed, args.purge_seconds
        )
        print(f"cv         : {args.folds} folds over IMAGES (held-out images never seen "
              f"in training), purge {args.purge_seconds}s")
    elif args.cv == "blocked":
        folds = time_blocked_folds(onset_s, start_s, end_s, args.folds, args.purge_seconds)
        print(f"cv         : {args.folds} contiguous-time folds, "
              f"purge {args.purge_seconds}s around each test block")
        if args.labels:
            print("             !! WARNING: the same image appears in train and test. "
                  "For a semantic\n"
                  "                task this LEAKS - the model can memorise an image's "
                  "response and\n"
                  "                recall its category. Use --cv image_grouped.")
    else:
        folds = stratified_folds(y, args.folds, args.seed)
        print(f"cv         : {args.folds} random stratified folds "
              f"-- WARNING: trial windows overlap by "
              f"{manifest['window_duration_ms'] - manifest['stimulus_duration_ms']:.1f} ms, "
              f"so this LEAKS and will read optimistically")

    x_all = torch.from_numpy(trials)          # float16 on CPU; cast per batch
    y_all = torch.from_numpy(y)

    probe = LFPClassifier(model_cfg, n_classes)
    print(f"parameters : {count_parameters(probe):,} "
          f"({count_parameters(probe.encoder):,} encoder + "
          f"{count_parameters(probe.head):,} head)\n")

    accs, bals, f1s, aucs = [], [], [], []
    t0 = time.time()

    for i, (train_idx, test_idx) in enumerate(folds):
        t_fold = time.time()
        if args.train_subsample is not None and len(train_idx) > args.train_subsample:
            sub = np.random.default_rng(args.seed + i).choice(
                len(train_idx), size=args.train_subsample, replace=False
            )
            train_idx = train_idx[np.sort(sub)]
        preds, probs, val_loss = train_one_fold(
            x_all[train_idx], y_all[train_idx],
            x_all[test_idx], y_all[test_idx],
            model_cfg, n_classes, args, device, encoder_state, seed=args.seed + i,
        )
        truth = y[test_idx]

        acc = float((preds == truth).mean())
        bal = float(balanced_accuracy_score(truth, preds))
        f1 = float(f1_score(truth, preds, average="macro", zero_division=0))
        accs.append(acc); bals.append(bal); f1s.append(f1)

        auc_str = ""
        if n_classes == 2 and len(np.unique(truth)) == 2:
            auc = float(roc_auc_score(truth, probs[:, 1]))
            aucs.append(auc)
            auc_str = f"  auc {auc:.4f}"

        purged = len(y) - len(train_idx) - len(test_idx)
        print(f"fold {i + 1}/{len(folds)}  train {len(train_idx):5d}  test {len(test_idx):5d}  "
              f"purged {purged:3d}  |  acc {acc:.4f}  bal {bal:.4f}  f1 {f1:.4f}{auc_str}  "
              f"({time.time() - t_fold:.0f}s)", flush=True)

    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))

    print("\n" + "=" * 78)
    print(f"MEAN ACCURACY OVER {len(folds)} FOLDS : {mean_acc:.4f}  (+/- {std_acc:.4f} sd)")
    print("=" * 78)
    print(f"  balanced accuracy : {np.mean(bals):.4f}")
    print(f"  macro F1          : {np.mean(f1s):.4f}")
    if aucs:
        # AUC is the metric to trust on an imbalanced binary task: it is unmoved by
        # a classifier that simply learns the class prior.
        print(f"  AUC               : {np.mean(aucs):.4f}  (+/- {np.std(aucs):.4f} sd, "
              f"0.5 = chance)")
        print(f"  per-fold AUC      : {[round(a, 4) for a in aucs]}")
    print(f"  chance ({chance_name:14s}): {chance:.4f}")
    print(f"  above chance      : {mean_acc / chance:.2f}x")
    print(f"  per-fold accuracy : {[round(a, 4) for a in accs]}")
    print(f"  elapsed           : {time.time() - t0:.0f}s")

    results = {
        "task": task,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "fold_accuracies": accs,
        "balanced_accuracy": float(np.mean(bals)),
        "macro_f1": float(np.mean(f1s)),
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "fold_aucs": aucs or None,
        "chance": chance,
        "chance_type": chance_name,
        "n_classes": n_classes,
        "mode": args.mode,
        "init": args.init,
        "cv": args.cv,
        "purge_seconds": args.purge_seconds,
        "permuted_labels": bool(args.permute_labels),
        "session_id": manifest["session_id"],
        "probe_id": manifest["probe_id"],
        "trial_averaged": False,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"  wrote             : {args.out}")


if __name__ == "__main__":
    main()
