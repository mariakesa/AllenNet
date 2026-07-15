"""Baselines and encoder readouts for natural-scenes decoding, all under one CV.

Answers two questions that decide whether scaling the pretraining is worth it:

1. IS THE ENCODER EARNING ITS KEEP? Compare it against the baselines instructions.txt
   asks for - raw flattened logistic regression, PCA + logistic regression, and a plain
   evoked-response (mean LFP in the response window) readout. If PCA+LR matches the
   encoder, the encoder is decoration.

2. IS THE POOLED EMBEDDING THROWING THE SIGNAL AWAY? The encoder ends in
   AdaptiveAvgPool2d(1), which averages the (64, 38, 24) latent over the ENTIRE 300 ms
   window and the ENTIRE shank. But the scene-evoked response is a transient (~50-150 ms
   post-onset) concentrated at specific depths. Averaging over everything dilutes it with
   250 ms of ongoing activity. So we also read out from the latent directly, and from a
   latent pooled only over the response window.

Every readout is fit inside the fold: scaler on training trials, PCA on training trials,
classifier on training trials. Folds hold out whole IMAGES (see finetune_decode.py), so
nothing is scored on an exemplar it was trained on.

No fine-tuning anywhere here - the encoder is frozen. This isolates what the pretrained
representation *contains*, which is the question that matters for whether more pretraining
would help.

Example
-------
    python evaluate_readouts.py --config config.yaml --labels artifacts/fourclass_labels.csv \
        --binarize animals
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from finetune_decode import image_grouped_folds, load_external_labels
from model import Encoder, load_encoder_trunk


def encode_all(
    trials: np.ndarray, model_cfg: dict, state: dict | None, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run the frozen encoder over every trial.

    Returns (pooled embedding (n, 128), latent (n, 64, T', C')).
    """
    encoder = Encoder(model_cfg).to(device).eval()
    if state is not None:
        load_encoder_trunk(encoder, state)

    embeddings, latents = [], []
    with torch.no_grad():
        for i in range(0, len(trials), 256):
            batch = torch.from_numpy(trials[i : i + 256]).float().to(device)
            x = batch.unsqueeze(1)
            # Second channel is the mask the encoder was pretrained with; nothing is
            # hidden at decode time, so it is all zeros.
            x = torch.cat([x, torch.zeros_like(x)], dim=1)
            latent, emb = encoder(x)
            embeddings.append(emb.cpu().numpy())
            latents.append(latent.cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(latents)


def logistic_cv(
    features: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_pca: int | None = None,
    seed: int = 0,
) -> tuple[float, float]:
    """Logistic regression across folds. Scaler and PCA are fit on TRAINING trials only.

    Returns (mean AUC, mean accuracy).
    """
    aucs, accs = [], []
    for train_idx, test_idx in folds:
        x_tr = features[train_idx].reshape(len(train_idx), -1)
        x_te = features[test_idx].reshape(len(test_idx), -1)
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(x_tr)
        x_tr, x_te = scaler.transform(x_tr), scaler.transform(x_te)

        if n_pca is not None and n_pca < min(x_tr.shape):
            pca = PCA(n_components=n_pca, svd_solver="randomized", random_state=seed).fit(x_tr)
            x_tr, x_te = pca.transform(x_tr), pca.transform(x_te)

        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(x_tr, y_tr)
        prob = clf.predict_proba(x_te)[:, 1]
        aucs.append(roc_auc_score(y_te, prob))
        accs.append(float((clf.predict(x_te) == y_te).mean()))
    logistic_cv.last_fold_aucs = aucs  # kept so the caller can test small differences
    return float(np.mean(aucs)), float(np.mean(accs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ns-dir", default="/media/maria/notsudata/allennet_natural_scenes")
    parser.add_argument("--encoder", default="runs/pretrain/encoder_best.pt")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--binarize", default="animals")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pca-components", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    model_cfg = cfg["model"]

    with open(os.path.join(args.ns_dir, "ns_dataset.json")) as fh:
        manifest = json.load(fh)
    trials = np.load(os.path.join(args.ns_dir, "ns_trials_f16.npy"))
    meta = pd.read_csv(os.path.join(args.ns_dir, "ns_trials_meta.csv"))
    fs = float(manifest["sampling_rate"])
    pre = int(manifest["pre_samples"])

    frames = meta.frame.to_numpy()
    y, known, class_names = load_external_labels(args.labels, frames)
    positive = class_names.index(args.binarize)
    y = (y == positive).astype(np.int64)
    trials, meta, frames = trials[known], meta[known].reset_index(drop=True), frames[known]
    y = y[known]

    start_s = meta.start_sample.to_numpy() / fs
    end_s = start_s + int(manifest["window_samples"]) / fs
    folds = image_grouped_folds(frames, y, start_s, end_s, args.folds, args.seed, 0.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chance = float(np.bincount(y).max() / len(y))

    print("=" * 84)
    print(f"READOUT COMPARISON - session {manifest['session_id']} probe {manifest['probe_id']}")
    print(f"task: {args.binarize} vs rest | {len(y)} trials | "
          f"image-grouped CV ({args.folds} folds, held-out images unseen)")
    print(f"majority-class baseline: {chance:.4f} accuracy, 0.5 AUC")
    print("=" * 84)

    # Response window: the scene-evoked transient, ~40-160 ms after stimulus onset.
    r0 = pre + int(0.040 * fs)
    r1 = pre + int(0.160 * fs)
    print(f"\nresponse window: samples {r0}..{r1} "
          f"({(r0 - pre) / fs * 1000:.0f}-{(r1 - pre) / fs * 1000:.0f} ms post-onset)\n")

    results: dict[str, dict] = {}

    def report(name: str, auc: float, acc: float, dims: int, t0: float) -> None:
        results[name] = {"auc": auc, "accuracy": acc, "dims": dims,
                         "fold_aucs": list(logistic_cv.last_fold_aucs)}
        print(f"  {name:44s} {auc:.4f}   {acc:.4f}   {dims:>7d}  ({time.time() - t0:.0f}s)",
              flush=True)

    print(f"  {'readout':44s} {'AUC':>6s}   {'acc':>6s}   {'dims':>7s}")
    print("  " + "-" * 74)

    # ---- BASELINES: no encoder at all -------------------------------------
    x = trials.astype(np.float32)

    t0 = time.time()
    erp = x[:, r0:r1, :].mean(axis=1)                       # (n, 93) evoked response
    report("BASELINE evoked response (mean in window)", *logistic_cv(erp, y, folds), 93, t0)

    t0 = time.time()
    flat = x.reshape(len(x), -1)                            # (n, 376*93)
    report("BASELINE PCA + logistic regression",
           *logistic_cv(flat, y, folds, n_pca=args.pca_components), args.pca_components, t0)

    t0 = time.time()
    report("BASELINE raw flattened logistic regression",
           *logistic_cv(flat, y, folds), flat.shape[1], t0)
    del flat

    # ---- ENCODER READOUTS: frozen, pretrained vs random --------------------
    for tag, state in [
        ("pretrained", torch.load(args.encoder, map_location="cpu", weights_only=True)),
        ("RANDOM (control)", None),
    ]:
        print()
        emb, latent = encode_all(x, model_cfg, state, device)
        n_lat_t = latent.shape[2]

        t0 = time.time()
        report(f"{tag}: pooled embedding (what we used)",
               *logistic_cv(emb, y, folds), emb.shape[1], t0)

        # Map the response window onto the latent's time axis. The stem strides 5 and
        # the downsample strides 2, so one latent step is ~10 input samples.
        step = trials.shape[1] / n_lat_t
        l0, l1 = int(r0 / step), max(int(r1 / step) + 1, int(r0 / step) + 1)
        t0 = time.time()
        win = latent[:, :, l0:l1, :].mean(axis=2)           # (n, 64, C') pool time only
        report(f"{tag}: latent pooled over response window",
               *logistic_cv(win, y, folds), win[0].size, t0)

        t0 = time.time()
        report(f"{tag}: full latent + PCA",
               *logistic_cv(latent.reshape(len(latent), -1), y, folds,
                            n_pca=args.pca_components), args.pca_components, t0)
        del emb, latent

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"session": manifest["session_id"], "chance": chance,
                       "results": results}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
