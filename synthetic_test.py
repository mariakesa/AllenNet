"""Minimal end-to-end test on synthetic data. No Allen data, no GPU, ~30 seconds.

Run this before shipping anything to the training machine. It builds a fake memmap
in the real layout and checks the five things that, if wrong, would waste a training
run without ever throwing an error:

  1. masks are contiguous and land in the configured 40-60% band
  2. the loss reads masked entries ONLY (perturb the visible ones: loss must not move)
  3. the model actually learns - masked loss beats the predict-the-mean baseline
  4. the encoder handles T=1250 AND T=376 with a fixed 128-D embedding
  5. checkpoints round-trip: save -> load -> identical loss

Test 2 is the one that matters most. A loss that accidentally includes visible
entries still goes down, still produces pretty reconstructions, and still yields a
useless encoder - because the network can satisfy it by copying its input. Nothing
downstream would reveal that; only this test would.

Example
-------
    python synthetic_test.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import numpy as np
import torch
import yaml

from dataset import LFPWindowDataset
from masking import masked_fraction, sample_mask
from model import build_input, build_model, count_parameters, masked_huber_loss

N_WINDOWS = 64
N_TIME = 1250
N_CHAN = 93
FS = 1250.0


def make_synthetic_dataset(root: str, seed: int = 0) -> None:
    """Write a memmap + metadata with the same contract as prepare_dataset.py.

    The signal is a travelling wave across depth plus band-limited noise, so it has
    genuine structure along BOTH axes. That is deliberate: a model can only beat the
    predict-the-mean baseline (test 3) by exploiting spatiotemporal correlation, so
    if the data had none, test 3 could never pass and would tell us nothing.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_TIME) / FS

    windows = np.empty((N_WINDOWS, N_TIME, N_CHAN), dtype=np.float16)
    for i in range(N_WINDOWS):
        freq = rng.uniform(4.0, 12.0)                  # theta-ish
        speed = rng.uniform(0.002, 0.010)              # s of lag per channel
        phase = rng.uniform(0, 2 * np.pi)
        lags = np.arange(N_CHAN) * speed
        wave = np.sin(2 * np.pi * freq * (t[:, None] - lags[None, :]) + phase)

        # Smooth noise: a random walk lowpassed by a moving average, so the noise is
        # correlated in time too (white noise would be unlearnable by construction).
        noise = rng.standard_normal((N_TIME, N_CHAN)).cumsum(axis=0)
        noise = noise - noise.mean(axis=0)
        noise /= noise.std(axis=0) + 1e-8

        sig = 1.2 * wave + 0.4 * noise
        windows[i] = sig.astype(np.float16)

    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "lfp_windows_f16.dat")
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=windows.shape)
    mm[:] = windows
    mm.flush()
    del mm

    n_train = N_WINDOWS // 2
    import pandas as pd

    meta = pd.DataFrame(
        {
            "index": np.arange(N_WINDOWS),
            "session_id": [1] * n_train + [2] * (N_WINDOWS - n_train),
            "probe_id": [11] * n_train + [22] * (N_WINDOWS - n_train),
            "mouse_id": ["m1"] * n_train + ["m2"] * (N_WINDOWS - n_train),
            "start_sample": np.arange(N_WINDOWS) * N_TIME,
            "split": ["train"] * n_train + ["val"] * (N_WINDOWS - n_train),
        }
    )
    meta.to_csv(os.path.join(root, "windows_meta.csv"), index=False)

    with open(os.path.join(root, "dataset.json"), "w") as fh:
        json.dump(
            {
                "memmap": {
                    "file": "lfp_windows_f16.dat",
                    "dtype": "float16",
                    "shape": list(windows.shape),
                    "bytes": windows.nbytes,
                },
                "sampling_rate": FS,
                "n_windows": N_WINDOWS,
            },
            fh,
            indent=2,
        )


def test_masks_are_contiguous_and_in_range(cfg: dict) -> None:
    print("1. masks: contiguous, 40-60% coverage")
    rng = np.random.default_rng(0)
    lo = float(cfg["masking"]["min_fraction"])
    hi = float(cfg["masking"]["max_fraction"])

    fractions = []
    for _ in range(20):
        mask = sample_mask((N_TIME, N_CHAN), rng, cfg["masking"])
        frac = masked_fraction(mask)
        fractions.append(frac)

        assert mask.dtype == bool, f"mask must be bool, got {mask.dtype}"
        assert mask.any() and not mask.all(), "mask must hide some but not all entries"

        # Contiguity: an isolated masked sample (no masked neighbour in time or
        # depth) would mean the mask is speckled, which is the failure mode we are
        # guarding against.
        padded = np.pad(mask, 1, constant_values=False)
        has_neighbour = (
            padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]
        )
        isolated = int((mask & ~has_neighbour).sum())
        assert isolated == 0, f"{isolated} isolated masked samples - mask is not contiguous"

    lo_seen, hi_seen = min(fractions), max(fractions)
    # Regions are capped to the remaining budget, so the fraction converges to the
    # target from below and must never exceed the configured ceiling. A small
    # undershoot is expected (regions overlap); an overshoot is a real bug.
    assert hi_seen <= hi + 0.01, (
        f"masked fraction reached {hi_seen:.2f}, above the configured ceiling {hi} - "
        f"regions are not being capped to the remaining budget"
    )
    assert lo_seen >= lo - 0.03, (
        f"masked fraction fell to {lo_seen:.2f}, below the configured floor {lo}"
    )
    print(f"   realised fraction {lo_seen:.2f}..{hi_seen:.2f}, zero isolated samples  OK")


def test_loss_ignores_visible_entries(cfg: dict) -> None:
    print("2. loss reads masked entries only")
    torch.manual_seed(0)
    model = build_model(cfg["model"])

    target = torch.randn(4, 1, 200, N_CHAN)
    mask = torch.zeros_like(target)
    mask[:, :, 50:120, 10:40] = 1.0

    x = build_input(target, mask)
    recon, _, _ = model(x)

    base = masked_huber_loss(recon, target, mask, delta=cfg["model"]["huber_delta"])

    # Corrupt every VISIBLE entry beyond recognition. A loss confined to the masked
    # region cannot notice.
    corrupted = target + (1.0 - mask) * 1000.0
    after = masked_huber_loss(recon, corrupted, mask, delta=cfg["model"]["huber_delta"])

    assert torch.allclose(base, after, atol=1e-6), (
        f"loss changed by {abs(float(after - base)):.6f} when only VISIBLE entries were "
        f"corrupted - the loss is leaking into unmasked entries, and the model can "
        f"solve the task by copying its input"
    )

    # And the converse: corrupting a masked entry MUST move the loss.
    corrupted_masked = target + mask * 10.0
    moved = masked_huber_loss(recon, corrupted_masked, mask, delta=cfg["model"]["huber_delta"])
    assert not torch.allclose(base, moved, atol=1e-4), "loss ignores masked entries too"
    print(f"   visible corrupted: loss {float(base):.6f} -> {float(after):.6f} (unchanged)  OK")


def test_model_learns(cfg: dict, root: str) -> float:
    print("3. training reduces masked loss below the predict-the-mean baseline")
    torch.manual_seed(0)
    np.random.seed(0)

    ds = LFPWindowDataset(cfg, split="train", dataset_dir=root, deterministic_masks=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    model = build_model(cfg["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    delta = float(cfg["model"]["huber_delta"])

    # Baseline: predict the per-window mean of the VISIBLE entries for every masked
    # entry. Any model worth training must beat this.
    baselines = []
    for x, target, mask in loader:
        visible_mean = (target * (1 - mask)).sum(dim=(1, 2, 3), keepdim=True) / (
            (1 - mask).sum(dim=(1, 2, 3), keepdim=True).clamp_min(1)
        )
        guess = visible_mean.expand_as(target)
        baselines.append(float(masked_huber_loss(guess, target, mask, delta=delta)))
    baseline = float(np.mean(baselines))

    losses = []
    for epoch in range(12):
        epoch_losses = []
        for x, target, mask in loader:
            recon, _, _ = model(x)
            loss = masked_huber_loss(recon, target, mask, delta=delta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(float(loss))
        losses.append(float(np.mean(epoch_losses)))

    first, final = losses[0], losses[-1]
    print(f"   baseline (predict visible mean) : {baseline:.4f}")
    print(f"   masked loss  epoch 0 -> 11      : {first:.4f} -> {final:.4f}")

    assert final < first, f"loss did not decrease ({first:.4f} -> {final:.4f})"
    assert final < baseline, (
        f"masked loss {final:.4f} did not beat the predict-the-mean baseline "
        f"{baseline:.4f} - the model is not learning the spatiotemporal structure"
    )
    print(f"   beats baseline by {100 * (1 - final / baseline):.0f}%  OK")
    return final


def test_variable_length(cfg: dict) -> None:
    print("4. encoder handles T=1250 (pretrain) and T=376 (downstream)")
    model = build_model(cfg["model"])
    emb_dim = int(cfg["model"]["embedding_dim"])

    for n_time in (1250, 376):
        x = torch.randn(3, 2, n_time, N_CHAN)
        recon, latent, emb = model(x)
        assert emb.shape == (3, emb_dim), f"T={n_time}: embedding {tuple(emb.shape)}"
        assert recon.shape == (3, 1, n_time, N_CHAN), (
            f"T={n_time}: reconstruction {tuple(recon.shape)} must match the input"
        )
        print(f"   T={n_time:4d} -> latent {tuple(latent.shape)}  embedding {tuple(emb.shape)}  OK")


def test_checkpoint_roundtrip(cfg: dict) -> None:
    print("5. checkpoint save -> load reproduces the loss")
    torch.manual_seed(0)
    model = build_model(cfg["model"])
    model.eval()

    target = torch.randn(2, 1, 400, N_CHAN)
    mask = torch.zeros_like(target)
    mask[:, :, 100:250, 20:60] = 1.0
    x = build_input(target, mask)

    with torch.no_grad():
        before = float(masked_huber_loss(model(x)[0], target, mask))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        torch.save(model.state_dict(), path)

        restored = build_model(cfg["model"])
        restored.load_state_dict(torch.load(path, weights_only=True))
        restored.eval()
        with torch.no_grad():
            after = float(masked_huber_loss(restored(x)[0], target, mask))

    assert abs(before - after) < 1e-6, f"loss changed across save/load: {before} -> {after}"

    # The encoder is saved separately for the downstream stage; it must load alone.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "encoder.pt")
        torch.save(model.encoder.state_dict(), path)
        fresh = build_model(cfg["model"])
        fresh.encoder.load_state_dict(torch.load(path, weights_only=True))

    print(f"   loss {before:.6f} == {after:.6f}; encoder loads standalone  OK")


def main() -> None:
    with open("config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    model = build_model(cfg["model"])
    total = count_parameters(model)
    print("=" * 78)
    print("synthetic end-to-end test")
    print("=" * 78)
    print(f"parameters: {total:,} "
          f"({count_parameters(model.encoder):,} encoder + "
          f"{count_parameters(model.decoder):,} decoder)\n")

    root = tempfile.mkdtemp(prefix="allennet_synth_")
    try:
        make_synthetic_dataset(root)
        test_masks_are_contiguous_and_in_range(cfg)
        test_loss_ignores_visible_entries(cfg)
        test_model_learns(cfg, root)
        test_variable_length(cfg)
        test_checkpoint_roundtrip(cfg)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\n" + "=" * 78)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
