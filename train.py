"""Masked self-supervised pretraining of the LFP encoder.

Trains MaskedLFPAutoencoder to reconstruct hidden regions of LFP windows, scoring
only the masked entries. The decoder is thrown away afterwards; `encoder_best.pt` is
the artifact the downstream natural-scenes decoder consumes.

Everything needed for an offline two-week trip: resume, early stopping, CPU fallback,
reproducible seeds, and a history file that survives the process.

Example
-------
    python train.py --config config.yaml
    python train.py --config config.yaml --epochs 1 --limit-batches 20   # smoke test
    python train.py --config config.yaml --resume                        # continue
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import LFPWindowDataset, worker_init
from model import build_model, count_parameters, masked_huber_loss, masked_mae


@dataclass
class EpochStats:
    """One epoch's worth of numbers, mirrored into history.json."""

    epoch: int
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_mae: float = 0.0
    val_huber: float = 0.0
    lr: float = 0.0
    seconds: float = 0.0
    masked_fraction: float = 0.0


def set_seed(seed: int) -> None:
    """Seed every RNG the run touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str | None = None) -> torch.device:
    """CUDA when available, CPU otherwise. Never fail for lack of a GPU."""
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lr_at(epoch: int, cfg: dict) -> float:
    """Linear warmup then cosine decay, evaluated per epoch."""
    base = float(cfg["lr"])
    warmup = int(cfg["warmup_epochs"])
    total = int(cfg["epochs"])

    if epoch < warmup:
        return base * (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    limit_batches: int | None = None,
) -> dict[str, float]:
    """One pass. Trains when `optimizer` is given, otherwise evaluates.

    Returns mean masked loss / MAE / Huber, weighted by the number of masked
    entries per batch (not by batch count) - masks vary in size, so a plain mean
    over batches would silently overweight the lightly-masked ones.
    """
    train = optimizer is not None
    model.train(train)

    delta = float(cfg["model"]["huber_delta"])
    clip = float(cfg["training"]["grad_clip"])
    use_amp = bool(cfg["training"]["amp"]) and device.type == "cuda"

    total_loss = 0.0
    total_mae = 0.0
    total_masked = 0.0
    total_cells = 0.0
    n_batches = 0

    for i, (x, target, mask) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break

        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                recon, _, _ = model(x)
                loss = masked_huber_loss(recon, target, mask, delta=delta)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    # Unscale before clipping, or the clip threshold is applied to
                    # gradients still inflated by the loss scale.
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()

        with torch.no_grad():
            n_masked = float(mask.sum().item())
            total_loss += float(loss.item()) * n_masked
            total_mae += float(masked_mae(recon.float(), target, mask).item()) * n_masked
            total_masked += n_masked
            total_cells += float(mask.numel())
        n_batches += 1

    if total_masked == 0:
        raise RuntimeError("no masked entries seen; check the masking config")

    return {
        "loss": total_loss / total_masked,
        "mae": total_mae / total_masked,
        "masked_fraction": total_masked / max(total_cells, 1.0),
        "batches": float(n_batches),
    }


def save_checkpoint(path: str, model, optimizer, scaler, epoch: int, best: float, cfg: dict) -> None:
    """Full state for resume."""
    torch.save(
        {
            "epoch": epoch,
            "best_val": best,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": cfg,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-dir", default=None, help="override training.run_dir")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--limit-batches", type=int, default=None, help="smoke test")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    tcfg = cfg["training"]
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    if args.batch_size is not None:
        tcfg["batch_size"] = args.batch_size
    if args.lr is not None:
        tcfg["lr"] = args.lr
    if args.run_dir is not None:
        tcfg["run_dir"] = args.run_dir

    run_dir = tcfg["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    set_seed(int(tcfg["seed"]))

    device = pick_device(args.device)
    use_amp = bool(tcfg["amp"]) and device.type == "cuda"

    print("=" * 78)
    print("masked LFP pretraining")
    print("=" * 78)
    print(f"device        : {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")
    print(f"mixed precision: {use_amp}")

    train_ds = LFPWindowDataset(cfg, split="train")
    val_ds = LFPWindowDataset(cfg, split="val")
    print(f"  {train_ds.describe()}")
    print(f"  {val_ds.describe()}")

    # Sessions must not straddle the split, or the val loss measures memorisation.
    overlap = set(train_ds.meta.session_id) & set(val_ds.meta.session_id)
    if overlap:
        raise RuntimeError(f"train/val share sessions {sorted(overlap)} - split is leaking")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=pin,
        drop_last=True,
        worker_init_fn=worker_init,
        persistent_workers=int(tcfg["num_workers"]) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=pin,
        worker_init_fn=worker_init,
        persistent_workers=int(tcfg["num_workers"]) > 0,
    )

    model = build_model(cfg["model"]).to(device)
    n_enc = count_parameters(model.encoder)
    n_dec = count_parameters(model.decoder)
    print(f"\nparameters    : {count_parameters(model):,} total "
          f"({n_enc:,} encoder + {n_dec:,} decoder)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_val = float("inf")
    history: list[dict] = []

    last_path = os.path.join(run_dir, "last.pt")
    history_path = os.path.join(run_dir, "history.json")

    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt["best_val"])
        if os.path.exists(history_path):
            with open(history_path) as fh:
                history = json.load(fh)
        print(f"resumed from  : {last_path} (epoch {start_epoch}, best val {best_val:.5f})")
    elif args.resume:
        print(f"resume requested but {last_path} does not exist; starting fresh")

    # The exact config this run used, next to its checkpoints.
    with open(os.path.join(run_dir, "config.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    epochs = int(tcfg["epochs"])
    patience = int(tcfg["patience"])
    since_best = 0

    print(f"\ntraining for up to {epochs} epochs "
          f"({len(train_loader)} batches/epoch, batch {tcfg['batch_size']})\n")

    for epoch in range(start_epoch, epochs):
        lr = lr_at(epoch, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        t0 = time.time()
        tr = run_epoch(model, train_loader, device, cfg, optimizer, scaler, args.limit_batches)
        va = run_epoch(model, val_loader, device, cfg, limit_batches=args.limit_batches)
        elapsed = time.time() - t0

        stats = EpochStats(
            epoch=epoch,
            train_loss=tr["loss"],
            val_loss=va["loss"],
            val_mae=va["mae"],
            val_huber=va["loss"],  # the optimised loss IS the masked Huber
            lr=lr,
            seconds=elapsed,
            masked_fraction=tr["masked_fraction"],
        )
        history.append(stats.__dict__)

        improved = va["loss"] < best_val - 1e-6
        flag = ""
        if improved:
            best_val = va["loss"]
            since_best = 0
            torch.save(model.encoder.state_dict(), os.path.join(run_dir, "encoder_best.pt"))
            save_checkpoint(
                os.path.join(run_dir, "autoencoder_best.pt"),
                model, optimizer, scaler, epoch, best_val, cfg,
            )
            flag = "  *best"
        else:
            since_best += 1

        print(
            f"epoch {epoch:3d}  train {tr['loss']:.5f}  val {va['loss']:.5f}  "
            f"val_mae {va['mae']:.5f}  masked {tr['masked_fraction']:.2f}  "
            f"lr {lr:.2e}  {elapsed:5.1f}s{flag}",
            flush=True,
        )

        save_checkpoint(last_path, model, optimizer, scaler, epoch, best_val, cfg)
        with open(history_path, "w") as fh:
            json.dump(history, fh, indent=2)

        if since_best >= patience:
            print(f"\nearly stopping: no val improvement in {patience} epochs")
            break

    print("\n" + "=" * 78)
    print(f"best val masked-Huber : {best_val:.5f}")
    print(f"encoder (for downstream): {os.path.join(run_dir, 'encoder_best.pt')}")
    print(f"full autoencoder        : {os.path.join(run_dir, 'autoencoder_best.pt')}")
    print(f"history                 : {history_path}")


if __name__ == "__main__":
    main()
