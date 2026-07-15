"""Dataset over the float16 LFP memmap produced by prepare_dataset.py.

The memmap directory is self-describing (dataset.json + windows_meta.csv), so this
loads with numpy alone - no AllenSDK, no HDF5, no network. That is what makes the
dataset portable to a training machine.

Windows are stored float16 and converted to float32 here, per instructions.txt:
float16 halves the disk and page-cache cost, but accumulating a loss in float16
loses precision, so the cast happens at load time.

Example
-------
    train = LFPWindowDataset(cfg, split="train")
    val   = LFPWindowDataset(cfg, split="val")
    x, target, mask = train[0]
    # x: (2, 1250, 93)  target: (1, 1250, 93)  mask: (1, 1250, 93)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from masking import sample_mask


class LFPWindowDataset(Dataset):
    """Masked-reconstruction samples drawn from the pretraining memmap.

    Args:
        cfg: full parsed config.yaml.
        split: "train" or "val". Splits are session-disjoint (assigned in
            prepare_dataset.py) - never split random windows from one recording,
            or the val loss measures memorisation of that recording's statistics.
        deterministic_masks: if True, window i always gets the same mask. Used for
            validation so the loss is comparable across epochs; early stopping on a
            metric whose mask is redrawn every epoch is chasing noise. Training
            leaves this False so the mask varies every sample and every epoch.
    """

    def __init__(
        self,
        cfg: dict,
        split: str,
        deterministic_masks: bool | None = None,
        dataset_dir: str | None = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.cfg = cfg
        self.split = split
        self.mask_cfg = cfg["masking"]
        self.deterministic = (split == "val") if deterministic_masks is None else deterministic_masks
        self.base_seed = int(cfg["training"]["val_mask_seed"])

        self.dir = dataset_dir or cfg["paths"]["dataset_dir"]
        manifest_path = os.path.join(self.dir, "dataset.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"{manifest_path!r} not found - run prepare_dataset.py first"
            )
        with open(manifest_path) as fh:
            self.manifest = json.load(fh)

        mm = self.manifest["memmap"]
        self.path = os.path.join(self.dir, mm["file"])
        self.shape = tuple(int(s) for s in mm["shape"])
        self.dtype = np.dtype(mm["dtype"])
        self.n_time, self.n_chan = self.shape[1], self.shape[2]

        # Defensive: a truncated or half-copied memmap is otherwise silently read
        # as garbage at the tail.
        expected = int(np.prod(self.shape)) * self.dtype.itemsize
        actual = os.path.getsize(self.path)
        if actual != expected:
            raise RuntimeError(
                f"{self.path}: expected {expected} bytes for shape {self.shape} "
                f"({self.dtype}), found {actual} - the file is truncated or the "
                f"manifest is stale"
            )

        meta = pd.read_csv(os.path.join(self.dir, "windows_meta.csv"))
        if len(meta) != self.shape[0]:
            raise RuntimeError(
                f"windows_meta.csv has {len(meta)} rows but the memmap has "
                f"{self.shape[0]} windows"
            )
        self.meta = meta[meta.split == split].reset_index(drop=True)
        if self.meta.empty:
            raise RuntimeError(f"no windows in split {split!r}")
        self.indices = self.meta["index"].to_numpy(dtype=np.int64)

        # Opened lazily, per worker. A memmap handle created in the parent and
        # inherited across a fork is a well-known source of corrupt reads.
        self._data: np.memmap | None = None

    def _memmap(self) -> np.memmap:
        if self._data is None:
            self._data = np.memmap(self.path, dtype=self.dtype, mode="r", shape=self.shape)
        return self._data

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = int(self.indices[i])
        window = np.asarray(self._memmap()[row], dtype=np.float32)  # (T, C)

        if not np.isfinite(window).all():
            raise RuntimeError(f"window {row} contains non-finite values")

        if self.deterministic:
            # Seed by the window's position in the memmap, so the same window gets
            # the same mask on every epoch and in every worker.
            rng = np.random.default_rng(self.base_seed + row)
        else:
            rng = np.random.default_rng()

        mask = sample_mask((self.n_time, self.n_chan), rng, self.mask_cfg)

        target = torch.from_numpy(window).unsqueeze(0)                     # (1, T, C)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)    # (1, T, C)
        visible = target * (1.0 - mask_t)
        x = torch.cat([visible, mask_t], dim=0)                            # (2, T, C)
        return x, target, mask_t

    def describe(self) -> str:
        sessions = sorted(int(s) for s in self.meta.session_id.unique())
        return (
            f"{self.split:>5}: {len(self):>5} windows, {len(sessions):>2} sessions, "
            f"window ({self.n_time}, {self.n_chan}), "
            f"masks {'fixed' if self.deterministic else 'resampled each epoch'}\n"
            f"         sessions: {sessions}"
        )


def worker_init(worker_id: int) -> None:
    """Give each DataLoader worker a distinct entropy source.

    Without this, forked workers inherit the parent's numpy global state. We use
    default_rng() with OS entropy for training masks, so this mainly guards any
    future code that reaches for the legacy global RNG.
    """
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
