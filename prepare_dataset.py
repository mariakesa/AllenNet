"""Synthesize the masked-pretraining dataset: a disk-backed float16 memmap of LFP windows.

Produces a self-describing directory that can be copied to another machine and
loaded with nothing but numpy (no AllenSDK, no HDF5, no network):

    <dataset_dir>/
        lfp_windows_f16.dat   memmap, shape (N, window_samples, n_channels), float16
        windows_meta.csv      one row per window: provenance + split
        recordings_meta.csv   one row per probe: normalization stats, gaps, yield
        dataset.json          shapes, dtype, splits, exact config, provenance

Design decisions that are not obvious, and why:

* Natural scenes are excluded at the *session* level (config: exclusion.natural_scenes).
  Every Allen brain-observatory session contains natural scenes; only the
  functional-connectivity sessions never show them. Dropping windows around the
  natural-scenes blocks inside a contaminated session would still let the encoder
  see the same mouse, probe and cortical state that the downstream decoder is
  evaluated on. Keeping only the never-saw-natural-scenes sessions is the clean
  claim, and it still leaves ~66 recording-hours, far more than the 2.5 GB target.

* The LFP timestamps contain multi-minute gaps (the acquisition is stopped and
  restarted). Windows are cut only inside contiguous runs, never across a gap,
  otherwise a "1 s window" could silently span 17 minutes of wall-clock.

* HDF5 chunks are (~50000, 1): chunked along time, one channel per chunk. Reading
  a 1250-sample window across 93 channels therefore decompresses 93 whole chunks
  (~40x amplification). We instead read chunk-aligned blocks of ~one chunk row
  span and cut many consecutive windows out of each block, so every byte we
  decompress is a byte we keep.

* Normalization is robust (median / MAD), per recording, per channel, and applied
  *before* the float16 cast. Raw LFP is ~2e-4 V, which sits near the bottom of
  float16's precision; normalizing first puts the values at O(1) where float16 has
  ~3 decimal digits. Per-window normalization is deliberately NOT used - it would
  erase the amplitude differences the encoder should be learning.

Example
-------
    python prepare_dataset.py --config config.yaml
    python prepare_dataset.py --config config.yaml --target-gb 0.2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Iterator

import h5py
import numpy as np
import pandas as pd
import yaml

from inspect_data import lfp_data_path

# (start, stop) in seconds, on the same clock as the LFP timestamps.
Interval = tuple[float, float]


@dataclass
class Recording:
    """A probe selected for pretraining, plus everything learned while reading it."""

    session_id: int
    probe_id: int
    mouse_id: str
    probe_name: str
    source_file: str
    n_samples: int
    n_channels_raw: int
    sampling_rate: float
    chunk_rows: int
    channel_slice: tuple[int, int] = (0, 0)
    n_runs: int = 0
    n_gap_seconds: float = 0.0
    n_unusable_samples: int = 0
    split: str = ""
    n_windows_planned: int = 0
    n_windows_kept: int = 0
    n_rejected_nan: int = 0
    n_rejected_artifact: int = 0
    n_bad_channels: int = 0
    bad_channels: list[int] = field(default_factory=list)
    dropped_reason: str = ""
    median: np.ndarray | None = field(default=None, repr=False)
    mad: np.ndarray | None = field(default=None, repr=False)


# ----------------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------------


def select_recordings(inventory: pd.DataFrame, cfg: dict) -> list[Recording]:
    """Apply the exclusion / validation rules and return the probes to read."""
    want_ch = int(cfg["data"]["n_channels"])
    want_fs = float(cfg["data"]["sampling_rate"])
    tol = float(cfg["data"]["sampling_rate_tol"])
    window = int(cfg["data"]["window_samples"])
    ns_policy = cfg["exclusion"]["natural_scenes"]

    df = inventory.copy()
    n0 = len(df)

    def drop(mask: pd.Series, reason: str) -> pd.DataFrame:
        nonlocal df
        removed = int((~mask).sum())
        if removed:
            print(f"  drop {removed:3d} probes: {reason}")
        df = df[mask]
        return df

    if ns_policy == "session":
        drop(~df.has_natural_scenes, "session contains natural-scenes presentations")
    elif ns_policy != "interval":
        raise ValueError(f"exclusion.natural_scenes must be 'session' or 'interval'")

    drop(df.n_channels >= want_ch, f"fewer than {want_ch} channels")
    drop(df.sampling_rate.between(want_fs - tol, want_fs + tol), f"sampling rate outside {want_fs}+/-{tol} Hz")
    drop(df.channels_sorted_by_depth, "channels not ordered by depth")
    drop(df.n_samples >= 10 * window, "recording too short")
    drop(df["dtype"].isin(["float32", "float64"]), "unexpected dtype")

    print(f"  selected {len(df)}/{n0} probes, "
          f"{df.session_id.nunique()} sessions, {df.mouse_id.nunique()} mice")

    if df.empty:
        raise RuntimeError("no probes survived selection")

    return [
        Recording(
            session_id=int(r.session_id),
            probe_id=int(r.probe_id),
            mouse_id=str(r.mouse_id),
            probe_name=str(r.probe_name),
            source_file=str(r.source_file),
            n_samples=int(r.n_samples),
            n_channels_raw=int(r.n_channels),
            sampling_rate=float(r.sampling_rate),
            chunk_rows=int(r.chunk_rows),
        )
        for r in df.sort_values(["session_id", "probe_id"]).itertuples()
    ]


def assign_splits(recs: list[Recording], cfg: dict) -> None:
    """Split by session (== mouse here), never by window.

    Random windows from one recording share its electrode positions, its drift and
    its normalization statistics; splitting on them would report a validation loss
    that says nothing about generalisation to a new recording.
    """
    key = cfg["split"]["by"]
    if key not in {"session", "mouse"}:
        raise ValueError(f"split.by must be 'session' or 'mouse', got {key!r}")

    groups = sorted({(r.session_id if key == "session" else r.mouse_id) for r in recs})
    rng = np.random.default_rng(int(cfg["split"]["seed"]))
    order = rng.permutation(len(groups))
    n_val = max(1, int(round(len(groups) * float(cfg["split"]["val_fraction"]))))
    val = {groups[i] for i in order[:n_val]}

    for r in recs:
        g = r.session_id if key == "session" else r.mouse_id
        r.split = "val" if g in val else "train"

    n_tr = sum(1 for r in recs if r.split == "train")
    print(f"  split by {key}: {len(groups) - n_val} train / {n_val} val groups "
          f"({n_tr} / {len(recs) - n_tr} probes)")


def channel_slice(n_raw: int, want: int, mode: str) -> tuple[int, int]:
    """Pick a *contiguous* span of `want` channels out of `n_raw`.

    Contiguity is the point: electrodes are 40 um apart and sorted by depth, so a
    contiguous span keeps neighbouring columns physically neighbouring, which is
    what the (1, k_channel) spatial convolutions assume.
    """
    if n_raw < want:
        raise ValueError(f"probe has {n_raw} channels, need {want}")
    extra = n_raw - want
    if mode == "center":
        lo = extra // 2
    elif mode == "first":
        lo = 0
    elif mode == "last":
        lo = extra
    else:
        raise ValueError(f"unknown channel_crop {mode!r}")
    return lo, lo + want


# ----------------------------------------------------------------------------
# exclusion intervals and contiguous runs
# ----------------------------------------------------------------------------


def excluded_intervals(cache_root: str, session_id: int, cfg: dict) -> list[Interval]:
    """Time intervals (seconds) that no window may overlap.

    For the functional-connectivity sessions we keep, natural_scenes is always
    absent - we still look, so that the guarantee is enforced by code rather than
    by assumption, and so that exclusion.natural_scenes='interval' also works.
    """
    guard = float(cfg["exclusion"]["guard_seconds"])
    path = os.path.join(cache_root, f"session_{session_id}", f"session_{session_id}.nwb")
    out: list[Interval] = []

    with h5py.File(path, "r") as f:
        intervals = f.get("intervals")
        if intervals is None:
            return out
        wanted = ["natural_scenes_presentations"]
        if cfg["exclusion"]["drop_invalid_times"]:
            wanted.append("invalid_times")
        for name in wanted:
            if name not in intervals:
                continue
            grp = intervals[name]
            if "start_time" not in grp or "stop_time" not in grp:
                continue
            starts = grp["start_time"][:]
            stops = grp["stop_time"][:]
            out.extend(
                (float(a) - guard, float(b) + guard) for a, b in zip(starts, stops)
            )
    return _merge(out)


def _merge(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping intervals so overlap tests stay cheap."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, stop in intervals[1:]:
        last_start, last_stop = merged[-1]
        if start <= last_stop:
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def contiguous_runs(timestamps: np.ndarray, fs: float, min_len: int) -> list[tuple[int, int]]:
    """Split a timestamp vector into runs of validly, uniformly sampled data.

    Two distinct defects have to break a run:

    * Gaps. Acquisition is stopped and restarted, so timestamps can jump by many
      minutes (one brain-observatory probe jumps 1014 s). A window cut across such
      a jump would look like 1 s of LFP but actually span 17 minutes.
    * Sentinels. Some probes prefix the recording with a block of -1.0 timestamps
      (probe 836943715: ~296k samples, ~237 s). Those samples have no wall-clock
      time, so they cannot be checked against the stimulus table - we cannot prove
      they are not natural scenes, so they must not enter the dataset.

    Returns [start, stop) sample index pairs for runs of at least `min_len` samples.
    """
    valid = timestamps > 0
    dt = np.diff(timestamps)
    # A real step is ~1/fs; require strictly increasing and no more than 1.5x.
    step_ok = (dt > 0) & (dt <= 1.5 / fs)
    # Sample i links to i+1 only if both ends are valid and the step is sane.
    linked = step_ok & valid[:-1] & valid[1:]

    breaks = np.flatnonzero(~linked)
    edges = np.concatenate(([0], breaks + 1, [len(timestamps)]))
    runs = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b - a >= min_len]
    # A lone invalid sample yields a 1-length "run" at its own index; the min_len
    # filter above removes those, and also any run too short to hold a window.
    return [(a, b) for a, b in runs if valid[a] and valid[b - 1]]


# ----------------------------------------------------------------------------
# window planning
# ----------------------------------------------------------------------------


def plan_blocks(
    rec: Recording,
    runs: list[tuple[int, int]],
    excluded: list[Interval],
    timestamps: np.ndarray,
    quota: int,
    cfg: dict,
) -> list[tuple[int, int]]:
    """Choose chunk-aligned read blocks yielding ~`quota` windows, spread over time.

    Returns [(block_start, n_windows)] in sample indices. Each block is read once
    and sliced into `n_windows` consecutive non-overlapping windows.
    """
    window = int(cfg["data"]["window_samples"])
    stride = int(cfg["data"]["stride_samples"])

    # One HDF5 chunk spans `chunk_rows` samples of a single channel. Sizing a block
    # to a whole number of windows within one chunk row means each chunk we
    # decompress is consumed end to end.
    per_block = max(1, rec.chunk_rows // stride)
    block_len = per_block * stride

    # Candidate blocks: chunk-aligned starts inside gap-free, non-excluded time.
    candidates: list[int] = []
    for run_start, run_stop in runs:
        first = ((run_start + rec.chunk_rows - 1) // rec.chunk_rows) * rec.chunk_rows
        for start in range(first, run_stop - block_len + 1, block_len):
            t0 = float(timestamps[start])
            t1 = float(timestamps[start + block_len - 1])
            if any(t0 < stop and start_x < t1 for start_x, stop in excluded):
                continue
            candidates.append(start)

    if not candidates:
        return []

    n_blocks = min(len(candidates), max(1, -(-quota // per_block)))  # ceil div
    # Spread the chosen blocks evenly across the recording rather than taking a
    # contiguous chunk of it: brain state drifts over a session, and we want the
    # encoder to see all of it.
    idx = np.linspace(0, len(candidates) - 1, n_blocks).round().astype(int)
    chosen = sorted({candidates[i] for i in idx})

    plan: list[tuple[int, int]] = []
    remaining = quota
    for start in chosen:
        take = min(per_block, remaining)
        if take <= 0:
            break
        plan.append((start, take))
        remaining -= take
    return plan


# ----------------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------------


def read_windows(
    dset: h5py.Dataset,
    plan: list[tuple[int, int]],
    ch_lo: int,
    ch_hi: int,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the planned blocks and cut them into windows.

    Returns (windows (n, window, ch) float32, start_sample (n,) int64).
    """
    window = int(cfg["data"]["window_samples"])
    stride = int(cfg["data"]["stride_samples"])

    chunks: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    for block_start, n_win in plan:
        span = (n_win - 1) * stride + window
        block = dset[block_start : block_start + span, ch_lo:ch_hi].astype(np.float32)
        if block.shape[0] < span:  # defensive: short read at EOF
            continue
        # (n_win, window, ch) view via strides, then copy once.
        idx = np.arange(n_win) * stride
        cut = np.stack([block[i : i + window] for i in idx], axis=0)
        chunks.append(cut)
        starts.append(block_start + idx)

    if not chunks:
        empty_w = np.zeros((0, window, ch_hi - ch_lo), dtype=np.float32)
        return empty_w, np.zeros((0,), dtype=np.int64)
    return np.concatenate(chunks, axis=0), np.concatenate(starts).astype(np.int64)


def robust_stats(windows: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel median and MAD over every sample of this recording's windows.

    Median/MAD rather than mean/std so that a few artifact-laden windows cannot
    drag the scale of the whole recording.
    """
    flat = windows.reshape(-1, windows.shape[-1])  # (n*window, ch)
    median = np.median(flat, axis=0)
    mad = np.median(np.abs(flat - median), axis=0)
    mad = np.maximum(mad, float(cfg["normalization"]["mad_floor"]))
    return median.astype(np.float32), mad.astype(np.float32)


def detect_bad_channels(z: np.ndarray, cfg: dict) -> np.ndarray:
    """Find electrodes that misbehave across the whole recording.

    A bad channel is one where an implausible fraction of *all* its samples are
    extreme, which is a property of the electrode, not of any one moment. This is
    deliberately different from window-level QC: a window is rejected for a
    transient event, a channel is repaired for being persistently broken.

    Returns a boolean mask over channels.
    """
    q = cfg["quality"]
    frac_extreme = (np.abs(z) > float(q["extreme_z"])).mean(axis=(0, 1))  # (ch,)
    return frac_extreme > float(q["bad_channel_extreme_frac"])


def interpolate_bad_channels(z: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Replace bad channels by linear interpolation between their nearest good neighbours.

    Columns are depth-ordered electrodes on a 40 um pitch, so a bad channel's
    signal is well approximated by its neighbours. Interpolating keeps the array
    rectangular and keeps every column physically where the spatial convolutions
    expect it - dropping the column instead would silently move every deeper
    electrode one step up.

    z: (n_windows, time, channel), modified in place.
    """
    good = np.flatnonzero(~bad)
    if good.size == 0:
        raise RuntimeError("every channel is bad")

    for c in np.flatnonzero(bad):
        below = good[good < c]
        above = good[good > c]
        if below.size and above.size:
            lo, hi = int(below[-1]), int(above[0])
            w = (c - lo) / (hi - lo)
            z[:, :, c] = (1.0 - w) * z[:, :, lo] + w * z[:, :, hi]
        else:  # bad channel at an edge of the shank: copy the nearest good one
            nearest = int(below[-1]) if below.size else int(above[0])
            z[:, :, c] = z[:, :, nearest]
    return z


def quality_mask(z: np.ndarray, cfg: dict) -> tuple[np.ndarray, int, int]:
    """Flag windows to keep. Returns (keep mask, #NaN rejects, #artifact rejects)."""
    q = cfg["quality"]
    n = z.shape[0]

    nan_bad = np.zeros(n, dtype=bool)
    if q["reject_nan"]:
        nan_bad = ~np.isfinite(z).all(axis=(1, 2))

    absz = np.abs(z)
    too_big = absz.max(axis=(1, 2)) > float(q["max_abs_z"])
    frac_extreme = (absz > float(q["extreme_z"])).mean(axis=(1, 2))
    artifact = (too_big | (frac_extreme > float(q["max_extreme_frac"]))) & ~nan_bad

    keep = ~(nan_bad | artifact)
    return keep, int(nan_bad.sum()), int(artifact.sum())


def process_recording(
    rec: Recording,
    cache_root: str,
    quota: int,
    cfg: dict,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Read one probe end to end and return its accepted windows (float16) + metadata."""
    window = int(cfg["data"]["window_samples"])
    want_ch = int(cfg["data"]["n_channels"])
    path = os.path.join(cache_root, rec.source_file) if not os.path.isabs(rec.source_file) else rec.source_file

    rec.channel_slice = channel_slice(rec.n_channels_raw, want_ch, cfg["data"]["channel_crop"])
    excluded = excluded_intervals(cache_root, rec.session_id, cfg)

    if cfg["exclusion"]["natural_scenes"] == "session":
        # Belt and braces: the inventory already dropped natural-scenes sessions.
        pass

    with h5py.File(path, "r") as f:
        dset = f[lfp_data_path(f, rec.probe_id)]
        ts = f[lfp_data_path(f, rec.probe_id).replace("/data", "/timestamps")][:]

        runs = contiguous_runs(ts, rec.sampling_rate, min_len=window)
        rec.n_runs = len(runs)
        # Samples that no run covers: gap boundaries plus any sentinel region.
        usable = sum(b - a for a, b in runs)
        rec.n_unusable_samples = int(rec.n_samples - usable)
        rec.n_gap_seconds = float(rec.n_unusable_samples / rec.sampling_rate)

        plan = plan_blocks(rec, runs, excluded, ts, quota, cfg)
        rec.n_windows_planned = sum(n for _, n in plan)
        if not plan:
            print(f"    ! {rec.probe_id}: no usable blocks")
            return np.zeros((0, window, want_ch), np.float16), pd.DataFrame()

        raw, starts = read_windows(dset, plan, *rec.channel_slice, cfg)

    if raw.shape[0] == 0:
        return np.zeros((0, window, want_ch), np.float16), pd.DataFrame()

    median, mad = robust_stats(raw, cfg)
    rec.median, rec.mad = median, mad

    # z = (x - median) / (1.4826 * MAD): 1.4826 makes MAD a std estimate for
    # Gaussian data, so z is comparable to a z-score across recordings.
    z = (raw - median) / (1.4826 * mad)
    del raw

    # Repair persistently-broken electrodes before judging individual windows,
    # otherwise one bad channel condemns every window on the probe.
    bad = detect_bad_channels(z, cfg)
    rec.bad_channels = [int(c) + rec.channel_slice[0] for c in np.flatnonzero(bad)]
    rec.n_bad_channels = int(bad.sum())
    if rec.n_bad_channels > int(cfg["quality"]["max_bad_channels"]):
        print(f"    ! {rec.probe_id}: {rec.n_bad_channels} bad channels "
              f"(> max {cfg['quality']['max_bad_channels']}), dropping recording")
        rec.dropped_reason = f"{rec.n_bad_channels} bad channels"
        rec.n_windows_kept = 0
        return np.zeros((0, window, want_ch), np.float16), pd.DataFrame()
    if rec.n_bad_channels:
        z = interpolate_bad_channels(z, bad)

    keep, n_nan, n_art = quality_mask(z, cfg)
    rec.n_rejected_nan, rec.n_rejected_artifact = n_nan, n_art
    z, starts = z[keep], starts[keep]
    rec.n_windows_kept = int(z.shape[0])

    meta = pd.DataFrame(
        {
            "session_id": rec.session_id,
            "probe_id": rec.probe_id,
            "mouse_id": rec.mouse_id,
            "probe_name": rec.probe_name,
            "source_file": rec.source_file,
            "start_sample": starts,
            "start_time_s": np.nan,  # filled by caller-free path below
            "sampling_rate": rec.sampling_rate,
            "channel_count": want_ch,
            "channel_lo": rec.channel_slice[0],
            "channel_hi": rec.channel_slice[1],
            "split": rec.split,
        }
    )
    meta["start_time_s"] = starts / rec.sampling_rate

    return z.astype(np.float16), meta


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--target-gb", type=float, default=None, help="override data.target_gb")
    parser.add_argument("--dataset-dir", default=None, help="override paths.dataset_dir")
    parser.add_argument("--limit-probes", type=int, default=None, help="use only the first N probes (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="plan and report, write nothing")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.target_gb is not None:
        cfg["data"]["target_gb"] = args.target_gb
    if args.dataset_dir is not None:
        cfg["paths"]["dataset_dir"] = args.dataset_dir

    cache_root = cfg["paths"]["allen_cache"]
    window = int(cfg["data"]["window_samples"])
    want_ch = int(cfg["data"]["n_channels"])
    out_dir = cfg["paths"]["dataset_dir"]

    inv_path = cfg["paths"]["inventory_csv"]
    if not os.path.exists(inv_path):
        raise SystemExit(f"inventory {inv_path!r} missing - run inspect_data.py first")
    inventory = pd.read_csv(inv_path)

    print("selecting recordings")
    recs = select_recordings(inventory, cfg)
    if args.limit_probes:
        recs = recs[: args.limit_probes]
        print(f"  --limit-probes: using {len(recs)} probes")
    assign_splits(recs, cfg)

    # Confirm the headline guarantee rather than assuming it.
    contaminated = [r.session_id for r in recs
                    if bool(inventory.set_index("probe_id").loc[r.probe_id, "has_natural_scenes"])]
    if cfg["exclusion"]["natural_scenes"] == "session" and contaminated:
        raise RuntimeError(f"natural-scenes sessions survived selection: {sorted(set(contaminated))}")
    print(f"  CONFIRMED: 0 of {len(recs)} selected probes come from a session containing natural scenes")

    bytes_per_window = window * want_ch * 2
    total_windows = int(float(cfg["data"]["target_gb"]) * 1e9 // bytes_per_window)
    quota = max(1, total_windows // len(recs))
    print(
        f"\ntarget {cfg['data']['target_gb']} GB -> {total_windows} windows of "
        f"({window}, {want_ch}) float16 -> ~{quota} windows/probe over {len(recs)} probes"
    )

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    os.makedirs(out_dir, exist_ok=True)
    dat_path = os.path.join(out_dir, "lfp_windows_f16.dat")

    # Allocate for the planned maximum; truncate to what actually survived QC.
    capacity = quota * len(recs)
    mm = np.memmap(dat_path, dtype=np.float16, mode="w+", shape=(capacity, window, want_ch))

    written = 0
    metas: list[pd.DataFrame] = []
    t0 = time.time()

    for i, rec in enumerate(recs, 1):
        t_rec = time.time()
        print(f"[{i:2d}/{len(recs)}] session {rec.session_id} probe {rec.probe_id} "
              f"({rec.n_channels_raw} ch, {rec.split})", flush=True)
        wins, meta = process_recording(rec, cache_root, quota, cfg)
        n = wins.shape[0]
        if n:
            mm[written : written + n] = wins
            meta.insert(0, "index", np.arange(written, written + n))
            metas.append(meta)
            written += n
        repaired = f", repaired ch {rec.bad_channels}" if rec.bad_channels else ""
        print(f"         runs={rec.n_runs} gaps={rec.n_gap_seconds:6.1f}s  "
              f"kept {n}/{rec.n_windows_planned}  "
              f"(nan {rec.n_rejected_nan}, artifact {rec.n_rejected_artifact}{repaired})  "
              f"{time.time() - t_rec:.1f}s", flush=True)

    mm.flush()
    del mm

    # Trim the tail left over from rejected windows.
    os.truncate(dat_path, written * bytes_per_window)

    windows_meta = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
    windows_meta.to_csv(os.path.join(out_dir, "windows_meta.csv"), index=False)

    rec_rows = []
    for r in recs:
        rec_rows.append(
            {
                "session_id": r.session_id,
                "probe_id": r.probe_id,
                "mouse_id": r.mouse_id,
                "probe_name": r.probe_name,
                "source_file": r.source_file,
                "split": r.split,
                "sampling_rate": r.sampling_rate,
                "n_samples": r.n_samples,
                "n_channels_raw": r.n_channels_raw,
                "channel_lo": r.channel_slice[0],
                "channel_hi": r.channel_slice[1],
                "n_runs": r.n_runs,
                "gap_seconds": r.n_gap_seconds,
                "n_unusable_samples": r.n_unusable_samples,
                "n_windows_kept": r.n_windows_kept,
                "n_rejected_nan": r.n_rejected_nan,
                "n_rejected_artifact": r.n_rejected_artifact,
                # Interpolated electrodes: these columns are reconstructed, not
                # measured. Recorded so the fact never gets lost downstream.
                "n_bad_channels": r.n_bad_channels,
                "bad_channels_interpolated": json.dumps(r.bad_channels),
                "dropped_reason": r.dropped_reason,
                # Normalization stats: kept so the float16 z-scores can be mapped
                # back to volts, and so the downstream natural-scenes data can be
                # normalized the same way.
                "median": json.dumps(r.median.tolist()) if r.median is not None else "",
                "mad": json.dumps(r.mad.tolist()) if r.mad is not None else "",
            }
        )
    pd.DataFrame(rec_rows).to_csv(os.path.join(out_dir, "recordings_meta.csv"), index=False)

    n_train = int((windows_meta.split == "train").sum()) if len(windows_meta) else 0
    n_val = written - n_train

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_revision": git_revision(),
        "memmap": {
            "file": "lfp_windows_f16.dat",
            "dtype": "float16",
            "shape": [written, window, want_ch],
            "order": "C",
            "axes": ["window", "time", "channel"],
            "bytes": written * bytes_per_window,
        },
        "units": "robust z-score: (volts - median) / (1.4826 * MAD), per recording, per channel",
        "sampling_rate": float(cfg["data"]["sampling_rate"]),
        "n_windows": written,
        "n_train": n_train,
        "n_val": n_val,
        "n_recordings": len(recs),
        "n_sessions": len({r.session_id for r in recs}),
        "n_mice": len({r.mouse_id for r in recs}),
        "natural_scenes_excluded": True,
        "natural_scenes_exclusion_policy": cfg["exclusion"]["natural_scenes"],
        "n_recordings_dropped": sum(1 for r in recs if r.dropped_reason),
        "n_channels_interpolated": sum(r.n_bad_channels for r in recs),
        "config": cfg,
    }
    with open(os.path.join(out_dir, "dataset.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    gb = written * bytes_per_window / 1e9
    print("\n" + "=" * 78)
    print(f"wrote {written} windows ({gb:.2f} GB float16) to {dat_path}")
    print(f"  shape       : ({written}, {window}, {want_ch})")
    print(f"  train / val : {n_train} / {n_val} windows "
          f"({len({r.session_id for r in recs if r.split == 'train'})} / "
          f"{len({r.session_id for r in recs if r.split == 'val'})} sessions)")
    print(f"  elapsed     : {time.time() - t0:.0f}s")
    print(f"  manifest    : {os.path.join(out_dir, 'dataset.json')}")
    print("\nThis directory is self-contained: copy it to the training machine as-is.")
    print("Load with:  np.memmap(path, dtype=np.float16, mode='r', shape=tuple(shape))")


if __name__ == "__main__":
    main()
