"""Inspect the raw Allen Neuropixels cache and emit an inventory of LFP probes.

Nothing downstream is allowed to assume the on-disk format, so this script is the
one place that pokes at the raw NWB files and reports what is actually there.
It reads HDF5 metadata only (plus a short slice of timestamps to measure the
sampling rate), so it runs over the whole 700 GB cache in seconds.

Layout discovered in the Allen `EcephysProjectCache` download:

    <cache>/session_<sid>/session_<sid>.nwb
        intervals/<stimulus>_presentations/{start_time,stop_time}
    <cache>/session_<sid>/probe_<pid>_lfp.nwb
        acquisition/probe_<pid>_lfp/probe_<pid>_lfp_data/data        (T, C) float32
        acquisition/probe_<pid>_lfp/probe_<pid>_lfp_data/timestamps  (T,)   float64
        general/extracellular_ephys/electrodes/probe_vertical_position (C,) int64
        general/subject/subject_id

Example
-------
    python inspect_data.py --config config.yaml
    python inspect_data.py --config config.yaml --probe-detail 832810582
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml

# Timestamps are sampled in three slices of this length (head / middle / tail)
# to estimate the sampling rate. One HDF5 chunk is ~50k samples, so each slice
# costs a chunk or two rather than the whole 100 MB array.
_FS_PROBE_SAMPLES = 50_000


@dataclass
class ProbeRecord:
    """One row of the inventory: a single probe's LFP recording."""

    session_id: int
    probe_id: int
    mouse_id: str
    probe_name: str
    source_file: str
    n_samples: int
    n_channels: int
    sampling_rate: float
    n_nonpositive_timestamps: int
    duration_s: float
    chunk_rows: int
    dtype: str
    vertical_span_um: int
    channels_sorted_by_depth: bool
    structures: str
    has_natural_scenes: bool
    n_natural_scenes_presentations: int
    has_invalid_times: bool


def _decode(value: Any) -> str:
    """HDF5 scalar string datasets come back as bytes; normalise to str."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def session_stimulus_summary(session_nwb: str) -> tuple[set[str], int, bool]:
    """Return (stimulus names, #natural-scenes presentations, has invalid_times).

    The session NWB is the only place that knows which stimuli were shown, so it
    is the authority on whether a session is contaminated with natural scenes.
    """
    with h5py.File(session_nwb, "r") as f:
        if "intervals" not in f:
            return set(), 0, False
        intervals = f["intervals"]
        names = {k.replace("_presentations", "") for k in intervals.keys()}
        n_ns = 0
        if "natural_scenes_presentations" in intervals:
            ns = intervals["natural_scenes_presentations"]
            n_ns = int(ns["start_time"].shape[0]) if "start_time" in ns else 0
        has_invalid = "invalid_times" in intervals
    return names, n_ns, has_invalid


def estimate_sampling_rate(ts: h5py.Dataset, n_samples: int) -> tuple[float, int]:
    """Estimate fs from head/middle/tail slices of the timestamp vector.

    Some probes prefix the recording with a block of -1.0 sentinel timestamps
    (probe 836943715 has ~296k of them). Those produce zero diffs, so a naive
    median over the head returns 0 and the sampling rate comes out NaN or inf,
    silently disqualifying an otherwise perfectly good recording. Take the median
    over *positive* diffs only, and report how many non-positive timestamps were
    seen so the caller knows the recording needs the sentinel region masked out.

    Returns (fs, n_nonpositive_timestamps_seen_in_the_sampled_slices).
    """
    n = min(_FS_PROBE_SAMPLES, n_samples)
    offsets = [0, max(0, n_samples // 2 - n // 2), max(0, n_samples - n)]

    diffs: list[np.ndarray] = []
    n_nonpositive = 0
    for off in sorted(set(offsets)):
        chunk = ts[off : off + n]
        n_nonpositive += int((chunk <= 0).sum())
        diffs.append(np.diff(chunk))

    all_diffs = np.concatenate(diffs) if diffs else np.zeros(0)
    positive = all_diffs[all_diffs > 0]
    fs = float(1.0 / np.median(positive)) if positive.size else float("nan")
    return fs, n_nonpositive


def lfp_data_path(f: h5py.File, probe_id: int) -> str:
    """Path to the (T, C) LFP data array inside a probe NWB.

    Defensive: the Allen files name the group after the probe id, but we verify
    rather than trust, and fall back to a search if the convention ever changes.
    """
    expected = f"acquisition/probe_{probe_id}_lfp/probe_{probe_id}_lfp_data/data"
    if expected in f:
        return expected

    found: list[str] = []

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset) and name.endswith("_lfp_data/data"):
            found.append(name)

    f.visititems(visit)
    if len(found) != 1:
        raise RuntimeError(
            f"cannot locate a unique LFP data array in {f.filename!r}; found {found}"
        )
    return found[0]


def inspect_probe(
    lfp_file: str,
    session_id: int,
    has_natural_scenes: bool,
    n_ns: int,
    has_invalid: bool,
    cache_root: str,
) -> ProbeRecord:
    """Read metadata (not signal) for one probe LFP file."""
    match = re.search(r"probe_(\d+)_lfp\.nwb$", lfp_file)
    if match is None:
        raise ValueError(f"unexpected LFP filename: {lfp_file}")
    probe_id = int(match.group(1))

    with h5py.File(lfp_file, "r") as f:
        data = f[lfp_data_path(f, probe_id)]
        if data.ndim != 2:
            raise RuntimeError(f"{lfp_file}: expected 2-D (time, channel), got {data.shape}")
        n_samples, n_channels = int(data.shape[0]), int(data.shape[1])

        ts = f[lfp_data_path(f, probe_id).replace("/data", "/timestamps")]
        fs, n_nonpositive = estimate_sampling_rate(ts, n_samples)
        duration = float(ts[-1] - ts[0])

        electrodes = f["general/extracellular_ephys/electrodes"]
        vpos = electrodes["probe_vertical_position"][:]
        # Column order must equal depth order, otherwise the (1, k_channel)
        # spatial convolutions would be mixing non-adjacent electrodes.
        sorted_by_depth = bool(np.all(np.diff(vpos) >= 0))
        span = int(vpos.max() - vpos.min()) if n_channels else 0
        structures = sorted(
            {_decode(x) for x in electrodes["location"][:] if _decode(x).strip()}
        )
        group_names = electrodes["group_name"][:]
        probe_name = _decode(group_names[0]) if len(group_names) else ""
        mouse_id = _decode(f["general/subject/subject_id"][()])

        chunk_rows = int(data.chunks[0]) if data.chunks else n_samples
        dtype = str(data.dtype)

    return ProbeRecord(
        session_id=session_id,
        probe_id=probe_id,
        mouse_id=mouse_id,
        probe_name=probe_name,
        # Relative to the cache root, so the inventory stays valid if the cache is
        # moved or mounted elsewhere on the training machine.
        source_file=os.path.relpath(lfp_file, cache_root),
        n_samples=n_samples,
        n_channels=n_channels,
        sampling_rate=fs,
        n_nonpositive_timestamps=n_nonpositive,
        duration_s=duration,
        chunk_rows=chunk_rows,
        dtype=dtype,
        vertical_span_um=span,
        channels_sorted_by_depth=sorted_by_depth,
        structures=";".join(structures),
        has_natural_scenes=has_natural_scenes,
        n_natural_scenes_presentations=n_ns,
        has_invalid_times=has_invalid,
    )


def build_inventory(cache_root: str) -> pd.DataFrame:
    """Walk the cache and inspect every probe LFP file found."""
    session_dirs = sorted(glob.glob(os.path.join(cache_root, "session_*")))
    if not session_dirs:
        raise FileNotFoundError(f"no session_* directories under {cache_root!r}")

    records: list[ProbeRecord] = []
    for sdir in session_dirs:
        session_id = int(os.path.basename(sdir).split("_")[1])
        session_nwb = os.path.join(sdir, f"session_{session_id}.nwb")
        if not os.path.exists(session_nwb):
            print(f"  ! session {session_id}: session NWB missing, skipping session")
            continue

        stimuli, n_ns, has_invalid = session_stimulus_summary(session_nwb)
        has_ns = "natural_scenes" in stimuli

        for lfp_file in sorted(glob.glob(os.path.join(sdir, "probe_*_lfp.nwb"))):
            try:
                records.append(
                    inspect_probe(
                        lfp_file, session_id, has_ns, n_ns, has_invalid, cache_root
                    )
                )
            except Exception as exc:  # a corrupt file must not kill the sweep
                print(f"  ! {lfp_file}: {exc!r}")

    return pd.DataFrame([asdict(r) for r in records])


def summarise(df: pd.DataFrame, cfg: dict) -> None:
    """Print the facts that drive the dataset-synthesis decisions."""
    want_ch = int(cfg["data"]["n_channels"])
    want_fs = float(cfg["data"]["sampling_rate"])
    tol = float(cfg["data"]["sampling_rate_tol"])

    print("\n" + "=" * 78)
    print("INVENTORY")
    print("=" * 78)
    print(f"probes:   {len(df)}")
    print(f"sessions: {df.session_id.nunique()}   mice: {df.mouse_id.nunique()}")
    print(f"dtypes:   {sorted(df.dtype.unique())}")

    fs_ok = df.sampling_rate.between(want_fs - tol, want_fs + tol)
    print(
        f"sampling rate: min {df.sampling_rate.min():.3f}  max {df.sampling_rate.max():.3f} Hz"
        f"  |  within {want_fs}+/-{tol}: {int(fs_ok.sum())}/{len(df)}"
    )
    print(f"channels sorted by depth: {int(df.channels_sorted_by_depth.sum())}/{len(df)}")

    print("\nchannel-count histogram:")
    hist = df.n_channels.value_counts().sort_index()
    for n_ch, count in hist.items():
        flag = " <- usable" if n_ch >= want_ch else ""
        print(f"  {n_ch:3d} ch : {count:3d} probes{flag}")

    ns_sessions = df[df.has_natural_scenes].session_id.nunique()
    clean_sessions = df[~df.has_natural_scenes].session_id.nunique()
    print(f"\nsessions WITH natural scenes (excluded): {ns_sessions}")
    print(f"sessions WITHOUT natural scenes (usable): {clean_sessions}")

    usable = df[(~df.has_natural_scenes) & (df.n_channels >= want_ch) & fs_ok]
    total = int(usable.n_samples.sum())
    win = int(cfg["data"]["window_samples"])
    n_win = total // win
    gb = n_win * win * want_ch * 2 / 1e9
    print(
        f"\nUSABLE for pretraining (no natural scenes, >= {want_ch} ch, fs ok):"
        f"\n  {len(usable)} probes / {usable.session_id.nunique()} sessions / "
        f"{usable.mouse_id.nunique()} mice"
        f"\n  {total/1250/3600:.1f} recording-hours -> up to {n_win} windows "
        f"({gb:.1f} GB float16)"
    )
    print(f"  target is {cfg['data']['target_gb']} GB, so this is comfortably sufficient.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--probe-detail",
        type=int,
        default=None,
        help="dump the raw HDF5 tree for one probe id and exit",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    cache_root = cfg["paths"]["allen_cache"]

    if args.probe_detail is not None:
        hits = glob.glob(
            os.path.join(cache_root, "session_*", f"probe_{args.probe_detail}_lfp.nwb")
        )
        if not hits:
            raise SystemExit(f"probe {args.probe_detail} not found under {cache_root}")
        with h5py.File(hits[0], "r") as f:
            print(hits[0])
            f.visititems(
                lambda n, o: print(f"  {n:70s} {o.shape} {o.dtype}")
                if isinstance(o, h5py.Dataset)
                else None
            )
        return

    print(f"scanning {cache_root} ...")
    df = build_inventory(cache_root)

    out_csv = cfg["paths"]["inventory_csv"]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.sort_values(["session_id", "probe_id"]).to_csv(out_csv, index=False)
    summarise(df, cfg)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
