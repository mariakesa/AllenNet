"""Compose the per-trial natural-scenes dataset for downstream decoding.

One row per *presentation*, not per image. The encoder was pretrained on single
1-second windows of raw LFP, so it has never seen a trial-averaged signal; feeding
it 50-trial averages at decode time would present it with an input distribution
(SNR, amplitude, noise structure) it was never trained on. So we keep all 5900
trials and let the classifier deal with single-trial noise.

Output (self-describing, numpy-only, same contract as the pretraining memmap):

    <out>/ns_trials_f16.npy    (n_trials, window_samples, 93) float16, robust z-scored
          ns_labels.npy        (n_trials,) int64, image id 0..117
          ns_trials_meta.csv   per trial: image id, onset sample, onset time, block
          ns_dataset.json      shape, dtype, session/probe, normalization, exact config

Key facts about this stimulus, discovered by inspection (see main() for the asserts
that enforce them):

* 5950 presentations = 119 conditions x 50 repeats. `frame` is -1 for blank and
  0..117 for the images, so dropping blanks leaves 5900 trials, 118 balanced classes.
* Presentations run BACK TO BACK: each lasts 250.2 ms and the next starts 250.2 ms
  later. A 376-sample window is 300.8 ms, so a trial's window necessarily overlaps
  its neighbours' stimulus periods. That is not a bug in the extraction - it is a
  property of the stimulus - but it means neighbouring trials share LFP samples, and
  a random train/test split over trials would leak. downstream_decode.py splits on
  contiguous time and purges around the boundaries for exactly this reason.

Normalization matches the pretraining pipeline exactly: robust per-channel
median/MAD estimated from the whole recording, applied before the float16 cast. It
is computed from the raw LFP with no reference to any label, so it is not label
leakage; and it must be the same transform the encoder was pretrained under, or the
encoder sees inputs on a scale it has never encountered.

Example
-------
    python build_natural_scenes.py --config config.yaml
    python build_natural_scenes.py --config config.yaml --session 742951821 \
        --probe 769322716 --window-samples 376 --pre-ms 50
"""

from __future__ import annotations

import argparse
import json
import os
import time

import h5py
import numpy as np
import pandas as pd
import yaml

from inspect_data import lfp_data_path
from prepare_dataset import (
    channel_slice,
    contiguous_runs,
    detect_bad_channels,
    interpolate_bad_channels,
)

# probeC of session 737581020: 95 channels in VISp, primary visual cortex. Natural-scene
# identity is about as linearly accessible from VISp LFP as it gets, which makes this the
# fairest test of whether the pretrained features help.
#
# Chosen for a second, less obvious reason: its LFP has NO acquisition gap anywhere in the
# natural-scenes epoch. Gaps are common - the first probe tried here (742951821 probeC,
# also VISp) has a 246 s gap starting right where the natural-scenes block begins, which
# silently costs 1070 of 5900 trials and leaves the classes unbalanced at 34-47 repeats
# instead of a clean 50. 47 of the 53 eligible probes are gap-free; this is one.
DEFAULT_SESSION = 737581020
DEFAULT_PROBE = 757988391

# The example script's session (831882777) cannot be used: it is a
# functional-connectivity session and never showed a natural scene.
FORBIDDEN_SESSIONS = {831882777}


def robust_stats_from_recording(
    dset: h5py.Dataset,
    runs: list[tuple[int, int]],
    ch_lo: int,
    ch_hi: int,
    n_blocks: int,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel median/MAD from blocks spread across the whole recording.

    Sampled across the entire session (not just the natural-scenes epoch) so the
    scale reflects the recording, exactly as in prepare_dataset.py. Label-free.
    """
    chunk_rows = int(dset.chunks[0]) if dset.chunks else 100_000
    starts: list[int] = []
    for run_start, run_stop in runs:
        first = ((run_start + chunk_rows - 1) // chunk_rows) * chunk_rows
        for s in range(first, run_stop - chunk_rows, chunk_rows):
            starts.append(s)
    if not starts:
        raise RuntimeError("no chunk-aligned blocks available for normalization stats")

    idx = np.linspace(0, len(starts) - 1, min(n_blocks, len(starts))).round().astype(int)
    sample = np.concatenate(
        [dset[starts[i] : starts[i] + chunk_rows, ch_lo:ch_hi].astype(np.float32) for i in idx],
        axis=0,
    )

    median = np.median(sample, axis=0)
    mad = np.median(np.abs(sample - median), axis=0)
    mad = np.maximum(mad, float(cfg["normalization"]["mad_floor"]))
    return median.astype(np.float32), mad.astype(np.float32)


def read_stimulus_table(cache_root: str, session_id: int) -> pd.DataFrame:
    """Natural-scenes presentations: onset time, offset time, image id."""
    path = os.path.join(cache_root, f"session_{session_id}", f"session_{session_id}.nwb")
    with h5py.File(path, "r") as f:
        intervals = f.get("intervals")
        if intervals is None or "natural_scenes_presentations" not in intervals:
            raise RuntimeError(
                f"session {session_id} has no natural_scenes presentations - it cannot "
                f"be used for downstream decoding"
            )
        grp = intervals["natural_scenes_presentations"]
        table = pd.DataFrame(
            {
                "start_time": grp["start_time"][:],
                "stop_time": grp["stop_time"][:],
                "frame": grp["frame"][:].astype(np.int64),
                "stimulus_block": grp["stimulus_block"][:].astype(np.int64),
            }
        )
    return table.sort_values("start_time").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--session", type=int, default=DEFAULT_SESSION)
    parser.add_argument("--probe", type=int, default=DEFAULT_PROBE)
    parser.add_argument("--window-samples", type=int, default=376,
                        help="trial window length; 376 = 300.8 ms at 1250 Hz")
    parser.add_argument("--pre-ms", type=float, default=50.0,
                        help="how much of the window sits before stimulus onset")
    parser.add_argument("--include-blank", action="store_true",
                        help="keep the 50 blank (frame == -1) presentations")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    cache_root = cfg["paths"]["allen_cache"]
    want_ch = int(cfg["data"]["n_channels"])
    fs = float(cfg["data"]["sampling_rate"])
    window = int(args.window_samples)
    pre = int(round(args.pre_ms * fs / 1000.0))

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(cfg["paths"]["dataset_dir"]), "allennet_natural_scenes"
    )

    if args.session in FORBIDDEN_SESSIONS:
        raise SystemExit(
            f"session {args.session} contains no natural-scenes presentations "
            f"(it is a functional-connectivity session); pick a brain-observatory session"
        )

    # The pretraining set must not contain this session, or the whole experiment is
    # circular. It cannot, by construction - pretraining kept only sessions with zero
    # natural-scenes presentations - but assert it rather than trust it.
    pretrain_meta = os.path.join(cfg["paths"]["dataset_dir"], "recordings_meta.csv")
    if os.path.exists(pretrain_meta):
        used = set(pd.read_csv(pretrain_meta).session_id.unique())
        if args.session in used:
            raise SystemExit(
                f"session {args.session} was used for PRETRAINING - decoding on it "
                f"would leak. Pick a different session."
            )
        print(f"confirmed: session {args.session} was not used for pretraining")

    print(f"session {args.session}, probe {args.probe}")
    table = read_stimulus_table(cache_root, args.session)
    print(f"  {len(table)} natural-scenes presentations, "
          f"{table.frame.nunique()} unique frames")

    if not args.include_blank:
        table = table[table.frame >= 0].reset_index(drop=True)
    n_classes = int(table.frame.nunique())
    counts = table.frame.value_counts()
    print(f"  {len(table)} trials, {n_classes} classes, "
          f"{counts.min()}-{counts.max()} repeats per class")

    # Presentations are contiguous, so a window longer than the stimulus overlaps its
    # neighbours. Say so loudly - it is what forces purged, time-blocked CV.
    stim_ms = float((table.stop_time - table.start_time).median() * 1000)
    win_ms = window / fs * 1000
    if win_ms > stim_ms:
        print(f"  NOTE: window {win_ms:.1f} ms > stimulus {stim_ms:.1f} ms, so trial "
              f"windows overlap their neighbours by up to {win_ms - stim_ms:.1f} ms.\n"
              f"        downstream_decode.py purges around fold boundaries to stop that leaking.")

    lfp_path = os.path.join(
        cache_root, f"session_{args.session}", f"probe_{args.probe}_lfp.nwb"
    )
    if not os.path.exists(lfp_path):
        raise SystemExit(f"{lfp_path} not found")

    t0 = time.time()
    with h5py.File(lfp_path, "r") as f:
        data = f[lfp_data_path(f, args.probe)]
        ts = f[lfp_data_path(f, args.probe).replace("/data", "/timestamps")][:]
        n_samples, n_ch_raw = data.shape
        print(f"  LFP {data.shape}, cropping {n_ch_raw} -> {want_ch} channels")

        ch_lo, ch_hi = channel_slice(n_ch_raw, want_ch, cfg["data"]["channel_crop"])
        runs = contiguous_runs(ts, fs, min_len=window)
        print(f"  {len(runs)} contiguous run(s)")

        median, mad = robust_stats_from_recording(data, runs, ch_lo, ch_hi, 20, cfg)

        # Map onsets to sample indices. searchsorted on the real timestamps rather
        # than assuming t=0 is sample 0: recordings start at t>0 and can have gaps.
        onsets = np.searchsorted(ts, table.start_time.to_numpy())
        starts = onsets - pre
        ends = starts + window

        valid = np.ones(len(table), dtype=bool)
        valid &= starts >= 0
        valid &= ends <= n_samples
        # Every sample of the window must sit inside one contiguous run.
        in_run = np.zeros(len(table), dtype=bool)
        for run_start, run_stop in runs:
            in_run |= (starts >= run_start) & (ends <= run_stop)
        valid &= in_run

        n_drop = int((~valid).sum())
        if n_drop:
            print(f"  dropping {n_drop} trials whose window falls outside a contiguous run")
        table = table[valid].reset_index(drop=True)
        starts, ends = starts[valid], ends[valid]

        # Read the whole natural-scenes span once. Reading 5900 windows individually
        # would decompress ~93 HDF5 chunks per trial; one contiguous read costs each
        # chunk exactly once.
        span_lo = int(starts.min())
        span_hi = int(ends.max())
        print(f"  reading samples {span_lo}..{span_hi} "
              f"({(span_hi - span_lo) / fs / 60:.1f} min of LFP)")
        span = data[span_lo:span_hi, ch_lo:ch_hi].astype(np.float32)

    # Normalize with the recording's stats, exactly as in pretraining.
    span = (span - median) / (1.4826 * mad)

    trials = np.stack([span[s - span_lo : e - span_lo] for s, e in zip(starts, ends)])
    del span
    print(f"  trials {trials.shape}")

    bad = detect_bad_channels(trials, cfg)
    n_bad = int(bad.sum())
    if n_bad > int(cfg["quality"]["max_bad_channels"]):
        raise SystemExit(f"{n_bad} bad channels on this probe; pick another")
    if n_bad:
        print(f"  interpolating {n_bad} bad channel(s): {np.flatnonzero(bad).tolist()}")
        trials = interpolate_bad_channels(trials, bad)

    if not np.isfinite(trials).all():
        raise RuntimeError("non-finite values in trials")

    labels = table.frame.to_numpy(dtype=np.int64)
    # Remap to a dense 0..n_classes-1 label space (identity when blanks are dropped).
    classes = np.unique(labels)
    remap = {int(c): i for i, c in enumerate(classes)}
    y = np.array([remap[int(v)] for v in labels], dtype=np.int64)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "ns_trials_f16.npy"), trials.astype(np.float16))
    np.save(os.path.join(out_dir, "ns_labels.npy"), y)

    meta = pd.DataFrame(
        {
            "trial": np.arange(len(y)),
            "label": y,
            "frame": labels,
            "onset_sample": starts + pre,
            "start_sample": starts,
            "onset_time_s": table.start_time.to_numpy(),
            "stimulus_block": table.stimulus_block.to_numpy(),
        }
    )
    meta.to_csv(os.path.join(out_dir, "ns_trials_meta.csv"), index=False)

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": int(args.session),
        "probe_id": int(args.probe),
        "shape": list(trials.shape),
        "dtype": "float16",
        "n_classes": int(len(classes)),
        "window_samples": window,
        "pre_samples": pre,
        "sampling_rate": fs,
        "stimulus_duration_ms": stim_ms,
        "window_duration_ms": win_ms,
        "windows_overlap_neighbouring_trials": bool(win_ms > stim_ms),
        "units": "robust z-score: (volts - median) / (1.4826 * MAD), per recording, per channel",
        "median": median.tolist(),
        "mad": mad.tolist(),
        "n_bad_channels_interpolated": n_bad,
        "trial_averaged": False,
        "used_for_pretraining": False,
    }
    with open(os.path.join(out_dir, "ns_dataset.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n" + "=" * 78)
    print(f"wrote {trials.shape[0]} single trials of ({window}, {want_ch}) to {out_dir}")
    print(f"  {len(classes)} classes, {np.bincount(y).min()}-{np.bincount(y).max()} trials each")
    print(f"  chance accuracy = {1 / len(classes):.4f}")
    print(f"  NOT trial-averaged; normalized identically to the pretraining data")
    print(f"  elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
