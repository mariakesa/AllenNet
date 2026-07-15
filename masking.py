"""Contiguous spatiotemporal masks for the reconstruction pretext task.

The point of the pretext task is to force the encoder to infer missing LFP from
spatial and temporal *context*. That only works if what is removed cannot be
recovered locally. Masking isolated samples fails this test completely: a hole of
one sample sits between two samples 0.8 ms away on a signal with no meaningful
power at 625 Hz, so linear interpolation solves it and the encoder learns nothing.

So every masked region is contiguous, and the mask is a mixture of three shapes:

    temporal block   all 93 channels, a span of time      -> "what happened during
                                                              this stretch?"
    channel block    all 1250 samples, a span of depth    -> "what does this part of
                                                              the shank see?"
    rectangle        bounded in both                      -> the general case

The mixture matters. Temporal-only masking can be solved by copying neighbouring
channels (LFP is highly correlated across 40 um); channel-only masking can be
solved by copying neighbouring time. Only a mixture forces the model to use both
axes.

Example
-------
    rng = np.random.default_rng(0)
    mask = sample_mask((1250, 93), rng, cfg["masking"])   # bool, True = masked
"""

from __future__ import annotations

import numpy as np

# Region kinds, in the order their weights are read from the config.
_KINDS = ("temporal_block", "channel_block", "rectangle")


def _span(
    rng: np.random.Generator,
    extent: int,
    width_range: list[int],
    max_width: int | None = None,
) -> tuple[int, int]:
    """A contiguous [lo, hi) span of random width, clipped to [0, extent).

    `max_width` caps the span to the masking budget still available. Without that
    cap the fraction overshoots: a temporal block spans all 93 channels, so one
    250-sample block adds 20% coverage in a single step and a mask targeting 55%
    can land at 65%, outside the 40-60% band instructions.txt asks for.
    """
    lo_w, hi_w = int(width_range[0]), int(width_range[1])
    width = int(rng.integers(lo_w, hi_w + 1))
    width = min(width, extent)
    if max_width is not None:
        width = min(width, max(1, int(max_width)))
    start = int(rng.integers(0, extent - width + 1))
    return start, start + width


def sample_mask(
    shape: tuple[int, int],
    rng: np.random.Generator,
    cfg: dict,
) -> np.ndarray:
    """Draw a boolean mask over a (time, channel) window. True == masked/hidden.

    Regions are added until the target masked fraction is reached, so regions may
    overlap and the realised fraction lands close to (never far above) the target.

    Args:
        shape: (T, C) of the window.
        rng: caller-owned generator; the caller decides whether this is a fresh
            draw (training) or seeded by window index (validation).
        cfg: the `masking` block of config.yaml.

    Returns:
        (T, C) bool array.
    """
    n_time, n_chan = int(shape[0]), int(shape[1])
    if n_time <= 0 or n_chan <= 0:
        raise ValueError(f"bad window shape {shape}")

    target = float(rng.uniform(float(cfg["min_fraction"]), float(cfg["max_fraction"])))

    weights = np.array([float(cfg["weights"][k]) for k in _KINDS], dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("masking.weights must not be all zero")
    weights = weights / weights.sum()

    t_range = cfg["temporal_width"]
    c_range = cfg["channel_width"]
    max_regions = int(cfg["max_regions"])

    mask = np.zeros((n_time, n_chan), dtype=bool)
    n_cells = n_time * n_chan
    n_masked = 0

    for _ in range(max_regions):
        if n_masked / n_cells >= target:
            break

        # Cells we may still mask before hitting the target. Regions are capped to
        # this so the realised fraction converges to the target from below instead
        # of overshooting past it.
        budget = max(1, int(target * n_cells) - n_masked)

        kind = _KINDS[int(rng.choice(len(_KINDS), p=weights))]
        if kind == "temporal_block":
            t0, t1 = _span(rng, n_time, t_range, max_width=budget // n_chan)
            c0, c1 = 0, n_chan
        elif kind == "channel_block":
            t0, t1 = 0, n_time
            c0, c1 = _span(rng, n_chan, c_range, max_width=budget // n_time)
        else:  # rectangle
            t0, t1 = _span(rng, n_time, t_range)
            c0, c1 = _span(rng, n_chan, c_range, max_width=budget // max(t1 - t0, 1))

        mask[t0:t1, c0:c1] = True
        # Recount rather than accumulate: regions overlap, so adding their areas
        # would overshoot and stop the loop early.
        n_masked = int(mask.sum())

    # Degenerate masks make the loss undefined (division by zero) or trivial.
    if n_masked == 0:
        t0, t1 = _span(rng, n_time, t_range)
        mask[t0:t1, :] = True
    elif n_masked == n_cells:
        # Nothing visible to reconstruct *from*; unmask a strip.
        mask[:, : max(1, n_chan // 8)] = False

    return mask


def masked_fraction(mask: np.ndarray) -> float:
    """Fraction of entries hidden by `mask`."""
    return float(mask.mean())
