"""Auto-identify the phase contrast channel by intensity statistics.

The single most reliable discriminator between phase contrast and fluorescence
is the **absolute skewness of the intensity histogram**:

- Phase contrast fills the image with background texture, so the histogram is
  roughly symmetric around the background level — ``|skew|`` is typically < 1.
- Fluorescence is mostly dark with sparse bright signal — the histogram is
  strongly right-skewed, usually ``skew`` > 2 and often > 5.

Absolute skewness is scale-invariant, so it does not depend on dtype or
camera offset, and it works whether the phase background sits at 15 % or
60 % of the dynamic range. We pick the channel with the smallest ``|skew|``
and use the fraction of near-background pixels as a tiebreaker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import skew


@dataclass
class ChannelStats:
    index: int
    name: str
    mean: float
    std: float
    skewness: float
    near_bg_fraction: float  # pixels within ±5 % of the median

    @property
    def phase_likeness(self) -> float:
        # Lower |skew| ⇒ more phase-like. Return something *higher* = more phase-like
        # so callers can argmax.
        return -abs(self.skewness)


def _stats(plane: np.ndarray, index: int, name: str) -> ChannelStats:
    img = plane.astype(np.float64, copy=False)
    mean = float(img.mean())
    std = float(img.std())
    sk = float(skew(img, axis=None))
    if not np.isfinite(sk):
        sk = 0.0
    median = float(np.median(img))
    span = max(median * 0.05, 1.0)
    near_bg = float(np.mean(np.abs(img - median) < span))
    return ChannelStats(
        index=index,
        name=name,
        mean=mean,
        std=std,
        skewness=sk,
        near_bg_fraction=near_bg,
    )


def channel_stats(
    array_zcyx: np.ndarray, channel_names: list[str] | None = None
) -> list[ChannelStats]:
    """Return per-channel diagnostic statistics computed on the middle Z slice."""
    if array_zcyx.ndim != 4:
        raise ValueError(f"expected (Z, C, Y, X) array, got shape {array_zcyx.shape}")
    z_mid = array_zcyx.shape[0] // 2
    n_channels = array_zcyx.shape[1]
    names = channel_names or [f"C{i}" for i in range(n_channels)]
    return [_stats(array_zcyx[z_mid, c], c, names[c]) for c in range(n_channels)]


def detect_phase_channel(array_zcyx: np.ndarray) -> int:
    """Return the channel index most likely to be phase contrast for a single scene."""
    return detect_phase_channel_multi([array_zcyx])


def detect_phase_channel_multi(arrays_zcyx: list[np.ndarray]) -> int:
    """Decide the phase channel from one or more scenes of the same CZI.

    All scenes in a CZI share the same channel layout, so the decision should
    be made once per file, not per scene — otherwise a single unusual scene
    (bright artifact, strongly skewed phase field) can flip the answer. We
    average ``|skew|`` and ``near_bg_fraction`` across scenes and pick the
    channel with the smallest mean ``|skew|``, tiebroken by the largest mean
    near-background fraction.
    """
    if not arrays_zcyx:
        raise ValueError("need at least one scene array")
    all_stats = [channel_stats(a) for a in arrays_zcyx]
    n_channels = len(all_stats[0])
    if any(len(s) != n_channels for s in all_stats):
        raise ValueError("all scenes must have the same channel count")

    aggregated = []
    for c in range(n_channels):
        mean_abs_skew = float(np.mean([abs(s[c].skewness) for s in all_stats]))
        mean_near_bg = float(np.mean([s[c].near_bg_fraction for s in all_stats]))
        aggregated.append((c, mean_abs_skew, mean_near_bg))

    # Primary: smallest mean |skew|. Tiebreak: largest mean near_bg_fraction.
    best = min(aggregated, key=lambda row: (round(row[1], 1), -row[2]))
    return best[0]


def resolve_phase_channel(
    array_zcyx: np.ndarray,
    channel_names: list[str],
    override: str | int | None,
) -> int:
    """Resolve the phase-channel index from an optional user override.

    ``override`` may be an int (index) or a string (channel name, case-insensitive).
    If ``None``, fall back to ``detect_phase_channel``.
    """
    if override is None:
        return detect_phase_channel(array_zcyx)
    if isinstance(override, int):
        if not 0 <= override < array_zcyx.shape[1]:
            raise ValueError(f"phase channel index {override} out of range")
        return override
    lowered = [n.lower() for n in channel_names]
    target = override.lower()
    if target in lowered:
        return lowered.index(target)
    raise ValueError(
        f"phase channel name {override!r} not found in channels {channel_names!r}"
    )
