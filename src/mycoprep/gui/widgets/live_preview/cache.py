"""Per-FOV cache of pipeline-stage results.

Keys are tuples of the relevant option fields; if a key matches between
runs the cached payload is returned and the stage is skipped. This is
what makes navigating between Focus / Segment / Features tabs feel
instantaneous: each stage downstream of an unchanged stage stays valid.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# Hashable key types for each stage. Plain tuples (rather than dataclasses)
# so dict-keying and equality just work.
FocusKey = tuple
SegmentKey = tuple
ClassifyKey = tuple
FeaturesKey = tuple


@dataclass
class CacheEntry:
    """One FOV's most-recent stage results."""

    # Focus stage outputs
    focus_key: Optional[FocusKey] = None
    phase: Optional[np.ndarray] = None
    image_channels: Optional[np.ndarray] = None  # (C, Y, X)
    channel_names: Optional[list[str]] = None
    # The phase-channel index resolved from the image data (after
    # auto-detection by skewness). Cached so subsequent renders that
    # hit the focus cache can skip re-running the detection.
    resolved_phase_channel: Optional[int] = None

    # Segment stage outputs
    segment_key: Optional[SegmentKey] = None
    mask: Optional[np.ndarray] = None
    n_cells: int = 0

    # Classify stage outputs
    classify_key: Optional[ClassifyKey] = None
    decisions: Optional[dict[int, str]] = None

    # Features stage outputs
    features_key: Optional[FeaturesKey] = None
    features_df: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Key builders — explicit tuples of the option fields that affect each stage.
# Keep these aligned with the upstream dataclasses (FocusOpts, SegmentOpts,
# ClassifyOpts, ExtractOpts in mycoprep.core.api).
# ---------------------------------------------------------------------------

def _phase_part(phase_channel: Any) -> Any:
    """Render ``phase_channel`` as a stable, hashable cache-key part.

    Plain int → ``int``; the "auto" sentinel (``None`` or a string
    label) is preserved verbatim so two renders with the same
    auto-detect setting share a cache entry.
    """
    if isinstance(phase_channel, int):
        return int(phase_channel)
    return phase_channel  # None / "auto" / channel-name string


def focus_key(sample_path: Path, fov_index: int, focus_opts: Any,
              phase_channel: Any) -> FocusKey:
    return (
        "focus",
        str(sample_path),
        int(fov_index),
        getattr(focus_opts, "mode", None),
        getattr(focus_opts, "metric", None),
        tuple(getattr(focus_opts, "tile_grid", ()) or ()),
        getattr(focus_opts, "save_zmaps", None),
        getattr(focus_opts, "save_mip", None),
        _phase_part(phase_channel),
    )


def disk_focus_key(sample_path: Path, fov_index: int, phase_channel: Any
                   ) -> FocusKey:
    """Focus key when the focused image comes from disk (Phase 2 path).

    Disk-loaded focus is invariant to focus_opts — it's whatever the
    user already produced — so the key only depends on file + FOV.
    """
    return ("disk_focus", str(sample_path), int(fov_index), _phase_part(phase_channel))


def segment_key(focus_k: FocusKey, segment_opts: Any, phase_channel: Any,
                roi: Optional[tuple[int, int, int, int]] = None
                ) -> SegmentKey:
    return (
        "segment",
        focus_k,
        getattr(segment_opts, "model_type", None),
        getattr(segment_opts, "diameter", None),
        getattr(segment_opts, "flow_threshold", None),
        getattr(segment_opts, "cellprob_threshold", None),
        getattr(segment_opts, "min_size", None),
        getattr(segment_opts, "gpu", None),
        getattr(segment_opts, "pixels_per_um", None),
        _phase_part(phase_channel),
        tuple(roi) if roi is not None else None,
    )


def classify_key(seg_k: SegmentKey, classify_opts: Any) -> ClassifyKey:
    mp = getattr(classify_opts, "model_path", None)
    return (
        "classify",
        seg_k,
        str(mp) if mp else None,
        getattr(classify_opts, "use_rules", None),
        getattr(classify_opts, "confidence_threshold", None),
        tuple(getattr(classify_opts, "keep_classes", ()) or ()),
        getattr(classify_opts, "pixels_per_um", None),
    )


def features_key(seg_or_class_k: tuple, features_opts: Any) -> FeaturesKey:
    # Hash the public option fields of ExtractOpts.
    pieces: list[Any] = ["features", seg_or_class_k]
    for attr in (
        "do_morphology", "do_intensity", "do_midline", "refine_contour",
        "save_crops", "crop_pad", "crop_size",
        "intensity_channels",
    ):
        v = getattr(features_opts, attr, None)
        if isinstance(v, (list, tuple, set)):
            v = tuple(v)
        pieces.append(v)
    return tuple(pieces)


# ---------------------------------------------------------------------------

class PreviewCache:
    """LRU-bounded ``(sample, fov) → CacheEntry`` map."""

    def __init__(self, max_entries: int = 8) -> None:
        self._entries: "OrderedDict[tuple[str, int], CacheEntry]" = OrderedDict()
        self._max = max(1, int(max_entries))

    def get(self, sample_path: Path, fov_index: int) -> CacheEntry:
        k = (str(sample_path), int(fov_index))
        if k in self._entries:
            self._entries.move_to_end(k)
            return self._entries[k]
        e = CacheEntry()
        self._entries[k] = e
        self._enforce_lru()
        return e

    def _enforce_lru(self) -> None:
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
