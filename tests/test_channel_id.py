"""Phase-vs-fluorescence channel auto-detection.

Covers ``mycoprep.core.focus.channel_id``: ``_stats``,
``detect_phase_channel_multi``, and ``resolve_phase_channel``. Picking
the wrong phase channel breaks every downstream stage, and the decision
is made silently from metadata, so it's worth pinning the behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from mycoprep.core.focus.channel_id import (
    _stats,
    channel_stats,
    detect_phase_channel,
    detect_phase_channel_multi,
    resolve_phase_channel,
)


def _phase_like(rng, shape=(64, 64)):
    """Symmetric histogram around a mid-grey background — small |skew|."""
    return rng.normal(loc=1000.0, scale=50.0, size=shape).astype(np.float32)


def _fluor_like(rng, shape=(64, 64)):
    """Mostly dark with a handful of bright spots — strong right skew."""
    img = rng.normal(loc=10.0, scale=5.0, size=shape).astype(np.float32)
    # Inject a few very bright pixels (fluorescent puncta).
    ys = rng.integers(0, shape[0], size=20)
    xs = rng.integers(0, shape[1], size=20)
    img[ys, xs] += 5000.0
    return img


class TestStats:
    def test_phase_like_has_low_abs_skew(self, rng):
        plane = _phase_like(rng)
        s = _stats(plane, index=0, name="phase")
        assert abs(s.skewness) < 1.0

    def test_fluor_like_has_high_abs_skew(self, rng):
        plane = _fluor_like(rng)
        s = _stats(plane, index=1, name="rfp")
        assert abs(s.skewness) > 2.0

    def test_phase_likeness_orders_phase_above_fluor(self, rng):
        sp = _stats(_phase_like(rng), 0, "phase")
        sf = _stats(_fluor_like(rng), 1, "rfp")
        assert sp.phase_likeness > sf.phase_likeness

    def test_skew_nan_is_clamped(self):
        # Constant plane → undefined skew → coerced to 0 (not NaN).
        plane = np.full((16, 16), 500.0, dtype=np.float32)
        s = _stats(plane, 0, "c0")
        assert s.skewness == 0.0


class TestDetectPhaseChannelMulti:
    def test_picks_phase_among_two_channels(self, rng):
        # (Z=3, C=2, Y=64, X=64). Channel 1 is phase, channel 0 is fluorescence.
        z, y, x = 3, 64, 64
        arr = np.zeros((z, 2, y, x), dtype=np.float32)
        for zi in range(z):
            arr[zi, 0] = _fluor_like(rng, (y, x))
            arr[zi, 1] = _phase_like(rng, (y, x))
        assert detect_phase_channel_multi([arr]) == 1

    def test_picks_phase_among_three_channels(self, rng):
        z, y, x = 3, 64, 64
        arr = np.zeros((z, 3, y, x), dtype=np.float32)
        for zi in range(z):
            arr[zi, 0] = _fluor_like(rng, (y, x))
            arr[zi, 1] = _fluor_like(rng, (y, x))
            arr[zi, 2] = _phase_like(rng, (y, x))
        assert detect_phase_channel_multi([arr]) == 2

    def test_averages_across_multiple_scenes(self, rng):
        # Two scenes, both with channel 0 = phase. Per-scene noise should
        # not flip the answer.
        z, y, x = 3, 64, 64
        scenes = []
        for _ in range(2):
            arr = np.zeros((z, 2, y, x), dtype=np.float32)
            for zi in range(z):
                arr[zi, 0] = _phase_like(rng, (y, x))
                arr[zi, 1] = _fluor_like(rng, (y, x))
            scenes.append(arr)
        assert detect_phase_channel_multi(scenes) == 0

    def test_single_scene_helper(self, rng):
        z, y, x = 3, 64, 64
        arr = np.zeros((z, 2, y, x), dtype=np.float32)
        for zi in range(z):
            arr[zi, 0] = _phase_like(rng, (y, x))
            arr[zi, 1] = _fluor_like(rng, (y, x))
        assert detect_phase_channel(arr) == 0

    def test_rejects_inconsistent_channel_counts(self, rng):
        a = np.zeros((1, 2, 8, 8), dtype=np.float32)
        b = np.zeros((1, 3, 8, 8), dtype=np.float32)
        with pytest.raises(ValueError):
            detect_phase_channel_multi([a, b])

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            detect_phase_channel_multi([])


class TestChannelStats:
    def test_returns_one_entry_per_channel(self, rng):
        arr = np.zeros((1, 3, 16, 16), dtype=np.float32)
        for c in range(3):
            arr[0, c] = _phase_like(rng, (16, 16))
        stats = channel_stats(arr, channel_names=["a", "b", "c"])
        assert [s.name for s in stats] == ["a", "b", "c"]
        assert [s.index for s in stats] == [0, 1, 2]

    def test_default_channel_names(self, rng):
        arr = np.zeros((1, 2, 16, 16), dtype=np.float32)
        stats = channel_stats(arr)
        assert [s.name for s in stats] == ["C0", "C1"]

    def test_rejects_non_4d(self):
        with pytest.raises(ValueError):
            channel_stats(np.zeros((3, 16, 16), dtype=np.float32))


class TestResolvePhaseChannel:
    def test_int_override_is_used_directly(self, rng):
        arr = np.zeros((1, 3, 8, 8), dtype=np.float32)
        assert resolve_phase_channel(arr, ["a", "b", "c"], 2) == 2

    def test_int_override_out_of_range(self):
        arr = np.zeros((1, 3, 8, 8), dtype=np.float32)
        with pytest.raises(ValueError):
            resolve_phase_channel(arr, ["a", "b", "c"], 7)

    def test_string_override_case_insensitive(self):
        arr = np.zeros((1, 3, 8, 8), dtype=np.float32)
        assert resolve_phase_channel(arr, ["Phase", "GFP", "RFP"], "phase") == 0
        assert resolve_phase_channel(arr, ["Phase", "GFP", "RFP"], "RFP") == 2

    def test_unknown_string_override_raises(self):
        arr = np.zeros((1, 3, 8, 8), dtype=np.float32)
        with pytest.raises(ValueError):
            resolve_phase_channel(arr, ["Phase", "GFP", "RFP"], "DAPI")

    def test_none_falls_through_to_detection(self, rng):
        z, y, x = 3, 64, 64
        arr = np.zeros((z, 2, y, x), dtype=np.float32)
        for zi in range(z):
            arr[zi, 0] = _phase_like(rng, (y, x))
            arr[zi, 1] = _fluor_like(rng, (y, x))
        assert resolve_phase_channel(arr, ["phase", "rfp"], None) == 0
