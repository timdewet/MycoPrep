"""Rule-based cell quality filters and crop extraction.

Covers ``mycoprep.core.cell_quality_classifier``: ``_normalise_area``,
``detect_edge_cells``, ``detect_debris_by_area``, ``detect_large_clumps``,
and the geometry-heavy ``extract_cell_crop``. These run before the CNN
and silently shape what the model sees, so threshold off-by-one and
coordinate bugs here are particularly damaging.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mycoprep.core.cell_quality_classifier import (
    CROP_SIZE,
    EDGE_MARGIN,
    _AREA_LOG_MAX,
    _AREA_LOG_MIN,
    _normalise_area,
    detect_debris_by_area,
    detect_edge_cells,
    detect_large_clumps,
    extract_cell_crop,
)


class TestNormaliseArea:
    def test_clipped_below_min(self):
        assert _normalise_area(0) == 0.0
        assert _normalise_area(1) == 0.0
        assert _normalise_area(10) == pytest.approx(0.0)

    def test_clipped_above_max(self):
        assert _normalise_area(10_000) == pytest.approx(1.0)
        assert _normalise_area(1_000_000) == 1.0

    def test_geometric_midpoint_is_half(self):
        # log-scale midpoint: sqrt(10 * 10_000) = 316.228...
        midpoint = math.exp((_AREA_LOG_MIN + _AREA_LOG_MAX) / 2)
        assert _normalise_area(midpoint) == pytest.approx(0.5, abs=1e-6)

    def test_monotonic(self):
        values = [_normalise_area(a) for a in (15, 50, 200, 1000, 5000)]
        assert values == sorted(values)


class TestDetectEdgeCells:
    def test_detects_cells_touching_top_and_right(self, edge_touching_mask):
        edges = detect_edge_cells(edge_touching_mask, margin=EDGE_MARGIN)
        assert edges == {2, 3}

    def test_interior_cells_are_not_edges(self, labeled_mask_3cells):
        # All three fixture cells are at least 5 px from any border.
        assert detect_edge_cells(labeled_mask_3cells, margin=EDGE_MARGIN) == set()

    def test_margin_zero_only_flags_cells_at_exact_border(self):
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[0, 5] = 1     # touches top row
        mask[2:5, 2:5] = 2  # interior
        edges = detect_edge_cells(mask, margin=0)
        # margin=0 → ys.min() < 0 is False; only the y.max() >= h - 0 == 10
        # check catches a cell at y == 9. Top row (y=0) does NOT trigger.
        # This pins the existing asymmetric "< margin" / ">= h - margin" rule.
        assert 1 not in edges
        assert 2 not in edges

    def test_large_margin_flags_more(self):
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[5:7, 5:7] = 1
        assert 1 not in detect_edge_cells(mask, margin=2)
        # margin=6 means ys.min()==5 < 6 → flagged.
        assert 1 in detect_edge_cells(mask, margin=6)


class TestDetectDebrisByArea:
    def test_below_threshold_flagged(self):
        mask = np.zeros((40, 40), dtype=np.int32)
        # 2x2 = 4 px; at 1 px/µm that's 4 µm² — above default 0.3 µm² floor.
        # Force a tiny min_area to make it debris.
        mask[5:7, 5:7] = 1
        debris = detect_debris_by_area(mask, pixels_per_um=1.0, min_area_um2=10.0)
        assert debris == {1}

    def test_above_threshold_not_flagged(self):
        mask = np.zeros((40, 40), dtype=np.int32)
        mask[5:25, 5:25] = 1  # 400 px
        debris = detect_debris_by_area(mask, pixels_per_um=1.0, min_area_um2=10.0)
        assert debris == set()

    def test_pixel_scaling(self):
        # At 2 px/µm, 100 px == 25 µm². With min_area_um2=20 that's not debris;
        # with min_area_um2=30 it is.
        mask = np.zeros((40, 40), dtype=np.int32)
        mask[0:10, 0:10] = 1  # 100 px
        assert detect_debris_by_area(mask, pixels_per_um=2.0, min_area_um2=20.0) == set()
        assert detect_debris_by_area(mask, pixels_per_um=2.0, min_area_um2=30.0) == {1}


class TestDetectLargeClumps:
    def test_above_threshold_flagged(self):
        mask = np.zeros((40, 40), dtype=np.int32)
        mask[5:25, 5:25] = 1  # 400 px
        assert detect_large_clumps(mask, pixels_per_um=1.0, max_area_um2=100.0) == {1}

    def test_below_threshold_not_flagged(self):
        mask = np.zeros((40, 40), dtype=np.int32)
        mask[5:25, 5:25] = 1
        assert detect_large_clumps(mask, pixels_per_um=1.0, max_area_um2=500.0) == set()


class TestExtractCellCrop:
    def _build_three_channel_image(self, mask):
        # 3 imaging channels filled with distinct constants so we can verify
        # which channel ends up where post-crop.
        h, w = mask.shape
        img = np.zeros((3, h, w), dtype=np.float32)
        img[0] = 100.0
        img[1] = 200.0
        img[2] = 300.0
        return img

    def test_shape_and_channel_count_default(self, labeled_mask_3cells):
        img = self._build_three_channel_image(labeled_mask_3cells)
        crop, bbox = extract_cell_crop(img, labeled_mask_3cells, cell_label=1)
        # 3 imaging channels + 1 mask channel.
        assert crop is not None
        assert crop.shape == (4, CROP_SIZE, CROP_SIZE)
        assert crop.dtype == np.float32
        assert bbox is not None
        assert len(bbox) == 4

    def test_phase_channel_mode_returns_two_channels(self, labeled_mask_3cells):
        img = self._build_three_channel_image(labeled_mask_3cells)
        crop, _ = extract_cell_crop(
            img, labeled_mask_3cells, cell_label=1, phase_channel=1
        )
        # 1 phase imaging channel + 1 mask channel.
        assert crop.shape == (2, CROP_SIZE, CROP_SIZE)

    def test_returns_none_when_label_absent(self, labeled_mask_3cells):
        img = self._build_three_channel_image(labeled_mask_3cells)
        crop, bbox = extract_cell_crop(img, labeled_mask_3cells, cell_label=999)
        assert crop is None
        assert bbox is None

    def test_mask_channel_is_binary_after_resize(self, labeled_mask_3cells):
        img = self._build_three_channel_image(labeled_mask_3cells)
        crop, _ = extract_cell_crop(img, labeled_mask_3cells, cell_label=2)
        mask = crop[-1]
        assert set(np.unique(mask).tolist()).issubset({0.0, 1.0})
        # The cell should occupy a non-trivial fraction of the cropped frame.
        assert mask.sum() > 0

    def test_constant_channel_normalises_to_zero(self, labeled_mask_3cells):
        img = self._build_three_channel_image(labeled_mask_3cells)
        crop, _ = extract_cell_crop(img, labeled_mask_3cells, cell_label=1)
        # Every imaging channel is filled with a single constant value, so the
        # cmax == cmin branch should set it to all zeros.
        for c in range(crop.shape[0] - 1):
            assert (crop[c] == 0.0).all(), f"channel {c} not zeroed"

    def test_non_constant_channel_normalises_to_unit_range(self, labeled_mask_3cells):
        # Make channel 0 a gradient so cmax > cmin → normal min-max scaling.
        h, w = labeled_mask_3cells.shape
        img = np.zeros((1, h, w), dtype=np.float32)
        img[0] = np.linspace(0, 1000, h * w, dtype=np.float32).reshape(h, w)
        crop, _ = extract_cell_crop(img, labeled_mask_3cells, cell_label=2)
        # Channel 0 (imaging): clamp inspected after normalisation.
        assert crop[0].min() >= 0.0
        assert crop[0].max() <= 1.0
        # Not constant → should span more than zero.
        assert crop[0].max() > crop[0].min()

    def test_bbox_respects_image_bounds(self, edge_touching_mask):
        img = self._build_three_channel_image(edge_touching_mask)
        _, bbox = extract_cell_crop(img, edge_touching_mask, cell_label=2)
        y_min, y_max, x_min, x_max = bbox
        h, w = edge_touching_mask.shape
        assert 0 <= y_min < y_max <= h
        assert 0 <= x_min < x_max <= w
