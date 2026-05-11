"""Tile-grid parsing, slicing, and stitch geometry.

Covers ``mycoprep.core.focus.tiling``: ``parse_grid``, ``iter_tile_slices``,
and ``assemble_tiled_plane``. Off-by-one errors in the last-tile remainder
or in stitching would silently corrupt every tile-focus output.
"""

from __future__ import annotations

import numpy as np
import pytest

from mycoprep.core.focus.tiling import (
    assemble_tiled_plane,
    iter_tile_slices,
    parse_grid,
)


class TestParseGrid:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("2x2", (2, 2)),
            ("3x4", (3, 4)),
            ("3X4", (3, 4)),
            ("1x1", (1, 1)),
            ("  2x3  ", (2, 3)),
            (" 10 x 20 ", (10, 20)),
        ],
    )
    def test_valid_specs(self, spec, expected):
        assert parse_grid(spec) == expected

    @pytest.mark.parametrize("spec", ["", "2", "2x", "x2", "2y2", "abc", "0x2", "2x0"])
    def test_rejects_invalid_specs(self, spec):
        with pytest.raises(ValueError):
            parse_grid(spec)

    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            parse_grid(22)  # type: ignore[arg-type]


class TestIterTileSlices:
    def test_evenly_divisible(self):
        slices = list(iter_tile_slices((100, 200), (2, 4)))
        assert len(slices) == 8
        # Tiles tile contiguously and cover the entire frame with no overlap.
        coords = [c for _, c in slices]
        assert sorted(coords) == [(i, j) for i in range(2) for j in range(4)]
        for (ys, xs), _ in slices:
            assert ys.stop - ys.start == 50
            assert xs.stop - xs.start == 50

    def test_last_tile_absorbs_remainder(self):
        # 101x203 / 2x4: first 1x3 tile is 50x50, last col is 50+3 wide, last row is 50+1 tall.
        slices = list(iter_tile_slices((101, 203), (2, 4)))
        spans = {coord: (ys, xs) for (ys, xs), coord in slices}

        first_y, first_x = spans[(0, 0)]
        last_row_y, _ = spans[(1, 0)]
        _, last_col_x = spans[(0, 3)]

        assert first_y == slice(0, 50)
        assert first_x == slice(0, 50)
        # Bottom row picks up the extra row.
        assert last_row_y == slice(50, 101)
        # Right column picks up the extra columns.
        assert last_col_x == slice(150, 203)

    def test_full_frame_coverage(self):
        h, w = 73, 89
        mask = np.zeros((h, w), dtype=np.int32)
        for (ys, xs), _ in iter_tile_slices((h, w), (3, 5)):
            mask[ys, xs] += 1
        # Every pixel covered exactly once.
        assert (mask == 1).all()

    def test_rejects_grid_larger_than_image(self):
        with pytest.raises(ValueError):
            list(iter_tile_slices((4, 4), (5, 5)))


class TestAssembleTiledPlane:
    def test_stitch_picks_correct_z_per_tile(self):
        # Z stack of 3 slices, each filled with a constant; the picked slice
        # determines what value appears in each tile region of the stitched plane.
        z = np.zeros((3, 20, 20), dtype=np.uint16)
        z[0] = 10
        z[1] = 20
        z[2] = 30

        z_per_tile = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 0,
        }
        out = assemble_tiled_plane(z, z_per_tile, (2, 2))

        assert out.shape == (20, 20)
        assert out.dtype == np.uint16
        assert (out[0:10, 0:10] == 10).all()
        assert (out[0:10, 10:20] == 20).all()
        assert (out[10:20, 0:10] == 30).all()
        assert (out[10:20, 10:20] == 10).all()

    def test_rejects_out_of_range_z(self):
        z = np.zeros((2, 4, 4), dtype=np.uint16)
        with pytest.raises(IndexError):
            assemble_tiled_plane(z, {(0, 0): 5}, (1, 1))

    def test_missing_tile_raises(self):
        z = np.zeros((2, 4, 4), dtype=np.uint16)
        with pytest.raises(KeyError):
            assemble_tiled_plane(z, {(0, 0): 0}, (2, 2))

    def test_rejects_non_3d(self):
        with pytest.raises(ValueError):
            assemble_tiled_plane(np.zeros((4, 4)), {(0, 0): 0}, (1, 1))
