"""Tile-based focus picking.

Splits a Z-stack into N×M rectangular tiles, picks the best Z per tile using
the validated ``focus.score_stack`` + ``focus.pick_best_slice`` path, and
stitches the per-tile picks back into a single (Y, X) image. Used for FOVs
where only part of the frame is in focus on any given Z slice — different
regions can be sharpest on different slices, and the per-FOV picker leaves
most of the frame defocused.
"""

from __future__ import annotations

import re
from typing import Iterator

import numpy as np

from . import focus

GridSpec = tuple[int, int]
TileSlices = tuple[slice, slice]
TileCoord = tuple[int, int]

_GRID_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")


def parse_grid(spec: str) -> GridSpec:
    """Parse a tile grid spec like ``"2x2"`` or ``"3X4"`` into ``(rows, cols)``."""
    if not isinstance(spec, str):
        raise ValueError(f"grid spec must be a string, got {type(spec).__name__}")
    m = _GRID_RE.match(spec)
    if not m:
        raise ValueError(
            f"invalid tile grid {spec!r}; expected NxM (e.g. '2x2', '3x4')"
        )
    rows, cols = int(m.group(1)), int(m.group(2))
    if rows < 1 or cols < 1:
        raise ValueError(f"tile grid must have positive dimensions, got {rows}x{cols}")
    return rows, cols


def iter_tile_slices(
    shape_yx: tuple[int, int], grid: GridSpec
) -> Iterator[tuple[TileSlices, TileCoord]]:
    """Yield ``((y_slice, x_slice), (tile_i, tile_j))`` for each tile.

    The last row/column tile absorbs any remainder so non-divisible shapes
    still tile cleanly without losing pixels.
    """
    h, w = shape_yx
    rows, cols = grid
    if rows < 1 or cols < 1:
        raise ValueError(f"grid must be positive, got {grid}")
    tile_h = h // rows
    tile_w = w // cols
    if tile_h < 1 or tile_w < 1:
        raise ValueError(
            f"shape {shape_yx} too small for grid {grid} (tile would be "
            f"{tile_h}x{tile_w} px)"
        )
    for i in range(rows):
        y0 = i * tile_h
        y1 = h if i == rows - 1 else (i + 1) * tile_h
        for j in range(cols):
            x0 = j * tile_w
            x1 = w if j == cols - 1 else (j + 1) * tile_w
            yield (slice(y0, y1), slice(x0, x1)), (i, j)


def pick_best_z_per_tile(
    stack_zyx: np.ndarray,
    grid: GridSpec,
    metric: str = focus.DEFAULT_METRIC,
) -> dict[TileCoord, tuple[int, dict[str, np.ndarray]]]:
    """Return ``{(i, j): (chosen_z, scores_dict)}`` for each tile.

    Each tile is scored independently so per-region focus differences are
    captured. Reuses the validated ``focus.score_stack`` (mask on by default)
    and ``focus.pick_best_slice`` so the proven `vollath_f4 + mask` path
    applies per tile.
    """
    if stack_zyx.ndim != 3:
        raise ValueError(f"expected (Z, Y, X) stack, got shape {stack_zyx.shape}")
    out: dict[TileCoord, tuple[int, dict[str, np.ndarray]]] = {}
    for (ys, xs), (i, j) in iter_tile_slices(stack_zyx.shape[1:], grid):
        sub = stack_zyx[:, ys, xs]
        scores = focus.score_stack(sub)
        chosen_z = focus.pick_best_slice(scores, metric=metric)
        out[(i, j)] = (chosen_z, scores)
    return out


def assemble_tiled_plane(
    stack_zyx: np.ndarray,
    z_per_tile: dict[TileCoord, int],
    grid: GridSpec,
) -> np.ndarray:
    """Stitch a single (Y, X) plane by copying each tile from its picked Z.

    ``stack_zyx`` is one channel's worth of data (Z, Y, X). ``z_per_tile``
    maps each tile coordinate to the Z index to copy from. Channels stay
    spatially coherent because all channels share the same phase-derived
    tile-Z map.
    """
    if stack_zyx.ndim != 3:
        raise ValueError(f"expected (Z, Y, X) stack, got shape {stack_zyx.shape}")
    n_z = stack_zyx.shape[0]
    out = np.empty(stack_zyx.shape[1:], dtype=stack_zyx.dtype)
    for (ys, xs), coord in iter_tile_slices(stack_zyx.shape[1:], grid):
        if coord not in z_per_tile:
            raise KeyError(f"missing tile {coord} in z_per_tile")
        z = z_per_tile[coord]
        if not 0 <= z < n_z:
            raise IndexError(f"tile {coord} z={z} outside stack of size {n_z}")
        out[ys, xs] = stack_zyx[z, ys, xs]
    return out
