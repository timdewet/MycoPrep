"""Midline-based morphology for rod-shaped bacterial cells (Phase B).

Replaces ``regionprops.major_axis_length`` / ``feret_diameter_max`` (both
inaccurate for curved Mtb cells) with a skeleton-derived **arc length** and a
per-midline-point **width vector**, plus an optional **gradient-based
sub-pixel contour refinement** that pushes precision past the half-pixel
floor of ``find_contours`` on a binary mask.

Algorithmic structure follows the published MicrobeJ / Oufti / PSICIC
methodology — sub-pixel contour, medial axis, per-vertex normal-snap to
the image-intensity-gradient peak, closed-spline contour smoothing — but
the implementation is rewritten on modern skimage + scipy primitives. No
vendored upstream code.

References
----------
- Ducret, A., Quardokus, E. M. & Brun, Y. V. *MicrobeJ, a tool for high
  throughput bacterial cell detection and quantitative analysis*.
  Nat. Microbiol. 1, 16077 (2016). doi:10.1038/nmicrobiol.2016.77
- Paintdakhi, A. et al. *Oufti: an integrated software package for
  high-accuracy, high-throughput quantitative microscopy analysis*.
  Mol. Microbiol. 99, 767-777 (2016).
- Guberman, J. M. et al. *PSICIC: noise and asymmetry in bacterial
  division revealed by computational image analysis at sub-pixel
  resolution*. PLoS Comput. Biol. 4, e1000233 (2008).
- Inspired by MOMIA (`jzrolling/MOMIA <https://github.com/jzrolling/MOMIA>`_,
  MIT licence).

Per-cell pipeline
-----------------
1. **Skeleton**  ``skimage.morphology.skeletonize`` of the cell's binary mask.
2. **Ordering**  Build a graph of skeleton pixels, find the two longest-path
   endpoints, walk tip→tip to produce an ordered midline polyline. Records
   ``branch_count`` = number of degree>2 nodes (post-division V-shapes).
3. **Pole extension**  Skeletonize stops ~1-2 px short of the true poles. Add
   the distance-transform value at each tip to the chord length so the arc
   length approximates the pole-to-pole measurement.
4. **Widths**  ``2 × distance_transform_edt`` evaluated at each midline
   pixel — exact for the locally-orthogonal width of a rod.

Returned per cell
-----------------
``length_um, width_median_um, width_mean_um, width_max_um, width_min_um,
width_std_um, max_width_position_frac, min_width_position_frac, sinuosity,
branch_count``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import find_contours
from skimage.morphology import skeletonize


@dataclass
class MidlineFeatures:
    length_um: float
    width_median_um: float
    width_mean_um: float
    width_max_um: float
    width_min_um: float
    width_std_um: float
    max_width_position_frac: float
    min_width_position_frac: float
    sinuosity: float
    branch_count: int
    # Sub-pixel contour derivatives — replace pixel-quantised regionprops
    # equivalents for downstream distribution analysis.
    area_um2_subpixel: float
    perimeter_um_subpixel: float


# ──────────────────────────────────────────────────────────────────────────
# Skeleton ordering
# ──────────────────────────────────────────────────────────────────────────


_NEIGHBORS_8 = np.array(
    [(-1, -1), (-1, 0), (-1, 1),
     (0, -1),           (0, 1),
     (1, -1),  (1, 0),  (1, 1)],
    dtype=np.int8,
)


def _skeleton_neighbors(skel: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Build a {pixel: [neighbours]} adjacency dict over the skeleton."""
    h, w = skel.shape
    neigh: dict[tuple[int, int], list[tuple[int, int]]] = {}
    ys, xs = np.where(skel)
    pset = set(zip(ys.tolist(), xs.tolist()))
    for y, x in zip(ys, xs):
        ns = []
        for dy, dx in _NEIGHBORS_8:
            ny, nx = int(y) + int(dy), int(x) + int(dx)
            if (ny, nx) in pset:
                ns.append((ny, nx))
        neigh[(int(y), int(x))] = ns
    return neigh


def _longest_path(neigh: dict) -> list[tuple[int, int]]:
    """Two-BFS: find one endpoint, then the furthest endpoint, then walk back.

    Skeletons may have branches (degree > 2 nodes). The longest tip-to-tip
    path is the canonical midline; remaining branches are short spurs we
    discard. We also accept a single-pixel skeleton (very small cell).
    """
    if not neigh:
        return []
    if len(neigh) == 1:
        return list(neigh.keys())

    def _bfs_furthest(start):
        from collections import deque
        seen = {start: None}
        q = deque([start])
        last = start
        while q:
            u = q.popleft()
            last = u
            for v in neigh[u]:
                if v not in seen:
                    seen[v] = u
                    q.append(v)
        return last, seen

    start = next(iter(neigh))
    a, _ = _bfs_furthest(start)
    b, parents = _bfs_furthest(a)

    path: list[tuple[int, int]] = []
    cur = b
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    return path


def _branch_count(neigh: dict) -> int:
    return sum(1 for _, ns in neigh.items() if len(ns) > 2)


# ──────────────────────────────────────────────────────────────────────────
# Polyline arc length
# ──────────────────────────────────────────────────────────────────────────


def _polyline_length_px(path: list[tuple[int, int]]) -> float:
    """Sum of inter-pixel Euclidean distances along an ordered path."""
    if len(path) < 2:
        return 0.0
    arr = np.asarray(path, dtype=np.float64)
    diffs = np.diff(arr, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


# ──────────────────────────────────────────────────────────────────────────
# Sub-pixel widths via local-normal × subpixel-contour intersection
# ──────────────────────────────────────────────────────────────────────────


def _midline_tangents_normals(midline_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangents and 90°-rotated normals at each midline point."""
    tangents = np.zeros_like(midline_arr, dtype=np.float64)
    tangents[1:-1] = midline_arr[2:] - midline_arr[:-2]
    tangents[0] = midline_arr[1] - midline_arr[0]
    tangents[-1] = midline_arr[-1] - midline_arr[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents = tangents / norms
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return tangents, normals


def _subpixel_contour(cell_mask: np.ndarray) -> Optional[np.ndarray]:
    """Longest sub-pixel contour from ``find_contours`` at level 0.5.

    Returns an ``(N, 2)`` float array of (y, x) points in the original
    (un-padded) coordinate frame. ``None`` if too small or no contour
    found.

    The mask is padded by one row/column of zeros before contouring so the
    contour closes properly even when the cell touches the array edge —
    without that, ``find_contours`` returns an open path going from one
    edge to another and the shoelace area collapses to nonsense.
    """
    padded = np.pad(cell_mask.astype(float), 1, mode="constant", constant_values=0)
    contours = find_contours(padded, 0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    if len(contour) < 4:
        return None
    # Undo the 1-pixel pad so coordinates align with cell_mask's frame
    # (which is what the midline lives in).
    return contour - 1.0


def _refine_contour_against_image(
    contour: np.ndarray,
    gradient_magnitude: np.ndarray,
    search_radius_px: float = 2.0,
    n_samples: int = 17,
    smooth: bool = True,
) -> np.ndarray:
    """Snap each contour vertex to the local intensity-gradient peak along
    its outward normal, then smooth the result with a closed periodic
    cubic spline.

    This is the refinement step used by MicrobeJ / Oufti / PSICIC to reach
    sub-pixel precision. The cellpose binary mask gives us a starting
    contour at half-pixel resolution; refining each vertex toward the
    peak of the **image** intensity gradient pulls the contour onto the
    actual cell-edge transition rather than the mask's pixel-stair edge.

    Parameters
    ----------
    contour
        Initial contour from ``find_contours``, ``(N, 2)`` ``(y, x)`` array.
        Assumed to be in the same coordinate frame as ``gradient_magnitude``.
    gradient_magnitude
        2D array of pre-computed Sobel gradient magnitudes for the cell's
        image channel (typically phase).
    search_radius_px
        Walk along ±this distance on either side of each vertex when looking
        for the gradient peak. Default 2 px is enough to cross a typical
        Mtb cell edge while staying inside the cell.
    n_samples
        Number of points sampled along each normal. 17 → 0.25 px resolution
        for the default 2 px search radius.
    smooth
        If True, fit a closed periodic cubic spline through the snapped
        vertices and resample at the same density. Removes the high-
        frequency vertex jitter that snapping can introduce.
    """
    from scipy.ndimage import map_coordinates

    n = len(contour)
    if n < 4:
        return contour

    # Tangents via central differences with periodic wrap.
    nxt = np.roll(contour, -1, axis=0)
    prv = np.roll(contour, 1, axis=0)
    tangents = nxt - prv
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents = tangents / norms
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    # Build sample grid: shape (N, S, 2) where S is samples-per-normal.
    ts = np.linspace(-search_radius_px, +search_radius_px, n_samples)
    sample_points = contour[:, None, :] + ts[None, :, None] * normals[:, None, :]

    # Bilinear-interpolate the gradient magnitude at each sample point.
    sp_flat = sample_points.reshape(-1, 2).T   # (2, N*S) — (y_coords, x_coords)
    grad_at_samples = map_coordinates(
        gradient_magnitude, sp_flat, order=1, mode="nearest",
    ).reshape(n, n_samples)

    # Peak along each vertex's normal.
    best_idx = np.argmax(grad_at_samples, axis=1)
    best_t = ts[best_idx]
    refined = contour + best_t[:, None] * normals

    if smooth and n >= 4:
        try:
            from scipy.interpolate import splev, splprep
            tck, _ = splprep(
                [refined[:, 1], refined[:, 0]],
                s=float(n) * 0.5, per=True, k=3,
            )
            u_dense = np.linspace(0.0, 1.0, n, endpoint=False)
            x_d, y_d = splev(u_dense, tck)
            refined = np.column_stack([y_d, x_d])
        except Exception:  # noqa: BLE001
            pass

    return refined


def _polygon_area_px(contour: np.ndarray) -> float:
    """Shoelace formula on a closed polygon. ``contour`` need not be
    explicitly closed; we wrap to the first point."""
    if contour is None or len(contour) < 3:
        return 0.0
    y = contour[:, 0]
    x = contour[:, 1]
    return 0.5 * abs(
        float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )


def _polygon_perimeter_px(contour: np.ndarray) -> float:
    """Closed-polygon perimeter from sub-pixel contour points."""
    if contour is None or len(contour) < 2:
        return 0.0
    closed = np.vstack([contour, contour[:1]])
    diffs = np.diff(closed, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def _spline_arc_length_px(midline_arr: np.ndarray) -> float:
    """B-spline-smoothed arc length through the integer-pixel midline path.

    Uses ``scipy.interpolate.splprep`` with a small smoothing parameter so
    the path follows the skeleton's centerline but ignores ±1 px staircase
    noise. Falls back to the polyline length on too-short inputs or
    spline-fit failures.
    """
    n = len(midline_arr)
    if n < 4:
        return _polyline_length_px([tuple(p) for p in midline_arr])
    try:
        from scipy.interpolate import splev, splprep
        # k=3 cubic spline; small s lets it smooth out 1-px wobble.
        k = min(3, n - 1)
        s_param = max(0.5, n * 0.5)
        tck, _ = splprep(
            [midline_arr[:, 1].astype(float), midline_arr[:, 0].astype(float)],
            s=s_param, k=k,
        )
        u_dense = np.linspace(0.0, 1.0, max(200, n * 10))
        x_d, y_d = splev(u_dense, tck)
        diffs = np.diff(np.column_stack([y_d, x_d]), axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum())
    except Exception:  # noqa: BLE001
        return _polyline_length_px([tuple(p) for p in midline_arr])


def _subpixel_widths_px(
    midline_arr: np.ndarray,
    contour: np.ndarray,
) -> Optional[np.ndarray]:
    """Width at each midline point via line-segment intersection of the
    local normal with the supplied sub-pixel contour.

    Returns one width value per midline point in pixel space. Falls back to
    0 for any midline point whose normal misses the contour on either side —
    callers are expected to substitute ``2 × dt`` at those points.
    """
    if contour is None or len(contour) < 4:
        return None

    _, normals = _midline_tangents_normals(midline_arr.astype(np.float64))

    a = contour[:-1]                       # (N, 2)
    b = contour[1:]
    bm_a = b - a                           # (N, 2)

    widths = np.zeros(len(midline_arr), dtype=np.float64)
    for i, (p, n) in enumerate(zip(midline_arr.astype(np.float64), normals)):
        # Solve  p + v·n = a + u·(b - a)  for each segment.
        # 2D cross convention: (x, y) × (s, t) ≡ x·t − y·s.
        det = bm_a[:, 0] * n[1] - bm_a[:, 1] * n[0]   # (b - a) × n
        with np.errstate(divide="ignore", invalid="ignore"):
            valid = np.abs(det) > 1e-9
            a_minus_p = a - p
            v = (bm_a[:, 0] * a_minus_p[:, 1] - bm_a[:, 1] * a_minus_p[:, 0]) / det
            u = (n[0] * a_minus_p[:, 1] - n[1] * a_minus_p[:, 0]) / det
        valid &= (u >= 0.0) & (u <= 1.0)
        v_valid = v[valid]
        if v_valid.size == 0:
            continue
        v_pos = v_valid[v_valid > 0]
        v_neg = v_valid[v_valid < 0]
        if v_pos.size and v_neg.size:
            widths[i] = float(v_pos.min()) - float(v_neg.max())
    return widths


# ──────────────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────────────


def midline_features(
    cell_mask: np.ndarray,
    pixels_per_um: float,
    refinement_image: Optional[np.ndarray] = None,
    refine_search_radius_px: float = 2.0,
) -> Optional[MidlineFeatures]:
    """Compute midline-based morphology for a single cell.

    Parameters
    ----------
    cell_mask
        2D boolean (or 0/non-zero) mask containing exactly one connected
        component — this cell. Caller is responsible for cropping to the
        cell's bounding box.
    pixels_per_um
        Pixel scale used to convert px-space measurements to micrometres.
    refinement_image
        Optional 2D **gradient-magnitude** image (same shape as
        ``cell_mask``) used to refine the sub-pixel contour by snapping
        each vertex toward the peak intensity-gradient direction along its
        normal. Improves area/perimeter/width precision past the half-pixel
        floor imposed by ``find_contours`` on a binary mask. The MicrobeJ
        paper's ~50 nm precision benchmark depends on this step.
    refine_search_radius_px
        ± distance, in pixels, that each contour vertex is allowed to move
        during refinement.

    Returns
    -------
    A ``MidlineFeatures`` instance, or ``None`` if the cell is too small /
    pathological for midline construction (in which case callers should
    fall back to NaN columns).
    """
    binary = cell_mask.astype(bool, copy=False)
    if binary.sum() < 4:
        return None

    px_per_um = 1.0 / float(pixels_per_um)
    dt = distance_transform_edt(binary)
    skel = skeletonize(binary)

    if skel.sum() == 0:
        return None

    # Sub-pixel contour computed once; reused for area, perimeter, and
    # widths so we never call ``find_contours`` more than once per cell.
    contour = _subpixel_contour(binary)
    if (
        contour is not None
        and refinement_image is not None
        and refinement_image.shape == cell_mask.shape
    ):
        contour = _refine_contour_against_image(
            contour, refinement_image,
            search_radius_px=refine_search_radius_px,
        )
    area_um2_subpixel = _polygon_area_px(contour) * (px_per_um ** 2) if contour is not None else 0.0
    perimeter_um_subpixel = _polygon_perimeter_px(contour) * px_per_um if contour is not None else 0.0

    neigh = _skeleton_neighbors(skel)
    midline = _longest_path(neigh)
    if len(midline) < 2:
        # Single-pixel skeleton — fall back to using the distance transform
        # peak as the only "midline point". Length ≈ 2× max distance, which
        # is a chunky rod's diameter. Better than nothing.
        peak_idx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        d = float(dt[peak_idx])
        return MidlineFeatures(
            length_um=2 * d * px_per_um,
            width_median_um=2 * d * px_per_um,
            width_mean_um=2 * d * px_per_um,
            width_max_um=2 * d * px_per_um,
            width_min_um=2 * d * px_per_um,
            width_std_um=0.0,
            max_width_position_frac=0.5,
            min_width_position_frac=0.5,
            sinuosity=1.0,
            branch_count=0,
            area_um2_subpixel=area_um2_subpixel,
            perimeter_um_subpixel=perimeter_um_subpixel,
        )

    midline_arr = np.asarray(midline, dtype=np.int32)
    # Spline-smoothed midline arc length: integrates a cubic-B-spline fit
    # to the integer-pixel skeleton path, smoothing out the ±1 px staircase
    # noise that quantises a raw polyline sum into {1, √2}-multiples.
    chord_length_px = _spline_arc_length_px(midline_arr)

    # Pole extension: skeleton tips sit 1–2 px short of the cell's true
    # pole. The local distance-transform value at each tip is roughly the
    # remaining radius; add it at both ends to approximate the true
    # pole-to-pole midline length.
    head = midline[0]
    tail = midline[-1]
    pole_correction_px = float(dt[head] + dt[tail])
    length_px = chord_length_px + pole_correction_px

    # Widths: line-segment intersection of the local midline normal with
    # the sub-pixel contour. Falls back to ``2 × dt`` (continuous in 1D
    # but quantised because dt = √integer) at points where the normal
    # misses the contour on either side.
    dt_widths_px = 2.0 * dt[midline_arr[:, 0], midline_arr[:, 1]].astype(np.float64)
    sub_widths_px = _subpixel_widths_px(midline_arr, contour) if contour is not None else None
    if sub_widths_px is not None:
        widths_px = np.where(sub_widths_px > 0, sub_widths_px, dt_widths_px)
    else:
        widths_px = dt_widths_px
    widths_um = widths_px * px_per_um

    # Densify the width vector by linear interpolation along the midline.
    # The raw vector has only N ≈ 10–20 samples for a typical Mtb cell, so
    # quantile-based statistics (median, max, min) end up landing on one of
    # those few specific values — very few unique medians across thousands
    # of cells. A 200-sample dense vector spreads the mass continuously and
    # makes per-cell distributions usable for downstream analysis.
    if len(widths_um) >= 2:
        t_sparse = np.linspace(0.0, 1.0, len(widths_um))
        t_dense = np.linspace(0.0, 1.0, 200)
        dense_widths_um = np.interp(t_dense, t_sparse, widths_um)
    else:
        dense_widths_um = widths_um

    # Sinuosity: skeleton arc length / skeleton-tip chord. Both are measured
    # WITHOUT the pole-extension correction so the ratio reflects body
    # curvature alone — round poles shouldn't make a straight cell look
    # bent. ≥1.0 by construction.
    head_arr = np.asarray(head, dtype=np.float64)
    tail_arr = np.asarray(tail, dtype=np.float64)
    chord_px = float(np.linalg.norm(head_arr - tail_arr))
    length_um = length_px * px_per_um
    if chord_px > 0:
        sinuosity = max(1.0, chord_length_px / chord_px)
    else:
        sinuosity = 1.0

    # Position fraction: argmax/argmin in the DENSE width vector, folded so
    # that 0 = pole, 0.5 = mid-cell. Dense interpolation gives 200 candidate
    # positions instead of N≈10, removing the worst quantisation in the
    # original argmax-of-small-array.
    n = len(dense_widths_um)
    arg_max = int(np.argmax(dense_widths_um))
    arg_min = int(np.argmin(dense_widths_um))
    frac_max = arg_max / max(1, n - 1)
    frac_min = arg_min / max(1, n - 1)
    if frac_max > 0.5:
        frac_max = 1.0 - frac_max
    if frac_min > 0.5:
        frac_min = 1.0 - frac_min

    return MidlineFeatures(
        length_um=length_um,
        width_median_um=float(np.median(dense_widths_um)),
        width_mean_um=float(np.mean(dense_widths_um)),
        width_max_um=float(np.max(dense_widths_um)),
        width_min_um=float(np.min(dense_widths_um)),
        width_std_um=float(np.std(dense_widths_um)),
        max_width_position_frac=float(frac_max),
        min_width_position_frac=float(frac_min),
        sinuosity=float(sinuosity),
        branch_count=int(_branch_count(neigh)),
        area_um2_subpixel=float(area_um2_subpixel),
        perimeter_um_subpixel=float(perimeter_um_subpixel),
    )
