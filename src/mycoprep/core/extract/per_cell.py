"""Per-cell morphology + per-channel intensity table for one FOV.

Phase A used ``skimage.measure.regionprops_table`` exclusively. Phase B
adds midline-derived columns (``length_um``, ``width_*``, ``sinuosity``,
``branch_count``) computed by ``_midline.midline_features`` — these are
the headline differentiator over plain regionprops for curved Mtb cells.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from . import _midline


# Properties pulled in one regionprops call. ``major_axis_length`` and
# ``feret_diameter_max`` are emitted but documented as crude proxies for
# length — Phase B will add a midline-based ``length_um``.
_BASE_PROPS = (
    "label",
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "equivalent_diameter_area",
    "orientation",
    "major_axis_length",
    "minor_axis_length",
    "feret_diameter_max",
    "centroid",
    "bbox",
)

_INTENSITY_PROPS = (
    "label",
    "intensity_mean",
    "intensity_min",
    "intensity_max",
)


def _intensity_extras(image_channel: np.ndarray, labeled_mask: np.ndarray) -> dict[int, dict[str, float]]:
    """Per-cell median, p95, sum — not in regionprops_table directly."""
    out: dict[int, dict[str, float]] = {}
    n_labels = int(labeled_mask.max())
    if n_labels == 0:
        return out
    flat_lab = labeled_mask.ravel()
    flat_int = image_channel.ravel().astype(np.float64, copy=False)
    order = np.argsort(flat_lab, kind="stable")
    sorted_lab = flat_lab[order]
    sorted_int = flat_int[order]
    edges = np.searchsorted(sorted_lab, np.arange(1, n_labels + 1, dtype=sorted_lab.dtype), side="left")
    edges = np.append(edges, np.searchsorted(sorted_lab, n_labels, side="right"))
    for label in range(1, n_labels + 1):
        lo = int(edges[label - 1])
        hi = int(edges[label])
        if hi <= lo:
            continue
        vals = sorted_int[lo:hi]
        out[label] = {
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
            "total": float(vals.sum()),
        }
    return out


def extract_fov_features(
    image_channels: np.ndarray,
    labeled_mask: np.ndarray,
    channel_names: list[str],
    pixels_per_um: float,
    *,
    run_id: str,
    well: str,
    fov_index: int,
    source_czi: str,
    plate_acquisition_datetime: Optional[str],
    fov_acquisition_time: Optional[str],
    intensity_channels: Optional[Iterable[int]] = None,
    midline_features: bool = True,
    refinement_channel: Optional[int] = None,
    refine_search_radius_px: float = 2.0,
) -> pd.DataFrame:
    """Build a per-cell DataFrame for one FOV.

    Parameters
    ----------
    image_channels
        ``(C, Y, X)`` array. The mask channel must NOT be included.
    labeled_mask
        ``(Y, X)`` int32 with 0=background, 1..N=cell labels.
    channel_names
        Length-C list of human-readable names (e.g. ``["Phase", "mCherry"]``).
    pixels_per_um
        Pixel scale, used to convert area/perimeter/length to micrometres.
    intensity_channels
        Subset of channel indices to compute intensity stats for; ``None`` =
        all channels.
    """
    if labeled_mask.max() == 0:
        return pd.DataFrame()

    px2_per_um2 = 1.0 / (pixels_per_um ** 2)
    px_per_um = 1.0 / pixels_per_um

    base = regionprops_table(labeled_mask, properties=_BASE_PROPS)
    n_cells = len(base["label"])

    cell_ids = np.asarray(base["label"], dtype=np.int64)
    df = pd.DataFrame({
        "cell_id": cell_ids,
        "area_um2": base["area"] * px2_per_um2,
        "area_px": base["area"].astype(np.int64),
        "perimeter_um": base["perimeter"] * px_per_um,
        "eccentricity": base["eccentricity"],
        "solidity": base["solidity"],
        "equivalent_diameter_um": base["equivalent_diameter_area"] * px_per_um,
        "orientation_rad": base["orientation"],
        # Crude length proxies — Phase B adds midline-based length_um.
        "major_axis_length_um": base["major_axis_length"] * px_per_um,
        "minor_axis_length_um": base["minor_axis_length"] * px_per_um,
        "feret_diameter_max_um": base["feret_diameter_max"] * px_per_um,
        "centroid_y": base["centroid-0"],
        "centroid_x": base["centroid-1"],
        "bbox_y0": base["bbox-0"].astype(np.int32),
        "bbox_x0": base["bbox-1"].astype(np.int32),
        "bbox_y1": base["bbox-2"].astype(np.int32),
        "bbox_x1": base["bbox-3"].astype(np.int32),
    })

    # Per-channel intensity. regionprops_table accepts a 2D image; we call
    # it per-channel so each channel becomes its own labelled set of columns.
    chans = list(intensity_channels) if intensity_channels is not None else list(range(len(channel_names)))
    for ch in chans:
        if ch < 0 or ch >= image_channels.shape[0]:
            continue
        cname = channel_names[ch] if ch < len(channel_names) else f"C{ch}"
        cname_safe = _sanitise(cname)
        plane = image_channels[ch]
        rp = regionprops_table(
            labeled_mask,
            intensity_image=plane,
            properties=_INTENSITY_PROPS,
        )
        # regionprops_table preserves label order, which matches the base
        # call above.
        df[f"intensity_mean_{cname_safe}"] = rp["intensity_mean"]
        df[f"intensity_min_{cname_safe}"] = rp["intensity_min"]
        df[f"intensity_max_{cname_safe}"] = rp["intensity_max"]
        extras = _intensity_extras(plane, labeled_mask)
        df[f"intensity_median_{cname_safe}"] = [
            extras.get(int(c), {}).get("median", np.nan) for c in cell_ids
        ]
        df[f"intensity_p95_{cname_safe}"] = [
            extras.get(int(c), {}).get("p95", np.nan) for c in cell_ids
        ]
        df[f"intensity_total_{cname_safe}"] = [
            extras.get(int(c), {}).get("total", np.nan) for c in cell_ids
        ]

    # ── Midline-derived morphology (Phase B) ─────────────────────────
    # Compute per cell from a bbox crop of the labelled mask. Cheap (a
    # skeletonize + distance_transform on a small ROI per cell). When
    # disabled or when a cell is too small for a clean midline, columns
    # are filled with NaN.
    if midline_features:
        # Pre-compute Sobel gradient magnitude of the refinement channel
        # ONCE for the whole FOV. Per-cell we'll crop this to each cell's
        # bbox. Computing on the full FOV avoids edge-effect artifacts at
        # the cell-bbox boundary.
        grad_full: Optional[np.ndarray] = None
        if (
            refinement_channel is not None
            and 0 <= refinement_channel < image_channels.shape[0]
        ):
            from scipy.ndimage import sobel
            plane = image_channels[refinement_channel].astype(np.float64)
            gy = sobel(plane, axis=0)
            gx = sobel(plane, axis=1)
            grad_full = np.hypot(gy, gx)

        length_um = np.full(n_cells, np.nan, dtype=np.float64)
        w_med = np.full(n_cells, np.nan, dtype=np.float64)
        w_mean = np.full(n_cells, np.nan, dtype=np.float64)
        w_max = np.full(n_cells, np.nan, dtype=np.float64)
        w_min = np.full(n_cells, np.nan, dtype=np.float64)
        w_std = np.full(n_cells, np.nan, dtype=np.float64)
        max_pos = np.full(n_cells, np.nan, dtype=np.float64)
        min_pos = np.full(n_cells, np.nan, dtype=np.float64)
        sinuosity = np.full(n_cells, np.nan, dtype=np.float64)
        branches = np.full(n_cells, -1, dtype=np.int32)
        area_subpix = np.full(n_cells, np.nan, dtype=np.float64)
        perim_subpix = np.full(n_cells, np.nan, dtype=np.float64)
        for i, cid in enumerate(cell_ids):
            y0 = int(base["bbox-0"][i]); x0 = int(base["bbox-1"][i])
            y1 = int(base["bbox-2"][i]); x1 = int(base["bbox-3"][i])
            sub = labeled_mask[y0:y1, x0:x1] == cid
            ref_img = grad_full[y0:y1, x0:x1] if grad_full is not None else None
            mf = _midline.midline_features(
                sub, pixels_per_um,
                refinement_image=ref_img,
                refine_search_radius_px=refine_search_radius_px,
            )
            if mf is None:
                continue
            length_um[i] = mf.length_um
            w_med[i] = mf.width_median_um
            w_mean[i] = mf.width_mean_um
            w_max[i] = mf.width_max_um
            w_min[i] = mf.width_min_um
            w_std[i] = mf.width_std_um
            max_pos[i] = mf.max_width_position_frac
            min_pos[i] = mf.min_width_position_frac
            sinuosity[i] = mf.sinuosity
            branches[i] = mf.branch_count
            area_subpix[i] = mf.area_um2_subpixel
            perim_subpix[i] = mf.perimeter_um_subpixel
        df["length_um"] = length_um
        df["width_median_um"] = w_med
        df["width_mean_um"] = w_mean
        df["width_max_um"] = w_max
        df["width_min_um"] = w_min
        df["width_std_um"] = w_std
        df["max_width_position_frac"] = max_pos
        df["min_width_position_frac"] = min_pos
        df["sinuosity"] = sinuosity
        df["branch_count"] = branches
        df["area_um2_subpixel"] = area_subpix
        df["perimeter_um_subpixel"] = perim_subpix

    # Identifiers + provenance — these are constant for the FOV.
    cell_uids = [
        f"{run_id}__{well}__{fov_index:03d}__{int(c):05d}" for c in cell_ids
    ]
    df.insert(0, "cell_uid", cell_uids)
    df.insert(1, "run_id", run_id)
    df.insert(2, "well", well)
    df.insert(3, "fov", fov_index)
    df["source_czi"] = source_czi
    df["plate_acquisition_datetime"] = plate_acquisition_datetime
    df["acquisition_time"] = fov_acquisition_time

    return df


def _sanitise(name: str) -> str:
    """Make a channel name safe to embed in a column key."""
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("_",):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s or "chan"
