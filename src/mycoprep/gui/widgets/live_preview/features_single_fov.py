"""Single-FOV in-memory feature extraction for the live preview.

Wraps :func:`mycoprep.core.extract.per_cell.extract_fov_features` so we
can run the feature stage on cached arrays (no disk read, no per-well
parquet, no HDF5 crops). Returns just the per-cell DataFrame; that's
all the live preview needs to place per-object text labels.

The full pipeline's :func:`extract_features_tiff` does plate / well /
condition bookkeeping; for a one-shot preview those fields are filled
in with neutral defaults so the columns the user actually cares about
(``length_um``, ``width_mean_um``, ``area_um2``, ``*_intensity_mean``,
``centroid_x``, ``centroid_y``) are populated identically to a real
run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run(
    image_channels: np.ndarray,
    mask: np.ndarray,
    features_opts: Any,
    channel_names: list[str],
    pixels_per_um: float | None,
) -> pd.DataFrame:
    """Compute the per-cell features DataFrame for one FOV.

    ``image_channels`` is the ``(C, Y, X)`` array (same one the worker
    handed to segment / classify) — the mask channel is NOT included.
    ``mask`` is the ``(Y, X)`` labeled mask from the segment stage.

    Returns an empty DataFrame when no cells are present.
    """
    from mycoprep.core.extract.per_cell import extract_fov_features

    if mask is None or int(mask.max()) == 0:
        return pd.DataFrame()

    px = float(pixels_per_um) if pixels_per_um else 1.0

    intensity_channels = None
    if features_opts is not None:
        # ExtractOpts.fluorescence_channels is the canonical field
        # (per ``api._build_extract_kwargs``); fall back to attribute
        # variations defensively.
        for attr in ("fluorescence_channels", "intensity_channels"):
            v = getattr(features_opts, attr, None)
            if v is not None:
                intensity_channels = list(v)
                break

    midline_flag = bool(getattr(features_opts, "midline_features", True))

    refinement_channel = None
    if bool(getattr(features_opts, "refine_contour", False)):
        # The full pipeline picks the phase channel for refinement. The
        # worker doesn't pass phase index here, so default to 0 (which
        # matches the typical Phase channel layout in our CZIs).
        refinement_channel = 0

    df = extract_fov_features(
        image_channels=image_channels,
        labeled_mask=mask.astype(np.int32, copy=False),
        channel_names=list(channel_names) if channel_names else [
            f"C{i}" for i in range(image_channels.shape[0])
        ],
        pixels_per_um=px,
        run_id="preview",
        well="preview",
        fov_index=0,
        source_czi="preview",
        plate_acquisition_datetime=None,
        fov_acquisition_time=None,
        intensity_channels=intensity_channels,
        midline_features=midline_flag,
        refinement_channel=refinement_channel,
    )
    return df
