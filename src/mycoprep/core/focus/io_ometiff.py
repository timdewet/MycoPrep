"""TIFF writers for focus output.

Historically these wrote OME-TIFFs, but the rest of the pipeline
(Split / Segment / Classify) uses ImageJ-hyperstack TIFFs written by
``cellpose_pipeline.save_hyperstack``. To keep downstream tooling
(Fiji / MicrobeJ / scikit-image) happy with a single metadata
convention, we emit the same ImageJ-hyperstack layout here too.

The module name is retained for import-compatibility with the rest of
the focus package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def _imagej_metadata(
    axes: str,
    channel_names: list[str],
    px_um: tuple[float | None, float | None],
    description: str,
    *,
    n_slices: int = 1,
    n_frames: int = 1,
) -> dict:
    """Build an ImageJ hyperstack metadata dict consistent with save_hyperstack."""
    meta: dict = {
        "axes": axes,
        "channels": len(channel_names),
        "slices": n_slices,
        "frames": n_frames,
        "hyperstack": True,
        "mode": "composite",
        "Labels": list(channel_names),
        "unit": "um",
    }
    px_x, px_y = px_um
    if px_x is not None:
        meta["spacing"] = float(px_x)
    if description:
        meta["Info"] = description
    return meta


def _resolution(px_um: tuple[float | None, float | None]):
    px_x, px_y = px_um
    if px_x and px_y:
        # ``px_um`` is micrometres-per-pixel; the TIFF resolution tag is
        # pixels-per-(resolution unit). With resolutionunit=3 (cm) that's
        # px/cm = (1 cm / 1 µm) / (µm/px) = 10_000 / px_um.
        return (10_000.0 / px_x, 10_000.0 / px_y)
    return None


def write(
    path: Path,
    planes_cyx: np.ndarray,
    channel_names: list[str],
    pixel_size_um: tuple[float | None, float | None],
    description: str = "",
) -> None:
    """Write a (C, Y, X) array as an ImageJ-hyperstack TIFF."""
    if planes_cyx.ndim != 3:
        raise ValueError(f"expected (C, Y, X) array, got shape {planes_cyx.shape}")
    if len(channel_names) != planes_cyx.shape[0]:
        raise ValueError(
            f"got {planes_cyx.shape[0]} channels but {len(channel_names)} names"
        )

    # ImageJ hyperstacks want ZCYX even for a single slice; match save_hyperstack's
    # (1, C, Y, X) → ZCYX convention so downstream readers treat it uniformly.
    planes_zcyx = planes_cyx[np.newaxis, :, :, :]
    metadata = _imagej_metadata(
        axes="ZCYX",
        channel_names=channel_names,
        px_um=pixel_size_um,
        description=description,
        n_slices=1,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path),
        planes_zcyx,
        imagej=True,
        metadata=metadata,
        resolution=_resolution(pixel_size_um),
        resolutionunit=3 if _resolution(pixel_size_um) else None,  # 3 = CENTIMETER
    )


def write_tcyx(
    path: Path,
    planes_tcyx: np.ndarray,
    channel_names: list[str],
    pixel_size_um: tuple[float | None, float | None],
    description: str = "",
) -> None:
    """Write a (T, C, Y, X) per-well stack as an ImageJ-hyperstack TIFF.

    FOVs are stored along the Z axis (matching ``save_hyperstack``'s
    convention of ``slices=n_fov``) so Fiji / MicrobeJ treat each FOV as
    a browseable slice instead of a time point.
    """
    if planes_tcyx.ndim != 4:
        raise ValueError(f"expected (T, C, Y, X) array, got shape {planes_tcyx.shape}")
    n_t, n_c, _, _ = planes_tcyx.shape
    if len(channel_names) != n_c:
        raise ValueError(f"got {n_c} channels but {len(channel_names)} names")

    # Re-label T as Z so the output matches Split's (N_FOV, C, Y, X) → ZCYX layout.
    metadata = _imagej_metadata(
        axes="ZCYX",
        channel_names=channel_names,
        px_um=pixel_size_um,
        description=description,
        n_slices=n_t,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path),
        planes_tcyx,
        imagej=True,
        metadata=metadata,
        resolution=_resolution(pixel_size_um),
        resolutionunit=3 if _resolution(pixel_size_um) else None,
    )
