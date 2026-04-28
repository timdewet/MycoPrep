"""Single-FOV in-memory focus, mirroring ``mycoprep.core.focus.pipeline._process_scene``.

The full pipeline's ``run_focus`` walks every scene of a CZI and writes
one OME-TIFF per well. For the live preview we only need the focused
``(C, Y, X)`` array of one scene at one set of focus options. This shim
calls the same primitives the real pipeline uses (``score_stack``,
``pick_best_slice``, ``pick_per_pixel_z``, ``tiling.pick_best_z_per_tile``,
and the ``_assemble_*_planes`` helpers) so the preview output matches a
real run for the same scene.

Whichever preview-only behavior we want diverges from the real pipeline
(currently nothing), it should live here so the mycoprep.core module
stays untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def run(
    czi_path: Path,
    fov_index: int,
    focus_opts: Any,
    phase_channel: int | str | None,
) -> tuple[np.ndarray, list[str], int]:
    """Focus a single scene of ``czi_path`` and return its planes.

    Returns ``(planes_cyx, channel_names, phase_idx)``. ``phase_idx``
    is the resolved index of the phase channel inside the returned
    planes; downstream stages (segment / classify) use it to find the
    grayscale plane.
    """
    from mycoprep.core.focus import (
        channel_id,
        focus as focus_metrics,
        io_czi,
        tiling,
    )
    from mycoprep.core.focus.pipeline import (
        Options,
        _assemble_edf_planes,
        _assemble_output_planes,
        _assemble_tiled_planes,
    )

    scene = io_czi.read_scene(Path(czi_path), int(fov_index))
    phase_override = phase_channel if phase_channel not in (None, -1) else None
    phase_idx = channel_id.resolve_phase_channel(
        scene.array_zcyx, scene.channel_names, phase_override
    )

    # Translate the GUI's FocusOpts into the lower-level Options the
    # primitives expect — same field passthroughs ``run_focus`` uses
    # (api.py:171-177) so preview output stays aligned with real runs.
    # ``save_mip`` is forced False here because MIP companion channels
    # would shift the phase index inside the returned planes; the
    # preview's downstream stages key on phase index, so we keep the
    # output shape == CZI channel count.
    opts = Options(
        metric=getattr(focus_opts, "metric", "ensemble"),
        focus_mode=getattr(focus_opts, "mode", "edf"),
        tile_grid=tuple(getattr(focus_opts, "tile_grid", (3, 3))),
        phase_channel=phase_override,
        save_mip=False,
    )

    phase_stack = scene.array_zcyx[:, phase_idx]

    if opts.focus_mode == "whole":
        scores = focus_metrics.score_stack(
            phase_stack,
            crop_fraction=opts.crop_fraction,
            preblur_sigma=opts.preblur_sigma,
        )
        chosen_z = focus_metrics.pick_best_slice(scores, metric=opts.metric)
        planes, names = _assemble_output_planes(
            scene.array_zcyx, scene.channel_names, phase_idx, chosen_z,
            save_mip=opts.save_mip,
        )
    elif opts.focus_mode == "tiles":
        tile_picks = tiling.pick_best_z_per_tile(
            phase_stack, opts.tile_grid, metric=opts.metric
        )
        z_per_tile = {coord: chosen for coord, (chosen, _) in tile_picks.items()}
        planes, names = _assemble_tiled_planes(
            scene.array_zcyx, scene.channel_names, phase_idx,
            z_per_tile, opts.tile_grid,
            save_mip=opts.save_mip,
        )
    elif opts.focus_mode == "edf":
        z_map = focus_metrics.pick_per_pixel_z(phase_stack)
        planes, names = _assemble_edf_planes(
            scene.array_zcyx, scene.channel_names, phase_idx, z_map,
            save_mip=opts.save_mip,
        )
    else:
        raise ValueError(f"unknown focus_mode {opts.focus_mode!r}")

    # ``planes`` is (C, Y, X). The phase channel index in this output
    # array is the *original* phase_idx because the assemble helpers
    # preserve channel order (MIP channels are appended after each
    # fluorescence one). Return it so the worker doesn't have to
    # second-guess.
    return planes, list(names), int(phase_idx)
