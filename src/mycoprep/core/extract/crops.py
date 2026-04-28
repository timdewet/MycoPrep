"""Single-cell crop extraction in the MorphologicalProfiling_Mtb HDF5 schema.

The output drops directly into ``MorphologicalProfiling_Mtb`` 's
``CropHDF5Dataset``: a single ``crops`` dataset shaped ``(N, C, H, W)`` plus
parallel 1D metadata datasets and file-level attrs. Additional datasets
(``cell_uids``, ``conditions``, ``reporters``, ``acquisition_times``,
``source_czis``, ``fov_indices``, ``cell_labels``) are written for joinability
and ignored by the upstream consumer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.transform import resize


@dataclass
class CropOpts:
    crop_size: int = 96
    pad: int = 10
    crop_channels: Optional[list[int]] = None  # None = all non-mask channels
    include_mask_channel: bool = True
    mask_background: bool = True
    dilate_px: int = 3
    normalise_per_crop: bool = True  # match MorphologicalProfiling_Mtb behaviour


# ──────────────────────────────────────────────────────────────────────────────
# Crop construction
# ──────────────────────────────────────────────────────────────────────────────


def extract_cell_crop(
    image_channels: np.ndarray,
    labeled_mask: np.ndarray,
    cell_label: int,
    selected_channels: list[int],
    opts: CropOpts,
) -> tuple[Optional[np.ndarray], int]:
    """Return a ``(C_out, crop_size, crop_size)`` float32 crop and the cell's
    pixel area, or ``(None, 0)`` if the label is empty.

    ``C_out`` = ``len(selected_channels)`` + (1 if ``include_mask_channel``).
    """
    h, w = labeled_mask.shape
    cell_pixels = labeled_mask == cell_label
    if not cell_pixels.any():
        return None, 0

    ys, xs = np.where(cell_pixels)
    pad = opts.pad
    y_min = max(int(ys.min()) - pad, 0)
    y_max = min(int(ys.max()) + pad + 1, h)
    x_min = max(int(xs.min()) - pad, 0)
    x_max = min(int(xs.max()) + pad + 1, w)

    mask_crop = cell_pixels[y_min:y_max, x_min:x_max]

    if opts.mask_background:
        dilated = binary_dilation(mask_crop, iterations=opts.dilate_px)
    else:
        dilated = np.ones_like(mask_crop)

    channel_planes: list[np.ndarray] = []
    for ch in selected_channels:
        plane = image_channels[ch, y_min:y_max, x_min:x_max].astype(np.float32)
        if opts.mask_background:
            bg = float(np.median(plane[~dilated])) if (~dilated).any() else 0.0
            plane = np.where(dilated, plane, bg).astype(np.float32)
        channel_planes.append(plane)

    if opts.include_mask_channel:
        channel_planes.append(mask_crop.astype(np.float32))

    combined = np.stack(channel_planes, axis=0)

    # Pad to square then resize to the requested side.
    _, ch_h, ch_w = combined.shape
    side = max(ch_h, ch_w)
    padded = np.zeros((combined.shape[0], side, side), dtype=np.float32)
    y_off = (side - ch_h) // 2
    x_off = (side - ch_w) // 2
    padded[:, y_off:y_off + ch_h, x_off:x_off + ch_w] = combined

    n_out = padded.shape[0]
    resized = np.zeros((n_out, opts.crop_size, opts.crop_size), dtype=np.float32)
    for c in range(n_out):
        resized[c] = resize(
            padded[c],
            (opts.crop_size, opts.crop_size),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )

    if opts.normalise_per_crop:
        # Per-image-channel min-max into [0, 1]; the mask channel is rebound.
        n_image = len(selected_channels)
        for c in range(n_image):
            cmin, cmax = resized[c].min(), resized[c].max()
            if cmax > cmin:
                resized[c] = (resized[c] - cmin) / (cmax - cmin)
            else:
                resized[c] = 0.0
        if opts.include_mask_channel:
            resized[-1] = (resized[-1] > 0.5).astype(np.float32)
    elif opts.include_mask_channel:
        resized[-1] = (resized[-1] > 0.5).astype(np.float32)

    return resized, int(cell_pixels.sum())


# ──────────────────────────────────────────────────────────────────────────────
# Filename → condition fields (best-effort, tunable downstream)
# ──────────────────────────────────────────────────────────────────────────────


_CONTROL_HINTS = ("plasmid", "control", "wt", "empty", "untreated", "ctrl")


def derive_condition_fields(tiff_stem: str) -> dict[str, str]:
    """Map a MycoPrep per-well TIFF stem onto the MorphologicalProfiling_Mtb
    condition vocabulary.

    MycoPrep names per-well TIFFs as
    ``{condition}__{reporter}__{mutant_or_drug}[__R{n}]`` (see
    ``split_czi_plate.build_output_filename``). This helper splits on
    ``__`` and assigns roles. The ``condition_type`` heuristic looks for
    common control hints in ``mutant_or_drug``.
    """
    parts = tiff_stem.split("__")
    # Drop a focus-stage filename suffix (e.g. ``_focused``) from the last
    # meaningful part so the replica/mutant fields aren't polluted.
    parts = [_strip_suffix(p) for p in parts]

    condition = parts[0] if len(parts) > 0 else ""
    reporter = parts[1] if len(parts) > 1 else ""
    mutant_or_drug = parts[2] if len(parts) > 2 else ""
    replica = ""
    if len(parts) > 3 and re.match(r"^R\d+$", parts[3], re.IGNORECASE):
        replica = parts[3][1:]

    is_control = any(h in mutant_or_drug.lower() for h in _CONTROL_HINTS)
    condition_type = "control" if is_control else "mutant"
    condition_label = mutant_or_drug or condition

    return {
        "condition": condition,
        "reporter": reporter,
        "mutant_or_drug": mutant_or_drug,
        "replica": replica,
        "condition_label": condition_label,
        "condition_type": condition_type,
        "gene": mutant_or_drug if condition_type == "mutant" else "",
        "drug": "",
        "concentration": "",
        "is_control": is_control,
        "is_drug": False,
    }


def _strip_suffix(part: str) -> str:
    for suf in ("_focused", "_filtered"):
        if part.endswith(suf):
            return part[: -len(suf)]
    return part


# ──────────────────────────────────────────────────────────────────────────────
# Per-well HDF5 writer
# ──────────────────────────────────────────────────────────────────────────────


HDF5_SCHEMA_VERSION = "1.0"


def open_well_h5(
    h5_path: Path,
    crop_size: int,
    n_channels: int,
    channel_names: list[str],
):
    """Open a per-well HDF5 file with the canonical schema, ready for streaming.

    Returns ``(h5_file, crops_ds)`` for incremental ``crops_ds.resize`` /
    ``crops_ds[lo:hi] = batch`` writes. The metadata datasets are created at
    close time via ``finalise_well_h5`` once cell counts are known.
    """
    import h5py

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(str(h5_path), "w")
    crops_ds = f.create_dataset(
        "crops",
        shape=(0, n_channels, crop_size, crop_size),
        maxshape=(None, n_channels, crop_size, crop_size),
        dtype=np.float32,
        chunks=(min(256, max(1, 256)), n_channels, crop_size, crop_size),
        compression="gzip",
        compression_opts=4,
    )
    f.attrs["crop_size"] = int(crop_size)
    f.attrs["n_channels"] = int(n_channels)
    f.attrs["channel_names"] = list(channel_names)
    f.attrs["schema_version"] = HDF5_SCHEMA_VERSION
    return f, crops_ds


def finalise_well_h5(h5_file, metadata: list[dict]) -> None:
    """Write the parallel metadata datasets and close the file."""
    import h5py

    n = len(metadata)
    str_dt = h5py.string_dtype()

    fov_key_to_id: dict[tuple[str, int], int] = {}
    fov_ids = np.empty(n, dtype=np.int32)
    for i, m in enumerate(metadata):
        key = (m["tiff_file"], int(m["fov_index"]))
        if key not in fov_key_to_id:
            fov_key_to_id[key] = len(fov_key_to_id)
        fov_ids[i] = fov_key_to_id[key]

    def _str_ds(name: str, key: str) -> None:
        h5_file.create_dataset(name, data=[m[key] for m in metadata], dtype=str_dt)

    _str_ds("condition_labels", "condition_label")
    _str_ds("condition_types", "condition_type")
    _str_ds("genes", "gene")
    _str_ds("drugs", "drug")
    _str_ds("concentrations", "concentration")
    _str_ds("replicas", "replica")
    _str_ds("tiff_files", "tiff_file")
    h5_file.create_dataset("fov_ids", data=fov_ids)
    h5_file.create_dataset(
        "is_control",
        data=[bool(m["is_control"]) for m in metadata],
        dtype=bool,
    )
    h5_file.create_dataset(
        "is_drug",
        data=[bool(m["is_drug"]) for m in metadata],
        dtype=bool,
    )
    h5_file.create_dataset(
        "areas_px",
        data=[int(m["area_px"]) for m in metadata],
        dtype=np.int32,
    )

    # MycoPrep-specific additions (additive — consumer ignores extras).
    _str_ds("cell_uids", "cell_uid")
    _str_ds("conditions", "condition")
    _str_ds("reporters", "reporter")
    _str_ds("acquisition_times", "acquisition_time")
    _str_ds("source_czis", "source_czi")
    h5_file.create_dataset(
        "fov_indices",
        data=[int(m["fov_index"]) for m in metadata],
        dtype=np.int32,
    )
    h5_file.create_dataset(
        "cell_labels",
        data=[int(m["cell_label"]) for m in metadata],
        dtype=np.int32,
    )

    h5_file.attrs["total_cells"] = n
    h5_file.attrs["n_fovs"] = len(fov_key_to_id)
    h5_file.close()


# ──────────────────────────────────────────────────────────────────────────────
# Multi-well consolidation
# ──────────────────────────────────────────────────────────────────────────────


def consolidate_well_h5_files(
    well_files: Iterable[Path],
    out_path: Path,
) -> Path:
    """Concatenate per-well ``<well>__crops.h5`` into a single ``all_crops.h5``.

    Schema is identical to the per-well files. Crops are streamed batch-wise
    so peak RAM is bounded.
    """
    import h5py

    well_files = [Path(p) for p in well_files if Path(p).exists()]
    if not well_files:
        raise ValueError("consolidate_well_h5_files: no input files")

    with h5py.File(str(well_files[0]), "r") as first:
        crop_size = int(first.attrs["crop_size"])
        n_channels = int(first.attrs["n_channels"])
        channel_names = [str(s) for s in first.attrs["channel_names"]]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = h5py.File(str(out_path), "w")
    crops_ds = out.create_dataset(
        "crops",
        shape=(0, n_channels, crop_size, crop_size),
        maxshape=(None, n_channels, crop_size, crop_size),
        dtype=np.float32,
        chunks=(256, n_channels, crop_size, crop_size),
        compression="gzip",
        compression_opts=4,
    )

    # 1D datasets we copy across — created lazily once we know the total length.
    str_keys = (
        "condition_labels",
        "condition_types",
        "genes",
        "drugs",
        "concentrations",
        "replicas",
        "tiff_files",
        "cell_uids",
        "conditions",
        "reporters",
        "acquisition_times",
        "source_czis",
    )
    int_keys = ("fov_indices", "cell_labels")
    bool_keys = ("is_control", "is_drug")
    int32_keys = ("areas_px",)

    str_buffers: dict[str, list] = {k: [] for k in str_keys}
    int_buffers: dict[str, list] = {k: [] for k in int_keys + int32_keys}
    bool_buffers: dict[str, list] = {k: [] for k in bool_keys}
    fov_id_offset = 0
    consolidated_fov_ids: list[int] = []
    n_total = 0

    for src_path in well_files:
        with h5py.File(str(src_path), "r") as src:
            src_n = int(src["crops"].shape[0])
            if src_n == 0:
                continue
            crops_ds.resize(n_total + src_n, axis=0)
            crops_ds[n_total: n_total + src_n] = src["crops"][:]
            n_total += src_n

            for k in str_keys:
                if k in src:
                    str_buffers[k].extend([s.decode() if isinstance(s, bytes) else str(s) for s in src[k][:]])
                else:
                    str_buffers[k].extend([""] * src_n)
            for k in int_keys + int32_keys:
                if k in src:
                    int_buffers[k].extend([int(v) for v in src[k][:]])
                else:
                    int_buffers[k].extend([0] * src_n)
            for k in bool_keys:
                if k in src:
                    bool_buffers[k].extend([bool(v) for v in src[k][:]])
                else:
                    bool_buffers[k].extend([False] * src_n)

            # Renumber per-well fov_ids into the consolidated namespace.
            if "fov_ids" in src:
                src_fov = src["fov_ids"][:].astype(np.int32)
                consolidated_fov_ids.extend((src_fov + fov_id_offset).tolist())
                fov_id_offset += int(src_fov.max()) + 1 if src_fov.size else 0
            else:
                consolidated_fov_ids.extend([fov_id_offset] * src_n)
                fov_id_offset += 1

    str_dt = h5py.string_dtype()
    for k, vals in str_buffers.items():
        out.create_dataset(k, data=vals, dtype=str_dt)
    for k, vals in int_buffers.items():
        dtype = np.int32
        out.create_dataset(k, data=np.asarray(vals, dtype=dtype))
    for k, vals in bool_buffers.items():
        out.create_dataset(k, data=np.asarray(vals, dtype=bool))
    out.create_dataset("fov_ids", data=np.asarray(consolidated_fov_ids, dtype=np.int32))

    out.attrs["crop_size"] = crop_size
    out.attrs["n_channels"] = n_channels
    out.attrs["channel_names"] = channel_names
    out.attrs["total_cells"] = n_total
    out.attrs["n_fovs"] = fov_id_offset
    out.attrs["schema_version"] = HDF5_SCHEMA_VERSION
    out.close()
    return out_path
