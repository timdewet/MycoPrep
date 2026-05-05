"""Top-level Features-stage entry points.

``extract_features_tiff`` walks one segmented hyperstack and produces:
  - ``<out_path>.parquet`` — per-cell morphology + intensity (Phase A: regionprops).
  - ``<out_path>__crops.h5`` — single-cell crops in the MorphologicalProfiling_Mtb
    schema, when ``opts.save_crops`` is True.

A per-FOV checkpoint dir mirrors the pattern in ``classify_filter_tiff`` so a
killed run can be resumed.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import crops as crops_mod
from . import per_cell


ProgressCB = Callable[[float, str], None]


def _noop(_: float, __: str) -> None:
    pass


@dataclass
class ExtractOpts:
    # Per-cell features
    morphology: bool = True
    intensity: bool = True
    midline_features: bool = True   # length_um, width_*, sinuosity, branch_count
    # Sub-pixel contour refinement against the phase-channel intensity
    # gradient. Algorithm follows MicrobeJ / Oufti (Ducret et al., Nat.
    # Microbiol. 2016, doi:10.1038/nmicrobiol.2016.77; Paintdakhi et al.,
    # Mol. Microbiol. 2016) — each contour vertex is snapped along its
    # outward normal to the local Sobel-gradient peak, then the result
    # is smoothed with a closed periodic cubic spline. Pushes area /
    # perimeter / width precision past the half-pixel resolution that
    # ``find_contours`` of a binary mask produces.
    refine_contour: bool = True
    refine_search_radius_px: float = 2.0
    fluorescence_channels: Optional[list[int]] = None  # None = all non-mask
    pixels_per_um: float = 13.8767  # auto-overridden by TIFF metadata

    # Output formats
    save_csv: bool = True   # write <well>.csv alongside <well>.parquet
    # Generate QC plots from all_features.parquet at end of run.
    # Outputs to <features_dir>/qc_plots/.
    make_qc_plots: bool = True

    # Feature library — accumulates features across runs for clustering
    add_to_library: bool = False
    library_dir: Optional[Path] = None   # default: ~/.mycoprep/feature_library/
    species: str = ""                    # e.g. "M. tuberculosis"
    experiment_type: str = "knockdown"   # "knockdown" or "drug"
    # Comma-separated list of mutant/condition tokens to treat as controls
    # for S-score computation (e.g. "NT1, NT2, WT, DMSO"). Matched as
    # whole-word case-insensitive tokens against the condition label.
    control_labels: str = ""

    # Single-cell crops
    save_crops: bool = True
    crop_size: int = 96
    crop_pad: int = 10
    crop_channels: Optional[list[int]] = None  # None = all non-mask channels
    include_mask_channel: bool = True
    mask_background: bool = True
    normalise_per_crop: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _read_acquisition_sidecar(tiff_path: Path) -> tuple[Optional[str], list[Optional[str]], Optional[str]]:
    """Return ``(plate_acq_dt, fov_acq_times, source_czi)`` from the sidecar.

    Falls back to ``(None, [], None)`` if the sidecar is missing/unparseable.
    The Features stage carries on with null provenance columns.
    """
    sidecar = tiff_path.with_name(tiff_path.stem + "__acquisition.json")
    if not sidecar.exists():
        return None, [], None
    try:
        payload = json.loads(sidecar.read_text())
    except Exception:  # noqa: BLE001
        return None, [], None
    plate = payload.get("plate_acquisition_datetime")
    fov_times = payload.get("fov_acquisition_times") or []
    source = payload.get("source_czi")
    return plate, list(fov_times), source


def _resolve_intensity_channels(
    n_image_channels: int,
    requested: Optional[list[int]],
) -> list[int]:
    if requested is None:
        return list(range(n_image_channels))
    return [c for c in requested if 0 <= c < n_image_channels]


def _label_mask(raw_mask: np.ndarray) -> np.ndarray:
    """Re-label a binary or already-labelled mask channel.

    Mirrors the logic in ``classify_filter_tiff`` so the Features stage doesn't
    care whether its input came from Segment (binary) or Classify (relabelled).
    """
    from skimage.measure import label as sk_label

    unique_positive = np.unique(raw_mask[raw_mask > 0])
    if len(unique_positive) <= 1:
        return sk_label(raw_mask > 0, connectivity=1).astype(np.int32)
    return raw_mask.astype(np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# Per-FOV worker
# ──────────────────────────────────────────────────────────────────────────────


def _process_fov(
    fov: np.ndarray,
    fov_index: int,
    *,
    image_channel_names: list[str],
    pixels_per_um: float,
    run_id: str,
    well: str,
    source_czi: str,
    plate_acq_dt: Optional[str],
    fov_acq_time: Optional[str],
    intensity_channels: list[int],
    crop_opts: Optional[crops_mod.CropOpts],
    crop_channels: list[int],
    condition_fields: dict[str, str],
    tiff_filename: str,
    midline_features_flag: bool = True,
    refinement_channel: Optional[int] = None,
    refine_search_radius_px: float = 2.0,
) -> tuple[pd.DataFrame, Optional[np.ndarray], list[dict]]:
    """Return ``(features_df, crops_array, crop_metadata)`` for one FOV.

    ``crops_array`` is ``None`` when ``crop_opts`` is None.
    """
    image_channels = fov[:-1]
    raw_mask = fov[-1]
    labeled_mask = _label_mask(raw_mask)

    df = per_cell.extract_fov_features(
        image_channels=image_channels,
        labeled_mask=labeled_mask,
        channel_names=image_channel_names,
        pixels_per_um=pixels_per_um,
        run_id=run_id,
        well=well,
        fov_index=fov_index,
        source_czi=source_czi,
        plate_acquisition_datetime=plate_acq_dt,
        fov_acquisition_time=fov_acq_time,
        intensity_channels=intensity_channels,
        midline_features=midline_features_flag,
        refinement_channel=refinement_channel,
        refine_search_radius_px=refine_search_radius_px,
    )

    if crop_opts is None or df.empty:
        return df, None, []

    crops_list: list[np.ndarray] = []
    crop_meta: list[dict] = []
    for row in df.itertuples(index=False):
        cell_id = int(row.cell_id)
        crop, area_px = crops_mod.extract_cell_crop(
            image_channels=image_channels,
            labeled_mask=labeled_mask,
            cell_label=cell_id,
            selected_channels=crop_channels,
            opts=crop_opts,
        )
        if crop is None:
            continue
        crops_list.append(crop)
        crop_meta.append({
            "cell_uid": row.cell_uid,
            "tiff_file": tiff_filename,
            "fov_index": fov_index,
            "cell_label": cell_id,
            "area_px": int(area_px),
            "condition": condition_fields["condition"],
            "reporter": condition_fields["reporter"],
            "mutant_or_drug": condition_fields["mutant_or_drug"],
            "replica": condition_fields["replica"],
            "condition_label": condition_fields["condition_label"],
            "condition_type": condition_fields["condition_type"],
            "gene": condition_fields["gene"],
            "drug": condition_fields["drug"],
            "concentration": condition_fields["concentration"],
            "is_control": condition_fields["is_control"],
            "is_drug": condition_fields["is_drug"],
            "acquisition_time": fov_acq_time or "",
            "source_czi": source_czi,
        })

    crops_array = np.stack(crops_list, axis=0) if crops_list else None
    return df, crops_array, crop_meta


# ──────────────────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────────────────


def extract_features_tiff(
    tiff_path: Path,
    out_path: Path,
    opts: ExtractOpts = field(default_factory=ExtractOpts),  # type: ignore[arg-type]
    *,
    run_id: Optional[str] = None,
    channel_labels: Optional[list[str]] = None,
    phase_channel: Optional[int] = None,
    progress_cb: ProgressCB = _noop,
) -> Path:
    """Process one segmented hyperstack into a Parquet table (and optionally
    HDF5 crops). ``out_path`` is the Parquet path; the ``__crops.h5``
    companion is written next to it.
    """
    from ..label_cells import load_hyperstack
    from ..api import _read_imagej_labels, _read_pixels_per_um  # type: ignore[attr-defined]

    tiff_path = Path(tiff_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    progress_cb(0.0, f"Loading {tiff_path.name}")
    data, _meta = load_hyperstack(tiff_path)
    n_fov = int(data.shape[0])
    n_total_channels = int(data.shape[1])
    n_image_channels = n_total_channels - 1

    embedded = _read_imagej_labels(tiff_path)
    if embedded and len(embedded) == n_total_channels:
        image_channel_names = list(embedded[:-1])
    elif channel_labels is not None and len(channel_labels) == n_image_channels:
        image_channel_names = list(channel_labels)
    else:
        image_channel_names = [f"C{i}" for i in range(n_image_channels)]

    detected_px = _read_pixels_per_um(tiff_path)
    pixels_per_um = detected_px if detected_px is not None else opts.pixels_per_um

    plate_acq_dt, fov_acq_times, sidecar_source = _read_acquisition_sidecar(tiff_path)
    source_czi = sidecar_source or tiff_path.name
    well = tiff_path.stem
    if run_id is None:
        run_id = out_path.parent.parent.name or "run"
    condition_fields = crops_mod.derive_condition_fields(well)

    intensity_channels = _resolve_intensity_channels(n_image_channels, opts.fluorescence_channels)
    crop_channels = _resolve_intensity_channels(n_image_channels, opts.crop_channels)
    if not crop_channels:
        crop_channels = list(range(n_image_channels))

    crop_opts: Optional[crops_mod.CropOpts] = None
    if opts.save_crops:
        crop_opts = crops_mod.CropOpts(
            crop_size=opts.crop_size,
            pad=opts.crop_pad,
            crop_channels=crop_channels,
            include_mask_channel=opts.include_mask_channel,
            mask_background=opts.mask_background,
            normalise_per_crop=opts.normalise_per_crop,
        )

    n_crop_channels = len(crop_channels) + (1 if opts.include_mask_channel else 0)
    crop_channel_names = [image_channel_names[c] for c in crop_channels]
    if opts.include_mask_channel:
        crop_channel_names.append("mask")

    # Per-FOV checkpoint dir, mirroring classify_filter_tiff's resumability.
    partial_dir = out_path.with_suffix(out_path.suffix + ".partial")
    partial_dir.mkdir(parents=True, exist_ok=True)

    n_existing = sum(1 for i in range(n_fov) if (partial_dir / f"fov_{i:03d}.npz").exists())
    if n_existing:
        progress_cb(0.04, f"Resuming: {n_existing}/{n_fov} FOV checkpoints already on disk")

    crops_h5_path = out_path.with_name(out_path.stem + "__crops.h5") if opts.save_crops else None
    h5_file = crops_ds = None
    if crops_h5_path is not None:
        h5_file, crops_ds = crops_mod.open_well_h5(
            crops_h5_path,
            crop_size=opts.crop_size,
            n_channels=n_crop_channels,
            channel_names=crop_channel_names,
        )

    all_dfs: list[pd.DataFrame] = []
    all_crop_meta: list[dict] = []
    n_crops_written = 0

    try:
        for i in range(n_fov):
            progress_cb(0.05 + 0.9 * (i / max(n_fov, 1)),
                        f"FOV {i+1}/{n_fov}")

            ckpt = partial_dir / f"fov_{i:03d}.npz"
            df_npz = partial_dir / f"fov_{i:03d}.parquet"
            meta_npz = partial_dir / f"fov_{i:03d}__meta.json"

            if ckpt.exists() and df_npz.exists():
                df = pd.read_parquet(df_npz)
                if df_npz.exists():
                    all_dfs.append(df)
                if h5_file is not None and ckpt.exists():
                    with np.load(ckpt) as npz:
                        if "crops" in npz and npz["crops"].size:
                            arr = npz["crops"]
                            crops_ds.resize(n_crops_written + arr.shape[0], axis=0)
                            crops_ds[n_crops_written: n_crops_written + arr.shape[0]] = arr
                            n_crops_written += arr.shape[0]
                if meta_npz.exists():
                    all_crop_meta.extend(json.loads(meta_npz.read_text()))
                continue

            fov_acq_time = fov_acq_times[i] if i < len(fov_acq_times) else None

            # Resolve which channel drives contour refinement: caller's
            # phase_channel if given, else channel 0.
            ref_ch = (
                phase_channel
                if (opts.refine_contour and phase_channel is not None
                    and 0 <= phase_channel < n_image_channels)
                else (0 if opts.refine_contour and n_image_channels > 0 else None)
            )
            df, crops_arr, crop_meta = _process_fov(
                data[i],
                i,
                image_channel_names=image_channel_names,
                pixels_per_um=pixels_per_um,
                run_id=run_id,
                well=well,
                source_czi=source_czi,
                plate_acq_dt=plate_acq_dt,
                fov_acq_time=fov_acq_time,
                intensity_channels=intensity_channels,
                crop_opts=crop_opts,
                crop_channels=crop_channels,
                condition_fields=condition_fields,
                tiff_filename=tiff_path.name,
                midline_features_flag=opts.midline_features,
                refinement_channel=ref_ch,
                refine_search_radius_px=opts.refine_search_radius_px,
            )

            # Stage checkpoint (atomic via .tmp + replace)
            tmp_df = df_npz.with_name(df_npz.name + ".tmp")
            df.to_parquet(tmp_df, index=False)
            tmp_df.replace(df_npz)

            if crops_arr is None:
                np.savez(ckpt, crops=np.zeros((0,), dtype=np.float32))
            else:
                tmp_ckpt = ckpt.with_name(ckpt.name + ".tmp")
                with open(tmp_ckpt, "wb") as fh:
                    np.savez_compressed(fh, crops=crops_arr)
                tmp_ckpt.replace(ckpt)
            tmp_meta = meta_npz.with_name(meta_npz.name + ".tmp")
            tmp_meta.write_text(json.dumps(crop_meta))
            tmp_meta.replace(meta_npz)

            all_dfs.append(df)
            if crops_arr is not None and h5_file is not None:
                crops_ds.resize(n_crops_written + crops_arr.shape[0], axis=0)
                crops_ds[n_crops_written: n_crops_written + crops_arr.shape[0]] = crops_arr
                n_crops_written += crops_arr.shape[0]
            all_crop_meta.extend(crop_meta)

        # Concatenate and write final Parquet (and optionally CSV).
        if all_dfs:
            full = pd.concat(all_dfs, ignore_index=True)
        else:
            full = pd.DataFrame()
        full.to_parquet(out_path, index=False)
        if opts.save_csv:
            csv_path = out_path.with_suffix(".csv")
            full.to_csv(csv_path, index=False)

        if h5_file is not None:
            crops_mod.finalise_well_h5(h5_file, all_crop_meta)
            h5_file = None  # closed by finalise

        # Final write succeeded → drop the partial checkpoints.
        try:
            shutil.rmtree(partial_dir)
        except OSError:
            pass

        msg = f"Wrote {len(full)} cells → {out_path.name}"
        if opts.save_csv:
            msg += f" (+ {out_path.with_suffix('.csv').name})"
        if crops_h5_path:
            msg += f" + {n_crops_written} crops → {crops_h5_path.name}"
        progress_cb(1.0, msg)
        return out_path

    finally:
        if h5_file is not None:
            try:
                h5_file.close()
            except Exception:  # noqa: BLE001
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Multi-well consolidation entry
# ──────────────────────────────────────────────────────────────────────────────


def consolidate_crops(
    well_h5_files: list[Path],
    out_path: Path,
) -> Path:
    """Merge per-well ``<well>__crops.h5`` files into a single ``all_crops.h5``."""
    return crops_mod.consolidate_well_h5_files(well_h5_files, out_path)


def consolidate_features(
    well_parquet_files: list[Path],
    out_path: Path,
    *,
    write_csv: bool = True,
) -> Path:
    """Concatenate per-well Parquet files into a single ``all_features.parquet``
    (and a matching CSV when ``write_csv``).

    Returns the Parquet path. Files missing on disk are silently skipped.
    """
    paths = [Path(p) for p in well_parquet_files if Path(p).exists()]
    if not paths:
        raise ValueError("consolidate_features: no input files")
    frames = [pd.read_parquet(p) for p in paths]
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)
    if write_csv:
        full.to_csv(out_path.with_suffix(".csv"), index=False)
    return out_path
