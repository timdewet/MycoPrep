"""GUI-friendly facade for the imaging pipeline.

This module exposes the pre-processing stages as callable library
functions with a uniform `progress_cb(fraction: float, message: str)`
contract. Each adapter wraps the lower-level functions defined in the
source modules without re-implementing them.

Stage adapters:
    split_plate(czi, layout_df, out_dir, ...)
    run_focus(czi, out_dir, opts, ...)
    segment_tiff(tiff, out_path, opts, ...)
    classify_filter_tiff(tiff, model_path, out_path, opts, ...)
    extract_features_tiff(tiff, out_path, opts, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

ProgressCB = Callable[[float, str], None]


def _noop(_: float, __: str) -> None:
    pass


def _read_imagej_labels(tiff_path: Path) -> Optional[list[str]]:
    """Return the ImageJ-hyperstack channel labels embedded in ``tiff_path``.

    Returns None if the file doesn't exist, isn't an ImageJ hyperstack,
    or doesn't carry a ``Labels`` key.
    """
    import tifffile
    try:
        with tifffile.TiffFile(str(tiff_path)) as tif:
            meta = tif.imagej_metadata or {}
    except Exception:  # noqa: BLE001
        return None
    labels = meta.get("Labels")
    if isinstance(labels, (list, tuple)) and labels:
        return [str(x) for x in labels]
    return None


def _read_pixels_per_um(tiff_path: Path) -> Optional[float]:
    """Return pixels-per-µm inferred from a TIFF's metadata.

    Prefers ImageJ ``spacing`` (µm/px). Falls back to TIFF ``XResolution``
    in pixels-per-centimetre. Returns None if no reliable pixel size is
    present.
    """
    import tifffile
    try:
        with tifffile.TiffFile(str(tiff_path)) as tif:
            meta = tif.imagej_metadata or {}
            first_page = tif.pages[0] if tif.pages else None
            tags = first_page.tags if first_page is not None else {}
            xres = tags.get("XResolution")
            resunit = tags.get("ResolutionUnit")
    except Exception:  # noqa: BLE001
        return None

    spacing = meta.get("spacing")
    if isinstance(spacing, (int, float)) and spacing > 0:
        return 1.0 / float(spacing)

    if xres is not None:
        try:
            num, den = xres.value
            px_per_unit = float(num) / float(den) if den else 0.0
        except Exception:  # noqa: BLE001
            px_per_unit = 0.0
        if px_per_unit <= 0:
            return None
        # ResolutionUnit: 1 = none, 2 = inch, 3 = centimetre.
        unit = int(resunit.value) if resunit is not None else 3
        if unit == 3:       # px/cm → px/µm
            return px_per_unit / 10_000.0
        if unit == 2:       # px/inch → px/µm
            return px_per_unit / 25_400.0
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: CZI → per-well TIFF split
# ─────────────────────────────────────────────────────────────────────────────

def split_plate(
    czi_path: Path,
    layout_df,                     # pandas.DataFrame with cols well/condition/reporter/mutant_or_drug
    out_dir: Path,
    channel_names: Optional[Iterable[str]] = None,
    pixels_per_um: Optional[float] = None,  # None → auto-detect from CZI metadata
    progress_cb: ProgressCB = _noop,
) -> list[Path]:
    """Split a multi-position CZI into per-well TIFF hyperstacks.

    Wraps `split_czi_plate.split_and_save`, accepting a pandas DataFrame
    layout (which the GUI's PlateLayout model produces) instead of an
    on-disk CSV.
    """
    from .split_czi_plate import split_and_save, normalize_well_id

    layout = {}
    for _, row in layout_df.iterrows():
        well = normalize_well_id(str(row["well"]))
        if not well:
            continue
        layout[well] = {
            "condition":      str(row.get("condition", "")).strip(),
            "reporter":       str(row.get("reporter", "")).strip(),
            "mutant_or_drug": str(row.get("mutant_or_drug", "")).strip(),
            "replica":        str(row.get("replica", "")).strip(),
        }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_cb(0.0, f"Splitting {czi_path.name} into {len(layout)} wells")
    split_and_save(
        czi_path=czi_path,
        layout=layout,
        output_dir=out_dir,
        channel_names=list(channel_names) if channel_names else None,
        pixels_per_um=pixels_per_um,
    )
    progress_cb(1.0, f"Wrote {len(layout)} per-well TIFFs to {out_dir}")
    return sorted(out_dir.glob("*.tif"))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Focus picking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FocusOpts:
    metric: str = "ensemble"
    mode: str = "edf"              # "whole" | "tiles" | "edf"
    tile_grid: tuple[int, int] = (3, 3)
    phase_channel: Optional[int | str] = None  # None = auto-ID
    save_zmaps: bool = False       # EDF mode only
    save_mip: bool = False         # Add <name>_MIP channel per fluorescence channel
    filename_suffix: str = "_focused"


def run_focus(
    czi_path: Path,
    out_dir: Path,
    opts: FocusOpts = field(default_factory=FocusOpts),  # type: ignore[arg-type]
    layout_df=None,
    progress_cb: ProgressCB = _noop,
    single_bucket_label: Optional[str] = None,
) -> Path:
    """Run FocusPicker on a CZI; write per-well OME-TIFFs to ``out_dir``.

    If ``layout_df`` is provided, wells are named with the layout's
    condition/reporter/mutant/replica template so Focus output matches
    Split output naming (with ``filename_suffix`` appended).

    If ``single_bucket_label`` is set, the CZI is treated as a single
    sample (all scenes lumped together) and written to one TIFF with
    that label as its stem — for non-plate / bulk-mode workflows.
    """
    from .focus.pipeline import Options, process_czi
    from .split_czi_plate import normalize_well_id

    opts_obj = Options(
        metric=opts.metric,
        focus_mode=opts.mode,
        tile_grid=opts.tile_grid,
        phase_channel=opts.phase_channel,
        save_mip=opts.save_mip,
    ) if not isinstance(opts, Options) else opts

    well_filenames: dict[str, str] | None = None
    if layout_df is not None:
        well_filenames = {}
        for _, row in layout_df.iterrows():
            well = normalize_well_id(str(row["well"]))
            cond = str(row.get("condition", "")).strip()
            rep  = str(row.get("reporter", "")).strip()
            mut  = str(row.get("mutant_or_drug", "")).strip()
            rn   = str(row.get("replica", "")).strip()
            if not cond and not rep and not mut:
                continue
            parts = [cond, rep, mut]
            if rn:
                parts.append(f"R{rn}")
            well_filenames[well] = "__".join(p.replace(" ", "_") for p in parts)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_cb(0.0, f"Focus picking on {czi_path.name} (mode={opts.mode})")
    process_czi(
        czi_path=czi_path,
        opts=opts_obj,
        out_dir=out_dir,
        well_filenames=well_filenames,
        save_zmaps=opts.save_zmaps,
        filename_suffix=opts.filename_suffix,
        progress_cb=progress_cb,
        single_bucket_label=single_bucket_label,
    )
    progress_cb(1.0, f"Focus-picked output written to {out_dir}")
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Cellpose-SAM segmentation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SegmentOpts:
    model_type: str = "cpsam"
    diameter: Optional[float] = None
    gpu: bool = True
    pixels_per_um: float = 13.8767
    # Cellpose tunables. Lower cellprob_threshold (e.g. -2.0) keeps more
    # marginal cells; raise flow_threshold (e.g. 0.6) to be more permissive
    # about mask shape; lower min_size to keep smaller objects.
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15


def segment_tiff(
    tiff_path: Path,
    out_path: Path,
    phase_channel: int,
    opts: SegmentOpts = field(default_factory=SegmentOpts),  # type: ignore[arg-type]
    channel_labels: Optional[list[str]] = None,
    progress_cb: ProgressCB = _noop,
) -> Path:
    """Segment all FOVs in a multi-FOV TIFF; append mask channel; save hyperstack."""
    from cellpose.models import CellposeModel

    from .cellpose_pipeline import process_tiff_unit, save_hyperstack

    progress_cb(0.0, f"Loading Cellpose model ({opts.model_type}, gpu={opts.gpu})")
    model = CellposeModel(gpu=opts.gpu, model_type=opts.model_type)

    progress_cb(0.1, f"Segmenting FOVs in {tiff_path.name}")
    stacked, fov_names, total_cells = process_tiff_unit(
        tiff_path=tiff_path,
        model=model,
        phase_channel=phase_channel,
        diameter=opts.diameter,
        classify_opts=None,           # classification handled by classify_filter_tiff
        model_type=opts.model_type,
        flow_threshold=opts.flow_threshold,
        cellprob_threshold=opts.cellprob_threshold,
        min_size=opts.min_size,
    )

    if stacked is None:
        progress_cb(1.0, f"No FOVs segmented from {tiff_path.name}")
        return out_path

    # Prefer channel labels embedded in the input TIFF (written by Focus or
    # Split) — those match the actual channel content even when optional
    # add-ons like MIP companions changed the channel count. Fall back to the
    # caller-supplied list (from the Input tab), then to anonymous names.
    n_image_channels = stacked.shape[1] - 1
    embedded = _read_imagej_labels(tiff_path)

    if embedded and len(embedded) == n_image_channels:
        base = list(embedded)
    elif channel_labels is not None and len(channel_labels) == n_image_channels:
        base = list(channel_labels)
    else:
        base = [f"C{i}" for i in range(n_image_channels)]

    channel_labels = base + ["BinaryMask"]

    detected_px = _read_pixels_per_um(tiff_path)
    pixels_per_um = detected_px if detected_px is not None else opts.pixels_per_um

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_hyperstack(
        stacked=stacked,
        output_path=out_path,
        condition_name=tiff_path.stem,
        filenames=fov_names,
        channel_labels=channel_labels,
        pixels_per_um=pixels_per_um,
    )

    # Carry the acquisition sidecar from the input TIFF to the segmented
    # output so a Focus → Segment → Features pipeline (no Classify) still
    # surfaces per-FOV provenance to the Features stage.
    src_sidecar = tiff_path.with_name(tiff_path.stem + "__acquisition.json")
    if src_sidecar.exists():
        dst_sidecar = out_path.with_name(out_path.stem + "__acquisition.json")
        try:
            import shutil as _shutil
            _shutil.copyfile(src_sidecar, dst_sidecar)
        except OSError:
            pass

    progress_cb(1.0, f"Segmented {total_cells} cells across {len(fov_names)} FOVs → {out_path.name}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Cell-quality classification (post-segmentation filter)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassifyOpts:
    model_path: Optional[Path] = None       # None = rules-only
    keep_classes: tuple[str, ...] = ("good",)
    confidence_threshold: float = 0.5
    use_rules: bool = True
    pixels_per_um: float = 13.8767


# Bundled model presets (resolved to absolute paths at the package root)
PRESET_MODELS = {
    "mtb": Path(__file__).resolve().parents[2] / "models_mtb" / "best_model.pth",
    # "msm": Path(...)  # add when an Msm model is trained
}


def resolve_classifier_preset(name: str) -> Path:
    if name not in PRESET_MODELS:
        raise KeyError(f"Unknown classifier preset '{name}'. Known: {sorted(PRESET_MODELS)}")
    p = PRESET_MODELS[name]
    if not p.exists():
        raise FileNotFoundError(f"Preset '{name}' resolves to {p} which does not exist")
    return p


def classify_filter_tiff(
    tiff_path: Path,
    out_path: Path,
    phase_channel: int,
    opts: ClassifyOpts = field(default_factory=ClassifyOpts),  # type: ignore[arg-type]
    channel_labels: Optional[list[str]] = None,
    progress_cb: ProgressCB = _noop,
) -> Path:
    """Read a segmented TIFF, filter clumps/OOF/edge cells per FOV, write filtered TIFF.

    The input is expected to be a hyperstack written by `segment_tiff`,
    i.e. the final channel is the binary mask. The output preserves the
    same layout but with rejected cells zeroed out of the mask.
    """
    import tifffile

    from .cell_quality_classifier import classify_and_filter_mask
    from .cellpose_pipeline import save_hyperstack
    from .label_cells import load_hyperstack

    progress_cb(0.0, f"Loading segmented hyperstack {tiff_path.name}")
    data, _meta = load_hyperstack(tiff_path)        # (N_FOV, C, Y, X)
    n_fov = data.shape[0]

    # IMPORTANT: use the value from `opts` (i.e. the Segment/Classify panel),
    # NOT a freshly-detected one from the TIFF metadata. The trained CNN's
    # rule-based companion filters (MIN_AREA_UM2 / MAX_AREA_UM2) were
    # calibrated to whatever pixel-size constant the *training script* used
    # (13.8767 by default). Substituting the file's "true" pixel size at
    # inference time would shift the decision boundary and reject good cells.
    pixels_per_um = opts.pixels_per_um

    # Log which channel we're feeding the CNN as phase so wrong assignments
    # are diagnosable in the run log.
    embedded = _read_imagej_labels(tiff_path)
    if embedded and 0 <= phase_channel < len(embedded):
        progress_cb(0.03, f"Phase channel = {phase_channel} ('{embedded[phase_channel]}'); "
                          f"input channels: {embedded}  ·  filter px/µm = {pixels_per_um}")
    else:
        progress_cb(0.03, f"Phase channel = {phase_channel}  ·  filter px/µm = {pixels_per_um}")

    # Mask is the last channel; image channels are everything else.
    from skimage.measure import label as sk_label

    # ── Per-FOV crash safety ────────────────────────────────────────────
    # Each FOV is checkpointed to a sibling .partial/ directory as soon as
    # it's processed. If the run is stopped or crashes mid-well, restarting
    # will skip FOVs whose .npz already exists. When all FOVs are present,
    # the final hyperstack is assembled and the .partial dir is removed.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_dir = out_path.with_suffix(out_path.suffix + ".partial")
    partial_dir.mkdir(parents=True, exist_ok=True)

    totals = {"total": 0, "kept": 0, "edge": 0, "debris": 0, "cnn": 0}

    def _fov_path(idx: int) -> Path:
        return partial_dir / f"fov_{idx:03d}.npz"

    n_existing = sum(1 for i in range(n_fov) if _fov_path(i).exists())
    if n_existing:
        progress_cb(0.04, f"Resuming: {n_existing}/{n_fov} FOV checkpoints already on disk")

    for i in range(n_fov):
        progress_cb(0.05 + 0.9 * (i / max(n_fov, 1)), f"Filtering FOV {i+1}/{n_fov}")

        fov_file = _fov_path(i)
        if fov_file.exists():
            # Already processed in a prior run; just account for stats and skip.
            try:
                with np.load(fov_file) as npz:
                    for k in totals:
                        totals[k] += int(npz["totals"][list(totals).index(k)])
            except Exception:
                pass
            continue

        fov = data[i]
        image_channels = fov[:-1]
        # Segment writes a *binary* (0/255) mask for MicrobeJ compatibility,
        # so we have to re-label connected components into unique integer IDs
        # before the classifier can iterate cells.
        raw_mask = fov[-1]
        unique_positive = np.unique(raw_mask[raw_mask > 0])
        if len(unique_positive) <= 1:
            labeled_mask = sk_label(raw_mask > 0, connectivity=1).astype(np.int32)
        else:
            labeled_mask = raw_mask.astype(np.int32)

        filtered_mask, report = classify_and_filter_mask(
            labeled_mask=labeled_mask,
            image_channels=image_channels,
            phase_channel=phase_channel,
            model_path=str(opts.model_path) if opts.model_path else None,
            pixels_per_um=pixels_per_um,
            keep_classes=opts.keep_classes,
            confidence_threshold=opts.confidence_threshold,
            use_rules=opts.use_rules,
            verbose=False,
        )
        fov_totals = (
            int(report.get("total_cells", 0)),
            int(report.get("kept", 0)),
            int(report.get("removed_edge", 0)),
            int(report.get("removed_debris", 0)),
            int(report.get("removed_cnn", 0)),
        )
        for key, val in zip(totals.keys(), fov_totals):
            totals[key] += val

        new_fov = np.concatenate(
            [image_channels, filtered_mask[None].astype(image_channels.dtype)],
            axis=0,
        )
        # Atomic write: stage to .tmp then rename so a crash mid-write
        # doesn't leave a half-finished checkpoint behind. (Use an open file
        # handle because ``np.savez_compressed`` silently appends ``.npz`` to
        # any path that doesn't already end that way, which would defeat the
        # rename below.)
        tmp = fov_file.with_name(fov_file.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, fov=new_fov, totals=np.asarray(fov_totals, dtype=np.int64))
        tmp.replace(fov_file)

    # Re-assemble final hyperstack from per-FOV checkpoints.
    out_fovs = []
    for i in range(n_fov):
        with np.load(_fov_path(i)) as npz:
            out_fovs.append(npz["fov"])
    stacked = np.stack(out_fovs, axis=0)

    # Channel labels: prefer those embedded in the segment-stage input TIFF
    # (which already has BinaryMask tagged on the end), fall back to the
    # caller-supplied list, then to anonymous names.
    n_image_channels = stacked.shape[1] - 1
    embedded = _read_imagej_labels(tiff_path)

    if embedded and len(embedded) == stacked.shape[1]:
        # Full labels including BinaryMask — keep as-is.
        channel_labels = list(embedded)
    elif embedded and len(embedded) == n_image_channels:
        channel_labels = list(embedded) + ["BinaryMask"]
    elif channel_labels is not None and len(channel_labels) == n_image_channels:
        channel_labels = list(channel_labels) + ["BinaryMask"]
    else:
        channel_labels = [f"C{i}" for i in range(n_image_channels)] + ["BinaryMask"]

    save_hyperstack(
        stacked=stacked,
        output_path=out_path,
        condition_name=tiff_path.stem + "__filtered",
        filenames=[f"fov_{i:03d}" for i in range(stacked.shape[0])],
        channel_labels=channel_labels,
        pixels_per_um=pixels_per_um,
    )

    # Carry the acquisition sidecar from the input TIFF to the classified
    # output so the Features stage (which reads from 03_classify/) can find
    # per-FOV provenance without having to walk back to the focus dir.
    src_sidecar = tiff_path.with_name(tiff_path.stem + "__acquisition.json")
    if src_sidecar.exists():
        dst_sidecar = out_path.with_name(out_path.stem + "__acquisition.json")
        try:
            import shutil as _shutil
            _shutil.copyfile(src_sidecar, dst_sidecar)
        except OSError:
            pass

    # Final write succeeded → drop the partial checkpoints.
    import shutil
    try:
        shutil.rmtree(partial_dir)
    except OSError:
        pass

    kept_pct = 100.0 * totals["kept"] / totals["total"] if totals["total"] else 0.0
    progress_cb(1.0,
        f"Kept {totals['kept']}/{totals['total']} cells ({kept_pct:.0f}%) "
        f"→ rejected: edge={totals['edge']}, debris/clump={totals['debris']}, "
        f"cnn={totals['cnn']}  →  {out_path.name}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Per-cell feature extraction (Phase A: regionprops baseline)
# ─────────────────────────────────────────────────────────────────────────────

# Re-export the Features-stage entry points so callers can keep importing
# everything from ``mycoprep.core.api``. The module-level lazy import keeps
# heavy deps (pandas, h5py, pyarrow) out of the cold-start path for users who
# only run earlier stages.

from .extract import (  # noqa: E402,F401
    ExtractOpts,
    consolidate_crops,
    consolidate_features,
    extract_features_tiff,
    make_qc_plots,
)
