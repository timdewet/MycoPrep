"""Stage protocol + adapters wrapping mycoprep.core.api functions.

Each adapter walks its input files, invokes the corresponding api.py
function, and forwards progress via a stage-scaled progress callback so
multi-file stages report a smooth 0→1 to the outer runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Protocol

from mycoprep.core.api import (
    classify_filter_tiff,
    consolidate_crops,
    consolidate_features,
    extract_features_tiff,
    make_qc_plots,
    run_focus,
    segment_tiff,
    split_plate,
)

from .context import RunContext

ProgressCB = Callable[[float, str], None]


class Stage(Protocol):
    name: str
    # Stages that do per-CZI / per-input reuse logic themselves set this to
    # True. The runner then skips its top-level "if any output files exist,
    # skip the whole stage" shortcut and lets the stage decide what to run
    # vs. reuse. Without this, adding a new CZI to a plate and re-running
    # with reuse-existing on would skip Focus entirely because the previous
    # run's TIFFs are still on disk.
    handles_own_reuse: bool = False

    def enabled(self, ctx: RunContext) -> bool: ...
    def validate(self, ctx: RunContext) -> list[str]: ...
    def output_dir(self, ctx: RunContext) -> Path: ...
    def run(self, ctx: RunContext, progress_cb: ProgressCB) -> list[Path]: ...


def _scaled_cb(outer: ProgressCB, base: float, span: float) -> ProgressCB:
    def inner(f: float, msg: str) -> None:
        outer(base + f * span, msg)
    return inner


def _iter_tiffs(directory: Path) -> list[Path]:
    """All TIFFs directly in *directory*, including OME-TIFFs (``*.ome.tiff``)."""
    paths: set[Path] = set()
    for pattern in ("*.tif", "*.tiff", "*.ome.tif", "*.ome.tiff"):
        paths.update(directory.glob(pattern))
    return sorted(paths)


# ─────────────────────────────────────────────────────────────────────────────

def _expected_stems_for_czi(layout_active_df, czi_filename: str, focus_suffix: str = "") -> list[str]:
    """Stems (without ``.tif``) that the layout expects to exist on disk for
    a particular CZI. ``focus_suffix`` is appended (e.g. ``_focused``) for
    the Focus stage's output dir.
    """
    from mycoprep.core.split_czi_plate import build_output_filename
    stems: list[str] = []
    if "source_czi" in layout_active_df.columns:
        sub = layout_active_df[layout_active_df["source_czi"].astype(str) == czi_filename]
    else:
        sub = layout_active_df
    for _, row in sub.iterrows():
        stem = build_output_filename(
            str(row.get("condition", "")),
            str(row.get("reporter", "")),
            str(row.get("mutant_or_drug", "")),
            str(row.get("replica", "")),
        ).removesuffix(".tif")
        if stem:
            stems.append(stem + focus_suffix)
    return stems


def _all_stems_present(stage_dir: Path, stems: list[str]) -> bool:
    """True iff every expected stem already exists as a ``.tif`` in ``stage_dir``."""
    if not stems:
        return False
    if not stage_dir.exists():
        return False
    for s in stems:
        if not (stage_dir / f"{s}.tif").exists():
            return False
    return True


def _expected_stems_for_layout(ctx, focus_suffix: str = "") -> set[str]:
    """All output stems the current layout expects across every CZI.

    Used by downstream stages (Segment/Classify/Features) to filter out
    stale TIFFs left in the focus dir from a prior experiment whose labels
    no longer match the current layout. Without this filter, the
    downstream stages would faithfully process those orphans too —
    wasting compute and producing rows in the features table that don't
    correspond to any current well.

    Handles both context types:

    - :class:`RunContext` (single-plate): pulls from
      ``ctx.layout.disambiguated_active_rows()``.
    - :class:`BulkRunContext` (bulk / single-file): pulls from
      ``ctx.czi_entries`` — each row already carries condition /
      reporter / mutant / replica.
    """
    from mycoprep.core.split_czi_plate import build_output_filename

    if hasattr(ctx, "czi_entries"):
        stems: set[str] = set()
        for entry in getattr(ctx, "czi_entries", []) or []:
            stem = build_output_filename(
                str(entry.get("condition", "")),
                str(entry.get("reporter", "")),
                str(entry.get("mutant_or_drug", "")),
                str(entry.get("replica", "")),
            ).removesuffix(".tif")
            if stem:
                stems.add(stem + focus_suffix)
        return stems

    active = ctx.layout.disambiguated_active_rows()
    stems = set()
    for czi in ctx.all_czi_paths:
        if "source_czi" in active.columns:
            sub = active[active["source_czi"].astype(str) == Path(czi).name]
        else:
            sub = active
        if sub.empty and len(ctx.all_czi_paths) > 1:
            continue
        sub = sub if not sub.empty else active
        for s in _expected_stems_for_czi(sub, Path(czi).name, focus_suffix=focus_suffix):
            stems.add(s)
    return stems


def _filter_to_layout(
    inputs: list[Path],
    expected_stems: set[str],
    progress_cb,
) -> list[Path]:
    """Drop TIFFs whose stems aren't in the layout's expected set.

    Returns the filtered list. If any TIFFs are dropped, log how many were
    skipped and the first few names so the user can spot stale orphans.
    """
    if not expected_stems:
        return inputs
    kept: list[Path] = []
    skipped: list[Path] = []
    for tiff in inputs:
        if tiff.stem in expected_stems:
            kept.append(tiff)
        else:
            skipped.append(tiff)
    if skipped:
        sample = ", ".join(p.name for p in skipped[:3])
        more = "" if len(skipped) <= 3 else f" (+ {len(skipped) - 3} more)"
        progress_cb(
            0.0,
            f"Skipping {len(skipped)} stale TIFF(s) not in current layout: "
            f"{sample}{more}",
        )
    return kept


class SplitStage:
    name = "Split"
    # Per-CZI reuse: skip CZIs whose outputs are already complete, run
    # CZIs whose outputs are missing (e.g. newly-added CZI).
    handles_own_reuse = True

    def enabled(self, ctx: RunContext) -> bool:
        # Focus already groups scenes by well using the layout naming and
        # produces the single "per-well" output. Running Split alongside
        # Focus would duplicate work and outputs, so Split is skipped.
        if ctx.do_focus:
            return False
        return ctx.do_split

    def validate(self, ctx: RunContext) -> list[str]:
        issues = []
        for p in ctx.all_czi_paths:
            if not Path(p).exists():
                issues.append(f"CZI file not found: {p}")
        issues.extend(ctx.layout.validate())
        return issues

    def output_dir(self, ctx: RunContext) -> Path:
        return ctx.split_dir

    def run(self, ctx: RunContext, progress_cb: ProgressCB) -> list[Path]:
        # Multi-CZI plate: each CZI splits its own subset of wells (matched
        # by ``source_czi`` in the layout) into the shared split_dir. Per-CZI
        # reuse: when ``ctx.reuse_existing`` is on, a CZI whose expected
        # outputs are all already on disk is skipped; CZIs with any missing
        # output (e.g. a newly-added CZI) still run.
        paths = ctx.all_czi_paths
        outputs: list[Path] = []
        active = ctx.layout.disambiguated_active_rows()
        n_skipped = 0
        for i, czi in enumerate(paths):
            if "source_czi" in active.columns:
                sub = active[active["source_czi"].astype(str) == Path(czi).name]
            else:
                sub = active
            if sub.empty:
                continue

            expected_stems = _expected_stems_for_czi(sub, Path(czi).name, focus_suffix="")
            if ctx.reuse_existing and _all_stems_present(ctx.split_dir, expected_stems):
                progress_cb(
                    (i + 1) / len(paths),
                    f"[{i+1}/{len(paths)}] Reused {Path(czi).name} "
                    f"({len(expected_stems)} TIFFs already on disk)",
                )
                outputs.extend(ctx.split_dir / f"{s}.tif" for s in expected_stems)
                n_skipped += 1
                continue

            cb = _scaled_cb(progress_cb, i / len(paths), 1 / len(paths))
            res = split_plate(
                czi_path=Path(czi),
                layout_df=sub,
                out_dir=ctx.split_dir,
                channel_names=ctx.channel_labels,
                progress_cb=cb,
            )
            outputs.extend(res)
        progress_cb(
            1.0,
            f"Split {len(paths)} CZI(s) → {len(outputs)} TIFFs"
            + (f" ({n_skipped} reused)" if n_skipped else ""),
        )
        return outputs


class FocusStage:
    name = "Focus"
    handles_own_reuse = True

    def enabled(self, ctx: RunContext) -> bool:
        return ctx.do_focus

    def validate(self, ctx: RunContext) -> list[str]:
        if not ctx.do_focus:
            return []
        issues = []
        for p in ctx.all_czi_paths:
            if not Path(p).exists():
                issues.append(f"Focus needs CZI but not found: {p}")
        return issues

    def output_dir(self, ctx: RunContext) -> Path:
        return ctx.focus_dir

    def run(self, ctx: RunContext, progress_cb: ProgressCB) -> list[Path]:
        # Multi-CZI plate: focus each CZI's subset of wells into the shared
        # focus_dir. The per-CZI sub-layout is matched via ``source_czi``;
        # falling back to the full layout in legacy single-CZI mode where
        # the column is empty. Per-CZI reuse: a CZI whose expected outputs
        # are all already on disk is skipped when ``ctx.reuse_existing`` is
        # on; newly-added CZIs still run.
        paths = ctx.all_czi_paths
        active = ctx.layout.disambiguated_active_rows()
        focus_suffix = ctx.focus_opts.filename_suffix or ""
        n_skipped = 0
        for i, czi in enumerate(paths):
            if "source_czi" in active.columns:
                sub = active[active["source_czi"].astype(str) == Path(czi).name]
            else:
                sub = active
            if sub.empty and len(paths) > 1:
                continue
            sub_layout = sub if not sub.empty else active

            expected_stems = _expected_stems_for_czi(
                sub_layout, Path(czi).name, focus_suffix=focus_suffix,
            )
            if ctx.reuse_existing and _all_stems_present(ctx.focus_dir, expected_stems):
                progress_cb(
                    (i + 1) / len(paths),
                    f"[{i+1}/{len(paths)}] Reused {Path(czi).name} "
                    f"({len(expected_stems)} focus TIFFs already on disk)",
                )
                n_skipped += 1
                continue

            cb = _scaled_cb(progress_cb, i / len(paths), 1 / len(paths))
            run_focus(
                czi_path=Path(czi),
                out_dir=ctx.focus_dir,
                opts=ctx.focus_opts,
                layout_df=sub_layout,
                progress_cb=cb,
            )
        progress_cb(
            1.0,
            f"Focus {len(paths)} CZI(s)"
            + (f" ({n_skipped} reused)" if n_skipped else ""),
        )
        return _iter_tiffs(ctx.focus_dir)


class SegmentStage:
    name = "Segment"
    handles_own_reuse = True

    def enabled(self, ctx: RunContext) -> bool:
        return ctx.do_segment

    def validate(self, ctx: RunContext) -> list[str]:
        # Need either focus or split output to exist (after upstream stages run)
        return []

    def output_dir(self, ctx: RunContext) -> Path:
        return ctx.segment_dir

    def run(self, ctx: RunContext, progress_cb: ProgressCB) -> list[Path]:
        # Prefer focus output if it exists on disk, else split output. We
        # accept either regardless of whether do_focus / do_split are set
        # for THIS run — the user may be running Segment alone against
        # outputs produced by an earlier run.
        inputs: list[Path] = []
        if ctx.focus_dir.exists():
            inputs = _iter_tiffs(ctx.focus_dir)
        if not inputs and ctx.split_dir.exists():
            inputs = _iter_tiffs(ctx.split_dir)

        if not inputs:
            raise RuntimeError(
                "SegmentStage has no input TIFFs. Run Focus or Split first, "
                "or point the output directory at a folder that already "
                "contains 01_split_and_focused/ or 01_split/."
            )

        # Filter to TIFFs the current layout actually expects so we don't
        # process stale orphans from a previous experiment that may still
        # be sitting in 01_split_and_focused/.
        focus_suffix = ctx.focus_opts.filename_suffix or ""
        expected = _expected_stems_for_layout(ctx, focus_suffix=focus_suffix)
        inputs = _filter_to_layout(inputs, expected, progress_cb)
        if not inputs:
            raise RuntimeError(
                "SegmentStage: no input TIFFs match the current layout. "
                "Either the layout's labels don't line up with on-disk "
                "filenames, or those wells haven't been focused yet."
            )

        outputs: list[Path] = []
        n = len(inputs)
        n_skipped = 0
        for i, tiff in enumerate(inputs):
            out = ctx.segment_dir / tiff.name
            # Per-TIFF reuse: skip if output already exists. Avoids
            # re-segmenting wells from a previous run when the user has
            # added a new CZI and ticked reuse-existing.
            if ctx.reuse_existing and out.exists():
                progress_cb(
                    (i + 1) / n,
                    f"[{i+1}/{n}] Reused {tiff.name}",
                )
                outputs.append(out)
                n_skipped += 1
                continue
            cb = _scaled_cb(progress_cb, i / n, 1 / n)
            segment_tiff(
                tiff_path=tiff,
                out_path=out,
                phase_channel=ctx.phase_channel or 0,
                opts=ctx.segment_opts,
                channel_labels=ctx.channel_labels,
                progress_cb=cb,
            )
            outputs.append(out)
        progress_cb(
            1.0,
            f"Segmented {n} TIFFs"
            + (f" ({n_skipped} reused)" if n_skipped else ""),
        )
        return outputs


class ClassifyStage:
    name = "Classify"
    handles_own_reuse = True

    def enabled(self, ctx: RunContext) -> bool:
        return ctx.do_classify

    def validate(self, ctx: RunContext) -> list[str]:
        # Classify needs segmented input, but that input can come from either
        # a Segment stage running in the same run OR from a previous run's
        # output directory on disk. If neither is present, ClassifyStage.run
        # will raise a clear error at execution time.
        if ctx.do_classify and not ctx.do_segment and not ctx.segment_dir.exists():
            return [
                f"Classify needs segmented TIFFs in {ctx.segment_dir} but that "
                "directory doesn't exist. Enable Segment or run it first."
            ]
        return []

    def output_dir(self, ctx: RunContext) -> Path:
        return ctx.classify_dir

    def run(self, ctx: RunContext, progress_cb: ProgressCB) -> list[Path]:
        inputs = _iter_tiffs(ctx.segment_dir)
        if not inputs:
            raise RuntimeError("ClassifyStage has no segmented TIFFs to process.")

        focus_suffix = ctx.focus_opts.filename_suffix or ""
        expected = _expected_stems_for_layout(ctx, focus_suffix=focus_suffix)
        inputs = _filter_to_layout(inputs, expected, progress_cb)
        if not inputs:
            raise RuntimeError(
                "ClassifyStage: no segmented TIFFs match the current layout."
            )

        outputs: list[Path] = []
        n = len(inputs)
        n_skipped = 0
        for i, tiff in enumerate(inputs):
            out = ctx.classify_dir / tiff.name
            # Per-TIFF reuse: skip already-classified wells when reuse-existing.
            if ctx.reuse_existing and out.exists():
                progress_cb(
                    (i + 1) / n,
                    f"[{i+1}/{n}] Reused {tiff.name}",
                )
                outputs.append(out)
                n_skipped += 1
                continue
            cb = _scaled_cb(progress_cb, i / n, 1 / n)
            classify_filter_tiff(
                tiff_path=tiff,
                out_path=out,
                phase_channel=ctx.phase_channel or 0,
                opts=ctx.classify_opts,
                channel_labels=ctx.channel_labels,
                progress_cb=cb,
            )
            outputs.append(out)
        progress_cb(
            1.0,
            f"Filtered {n} TIFFs"
            + (f" ({n_skipped} reused)" if n_skipped else ""),
        )
        return outputs


class ExtractStage:
    """Per-cell feature extraction (Phase A: regionprops + HDF5 crops).

    Reads the most-recently-produced TIFFs upstream — preferring the
    Classify output (filtered cells) and falling back to Segment if Classify
    wasn't run. Writes one Parquet (and optionally one HDF5) per well into
    ``ctx.features_dir``, then consolidates per-well crop files into a single
    ``all_crops.h5`` matching the MorphologicalProfiling_Mtb schema.
    """

    name = "Features"
    handles_own_reuse = True

    def enabled(self, ctx) -> bool:
        return bool(getattr(ctx, "do_features", False))

    def validate(self, ctx) -> list[str]:
        if not self.enabled(ctx):
            return []
        # Same fallback as `run`: classify_dir → segment_dir.
        if (not ctx.classify_dir.exists()
                and not ctx.segment_dir.exists()
                and not getattr(ctx, "do_segment", False)
                and not getattr(ctx, "do_classify", False)):
            return [
                f"Features needs segmented or classified TIFFs in "
                f"{ctx.classify_dir} or {ctx.segment_dir}, but neither exists."
            ]
        return []

    def output_dir(self, ctx) -> Path:
        return ctx.features_dir

    def run(self, ctx, progress_cb: ProgressCB) -> list[Path]:
        # Prefer classify output (quality-filtered cells); fall back to segment.
        inputs: list[Path] = []
        if ctx.classify_dir.exists():
            inputs = _iter_tiffs(ctx.classify_dir)
        if not inputs and ctx.segment_dir.exists():
            inputs = _iter_tiffs(ctx.segment_dir)
        if not inputs:
            raise RuntimeError(
                "ExtractStage has no segmented input TIFFs. Run Segment "
                "(and optionally Classify) first."
            )

        focus_suffix = ctx.focus_opts.filename_suffix or ""
        expected = _expected_stems_for_layout(ctx, focus_suffix=focus_suffix)
        inputs = _filter_to_layout(inputs, expected, progress_cb)
        if not inputs:
            raise RuntimeError(
                "ExtractStage: no input TIFFs match the current layout."
            )

        ctx.features_dir.mkdir(parents=True, exist_ok=True)
        opts = getattr(ctx, "features_opts", None)
        if opts is None:
            from mycoprep.core.api import ExtractOpts as _EO
            opts = _EO()

        run_id = ctx.output_dir.name or "run"

        outputs: list[Path] = []
        crop_files: list[Path] = []
        n = len(inputs)
        n_skipped = 0
        for i, tiff in enumerate(inputs):
            out = ctx.features_dir / (tiff.stem + ".parquet")
            # Per-TIFF reuse: skip already-extracted wells when reuse-existing.
            # Also picks up the optional per-well crops .h5 if present.
            if ctx.reuse_existing and out.exists():
                progress_cb(
                    (i + 1) / n,
                    f"[{i+1}/{n}] Reused {tiff.name}",
                )
                outputs.append(out)
                crop_h5 = out.with_name(out.stem + "__crops.h5")
                if crop_h5.exists():
                    crop_files.append(crop_h5)
                n_skipped += 1
                continue
            cb = _scaled_cb(progress_cb, i / n, 1 / n)
            extract_features_tiff(
                tiff_path=tiff,
                out_path=out,
                opts=opts,
                run_id=run_id,
                channel_labels=ctx.channel_labels,
                phase_channel=ctx.phase_channel,
                progress_cb=cb,
            )
            outputs.append(out)
            crop_h5 = out.with_name(out.stem + "__crops.h5")
            if crop_h5.exists():
                crop_files.append(crop_h5)

        # Consolidate per-well Parquet → all_features.parquet (+ all_features.csv).
        if outputs:
            try:
                write_csv = bool(getattr(opts, "save_csv", True))
                consolidate_features(
                    outputs, ctx.features_dir / "all_features.parquet",
                    write_csv=write_csv,
                )
                msg = f"Consolidated {len(outputs)} per-well features → all_features.parquet"
                if write_csv:
                    msg += " (+ all_features.csv)"
                progress_cb(1.0, msg)
            except Exception as e:  # noqa: BLE001
                progress_cb(1.0, f"Per-well features OK but consolidation failed: {e}")

        # Consolidate per-well HDF5 crops into a single all_crops.h5
        # matching the MorphologicalProfiling_Mtb schema.
        if crop_files:
            try:
                consolidate_crops(crop_files, ctx.features_dir / "all_crops.h5")
                progress_cb(1.0, f"Consolidated {len(crop_files)} per-well .h5 → all_crops.h5")
            except Exception as e:  # noqa: BLE001
                progress_cb(1.0, f"Per-well crops OK but consolidation failed: {e}")

        # Auto-QC plots — runs after consolidation so the plots use the
        # canonical ``all_features.parquet``. Off via ``opts.make_qc_plots``
        # for users who don't want the matplotlib import at end of run.
        if outputs and getattr(opts, "make_qc_plots", True):
            try:
                make_qc_plots(ctx.features_dir, progress_cb=progress_cb)
            except Exception as e:  # noqa: BLE001
                progress_cb(1.0, f"qc_plots skipped ({e})")

        if not outputs and not crop_files:
            progress_cb(1.0, f"Extracted features for {n} TIFFs")
        return outputs


ALL_STAGES: list[Stage] = [
    SplitStage(),
    FocusStage(),
    SegmentStage(),
    ClassifyStage(),
    ExtractStage(),
]
