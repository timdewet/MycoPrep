"""PipelineRunner — walks enabled stages on a QThread, emits progress signals."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .bulk_layout import _output_filename
from .context import BulkRunContext, RunContext
from .stages import ALL_STAGES, ClassifyStage, ExtractStage, SegmentStage, Stage, _iter_tiffs


class StopRequested(Exception):
    """Raised inside a progress callback when the user clicks Stop.

    Stages call progress_cb frequently (per-FOV); raising from there gives
    a near-immediate cancellation instead of having to wait for the next
    stage boundary.
    """


class PipelineRunner(QObject):
    """Executes enabled stages sequentially on its own thread.

    Emits, in order: runStarted → stageStarted → stageProgress* → stageFinished
    (repeated per stage) → runFinished / runFailed.
    """

    runStarted = pyqtSignal()
    stagesPlanned = pyqtSignal(list)             # names of stages that will actually run
    stageStarted = pyqtSignal(str)               # stage name
    stageProgress = pyqtSignal(str, float, str)  # stage name, fraction, message
    stageFinished = pyqtSignal(str, int)         # stage name, num_outputs
    runFinished = pyqtSignal(Path)               # manifest path
    runFailed = pyqtSignal(str)                  # error message

    def __init__(
        self,
        ctx: RunContext,
        stages: list[Stage] | None = None,
        reuse_existing: bool = False,
    ) -> None:
        super().__init__()
        self._ctx = ctx
        self._stages = stages or ALL_STAGES
        self._thread: QThread | None = None
        self._reuse_existing = reuse_existing
        self._stop_requested = False

    def request_stop(self) -> None:
        """Ask the runner to stop at the next stage boundary."""
        self._stop_requested = True

    def _rename_outputs_for_layout_changes(self, log_cb) -> None:
        """Detect on-disk TIFFs whose filenames don't match the current
        layout's labels and rename them in place.

        We can't rely on the previous ``plate_layout.csv`` because it gets
        overwritten at end-of-run — by the time the user reruns after a
        label edit, the prior layout snapshot is already gone. Instead we
        bridge old↔new via the ``scene_indices`` recorded in each TIFF's
        ``__acquisition.json`` sidecar (immutable since acquisition).

        Procedure:
          1. Build {frozenset(scene_indices) → well_id} from the current
             layout's active rows.
          2. For each TIFF in the focus/split dir, read its sidecar's
             scene_indices, look up which well it now belongs to, and
             compute the expected stem from the current layout.
          3. If the TIFF's actual stem differs from the expected stem,
             schedule a rename and propagate it through downstream stages.

        Files with no sidecar, or whose scene_indices match no current
        well, are left alone.
        """
        ctx = self._ctx
        try:
            from mycoprep.core.split_czi_plate import (
                build_output_filename, normalize_well_id,
            )
        except Exception:  # noqa: BLE001
            return

        # Build {(source_czi, frozenset(scene_indices)) → (well_id,
        # expected_stem)} from the current layout. The source_czi axis is
        # required for multi-CZI plates: scene indices repeat across CZIs
        # (CZI1's scene 5 ≠ CZI2's scene 5) so without the CZI tag we would
        # rename TIFFs to the wrong well. Wells without a recorded
        # source_czi (legacy single-CZI layouts) are matched on
        # scene_indices alone.
        # Use the auto-disambiguated rows so duplicate-label control wells
        # (e.g. several "NT" wells) line up with the actual on-disk
        # filenames Split/Focus produced.
        rows = ctx.layout.disambiguated_active_rows()
        scene_to_well: dict[tuple[str, frozenset], tuple[str, str]] = {}
        scene_to_well_no_czi: dict[frozenset, tuple[str, str]] = {}
        for _, row in rows.iterrows():
            well_id = normalize_well_id(str(row.get("well", "")))
            if not well_id:
                continue
            sis = row.get("scene_indices") or []
            if isinstance(sis, str):
                # PlateLayout sometimes stores as ";"-joined string (post-CSV
                # round-trip); normalize.
                sis = [int(x) for x in sis.split(";") if x.strip().isdigit()]
            try:
                key = frozenset(int(x) for x in sis)
            except (TypeError, ValueError):
                continue
            if not key:
                continue
            src_czi = str(row.get("source_czi", "")).strip()
            expected_stem = build_output_filename(
                str(row.get("condition", "")),
                str(row.get("reporter", "")),
                str(row.get("mutant_or_drug", "")),
                str(row.get("replica", "")),
            ).removesuffix(".tif")
            if src_czi:
                scene_to_well[(src_czi, key)] = (well_id, expected_stem)
            else:
                scene_to_well_no_czi[key] = (well_id, expected_stem)

        if not scene_to_well and not scene_to_well_no_czi:
            return

        # Walk the focus/split dir for TIFFs whose sidecar (source_czi,
        # scene_indices) matches a current-layout well but whose filename
        # doesn't.
        focus_suffix = ctx.focus_opts.filename_suffix or ""
        rename_map: dict[str, str] = {}
        # Diagnostic counters so the user can see why nothing got renamed
        # if the rename pass appears to no-op.
        n_seen = 0
        n_no_sidecar = 0
        n_no_match = 0
        n_already_correct = 0
        scan_dirs = [ctx.focus_dir, ctx.split_dir]
        for d in scan_dirs:
            if not d.exists():
                continue
            for tiff in sorted(d.glob("*.tif")):
                n_seen += 1
                sidecar = tiff.with_name(tiff.stem + "__acquisition.json")
                if not sidecar.exists():
                    n_no_sidecar += 1
                    continue
                try:
                    payload = json.loads(sidecar.read_text())
                    sis = payload.get("scene_indices") or []
                    key = frozenset(int(x) for x in sis)
                    src_czi = str(payload.get("source_czi", "")).strip()
                except Exception:  # noqa: BLE001
                    n_no_match += 1
                    continue
                match = scene_to_well.get((src_czi, key)) if src_czi else None
                if match is None:
                    match = scene_to_well_no_czi.get(key)
                if not match:
                    n_no_match += 1
                    continue
                _well_id, expected_stem = match
                expected_full = expected_stem + focus_suffix
                actual_stem = tiff.stem  # already includes focus_suffix
                actual_base = (
                    actual_stem[: -len(focus_suffix)]
                    if focus_suffix and actual_stem.endswith(focus_suffix)
                    else actual_stem
                )
                if actual_base != expected_stem:
                    rename_map[actual_base] = expected_stem
                else:
                    n_already_correct += 1

        # Always log the rename-pass summary so the user can see why a label
        # edit didn't appear to take effect.
        #
        # The buckets are mutually exclusive and add to ``n_seen``:
        #   - missing sidecar  : TIFF has no __acquisition.json → can't bridge
        #   - no layout match  : sidecar exists, but its (source_czi,
        #                        scene_indices) doesn't connect to any well in
        #                        the current layout → layout was rebuilt
        #                        without matching scenes, or the CZI name in
        #                        the sidecar differs from the layout's
        #   - already correct  : matched a well AND filename already right
        #   - to rename        : matched a well AND filename differs
        log_cb(0.0,
            f"Layout-rename pass: {n_seen} TIFF(s) scanned · "
            f"{n_no_sidecar} no sidecar · "
            f"{n_no_match} sidecar present but no layout match · "
            f"{n_already_correct} matched + already correct · "
            f"{len(rename_map)} matched + to rename"
        )

        # When everything fell into "no layout match", dump one sidecar +
        # one layout entry so the user can see what's actually mismatching.
        if (n_no_match == n_seen - n_no_sidecar and n_no_match > 0
                and len(rename_map) == 0 and n_already_correct == 0):
            for d in scan_dirs:
                if not d.exists():
                    continue
                sample_tiff = next(iter(d.glob("*.tif")), None)
                if sample_tiff is None:
                    continue
                sidecar = sample_tiff.with_name(sample_tiff.stem + "__acquisition.json")
                if not sidecar.exists():
                    continue
                try:
                    payload = json.loads(sidecar.read_text())
                except Exception:  # noqa: BLE001
                    break
                log_cb(0.0,
                    f"  e.g. {sample_tiff.name}: sidecar source_czi="
                    f"{payload.get('source_czi')!r}, "
                    f"scene_indices={sorted(payload.get('scene_indices') or [])[:6]}"
                    + ("…" if len(payload.get('scene_indices') or []) > 6 else "")
                )
                break
            if scene_to_well:
                (src, key), (well, stem) = next(iter(scene_to_well.items()))
                log_cb(0.0,
                    f"  vs. layout entry: source_czi={src!r}, "
                    f"scene_indices={sorted(key)[:6]}"
                    + ("…" if len(key) > 6 else "")
                    + f", expected_stem={stem!r}"
                )
            elif scene_to_well_no_czi:
                key, (well, stem) = next(iter(scene_to_well_no_czi.items()))
                log_cb(0.0,
                    f"  vs. layout entry (no source_czi recorded): "
                    f"scene_indices={sorted(key)[:6]}"
                    + ("…" if len(key) > 6 else "")
                    + f", expected_stem={stem!r}"
                )
            else:
                log_cb(0.0,
                    "  layout has no wells with scene_indices — "
                    "the layout couldn't be reverse-matched at all"
                )

        if not rename_map:
            return

        # Stage dirs to walk. Each entry: (dir, suffixes appended to the stem
        # for that stage's files). The Focus stage appends ``_focused`` (or
        # whatever ``filename_suffix`` is). Segment/Classify keep the same
        # stem as the focus stem because they overwrite ``<stem>.tif``.
        focus_suffix = ctx.focus_opts.filename_suffix or ""
        stage_dirs: list[tuple[Path, str]] = [
            (ctx.split_dir,    ""),               # 01_split
            (ctx.focus_dir,    focus_suffix),     # 01_focus / 01_split_and_focused
            (ctx.segment_dir,  focus_suffix),     # 02_segment (keeps focus stem)
            (ctx.classify_dir, focus_suffix),     # 03_classify (keeps focus stem)
        ]
        for d, suf in stage_dirs:
            if not d.exists():
                continue
            for old_stem, new_stem in rename_map.items():
                old_full = old_stem + suf
                new_full = new_stem + suf
                # TIFFs and acquisition sidecars share the same stem.
                for ext in (".tif", ".tiff", "__acquisition.json"):
                    src = d / f"{old_full}{ext}"
                    if src.exists():
                        dst = d / f"{new_full}{ext}"
                        try:
                            src.rename(dst)
                            log_cb(0.0, f"  {d.name}/  {src.name} → {dst.name}")
                        except OSError as e:
                            log_cb(0.0, f"  rename failed: {src.name} ({e})")

        # Features dir: rename sidecar files AND rewrite Parquet/CSV/h5
        # column values that embed the well stem.
        feat_dir = ctx.features_dir
        if feat_dir.exists():
            self._rename_features_outputs(feat_dir, rename_map, focus_suffix, log_cb)

    def _rename_features_outputs(
        self, feat_dir: Path,
        rename_map: dict[str, str],
        focus_suffix: str,
        log_cb,
    ) -> None:
        """Rename and rewrite Features-stage outputs in-place."""
        import pandas as pd  # local import: features is optional

        for old_stem, new_stem in rename_map.items():
            old_full = old_stem + focus_suffix
            new_full = new_stem + focus_suffix

            # Per-well Parquet: rewrite ``well`` and ``cell_uid`` columns
            # (cell_uid embeds the well stem) before renaming the file.
            old_pq = feat_dir / f"{old_full}.parquet"
            new_pq = feat_dir / f"{new_full}.parquet"
            if old_pq.exists():
                try:
                    df = pd.read_parquet(old_pq)
                    if "well" in df.columns:
                        df["well"] = df["well"].astype(str).str.replace(
                            old_full, new_full, regex=False,
                        )
                    if "cell_uid" in df.columns:
                        df["cell_uid"] = df["cell_uid"].astype(str).str.replace(
                            old_full, new_full, regex=False,
                        )
                    df.to_parquet(new_pq, index=False)
                    if old_pq != new_pq:
                        old_pq.unlink(missing_ok=True)
                    # CSV companion: regenerate from the rewritten frame to
                    # avoid stale file hanging around.
                    old_csv = feat_dir / f"{old_full}.csv"
                    new_csv = feat_dir / f"{new_full}.csv"
                    df.to_csv(new_csv, index=False)
                    if old_csv.exists() and old_csv != new_csv:
                        old_csv.unlink(missing_ok=True)
                    log_cb(0.0, f"  {feat_dir.name}/  rewrote {old_full}.parquet → {new_full}.parquet")
                except Exception as e:  # noqa: BLE001
                    log_cb(0.0, f"  features rewrite failed for {old_full}: {e}")

            # Per-well HDF5 crops: just rename the file. The cell_uids dataset
            # inside also embeds the well stem, but rewriting an HDF5 in place
            # is heavier; for now mark this as a known limitation in the log.
            old_h5 = feat_dir / f"{old_full}__crops.h5"
            if old_h5.exists():
                new_h5 = feat_dir / f"{new_full}__crops.h5"
                try:
                    old_h5.rename(new_h5)
                    log_cb(0.0,
                        f"  {feat_dir.name}/  {old_h5.name} → {new_h5.name} "
                        f"(NB: cell_uids inside still reference old well; rerun "
                        f"Features to refresh)",
                    )
                except OSError:
                    pass

        # Drop stale ``all_features.*`` and ``all_crops.h5`` so Features
        # reuse-existing repopulates them with the renamed inputs.
        for stale in ("all_features.parquet", "all_features.csv", "all_crops.h5"):
            p = feat_dir / stale
            if p.exists():
                try:
                    p.unlink()
                    log_cb(0.0, f"  {feat_dir.name}/  removed stale {stale}")
                except OSError:
                    pass

    def _migrate_legacy_dirs(self) -> None:
        """Rename pre-renumbering output dirs so old runs can be reused.

        Earlier versions used 02_focus / 03_segment / 04_classify. The
        current scheme is 01_focus (or 01_split_and_focused) / 02_segment /
        03_classify. If a user's output folder still has the old names but
        not the new ones, silently rename so reuse-existing keeps working.
        """
        out = self._ctx.output_dir
        renames = [
            ("02_focus",    self._ctx.focus_dir.name),
            ("03_segment",  self._ctx.segment_dir.name),
            ("04_classify", self._ctx.classify_dir.name),
        ]
        for old, new in renames:
            if old == new:
                continue
            old_p = out / old
            new_p = out / new
            if old_p.exists() and not new_p.exists():
                try:
                    old_p.rename(new_p)
                except OSError:
                    pass

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.runStarted.emit()
            self._migrate_legacy_dirs()

            # Layout-driven filename refresh. Detects on-disk TIFFs whose
            # filenames don't match the current layout's labels (using each
            # TIFF's __acquisition.json scene_indices to bridge old↔new) and
            # renames them in place. Runs unconditionally so a label edit
            # propagates whether reuse-existing is on or off.
            def _layout_log(_f: float, msg: str) -> None:
                self.stageProgress.emit("Split", 0.0, msg)
            try:
                self._rename_outputs_for_layout_changes(_layout_log)
            except Exception as e:  # noqa: BLE001
                self.stageProgress.emit("Split", 0.0,
                                        f"Layout-rename pass failed: {e}")

            # Validate everything up front. Skip validation for stages whose
            # output already exists on disk when reuse-existing is on — those
            # stages won't actually re-run, so their inputs (e.g. the original
            # CZI) don't need to still be present.
            issues: list[str] = []
            enabled = [s for s in self._stages if s.enabled(self._ctx)]
            self.stagesPlanned.emit([s.name for s in enabled])
            for s in enabled:
                handles_own = bool(getattr(s, "handles_own_reuse", False))
                if self._reuse_existing and not handles_own:
                    out_dir = s.output_dir(self._ctx)
                    if out_dir.exists() and (
                        any(out_dir.glob("*.tif")) or any(out_dir.glob("*.tiff"))
                    ):
                        continue
                issues.extend(s.validate(self._ctx))
            if issues:
                raise RuntimeError("Validation failed:\n  - " + "\n  - ".join(issues))

            self._ctx.output_dir.mkdir(parents=True, exist_ok=True)
            # Make reuse-existing visible to stages that do their own
            # per-input reuse logic (e.g. SplitStage / FocusStage skip a
            # CZI whose outputs are already complete but still run a
            # newly-added CZI).
            self._ctx.reuse_existing = self._reuse_existing
            manifest: dict = {
                "started_at": time.time(),
                "czi": str(self._ctx.czi_path),
                "output_dir": str(self._ctx.output_dir),
                "stages": [],
            }

            for stage in enabled:
                if self._stop_requested:
                    break
                self.stageStarted.emit(stage.name)
                t0 = time.time()

                def cb(fraction: float, message: str, name: str = stage.name) -> None:
                    if self._stop_requested:
                        raise StopRequested()
                    self.stageProgress.emit(name, fraction, message)

                # Reuse-existing: if the stage's output dir already has TIFF
                # output, skip re-running and report the existing files.
                #
                # Stages that handle their own reuse (SplitStage / FocusStage
                # do per-CZI reuse so a newly-added CZI still runs) bypass
                # this top-level shortcut.
                stage_out = stage.output_dir(self._ctx)
                reused = False
                handles_own = bool(getattr(stage, "handles_own_reuse", False))
                if self._reuse_existing and stage_out.exists() and not handles_own:
                    existing = sorted(stage_out.glob("*.tif")) + sorted(stage_out.glob("*.tiff"))
                    if existing:
                        cb(1.0, f"Reused existing output ({len(existing)} files)")
                        outputs = existing
                        reused = True

                if not reused:
                    try:
                        outputs = stage.run(self._ctx, cb)
                    except StopRequested:
                        manifest["stages"].append({
                            "name": stage.name,
                            "reused": False,
                            "stopped": True,
                            "elapsed_s": round(time.time() - t0, 2),
                            "opts": _stage_opts(self._ctx, stage.name),
                            "outputs": [],
                        })
                        self.stageFinished.emit(stage.name, 0)
                        break

                manifest["stages"].append({
                    "name": stage.name,
                    "reused": reused,
                    "elapsed_s": round(time.time() - t0, 2),
                    "opts": _stage_opts(self._ctx, stage.name),
                    "outputs": [str(p) for p in outputs],
                })
                self.stageFinished.emit(stage.name, len(outputs))

                # Focus subsumes Split — when both flags are on, Focus writes
                # per-well TIFFs to ``01_split_and_focused`` (which is the
                # value of ``focus_dir`` in that case) instead of producing a
                # separate ``01_split`` dir. Either path means the Split work
                # is done; mark it completed in the stepper so the user sees
                # the byproduct. Skipped when ``do_split`` was off — in that
                # case Split is not part of the run and shouldn't turn green.
                if stage.name == "Focus" and self._ctx.do_split:
                    candidates: list[Path] = []
                    if self._ctx.split_dir.exists():
                        candidates.append(self._ctx.split_dir)
                    if (self._ctx.focus_dir.exists()
                            and self._ctx.focus_dir != self._ctx.split_dir):
                        candidates.append(self._ctx.focus_dir)
                    split_outputs: list[Path] = []
                    for d in candidates:
                        split_outputs.extend(sorted(d.glob("*.tif")))
                        split_outputs.extend(sorted(d.glob("*.tiff")))
                    if split_outputs:
                        self.stageFinished.emit("Split", len(split_outputs))

            manifest["finished_at"] = time.time()
            manifest["stopped_early"] = self._stop_requested
            # Persist the plate layout alongside the manifest
            layout_csv = self._ctx.output_dir / "plate_layout.csv"
            self._ctx.layout.to_csv(layout_csv)
            manifest["plate_layout_csv"] = str(layout_csv)

            self._ctx.manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
            self.runFinished.emit(self._ctx.manifest_path)

        except Exception as e:  # noqa: BLE001
            self.runFailed.emit(str(e))
        finally:
            if self._thread is not None:
                self._thread.quit()


class BulkPipelineRunner(QObject):
    """Runs Focus once per CZI, then Segment + Classify once over the
    accumulated TIFF directory. Mirrors PipelineRunner's signal surface
    so the Run panel can stay agnostic about which mode is active.
    """

    runStarted = pyqtSignal()
    stagesPlanned = pyqtSignal(list)
    stageStarted = pyqtSignal(str)
    stageProgress = pyqtSignal(str, float, str)
    stageFinished = pyqtSignal(str, int)
    runFinished = pyqtSignal(Path)
    runFailed = pyqtSignal(str)

    def __init__(self, ctx: BulkRunContext, reuse_existing: bool = False) -> None:
        super().__init__()
        self._ctx = ctx
        self._thread: QThread | None = None
        self._reuse_existing = reuse_existing
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def start(self) -> None:
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.start()

    # --------------------------------------------------------------------------

    def _run(self) -> None:
        from mycoprep.core.api import run_focus

        try:
            self.runStarted.emit()
            ctx = self._ctx
            # Mirror the runner's reuse-existing flag onto the context so
            # the per-input reuse logic inside SegmentStage / ClassifyStage
            # / ExtractStage (which is shared with the plate runner) can
            # consult ``ctx.reuse_existing`` uniformly.
            ctx.reuse_existing = bool(self._reuse_existing)

            entries = list(ctx.czi_entries)
            if not entries:
                raise RuntimeError("Validation failed: no CZIs with conditions assigned.")

            # Existence check up front so we don't half-run.
            issues: list[str] = []
            for e in entries:
                p = Path(e["czi_path"])
                if not p.exists():
                    issues.append(f"CZI not found: {p}")
            if issues:
                raise RuntimeError("Validation failed:\n  - " + "\n  - ".join(issues))

            ctx.output_dir.mkdir(parents=True, exist_ok=True)
            manifest: dict = {
                "started_at": time.time(),
                "mode": "bulk",
                "output_dir": str(ctx.output_dir),
                "stages": [],
            }

            planned: list[str] = []
            if ctx.do_focus:
                planned.append("Focus")
            if ctx.do_segment:
                planned.append("Segment")
            if ctx.do_classify:
                planned.append("Classify")
            if getattr(ctx, "do_features", False):
                planned.append("Features")
            self.stagesPlanned.emit(planned)

            # ── Focus stage: per-CZI ──────────────────────────────────
            if ctx.do_focus:
                if self._stop_requested:
                    self._finalise(manifest, stopped=True)
                    return

                self.stageStarted.emit("Focus")
                t0 = time.time()
                ctx.focus_dir.mkdir(parents=True, exist_ok=True)
                focus_outputs: list[Path] = []
                n = len(entries)

                for i, entry in enumerate(entries):
                    if self._stop_requested:
                        break

                    label = _output_filename(
                        str(entry.get("condition", "")),
                        str(entry.get("reporter", "")),
                        str(entry.get("mutant_or_drug", "")),
                        str(entry.get("replica", "")),
                    ).removesuffix(".tif")

                    out_file = ctx.focus_dir / f"{label}{ctx.focus_opts.filename_suffix}.tif"

                    if self._reuse_existing and out_file.exists():
                        cb_msg = f"[{i+1}/{n}] Reused existing {out_file.name}"
                        self.stageProgress.emit("Focus", (i + 1) / n, cb_msg)
                        focus_outputs.append(out_file)
                        continue

                    def per_czi_cb(frac: float, msg: str, _i=i, _n=n,
                                   _name=Path(entry["czi_path"]).name) -> None:
                        if self._stop_requested:
                            raise StopRequested()
                        outer_frac = (_i + frac) / _n
                        self.stageProgress.emit(
                            "Focus", outer_frac, f"[{_i+1}/{_n}] {_name}: {msg}",
                        )

                    try:
                        run_focus(
                            czi_path=Path(entry["czi_path"]),
                            out_dir=ctx.focus_dir,
                            opts=ctx.focus_opts,
                            single_bucket_label=label,
                            progress_cb=per_czi_cb,
                        )
                    except StopRequested:
                        break
                    focus_outputs.append(out_file)

                manifest["stages"].append({
                    "name": "Focus",
                    "elapsed_s": round(time.time() - t0, 2),
                    "opts": _stage_opts_bulk(ctx, "Focus"),
                    "outputs": [str(p) for p in focus_outputs],
                })
                self.stageFinished.emit("Focus", len(focus_outputs))

            # ── Segment + Classify + Features: reuse existing stage objects ────
            for stage_obj in (SegmentStage(), ClassifyStage(), ExtractStage()):
                if self._stop_requested:
                    break
                if not stage_obj.enabled(ctx):  # type: ignore[arg-type]
                    continue

                self.stageStarted.emit(stage_obj.name)
                t0 = time.time()

                def cb(fraction: float, message: str,
                       name: str = stage_obj.name) -> None:
                    if self._stop_requested:
                        raise StopRequested()
                    self.stageProgress.emit(name, fraction, message)

                # Reuse-existing: skip if outputs already present.
                stage_out = stage_obj.output_dir(ctx)  # type: ignore[arg-type]
                reused = False
                outputs: list[Path] = []
                if self._reuse_existing and stage_out.exists():
                    existing = _iter_tiffs(stage_out)
                    if existing:
                        cb(1.0, f"Reused existing output ({len(existing)} files)")
                        outputs = existing
                        reused = True

                if not reused:
                    try:
                        outputs = stage_obj.run(ctx, cb)  # type: ignore[arg-type]
                    except StopRequested:
                        manifest["stages"].append({
                            "name": stage_obj.name,
                            "stopped": True,
                            "elapsed_s": round(time.time() - t0, 2),
                            "opts": _stage_opts_bulk(ctx, stage_obj.name),
                            "outputs": [],
                        })
                        self.stageFinished.emit(stage_obj.name, 0)
                        break

                manifest["stages"].append({
                    "name": stage_obj.name,
                    "reused": reused,
                    "elapsed_s": round(time.time() - t0, 2),
                    "opts": _stage_opts_bulk(ctx, stage_obj.name),
                    "outputs": [str(p) for p in outputs],
                })
                self.stageFinished.emit(stage_obj.name, len(outputs))

            self._finalise(manifest, stopped=self._stop_requested)

        except Exception as e:  # noqa: BLE001
            self.runFailed.emit(str(e))
        finally:
            if self._thread is not None:
                self._thread.quit()

    def _finalise(self, manifest: dict, stopped: bool) -> None:
        manifest["finished_at"] = time.time()
        manifest["stopped_early"] = stopped
        # Persist the entry list alongside the manifest for traceability.
        try:
            entries_csv = self._ctx.output_dir / "bulk_entries.csv"
            import csv as _csv
            with open(entries_csv, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["czi_path", "condition", "reporter",
                            "mutant_or_drug", "replica", "notes"])
                for e in self._ctx.czi_entries:
                    w.writerow([e.get(k, "") for k in (
                        "czi_path", "condition", "reporter",
                        "mutant_or_drug", "replica", "notes",
                    )])
            manifest["bulk_entries_csv"] = str(entries_csv)
        except Exception:  # noqa: BLE001
            pass
        self._ctx.manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        self.runFinished.emit(self._ctx.manifest_path)


def _stage_opts_bulk(ctx: BulkRunContext, name: str) -> dict:
    opts_map: dict = {
        "Focus": ctx.focus_opts,
        "Segment": ctx.segment_opts,
        "Classify": ctx.classify_opts,
        "Features": getattr(ctx, "features_opts", None),
    }
    o = opts_map.get(name)
    if o is None:
        return {}
    if is_dataclass(o):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(o).items()}
    return {}


def _stage_opts(ctx: RunContext, name: str) -> dict:
    """Serialise the options dataclass for a given stage into the manifest."""
    opts_map = {
        "Focus": ctx.focus_opts,
        "Segment": ctx.segment_opts,
        "Classify": ctx.classify_opts,
        "Features": getattr(ctx, "features_opts", None),
    }
    o = opts_map.get(name)
    if o is None:
        return {}
    if is_dataclass(o):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(o).items()}
    return {}
