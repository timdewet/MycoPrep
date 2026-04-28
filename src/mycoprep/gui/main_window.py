"""Main window — sidebar nav across stages, orchestrates RunContext + PipelineRunner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt6.QtCore import QByteArray, QSettings, QSize, Qt as _Qt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .panels.features_panel import FeaturesPanel
from .panels.input_panel import InputMode, InputPanel
from .panels.label_train_panel import LabelTrainPanel
from .panels.layout_panel import LayoutPanel
from .panels.run_panel import RunPanel
from .panels.stage_panels import FocusPanel, SegmentClassifyPanel
from .pipeline.context import BulkRunContext, RunContext
from .pipeline.runner import BulkPipelineRunner, PipelineRunner
from .ui import icons, tokens
from .ui.elevation import apply_shadow
from .ui.nav_sidebar import NavEntry, NavSidebar, StageStatus
from .widgets.live_preview import LivePreviewPanel


def _gpu_available() -> bool:
    """Best-effort GPU probe for the preflight check.

    Returns True when either CUDA (NVIDIA) or MPS (Apple Silicon) is usable.
    """
    try:
        import torch  # type: ignore
    except Exception:  # noqa: BLE001
        return False
    try:
        if torch.cuda.is_available():
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


# Stage keys used by the sidebar and the (key, panel, label) registry.
NAV_ENTRIES = [
    NavEntry("input",    "Input",              "input"),
    NavEntry("plate",    "Plate layout",       "plate"),
    NavEntry("focus",    "Focus",              "focus"),
    NavEntry("segment",  "Segment & Classify", "segment"),
    NavEntry("features", "Features",           "features"),
    NavEntry("run",      "Run",                "run"),
]


class _LabelTrainerWindow(QMainWindow):
    def __init__(self, panel: LabelTrainPanel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Labeller & Classifier Trainer")
        self.resize(1100, 800)
        self.setCentralWidget(panel)


class _ModelDetailsWindow(QMainWindow):
    def __init__(self, inspector, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Classifier Model Details")
        self.resize(720, 380)
        self.setCentralWidget(inspector)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MycoPrep")
        from ._resources import resource_root
        logo_path = resource_root() / "logo" / "logo.svg"
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))
        self._logo_path = logo_path
        # Aim for enough room to show the 12-column plate uncompressed, but
        # cap to 90% of the available screen so the window fits on smaller
        # (non-retina) laptop displays without overflowing.
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        target_w, target_h = 1600, 1080
        if avail is not None:
            target_w = min(target_w, int(avail.width() * 0.9))
            target_h = min(target_h, int(avail.height() * 0.9))
        self.resize(target_w, target_h)

        self._runner: PipelineRunner | None = None
        self._labeller_window: _LabelTrainerWindow | None = None
        self._model_details_window: _ModelDetailsWindow | None = None

        self.input_panel = InputPanel()
        self.layout_panel = LayoutPanel()
        self.focus_panel = FocusPanel()
        self.segclass_panel = SegmentClassifyPanel()
        self.features_panel = FeaturesPanel()
        self.run_panel = RunPanel()
        self.label_train_panel = LabelTrainPanel()
        self.live_preview = LivePreviewPanel()

        # Convenience aliases — many other places still reference these.
        self.segment_panel = self.segclass_panel.segment_panel
        self.classify_panel = self.segclass_panel.classify_panel

        # Wire panel signals BEFORE building chrome so callbacks can reach widgets.
        # cziPathsSelected fires the multi-CZI list (single-CZI runs send
        # a 1-element list). Connect it BEFORE cziSelected so the layout
        # panel populates from the FULL list before single-CZI consumers
        # (focus/segment) refresh — those only need channel metadata, not
        # the layout. Falls back to the single-CZI signal for legacy callers.
        self.input_panel.cziPathsSelected.connect(self.layout_panel.set_czi_paths)
        self.input_panel.cziPathsSelected.connect(lambda _ps: self._refresh_preview_dirs())
        self.input_panel.cziSelected.connect(self._on_czi_selected)
        self.input_panel.outputDirSelected.connect(self.layout_panel.set_output_dir)
        self.input_panel.outputDirSelected.connect(lambda _p: self._refresh_preview_dirs())
        self.input_panel.outputDirSelected.connect(lambda _p: self._refresh_input_status())
        self.layout_panel.layoutValidityChanged.connect(self._on_layout_validity_changed)
        self.input_panel.resetRequested.connect(self._on_input_reset)
        self.input_panel.channelsChanged.connect(self._on_channels_changed)
        self.input_panel.modeChanged.connect(self._on_input_mode_changed)
        self.input_panel.modeChanged.connect(lambda _m: self._refresh_preview_layouts())
        # Sample labels in the live preview pull condition / reporter
        # / mutant from the plate layout — refresh whenever the user
        # edits a well.
        self.layout_panel._editor.layoutChanged.connect(self._refresh_preview_layouts)
        # Same for bulk-mode label edits (single-file mode also writes
        # through the bulk table model).
        self.input_panel._bulk_model.dataChanged.connect(
            lambda *_args: self._refresh_preview_layouts()
        )
        self.input_panel._bulk_model.modelReset.connect(self._refresh_preview_layouts)
        self.run_panel.runRequested.connect(self._on_run_requested)
        self.run_panel.stageEnablesChanged.connect(self._recompute_stage_readiness)
        self.classify_panel.openLabelTrainerRequested.connect(self._open_label_trainer)
        self.classify_panel.showModelDetailsRequested.connect(self._open_model_details)

        # Stage-state tracking for sidebar status dots.
        self._stage_status: dict[str, StageStatus] = {e.key: StageStatus.IDLE for e in NAV_ENTRIES}
        self._run_active = False
        self.run_panel.stageRunStarted.connect(self._on_stage_run_started)
        self.run_panel.stageRunFinished.connect(self._on_stage_run_finished)
        self.run_panel.runFinishedAll.connect(self._on_run_all_finished)
        self.run_panel.runFailedAll.connect(self._on_run_all_failed)

        self._build_chrome()
        self._apply_card_shadows()
        self._wire_live_preview()
        self._on_input_mode_changed(self.input_panel.mode)
        self._restore_state()
        self._refresh_input_status()

    # ------------------------------------------------------------------ live preview

    def _wire_live_preview(self) -> None:
        """Hand the live preview the option providers + change signals
        it needs to drive the per-FOV pipeline."""
        self.live_preview.wire_pipeline(
            focus_opts=self.focus_panel.opts,
            segment_opts=self.segment_panel.opts,
            classify_opts=self.classify_panel.opts,
            features_opts=self.features_panel.opts,
            # Pass the InputPanel's phase_channel through verbatim —
            # ``None`` is the "auto-detect by skewness" sentinel, which
            # the worker resolves from the actual image data once focus
            # has loaded it.
            phase_channel=lambda: self.input_panel.phase_channel,
            channel_labels=lambda: self.input_panel.channel_labels,
        )
        # Re-render on any option change in the four stage panels.
        self.focus_panel.optionsChanged.connect(self.live_preview.trigger_render)
        self.segment_panel.optionsChanged.connect(self.live_preview.trigger_render)
        self.classify_panel.optionsChanged.connect(self.live_preview.trigger_render)
        self.features_panel.optionsChanged.connect(self.live_preview.trigger_render)

    # ------------------------------------------------------------------ chrome

    def _build_chrome(self) -> None:
        # Header
        self._header = QFrame()
        self._header.setObjectName("header")
        self._header.setFixedHeight(tokens.HEADER_HEIGHT)
        hl = QHBoxLayout(self._header)
        hl.setContentsMargins(tokens.S5, 0, tokens.S5, 0)
        hl.setSpacing(tokens.S3)

        self._breadcrumb = QLabel(f"Step 1 of {len(NAV_ENTRIES)} · Input")
        self._breadcrumb.setObjectName("breadcrumb")
        hl.addWidget(self._breadcrumb)
        hl.addStretch(1)

        brand_name = QLabel("MycoPrep")
        brand_name.setObjectName("brandName")
        hl.addWidget(brand_name)

        if self._logo_path.exists():
            logo_lbl = QLabel()
            logo_lbl.setFixedSize(QSize(50, 50))
            renderer = QSvgRenderer(str(self._logo_path))
            pix = QPixmap(QSize(50, 50))
            pix.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pix)
            renderer.render(painter)
            painter.end()
            logo_lbl.setPixmap(pix)
            logo_lbl.setStyleSheet("background: transparent;")
            hl.addWidget(logo_lbl)

        header_div = QFrame()
        header_div.setObjectName("headerDivider")

        # Sidebar + stack
        self._sidebar = NavSidebar(NAV_ENTRIES)
        self._sidebar.set_gpu_available(_gpu_available())
        self._sidebar.currentChanged.connect(self._on_nav_changed)

        from PyQt6.QtWidgets import QScrollArea, QSizePolicy

        self._stack = QStackedWidget()
        self._stack.setContentsMargins(0, 0, 0, 0)
        # Pages that should fill the page (their own widgets manage stretch).
        FILL_KEYS = {"plate", "segment", "run"}
        for entry, panel in zip(NAV_ENTRIES, [
            self.input_panel,
            self.layout_panel,
            self.focus_panel,
            self.segclass_panel,
            self.features_panel,
            self.run_panel,
        ]):
            wrap = QWidget()
            wv = QVBoxLayout(wrap)
            wv.setContentsMargins(tokens.S5, tokens.S5, tokens.S5, tokens.S5)
            wv.setSpacing(tokens.S3)
            if entry.key in FILL_KEYS:
                # Plate/Segment/Run own their full vertical layout — give them all the space.
                wv.addWidget(panel, stretch=1)
            else:
                # Forms (Input, Focus): wrap in a scroll area pinned to the top so
                # cards don't stretch to fill empty vertical space.
                scroll = QScrollArea()
                scroll.setFrameShape(scroll.Shape.NoFrame)
                scroll.setWidgetResizable(True)
                scroll.setHorizontalScrollBarPolicy(_Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                inner = QWidget()
                iv = QVBoxLayout(inner)
                iv.setContentsMargins(0, 0, 0, 0)
                iv.setSpacing(tokens.S3)
                panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
                iv.addWidget(panel)
                iv.addStretch(1)
                scroll.setWidget(inner)
                wv.addWidget(scroll)
            self._stack.addWidget(wrap)

        # Body row: sidebar | divider | (stack | live preview) inside a
        # horizontal QSplitter so the user can drag the boundary between
        # the option column and the preview column. The sidebar stays
        # fixed-width (a normal QHBoxLayout child) — only the stack and
        # the preview share the splitter.
        body = QWidget()
        body_l = QHBoxLayout(body)
        body_l.setContentsMargins(0, 0, 0, 0)
        body_l.setSpacing(0)
        body_l.addWidget(self._sidebar)
        divider = QFrame()
        divider.setObjectName("sidebarDivider")
        divider.setFrameShape(QFrame.Shape.NoFrame)
        body_l.addWidget(divider)

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.setHandleWidth(1)
        self._main_splitter.addWidget(self._stack)
        self._main_splitter.addWidget(self.live_preview)
        # Default split: ~45% options, ~55% preview. Restored from
        # QSettings in _restore_state if a saved width is available.
        self._main_splitter.setSizes([520, 640])
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        body_l.addWidget(self._main_splitter, stretch=1)

        # Compose
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._header)
        v.addWidget(header_div)
        v.addWidget(body, stretch=1)
        self.setCentralWidget(wrap)

        # Status bar
        sb = QStatusBar()
        self._status_msg = QLabel("Ready.")
        self._status_msg.setObjectName("muted")
        sb.addWidget(self._status_msg, 1)
        self._status_outdir = QLabel("")
        self._status_outdir.setObjectName("caption")
        sb.addPermanentWidget(self._status_outdir)
        self.setStatusBar(sb)

    # ------------------------------------------------------------------ elevation

    def _apply_card_shadows(self) -> None:
        """Attach a subtle drop shadow to every card surface and tighten card heights."""
        from PyQt6.QtWidgets import QFormLayout, QFrame, QGroupBox, QSizePolicy
        for panel in (
            self.input_panel,
            self.layout_panel,
            self.focus_panel,
            self.segclass_panel,
            self.features_panel,
            self.run_panel,
            self.label_train_panel,
        ):
            for box in panel.findChildren(QGroupBox):
                apply_shadow(box, level=1)
                # Pin card height to its content height so short cards don't
                # stretch into seas of empty space.
                box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
            for frame in panel.findChildren(QFrame):
                if frame.objectName() == "card":
                    apply_shadow(frame, level=1)
                    frame.setSizePolicy(
                        QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
                    )
            # Make form fields (and our helper-wrapped rows) grow horizontally so
            # captions can wrap properly instead of eliding.
            for form in panel.findChildren(QFormLayout):
                form.setFieldGrowthPolicy(
                    QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
                )

    # ------------------------------------------------------------------ nav

    def _on_nav_changed(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)
        if not (0 <= idx < len(NAV_ENTRIES)):
            return
        entry = NAV_ENTRIES[idx]
        # The sidebar tracks which keys are visible (e.g. plate is hidden in
        # bulk mode); use that to drive the step counter.
        visible_entries = [e for e in NAV_ENTRIES if e.key in self._sidebar._visible_keys]
        visible_idx = next(
            (i + 1 for i, e in enumerate(visible_entries) if e.key == entry.key),
            1,
        )
        self._breadcrumb.setText(
            f"Step {visible_idx} of {len(visible_entries)} · {entry.label}"
        )
        # Live preview: visible only on the focus / segment / features
        # tabs, hidden on input / plate / run.
        from .widgets.live_preview.panel import RENDER_TAB_KEYS
        show_preview = entry.key in RENDER_TAB_KEYS
        self.live_preview.setVisible(show_preview)
        if show_preview:
            self.live_preview.set_current_tab(entry.key)

    # ------------------------------------------------------------------ persistence

    def _settings(self) -> QSettings:
        return QSettings("MMRU", "MycoPrep")

    def _save_state(self) -> None:
        s = self._settings()
        try:
            s.setValue("window/geometry", self.saveGeometry())
            s.setValue("window/state", self.saveState())
            s.setValue("window/nav_index", self._stack.currentIndex())
            s.setValue("window/main_splitter", self._main_splitter.saveState())
            s.setValue("input_panel", json.dumps(self.input_panel.state()))
            s.setValue("focus_panel", json.dumps(self.focus_panel.state()))
            s.setValue("segment_panel", json.dumps(self.segment_panel.state()))
            s.setValue("classify_panel", json.dumps(self.classify_panel.state()))
            s.setValue("features_panel", json.dumps(self.features_panel.state()))
            s.setValue("run_panel", json.dumps(self.run_panel.state()))
            s.setValue("live_preview", json.dumps(self.live_preview.state()))
        except Exception as e:  # noqa: BLE001
            print(f"[mycoprep] Failed to save settings: {e}", file=sys.stderr)

    def _restore_state(self) -> None:
        s = self._settings()
        try:
            geom = s.value("window/geometry")
            if isinstance(geom, QByteArray) and not geom.isEmpty():
                self.restoreGeometry(geom)
        except Exception as e:  # noqa: BLE001
            print(f"[mycoprep] geometry restore failed: {e}", file=sys.stderr)
        try:
            wstate = s.value("window/state")
            if isinstance(wstate, QByteArray) and not wstate.isEmpty():
                self.restoreState(wstate)
        except Exception as e:  # noqa: BLE001
            print(f"[mycoprep] window state restore failed: {e}", file=sys.stderr)
        try:
            split = s.value("window/main_splitter")
            if isinstance(split, QByteArray) and not split.isEmpty():
                self._main_splitter.restoreState(split)
        except Exception as e:  # noqa: BLE001
            print(f"[mycoprep] splitter restore failed: {e}", file=sys.stderr)
        try:
            raw = s.value("live_preview")
            if raw:
                self.live_preview.restore_state(json.loads(raw))
        except Exception as e:  # noqa: BLE001
            print(f"[mycoprep] live_preview restore failed: {e}", file=sys.stderr)
        for key, panel in (
            ("input_panel",    self.input_panel),
            ("focus_panel",    self.focus_panel),
            ("segment_panel",  self.segment_panel),
            ("classify_panel", self.classify_panel),
            ("features_panel", self.features_panel),
            ("run_panel",      self.run_panel),
        ):
            raw = s.value(key)
            if not raw:
                continue
            try:
                panel.restore_state(json.loads(raw))
            except Exception as e:  # noqa: BLE001
                print(f"[mycoprep] {key} restore failed: {e}", file=sys.stderr)
        # InputPanel.restore_state emits outputDirSelected (which auto-loads
        # plate_layout.csv) BEFORE cziSelected (which calls
        # LayoutPanel.set_czi_path → PlateLayout.from_czi, building an
        # empty-conditions layout that clobbers the CSV-loaded one). Re-import
        # the CSV here, after both have fired, so the previous run's plate
        # layout (conditions / reporters / mutants) survives app restart.
        out = self.input_panel.output_dir
        if out is not None and Path(out).exists():
            self.layout_panel.set_output_dir(out)

        # Now that all panels and the output dir are restored, paint the
        # initial sidebar readiness state.
        self._recompute_stage_readiness()

        # Always start on the Input tab — it's the natural beginning of
        # the workflow and avoids reopening on a stale stage from last session.
        input_idx = self._sidebar.index_of("input")
        if input_idx >= 0:
            self._sidebar.set_current(input_idx)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._save_state()
        super().closeEvent(event)

    # ---------------------------------------------------------------- preflight

    def _preflight_ok(self, out: Path, enables: dict[str, bool]) -> bool:
        blocking: list[str] = []
        warnings: list[str] = []

        if enables.get("Classify") and not enables.get("Segment"):
            seg_dir = out / "02_segment"
            if not seg_dir.exists() or not any(seg_dir.iterdir()):
                blocking.append(
                    "Classify is enabled but Segment is disabled and no "
                    f"segment output exists at {seg_dir}."
                )

        if enables.get("Segment") and not enables.get("Focus"):
            focus_candidates = [
                out / "01_split_and_focused",
                out / "01_focus",
                out / "01_split",
            ]
            has_focus = any(d.exists() and any(d.iterdir()) for d in focus_candidates)
            if not has_focus:
                blocking.append(
                    "Segment is enabled but Focus is disabled and no focused "
                    "output exists in any of:\n  "
                    + "\n  ".join(str(d) for d in focus_candidates)
                )

        if enables.get("Features") and not (
            enables.get("Segment") or enables.get("Classify")
        ):
            features_candidates = [out / "03_classify", out / "02_segment"]
            has_input = any(d.exists() and any(d.iterdir()) for d in features_candidates)
            if not has_input:
                blocking.append(
                    "Features is enabled but neither Segment nor Classify is "
                    "enabled, and no upstream output exists in any of:\n  "
                    + "\n  ".join(str(d) for d in features_candidates)
                )

        if enables.get("Segment") and self.segment_panel.opts().gpu:
            if not _gpu_available():
                warnings.append(
                    "GPU requested for segmentation but no CUDA or MPS device "
                    "was found; the run will fall back to CPU (much slower)."
                )

        if (out.exists() and any(out.iterdir())
                and not self.run_panel.reuse_existing.isChecked()):
            warnings.append(
                f"Output directory already contains files and 'Reuse existing' "
                f"is off. Existing files in {out} may be overwritten."
            )

        if self.input_panel.mode in (InputMode.SINGLE_FILE, InputMode.BULK):
            issues = self.input_panel.bulk_layout.validate()
            if issues:
                blocking.extend(issues)

        if blocking:
            QMessageBox.critical(self, "Cannot run", "\n\n".join(blocking))
            return False

        if warnings:
            for w in warnings:
                self.run_panel.log.log(f"⚠ {w}", level="warning")
            answer = QMessageBox.question(
                self, "Run with warnings?",
                "\n\n".join(warnings) + "\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        return True

    # ---------------------------------------------------------------- slots

    def _on_input_mode_changed(self, mode: InputMode) -> None:
        """Plate-layout step is only meaningful in single-plate mode."""
        plate_idx = self._sidebar.index_of("plate")
        visible = mode == InputMode.SINGLE_PLATE
        self._sidebar.set_visible("plate", visible)
        # Hide the corresponding stack page so navigation stays consistent.
        if 0 <= plate_idx < self._stack.count():
            self._stack.widget(plate_idx).setEnabled(visible)
            # If plate is currently selected but no longer reachable, jump to Input.
            if not visible and self._stack.currentIndex() == plate_idx:
                self._sidebar.set_current(self._sidebar.index_of("input"))
        self._refresh_input_status()
        # Step counter changes when the visible set changes.
        self._on_nav_changed(self._stack.currentIndex())

    def _on_layout_validity_changed(self, ok: bool) -> None:
        self._sidebar.set_status("plate", StageStatus.DONE if ok else StageStatus.IDLE)
        self._recompute_stage_readiness()

    def _on_input_reset(self) -> None:
        # Wipe downstream state so the user really is starting from scratch.
        self.layout_panel.clear()
        self._refresh_input_status()
        self._refresh_preview_dirs()

    def _refresh_input_status(self) -> None:
        ready = (
            self.input_panel.output_dir is not None
            and self.input_panel.has_czi_input
        )
        self._sidebar.set_status("input", StageStatus.DONE if ready else StageStatus.IDLE)
        self._recompute_stage_readiness()

    # ---------------------------------------------------------------- readiness

    def _recompute_stage_readiness(self) -> None:
        """Drive the sidebar status dots for the configurable stages.

        Green = "this step won't block the Run" — either the stage is enabled
        in the Run panel (and so will execute) or its output already exists on
        disk from a prior run (and so will be reused). Stages that are disabled
        AND have no prior output show as DISABLED so the user can see they'll
        be skipped.

        Input/Plate keep their existing validation-based rules (set elsewhere);
        Run keeps its run-completion rule. Skipped during/after a live run so
        the runner's own status updates don't get clobbered.
        """
        if self._run_active:
            return

        out = self.input_panel.output_dir
        enables = self.run_panel.stage_enables()

        # Stage → predicate that returns True when prior output exists.
        def _has_tiffs(d: Path) -> bool:
            return d.exists() and (
                any(d.glob("*.tif")) or any(d.glob("*.tiff"))
            )

        def _has_parquet(d: Path) -> bool:
            return d.exists() and any(d.glob("*.parquet"))

        focus_existing = (
            out is not None
            and (_has_tiffs(out / "01_focus")
                 or _has_tiffs(out / "01_split_and_focused")
                 or _has_tiffs(out / "01_split"))
        )
        segment_existing = out is not None and _has_tiffs(out / "02_segment")
        # Classify shares the Segment & Classify panel; treat its readiness
        # via the same sidebar dot.
        classify_existing = out is not None and _has_tiffs(out / "03_classify")
        features_existing = out is not None and _has_parquet(out / "04_features")

        focus_ready = enables.get("Focus", False) or focus_existing
        # The Segment & Classify tab is one entry in the sidebar — green if
        # either Segment or Classify is enabled, or either's output is on disk.
        segclass_ready = (
            enables.get("Segment", False)
            or enables.get("Classify", False)
            or segment_existing
            or classify_existing
        )
        features_ready = enables.get("Features", False) or features_existing

        self._sidebar.set_status(
            "focus",
            StageStatus.DONE if focus_ready else StageStatus.DISABLED,
        )
        self._sidebar.set_status(
            "segment",
            StageStatus.DONE if segclass_ready else StageStatus.DISABLED,
        )
        self._sidebar.set_status(
            "features",
            StageStatus.DONE if features_ready else StageStatus.DISABLED,
        )

    def _on_channels_changed(self) -> None:
        self.focus_panel.set_phase_channel(self.input_panel.phase_channel)
        ch = self.input_panel.phase_channel
        ch_int = ch if isinstance(ch, int) else 0
        self.label_train_panel.set_phase_channel(ch_int)
        self.live_preview.set_phase_channel(ch_int)
        labels = self.input_panel.channel_labels or []
        self.features_panel.set_channels(list(labels))

    def _on_czi_selected(self, czi_path) -> None:
        try:
            from mycoprep.core.split_czi_plate import _read_czi_pixels_per_um
            px = _read_czi_pixels_per_um(czi_path)
        except Exception:  # noqa: BLE001
            px = None
        self.segclass_panel.set_detected_pixels_per_um(px)
        self._refresh_preview_dirs()
        self._refresh_input_status()

    def _refresh_preview_dirs(self) -> None:
        out = self.input_panel.output_dir
        if out is not None:
            self._status_outdir.setText(str(out))
        else:
            self._status_outdir.setText("")
        # Always feed the live preview the up-to-date list of CZIs from
        # the Input panel — the Focus tab pulls samples from this list.
        self.live_preview.set_czi_paths(self._all_input_czi_paths())
        # And the layout context, so samples can be labelled by
        # well / condition / reporter rather than raw filename.
        self._refresh_preview_layouts()
        if out is None:
            self.live_preview.set_search_dirs([])
            self.label_train_panel.set_segment_dir(None)
            return
        candidates = [
            out / "01_split_and_focused",
            out / "01_focus",
            out / "01_split",
        ]
        self.live_preview.set_search_dirs([d for d in candidates if d.exists()])
        seg_dir = out / "02_segment"
        self.label_train_panel.set_segment_dir(seg_dir if seg_dir.exists() else None)

    def _refresh_preview_layouts(self) -> None:
        """Hand the live preview the current plate / bulk layout DataFrames.

        Called whenever any of the layouts change so the sample combo
        re-renders with the latest condition / reporter / mutant labels.

        Important: read ``self.input_panel._bulk_layout`` (the private
        attribute), not the public ``bulk_layout`` property. The latter
        calls ``_sync_single_file_to_layout()``, which fires
        ``_bulk_model.modelReset`` — the same signal that triggers
        this method — and creates an infinite loop.
        """
        try:
            plate_df = self.layout_panel.layout_model.df
        except Exception:  # noqa: BLE001
            plate_df = None
        try:
            bulk_df = self.input_panel._bulk_layout.df
        except Exception:  # noqa: BLE001
            bulk_df = None
        self.live_preview.set_layouts(
            self.input_panel.mode.value, plate_df, bulk_df,
        )

    def _all_input_czi_paths(self) -> list[Path]:
        """Aggregate every CZI the user has lined up across input modes.

        Single-plate uses the plate CZI list; single-file/bulk pull from
        the bulk layout. Used to seed the live preview's Focus-tab
        sample combo regardless of input mode.
        """
        mode = self.input_panel.mode
        if mode == InputMode.SINGLE_PLATE:
            return [Path(p) for p in self.input_panel.czi_paths]
        try:
            df = self.input_panel.bulk_layout.df
            return [Path(p) for p in df["czi_path"].astype(str) if p]
        except Exception:  # noqa: BLE001
            return []

    def _open_label_trainer(self) -> None:
        if self._labeller_window is None:
            self._labeller_window = _LabelTrainerWindow(self.label_train_panel, parent=self)
        self._labeller_window.show()
        self._labeller_window.raise_()
        self._labeller_window.activateWindow()

    def _open_model_details(self) -> None:
        if self._model_details_window is None:
            self._model_details_window = _ModelDetailsWindow(
                self.classify_panel.inspector, parent=self,
            )
        self._model_details_window.show()
        self._model_details_window.raise_()
        self._model_details_window.activateWindow()

    # ---------------------------------------------------------------- run state -> sidebar

    def _on_stage_run_started(self, name: str) -> None:
        self._run_active = True
        self._sidebar.set_status("run", StageStatus.RUNNING)
        self._status_msg.setText(f"Running: {name}")

    def _on_stage_run_finished(self, name: str, _n: int) -> None:
        self._status_msg.setText(f"Finished: {name}")

    def _on_run_all_finished(self, _manifest: object) -> None:
        self._run_active = False
        self._sidebar.set_status("run", StageStatus.DONE)
        self._status_msg.setText("Run complete.")
        # Outputs now exist on disk for stages that just ran — re-evaluate so
        # any newly-produced output sets its sidebar dot green.
        self._recompute_stage_readiness()

    def _on_run_all_failed(self, msg: str) -> None:
        self._run_active = False
        self._sidebar.set_status("run", StageStatus.ERROR)
        self._status_msg.setText(f"Failed: {msg}")
        self._recompute_stage_readiness()

    def _on_run_requested(self) -> None:
        out = self.input_panel.output_dir
        if out is None:
            QMessageBox.warning(self, "Missing output", "Select an output directory.")
            return

        enables = self.run_panel.stage_enables()
        mode = self.input_panel.mode

        if not self._preflight_ok(out, enables):
            return

        if mode == InputMode.SINGLE_PLATE:
            czi = self.input_panel.czi_path
            czi_paths = self.input_panel.czi_paths
            if czi is None or not czi_paths:
                QMessageBox.warning(self, "Missing input", "Select a CZI file.")
                return
            ctx = RunContext(
                czi_path=czi,
                czi_paths=list(czi_paths),
                output_dir=out,
                layout=self.layout_panel.layout_model,
                do_split=enables["Split"],
                do_focus=enables["Focus"],
                do_segment=enables["Segment"],
                do_classify=enables["Classify"],
                do_features=enables.get("Features", False),
                focus_opts=self.focus_panel.opts(),
                segment_opts=self.segment_panel.opts(),
                classify_opts=self.classify_panel.opts(),
                features_opts=self.features_panel.opts(),
                phase_channel=(self.input_panel.phase_channel
                               if isinstance(self.input_panel.phase_channel, int) else 0),
                channel_labels=self.input_panel.channel_labels,
            )
            self._runner = PipelineRunner(
                ctx, reuse_existing=self.run_panel.reuse_existing.isChecked()
            )
        else:
            bulk = self.input_panel.bulk_layout
            issues = bulk.validate()
            if issues:
                QMessageBox.warning(self, "Validation failed", "\n".join(issues))
                return
            entries = bulk.active_rows().to_dict(orient="records")
            ctx = BulkRunContext(
                czi_entries=entries,
                output_dir=out,
                do_focus=enables["Focus"],
                do_segment=enables["Segment"],
                do_classify=enables["Classify"],
                do_features=enables.get("Features", False),
                focus_opts=self.focus_panel.opts(),
                segment_opts=self.segment_panel.opts(),
                classify_opts=self.classify_panel.opts(),
                features_opts=self.features_panel.opts(),
                phase_channel=(self.input_panel.phase_channel
                               if isinstance(self.input_panel.phase_channel, int) else 0),
                channel_labels=self.input_panel.channel_labels,
            )
            self._runner = BulkPipelineRunner(
                ctx, reuse_existing=self.run_panel.reuse_existing.isChecked()
            )

        self._runner.stagesPlanned.connect(self.run_panel.on_stages_planned)
        self._runner.stageStarted.connect(self.run_panel.on_stage_started)
        self._runner.stageProgress.connect(self.run_panel.on_stage_progress)
        self._runner.stageFinished.connect(self.run_panel.on_stage_finished)
        self._runner.runFinished.connect(self.run_panel.on_run_finished)
        self._runner.runFailed.connect(self.run_panel.on_run_failed)
        self.run_panel.stopRequested.connect(self._runner.request_stop)

        self.run_panel.run_btn.setEnabled(False)
        self.run_panel.stop_btn.setEnabled(True)

        def _end(*_args: object) -> None:
            self.run_panel.run_btn.setEnabled(True)
            self.run_panel.stop_btn.setEnabled(False)

        self._runner.runFinished.connect(_end)
        self._runner.runFailed.connect(_end)
        self._runner.start()
