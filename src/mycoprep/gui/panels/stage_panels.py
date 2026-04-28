"""Per-stage option panels — small forms wrapping api.py dataclasses."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from mycoprep.core.api import (
    PRESET_MODELS,
    ClassifyOpts,
    FocusOpts,
    SegmentOpts,
    resolve_classifier_preset,
)

from ..ui import icons, tokens
from ..ui.labeled_slider import LabeledSlider


def _with_helper(widget: QWidget, helper: str) -> QWidget:
    """Wrap a form-row widget with helper text below it."""
    from PyQt6.QtWidgets import QVBoxLayout
    holder = QWidget()
    v = QVBoxLayout(holder)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(2)
    v.addWidget(widget)
    if helper:
        cap = QLabel(helper)
        cap.setObjectName("caption")
        cap.setStyleSheet(
            f"color: {tokens.active().text_subtle}; font-size: {tokens.FS_CAPTION}px;"
        )
        cap.setWordWrap(True)
        v.addWidget(cap)
    return holder


class FocusPanel(QWidget):
    # Fires whenever any option widget on this panel changes value. The
    # live preview controller listens for this and re-runs focus +
    # downstream stages. Suppressed during ``restore_state`` via the
    # ``_loading`` guard so app startup doesn't cause spurious renders.
    optionsChanged = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PyQt6.QtWidgets import QGroupBox, QVBoxLayout

        self._loading = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.S3)

        opt_box = QGroupBox("Focus options")
        self._form = QFormLayout(opt_box)
        self._form.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        self._form.setHorizontalSpacing(tokens.S4)
        self._form.setVerticalSpacing(tokens.S3)
        root.addWidget(opt_box)
        root.addStretch(1)

        self.mode = QComboBox()
        self.mode.addItems(["edf", "tiles", "whole"])      # EDF is the default
        self.metric = QComboBox()
        self.metric.addItems(["ensemble", "normalized_variance", "brenner",
                              "tenengrad", "laplacian", "sml", "vollath"])
        self.metric.setCurrentText("ensemble")             # tested default

        # Tile-grid row, hidden unless mode == "tiles"
        self.tile_rows = QSpinBox(); self.tile_rows.setRange(1, 16); self.tile_rows.setValue(3)
        self.tile_cols = QSpinBox(); self.tile_cols.setRange(1, 16); self.tile_cols.setValue(3)
        tile_row = QHBoxLayout()
        tile_row.addWidget(self.tile_rows); tile_row.addWidget(QLabel("×")); tile_row.addWidget(self.tile_cols)
        tile_row.addStretch(1)
        self._tile_wrap = QWidget(); self._tile_wrap.setLayout(tile_row)
        self._tile_label = QLabel("Tile grid (rows × cols):")

        self._form.addRow("Mode:", _with_helper(
            self.mode,
            "edf = stack the per-tile in-focus pixels (recommended). "
            "tiles = pick best Z per tile. whole = single best Z per FOV.",
        ))
        self._form.addRow("Metric:", _with_helper(
            self.metric,
            "Sharpness measure used to score Z-planes. 'ensemble' averages "
            "several robust metrics and is a safe default.",
        ))
        self._form.addRow(self._tile_label, self._tile_wrap)

        self.save_zmaps = QCheckBox("Save per-FOV Z-map TIFFs (EDF debug output)")
        self.save_zmaps.setChecked(False)
        self._form.addRow("", self.save_zmaps)

        self.save_mip = QCheckBox("Add MIP companion channel for each fluorescence channel")
        self.save_mip.setChecked(False)
        self._form.addRow("", self.save_mip)

        self.mode.currentTextChanged.connect(self._refresh_tile_visibility)
        self._refresh_tile_visibility(self.mode.currentText())

        # External phase-channel override — set from the Input tab.
        self._phase_channel: int | str | None = None

        # Fan in every option-widget value-changed signal into the
        # single ``optionsChanged`` signal the controller listens to.
        for w in (self.mode, self.metric):
            w.currentIndexChanged.connect(self._emit_options_changed)
        for w in (self.tile_rows, self.tile_cols):
            w.valueChanged.connect(self._emit_options_changed)
        for w in (self.save_zmaps, self.save_mip):
            w.toggled.connect(self._emit_options_changed)

    def _emit_options_changed(self, *_args) -> None:
        if not self._loading:
            self.optionsChanged.emit()

    def _refresh_tile_visibility(self, mode: str) -> None:
        show = mode == "tiles"
        self._tile_wrap.setVisible(show)
        self._tile_label.setVisible(show)

    def set_phase_channel(self, phase: int | str | None) -> None:
        self._phase_channel = phase

    def opts(self) -> FocusOpts:
        return FocusOpts(
            metric=self.metric.currentText(),
            mode=self.mode.currentText(),
            tile_grid=(self.tile_rows.value(), self.tile_cols.value()),
            phase_channel=self._phase_channel,
            save_zmaps=self.save_zmaps.isChecked(),
            save_mip=self.save_mip.isChecked(),
        )

    def state(self) -> dict:
        return {
            "metric": self.metric.currentText(),
            "mode": self.mode.currentText(),
            "tile_rows": self.tile_rows.value(),
            "tile_cols": self.tile_cols.value(),
            "save_zmaps": self.save_zmaps.isChecked(),
            "save_mip": self.save_mip.isChecked(),
        }

    def restore_state(self, s: dict) -> None:
        if not isinstance(s, dict):
            return
        self._loading = True
        try:
            if "mode" in s:
                i = self.mode.findText(str(s["mode"]))
                if i >= 0:
                    self.mode.setCurrentIndex(i)
            if "metric" in s:
                i = self.metric.findText(str(s["metric"]))
                if i >= 0:
                    self.metric.setCurrentIndex(i)
            if "tile_rows" in s:
                self.tile_rows.setValue(int(s["tile_rows"]))
            if "tile_cols" in s:
                self.tile_cols.setValue(int(s["tile_cols"]))
            if "save_zmaps" in s:
                self.save_zmaps.setChecked(bool(s["save_zmaps"]))
            if "save_mip" in s:
                self.save_mip.setChecked(bool(s["save_mip"]))
        finally:
            self._loading = False


class SegmentPanel(QWidget):
    """Pure options widget for the Segment stage.

    The single-FOV preview now lives in the persistent live preview
    column to the right of this tab; this panel only carries options.
    """

    optionsChanged = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PyQt6.QtWidgets import QGroupBox, QVBoxLayout

        self._loading = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        opt_box = QGroupBox("Segmentation options")
        form = QFormLayout(opt_box)
        form.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        form.setHorizontalSpacing(tokens.S4)
        form.setVerticalSpacing(tokens.S3)
        self.model_type = QComboBox()
        self.model_type.addItems(["cpsam", "cyto3", "cyto2"])
        self.diameter = QDoubleSpinBox(); self.diameter.setRange(0.0, 500.0); self.diameter.setSpecialValueText("auto")
        # Default to a concrete value (rather than ``auto``) so cellpose's
        # rescale-by-diameter pass lands on a sensible image size — the
        # cpsam ViT-SAM backbone has fixed-size positional embeddings
        # and crashes on unusually large rescaled grids. 25 px matches
        # typical *Mtb* imaging at our standard 100×/1.4 objective; the
        # user can override if their imaging differs.
        self.diameter.setValue(25.0)
        self.gpu = QCheckBox("Use GPU")
        self.gpu.setChecked(True)
        self.pixels_per_um = QDoubleSpinBox()
        self.pixels_per_um.setRange(0.0, 10000.0)
        self.pixels_per_um.setValue(13.8767)
        self.pixels_per_um.setDecimals(4)
        self._pixel_hint = QLabel("(auto-detect from CZI once a file is loaded)")
        self._pixel_hint.setObjectName("muted")
        self._user_overrode_pixels = False
        self.pixels_per_um.valueChanged.connect(self._mark_user_override)

        # Cellpose tunables — sliders for thresholds (visual feedback for
        # tuning) plus precise numeric input.
        self.flow_threshold = LabeledSlider(
            "Flow threshold", 0.0, 3.0, 0.4, step=0.05, decimals=2,
            helper="Higher → keep more marginal masks. Default 0.4. Try 0.6–1.0 if cells are missed.",
        )
        self.cellprob_threshold = LabeledSlider(
            "Cell-prob threshold", -6.0, 6.0, 0.0, step=0.1, decimals=2,
            helper="Lower (more negative) → keep more low-probability cells. Default 0. Try −1 to −2 if cells are missed.",
        )

        self.min_size = QSpinBox()
        self.min_size.setRange(0, 5000); self.min_size.setValue(15)
        self.min_size.setToolTip("Minimum mask area in pixels. Default 15. Set to 0 to disable.")

        form.addRow("Cellpose model:", _with_helper(
            self.model_type,
            "cpsam = newest (recommended). cyto3/cyto2 = older Cellpose builds.",
        ))
        form.addRow("Diameter (px):", _with_helper(
            self.diameter,
            "Median cell diameter in pixels; 0 = let Cellpose estimate.",
        ))
        form.addRow("Min size (px):", _with_helper(
            self.min_size,
            "Drop masks smaller than this many pixels. 0 disables the filter.",
        ))
        form.addRow("", self.gpu)
        form.addRow("Pixels per µm:", _with_helper(
            self.pixels_per_um, ""  # hint label below carries the helper
        ))
        form.addRow("", self._pixel_hint)
        # Sliders sit below the form, full-width, with their own helper text.
        sliders_label = QLabel("Mask-quality thresholds")
        sliders_label.setObjectName("h3")
        from PyQt6.QtWidgets import QVBoxLayout as _QVB
        opt_box_layout = opt_box.layout()  # the form
        # Move out of QFormLayout: stack the two sliders below
        _holder = QWidget()
        _holder_l = _QVB(_holder)
        _holder_l.setContentsMargins(0, tokens.S2, 0, 0)
        _holder_l.setSpacing(tokens.S3)
        _holder_l.addWidget(sliders_label)
        _holder_l.addWidget(self.flow_threshold)
        _holder_l.addWidget(self.cellprob_threshold)
        # Add sliders as a wide form row spanning both columns
        form.addRow(_holder)

        root.addWidget(opt_box)
        root.addStretch(1)

        # Fan in option-widget signals → optionsChanged.
        self.model_type.currentIndexChanged.connect(self._emit_options_changed)
        self.diameter.valueChanged.connect(self._emit_options_changed)
        self.min_size.valueChanged.connect(self._emit_options_changed)
        self.gpu.toggled.connect(self._emit_options_changed)
        self.pixels_per_um.valueChanged.connect(self._emit_options_changed)
        self.flow_threshold.valueChanged.connect(self._emit_options_changed)
        self.cellprob_threshold.valueChanged.connect(self._emit_options_changed)

    def _emit_options_changed(self, *_args) -> None:
        if not self._loading:
            self.optionsChanged.emit()

    def _mark_user_override(self, _value: float) -> None:
        # Any manual edit sticks; auto-detection from CZI won't clobber it.
        self._user_overrode_pixels = True

    def set_detected_pixels_per_um(self, value: float | None) -> None:
        """Called from the Input panel when a CZI is selected."""
        if value is None:
            self._pixel_hint.setText("(pixel size could not be read from CZI)")
            return
        self._pixel_hint.setText(f"(auto-detected from CZI: {value:.3f} px/µm)")
        if not self._user_overrode_pixels:
            self.pixels_per_um.blockSignals(True)
            self.pixels_per_um.setValue(value)
            self.pixels_per_um.blockSignals(False)

    def opts(self) -> SegmentOpts:
        diameter = None if self.diameter.value() == 0.0 else self.diameter.value()
        return SegmentOpts(
            model_type=self.model_type.currentText(),
            diameter=diameter,
            flow_threshold=self.flow_threshold.value(),
            cellprob_threshold=self.cellprob_threshold.value(),
            min_size=self.min_size.value(),
            gpu=self.gpu.isChecked(),
            pixels_per_um=self.pixels_per_um.value(),
        )

    def state(self) -> dict:
        return {
            "model_type": self.model_type.currentText(),
            "diameter": self.diameter.value(),
            "flow_threshold": self.flow_threshold.value(),
            "cellprob_threshold": self.cellprob_threshold.value(),
            "min_size": self.min_size.value(),
            "gpu": self.gpu.isChecked(),
            "pixels_per_um": self.pixels_per_um.value(),
            "user_overrode_pixels": self._user_overrode_pixels,
        }

    def restore_state(self, s: dict) -> None:
        if not isinstance(s, dict):
            return
        self._loading = True
        try:
            if "model_type" in s:
                i = self.model_type.findText(str(s["model_type"]))
                if i >= 0:
                    self.model_type.setCurrentIndex(i)
            if "diameter" in s:
                self.diameter.setValue(float(s["diameter"]))
            if "flow_threshold" in s:
                self.flow_threshold.setValue(float(s["flow_threshold"]))
            if "cellprob_threshold" in s:
                self.cellprob_threshold.setValue(float(s["cellprob_threshold"]))
            if "min_size" in s:
                self.min_size.setValue(int(s["min_size"]))
            if "gpu" in s:
                self.gpu.setChecked(bool(s["gpu"]))
            if "pixels_per_um" in s:
                # Block the override flag while restoring; the user's manual
                # state is captured separately below.
                self.pixels_per_um.blockSignals(True)
                self.pixels_per_um.setValue(float(s["pixels_per_um"]))
                self.pixels_per_um.blockSignals(False)
            if "user_overrode_pixels" in s:
                self._user_overrode_pixels = bool(s["user_overrode_pixels"])
        finally:
            self._loading = False


class ClassifyPanel(QWidget):
    """Classifier options + small model-inspector. The combined
    Segment & Classify tab embeds this directly; the inspector lives
    inside this panel so it stays adjacent to the threshold control.
    """

    # Emitted when the user clicks one of the "Show …" buttons.
    # Connected by MainWindow to pop the corresponding standalone window.
    openLabelTrainerRequested = pyqtSignal()
    showModelDetailsRequested = pyqtSignal()
    optionsChanged = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PyQt6.QtWidgets import QGroupBox, QVBoxLayout
        from ..widgets.model_inspector import ModelInspector

        self._loading = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # ── Options group ────────────────────────────────────────────────
        opt_box = QGroupBox("Classifier options")
        form = QFormLayout(opt_box)
        form.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        form.setHorizontalSpacing(tokens.S4)
        form.setVerticalSpacing(tokens.S3)

        self.preset = QComboBox()
        self.preset.addItems(["none (rules only)"] + sorted(PRESET_MODELS.keys()) + ["custom…"])
        self.preset.currentIndexChanged.connect(self._on_preset)

        self.custom_path = QLineEdit()
        self.custom_path.setReadOnly(True)
        self.custom_path.setPlaceholderText("(no custom model selected)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_custom)
        path_row = QHBoxLayout()
        path_row.addWidget(self.custom_path); path_row.addWidget(browse)
        path_wrap = QWidget(); path_wrap.setLayout(path_row)

        self.use_rules = QCheckBox("Apply rule-based pre-filter (edge / debris)")
        self.use_rules.setChecked(True)

        self.confidence = LabeledSlider(
            "Confidence threshold", 0.0, 1.0, 0.5, step=0.01, decimals=2,
            helper="Cells classified as 'good' below this score are rejected. "
                   "Higher = stricter. Default 0.5.",
        )
        self.confidence.valueChanged.connect(self._on_confidence_changed)

        form.addRow("Classifier:", self.preset)
        form.addRow("Custom model (.pth):", path_wrap)
        form.addRow("", self.use_rules)
        from PyQt6.QtWidgets import QVBoxLayout as _QVB
        _conf_holder = QWidget()
        _conf_l = _QVB(_conf_holder)
        _conf_l.setContentsMargins(0, tokens.S2, 0, 0)
        _conf_l.addWidget(self.confidence)
        form.addRow(_conf_holder)
        root.addWidget(opt_box)

        # Pop-out buttons grouped in a card: model details and the
        # labeller / trainer live in their own windows so this tab can
        # stay focused on the controls that drive the pipeline. The
        # single-FOV segmentation preview now lives in the always-on
        # live preview column to the right of this tab.
        from PyQt6.QtWidgets import QFrame
        actions_card = QFrame()
        actions_card.setObjectName("card")
        button_row = QHBoxLayout(actions_card)
        button_row.setContentsMargins(tokens.S3, tokens.S2, tokens.S3, tokens.S2)
        button_row.setSpacing(tokens.S2)

        self._show_details_btn = QPushButton("  Model details")
        self._show_details_btn.setIcon(icons.icon("model", role="muted"))
        self._show_details_btn.setToolTip(
            "Show ROC / precision-recall curves and training stats for the "
            "currently-selected classifier model in a separate window. The live "
            "threshold marker on the plot tracks this tab's confidence spinner."
        )
        self._show_details_btn.clicked.connect(self.showModelDetailsRequested.emit)

        self._open_labeler_btn = QPushButton("  Labeller / trainer")
        self._open_labeler_btn.setIcon(icons.icon("label", role="muted"))
        self._open_labeler_btn.setToolTip(
            "Open the active-learning labeller and classifier fine-tuner in a "
            "separate window. Use it to build training data from the current "
            "segmented output and retrain the model."
        )
        self._open_labeler_btn.clicked.connect(self.openLabelTrainerRequested.emit)

        button_row.addWidget(self._show_details_btn)
        button_row.addWidget(self._open_labeler_btn)
        button_row.addStretch(1)
        root.addWidget(actions_card)
        root.addStretch(1)

        # The inspector widget is still constructed (and kept fed with the
        # current model and threshold), but it lives in a pop-up window
        # rather than in this panel — see MainWindow._open_model_details.
        self.inspector = ModelInspector()

        self.custom_path.textChanged.connect(self._refresh_inspector)
        self._refresh_inspector()

        # Fan in option-widget signals → optionsChanged.
        self.preset.currentIndexChanged.connect(self._emit_options_changed)
        self.custom_path.textChanged.connect(self._emit_options_changed)
        self.use_rules.toggled.connect(self._emit_options_changed)
        self.confidence.valueChanged.connect(self._emit_options_changed)

    def _emit_options_changed(self, *_args) -> None:
        if not self._loading:
            self.optionsChanged.emit()

    def _on_preset(self) -> None:
        text = self.preset.currentText()
        self.custom_path.setEnabled(text == "custom…")
        self._refresh_inspector()

    def _pick_custom(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select classifier .pth", "", "PyTorch (*.pth)")
        if path:
            self.custom_path.setText(path)

    def _on_confidence_changed(self, value: float) -> None:
        self.inspector.set_threshold(value)

    def _refresh_inspector(self) -> None:
        """Point the inspector at the currently-selected model (or None)."""
        text = self.preset.currentText()
        if text.startswith("none"):
            self.inspector.set_model(None)
            return
        if text == "custom…":
            p = self.custom_path.text().strip()
            self.inspector.set_model(Path(p) if p else None)
            return
        try:
            self.inspector.set_model(resolve_classifier_preset(text))
        except (KeyError, FileNotFoundError):
            self.inspector.set_model(None)
        self.inspector.set_threshold(self.confidence.value())

    def opts(self) -> ClassifyOpts:
        text = self.preset.currentText()
        model_path: Path | None = None
        if text.startswith("none"):
            model_path = None
        elif text == "custom…":
            p = self.custom_path.text().strip()
            model_path = Path(p) if p else None
        else:
            try:
                model_path = resolve_classifier_preset(text)
            except (KeyError, FileNotFoundError):
                model_path = None
        return ClassifyOpts(
            model_path=model_path,
            use_rules=self.use_rules.isChecked(),
            confidence_threshold=self.confidence.value(),
        )

    def state(self) -> dict:
        return {
            "preset": self.preset.currentText(),
            "custom_path": self.custom_path.text(),
            "use_rules": self.use_rules.isChecked(),
            "confidence": self.confidence.value(),
        }

    def restore_state(self, s: dict) -> None:
        if not isinstance(s, dict):
            return
        self._loading = True
        try:
            if "preset" in s:
                i = self.preset.findText(str(s["preset"]))
                if i >= 0:
                    self.preset.setCurrentIndex(i)
            if "custom_path" in s:
                self.custom_path.setText(str(s["custom_path"]))
            if "use_rules" in s:
                self.use_rules.setChecked(bool(s["use_rules"]))
            if "confidence" in s:
                self.confidence.setValue(float(s["confidence"]))
        finally:
            self._loading = False


class SegmentClassifyPanel(QWidget):
    """Combined Segment + Classify tab.

    Stacks SegmentPanel above ClassifyPanel — both option groups feed
    the live preview column on the right. Re-exposes the inner panels
    as ``.segment_panel`` and ``.classify_panel`` so MainWindow can wire
    signals/options through.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PyQt6.QtWidgets import QVBoxLayout

        self.segment_panel = SegmentPanel()
        self.classify_panel = ClassifyPanel()

        # Vertical stack: Segment above, Classify below. The combined tab
        # used to put these side-by-side, but the live preview reclaims
        # the right side of the window so the options column is now too
        # narrow for two cards across.
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.S4)
        root.addWidget(self.segment_panel)
        root.addWidget(self.classify_panel)
        root.addStretch(1)

    def set_detected_pixels_per_um(self, value):
        self.segment_panel.set_detected_pixels_per_um(value)
