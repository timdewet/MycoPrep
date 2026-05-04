"""Per-cell feature extraction (Phase A) panel.

Phase A surface: a compact form with two grouped sections — per-cell features
(regionprops + per-channel intensity) and single-cell HDF5 crops for embedding
work. Phase B will add midline-derived features (length_um, demograph, polar
bias) once MOMIA is vendored; the panel is built to make those additions
straightforward.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mycoprep.core.api import ExtractOpts

from ..ui import tokens


def _with_helper(widget: QWidget, helper: str) -> QWidget:
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


class _ChannelMultiSelect(QWidget):
    """Compact multi-select box backed by a QListWidget. Channel names are
    populated lazily (via ``set_channels``) once they're known from the
    Input panel; selections persist via integer indices.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._list.setMaximumHeight(110)
        layout.addWidget(self._list)
        self._channels: list[str] = []
        self._pending_selection: list[int] | None = None

    def set_channels(self, names: list[str]) -> None:
        self._channels = list(names)
        prev_idx = self.selected_indices() or self._pending_selection
        self._list.clear()
        for i, name in enumerate(self._channels):
            item = QListWidgetItem(f"{i}: {name}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable)
            self._list.addItem(item)
        if prev_idx:
            for i in prev_idx:
                if 0 <= i < self._list.count():
                    self._list.item(i).setSelected(True)
            self._pending_selection = None
        else:
            for i in range(self._list.count()):
                self._list.item(i).setSelected(True)

    def selected_indices(self) -> list[int]:
        return [self._list.row(item) for item in self._list.selectedItems()]

    def set_selected_indices(self, indices: list[int]) -> None:
        if self._list.count() == 0:
            self._pending_selection = list(indices)
            return
        self._list.clearSelection()
        for i in indices:
            if 0 <= i < self._list.count():
                self._list.item(i).setSelected(True)


class FeaturesPanel(QWidget):
    """Options for the Features stage."""

    optionsChanged = pyqtSignal()
    openLibraryBrowserRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._loading = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.S3)

        # ── Per-cell features ──────────────────────────────────────
        feat_box = QGroupBox("Per-cell features (Parquet)")
        feat_form = QFormLayout(feat_box)
        feat_form.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        feat_form.setHorizontalSpacing(tokens.S4)
        feat_form.setVerticalSpacing(tokens.S3)

        self.do_morphology = QCheckBox("Morphology (regionprops)")
        self.do_morphology.setChecked(True)
        self.do_intensity = QCheckBox("Per-channel intensity")
        self.do_intensity.setChecked(True)
        self.do_midline = QCheckBox(
            "Midline-derived features (length_um, width_*, sinuosity, branch_count)"
        )
        self.do_midline.setChecked(True)
        self.do_refine_contour = QCheckBox(
            "Refine contour against phase-gradient (sub-pixel precision; "
            "MicrobeJ-style)"
        )
        self.do_refine_contour.setChecked(True)

        feat_form.addRow("", self.do_morphology)
        feat_form.addRow("", self.do_intensity)
        feat_form.addRow("", self.do_midline)
        feat_form.addRow("", self.do_refine_contour)

        self.intensity_channels = _ChannelMultiSelect()
        feat_form.addRow("Intensity channels:", _with_helper(
            self.intensity_channels,
            "Channels to compute intensity stats for. The mask channel is "
            "excluded automatically. Default: all non-mask channels.",
        ))

        self.save_csv = QCheckBox("Also write CSV alongside Parquet")
        self.save_csv.setChecked(True)
        feat_form.addRow("", _with_helper(
            self.save_csv,
            "CSV is human-readable and opens in Excel; Parquet is faster and "
            "preserves dtypes. Both contain identical rows.",
        ))

        self.make_qc_plots = QCheckBox(
            "Auto-generate QC plots at end of run"
        )
        self.make_qc_plots.setChecked(True)
        feat_form.addRow("", _with_helper(
            self.make_qc_plots,
            "Writes per-mutant violins (length / width / area / intensity), "
            "pooled-by-ATc figures, length-vs-intensity facets, and "
            "intensity-variation bars + ridge to "
            "<features_dir>/qc_plots/.",
        ))

        root.addWidget(feat_box)

        # ── Single-cell HDF5 crops ────────────────────────────────
        crops_box = QGroupBox("Single-cell crops (HDF5, for embedding work)")
        crops_form = QFormLayout(crops_box)
        crops_form.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        crops_form.setHorizontalSpacing(tokens.S4)
        crops_form.setVerticalSpacing(tokens.S3)

        self.save_crops = QCheckBox("Save HDF5 crops (drops into MorphologicalProfiling_Mtb pipeline)")
        self.save_crops.setChecked(True)
        crops_form.addRow("", self.save_crops)

        self.crop_size = QSpinBox()
        self.crop_size.setRange(16, 512)
        self.crop_size.setSingleStep(16)
        self.crop_size.setValue(96)
        crops_form.addRow("Crop size (px):", _with_helper(
            self.crop_size,
            "Output crop side length. Existing SupCon training pipeline uses 128.",
        ))

        self.crop_pad = QSpinBox()
        self.crop_pad.setRange(0, 100)
        self.crop_pad.setValue(10)
        crops_form.addRow("Pad (px):", self.crop_pad)

        self.include_mask = QCheckBox("Include mask channel")
        self.include_mask.setChecked(True)
        crops_form.addRow("", self.include_mask)

        self.mask_background = QCheckBox("Suppress background outside dilated cell mask")
        self.mask_background.setChecked(True)
        crops_form.addRow("", self.mask_background)

        self.normalise_per_crop = QCheckBox("Per-crop min-max normalise to [0, 1]")
        self.normalise_per_crop.setChecked(True)
        crops_form.addRow("", self.normalise_per_crop)

        self.crop_channels = _ChannelMultiSelect()
        crops_form.addRow("Crop channels:", _with_helper(
            self.crop_channels,
            "Channels to include in the crop tensor. Default: all non-mask channels.",
        ))

        # Disable crop sub-widgets when the master checkbox is off.
        self.save_crops.toggled.connect(self._refresh_crops_enable)
        self._crops_widgets = (
            self.crop_size, self.crop_pad,
            self.include_mask, self.mask_background,
            self.normalise_per_crop, self.crop_channels,
        )

        root.addWidget(crops_box)

        # ── Feature library ───────────────────────────────────────
        lib_box = QGroupBox("Feature library (cross-experiment clustering)")
        lib_form = QFormLayout(lib_box)
        lib_form.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        lib_form.setHorizontalSpacing(tokens.S4)
        lib_form.setVerticalSpacing(tokens.S3)

        self.add_to_library = QCheckBox("Register this run in the feature library")
        self.add_to_library.setChecked(False)
        lib_form.addRow("", _with_helper(
            self.add_to_library,
            "When enabled, the consolidated all_features.parquet is copied "
            "into the library after extraction completes.",
        ))

        self.species = QComboBox()
        # Plots can't combine across species, so we don't offer an
        # "any species" choice — pick a real one. M. tuberculosis is
        # the most common project default.
        self.species.addItems(["M. tuberculosis", "M. smegmatis"])
        self.species.setCurrentText("M. tuberculosis")
        lib_form.addRow("Species:", self.species)

        self.experiment_type = QComboBox()
        self.experiment_type.addItems(["knockdown", "drug"])
        lib_form.addRow("Experiment type:", _with_helper(
            self.experiment_type,
            "Label this run as a genetic knockdown or drug treatment "
            "experiment. Used to distinguish points in clustering plots.",
        ))

        lib_dir_row = QHBoxLayout()
        self.library_dir = QLineEdit()
        self.library_dir.setPlaceholderText("~/.mycoprep/feature_library/")
        lib_dir_row.addWidget(self.library_dir)
        self._lib_browse_btn = QPushButton("Browse\u2026")
        self._lib_browse_btn.setFixedWidth(80)
        self._lib_browse_btn.clicked.connect(self._browse_library_dir)
        lib_dir_row.addWidget(self._lib_browse_btn)
        lib_dir_widget = QWidget()
        lib_dir_widget.setLayout(lib_dir_row)
        lib_form.addRow("Library dir:", lib_dir_widget)

        lib_buttons = QHBoxLayout()
        self._import_btn = QPushButton("Import existing parquet\u2026")
        self._import_btn.setToolTip(
            "Register an already-processed all_features.parquet into the "
            "library without rerunning the pipeline."
        )
        self._import_btn.clicked.connect(self._import_existing_parquet)
        lib_buttons.addWidget(self._import_btn)

        self._browse_lib_btn = QPushButton("Browse library\u2026")
        self._browse_lib_btn.setToolTip(
            "Open the library browser to view, filter, and manage registered runs."
        )
        self._browse_lib_btn.clicked.connect(self.openLibraryBrowserRequested.emit)
        lib_buttons.addWidget(self._browse_lib_btn)
        lib_buttons.addStretch(1)

        lib_buttons_widget = QWidget()
        lib_buttons_widget.setLayout(lib_buttons)
        lib_form.addRow("", lib_buttons_widget)

        # Disable library sub-widgets when the master checkbox is off.
        self._lib_widgets = (
            self.species, self.experiment_type, self.library_dir,
            self._lib_browse_btn,
        )
        self.add_to_library.toggled.connect(self._refresh_library_enable)

        root.addWidget(lib_box)
        root.addStretch(1)

        self._refresh_crops_enable(self.save_crops.isChecked())
        self._refresh_library_enable(self.add_to_library.isChecked())

        # Fan in option signals → optionsChanged.
        for cb in (
            self.do_morphology, self.do_intensity, self.do_midline,
            self.do_refine_contour, self.save_csv, self.make_qc_plots,
            self.save_crops, self.include_mask, self.mask_background,
            self.normalise_per_crop, self.add_to_library,
        ):
            cb.toggled.connect(self._emit_options_changed)
        for sb in (self.crop_size, self.crop_pad):
            sb.valueChanged.connect(self._emit_options_changed)
        self.library_dir.textChanged.connect(self._emit_options_changed)
        self.species.currentTextChanged.connect(self._emit_options_changed)
        self.experiment_type.currentTextChanged.connect(self._emit_options_changed)
        # The channel multi-select widgets emit selection signals through
        # their internal QListWidget; hook those up too.
        self.intensity_channels._list.itemSelectionChanged.connect(self._emit_options_changed)
        self.crop_channels._list.itemSelectionChanged.connect(self._emit_options_changed)

    def _emit_options_changed(self, *_args) -> None:
        if not self._loading:
            self.optionsChanged.emit()

    # ─────────────────────────────────────────────────────────────
    # External wiring
    # ─────────────────────────────────────────────────────────────

    def set_channels(self, names: list[str]) -> None:
        """Populate the channel multi-selects from the Input panel."""
        # The mask channel is appended at Segment time; downstream of segment
        # the *image* channels are everything except the last. The Input panel
        # already only knows image-channel names, so pass them through.
        self.intensity_channels.set_channels(list(names))
        self.crop_channels.set_channels(list(names))

    def _refresh_crops_enable(self, enabled: bool) -> None:
        for w in self._crops_widgets:
            w.setEnabled(enabled)

    def _refresh_library_enable(self, enabled: bool) -> None:
        for w in self._lib_widgets:
            w.setEnabled(enabled)

    def _browse_library_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select library directory")
        if d:
            self.library_dir.setText(d)

    def _import_existing_parquet(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select all_features.parquet", "",
            "Parquet files (*.parquet);;All files (*)",
        )
        if not path:
            return
        from mycoprep.core.extract.feature_library import (
            FeatureLibrary,
            derive_run_id_from_parquet,
        )

        lib_dir_text = self.library_dir.text().strip()
        lib = FeatureLibrary(Path(lib_dir_text) if lib_dir_text else None)
        run_id = derive_run_id_from_parquet(Path(path))
        lib.register_run(
            run_id=run_id,
            features_parquet=Path(path),
            species=self.species.currentText().strip(),
            experiment_type=self.experiment_type.currentText(),
        )

    # ─────────────────────────────────────────────────────────────
    # Options / persistence
    # ─────────────────────────────────────────────────────────────

    def opts(self) -> ExtractOpts:
        sel_intensity = self.intensity_channels.selected_indices() or None
        sel_crop = self.crop_channels.selected_indices() or None
        lib_dir_text = self.library_dir.text().strip()
        return ExtractOpts(
            morphology=self.do_morphology.isChecked(),
            intensity=self.do_intensity.isChecked(),
            midline_features=self.do_midline.isChecked(),
            refine_contour=self.do_refine_contour.isChecked(),
            fluorescence_channels=sel_intensity,
            save_csv=self.save_csv.isChecked(),
            make_qc_plots=self.make_qc_plots.isChecked(),
            add_to_library=self.add_to_library.isChecked(),
            library_dir=Path(lib_dir_text) if lib_dir_text else None,
            species=self.species.currentText().strip(),
            experiment_type=self.experiment_type.currentText(),
            save_crops=self.save_crops.isChecked(),
            crop_size=int(self.crop_size.value()),
            crop_pad=int(self.crop_pad.value()),
            crop_channels=sel_crop,
            include_mask_channel=self.include_mask.isChecked(),
            mask_background=self.mask_background.isChecked(),
            normalise_per_crop=self.normalise_per_crop.isChecked(),
        )

    def state(self) -> dict:
        return {
            "morphology": self.do_morphology.isChecked(),
            "intensity": self.do_intensity.isChecked(),
            "midline": self.do_midline.isChecked(),
            "refine_contour": self.do_refine_contour.isChecked(),
            "intensity_channels": self.intensity_channels.selected_indices(),
            "save_csv": self.save_csv.isChecked(),
            "make_qc_plots": self.make_qc_plots.isChecked(),
            "add_to_library": self.add_to_library.isChecked(),
            "library_dir": self.library_dir.text(),
            "species": self.species.currentText(),
            "experiment_type": self.experiment_type.currentText(),
            "save_crops": self.save_crops.isChecked(),
            "crop_size": int(self.crop_size.value()),
            "crop_pad": int(self.crop_pad.value()),
            "include_mask_channel": self.include_mask.isChecked(),
            "mask_background": self.mask_background.isChecked(),
            "normalise_per_crop": self.normalise_per_crop.isChecked(),
            "crop_channels": self.crop_channels.selected_indices(),
        }

    def restore_state(self, s: dict) -> None:
        if not isinstance(s, dict):
            return
        self._loading = True
        try:
            if "morphology" in s:
                self.do_morphology.setChecked(bool(s["morphology"]))
            if "intensity" in s:
                self.do_intensity.setChecked(bool(s["intensity"]))
            if "midline" in s:
                self.do_midline.setChecked(bool(s["midline"]))
            if "refine_contour" in s:
                self.do_refine_contour.setChecked(bool(s["refine_contour"]))
            if "save_csv" in s:
                self.save_csv.setChecked(bool(s["save_csv"]))
            if "make_qc_plots" in s:
                self.make_qc_plots.setChecked(bool(s["make_qc_plots"]))
            if "save_crops" in s:
                self.save_crops.setChecked(bool(s["save_crops"]))
            if "crop_size" in s:
                self.crop_size.setValue(int(s["crop_size"]))
            if "crop_pad" in s:
                self.crop_pad.setValue(int(s["crop_pad"]))
            if "include_mask_channel" in s:
                self.include_mask.setChecked(bool(s["include_mask_channel"]))
            if "mask_background" in s:
                self.mask_background.setChecked(bool(s["mask_background"]))
            if "normalise_per_crop" in s:
                self.normalise_per_crop.setChecked(bool(s["normalise_per_crop"]))
            if "add_to_library" in s:
                self.add_to_library.setChecked(bool(s["add_to_library"]))
            if "library_dir" in s:
                self.library_dir.setText(str(s["library_dir"]))
            if "species" in s:
                idx = self.species.findText(str(s["species"]))
                if idx >= 0:
                    self.species.setCurrentIndex(idx)
            if "experiment_type" in s:
                idx = self.experiment_type.findText(str(s["experiment_type"]))
                if idx >= 0:
                    self.experiment_type.setCurrentIndex(idx)
            if "intensity_channels" in s:
                self.intensity_channels.set_selected_indices(list(s["intensity_channels"] or []))
            if "crop_channels" in s:
                self.crop_channels.set_selected_indices(list(s["crop_channels"] or []))
        finally:
            self._loading = False
