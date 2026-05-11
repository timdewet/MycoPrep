"""CNN embeddings panel — autoencoder configuration for the Embeddings stage.

Sits in the sidebar between Features and Run. Surfaces model type, training
source, channel selection, and training hyperparameters. The stage itself is
toggled on/off in the Run panel's "Stages to run" checklist.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..pipeline.context import EmbeddingOpts
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


class EmbeddingsPanel(QWidget):
    """Options for the CNN Embeddings stage."""

    optionsChanged = pyqtSignal()
    openLibraryBrowserRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._loading = False
        self._train_only = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.S3)

        # ── Model section ────────────────────────────────────────────
        model_box = QGroupBox("Autoencoder model")
        model_form = QFormLayout(model_box)
        model_form.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        model_form.setHorizontalSpacing(tokens.S4)
        model_form.setVerticalSpacing(tokens.S3)

        self.model_type = QComboBox()
        self.model_type.addItems([
            "ResNet-18",
            "Lightweight",
            "ResNet-18 (SupCon)",
            "Lightweight (SupCon)",
        ])
        model_form.addRow("Architecture:", _with_helper(
            self.model_type,
            "ResNet-18 / Lightweight: autoencoder, label-free reconstruction loss.\n"
            "ResNet-18 (SupCon) / Lightweight (SupCon): supervised contrastive — "
            "uses gene labels from condition_label to push different genes apart "
            "and pull same-gene cells together. Sharper class clusters when labels "
            "exist. Lightweight (~few hundred K params, no ImageNet pretraining) "
            "is often preferable for bacteria where ResNet's depth and natural-image "
            "priors don't help.",
        ))

        self.model_source = QComboBox()
        self.model_source.addItems([
            "Auto (train/fine-tune)",
            "Train from scratch",
            "Use existing model",
        ])
        self._source_helper = _with_helper(
            self.model_source,
            "Auto: trains from scratch on the first run, fine-tunes on later runs.\n"
            "Train from scratch: ignores any existing library model and starts fresh "
            "(needed when changing the channel set, since you can't re-shape conv1).\n"
            "Use existing: point to a trained .pth file.",
        )
        self._source_label = QLabel("Source:")
        model_form.addRow(self._source_label, self._source_helper)

        path_row = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("(auto-detected from library)")
        self.model_path.setEnabled(False)
        path_row.addWidget(self.model_path)
        self._browse_btn = QPushButton("Browse\u2026")
        self._browse_btn.setFixedWidth(80)
        self._browse_btn.setEnabled(False)
        self._browse_btn.clicked.connect(self._browse_model_path)
        path_row.addWidget(self._browse_btn)
        path_widget = QWidget()
        path_widget.setLayout(path_row)
        self._path_widget = path_widget
        self._path_label = QLabel("Model path:")
        model_form.addRow(self._path_label, path_widget)

        self.in_channels = QComboBox()
        self.in_channels.addItems([
            "1 image channel (brightfield only)",
            "2 image channels (BF + 1 fluor)",
            "3 image channels (BF + 2 fluors)",
        ])
        model_form.addRow("Image channels:", _with_helper(
            self.in_channels,
            "Number of non-mask image channels fed to the encoder, taken in "
            "their H5 order. Set this to match how many imaging channels "
            "your acquisition has.",
        ))

        self.include_mask = QCheckBox("Include segmentation mask channel")
        self.include_mask.setChecked(True)
        model_form.addRow("", _with_helper(
            self.include_mask,
            "Append the binarised mask as an extra input channel. The "
            "MorphologicalProfiling_Mtb SupCon training used phase + ParB + "
            "mask — the mask gives the encoder the segmentation prior for "
            "free instead of forcing it to re-derive cell boundaries from "
            "intensity. Default ON.",
        ))

        root.addWidget(model_box)

        # ── Training section ─────────────────────────────────────────
        train_box = QGroupBox("Training")
        train_form = QFormLayout(train_box)
        train_form.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        train_form.setHorizontalSpacing(tokens.S4)
        train_form.setVerticalSpacing(tokens.S3)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(50)
        train_form.addRow("Epochs:", self.epochs)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(8, 2048)
        self.batch_size.setValue(256)
        self.batch_size.setSingleStep(64)
        train_form.addRow("Batch size:", self.batch_size)

        self.include_library_crops = QCheckBox("Include library crops in training")
        self.include_library_crops.setChecked(True)
        train_form.addRow("", _with_helper(
            self.include_library_crops,
            "Pool crops from all registered library runs for training, not just "
            "the current run. Strongly recommended for robust embeddings.",
        ))

        root.addWidget(train_box)

        # ── Library access ──────────────────────────────────────────
        lib_row = QHBoxLayout()
        lib_row.setContentsMargins(0, 0, 0, 0)
        lib_row.addStretch(1)
        self._lib_btn = QPushButton("Library browser\u2026")
        self._lib_btn.setToolTip(
            "Open the morphology library browser to inspect registered runs "
            "and trained models."
        )
        self._lib_btn.clicked.connect(self.openLibraryBrowserRequested.emit)
        lib_row.addWidget(self._lib_btn)
        lib_row_widget = QWidget()
        lib_row_widget.setLayout(lib_row)
        root.addWidget(lib_row_widget)

        root.addStretch(1)

        self.model_source.currentIndexChanged.connect(self._refresh_model_path_enable)

        # Fan in option signals
        for cb in (self.include_library_crops, self.include_mask):
            cb.toggled.connect(self._emit_options_changed)
        for sb in (self.epochs, self.batch_size):
            sb.valueChanged.connect(self._emit_options_changed)
        for combo in (self.model_type, self.model_source, self.in_channels):
            combo.currentTextChanged.connect(self._emit_options_changed)
        self.model_path.textChanged.connect(self._emit_options_changed)

    def _emit_options_changed(self, *_args) -> None:
        if not self._loading:
            self.optionsChanged.emit()

    def _refresh_model_path_enable(self, _idx: int = 0) -> None:
        # The dropdown order is now Auto / Train from scratch / Use existing.
        is_use_existing = (
            "existing" in self.model_source.currentText().lower()
            and not self._train_only
        )
        self.model_path.setEnabled(is_use_existing)
        self._browse_btn.setEnabled(is_use_existing)
        self.epochs.setEnabled(not is_use_existing)
        self.batch_size.setEnabled(not is_use_existing)
        self.include_library_crops.setEnabled(not is_use_existing)

    def set_train_only_mode(self, train_only: bool) -> None:
        """Hide source/model-path rows when the run is a library-only training job.

        In this mode 'Use existing model' is meaningless (training from scratch
        is the whole point), so we collapse those rows entirely.
        """
        self._train_only = bool(train_only)
        self._source_label.setVisible(not self._train_only)
        self._source_helper.setVisible(not self._train_only)
        self._path_label.setVisible(not self._train_only)
        self._path_widget.setVisible(not self._train_only)
        if self._train_only:
            # Force model_source to "Auto" so opts() reports the right value.
            self.model_source.setCurrentIndex(0)
        self._refresh_model_path_enable()

    def _browse_model_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select trained model", "",
            "PyTorch model (*.pth);;All files (*)",
        )
        if path:
            self.model_path.setText(path)

    # ─────────────────────────────────────────────────────────────
    # Options / persistence
    # ─────────────────────────────────────────────────────────────

    def opts(self) -> EmbeddingOpts:
        return EmbeddingOpts(
            model_type=self.model_type.currentText(),
            model_source=self.model_source.currentText(),
            model_path=self.model_path.text().strip(),
            in_channels=int(self.in_channels.currentIndex()) + 1,
            include_mask=self.include_mask.isChecked(),
            epochs=int(self.epochs.value()),
            batch_size=int(self.batch_size.value()),
            include_library_crops=self.include_library_crops.isChecked(),
            train_only=self._train_only,
        )

    def state(self) -> dict:
        return {
            "model_type": self.model_type.currentText(),
            "model_source": self.model_source.currentText(),
            "model_path": self.model_path.text(),
            "in_channels": self.in_channels.currentIndex(),
            "include_mask": self.include_mask.isChecked(),
            "epochs": int(self.epochs.value()),
            "batch_size": int(self.batch_size.value()),
            "include_library_crops": self.include_library_crops.isChecked(),
        }

    def restore_state(self, s: dict) -> None:
        if not isinstance(s, dict):
            return
        self._loading = True
        try:
            if "model_type" in s:
                idx = self.model_type.findText(str(s["model_type"]))
                if idx >= 0:
                    self.model_type.setCurrentIndex(idx)
            if "model_source" in s:
                idx = self.model_source.findText(str(s["model_source"]))
                if idx >= 0:
                    self.model_source.setCurrentIndex(idx)
            if "model_path" in s:
                self.model_path.setText(str(s["model_path"]))
            if "in_channels" in s:
                idx = int(s["in_channels"])
                idx = max(0, min(idx, self.in_channels.count() - 1))
                self.in_channels.setCurrentIndex(idx)
            if "include_mask" in s:
                self.include_mask.setChecked(bool(s["include_mask"]))
            if "epochs" in s:
                self.epochs.setValue(int(s["epochs"]))
            if "batch_size" in s:
                self.batch_size.setValue(int(s["batch_size"]))
            if "include_library_crops" in s:
                self.include_library_crops.setChecked(bool(s["include_library_crops"]))
        finally:
            self._loading = False
        self._refresh_model_path_enable()
