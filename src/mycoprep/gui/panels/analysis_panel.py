"""Analysis panel — post-pipeline clustering and feature library tools.

This page lives after the Run stage and houses workflows that operate on
already-extracted features rather than raw images:

- Embedded feature library browser (manage registered runs, import existing)
- Re-run morphology clustering on a chosen features directory + library
  selection, regenerating the clustering QC plots without rerunning the
  full pipeline

Long-term this page is the home of the interactive cluster viewer.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..ui import tokens
from ..widgets.library_browser import LibraryBrowser

SPECIES_OPTIONS = ["M. tuberculosis", "M. smegmatis"]

_FEATURE_SUBDIRS = {"04_features", "features"}


def _default_run_id(features_dir: Path) -> str:
    """Best-effort run label, walking past conventional feature subdirs."""
    if features_dir.name in _FEATURE_SUBDIRS and features_dir.parent != features_dir:
        return features_dir.parent.name or features_dir.name
    return features_dir.name


class AnalysisPanel(QWidget):
    """Clustering and feature library management."""

    statusMessage = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.S3)

        # ── Re-run clustering on existing features ────────────────────
        cluster_box = QGroupBox("Re-run clustering on existing features")
        cluster_form = QFormLayout(cluster_box)
        cluster_form.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        cluster_form.setHorizontalSpacing(tokens.S4)
        cluster_form.setVerticalSpacing(tokens.S3)

        feat_row = QHBoxLayout()
        self._features_dir = QLineEdit()
        self._features_dir.setPlaceholderText(
            "Path to a features dir containing all_features.parquet"
        )
        feat_row.addWidget(self._features_dir)
        self._browse_features_btn = QPushButton("Browse\u2026")
        self._browse_features_btn.setFixedWidth(80)
        self._browse_features_btn.clicked.connect(self._browse_features_dir)
        feat_row.addWidget(self._browse_features_btn)
        feat_widget = QWidget(); feat_widget.setLayout(feat_row)
        cluster_form.addRow("Features dir:", feat_widget)

        self._species = QComboBox()
        self._species.addItems([""] + SPECIES_OPTIONS)
        cluster_form.addRow("Species:", self._species)

        self._run_id = QLineEdit()
        self._run_id.setPlaceholderText(
            "Used in the figure title (default: features dir name)"
        )
        cluster_form.addRow("Run label:", self._run_id)

        self._control_labels = QLineEdit()
        self._control_labels.setPlaceholderText("e.g. NT1, NT2, WT, DMSO")
        self._control_labels.setToolTip(
            "Comma-separated mutant tokens to treat as controls when "
            "computing S-scores. Leave blank to fall back to a global "
            "z-score (no control anchor)."
        )
        cluster_form.addRow("Control labels:", self._control_labels)

        action_row = QHBoxLayout()
        self._run_btn = QPushButton("Generate clustering plots")
        self._run_btn.clicked.connect(self._run_clustering)
        action_row.addWidget(self._run_btn)

        self._open_plots_btn = QPushButton("Open plots folder")
        self._open_plots_btn.clicked.connect(self._open_plots_folder)
        self._open_plots_btn.setEnabled(False)
        action_row.addWidget(self._open_plots_btn)
        action_row.addStretch(1)
        action_widget = QWidget(); action_widget.setLayout(action_row)
        cluster_form.addRow("", action_widget)

        self._status = QLabel("")
        self._status.setStyleSheet(
            f"color: {tokens.active().text_subtle}; font-size: {tokens.FS_CAPTION}px;"
        )
        self._status.setWordWrap(True)
        cluster_form.addRow("", self._status)

        root.addWidget(cluster_box)

        # ── Embedded library browser ──────────────────────────────────
        lib_box = QGroupBox("Feature library")
        lib_layout = QVBoxLayout(lib_box)
        lib_layout.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        self._library_browser = LibraryBrowser()
        lib_layout.addWidget(self._library_browser)
        root.addWidget(lib_box, stretch=1)

        self._last_plots_dir: Path | None = None

    # ------------------------------------------------------------------
    # External wiring
    # ------------------------------------------------------------------

    def set_library_dir(self, library_dir: Path | None) -> None:
        """Sync the embedded browser with the library dir from FeaturesPanel."""
        self._library_browser.set_library_dir(library_dir)

    def refresh_library(self) -> None:
        self._library_browser.refresh()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _browse_features_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select features directory")
        if d:
            self._features_dir.setText(d)

    def _open_plots_folder(self) -> None:
        if self._last_plots_dir and self._last_plots_dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_plots_dir)))

    def _run_clustering(self) -> None:
        text = self._features_dir.text().strip()
        if not text:
            QMessageBox.warning(
                self, "Missing input",
                "Choose a features directory containing all_features.parquet.",
            )
            return
        feat_dir = Path(text)
        if not (feat_dir / "all_features.parquet").exists():
            QMessageBox.warning(
                self, "Not found",
                f"No all_features.parquet under {feat_dir}",
            )
            return

        species = self._species.currentText().strip()
        run_id = self._run_id.text().strip() or _default_run_id(feat_dir)
        library_dir = self._library_browser._library_dir
        control_labels = [
            t.strip() for t in self._control_labels.text().split(",")
            if t and t.strip()
        ]

        self._status.setText("Running clustering\u2026")
        self._run_btn.setEnabled(False)
        try:
            from mycoprep.core.extract.qc_plots import make_qc_plots
            messages: list[str] = []
            def cb(_f: float, m: str) -> None:
                messages.append(m)
                self._status.setText(m)
            out_dir = make_qc_plots(
                feat_dir,
                library_dir=library_dir,
                species=species,
                current_run_id=run_id,
                control_labels=control_labels,
                progress_cb=cb,
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Clustering failed", str(e))
            self._status.setText(f"Failed: {e}")
            return
        finally:
            self._run_btn.setEnabled(True)

        if out_dir is None:
            self._status.setText("No plots generated (see status above).")
            return

        self._last_plots_dir = out_dir
        self._open_plots_btn.setEnabled(True)
        self._status.setText(f"Plots written to {out_dir}")
        self.statusMessage.emit(f"Clustering done: {out_dir}")
