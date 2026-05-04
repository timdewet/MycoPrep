"""Analysis panel — post-pipeline clustering and feature library tools.

This page lives after the Run stage and houses workflows that operate on
already-extracted features rather than raw images:

- Embedded interactive Plotly UMAP. Default view shows the feature
  library on its own; switches to a current-run-vs-library comparison
  after the user re-runs clustering.
- Embedded feature library browser (manage registered runs, import existing).
- Re-run morphology clustering on a chosen features directory + library
  selection, regenerating the clustering QC plots without rerunning the
  full pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWebEngineWidgets import QWebEngineView
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


_PLACEHOLDER_HTML = """\
<!doctype html>
<html><head><meta charset="utf-8"><style>
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;
     color:#666;text-align:center;padding:60px 20px;background:#fafafa;}}
h2{{color:#333;font-weight:500;margin-bottom:6px;}}
p{{margin:6px 0;font-size:13px;}}
</style></head><body>
<h2>{title}</h2>
<p>{body}</p>
</body></html>
"""


class AnalysisPanel(QWidget):
    """Clustering and feature library management with an embedded interactive
    plot view."""

    statusMessage = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Hold rendered HTMLs alive for the QWebEngineView.
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="mycoprep_analysis_"))
        self._library_html_path = self._tmp_dir / "library.html"
        self._last_plots_dir: Path | None = None
        self._current_view: str = "library"  # "library" or "comparison"

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.S3)

        # ── Interactive plot view (top, largest) ───────────────────────
        self._web_view = QWebEngineView()
        self._web_view.setMinimumHeight(420)
        root.addWidget(self._web_view, stretch=3)

        # ── Bottom: re-run controls (left) | library browser (right) ───
        self._bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

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
        self._species.currentTextChanged.connect(
            lambda _t: self._refresh_library_view()
        )
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

        self._show_library_btn = QPushButton("Show library only")
        self._show_library_btn.setToolTip(
            "Reload the interactive plot of just the feature library "
            "(no current run overlaid)."
        )
        self._show_library_btn.clicked.connect(self._refresh_library_view)
        action_row.addWidget(self._show_library_btn)

        self._open_plots_btn = QPushButton("Open plots folder")
        self._open_plots_btn.clicked.connect(self._open_plots_folder)
        self._open_plots_btn.setEnabled(False)
        action_row.addWidget(self._open_plots_btn)

        self._open_in_browser_btn = QPushButton("Open in browser")
        self._open_in_browser_btn.setToolTip(
            "Open the currently displayed interactive plot in your "
            "default browser (more screen space, native zoom)."
        )
        self._open_in_browser_btn.clicked.connect(self._open_current_in_browser)
        action_row.addWidget(self._open_in_browser_btn)

        action_row.addStretch(1)
        action_widget = QWidget(); action_widget.setLayout(action_row)
        cluster_form.addRow("", action_widget)

        self._status = QLabel("")
        self._status.setStyleSheet(
            f"color: {tokens.active().text_subtle}; font-size: {tokens.FS_CAPTION}px;"
        )
        self._status.setWordWrap(True)
        cluster_form.addRow("", self._status)

        self._bottom_splitter.addWidget(cluster_box)

        # ── Library browser (right of bottom splitter) ─────────────────
        lib_box = QGroupBox("Feature library")
        lib_layout = QVBoxLayout(lib_box)
        lib_layout.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        self._library_browser = LibraryBrowser()
        self._library_browser.libraryChanged.connect(self._refresh_library_view)
        lib_layout.addWidget(self._library_browser)
        self._bottom_splitter.addWidget(lib_box)

        self._bottom_splitter.setStretchFactor(0, 1)
        self._bottom_splitter.setStretchFactor(1, 1)
        root.addWidget(self._bottom_splitter, stretch=2)

        self._set_placeholder(
            "Feature library", "Loading\u2026 (will appear here once the library has data).",
        )

    # ------------------------------------------------------------------
    # External wiring
    # ------------------------------------------------------------------

    def set_library_dir(self, library_dir: Path | None) -> None:
        """Sync the embedded browser with the library dir from FeaturesPanel."""
        self._library_browser.set_library_dir(library_dir)
        self._refresh_library_view()

    def refresh_library(self) -> None:
        """Called when the user navigates to the Analysis page."""
        self._library_browser.refresh()
        if self._current_view == "library":
            self._refresh_library_view()

    # ------------------------------------------------------------------
    # Plot view management
    # ------------------------------------------------------------------

    def _set_placeholder(self, title: str, body: str) -> None:
        html = _PLACEHOLDER_HTML.format(title=title, body=body)
        self._library_html_path.write_text(html, encoding="utf-8")
        self._web_view.setUrl(QUrl.fromLocalFile(str(self._library_html_path)))

    def _refresh_library_view(self) -> None:
        """Render the library-only interactive plot and load it into the
        embedded view. Falls back to a placeholder when the library is
        empty or has too few conditions."""
        from mycoprep.core.extract.qc_plots import render_library_html

        species = self._species.currentText().strip()
        library_dir = self._library_browser._library_dir

        try:
            written = render_library_html(
                self._library_html_path,
                library_dir=library_dir,
                species=species,
            )
        except Exception as e:  # noqa: BLE001
            self._set_placeholder(
                "Library plot failed", f"{e}",
            )
            self._current_view = "library"
            return

        if written is None:
            sp_label = species or "any species"
            self._set_placeholder(
                "No library data",
                f"No registered runs match {sp_label}. Add runs from the "
                "Features panel during a pipeline run, or use \u201cImport "
                "existing parquet\u201d in the library browser to seed it.",
            )
        else:
            self._web_view.setUrl(QUrl.fromLocalFile(str(written)))
        self._current_view = "library"

    def _show_comparison(self, html_path: Path) -> None:
        if not html_path.exists():
            QMessageBox.information(
                self, "Plot not available",
                f"{html_path.name} was not produced for the last run.",
            )
            return
        self._web_view.setUrl(QUrl.fromLocalFile(str(html_path)))
        self._current_view = "comparison"

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _browse_features_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select features directory")
        if d:
            self._features_dir.setText(d)

    def _open_plots_folder(self) -> None:
        if self._last_plots_dir and self._last_plots_dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_plots_dir)))

    def _open_current_in_browser(self) -> None:
        url = self._web_view.url()
        if url.isValid():
            QDesktopServices.openUrl(url)

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
            def cb(_f: float, m: str) -> None:
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

        # Prefer the comparison-with-library plot; fall back to solo.
        comparison_html = out_dir / "morphology_clustering_library.html"
        solo_html = out_dir / "morphology_clustering_run.html"
        if comparison_html.exists():
            self._show_comparison(comparison_html)
            self._status.setText(
                f"Showing current run vs library. Plots in {out_dir}."
            )
        elif solo_html.exists():
            self._show_comparison(solo_html)
            self._status.setText(
                f"Library was empty — showing solo run. Plots in {out_dir}."
            )
        else:
            self._status.setText(f"Plots written to {out_dir}")
        self.statusMessage.emit(f"Clustering done: {out_dir}")
