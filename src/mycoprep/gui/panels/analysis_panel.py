"""Analysis panel — post-pipeline clustering and feature library tools.

This page lives after the Run stage and houses workflows that operate on
already-extracted features rather than raw images:

- Embedded interactive Plotly UMAP at the top of the page. Default view
  shows the feature library on its own; switches to a current-run-vs-
  library comparison after the user re-runs clustering.
- Embedded feature library browser (manage registered runs, import existing).
- Re-run morphology clustering on a chosen features directory + library
  selection, regenerating the clustering QC plots without rerunning the
  full pipeline.

Plot rendering runs on a worker QThread so the UI stays responsive and a
progress bar reflects the current step.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, Qt, QThread, QUrl, pyqtSignal
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
    QProgressBar,
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


# ─────────────────────────────────────────────────────────────────────────────
# Worker QObjects — run plot rendering off the GUI thread.
# ─────────────────────────────────────────────────────────────────────────────


class _LibraryWorker(QObject):
    """Render the library-only Plotly HTML in a worker thread."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)   # Path | None
    failed = pyqtSignal(str)

    def __init__(
        self,
        out_path: Path,
        library_dir: Optional[Path],
        species: str,
    ) -> None:
        super().__init__()
        self._out_path = out_path
        self._library_dir = library_dir
        self._species = species

    def run(self) -> None:
        try:
            from mycoprep.core.extract.qc_plots import render_library_html

            self.progress.emit(5, "Loading library\u2026")
            self.progress.emit(20, "Computing S-scores\u2026")
            written = render_library_html(
                self._out_path,
                library_dir=self._library_dir,
                species=self._species,
            )
            self.progress.emit(100, "Done")
            self.finished.emit(written)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _ClusteringWorker(QObject):
    """Run ``make_qc_plots`` (current run + optional library comparison)."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)   # Path | None (qc_plots dir)
    failed = pyqtSignal(str)

    def __init__(
        self,
        features_dir: Path,
        library_dir: Optional[Path],
        species: str,
        current_run_id: str,
        control_labels: list[str],
    ) -> None:
        super().__init__()
        self._features_dir = features_dir
        self._library_dir = library_dir
        self._species = species
        self._current_run_id = current_run_id
        self._control_labels = control_labels

    def run(self) -> None:
        try:
            from mycoprep.core.extract.qc_plots import make_qc_plots

            def cb(f: float, m: str) -> None:
                self.progress.emit(int(round(max(0.0, min(1.0, f)) * 100)), m)

            out = make_qc_plots(
                self._features_dir,
                library_dir=self._library_dir,
                species=self._species,
                current_run_id=self._current_run_id,
                control_labels=self._control_labels,
                progress_cb=cb,
            )
            self.finished.emit(out)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────


class AnalysisPanel(QWidget):
    """Clustering and feature library management with an embedded interactive
    plot view."""

    statusMessage = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._tmp_dir = Path(tempfile.mkdtemp(prefix="mycoprep_analysis_"))
        self._library_html_path = self._tmp_dir / "library.html"
        self._last_plots_dir: Path | None = None
        self._current_view: str = "library"  # "library" or "comparison"
        self._worker_thread: QThread | None = None
        self._worker: QObject | None = None
        # Track when we last saw the library change so we know whether
        # to re-render on tab entry.
        self._last_library_state: tuple[Path | None, str, float] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.S3)

        # ── Interactive plot view (top, largest) ───────────────────────
        self._web_view = QWebEngineView()
        self._web_view.setMinimumHeight(420)
        root.addWidget(self._web_view, stretch=3)

        # Progress strip — shown only while a worker is running.
        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(tokens.S3, 0, tokens.S3, 0)
        progress_row.setSpacing(tokens.S2)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setMinimumWidth(160)
        self._progress.setMaximumHeight(14)
        self._progress.setTextVisible(False)
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(
            f"color: {tokens.active().text_subtle}; font-size: {tokens.FS_CAPTION}px;"
        )
        progress_row.addWidget(self._progress)
        progress_row.addWidget(self._progress_label, stretch=1)
        progress_widget = QWidget(); progress_widget.setLayout(progress_row)
        root.addWidget(progress_widget)
        progress_widget.setVisible(False)
        self._progress_widget = progress_widget

        # ── Bottom: re-run controls (left) | library browser (right) ───
        self._bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

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

        lib_box = QGroupBox("Feature library")
        lib_layout = QVBoxLayout(lib_box)
        lib_layout.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        self._library_browser = LibraryBrowser()
        self._library_browser.libraryChanged.connect(self._on_library_changed)
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
        """Called when the user navigates to the Analysis page or after a
        pipeline run finishes. Re-renders the library plot only when the
        underlying library has actually changed since we last saw it, so
        switching tabs doesn't flicker."""
        self._library_browser.refresh()
        if self._library_state_changed():
            self._refresh_library_view()

    def on_pipeline_finished(self) -> None:
        """Hook invoked from MainWindow when a pipeline run finishes —
        the library may have just gained a new run, so re-render."""
        self._library_browser.refresh()
        self._refresh_library_view()

    # ------------------------------------------------------------------
    # Plot view management
    # ------------------------------------------------------------------

    def _set_placeholder(self, title: str, body: str) -> None:
        html = _PLACEHOLDER_HTML.format(title=title, body=body)
        self._library_html_path.write_text(html, encoding="utf-8")
        self._web_view.setUrl(QUrl.fromLocalFile(str(self._library_html_path)))

    def _library_state_changed(self) -> bool:
        """Detect whether the library_dir, species, or library.parquet
        mtime has changed since the last successful render."""
        species = self._species.currentText().strip()
        library_dir = self._library_browser._library_dir
        from mycoprep.core.extract.feature_library import FeatureLibrary
        try:
            lib = FeatureLibrary(library_dir)
            idx_path = lib.library_dir / "library.parquet"
            mtime = idx_path.stat().st_mtime if idx_path.exists() else 0.0
        except Exception:  # noqa: BLE001
            mtime = 0.0
        state = (library_dir, species, mtime)
        if state != self._last_library_state:
            return True
        return False

    def _on_library_changed(self) -> None:
        """The embedded browser flagged a content change (import / edit /
        remove). Always re-render, since any displayed plot is now stale."""
        self._refresh_library_view()

    def _refresh_library_view(self) -> None:
        """Render the library-only interactive plot (in a worker thread)
        and load it into the embedded view. Falls back to a placeholder
        when the library is empty or has too few conditions."""
        if self._is_busy():
            return
        species = self._species.currentText().strip()
        library_dir = self._library_browser._library_dir

        worker = _LibraryWorker(
            self._library_html_path, library_dir, species,
        )
        self._start_worker(
            worker,
            on_finished=self._on_library_render_finished,
            status="Rendering library plot\u2026",
        )

    def _on_library_render_finished(self, written: object) -> None:
        species = self._species.currentText().strip()
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
        # Cache the state we just rendered for so refresh_library() can
        # decide whether to re-render on next tab entry.
        library_dir = self._library_browser._library_dir
        from mycoprep.core.extract.feature_library import FeatureLibrary
        try:
            lib = FeatureLibrary(library_dir)
            idx_path = lib.library_dir / "library.parquet"
            mtime = idx_path.stat().st_mtime if idx_path.exists() else 0.0
        except Exception:  # noqa: BLE001
            mtime = 0.0
        self._last_library_state = (library_dir, species, mtime)

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
    # Worker plumbing
    # ------------------------------------------------------------------

    def _is_busy(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.isRunning()

    def _start_worker(
        self,
        worker: QObject,
        *,
        on_finished,
        status: str,
    ) -> None:
        self._progress.setValue(0)
        self._progress_label.setText(status)
        self._progress_widget.setVisible(True)
        self._run_btn.setEnabled(False)
        self._show_library_btn.setEnabled(False)
        self._status.setText(status)

        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_worker_progress)
        worker.finished.connect(lambda result: self._on_worker_done(
            on_finished, result, success=True,
        ))
        worker.failed.connect(lambda err: self._on_worker_done(
            on_finished, err, success=False,
        ))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._worker = worker
        self._worker_thread = thread
        thread.start()

    def _on_worker_progress(self, pct: int, msg: str) -> None:
        self._progress.setValue(int(pct))
        self._progress_label.setText(msg)
        self._status.setText(msg)

    def _on_worker_done(self, callback, result, *, success: bool) -> None:
        self._progress_widget.setVisible(False)
        self._run_btn.setEnabled(True)
        self._show_library_btn.setEnabled(True)
        self._worker = None
        self._worker_thread = None
        if success:
            try:
                callback(result)
            except Exception as e:  # noqa: BLE001
                self._status.setText(f"Render callback failed: {e}")
        else:
            QMessageBox.critical(self, "Plot generation failed", str(result))
            self._status.setText(f"Failed: {result}")

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
        if self._is_busy():
            return
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

        worker = _ClusteringWorker(
            feat_dir, library_dir, species, run_id, control_labels,
        )
        self._start_worker(
            worker,
            on_finished=self._on_clustering_finished,
            status="Running clustering\u2026",
        )

    def _on_clustering_finished(self, out_dir: object) -> None:
        if out_dir is None:
            self._status.setText("No plots generated (see status above).")
            return
        out_dir = Path(out_dir)
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
