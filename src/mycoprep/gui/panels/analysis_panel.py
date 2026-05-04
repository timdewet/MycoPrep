"""Analysis panel — interactive UMAP of the feature library.

The page is a single full-bleed plot view. The library is the source of
truth: anything you want to see lives in the library, registered either
during a pipeline run or via the library browser's "Import existing
parquet" action. Removing this surface entirely would leave no place for
the interactive plot, but everything else has been pared back so the
UMAP gets the room it deserves.

Workers render plots on a QThread; a translucent overlay greys out the
plot area and shows a progress bar while a render is running.
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
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from ..ui import tokens

SPECIES_OPTIONS = ["M. tuberculosis", "M. smegmatis"]


_PLACEHOLDER_HTML = """\
<!doctype html>
<html><head><meta charset="utf-8"><style>
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;
     color:#666;text-align:center;padding:80px 20px;background:#fafafa;}}
h2{{color:#333;font-weight:500;margin-bottom:6px;}}
p{{margin:6px 0;font-size:13px;}}
</style></head><body>
<h2>{title}</h2>
<p>{body}</p>
</body></html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Translucent overlay used as a "loading" curtain over the plot area
# ─────────────────────────────────────────────────────────────────────────────


class _PlotOverlay(QWidget):
    """A semi-transparent grey curtain with a centred progress bar."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAutoFillBackground(False)
        self.setStyleSheet(
            "background-color: rgba(40, 40, 40, 110);"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch(1)

        card = QFrame()
        card.setObjectName("loadingCard")
        card.setStyleSheet(
            "QFrame#loadingCard {"
            f"  background-color: white; border-radius: {tokens.R_LG}px;"
            "}"
        )
        card.setMinimumWidth(320)
        card.setMaximumWidth(420)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(tokens.S5, tokens.S4, tokens.S5, tokens.S4)
        card_layout.setSpacing(tokens.S2)

        self._label = QLabel("Rendering\u2026")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(f"color: {tokens.active().text_muted};")
        card_layout.addWidget(self._label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        self._progress.setMaximumHeight(10)
        card_layout.addWidget(self._progress)

        # Center the card horizontally
        card_row = QHBoxLayout()
        card_row.addStretch(1)
        card_row.addWidget(card)
        card_row.addStretch(1)
        outer.addLayout(card_row)
        outer.addStretch(1)

        self.hide()

    def set_progress(self, pct: int, msg: str) -> None:
        self._progress.setValue(int(pct))
        if msg:
            self._label.setText(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Worker — renders the library Plotly HTML in a thread
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
        color_by: str = "cluster",
        feature_col: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._out_path = out_path
        self._library_dir = library_dir
        self._species = species
        self._color_by = color_by
        self._feature_col = feature_col

    def run(self) -> None:
        try:
            from mycoprep.core.extract.qc_plots import render_library_html

            self.progress.emit(10, "Loading library\u2026")
            self.progress.emit(35, "Computing S-scores\u2026")
            self.progress.emit(60, "Embedding (UMAP)\u2026")
            written = render_library_html(
                self._out_path,
                library_dir=self._library_dir,
                species=self._species,
                color_by=self._color_by,
                feature_col=self._feature_col,
            )
            self.progress.emit(100, "Done")
            self.finished.emit(written)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────


class AnalysisPanel(QWidget):
    """Full-bleed interactive UMAP of the feature library."""

    statusMessage = pyqtSignal(str)
    openLibraryBrowserRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._tmp_dir = Path(tempfile.mkdtemp(prefix="mycoprep_analysis_"))
        self._library_html_path = self._tmp_dir / "library.html"
        self._library_dir: Path | None = None
        self._worker_thread: QThread | None = None
        self._worker: QObject | None = None
        # Cached library state (path, species, mtime) — avoid re-rendering
        # on tab switches when nothing has changed.
        self._last_library_state: tuple[Path | None, str, float] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.S2)

        # ── Top toolbar ────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(tokens.S4, tokens.S3, tokens.S4, 0)
        toolbar.setSpacing(tokens.S2)

        toolbar.addWidget(QLabel("Species:"))
        self._species = QComboBox()
        self._species.addItems([""] + SPECIES_OPTIONS)
        self._species.currentTextChanged.connect(
            lambda _t: self._refresh_library_view(force=True)
        )
        toolbar.addWidget(self._species)
        toolbar.addStretch(1)

        self._library_btn = QPushButton("Library browser\u2026")
        self._library_btn.setToolTip(
            "Open the feature library to add, edit, or remove registered runs."
        )
        self._library_btn.clicked.connect(self.openLibraryBrowserRequested.emit)
        toolbar.addWidget(self._library_btn)

        self._open_browser_btn = QPushButton("Open in browser")
        self._open_browser_btn.setToolTip(
            "Open the current interactive plot in your default browser."
        )
        self._open_browser_btn.clicked.connect(self._open_current_in_browser)
        toolbar.addWidget(self._open_browser_btn)

        root.addLayout(toolbar)

        # ── Plot area with translucent overlay ─────────────────────────
        plot_container = QWidget()
        plot_stack = QStackedLayout(plot_container)
        plot_stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        plot_stack.setContentsMargins(tokens.S3, tokens.S2, tokens.S3, tokens.S3)

        self._web_view = QWebEngineView()
        self._web_view.setMinimumHeight(420)
        plot_stack.addWidget(self._web_view)

        self._overlay = _PlotOverlay(plot_container)
        plot_stack.addWidget(self._overlay)

        root.addWidget(plot_container, stretch=1)

        # ── Bottom: colour-by selector + status pill ───────────────────
        colour_row = QHBoxLayout()
        colour_row.setContentsMargins(tokens.S4, 0, tokens.S4, 0)
        colour_row.setSpacing(tokens.S2)

        colour_row.addWidget(QLabel("Colour by:"))
        self._color_by = QComboBox()
        self._color_by.addItem("Cluster", userData="cluster")
        self._color_by.addItem("Run ID", userData="run_id")
        self._color_by.addItem("Feature gradient\u2026", userData="feature")
        self._color_by.currentIndexChanged.connect(self._on_color_by_changed)
        colour_row.addWidget(self._color_by)

        self._feature_col = QComboBox()
        self._feature_col.setMinimumWidth(220)
        self._feature_col.setEnabled(False)
        self._feature_col.currentIndexChanged.connect(
            lambda _i: self._refresh_library_view(force=True)
            if self._color_by.currentData() == "feature" else None
        )
        colour_row.addWidget(self._feature_col)

        colour_row.addStretch(1)

        self._status = QLabel("")
        self._status.setStyleSheet(
            f"color: {tokens.active().text_subtle}; "
            f"font-size: {tokens.FS_CAPTION}px;"
        )
        self._status.setWordWrap(True)
        colour_row.addWidget(self._status, stretch=2)

        root.addLayout(colour_row)

        self._set_placeholder(
            "Feature library",
            "Loading\u2026 (will appear here once the library has data).",
        )

    # ------------------------------------------------------------------
    # External wiring
    # ------------------------------------------------------------------

    def set_library_dir(self, library_dir: Path | None) -> None:
        """Sync the library dir from FeaturesPanel and re-render the plot."""
        self._library_dir = library_dir
        self._refresh_library_view(force=True)

    def refresh_library(self) -> None:
        """Called when the user navigates to the Analysis page. Re-renders
        only when the underlying library has actually changed since we
        last saw it."""
        if self._library_state_changed():
            self._refresh_library_view(force=True)

    def on_pipeline_finished(self) -> None:
        """Hook from MainWindow when a pipeline run completes — the library
        may have just gained a new run, so re-render unconditionally."""
        self._refresh_library_view(force=True)

    def on_library_changed_external(self) -> None:
        """The popup library browser changed (import / edit / remove)."""
        self._refresh_library_view(force=True)

    # ------------------------------------------------------------------
    # Plot view management
    # ------------------------------------------------------------------

    def _set_placeholder(self, title: str, body: str) -> None:
        html = _PLACEHOLDER_HTML.format(title=title, body=body)
        self._library_html_path.write_text(html, encoding="utf-8")
        self._web_view.setUrl(QUrl.fromLocalFile(str(self._library_html_path)))

    def _library_state_changed(self) -> bool:
        species = self._species.currentText().strip()
        from mycoprep.core.extract.feature_library import FeatureLibrary
        try:
            lib = FeatureLibrary(self._library_dir)
            idx_path = lib.library_dir / "library.parquet"
            mtime = idx_path.stat().st_mtime if idx_path.exists() else 0.0
        except Exception:  # noqa: BLE001
            mtime = 0.0
        state = (self._library_dir, species, mtime)
        return state != self._last_library_state

    def _refresh_library_view(self, force: bool = False) -> None:
        if self._is_busy():
            return
        if not force and not self._library_state_changed():
            return

        species = self._species.currentText().strip()
        # Make sure the feature dropdown reflects the current library state
        # before we kick off the worker (so a feature-mode render finds a
        # valid column, not a stale one from a different species).
        self._populate_feature_combo()
        color_by = self._color_by.currentData() or "cluster"
        feature_col = (
            self._feature_col.currentText() or None
            if color_by == "feature" else None
        )
        worker = _LibraryWorker(
            self._library_html_path, self._library_dir, species,
            color_by=color_by, feature_col=feature_col,
        )
        self._start_worker(
            worker,
            on_finished=self._on_library_render_finished,
            status="Rendering library plot\u2026",
        )

    def _on_color_by_changed(self, _idx: int) -> None:
        mode = self._color_by.currentData() or "cluster"
        is_feature = (mode == "feature")
        self._feature_col.setEnabled(is_feature)
        if is_feature:
            self._populate_feature_combo()
            if self._feature_col.count() == 0:
                self._status.setText(
                    "No feature columns available — library is empty for this species."
                )
                return
        # Re-render with the new colouring.
        self._refresh_library_view(force=True)

    def _populate_feature_combo(self) -> None:
        """Populate the feature combo from the library if not already done.

        Preserves the user's current selection when possible so a render
        triggered by another control change doesn't reset it.
        """
        from mycoprep.core.extract.qc_plots import library_feature_columns

        species = self._species.currentText().strip()
        try:
            cols = library_feature_columns(
                library_dir=self._library_dir, species=species,
            )
        except Exception:  # noqa: BLE001
            cols = []
        prev = self._feature_col.currentText()
        if [self._feature_col.itemText(i) for i in range(self._feature_col.count())] == cols:
            return  # no change
        self._feature_col.blockSignals(True)
        self._feature_col.clear()
        self._feature_col.addItems(cols)
        if prev and prev in cols:
            self._feature_col.setCurrentText(prev)
        self._feature_col.blockSignals(False)

    def _on_library_render_finished(self, written: object) -> None:
        species = self._species.currentText().strip()
        if written is None:
            sp_label = species or "any species"
            self._set_placeholder(
                "No library data",
                f"No registered runs match {sp_label}. Add runs from the "
                "Features panel during a pipeline run, or use \u201cLibrary "
                "browser\u201d \u2192 \u201cImport existing parquet\u201d.",
            )
            self._status.setText("")
        else:
            self._web_view.setUrl(QUrl.fromLocalFile(str(written)))
            self._status.setText("")
        from mycoprep.core.extract.feature_library import FeatureLibrary
        try:
            lib = FeatureLibrary(self._library_dir)
            idx_path = lib.library_dir / "library.parquet"
            mtime = idx_path.stat().st_mtime if idx_path.exists() else 0.0
        except Exception:  # noqa: BLE001
            mtime = 0.0
        self._last_library_state = (self._library_dir, species, mtime)

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
        self._overlay.set_progress(0, status)
        self._overlay.show()
        self._overlay.raise_()
        self._library_btn.setEnabled(False)
        self._open_browser_btn.setEnabled(False)

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
        self._overlay.set_progress(pct, msg)

    def _on_worker_done(self, callback, result, *, success: bool) -> None:
        self._overlay.hide()
        self._library_btn.setEnabled(True)
        self._open_browser_btn.setEnabled(True)
        self._worker = None
        self._worker_thread = None
        if success:
            try:
                callback(result)
            except Exception as e:  # noqa: BLE001
                self._status.setText(f"Render callback failed: {e}")
        else:
            self._status.setText(f"Failed: {result}")
            self._set_placeholder("Plot generation failed", str(result))

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _open_current_in_browser(self) -> None:
        url = self._web_view.url()
        if url.isValid():
            QDesktopServices.openUrl(url)
