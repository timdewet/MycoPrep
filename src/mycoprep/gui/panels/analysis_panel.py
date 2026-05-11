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
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedLayout,
    QToolButton,
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
# Multi-select dropdown — QToolButton with a popup of checkable actions
# ─────────────────────────────────────────────────────────────────────────────


class _MultiSelectButton(QToolButton):
    """A button that opens a popup of checkable items.

    The label summarises the current selection ("(none)", "<single name>",
    or "N selected"). Emits ``selectionChanged`` whenever the user toggles
    an entry.
    """

    selectionChanged = pyqtSignal(list)

    def __init__(self, placeholder: str = "(none)", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._placeholder = placeholder
        self._items: list[str] = []
        self._checked: set[str] = set()
        self._menu = QMenu(self)
        self.setMenu(self._menu)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setMinimumWidth(180)
        self.setText(placeholder)

    def set_items(self, items: list[str]) -> None:
        self._menu.clear()
        self._items = list(items)
        self._checked = {c for c in self._checked if c in items}
        if self._items:
            clear_action = QAction("Clear selection", self._menu)
            clear_action.triggered.connect(self.clear_selection)
            self._menu.addAction(clear_action)
            self._menu.addSeparator()
        for item in self._items:
            action = QAction(item, self._menu)
            action.setCheckable(True)
            action.setChecked(item in self._checked)
            action.triggered.connect(
                lambda _checked, n=item: self._toggle(n)
            )
            self._menu.addAction(action)
        self._update_label()

    def selected(self) -> list[str]:
        return [n for n in self._items if n in self._checked]

    def set_selected(self, names: list[str]) -> None:
        self._checked = {n for n in names if n in self._items}
        # Refresh the menu's checked states
        for action in self._menu.actions():
            if action.isCheckable():
                action.setChecked(action.text() in self._checked)
        self._update_label()

    def clear_selection(self) -> None:
        if not self._checked:
            return
        self._checked.clear()
        for action in self._menu.actions():
            if action.isCheckable():
                action.setChecked(False)
        self._update_label()
        self.selectionChanged.emit(self.selected())

    def _toggle(self, name: str) -> None:
        if name in self._checked:
            self._checked.discard(name)
        else:
            self._checked.add(name)
        self._update_label()
        self.selectionChanged.emit(self.selected())

    def _update_label(self) -> None:
        n = len(self._checked)
        if n == 0:
            self.setText(self._placeholder)
        elif n == 1:
            self.setText(next(iter(self._checked)))
        else:
            self.setText(f"{n} selected")


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


class _CompareWorker(QObject):
    """Render a per-feature comparison heatmap for a given gene set."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        out_path: Path,
        library_dir: Optional[Path],
        species: str,
        genes: list[str],
        baseline_mode: str,
    ) -> None:
        super().__init__()
        self._out_path = out_path
        self._library_dir = library_dir
        self._species = species
        self._genes = list(genes)
        self._baseline_mode = baseline_mode

    def run(self) -> None:
        try:
            from mycoprep.core.extract.qc_plots import render_comparison_html

            self.progress.emit(15, "Loading library\u2026")
            self.progress.emit(40, "Computing S-scores\u2026")
            self.progress.emit(75, "Building heatmap\u2026")
            written = render_comparison_html(
                self._out_path,
                library_dir=self._library_dir,
                species=self._species,
                genes=self._genes,
                baseline_mode=self._baseline_mode,
            )
            self.progress.emit(100, "Done")
            self.finished.emit(written)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _EmbeddingsWorker(QObject):
    """Render UMAP of CNN embeddings in a worker thread."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        out_path: Path,
        library_dir: Optional[Path],
        species: str,
        color_by: str = "cluster",
        feature_col: Optional[str] = None,
        highlight_genes: Optional[list[str]] = None,
        batch_correct: bool = True,
        model_type: str = "",
    ) -> None:
        super().__init__()
        self._out_path = out_path
        self._library_dir = library_dir
        self._species = species
        self._color_by = color_by
        self._feature_col = feature_col
        self._highlight_genes = list(highlight_genes or [])
        self._batch_correct = batch_correct
        self._model_type = model_type

    def run(self) -> None:
        try:
            from mycoprep.core.extract.qc_plots import render_embeddings_html

            self.progress.emit(10, "Loading embeddings\u2026")
            self.progress.emit(50, "Computing UMAP\u2026")
            written = render_embeddings_html(
                self._out_path,
                library_dir=self._library_dir,
                species=self._species,
                color_by=self._color_by,
                feature_col=self._feature_col,
                highlight_genes=self._highlight_genes,
                batch_correct=self._batch_correct,
                model_type=self._model_type,
            )
            self.progress.emit(100, "Done")
            self.finished.emit(written)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _EmbeddingsOTWorker(QObject):
    """Render UMAP of CNN embeddings via Optimal Transport in a worker thread."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        out_path: Path,
        library_dir: Optional[Path],
        species: str,
        color_by: str = "cluster",
        feature_col: Optional[str] = None,
        highlight_genes: Optional[list[str]] = None,
        batch_correct: bool = True,
        model_type: str = "",
        n_neighbors: int = 5,
        pathway_csv: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self._out_path = out_path
        self._library_dir = library_dir
        self._species = species
        self._color_by = color_by
        self._feature_col = feature_col
        self._highlight_genes = list(highlight_genes or [])
        self._batch_correct = batch_correct
        self._model_type = model_type
        self._n_neighbors = int(n_neighbors)
        self._pathway_csv = pathway_csv

    def run(self) -> None:
        try:
            from mycoprep.core.extract.qc_plots import render_embeddings_ot_html

            self.progress.emit(5, "Loading embeddings…")

            def _cb(f: float, msg: str = "") -> None:
                pct = int(5 + 90 * max(0.0, min(1.0, f)))
                self.progress.emit(
                    pct,
                    msg if msg else f"Computing OT distance matrix… {pct}%",
                )

            written = render_embeddings_ot_html(
                self._out_path,
                library_dir=self._library_dir,
                species=self._species,
                color_by=self._color_by,
                feature_col=self._feature_col,
                highlight_genes=self._highlight_genes,
                batch_correct=self._batch_correct,
                model_type=self._model_type,
                n_neighbors=self._n_neighbors,
                pathway_csv=self._pathway_csv,
                progress_cb=_cb,
            )
            self.progress.emit(100, "Done")
            self.finished.emit(written)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _FeaturesOTWorker(QObject):
    """Render UMAP of S-score features via Optimal Transport in a worker thread."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        out_path: Path,
        library_dir: Optional[Path],
        species: str,
        color_by: str = "cluster",
        feature_col: Optional[str] = None,
        highlight_genes: Optional[list[str]] = None,
        batch_correct: bool = True,
        n_neighbors: int = 5,
        pathway_csv: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self._out_path = out_path
        self._library_dir = library_dir
        self._species = species
        self._color_by = color_by
        self._feature_col = feature_col
        self._highlight_genes = list(highlight_genes or [])
        self._batch_correct = batch_correct
        self._n_neighbors = int(n_neighbors)
        self._pathway_csv = pathway_csv

    def run(self) -> None:
        try:
            from mycoprep.core.extract.qc_plots import render_features_ot_html

            self.progress.emit(5, "Loading features…")

            def _cb(f: float, msg: str = "") -> None:
                pct = int(5 + 90 * max(0.0, min(1.0, f)))
                self.progress.emit(
                    pct,
                    msg if msg else f"Computing OT distance matrix… {pct}%",
                )

            written = render_features_ot_html(
                self._out_path,
                library_dir=self._library_dir,
                species=self._species,
                color_by=self._color_by,
                feature_col=self._feature_col,
                highlight_genes=self._highlight_genes,
                batch_correct=self._batch_correct,
                n_neighbors=self._n_neighbors,
                pathway_csv=self._pathway_csv,
                progress_cb=_cb,
            )
            self.progress.emit(100, "Done")
            self.finished.emit(written)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


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
        highlight_genes: Optional[list[str]] = None,
        baseline_mode: str = "pooled",
        batch_correct: bool = True,
    ) -> None:
        super().__init__()
        self._out_path = out_path
        self._library_dir = library_dir
        self._species = species
        self._color_by = color_by
        self._feature_col = feature_col
        self._highlight_genes = list(highlight_genes or [])
        self._baseline_mode = baseline_mode
        self._batch_correct = batch_correct

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
                highlight_genes=self._highlight_genes,
                baseline_mode=self._baseline_mode,
                batch_correct=self._batch_correct,
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
        self._comparison_html_path = self._tmp_dir / "comparison.html"
        self._library_dir: Path | None = None
        self._worker_thread: QThread | None = None
        self._worker: QObject | None = None
        self._compare_window: "_ComparisonWindow | None" = None
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

        toolbar.addWidget(QLabel("View:"))
        self._view_mode = QComboBox()
        self._view_mode.addItem("Feature profiles (mean)", userData="features")
        self._view_mode.addItem("Feature profiles (OT)", userData="features_ot")
        self._view_mode.addItem("CNN embeddings (mean)", userData="embeddings")
        self._view_mode.addItem("CNN embeddings (OT)", userData="embeddings_ot")
        self._view_mode.setToolTip(
            "Mean views: UMAP of per-condition mean profiles (cosine sim).\n"
            "OT views: UMAP of pairwise Sinkhorn (Optimal Transport) "
            "distances between full per-condition cell distributions — "
            "captures distribution shape, not just the mean. Reveals "
            "heterogeneous knockdowns the mean view hides.\n\n"
            "Feature views read S-score morphological features.\n"
            "CNN embedding views read 512-d learned encoder features."
        )
        self._view_mode.currentIndexChanged.connect(
            lambda _i: self._on_view_mode_changed()
        )
        toolbar.addWidget(self._view_mode)
        toolbar.addSpacing(tokens.S3)

        # Model dropdown — only meaningful in CNN embeddings mode. Lists
        # each trained architecture (resnet18, supcon_resnet18, …) that
        # has saved embeddings; "Latest" picks the most recently-extracted.
        self._model_label = QLabel("Model:")
        self._model_select = QComboBox()
        self._model_select.addItem("Latest", userData="")
        self._model_select.setToolTip(
            "Which trained CNN embedding to display.\n"
            "Each architecture (autoencoder, SupCon, …) writes its embeddings\n"
            "to its own subdirectory and stays available for comparison."
        )
        self._model_select.currentIndexChanged.connect(
            lambda _i: self._refresh_library_view(force=True)
        )
        toolbar.addWidget(self._model_label)
        toolbar.addWidget(self._model_select)
        toolbar.addSpacing(tokens.S3)
        # Hide initially; shown when the user picks "CNN embeddings".
        self._model_label.setVisible(False)
        self._model_select.setVisible(False)

        toolbar.addWidget(QLabel("Species:"))
        self._species = QComboBox()
        # Species are not combinable on a single embedding (their
        # control sets, condition libraries, and morphology
        # distributions don't overlap meaningfully). Force a real
        # selection rather than an "any species" default.
        self._species.addItems(SPECIES_OPTIONS)
        self._species.setCurrentText("M. tuberculosis")
        self._species.currentTextChanged.connect(
            lambda _t: self._refresh_library_view(force=True)
        )
        toolbar.addWidget(self._species)
        toolbar.addStretch(1)

        self._library_btn = QPushButton("Library browser\u2026")
        self._library_btn.setToolTip(
            "Open the morphology library to add, edit, or remove registered runs."
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
        # "Condition" colours by the inducer/treatment state alone
        # ("ATc-", "ATc+", drug concentration, etc.). The gene/mutant
        # is separate — use the highlight-genes selector for that.
        self._color_by.addItem("Condition", userData="atc")
        self._color_by.addItem("Reporter", userData="reporter")
        self._color_by.addItem("Replica", userData="replica")
        self._color_by.addItem("Pathway", userData="pathway")
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

        colour_row.addSpacing(tokens.S4)
        colour_row.addWidget(QLabel("Baseline:"))
        self._baseline_mode = QComboBox()
        self._baseline_mode.addItem("Pooled controls", userData="pooled")
        self._baseline_mode.addItem("Per-run controls", userData="per_run")
        self._baseline_mode.setToolTip(
            "How to compute the S-score baseline.\n\n"
            "Pooled: every control profile across every run contributes\n"
            "  to a single (mu, sigma). Stable but lets between-run\n"
            "  variance leak into sigma.\n\n"
            "Per-run: each run's controls form their own baseline; each\n"
            "  perturbation is z-scored against its own run's controls.\n"
            "  Removes batch effects but needs \u22652 controls per run."
        )
        self._baseline_mode.currentIndexChanged.connect(
            lambda _i: self._refresh_library_view(force=True)
        )
        colour_row.addWidget(self._baseline_mode)

        colour_row.addSpacing(tokens.S4)
        self._batch_correct = QCheckBox("Batch correct")
        self._batch_correct.setChecked(True)
        self._batch_correct.setToolTip(
            "Apply Harmony batch correction to align PCA embeddings\n"
            "across runs before UMAP. Recommended when combining\n"
            "data from multiple experiments — controls from different\n"
            "runs should converge instead of clustering per-experiment.\n\n"
            "Requires the harmonypy package; silently skipped otherwise."
        )
        self._batch_correct.stateChanged.connect(
            lambda _s: self._refresh_library_view(force=True)
        )
        colour_row.addWidget(self._batch_correct)

        colour_row.addSpacing(tokens.S4)
        colour_row.addWidget(QLabel("Highlight gene(s):"))
        self._highlight_genes = _MultiSelectButton("(none)")
        self._highlight_genes.setToolTip(
            "Highlight points whose mutant/gene matches the selected "
            "name(s); other points are dimmed. Empty = no highlighting."
        )
        self._highlight_genes.selectionChanged.connect(self._on_highlight_changed)
        colour_row.addWidget(self._highlight_genes)

        self._compare_btn = QPushButton("Compare\u2026")
        self._compare_btn.setToolTip(
            "Open a per-feature S-score heatmap for the highlighted "
            "gene(s) so you can see exactly where their profiles "
            "diverge across runs / ATc states."
        )
        self._compare_btn.clicked.connect(self._open_comparison)
        self._compare_btn.setEnabled(False)
        colour_row.addWidget(self._compare_btn)

        # \u2500\u2500 OT-only controls \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        # Visible only when an OT view is selected.
        colour_row.addSpacing(tokens.S4)
        self._ot_nn_label = QLabel("OT n_neighbors:")
        colour_row.addWidget(self._ot_nn_label)
        self._ot_nn = QSpinBox()
        self._ot_nn.setRange(2, 50)
        self._ot_nn.setValue(5)
        self._ot_nn.setToolTip(
            "n_neighbors for the precomputed-distance UMAP. Smaller =\n"
            "tighter local clusters (matches the Mtb reference, default 5).\n"
            "Larger = more global structure but tends to dissolve real\n"
            "phenotype groups into one big blob."
        )
        self._ot_nn.valueChanged.connect(
            lambda _v: self._refresh_library_view(force=True),
        )
        colour_row.addWidget(self._ot_nn)

        self._pathway_btn = QPushButton("Pathway map\u2026")
        self._pathway_btn.setToolTip(
            "Load a CSV with 'gene' and 'pathway' columns to colour points\n"
            "by pathway membership. Selected when 'Colour by' = Pathway."
        )
        self._pathway_btn.clicked.connect(self._pick_pathway_csv)
        colour_row.addWidget(self._pathway_btn)
        self._pathway_csv: Path | None = None

        self._ranked_btn = QPushButton("Ranked matches CSV\u2026")
        self._ranked_btn.setToolTip(
            "From the cached OT distance matrix, export each condition's\n"
            "top-K nearest other conditions ordered by Sinkhorn distance.\n"
            "Run an OT view first to populate the cache."
        )
        self._ranked_btn.clicked.connect(self._export_ranked_matches)
        colour_row.addWidget(self._ranked_btn)

        self._perm_btn = QPushButton("Permutation test CSV\u2026")
        self._perm_btn.setToolTip(
            "Run a 1000-permutation test on each query's top-1 nearest\n"
            "match (CNN embeddings only); export distances + BH-FDR\n"
            "q-values."
        )
        self._perm_btn.clicked.connect(self._export_permutation_test)
        colour_row.addWidget(self._perm_btn)

        colour_row.addStretch(1)

        self._status = QLabel("")
        self._status.setStyleSheet(
            f"color: {tokens.active().text_subtle}; "
            f"font-size: {tokens.FS_CAPTION}px;"
        )
        self._status.setWordWrap(True)
        colour_row.addWidget(self._status, stretch=2)

        root.addLayout(colour_row)

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
        view_mode = self._view_mode.currentData() or "features"
        from mycoprep.core.extract.feature_library import FeatureLibrary
        try:
            lib = FeatureLibrary(self._library_dir)
            idx_path = lib.library_dir / "library.parquet"
            mtime = idx_path.stat().st_mtime if idx_path.exists() else 0.0
            # Embeddings live in a separate file; track its mtime too so a
            # fresh training run is reflected on the Analysis tab even
            # though library.parquet itself is unchanged.
            emb_path = lib.models_dir / "embeddings" / "cnn_embeddings.parquet"
            emb_mtime = emb_path.stat().st_mtime if emb_path.exists() else 0.0
        except Exception:  # noqa: BLE001
            mtime = 0.0
            emb_mtime = 0.0
        state = (self._library_dir, species, view_mode, mtime, emb_mtime)
        return state != self._last_library_state

    def _refresh_library_view(self, force: bool = False) -> None:
        if self._is_busy():
            return
        if not force and not self._library_state_changed():
            return

        species = self._species.currentText().strip()
        # Refresh combos that depend on library state before kicking off
        # the worker (so a feature-mode render finds a valid column and
        # the gene highlighter offers the right names).
        self._populate_feature_combo()
        self._populate_gene_combo()
        color_by = self._color_by.currentData() or "cluster"
        feature_col = (
            self._feature_col.currentText() or None
            if color_by == "feature" else None
        )
        baseline_mode = self._baseline_mode.currentData() or "pooled"
        view_mode = self._view_mode.currentData() or "features"
        model_type = self._model_select.currentData() or ""
        n_neighbors = int(self._ot_nn.value())
        pathway_csv = self._pathway_csv

        if view_mode == "features_ot":
            worker = _FeaturesOTWorker(
                self._library_html_path, self._library_dir, species,
                color_by=color_by, feature_col=feature_col,
                highlight_genes=self._highlight_genes.selected(),
                batch_correct=self._batch_correct.isChecked(),
                n_neighbors=n_neighbors,
                pathway_csv=pathway_csv,
            )
            status = "Rendering Feature Profiles (OT)\u2026"
        elif view_mode == "embeddings":
            worker = _EmbeddingsWorker(
                self._library_html_path, self._library_dir, species,
                color_by=color_by, feature_col=feature_col,
                highlight_genes=self._highlight_genes.selected(),
                batch_correct=self._batch_correct.isChecked(),
                model_type=str(model_type),
            )
            status = "Rendering CNN Embeddings (mean)\u2026"
        elif view_mode == "embeddings_ot":
            worker = _EmbeddingsOTWorker(
                self._library_html_path, self._library_dir, species,
                color_by=color_by, feature_col=feature_col,
                highlight_genes=self._highlight_genes.selected(),
                batch_correct=self._batch_correct.isChecked(),
                model_type=str(model_type),
                n_neighbors=n_neighbors,
                pathway_csv=pathway_csv,
            )
            status = "Rendering CNN Embeddings (OT)\u2026"
        else:
            worker = _LibraryWorker(
                self._library_html_path, self._library_dir, species,
                color_by=color_by, feature_col=feature_col,
                highlight_genes=self._highlight_genes.selected(),
                baseline_mode=baseline_mode,
                batch_correct=self._batch_correct.isChecked(),
            )
            status = "Rendering library plot\u2026"

        self._start_worker(
            worker,
            on_finished=self._on_library_render_finished,
            status=status,
        )

    def _on_view_mode_changed(self) -> None:
        view_mode = self._view_mode.currentData() or "features"
        is_embeddings = view_mode in ("embeddings", "embeddings_ot")
        self._model_label.setVisible(is_embeddings)
        self._model_select.setVisible(is_embeddings)
        is_ot = view_mode in ("features_ot", "embeddings_ot")
        self._ot_nn_label.setVisible(is_ot)
        self._ot_nn.setVisible(is_ot)
        self._ranked_btn.setEnabled(is_ot)
        self._perm_btn.setEnabled(view_mode == "embeddings_ot")
        self._refresh_library_view(force=True)

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

    # ------------------------------------------------------------------
    # OT analysis hooks (PR5 — ranked matches, permutation test, pathway)
    # ------------------------------------------------------------------

    def _pick_pathway_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select gene→pathway CSV", "",
            "CSV (*.csv);;All files (*)",
        )
        if not path:
            return
        self._pathway_csv = Path(path)
        self._status.setText(
            f"Pathway map loaded: {self._pathway_csv.name}"
        )
        # Trigger re-render so the pathway colouring picks up the new file.
        if self._color_by.currentData() == "pathway":
            self._refresh_library_view(force=True)

    def _ot_sidecar_for_current_view(self) -> Optional[Path]:
        from mycoprep.core.extract.qc_plots import _ot_sidecar_path
        candidate = _ot_sidecar_path(self._library_html_path)
        return candidate if candidate.exists() else None

    def _export_ranked_matches(self) -> None:
        sidecar = self._ot_sidecar_for_current_view()
        if sidecar is None:
            QMessageBox.information(
                self, "Ranked matches",
                "No cached OT distance matrix yet. Switch to an OT view "
                "(Feature profiles (OT) or CNN embeddings (OT)) and let it "
                "render once, then try again.",
            )
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "Export ranked matches CSV",
            "ranked_matches.csv", "CSV (*.csv)",
        )
        if not out:
            return
        try:
            from mycoprep.core.extract.ot_analysis import rank_condition_matches
            df = rank_condition_matches(sidecar, top_k=10)
            df.to_csv(out, index=False)
            self._status.setText(
                f"Wrote {len(df)} ranked matches → {Path(out).name}"
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Ranked matches failed", str(exc),
            )

    def _export_permutation_test(self) -> None:
        if self._view_mode.currentData() != "embeddings_ot":
            QMessageBox.information(
                self, "Permutation test",
                "Permutation testing reads per-cell embeddings; it's "
                "only available in the CNN embeddings (OT) view.",
            )
            return
        sidecar = self._ot_sidecar_for_current_view()
        if sidecar is None:
            QMessageBox.information(
                self, "Permutation test",
                "No cached OT distance matrix yet. Render the OT view "
                "first, then try again.",
            )
            return
        # Locate the per-cell embeddings parquet.
        from mycoprep.core.extract.feature_library import FeatureLibrary
        try:
            lib = FeatureLibrary(self._library_dir)
            model_type = self._model_select.currentData() or ""
            emb_dir = lib.models_dir / "embeddings"
            if model_type:
                emb_path = emb_dir / str(model_type) / "cnn_embeddings.parquet"
            else:
                # Pick the most recent.
                cands = list(emb_dir.glob("**/cnn_embeddings.parquet"))
                emb_path = (
                    max(cands, key=lambda p: p.stat().st_mtime)
                    if cands else None
                )
        except Exception:  # noqa: BLE001
            emb_path = None
        if emb_path is None or not emb_path.exists():
            QMessageBox.information(
                self, "Permutation test",
                "No CNN embeddings parquet found in the library.",
            )
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "Export permutation test CSV",
            "permutation_test.csv", "CSV (*.csv)",
        )
        if not out:
            return
        try:
            from mycoprep.core.extract.ot_analysis import permutation_test
            df = permutation_test(
                emb_path, sidecar,
                top_k_per_query=1, n_perm=1000, sub_n=200,
            )
            df.to_csv(out, index=False)
            n_sig = int(df["significant"].sum()) if "significant" in df else 0
            self._status.setText(
                f"Permutation test: {len(df)} pairs, {n_sig} significant (FDR≤0.05) → {Path(out).name}"
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Permutation test failed", str(exc),
            )

    def _populate_gene_combo(self) -> None:
        """Refresh the gene multi-select with the current library's genes."""
        from mycoprep.core.extract.qc_plots import library_gene_list

        species = self._species.currentText().strip()
        try:
            genes = library_gene_list(
                library_dir=self._library_dir, species=species,
            )
        except Exception:  # noqa: BLE001
            genes = []
        # set_items preserves selections that are still valid.
        self._highlight_genes.set_items(genes)

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
        view_mode = self._view_mode.currentData() or "features"
        from mycoprep.core.extract.feature_library import FeatureLibrary
        try:
            lib = FeatureLibrary(self._library_dir)
            idx_path = lib.library_dir / "library.parquet"
            mtime = idx_path.stat().st_mtime if idx_path.exists() else 0.0
            emb_path = lib.models_dir / "embeddings" / "cnn_embeddings.parquet"
            emb_mtime = emb_path.stat().st_mtime if emb_path.exists() else 0.0
        except Exception:  # noqa: BLE001
            mtime = 0.0
            emb_mtime = 0.0
        self._last_library_state = (
            self._library_dir, species, view_mode, mtime, emb_mtime,
        )

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

    def _on_highlight_changed(self, names: list[str]) -> None:
        # Compare button only makes sense once the user has picked some
        # genes to compare. Otherwise prompt them via the disabled state.
        self._compare_btn.setEnabled(bool(names))
        self._refresh_library_view(force=True)

    def _open_comparison(self) -> None:
        genes = self._highlight_genes.selected()
        if not genes:
            QMessageBox.information(
                self, "Pick genes first",
                "Select one or more genes in \u201cHighlight gene(s)\u201d "
                "to compare their per-feature S-scores.",
            )
            return
        if self._is_busy():
            return

        species = self._species.currentText().strip()
        baseline_mode = self._baseline_mode.currentData() or "pooled"

        worker = _CompareWorker(
            self._comparison_html_path,
            self._library_dir, species, genes, baseline_mode,
        )
        self._start_worker(
            worker,
            on_finished=self._on_comparison_finished,
            status="Building comparison heatmap\u2026",
        )

    def _on_comparison_finished(self, written: object) -> None:
        if not written:
            QMessageBox.information(
                self, "Comparison empty",
                "No matching profiles found for the selected gene(s) "
                "in the current library / species.",
            )
            return
        if self._compare_window is None:
            self._compare_window = _ComparisonWindow(parent=self)
        self._compare_window.set_html(Path(written))
        self._compare_window.show()
        self._compare_window.raise_()
        self._compare_window.activateWindow()


class _ComparisonWindow(QMainWindow):
    """Popup window hosting the per-feature S-score heatmap."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Per-feature comparison")
        self.resize(1200, 700)
        self._view = QWebEngineView()
        self.setCentralWidget(self._view)

    def set_html(self, path: Path) -> None:
        self._view.setUrl(QUrl.fromLocalFile(str(path)))
