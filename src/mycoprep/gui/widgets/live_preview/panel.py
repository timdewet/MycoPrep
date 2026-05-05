"""Right-column live preview panel.

Phase 1 scope: sample/FOV picker that drives a :class:`MultiOverlayCanvas`
with the phase plane only. No live re-compute yet; on Segment & Classify
and Features tabs the panel loads the focused TIFF for the chosen FOV
directly off disk. On the Focus tab a placeholder message points the
user at later phases.

Sample/FOV selection persists per-tab so navigating tabs restores the
user's last context. The same `(sample_id, fov_index)` is shared with
the cache key used by future phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from ...ui import tokens
from .canvas import (
    ChannelLayer,
    MultiOverlayCanvas,
    TextLabel,
    color_for_channel_name,
)
from .controller import (
    CanvasSink,
    OptsProviders,
    PreviewController,
    ProgressSink,
)


# Color choices offered in the per-channel overlay picker.
COLOR_CHOICES = ("green", "red", "blue", "magenta", "yellow", "cyan", "white")

# Default feature columns for the per-cell label, in priority order. The
# panel tries each in turn and uses the ones that exist in the live
# features DataFrame. ``length_um`` requires midline_features=True;
# the panel falls back gracefully when it's missing.
DEFAULT_LABEL_COLUMNS = (
    "length_um",
    "width_mean_um",
    "area_um2",
)

# Cap rendered labels at this count when the user is zoomed out.
LABEL_BUDGET = 500
LABEL_FONT_PT = 7
# Friendly short names for the label rendering. Falls back to the raw
# column name if not in this map.
LABEL_NICE_NAMES = {
    "length_um": "L",
    "width_mean_um": "W",
    "area_um2": "A",
    "perimeter_um": "P",
    "eccentricity": "ecc",
    "solidity": "sol",
    "equivalent_diameter_um": "d",
    "major_axis_length_um": "maj",
    "minor_axis_length_um": "min",
}


# Tab keys this panel knows how to render for. Anything else → hidden.
RENDER_TAB_KEYS = ("focus", "segment", "features")


@dataclass
class _Selection:
    """Shared sample + FOV state across all three preview tabs.

    Switching tabs preserves the same logical sample (matched by
    :attr:`sample_id`) and FOV — so a user tuning focus on well A1
    FOV 2 can flip to the Segment & Classify tab and see segmentation
    of the same well/FOV without having to re-select.
    """

    sample_id: str = ""
    fov_index: int = 0       # local FOV inside the sample


@dataclass
class _Sample:
    """One entry in the sample combo."""

    sample_id: str           # stable identity across tabs (well id, czi name, or tiff stem)
    label: str               # e.g. "A1 · Wt + GFP · R1" or "myfile.czi · Wt"
    path: Path
    fov_count: int
    kind: str                # "tiff" or "czi"
    # When the sample is one well of a multi-well plate CZI, this is the
    # absolute CZI scene-index list for that well — the FOV spinner
    # navigates 0..len(scene_indices)-1, and the panel translates back
    # into a CZI scene index before talking to the worker.
    # ``None`` means "use the FOV spinner value as-is" (single-CZI bulk
    # samples and focused TIFFs).
    scene_indices: Optional[list[int]] = None


@dataclass
class _ChannelConfig:
    """Per-channel overlay state — visibility, opacity, color, contrast."""

    name: str
    visible: bool = True
    opacity: float = 0.85
    color: str = "white"
    is_phase: bool = False    # the phase channel is rendered at z=0, not as an overlay
    # Manual contrast bounds. ``None`` = auto (1st / 99th percentile of
    # the image). When the user edits a min/max spinbox, the bound
    # becomes a concrete float and stops auto-tracking.
    level_lo: Optional[float] = None
    level_hi: Optional[float] = None


class LivePreviewPanel(QWidget):
    """Persistent right-column preview.

    Phase 1: sample/FOV picker + display-only canvas.
    """

    # Notifies MainWindow that the user picked a different sample/FOV.
    # MainWindow can use this to update the per-tab state cache or to
    # trigger a render in later phases.
    selectionChanged = pyqtSignal(str, object, int)
    # Args: tab_key ("focus"|"segment"|"features"), sample_path (Path|None), fov_index (int).

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("livePreviewPanel")

        self._current_tab: str = "focus"
        # One shared selection across all three preview tabs — the user
        # navigates by tab to tune different stage options, but the
        # sample + FOV stay put.
        self._selection: _Selection = _Selection()
        # Sources, refreshed when the input panel's CZIs / output dir change.
        self._search_dirs: list[Path] = []
        self._czi_paths: list[Path] = []
        # Plate / bulk layout context for sample naming. Held so we can
        # rebuild the sample combo whenever the user re-edits a layout.
        # ``mode`` is the InputMode value (single_plate / bulk / single_file).
        self._input_mode: str = ""
        self._plate_layout_df: Any = None
        self._bulk_layout_df: Any = None
        # Phase channel comes from the Input panel; used to pick which
        # plane of the loaded hyperstack to display.
        self._phase_channel: int = 0
        # Toggleable: whether to draw the segmentation boundary overlay
        # at all. Classification still runs in the background so the
        # boundaries pick up classifier colours when the overlay is on.
        self._segmentation_overlay_enabled: bool = True
        # Toggleable: limit segmentation (and downstream classify /
        # features) to a draggable ROI rectangle on the canvas. ON by
        # default so iteration on large images stays fast.
        self._roi_enabled: bool = True

        # Cached arrays + per-channel overlay state. Set when focus
        # results land via :meth:`_on_image_channels`. The list of
        # ``_ChannelConfig`` records owns visibility/opacity/color for
        # each channel and is rebuilt only when the channel set
        # actually changes (so toggling a fluor channel doesn't lose
        # the user's tweaks to other channels).
        self._image_channels: Optional[Any] = None
        self._image_channel_names: list[str] = []
        self._image_phase_idx: int = 0
        self._channel_configs: list[_ChannelConfig] = []
        # Per-channel widgets, keyed by channel index. Built lazily.
        self._channel_widgets: list[dict] = []

        # Per-cell features state — DataFrame from the worker, plus the
        # user's choice of which columns to render as labels.
        self._features_df: Any = None
        self._label_columns: list[str] = list(DEFAULT_LABEL_COLUMNS)
        self._labels_visible: bool = True

        # The controller is created on first use of wire_pipeline so the
        # panel is usable in display-only mode (Phase 1 callers / tests)
        # too.
        self._controller: Optional[PreviewController] = None

        self._build_ui()
        self._refresh_for_tab(self._current_tab)

    # ----------------------------------------------------------------- build

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(tokens.S4, tokens.S4, tokens.S4, tokens.S4)
        outer.setSpacing(tokens.S3)

        title = QLabel("Live preview")
        title.setObjectName("h2")
        outer.addWidget(title)

        # ── Sample / FOV picker card ──────────────────────────────────────
        picker = QFrame()
        picker.setObjectName("card")
        pv = QVBoxLayout(picker)
        pv.setContentsMargins(tokens.S3, tokens.S3, tokens.S3, tokens.S3)
        pv.setSpacing(tokens.S2)

        pv.addWidget(QLabel("Sample:"))
        self._sample_combo = QComboBox()
        self._sample_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._sample_combo.currentIndexChanged.connect(self._on_sample_changed)
        pv.addWidget(self._sample_combo)

        fov_row = QHBoxLayout()
        fov_row.setSpacing(tokens.S2)
        fov_row.addWidget(QLabel("FOV:"))
        self._fov_spin = QSpinBox()
        self._fov_spin.setRange(0, 0)
        self._fov_spin.setPrefix("FOV ")
        self._fov_spin.valueChanged.connect(self._on_fov_changed)
        fov_row.addWidget(self._fov_spin, stretch=1)
        pv.addLayout(fov_row)

        outer.addWidget(picker)

        # ── Stage notice (used on Focus tab while live focus is unimplemented).
        self._notice = QLabel("")
        self._notice.setObjectName("muted")
        self._notice.setWordWrap(True)
        self._notice.setVisible(False)
        outer.addWidget(self._notice)

        # ── Canvas ────────────────────────────────────────────────────────
        # The canvas is the dominant surface; keep it stretched so it
        # absorbs all vertical/horizontal slack.
        self._canvas = MultiOverlayCanvas()
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        outer.addWidget(self._canvas, stretch=1)

        # ── Channel overlay controls ───────────────────────────────────────
        # The card is collapsible (click the header to toggle). Per-channel
        # contrast histograms are also hidden by default — each row shows
        # a "Contrast" disclosure button to reveal its histogram on demand.
        self._channels_card = QFrame()
        self._channels_card.setObjectName("card")
        self._channels_layout = QVBoxLayout(self._channels_card)
        self._channels_layout.setContentsMargins(
            tokens.S3, tokens.S3, tokens.S3, tokens.S3
        )
        self._channels_layout.setSpacing(tokens.S2)

        # Clickable header row: chevron + label. Toggles ``_channels_body``.
        self._channels_collapsed = False
        header_btn = QPushButton()
        header_btn.setObjectName("ghost")
        header_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        header_btn.setFlat(True)
        header_btn.setStyleSheet(
            "QPushButton { text-align: left; padding: 0; border: none; }"
        )
        self._channels_header_btn = header_btn
        header_btn.clicked.connect(self._toggle_channels_collapsed)
        self._channels_layout.addWidget(header_btn)

        # Body container holds the per-channel rows; collapsing the card
        # just hides this widget so the header stays visible.
        self._channels_body = QFrame()
        body_v = QVBoxLayout(self._channels_body)
        body_v.setContentsMargins(0, 0, 0, 0)
        body_v.setSpacing(tokens.S2)
        self._channels_body_layout = body_v
        self._channels_layout.addWidget(self._channels_body)

        self._channels_card.setVisible(False)
        outer.addWidget(self._channels_card)
        self._update_channels_header_label()

        # ── Labels card (Features tab only) ───────────────────────────────
        # Multi-select of which feature columns to display next to each
        # cell. Built lazily from the columns of the latest features_df.
        self._labels_card = QFrame()
        self._labels_card.setObjectName("card")
        labels_layout = QVBoxLayout(self._labels_card)
        labels_layout.setContentsMargins(
            tokens.S3, tokens.S3, tokens.S3, tokens.S3
        )
        labels_layout.setSpacing(tokens.S2)
        labels_header_row = QHBoxLayout()
        labels_header_row.setSpacing(tokens.S2)
        labels_header = QLabel("Per-cell labels")
        labels_header.setObjectName("h3")
        labels_header_row.addWidget(labels_header)
        labels_header_row.addStretch(1)
        self._labels_visible_cb = QCheckBox("Show")
        self._labels_visible_cb.setChecked(True)
        self._labels_visible_cb.toggled.connect(self._on_labels_visible_toggled)
        labels_header_row.addWidget(self._labels_visible_cb)
        labels_layout.addLayout(labels_header_row)

        from PyQt6.QtWidgets import QListWidget
        self._labels_list = QListWidget()
        self._labels_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._labels_list.setMaximumHeight(110)
        self._labels_list.itemSelectionChanged.connect(self._on_label_columns_changed)
        labels_layout.addWidget(self._labels_list)
        self._labels_card.setVisible(False)
        outer.addWidget(self._labels_card)

        # ── Overlay toggles + progress strip ───────────────────────────────
        toggles_row = QHBoxLayout()
        toggles_row.setSpacing(tokens.S3)
        self._segmentation_overlay_cb = QCheckBox("Show segmentation")
        self._segmentation_overlay_cb.setChecked(True)
        self._segmentation_overlay_cb.setToolTip(
            "Show the segmentation boundary overlay on top of the image. "
            "When on, boundaries are coloured by the classifier's "
            "decision (green = kept, grey = edge, orange = debris, "
            "red = rejected). Turn off to see the bare phase + "
            "fluorescence channels with no overlay."
        )
        self._segmentation_overlay_cb.toggled.connect(
            self._on_segmentation_overlay_toggled
        )
        toggles_row.addWidget(self._segmentation_overlay_cb)
        # Crop-to-ROI toggle: on by default to keep iteration fast on
        # large images. Drag the yellow rectangle on the canvas to pick
        # a representative region; segmentation runs only inside it.
        self._roi_enabled_cb = QCheckBox("Crop segmentation to ROI")
        self._roi_enabled_cb.setChecked(True)
        self._roi_enabled_cb.setToolTip(
            "When on, segmentation only runs inside the yellow ROI "
            "rectangle drawn on the image. Useful for tuning options "
            "on a representative region — full plates can be slow on "
            "every parameter change. Drag the rectangle to move; drag "
            "its handles to resize."
        )
        self._roi_enabled_cb.toggled.connect(self._on_roi_enabled_toggled)
        toggles_row.addWidget(self._roi_enabled_cb)
        toggles_row.addStretch(1)
        outer.addLayout(toggles_row)

        progress_row = QHBoxLayout()
        progress_row.setSpacing(tokens.S2)
        self._progress_label = QLabel("")
        self._progress_label.setObjectName("muted")
        progress_row.addWidget(self._progress_label, stretch=1)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)            # busy indicator
        self._progress_bar.setMaximumHeight(8)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setVisible(False)
        progress_row.addWidget(self._progress_bar, stretch=1)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        progress_row.addWidget(self._cancel_btn)
        outer.addLayout(progress_row)

    # ----------------------------------------------------------------- API

    def wire_pipeline(
        self,
        focus_opts: Callable[[], Any],
        segment_opts: Callable[[], Any],
        classify_opts: Callable[[], Any],
        features_opts: Callable[[], Any],
        phase_channel: Callable[[], int],
        channel_labels: Callable[[], Optional[list[str]]] = lambda: None,
    ) -> None:
        """Install the live opts providers + spin up the render controller.

        MainWindow calls this once after the panels are constructed so
        the controller can read live opts from them whenever the user
        moves a slider. The classify provider can return ``None`` for
        "rules-only / no model"; the panel handles that gracefully.
        """
        opts = OptsProviders(
            focus_opts=focus_opts,
            segment_opts=segment_opts,
            classify_opts=classify_opts,
            features_opts=features_opts,
            phase_channel=phase_channel,
            channel_labels=channel_labels,
            roi=self._current_roi,
        )
        canvas = CanvasSink(
            set_phase=self._canvas.set_phase,
            set_image_channels=self._on_image_channels,
            # Route mask updates through the panel so we can suppress
            # the overlay when the user has toggled segmentation off.
            set_mask=self._on_canvas_mask,
            set_features_df=self._on_features_df,
            set_labels=self._canvas.set_labels,
            clear=self._on_canvas_clear,
        )
        progress = ProgressSink(
            started=self._on_progress_started,
            finished=self._on_progress_finished,
            failed=self._on_progress_failed,
        )
        self._controller = PreviewController(opts, canvas, progress, parent=self)
        # Re-render when the user drags / resizes the ROI rectangle.
        self._canvas.add_roi_change_listener(self._on_roi_geometry_changed)
        # Apply the default ROI-on state so the rectangle is visible
        # the first time the canvas has image data.
        self._apply_roi_visibility()

    def trigger_render(self) -> None:
        """Force a re-render against the current selection.

        Hooked to each option panel's ``optionsChanged`` signal in
        MainWindow so live edits propagate without each panel needing a
        controller reference.
        """
        if self._controller is not None:
            self._controller.request_render(reason="options")

    def set_current_tab(self, tab_key: str) -> None:
        """Called by MainWindow when the user navigates between tabs.

        Ignored for non-render tabs (input/plate/run); MainWindow shows/
        hides the entire panel for those.
        """
        if tab_key not in RENDER_TAB_KEYS:
            return
        self._current_tab = tab_key
        # Labels card is only relevant on the Features tab — hide it
        # elsewhere so the panel doesn't show empty controls.
        self._labels_card.setVisible(tab_key == "features" and self._features_df is not None)
        self._refresh_for_tab(tab_key)

    def set_search_dirs(self, dirs: list[Path]) -> None:
        """Where to look for previewable TIFFs (from the project's output dir)."""
        self._search_dirs = [Path(d) for d in dirs if Path(d).exists()]
        if self._current_tab in ("segment", "features"):
            self._refresh_for_tab(self._current_tab)

    def set_czi_paths(self, paths: list[Path]) -> None:
        """The list of CZI files known to the InputPanel.

        Used as a fallback when no plate/bulk layout context has been
        provided yet — :meth:`set_layouts` is the richer path.
        """
        self._czi_paths = [Path(p) for p in paths if Path(p).exists()]
        self._refresh_for_tab(self._current_tab)

    def set_layouts(
        self,
        mode: str,
        plate_layout_df: Any,
        bulk_layout_df: Any,
    ) -> None:
        """Hand the panel the plate / bulk layout DataFrames.

        ``mode`` is the InputMode value string. The panel uses these to
        label samples by condition / reporter / mutant / replica rather
        than by raw filename, and (in single-plate mode) to map well
        selections to the right CZI scene indices.
        """
        self._input_mode = str(mode)
        self._plate_layout_df = plate_layout_df
        self._bulk_layout_df = bulk_layout_df
        self._refresh_for_tab(self._current_tab)

    def set_phase_channel(self, ch: int) -> None:
        """Phase channel index from the Input panel — used to pick the
        plane of a loaded multi-channel TIFF to display."""
        self._phase_channel = int(ch) if isinstance(ch, int) else 0
        # Phase channel is part of the focus key; tell the controller to
        # re-render so the cache miss handles the rest.
        if self._controller is not None:
            self._controller.invalidate_cache(("focus", "segment", "classify", "features"))
            self._controller.request_render(reason="phase_channel")
        elif self._sample_combo.currentIndex() >= 0:
            self._reload_current_fov()

    # ----------------------------------------------------------------- internals

    def _refresh_for_tab(self, tab_key: str) -> None:
        """Repopulate the sample combo for the given tab.

        The shared :attr:`_selection` is reapplied so switching tabs
        keeps the same logical sample selected when it exists in the
        new tab's source list (e.g., the same plate well shows up on
        Focus, Segment & Classify, and Features).
        """
        self._sample_combo.blockSignals(True)
        self._sample_combo.clear()

        samples = self._samples_for_tab(tab_key)
        for s in samples:
            self._sample_combo.addItem(s.label, userData=s)

        # Restore the shared selection by sample_id when possible.
        target_idx = -1
        if self._selection.sample_id:
            for i in range(self._sample_combo.count()):
                s = self._sample_combo.itemData(i)
                if isinstance(s, _Sample) and s.sample_id == self._selection.sample_id:
                    target_idx = i
                    break
        if target_idx < 0 and self._sample_combo.count() > 0:
            target_idx = 0
        if target_idx >= 0:
            self._sample_combo.setCurrentIndex(target_idx)
        self._sample_combo.blockSignals(False)

        # Notice/placeholder per tab.
        if self._sample_combo.count() == 0:
            self._notice.setText(
                "No samples available yet. Add CZIs on the Input tab "
                "to start previewing — the panel will focus and segment "
                "them on the fly per FOV."
            )
            self._notice.setVisible(True)
            self._canvas.clear()
        else:
            self._notice.setVisible(False)
            # Trigger initial draw for the restored selection.
            self._on_sample_changed(self._sample_combo.currentIndex())

    def _samples_for_tab(self, tab_key: str) -> list[_Sample]:
        # Tab-specific source priority:
        # - focus: raw CZIs (the user is tuning focus options).
        # - segment / features: focused TIFFs first (cheap to load), then
        #   raw CZIs as a fallback so the preview works even before the
        #   real Focus stage has produced any disk output.
        out: list[_Sample] = []
        if tab_key in ("segment", "features"):
            out.extend(self._tiff_samples())
        out.extend(self._czi_samples(suffix=" (raw CZI — focus on the fly)"
                                     if tab_key in ("segment", "features") else ""))
        return out

    def _czi_samples(self, *, suffix: str = "") -> list[_Sample]:
        """Build CZI samples.

        - Single-plate mode: one entry per labelled well (so each FOV is
          tied to its condition / reporter / mutant). For data where each
          well has multiple positions, the FOV spinner navigates those
          positions; for data with a single position per well, the user
          navigates between wells via the sample dropdown.
        - In addition, **always append a per-CZI ``all FOVs`` entry**.
          That gives the user flat scene navigation across the entire
          CZI as a fallback — useful when wells haven't been labelled
          yet, or when they just want to scroll through every position
          without filtering by well.
        - Bulk / single-file modes already work per-CZI, so no fallback
          is added there.
        """
        out: list[_Sample] = []
        if self._input_mode == "single_plate" and self._plate_layout_df is not None:
            out.extend(self._plate_well_samples(suffix=suffix))
            out.extend(self._raw_czi_samples(
                exclude_paths={s.path for s in out},
                suffix=suffix,
                label_tail="all FOVs",
            ))
            return out
        if self._bulk_layout_df is not None:
            return self._bulk_czi_samples(suffix=suffix)
        # Last-resort fallback (no layout wired yet): raw filenames.
        return self._raw_czi_samples(suffix=suffix)

    def _raw_czi_samples(
        self,
        *,
        exclude_paths: Optional[set[Path]] = None,
        suffix: str = "",
        label_tail: str = "",
    ) -> list[_Sample]:
        """One ``_Sample`` per CZI, navigating every scene flat.

        ``scene_indices`` is populated with the CZI's *actual* scene
        IDs from pylibCZIrw — those are not always ``0..N-1``, so we
        translate the FOV spinner's local position to the real index
        before reading.
        """
        out: list[_Sample] = []
        for p in self._czi_paths:
            indices = _list_czi_scene_indices(p)
            tail = f" · {label_tail}" if label_tail else ""
            if indices:
                fov_count = len(indices)
                scene_indices: Optional[list[int]] = indices
            else:
                # Single-position CZI: no scenes; the worker reads it
                # without a scene argument. One FOV, no remap.
                fov_count = 1
                scene_indices = None
            out.append(
                _Sample(
                    sample_id=f"czi-flat:{p.name}",
                    label=f"{p.name}{tail}{suffix}",
                    path=p,
                    fov_count=fov_count,
                    kind="czi",
                    scene_indices=scene_indices,
                )
            )
        return out

    def _plate_well_samples(self, *, suffix: str = "") -> list[_Sample]:
        df = self._plate_layout_df
        if df is None:
            return []
        out: list[_Sample] = []
        czi_by_name = {p.name: p for p in self._czi_paths}
        for _, row in df.iterrows():
            cond = str(row.get("condition", "") or "").strip()
            if not cond:
                # Inactive well — skip; only labelled wells are samples.
                continue
            scenes = _parse_scene_indices(row.get("scene_indices"))
            source_czi = str(row.get("source_czi", "") or "").strip()
            czi_path = czi_by_name.get(source_czi)
            if czi_path is None:
                # The well's source CZI isn't in the current input list
                # (user may have removed it) — drop the sample rather
                # than silently mapping to the wrong file.
                continue
            # Defensive fallback: if the layout has no scene list for
            # this well (legacy CSV import, missing scene metadata, etc.)
            # fall back to the CZI's full scene count so the FOV spinner
            # can still navigate something.
            if not scenes:
                fov_count = _count_czi_scenes(czi_path)
                scene_indices = None
            else:
                fov_count = len(scenes)
                scene_indices = scenes
            well = str(row.get("well", "") or "").strip() or "?"
            label = _build_label(
                prefix=well,
                condition=cond,
                reporter=str(row.get("reporter", "") or "").strip(),
                mutant=str(row.get("mutant_or_drug", "") or "").strip(),
                replica=str(row.get("replica", "") or "").strip(),
            ) + suffix
            out.append(
                _Sample(
                    sample_id=f"well:{well}",
                    label=label,
                    path=czi_path,
                    fov_count=fov_count,
                    kind="czi",
                    scene_indices=scene_indices,
                )
            )
        return out

    def _bulk_czi_samples(self, *, suffix: str = "") -> list[_Sample]:
        df = self._bulk_layout_df
        if df is None or df.empty:
            return []
        out: list[_Sample] = []
        for _, row in df.iterrows():
            czi_str = str(row.get("czi_path", "") or "").strip()
            if not czi_str:
                continue
            p = Path(czi_str)
            if not p.exists():
                continue
            cond = str(row.get("condition", "") or "").strip()
            label_prefix = p.name if not cond else _build_label(
                prefix=p.name,
                condition=cond,
                reporter=str(row.get("reporter", "") or "").strip(),
                mutant=str(row.get("mutant_or_drug", "") or "").strip(),
                replica=str(row.get("replica", "") or "").strip(),
            )
            indices = _list_czi_scene_indices(p)
            if indices:
                fov_count = len(indices)
                scene_indices: Optional[list[int]] = indices
            else:
                fov_count = 1
                scene_indices = None
            out.append(
                _Sample(
                    sample_id=f"czi:{p.name}",
                    label=label_prefix + suffix,
                    path=p,
                    fov_count=fov_count,
                    kind="czi",
                    scene_indices=scene_indices,
                )
            )
        return out

    def _tiff_samples(self) -> list[_Sample]:
        """Focused-output TIFFs from the project's search dirs.

        ``sample_id`` is derived from the TIFF's stem (with the
        ``_focused`` suffix stripped) so a single logical sample lines
        up with the equivalent CZI well across tabs.
        """
        out: list[_Sample] = []
        seen: set[Path] = set()
        for d in self._search_dirs:
            for p in sorted(d.glob("*.tif")) + sorted(d.glob("*.tiff")):
                if p in seen or "_zmaps" in p.stem:
                    continue
                seen.add(p)
                stem = p.stem
                if stem.endswith("_focused"):
                    stem = stem[: -len("_focused")]
                out.append(
                    _Sample(
                        sample_id=f"tiff:{stem}",
                        label=_friendly_tiff_label(d.name, p),
                        path=p,
                        fov_count=_count_tiff_fovs(p),
                        kind="tiff",
                    )
                )
        return out

    def _on_sample_changed(self, idx: int) -> None:
        if idx < 0:
            self._fov_spin.setRange(0, 0)
            self._canvas.clear()
            self._selection = _Selection()
            self._notify_selection(None, 0, None)
            return
        sample: _Sample = self._sample_combo.itemData(idx)
        if sample is None:
            return
        # Carry the FOV across samples when the user lands on the same
        # logical sample as before (e.g., tab switch); otherwise start
        # at 0.
        same_sample = (sample.sample_id == self._selection.sample_id)
        fov = self._selection.fov_index if same_sample else 0
        fov = min(fov, max(sample.fov_count - 1, 0))

        self._fov_spin.blockSignals(True)
        self._fov_spin.setRange(0, max(sample.fov_count - 1, 0))
        self._fov_spin.setValue(fov)
        self._fov_spin.blockSignals(False)

        self._selection = _Selection(sample_id=sample.sample_id, fov_index=fov)
        self._notify_selection(sample.path, fov, sample.scene_indices)
        self.selectionChanged.emit(self._current_tab, sample.path, fov)

    def _on_fov_changed(self, fov: int) -> None:
        self._selection.fov_index = int(fov)
        sample = self._current_sample()
        if sample is None:
            return
        self._notify_selection(sample.path, fov, sample.scene_indices)
        self.selectionChanged.emit(self._current_tab, sample.path, fov)

    def _current_sample(self) -> Optional[_Sample]:
        idx = self._sample_combo.currentIndex()
        if idx < 0:
            return None
        s = self._sample_combo.itemData(idx)
        return s if isinstance(s, _Sample) else None

    def _notify_selection(
        self,
        sample_path: Optional[Path],
        fov_local: int,
        scene_indices: Optional[list[int]],
    ) -> None:
        """Hand the selection to the controller, translating the
        spinner's local FOV index to an absolute CZI scene index when
        the sample is a multi-scene plate well."""
        if self._controller is None:
            self._reload_current_fov()
            return
        if scene_indices and 0 <= fov_local < len(scene_indices):
            absolute_fov = int(scene_indices[fov_local])
        else:
            absolute_fov = int(fov_local)
        self._controller.set_selection(self._current_tab, sample_path, absolute_fov)

    def _reload_current_fov(self) -> None:
        """Phase 1: just load the phase plane off disk and display it.

        Later phases replace this with a controller-driven render that
        runs the cached pipeline chain and updates the canvas.
        """
        if self._current_tab == "focus":
            # Live focus preview lands in Phase 3; no on-disk fallback for
            # raw CZI z-stacks here.
            return

        idx = self._sample_combo.currentIndex()
        if idx < 0:
            return
        sample: _Sample = self._sample_combo.itemData(idx)
        if sample is None or sample.kind != "tiff":
            return
        fov = self._fov_spin.value()
        try:
            from mycoprep.core.label_cells import load_hyperstack
            data, _meta = load_hyperstack(sample.path)
            if fov >= data.shape[0]:
                self._canvas.clear()
                return
            channels = data[fov]
            phase_idx = min(self._phase_channel, channels.shape[0] - 1)
            self._canvas.set_phase(channels[phase_idx])
            # Phase 1: mask/labels/fluor channels stay clear; future phases
            # populate them from the cache.
            self._canvas.set_mask(None)
            self._canvas.set_channels(None)
            self._canvas.set_labels(None)
        except Exception as e:  # noqa: BLE001
            self._notice.setText(f"Failed to load FOV {fov} of {sample.path.name}: {e}")
            self._notice.setVisible(True)
            self._canvas.clear()

    # ------------------------------------------------------- progress + toggles

    # Friendlier names for each stage shown in the progress strip.
    _STAGE_LABEL = {
        "focus":    "focus",
        "segment":  "segmentation",
        "classify": "classification",
        "features": "feature extraction",
    }

    def _on_progress_started(self, stage: str) -> None:
        nice = self._STAGE_LABEL.get(stage, stage)
        self._progress_label.setText(f"Running {nice}…")
        self._progress_bar.setVisible(True)
        self._cancel_btn.setVisible(True)

    def _on_progress_finished(self) -> None:
        self._progress_label.setText("")
        self._progress_bar.setVisible(False)
        self._cancel_btn.setVisible(False)

    def _on_progress_failed(self, stage: str, msg: str) -> None:
        nice = self._STAGE_LABEL.get(stage, stage)
        self._progress_label.setText(f"{nice} failed: {msg}")
        self._progress_label.setWordWrap(True)
        self._progress_bar.setVisible(False)
        self._cancel_btn.setVisible(False)

    def _on_cancel_clicked(self) -> None:
        if self._controller is not None:
            self._controller._cancel_worker()
            self._on_progress_finished()

    def _on_roi_enabled_toggled(self, on: bool) -> None:
        self._roi_enabled = bool(on)
        self._apply_roi_visibility()
        if self._controller is not None:
            # Toggling the ROI changes what region the worker
            # processes — kick off a re-render with the new bounds.
            self._controller.invalidate_cache(("segment", "classify", "features"))
            self._controller.request_render(reason="roi_toggle")

    def _on_roi_geometry_changed(self) -> None:
        if self._controller is None:
            return
        self._controller.invalidate_cache(("segment", "classify", "features"))
        self._controller.request_render(reason="roi_geometry")

    def _apply_roi_visibility(self) -> None:
        """Show / hide the ROI rectangle on the canvas to match the
        ``Crop segmentation to ROI`` toggle state."""
        self._canvas.set_roi_visible(self._roi_enabled)

    def _current_roi(self) -> Optional[tuple[int, int, int, int]]:
        """Provider passed into the controller's :class:`OptsProviders` —
        returns the current ROI bounds when crop-to-ROI is on, else
        ``None`` so the worker processes the whole image."""
        if not self._roi_enabled:
            return None
        return self._canvas.get_roi_bounds()

    def _on_segmentation_overlay_toggled(self, on: bool) -> None:
        """Show or hide the boundary overlay without changing what the
        worker computes — classification still runs in the background
        so its colours remain available when the overlay is shown again.
        """
        self._segmentation_overlay_enabled = bool(on)
        if self._controller is not None:
            # Repaint from cache; the panel's mask sink will respect the
            # new toggle when it forwards the cached mask to the canvas.
            self._controller.repaint_cached()

    def _on_canvas_mask(self, mask: Any, decisions: Any) -> None:
        """Mask sink — gates display of the boundary overlay on the
        ``Show segmentation`` toggle."""
        if not self._segmentation_overlay_enabled:
            self._canvas.set_mask(None)
            return
        self._canvas.set_mask(mask, decisions)

    # ----------------------------------------------------- channel overlays

    def _on_image_channels(
        self,
        image_channels: Any,
        names: list[str],
        phase_idx: int,
    ) -> None:
        """Receive raw channel arrays from the controller and refresh
        the per-channel overlay UI + canvas layers."""
        self._image_channels = image_channels
        self._image_channel_names = list(names) if names else [
            f"C{i}" for i in range(image_channels.shape[0])
        ]
        self._image_phase_idx = int(phase_idx)
        self._sync_channel_configs()
        self._rebuild_channel_widgets()
        self._channels_card.setVisible(True)
        self._refresh_canvas_channels()

    def _on_canvas_clear(self) -> None:
        self._image_channels = None
        self._image_channel_names = []
        self._channels_card.setVisible(False)
        self._features_df = None
        self._labels_card.setVisible(False)
        self._canvas.clear()

    def _sync_channel_configs(self) -> None:
        """Keep per-channel state in step with the current channel set.

        Existing configs are preserved by name; new channels get their
        defaults via the name-to-color heuristic.
        """
        new: list[_ChannelConfig] = []
        prev_by_name = {c.name: c for c in self._channel_configs}
        for i, name in enumerate(self._image_channel_names):
            is_phase = (i == self._image_phase_idx)
            existing = prev_by_name.get(name)
            if existing is not None:
                existing.is_phase = is_phase
                new.append(existing)
            else:
                new.append(_ChannelConfig(
                    name=name,
                    is_phase=is_phase,
                    visible=not is_phase,           # phase is on the base layer
                    color=color_for_channel_name(name),
                    opacity=0.85,
                ))
        self._channel_configs = new

    def _rebuild_channel_widgets(self) -> None:
        """Lay out controls for each non-phase channel.

        Each channel row is one tight line — visibility toggle (with
        channel name), colour combo, opacity slider, and a "Contrast"
        disclosure button. The histogram is hidden by default and
        only takes up space when the user clicks the disclosure for
        that channel. The whole card is also collapsible from its
        header.
        """
        import pyqtgraph as pg
        from PyQt6.QtWidgets import QGridLayout, QSlider

        # Drop existing per-channel widgets from the body container.
        while self._channels_body_layout.count() > 0:
            item = self._channels_body_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._channel_widgets = []

        non_phase_count = sum(1 for c in self._channel_configs if not c.is_phase)
        self._canvas.ensure_channel_item_count(non_phase_count)

        for i, cfg in enumerate(self._channel_configs):
            if cfg.is_phase:
                continue
            block = QFrame()
            grid = QGridLayout(block)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(tokens.S2)
            grid.setVerticalSpacing(2)

            cb = QCheckBox(cfg.name)
            cb.setChecked(cfg.visible)
            grid.addWidget(cb, 0, 0)

            color_combo = QComboBox()
            color_combo.addItems(COLOR_CHOICES)
            color_combo.setCurrentIndex(
                COLOR_CHOICES.index(cfg.color) if cfg.color in COLOR_CHOICES else 0
            )
            color_combo.setMaximumWidth(72)
            grid.addWidget(color_combo, 0, 1)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(cfg.opacity * 100))
            slider.setToolTip("Overlay opacity")
            grid.addWidget(slider, 0, 2)

            # Contrast disclosure: hidden histogram by default, click
            # the button to show it for fine-tuning min/max.
            tune_btn = QPushButton("⌃")
            tune_btn.setCheckable(True)
            tune_btn.setMaximumWidth(28)
            tune_btn.setToolTip("Show / hide the contrast histogram")
            tune_btn.setStyleSheet(
                "QPushButton { padding: 0 4px; }"
            )
            grid.addWidget(tune_btn, 0, 3)

            hist = pg.HistogramLUTWidget(orientation="horizontal")
            hist.setMaximumHeight(60)
            hist.setVisible(False)
            try:
                hist.item.gradient.setVisible(False)
            except Exception:  # noqa: BLE001
                pass
            grid.addWidget(hist, 1, 0, 1, 4)

            grid.setColumnStretch(2, 1)  # opacity slider takes the slack

            cb.toggled.connect(lambda on, ci=i: self._set_channel_visible(ci, on))
            slider.valueChanged.connect(
                lambda v, ci=i: self._set_channel_opacity(ci, v / 100.0)
            )
            color_combo.currentTextChanged.connect(
                lambda txt, ci=i: self._set_channel_color(ci, txt)
            )
            hist.item.region.sigRegionChangeFinished.connect(
                lambda _r=None, ci=i, h=hist: self._on_histogram_changed(ci, h)
            )
            tune_btn.toggled.connect(
                lambda on, h=hist, btn=tune_btn: (
                    h.setVisible(on),
                    btn.setText("⌄" if on else "⌃"),
                )
            )

            self._channel_widgets.append({
                "cb": cb,
                "color": color_combo,
                "slider": slider,
                "hist": hist,
                "tune_btn": tune_btn,
                "row": block,
            })
            self._channels_body_layout.addWidget(block)

        self._sync_histograms_to_canvas()
        self._update_channels_header_label()

    def _set_channel_visible(self, idx: int, visible: bool) -> None:
        if 0 <= idx < len(self._channel_configs):
            self._channel_configs[idx].visible = bool(visible)
            self._refresh_canvas_channels()

    def _set_channel_opacity(self, idx: int, opacity: float) -> None:
        if 0 <= idx < len(self._channel_configs):
            self._channel_configs[idx].opacity = float(np.clip(opacity, 0.0, 1.0))
            self._refresh_canvas_channels()

    def _set_channel_color(self, idx: int, color: str) -> None:
        if 0 <= idx < len(self._channel_configs):
            self._channel_configs[idx].color = color
            self._refresh_canvas_channels()

    def _set_channel_level(
        self,
        idx: int,
        *,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
    ) -> None:
        if not (0 <= idx < len(self._channel_configs)):
            return
        cfg = self._channel_configs[idx]
        if lo is not None:
            cfg.level_lo = float(lo)
        if hi is not None:
            cfg.level_hi = float(hi)
        self._refresh_canvas_channels()

    def _sync_histograms_to_canvas(self) -> None:
        """Bind each channel's ``HistogramLUTWidget`` to its canvas
        ImageItem and seed the region with the user's stored bounds
        (or auto-detected percentiles when no bounds are set yet).

        ``ui_idx`` (the row's position in the channel card) matches
        the canvas ImageItem index because both iterate non-phase
        channels in order. ``cfg_idx`` is the index into the full
        config list including phase, used for fetching image data.
        """
        if self._image_channels is None:
            return
        for ui_idx, widgets in enumerate(self._channel_widgets):
            cfg_idx = self._nth_non_phase_config(ui_idx)
            if cfg_idx is None or cfg_idx >= self._image_channels.shape[0]:
                continue
            cfg = self._channel_configs[cfg_idx]
            plane = self._image_channels[cfg_idx]
            # Default bounds = 1st / 99th percentile when the user
            # hasn't dragged the histogram yet.
            if cfg.level_lo is None:
                cfg.level_lo = float(np.percentile(plane, 1))
            if cfg.level_hi is None:
                cfg.level_hi = float(np.percentile(plane, 99))

            item = self._canvas.channel_image_item(ui_idx)
            hist = widgets["hist"]
            if item is not None:
                # Bind the histogram to the underlying ImageItem so the
                # histogram bars reflect that channel's pixel
                # distribution. ``setImageItem`` resets the region from
                # the item's current levels — push our stored levels in
                # afterwards so the user's tweaks survive across FOVs.
                hist.item.setImageItem(item)
            region = hist.item.region
            region.blockSignals(True)
            region.setRegion((cfg.level_lo, cfg.level_hi))
            region.blockSignals(False)

    def _toggle_channels_collapsed(self) -> None:
        self._channels_collapsed = not self._channels_collapsed
        self._channels_body.setVisible(not self._channels_collapsed)
        self._update_channels_header_label()

    def _update_channels_header_label(self) -> None:
        """Render the chevron + channel-count summary on the card header."""
        chevron = "▶" if self._channels_collapsed else "▼"
        n_fluor = sum(
            1 for c in self._channel_configs if not c.is_phase
        )
        suffix = f" ({n_fluor})" if n_fluor else ""
        self._channels_header_btn.setText(f"{chevron}  Channels{suffix}")

    def _on_histogram_changed(self, idx: int, hist) -> None:
        """User dragged the histogram region — sync into config + canvas."""
        if not (0 <= idx < len(self._channel_configs)):
            return
        lo, hi = hist.item.region.getRegion()
        cfg = self._channel_configs[idx]
        cfg.level_lo = float(lo)
        cfg.level_hi = float(hi)
        self._refresh_canvas_channels()

    def _nth_non_phase_config(self, ui_idx: int) -> Optional[int]:
        """Map a UI-row index (over non-phase channels only) back to an
        index into ``self._channel_configs``."""
        seen = -1
        for i, cfg in enumerate(self._channel_configs):
            if cfg.is_phase:
                continue
            seen += 1
            if seen == ui_idx:
                return i
        return None

    def _refresh_canvas_channels(self) -> None:
        """Build the canvas's overlay layer list from the cached arrays
        and the user's per-channel config.

        Layers are aligned 1:1 with non-phase ``_channel_configs`` (in
        config order). Invisible channels are sent through with
        ``image=None`` so the canvas's ImageItem pool stays a stable
        size across renders — important because the panel's histograms
        are bound to those items by index.
        """
        if self._image_channels is None:
            self._canvas.set_channels(None)
            return
        layers: list[ChannelLayer] = []
        for cfg_idx, cfg in enumerate(self._channel_configs):
            if cfg.is_phase:
                continue
            visible = cfg.visible and cfg_idx < self._image_channels.shape[0]
            layers.append(ChannelLayer(
                image=self._image_channels[cfg_idx] if visible else None,
                color=cfg.color,
                opacity=cfg.opacity,
                level_lo=cfg.level_lo,
                level_hi=cfg.level_hi,
            ))
        self._canvas.set_channels(layers)

    # ----------------------------------------------------- per-cell labels

    def _on_features_df(self, df: Any) -> None:
        """Receive per-cell features from the worker."""
        self._features_df = df
        self._rebuild_labels_list()
        # Show the card on the Features tab as soon as we have a df.
        self._labels_card.setVisible(
            self._current_tab == "features" and df is not None
        )
        self._refresh_canvas_labels()

    def _rebuild_labels_list(self) -> None:
        """Populate the multi-select with the numeric columns of the df."""
        from PyQt6.QtCore import Qt as _Qt
        from PyQt6.QtWidgets import QListWidgetItem
        self._labels_list.blockSignals(True)
        self._labels_list.clear()
        df = self._features_df
        if df is None or len(df) == 0:
            self._labels_list.blockSignals(False)
            return
        # Sort by a small bias: "headline" measurements first, then the rest
        # alphabetical.
        cols = list(df.columns)
        priority = list(DEFAULT_LABEL_COLUMNS) + [
            "perimeter_um", "eccentricity", "solidity", "equivalent_diameter_um",
            "major_axis_length_um", "minor_axis_length_um",
        ]
        ordered: list[str] = []
        seen: set[str] = set()
        for c in priority:
            if c in cols and c not in seen:
                ordered.append(c)
                seen.add(c)
        for c in cols:
            if c in seen:
                continue
            # Skip non-numeric / bookkeeping columns the user wouldn't
            # plot per cell.
            if c in ("cell_id", "cell_uid", "fov_index", "well", "run_id",
                     "source_czi", "tiff_file",
                     "centroid_x", "centroid_y",
                     "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"):
                continue
            try:
                if not df[c].dtype.kind in ("i", "u", "f"):
                    continue
            except Exception:  # noqa: BLE001
                continue
            ordered.append(c)
            seen.add(c)
        for c in ordered:
            item = QListWidgetItem(c)
            item.setFlags(item.flags() | _Qt.ItemFlag.ItemIsSelectable)
            self._labels_list.addItem(item)
            if c in self._label_columns:
                item.setSelected(True)
        # If nothing matched the saved selection, fall back to defaults
        # that are present.
        if not any(item.isSelected() for item in
                   [self._labels_list.item(i) for i in range(self._labels_list.count())]):
            for i in range(self._labels_list.count()):
                item = self._labels_list.item(i)
                if item.text() in DEFAULT_LABEL_COLUMNS:
                    item.setSelected(True)
        self._labels_list.blockSignals(False)
        # Sync internal state from the actual selection.
        self._label_columns = [
            self._labels_list.item(i).text()
            for i in range(self._labels_list.count())
            if self._labels_list.item(i).isSelected()
        ]

    def _on_label_columns_changed(self) -> None:
        self._label_columns = [
            self._labels_list.item(i).text()
            for i in range(self._labels_list.count())
            if self._labels_list.item(i).isSelected()
        ]
        self._refresh_canvas_labels()

    def _on_labels_visible_toggled(self, on: bool) -> None:
        self._labels_visible = bool(on)
        self._refresh_canvas_labels()

    def _refresh_canvas_labels(self) -> None:
        """Place a TextItem at each cell centroid with the selected
        columns rendered as multi-line text."""
        if self._features_df is None or not self._labels_visible or not self._label_columns:
            self._canvas.set_labels(None)
            return
        df = self._features_df
        if "centroid_x" not in df.columns or "centroid_y" not in df.columns or len(df) == 0:
            self._canvas.set_labels(None)
            return
        # Cap rendered count so very dense FOVs don't kill the framerate.
        # Phase 5b can lift this cap once the canvas culls offscreen items.
        rows = df.head(LABEL_BUDGET) if len(df) > LABEL_BUDGET else df
        labels: list[TextLabel] = []
        for row in rows.itertuples(index=False):
            lines: list[str] = []
            for col in self._label_columns:
                if not hasattr(row, col):
                    continue
                v = getattr(row, col)
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                short = LABEL_NICE_NAMES.get(col, col)
                lines.append(f"{short}: {fv:.2f}")
            if not lines:
                continue
            labels.append(TextLabel(
                text="\n".join(lines),
                x=float(getattr(row, "centroid_x")),
                y=float(getattr(row, "centroid_y")),
            ))
        self._canvas.set_labels(labels)

    # ----------------------------------------------------------------- persistence

    def state(self) -> dict:
        """Snapshot for QSettings round-trip."""
        return {
            "selection": {
                "sample_id": self._selection.sample_id,
                "fov_index": int(self._selection.fov_index),
            }
        }

    def restore_state(self, s: dict) -> None:
        if not isinstance(s, dict):
            return
        sel = s.get("selection")
        if isinstance(sel, dict):
            self._selection = _Selection(
                sample_id=str(sel.get("sample_id", "") or ""),
                fov_index=int(sel.get("fov_index", 0) or 0),
            )


# ---------------------------------------------------------------------------
# FOV-count probes — cheap metadata reads, no full hyperstack load.
# ---------------------------------------------------------------------------

def _parse_scene_indices(raw) -> list[int]:
    """Coerce a layout row's ``scene_indices`` value to a list of ints.

    Accepts:
      - ``None`` / empty → ``[]``
      - a list / tuple / numpy array of int-likes → those ints
      - a string like ``"0;1;2"``, ``"0,1,2"``, or ``"[0, 1, 2]"`` —
        split on ``[;, \\s]`` and parsed. ``list("0;1;2")`` would yield
        per-character iteration which previously crashed with
        ``ValueError: invalid literal for int() with base 10: ';'``.
    """
    import re

    if raw is None:
        return []
    if isinstance(raw, str):
        cleaned = raw.strip().strip("[](){}")
        if not cleaned:
            return []
        tokens = re.split(r"[;,\s]+", cleaned)
        out: list[int] = []
        for t in tokens:
            t = t.strip()
            if not t:
                continue
            try:
                out.append(int(t))
            except ValueError:
                continue
        return out
    try:
        return [int(s) for s in raw]
    except (TypeError, ValueError):
        return []


def _count_tiff_fovs(path: Path) -> int:
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tif:
            meta = tif.imagej_metadata or {}
            return int(meta.get("slices", 1))
    except Exception:  # noqa: BLE001
        return 1


def _build_label(
    *,
    prefix: str,
    condition: str,
    reporter: str,
    mutant: str,
    replica: str,
) -> str:
    """Render a human-readable sample label.

    Format: ``"{prefix} · {condition} + {reporter} · {mutant} · R{replica}"``
    with empty parts elided. ``prefix`` is typically the well id (plate
    mode) or the CZI filename (bulk mode).
    """
    body_bits: list[str] = []
    cond_rep = " + ".join([s for s in (condition, reporter) if s])
    if cond_rep:
        body_bits.append(cond_rep)
    if mutant:
        body_bits.append(mutant)
    if replica:
        body_bits.append(f"R{replica}")
    body = " · ".join(body_bits)
    return f"{prefix} · {body}" if body else str(prefix)


def _friendly_tiff_label(dir_name: str, path: Path) -> str:
    """Turn a focused-TIFF stem into a sample label.

    Stems look like ``condition__reporter__mutant__R{replica}_focused``
    (per ``BulkLayout._output_filename`` + the focus suffix). We split
    on ``__`` and drop the trailing ``_focused`` so the user sees the
    original condition fields.
    """
    stem = path.stem
    if stem.endswith("_focused"):
        stem = stem[: -len("_focused")]
    parts = [p for p in stem.split("__") if p]
    if len(parts) >= 2:
        cond, rep, *rest = parts
        mutant = rest[0] if rest else ""
        replica = rest[1].lstrip("R") if len(rest) > 1 else ""
        return _build_label(
            prefix=path.name,
            condition=cond.replace("_", " "),
            reporter=rep.replace("_", " "),
            mutant=mutant.replace("_", " "),
            replica=replica,
        )
    return f"{dir_name}/{path.name}"


def _count_czi_scenes(path: Path) -> int:
    return max(len(_list_czi_scene_indices(path)), 1)


def _list_czi_scene_indices(path: Path) -> list[int]:
    """Return the CZI's actual scene indices (e.g. ``[5, 6, 7]`` or ``[]``).

    Used so the panel can translate the FOV spinner's local position
    into the absolute CZI scene index pylibCZIrw expects — those
    indices are not always ``0..N-1``.
    """
    try:
        from mycoprep.core.focus.io_czi import list_scene_indices
        # ``list_scene_indices`` falls back to ``[0]`` for non-scene
        # CZIs (single-position files); strip that sentinel here so
        # callers can detect "no scenes" via empty list.
        with __import__("contextlib").suppress(Exception):
            from mycoprep.core.focus.io_czi import _open
            with _open(path) as doc:
                rects = doc.scenes_bounding_rectangle
                if not rects:
                    return []
        return [int(s) for s in list_scene_indices(path)]
    except Exception:  # noqa: BLE001
        return []
