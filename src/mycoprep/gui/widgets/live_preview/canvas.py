"""Stacked pyqtgraph canvas for the live preview.

Layers, bottom → top:

- z=0   phase channel (grayscale, percentile-clipped contrast).
- z=1+  one ImageItem per fluorescence channel, with a single-color LUT
        and additive blend. Per-channel toggle, opacity, and color are
        managed by the parent panel.
- z=10  segmentation boundary overlay (RGBA, colored by classifier
        decision when available; plain red otherwise).
- z=20+ per-object pg.TextItem labels at centroid_x/centroid_y.

Channel layers, the boundary overlay, and the label layer are reused
across renders (we never add/remove pyqtgraph items, only ``setImage`` /
``setText`` / ``setPos``) so re-renders don't churn the scene graph.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter
from PyQt6.QtWidgets import QGraphicsSimpleTextItem, QVBoxLayout, QWidget


# pyqtgraph defaults to column-major (x, y) image order; switch to row-major
# (y, x) so we can pass numpy arrays straight through without transposing.
pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)


# 0–255 RGBA tuples for the boundary overlay categories.
DECISION_COLORS: dict[str, tuple[int, int, int, int]] = {
    "good":   (51, 191, 76, 245),    # green
    "edge":   (140, 140, 140, 215),  # grey
    "debris": (242, 153, 26, 245),   # orange
    "bad":    (245, 69, 43, 245),    # red
}


# Per-cell label rendering. The text item lives in image-coordinate space
# (NOT screen-space) so it stays a fixed size *on the image* and shrinks
# as the user zooms out. This means labels are tiny / invisible at zoom
# out and readable once you zoom in to a cell — exactly what dense FOVs
# need to avoid label clutter.
LABEL_FONT_POINT_SIZE = 3.5
LABEL_FONT_FAMILY = ""  # use Qt default; family-agnostic
LABEL_COLOR = "#ffd400"   # warm yellow — high contrast against phase + most fluors


def _percentile_levels(image: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0
                       ) -> tuple[float, float]:
    """Robust contrast bounds; falls back to (0, 1) on empty input."""
    if image is None or image.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(image, (lo_pct, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


class MultiOverlayCanvas(QWidget):
    """ImageView-based canvas with stacked overlay layers.

    Public surface (used by :class:`LivePreviewPanel` and the worker
    callbacks):

    - :meth:`set_phase` — sets the bottom phase plane.
    - :meth:`set_channels` — sets fluorescence channel layers (call
      again to update; layers are reused).
    - :meth:`set_mask` — sets the boundary overlay from a labeled mask
      and (optionally) a per-label decisions dict.
    - :meth:`set_labels` — sets per-object text labels.
    - :meth:`clear` — wipe all layers (used when no FOV is selected).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()

        # Internal state tracked across renders so layers can be reused.
        self._phase_shape: tuple[int, int] | None = None
        self._channel_layers: list[pg.ImageItem] = []   # z=1+
        self._label_items: list[pg.TextItem] = []       # z=20+
        self._image_initialised = False

        # ROI handle (lazy: created on first show). Lives on top of all
        # image overlays so the user can always see + drag it.
        self._roi: Optional[pg.RectROI] = None
        self._roi_change_listeners: list = []

    # ----------------------------------------------------------------- build

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._image_view = pg.ImageView()
        self._image_view.ui.roiBtn.hide()
        self._image_view.ui.menuBtn.hide()
        # Default to grayscale instead of pyqtgraph's blue-orange LUT.
        self._image_view.setColorMap(pg.colormap.get("CET-L1"))
        self._image_view.view.setAspectLocked(True)
        self._image_view.view.invertY(True)  # image-style coords (origin top-left)
        self._image_view.setMinimumHeight(360)

        # Boundary overlay (RGBA, on top of all channel layers).
        self._overlay_item = pg.ImageItem(axisOrder="row-major")
        self._overlay_item.setZValue(10)
        self._image_view.view.addItem(self._overlay_item)

        outer.addWidget(self._image_view)

    # ----------------------------------------------------------------- API

    def clear(self) -> None:
        """Wipe phase, channels, mask, and labels."""
        # ImageView's setImage doesn't accept None; just hide via empty array.
        self._image_view.clear()
        for it in self._channel_layers:
            it.clear()
        self._overlay_item.clear()
        self._set_label_count(0)
        self._phase_shape = None
        self._image_initialised = False

    def set_phase(self, phase: np.ndarray) -> None:
        """Set the grayscale phase plane (z=0)."""
        if phase is None:
            self._image_view.clear()
            self._phase_shape = None
            return
        lo, hi = _percentile_levels(phase)
        # autoRange=False preserves the user's zoom across re-renders.
        self._image_view.setImage(
            phase.astype(np.float32, copy=False),
            autoRange=False,
            autoLevels=False,
            levels=(lo, hi),
        )
        if not self._image_initialised:
            self._image_view.view.autoRange()
            self._image_initialised = True
        self._phase_shape = phase.shape

    def ensure_channel_item_count(self, n: int) -> None:
        """Pre-allocate ``n`` channel ``ImageItem`` slots.

        Used by the panel to bind a ``HistogramLUTWidget`` per channel
        before image data has been pushed in — the items have to exist
        first for the histogram to attach to.
        """
        while len(self._channel_layers) < n:
            it = pg.ImageItem(axisOrder="row-major")
            it.setZValue(1 + len(self._channel_layers))
            it.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            self._image_view.view.addItem(it)
            self._channel_layers.append(it)
        while len(self._channel_layers) > n:
            it = self._channel_layers.pop()
            self._image_view.view.removeItem(it)

    def channel_image_item(self, idx: int) -> Optional[pg.ImageItem]:
        """Return the ImageItem for channel ``idx`` (None if out of range)."""
        if 0 <= idx < len(self._channel_layers):
            return self._channel_layers[idx]
        return None

    def set_channels(
        self,
        channels: list["ChannelLayer"] | None,
    ) -> None:
        """Set or update the fluorescence channel layers.

        ``channels`` is a list of :class:`ChannelLayer` records, one
        per channel, in z-order. The ``image`` field of an invisible
        channel can be ``None`` — the corresponding ImageItem is then
        cleared. ``ImageItem`` instances are reused across calls so
        external bindings (e.g. a ``HistogramLUTWidget``) stay valid.
        """
        channels = channels or []
        self.ensure_channel_item_count(len(channels))

        for layer, item in zip(channels, self._channel_layers):
            if layer.image is None:
                item.clear()
                continue
            # Manual override > auto-detected percentile bounds.
            auto_lo, auto_hi = _percentile_levels(layer.image)
            lo = float(layer.level_lo) if layer.level_lo is not None else auto_lo
            hi = float(layer.level_hi) if layer.level_hi is not None else auto_hi
            if hi <= lo:
                hi = lo + 1.0
            lut = _single_color_lut(layer.color)
            item.setImage(
                layer.image.astype(np.float32, copy=False),
                autoLevels=False,
                levels=(lo, hi),
            )
            item.setLookupTable(lut)
            item.setOpacity(float(layer.opacity))

    def set_mask(
        self,
        mask: np.ndarray | None,
        decisions: dict[int, str] | None = None,
    ) -> None:
        """Render boundary overlay from a labeled mask.

        ``decisions`` maps label → category (good/edge/debris/bad); if
        omitted, all boundaries are drawn red. ``None`` mask clears the
        overlay.
        """
        if mask is None or mask.size == 0 or int(mask.max()) == 0:
            self._overlay_item.clear()
            return

        from skimage.segmentation import find_boundaries

        overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
        if decisions:
            for category, color in DECISION_COLORS.items():
                cat_labels = [lbl for lbl, dec in decisions.items() if dec == category]
                if not cat_labels:
                    continue
                cat_mask = np.isin(mask, cat_labels)
                if not cat_mask.any():
                    continue
                bnd = find_boundaries(np.where(cat_mask, mask, 0), mode="outer")
                overlay[bnd] = color
        else:
            bnd = find_boundaries(mask, mode="outer")
            overlay[bnd] = DECISION_COLORS["bad"]
        self._overlay_item.setImage(overlay, autoLevels=False)

    def set_labels(
        self,
        labels: list["TextLabel"] | None,
    ) -> None:
        """Place per-object text labels at given coordinates.

        ``labels`` is a list of :class:`TextLabel`. Reuses existing
        text items where possible.
        """
        labels = labels or []
        self._set_label_count(len(labels))
        for txt, item in zip(labels, self._label_items):
            # ``txt.color`` (TextLabel default white) is overridden by the
            # canvas-wide LABEL_COLOR for visual consistency, while still
            # letting individual labels override if needed.
            item.setText(txt.text)
            if txt.color and txt.color.lower() != "#ffffff":
                item.setBrush(QBrush(QColor(txt.color)))
            else:
                item.setBrush(QBrush(QColor(LABEL_COLOR)))
            # Centre the item over (x, y) in image coords. With Qt's
            # Y-down convention (which matches the view's invertY=True
            # display), shifting by ``-h/2`` puts the bounding rect's
            # midpoint on the centroid.
            br = item.boundingRect()
            item.setPos(txt.x - br.width() / 2.0,
                        txt.y - br.height() / 2.0)
            item.setVisible(True)

    # ----------------------------------------------------------------- helpers

    # --------------------------------------------------- ROI (segmentation crop)

    def set_roi_visible(self, visible: bool) -> None:
        """Show or hide the segmentation-crop ROI rectangle.

        On first show, the ROI is centred on the current image at ~50%
        of the smaller dimension. Subsequent toggles preserve the
        user's last position / size.
        """
        if visible and self._roi is None:
            self._build_roi()
        if self._roi is not None:
            self._roi.setVisible(visible)
            for handle in self._roi.handles:
                handle["item"].setVisible(visible)

    def get_roi_bounds(self) -> Optional[tuple[int, int, int, int]]:
        """Return the ROI bounds clamped to the image, as ``(x0, y0, x1, y1)``,
        or ``None`` when no ROI is shown / valid."""
        if self._roi is None or not self._roi.isVisible():
            return None
        if self._phase_shape is None:
            return None
        h, w = self._phase_shape
        pos = self._roi.pos()
        size = self._roi.size()
        x0 = int(max(0.0, pos.x()))
        y0 = int(max(0.0, pos.y()))
        x1 = int(min(float(w), pos.x() + size.x()))
        y1 = int(min(float(h), pos.y() + size.y()))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None  # too small to bother with
        return (x0, y0, x1, y1)

    def add_roi_change_listener(self, callback) -> None:
        """Register a callback invoked when the user finishes dragging
        / resizing the ROI."""
        self._roi_change_listeners.append(callback)

    def _build_roi(self) -> None:
        if self._phase_shape is None:
            h = w = 512
        else:
            h, w = self._phase_shape
        side = max(64, int(min(h, w) * 0.5))
        x = max(0, (w - side) // 2)
        y = max(0, (h - side) // 2)
        roi = pg.RectROI(
            pos=[x, y],
            size=[side, side],
            pen=pg.mkPen("y", width=2),
            movable=True,
            resizable=True,
        )
        roi.setZValue(50)
        self._image_view.view.addItem(roi)
        roi.sigRegionChangeFinished.connect(self._on_roi_changed)
        self._roi = roi

    def _on_roi_changed(self, *_args) -> None:
        for cb in list(self._roi_change_listeners):
            try:
                cb()
            except Exception:  # noqa: BLE001
                pass

    def _set_label_count(self, n: int) -> None:
        """Grow / shrink the pool of label items to ``n``.

        We use ``QGraphicsSimpleTextItem`` directly rather than
        ``pg.TextItem`` because the latter overrides its own transform
        on every view change to keep itself at screen size — which
        fights the "scale with the image" behavior we want here.

        - ``ItemIgnoresTransformations`` stays at its default
          ``False`` so the text scales with the view: small / hidden
          when zoomed out, larger when zoomed in. That keeps the
          font fixed in *image* space, the same size relative to
          the bacteria at every zoom level.
        - No per-item Y-flip is applied. With pyqtgraph's
          ``invertY=True``, the scene's Y axis already runs
          Qt-default (top-down), which matches the orientation of
          glyph rendering — text comes out right-side-up.
        - ``setPointSizeF`` is interpreted in scene/image-pixel
          units since the item participates in the view transform.
        """
        while len(self._label_items) < n:
            item = QGraphicsSimpleTextItem()
            item.setZValue(20 + len(self._label_items))
            item.setBrush(QBrush(QColor(LABEL_COLOR)))
            font = QFont() if not LABEL_FONT_FAMILY else QFont(LABEL_FONT_FAMILY)
            font.setPointSizeF(LABEL_FONT_POINT_SIZE)
            item.setFont(font)
            self._image_view.view.addItem(item)
            self._label_items.append(item)
        # Hide (don't remove) excess items; setText() costs are bounded.
        for item in self._label_items[n:]:
            item.setVisible(False)


# ---------------------------------------------------------------------------
# Small data records — kept here so the canvas defines its own input contract.
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class ChannelLayer:
    """One fluorescence channel layer to composite over the phase plane."""

    image: np.ndarray | None
    color: str           # one of "green", "red", "blue", "magenta", "yellow", "cyan", "white"
    opacity: float       # 0.0 → 1.0
    # Manual contrast bounds in raw image-data units. ``None`` means
    # "auto" — fall back to the 1st / 99th percentile of the image.
    level_lo: float | None = None
    level_hi: float | None = None


@dataclass
class TextLabel:
    """One per-object label rendered above the segmentation overlay."""

    text: str
    x: float
    y: float
    color: str = "#ffffff"


# ---------------------------------------------------------------------------
# Color → LUT helpers
# ---------------------------------------------------------------------------

_NAMED_RGB: dict[str, tuple[int, int, int]] = {
    "green":   (0, 255, 0),
    "red":     (255, 0, 0),
    "blue":    (0, 128, 255),
    "magenta": (255, 0, 255),
    "yellow":  (255, 255, 0),
    "cyan":    (0, 255, 255),
    "white":   (255, 255, 255),
}


def _single_color_lut(color: str) -> np.ndarray:
    """Build a 256×4 uint8 LUT that ramps black → ``color``.

    Used to recolor a single-channel intensity image with one chosen hue
    while keeping the dynamic range as alpha-modulated luminance.
    """
    r, g, b = _NAMED_RGB.get(color, _NAMED_RGB["white"])
    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.zeros((256, 4), dtype=np.uint8)
    lut[:, 0] = (ramp * r).astype(np.uint8)
    lut[:, 1] = (ramp * g).astype(np.uint8)
    lut[:, 2] = (ramp * b).astype(np.uint8)
    lut[:, 3] = 255
    return lut


def color_for_channel_name(name: str) -> str:
    """Heuristic mapping from channel label to a default display color."""
    if not name:
        return "white"
    n = name.lower()
    if any(k in n for k in ("gfp", "yfp", "fitc", "alexa488", "488")):
        return "green"
    if any(k in n for k in ("rfp", "mcherry", "texas", "alexa594", "594", "555")):
        return "red"
    if any(k in n for k in ("dapi", "hoechst", "405")):
        return "blue"
    if any(k in n for k in ("cy5", "alexa647", "647")):
        return "magenta"
    return "white"
