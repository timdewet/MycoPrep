"""Visual plate-map editor.

Renders a grid of wells (A1, A2, ... H12 for a 96-well plate) as clickable
buttons. Click/drag to select a group of wells, then use the condition
editor panel on the right to fill in condition / reporter / mutant_or_drug
for the whole selection at once. Wells are auto-coloured by condition so
the plate layout is visible at a glance.

The widget works on a PlateLayout instance (wrapping a pandas DataFrame).
Mutations go directly into `layout.df`; consumers subscribe to the
`layoutChanged` signal to react.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..pipeline.layout import PlateLayout, normalize_well_id
from ..ui import theme, tokens


# ─────────────────────────────────────────────────────────────────────────────
# Condition → colour (stable hue from hash; saturation/value from active palette)
# ─────────────────────────────────────────────────────────────────────────────


def empty_fill_color() -> QColor:
    return QColor(tokens.active().well_empty_fill)


def empty_ring_color() -> QColor:
    return QColor(tokens.active().well_empty_ring)


def condition_color(condition: str) -> QColor:
    """Fill colour for the well (hue-stable, saturation/value from theme)."""
    if not condition:
        return empty_fill_color()
    p = tokens.active()
    h = hashlib.md5(condition.encode("utf-8")).digest()
    hue = h[0] / 255.0 * 360.0
    return QColor.fromHsv(int(hue), p.well_fill_s, p.well_fill_v)


def reporter_color(reporter: str) -> QColor:
    """Ring colour (hue-stable, saturation/value from theme)."""
    if not reporter:
        return empty_ring_color()
    p = tokens.active()
    h = hashlib.md5(("reporter:" + reporter).encode("utf-8")).digest()
    hue = h[0] / 255.0 * 360.0
    return QColor.fromHsv(int(hue), p.well_ring_s, p.well_ring_v)


# ─────────────────────────────────────────────────────────────────────────────
# Well button
# ─────────────────────────────────────────────────────────────────────────────

class WellButton(QToolButton):
    """Checkable circular button representing one well.

    Painted manually so we can show, at the same time:
      - condition (fill colour)
      - reporter (outer ring colour)
      - the well ID as primary text
      - mutant/drug + replica as small secondary text underneath
    """

    SIDE_PX = 56

    def __init__(self, well_id: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.well_id = well_id
        self.setCheckable(True)
        # Text is drawn manually in paintEvent — disable the builtin label.
        self.setText("")
        self.setFixedSize(self.SIDE_PX, self.SIDE_PX)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # Model-driven state
        self._fill = empty_fill_color()
        self._ring = empty_ring_color()
        self._is_empty_fill = True
        self._sub_text = ""       # drawn underneath the well ID
        self._has_data = True
        self._hovered = False
        self.setMouseTracking(True)

    # ----------------------------------------------------------------- setters

    def set_colors(self, fill: QColor, ring: QColor, is_empty: bool = False) -> None:
        self._fill = fill
        self._ring = ring
        self._is_empty_fill = is_empty
        self.update()

    def set_sub_text(self, text: str) -> None:
        self._sub_text = text
        self.update()

    def set_has_data(self, has_data: bool) -> None:
        self._has_data = has_data
        if not has_data:
            self.setEnabled(False)
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    # ----------------------------------------------------------------- paint

    # Drag-paint hooks. Set by PlateMapEditor when constructing each button.
    on_drag_start: "Optional[callable]" = None    # type: ignore[assignment]
    on_drag_enter: "Optional[callable]" = None    # type: ignore[assignment]
    on_drag_end:   "Optional[callable]" = None    # type: ignore[assignment]

    def mousePressEvent(self, event) -> None:     # noqa: N802
        super().mousePressEvent(event)
        self.update()
        if event.button() == Qt.MouseButton.LeftButton and self.on_drag_start:
            self.on_drag_start(self.well_id, self.isChecked())

    def mouseMoveEvent(self, event) -> None:      # noqa: N802
        super().mouseMoveEvent(event)
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if self.on_drag_enter is None:
            return
        parent = self.parentWidget()
        if parent is None:
            return
        gpos = self.mapTo(parent, event.position().toPoint())
        for w in parent.findChildren(WellButton):
            if w is self:
                continue
            if w.geometry().contains(gpos):
                self.on_drag_enter(w.well_id)
                break

    def mouseReleaseEvent(self, event) -> None:   # noqa: N802
        super().mouseReleaseEvent(event)
        self.update()
        if event.button() == Qt.MouseButton.LeftButton and self.on_drag_end:
            self.on_drag_end()

    def nextCheckState(self) -> None:             # noqa: N802
        super().nextCheckState()
        self.update()

    def enterEvent(self, e) -> None:   # noqa: N802
        self._hovered = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:   # noqa: N802
        self._hovered = False
        self.update()
        super().leaveEvent(e)

    def paintEvent(self, _event) -> None:   # noqa: N802
        from PyQt6.QtGui import QFont, QPainter, QPen
        from PyQt6.QtCore import QRectF

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(self.rect()).adjusted(2, 2, -2, -2)

        pal = tokens.active()

        if not self._has_data:
            # Empty slot — dashed outline, no text
            pen = QPen(QColor(pal.border_strong))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidthF(1.0)
            p.setPen(pen)
            p.setBrush(QColor(pal.surface_alt))
            p.drawEllipse(rect)
            return

        # Ring (reporter colour). Checked = text overlay; hover = primary overlay.
        if self.isChecked():
            ring = QColor(pal.text)
            ring_w = 2.5
        elif self._hovered:
            ring = QColor(pal.primary)
            ring_w = 2.0
        else:
            ring = self._ring
            ring_w = 2.0

        p.setPen(QPen(ring, ring_w))
        p.setBrush(self._fill)
        p.drawEllipse(rect.adjusted(ring_w / 2, ring_w / 2, -ring_w / 2, -ring_w / 2))

        # Text — primary well ID on top, sub-text below
        text_color = QColor(pal.text) if not self._is_empty_fill else QColor(pal.text_subtle)
        p.setPen(text_color)

        if self._sub_text:
            top_font = QFont(self.font()); top_font.setPointSize(9);  top_font.setBold(True)
            bot_font = QFont(self.font()); bot_font.setPointSize(7)
            p.setFont(top_font)
            top_rect = QRectF(rect.left(), rect.top() + 6, rect.width(), rect.height() * 0.5)
            p.drawText(top_rect, int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
                       self.well_id)
            p.setFont(bot_font)
            bot_rect = QRectF(rect.left(), rect.center().y(), rect.width(), rect.height() * 0.5 - 4)
            elided = p.fontMetrics().elidedText(self._sub_text, Qt.TextElideMode.ElideRight,
                                                int(rect.width() - 6))
            p.drawText(bot_rect, int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
                       elided)
        else:
            f = QFont(self.font()); f.setPointSize(10); f.setBold(True)
            p.setFont(f)
            p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), self.well_id)

# ─────────────────────────────────────────────────────────────────────────────
# Plate editor
# ─────────────────────────────────────────────────────────────────────────────

class PlateMapEditor(QWidget):
    """Visual editor for a PlateLayout."""

    layoutChanged = pyqtSignal()

    def __init__(self, layout: PlateLayout, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._layout_model = layout
        self._buttons: dict[str, WellButton] = {}
        # Drag-paint state
        self._drag_active = False
        self._drag_paint_value = False  # what to set hovered wells to
        self._drag_origin: Optional[str] = None
        self._build_ui()
        self._rebuild_grid()
        self._refresh_colors()
        # Recolour when the theme flips.
        theme.add_theme_listener(lambda _p: self._on_theme_changed())

    # ----------------------------------------------------------------- model

    def set_layout(self, layout: PlateLayout) -> None:
        self._layout_model = layout
        self._rebuild_grid()
        self._refresh_colors()

    @property
    def layout_model(self) -> PlateLayout:
        return self._layout_model

    # -------------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        main = QHBoxLayout(self)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(16)

        # Left: plate grid + legend
        left = QVBoxLayout()
        left.setSpacing(8)
        self._grid_host = QWidget()
        self._grid_host.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self._grid = QGridLayout(self._grid_host)
        self._grid.setSpacing(5)
        self._grid.setContentsMargins(8, 8, 8, 8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)  # ← centre the grid in its viewport
        scroll.setWidget(self._grid_host)
        left.addWidget(scroll, stretch=1)

        legend_title = QLabel("Conditions")
        legend_title.setObjectName("sectionHeader")
        self._legend_host = QWidget()
        self._legend_layout = QHBoxLayout(self._legend_host)
        self._legend_layout.setContentsMargins(0, 0, 0, 0)
        self._legend_layout.setSpacing(10)
        legend_scroll = QScrollArea()
        legend_scroll.setWidgetResizable(True)
        legend_scroll.setFrameShape(legend_scroll.Shape.NoFrame)
        legend_scroll.setFixedHeight(46)
        legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        legend_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        legend_scroll.setWidget(self._legend_host)
        left.addWidget(legend_title)
        left.addWidget(legend_scroll)

        left_wrap = QWidget(); left_wrap.setLayout(left)
        main.addWidget(left_wrap, stretch=3)

        # Right: selection-driven condition editor
        editor = QWidget()
        form = QVBoxLayout(editor)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        title = QLabel("Selected wells")
        title.setObjectName("sectionHeader")
        self._selection_label = QLabel("(none)")
        self._selection_label.setObjectName("muted")
        self._selection_label.setWordWrap(True)

        self._condition_edit = QLineEdit()
        self._reporter_edit = QLineEdit()
        self._mutant_edit = QLineEdit()
        self._replica_edit = QSpinBox()
        self._replica_edit.setRange(0, 999)
        self._replica_edit.setSpecialValueText("—")   # 0 = unset
        self._replica_edit.setValue(0)
        self._notes_edit = QLineEdit()

        apply_btn = QPushButton("Apply to selection")
        apply_btn.setObjectName("primary")
        apply_btn.clicked.connect(self._apply_fields)

        clear_fields_btn = QPushButton("Clear fields")
        clear_fields_btn.clicked.connect(self._clear_selection_fields)

        select_all_btn = QPushButton("Select all")
        select_all_btn.clicked.connect(self._select_all)
        select_none_btn = QPushButton("Clear selection")
        select_none_btn.clicked.connect(self._select_none)

        form.addWidget(title)
        form.addWidget(self._selection_label)
        form.addSpacing(10)
        for label_text, widget in [
            ("Condition", self._condition_edit),
            ("Reporter", self._reporter_edit),
            ("Mutant / drug", self._mutant_edit),
            ("Replica", self._replica_edit),
            ("Notes", self._notes_edit),
        ]:
            lbl = QLabel(label_text)
            lbl.setObjectName("muted")
            form.addWidget(lbl)
            form.addWidget(widget)
        form.addSpacing(8)
        form.addWidget(apply_btn)
        form.addWidget(clear_fields_btn)
        form.addSpacing(12)
        sel_row = QHBoxLayout()
        sel_row.addWidget(select_all_btn); sel_row.addWidget(select_none_btn)
        form.addLayout(sel_row)
        form.addStretch(1)

        editor.setMaximumWidth(280)
        main.addWidget(editor, stretch=2)

    def _rebuild_grid(self) -> None:
        # Clear previous buttons
        for i in reversed(range(self._grid.count())):
            item = self._grid.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        self._buttons.clear()

        rows = self._layout_model.rows
        cols = self._layout_model.cols
        row0 = ord(self._layout_model.row_offset) - ord("A")
        col0 = self._layout_model.col_offset

        header_style = (
            f"color: {tokens.active().text_muted}; font-weight: 600; font-size: 11px;"
        )

        # Column headers
        for c in range(cols):
            lbl = QLabel(str(col0 + c))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(header_style)
            self._grid.addWidget(lbl, 0, c + 1)

        # Row headers + wells
        for r in range(rows):
            row_letter = chr(ord("A") + row0 + r)
            row_label = QLabel(row_letter)
            row_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row_label.setStyleSheet(header_style)
            self._grid.addWidget(row_label, r + 1, 0)
            for c in range(cols):
                well = f"{row_letter}{col0 + c}"
                btn = WellButton(well, parent=self._grid_host)
                btn.toggled.connect(self._on_well_toggled)
                btn.on_drag_start = self._on_drag_start
                btn.on_drag_enter = self._on_drag_enter
                btn.on_drag_end = self._on_drag_end
                btn.set_has_data(self._layout_model.has_data(well))
                self._grid.addWidget(btn, r + 1, c + 1)
                self._buttons[well] = btn

    # ----------------------------------------------------------------- events

    def _on_drag_start(self, well: str, paint_value: bool) -> None:
        self._drag_active = True
        self._drag_paint_value = paint_value
        self._drag_origin = well

    def _on_drag_enter(self, well: str) -> None:
        if not self._drag_active:
            return
        btn = self._buttons.get(well)
        if btn is None or not btn.isEnabled():
            return
        if btn.isChecked() != self._drag_paint_value:
            btn.setChecked(self._drag_paint_value)

    def _on_drag_end(self) -> None:
        self._drag_active = False
        self._drag_origin = None

    def _on_well_toggled(self, _checked: bool) -> None:
        selected = self.selected_wells()
        if not selected:
            self._selection_label.setText("(none)")
        else:
            self._selection_label.setText(", ".join(sorted(selected)))
            # If all selected wells share the same values, prefill the editor
            self._prefill_from_selection(selected)

    def _prefill_from_selection(self, wells: list[str]) -> None:
        df = self._layout_model.df
        rows = df[df["well"].isin(wells)]
        if rows.empty:
            return
        # Text-style fields
        for col, widget in [
            ("condition", self._condition_edit),
            ("reporter", self._reporter_edit),
            ("mutant_or_drug", self._mutant_edit),
            ("notes", self._notes_edit),
        ]:
            values = set(rows[col].astype(str))
            if len(values) == 1:
                widget.setText(next(iter(values)))
            else:
                widget.setText("")
                widget.setPlaceholderText("(multiple values)")
        # Numeric replica
        rep_values = set(str(v).strip() for v in rows["replica"])
        if len(rep_values) == 1:
            v = next(iter(rep_values))
            try:
                self._replica_edit.setValue(int(v) if v else 0)
            except ValueError:
                self._replica_edit.setValue(0)
        else:
            self._replica_edit.setValue(0)

    def _apply_fields(self) -> None:
        wells = self.selected_wells()
        if not wells:
            return
        rep_val = self._replica_edit.value()
        fields = {
            "condition": self._condition_edit.text().strip(),
            "reporter": self._reporter_edit.text().strip(),
            "mutant_or_drug": self._mutant_edit.text().strip(),
            "replica": str(rep_val) if rep_val > 0 else "",
            "notes": self._notes_edit.text().strip(),
        }
        # Only apply fields the user actually set (non-empty)
        fields = {k: v for k, v in fields.items() if v != ""}
        if not fields:
            return
        self._layout_model.set_wells(wells, **fields)
        self._refresh_colors()
        self.layoutChanged.emit()
        # Clear the selection so the next label-apply is a deliberate re-select,
        # not an accidental overwrite of the wells we just edited.
        self._select_none()
        self._clear_selection_fields()

    def _clear_selection_fields(self) -> None:
        for w in [self._condition_edit, self._reporter_edit, self._mutant_edit,
                  self._notes_edit]:
            w.clear()
        self._replica_edit.setValue(0)

    def _select_all(self) -> None:
        for b in self._buttons.values():
            b.setChecked(True)

    def _select_none(self) -> None:
        for b in self._buttons.values():
            b.setChecked(False)

    # ----------------------------------------------------------------- helpers

    def selected_wells(self) -> list[str]:
        return [w for w, b in self._buttons.items() if b.isChecked()]

    def _on_theme_changed(self) -> None:
        # Rebuild grid headers with the new muted color, then recolour wells.
        self._rebuild_grid()
        self._refresh_colors()

    def _refresh_colors(self) -> None:
        df = self._layout_model.df
        for well, btn in self._buttons.items():
            rows = df[df["well"] == well]
            if rows.empty:
                btn.set_colors(empty_fill_color(), empty_ring_color(), is_empty=True)
                btn.set_sub_text("")
                continue
            row = rows.iloc[0]
            condition = str(row["condition"])
            reporter  = str(row["reporter"])
            mutant    = str(row["mutant_or_drug"]).strip()
            replica   = str(row["replica"]).strip()

            btn.set_colors(
                condition_color(condition),
                reporter_color(reporter),
                is_empty=not bool(condition),
            )

            sub_parts = [p for p in [mutant, f"R{replica}" if replica else ""] if p]
            btn.set_sub_text(" ".join(sub_parts))
        self._refresh_legend()

    def _refresh_legend(self) -> None:
        # Clear existing
        for i in reversed(range(self._legend_layout.count())):
            item = self._legend_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
            else:
                self._legend_layout.removeItem(item)

        df = self._layout_model.df
        conditions = sorted({str(c) for c in df["condition"] if str(c).strip()})
        reporters  = sorted({str(r) for r in df["reporter"]  if str(r).strip()})

        if not conditions and not reporters:
            placeholder = QLabel("(no conditions or reporters assigned yet)")
            placeholder.setObjectName("muted")
            self._legend_layout.addWidget(placeholder)
            self._legend_layout.addStretch(1)
            return

        def _chip(swatch_color: QColor, border_color: QColor,
                  label_text: str, shape: str = "circle") -> QWidget:
            chip = QWidget()
            h = QHBoxLayout(chip)
            h.setContentsMargins(8, 4, 10, 4); h.setSpacing(6)
            sw = QLabel()
            sw.setFixedSize(14, 14)
            if shape == "ring":
                sw.setStyleSheet(
                    f"background: transparent; border-radius: 7px; "
                    f"border: 3px solid {swatch_color.name()};"
                )
            else:
                sw.setStyleSheet(
                    f"background: {swatch_color.name()}; border-radius: 7px; "
                    f"border: 1px solid {border_color.name()};"
                )
            txt = QLabel(label_text)
            h.addWidget(sw); h.addWidget(txt)
            pal = tokens.active()
            chip.setStyleSheet(
                f"QWidget {{ background: {pal.surface}; border: 1px solid {pal.border}; "
                f"border-radius: 14px; }}"
                "QLabel { background: transparent; border: none; }"
            )
            return chip

        if conditions:
            hdr = QLabel("Condition:"); hdr.setObjectName("muted")
            self._legend_layout.addWidget(hdr)
            for cond in conditions:
                n = int((df["condition"] == cond).sum())
                self._legend_layout.addWidget(
                    _chip(condition_color(cond),
                          condition_color(cond).darker(135),
                          f"{cond}  ·  {n}",
                          shape="circle")
                )

        if reporters:
            hdr = QLabel("Reporter:"); hdr.setObjectName("muted")
            self._legend_layout.addWidget(hdr)
            for rep in reporters:
                n = int((df["reporter"] == rep).sum())
                self._legend_layout.addWidget(
                    _chip(reporter_color(rep), reporter_color(rep),
                          f"{rep}  ·  {n}", shape="ring")
                )
        self._legend_layout.addStretch(1)
