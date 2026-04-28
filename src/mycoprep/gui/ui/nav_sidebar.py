"""Vertical navigation sidebar: one row per pipeline stage.

Each row is a checkable, auto-exclusive QPushButton with an icon, label,
and a status dot. Emits `currentChanged(int)` for driving a QStackedWidget.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import icons, tokens, theme


class StageStatus(str, Enum):
    IDLE = "idle"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    DISABLED = "disabled"


_STATUS_TO_ICON = {
    StageStatus.IDLE:     ("status-idle",     "muted"),
    StageStatus.READY:    ("status-ready",    "primary"),
    StageStatus.RUNNING:  ("status-running",  "warning"),
    StageStatus.DONE:     ("status-done",     "success"),
    StageStatus.ERROR:    ("status-error",    "danger"),
    StageStatus.DISABLED: ("status-disabled", "muted"),
}


class _StatusBadge(QLabel):
    """Small icon badge showing the stage's status."""

    SIZE = 16

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status = StageStatus.IDLE
        self._refresh()
        icons.on_theme_change(self._refresh)

    def set_status(self, s: StageStatus) -> None:
        self._status = s
        self._refresh()

    def _refresh(self) -> None:
        from PyQt6.QtCore import QSize
        glyph, role = _STATUS_TO_ICON[self._status]
        ic = icons.icon(glyph, role=role)  # type: ignore[arg-type]
        pix = ic.pixmap(QSize(self.SIZE, self.SIZE))
        if not pix.isNull():
            self.setPixmap(pix)
            self.setText("")
        else:
            # Fallback bullet glyphs when qtawesome is unavailable.
            fallback = {
                StageStatus.IDLE:     "○",
                StageStatus.READY:    "•",
                StageStatus.RUNNING:  "◐",
                StageStatus.DONE:     "✓",
                StageStatus.ERROR:    "!",
                StageStatus.DISABLED: "–",
            }[self._status]
            self.setText(fallback)


class _NavItem(QPushButton):
    def __init__(self, text: str, icon_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("navItem")
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._icon_name = icon_name
        self._label_text = text
        self._dot = _StatusBadge(self)
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # Build the row as inner child widgets so we can explicitly control
        # spacing between the icon and the label (Qt's setIcon path glues
        # them too tightly).
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(tokens.S3)  # 12px between icon and label

        self._icon_lbl = QLabel()
        self._icon_lbl.setFixedSize(QSize(18, 18))
        self._icon_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._icon_lbl.setStyleSheet("background: transparent;")
        layout.addWidget(self._icon_lbl)

        self._lbl = QLabel(text)
        self._lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._lbl.setStyleSheet("background: transparent;")
        layout.addWidget(self._lbl)
        layout.addStretch(1)
        layout.addWidget(self._dot)

        self._refresh_icon()
        icons.on_theme_change(self._refresh_icon)
        # autoExclusive un-checks the previous button via Qt's C++ path, which
        # bypasses our Python setChecked override. Hook toggled so the icon
        # and label restyle on both check AND uncheck regardless of cause.
        self.toggled.connect(lambda _on: self._refresh_icon())

    def _refresh_icon(self) -> None:
        role = "primary" if self.isChecked() else "muted"
        ic = icons.icon(self._icon_name, role=role)
        pix = ic.pixmap(QSize(18, 18))
        if not pix.isNull():
            self._icon_lbl.setPixmap(pix)
        else:
            # qtawesome unavailable — leave blank, label still readable.
            self._icon_lbl.clear()
        # Also update the label colour to match the row's selection state,
        # since the QPushButton's own color: rule doesn't reach the QLabel.
        pal = tokens.active()
        if self.isChecked():
            self._lbl.setStyleSheet(
                f"background: transparent; color: {pal.text}; "
                f"font-weight: {tokens.FW_SEMIBOLD};"
            )
        else:
            self._lbl.setStyleSheet(
                f"background: transparent; color: {pal.text_muted};"
            )

    def setChecked(self, checked: bool) -> None:  # noqa: N802
        super().setChecked(checked)
        self._refresh_icon()

    def set_status(self, s: StageStatus) -> None:
        self._dot.set_status(s)


@dataclass
class NavEntry:
    key: str
    label: str
    icon: str


class NavSidebar(QFrame):
    """Sidebar emitting `currentChanged(int)` when the user picks a stage."""

    currentChanged = pyqtSignal(int)

    def __init__(self, entries: list[NavEntry], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(tokens.SIDEBAR_WIDTH)
        self._entries = entries
        self._items: list[_NavItem] = []
        self._visible_keys: set[str] = {e.key for e in entries}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, tokens.S4, 0, tokens.S3)
        root.setSpacing(0)

        brand = QLabel("MycoPrep")
        brand.setObjectName("brand")
        brand.setContentsMargins(tokens.S4, tokens.S2, tokens.S4, tokens.S4)
        root.addWidget(brand)

        for i, entry in enumerate(entries):
            item = _NavItem(entry.label, entry.icon, self)
            item.clicked.connect(lambda _checked=False, idx=i: self._on_clicked(idx))
            root.addWidget(item)
            self._items.append(item)

        if self._items:
            self._items[0].setChecked(True)

        root.addStretch(1)

        # Footer
        footer = QWidget()
        fl = QVBoxLayout(footer)
        fl.setContentsMargins(tokens.S4, tokens.S2, tokens.S4, tokens.S2)
        fl.setSpacing(tokens.S1)
        self._gpu_label = QLabel()
        self._gpu_label.setObjectName("caption")
        fl.addWidget(self._gpu_label)
        root.addWidget(footer)
        self.set_gpu_available(False)

    def _on_clicked(self, idx: int) -> None:
        # autoExclusive handles the checked state; emit visible-index.
        self._items[idx].setChecked(True)
        self.currentChanged.emit(idx)

    # ── public API ─────────────────────────────────────────────────────
    def set_current(self, idx: int) -> None:
        if 0 <= idx < len(self._items):
            self._items[idx].setChecked(True)
            self.currentChanged.emit(idx)

    def set_status(self, key: str, status: StageStatus) -> None:
        for entry, item in zip(self._entries, self._items):
            if entry.key == key:
                item.set_status(status)
                return

    def set_visible(self, key: str, visible: bool) -> None:
        for entry, item in zip(self._entries, self._items):
            if entry.key == key:
                item.setVisible(visible)
                if visible:
                    self._visible_keys.add(key)
                else:
                    self._visible_keys.discard(key)
                return

    def set_gpu_available(self, ok: bool) -> None:
        glyph = "●" if ok else "○"
        text = "GPU available" if ok else "GPU unavailable (CPU)"
        self._gpu_label.setText(f"{glyph}  {text}")

    def index_of(self, key: str) -> int:
        for i, entry in enumerate(self._entries):
            if entry.key == key:
                return i
        return -1
