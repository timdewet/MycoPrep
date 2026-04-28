"""Append-only log panel fed by stage progress callbacks."""

from __future__ import annotations

import datetime as _dt
from typing import Literal

from PyQt6.QtGui import QTextCharFormat, QColor
from PyQt6.QtWidgets import QPlainTextEdit, QWidget

from ..ui import tokens, theme


Level = Literal["info", "warning", "error", "success"]


class LogView(QPlainTextEdit):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(2000)
        self._follow = True
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)

    def set_follow(self, follow: bool) -> None:
        self._follow = follow
        if follow:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def follow(self) -> bool:
        return self._follow

    def _on_scroll(self, value: int) -> None:
        # Auto-detach follow mode if the user scrolls away from the bottom.
        sb = self.verticalScrollBar()
        if value < sb.maximum() - 4:
            self._follow = False

    def log(self, message: str, level: Level = "info") -> None:
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        cursor = self.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        fmt = QTextCharFormat()
        pal = tokens.active()
        color = {
            "info":    pal.log_text,
            "warning": pal.log_warn,
            "error":   pal.log_error,
            "success": pal.log_success,
        }[level]
        fmt.setForeground(QColor(color))
        cursor.insertText(line + "\n", fmt)
        if self._follow:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def clear_log(self) -> None:
        self.clear()

    def copy_all(self) -> str:
        return self.toPlainText()
