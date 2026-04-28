"""Multi-stage progress stepper.

A horizontal row of stages: each is a numbered circle + label + thin
progress bar that fills as the stage runs. Replaces the four stacked
QProgressBars in the Run panel.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import tokens, theme


class StepState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    SKIPPED = "skipped"


class _StepBadge(QWidget):
    DIAM = 28

    def __init__(self, idx: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._idx = idx
        self._state = StepState.PENDING
        self.setFixedSize(self.DIAM, self.DIAM)

    def set_state(self, s: StepState) -> None:
        self._state = s
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pal = tokens.active()
        if self._state is StepState.DONE:
            fill = QColor(pal.success); text = pal.on_primary; mark = "✓"
        elif self._state is StepState.RUNNING:
            fill = QColor(pal.primary); text = pal.on_primary; mark = str(self._idx)
        elif self._state is StepState.ERROR:
            fill = QColor(pal.danger); text = pal.on_primary; mark = "!"
        elif self._state is StepState.SKIPPED:
            fill = QColor(pal.surface_alt); text = pal.text_subtle; mark = "–"
        else:
            fill = QColor(pal.surface_alt); text = pal.text_muted; mark = str(self._idx)

        p.setPen(QPen(QColor(pal.border_strong), 1))
        p.setBrush(fill)
        p.drawEllipse(1, 1, self.DIAM - 2, self.DIAM - 2)
        p.setPen(QColor(text))
        f = QFont(); f.setPointSize(10); f.setBold(True); p.setFont(f)
        p.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), mark)


class _Step(QWidget):
    def __init__(self, idx: int, name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.name = name
        self._fraction = 0.0
        self._state = StepState.PENDING

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(tokens.S1)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(tokens.S2)
        self._badge = _StepBadge(idx, self)
        self._label = QLabel(name)
        self._label.setObjectName("h3")
        self._pct = QLabel("")
        self._pct.setObjectName("muted")
        head.addWidget(self._badge)
        head.addWidget(self._label)
        head.addStretch(1)
        head.addWidget(self._pct)
        v.addLayout(head)

        # Slim progress strip drawn manually
        self._strip = _ProgressStrip(self)
        self._strip.setFixedHeight(4)
        v.addWidget(self._strip)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_state(self, s: StepState) -> None:
        self._state = s
        self._badge.set_state(s)
        self._strip.set_state(s)
        if s is StepState.PENDING:
            self._pct.setText("")
        elif s is StepState.SKIPPED:
            self._pct.setText("skipped")
        elif s is StepState.DONE:
            self._pct.setText("done")
            self._fraction = 1.0
            self._strip.set_fraction(1.0)
        elif s is StepState.ERROR:
            self._pct.setText("error")

    def set_fraction(self, f: float) -> None:
        self._fraction = max(0.0, min(1.0, f))
        self._strip.set_fraction(self._fraction)
        if self._state is StepState.RUNNING:
            self._pct.setText(f"{int(self._fraction * 100)}%")


class _ProgressStrip(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._f = 0.0
        self._state = StepState.PENDING

    def set_fraction(self, f: float) -> None:
        self._f = f
        self.update()

    def set_state(self, s: StepState) -> None:
        self._state = s
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pal = tokens.active()
        w, h = self.width(), self.height()
        # Track
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(pal.surface_alt))
        p.drawRoundedRect(0, 0, w, h, h / 2, h / 2)
        # Fill
        if self._state is StepState.ERROR:
            fill = QColor(pal.danger)
        elif self._state is StepState.DONE:
            fill = QColor(pal.success)
        elif self._state is StepState.SKIPPED:
            fill = QColor(pal.border_strong)
        else:
            fill = QColor(pal.primary)
        fw = int(w * self._f)
        if fw > 0:
            p.setBrush(fill)
            p.drawRoundedRect(0, 0, fw, h, h / 2, h / 2)


class Stepper(QWidget):
    """Horizontal stepper with one step per stage."""

    def __init__(self, names: list[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._steps: dict[str, _Step] = {}
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(tokens.S4)
        for i, name in enumerate(names, start=1):
            step = _Step(i, name, self)
            self._steps[name] = step
            h.addWidget(step, stretch=1)
        theme.add_theme_listener(lambda _p: self._repaint_all())

    def _repaint_all(self) -> None:
        for s in self._steps.values():
            s._badge.update()
            s._strip.update()

    # ── public API ─────────────────────────────────────────────────────
    def set_state(self, name: str, state: StepState) -> None:
        if name in self._steps:
            self._steps[name].set_state(state)

    def set_fraction(self, name: str, fraction: float) -> None:
        if name in self._steps:
            self._steps[name].set_fraction(fraction)

    def reset(self) -> None:
        for s in self._steps.values():
            s.set_state(StepState.PENDING)
            s.set_fraction(0.0)
