"""Slider + numeric spinbox + helper text — for scientific threshold parameters.

Both controls stay in sync. Helper text sits below in muted style so it never
fights the active value visually.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import tokens


class LabeledSlider(QWidget):
    """Float-valued slider (under the hood QSlider is integer; we scale).

    Layout:
        [Label]                                   [spinbox  ]
        [——————slider——————————————————————]
        [helper text                                       ]
    """

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        step: float = 0.05,
        decimals: int = 2,
        suffix: str = "",
        helper: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._step = step
        self._decimals = decimals

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(tokens.S1)

        # Header row: name + numeric control
        head = QHBoxLayout()
        head.setSpacing(tokens.S2)
        self._label = QLabel(label)
        self._label.setObjectName("h3")
        self._label.setStyleSheet(
            f"font-size: {tokens.FS_BODY}px; font-weight: {tokens.FW_MEDIUM};"
        )
        head.addWidget(self._label)
        head.addStretch(1)

        self._spin = QDoubleSpinBox()
        self._spin.setRange(minimum, maximum)
        self._spin.setSingleStep(step)
        self._spin.setDecimals(decimals)
        self._spin.setValue(value)
        if suffix:
            self._spin.setSuffix(f" {suffix}")
        self._spin.setFixedWidth(96)
        self._spin.valueChanged.connect(self._on_spin)
        head.addWidget(self._spin)
        v.addLayout(head)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._steps())
        self._slider.setSingleStep(1)
        self._slider.setPageStep(max(1, self._steps() // 10))
        self._slider.setValue(self._to_slider(value))
        self._slider.valueChanged.connect(self._on_slider)
        self._slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        v.addWidget(self._slider)

        # Helper text
        self._helper = QLabel(helper)
        self._helper.setObjectName("caption")
        self._helper.setWordWrap(True)
        self._helper.setStyleSheet(
            f"color: {tokens.active().text_subtle}; font-size: {tokens.FS_CAPTION}px;"
        )
        if not helper:
            self._helper.hide()
        v.addWidget(self._helper)

    def _steps(self) -> int:
        return max(1, int(round((self._max - self._min) / self._step)))

    def _to_slider(self, value: float) -> int:
        return int(round((value - self._min) / self._step))

    def _to_value(self, slider_int: int) -> float:
        return self._min + slider_int * self._step

    # ── slot wiring ────────────────────────────────────────────────────
    def _on_slider(self, slider_int: int) -> None:
        v = self._to_value(slider_int)
        if abs(v - self._spin.value()) > self._step / 2:
            self._spin.blockSignals(True)
            self._spin.setValue(v)
            self._spin.blockSignals(False)
        self.valueChanged.emit(v)

    def _on_spin(self, v: float) -> None:
        s = self._to_slider(v)
        if s != self._slider.value():
            self._slider.blockSignals(True)
            self._slider.setValue(s)
            self._slider.blockSignals(False)
        self.valueChanged.emit(v)

    # ── public API ─────────────────────────────────────────────────────
    def value(self) -> float:
        return self._spin.value()

    def setValue(self, v: float) -> None:  # noqa: N802 (Qt API style)
        self._spin.setValue(v)

    def set_helper(self, text: str) -> None:
        self._helper.setText(text)
        self._helper.setVisible(bool(text))

    def setToolTip(self, text: str) -> None:  # noqa: N802
        super().setToolTip(text)
        self._spin.setToolTip(text)
        self._slider.setToolTip(text)
