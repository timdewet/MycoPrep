"""Subtle drop-shadow elevation for cards.

Qt stylesheets can't render shadows, so we attach a `QGraphicsDropShadowEffect`
to the widget. Kept lightweight — every visible card gets a shadow but they
share the same effect parameters. Theme-aware so dark mode shadows lift the
surface against the darker bg.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect, QWidget

from . import theme, tokens


def apply_shadow(widget: QWidget, level: int = 1) -> None:
    """Attach a soft drop shadow.

    `level=1` is for default cards; higher values can be reserved for
    elevated surfaces (sticky bars).
    """
    effect = QGraphicsDropShadowEffect(widget)
    if level >= 2:
        effect.setBlurRadius(22)
        effect.setOffset(0, 6)
    else:
        effect.setBlurRadius(14)
        effect.setOffset(0, 2)
    effect.setColor(_shadow_color())
    widget.setGraphicsEffect(effect)
    # Re-tint when theme flips.
    theme.add_theme_listener(lambda _p: effect.setColor(_shadow_color()))


def _shadow_color() -> QColor:
    p = tokens.active()
    if p.name == "dark":
        return QColor(0, 0, 0, 140)
    return QColor(15, 23, 42, 30)  # slate, alpha 30
