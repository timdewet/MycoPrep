"""Theme builder + system-theme follower.

`apply_theme(app)` installs a token-driven QSS and attaches a listener to
`QGuiApplication.styleHints().colorSchemeChanged` so the UI re-skins when the
user toggles macOS Appearance (or the equivalent on Win/Linux).

`set_theme_override` lets the user force Light/Dark/Auto via the header pill.
"""

from __future__ import annotations

import tempfile
from enum import Enum
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication, QPalette, QColor
from PyQt6.QtWidgets import QApplication

from . import tokens
from .tokens import DARK, LIGHT, Palette


class ThemeMode(str, Enum):
    AUTO = "auto"
    LIGHT = "light"
    DARK = "dark"


_CHECK_SVG_LIGHT = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" '
    'viewBox="0 0 16 16">'
    '<path d="M3.5 8.3 L6.5 11.2 L12.5 5.0" '
    'fill="none" stroke="white" stroke-opacity="0.95" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)


def _arrow_svg(stroke: str, direction: str) -> str:
    """Generic chevron arrow. direction in {down, up}."""
    if direction == "up":
        d = "M3 9 L8 4 L13 9"
    elif direction == "down":
        d = "M3 6 L8 11 L13 6"
    else:
        d = "M3 6 L8 11 L13 6"
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" '
        'viewBox="0 0 16 16">'
        f'<path d="{d}" fill="none" stroke="{stroke}" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>'
    )


def _write_assets(p) -> dict:
    """Generate cached SVG paths for the active palette."""
    cache_dir = Path(tempfile.gettempdir()) / "mycoprep" / p.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    paths["check"] = str(cache_dir / "check.svg")
    Path(paths["check"]).write_text(_CHECK_SVG_LIGHT)
    paths["chevron-down"] = str(cache_dir / "chev-down.svg")
    Path(paths["chevron-down"]).write_text(_arrow_svg(p.text_muted, "down"))
    paths["chevron-up"] = str(cache_dir / "chev-up.svg")
    Path(paths["chevron-up"]).write_text(_arrow_svg(p.text_muted, "up"))
    return {k: Path(v).as_posix() for k, v in paths.items()}


_mode: ThemeMode = ThemeMode.LIGHT
_listeners: list[Callable[[Palette], None]] = []


def add_theme_listener(callback: Callable[[Palette], None]) -> None:
    """Register a callback fired whenever the active palette changes.

    Used by widgets that paint manually (icons, plate map) and need to
    refresh when the theme flips.
    """
    _listeners.append(callback)


def _detect_system() -> Palette:
    try:
        scheme = QGuiApplication.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return DARK
    except Exception:  # noqa: BLE001
        pass
    return LIGHT


def current_palette() -> Palette:
    if _mode is ThemeMode.LIGHT:
        return LIGHT
    if _mode is ThemeMode.DARK:
        return DARK
    return _detect_system()


def build_qss(p: Palette) -> str:
    assets = _write_assets(p)
    check_url = assets["check"]
    chev_down = assets["chevron-down"]
    chev_up = assets["chevron-up"]
    return f"""
* {{
    font-family: {tokens.FONT_FAMILY};
    font-size: {tokens.FS_BODY}px;
    color: {p.text};
}}

QMainWindow, QWidget {{ background-color: {p.bg}; }}
QLabel {{ background: transparent; }}
QToolTip {{
    background: {p.surface};
    color: {p.text};
    border: 1px solid {p.border};
    padding: 4px 8px;
}}

/* ── Cards (group boxes) ─────────────────────────────────────────── */
QGroupBox {{
    font-weight: {tokens.FW_SEMIBOLD};
    font-size: {tokens.FS_H3}px;
    color: {p.text};
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: {tokens.R_LG}px;
    /* margin-top reserves room above the border for the title PLUS a
     * breathing gap between the title baseline and the border line. */
    margin-top: {tokens.FS_H3 + 8}px;
    padding: {tokens.S5}px {tokens.S5}px {tokens.S5}px {tokens.S5}px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: {tokens.S5}px;
    top: 0;
    padding: 0 {tokens.S3}px;
    background: transparent;
    color: {p.text};
}}

/* ── Inputs ───────────────────────────────────────────────────────── */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {p.surface};
    border: 1px solid {p.border_strong};
    border-radius: {tokens.R_SM}px;
    padding: 6px 8px;
    color: {p.text};
    selection-background-color: {p.selection};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {p.focus_ring};
}}
QLineEdit:read-only {{
    background: {p.surface_alt};
    color: {p.text_muted};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {p.text_subtle};
    background: {p.surface_alt};
    border-color: {p.border};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{
    image: url({chev_down});
    width: 12px; height: 12px;
    margin-right: 4px;
}}

QSpinBox, QDoubleSpinBox {{
    padding-right: 18px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px; height: 12px;
    border-left: 1px solid {p.border_strong};
    border-bottom: 1px solid {p.border_strong};
    background: {p.surface};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px; height: 12px;
    border-left: 1px solid {p.border_strong};
    background: {p.surface};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({chev_up}); width: 10px; height: 10px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({chev_down}); width: 10px; height: 10px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {p.surface_alt};
}}

QPushButton:focus, QCheckBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QLineEdit:focus, QComboBox:focus {{ outline: none; }}

/* ── Buttons ──────────────────────────────────────────────────────── */
QPushButton {{
    background: {p.surface};
    border: 1px solid {p.border_strong};
    border-radius: {tokens.R_MD}px;
    padding: 7px 14px;
    color: {p.text};
}}
QPushButton:hover  {{ background: {p.surface_alt}; border-color: {p.text_subtle}; }}
QPushButton:pressed{{ background: {p.surface_alt}; }}
QPushButton:disabled {{
    color: {p.text_subtle};
    background: {p.surface_alt};
    border-color: {p.border};
}}

QPushButton#primary {{
    background: {p.primary};
    border: 1px solid {p.primary};
    color: {p.on_primary};
    font-weight: {tokens.FW_BOLD};
    padding: 10px 24px;
    font-size: {tokens.FS_BODY}px;
}}
QPushButton#primary:hover    {{ background: {p.primary_hover};   border-color: {p.primary_hover}; }}
QPushButton#primary:pressed  {{ background: {p.primary_pressed}; border-color: {p.primary_pressed}; }}
QPushButton#primary:disabled {{
    background: {p.primary_disabled};
    border-color: {p.primary_disabled};
    color: {p.on_primary};
}}
QPushButton#primary:focus {{ border: 2px solid {p.focus_ring}; }}

QPushButton#danger {{
    background: {p.danger};
    border: 1px solid {p.danger};
    color: #ffffff;
    font-weight: {tokens.FW_SEMIBOLD};
    padding: 10px 20px;
}}
QPushButton#danger:hover {{ background: #b91c1c; border-color: #b91c1c; }}
QPushButton#danger:disabled {{
    background: rgba(220, 38, 38, 0.45);
    border-color: rgba(220, 38, 38, 0.45);
    color: rgba(255, 255, 255, 0.75);
}}

QPushButton#ghost {{
    background: transparent;
    border: 1px solid transparent;
    color: {p.text_muted};
    padding: 5px 10px;
}}
QPushButton#ghost:hover {{ background: {p.surface_alt}; color: {p.text}; }}

/* ── Sidebar items ────────────────────────────────────────────────── */
QPushButton#navItem {{
    background: transparent;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 0;
    padding: 0;
    color: {p.text_muted};
    text-align: left;
    font-size: {tokens.FS_BODY}px;
}}
QPushButton#navItem:hover {{
    background: {p.surface};
    color: {p.text};
}}
QPushButton#navItem:checked {{
    background: {p.surface};
    color: {p.text};
    font-weight: {tokens.FW_SEMIBOLD};
    border-left: 3px solid {p.primary};
}}
QPushButton#navItem:disabled {{ color: {p.text_subtle}; }}

QFrame#sidebar {{ background: {p.surface_alt}; border: none; }}
QFrame#sidebarDivider {{
    background: {p.border};
    border: none;
    max-width: 1px;
    min-width: 1px;
}}
QFrame#header {{ background: {p.surface}; border: none; }}
QFrame#headerDivider {{ background: {p.border}; min-height: 1px; max-height: 1px; }}

QFrame#updateBanner {{
    background: {p.surface_alt};
    border: none;
    border-bottom: 1px solid {p.border};
}}
QLabel#updateBannerMsg {{ color: {p.text}; font-weight: {tokens.FW_MEDIUM}; }}
QPushButton#updateBannerView {{
    background: {p.primary};
    border: 1px solid {p.primary};
    color: {p.on_primary};
    font-weight: {tokens.FW_SEMIBOLD};
    padding: 5px 14px;
}}
QPushButton#updateBannerView:hover {{
    background: {p.primary_hover};
    border-color: {p.primary_hover};
}}
QPushButton#updateBannerDismiss {{
    background: transparent;
    border: 1px solid transparent;
    color: {p.text_muted};
    padding: 5px 10px;
}}
QPushButton#updateBannerDismiss:hover {{ color: {p.text}; }}

QLabel#brand {{
    font-size: {tokens.FS_H2}px;
    font-weight: {tokens.FW_SEMIBOLD};
    color: {p.text};
}}
QLabel#breadcrumb {{ color: {p.text_muted}; }}
QLabel#brandName {{
    font-size: {tokens.FS_H2}px;
    font-weight: {tokens.FW_SEMIBOLD};
    color: {p.text};
}}
QLabel#h1 {{ font-size: {tokens.FS_H1}px; font-weight: {tokens.FW_BOLD}; color: {p.text}; }}
QLabel#h2 {{ font-size: {tokens.FS_H2}px; font-weight: {tokens.FW_SEMIBOLD}; color: {p.text}; }}
QLabel#h3, QLabel#sectionHeader {{
    font-size: {tokens.FS_H3}px;
    font-weight: {tokens.FW_SEMIBOLD};
    color: {p.text};
}}
QLabel#caption {{ font-size: {tokens.FS_CAPTION}px; color: {p.text_subtle}; }}
QLabel#muted   {{ color: {p.text_muted}; }}

/* Theme pill in header */
QPushButton#themePill {{
    background: {p.surface_alt};
    border: 1px solid {p.border};
    border-radius: {tokens.R_PILL}px;
    padding: 4px 12px;
    color: {p.text_muted};
    font-size: {tokens.FS_CAPTION}px;
}}
QPushButton#themePill:hover {{ color: {p.text}; }}

/* Segmented control (Input mode picker) */
QPushButton#segLeft, QPushButton#segMid, QPushButton#segRight {{
    background: {p.surface};
    border: 1px solid {p.border_strong};
    color: {p.text_muted};
    padding: 7px 16px;
    font-weight: {tokens.FW_MEDIUM};
}}
QPushButton#segLeft  {{ border-top-left-radius: {tokens.R_MD}px; border-bottom-left-radius: {tokens.R_MD}px; border-right: none; }}
QPushButton#segMid   {{ border-right: none; }}
QPushButton#segRight {{ border-top-right-radius: {tokens.R_MD}px; border-bottom-right-radius: {tokens.R_MD}px; }}
QPushButton#segLeft:checked, QPushButton#segMid:checked, QPushButton#segRight:checked {{
    background: {p.primary};
    border-color: {p.primary};
    color: {p.on_primary};
}}
QPushButton#segLeft:hover:!checked, QPushButton#segMid:hover:!checked, QPushButton#segRight:hover:!checked {{
    color: {p.text};
    background: {p.surface_alt};
}}

/* ── Checkboxes ───────────────────────────────────────────────────── */
QCheckBox {{ spacing: 7px; padding: 2px; color: {p.text}; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid {p.border_strong}; background: {p.surface};
}}
QCheckBox::indicator:hover {{ border: 1px solid {p.primary}; }}
QCheckBox::indicator:checked {{
    background: {p.primary};
    border-color: {p.primary};
    image: url({check_url});
}}
QCheckBox::indicator:checked:hover {{
    background: {p.primary_hover};
    border-color: {p.primary_hover};
}}

QRadioButton {{ spacing: 7px; padding: 2px; color: {p.text}; }}
QRadioButton::indicator {{
    width: 16px; height: 16px; border-radius: 8px;
    border: 1px solid {p.border_strong}; background: {p.surface};
}}
QRadioButton::indicator:checked {{
    background: {p.primary};
    border: 4px solid {p.primary};
}}

/* ── Progress bar ─────────────────────────────────────────────────── */
QProgressBar {{
    border: 1px solid {p.border};
    border-radius: {tokens.R_SM}px;
    background: {p.surface_alt};
    text-align: center;
    color: {p.text};
    height: 18px;
}}
QProgressBar::chunk {{
    background: {p.primary};
    border-radius: 3px;
}}

/* ── Tabs (used by some embedded widgets) ─────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {p.border};
    border-radius: {tokens.R_MD}px;
    background: {p.surface};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    padding: 8px 18px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-bottom: none;
    border-top-left-radius: {tokens.R_MD}px;
    border-top-right-radius: {tokens.R_MD}px;
    color: {p.text_muted};
}}
QTabBar::tab:selected {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-bottom: 1px solid {p.surface};
    color: {p.text};
    font-weight: {tokens.FW_SEMIBOLD};
}}
QTabBar::tab:hover:!selected {{ color: {p.text}; }}

/* ── Log (dark, monospace, regardless of theme) ──────────────────── */
QPlainTextEdit {{
    font-family: {tokens.FONT_FAMILY_MONO};
    font-size: {tokens.FS_MONO}px;
    background: {p.log_bg};
    color: {p.log_text};
    border: 1px solid {p.border};
    border-radius: {tokens.R_SM}px;
    padding: {tokens.S2}px;
    selection-background-color: {p.selection};
}}

/* ── Status bar ───────────────────────────────────────────────────── */
QStatusBar {{
    background: {p.surface};
    border-top: 1px solid {p.border};
    color: {p.text_muted};
}}
QStatusBar::item {{ border: none; }}

/* ── Scroll bars ─────────────────────────────────────────────────── */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {p.border_strong};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {p.text_subtle}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{
    background: {p.border_strong};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* Splitter handle: a thin visible line plus a few px of grab area on
 * either side via padding so the user can actually drag it. Hover
 * highlights so the affordance is obvious. */
QSplitter::handle {{ background: {p.border}; }}
QSplitter::handle:horizontal {{ width: 5px; margin: 0 -2px; }}
QSplitter::handle:vertical   {{ height: 5px; margin: -2px 0; }}
QSplitter::handle:hover {{ background: {p.text_subtle}; }}

QFrame#card {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: {tokens.R_LG}px;
}}
QFrame#statusDot {{
    border-radius: 5px;
    min-width: 10px; max-width: 10px;
    min-height: 10px; max-height: 10px;
}}
"""


def _apply_qpalette(app: QApplication, p: Palette) -> None:
    """Best-effort QPalette so non-styled native widgets blend in."""
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor(p.bg))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(p.text))
    pal.setColor(QPalette.ColorRole.Base, QColor(p.surface))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(p.surface_alt))
    pal.setColor(QPalette.ColorRole.Text, QColor(p.text))
    pal.setColor(QPalette.ColorRole.Button, QColor(p.surface))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(p.text))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(p.primary))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(p.on_primary))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(p.surface))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(p.text))
    app.setPalette(pal)


def _refresh(app: QApplication) -> None:
    p = current_palette()
    tokens.set_active(p)
    _apply_qpalette(app, p)
    app.setStyleSheet(build_qss(p))
    for cb in list(_listeners):
        try:
            cb(p)
        except Exception:  # noqa: BLE001
            pass


def set_theme_override(mode: ThemeMode) -> None:
    """Manually force a theme; pass AUTO to follow the system."""
    global _mode
    _mode = mode
    app = QApplication.instance()
    if isinstance(app, QApplication):
        _refresh(app)


def theme_mode() -> ThemeMode:
    return _mode


def apply_theme(app: QApplication) -> None:
    """Install the theme + hook the system colorScheme signal."""
    _refresh(app)
    try:
        hints = QGuiApplication.styleHints()
        hints.colorSchemeChanged.connect(lambda _scheme: _refresh(app))
    except Exception:  # noqa: BLE001
        # Pre-Qt 6.5 — system follow won't work but manual override still does.
        pass
