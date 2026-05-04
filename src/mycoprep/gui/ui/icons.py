"""Theme-aware icon helper, backed by qtawesome (Phosphor / MDI).

Falls back to a transparent QIcon when qtawesome isn't installed so the app
still launches; callers can still set fallback text glyphs as button labels.
"""

from __future__ import annotations

from typing import Literal

from PyQt6.QtGui import QIcon

from . import tokens, theme

try:
    import qtawesome as qta  # type: ignore
    _HAS_QTA = True
except Exception:  # noqa: BLE001
    qta = None  # type: ignore
    _HAS_QTA = False


Role = Literal["text", "muted", "primary", "danger", "success", "warning", "on_primary"]


# Map our semantic icon names to qtawesome glyph IDs.
# Using MDI (always available with qtawesome) keeps things universal.
ICONS: dict[str, str] = {
    "input":         "mdi.import",
    "plate":         "mdi.view-grid",
    "focus":         "mdi.target",
    "segment":       "mdi.shape",
    "features":      "mdi.table-large",
    "run":           "mdi.play-circle",
    "analysis":      "mdi.chart-scatter-plot",
    "play":          "mdi.play",
    "stop":          "mdi.stop",
    "folder":        "mdi.folder",
    "folder-open":   "mdi.folder-open",
    "browse":        "mdi.dots-horizontal",
    "preview":       "mdi.image-search",
    "model":         "mdi.brain",
    "label":         "mdi.tag",
    "copy":          "mdi.content-copy",
    "clear":         "mdi.broom",
    "follow":        "mdi.arrow-down-bold-circle-outline",
    "sun":           "mdi.weather-sunny",
    "moon":          "mdi.weather-night",
    "auto":          "mdi.theme-light-dark",
    "gpu":           "mdi.memory",
    "warning":       "mdi.alert",
    "check":         "mdi.check",
    "check-circle":  "mdi.check-circle",
    "cog":           "mdi.cog",
    # Status-dot replacements
    "status-idle":      "mdi.circle-outline",
    "status-ready":     "mdi.circle-medium",
    "status-running":   "mdi.timer-sand",
    "status-done":      "mdi.check-circle",
    "status-error":     "mdi.alert-circle",
    "status-disabled":  "mdi.minus-circle-outline",
    # Misc
    "expand":   "mdi.chevron-down",
    "collapse": "mdi.chevron-up",
    "info":     "mdi.information-outline",
}


def _resolve_color(role: Role) -> str:
    p = tokens.active()
    return {
        "text":       p.text,
        "muted":      p.text_muted,
        "primary":    p.primary,
        "danger":     p.danger,
        "success":    p.success,
        "warning":    p.warning,
        "on_primary": p.on_primary,
    }[role]


_warned: set[str] = set()


def icon(name: str, role: Role = "text", disabled_role: Role | None = None) -> QIcon:
    """Return a QIcon for the given semantic name, coloured for the active palette.

    If `disabled_role` is given, the icon also carries a Disabled-mode tint so
    Qt auto-swaps the colour when the host widget is disabled.
    """
    if not _HAS_QTA:
        return QIcon()
    glyph = ICONS.get(name, name)
    try:
        kwargs = {"color": _resolve_color(role)}
        if disabled_role is not None:
            kwargs["color_disabled"] = _resolve_color(disabled_role)
        return qta.icon(glyph, **kwargs)
    except Exception as e:  # noqa: BLE001
        if glyph not in _warned:
            _warned.add(glyph)
            import sys
            print(f"[icons] qtawesome failed for {glyph!r}: {e}", file=sys.stderr)
        return QIcon()


def has_qtawesome() -> bool:
    return _HAS_QTA


# ---------------------------------------------------------------------------
# Theme refresh: when the palette flips, every cached qtawesome icon needs
# rebuilding. We keep a registry of (widget, slot) pairs and re-fire them.
# ---------------------------------------------------------------------------

_subscribers: list = []


def on_theme_change(callback) -> None:
    """Register a no-arg callback to re-set icons after a theme switch."""
    _subscribers.append(callback)


def _refresh_all(_palette) -> None:
    for cb in list(_subscribers):
        try:
            cb()
        except Exception:  # noqa: BLE001
            pass


theme.add_theme_listener(_refresh_all)
