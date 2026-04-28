"""Design tokens: spacing, type, radius, color palettes (light + dark).

Single source of truth for every visual value in the app. Widgets and the
theme builder pull from here; nothing else should hard-code a colour or
margin.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── Spacing scale (px) ──────────────────────────────────────────────────────
S1 = 4
S2 = 8
S3 = 12
S4 = 16
S5 = 20
S6 = 24
S7 = 32
S8 = 48


# ── Radius (px) ─────────────────────────────────────────────────────────────
R_SM = 4
R_MD = 6
R_LG = 10
R_PILL = 999


# ── Type scale ──────────────────────────────────────────────────────────────
FONT_FAMILY = '"SF Pro Text", "Helvetica Neue", "Segoe UI", "Inter", sans-serif'
FONT_FAMILY_MONO = '"SF Mono", Menlo, Consolas, monospace'

FS_CAPTION = 11
FS_BODY = 13
FS_H3 = 14
FS_H2 = 16
FS_H1 = 20
FS_MONO = 12

FW_REGULAR = 400
FW_MEDIUM = 500
FW_SEMIBOLD = 600
FW_BOLD = 700


# ── Sidebar / chrome geometry ───────────────────────────────────────────────
SIDEBAR_WIDTH = 224
HEADER_HEIGHT = 72


@dataclass(frozen=True)
class Palette:
    """Semantic color tokens. Two instances exist: LIGHT and DARK."""

    name: str  # "light" or "dark"

    bg: str
    surface: str
    surface_alt: str

    border: str
    border_strong: str

    text: str
    text_muted: str
    text_subtle: str

    primary: str
    primary_hover: str
    primary_pressed: str
    primary_disabled: str
    on_primary: str

    accent: str
    success: str
    warning: str
    danger: str

    focus_ring: str
    selection: str

    log_bg: str
    log_text: str
    log_warn: str
    log_error: str
    log_success: str

    # Plate-map well palette anchors (HSV saturation/value); hue is hashed.
    well_fill_s: int
    well_fill_v: int
    well_ring_s: int
    well_ring_v: int
    well_empty_fill: str
    well_empty_ring: str


LIGHT = Palette(
    name="light",
    bg="#fafbfc",
    surface="#ffffff",
    surface_alt="#f4f6f8",
    border="#e2e6eb",
    border_strong="#d8dde3",
    text="#1f2329",
    text_muted="#5b6571",
    text_subtle="#8b939d",
    primary="#2563eb",
    primary_hover="#1d4fc4",
    primary_pressed="#173f9c",
    primary_disabled="#b8c8e8",
    on_primary="#ffffff",
    accent="#7c3aed",
    success="#16a34a",
    warning="#d97706",
    danger="#dc2626",
    focus_ring="#4a90e2",
    selection="#cfe3ff",
    log_bg="#1f2329",
    log_text="#e6e8eb",
    log_warn="#fbbf24",
    log_error="#f87171",
    log_success="#86efac",
    well_fill_s=130,
    well_fill_v=235,
    well_ring_s=200,
    well_ring_v=200,
    well_empty_fill="#eef1f5",
    well_empty_ring="#c8ced6",
)


DARK = Palette(
    name="dark",
    bg="#0e1116",
    surface="#161b22",
    surface_alt="#1c222b",
    border="#2a313c",
    border_strong="#3a424d",
    text="#e6edf3",
    text_muted="#9ba6b3",
    text_subtle="#6e7681",
    primary="#4493f8",
    primary_hover="#5aa3ff",
    primary_pressed="#2f7ee0",
    primary_disabled="#26385a",
    on_primary="#ffffff",
    accent="#a78bfa",
    success="#4ade80",
    warning="#fbbf24",
    danger="#f87171",
    focus_ring="#4493f8",
    selection="#1f3a5f",
    log_bg="#0a0d12",
    log_text="#d0d7de",
    log_warn="#fbbf24",
    log_error="#f87171",
    log_success="#4ade80",
    well_fill_s=150,
    well_fill_v=170,
    well_ring_s=210,
    well_ring_v=220,
    well_empty_fill="#1c222b",
    well_empty_ring="#3a424d",
)


# Active palette is held in a small mutable container so widgets that
# resolve colors at paint time (not via QSS) can read the current one
# without each holding a ref. Theme.apply_theme writes to it.
class _Active:
    palette: Palette = LIGHT


def active() -> Palette:
    """Return the currently-applied palette."""
    return _Active.palette


def set_active(palette: Palette) -> None:
    _Active.palette = palette
