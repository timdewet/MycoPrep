"""Locate bundled resources (logo, icons, etc.) in both dev and frozen builds.

When the app is run from source, files like the logo live under
``<repo>/assets/``. When packaged with PyInstaller, the same files are
extracted into a temp directory whose path is exposed as
``sys._MEIPASS``. ``resource_root`` returns whichever applies, so call
sites can write ``resource_root() / "logo" / "logo.svg"`` and not care
which mode they're running in.
"""

from __future__ import annotations

import sys
from pathlib import Path


def resource_root() -> Path:
    """Directory holding bundled non-Python assets (logo, icons, ...)."""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
    # Dev mode: this file lives at src/mycoprep/gui/_resources.py;
    # repo root is three parents up, assets/ sits beside src/.
    return Path(__file__).resolve().parents[3] / "assets"
