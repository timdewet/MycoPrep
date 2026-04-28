"""Qt application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from ._resources import resource_root
from .main_window import MainWindow
from .ui.theme import apply_theme


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("MycoPrep")
    app.setApplicationDisplayName("MycoPrep")
    logo = resource_root() / "logo" / "logo.svg"
    if logo.exists():
        app.setWindowIcon(QIcon(str(logo)))
    apply_theme(app)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
