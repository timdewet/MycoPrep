"""Qt application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

# QtWebEngine requires either an early import or Qt.AA_ShareOpenGLContexts
# to be set before the QApplication is constructed. Set the attribute
# upfront so the embedded interactive plot in the Analysis page works
# regardless of import order downstream.
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

from ._resources import resource_root
from .main_window import MainWindow
from .ui.theme import apply_theme


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("MycoPrep")
    app.setApplicationDisplayName("MycoPrep")
    # Ensure closing the last window actually exits the process. Qt's
    # default is to keep the event loop alive while any QObject is
    # still attached — QWebEngineView (used by the Analysis panel)
    # spawns helper processes that occasionally outlive ``win.close()``
    # and leave the terminal hanging on Windows.
    app.setQuitOnLastWindowClosed(True)
    logo = resource_root() / "logo" / "logo.svg"
    if logo.exists():
        app.setWindowIcon(QIcon(str(logo)))
    apply_theme(app)
    win = MainWindow()
    win.show()
    rc = app.exec()
    # Force-quit any lingering Qt resources. ``os._exit`` bypasses
    # Python's atexit cleanup — safe here because the GUI is the
    # only thing this entry-point manages, and it's the documented
    # escape hatch for Qt apps that won't release the terminal when
    # WebEngine helper processes outlive the main loop.
    import os
    os._exit(rc)


if __name__ == "__main__":
    sys.exit(main())
