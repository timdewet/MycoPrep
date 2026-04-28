"""Global Qt stylesheet — applied once in app.py."""

QSS = """
* {
    font-family: -apple-system, "SF Pro Text", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 13px;
    color: #1f2329;
}

QMainWindow, QWidget {
    background-color: #fafbfc;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #d8dde3;
    border-radius: 6px;
    background: #ffffff;
    top: -1px;
}
QTabBar::tab {
    background: transparent;
    padding: 8px 18px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #5b6571;
}
QTabBar::tab:selected {
    background: #ffffff;
    border: 1px solid #d8dde3;
    border-bottom: 1px solid #ffffff;
    color: #1f2329;
    font-weight: 600;
}
QTabBar::tab:hover:!selected { color: #1f2329; }

/* ── Group boxes ─────────────────────────────────────────────────────── */
QGroupBox {
    font-weight: 600;
    color: #3a424d;
    border: 1px solid #e2e6eb;
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    background: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    background: #fafbfc;
}

/* ── Inputs ──────────────────────────────────────────────────────────── */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background: #ffffff;
    border: 1px solid #d8dde3;
    border-radius: 4px;
    padding: 5px 7px;
    selection-background-color: #cfe3ff;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPlainTextEdit:focus {
    border: 1px solid #4a90e2;
}
QLineEdit:read-only { background: #f4f6f8; color: #5b6571; }

/* ── Buttons ─────────────────────────────────────────────────────────── */
QPushButton {
    background: #ffffff;
    border: 1px solid #ccd2d9;
    border-radius: 5px;
    padding: 6px 14px;
    color: #1f2329;
}
QPushButton:hover { background: #f0f3f6; border-color: #b2bac4; }
QPushButton:pressed { background: #e3e8ed; }
QPushButton:disabled { color: #a8aeb6; background: #f4f6f8; border-color: #e2e6eb; }

QPushButton#primary {
    background: #2563eb;
    border: 1px solid #1d4fc4;
    color: #ffffff;
    font-weight: 600;
    padding: 8px 22px;
}
QPushButton#primary:hover  { background: #1d4fc4; }
QPushButton#primary:pressed{ background: #173f9c; }
QPushButton#primary:disabled { background: #b8c8e8; border-color: #b8c8e8; color: #ffffff; }

/* ── Checkboxes ──────────────────────────────────────────────────────── */
QCheckBox { spacing: 7px; padding: 2px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px;
                       border: 1px solid #b8bfc7; background: #ffffff; }
QCheckBox::indicator:hover { border: 1px solid #2563eb; }
QCheckBox::indicator:checked {
    background: #2563eb;
    border-color: #2563eb;
    image: url(CHECK_ICON);
}
QCheckBox::indicator:checked:hover { background: #1d4fc4; border-color: #1d4fc4; }

/* ── Progress bar ────────────────────────────────────────────────────── */
QProgressBar {
    border: 1px solid #d8dde3;
    border-radius: 4px;
    background: #f0f3f6;
    text-align: center;
    color: #1f2329;
    height: 18px;
}
QProgressBar::chunk {
    background: #2563eb;
    border-radius: 3px;
}

/* ── Scroll area / log ───────────────────────────────────────────────── */
QPlainTextEdit {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px;
    background: #1f2329;
    color: #e6e8eb;
    border: 1px solid #2b313a;
}

QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #c8ced6; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #a8b0ba; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #c8ced6; border-radius: 5px; min-width: 30px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

QLabel#sectionHeader { font-size: 14px; font-weight: 600; color: #3a424d; }
QLabel#muted        { color: #6b7280; }
"""
