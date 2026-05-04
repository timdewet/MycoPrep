"""Feature library browser — table view of registered runs with filter,
remove, and import actions.

The library is a persistent store under ``~/.mycoprep/feature_library/`` (or
a user-chosen directory). Each registered run contributes a parquet copy
of its ``all_features.parquet`` plus a row in ``library.parquet`` with run
metadata (species, experiment type, cell counts, conditions, date).

This widget is the GUI surface for inspecting and managing that store.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mycoprep.core.extract.feature_library import (
    FeatureLibrary,
    derive_run_id_from_parquet,
)

from ..ui import tokens


class LibraryBrowser(QWidget):
    """Browse, filter, and manage the persistent feature library."""

    libraryChanged = pyqtSignal()

    _COLUMNS = [
        ("run_id", "Run ID"),
        ("species", "Species"),
        ("experiment_type", "Type"),
        ("n_cells", "# Cells"),
        ("n_conditions", "# Conds"),
        ("condition_labels", "Conditions"),
        ("source_czi", "Source CZI"),
        ("date_added", "Added"),
    ]

    def __init__(
        self,
        library_dir: Optional[Path] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._library_dir = Path(library_dir) if library_dir else None

        root = QVBoxLayout(self)
        root.setContentsMargins(tokens.S4, tokens.S4, tokens.S4, tokens.S4)
        root.setSpacing(tokens.S3)

        # ── Header: library dir path + actions ──────────────────────────
        header = QHBoxLayout()
        header.setSpacing(tokens.S2)
        self._dir_label = QLabel()
        self._dir_label.setStyleSheet(
            f"color: {tokens.active().text_subtle}; font-size: {tokens.FS_CAPTION}px;"
        )
        self._dir_label.setWordWrap(True)
        header.addWidget(self._dir_label, stretch=1)

        self._browse_btn = QPushButton("Change dir\u2026")
        self._browse_btn.clicked.connect(self._change_library_dir)
        header.addWidget(self._browse_btn)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        header.addWidget(self._refresh_btn)
        root.addLayout(header)

        # ── Filter bar ──────────────────────────────────────────────────
        filter_box = QGroupBox("Filter")
        filter_layout = QHBoxLayout(filter_box)
        filter_layout.setContentsMargins(tokens.S4, tokens.S4, tokens.S4, tokens.S3)
        filter_layout.setSpacing(tokens.S3)

        filter_layout.addWidget(QLabel("Species:"))
        self._species_filter = QComboBox()
        self._species_filter.setEditable(True)
        self._species_filter.setMinimumWidth(160)
        self._species_filter.currentTextChanged.connect(self._apply_filter)
        filter_layout.addWidget(self._species_filter)

        filter_layout.addWidget(QLabel("Type:"))
        self._type_filter = QComboBox()
        self._type_filter.addItems(["", "knockdown", "drug"])
        self._type_filter.currentTextChanged.connect(self._apply_filter)
        filter_layout.addWidget(self._type_filter)

        filter_layout.addWidget(QLabel("Search:"))
        self._search = QLineEdit()
        self._search.setPlaceholderText("Run ID or condition substring\u2026")
        self._search.textChanged.connect(self._apply_filter)
        filter_layout.addWidget(self._search, stretch=1)
        root.addWidget(filter_box)

        # ── Splitter: table on left, summary on right ──────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._table = QTableWidget()
        self._table.setColumnCount(len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels([h for _, h in self._COLUMNS])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        h = self._table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        h.setStretchLastSection(True)
        splitter.addWidget(self._table)

        # Right panel: summary
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        summary_layout.setContentsMargins(tokens.S3, 0, 0, 0)
        summary_layout.setSpacing(tokens.S2)

        summary_layout.addWidget(QLabel("<b>Summary</b>"))
        self._summary_label = QLabel("\u2014")
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._summary_label.setStyleSheet(
            f"font-family: monospace; font-size: {tokens.FS_CAPTION}px;"
        )
        summary_layout.addWidget(self._summary_label)
        summary_layout.addStretch(1)
        splitter.addWidget(summary_widget)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        # ── Action buttons ──────────────────────────────────────────────
        actions = QHBoxLayout()
        actions.setSpacing(tokens.S2)
        self._import_btn = QPushButton("Import existing parquet\u2026")
        self._import_btn.clicked.connect(self._import_parquet)
        actions.addWidget(self._import_btn)

        actions.addStretch(1)

        self._edit_btn = QPushButton("Edit selected\u2026")
        self._edit_btn.setToolTip("Change species or experiment type for the selected run.")
        self._edit_btn.clicked.connect(self._edit_selected)
        self._edit_btn.setEnabled(False)
        actions.addWidget(self._edit_btn)

        self._remove_btn = QPushButton("Remove selected")
        self._remove_btn.clicked.connect(self._remove_selected)
        self._remove_btn.setEnabled(False)
        actions.addWidget(self._remove_btn)
        root.addLayout(actions)

        self._table.itemSelectionChanged.connect(self._update_action_enabled)

        # Initial load
        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_library_dir(self, path: Optional[Path]) -> None:
        self._library_dir = Path(path) if path else None
        self.refresh()

    def refresh(self) -> None:
        """Reload the library index and repopulate the table."""
        lib = FeatureLibrary(self._library_dir)
        self._dir_label.setText(f"<i>Library dir:</i> {lib.library_dir}")
        runs = lib.list_runs()
        self._all_runs = runs

        # Refresh species filter options
        prev_species = self._species_filter.currentText()
        self._species_filter.blockSignals(True)
        self._species_filter.clear()
        self._species_filter.addItem("")
        if not runs.empty and "species" in runs.columns:
            for sp in sorted(runs["species"].dropna().unique()):
                self._species_filter.addItem(str(sp))
        self._species_filter.setCurrentText(prev_species)
        self._species_filter.blockSignals(False)

        self._apply_filter()
        self._update_summary(lib)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_filter(self) -> None:
        runs = getattr(self, "_all_runs", None)
        if runs is None or runs.empty:
            self._table.setRowCount(0)
            return
        sp = self._species_filter.currentText().strip().lower()
        et = self._type_filter.currentText().strip().lower()
        q = self._search.text().strip().lower()

        df = runs
        if sp:
            df = df[df["species"].astype(str).str.lower() == sp]
        if et:
            df = df[df["experiment_type"].astype(str).str.lower() == et]
        if q:
            mask = df["run_id"].astype(str).str.lower().str.contains(q, na=False)
            if "condition_labels" in df.columns:
                mask |= df["condition_labels"].astype(str).str.lower().str.contains(q, na=False)
            df = df[mask]

        self._populate_table(df)

    def _populate_table(self, df) -> None:
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            for c, (key, _label) in enumerate(self._COLUMNS):
                val = row.get(key, "")
                if key == "date_added":
                    val = _format_date(val)
                if key in ("n_cells", "n_conditions"):
                    item = QTableWidgetItem()
                    try:
                        item.setData(Qt.ItemDataRole.DisplayRole, int(val))
                    except (ValueError, TypeError):
                        item.setText(str(val))
                else:
                    item = QTableWidgetItem(str(val))
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()
        self._table.setSortingEnabled(True)

    def _update_summary(self, lib: FeatureLibrary) -> None:
        s = lib.summary()
        if s.empty:
            self._summary_label.setText("Library is empty.")
            return
        lines = [f"{int(r['n_runs'])} run(s), {int(r['total_cells']):,} cells"
                 f"\n  {r['species']} / {r['experiment_type']}"
                 for _, r in s.iterrows()]
        total_runs = int(s["n_runs"].sum())
        total_cells = int(s["total_cells"].sum())
        header = f"<b>{total_runs}</b> run(s), <b>{total_cells:,}</b> cells\n\n"
        self._summary_label.setText(header + "\n".join(lines))

    def _update_action_enabled(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        self._remove_btn.setEnabled(bool(rows))
        self._edit_btn.setEnabled(len(rows) == 1)

    def _selected_run_ids(self) -> list[str]:
        rows = self._table.selectionModel().selectedRows()
        ids: list[str] = []
        for idx in rows:
            item = self._table.item(idx.row(), 0)
            if item is not None:
                ids.append(item.text())
        return ids

    def _edit_selected(self) -> None:
        run_ids = self._selected_run_ids()
        if len(run_ids) != 1:
            return
        run_id = run_ids[0]
        # Look up current values from the cached index.
        runs = getattr(self, "_all_runs", None)
        if runs is None or runs.empty:
            return
        match = runs[runs["run_id"] == run_id]
        if match.empty:
            return
        current_species = str(match.iloc[0].get("species", ""))
        current_type = str(match.iloc[0].get("experiment_type", "knockdown"))

        species_options = ["M. tuberculosis", "M. smegmatis"]
        try:
            sp_idx = species_options.index(current_species)
        except ValueError:
            sp_idx = 0
        species, ok = QInputDialog.getItem(
            self, "Edit species", f"Species for '{run_id}':",
            species_options, sp_idx, False,
        )
        if not ok:
            return

        type_options = ["knockdown", "drug"]
        try:
            t_idx = type_options.index(current_type)
        except ValueError:
            t_idx = 0
        exp_type, ok = QInputDialog.getItem(
            self, "Edit experiment type", f"Type for '{run_id}':",
            type_options, t_idx, False,
        )
        if not ok:
            return

        lib = FeatureLibrary(self._library_dir)
        lib.update_run(run_id, species=species, experiment_type=exp_type)
        self.refresh()
        self.libraryChanged.emit()

    def _remove_selected(self) -> None:
        run_ids = self._selected_run_ids()
        if not run_ids:
            return
        msg = f"Remove {len(run_ids)} run(s) from the library?\n\n" + "\n".join(
            f"  \u2022 {rid}" for rid in run_ids[:8]
        )
        if len(run_ids) > 8:
            msg += f"\n  \u2026 and {len(run_ids) - 8} more"
        reply = QMessageBox.question(
            self, "Remove from library", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        lib = FeatureLibrary(self._library_dir)
        for rid in run_ids:
            lib.remove_run(rid)
        self.refresh()
        self.libraryChanged.emit()

    def _import_parquet(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select all_features.parquet", "",
            "Parquet files (*.parquet);;All files (*)",
        )
        if not path:
            return

        species_options = ["M. tuberculosis", "M. smegmatis"]
        current = self._species_filter.currentText()
        try:
            current_idx = species_options.index(current)
        except ValueError:
            current_idx = 0
        species, ok = QInputDialog.getItem(
            self, "Species", "Species:",
            species_options, current_idx, False,
        )
        if not ok:
            return

        exp_type, ok = QInputDialog.getItem(
            self, "Experiment type", "Type:",
            ["knockdown", "drug"], 0, False,
        )
        if not ok:
            return

        run_id, ok = QInputDialog.getText(
            self, "Run ID", "Run ID:",
            text=derive_run_id_from_parquet(Path(path)),
        )
        if not ok or not run_id.strip():
            return

        try:
            lib = FeatureLibrary(self._library_dir)
            lib.register_run(
                run_id=run_id.strip(),
                features_parquet=Path(path),
                species=species.strip(),
                experiment_type=exp_type,
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Import failed", str(e))
            return

        self.refresh()
        self.libraryChanged.emit()

    def _change_library_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select library directory",
            str(self._library_dir) if self._library_dir else "",
        )
        if d:
            self.set_library_dir(Path(d))


def _format_date(val) -> str:
    if not val:
        return ""
    s = str(val)
    try:
        # ISO format with microseconds or seconds
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return s
