"""Input panel: pick CZI(s) + output dir + channel config.

Three modes are available, switched via radio buttons at the top:

- **Single plate**  — one multi-position CZI with Zeiss well metadata.
                      The Plate-layout tab handles condition labelling.
- **Single file**   — one CZI without plate metadata; inline labels.
- **Bulk**          — many CZIs, each with its own labels (table editor).

Single-file and Bulk both feed the shared ``BulkLayout`` model, so the
runtime path is identical for them. Single-plate is the legacy flow and
its labels still come from the ``PlateLayout`` on the Plate-layout tab.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..pipeline.bulk_layout import COLUMNS as BULK_COLS
from ..pipeline.bulk_layout import BulkLayout


class InputMode(Enum):
    SINGLE_FILE = "single_file"
    SINGLE_PLATE = "single_plate"
    BULK = "bulk"


# ─────────────────────────────────────────────────────────────────────────────
# QAbstractTableModel for the BulkLayout DataFrame
# ─────────────────────────────────────────────────────────────────────────────

from PyQt6.QtCore import QAbstractTableModel, QModelIndex


class _BulkTableModel(QAbstractTableModel):
    """Two-way binding between a BulkLayout's DataFrame and a QTableView."""

    HEADERS = ["File", "Condition", "Reporter", "Mutant / drug", "Replica", "Notes"]
    EDITABLE_COLS = {1, 2, 3, 4, 5}     # everything except the path

    def __init__(self, layout: BulkLayout, parent=None) -> None:
        super().__init__(parent)
        self._layout = layout

    def set_layout(self, layout: BulkLayout) -> None:
        self.beginResetModel()
        self._layout = layout
        self.endResetModel()

    @property
    def layout(self) -> BulkLayout:
        return self._layout

    def rowCount(self, _parent=QModelIndex()) -> int:    # noqa: N802
        return len(self._layout.df)

    def columnCount(self, _parent=QModelIndex()) -> int:  # noqa: N802
        return len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return str(section + 1)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            value = self._layout.df.iloc[row][BULK_COLS[col]]
            if col == 0:    # show just the filename, not the full path
                return Path(str(value)).name if value else ""
            return "" if value is None else str(value)
        if role == Qt.ItemDataRole.ToolTipRole and col == 0:
            return str(self._layout.df.iloc[row]["czi_path"])
        return None

    def flags(self, index: QModelIndex):
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() in self.EDITABLE_COLS:
            return base | Qt.ItemFlag.ItemIsEditable
        return base

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        row, col = index.row(), index.column()
        if col not in self.EDITABLE_COLS:
            return False
        self._layout.df.iat[row, col] = str(value)
        self.dataChanged.emit(index, index, [role])
        return True

    def reset(self) -> None:
        self.beginResetModel()
        self.endResetModel()


# ─────────────────────────────────────────────────────────────────────────────
# InputPanel
# ─────────────────────────────────────────────────────────────────────────────

class InputPanel(QWidget):
    """Combined source + channels editor with a mode toggle at the top."""

    cziSelected = pyqtSignal(Path)        # fires for single-plate AND single-file
    cziPathsSelected = pyqtSignal(list)   # full list of plate CZIs (single-plate only)
    outputDirSelected = pyqtSignal(Path)
    channelsChanged = pyqtSignal()
    modeChanged = pyqtSignal(object)      # InputMode
    resetRequested = pyqtSignal()         # user clicked the Reset button

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Single-plate state. ``_czi_path`` is the FIRST plate CZI (kept for
        # legacy callers); ``_czi_paths`` is the full list when the user
        # has added multiple CZIs that together make up one logical plate.
        self._czi_path: Path | None = None
        self._czi_paths: list[Path] = []
        # Shared output / channels state
        self._out_dir: Path | None = None
        self._channel_name_edits: list[QLineEdit] = []
        self._original_channel_names: list[str] = []
        # Bulk + single-file state
        self._bulk_layout = BulkLayout.empty()
        self._bulk_model = _BulkTableModel(self._bulk_layout)
        # Active mode (default to plate to preserve legacy startup behaviour)
        self._mode = InputMode.SINGLE_FILE

        self._build_ui()

    # ────────────────────────────────────────────────────────────────── build

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(20)

        # ── Mode selector (segmented control) ───────────────────────────
        mode_box = QGroupBox("Input mode")
        mode_row = QHBoxLayout(mode_box)
        mode_row.setContentsMargins(16, 20, 16, 16)
        mode_row.setSpacing(0)

        self._radio_single_file = QPushButton("Single file")
        self._radio_single_file.setObjectName("segLeft")
        self._radio_bulk = QPushButton("Multiple files")
        self._radio_bulk.setObjectName("segMid")
        self._radio_single_plate = QPushButton("Single plate")
        self._radio_single_plate.setObjectName("segRight")

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for b in (self._radio_single_file, self._radio_bulk, self._radio_single_plate):
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self._mode_group.addButton(b)
            mode_row.addWidget(b)
        self._radio_single_file.setChecked(True)
        mode_row.addStretch(1)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setObjectName("ghost")
        self._reset_btn.setToolTip(
            "Clear the current CZI, output directory, channel labels, and "
            "any per-file labels — start a new run from scratch."
        )
        self._reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        mode_row.addWidget(self._reset_btn)
        outer.addWidget(mode_box)

        self._radio_single_file.toggled.connect(
            lambda on: self._on_mode_changed(InputMode.SINGLE_FILE) if on else None
        )
        self._radio_single_plate.toggled.connect(
            lambda on: self._on_mode_changed(InputMode.SINGLE_PLATE) if on else None
        )
        self._radio_bulk.toggled.connect(
            lambda on: self._on_mode_changed(InputMode.BULK) if on else None
        )

        # ── Stacked source pages ────────────────────────────────────────
        # Custom QStackedWidget that reports the size of the *current* page
        # rather than the union of all pages, so switching modes doesn't
        # leave a tall empty gap below short pages (e.g. Single plate vs Bulk).
        class _AdaptiveStack(QStackedWidget):
            def sizeHint(self):  # type: ignore[override]
                cur = self.currentWidget()
                return cur.sizeHint() if cur is not None else super().sizeHint()

            def minimumSizeHint(self):  # type: ignore[override]
                cur = self.currentWidget()
                return cur.minimumSizeHint() if cur is not None else super().minimumSizeHint()

        from PyQt6.QtWidgets import QSizePolicy as _QSP
        self._stack = _AdaptiveStack()
        self._stack.addWidget(self._build_single_plate_page())  # idx 0
        self._stack.addWidget(self._build_single_file_page())   # idx 1
        self._stack.addWidget(self._build_bulk_page())          # idx 2
        self._stack.setSizePolicy(_QSP.Policy.Preferred, _QSP.Policy.Maximum)
        # Re-layout the panel when the page changes so the new (taller or
        # shorter) page is reflected.
        self._stack.currentChanged.connect(lambda _i: self._stack.updateGeometry())
        outer.addWidget(self._stack)

        self._info_label = QLabel("")
        self._info_label.setObjectName("muted")
        self._info_label.setWordWrap(True)
        self._info_label.setVisible(False)
        outer.addWidget(self._info_label)

        # ── Output dir (shared by all modes) ────────────────────────────
        out_box = QGroupBox("Output directory")
        out_l = QHBoxLayout(out_box)
        out_l.setContentsMargins(16, 20, 16, 16)
        out_l.setSpacing(8)
        self._out_edit = QLineEdit()
        self._out_edit.setReadOnly(True)
        self._out_edit.setPlaceholderText("(none selected — click Browse)")
        out_btn = QPushButton("Browse")
        out_btn.clicked.connect(self._pick_out)
        out_l.addWidget(self._out_edit, stretch=1)
        out_l.addWidget(out_btn)
        outer.addWidget(out_box)

        # ── Channels group (shared) ─────────────────────────────────────
        self._channels_box = QGroupBox("Channels")
        self._channels_form = QFormLayout(self._channels_box)
        self._channels_form.setContentsMargins(16, 20, 16, 16)
        self._channels_form.setHorizontalSpacing(14)
        self._channels_form.setVerticalSpacing(8)

        self._phase_combo = QComboBox()
        self._phase_combo.addItem("auto (detect by skewness)", userData=None)
        self._phase_combo.currentIndexChanged.connect(lambda _i: self.channelsChanged.emit())
        self._channels_form.addRow("Phase channel:", self._phase_combo)

        outer.addWidget(self._channels_box)
        outer.addStretch(1)

        # Initial visibility
        self._sync_stack_to_mode()

    # ─────────────────────────────────────────────────────────── mode pages

    def _build_single_plate_page(self) -> QWidget:
        from PyQt6.QtWidgets import QListWidget, QAbstractItemView

        page = QWidget()
        v = QVBoxLayout(page); v.setContentsMargins(0, 0, 0, 0)
        box = QGroupBox("Plate CZIs")
        v_box = QVBoxLayout(box)
        v_box.setContentsMargins(16, 20, 16, 16)
        v_box.setSpacing(8)

        # Helper text — surfaces the multi-CZI capability for users who used
        # to single-pick a CZI here.
        hint = QLabel(
            "Select one CZI for a normal single-plate run, or add multiple "
            "CZIs that together make up one plate (e.g. row-band split). "
            "Wells from each CZI merge into one layout. The first CZI's "
            "channels apply to all subsequent CZIs."
        )
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        v_box.addWidget(hint)

        # CZI list + buttons.
        self._plate_czi_list = QListWidget()
        self._plate_czi_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection,
        )
        self._plate_czi_list.setMinimumHeight(70)
        self._plate_czi_list.setMaximumHeight(120)
        v_box.addWidget(self._plate_czi_list)

        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        add_btn = QPushButton("Add CZI(s)…")
        add_btn.clicked.connect(self._pick_plate_czi)
        rm_btn = QPushButton("Remove selected")
        rm_btn.clicked.connect(self._remove_selected_plate_czis)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(rm_btn)
        btn_row.addStretch(1)
        v_box.addLayout(btn_row)

        v.addWidget(box)
        return page

    def _build_single_file_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page); v.setContentsMargins(0, 0, 0, 0)
        box = QGroupBox("Single CZI file")
        form = QFormLayout(box)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)

        row = QHBoxLayout(); row.setSpacing(8)
        self._sf_czi_edit = QLineEdit()
        self._sf_czi_edit.setReadOnly(True)
        self._sf_czi_edit.setPlaceholderText("(none selected)")
        btn = QPushButton("Browse")
        btn.clicked.connect(self._pick_single_file_czi)
        row.addWidget(self._sf_czi_edit); row.addWidget(btn)
        wrap = QWidget(); wrap.setLayout(row)
        form.addRow("Input CZI:", wrap)

        # Inline labels — same fields as the bulk row
        self._sf_condition = QLineEdit()
        self._sf_reporter  = QLineEdit()
        self._sf_mutant    = QLineEdit()
        self._sf_replica   = QLineEdit()
        self._sf_notes     = QLineEdit()

        for w in (self._sf_condition, self._sf_reporter, self._sf_mutant,
                  self._sf_replica, self._sf_notes):
            w.editingFinished.connect(self._sync_single_file_to_layout)

        form.addRow("Condition:",     self._sf_condition)
        form.addRow("Reporter:",      self._sf_reporter)
        form.addRow("Mutant / drug:", self._sf_mutant)
        form.addRow("Replica:",       self._sf_replica)
        form.addRow("Notes:",         self._sf_notes)

        v.addWidget(box)
        return page

    def _build_bulk_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page); v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        box = QGroupBox("Bulk CZI batch")
        bv = QVBoxLayout(box)

        # Toolbar
        bar = QHBoxLayout(); bar.setSpacing(8)
        add_files_btn = QPushButton("Add files");    add_files_btn.clicked.connect(self._bulk_add_files)
        add_folder_btn = QPushButton("Add folder");  add_folder_btn.clicked.connect(self._bulk_add_folder)
        remove_btn = QPushButton("Remove selected"); remove_btn.clicked.connect(self._bulk_remove_selected)
        save_csv_btn = QPushButton("Save batch CSV");  save_csv_btn.clicked.connect(self._bulk_save_csv)
        load_csv_btn = QPushButton("Load batch CSV");  load_csv_btn.clicked.connect(self._bulk_load_csv)
        for b in (add_files_btn, add_folder_btn, remove_btn, save_csv_btn, load_csv_btn):
            bar.addWidget(b)
        bar.addStretch(1)
        bv.addLayout(bar)

        # Table
        self._bulk_table = QTableView()
        self._bulk_table.setModel(self._bulk_model)
        self._bulk_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._bulk_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        hdr = self._bulk_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr.setStretchLastSection(True)
        self._bulk_model.dataChanged.connect(lambda *_: self._refresh_bulk_status())
        self._bulk_model.modelReset.connect(self._refresh_bulk_status)
        bv.addWidget(self._bulk_table)

        self._bulk_status = QLabel("")
        self._bulk_status.setObjectName("muted")
        bv.addWidget(self._bulk_status)

        v.addWidget(box)
        return page

    # ─────────────────────────────────────────────────────────── mode events

    def _on_mode_changed(self, mode: InputMode) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        self._sync_stack_to_mode()
        # Refresh the channels/info display for the active source.
        self._refresh_info_and_channels()
        self.modeChanged.emit(mode)

    def _sync_stack_to_mode(self) -> None:
        idx = {InputMode.SINGLE_PLATE: 0,
               InputMode.SINGLE_FILE:  1,
               InputMode.BULK:         2}[self._mode]
        self._stack.setCurrentIndex(idx)

    # ─────────────────────────────────────────── single-plate / single-file

    def _pick_plate_czi(self) -> None:
        """Multi-select CZIs to add to the plate. The first CZI defines
        channel layout; subsequent CZIs must match (same channel count and
        names) or they're rejected with a clear error."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select one or more CZI files", "",
            "CZI files (*.czi);;All files (*)",
        )
        if not paths:
            return

        added: list[Path] = []
        rejected: list[tuple[str, str]] = []  # (filename, reason)
        existing_set = {p.resolve() for p in self._czi_paths}
        for raw in paths:
            p = Path(raw).resolve()
            if p in existing_set:
                rejected.append((p.name, "already in list"))
                continue
            ok, why = self._validate_plate_czi_compatibility(p)
            if not ok:
                rejected.append((p.name, why))
                continue
            self._czi_paths.append(p)
            existing_set.add(p)
            added.append(p)

        if added and self._czi_path is None:
            self._czi_path = self._czi_paths[0]

        self._refresh_plate_czi_list()
        self._refresh_info_and_channels()
        if self._czi_path is not None:
            # Keep the legacy single-CZI signal alive (channels box, focus
            # panel, segment panel still rely on it) AND emit the multi-CZI
            # list so the layout panel can repopulate from all CZIs.
            self.cziSelected.emit(self._czi_path)
        self.cziPathsSelected.emit(list(self._czi_paths))
        self.channelsChanged.emit()

        if rejected:
            QMessageBox.warning(
                self, "Some CZIs were not added",
                "\n".join(f"• {n} — {r}" for n, r in rejected),
            )

    def _remove_selected_plate_czis(self) -> None:
        """Remove the user-selected rows from the CZI list."""
        selected_rows = sorted(
            {self._plate_czi_list.row(it) for it in self._plate_czi_list.selectedItems()},
            reverse=True,
        )
        if not selected_rows:
            return
        for r in selected_rows:
            if 0 <= r < len(self._czi_paths):
                del self._czi_paths[r]
        self._czi_path = self._czi_paths[0] if self._czi_paths else None
        self._refresh_plate_czi_list()
        self._refresh_info_and_channels()
        if self._czi_path is not None:
            self.cziSelected.emit(self._czi_path)
        self.cziPathsSelected.emit(list(self._czi_paths))
        self.channelsChanged.emit()

    def _refresh_plate_czi_list(self) -> None:
        self._plate_czi_list.clear()
        for p in self._czi_paths:
            self._plate_czi_list.addItem(p.name)

    def _validate_plate_czi_compatibility(self, czi_path: Path) -> tuple[bool, str]:
        """Verify a candidate CZI matches the channel layout of the first.

        First-added CZI defines the contract: number of channels and channel
        names. Subsequent CZIs must match, or the multi-CZI plate would
        produce inconsistent per-well TIFFs downstream.
        """
        if not self._czi_paths:
            return True, ""
        try:
            from mycoprep.core.split_czi_plate import extract_channel_names
        except Exception:  # noqa: BLE001
            return True, ""
        try:
            ref_names = extract_channel_names(self._czi_paths[0]) or []
            new_names = extract_channel_names(czi_path) or []
        except Exception as e:  # noqa: BLE001
            return False, f"could not read channel metadata ({e})"
        if len(ref_names) != len(new_names):
            return False, (
                f"channel count {len(new_names)} does not match the first "
                f"CZI's {len(ref_names)}"
            )
        if ref_names and new_names and ref_names != new_names:
            return False, (
                f"channel names {new_names} do not match the first CZI's "
                f"{ref_names}"
            )
        return True, ""

    def _pick_single_file_czi(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a CZI file", "", "CZI files (*.czi);;All files (*)"
        )
        if not path:
            return
        self._sf_czi_edit.setText(path)
        self._sync_single_file_to_layout()
        # Single-file CZI also drives the same metadata-detection paths as
        # plate mode (channel names + pixel size), so emit cziSelected too.
        self._czi_path = Path(path)
        self._refresh_info_and_channels()
        self.cziSelected.emit(self._czi_path)
        self.channelsChanged.emit()

    def _sync_single_file_to_layout(self) -> None:
        """Single-file mode keeps the BulkLayout in sync (single row)."""
        path_str = self._sf_czi_edit.text().strip()
        if not path_str:
            self._bulk_layout = BulkLayout.empty()
        else:
            self._bulk_layout.set_single(
                Path(path_str),
                condition=self._sf_condition.text().strip(),
                reporter=self._sf_reporter.text().strip(),
                mutant_or_drug=self._sf_mutant.text().strip(),
                replica=self._sf_replica.text().strip(),
                notes=self._sf_notes.text().strip(),
            )
        self._bulk_model.set_layout(self._bulk_layout)

    # ────────────────────────────────────────────────────── bulk operations

    def _bulk_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add CZI files", "", "CZI files (*.czi);;All files (*)"
        )
        if not paths:
            return
        n = self._bulk_layout.add_files([Path(p) for p in paths])
        self._bulk_model.set_layout(self._bulk_layout)
        self._refresh_bulk_status(extra=f"Added {n} file(s).")

    def _bulk_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder of CZIs")
        if not folder:
            return
        n = self._bulk_layout.add_folder(Path(folder))
        self._bulk_model.set_layout(self._bulk_layout)
        self._refresh_bulk_status(extra=f"Added {n} file(s) from {folder}.")

    def _bulk_remove_selected(self) -> None:
        sel = self._bulk_table.selectionModel().selectedRows() if self._bulk_table.selectionModel() else []
        if not sel:
            return
        rows = sorted({s.row() for s in sel})
        self._bulk_layout.remove_rows(rows)
        self._bulk_model.set_layout(self._bulk_layout)
        self._refresh_bulk_status(extra=f"Removed {len(rows)} row(s).")

    def _bulk_save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save batch CSV", "bulk_batch.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            self._bulk_layout.to_csv(Path(path))
            self._refresh_bulk_status(extra=f"Saved batch to {path}.")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(e))

    def _bulk_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load batch CSV", "", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            self._bulk_layout = BulkLayout.from_csv(Path(path))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self._bulk_model.set_layout(self._bulk_layout)
        self._refresh_bulk_status(extra=f"Loaded {len(self._bulk_layout.df)} row(s) from {path}.")

    def _refresh_bulk_status(self, extra: str = "") -> None:
        n_total = len(self._bulk_layout.df)
        n_active = len(self._bulk_layout.active_rows())
        msg = f"{n_total} CZIs · {n_active} with conditions assigned."
        if extra:
            msg += "  " + extra
        self._bulk_status.setText(msg)

    # ───────────────────────────────────────────────────────────── shared

    def _pick_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if not path:
            return
        self._out_dir = Path(path)
        self._out_edit.setText(str(self._out_dir))
        self.outputDirSelected.emit(self._out_dir)

    def _refresh_info_and_channels(self) -> None:
        # Source CZI for metadata detection: the explicit single-CZI path
        # in plate or single-file mode; otherwise the first CZI in the bulk
        # batch (so channel labels still get detected for batch mode).
        czi: Path | None = None
        if self._mode == InputMode.SINGLE_PLATE:
            czi = self._czi_path
        elif self._mode == InputMode.SINGLE_FILE:
            sp = self._sf_czi_edit.text().strip()
            czi = Path(sp) if sp else None
        else:  # BULK
            df = self._bulk_layout.df
            if len(df):
                czi = Path(str(df.iloc[0]["czi_path"]))

        if czi is None or not czi.exists():
            self._info_label.setText("")
            self._info_label.setVisible(False)
            self._clear_channel_rows()
            return
        try:
            from mycoprep.core.split_czi_plate import (
                extract_channel_names,
                extract_scene_well_map,
            )
            try:
                scenes = extract_scene_well_map(czi)
                n_scenes = len(scenes)
            except Exception:
                # Non-plate CZIs raise — that's fine; just don't show a count.
                n_scenes = None
            chans = extract_channel_names(czi) or []
            if n_scenes is not None:
                self._info_label.setText(
                    f"{czi.name}: {n_scenes} scenes · {len(chans)} channels detected"
                )
            else:
                self._info_label.setText(
                    f"{czi.name}: {len(chans)} channels detected"
                )
            self._info_label.setVisible(True)
            self._populate_channel_rows(chans)
        except Exception as e:  # noqa: BLE001
            self._info_label.setText(f"(could not read metadata: {e})")
            self._info_label.setVisible(True)
            self._clear_channel_rows()

    # ───────────────────────────────────────────────── channel row helpers

    def _clear_channel_rows(self) -> None:
        while self._channels_form.rowCount() > 1:
            self._channels_form.removeRow(1)
        self._channel_name_edits.clear()
        self._original_channel_names.clear()
        self._phase_combo.blockSignals(True)
        self._phase_combo.clear()
        self._phase_combo.addItem("auto (detect by skewness)", userData=None)
        self._phase_combo.blockSignals(False)

    def _on_reset_clicked(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Reset inputs?",
            "Clear the current CZI, output directory, channel labels, and "
            "any per-file labels? Settings on other tabs are not affected.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        # CZI / output dir
        self._czi_path = None
        self._czi_paths = []
        self._plate_czi_list.clear()
        self._sf_czi_edit.setText("")
        self._out_dir = None
        self._out_edit.setText("")
        # Single-file inline labels
        for edit in (self._sf_condition, self._sf_reporter, self._sf_mutant,
                     self._sf_replica, self._sf_notes):
            edit.setText("")
        # Bulk layout
        self._bulk_layout = BulkLayout.empty()
        self._bulk_model.set_layout(self._bulk_layout)
        # Channels + info banner
        self._clear_channel_rows()
        self._info_label.setText("")
        self._info_label.setVisible(False)
        # Reset mode to Single file
        self._radio_single_file.setChecked(True)
        self.resetRequested.emit()

    def _populate_channel_rows(self, names: list[str]) -> None:
        self._clear_channel_rows()
        self._original_channel_names = list(names)
        self._phase_combo.blockSignals(True)
        for i, n in enumerate(names):
            self._phase_combo.addItem(f"{i}:  {n}", userData=i)
        self._phase_combo.setCurrentIndex(0)
        self._phase_combo.blockSignals(False)
        for i, n in enumerate(names):
            edit = QLineEdit(n)
            edit.textChanged.connect(lambda _t: self.channelsChanged.emit())
            self._channel_name_edits.append(edit)
            self._channels_form.addRow(f"Channel {i} label:", edit)

    # ──────────────────────────────────────────────────────────── getters

    @property
    def mode(self) -> InputMode:
        return self._mode

    @property
    def czi_path(self) -> Path | None:
        """First plate CZI path (legacy single-CZI accessor). None outside
        plate mode. Use :attr:`czi_paths` for the full multi-CZI list."""
        return self._czi_path if self._mode == InputMode.SINGLE_PLATE else None

    @property
    def czi_paths(self) -> list[Path]:
        """Full list of plate CZIs. Single-CZI runs return ``[the_one]``;
        multi-CZI runs return all selected paths in user-add order. Empty
        outside plate mode."""
        if self._mode != InputMode.SINGLE_PLATE:
            return []
        return list(self._czi_paths)

    @property
    def output_dir(self) -> Path | None:
        return self._out_dir

    @property
    def has_czi_input(self) -> bool:
        """True iff a CZI is selected for the current mode (any mode)."""
        if self._mode == InputMode.SINGLE_PLATE:
            return self._czi_path is not None and self._czi_path.exists()
        if self._mode == InputMode.SINGLE_FILE:
            txt = self._sf_czi_edit.text().strip()
            return bool(txt) and Path(txt).exists()
        # BULK: any active row counts as "input present".
        try:
            return len(self._bulk_layout.active_rows()) > 0
        except Exception:  # noqa: BLE001
            return False

    @property
    def bulk_layout(self) -> BulkLayout:
        """Live BulkLayout — populated for single-file and bulk modes."""
        # Make sure single-file edits are flushed before the runner reads it.
        if self._mode == InputMode.SINGLE_FILE:
            self._sync_single_file_to_layout()
        return self._bulk_layout

    @property
    def channel_labels(self) -> Optional[list[str]]:
        if not self._channel_name_edits:
            return None
        return [e.text().strip() or self._original_channel_names[i]
                for i, e in enumerate(self._channel_name_edits)]

    @property
    def phase_channel(self) -> int | str | None:
        return self._phase_combo.currentData()

    # ──────────────────────────────────────────────────────── persistence

    def state(self) -> dict:
        """Snapshot of the panel's UI state for QSettings round-trip."""
        return {
            "mode": self._mode.value,
            "plate_czi": str(self._czi_path) if self._czi_path else "",
            "plate_czi_paths": [str(p) for p in self._czi_paths],
            "single_file_czi": self._sf_czi_edit.text(),
            "single_file_condition": self._sf_condition.text(),
            "single_file_reporter": self._sf_reporter.text(),
            "single_file_mutant": self._sf_mutant.text(),
            "single_file_replica": self._sf_replica.text(),
            "single_file_notes": self._sf_notes.text(),
            "output_dir": str(self._out_dir) if self._out_dir else "",
            "phase_channel": self._phase_combo.currentData(),
            "channel_labels": [e.text() for e in self._channel_name_edits],
        }

    def restore_state(self, s: dict) -> None:
        if not isinstance(s, dict):
            return

        # Restore output dir first (cheap, no metadata reads).
        out = s.get("output_dir") or ""
        if out and Path(out).exists():
            self._out_dir = Path(out)
            self._out_edit.setText(out)
            self.outputDirSelected.emit(self._out_dir)

        # Restore single-file inline labels (so a later channel-restore on
        # this CZI sees the right BulkLayout).
        for key, edit in (
            ("single_file_condition", self._sf_condition),
            ("single_file_reporter",  self._sf_reporter),
            ("single_file_mutant",    self._sf_mutant),
            ("single_file_replica",   self._sf_replica),
            ("single_file_notes",     self._sf_notes),
        ):
            v = s.get(key)
            if v:
                edit.setText(str(v))

        # Restore mode (radio buttons emit signals that flip the stack).
        mode_str = s.get("mode")
        if mode_str:
            try:
                target = InputMode(mode_str)
            except ValueError:
                target = self._mode
            radio = {
                InputMode.SINGLE_FILE:  self._radio_single_file,
                InputMode.SINGLE_PLATE: self._radio_single_plate,
                InputMode.BULK:         self._radio_bulk,
            }[target]
            radio.setChecked(True)

        # Restore CZI path appropriate to the active mode. Skip silently if
        # files are gone — stale paths must not block app startup. Prefer
        # the multi-CZI list when present (new format); fall back to the
        # legacy single ``plate_czi`` field.
        restored_paths: list[Path] = []
        for raw in (s.get("plate_czi_paths") or []):
            p = Path(raw)
            if p.exists():
                restored_paths.append(p)
        if not restored_paths:
            legacy = s.get("plate_czi") or ""
            if legacy and Path(legacy).exists():
                restored_paths = [Path(legacy)]
        if restored_paths:
            self._czi_paths = restored_paths
            self._czi_path = restored_paths[0]
            self._refresh_plate_czi_list()

        sf_czi = s.get("single_file_czi") or ""
        if sf_czi and Path(sf_czi).exists():
            self._sf_czi_edit.setText(sf_czi)
            self._sync_single_file_to_layout()
            if self._mode == InputMode.SINGLE_FILE:
                self._czi_path = Path(sf_czi)

        # Trigger metadata-driven channel population for whichever CZI is
        # active. This is what populates _phase_combo and the label edits.
        if self._czi_path is not None and self._czi_path.exists():
            self._refresh_info_and_channels()
            self.cziSelected.emit(self._czi_path)
            if self._mode == InputMode.SINGLE_PLATE and self._czi_paths:
                self.cziPathsSelected.emit(list(self._czi_paths))

        # Restore channel labels (after _populate_channel_rows ran above).
        labels = s.get("channel_labels") or []
        for i, text in enumerate(labels):
            if i < len(self._channel_name_edits) and text:
                self._channel_name_edits[i].setText(str(text))

        # Restore phase channel selection.
        phase = s.get("phase_channel")
        if phase is not None:
            for i in range(self._phase_combo.count()):
                if self._phase_combo.itemData(i) == phase:
                    self._phase_combo.setCurrentIndex(i)
                    break

        # Notify listeners now that the panel reflects the saved state.
        self.channelsChanged.emit()
