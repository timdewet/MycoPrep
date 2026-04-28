"""Layout panel — wraps the visual PlateMapEditor with CSV import/export."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..pipeline.layout import PlateLayout
from ..ui import icons, tokens
from ..widgets.plate_map import PlateMapEditor


class LayoutPanel(QWidget):
    # True when the layout has at least one active well AND validates
    # clean; False otherwise. Drives the sidebar status indicator.
    layoutValidityChanged = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = PlateLayout.empty()
        self._editor = PlateMapEditor(self._layout)
        self._build_ui()
        # Re-evaluate validity whenever the user touches the plate map
        # (selecting wells, applying labels, clearing, etc.).
        self._editor.layoutChanged.connect(self._emit_validity)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.S3)

        # Toolbar card
        toolbar_card = QFrame()
        toolbar_card.setObjectName("card")
        toolbar = QHBoxLayout(toolbar_card)
        toolbar.setContentsMargins(tokens.S3, tokens.S3, tokens.S3, tokens.S3)
        toolbar.setSpacing(tokens.S2)

        self._import_btn = QPushButton("  Import CSV")
        self._import_btn.setIcon(icons.icon("folder", role="muted"))
        self._export_btn = QPushButton("  Export CSV")
        self._export_btn.setIcon(icons.icon("copy", role="muted"))
        self._validate_btn = QPushButton("  Validate")
        self._validate_btn.setIcon(icons.icon("check", role="muted"))

        self._import_btn.clicked.connect(self._import_clicked)
        self._export_btn.clicked.connect(self._export_clicked)
        self._validate_btn.clicked.connect(self._validate_clicked)

        for b in [self._import_btn, self._export_btn, self._validate_btn]:
            toolbar.addWidget(b)
        toolbar.addStretch(1)

        # Inline status pill (replaces a popup for the common-case Validate result)
        self._status_label = QLabel("")
        self._status_label.setObjectName("muted")
        toolbar.addWidget(self._status_label)

        root.addWidget(toolbar_card)
        root.addWidget(self._editor, stretch=1)

        self._czi_path: Path | None = None
        self._czi_paths: list[Path] = []

    # ---------------------------------------------------------------- slots

    def set_czi_path(self, path: Path) -> None:
        """Single-CZI shim around :meth:`set_czi_paths`.

        Kept so MainWindow's existing ``input_panel.cziSelected`` signal
        keeps working unchanged.
        """
        self.set_czi_paths([path] if path is not None else [])

    def set_czi_paths(self, paths: list[Path]) -> None:
        """Populate the layout from one or more plate CZIs.

        Wells from each CZI contribute their scene_indices and source_czi
        to the merged layout. The first CZI added wins on well-ID
        conflicts; conflicts are surfaced via the status label.
        """
        cleaned = [Path(p) for p in (paths or []) if p is not None]
        self._czi_paths = list(cleaned)
        self._czi_path = cleaned[0] if cleaned else None
        if not cleaned:
            return
        conflicts: list[tuple[str, str, str]] = []
        try:
            self._layout = PlateLayout.from_czis(
                cleaned,
                on_conflict=lambda w, kept, skipped: conflicts.append((w, kept, skipped)),
            )
        except Exception:  # noqa: BLE001
            return
        self._editor.set_layout(self._layout)
        if conflicts:
            self._status_label.setText(
                f"⚠ {len(conflicts)} well conflict(s) — kept first CZI's "
                f"wells, see log"
            )
            for well, kept, skipped in conflicts:
                print(
                    f"[layout] well {well} appears in {kept} and {skipped}; "
                    f"keeping {kept}"
                )
        self._emit_validity()

    def clear(self) -> None:
        """Reset to an empty plate layout (e.g. when the user resets inputs)."""
        self._czi_path = None
        self._layout = PlateLayout.empty()
        self._editor.set_layout(self._layout)
        self._status_label.setText("")
        self._emit_validity()

    def set_output_dir(self, path: Path) -> None:
        """If the selected output directory already has a plate_layout.csv
        from a previous run, load it so the user doesn't have to redo the
        well assignments. Silently ignored if absent or unreadable.

        When CZIs have already been loaded into the panel, the CSV is
        OVERLAID onto the existing wells (labels only) rather than
        replacing the layout — otherwise wells from CZIs that aren't
        mentioned in the CSV would disappear along with their live
        scene_indices.
        """
        if path is None:
            return
        csv = Path(path) / "plate_layout.csv"
        if not csv.exists():
            return
        try:
            imported = PlateLayout.from_csv(csv)
        except Exception:  # noqa: BLE001
            return
        if self._has_live_czi_metadata():
            self._layout.merge_labels_from(imported)
        else:
            self._layout = imported
        self._editor.set_layout(self._layout)
        self._emit_validity()

    def _import_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import plate layout CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            imported = PlateLayout.from_csv(Path(path))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Import failed", str(e))
            return

        if self._has_live_czi_metadata():
            # CZIs have populated wells with scene_indices / source_czi —
            # overlay the CSV's labels onto those wells without dropping
            # wells the CSV doesn't mention (e.g. a 3rd CZI's wells when
            # the CSV only described the first two).
            n_updated = self._layout.merge_labels_from(imported)
            csv_wells = set(imported.df["well"].astype(str))
            live_wells = set(self._layout.df["well"].astype(str))
            unmatched = sorted(csv_wells - live_wells)
            unlabelled = len(self._layout.active_rows()) == 0 and n_updated == 0
            msg_parts = [f"Updated labels on {n_updated} well(s)."]
            if unmatched:
                msg_parts.append(
                    f"{len(unmatched)} CSV well(s) had no matching CZI scenes "
                    f"and were skipped: {', '.join(unmatched[:8])}"
                    + ("…" if len(unmatched) > 8 else "")
                )
            self._status_label.setText("  ·  ".join(msg_parts))
            if unlabelled:
                # Probably an empty/legacy layout state — fall back to
                # replacing so the user actually gets data.
                self._layout = imported
        else:
            self._layout = imported

        self._editor.set_layout(self._layout)
        self._emit_validity()

    def _has_live_czi_metadata(self) -> bool:
        """True if any well in the current layout has scene_indices populated.

        Indicates the user has loaded CZIs into the panel — and that a CSV
        import should overlay labels rather than replace the whole layout.
        """
        try:
            for sis in self._layout.df.get("scene_indices", []):
                if isinstance(sis, (list, tuple)) and len(sis) > 0:
                    return True
                if isinstance(sis, str) and sis.strip() not in ("", "[]"):
                    return True
        except Exception:  # noqa: BLE001
            return False
        return False

    def _export_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export plate layout CSV", "plate_layout.csv", "CSV (*.csv)")
        if not path:
            return
        self._layout.to_csv(Path(path))

    def _is_valid(self) -> bool:
        try:
            return (
                len(self._editor.layout_model.active_rows()) > 0
                and not self._editor.layout_model.validate()
            )
        except Exception:  # noqa: BLE001
            return False

    def _emit_validity(self) -> None:
        self.layoutValidityChanged.emit(self._is_valid())

    def _validate_clicked(self) -> None:
        from ..ui import tokens as _t
        issues = self._editor.layout_model.validate()
        if issues:
            self._status_label.setText(f"⚠ {len(issues)} issue(s)")
            self._status_label.setStyleSheet(f"color: {_t.active().warning};")
            QMessageBox.warning(self, "Layout issues", "\n".join(issues))
        else:
            active = len(self._editor.layout_model.active_rows())
            self._status_label.setText(f"✓ Valid · {active} well(s) ready")
            self._status_label.setStyleSheet(f"color: {_t.active().success};")
        self._emit_validity()

    # ---------------------------------------------------------------- getters

    @property
    def layout_model(self) -> PlateLayout:
        return self._editor.layout_model
