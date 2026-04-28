"""Active-learning labeler + classifier fine-tuning tab.

Workflow:
  1. Pick a segmented-output folder (defaults to the project's 02_segment).
  2. Pick a base classifier model (defaults to bundled mtb).
  3. Click "Load uncertain cells" — the panel runs the model on the
     segmented FOVs, ranks every cell by classifier uncertainty
     (probability nearest 0.5), and shows the top-N as a clickable grid.
  4. Click each cell to cycle ``unlabeled → good → bad → unlabeled``;
     keyboard shortcuts ``g`` / ``b`` / ``u`` toggle the focused cell.
  5. "Save labels" appends the labeled crops to
     ``data/labeled_data/`` in the format
     ``cell_quality_classifier.save_crop_dataset`` produces.
  6. The training section retrains the model on all labeled crops,
     fine-tuning from the currently-selected preset by default.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from mycoprep.core.api import PRESET_MODELS, resolve_classifier_preset


# ─────────────────────────────────────────────────────────────────────────────
# Active-learning worker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CellSample:
    crop: np.ndarray            # (C, H, W) — to be saved as the training crop
    thumbnail: np.ndarray       # phase plane only, for fast UI display
    source_file: str            # which input TIFF this came from
    fov_index: int
    cell_label: int             # label inside its FOV's mask
    confidence_good: float      # CNN P(good) at the time of ranking


class _ActiveLearningWorker(QThread):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object, str)   # list[_CellSample], error

    def __init__(
        self,
        segment_dir: Path,
        model_path: Optional[Path],
        phase_channel: int,
        n_target: int,
        seen_ids: Optional[set] = None,
    ) -> None:
        super().__init__()
        self.segment_dir = segment_dir
        self.model_path = model_path
        self.phase_channel = phase_channel
        self.n_target = n_target
        # Set of (source_stem, fov_index, cell_label) triples to skip — cells
        # that have already been labeled (on disk) or shown in this session.
        self.seen_ids: set = seen_ids or set()
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:    # noqa: D401
        try:
            from skimage.measure import label as sk_label
            from skimage.segmentation import find_boundaries  # noqa: F401

            from mycoprep.core.cell_quality_classifier import (
                CROP_SIZE,
                CellQualityClassifier,
                _append_area_channel_single,
                extract_cell_crop,
            )
            from mycoprep.core.label_cells import load_hyperstack

            tiffs = (
                sorted(self.segment_dir.glob("*.tif"))
                + sorted(self.segment_dir.glob("*.tiff"))
            )
            if not tiffs:
                self.finished.emit([], f"No TIFFs found in {self.segment_dir}")
                return

            # Load classifier (only if a model is provided — otherwise rank cells
            # by random order so the user can still hand-label without a model).
            clf = None
            if self.model_path is not None:
                clf = CellQualityClassifier(str(self.model_path), in_channels=3)

            samples: list[_CellSample] = []
            for ti, tiff_path in enumerate(tiffs):
                if self._cancel:
                    break
                self.progress.emit(
                    ti / max(len(tiffs), 1),
                    f"Reading {tiff_path.name} ({ti+1}/{len(tiffs)})",
                )
                data, _ = load_hyperstack(tiff_path)   # (N_FOV, C, Y, X)
                n_fov = data.shape[0]
                for fov_idx in range(n_fov):
                    if self._cancel:
                        break
                    fov = data[fov_idx]
                    image_channels = fov[:-1]
                    raw_mask = fov[-1]
                    uniq = np.unique(raw_mask[raw_mask > 0])
                    if len(uniq) <= 1:
                        labeled = sk_label(raw_mask > 0, connectivity=1).astype(np.int32)
                    else:
                        labeled = raw_mask.astype(np.int32)
                    cell_labels = sorted(set(np.unique(labeled)) - {0})
                    if not cell_labels:
                        continue
                    # Skip cells we've already labelled (or shown this session)
                    src_stem = Path(tiff_path).stem
                    cell_labels = [
                        lbl for lbl in cell_labels
                        if (src_stem, fov_idx, int(lbl)) not in self.seen_ids
                    ]
                    if not cell_labels:
                        continue
                    # Build crops for every cell, score them
                    crops_list = []
                    valid_labels = []
                    for lbl in cell_labels:
                        crop, _ = extract_cell_crop(
                            image_channels, labeled, lbl,
                            phase_channel=self.phase_channel,
                        )
                        if crop is None:
                            continue
                        true_area = int(np.sum(labeled == lbl))
                        crop = _append_area_channel_single(crop, true_area_px=true_area)
                        crops_list.append(crop)
                        valid_labels.append(lbl)
                    if not crops_list:
                        continue
                    crops_arr = np.stack(crops_list, axis=0)
                    if clf is not None:
                        _, probs = clf.predict_batch(crops_arr)   # (N, num_classes)
                        # Confidence-good = P(class 0). Uncertainty = |0.5 - p|.
                        p_good = probs[:, 0]
                    else:
                        p_good = np.full(len(crops_list), 0.5, dtype=np.float32)
                    for crop, lbl, pg in zip(crops_list, valid_labels, p_good):
                        # Phase plane: index 0 of the saved crop is phase.
                        thumb = crop[0]
                        samples.append(_CellSample(
                            crop=crop,
                            thumbnail=thumb,
                            source_file=tiff_path.name,
                            fov_index=fov_idx,
                            cell_label=int(lbl),
                            confidence_good=float(pg),
                        ))

            # Rank by uncertainty (closest to 0.5 first), trim to n_target
            samples.sort(key=lambda s: abs(0.5 - s.confidence_good))
            samples = samples[: self.n_target]

            self.progress.emit(1.0, f"Loaded {len(samples)} cells for labeling")
            self.finished.emit(samples, "")
        except Exception as e:
            self.finished.emit([], str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Cell thumbnail widget
# ─────────────────────────────────────────────────────────────────────────────

LABEL_COLORS = {
    "unlabeled": "#d8dde3",
    "good":      "#2ca02c",
    "bad":       "#e0402b",
}


class _CellThumb(QPushButton):
    """Square button showing a single phase crop, recoloured by current label."""

    SIDE_PX = 96

    def __init__(self, sample: _CellSample, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.sample = sample
        self.label = "unlabeled"          # 'unlabeled' | 'good' | 'bad'
        self.setFixedSize(self.SIDE_PX, self.SIDE_PX)
        self.setIconSize(self._icon_size())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._render()
        self.clicked.connect(self._cycle)

    def _icon_size(self):
        from PyQt6.QtCore import QSize
        return QSize(self.SIDE_PX - 12, self.SIDE_PX - 12)

    def _render(self) -> None:
        from PyQt6.QtGui import QImage, QPixmap, QIcon
        from skimage.segmentation import find_boundaries

        # Crop layout from extract_cell_crop + _append_area_channel_single is
        # (3, H, W): [phase, mask, area]. We want the phase as the visual base
        # and the mask boundary as a coloured overlay so the user can tell
        # exactly which object Cellpose segmented.
        phase = self.sample.crop[0]
        mask = self.sample.crop[1] > 0.5

        # Auto-contrast phase
        lo, hi = np.percentile(phase, (1, 99))
        scaled = np.clip((phase - lo) / max(hi - lo, 1e-6), 0, 1)
        gray = (scaled * 255).astype(np.uint8)
        h, w = gray.shape

        # Build RGB so we can paint a coloured boundary on top
        rgb = np.stack([gray, gray, gray], axis=-1)
        if mask.any():
            boundary = find_boundaries(mask, mode="outer")
            rgb[boundary] = (0x21, 0xD4, 0xFD)   # cyan — readable over light & dark phase

        # Make sure the buffer is C-contiguous before handing it to QImage
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self._icon_size().width(), self._icon_size().height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setIcon(QIcon(pix))

        from ..ui import tokens as _t
        pal = _t.active()
        # Map symbolic labels to theme tokens so colours track light/dark.
        ring = {
            "unlabeled": pal.border_strong,
            "good":      pal.success,
            "bad":       pal.danger,
        }[self.label]
        self.setStyleSheet(
            f"QPushButton {{ background: {pal.surface}; border: 3px solid {ring}; "
            f"border-radius: 6px; padding: 2px; }}"
            f"QPushButton:hover {{ border: 3px solid {pal.primary}; }}"
        )

    def _cycle(self) -> None:
        order = ["unlabeled", "good", "bad"]
        self.label = order[(order.index(self.label) + 1) % len(order)]
        self._render()

    def set_label(self, value: str) -> None:
        if value in LABEL_COLORS:
            self.label = value
            self._render()


# ─────────────────────────────────────────────────────────────────────────────
# Training worker
# ─────────────────────────────────────────────────────────────────────────────

class _TrainWorker(QThread):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object, str)   # summary dict, error

    def __init__(self, data_dir: Path, output_dir: Path,
                 epochs: int, lr: float, pretrained: Optional[Path]) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.epochs = epochs
        self.lr = lr
        self.pretrained = pretrained

    def run(self) -> None:
        try:
            from mycoprep.core.train_classifier import train
            summary = train(
                data_dir=self.data_dir,
                output_dir=self.output_dir,
                epochs=self.epochs,
                lr=self.lr,
                pretrained_path=str(self.pretrained) if self.pretrained else None,
                progress_cb=lambda f, m: self.progress.emit(f, m),
            )
            # Belt-and-braces: even if train() somehow forgot to emit a final
            # progress(1.0, …), make sure the bar finalises here.
            self.progress.emit(1.0, "Training finished.")
            self.finished.emit(summary or {}, "")
        except Exception as e:
            self.finished.emit({}, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Panel
# ─────────────────────────────────────────────────────────────────────────────

# Where labeled crops accumulate across runs. Resolved relative to this file.
SHARED_LABELED_DIR = (
    Path(__file__).resolve().parents[4] / "data" / "labeled_data"
)


def _parse_crop_filename(fname: str) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """Pull (source_stem, fov_index, cell_label) out of a saved crop filename.

    Filenames are written as:
        {source_stem}__fov{fov:03d}__cell{cell:04d}__{ts}_{i:03d}.npy
    Returns (None, None, None) if the filename doesn't match.
    """
    import re
    m = re.match(r"^(?P<stem>.+?)__fov(?P<fov>\d+)__cell(?P<cell>\d+)__", fname)
    if not m:
        return None, None, None
    return m.group("stem"), int(m.group("fov")), int(m.group("cell"))


class LabelTrainPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._segment_dir: Path | None = None
        self._phase_channel: int = 0
        # Pool: every cell loaded by the most recent _ActiveLearningWorker run,
        # ranked by uncertainty. We paginate through this locally so the user
        # gets near-instant page turns instead of paying the cellpose+CNN cost
        # for every fresh batch of 48.
        self._pool: list[_CellSample] = []
        self._page_index = 0
        # In-memory label cache, keyed by absolute pool index so that
        # navigating back/forth between pages preserves what the user has
        # already marked but not yet saved to disk.
        self._page_labels: dict[int, str] = {}
        self._thumbs: list[_CellThumb] = []
        self._al_worker: _ActiveLearningWorker | None = None
        self._train_worker: _TrainWorker | None = None
        # Cells already shown this session so re-loading shows new ones.
        self._session_seen_ids: set = set()
        self._build_ui()
        self._refresh_dataset_stats()

    # ----------------------------------------------------------------- public

    def set_segment_dir(self, path: Path | None) -> None:
        if path is None:
            self._segment_dir = None
            self._dir_edit.setText("")
            return
        self._segment_dir = Path(path)
        self._dir_edit.setText(str(self._segment_dir))

    def set_phase_channel(self, ch: int) -> None:
        self._phase_channel = int(ch) if isinstance(ch, int) else 0

    # -------------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        from ..ui import tokens as _t
        root = QVBoxLayout(self)
        root.setContentsMargins(_t.S4, _t.S4, _t.S4, _t.S4)
        root.setSpacing(_t.S3)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_label_section())
        splitter.addWidget(self._build_train_section())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

    def _build_label_section(self) -> QWidget:
        from ..ui import tokens as _t
        box = QGroupBox("Label cells")
        v = QVBoxLayout(box)
        v.setContentsMargins(_t.S4, _t.S5, _t.S4, _t.S4)
        v.setSpacing(_t.S2)

        # Toolbar
        bar = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setReadOnly(True)
        self._dir_edit.setPlaceholderText("(no segmented folder selected)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_dir)

        self._model_combo = QComboBox()
        self._model_combo.addItem("none (random order)", userData=None)
        for name in sorted(PRESET_MODELS.keys()):
            try:
                p = resolve_classifier_preset(name)
                self._model_combo.addItem(f"preset: {name}", userData=str(p))
            except FileNotFoundError:
                pass

        # Pool = how many cells to load up-front per "Load pool" click.
        # Page = how many of the pool to show at once. The user labels a
        # page, hits "Save & next page", and the next slice of the pool
        # is shown without re-running cellpose/CNN.
        self._pool_size = QSpinBox(); self._pool_size.setRange(48, 5000); self._pool_size.setValue(1000)
        self._pool_size.setToolTip("Cells to preload per fetch. Higher = fewer waits, more upfront time.")
        self._page_size = QSpinBox(); self._page_size.setRange(8, 256); self._page_size.setValue(48)
        self._page_size.setToolTip("Cells shown per page. Smaller pages render faster and feel less overwhelming.")

        self._load_btn = QPushButton("Load pool")
        self._load_btn.setToolTip(
            "Run the model across all FOVs and load the most-uncertain cells "
            "into memory in one shot. Subsequent pages turn instantly."
        )
        self._load_btn.clicked.connect(self._load_pool)

        self._save_btn = QPushButton("Save labels")
        self._save_btn.clicked.connect(self._save_batch)
        self._save_btn.setEnabled(False)

        # Plain page navigation — labels you've made on the current page are
        # cached in memory and restored when you come back, so flipping
        # between pages without saving is non-destructive.
        self._prev_btn = QPushButton("← Prev")
        self._prev_btn.clicked.connect(self._prev_page)
        self._prev_btn.setEnabled(False)
        self._next_btn = QPushButton("Next →")
        self._next_btn.clicked.connect(self._next_page)
        self._next_btn.setEnabled(False)

        # Primary action: save current page's labels and advance to the
        # next page in the already-loaded pool. No model re-run.
        self._save_next_btn = QPushButton("Save & next page")
        self._save_next_btn.setObjectName("primary")
        self._save_next_btn.clicked.connect(self._save_then_next_page)
        self._save_next_btn.setEnabled(False)

        self._reset_btn = QPushButton("Reset session")
        self._reset_btn.setToolTip(
            "Forget which cells have been shown this session, so previously-skipped "
            "cells become eligible again. (Already-saved labels are still excluded.)"
        )
        self._reset_btn.clicked.connect(self._reset_session)

        bar.addWidget(QLabel("Source:"))
        bar.addWidget(self._dir_edit, stretch=1)
        bar.addWidget(browse)
        bar.addWidget(QLabel("Model:"))
        bar.addWidget(self._model_combo)
        bar.addWidget(QLabel("Pool:"))
        bar.addWidget(self._pool_size)
        bar.addWidget(QLabel("Page:"))
        bar.addWidget(self._page_size)
        bar.addWidget(self._load_btn)
        bar.addWidget(self._save_btn)
        bar.addWidget(self._prev_btn)
        bar.addWidget(self._next_btn)
        bar.addWidget(self._save_next_btn)
        bar.addWidget(self._reset_btn)
        v.addLayout(bar)

        # Status + progress
        self._status = QLabel("Pick a segmented-output folder and load a batch.")
        self._status.setObjectName("muted")
        self._status.setWordWrap(True)
        v.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000); self._progress.setValue(0)
        v.addWidget(self._progress)

        # Hint
        hint = QLabel(
            "Click a thumbnail to cycle: unlabeled → good → bad → unlabeled.  "
            "Outer border shows your label (grey/green/red); inner cyan outline "
            "shows the segmented cell.  "
            "Use ← Prev / Next → to flip pages without saving — your labels "
            "are remembered.  Click 'Save & next page' to persist labels and "
            "advance, or 'Save labels' to commit without moving."
        )
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        v.addWidget(hint)

        # Grid of thumbnails (in a scroll area)
        self._grid_host = QWidget()
        from PyQt6.QtWidgets import QGridLayout
        self._grid = QGridLayout(self._grid_host)
        self._grid.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._grid_host)
        v.addWidget(scroll, stretch=1)

        return box

    def _build_train_section(self) -> QWidget:
        from ..ui import tokens as _t
        box = QGroupBox("Fine-tune classifier")
        v = QVBoxLayout(box)
        v.setContentsMargins(_t.S4, _t.S5, _t.S4, _t.S4)
        v.setSpacing(_t.S2)

        # Stats
        self._stats_label = QLabel("")
        self._stats_label.setObjectName("muted")
        v.addWidget(self._stats_label)

        # Controls
        bar = QHBoxLayout()
        self._epochs = QSpinBox(); self._epochs.setRange(1, 500); self._epochs.setValue(40)
        # Lower default LR than the from-scratch trainer (1e-3) — fine-tuning
        # a pretrained model on a small batch of labels needs gentle updates,
        # otherwise epoch 1 obliterates the pretrained features (val acc
        # collapses to chance and never recovers).
        self._lr = QDoubleSpinBox()
        self._lr.setRange(1e-6, 1.0); self._lr.setDecimals(6)
        self._lr.setValue(1e-4); self._lr.setSingleStep(5e-5)
        self._fine_tune = QCheckBox("Fine-tune from selected base model"); self._fine_tune.setChecked(True)
        self._train_btn = QPushButton("Start training")
        self._train_btn.setObjectName("primary")
        self._train_btn.clicked.connect(self._start_training)

        bar.addWidget(QLabel("Epochs:")); bar.addWidget(self._epochs)
        bar.addWidget(QLabel("LR:"));     bar.addWidget(self._lr)
        bar.addWidget(self._fine_tune)
        bar.addStretch(1)
        bar.addWidget(self._train_btn)
        v.addLayout(bar)

        self._train_progress = QProgressBar()
        self._train_progress.setRange(0, 1000); self._train_progress.setValue(0)
        v.addWidget(self._train_progress)

        self._train_status = QLabel("")
        self._train_status.setObjectName("muted")
        self._train_status.setWordWrap(True)
        v.addWidget(self._train_status)

        return box

    # ---------------------------------------------------------------- events

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select segmented-output folder",
            str(self._segment_dir) if self._segment_dir else "",
        )
        if path:
            self.set_segment_dir(Path(path))

    def _load_pool(self) -> None:
        """Run cellpose+CNN once over all FOVs to build a large pool of the
        most-uncertain cells. Subsequent page turns are local — no model
        re-runs — until the pool is exhausted."""
        if self._segment_dir is None or not self._segment_dir.exists():
            QMessageBox.warning(self, "No source", "Pick a segmented-output folder first.")
            return

        model_path_str = self._model_combo.currentData()
        model_path = Path(model_path_str) if model_path_str else None

        seen = self._load_disk_seen_ids() | self._session_seen_ids

        self._load_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._save_next_btn.setEnabled(False)
        self._progress.setValue(0)
        self._status.setText(
            f"Loading pool of up to {self._pool_size.value()} cells "
            f"(skipping {len(seen)} already-shown). This is a one-time wait per pool."
        )

        self._al_worker = _ActiveLearningWorker(
            segment_dir=self._segment_dir,
            model_path=model_path,
            phase_channel=self._phase_channel,
            n_target=self._pool_size.value(),
            seen_ids=seen,
        )
        self._al_worker.progress.connect(self._on_al_progress)
        self._al_worker.finished.connect(self._on_al_finished)
        self._al_worker.start()

    def _save_then_next_page(self) -> None:
        """Capture current page's labels into the cache, persist everything
        cached so far, then advance to the next page from the pool."""
        self._capture_current_page()
        self._save_batch(silent_if_empty=True)
        self._page_index += 1
        page_size = self._page_size.value()
        if self._page_index * page_size >= len(self._pool):
            # Pool exhausted — no more pages to show.
            self._populate_grid_from_page()
            self._save_btn.setEnabled(False)
            self._save_next_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            self._status.setText(
                f"Pool exhausted (cycled through all {len(self._pool)} cells). "
                "Click 'Load pool' to fetch more from the segmented folder."
            )
            return
        self._populate_grid_from_page()
        self._update_page_status()

    def _reset_session(self) -> None:
        """Clear the in-session "already shown" set so re-loading shows
        previously-skipped cells again. Saved labels are still excluded.
        Also drops any cached-but-unsaved labels."""
        if self._session_seen_ids or self._page_labels:
            n_seen = len(self._session_seen_ids)
            n_cached = len(self._page_labels)
            self._session_seen_ids.clear()
            self._page_labels.clear()
            self._status.setText(
                f"Cleared {n_seen} cells from session-shown set "
                f"and {n_cached} unsaved label(s) from the cache. "
                "Saved labels on disk are still excluded."
            )

    # ---------------------------------------------------------------- dedup

    def _load_disk_seen_ids(self) -> set:
        """Parse labels.csv to recover (source_stem, fov_index, cell_label)
        triples for every previously-saved labelled crop."""
        labels_csv = SHARED_LABELED_DIR / "labels.csv"
        if not labels_csv.exists():
            return set()
        seen: set = set()
        with open(labels_csv) as f:
            for r in csv.DictReader(f):
                fname = r.get("filename", "")
                stem, fov, cell = _parse_crop_filename(fname)
                if stem is not None:
                    seen.add((stem, fov, cell))
        return seen

    def _on_al_progress(self, frac: float, msg: str) -> None:
        self._progress.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self._status.setText(msg)

    def _on_al_finished(self, samples, error: str) -> None:
        self._load_btn.setEnabled(True)
        if error:
            self._status.setText(f"Failed: {error}")
            return

        self._pool = list(samples)
        self._page_index = 0
        # The cache is keyed by index into _pool, so a fresh pool invalidates
        # any cached labels from a previous one.
        self._page_labels.clear()

        # Mark every cell in the pool as session-shown so subsequent pool
        # loads skip them, even if the user never labels them all.
        for s in self._pool:
            self._session_seen_ids.add(
                (Path(s.source_file).stem, s.fov_index, s.cell_label)
            )

        if not self._pool:
            self._populate_grid_from_page()
            self._save_btn.setEnabled(False)
            self._save_next_btn.setEnabled(False)
            self._status.setText(
                "No more unseen cells in this folder. "
                "Use 'Reset session' to revisit skipped cells, "
                "or pick a different segmented folder."
            )
            return

        self._populate_grid_from_page()
        self._save_btn.setEnabled(True)
        self._save_next_btn.setEnabled(True)
        self._update_page_status()

    def _update_page_status(self) -> None:
        page_size = self._page_size.value()
        start = self._page_index * page_size
        end = min(start + page_size, len(self._pool))
        n_pages = (len(self._pool) + page_size - 1) // page_size
        self._status.setText(
            f"Pool: {len(self._pool)} cells loaded · "
            f"page {self._page_index + 1}/{n_pages} "
            f"(cells {start + 1}–{end}). "
            "Click 'Save & next page' for the next set — pages turn instantly."
        )

    def _populate_grid_from_page(self) -> None:
        # Clear current thumbnails
        for i in reversed(range(self._grid.count())):
            item = self._grid.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        self._thumbs.clear()

        page_size = self._page_size.value()
        start = self._page_index * page_size
        end = min(start + page_size, len(self._pool))

        cols = 8
        for i, s in enumerate(self._pool[start:end]):
            thumb = _CellThumb(s)
            # Restore any cached label for this absolute pool index.
            cached = self._page_labels.get(start + i)
            if cached:
                thumb.set_label(cached)
            self._thumbs.append(thumb)
            self._grid.addWidget(thumb, i // cols, i % cols)

        # Refresh prev/next button enabled-state to match where we are.
        self._prev_btn.setEnabled(self._page_index > 0)
        self._next_btn.setEnabled((end < len(self._pool)))

    def _capture_current_page(self) -> None:
        """Snapshot the labels from the visible page into the pool-level cache."""
        page_size = self._page_size.value()
        start = self._page_index * page_size
        for i, t in enumerate(self._thumbs):
            self._page_labels[start + i] = t.label

    def _prev_page(self) -> None:
        if self._page_index <= 0:
            return
        self._capture_current_page()
        self._page_index -= 1
        self._populate_grid_from_page()
        self._update_page_status()

    def _next_page(self) -> None:
        page_size = self._page_size.value()
        if (self._page_index + 1) * page_size >= len(self._pool):
            return
        self._capture_current_page()
        self._page_index += 1
        self._populate_grid_from_page()
        self._update_page_status()

    # ---------------------------------------------------------------- saving

    def _save_batch(self, _checked: bool = False, silent_if_empty: bool = False) -> int:
        # Always include what's currently visible in the cache so saving from
        # a single visible page works even if no navigation happened.
        self._capture_current_page()

        # All cells with a non-"unlabeled" tag in the cache, regardless of
        # which page they were assigned on.
        to_save = [
            (idx, lbl) for idx, lbl in self._page_labels.items()
            if lbl in ("good", "bad") and 0 <= idx < len(self._pool)
        ]
        if not to_save:
            if not silent_if_empty:
                QMessageBox.information(self, "Nothing to save", "Label some cells first.")
            return 0

        SHARED_LABELED_DIR.mkdir(parents=True, exist_ok=True)
        crops_dir = SHARED_LABELED_DIR / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        labels_csv = SHARED_LABELED_DIR / "labels.csv"

        ts = int(time.time())
        rows = []

        from mycoprep.core.cell_quality_classifier import CLASS_NAMES
        good_idx = CLASS_NAMES.index("good") if "good" in CLASS_NAMES else 0
        bad_idx  = CLASS_NAMES.index("bad")  if "bad"  in CLASS_NAMES else 1

        for i, (pool_idx, label_name) in enumerate(to_save):
            sample = self._pool[pool_idx]
            label_idx = good_idx if label_name == "good" else bad_idx
            stem = (
                f"{Path(sample.source_file).stem}__"
                f"fov{sample.fov_index:03d}__cell{sample.cell_label:04d}__{ts}_{i:03d}"
            )
            fname = f"{stem}.npy"
            np.save(crops_dir / fname, sample.crop)
            rows.append([fname, label_idx, label_name])

        # Append to labels.csv
        new_file = not labels_csv.exists()
        with open(labels_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["filename", "label_idx", "label_name"])
            w.writerows(rows)

        # Drop the saved entries from the cache so they don't get persisted
        # again on the next save.
        for idx, _ in to_save:
            self._page_labels.pop(idx, None)

        # Refresh metadata.json
        meta_path = SHARED_LABELED_DIR / "metadata.json"
        from mycoprep.core.cell_quality_classifier import CROP_SIZE
        first_pool_idx = to_save[0][0]
        n_channels = self._pool[first_pool_idx].crop.shape[0]
        meta = {
            "crop_size": CROP_SIZE,
            "n_channels": n_channels,
            "class_names": CLASS_NAMES,
            "n_samples": (
                sum(1 for _ in csv.reader(open(labels_csv))) - 1
            ),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        n_saved = len(rows)
        self._status.setText(
            f"Saved {n_saved} labels → {SHARED_LABELED_DIR}. "
            "Use Prev/Next to keep paging, or 'Save & next page' for the next set."
        )
        self._refresh_dataset_stats()
        return n_saved

    # ---------------------------------------------------------------- training

    def _refresh_dataset_stats(self) -> None:
        labels_csv = SHARED_LABELED_DIR / "labels.csv"
        if not labels_csv.exists():
            self._stats_label.setText(
                f"No labeled crops yet. Saved batches will accumulate at {SHARED_LABELED_DIR}."
            )
            self._train_btn.setEnabled(False)
            return
        good = bad = 0
        with open(labels_csv) as f:
            for r in csv.DictReader(f):
                if r["label_name"] == "good":
                    good += 1
                elif r["label_name"] == "bad":
                    bad += 1
        total = good + bad
        self._stats_label.setText(
            f"{total} labeled crops on disk · good={good}, bad={bad}  "
            f"({SHARED_LABELED_DIR})"
        )
        self._train_btn.setEnabled(total >= 20)

    def _start_training(self) -> None:
        # Output dir alongside the labeled data
        out_dir = SHARED_LABELED_DIR / f"trained_{int(time.time())}"
        pretrained = None
        if self._fine_tune.isChecked():
            mdl = self._model_combo.currentData()
            pretrained = Path(mdl) if mdl else None

        self._train_btn.setEnabled(False)
        self._train_progress.setValue(0)
        self._train_status.setText(
            f"Training to {out_dir} (epochs={self._epochs.value()}, "
            f"lr={self._lr.value()}, fine-tune={'yes' if pretrained else 'no'})…"
        )

        self._train_worker = _TrainWorker(
            data_dir=SHARED_LABELED_DIR,
            output_dir=out_dir,
            epochs=self._epochs.value(),
            lr=self._lr.value(),
            pretrained=pretrained,
        )
        self._train_worker.progress.connect(self._on_train_progress)
        self._train_worker.finished.connect(self._on_train_finished)
        self._train_worker.start()

    def _on_train_progress(self, frac: float, msg: str) -> None:
        self._train_progress.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self._train_status.setText(msg)

    def _on_train_finished(self, summary: dict, error: str) -> None:
        self._train_btn.setEnabled(True)
        # Make sure the bar reads 100% regardless of how training ended.
        self._train_progress.setValue(1000)

        if error:
            self._train_status.setText(f"Training failed: {error}")
            return

        best_acc = summary.get("best_val_acc", 0.0)
        n_train = summary.get("n_train", 0)
        n_val = summary.get("n_val", 0)
        model_path = summary.get("model_path", "(unknown)")

        warning = ""
        if best_acc <= 0.55:
            warning = (
                "  ⚠  Validation accuracy is at chance — the model didn't learn "
                "anything generalisable. With only a few dozen labels (and those "
                "drawn from the *most uncertain* cells), the training signal is "
                "very noisy. Try labeling many more cells (300+ ideally) before "
                "retraining, or lower the learning rate further."
            )
        elif n_train + n_val < 100:
            warning = (
                "  ⚠  Small dataset — results will be noisy. Label more cells "
                "and retrain to improve."
            )

        self._train_status.setText(
            f"Done · best val acc {best_acc:.3f}  "
            f"(train={n_train}, val={n_val})  →  {model_path}.{warning}"
        )
