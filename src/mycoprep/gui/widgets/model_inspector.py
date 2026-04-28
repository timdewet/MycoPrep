"""Classifier-model inspector.

Given a path to a trained .pth, looks in the same directory for:
  - metrics.json         → raw ROC/PR curves (live overlay with threshold line)
  - roc_pr_curves.png    → fallback static image
  - training_config.json → summary stats

The live plot re-renders whenever the user changes the confidence threshold;
the threshold is drawn as a vertical line on both the ROC (in threshold
space) and PR curves, with the operating point highlighted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class ModelInspector(QWidget):
    """Renders classifier metadata for the currently selected model."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model_dir: Path | None = None
        self._metrics: dict | None = None
        self._threshold = 0.5
        self._build_ui()
        self.set_model(None)

    # ------------------------------------------------------------------ setup

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self._header = QLabel()
        self._header.setObjectName("sectionHeader")
        self._header.setWordWrap(True)
        root.addWidget(self._header)

        self._stats = QLabel()
        self._stats.setObjectName("muted")
        self._stats.setWordWrap(True)
        root.addWidget(self._stats)

        # Live plot (matplotlib) — compact (~25% of the original training figure)
        self._fig = Figure(figsize=(4.6, 2.0), tight_layout=True, facecolor="#ffffff")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._canvas.setFixedHeight(220)
        self._canvas.setFixedWidth(660)
        root.addWidget(self._canvas, alignment=Qt.AlignmentFlag.AlignLeft)

        # Static image fallback — scaled down to match
        self._png_label = QLabel()
        self._png_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._png_label.setFixedHeight(220)
        self._png_scroll = QScrollArea()
        self._png_scroll.setWidgetResizable(False)
        self._png_scroll.setFrameShape(self._png_scroll.Shape.NoFrame)
        self._png_scroll.setFixedHeight(240)
        self._png_scroll.setMaximumWidth(680)
        self._png_scroll.setWidget(self._png_label)
        root.addWidget(self._png_scroll, alignment=Qt.AlignmentFlag.AlignLeft)

    # ----------------------------------------------------------------- public

    def set_model(self, model_path: Path | None) -> None:
        """Point the inspector at a specific .pth (or None = no model)."""
        if model_path is None:
            self._model_dir = None
            self._metrics = None
            self._header.setText("No classifier selected")
            self._stats.setText("Rule-based filtering only (edge / debris).")
            self._canvas.hide()
            self._png_scroll.hide()
            return

        self._model_dir = Path(model_path).parent
        self._load_metadata()
        self._refresh()

    def set_threshold(self, t: float) -> None:
        self._threshold = float(t)
        self._refresh_live_plot()

    # ---------------------------------------------------------------- loading

    def _load_metadata(self) -> None:
        assert self._model_dir is not None
        self._metrics = None
        metrics_path = self._model_dir / "metrics.json"
        if metrics_path.exists():
            try:
                self._metrics = json.loads(metrics_path.read_text())
            except Exception:  # noqa: BLE001
                self._metrics = None

    # ---------------------------------------------------------------- render

    def _refresh(self) -> None:
        if self._model_dir is None:
            return

        # Header / stats
        name = self._model_dir.name
        cfg_path = self._model_dir / "training_config.json"
        stats_lines: list[str] = []
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                acc = cfg.get("best_val_acc")
                n_train = cfg.get("n_train")
                n_val = cfg.get("n_val")
                classes = cfg.get("class_names")
                if acc is not None:
                    stats_lines.append(f"Best val accuracy: {acc:.1%}")
                if n_train and n_val:
                    stats_lines.append(f"Trained on {n_train:,} (val: {n_val:,})")
                if classes:
                    stats_lines.append(f"Classes: {', '.join(classes)}")
            except Exception:  # noqa: BLE001
                pass

        self._header.setText(f"Model: {name}")
        self._stats.setText("  ·  ".join(stats_lines) if stats_lines else "")

        # Prefer live plot if raw curves are available
        if self._metrics and self._metrics.get("binary"):
            self._canvas.show()
            self._png_scroll.hide()
            self._refresh_live_plot()
        else:
            # Fallback to the rendered PNG, if any
            png = self._model_dir / "roc_pr_curves.png"
            if png.exists():
                pix = QPixmap(str(png))
                pix = pix.scaledToHeight(
                    self._png_label.height(),
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._png_label.setPixmap(pix)
                self._png_label.setFixedWidth(pix.width())
                self._png_scroll.show()
            else:
                self._png_scroll.hide()
            self._canvas.hide()

    def _refresh_live_plot(self) -> None:
        if not self._metrics or "binary" not in self._metrics:
            return
        b = self._metrics["binary"]
        fpr   = np.asarray(b["roc"]["fpr"])
        tpr   = np.asarray(b["roc"]["tpr"])
        thr_r = np.asarray(b["roc"]["thresholds"])
        auc_v = b["roc"]["auc"]

        prec  = np.asarray(b["pr"]["precision"])
        rec   = np.asarray(b["pr"]["recall"])
        thr_p = np.asarray(b["pr"]["thresholds"])
        ap_v  = b["pr"]["ap"]
        base  = b["pr"]["baseline"]

        t = self._threshold
        # Nearest operating point on each curve
        idx_roc = int(np.argmin(np.abs(thr_r - t))) if len(thr_r) else 0
        # Note: PR thresholds have len = len(precision) - 1
        idx_pr = int(np.argmin(np.abs(thr_p - t))) if len(thr_p) else 0
        idx_pr_clamped = min(idx_pr, len(prec) - 1)

        self._fig.clear()
        ax_roc, ax_pr = self._fig.subplots(1, 2)

        for ax in (ax_roc, ax_pr):
            ax.tick_params(labelsize=7)

        # ROC
        ax_roc.plot(fpr, tpr, color="#2166ac", lw=1.2, label=f"AUC {auc_v:.2f}")
        ax_roc.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.4)
        ax_roc.plot(fpr[idx_roc], tpr[idx_roc], "o", ms=5, color="#e0402b",
                    label=f"τ {t:.2f}")
        if "optimal_threshold" in b:
            idx_opt = int(np.argmin(np.abs(thr_r - b["optimal_threshold"])))
            ax_roc.plot(fpr[idx_opt], tpr[idx_opt], "^", ms=5, color="#2ca02c",
                        label=f"opt {b['optimal_threshold']:.2f}")
        ax_roc.set_xlabel("FPR", fontsize=8); ax_roc.set_ylabel("TPR", fontsize=8)
        ax_roc.set_title("ROC", fontsize=8)
        ax_roc.grid(True, alpha=0.2)
        ax_roc.legend(loc="lower right", fontsize=7, frameon=False)

        # PR
        ax_pr.plot(rec, prec, color="#b2182b", lw=1.2, label=f"AP {ap_v:.2f}")
        ax_pr.axhline(base, color="k", ls="--", lw=0.7, alpha=0.4)
        ax_pr.plot(rec[idx_pr_clamped], prec[idx_pr_clamped], "o", ms=5, color="#e0402b",
                   label=f"τ {t:.2f}")
        ax_pr.set_xlabel("Recall", fontsize=8); ax_pr.set_ylabel("Precision", fontsize=8)
        ax_pr.set_title("PR", fontsize=8)
        ax_pr.grid(True, alpha=0.2)
        ax_pr.legend(loc="lower left", fontsize=7, frameon=False)

        self._canvas.draw_idle()
