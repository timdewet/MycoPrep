"""Run panel: stage enables, dependency checks, run button, stepper, log."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt6.QtCore import QUrl, pyqtSignal, Qt
from PyQt6.QtGui import QDesktopServices, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..ui import icons, tokens
from ..ui.stepper import StepState, Stepper
from ..widgets.log_view import LogView

STAGE_NAMES = ["Split", "Focus", "Segment", "Classify", "Features"]


class RunPanel(QWidget):
    """Stage toggles, Run button, stepper progress, log."""

    runRequested = pyqtSignal()
    stopRequested = pyqtSignal()
    stageEnablesChanged = pyqtSignal()  # any stage toggle flipped

    # Re-emit signals so MainWindow can drive sidebar status without
    # holding a ref to the runner directly.
    stageRunStarted = pyqtSignal(str)
    stageRunFinished = pyqtSignal(str, int)
    runFinishedAll = pyqtSignal(object)
    runFailedAll = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Stages the runner has decided will actually execute (set when the
        # runner emits stagesPlanned at the start of a run). Until then,
        # fall back to the user's UI checks. This keeps the overall progress
        # bar accurate when a stage is checked but skipped at runtime — e.g.
        # Split is suppressed when Focus is also enabled (Focus subsumes it).
        self._planned_stages: list[str] | None = None
        self.checks: dict[str, QCheckBox] = {}
        for name in STAGE_NAMES:
            cb = QCheckBox(name)
            # Phase A: Features is an opt-in extra step; default it OFF so
            # existing four-stage runs aren't slowed down by feature
            # extraction users haven't asked for.
            cb.setChecked(name != "Features")
            cb.toggled.connect(self._refresh_run_enable)
            cb.toggled.connect(lambda _checked: self.stageEnablesChanged.emit())
            self.checks[name] = cb

        self.stepper = Stepper(STAGE_NAMES)

        self.run_btn = QPushButton("  Run pipeline")
        self.run_btn.setObjectName("primary")
        self.run_btn.setIcon(icons.icon("play", role="on_primary"))
        self.run_btn.clicked.connect(self.runRequested.emit)

        self.stop_btn = QPushButton("  Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setIcon(icons.icon("stop", role="on_primary"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stopRequested.emit)

        self.reuse_existing = QCheckBox("Reuse existing outputs")
        self.reuse_existing.setChecked(True)
        self.reuse_existing.setToolTip(
            "Skip stages that already have output in the target directory."
        )

        self.open_output_btn = QPushButton("  Open output folder")
        self.open_output_btn.setIcon(icons.icon("folder-open", role="text"))
        self.open_output_btn.setEnabled(False)
        self.open_output_btn.setToolTip("Reveal the most recent run's output directory.")
        self.open_output_btn.clicked.connect(self._on_open_output_clicked)
        self._last_manifest_path: Path | None = None

        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("muted")

        self.log = LogView()

        # Log toolbar buttons
        self._copy_log_btn = QPushButton("Copy")
        self._copy_log_btn.setObjectName("ghost")
        self._copy_log_btn.setIcon(icons.icon("copy", role="muted"))
        self._copy_log_btn.clicked.connect(self._copy_log)

        self._clear_log_btn = QPushButton("Clear")
        self._clear_log_btn.setObjectName("ghost")
        self._clear_log_btn.setIcon(icons.icon("clear", role="muted"))
        self._clear_log_btn.clicked.connect(self.log.clear_log)

        self._follow_btn = QPushButton("Follow")
        self._follow_btn.setObjectName("ghost")
        self._follow_btn.setIcon(icons.icon("follow", role="muted"))
        self._follow_btn.setCheckable(True)
        self._follow_btn.setChecked(True)
        self._follow_btn.toggled.connect(self.log.set_follow)

        self._build_ui()
        self._refresh_run_enable()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        # Cards on this tab are one logical settings stack — flush is cleaner.
        root.setSpacing(0)

        # ── Stage toggles card ─────────────────────────────────────────
        stages_box = QGroupBox("Stages to run")
        h = QHBoxLayout(stages_box)
        h.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        h.setSpacing(tokens.S5)
        for name in STAGE_NAMES:
            h.addWidget(self.checks[name])
        h.addStretch(1)
        root.addWidget(stages_box)

        # ── Progress card: overall bar + per-stage stepper ─────────────
        progress_box = QGroupBox("Progress")
        pv = QVBoxLayout(progress_box)
        pv.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        pv.setSpacing(tokens.S3)

        overall_row = QHBoxLayout()
        overall_row.setContentsMargins(0, 0, 0, 0)
        overall_row.setSpacing(tokens.S3)
        overall_label = QLabel("Overall")
        overall_label.setObjectName("h3")
        overall_label.setFixedWidth(72)
        self.overall_bar = QProgressBar()
        self.overall_bar.setRange(0, 1000)
        self.overall_bar.setValue(0)
        self.overall_bar.setFormat("%p%  ·  ready")
        self.overall_bar.setFixedHeight(22)
        overall_row.addWidget(overall_label)
        overall_row.addWidget(self.overall_bar, stretch=1)
        pv.addLayout(overall_row)
        pv.addWidget(self.stepper)
        root.addWidget(progress_box)

        # ── Collapsible log card ──────────────────────────────────────
        self._log_box = QGroupBox("Log")
        lv = QVBoxLayout(self._log_box)
        lv.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        lv.setSpacing(tokens.S2)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(tokens.S1)
        self._collapse_log_btn = QPushButton()
        self._collapse_log_btn.setObjectName("ghost")
        self._collapse_log_btn.setCheckable(True)
        self._collapse_log_btn.setChecked(False)  # expanded by default
        self._collapse_log_btn.setIcon(icons.icon("collapse", role="muted"))
        self._collapse_log_btn.setText(" Hide")
        self._collapse_log_btn.toggled.connect(self._on_log_collapsed)
        toolbar.addWidget(self._collapse_log_btn)
        toolbar.addStretch(1)
        toolbar.addWidget(self._follow_btn)
        toolbar.addWidget(self._copy_log_btn)
        toolbar.addWidget(self._clear_log_btn)
        lv.addLayout(toolbar)
        self._log_body = QWidget()
        body_l = QVBoxLayout(self._log_body)
        body_l.setContentsMargins(0, 0, 0, 0)
        body_l.addWidget(self.log)
        lv.addWidget(self._log_body, stretch=1)
        root.addWidget(self._log_box, stretch=1)

        # ── Sticky action bar (bottom) ─────────────────────────────────
        action_bar = QFrame()
        action_bar.setObjectName("card")
        ab = QHBoxLayout(action_bar)
        ab.setContentsMargins(tokens.S4, tokens.S3, tokens.S4, tokens.S3)
        ab.setSpacing(tokens.S3)
        ab.addWidget(self.run_btn)
        ab.addWidget(self.stop_btn)
        ab.addSpacing(tokens.S3)
        ab.addWidget(self.reuse_existing)
        ab.addSpacing(tokens.S3)
        ab.addWidget(self.open_output_btn)
        ab.addStretch(1)
        ab.addWidget(self.status_label)
        root.addWidget(action_bar)

    # ---------------------------------------------------------------- enables

    def _refresh_run_enable(self) -> None:
        msg = self._dep_issue()
        self.run_btn.setEnabled(msg is None)
        self.run_btn.setToolTip(msg or "Run the enabled stages.")

    def _dep_issue(self) -> str | None:
        c = self.checks
        if not any(cb.isChecked() for cb in c.values()):
            return "At least one stage must be enabled."
        return None

    # ---------------------------------------------------------------- log toolbar

    def _copy_log(self) -> None:
        QGuiApplication.clipboard().setText(self.log.copy_all())

    def _on_log_collapsed(self, hidden: bool) -> None:
        self._log_body.setVisible(not hidden)
        self._collapse_log_btn.setIcon(
            icons.icon("expand" if hidden else "collapse", role="muted")
        )
        self._collapse_log_btn.setText(" Show" if hidden else " Hide")

    # ── overall progress helper ────────────────────────────────────────

    def _enabled_stage_count(self) -> int:
        return max(1, sum(1 for cb in self.checks.values() if cb.isChecked()))

    def _refresh_overall(self, current_name: str | None, current_fraction: float) -> None:
        """Aggregate per-stage progress into a single overall percentage.

        Each planned stage contributes 1/N. Done stages contribute fully;
        the active stage contributes its fraction; pending stages 0.
        """
        if self._planned_stages is not None:
            enabled = list(self._planned_stages)
        else:
            enabled = [n for n, cb in self.checks.items() if cb.isChecked()]
        if not enabled:
            self.overall_bar.setValue(0)
            self.overall_bar.setFormat("%p%  ·  no stages enabled")
            return
        weight = 1.0 / len(enabled)
        total = 0.0
        for n in enabled:
            step = self.stepper._steps.get(n)
            if step is None:
                continue
            if step._state is StepState.DONE:
                total += weight
            elif step._state is StepState.RUNNING and n == current_name:
                total += weight * max(0.0, min(1.0, current_fraction))
            elif step._state is StepState.SKIPPED:
                total += weight  # treat skipped as complete
        total = max(0.0, min(1.0, total))
        self.overall_bar.setValue(int(total * 1000))
        if current_name:
            self.overall_bar.setFormat(f"%p%  ·  {current_name}")
        else:
            self.overall_bar.setFormat("%p%")

    # ---------------------------------------------------------------- runner slots

    def on_stages_planned(self, names: list) -> None:
        """Runner has decided which stages will actually execute."""
        self._planned_stages = [str(n) for n in names]
        self._refresh_overall(None, 0.0)

    def on_stage_started(self, name: str) -> None:
        self.status_label.setText(f"Running: {name}")
        self.stepper.set_state(name, StepState.RUNNING)
        self.stepper.set_fraction(name, 0.0)
        self.log.log(f"▶ {name} started")
        self._refresh_overall(name, 0.0)
        self.stageRunStarted.emit(name)

    def on_stage_progress(self, name: str, fraction: float, message: str) -> None:
        self.stepper.set_fraction(name, fraction)
        self._refresh_overall(name, fraction)
        if message:
            self.log.log(f"  {name}: {message}")

    def on_stage_finished(self, name: str, n_outputs: int) -> None:
        self.stepper.set_state(name, StepState.DONE)
        self.log.log(f"✓ {name} finished — {n_outputs} output(s)", level="success")
        self._refresh_overall(None, 0.0)
        self.stageRunFinished.emit(name, n_outputs)

    def on_run_finished(self, manifest_path: Path) -> None:
        self.status_label.setText(f"Done — {manifest_path.name}")
        self.log.log(f"✓ Run complete. Manifest: {manifest_path}", level="success")
        self._last_manifest_path = Path(manifest_path)
        self.open_output_btn.setEnabled(True)
        self._log_manifest_summary(self._last_manifest_path)
        self.overall_bar.setValue(1000)
        self.overall_bar.setFormat("%p%  ·  done")
        self.runFinishedAll.emit(manifest_path)

    def on_run_failed(self, error: str) -> None:
        self.status_label.setText("Failed")
        self.log.log(f"✗ Run failed: {error}", level="error")
        for name, step in self.stepper._steps.items():
            if step._state is StepState.RUNNING:
                self.stepper.set_state(name, StepState.ERROR)
        self.overall_bar.setFormat("%p%  ·  failed")
        self.runFailedAll.emit(error)

    # ----------------------------------------------------- output folder

    def _on_open_output_clicked(self) -> None:
        target = self._last_manifest_path.parent if self._last_manifest_path else None
        if target is None or not target.exists():
            QMessageBox.warning(
                self, "Folder unavailable",
                "The output folder for the last run is not accessible.",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _log_manifest_summary(self, manifest_path: Path) -> None:
        try:
            data = json.loads(manifest_path.read_text())
        except Exception as e:  # noqa: BLE001
            self.log.log(f"  (could not read manifest: {e})", level="warning")
            return
        stages = data.get("stages") or []
        if not stages:
            return
        total = 0.0
        self.log.log("── Run summary ──")
        for entry in stages:
            name = entry.get("name") or entry.get("stage") or "?"
            elapsed = entry.get("elapsed_s")
            outputs = entry.get("outputs")
            reused = entry.get("reused")
            tag = "reused" if reused else "ran"
            parts = [f"  {name}: {tag}"]
            if isinstance(outputs, list):
                parts.append(f"{len(outputs)} output(s)")
            elif isinstance(outputs, int):
                parts.append(f"{outputs} output(s)")
            if isinstance(elapsed, (int, float)):
                parts.append(f"{elapsed:.1f}s")
                total += float(elapsed)
            self.log.log(" · ".join(parts))
        if total > 0:
            self.log.log(f"  total: {total:.1f}s")

    # ------------------------------------------------------ persistence

    def state(self) -> dict:
        return {
            "stage_enables": {n: cb.isChecked() for n, cb in self.checks.items()},
            "reuse_existing": self.reuse_existing.isChecked(),
        }

    def restore_state(self, s: dict) -> None:
        if not isinstance(s, dict):
            return
        enables = s.get("stage_enables") or {}
        for name, cb in self.checks.items():
            if name in enables:
                cb.setChecked(bool(enables[name]))
        if "reuse_existing" in s:
            self.reuse_existing.setChecked(bool(s["reuse_existing"]))

    # ---------------------------------------------------------------- getters

    def stage_enables(self) -> dict[str, bool]:
        return {name: cb.isChecked() for name, cb in self.checks.items()}
