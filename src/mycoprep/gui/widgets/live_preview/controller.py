"""Orchestrator for the live preview.

The :class:`PreviewController` owns:

- a :class:`PreviewCache` of per-FOV stage results
- a single in-flight :class:`PreviewWorker`
- a :class:`QTimer` that debounces a flurry of option-change signals
  into a single render request

Triggers (sample/FOV picked, options changed, tab navigated) call
:meth:`request_render`. The controller decides which stages need to run
based on the live opts vs cached keys, spawns a fresh worker, and
funnels its stage outputs both into the cache and the canvas/panel
callbacks.

Keeps zero references to specific Qt widgets — the panel passes in a
small set of callbacks (``opts_providers``, ``set_phase``, ``set_mask``,
``progress_started`` / ``progress_finished``) so this object stays
testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from .cache import (
    CacheEntry,
    PreviewCache,
)
from .worker import (
    ClassifyPayload,
    FeaturesPayload,
    FocusPayload,
    PreviewWorker,
    RenderRequest,
    SegmentPayload,
)


# How long after the last option-change signal before we kick off a render.
DEFAULT_DEBOUNCE_MS = 300


@dataclass
class OptsProviders:
    """Callbacks the panel installs so the controller can read live opts."""

    focus_opts: Callable[[], Any] = lambda: None
    segment_opts: Callable[[], Any] = lambda: None
    classify_opts: Callable[[], Any] = lambda: None
    features_opts: Callable[[], Any] = lambda: None
    phase_channel: Callable[[], int] = lambda: 0
    # User-edited channel labels from the Input panel ("Phase", "GFP",
    # "RFP", …). Used to label the per-channel overlay rows and the
    # focus payload so the panel doesn't fall back to "C0", "C1".
    channel_labels: Callable[[], Optional[list[str]]] = lambda: None
    # Current segmentation ROI (x0, y0, x1, y1) in image pixel coords,
    # or ``None`` to process the entire image.
    roi: Callable[[], Optional[tuple[int, int, int, int]]] = lambda: None


@dataclass
class CanvasSink:
    """Callbacks invoked on the GUI thread as stage results land.

    ``set_phase`` and ``set_image_channels`` are called when focus
    completes. The panel owns the channel rendering logic — given
    raw ``(C, Y, X)`` arrays, it composes them into the canvas's
    per-layer ``ChannelLayer`` records based on the user's per-channel
    visibility / opacity / color settings.

    ``set_features_df`` lands the per-cell DataFrame from the features
    stage; the panel owns the per-cell text-label rendering.
    """

    set_phase: Callable[[Any], None] = lambda _img: None
    set_image_channels: Callable[[Any, Any, int], None] = (
        lambda _channels, _names, _phase_idx: None
    )
    set_mask: Callable[[Any, Any], None] = lambda _mask, _decisions: None
    set_features_df: Callable[[Any], None] = lambda _df: None
    set_labels: Callable[[Any], None] = lambda _labels: None
    clear: Callable[[], None] = lambda: None


@dataclass
class ProgressSink:
    """Callbacks for spinner/progress UI on the GUI thread."""

    started: Callable[[str], None] = lambda _stage: None
    finished: Callable[[], None] = lambda: None
    failed: Callable[[str, str], None] = lambda _stage, _msg: None


class PreviewController(QObject):
    """Owns cache + worker + debounce timer for the live preview."""

    # Forwarded to the panel for high-level UI feedback.
    renderStateChanged = pyqtSignal(bool)   # True when work is in flight

    def __init__(
        self,
        opts: OptsProviders,
        canvas: CanvasSink,
        progress: ProgressSink,
        debounce_ms: int = DEFAULT_DEBOUNCE_MS,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._opts = opts
        self._canvas = canvas
        self._progress = progress

        self._cache = PreviewCache()
        self._worker: Optional[PreviewWorker] = None
        self._pending_request: Optional[RenderRequest] = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(debounce_ms)
        self._debounce.timeout.connect(self._fire_pending)

        # Selection state — driven by panel.set_selection.
        self._tab: str = "segment"
        self._sample_path: Optional[Path] = None
        self._fov_index: int = 0

    # ----------------------------------------------------------------- API

    def set_selection(self, tab: str, sample_path: Optional[Path],
                      fov_index: int) -> None:
        """Update which FOV the controller is rendering for.

        On a real sample/FOV change we eagerly wipe the stale mask
        boundary overlay and per-cell labels so the canvas doesn't
        show last-FOV's segmentation while the new one is still
        processing. The phase plane is left in place until the new
        focus payload lands — clearing it would just produce a flash
        of black for the typically-brief focus step.
        """
        new_path = Path(sample_path) if sample_path is not None else None
        new_fov = int(fov_index)
        fov_changed = (new_path != self._sample_path) or (new_fov != self._fov_index)
        self._tab = tab
        self._sample_path = new_path
        self._fov_index = new_fov
        if fov_changed:
            self._canvas.set_mask(None, None)
            self._canvas.set_features_df(None)
        self.request_render(reason="selection")

    def request_render(self, reason: str = "") -> None:
        """Schedule a render, debounced; cancels any in-flight worker."""
        if self._sample_path is None:
            self._cancel_worker()
            self._canvas.clear()
            return
        # Build the request now so it captures the live opts at the moment
        # of the trigger, not at fire time. (If the user keeps moving a
        # slider, each trigger overwrites the pending request.)
        self._pending_request = self._build_request()
        self._debounce.start()

    def repaint_cached(self) -> None:
        """Repaint the canvas from the cached entry for the current FOV.

        Called after a UI toggle (e.g. "Show classification colors") so
        the change shows up even if no stage actually re-runs.
        """
        if self._sample_path is None:
            self._canvas.clear()
            return
        entry = self._cache.get(self._sample_path, self._fov_index)
        if entry.phase is not None:
            self._canvas.set_phase(entry.phase)
        if entry.image_channels is not None:
            phase_idx = (
                entry.resolved_phase_channel
                if entry.resolved_phase_channel is not None
                else (self._opts.phase_channel() if isinstance(self._opts.phase_channel(), int) else 0)
            )
            self._canvas.set_image_channels(
                entry.image_channels,
                entry.channel_names or [],
                int(phase_idx),
            )
        if entry.mask is not None:
            self._canvas.set_mask(entry.mask, entry.decisions)
        else:
            self._canvas.set_mask(None, None)
        if entry.features_df is not None:
            self._canvas.set_features_df(entry.features_df)

    def invalidate_cache(self, levels: tuple[str, ...] = ("segment",)) -> None:
        """Drop selected stage results across all cache entries.

        Currently used to reset cache from external "channels changed"
        / "phase channel changed" events. Most stage invalidation is
        implicit via the keying — set_selection / option changes pick a
        new key and the cache miss handles the rest.
        """
        for entry in list(self._cache._entries.values()):
            for lvl in levels:
                if lvl == "focus":
                    entry.focus_key = None
                    entry.phase = None
                    entry.image_channels = None
                if lvl == "segment":
                    entry.segment_key = None
                    entry.mask = None
                if lvl == "classify":
                    entry.classify_key = None
                    entry.decisions = None
                if lvl == "features":
                    entry.features_key = None
                    entry.features_df = None

    # ------------------------------------------------------------ internals

    def _build_request(self) -> RenderRequest:
        # Run the full chain regardless of which tab the user is on,
        # so segmentation + classification appear on the Focus tab
        # too — the user shouldn't have to switch tabs to see whether
        # their focus options yield masks worth keeping. The cache
        # ensures that tweaks to (say) only the focus options don't
        # silently re-run segment + classify on every keystroke.
        target = "features" if self._tab == "features" else "classify"
        # Pick the focus shim based on the sample's file type:
        # focused TIFFs on disk → use the cheap disk-load shim,
        # raw CZIs → run the in-memory single-FOV focus chain. This
        # lets the user preview segmentation off raw CZIs before
        # they've ever run the real Focus stage.
        use_disk_focus = self._is_focused_tiff(self._sample_path)

        classify_opts = self._opts.classify_opts() if target in ("classify", "features") else None
        # ``phase_channel`` may be ``None`` ("auto") or a string label —
        # the worker resolves it to a concrete int from the image data
        # the first time focus loads channels for a given sample.
        raw_phase = self._opts.phase_channel()
        phase_channel: Any = raw_phase if isinstance(raw_phase, (int, str)) else None
        return RenderRequest(
            sample_path=self._sample_path,
            fov_index=self._fov_index,
            target_stage=target,
            phase_channel=phase_channel,
            focus_opts=self._opts.focus_opts(),
            segment_opts=self._opts.segment_opts(),
            classify_opts=classify_opts,
            features_opts=self._opts.features_opts() if target == "features" else None,
            use_disk_focus=use_disk_focus,
            channel_labels=self._opts.channel_labels(),
            roi=self._opts.roi(),
            cached_entry=self._cache.get(self._sample_path, self._fov_index),
        )

    def _effective_phase_index(self, payload: FocusPayload) -> int:
        """Pick the phase channel index to use for canvas rendering.

        Prefers the worker's resolved value (auto-detection result),
        then any explicit int the user picked in the InputPanel,
        finally falls back to 0.
        """
        if payload.resolved_phase_channel is not None:
            return int(payload.resolved_phase_channel)
        raw = self._opts.phase_channel()
        if isinstance(raw, int):
            return raw
        return 0

    @staticmethod
    def _is_focused_tiff(path: Optional[Path]) -> bool:
        """True when the sample is a focused TIFF on disk (vs. a raw CZI).

        The two source types use different focus shims in the worker —
        TIFFs are loaded via ``load_hyperstack`` (cheap) while CZIs go
        through the in-memory single-FOV focus chain.
        """
        if path is None:
            return True
        suffix = path.suffix.lower()
        return suffix in (".tif", ".tiff")

    def _fire_pending(self) -> None:
        if self._pending_request is None:
            return
        request = self._pending_request
        self._pending_request = None
        self._cancel_worker()
        self._start_worker(request)

    def _cancel_worker(self) -> None:
        if self._worker is None:
            return
        self._worker.request_stop()
        # Don't block waiting for the worker — Qt will dispatch its
        # ``cancelled`` / ``chainFinished`` signal when the cellpose call
        # comes back. Detach so a new worker can start.
        try:
            self._worker.stageStarted.disconnect(self._on_stage_started)
            self._worker.stageFinished.disconnect(self._on_stage_finished)
            self._worker.stageFailed.disconnect(self._on_stage_failed)
            self._worker.chainFinished.disconnect(self._on_chain_finished)
            self._worker.cancelled.disconnect(self._on_cancelled)
        except (TypeError, RuntimeError):
            # Already disconnected or deleted; ignore.
            pass
        self._worker = None

    def _start_worker(self, request: RenderRequest) -> None:
        worker = PreviewWorker(request, parent=self)
        worker.stageStarted.connect(self._on_stage_started)
        worker.stageFinished.connect(self._on_stage_finished)
        worker.stageFailed.connect(self._on_stage_failed)
        worker.chainFinished.connect(self._on_chain_finished)
        worker.cancelled.connect(self._on_cancelled)
        self._worker = worker
        self.renderStateChanged.emit(True)
        worker.start()

    # ------------------------------------------------------------- slots

    def _on_stage_started(self, stage: str) -> None:
        self._progress.started(stage)

    def _on_stage_finished(self, stage: str, payload: object) -> None:
        # Land the result both in the cache and on the canvas.
        if self._sample_path is None:
            return
        entry = self._cache.get(self._sample_path, self._fov_index)
        if isinstance(payload, FocusPayload):
            entry.focus_key = payload.key
            entry.phase = payload.phase
            entry.image_channels = payload.image_channels
            entry.channel_names = payload.channel_names
            if payload.resolved_phase_channel is not None:
                entry.resolved_phase_channel = int(payload.resolved_phase_channel)
            self._canvas.set_phase(payload.phase)
            self._canvas.set_image_channels(
                payload.image_channels,
                payload.channel_names or [],
                # Prefer the worker-resolved index (correct when the
                # InputPanel is set to "auto"), falling back to whatever
                # the user explicitly picked.
                self._effective_phase_index(payload),
            )
        elif isinstance(payload, SegmentPayload):
            entry.segment_key = payload.key
            entry.mask = payload.mask
            entry.n_cells = payload.n_cells
            self._canvas.set_mask(payload.mask, entry.decisions)
        elif isinstance(payload, ClassifyPayload):
            entry.classify_key = payload.key
            entry.decisions = payload.decisions
            # Repaint the mask so colors update with classification.
            self._canvas.set_mask(entry.mask, entry.decisions)
        elif isinstance(payload, FeaturesPayload):
            entry.features_key = payload.key
            entry.features_df = payload.df
            self._canvas.set_features_df(payload.df)

    def _on_stage_failed(self, stage: str, msg: str) -> None:
        self._progress.failed(stage, msg)

    def _on_chain_finished(self) -> None:
        self._progress.finished()
        self.renderStateChanged.emit(False)
        # If the user changed something while we were running, fire
        # again now (debounced re-trigger handled elsewhere).
        if self._pending_request is not None:
            self._fire_pending()

    def _on_cancelled(self) -> None:
        self._progress.finished()
        self.renderStateChanged.emit(False)
