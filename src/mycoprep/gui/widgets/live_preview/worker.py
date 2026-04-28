"""Background worker that runs the focus → segment → classify → features
chain on a single FOV.

Each stage checks the request's prior ``CacheEntry`` keys and skips if the
cache is still valid. Results are emitted as they're produced so the
preview can repaint progressively (focus first, mask next, labels last).

Phase 2 wires up segment + classify only. The focus step in this phase
just loads the focused image from disk (the user must have run the real
Focus stage already). Phase 3 replaces that with a single-FOV in-memory
focus shim.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from .cache import (
    CacheEntry,
    classify_key,
    disk_focus_key,
    features_key,
    segment_key,
)


class CancelledError(RuntimeError):
    """Raised in the worker body when the controller asks for a stop."""


@dataclass
class RenderRequest:
    """Everything the worker needs to render one FOV's chain.

    The worker treats this as immutable. The controller always
    constructs a fresh request per render.
    """

    sample_path: Path
    fov_index: int
    target_stage: str          # "focus" | "segment" | "classify" | "features"
    # ``int`` once known. ``None`` (or a string label) is the
    # "auto-detect" sentinel from the InputPanel — the worker
    # resolves it from the image data the first time focus runs
    # and mutates this field to the resolved int so downstream
    # stages see a concrete index.
    phase_channel: Any = None

    # Live-tunable options. ``None`` means "this stage isn't configured
    # / not requested for this run". Phase 2 always passes segment_opts.
    focus_opts: Any = None
    segment_opts: Any = None
    classify_opts: Any = None  # None when classification should be skipped
    features_opts: Any = None

    # Phase 2 hint — read focus output off disk rather than running a
    # single-FOV in-memory focus. Phase 3 will set this False on the
    # Focus tab so the worker actually re-focuses.
    use_disk_focus: bool = True

    # User-edited channel names from the Input panel (e.g. ``["Phase",
    # "GFP", "RFP"]``). Threaded into the FocusPayload so the panel
    # can label per-channel overlay rows with real names instead of
    # ``C0`` / ``C1``.
    channel_labels: Optional[list[str]] = None

    # Optional segmentation crop ``(x0, y0, x1, y1)`` in image pixel
    # coordinates. When set, segment / classify / features run only on
    # the cropped region; the resulting mask is padded back to the full
    # image so downstream coordinates (centroids, bbox) stay aligned
    # with the displayed phase plane.
    roi: Optional[tuple[int, int, int, int]] = None

    # The cache entry currently associated with (sample, fov). The
    # worker reads its `*_key` fields to decide which stages can be
    # skipped, but never mutates it.
    cached_entry: CacheEntry = field(default_factory=CacheEntry)


# Order in which stages execute and the names emitted on signals.
STAGE_ORDER = ("focus", "segment", "classify", "features")


def _stage_idx(name: str) -> int:
    try:
        return STAGE_ORDER.index(name)
    except ValueError:
        return -1


@dataclass
class FocusPayload:
    key: tuple
    phase: np.ndarray
    image_channels: np.ndarray  # (C, Y, X)
    channel_names: Optional[list[str]] = None
    # The phase channel index resolved from the actual image data,
    # in case the request asked for auto-detection.
    resolved_phase_channel: Optional[int] = None


@dataclass
class SegmentPayload:
    key: tuple
    mask: np.ndarray
    n_cells: int


@dataclass
class ClassifyPayload:
    key: tuple
    decisions: dict[int, str]


@dataclass
class FeaturesPayload:
    key: tuple
    df: Any  # pandas DataFrame; kept untyped to avoid the import here


class PreviewWorker(QThread):
    """Run the chain on a background thread.

    The controller cancels in-flight workers by calling
    :meth:`request_stop` (cooperative — checked between stages). After
    cancellation the worker emits ``cancelled()`` and exits cleanly.
    """

    # ``payload`` is one of the *Payload classes above.
    stageStarted = pyqtSignal(str)
    stageFinished = pyqtSignal(str, object)
    stageFailed = pyqtSignal(str, str)        # stage_name, error_message
    chainFinished = pyqtSignal()
    cancelled = pyqtSignal()

    # Cellpose models are expensive to construct (~3-5 s) — share one
    # across worker instances per (model_type, gpu) so re-renders don't
    # re-pay that cost. Cleared when the app exits.
    _cellpose_cache: dict[tuple[str, bool], Any] = {}

    def __init__(self, request: RenderRequest, parent=None) -> None:
        super().__init__(parent)
        self._req = request
        self._stop = threading.Event()

    # ----------------------------------------------------------------- API

    def request_stop(self) -> None:
        self._stop.set()

    # -------------------------------------------------------------- run loop

    def run(self) -> None:
        req = self._req
        try:
            target = _stage_idx(req.target_stage)
            if target < 0:
                return

            # FOCUS ───────────────────────────────────────────────────
            self._check_cancel()
            try:
                focus_payload = self._maybe_run_focus()
            except Exception as exc:  # noqa: BLE001
                self.stageFailed.emit("focus", str(exc))
                self.chainFinished.emit()
                return
            if focus_payload is not None:
                self.stageFinished.emit("focus", focus_payload)
            phase, channels = self._resolve_focus_arrays(focus_payload)
            if phase is None:
                self.chainFinished.emit()
                return
            if not isinstance(req.phase_channel, int):
                if focus_payload is not None and focus_payload.resolved_phase_channel is not None:
                    req.phase_channel = focus_payload.resolved_phase_channel
                elif req.cached_entry.resolved_phase_channel is not None:
                    req.phase_channel = req.cached_entry.resolved_phase_channel
                elif channels is not None:
                    from mycoprep.core.focus.channel_id import detect_phase_channel
                    req.phase_channel = int(detect_phase_channel(channels[None, ...]))
            if target == 0:
                self.chainFinished.emit()
                return

            # SEGMENT ─────────────────────────────────────────────────
            self._check_cancel()
            try:
                seg_payload = self._maybe_run_segment(phase, channels)
            except Exception as exc:  # noqa: BLE001
                self.stageFailed.emit("segment", str(exc))
                self.chainFinished.emit()
                return
            if seg_payload is not None:
                self.stageFinished.emit("segment", seg_payload)
            mask = self._resolve_mask(seg_payload)
            if target == 1:
                self.chainFinished.emit()
                return

            # CLASSIFY ────────────────────────────────────────────────
            if req.classify_opts is not None and mask is not None and mask.max() > 0:
                self._check_cancel()
                try:
                    cls_payload = self._maybe_run_classify(channels, mask)
                except Exception as exc:  # noqa: BLE001
                    self.stageFailed.emit("classify", str(exc))
                    self.chainFinished.emit()
                    return
                if cls_payload is not None:
                    self.stageFinished.emit("classify", cls_payload)
            if target == 2:
                self.chainFinished.emit()
                return

            # FEATURES ────────────────────────────────────────────────
            if req.features_opts is not None and mask is not None and mask.max() > 0:
                self._check_cancel()
                try:
                    feat_payload = self._maybe_run_features(channels, mask)
                except Exception as exc:  # noqa: BLE001
                    self.stageFailed.emit("features", str(exc))
                    self.chainFinished.emit()
                    return
                if feat_payload is not None:
                    self.stageFinished.emit("features", feat_payload)
            self.chainFinished.emit()

        except CancelledError:
            self.cancelled.emit()

    # ------------------------------------------------------------ stages

    def _maybe_run_focus(self) -> Optional[FocusPayload]:
        """Phase 2: load the focused image from disk if not cached."""
        req = self._req
        if req.use_disk_focus:
            new_key = disk_focus_key(req.sample_path, req.fov_index, req.phase_channel)
        else:
            # Phase 3 will populate this branch.
            from .cache import focus_key as _fk
            new_key = _fk(req.sample_path, req.fov_index, req.focus_opts, req.phase_channel)

        cached = req.cached_entry
        if cached.focus_key == new_key and cached.phase is not None:
            # Cache hit: caller falls back to cached arrays via _resolve_focus_arrays.
            return None

        self.stageStarted.emit("focus")
        if req.use_disk_focus:
            phase, channels = self._load_disk_focus()
            channel_names = list(req.channel_labels) if req.channel_labels else None
        else:
            phase, channels, in_mem_names = self._run_in_memory_focus()
            # Prefer the user's edited InputPanel labels if present —
            # those reflect any renaming the user has done in the GUI.
            channel_names = (
                list(req.channel_labels) if req.channel_labels else in_mem_names
            )
        # ``self._req.phase_channel`` was mutated to the resolved int
        # by the focus shim; expose it on the payload so the controller
        # can stash it in the cache for future renders.
        resolved = req.phase_channel if isinstance(req.phase_channel, int) else None
        return FocusPayload(
            key=new_key,
            phase=phase,
            image_channels=channels,
            channel_names=channel_names,
            resolved_phase_channel=resolved,
        )

    def _maybe_run_segment(self, phase: np.ndarray, channels: np.ndarray
                           ) -> Optional[SegmentPayload]:
        req = self._req
        focus_k = req.cached_entry.focus_key
        # If focus was just produced this render, the controller hasn't
        # written it to the cache yet — the actual key is on the
        # FocusPayload that we just emitted. We re-derive the key here to
        # match what the controller will store.
        if focus_k is None:
            if req.use_disk_focus:
                focus_k = disk_focus_key(req.sample_path, req.fov_index, req.phase_channel)
            else:
                from .cache import focus_key as _fk
                focus_k = _fk(req.sample_path, req.fov_index, req.focus_opts, req.phase_channel)

        new_key = segment_key(focus_k, req.segment_opts, req.phase_channel, req.roi)
        cached = req.cached_entry
        if cached.segment_key == new_key and cached.mask is not None:
            return None

        self.stageStarted.emit("segment")
        from cellpose.models import CellposeModel
        from mycoprep.core.cellpose_pipeline import segment_phase

        model_key = (str(req.segment_opts.model_type), bool(req.segment_opts.gpu))
        model = self._cellpose_cache.get(model_key)
        if model is None:
            model = CellposeModel(gpu=req.segment_opts.gpu, model_type=req.segment_opts.model_type)
            self._cellpose_cache[model_key] = model

        # If the user has set an ROI, crop the phase plane before
        # cellpose runs and pad the resulting mask back into a
        # full-size canvas so labels and overlays stay in image
        # coordinates.
        roi = req.roi
        if roi is not None:
            x0, y0, x1, y1 = roi
            phase_for_seg = phase[y0:y1, x0:x1]
        else:
            phase_for_seg = phase

        try:
            mask_local = segment_phase(
                phase_for_seg, model,
                diameter=req.segment_opts.diameter,
                model_type=req.segment_opts.model_type,
                flow_threshold=req.segment_opts.flow_threshold,
                cellprob_threshold=req.segment_opts.cellprob_threshold,
                min_size=req.segment_opts.min_size,
            )
        except Exception as exc:  # noqa: BLE001
            # Translate known-shape failure modes into actionable messages.
            # The cpsam (Cellpose-SAM) model has fixed-size positional
            # embeddings in its ViT-SAM backbone — when cellpose's
            # auto-rescale produces a non-square or oversized patch grid,
            # ``add_decomposed_rel_pos`` indexes out of range. The fix is
            # outside our codebase, so guide the user to either set a
            # concrete diameter (controls the rescale factor) or switch
            # to a CNN model that doesn't have this limit.
            msg = str(exc)
            looks_like_sam_relpos = (
                "rel_pos" in msg
                or "decomposed_rel_pos" in msg
                or "out of bounds" in msg.lower()
            )
            if (looks_like_sam_relpos
                    and getattr(req.segment_opts, "model_type", "") == "cpsam"):
                friendly = (
                    "Cellpose-SAM (cpsam) couldn't process this image at the "
                    "current settings: its position embeddings overflowed. "
                    "Two ways to fix: (1) set a concrete Diameter in the "
                    "Segment options instead of leaving it on auto, or "
                    "(2) switch the Cellpose model to cyto3 — it uses a "
                    "CNN backbone with no size limit."
                )
                raise RuntimeError(friendly) from exc
            raise

        # Pad the cropped mask back into a full-size canvas so the
        # mask coordinates align with the displayed phase plane. This
        # also keeps classify / features happy — they look up
        # ``mask.shape`` and centroids in image coords.
        if roi is not None:
            x0, y0, x1, y1 = roi
            mask = np.zeros(phase.shape, dtype=mask_local.dtype)
            mask[y0:y1, x0:x1] = mask_local
        else:
            mask = mask_local
        return SegmentPayload(key=new_key, mask=mask, n_cells=int(mask.max()))

    def _maybe_run_classify(self, channels: np.ndarray, mask: np.ndarray
                            ) -> Optional[ClassifyPayload]:
        req = self._req
        # Re-derive the segment key the same way segment did.
        if req.cached_entry.segment_key is not None:
            seg_k = req.cached_entry.segment_key
        else:
            if req.use_disk_focus:
                focus_k = disk_focus_key(req.sample_path, req.fov_index, req.phase_channel)
            else:
                from .cache import focus_key as _fk
                focus_k = _fk(req.sample_path, req.fov_index, req.focus_opts, req.phase_channel)
            seg_k = segment_key(focus_k, req.segment_opts, req.phase_channel, req.roi)
        new_key = classify_key(seg_k, req.classify_opts)
        cached = req.cached_entry
        if cached.classify_key == new_key and cached.decisions is not None:
            return None

        self.stageStarted.emit("classify")
        from mycoprep.core.cell_quality_classifier import classify_and_filter_mask

        opts = req.classify_opts
        _filtered, report = classify_and_filter_mask(
            labeled_mask=mask.astype(int),
            image_channels=channels,
            phase_channel=req.phase_channel,
            model_path=str(opts.model_path) if opts.model_path else None,
            pixels_per_um=getattr(opts, "pixels_per_um", None),
            keep_classes=getattr(opts, "keep_classes", None),
            confidence_threshold=opts.confidence_threshold,
            use_rules=opts.use_rules,
            verbose=False,
        )

        # Build per-label decision map matching the legacy preview's scheme.
        details = report.get("details", {})
        decisions: dict[int, str] = {}
        n_cells = int(mask.max())
        for lbl, (cls_name, conf) in details.items():
            if cls_name in ("good",) and conf >= opts.confidence_threshold:
                decisions[int(lbl)] = "good"
            elif cls_name == "edge_cell":
                decisions[int(lbl)] = "edge"
            elif cls_name == "debris":
                decisions[int(lbl)] = "debris"
            else:
                decisions[int(lbl)] = "bad"
        # Cells the classifier didn't see (e.g. rules-only mode without a model)
        # default to "good" so they show up as kept rather than rejected.
        for lbl in range(1, n_cells + 1):
            decisions.setdefault(int(lbl), "good")
        return ClassifyPayload(key=new_key, decisions=decisions)

    def _maybe_run_features(self, channels: np.ndarray, mask: np.ndarray
                            ) -> Optional[FeaturesPayload]:
        req = self._req
        # Use classify_key when classification ran this round, else
        # segment_key — same fallback the real ExtractStage uses
        # (stages.py:362-365 prefers classify_dir then segment_dir).
        upstream = req.cached_entry.classify_key or req.cached_entry.segment_key
        if upstream is None:
            # Re-derive segment key on cache miss.
            if req.use_disk_focus:
                fk = disk_focus_key(req.sample_path, req.fov_index, req.phase_channel)
            else:
                from .cache import focus_key as _fk
                fk = _fk(req.sample_path, req.fov_index, req.focus_opts, req.phase_channel)
            upstream = segment_key(fk, req.segment_opts, req.phase_channel, req.roi)
        new_key = features_key(upstream, req.features_opts)
        cached = req.cached_entry
        if cached.features_key == new_key and cached.features_df is not None:
            return None

        self.stageStarted.emit("features")
        from .features_single_fov import run as _run_features
        # ``image_channels`` here may include MIP companion channels in
        # disk-focus mode (TIFF on disk); features only cares about the
        # raw image channels, so just hand them over unchanged.
        df = _run_features(
            image_channels=channels,
            mask=mask,
            features_opts=req.features_opts,
            channel_names=req.cached_entry.channel_names or [],
            pixels_per_um=getattr(req.segment_opts, "pixels_per_um", None),
        )
        return FeaturesPayload(key=new_key, df=df)

    # ---------------------------------------------------------- helpers

    def _check_cancel(self) -> None:
        if self._stop.is_set():
            raise CancelledError()

    def _load_disk_focus(self) -> tuple[np.ndarray, np.ndarray]:
        """Phase 2 focus shim — read one FOV from a focused TIFF on disk.

        Resolves the phase channel via skewness if the request didn't
        come in with a concrete int (auto-detect mode), so the right
        plane is rendered as the grayscale base layer. Mutates
        ``self._req.phase_channel`` to the resolved int so downstream
        stages see a concrete index.
        """
        from mycoprep.core.label_cells import load_hyperstack
        data, _meta = load_hyperstack(self._req.sample_path)
        if self._req.fov_index >= data.shape[0]:
            raise IndexError(
                f"FOV {self._req.fov_index} out of range "
                f"(file has {data.shape[0]} FOV(s))"
            )
        channels = data[self._req.fov_index]
        phase_arg = self._req.phase_channel
        if not isinstance(phase_arg, int):
            from mycoprep.core.focus.channel_id import detect_phase_channel
            # ``detect_phase_channel`` expects (Z, C, Y, X); wrap
            # our (C, Y, X) FOV array as a singleton Z stack.
            phase_arg = int(detect_phase_channel(channels[None, ...]))
            self._req.phase_channel = phase_arg
        phase_idx = min(phase_arg, channels.shape[0] - 1)
        return channels[phase_idx], channels

    def _run_in_memory_focus(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Single-FOV focus from a CZI z-stack — mirrors the real
        pipeline's per-scene processing without writing to disk.

        Returns ``(phase_plane, channels_cyx, channel_names)``.
        """
        from .focus_single_fov import run as _run_focus
        planes, names, phase_idx = _run_focus(
            self._req.sample_path,
            self._req.fov_index,
            self._req.focus_opts,
            self._req.phase_channel,
        )
        # Cache the resolved phase index back into the request so segment
        # and classify use the right plane (the user may have asked for
        # auto-detect via phase_channel = -1 or None).
        self._req.phase_channel = int(phase_idx)
        return planes[phase_idx], planes, list(names)

    def _resolve_focus_arrays(self, payload: Optional[FocusPayload]
                              ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if payload is not None:
            return payload.phase, payload.image_channels
        cached = self._req.cached_entry
        return cached.phase, cached.image_channels

    def _resolve_mask(self, payload: Optional[SegmentPayload]
                      ) -> Optional[np.ndarray]:
        if payload is not None:
            return payload.mask
        return self._req.cached_entry.mask
