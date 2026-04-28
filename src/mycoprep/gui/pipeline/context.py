"""RunContext — immutable-ish bag of state passed from the GUI to the runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from mycoprep.core.api import ClassifyOpts, ExtractOpts, FocusOpts, SegmentOpts

from .layout import PlateLayout


@dataclass
class RunContext:
    czi_path: Path
    output_dir: Path
    layout: PlateLayout
    # Multi-CZI plate support: when the user has selected several CZIs that
    # together compose one logical plate (e.g. row-band split across files),
    # ``czi_paths`` carries the full list. Single-CZI runs leave this as a
    # one-element list and ``czi_path`` keeps its meaning for legacy code.
    czi_paths: list[Path] = field(default_factory=list)

    # Stage enables
    do_split: bool = True
    do_focus: bool = True
    do_segment: bool = True
    do_classify: bool = True
    do_features: bool = False

    # Per-stage options
    focus_opts: FocusOpts = field(default_factory=FocusOpts)
    segment_opts: SegmentOpts = field(default_factory=SegmentOpts)
    classify_opts: ClassifyOpts = field(default_factory=ClassifyOpts)
    features_opts: ExtractOpts = field(default_factory=ExtractOpts)

    # Channel handling (resolved early so all downstream stages agree)
    phase_channel: Optional[int] = None
    channel_labels: Optional[list[str]] = None

    # Reuse-existing flag — set by the runner so stages can do per-input
    # reuse logic (e.g. a multi-CZI plate where one CZI's wells are already
    # processed but a newly-added CZI still needs to run).
    reuse_existing: bool = False

    # ------------------------------------------------------------------ paths

    @property
    def all_czi_paths(self) -> list[Path]:
        """Canonical accessor: returns the full list of CZI paths whether
        the run was constructed with the legacy single-path or the new
        ``czi_paths`` list. Stages should call this rather than touching
        either field directly."""
        if self.czi_paths:
            return list(self.czi_paths)
        if self.czi_path:
            return [self.czi_path]
        return []

    @property
    def split_dir(self) -> Path:
        return self.output_dir / "01_split"

    @property
    def focus_dir(self) -> Path:
        # Split is skipped when Focus is on (Focus produces the per-well output
        # directly). In that case we merge the two stage roles into a single
        # folder so the output hierarchy doesn't advertise a missing step.
        if self.do_split and self.do_focus:
            return self.output_dir / "01_split_and_focused"
        return self.output_dir / "01_focus"

    @property
    def segment_dir(self) -> Path:
        return self.output_dir / "02_segment"

    @property
    def classify_dir(self) -> Path:
        return self.output_dir / "03_classify"

    @property
    def features_dir(self) -> Path:
        return self.output_dir / "04_features"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "run_manifest.json"


@dataclass
class BulkRunContext:
    """Run context for non-plate (Single file or Bulk) modes.

    A list of (czi_path, label-fields) entries that all share the same
    output directory and stage options. The Focus stage is invoked once
    per entry; Segment and Classify run once each over the resulting
    per-CZI TIFFs.
    """

    czi_entries: list[dict[str, Any]]
    output_dir: Path

    # Stage enables (Split is unused in bulk mode — each CZI is one sample).
    do_focus: bool = True
    do_segment: bool = True
    do_classify: bool = True
    do_features: bool = False

    # Per-stage options (shared across all CZIs in the batch)
    focus_opts: FocusOpts = field(default_factory=FocusOpts)
    segment_opts: SegmentOpts = field(default_factory=SegmentOpts)
    classify_opts: ClassifyOpts = field(default_factory=ClassifyOpts)
    features_opts: ExtractOpts = field(default_factory=ExtractOpts)

    phase_channel: Optional[int] = None
    channel_labels: Optional[list[str]] = None

    # ── duck-type compatibility with RunContext ──────────────────────
    # SegmentStage and ClassifyStage read these; provide stable values so
    # the existing stage code can be reused without branching on context
    # type.
    do_split: bool = False     # bulk mode never splits
    reuse_existing: bool = False   # set by BulkPipelineRunner from its own flag

    @property
    def split_dir(self) -> Path:
        # Path that never exists, so SegmentStage's split-fallback skips it.
        return self.output_dir / "_split_unused_in_bulk_mode"

    # Same flat directory layout as the plate flow so downstream tools
    # don't have to special-case anything.
    @property
    def focus_dir(self) -> Path:
        return self.output_dir / "01_focus"

    @property
    def segment_dir(self) -> Path:
        return self.output_dir / "02_segment"

    @property
    def classify_dir(self) -> Path:
        return self.output_dir / "03_classify"

    @property
    def features_dir(self) -> Path:
        return self.output_dir / "04_features"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "run_manifest.json"
