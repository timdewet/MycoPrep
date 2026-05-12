"""Generate every representation needed for the comparison diagnostic.

Idempotently ensures the artefacts the evaluator needs exist on disk:

- CNN encoder checkpoints (one per model_type) — trains missing ones in
  library-mode using all registered crop H5 files.
- ``cnn_embeddings.parquet`` per architecture, under
  ``<library>/models/embeddings/<model_type>/`` (canonical location).
- OT distance-matrix sidecars **for both Harmony states**:
    * Harmony on → canonical cache (shared with the GUI).
    * Harmony off → side-cache under ``<out_dir>/ot_matrices/`` so the
      canonical cache isn't clobbered.
- Features-OT sidecars, also for both Harmony states.

Mean-source representations (feature means / CNN-embedding means) do not
need precomputation — :mod:`representation_eval` builds them at score time.

All training / extraction / OT-render work is delegated to existing
functions in :mod:`autoencoder.train`, :mod:`autoencoder.supcon`,
:mod:`autoencoder.extract`, and :mod:`extract.qc_plots`. This module
just routes paths and skips work whose output already exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

ProgressCB = Optional[Callable[[float, str], None]]


# ─────────────────────────────────────────────────────────────────────────
# Plan / report dataclasses
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationPlan:
    species: str
    model_types: list[str] = field(
        default_factory=lambda: [
            "resnet18", "lightweight",
            "supcon_resnet18", "supcon_lightweight",
        ]
    )
    batch_correct_states: tuple[bool, ...] = (True, False)
    compute_features_ot: bool = True
    compute_embedding_ot: bool = True
    retrain: bool = False
    epochs: int = 50
    batch_size: int = 64


@dataclass
class ArtefactReport:
    kind: str          # 'model' | 'embeddings' | 'ot_features' | 'ot_embeddings'
    model_type: str    # '' for ot_features
    batch_correct: Optional[bool]
    path: Optional[Path]
    built: bool
    cached: bool
    note: str = ""


@dataclass
class GenerationReport:
    species: str
    artefacts: list[ArtefactReport] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[ArtefactReport]:
        return [a for a in self.artefacts if a.kind == kind]


# ─────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────

def _emb_dir_for(library_dir: Optional[Path], model_type: str) -> Path:
    from .feature_library import FeatureLibrary
    return FeatureLibrary(library_dir).models_dir / "embeddings" / model_type


def _bc_off_emb_ot_sidecar(out_dir: Path, model_type: str) -> Path:
    return out_dir / "ot_matrices" / "embeddings" / f"{model_type}__bc_off.ot_distance.parquet"


def _bc_off_emb_ot_render_path(out_dir: Path, model_type: str) -> Path:
    # render_embeddings_ot_html writes its sidecar next to this html path
    # via _ot_sidecar_path(out_path) — we point both at the bc-off subdir.
    return out_dir / "ot_matrices" / "embeddings" / f"{model_type}__bc_off.html"


def _bc_off_features_ot_sidecar(out_dir: Path, species: str) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in (species or "all"))
    return out_dir / "ot_matrices" / "features" / f"{safe}__bc_off.ot_distance.parquet"


def _bc_off_features_ot_render_path(out_dir: Path, species: str) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in (species or "all"))
    return out_dir / "ot_matrices" / "features" / f"{safe}__bc_off.html"


def canonical_emb_ot_sidecar(emb_parquet: Path) -> Path:
    """Return the canonical (BC=True) OT cache path for an embeddings parquet.

    Mirrors :func:`qc_plots._embedding_ot_cache_path`.
    """
    return emb_parquet.with_name(emb_parquet.stem + ".ot_default.ot_distance.parquet")


def canonical_features_ot_sidecar(library_dir: Optional[Path], species: str) -> Path:
    """Return the canonical (BC=True) features-OT cache path.

    Mirrors :func:`qc_plots._features_ot_cache_path`.
    """
    from .feature_library import FeatureLibrary
    lib = FeatureLibrary(library_dir)
    cache_dir = lib.library_dir / "features_ot_cache"
    safe = "".join(c if c.isalnum() else "_" for c in (species or "all"))
    return cache_dir / f"{safe}.ot_distance.parquet"


# ─────────────────────────────────────────────────────────────────────────
# Steps
# ─────────────────────────────────────────────────────────────────────────

def _ensure_model_and_embeddings(
    library_dir: Optional[Path],
    species: str,
    model_type: str,
    *,
    epochs: int,
    batch_size: int,
    retrain: bool,
    progress_cb: ProgressCB,
) -> tuple[Optional[Path], ArtefactReport, ArtefactReport]:
    """Train (if missing) and extract embeddings (if missing) for ``model_type``.

    Returns ``(emb_parquet_path, model_report, emb_report)``. The
    embeddings parquet path is None when generation failed.
    """
    from ..autoencoder import (
        AutoencoderConfig, extract_embeddings, train_autoencoder,
    )
    from ..autoencoder.supcon import train_supcon
    from .feature_library import FeatureLibrary

    lib = FeatureLibrary(library_dir)

    # ── 1. model checkpoint ────────────────────────────────────────────
    latest = None if retrain else lib.latest_model(model_type=model_type)
    model_built = False
    model_path: Optional[Path] = None

    if latest is not None:
        model_path = lib.get_model_path(latest["model_name"])
        if not (model_path and model_path.exists()):
            latest = None
            model_path = None

    if latest is None:
        # Need to train. Gather library crops for the requested species.
        h5_paths_with_runs = lib.crop_h5_paths_with_run_ids(species=species or None)
        if not h5_paths_with_runs:
            return None, ArtefactReport(
                kind="model", model_type=model_type, batch_correct=None,
                path=None, built=False, cached=False,
                note="no library crops available for training",
            ), ArtefactReport(
                kind="embeddings", model_type=model_type, batch_correct=None,
                path=None, built=False, cached=False,
                note="skipped: no model",
            )
        h5_paths = [p for _, p in h5_paths_with_runs]
        run_ids_per_file = [rid for rid, _ in h5_paths_with_runs]

        config = AutoencoderConfig(
            model_type=model_type,
            epochs=int(epochs),
            batch_size=int(batch_size),
        )
        is_supcon = config.is_supcon
        trainer = train_supcon if is_supcon else train_autoencoder
        emb_dir = _emb_dir_for(library_dir, model_type)
        emb_dir.mkdir(parents=True, exist_ok=True)

        if progress_cb:
            progress_cb(0.0, f"Training {model_type} on library crops ({epochs} epochs)...")
        try:
            summary = trainer(
                h5_paths, emb_dir, config,
                progress_cb=(
                    (lambda f, m: progress_cb(f * 0.6, m)) if progress_cb else None
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return None, ArtefactReport(
                kind="model", model_type=model_type, batch_correct=None,
                path=None, built=False, cached=False,
                note=f"training failed: {exc}",
            ), ArtefactReport(
                kind="embeddings", model_type=model_type, batch_correct=None,
                path=None, built=False, cached=False, note="skipped: no model",
            )
        model_path = Path(summary["model_path"])
        registered = lib.register_model(
            model_name=f"{model_type}_eval",
            model_path=model_path,
            model_type=model_type,
            run_ids=run_ids_per_file,
            epochs=int(epochs),
            val_loss=float(summary.get("best_val_loss", 0.0)),
            config=config.__dict__,
        )
        model_path = registered
        model_built = True
        model_report = ArtefactReport(
            kind="model", model_type=model_type, batch_correct=None,
            path=model_path, built=True, cached=False,
        )
    else:
        model_report = ArtefactReport(
            kind="model", model_type=model_type, batch_correct=None,
            path=model_path, built=False, cached=True,
            note=f"reusing manifest entry {latest['model_name']!r}",
        )

    # ── 2. embeddings parquet ──────────────────────────────────────────
    emb_dir = _emb_dir_for(library_dir, model_type)
    emb_parquet = emb_dir / "cnn_embeddings.parquet"

    if emb_parquet.exists() and not retrain and not model_built:
        emb_report = ArtefactReport(
            kind="embeddings", model_type=model_type, batch_correct=None,
            path=emb_parquet, built=False, cached=True,
        )
        return emb_parquet, model_report, emb_report

    # Need to extract.
    config = AutoencoderConfig(model_type=model_type, batch_size=int(batch_size))
    h5_paths_with_runs = lib.crop_h5_paths_with_run_ids(species=species or None)
    if not h5_paths_with_runs:
        emb_report = ArtefactReport(
            kind="embeddings", model_type=model_type, batch_correct=None,
            path=None, built=False, cached=False,
            note="no library crops available for extraction",
        )
        return None, model_report, emb_report
    h5_paths = [p for _, p in h5_paths_with_runs]
    run_ids_per_file = [rid for rid, _ in h5_paths_with_runs]

    emb_dir.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(0.6, f"Extracting {model_type} embeddings...")
    try:
        emb_parquet = extract_embeddings(
            h5_paths, model_path, emb_dir, config,
            progress_cb=(
                (lambda f, m: progress_cb(0.6 + f * 0.3, m)) if progress_cb else None
            ),
            run_ids_per_file=run_ids_per_file,
        )
    except Exception as exc:  # noqa: BLE001
        emb_report = ArtefactReport(
            kind="embeddings", model_type=model_type, batch_correct=None,
            path=None, built=False, cached=False,
            note=f"extraction failed: {exc}",
        )
        return None, model_report, emb_report

    emb_report = ArtefactReport(
        kind="embeddings", model_type=model_type, batch_correct=None,
        path=Path(emb_parquet), built=True, cached=False,
    )
    return Path(emb_parquet), model_report, emb_report


def _ensure_embedding_ot(
    library_dir: Optional[Path],
    species: str,
    model_type: str,
    emb_parquet: Path,
    *,
    batch_correct: bool,
    out_dir: Path,
    retrain: bool,
    progress_cb: ProgressCB,
) -> ArtefactReport:
    """Ensure the (model_type, batch_correct) OT sidecar exists."""
    from .qc_plots import (
        precompute_embedding_ot_cache, render_embeddings_ot_html,
    )

    if batch_correct:
        sidecar = canonical_emb_ot_sidecar(emb_parquet)
        if sidecar.exists() and not retrain:
            return ArtefactReport(
                kind="ot_embeddings", model_type=model_type,
                batch_correct=True, path=sidecar, built=False, cached=True,
            )
        if progress_cb:
            progress_cb(0.0, f"Precomputing OT (BC=on) for {model_type}...")
        try:
            precompute_embedding_ot_cache(
                emb_parquet,
                library_dir=library_dir,
                species=species,
                model_type=model_type,
                batch_correct=True,
                progress_cb=(
                    (lambda f, *_a: progress_cb(f, "OT precompute")) if progress_cb else None
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return ArtefactReport(
                kind="ot_embeddings", model_type=model_type,
                batch_correct=True, path=None, built=False, cached=False,
                note=f"OT precompute failed: {exc}",
            )
        return ArtefactReport(
            kind="ot_embeddings", model_type=model_type,
            batch_correct=True,
            path=sidecar if sidecar.exists() else None,
            built=sidecar.exists(),
            cached=False,
            note="" if sidecar.exists() else "precompute reported success but sidecar missing",
        )

    # BC=off — diagnostic-side path.
    sidecar = _bc_off_emb_ot_sidecar(out_dir, model_type)
    if sidecar.exists() and not retrain:
        return ArtefactReport(
            kind="ot_embeddings", model_type=model_type,
            batch_correct=False, path=sidecar, built=False, cached=True,
        )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    render_out = _bc_off_emb_ot_render_path(out_dir, model_type)
    if progress_cb:
        progress_cb(0.0, f"Precomputing OT (BC=off) for {model_type}...")
    try:
        render_embeddings_ot_html(
            render_out,
            library_dir=library_dir,
            species=species,
            model_type=model_type,
            batch_correct=False,
            progress_cb=(
                (lambda f, *_a: progress_cb(f, "OT BC=off")) if progress_cb else None
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return ArtefactReport(
            kind="ot_embeddings", model_type=model_type,
            batch_correct=False, path=None, built=False, cached=False,
            note=f"OT BC=off render failed: {exc}",
        )
    # render_embeddings_ot_html writes its sidecar next to render_out via
    # _ot_sidecar_path(render_out) — same suffix logic as our path helper.
    return ArtefactReport(
        kind="ot_embeddings", model_type=model_type,
        batch_correct=False,
        path=sidecar if sidecar.exists() else None,
        built=sidecar.exists(),
        cached=False,
        note="" if sidecar.exists() else "BC=off render reported success but sidecar missing",
    )


def _ensure_features_ot(
    library_dir: Optional[Path],
    species: str,
    *,
    batch_correct: bool,
    out_dir: Path,
    retrain: bool,
    progress_cb: ProgressCB,
) -> ArtefactReport:
    """Ensure the (features, batch_correct) OT sidecar exists."""
    from .qc_plots import (
        precompute_features_ot_cache, render_features_ot_html,
    )

    if batch_correct:
        sidecar = canonical_features_ot_sidecar(library_dir, species)
        if sidecar.exists() and not retrain:
            return ArtefactReport(
                kind="ot_features", model_type="", batch_correct=True,
                path=sidecar, built=False, cached=True,
            )
        if progress_cb:
            progress_cb(0.0, "Precomputing features-OT (BC=on)...")
        try:
            precompute_features_ot_cache(
                library_dir=library_dir,
                species=species,
                batch_correct=True,
                progress_cb=(
                    (lambda f, *_a: progress_cb(f, "features-OT")) if progress_cb else None
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return ArtefactReport(
                kind="ot_features", model_type="", batch_correct=True,
                path=None, built=False, cached=False,
                note=f"features-OT precompute failed: {exc}",
            )
        return ArtefactReport(
            kind="ot_features", model_type="", batch_correct=True,
            path=sidecar if sidecar.exists() else None,
            built=sidecar.exists(),
            cached=False,
        )

    sidecar = _bc_off_features_ot_sidecar(out_dir, species)
    if sidecar.exists() and not retrain:
        return ArtefactReport(
            kind="ot_features", model_type="", batch_correct=False,
            path=sidecar, built=False, cached=True,
        )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    render_out = _bc_off_features_ot_render_path(out_dir, species)
    if progress_cb:
        progress_cb(0.0, "Precomputing features-OT (BC=off)...")
    try:
        render_features_ot_html(
            render_out,
            library_dir=library_dir,
            species=species,
            batch_correct=False,
            progress_cb=(
                (lambda f, *_a: progress_cb(f, "features-OT BC=off")) if progress_cb else None
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return ArtefactReport(
            kind="ot_features", model_type="", batch_correct=False,
            path=None, built=False, cached=False,
            note=f"features-OT BC=off render failed: {exc}",
        )
    return ArtefactReport(
        kind="ot_features", model_type="", batch_correct=False,
        path=sidecar if sidecar.exists() else None,
        built=sidecar.exists(),
        cached=False,
    )


# ─────────────────────────────────────────────────────────────────────────
# Public orchestrator
# ─────────────────────────────────────────────────────────────────────────

def ensure_all_representations(
    library_dir: Optional[Path],
    plan: GenerationPlan,
    *,
    out_dir: Path,
    progress_cb: ProgressCB = None,
) -> GenerationReport:
    """Generate every artefact the evaluator needs for ``plan``.

    Idempotent: each step is skipped when its output already exists,
    unless ``plan.retrain`` is set. Failures are recorded in the
    returned :class:`GenerationReport` as ``ArtefactReport`` entries
    with ``built=False, cached=False, note=...``; the orchestrator
    continues with the remaining steps.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = GenerationReport(species=plan.species)

    # Per-model: train + extract + OT (both BC states).
    n_models = max(1, len(plan.model_types))
    for i, mt in enumerate(plan.model_types):
        def _sub_cb(f, msg, *, base=i / n_models, span=1.0 / n_models):
            if progress_cb:
                progress_cb(base + f * span, f"[{mt}] {msg}")

        emb_path, m_rep, e_rep = _ensure_model_and_embeddings(
            library_dir, plan.species, mt,
            epochs=plan.epochs,
            batch_size=plan.batch_size,
            retrain=plan.retrain,
            progress_cb=_sub_cb,
        )
        report.artefacts.append(m_rep)
        report.artefacts.append(e_rep)

        if emb_path and plan.compute_embedding_ot:
            for bc in plan.batch_correct_states:
                ot_rep = _ensure_embedding_ot(
                    library_dir, plan.species, mt, emb_path,
                    batch_correct=bc,
                    out_dir=out_dir,
                    retrain=plan.retrain,
                    progress_cb=_sub_cb,
                )
                report.artefacts.append(ot_rep)

    # Features-OT for both BC states. Allocate the trailing 5% of overall
    # progress equally between the BC states so the per-Sinkhorn-iteration
    # fraction reported by the inner callback actually advances the bar
    # instead of repeatedly logging the same hardcoded percentage.
    if plan.compute_features_ot:
        n_bcs = max(1, len(plan.batch_correct_states))
        for bi, bc in enumerate(plan.batch_correct_states):
            bc_lo = 0.95 + (bi / n_bcs) * 0.05
            bc_hi = 0.95 + ((bi + 1) / n_bcs) * 0.05

            def _sub_cb(f, msg, *, _bc=bc, _lo=bc_lo, _hi=bc_hi):
                if progress_cb:
                    clamped = max(0.0, min(1.0, float(f)))
                    progress_cb(
                        _lo + clamped * (_hi - _lo),
                        f"[features-OT BC={'on' if _bc else 'off'}] {msg}",
                    )

            ot_rep = _ensure_features_ot(
                library_dir, plan.species,
                batch_correct=bc,
                out_dir=out_dir,
                retrain=plan.retrain,
                progress_cb=_sub_cb,
            )
            report.artefacts.append(ot_rep)

    return report
