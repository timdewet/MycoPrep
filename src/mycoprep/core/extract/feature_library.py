"""Persistent morphology library for cross-experiment morphological profiling.

Accumulates ``all_features.parquet`` files across pipeline runs so that
clustering and downstream analyses (drug-gene comparisons, etc.) can draw
on an ever-growing reference set.  Also manages trained autoencoder models
and their provenance (which runs contributed to each model).

Storage layout::

    <library_dir>/
        library.parquet          # index / database of registered runs
        runs/
            <run_id_1>.parquet   # cell-level features copied from each run
            <run_id_2>.parquet
            ...
        models/
            model_manifest.json  # provenance: which runs trained each model
            <model_name>.pth     # trained autoencoder weights
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

_NEW_DEFAULT_DIR = Path.home() / ".mycoprep" / "morphology_library"
_OLD_DEFAULT_DIR = Path.home() / ".mycoprep" / "feature_library"


def _resolve_default_library_dir() -> Path:
    """Use the new default path, migrating from the old one if needed."""
    if _NEW_DEFAULT_DIR.exists():
        return _NEW_DEFAULT_DIR
    if _OLD_DEFAULT_DIR.exists():
        _OLD_DEFAULT_DIR.rename(_NEW_DEFAULT_DIR)
        return _NEW_DEFAULT_DIR
    return _NEW_DEFAULT_DIR


_DEFAULT_LIBRARY_DIR = _resolve_default_library_dir()
_INDEX_FILE = "library.parquet"
_RUNS_SUBDIR = "runs"

_INDEX_COLUMNS = [
    "run_id",
    "species",
    "experiment_type",
    "condition_labels",
    "control_labels",
    "n_cells",
    "n_conditions",
    "source_czi",
    "plate_acquisition_datetime",
    "date_added",
    "features_file",
    "crops_h5_path",
]


class MorphologyLibrary:
    """Manage a persistent library of per-run morphological features and models."""

    def __init__(self, library_dir: Optional[Path] = None) -> None:
        self._dir = Path(library_dir) if library_dir else _DEFAULT_LIBRARY_DIR
        self._runs_dir = self._dir / _RUNS_SUBDIR
        self._index_path = self._dir / _INDEX_FILE
        self._models_dir = self._dir / "models"

    @property
    def library_dir(self) -> Path:
        return self._dir

    def _ensure_dirs(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._runs_dir.mkdir(exist_ok=True)

    def _read_index(self) -> pd.DataFrame:
        if self._index_path.exists():
            return pd.read_parquet(self._index_path)
        return pd.DataFrame(columns=_INDEX_COLUMNS)

    def _write_index(self, idx: pd.DataFrame) -> None:
        self._ensure_dirs()
        idx.to_parquet(self._index_path, index=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_run(
        self,
        run_id: str,
        features_parquet: Path,
        species: str,
        experiment_type: str,
        source_czi: str = "",
        acquisition_datetime: str = "",
        control_labels: str = "",
        crops_h5_path: str = "",
    ) -> None:
        """Copy run features into the library and update the index."""
        features_parquet = Path(features_parquet)
        if not features_parquet.exists():
            raise FileNotFoundError(features_parquet)

        self._ensure_dirs()
        dest = self._runs_dir / f"{run_id}.parquet"
        shutil.copy2(features_parquet, dest)

        df = pd.read_parquet(features_parquet)
        conditions = _condition_labels(df)

        # Auto-detect crops path if not provided
        if not crops_h5_path:
            candidate = features_parquet.parent / "all_crops.h5"
            if candidate.exists():
                crops_h5_path = str(candidate)

        idx = self._read_index()
        idx = idx[idx["run_id"] != run_id]

        new_row = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "species": species or "unknown",
                    "experiment_type": experiment_type,
                    "condition_labels": ",".join(sorted(conditions)),
                    "control_labels": control_labels or "",
                    "n_cells": len(df),
                    "n_conditions": len(conditions),
                    "source_czi": source_czi,
                    "plate_acquisition_datetime": acquisition_datetime,
                    "date_added": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "features_file": f"{_RUNS_SUBDIR}/{run_id}.parquet",
                    "crops_h5_path": crops_h5_path,
                }
            ]
        )
        idx = pd.concat([idx, new_row], ignore_index=True)
        # Ensure schema parity with new columns when reading older indexes.
        for col in _INDEX_COLUMNS:
            if col not in idx.columns:
                idx[col] = ""
        self._write_index(idx[_INDEX_COLUMNS])

    def load_species(self, species: str) -> pd.DataFrame:
        """Load all cell features for *species* across registered runs."""
        idx = self._read_index()
        if species:
            idx = idx[idx["species"].str.lower() == species.lower()]
        if idx.empty:
            return pd.DataFrame()

        frames = []
        for _, row in idx.iterrows():
            path = self._dir / row["features_file"]
            if path.exists():
                part = pd.read_parquet(path)
                part["_library_run_id"] = row["run_id"]
                part["_library_experiment_type"] = row["experiment_type"]
                frames.append(part)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def list_runs(
        self,
        species: Optional[str] = None,
        experiment_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return the index, optionally filtered."""
        idx = self._read_index()
        if species:
            idx = idx[idx["species"].str.lower() == species.lower()]
        if experiment_type:
            idx = idx[idx["experiment_type"].str.lower() == experiment_type.lower()]
        return idx.reset_index(drop=True)

    def remove_run(self, run_id: str) -> bool:
        """Remove a run from the library. Returns True if found."""
        idx = self._read_index()
        match = idx[idx["run_id"] == run_id]
        if match.empty:
            return False

        for _, row in match.iterrows():
            path = self._dir / row["features_file"]
            if path.exists():
                path.unlink()

        idx = idx[idx["run_id"] != run_id]
        self._write_index(idx)
        return True

    def update_run(
        self,
        run_id: str,
        species: Optional[str] = None,
        experiment_type: Optional[str] = None,
        control_labels: Optional[str] = None,
    ) -> bool:
        """Update species, experiment_type, and/or control_labels.

        Returns True if the run was found and at least one field updated.
        """
        idx = self._read_index()
        mask = idx["run_id"] == run_id
        if not mask.any():
            return False
        if species is not None:
            idx.loc[mask, "species"] = species
        if experiment_type is not None:
            idx.loc[mask, "experiment_type"] = experiment_type
        if control_labels is not None:
            if "control_labels" not in idx.columns:
                idx["control_labels"] = ""
            idx.loc[mask, "control_labels"] = control_labels
        self._write_index(idx)
        return True

    def summary(self) -> pd.DataFrame:
        """Aggregated summary: species x experiment_type counts."""
        idx = self._read_index()
        if idx.empty:
            return pd.DataFrame(
                columns=["species", "experiment_type", "n_runs", "total_cells"]
            )
        return (
            idx.groupby(["species", "experiment_type"])
            .agg(n_runs=("run_id", "size"), total_cells=("n_cells", "sum"))
            .reset_index()
        )

    # ------------------------------------------------------------------
    # Model provenance
    # ------------------------------------------------------------------

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def _manifest_path(self) -> Path:
        return self._models_dir / "model_manifest.json"

    def _read_manifest(self) -> list[dict]:
        path = self._manifest_path()
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return []

    def _write_manifest(self, entries: list[dict]) -> None:
        self._models_dir.mkdir(parents=True, exist_ok=True)
        with open(self._manifest_path(), "w") as f:
            json.dump(entries, f, indent=2)

    def register_model(
        self,
        model_name: str,
        model_path: Path,
        model_type: str,
        run_ids: list[str],
        epochs: int,
        val_loss: float,
        config: Optional[dict] = None,
    ) -> Path:
        """Copy a trained model into the library and update the manifest."""
        self._models_dir.mkdir(parents=True, exist_ok=True)
        dest = self._models_dir / f"{model_name}.pth"
        shutil.copy2(model_path, dest)

        entries = self._read_manifest()
        entries = [e for e in entries if e.get("model_name") != model_name]
        entries.append({
            "model_name": model_name,
            "model_path": str(dest.relative_to(self._dir)),
            "model_type": model_type,
            "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_ids": run_ids,
            "epochs": epochs,
            "final_val_loss": val_loss,
            "config": config or {},
        })
        self._write_manifest(entries)
        return dest

    def update_model_runs(self, model_name: str, new_run_ids: list[str]) -> bool:
        """Append run_ids to an existing model's provenance."""
        entries = self._read_manifest()
        for entry in entries:
            if entry.get("model_name") == model_name:
                existing = set(entry.get("run_ids", []))
                entry["run_ids"] = sorted(existing | set(new_run_ids))
                entry["training_date"] = datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )
                self._write_manifest(entries)
                return True
        return False

    def list_models(self) -> list[dict]:
        """Return all registered model entries."""
        return self._read_manifest()

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Resolve the .pth path for a named model."""
        for entry in self._read_manifest():
            if entry.get("model_name") == model_name:
                return self._dir / entry["model_path"]
        return None

    def latest_model(self, model_type: Optional[str] = None) -> Optional[dict]:
        """Return the most recently trained model entry."""
        entries = self._read_manifest()
        if model_type:
            entries = [e for e in entries if e.get("model_type") == model_type]
        return entries[-1] if entries else None

    def crop_h5_paths(self, species: Optional[str] = None) -> list[Path]:
        """Return paths to all_crops.h5 files for registered runs."""
        return [p for _, p in self.crop_h5_paths_with_run_ids(species=species)]

    def crop_h5_paths_with_run_ids(
        self, species: Optional[str] = None,
    ) -> list[tuple[str, Path]]:
        """Return ``(run_id, path)`` tuples for each registered run's crops.

        Use this instead of :meth:`crop_h5_paths` when you need to know which
        library run each h5 file belongs to — e.g. when computing per-cell
        ``run_id`` for batch correction. Filename stems aren't reliable as
        run identifiers because every run's consolidated h5 is conventionally
        named ``all_crops.h5``.
        """
        idx = self._read_index()
        if species:
            idx = idx[idx["species"].str.lower() == species.lower()]
        out: list[tuple[str, Path]] = []
        for _, row in idx.iterrows():
            run_id = str(row["run_id"])
            stored = row.get("crops_h5_path", "")
            if stored and Path(stored).exists():
                out.append((run_id, Path(stored)))
                continue
            features_path = self._dir / row["features_file"]
            crops_path = features_path.parent / "all_crops.h5"
            if not crops_path.exists():
                run_dir = features_path.parent.parent
                crops_path = run_dir / "04_features" / "all_crops.h5"
            if crops_path.exists():
                out.append((run_id, crops_path))
        return out


# Backward-compat alias
FeatureLibrary = MorphologyLibrary

_FEATURE_DIR_NAMES = {"04_features", "features"}


def derive_run_id_from_parquet(parquet_path: Path) -> str:
    """Best-effort run_id from an ``all_features.parquet`` path.

    Walks past conventional feature-subdir names (``04_features``,
    ``features``) so the result names the *output* directory rather than
    the per-stage subdir.
    """
    p = Path(parquet_path).resolve()
    parent = p.parent
    if parent.name in _FEATURE_DIR_NAMES and parent.parent != parent:
        return parent.parent.name or "imported"
    return parent.name or "imported"


def _condition_labels(df: pd.DataFrame) -> set[str]:
    """Derive condition labels from the well column."""
    if "well" not in df.columns:
        return set()
    parts = df["well"].astype(str).str.split("__", expand=True)
    atc = parts[0].str.replace("_focused", "", regex=False)
    mutant = parts[2] if 2 in parts.columns else df["well"].astype(str)
    return set((mutant + " " + atc).unique())
