"""Persistent feature library for cross-experiment morphological profiling.

Accumulates ``all_features.parquet`` files across pipeline runs so that
clustering and downstream analyses (drug-gene comparisons, etc.) can draw
on an ever-growing reference set.

Storage layout::

    <library_dir>/
        library.parquet          # index / database of registered runs
        runs/
            <run_id_1>.parquet   # cell-level features copied from each run
            <run_id_2>.parquet
            ...
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

_DEFAULT_LIBRARY_DIR = Path.home() / ".mycoprep" / "feature_library"
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
]


class FeatureLibrary:
    """Manage a persistent library of per-run morphological features."""

    def __init__(self, library_dir: Optional[Path] = None) -> None:
        self._dir = Path(library_dir) if library_dir else _DEFAULT_LIBRARY_DIR
        self._runs_dir = self._dir / _RUNS_SUBDIR
        self._index_path = self._dir / _INDEX_FILE

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
