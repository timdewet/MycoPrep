"""Shared plate-layout model used across all pipeline stages.

A PlateLayout is a pandas DataFrame with a fixed schema. It is the
single source of truth for condition labels: it is constructed once
(bootstrapped from a CZI, imported from CSV, or built from scratch in
the visual editor) and then passed to every downstream stage via the
RunContext, so labels never drift between stages.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# Fixed column order. Additional columns are allowed but won't drive stage outputs.
# ``source_czi`` is the filename (not full path) of the CZI a well's scenes
# came from, used by the multi-CZI plate flow to dispatch each well to the
# right CZI at Split/Focus time. Empty for legacy single-CZI layouts.
COLUMNS = [
    "well", "scene_indices", "condition", "reporter",
    "mutant_or_drug", "replica", "notes", "source_czi",
]


# ─────────────────────────────────────────────────────────────────────────────
# Well-ID helpers (mirror mycoprep.core.split_czi_plate so naming stays in sync)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_well_id(well: str) -> str:
    """'A01' → 'A1', 'a1' → 'A1'."""
    well = str(well).strip().upper()
    m = re.match(r"^([A-Z])0*(\d+)$", well)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return well


def well_sort_key(well: str):
    m = re.match(r"^([A-Z])(\d+)$", well)
    if m:
        return (m.group(1), int(m.group(2)))
    return (well, 0)


def infer_plate_shape(wells: list[str]) -> tuple[int, int]:
    """Infer (rows, cols) from a list of wells. Defaults to 96-well (8×12)."""
    if not wells:
        return 8, 12
    max_row = 0
    max_col = 0
    for w in wells:
        m = re.match(r"^([A-Z])(\d+)$", normalize_well_id(w))
        if m:
            max_row = max(max_row, ord(m.group(1)) - ord("A") + 1)
            max_col = max(max_col, int(m.group(2)))
    # Snap up to the common plate sizes
    for rows, cols in [(2, 3), (4, 6), (6, 8), (8, 12), (16, 24), (32, 48)]:
        if max_row <= rows and max_col <= cols:
            return rows, cols
    return max_row, max_col


def well_bounds(wells: list[str]) -> tuple[str, str, int, int]:
    """Return (min_row_letter, max_row_letter, min_col, max_col) over a set of wells."""
    if not wells:
        return "A", "A", 1, 1
    rows: list[int] = []
    cols: list[int] = []
    for w in wells:
        m = re.match(r"^([A-Z])(\d+)$", normalize_well_id(w))
        if m:
            rows.append(ord(m.group(1)) - ord("A"))
            cols.append(int(m.group(2)))
    return chr(ord("A") + min(rows)), chr(ord("A") + max(rows)), min(cols), max(cols)


def all_wells(rows: int, cols: int) -> list[str]:
    return [f"{chr(ord('A') + r)}{c + 1}" for r in range(rows) for c in range(cols)]


# ─────────────────────────────────────────────────────────────────────────────
# PlateLayout
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlateLayout:
    """Wraps a pandas DataFrame with the required columns + plate dimensions.

    `rows`/`cols` describe the *displayed* grid bounds (which can be a
    sub-region of a full plate). `row_offset` is the letter of the first
    visible row (e.g. 'C'); `col_offset` is the number of the first
    visible column (e.g. 1).
    """

    df: pd.DataFrame
    rows: int = 8
    cols: int = 12
    row_offset: str = "A"
    col_offset: int = 1

    # ------------------------------------------------------------------ build

    @classmethod
    def empty(cls, rows: int = 8, cols: int = 12) -> "PlateLayout":
        wells = all_wells(rows, cols)
        df = pd.DataFrame({
            "well": wells,
            "scene_indices": [[] for _ in wells],
            "condition": [""] * len(wells),
            "reporter": [""] * len(wells),
            "mutant_or_drug": [""] * len(wells),
            "replica": [""] * len(wells),
            "notes": [""] * len(wells),
            "source_czi": [""] * len(wells),
        })
        return cls(df=df, rows=rows, cols=cols)

    @classmethod
    def from_czi(cls, czi_path: Path) -> "PlateLayout":
        """Single-CZI shim around :meth:`from_czis`. Kept for callers that
        only ever pass one path."""
        return cls.from_czis([czi_path])

    @classmethod
    def from_czis(
        cls,
        czi_paths: list[Path],
        on_conflict=None,
    ) -> "PlateLayout":
        """Bootstrap from one or more plate CZIs.

        Wells from each CZI are read via that file's scene-well metadata;
        ``source_czi`` is set to the CZI's filename so the runner can
        dispatch each well to the right file at Split/Focus time. The
        displayed grid is the bounding box of the union of all wells.

        If a well_id appears in more than one CZI, the first-added CZI
        wins and the conflict is reported via ``on_conflict(well, kept_czi,
        skipped_czi)`` (default: silent — caller decides how to surface).
        """
        try:
            from mycoprep.core.split_czi_plate import extract_scene_well_map
        except ImportError as e:
            raise RuntimeError(
                "mycoprep.core not importable. Reinstall the project: `pip install -e .`."
            ) from e

        if not czi_paths:
            return cls.empty()

        # Per-well aggregation: {well_id: (scene_indices, source_czi_filename)}.
        per_well: dict[str, tuple[list[int], str]] = {}
        for path in czi_paths:
            path = Path(path)
            try:
                scene_well_map = extract_scene_well_map(path)
            except Exception:  # noqa: BLE001
                # Non-plate CZI (no scene metadata) — silently skip.
                continue
            for scene_idx, well in scene_well_map.items():
                well_norm = normalize_well_id(well)
                if well_norm in per_well:
                    existing_czi = per_well[well_norm][1]
                    # Cross-CZI conflict: a different CZI already owns
                    # this well — first-CZI wins, surface via callback.
                    if existing_czi != path.name:
                        if on_conflict is not None:
                            on_conflict(well_norm, existing_czi, path.name)
                        continue
                    # Same CZI, multi-position well (e.g. A1-1, A1-2):
                    # accumulate the additional scene index so the live
                    # preview can navigate every position.
                per_well.setdefault(well_norm, ([], path.name))[0].append(int(scene_idx))

        if not per_well:
            return cls.empty()

        detected_wells = list(per_well.keys())
        min_r, max_r, min_c, max_c = well_bounds(detected_wells)
        n_rows = ord(max_r) - ord(min_r) + 1
        n_cols = max_c - min_c + 1

        wells: list[str] = []
        for r in range(n_rows):
            for c in range(n_cols):
                wells.append(f"{chr(ord(min_r) + r)}{min_c + c}")

        scene_lists: list[list[int]] = []
        sources: list[str] = []
        for w in wells:
            entry = per_well.get(w)
            if entry is None:
                scene_lists.append([])
                sources.append("")
            else:
                scene_lists.append(sorted(entry[0]))
                sources.append(entry[1])

        df = pd.DataFrame({
            "well": wells,
            "scene_indices": scene_lists,
            "condition": [""] * len(wells),
            "reporter": [""] * len(wells),
            "mutant_or_drug": [""] * len(wells),
            "replica": [""] * len(wells),
            "notes": [""] * len(wells),
            "source_czi": sources,
        })
        return cls(df=df, rows=n_rows, cols=n_cols, row_offset=min_r, col_offset=min_c)

    @classmethod
    def from_csv(cls, csv_path: Path) -> "PlateLayout":
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        if "well" not in df.columns:
            raise ValueError("CSV missing required 'well' column.")
        df["well"] = df["well"].map(normalize_well_id)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = [] if col == "scene_indices" else ""
        if "scene_indices" in df.columns and df["scene_indices"].dtype == object:
            df["scene_indices"] = df["scene_indices"].map(_parse_scene_list)

        wells_in_csv = df["well"].tolist()
        if not wells_in_csv:
            return cls.empty()

        # Grid is the bounding box of the CSV's wells — same policy as from_czi
        # so imported layouts render as a tight grid rather than padding out to
        # a full plate shape.
        min_r, max_r, min_c, max_c = well_bounds(wells_in_csv)
        n_rows = ord(max_r) - ord(min_r) + 1
        n_cols = max_c - min_c + 1

        # Build the full bounded grid; merge in CSV rows keyed by well.
        wells: list[str] = [
            f"{chr(ord(min_r) + r)}{min_c + c}"
            for r in range(n_rows) for c in range(n_cols)
        ]
        base = pd.DataFrame({
            "well": wells,
            "scene_indices": [[] for _ in wells],
            "condition": [""] * len(wells),
            "reporter": [""] * len(wells),
            "mutant_or_drug": [""] * len(wells),
            "replica": [""] * len(wells),
            "notes": [""] * len(wells),
        })
        # Overlay CSV values (use .at so list-typed scene_indices doesn't get
        # pandas-broadcast-coerced).
        indexed = df.set_index("well")
        well_to_idx = {w: i for i, w in enumerate(base["well"])}
        for well in wells:
            if well not in indexed.index:
                continue
            src = indexed.loc[well]
            row_idx = well_to_idx[well]
            for col in COLUMNS:
                if col == "well":
                    continue
                default = [] if col == "scene_indices" else ""
                base.at[row_idx, col] = src.get(col, default)

        return cls(df=base[COLUMNS].copy(), rows=n_rows, cols=n_cols,
                   row_offset=min_r, col_offset=min_c)

    # -------------------------------------------------------------------- io

    def to_csv(self, path: Path) -> None:
        df = self.df.copy()
        df["scene_indices"] = df["scene_indices"].map(
            lambda xs: ";".join(str(x) for x in xs) if isinstance(xs, (list, tuple)) else str(xs)
        )
        df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)

    # ------------------------------------------------------------- mutations

    def set_wells(self, wells: list[str], **fields: str) -> None:
        """Apply field values to a set of wells in-place."""
        wells_norm = {normalize_well_id(w) for w in wells}
        for col, value in fields.items():
            if col not in self.df.columns:
                raise KeyError(f"Unknown column '{col}'")
            mask = self.df["well"].isin(wells_norm)
            self.df.loc[mask, col] = value

    # ---------------------------------------------------------------- views

    def has_data(self, well: str) -> bool:
        """True if this well participates in the layout.

        A well "has data" if it has scene indices from the source CZI OR has
        any label assigned (condition / reporter / mutant / replica). The
        second case matters when a layout is imported from CSV without
        scene_indices — the wells are still meaningful.
        """
        rows = self.df[self.df["well"] == normalize_well_id(well)]
        if rows.empty:
            return False
        row = rows.iloc[0]
        scenes = row["scene_indices"]
        if isinstance(scenes, (list, tuple)) and len(scenes) > 0:
            return True
        for col in ("condition", "reporter", "mutant_or_drug", "replica"):
            if str(row.get(col, "")).strip():
                return True
        return False

    def active_rows(self) -> pd.DataFrame:
        """Rows that have at least a condition set — what stages actually use."""
        return self.df[self.df["condition"].astype(str).str.len() > 0].copy()

    def merge_labels_from(self, source: "PlateLayout") -> int:
        """Overlay labels from another layout onto this one, matching on
        well_id. Returns the number of wells whose labels were updated.

        Used when the user imports a CSV after CZIs have been loaded: the
        CSV is the source of truth for ``condition`` / ``reporter`` /
        ``mutant_or_drug`` / ``replica`` / ``notes``, but the live CZI
        metadata (``scene_indices``, ``source_czi``) on ``self`` must not
        be clobbered — otherwise wells from CZIs that the CSV doesn't
        mention would lose their scene metadata.

        Wells present in ``source`` but not in ``self`` are silently
        ignored — we have no CZI scene metadata for them, so they
        couldn't be processed downstream anyway.
        """
        if source is None or source.df is None or source.df.empty:
            return 0
        label_cols = ["condition", "reporter", "mutant_or_drug", "replica", "notes"]
        src_df = source.df.set_index("well", drop=False)
        # Drop duplicate well-IDs in the source so .at[] doesn't return Series.
        src_df = src_df[~src_df.index.duplicated(keep="first")]
        n_updated = 0
        for i, well_id in enumerate(self.df["well"].astype(str)):
            if well_id not in src_df.index:
                continue
            for col in label_cols:
                if col in src_df.columns:
                    val = src_df.at[well_id, col]
                    if pd.isna(val):
                        val = ""
                    self.df.at[i, col] = str(val)
            n_updated += 1
        return n_updated

    def disambiguated_active_rows(self) -> pd.DataFrame:
        """``active_rows()``, but auto-numbers groups of wells that share
        the same ``(condition, reporter, mutant_or_drug, replica)`` so each
        well gets a unique output filename without the user having to
        manually renumber controls.

        Groups of size 1 keep their literal labels. For groups of size > 1,
        a ``_1, _2, ...`` suffix is appended to ``mutant_or_drug`` in
        well-ID order (A1 → ``NT_1``, B5 → ``NT_2``, ...). The user's
        literal labels are preserved on the original ``df``.

        Stages building per-well filenames (Split, Focus, rename pass)
        should call this; ``active_rows()`` is for callers that genuinely
        need the unrenumbered labels (e.g. surfacing them in a UI).
        """
        df = self.active_rows().copy().reset_index(drop=True)
        if df.empty:
            return df
        key_cols = ["condition", "reporter", "mutant_or_drug", "replica"]
        df["_ws"] = df["well"].apply(well_sort_key)
        df = df.sort_values("_ws").reset_index(drop=True)
        df["_grp_size"] = df.groupby(key_cols)["well"].transform("size")
        df["_grp_idx"] = df.groupby(key_cols).cumcount() + 1
        mask = df["_grp_size"] > 1
        df.loc[mask, "mutant_or_drug"] = (
            df.loc[mask, "mutant_or_drug"].astype(str)
            + "_"
            + df.loc[mask, "_grp_idx"].astype(str)
        )
        return df.drop(columns=["_ws", "_grp_size", "_grp_idx"])

    def validate(self) -> list[str]:
        """Return a list of human-readable validation issues (empty = OK).

        Filename collisions among wells with identical labels (e.g. several
        non-targeting controls all labelled ``NT``) are NOT flagged here —
        they auto-disambiguate at filename-build time via
        :meth:`disambiguated_active_rows`. We only flag the empty-layout
        case, which is a hard blocker.
        """
        issues = []
        active = self.active_rows()
        if active.empty:
            issues.append("No wells have a condition assigned — nothing to process.")
        return issues


def _parse_scene_list(value) -> list[int]:
    if isinstance(value, list):
        return value
    if not value or str(value).strip() == "":
        return []
    s = str(value).strip().strip("[]")
    parts = re.split(r"[;,\s]+", s)
    out = []
    for p in parts:
        if p == "":
            continue
        try:
            out.append(int(p))
        except ValueError:
            pass
    return out
