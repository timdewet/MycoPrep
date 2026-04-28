"""Pandas-backed model for non-plate (bulk) CZI runs.

One row per CZI file. Mirrors the public surface of `PlateLayout` so the
rest of the pipeline (filename templates, `active_rows()`, `validate()`)
behaves the same way regardless of source.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

COLUMNS = [
    "czi_path",
    "condition",
    "reporter",
    "mutant_or_drug",
    "replica",
    "notes",
]


def _output_filename(condition: str, reporter: str, mutant: str, replica: str) -> str:
    """The same template all stage adapters expect (matches PlateLayout)."""
    parts = [condition, reporter, mutant]
    parts = [p.replace(" ", "_") for p in parts]
    if str(replica).strip():
        parts.append(f"R{str(replica).strip()}")
    return "__".join(parts) + ".tif"


@dataclass
class BulkLayout:
    df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=COLUMNS))

    # ----------------------------------------------------------------- builders

    @classmethod
    def empty(cls) -> "BulkLayout":
        return cls(df=pd.DataFrame(columns=COLUMNS))

    @classmethod
    def from_csv(cls, csv_path: Path) -> "BulkLayout":
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = ""
        return cls(df=df[COLUMNS].copy())

    # ------------------------------------------------------------------ writers

    def to_csv(self, csv_path: Path) -> None:
        self.df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    # ------------------------------------------------------------ row mutations

    def add_files(self, paths: list[Path]) -> int:
        """Append files that aren't already in the layout. Returns # added."""
        existing = set(self.df["czi_path"].astype(str))
        new = []
        for p in paths:
            sp = str(Path(p))
            if sp in existing:
                continue
            existing.add(sp)
            row = {c: "" for c in COLUMNS}
            row["czi_path"] = sp
            new.append(row)
        if not new:
            return 0
        self.df = pd.concat([self.df, pd.DataFrame(new, columns=COLUMNS)], ignore_index=True)
        return len(new)

    def add_folder(self, folder: Path) -> int:
        """Recursively add every *.czi under `folder`."""
        return self.add_files(sorted(Path(folder).rglob("*.czi")))

    def remove_rows(self, indices: list[int]) -> int:
        if not indices:
            return 0
        keep = [i for i in range(len(self.df)) if i not in set(indices)]
        self.df = self.df.iloc[keep].reset_index(drop=True)
        return len(indices)

    def set_single(self, czi_path: Path, **fields: str) -> None:
        """Single-file mode: collapse to a single row, overwrite labels."""
        row = {c: "" for c in COLUMNS}
        row["czi_path"] = str(Path(czi_path))
        for k, v in fields.items():
            if k in COLUMNS:
                row[k] = v
        self.df = pd.DataFrame([row], columns=COLUMNS)

    # ----------------------------------------------------------------- queries

    def active_rows(self) -> pd.DataFrame:
        """Rows the runner should actually process — those with a condition."""
        if self.df.empty:
            return self.df.copy()
        mask = self.df["condition"].astype(str).str.strip().str.len() > 0
        return self.df[mask].copy()

    def validate(self) -> list[str]:
        """Return human-readable errors that should block a run."""
        issues: list[str] = []
        active = self.active_rows()
        if active.empty:
            issues.append("No CZIs have a condition assigned — nothing to process.")
            return issues

        # Existence check
        for _, r in active.iterrows():
            p = Path(str(r["czi_path"]))
            if not p.exists():
                issues.append(f"CZI not found: {p}")

        # Duplicate-output check (matches PlateLayout's contract)
        seen: dict[str, str] = {}
        for _, r in active.iterrows():
            fname = _output_filename(
                str(r["condition"]),
                str(r["reporter"]),
                str(r["mutant_or_drug"]),
                str(r["replica"]),
            )
            if fname in seen:
                issues.append(
                    f"CZIs '{Path(seen[fname]).name}' and '{Path(r['czi_path']).name}' "
                    f"would produce duplicate output '{fname}'."
                )
            else:
                seen[fname] = str(r["czi_path"])
        return issues

    # ------------------------------------------------------------- conveniences

    def output_label_for(self, row) -> str:
        """Stem (no extension) used as Focus output filename for `row`."""
        fname = _output_filename(
            str(row["condition"]), str(row["reporter"]),
            str(row["mutant_or_drug"]), str(row["replica"]),
        )
        return Path(fname).stem  # drop ".tif"
