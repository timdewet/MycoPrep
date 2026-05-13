"""Report per-architecture coverage of cnn_embeddings.parquet.

Cross-architecture comparison in ``library compare-representations`` is
only fair when every model's extraction produced a comparable set of
conditions. Run this script to see whether ResNet-18 / Lightweight /
SupCon variants all cover the same runs and conditions, or whether one
has dropped data.

Usage (from the MycoPrep dir, with the venv active):
    python scripts/inspect_embeddings.py
    python scripts/inspect_embeddings.py --library-dir <path>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--library-dir",
        default=os.path.join(os.path.expanduser("~"), ".mycoprep", "morphology_library"),
        help="Library root (default: ~/.mycoprep/morphology_library)",
    )
    args = ap.parse_args()

    emb_root = Path(args.library_dir) / "models" / "embeddings"
    if not emb_root.exists():
        print(f"No embeddings dir at {emb_root}")
        return

    rows = []
    for sub in sorted(emb_root.iterdir()):
        if not sub.is_dir():
            continue
        p = sub / "cnn_embeddings.parquet"
        if not p.exists():
            print(f"{sub.name}: no cnn_embeddings.parquet")
            continue
        df = pd.read_parquet(p)
        n_rows = len(df)
        n_conditions = (
            int(df["condition_label"].nunique())
            if "condition_label" in df.columns else None
        )
        n_runs = int(df["run_id"].nunique()) if "run_id" in df.columns else None
        n_genes = int(df["gene"].nunique()) if "gene" in df.columns else None
        rows.append({
            "model_type": sub.name,
            "rows": n_rows,
            "runs": n_runs,
            "unique_conditions": n_conditions,
            "unique_genes": n_genes,
            "mtime": pd.Timestamp(p.stat().st_mtime, unit="s"),
        })

    if not rows:
        print("No cnn_embeddings.parquet files found.")
        return

    summary = pd.DataFrame(rows)
    print("=== Per-architecture coverage ===")
    print(summary.to_string(index=False))

    # Per-run breakdown for each architecture (so you can see which runs
    # are missing from any model's extraction).
    print("\n=== Per-run condition counts ===")
    per_run_frames = []
    for sub in sorted(emb_root.iterdir()):
        p = sub / "cnn_embeddings.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "run_id" not in df.columns or "condition_label" not in df.columns:
            continue
        counts = df.groupby("run_id")["condition_label"].nunique()
        counts.name = sub.name
        per_run_frames.append(counts.to_frame())
    if per_run_frames:
        merged = pd.concat(per_run_frames, axis=1).fillna(0).astype(int)
        print(merged.to_string())
    else:
        print("(no per-run breakdown available)")

    # Sample condition_label strings per architecture so format / parsing
    # differences are visible without guessing.
    print("\n=== Sample condition_label strings (first 5 unique) ===")
    label_sets: dict[str, set[str]] = {}
    for sub in sorted(emb_root.iterdir()):
        p = sub / "cnn_embeddings.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "condition_label" not in df.columns:
            print(f"{sub.name}: (no condition_label column)")
            continue
        labels = sorted(df["condition_label"].astype(str).unique())
        label_sets[sub.name] = set(labels)
        print(f"\n{sub.name} ({len(labels)} unique):")
        for s in labels[:5]:
            print(f"    {s!r}")
        if len(labels) > 5:
            print(f"    ... and {len(labels) - 5} more")

    # Pairwise intersection counts to see which architectures share labels.
    if len(label_sets) > 1:
        print("\n=== Pairwise condition_label intersection ===")
        names = sorted(label_sets.keys())
        rows = []
        for a in names:
            row = {"_": a}
            for b in names:
                row[b] = len(label_sets[a] & label_sets[b])
            rows.append(row)
        print(pd.DataFrame(rows).set_index("_").to_string())


if __name__ == "__main__":
    main()
