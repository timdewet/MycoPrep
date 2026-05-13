"""Remove one or more trained models from the library so they get
re-trained / re-extracted on the next compare-representations run.

Deletes (idempotent — missing files are silently skipped):
  - The model manifest entries for each ``--model-type``.
  - The ``<library>/models/<name>.pth`` checkpoint files those entries pointed at.
  - The ``<library>/models/embeddings/<model_type>/`` directory (embeddings
    parquet + cached OT sidecars).

Lightweight / SupCon-lightweight variants are kept untouched so the next
compare-representations call only retrains what was dropped.

Usage:
    python scripts/drop_model.py --model-type resnet18 supcon_resnet18
    python scripts/drop_model.py --model-type resnet18 --library-dir <path>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mycoprep.core.extract.feature_library import FeatureLibrary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-type",
        nargs="+",
        required=True,
        choices=["resnet18", "lightweight", "supcon_resnet18", "supcon_lightweight"],
        help="One or more model_types to drop (e.g. resnet18 supcon_resnet18).",
    )
    ap.add_argument(
        "--library-dir",
        default=None,
        help="Library directory (default: ~/.mycoprep/morphology_library).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without making changes.",
    )
    args = ap.parse_args()

    lib = FeatureLibrary(Path(args.library_dir) if args.library_dir else None)
    to_drop = set(args.model_type)

    # Manifest entries.
    entries = lib._read_manifest()
    dropped_entries = [e for e in entries if e.get("model_type") in to_drop]
    kept_entries = [e for e in entries if e.get("model_type") not in to_drop]

    if not dropped_entries:
        print(f"No manifest entries match {sorted(to_drop)}.")
    else:
        print(f"Manifest entries to drop ({len(dropped_entries)}):")
        for e in dropped_entries:
            print(f"  - {e.get('model_name')} ({e.get('model_type')})")

    # .pth checkpoints those entries pointed at.
    pth_paths: list[Path] = []
    for e in dropped_entries:
        rel = e.get("model_path")
        if rel:
            p = lib.library_dir / rel
            if p.exists():
                pth_paths.append(p)

    # Embedding directories.
    emb_dirs: list[Path] = []
    for mt in to_drop:
        d = lib.models_dir / "embeddings" / mt
        if d.exists():
            emb_dirs.append(d)

    print(f"\n.pth checkpoints to delete ({len(pth_paths)}):")
    for p in pth_paths:
        print(f"  - {p}")
    print(f"\nEmbedding directories to delete ({len(emb_dirs)}):")
    for d in emb_dirs:
        print(f"  - {d}")

    if args.dry_run:
        print("\n(dry run — no changes made)")
        return

    if not (dropped_entries or pth_paths or emb_dirs):
        print("\nNothing to do.")
        return

    # Apply.
    if dropped_entries:
        lib._write_manifest(kept_entries)
        print(f"\nWrote {lib._manifest_path()} with {len(kept_entries)} remaining entries.")
    for p in pth_paths:
        p.unlink()
        print(f"  rm {p}")
    for d in emb_dirs:
        shutil.rmtree(d)
        print(f"  rm -r {d}")

    print(
        "\nDone. On the next compare-representations run, the dropped "
        "model_types will be re-trained on the current library."
    )


if __name__ == "__main__":
    main()
