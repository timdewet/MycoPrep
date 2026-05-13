"""Inspect and rewrite the ``crops_h5_path`` column in ``library.parquet``.

When you copy a morphology library between machines (e.g. Windows
``C:\\...`` paths → macOS ``/Users/...`` paths, or moving the Dropbox
mount point), the per-run ``crops_h5_path`` values still point at the
old host's locations. The Analysis panel doesn't care — it reads
``runs/<run_id>.parquet`` directly — but training, extraction, and
``compare-representations`` need the H5 files to be reachable.

Usage:

    # Just show what's stored now (no changes):
    python scripts/relink_library.py --library-dir <path>

    # Rewrite paths matching --windows-prefix to --mac-prefix:
    python scripts/relink_library.py --library-dir <path> \\
        --windows-prefix "C:/Users/User/Library/CloudStorage/Dropbox-MMRU" \\
        --mac-prefix    "/Users/timdewet/Library/CloudStorage/Dropbox-MMRU"

The script normalises backslashes to forward slashes before comparing,
so you can paste either form for ``--windows-prefix``.

Pass ``--dry-run`` to preview without writing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _normalise(s: str) -> str:
    return str(s).replace("\\", "/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--library-dir",
        required=True,
        help="Path to the morphology_library directory.",
    )
    ap.add_argument(
        "--windows-prefix",
        default=None,
        help="Prefix to replace in crops_h5_path (Windows-style or normalised).",
    )
    ap.add_argument(
        "--mac-prefix",
        default=None,
        help="Replacement prefix for matching paths.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing.",
    )
    args = ap.parse_args()

    lib_dir = Path(args.library_dir).expanduser().resolve()
    p = lib_dir / "library.parquet"
    if not p.exists():
        raise SystemExit(f"No library.parquet at {p}")

    df = pd.read_parquet(p)
    print(f"\n=== library.parquet: {len(df)} runs ===\n")
    cols = ["run_id", "species", "n_cells", "crops_h5_path"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))

    if not (args.windows_prefix and args.mac_prefix):
        print(
            "\n(no rewrite requested — pass --windows-prefix and --mac-prefix "
            "to translate)"
        )
        return

    win_pref = _normalise(args.windows_prefix).rstrip("/")
    mac_pref = args.mac_prefix.rstrip("/")

    new_paths: list[str] = []
    n_changed = 0
    for raw in df["crops_h5_path"].astype(str).tolist():
        norm = _normalise(raw)
        if norm.startswith(win_pref):
            tail = norm[len(win_pref):]
            new = mac_pref + tail
            new_paths.append(new)
            n_changed += 1
        else:
            new_paths.append(raw)

    print(f"\n=== Rewrite preview ({n_changed}/{len(df)} paths will change) ===\n")
    for raw, new in zip(df["crops_h5_path"].tolist(), new_paths):
        if raw != new:
            print(f"  {raw}\n    → {new}\n")

    if n_changed == 0:
        print(
            "No paths matched --windows-prefix. Check that the prefix you "
            "passed actually appears (case-sensitive) at the start of the "
            "stored paths shown above."
        )
        return

    if args.dry_run:
        print("(dry run — no changes written)")
        return

    df["crops_h5_path"] = new_paths
    df.to_parquet(p, index=False)
    print(f"Wrote {p} with {n_changed} updated paths.")

    # Quick existence check after rewrite.
    missing = [pth for pth in new_paths if pth and not Path(pth).exists()]
    if missing:
        print(
            f"\nWARNING: {len(missing)}/{n_changed} rewritten paths "
            f"still don't exist on disk. First few:"
        )
        for m in missing[:5]:
            print(f"  {m}")
    else:
        print("\nAll rewritten paths exist on disk. ✓")


if __name__ == "__main__":
    main()
