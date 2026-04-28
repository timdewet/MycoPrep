"""Move processed CZIs into a sibling ``raw/`` directory."""

from __future__ import annotations

from pathlib import Path


def move_to_raw(czi_path: Path) -> Path:
    """Move ``czi_path`` into ``<parent>/raw/<name>``.

    Refuses to clobber an existing file at the destination.
    """
    czi_path = Path(czi_path)
    raw_dir = czi_path.parent / "raw"
    raw_dir.mkdir(exist_ok=True)
    dest = raw_dir / czi_path.name
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {dest}")
    czi_path.rename(dest)
    return dest
