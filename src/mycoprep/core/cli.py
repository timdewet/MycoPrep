"""Unified Typer CLI exposing the four pre-processing stages.

The focus subcommand delegates to the existing FocusPicker CLI; the
others call into argparse-based main() functions via sys.argv rewriting.
"""

from __future__ import annotations

import sys

import typer

from .focus.cli import app as focus_app

app = typer.Typer(
    add_completion=False,
    help="Bacterial microscopy pre-processing pipeline.",
)
app.add_typer(focus_app, name="focus", help="Focus picking for CZI Z-stacks.")


def _delegate(module, argv: list[str]) -> None:
    """Call a legacy argparse main() with a rewritten sys.argv."""
    saved = sys.argv
    try:
        sys.argv = [module.__name__] + argv
        module.main()
    finally:
        sys.argv = saved


@app.command("split", help="Split a multi-position CZI plate into per-well TIFFs.")
def split(argv: list[str] = typer.Argument(None)) -> None:  # pragma: no cover
    from . import split_czi_plate as m
    _delegate(m, argv or [])


@app.command("segment", help="Run Cellpose-SAM segmentation on CZI or TIFF input.")
def segment(argv: list[str] = typer.Argument(None)) -> None:  # pragma: no cover
    from . import cellpose_pipeline as m
    _delegate(m, argv or [])


@app.command("review", help="Active-learning review of classifier predictions.")
def review(argv: list[str] = typer.Argument(None)) -> None:  # pragma: no cover
    from . import review_classifications as m
    _delegate(m, argv or [])


@app.command("label", help="Interactive labeling of segmented cells.")
def label(argv: list[str] = typer.Argument(None)) -> None:  # pragma: no cover
    from . import label_cells as m
    _delegate(m, argv or [])


@app.command("train", help="Train a cell-quality classifier.")
def train(argv: list[str] = typer.Argument(None)) -> None:  # pragma: no cover
    from . import train_classifier as m
    _delegate(m, argv or [])


# ── Feature library subcommands ───────────────────────────────────────────────

library_app = typer.Typer(
    add_completion=False,
    help="Manage the persistent morphological feature library.",
)
app.add_typer(library_app, name="library")


@library_app.command("list")
def library_list(
    species: str = typer.Option("", help="Filter by species name."),
    experiment_type: str = typer.Option(
        "", "--type", help="Filter by experiment type (knockdown / drug)."
    ),
    library_dir: str = typer.Option(
        "", "--dir", help="Library directory (default: ~/.mycoprep/feature_library/)."
    ),
) -> None:
    """List runs registered in the feature library."""
    from pathlib import Path
    from .extract.feature_library import FeatureLibrary

    lib = FeatureLibrary(Path(library_dir) if library_dir else None)
    runs = lib.list_runs(
        species=species or None,
        experiment_type=experiment_type or None,
    )
    if runs.empty:
        typer.echo("Library is empty.")
        return
    display_cols = [
        c for c in ["run_id", "species", "experiment_type", "n_cells",
                     "n_conditions", "date_added"]
        if c in runs.columns
    ]
    typer.echo(runs[display_cols].to_string(index=False))


@library_app.command("summary")
def library_summary(
    library_dir: str = typer.Option(
        "", "--dir", help="Library directory."
    ),
) -> None:
    """Show aggregated summary of the feature library."""
    from pathlib import Path
    from .extract.feature_library import FeatureLibrary

    lib = FeatureLibrary(Path(library_dir) if library_dir else None)
    s = lib.summary()
    if s.empty:
        typer.echo("Library is empty.")
        return
    typer.echo(s.to_string(index=False))


@library_app.command("add")
def library_add(
    parquet_path: str = typer.Argument(..., help="Path to all_features.parquet."),
    species: str = typer.Option("", help="Species name (e.g. 'M. tuberculosis')."),
    experiment_type: str = typer.Option(
        "knockdown", "--type", help="Experiment type: knockdown or drug."
    ),
    run_id: str = typer.Option(
        "", help="Run ID. Defaults to parent directory name."
    ),
    library_dir: str = typer.Option(
        "", "--dir", help="Library directory."
    ),
) -> None:
    """Import an existing all_features.parquet into the library."""
    from pathlib import Path
    from .extract.feature_library import (
        FeatureLibrary,
        derive_run_id_from_parquet,
    )

    pq = Path(parquet_path)
    if not pq.exists():
        typer.echo(f"File not found: {pq}")
        raise typer.Exit(code=1)
    rid = run_id or derive_run_id_from_parquet(pq)
    lib = FeatureLibrary(Path(library_dir) if library_dir else None)
    lib.register_run(
        run_id=rid,
        features_parquet=pq,
        species=species,
        experiment_type=experiment_type,
    )
    typer.echo(f"Registered '{rid}' ({species or 'unknown'}, {experiment_type}) in library.")


@library_app.command("update")
def library_update(
    run_id: str = typer.Argument(..., help="Run ID to update."),
    species: str = typer.Option("", help="New species value."),
    experiment_type: str = typer.Option(
        "", "--type", help="New experiment type (knockdown / drug)."
    ),
    library_dir: str = typer.Option(
        "", "--dir", help="Library directory."
    ),
) -> None:
    """Update species and/or experiment type for a registered run."""
    from pathlib import Path
    from .extract.feature_library import FeatureLibrary

    if not species and not experiment_type:
        typer.echo("Provide --species and/or --type to update.")
        raise typer.Exit(code=1)
    lib = FeatureLibrary(Path(library_dir) if library_dir else None)
    if lib.update_run(
        run_id,
        species=species or None,
        experiment_type=experiment_type or None,
    ):
        typer.echo(f"Updated '{run_id}'.")
    else:
        typer.echo(f"Run '{run_id}' not found in library.")
        raise typer.Exit(code=1)


@library_app.command("remove")
def library_remove(
    run_id: str = typer.Argument(..., help="Run ID to remove."),
    library_dir: str = typer.Option(
        "", "--dir", help="Library directory."
    ),
) -> None:
    """Remove a run from the feature library."""
    from pathlib import Path
    from .extract.feature_library import FeatureLibrary

    lib = FeatureLibrary(Path(library_dir) if library_dir else None)
    if lib.remove_run(run_id):
        typer.echo(f"Removed '{run_id}' from library.")
    else:
        typer.echo(f"Run '{run_id}' not found in library.")
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
