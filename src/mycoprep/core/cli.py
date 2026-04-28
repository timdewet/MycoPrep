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


if __name__ == "__main__":  # pragma: no cover
    app()
