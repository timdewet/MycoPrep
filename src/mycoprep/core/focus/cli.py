"""FocusPicker command-line entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from . import focus, tiling
from .pipeline import Options, process_czi
from .review import review_czi

_FOCUS_MODES = ("whole", "tiles", "edf")

app = typer.Typer(add_completion=False, help="Pick the in-focus slice from Zeiss CZI Z-stacks.")


@app.callback()
def _main() -> None:
    """FocusPicker — automated in-focus slice selection for Zeiss CZI Z-stacks."""
    # Presence of this callback forces Typer into multi-command mode, so the
    # `process` subcommand must be spelled explicitly on the command line.


def _parse_phase_channel(raw: str | None) -> str | int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


@app.command()
def process(
    czi: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    phase_channel: str | None = typer.Option(
        None,
        "--phase-channel",
        help="Override phase channel by name or index. Default: auto-detect.",
    ),
    metric: str = typer.Option(
        focus.DEFAULT_METRIC,
        "--metric",
        help=f"Focus metric used to pick the best slice. One of: {', '.join(focus.METRIC_NAMES)}",
    ),
    no_archive: bool = typer.Option(
        False, "--no-archive", help="Do not move the original CZI into raw/ on success."
    ),
    crop: float = typer.Option(
        1.0,
        "--crop",
        help="Score only the central fraction of each slice (0<crop<=1). 0.7 ignores edges.",
    ),
    preblur: float = typer.Option(
        0.0, "--preblur", help="Gaussian pre-blur sigma before scoring (suppress pixel noise)."
    ),
    focus_mode: str = typer.Option(
        "whole",
        "--focus-mode",
        help=(
            "How to pick focus across the FOV. "
            "'whole' picks one Z per FOV (default). "
            "'tiles' splits the FOV into a grid (see --tile-grid) and picks one Z per tile. "
            "'edf' picks Z per pixel for an extended depth of field result."
        ),
    ),
    tile_grid: str = typer.Option(
        "2x2",
        "--tile-grid",
        help="Tile grid as NxM (e.g. '2x2', '3x4'). Only used with --focus-mode tiles.",
    ),
) -> None:
    """Process a single CZI: pick best-focus slices and write OME-TIFFs per scene."""
    if metric not in focus.METRIC_NAMES:
        typer.echo(f"unknown metric {metric!r}; choose from {focus.METRIC_NAMES}", err=True)
        raise typer.Exit(code=2)
    if focus_mode not in _FOCUS_MODES:
        typer.echo(
            f"unknown focus mode {focus_mode!r}; choose from {_FOCUS_MODES}", err=True
        )
        raise typer.Exit(code=2)
    try:
        grid = tiling.parse_grid(tile_grid)
    except ValueError as exc:
        typer.echo(f"focuspicker: {exc}", err=True)
        raise typer.Exit(code=2)

    opts = Options(
        phase_channel=_parse_phase_channel(phase_channel),
        metric=metric,
        archive_original=not no_archive,
        crop_fraction=crop,
        preblur_sigma=preblur,
        focus_mode=focus_mode,  # type: ignore[arg-type]
        tile_grid=grid,
    )

    try:
        result = process_czi(czi, opts)
    except Exception as exc:  # surfaced to ZEN via stderr + non-zero exit
        typer.echo(f"focuspicker: failed to process {czi}: {exc}", err=True)
        raise typer.Exit(code=1)

    if not result.scene_results and not result.well_results:
        typer.echo(f"focuspicker: no scenes found in {czi}", err=True)
        raise typer.Exit(code=1)

    if result.well_results:
        typer.echo(
            f"focuspicker: wrote {len(result.well_results)} per-well stacks "
            f"({sum(len(w.scene_indices) for w in result.well_results)} FOVs total)"
        )
        for w in result.well_results:
            typer.echo(
                f"  {w.well}: {len(w.scene_indices)} FOVs -> {w.output_path.name}"
            )
    else:
        for r in result.scene_results:
            typer.echo(
                f"scene {r.scene_index:02d}: chose Z={r.chosen_z} -> {r.output_path.name}"
            )


@app.command()
def review(
    czi: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    n: int = typer.Option(10, "--n", help="Number of scenes to sample."),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output directory. Default: <czi_parent>/focuspicker_review/<czi_stem>/",
    ),
    phase_channel: str | None = typer.Option(
        None, "--phase-channel", help="Override phase channel by name or index."
    ),
    seed: int = typer.Option(0, "--seed", help="RNG seed for scene sampling."),
    crop: float = typer.Option(
        1.0,
        "--crop",
        help="Score only the central fraction of each slice (0<crop<=1). Use 0.7 to ignore edge debris/vignetting.",
    ),
    preblur: float = typer.Option(
        0.0,
        "--preblur",
        help="Gaussian pre-blur sigma applied before scoring. Use ~1.0 to suppress pixel noise.",
    ),
    mask: bool = typer.Option(
        False,
        "--mask/--no-mask",
        help="Restrict scoring to a foreground mask (Otsu on bandpass response). Off by default.",
    ),
    smooth: bool = typer.Option(
        False,
        "--smooth/--no-smooth",
        help="Smooth per-Z score curves with [1,2,1]/4 before picking. Off by default.",
    ),
) -> None:
    """Dump a random sample of scenes and run **every** focus metric in parallel.

    For each sampled scene this writes:
      • the full phase Z-stack as a TIFF (scrub through in Fiji)
      • one chosen-slice TIFF *per metric* (open them all together to compare)
      • a per-scene scores CSV with every metric's score for every slice

    Plus a top-level ``metric_comparison.csv`` summarising which slice each
    metric picked for each FOV. The original CZI is not modified.
    """
    if not 0 < crop <= 1.0:
        typer.echo(f"--crop must be in (0, 1], got {crop}", err=True)
        raise typer.Exit(code=2)
    if preblur < 0:
        typer.echo(f"--preblur must be >= 0, got {preblur}", err=True)
        raise typer.Exit(code=2)

    out_dir = out or (czi.parent / "focuspicker_review" / czi.stem)
    opts = Options(
        phase_channel=_parse_phase_channel(phase_channel),
        archive_original=False,
    )

    try:
        results = review_czi(
            czi,
            out_dir,
            n_scenes=n,
            opts=opts,
            seed=seed,
            crop_fraction=crop,
            preblur_sigma=preblur,
            use_mask=mask,
            smooth_z=smooth,
        )
    except Exception as exc:
        typer.echo(f"focuspicker: review failed for {czi}: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"wrote {len(results)} scenes to {out_dir}")


@app.command()
def label(
    review_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    no_resume: bool = typer.Option(
        False,
        "--no-resume",
        help="Re-label every scene from scratch instead of skipping already-labelled ones.",
    ),
) -> None:
    """Manually mark the best in-focus slice for each FOV in a review directory.

    Reads ``scene*_phase_stack.tif`` files written by ``focuspicker review``
    and opens a matplotlib popup. Use the arrow keys to scrub through Z,
    Enter/Space to accept the current slice, ``n`` to mark the FOV as having
    no in-focus slice, ``b`` to revisit the previous scene, ``s`` to skip,
    and ``q`` to save and quit. Labels are written atomically after every
    confirmation, so progress is never lost.

    The UI never reveals which slice any focus metric chose — labels are
    anchored to your eye, not to the algorithm.
    """
    from .labeling import label_review_dir

    try:
        out_path = label_review_dir(review_dir, resume=not no_resume)
    except Exception as exc:
        typer.echo(f"focuspicker: labelling failed: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"labels saved to {out_path}")


@app.command()
def compare(
    review_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True
    ),
) -> None:
    """Score every focus metric against the manual labels in a review dir.

    Requires that ``focuspicker label`` has already been run on the same
    directory. Prints a ranked table and writes ``metric_evaluation.csv``.
    """
    from .evaluation import evaluate_metrics

    try:
        evaluate_metrics(review_dir)
    except FileNotFoundError as exc:
        typer.echo(f"focuspicker: {exc}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"focuspicker: comparison failed: {exc}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
