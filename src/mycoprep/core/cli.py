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
    controls: str = typer.Option(
        "", "--controls", help="Comma-separated control labels (e.g. 'NT1,NT2,WT')."
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
        control_labels=controls,
    )
    typer.echo(f"Registered '{rid}' ({species or 'unknown'}, {experiment_type}) in library.")


@library_app.command("update")
def library_update(
    run_id: str = typer.Argument(..., help="Run ID to update."),
    species: str = typer.Option("", help="New species value."),
    experiment_type: str = typer.Option(
        "", "--type", help="New experiment type (knockdown / drug)."
    ),
    controls: str = typer.Option(
        None, "--controls",
        help="New control labels (comma-separated). Pass empty string to clear.",
    ),
    library_dir: str = typer.Option(
        "", "--dir", help="Library directory."
    ),
) -> None:
    """Update species, experiment type, and/or control labels for a run."""
    from pathlib import Path
    from .extract.feature_library import FeatureLibrary

    if not species and not experiment_type and controls is None:
        typer.echo("Provide --species, --type, and/or --controls to update.")
        raise typer.Exit(code=1)
    lib = FeatureLibrary(Path(library_dir) if library_dir else None)
    if lib.update_run(
        run_id,
        species=species or None,
        experiment_type=experiment_type or None,
        control_labels=controls,
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


@library_app.command("precompute-ot-views")
def library_precompute_ot_views(
    species: str = typer.Option(
        "", "--species", "-s",
        help="Restrict to one species; default: every species in the library.",
    ),
    models: str = typer.Option(
        "all",
        "--models", "-m",
        help="Comma-separated CNN model_types for embeddings-OT, or 'all' to "
             "auto-discover from the embeddings directory, or 'none' to skip.",
    ),
    features: bool = typer.Option(
        True, "--features/--no-features",
        help="Also precompute the features-OT variants.",
    ),
    combos: str = typer.Option(
        "all", "--combos",
        help="Which (batch_correct, merge_replicates) combos to populate: "
             "'all' for every variant, or a comma list like "
             "'bc-on:merge-off,bc-off:merge-on'.",
    ),
    n_cells_per_condition: int = typer.Option(400, "--n-cells"),
    sinkhorn_reg: float = typer.Option(0.05, "--reg"),
    library_dir: str = typer.Option(
        "", "--dir", help="Library directory."
    ),
) -> None:
    """Populate the variant-keyed OT cache for every toggle combination.

    Each (source, batch_correct, merge_replicates) cell gets its own
    sidecar under ``<canonical>.variants/`` so flipping the toggles in
    the Analysis panel doesn't trigger a fresh Sinkhorn compute. Time
    cost equals one full OT compute per cell — features-OT × 4 combos
    × N species + embeddings-OT × 4 combos × M model_types.
    """
    import itertools
    from pathlib import Path

    from .extract.feature_library import FeatureLibrary
    from .extract.qc_plots import (
        precompute_embedding_ot_cache,
        precompute_features_ot_cache,
    )

    # Parse combos.
    if combos.strip().lower() == "all":
        bc_mr_combos = list(itertools.product([True, False], [True, False]))
    else:
        bc_mr_combos = []
        for token in combos.split(","):
            t = token.strip().lower().replace(" ", "")
            if not t:
                continue
            bc = "bc-on" in t
            mr = "merge-on" in t
            if "bc-off" in t:
                bc = False
            if "merge-off" in t:
                mr = False
            bc_mr_combos.append((bc, mr))
        if not bc_mr_combos:
            typer.echo(f"No valid combos parsed from --combos {combos!r}")
            raise typer.Exit(code=1)

    lib_dir = Path(library_dir) if library_dir else None
    lib = FeatureLibrary(lib_dir)

    # Species.
    if species:
        species_list = [species]
    else:
        idx = lib.list_runs()
        if idx.empty:
            typer.echo("Library is empty.")
            raise typer.Exit(code=1)
        species_list = sorted(s for s in idx["species"].astype(str).unique() if s)

    # Models.
    if models.lower() == "none":
        model_types: list[str] = []
    elif models.lower() == "all":
        emb_root = lib.models_dir / "embeddings"
        model_types = (
            sorted(
                sub.name for sub in emb_root.iterdir()
                if sub.is_dir() and (sub / "cnn_embeddings.parquet").exists()
            ) if emb_root.exists() else []
        )
    else:
        model_types = [m.strip() for m in models.split(",") if m.strip()]

    total_jobs = (
        (len(species_list) if features else 0) * len(bc_mr_combos)
        + len(species_list) * len(model_types) * len(bc_mr_combos)
    )
    typer.echo(
        f"Planning {total_jobs} OT precompute jobs across "
        f"species={species_list}, model_types={model_types}, "
        f"combos={[('bc-on' if b else 'bc-off', 'merge-on' if m else 'merge-off') for b, m in bc_mr_combos]}"
    )

    done = 0
    for sp in species_list:
        if features:
            for bc, mr in bc_mr_combos:
                done += 1
                tag = f"features {sp!r} bc-{'on' if bc else 'off'}__merge-{'on' if mr else 'off'}"
                typer.echo(f"\n[{done}/{total_jobs}] {tag}")
                try:
                    precompute_features_ot_cache(
                        library_dir=lib_dir,
                        species=sp,
                        n_cells_per_condition=int(n_cells_per_condition),
                        sinkhorn_reg=float(sinkhorn_reg),
                        batch_correct=bc,
                        merge_replicates=mr,
                        progress_cb=lambda f, *_a: None,
                    )
                    typer.echo(f"  ok")
                except Exception as exc:  # noqa: BLE001
                    typer.echo(f"  FAILED: {exc}")

        for mt in model_types:
            emb_path = lib.models_dir / "embeddings" / mt / "cnn_embeddings.parquet"
            if not emb_path.exists():
                typer.echo(f"\n(skip — no embeddings parquet for {mt})")
                continue
            for bc, mr in bc_mr_combos:
                done += 1
                tag = f"embeddings {mt} {sp!r} bc-{'on' if bc else 'off'}__merge-{'on' if mr else 'off'}"
                typer.echo(f"\n[{done}/{total_jobs}] {tag}")
                try:
                    precompute_embedding_ot_cache(
                        emb_path,
                        library_dir=lib_dir,
                        species=sp,
                        model_type=mt,
                        n_cells_per_condition=int(n_cells_per_condition),
                        sinkhorn_reg=float(sinkhorn_reg),
                        batch_correct=bc,
                        merge_replicates=mr,
                        progress_cb=lambda f, *_a: None,
                    )
                    typer.echo(f"  ok")
                except Exception as exc:  # noqa: BLE001
                    typer.echo(f"  FAILED: {exc}")

    typer.echo(f"\nDone. Variants live under <cache>.variants/bc-*__merge-*.ot_distance.parquet")


@library_app.command("gene-template")
def library_gene_template(
    out_csv: str = typer.Argument(
        ..., help="Output CSV path."
    ),
    species: str = typer.Option(
        "", "--species", "-s",
        help="Restrict to one species; default: every gene across all species.",
    ),
    columns: str = typer.Option(
        "operon,family,functional_class", "--columns", "-c",
        help="Comma-separated empty grouping columns to include for you to fill in.",
    ),
    with_counts: bool = typer.Option(
        True, "--with-counts/--no-counts",
        help="Include an _n_conditions column showing how many conditions exist "
             "per gene. The underscore prefix means compare-representations "
             "ignores it as metadata.",
    ),
    library_dir: str = typer.Option(
        "", "--dir", help="Library directory."
    ),
) -> None:
    """Emit a starter groupings CSV pre-filled with the genes in the library.

    Each non-empty gene token across the library becomes a row; grouping
    columns are left blank for you to fill in. Pass the resulting file
    to `library compare-representations`.
    """
    from collections import Counter
    from pathlib import Path

    import pandas as pd

    from .extract.feature_library import FeatureLibrary

    lib = FeatureLibrary(Path(library_dir) if library_dir else None)

    if species:
        species_iter = [species]
    else:
        idx = lib.list_runs()
        if idx.empty:
            typer.echo("Library is empty.")
            raise typer.Exit(code=1)
        species_iter = sorted(s for s in idx["species"].astype(str).unique() if s)
        if not species_iter:
            typer.echo("No species detected; pass --species explicitly.")
            raise typer.Exit(code=1)

    from .extract.crops import derive_condition_fields

    def _gene_from_row(row, primary_col, fallback_col):
        """Best-effort gene extraction.

        For MycoPrep wells the canonical form is
        ``<atc>__<reporter>__<mutant>[__R<n>]`` so the gene is the
        third ``__`` token (parsed by ``derive_condition_fields``).
        When the row already carries a parsed ``condition_label`` like
        "hadA ATc+", first-whitespace-token suffices. The function tries
        both.
        """
        cand = ""
        if primary_col and primary_col in row.index:
            cand = str(row[primary_col]).strip()
        if not cand and fallback_col and fallback_col in row.index:
            cand = str(row[fallback_col]).strip()
        if not cand or cand.lower() == "nan":
            return ""
        if "__" in cand:
            return derive_condition_fields(cand).get("gene", "") or ""
        return cand.split()[0]

    counts: Counter[str] = Counter()
    diagnostics: list[str] = []
    for sp in species_iter:
        df = lib.load_species(sp)
        if df.empty:
            diagnostics.append(f"  species={sp!r}: no runs matched")
            continue
        # Prefer the underscore-delimited well stem (parsed by
        # derive_condition_fields) since that's the authoritative source
        # of the gene name; fall back to condition / condition_label.
        primary_col = next(
            (c for c in ("well", "condition", "condition_label") if c in df.columns),
            None,
        )
        fallback_col = next(
            (c for c in ("condition", "condition_label", "well")
             if c in df.columns and c != primary_col),
            None,
        )
        if primary_col is None:
            diagnostics.append(
                f"  species={sp!r}: loaded {len(df)} cells but no "
                f"well/condition/condition_label column "
                f"(columns: {list(df.columns)[:10]}...)"
            )
            continue
        # Collapse to per-(well, condition) — one row per unique condition
        # so the count column reflects condition-count, not cell-count.
        group_cols = [c for c in [primary_col, fallback_col] if c]
        per_cond = df[group_cols].drop_duplicates().reset_index(drop=True)
        per_cond["__gene"] = per_cond.apply(
            lambda r: _gene_from_row(r, primary_col, fallback_col), axis=1,
        )
        added = 0
        for gene, sub in per_cond.groupby("__gene"):
            if not gene or gene.lower() == "nan":
                continue
            counts[gene] += int(len(sub))
            added += 1
        diagnostics.append(
            f"  species={sp!r}: {len(df)} cells, {len(per_cond)} conditions, "
            f"{added} unique genes (parsed from {primary_col!r})"
        )

    if not counts:
        typer.echo("No genes found in the library.")
        if diagnostics:
            typer.echo("\nDiagnostics:")
            for line in diagnostics:
                typer.echo(line)
        # Show what's registered so the user can correct the species filter.
        idx_all = lib.list_runs()
        if not idx_all.empty:
            sp_present = sorted(
                s for s in idx_all["species"].astype(str).unique() if s
            )
            typer.echo(
                f"\nSpecies registered in the library: "
                f"{sp_present if sp_present else '(none)'}"
            )
            typer.echo(f"Library dir: {lib.library_dir}")
        raise typer.Exit(code=1)

    group_cols = [c.strip() for c in columns.split(",") if c.strip()]
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    out_df = pd.DataFrame({"gene": [g for g, _ in rows]})
    if with_counts:
        out_df["_n_conditions"] = [n for _, n in rows]
    for col in group_cols:
        out_df[col] = ""

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    typer.echo(
        f"Wrote {len(out_df)} genes to {out_path}.\n"
        f"Fill in {', '.join(group_cols)} (empty cells = unmapped), then run:\n"
        f"  mycoprep-cli library compare-representations {out_path}"
        + (f' --species "{species}"' if species else "")
    )


@library_app.command("compare-representations")
def library_compare_representations(
    group_csv: str = typer.Argument(
        ..., help="CSV/TSV with a 'gene' column and one or more grouping columns "
                  "(e.g. operon, family, functional_class)."
    ),
    species: str = typer.Option(
        "", "--species", "-s",
        help="Restrict to one species; default: evaluate each species separately.",
    ),
    models: str = typer.Option(
        "resnet18,lightweight,supcon_resnet18,supcon_lightweight",
        "--models", "-m",
        help="Comma-separated CNN architectures to evaluate (and train if missing).",
    ),
    batch_correct: str = typer.Option(
        "both", "--batch-correct",
        help="'both', 'on', or 'off' — which Harmony states to evaluate.",
    ),
    retrain: bool = typer.Option(
        False, "--retrain",
        help="Force retraining even if a model_type is already in the manifest.",
    ),
    epochs: int = typer.Option(
        50, "--epochs", help="Epochs used when training a missing model."
    ),
    batch_size: int = typer.Option(64, "--batch-size"),
    ot: bool = typer.Option(
        True, "--ot/--no-ot",
        help="Generate OT distance caches for features and embeddings.",
    ),
    umap: bool = typer.Option(
        True, "--umap/--no-umap",
        help="Add UMAP-projected evaluation rows for each source.",
    ),
    k: str = typer.Option(
        "1,3,5", "--k",
        help="Comma-separated k values for kNN same-group accuracy.",
    ),
    gene_key: str = typer.Option(
        "gene", "--gene-key",
        help="Replicate key: 'gene' (default; first token of condition_label) "
             "or 'condition' (full condition, treats different reporter "
             "backgrounds / treatments as distinct biological observations).",
    ),
    replicate_scope: bool = typer.Option(
        True, "--replicate-scope/--no-replicate-scope",
        help="Also emit a separate replicate-consistency score per source.",
    ),
    same_conditions_only: bool = typer.Option(
        False, "--same-conditions-only",
        help="Before scoring, restrict every source to the intersection "
             "of *genes* present across all loaded sources. Use when "
             "different architectures' extractions cover different "
             "subsets of genes and the absolute mAPs would otherwise "
             "not be directly comparable. Intersects by gene (not full "
             "condition label) so it's robust to differences in how "
             "each source formats its condition strings.",
    ),
    library_dir: str = typer.Option(
        "", "--dir", help="Library directory (default: ~/.mycoprep/morphology_library/)."
    ),
    out_dir: str = typer.Option(
        "", "--out",
        help="Output dir; default: <library>/representation_eval/<timestamp>/",
    ),
    top_n: int = typer.Option(
        15, "--top",
        help="Show this many top rows of the summary table on stdout.",
    ),
) -> None:
    """Score every representation against every grouping in GROUP_CSV.

    Generates missing CNN encoders, embeddings, and OT distance caches
    (with and without Harmony batch correction), then evaluates every
    (source, batch_correct, grouping) using kNN same-group accuracy and
    mAP. Writes a CSV/PNG summary under the output directory and prints
    the top rows to stdout.
    """
    from pathlib import Path

    from .extract.feature_library import FeatureLibrary
    from .extract.representation_eval import score_all_representations

    bc_arg = batch_correct.strip().lower()
    if bc_arg == "both":
        bc_states: tuple[bool, ...] = (True, False)
    elif bc_arg == "on":
        bc_states = (True,)
    elif bc_arg == "off":
        bc_states = (False,)
    else:
        typer.echo(f"--batch-correct must be one of: both, on, off (got {batch_correct!r})")
        raise typer.Exit(code=1)

    model_types = [m.strip() for m in models.split(",") if m.strip()]
    if not model_types:
        typer.echo("--models must list at least one model type.")
        raise typer.Exit(code=1)

    try:
        k_list = tuple(int(x) for x in k.split(",") if x.strip())
    except ValueError:
        typer.echo(f"--k must be comma-separated integers (got {k!r})")
        raise typer.Exit(code=1) from None
    if not k_list:
        k_list = (1, 3, 5)

    if gene_key not in ("gene", "condition"):
        typer.echo(f"--gene-key must be 'gene' or 'condition' (got {gene_key!r})")
        raise typer.Exit(code=1)

    lib_dir = Path(library_dir) if library_dir else None
    out_path = Path(out_dir) if out_dir else None

    # Resolve species list: a single one if specified, else iterate every
    # species present in the library.
    if species:
        species_list = [species]
    else:
        lib = FeatureLibrary(lib_dir)
        idx = lib.list_runs()
        if idx.empty:
            typer.echo("Library is empty.")
            raise typer.Exit(code=1)
        species_list = sorted(s for s in idx["species"].astype(str).unique() if s)
        if not species_list:
            typer.echo("No species detected in library; pass --species explicitly.")
            raise typer.Exit(code=1)

    all_metrics = []
    for sp in species_list:
        typer.echo(f"\n=== Species: {sp} ===")

        def _cb(f: float, msg: str, *, _sp=sp):
            typer.echo(f"  [{_sp}] {int(f * 100):3d}%  {msg}")

        metrics, _ = score_all_representations(
            lib_dir,
            sp,
            Path(group_csv),
            model_types=model_types,
            batch_correct_states=bc_states,
            include_features_ot=ot,
            include_embeddings_ot=ot,
            include_umap=umap,
            replicate_key_mode=gene_key,
            include_replicate_scope=replicate_scope,
            same_conditions_only=bool(same_conditions_only),
            epochs=int(epochs),
            batch_size=int(batch_size),
            retrain=bool(retrain),
            k_list=k_list,
            out_dir=out_path,
            progress_cb=_cb,
        )
        all_metrics.extend(metrics)

    # Print headline table.
    if not all_metrics:
        typer.echo("\nNo metrics produced.")
        return

    from .extract.representation_eval import metrics_to_summary_df
    df = metrics_to_summary_df(all_metrics)
    if df.empty:
        typer.echo("\nSummary table is empty.")
        return

    head = df.head(int(top_n))
    show_cols = [
        c for c in [
            "species", "grouping", "scope", "representation", "batch_correct",
            "map", "knn@1", "knn@5", "n_conditions",
        ] if c in head.columns
    ]
    typer.echo(f"\nTop {len(head)} rows by mAP:\n")
    typer.echo(head[show_cols].to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    app()
