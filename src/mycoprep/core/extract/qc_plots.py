"""Quick-look QC plots from ``all_features.parquet``.

Produces the same figures the standalone ``qc_plots.py`` script generates,
exposed as a callable so the Features stage can run them automatically at
end-of-run.

Plots written to ``<features_dir>/qc_plots/``:
- ``features_per_mutant.png`` — length / width-median / width-mean / area /
  intensity violins per (mutant × ATc) group, plus matching summary CSV.
- ``features_pooled_by_atc.png`` — pooled view across mutants.
- ``length_vs_intensity_facets.png`` — per-cell scatter facets per mutant.
- ``intensity_variation.png`` — CV + dynamic range bar charts.
- ``intensity_distributions.png`` — log-scale ridge of per-cell intensity.
- ``morphology_clustering_run.png`` — UMAP + HDBSCAN on current run.
- ``morphology_clustering_library.png`` — same with library context (when
  a feature library is available).

Failures (e.g. missing parquet, headless matplotlib not installed) are
caught and reported via the progress callback — they should never block
the pipeline run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

ProgressCB = Callable[[float, str], None]


def _noop(_f: float, _m: str) -> None:
    pass


def _pick_intensity_col(frame) -> str:
    """Auto-pick a fluorescence intensity column from a features DataFrame.

    Prefers columns mentioning 'mcherry'/'mcher' (case-insensitive); falls
    back to the first non-phase ``intensity_mean_*`` column.
    """
    candidates = [c for c in frame.columns if c.startswith("intensity_mean_")]
    if not candidates:
        return ""
    lc = [c for c in candidates if "cherry" in c.lower() or "mcher" in c.lower()]
    if lc:
        return lc[0]
    non_phase = [c for c in candidates if "phase" not in c.lower()]
    return (non_phase or candidates)[0]


def make_qc_plots(
    features_dir: Path,
    *,
    library_dir: Optional[Path] = None,
    species: str = "",
    current_run_id: str = "",
    control_labels: Optional[list[str]] = None,
    batch_correct: bool = True,
    progress_cb: ProgressCB = _noop,
) -> Optional[Path]:
    """Generate QC plots in ``<features_dir>/qc_plots/`` from
    ``all_features.parquet``. Returns the plots dir, or ``None`` on failure.
    """
    features_dir = Path(features_dir)
    pq = features_dir / "all_features.parquet"
    if not pq.exists():
        progress_cb(1.0, f"qc_plots: {pq.name} not found, skipping")
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")  # avoid Qt clash from a headless feature stage
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except Exception as e:  # noqa: BLE001
        progress_cb(1.0, f"qc_plots: dependency missing ({e}), skipping")
        return None

    df = pd.read_parquet(pq)
    if df.empty or "well" not in df.columns:
        progress_cb(1.0, "qc_plots: empty / malformed features table, skipping")
        return None

    out_dir = features_dir / "qc_plots"
    out_dir.mkdir(exist_ok=True)

    # Parse condition fields from the well stem.
    parts = df["well"].astype(str).str.split("__", expand=True)
    df["atc"] = parts[0].str.replace("_focused", "", regex=False)
    if 1 in parts.columns:
        df["reporter"] = parts[1]
    if 2 in parts.columns:
        df["mutant"] = parts[2]
    else:
        df["mutant"] = df["well"]

    intensity_col = _pick_intensity_col(df)
    intensity_label = (
        intensity_col.replace("intensity_mean_", "") if intensity_col else "intensity"
    )

    mutants = sorted(df["mutant"].astype(str).unique())
    PALETTE_DEFAULT = {"ATc-": "#5b8def", "ATc+": "#e07b6b"}
    atc_states = [a for a in ("ATc-", "ATc+") if (df["atc"] == a).any()]
    if not atc_states:
        atc_states = sorted(df["atc"].astype(str).unique().tolist())
    palette = {a: PALETTE_DEFAULT.get(a, "#888888") for a in atc_states}

    def violin_with_box(ax, data_by_group, labels, palette_seq, title, ylabel):
        positions = np.arange(len(labels))
        if not any(len(d) > 0 for d in data_by_group):
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(title, fontsize=11)
            return
        clean = [d if len(d) else np.array([0.0]) for d in data_by_group]
        parts = ax.violinplot(
            clean, positions=positions, widths=0.85,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for body, c in zip(parts["bodies"], palette_seq):
            body.set_facecolor(c); body.set_alpha(0.5)
            body.set_edgecolor("black"); body.set_linewidth(0.6)
        bp = ax.boxplot(
            clean, positions=positions, widths=0.18,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for box, c in zip(bp["boxes"], palette_seq):
            box.set_facecolor("white"); box.set_edgecolor("black"); box.set_linewidth(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── Figure 1: per-mutant per-ATc violins ────────────────────────────
    progress_cb(0.1, "qc_plots: building per-mutant figure")
    group_labels: list[str] = []
    group_palette: list[str] = []
    data_length: list = []
    data_width_med: list = []
    data_width_mean: list = []
    data_area: list = []
    data_intensity: list = []
    area_col = "area_um2_subpixel" if "area_um2_subpixel" in df.columns else "area_um2"
    for m in mutants:
        for atc in atc_states:
            sub = df[(df["mutant"] == m) & (df["atc"] == atc)]
            if len(sub) == 0:
                continue
            group_labels.append(f"{m}\n{atc}\n(n={len(sub)})")
            group_palette.append(palette[atc])
            data_length.append(sub.get("length_um", pd.Series(dtype=float)).to_numpy())
            data_width_med.append(sub.get("width_median_um", pd.Series(dtype=float)).to_numpy())
            data_width_mean.append(sub.get("width_mean_um", pd.Series(dtype=float)).to_numpy())
            data_area.append(sub.get(area_col, pd.Series(dtype=float)).to_numpy())
            data_intensity.append(
                sub.get(intensity_col, pd.Series(dtype=float)).to_numpy()
                if intensity_col else np.array([])
            )

    fig, axes = plt.subplots(5, 1, figsize=(max(7, 0.9 * len(group_labels)), 17), sharex=True)
    violin_with_box(axes[0], data_length, group_labels, group_palette,
                    "Length (midline arc)", "length / µm")
    violin_with_box(axes[1], data_width_med, group_labels, group_palette,
                    "Width (median along midline)", "width / µm")
    violin_with_box(axes[2], data_width_mean, group_labels, group_palette,
                    "Width (mean along midline)", "width / µm")
    violin_with_box(axes[3], data_area, group_labels, group_palette,
                    f"Area ({area_col})", "area / µm²")
    violin_with_box(axes[4], data_intensity, group_labels, group_palette,
                    f"{intensity_label} mean intensity (per cell)", "intensity (a.u.)")
    handles = [mpatches.Patch(color=palette[a], alpha=0.6, label=a) for a in atc_states]
    axes[0].legend(handles=handles, loc="upper right", frameon=False)
    fig.suptitle("Single-cell features by mutant × ATc state (Phase B)",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "features_per_mutant.png", dpi=160)
    plt.close(fig)

    # ── Figure 2: pooled by ATc ────────────────────────────────────────
    progress_cb(0.4, "qc_plots: pooled-ATc figure")
    fig2, axes2 = plt.subplots(1, 5, figsize=(17, 4))
    plot_cols = (
        ("length_um", "Length", "length / µm"),
        ("width_median_um", "Width (median)", "width / µm"),
        ("width_mean_um", "Width (mean)", "width / µm"),
        (area_col, "Area", "area / µm²"),
        (intensity_col, f"{intensity_label} mean", "intensity (a.u.)"),
    )
    for ax, (col, title, ylabel) in zip(axes2, plot_cols):
        if not col or col not in df.columns:
            ax.axis("off"); continue
        grouped = [df.loc[df["atc"] == a, col].dropna().to_numpy() for a in atc_states]
        labels = [f"{a}\n(n={len(g):,})" for a, g in zip(atc_states, grouped)]
        violin_with_box(ax, grouped, labels, [palette[a] for a in atc_states],
                        title, ylabel)
    fig2.suptitle("Pooled across mutants", fontsize=12, y=1.02)
    fig2.tight_layout()
    fig2.savefig(out_dir / "features_pooled_by_atc.png", dpi=160, bbox_inches="tight")
    plt.close(fig2)

    # ── Figure 3: length vs intensity scatter facets ────────────────────
    if intensity_col and "length_um" in df.columns:
        progress_cb(0.6, "qc_plots: length-vs-intensity facets")
        n_mut = len(mutants); ncols = min(4, max(1, n_mut))
        nrows = int(np.ceil(n_mut / ncols)) if n_mut else 1
        fig3, axes3 = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                                    sharex=True, sharey=True)
        axes3 = np.atleast_2d(axes3).reshape(nrows, ncols)
        for i, m in enumerate(mutants):
            r, c = divmod(i, ncols)
            ax = axes3[r][c]
            for atc in atc_states:
                sub = df[(df["mutant"] == m) & (df["atc"] == atc)]
                if len(sub) == 0:
                    continue
                ax.scatter(sub["length_um"], sub[intensity_col],
                           s=4, alpha=0.35, color=palette[atc],
                           label=atc, edgecolors="none")
            ax.set_title(m, fontsize=10)
            ax.grid(alpha=0.25, linestyle="--")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if i == 0:
                ax.legend(frameon=False, fontsize=8, markerscale=2)
        for j in range(n_mut, nrows * ncols):
            r, c = divmod(j, ncols); axes3[r][c].axis("off")
        for ax in axes3[-1]:
            ax.set_xlabel("length (µm)")
        for ax in axes3[:, 0]:
            ax.set_ylabel(f"{intensity_label} mean (a.u.)")
        fig3.suptitle("Length vs intensity per cell, by mutant", fontsize=12, y=1.005)
        fig3.tight_layout()
        fig3.savefig(out_dir / "length_vs_intensity_facets.png",
                     dpi=160, bbox_inches="tight")
        plt.close(fig3)

    # ── Summary table ──────────────────────────────────────────────────
    summary = (
        df.groupby(["mutant", "atc"])
          .agg(
              n=("cell_uid" if "cell_uid" in df.columns else "well", "size"),
              length_median=("length_um", "median"),
              length_iqr=("length_um", lambda s: s.quantile(0.75) - s.quantile(0.25)),
              width_median=("width_median_um", "median"),
              width_mean=("width_mean_um", "mean"),
              area_median=(area_col, "median"),
              area_mean=(area_col, "mean"),
          )
          .round(3)
    )
    if intensity_col:
        intensity_summary = df.groupby(["mutant", "atc"])[intensity_col].agg(
            mCherry_median="median",
            mCherry_p95=lambda s: s.quantile(0.95),
        ).round(3)
        summary = summary.join(intensity_summary)
    summary.to_csv(out_dir / "summary_by_mutant_atc.csv")

    # ── Figures 4 + 5: variation ────────────────────────────────────────
    if intensity_col and intensity_col in df.columns:
        progress_cb(0.8, "qc_plots: variation figures")
        var_stats = (
            df.groupby(["mutant", "atc"])[intensity_col]
              .agg(
                  n="size", mean="mean", median="median", std="std",
                  p5=lambda s: s.quantile(0.05),
                  p95=lambda s: s.quantile(0.95),
                  mad=lambda s: (s - s.median()).abs().median(),
              )
              .reset_index()
        )
        var_stats["cv"] = var_stats["std"] / var_stats["mean"]
        var_stats["robust_cv"] = var_stats["mad"] / var_stats["median"]
        var_stats["dynamic_range"] = var_stats["p95"] / var_stats["p5"].clip(lower=1e-9)
        var_stats = var_stats.sort_values("cv", ascending=False).reset_index(drop=True)
        var_stats.to_csv(out_dir / "intensity_variation.csv", index=False)

        fig4, axes4 = plt.subplots(1, 2, figsize=(max(7, 0.6 * len(var_stats) + 4), 5))
        labels = [f"{m}\n{a}\n(n={int(n)})"
                  for m, a, n in zip(var_stats["mutant"], var_stats["atc"], var_stats["n"])]
        bar_colors = [palette.get(a, "#888888") for a in var_stats["atc"]]
        pos = np.arange(len(var_stats))
        axes4[0].bar(pos, var_stats["cv"], color=bar_colors,
                     edgecolor="black", linewidth=0.5)
        axes4[0].axhline(1.0, color="red", linestyle="--", linewidth=0.8,
                         label="CV = 1 (heterogeneous)")
        axes4[0].set_xticks(pos); axes4[0].set_xticklabels(labels, rotation=30, ha="right")
        axes4[0].set_ylabel("CV (std / mean)")
        axes4[0].set_title(f"Intensity heterogeneity ({intensity_label})", fontsize=11)
        axes4[0].grid(axis="y", alpha=0.25, linestyle="--")
        axes4[0].spines["top"].set_visible(False); axes4[0].spines["right"].set_visible(False)
        axes4[0].legend(frameon=False, fontsize=8)
        axes4[1].bar(pos, var_stats["dynamic_range"], color=bar_colors,
                     edgecolor="black", linewidth=0.5)
        axes4[1].set_xticks(pos); axes4[1].set_xticklabels(labels, rotation=30, ha="right")
        axes4[1].set_ylabel("dynamic range (p95 / p5)")
        axes4[1].set_title(f"Bright-tail amplification ({intensity_label})", fontsize=11)
        axes4[1].grid(axis="y", alpha=0.25, linestyle="--")
        axes4[1].spines["top"].set_visible(False); axes4[1].spines["right"].set_visible(False)
        fig4.tight_layout()
        fig4.savefig(out_dir / "intensity_variation.png", dpi=160, bbox_inches="tight")
        plt.close(fig4)

        fig5, ax5 = plt.subplots(figsize=(8, max(4, 0.45 * len(var_stats))))
        log_max = np.log10(max(1, var_stats["p95"].max() * 1.1))
        log_min = np.log10(max(1, var_stats["p5"].min() / 1.1))
        bins = np.logspace(log_min, log_max, 50)
        centers = (bins[:-1] + bins[1:]) / 2
        for i, (_, row) in enumerate(var_stats.iterrows()):
            sub = df[(df["mutant"] == row["mutant"]) & (df["atc"] == row["atc"])]
            if sub.empty: continue
            vals = sub[intensity_col].dropna().to_numpy()
            if vals.size == 0: continue
            h, _ = np.histogram(vals, bins=bins)
            h = h / max(1, h.max())
            ax5.fill_between(centers, i + h * 0.85, i, alpha=0.55,
                             color=palette.get(row["atc"], "#888888"),
                             linewidth=0.6, edgecolor="black")
            ax5.text(bins[0] * 0.9, i + 0.4,
                     f"{row['mutant']} (CV={row['cv']:.2f}, n={int(row['n'])})",
                     ha="right", va="center", fontsize=8)
        ax5.set_xscale("log")
        ax5.set_xlabel(f"{intensity_label} (per cell, log)")
        ax5.set_yticks([]); ax5.set_xlim(bins[0], bins[-1])
        ax5.set_title(f"Per-cell {intensity_label} distributions, sorted by CV",
                      fontsize=11)
        ax5.grid(axis="x", alpha=0.25, linestyle="--", which="both")
        for sp in ("top", "right", "left"):
            ax5.spines[sp].set_visible(False)
        fig5.tight_layout()
        fig5.savefig(out_dir / "intensity_distributions.png",
                     dpi=160, bbox_inches="tight")
        plt.close(fig5)

    # ── Figure 6: morphology clustering ──────────────────────────────────
    try:
        _morphology_cluster_plots(
            df,
            out_dir=out_dir,
            palette=palette,
            atc_states=atc_states,
            library_dir=library_dir,
            species=species,
            current_run_id=current_run_id,
            control_labels=control_labels or [],
            batch_correct=batch_correct,
            progress_cb=progress_cb,
        )
    except Exception as e:  # noqa: BLE001
        progress_cb(0.95, f"qc_plots: clustering skipped ({e})")

    progress_cb(1.0, f"qc_plots: wrote → {out_dir.name}/")
    return out_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Morphology clustering helpers
# ═══════════════════════════════════════════════════════════════════════════════

_MORPHOLOGY_COLS_PREFERRED = [
    "length_um",
    "width_median_um",
    "width_mean_um",
    "width_max_um",
    "width_min_um",
    "width_std_um",
    "area_um2_subpixel",
    "area_um2",
    "perimeter_um_subpixel",
    "perimeter_um",
    "eccentricity",
    "solidity",
    "sinuosity",
    "feret_diameter_max_um",
    "major_axis_length_um",
    "minor_axis_length_um",
]


def _select_morphology_cols(df) -> list[str]:
    """Pick morphology columns present in *df*, preferring subpixel variants."""
    import pandas as pd

    available = set(df.columns)
    selected: list[str] = []
    seen_base: set[str] = set()
    for col in _MORPHOLOGY_COLS_PREFERRED:
        base = col.replace("_subpixel", "")
        if base in seen_base:
            continue
        if col in available:
            selected.append(col)
            seen_base.add(base)
        elif base in available and base != col:
            selected.append(base)
            seen_base.add(base)
    return selected


def _extract_run_ids(profile_index) -> "np.ndarray":
    """Extract run_id from 'condition @ run_id' index labels."""
    import numpy as np

    return np.array([
        str(lbl).rsplit(" @ ", 1)[1] if " @ " in str(lbl) else ""
        for lbl in profile_index
    ])


def _match_controls(condition_labels, control_labels: list[str]) -> "np.ndarray":
    """Boolean mask: which entries in *condition_labels* are controls.

    Each control_label is matched as a case-insensitive whole-word token
    against the condition label. ``"NT1"`` matches ``"NT1 ATc-"`` but not
    ``"mutNT11 ATc+"``. If *control_labels* is empty, no rows are flagged.
    """
    import re

    import numpy as np

    arr = np.array([str(c) for c in condition_labels])
    if not control_labels:
        return np.zeros(len(arr), dtype=bool)

    tokens = [re.escape(c.strip()) for c in control_labels if c and c.strip()]
    if not tokens:
        return np.zeros(len(arr), dtype=bool)
    pattern = r"\b(?:" + "|".join(tokens) + r")\b"
    rx = re.compile(pattern, re.IGNORECASE)
    return np.array([bool(rx.search(c)) for c in arr])


def _compute_condition_sscores(
    df, morph_cols, condition_col="condition",
    control_labels: list[str] | None = None,
    baseline_mode: str = "pooled",
):
    """Compute per-condition S-score profiles (mean + CV, Z-scored vs controls).

    Controls are identified explicitly via *control_labels* (e.g.
    ``["NT1", "NT2", "WT"]``).

    ``baseline_mode``:
      - ``"pooled"`` (default): every control profile across every run
        contributes equally to a single (mu, sigma) baseline. Matches
        the MorphologicalProfiling_Mtb R pipeline. Stable when each
        run has few controls, but lets between-run variance leak into
        sigma.
      - ``"per_run"``: each run's controls form their own (mu_r,
        sigma_r); perturbations are z-scored against their own run's
        baseline. Removes batch effects between runs but requires
        ≥ 2 control profiles per run for a non-degenerate sigma —
        falls back to the pooled sigma for runs that don't qualify.
        Requires the ``condition_col`` labels to encode run as
        ``"<condition> @ <run_id>"``; otherwise behaves like pooled.

    If no condition matches a control label, falls back to
    ``StandardScaler`` against the full dataset — geometry stays
    informative but values are no longer true S-scores.
    """
    import numpy as np
    import pandas as pd

    def _cv(s):
        m = s.mean()
        return s.std() / m if m != 0 else 0.0

    means = df.groupby(condition_col)[morph_cols].mean()
    cvs = df.groupby(condition_col)[morph_cols].agg(_cv)
    cvs.columns = [c + "_CV" for c in cvs.columns]

    profiles = pd.concat([means, cvs], axis=1)

    control_mask = _match_controls(profiles.index.tolist(), control_labels or [])

    if not control_mask.any():
        from sklearn.preprocessing import StandardScaler
        vals = StandardScaler().fit_transform(profiles.values)
        return pd.DataFrame(vals, index=profiles.index, columns=profiles.columns)

    if baseline_mode == "per_run":
        # Pull run_id out of the index labels. The library-mode
        # convention is "<condition> @ <run_id>"; solo mode just has
        # "<condition>". When the @ marker is missing we treat every
        # row as the same run, which collapses to pooled behaviour.
        run_ids = pd.Series(
            [str(lbl).rsplit(" @ ", 1)[1] if " @ " in str(lbl) else ""
             for lbl in profiles.index],
            index=profiles.index,
        )

        # Pooled baseline used as a fallback when a run has fewer than
        # two control profiles (sigma would otherwise be zero / NaN).
        pooled_ctrl = profiles.loc[control_mask]
        pooled_sigma = pooled_ctrl.std().replace(0, 1)

        out = profiles.copy()
        for run, idxs in run_ids.groupby(run_ids).groups.items():
            sub_mask = pd.Series(control_mask, index=profiles.index).loc[idxs]
            if not sub_mask.any():
                # No controls in this run — leave its rows un-normalised
                # (the user will see them clearly off-baseline). Could
                # alternatively use the pooled baseline; explicit "no
                # baseline" feels more honest.
                continue
            ctrl_rows = profiles.loc[idxs[sub_mask.values]]
            mu = ctrl_rows.mean()
            if len(ctrl_rows) < 2:
                # std is undefined / zero — borrow the pooled sigma.
                sigma = pooled_sigma
            else:
                sigma = ctrl_rows.std().replace(0, 1)
            out.loc[idxs] = (profiles.loc[idxs] - mu) / sigma
        return out

    # Default: pooled baseline.
    ctrl = profiles.loc[control_mask]
    mu = ctrl.mean()
    sigma = ctrl.std().replace(0, 1)
    return (profiles - mu) / sigma


def _subsample_stratified(df, max_n, group_col, rng):
    """Subsample to at most *max_n* rows, stratified by *group_col*."""
    if len(df) <= max_n:
        return df
    groups = df[group_col].unique()
    per_group = max(1, max_n // len(groups))
    parts = []
    for g in groups:
        sub = df[df[group_col] == g]
        if len(sub) > per_group:
            sub = sub.sample(n=per_group, random_state=rng)
        parts.append(sub)
    return __import__("pandas").concat(parts, ignore_index=True)


def _run_umap_hdbscan(X_scaled, n_neighbors=3, min_dist=0.0,
                       min_cluster_size=3, random_state=42,
                       batch_labels=None,
                       *,
                       n_consensus_runs: int = 25,
                       pca_threshold_features: int = 100):
    """[PCA] → [Harmony] → consensus UMAP+HDBSCAN → (embedding_2d, cluster_labels).

    Mirrors the MorphologicalProfiling_Mtb R pipeline's stability strategy:
    UMAP+HDBSCAN is rerun with ``n_consensus_runs`` different random seeds,
    a co-association proportion matrix is built across runs, and a single
    stable partition is recovered by hierarchical clustering on
    ``1 − co_assoc`` with ``k`` picked by silhouette in
    ``[2, min(8, N-1)]``. The 2D embedding for plotting comes from the
    ``random_state``-seeded run so the picture is reproducible.

    PCA is skipped when ``n_features <= pca_threshold_features`` — UMAP
    handles ~30–60-D inputs directly and PCA-to-95%-variance silently
    drops the low-variance shape axes (``width_std``, ``sinuosity`` …)
    that carry the discriminative phenotype signal.

    Defaults (``n_neighbors=3``, ``min_dist=0``, ``min_cluster_size=3``)
    match the Mtb R config; they capture local structure rather than
    dissolving into global geometry.

    Harmony, when batch labels are present, uses ``nclust = min(max(2,
    n_batches), 5)`` instead of an adaptive ``N // 5``-based heuristic
    that over-aligned and erased biology on small profile counts.
    """
    import numpy as np
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA

    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required for clustering plots")

    n_samples, n_features = X_scaled.shape

    # Skip PCA on low-D inputs (typical: 28–60-D shape S-scores). UMAP
    # handles them directly and PCA would discard low-variance axes that
    # carry phenotype signal. Keep PCA only when the input is genuinely
    # high-D (e.g. 512-D CNN embeddings), in which case it both denoises
    # and accelerates Harmony.
    if n_features > pca_threshold_features:
        n_components = min(n_features, n_samples, 50)
        if n_components >= 2:
            pca = PCA(
                n_components=min(n_components, 0.95),
                random_state=random_state,
            )
            X_proc = pca.fit_transform(X_scaled)
        else:
            X_proc = X_scaled
    else:
        X_proc = X_scaled

    if batch_labels is not None:
        unique_batches = np.unique(batch_labels)
        n_batches = len(unique_batches)
        if n_batches >= 2:
            try:
                import harmonypy
                import pandas as pd

                ho = harmonypy.run_harmony(
                    X_proc,
                    pd.DataFrame({"run_id": batch_labels}),
                    vars_use="run_id",
                    max_iter_harmony=20,
                    # Conservative pinned nclust — anchored on the actual
                    # batch count, capped at 5. The previous adaptive
                    # ``min(max(2, N//5), 20)`` over-aligned on small N
                    # and could erase real biology when run/condition
                    # confounding was partial.
                    nclust=min(max(2, n_batches), 5),
                )
                X_proc = ho.Z_corr
            except ImportError:
                pass

    effective_neighbors = max(2, min(n_neighbors, n_samples - 1))
    effective_min_cluster = max(2, min(min_cluster_size, n_samples // 2))

    def _single_run(seed: int):
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=min_dist,
            random_state=seed,
            n_jobs=1,
        )
        emb = reducer.fit_transform(X_proc)
        clusterer = HDBSCAN(min_cluster_size=effective_min_cluster, copy=False)
        return emb, clusterer.fit_predict(emb)

    # Single-shot path for tiny datasets where consensus is meaningless.
    if n_samples < 6 or n_consensus_runs <= 1:
        return _single_run(random_state)

    # Consensus: gather labels across many seeds, compute pairwise
    # co-association proportion, hclust, pick k by silhouette.
    embedding, _ = _single_run(random_state)

    label_runs: list[np.ndarray] = []
    seeds = [random_state] + list(range(random_state + 1,
                                        random_state + n_consensus_runs))
    for seed in seeds:
        if seed == random_state:
            _, lbl = _single_run(seed)
        else:
            _, lbl = _single_run(seed)
        # Remap HDBSCAN noise label (-1) to unique negative IDs per
        # sample so two noise points are NOT counted as co-clustered.
        lbl = np.asarray(lbl).astype(np.int64).copy()
        noise = lbl == -1
        if noise.any():
            offset = int(lbl.max()) + 1 if (lbl >= 0).any() else 0
            lbl[noise] = -(offset + 1 + np.arange(int(noise.sum())))
        label_runs.append(lbl)

    L = np.stack(label_runs, axis=1)  # (n_samples, n_runs)
    co_assoc = np.zeros((n_samples, n_samples), dtype=np.float64)
    for r in range(L.shape[1]):
        col = L[:, r][:, None]
        co_assoc += (col == col.T).astype(np.float64)
    co_assoc /= L.shape[1]
    consensus_dist = 1.0 - co_assoc
    np.fill_diagonal(consensus_dist, 0.0)

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        from sklearn.metrics import silhouette_score

        condensed = squareform(consensus_dist, checks=False)
        Z = linkage(condensed, method="average")

        scores: list[tuple[int, float]] = []
        k_max = min(8, n_samples - 1)
        for k in range(2, k_max + 1):
            cand = fcluster(Z, t=k, criterion="maxclust")
            if len(np.unique(cand)) < 2:
                continue
            try:
                score = silhouette_score(
                    consensus_dist, cand, metric="precomputed",
                )
            except Exception:  # noqa: BLE001
                continue
            scores.append((k, score))

        if scores:
            best_score = max(s for _, s in scores)
            # Bias toward parsimony: pick the SMALLEST k whose silhouette
            # is within ``tol`` of the maximum. Without this, silhouette
            # routinely splits a single dense biological cluster into
            # 2–3 noise-driven sub-clusters because the within-cluster
            # consensus distance is tiny.
            tol = 0.03
            eligible = [k for k, s in scores if s >= best_score - tol]
            best_k = min(eligible) if eligible else scores[0][0]
            labels = fcluster(Z, t=best_k, criterion="maxclust") - 1
        else:
            labels = label_runs[0]
    except Exception:  # noqa: BLE001
        # Fallback: just use the labels from the seeded run.
        labels = label_runs[0]

    return embedding, labels


def _embed_profiles(profiles, *, batch_correct: bool = True):
    """2D embedding + cluster labels from a per-condition S-score table.

    Uses UMAP+HDBSCAN when there are at least 6 conditions (UMAP needs
    enough neighbours to behave); falls back to PCA + a single dummy
    cluster otherwise. Returns (embedding[N×2], cluster_labels[N]).

    When *batch_correct* is True (default), batch labels are auto-extracted
    from the profile index (expects 'condition @ run_id' format) and
    passed to Harmony for inter-run alignment when multiple runs are
    present. Set to False to disable batch correction.
    """
    import numpy as np

    X = profiles.values
    n = len(profiles)

    if batch_correct:
        batch_labels = _extract_run_ids(profiles.index)
        has_batches = len(set(batch_labels) - {""}) >= 2
    else:
        batch_labels = None
        has_batches = False

    if n >= 6:
        emb, lbl = _run_umap_hdbscan(
            X,
            n_neighbors=min(3, n - 1),
            min_dist=0.0,
            min_cluster_size=3,
            batch_labels=batch_labels if has_batches else None,
        )
    else:
        from sklearn.decomposition import PCA as _PCA
        n_comp = min(2, X.shape[1], n)
        emb = _PCA(n_components=n_comp).fit_transform(X)
        if emb.shape[1] == 1:
            emb = np.column_stack([emb, np.zeros(n)])
        lbl = np.full(n, 0)
    return emb, lbl


def _condition_meta_table(
    combined_df, profile_labels, label_col,
    control_labels: list[str] | None = None,
):
    """Resolve per-profile metadata used by the static + interactive plots.

    Returns a list of dicts in profile order with: condition, gene,
    atc, reporter, replica, run_id, experiment_type, n_cells,
    is_current_run, is_control.

    The well-name convention is ``<atc>__<reporter>__<mutant>[__R<n>]``
    (see ``crops.derive_condition_fields``); this helper recovers each
    field from the underlying ``well`` column when present.
    """
    import pandas as pd

    rows = []
    for label in profile_labels:
        sub = combined_df[combined_df[label_col] == label]
        if sub.empty:
            rows.append({
                "label": label, "condition": label, "gene": "",
                "atc": "", "reporter": "", "replica": "",
                "run_id": "",
                "experiment_type": "knockdown", "n_cells": 0,
                "is_current_run": False, "is_control": False,
            })
            continue
        cond = sub["condition"].iloc[0] if "condition" in sub.columns else label
        run_id = (
            sub["_run_id"].iloc[0]
            if "_run_id" in sub.columns and pd.notna(sub["_run_id"].iloc[0])
            else ""
        )
        exp_type = (
            sub["_experiment_type"].iloc[0]
            if "_experiment_type" in sub.columns
            else "knockdown"
        )
        is_current = bool(sub.get("_is_current_run", pd.Series([True])).any())

        # Parse atc / reporter / mutant / replica from the well stem.
        atc = reporter = replica = ""
        if "well" in sub.columns:
            well = str(sub["well"].iloc[0])
            parts = [p for p in well.split("__")]
            # Strip a focus-stage suffix like "_focused" if present.
            parts = [p.replace("_focused", "") for p in parts]
            if len(parts) > 0:
                atc = parts[0]
            if len(parts) > 1:
                reporter = parts[1]
            if len(parts) > 3:
                p3 = parts[3]
                if len(p3) > 1 and p3[0].lower() == "r" and p3[1:].isdigit():
                    replica = p3[1:]
                else:
                    replica = p3

        # Gene is the first whitespace-separated token of the condition
        # ("<gene> <ATc-/+>").
        gene = str(cond).split()[0] if str(cond).strip() else ""
        rows.append({
            "label": label, "condition": cond, "gene": gene,
            "atc": atc, "reporter": reporter, "replica": replica,
            "run_id": run_id,
            "experiment_type": exp_type or "knockdown",
            "n_cells": int(len(sub)),
            "is_current_run": is_current,
            "is_control": False,
        })

    # Tag controls using the same whole-word matching as S-score baseline
    # selection — the condition label, not the run-tagged combined label.
    if control_labels:
        cond_arr = [r["condition"] for r in rows]
        ctrl_mask = _match_controls(cond_arr, control_labels)
        for r, is_ctrl in zip(rows, ctrl_mask):
            r["is_control"] = bool(is_ctrl)
    return rows


_CONTROL_COLOR = "#9aa0a6"   # mid-grey, distinguishable in both light and dark


def _plot_condition_panel_static(
    profiles, embedding, labels_cluster, meta_rows, title, ax,
):
    """Static matplotlib version: cluster-colored scatter with hover-style
    annotations for the current run. Controls are always grey, regardless
    of cluster assignment, so the baseline reference is visually distinct.
    """
    import numpy as np

    cmap_cl = __import__("matplotlib").colormaps.get_cmap("tab10")
    markers = {"knockdown": "o", "drug": "^"}

    has_control = any(r["is_control"] for r in meta_rows)

    seen_legend: set[str] = set()
    for cl in sorted(set(labels_cluster)):
        mask = np.array(labels_cluster) == cl
        cluster_color = "#cccccc" if cl == -1 else cmap_cl(cl % 10)
        cluster_label = "noise" if cl == -1 else f"cluster {cl}"
        for i in np.where(mask)[0]:
            r = meta_rows[i]
            marker = markers.get(r["experiment_type"], "o")
            is_curr = r["is_current_run"]
            color = _CONTROL_COLOR if r["is_control"] else cluster_color
            ax.scatter(
                embedding[i, 0], embedding[i, 1],
                s=90 if is_curr else 35, alpha=0.85,
                color=color, marker=marker,
                edgecolors="black" if is_curr else "none",
                linewidths=0.6,
            )
            if is_curr:
                ax.annotate(
                    r["condition"], (embedding[i, 0], embedding[i, 1]),
                    fontsize=6, alpha=0.8,
                    xytext=(4, 4), textcoords="offset points",
                )
        if cluster_label not in seen_legend:
            ax.scatter([], [], color=cluster_color, label=cluster_label, marker="o")
            seen_legend.add(cluster_label)
    if has_control:
        ax.scatter([], [], color=_CONTROL_COLOR, label="control", marker="o")
    ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
    ax.legend(fontsize=7, frameon=False, loc="best")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


_CATEGORICAL_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _build_hover_text(profiles, meta_rows, top_features_per_point: int = 5):
    import numpy as np

    feat_cols = list(profiles.columns)
    hover_text = []
    for i, r in enumerate(meta_rows):
        vals = profiles.iloc[i]
        top_idx = np.argsort(np.abs(vals.values))[::-1][:top_features_per_point]
        top_lines = [
            f"  {feat_cols[j]}: {vals.iloc[j]:+.2f}" for j in top_idx
        ]
        ctrl_tag = " · control" if r.get("is_control") else ""
        hover_text.append(
            f"<b>{r['condition']}</b>{ctrl_tag}"
            f"{(' · ' + r['run_id']) if r['run_id'] else ''}"
            f"<br>type: {r['experiment_type']}"
            f"<br>cells: {r['n_cells']:,}"
            f"<br>top S-scores:<br>" + "<br>".join(top_lines)
        )
    return hover_text


def _plot_condition_plotly(
    profiles, embedding, labels_cluster, meta_rows, title,
    color_by: str = "cluster",
    feature_col: str | None = None,
    highlight_genes: list[str] | None = None,
    top_features_per_point: int = 5,
):
    """Build an interactive Plotly figure: single scatter, configurable colour.

    ``color_by``:
      - ``"cluster"`` (default): HDBSCAN cluster id; controls grey
      - ``"run_id"``: each registered run gets its own colour; controls grey
      - ``"feature"``: continuous viridis gradient over ``feature_col``
        (an S-score column from ``profiles``); controls excluded from the
        gradient and overlaid as grey circles for reference

    ``highlight_genes``: when non-empty, points whose ``gene`` is not in
    the list are dimmed (smaller, semi-transparent, no edges). Highlighted
    points are drawn at full size and emphasised.
    Returns a ``plotly.graph_objects.Figure`` ready to ``write_html``.
    """
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()
    hover_text = _build_hover_text(profiles, meta_rows, top_features_per_point)

    highlight_set = {g for g in (highlight_genes or []) if g}
    has_highlight = bool(highlight_set)
    is_highlighted = np.array(
        [(not has_highlight) or (r.get("gene", "") in highlight_set)
         for r in meta_rows],
        dtype=bool,
    )

    sizes = np.array([
        (22 if r["is_current_run"] else 12) if is_highlighted[i]
        else 7
        for i, r in enumerate(meta_rows)
    ])
    line_widths = np.array([
        (2 if r["is_current_run"] else 0) if is_highlighted[i]
        else 0
        for i, r in enumerate(meta_rows)
    ])
    opacities = np.where(is_highlighted, 0.95, 0.25)
    is_ctrl = np.array([r["is_control"] for r in meta_rows], dtype=bool)
    exp_types = [r["experiment_type"] for r in meta_rows]

    if color_by == "feature":
        if feature_col is None or feature_col not in profiles.columns:
            # Fall back to cluster colouring if the feature is missing.
            color_by = "cluster"

    # ── Always draw controls in grey first (consistent across modes) ────
    ctrl_idxs = np.where(is_ctrl)[0].tolist()
    if ctrl_idxs:
        for exp_type in ("knockdown", "drug"):
            sel = [i for i in ctrl_idxs if exp_types[i] == exp_type]
            if not sel:
                continue
            symbol = "circle" if exp_type == "knockdown" else "triangle-up"
            fig.add_trace(go.Scatter(
                x=embedding[sel, 0], y=embedding[sel, 1],
                mode="markers",
                name="control",
                legendgroup="control",
                showlegend=(exp_type == "knockdown"),
                marker=dict(
                    size=sizes[sel].tolist(),
                    color=_CONTROL_COLOR,
                    symbol=symbol,
                    opacity=opacities[sel].tolist(),
                    line=dict(width=line_widths[sel].tolist(), color="black"),
                ),
                hovertext=[hover_text[i] for i in sel],
                hoverinfo="text",
            ))

    non_ctrl_idxs = np.where(~is_ctrl)[0].tolist()

    if color_by == "cluster":
        for cl in sorted(set(labels_cluster)):
            grp = [i for i in non_ctrl_idxs if labels_cluster[i] == cl]
            if not grp:
                continue
            color = "#cccccc" if cl == -1 else _CATEGORICAL_PALETTE[cl % len(_CATEGORICAL_PALETTE)]
            name = "noise" if cl == -1 else f"cluster {cl}"
            for exp_type in ("knockdown", "drug"):
                sel = [i for i in grp if exp_types[i] == exp_type]
                if not sel:
                    continue
                symbol = "circle" if exp_type == "knockdown" else "triangle-up"
                fig.add_trace(go.Scatter(
                    x=embedding[sel, 0], y=embedding[sel, 1],
                    mode="markers",
                    name=f"{name} ({exp_type})" if exp_type == "drug" else name,
                    legendgroup=f"cl{cl}",
                    showlegend=(exp_type == "knockdown"),
                    marker=dict(
                        size=sizes[sel].tolist(),
                        color=color,
                        symbol=symbol,
                        opacity=opacities[sel].tolist(),
                        line=dict(width=line_widths[sel].tolist(), color="black"),
                    ),
                    hovertext=[hover_text[i] for i in sel],
                    hoverinfo="text",
                ))

    elif color_by in ("run_id", "condition", "atc", "reporter", "replica"):
        # Categorical colouring keyed on whatever meta-row attribute the
        # user picked. Empty/missing values get bucketed under "(unknown)".
        key = color_by
        labels_per_point = [
            meta_rows[i].get(key) or "(unknown)" for i in non_ctrl_idxs
        ]
        unique_labels = sorted(set(labels_per_point))
        for k, val in enumerate(unique_labels):
            grp = [
                non_ctrl_idxs[j]
                for j, r in enumerate(labels_per_point) if r == val
            ]
            color = _CATEGORICAL_PALETTE[k % len(_CATEGORICAL_PALETTE)]
            for exp_type in ("knockdown", "drug"):
                sel = [i for i in grp if exp_types[i] == exp_type]
                if not sel:
                    continue
                symbol = "circle" if exp_type == "knockdown" else "triangle-up"
                fig.add_trace(go.Scatter(
                    x=embedding[sel, 0], y=embedding[sel, 1],
                    mode="markers",
                    name=f"{val} ({exp_type})" if exp_type == "drug" else val,
                    legendgroup=f"{key}_{val}",
                    showlegend=(exp_type == "knockdown"),
                    marker=dict(
                        size=sizes[sel].tolist(),
                        color=color,
                        symbol=symbol,
                        opacity=opacities[sel].tolist(),
                        line=dict(width=line_widths[sel].tolist(), color="black"),
                    ),
                    hovertext=[hover_text[i] for i in sel],
                    hoverinfo="text",
                ))

    elif color_by == "feature":
        # One trace per experiment type, coloured continuously along
        # the chosen feature's S-score. Plotly only renders one
        # colorbar per figure, so all non-control traces share the
        # ``coloraxis``.
        feature_vals = profiles[feature_col].values
        for exp_type in ("knockdown", "drug"):
            sel = [i for i in non_ctrl_idxs if exp_types[i] == exp_type]
            if not sel:
                continue
            symbol = "circle" if exp_type == "knockdown" else "triangle-up"
            fig.add_trace(go.Scatter(
                x=embedding[sel, 0], y=embedding[sel, 1],
                mode="markers",
                name=exp_type,
                showlegend=True,
                marker=dict(
                    size=sizes[sel].tolist(),
                    color=feature_vals[sel].tolist(),
                    coloraxis="coloraxis",
                    symbol=symbol,
                    opacity=opacities[sel].tolist(),
                    line=dict(width=line_widths[sel].tolist(), color="black"),
                ),
                hovertext=[hover_text[i] for i in sel],
                hoverinfo="text",
            ))
        fig.update_layout(coloraxis=dict(
            colorscale="Viridis",
            colorbar=dict(
                title=dict(text=feature_col, side="right"),
                thickness=12, len=0.7,
            ),
        ))

    # One label per highlighted point. Multiple (run, ATc) instances of
    # the same gene each get their own label at their own location, so
    # spread / reproducibility is visible directly on the plot rather
    # than only via hover.
    if has_highlight:
        for i, r in enumerate(meta_rows):
            if not is_highlighted[i] or r.get("is_control"):
                continue
            cond = r.get("condition", r.get("gene", ""))
            fig.add_annotation(
                x=float(embedding[i, 0]),
                y=float(embedding[i, 1]),
                text=f"<b>{cond}</b>",
                showarrow=False,
                font=dict(size=11, color="#222"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#444",
                borderwidth=1,
                borderpad=2,
                yshift=14,
            )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        autosize=True,
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(font=dict(size=10)),
        plot_bgcolor="white",
        xaxis=dict(title="Component 1", showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="Component 2", showgrid=True, gridcolor="#eee"),
    )
    return fig


def _make_plot_html_fullheight(html_path: "Path") -> None:
    """Inject a small CSS block into a plotly-generated HTML so the
    ``html`` and ``body`` elements take up the full viewport — without
    this, the body defaults to ``height: auto`` and the plot div sits
    inside extra whitespace at the bottom of the page."""
    try:
        text = html_path.read_text(encoding="utf-8")
    except OSError:
        return
    style = (
        "<style>"
        "html,body{height:100%;margin:0;padding:0;}"
        ".plotly-graph-div{height:100% !important;width:100% !important;}"
        "</style>"
    )
    if "</head>" in text and style not in text:
        text = text.replace("</head>", style + "</head>", 1)
        try:
            html_path.write_text(text, encoding="utf-8")
        except OSError:
            pass


def library_gene_list(
    *,
    library_dir: "Path | None" = None,
    species: str = "",
) -> list[str]:
    """Return the sorted unique gene/mutant tokens registered in the library.

    The condition convention is ``"<gene> <ATc-/+>"``, so we split on the
    first whitespace and dedupe. Used to populate the GUI's "highlight
    genes" multi-select.
    """
    from .feature_library import FeatureLibrary

    try:
        lib = FeatureLibrary(library_dir)
        df = lib.load_species(species)
    except Exception:  # noqa: BLE001
        return []
    if df.empty or "well" not in df.columns:
        return []
    parts = df["well"].astype(str).str.split("__", expand=True)
    if 2 not in parts.columns:
        return []
    mutants = parts[2].astype(str).str.strip()
    return sorted({m for m in mutants.unique() if m and m != "nan"})


def library_feature_columns(
    *,
    library_dir: "Path | None" = None,
    species: str = "",
) -> list[str]:
    """Return the S-score feature column names that ``render_library_html``
    would expose for ``color_by="feature"``. Empty list if the library
    has no matching runs.
    """
    from .feature_library import FeatureLibrary

    try:
        lib = FeatureLibrary(library_dir)
        df_lib = lib.load_species(species)
    except Exception:  # noqa: BLE001
        return []
    if df_lib.empty:
        return []
    morph_cols = _select_morphology_cols(df_lib)
    if len(morph_cols) < 3:
        return []
    # The S-score profile concatenates means + CVs.
    return list(morph_cols) + [c + "_CV" for c in morph_cols]


def render_comparison_html(
    out_path: "Path",
    *,
    library_dir: "Path | None" = None,
    species: str = "",
    genes: list[str] | None = None,
    baseline_mode: str = "pooled",
    top_features: int = 20,
) -> "Path | None":
    """Build an interactive horizontal-bar comparison of S-score profiles.

    All (run, condition) profiles whose gene is in *genes* (plus all
    controls) are drawn together so the user can see, feature-by-feature,
    where the selected gene's instances diverge across runs / ATc states.

    Features are ranked by *spread* (max - min of selected non-control
    profiles) and the top *top_features* are shown. Returns the written
    HTML path, or ``None`` if the library has no usable data.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    from .feature_library import FeatureLibrary

    out_path = Path(out_path)
    if not genes:
        return None

    try:
        lib = FeatureLibrary(library_dir)
        df_lib = lib.load_species(species)
        lib_index = lib.list_runs(species=species or None)
    except Exception:  # noqa: BLE001
        return None
    if df_lib.empty:
        return None

    morph_cols = _select_morphology_cols(df_lib)
    if len(morph_cols) < 3:
        return None

    df_lib = df_lib.dropna(subset=morph_cols).copy()
    if "condition" not in df_lib.columns:
        if "well" in df_lib.columns:
            parts = df_lib["well"].astype(str).str.split("__", expand=True)
            atc_lib = parts[0].str.replace("_focused", "", regex=False)
            mutant_lib = parts[2] if 2 in parts.columns else df_lib["well"].astype(str)
            df_lib["condition"] = (
                mutant_lib.astype(str) + " " + atc_lib.astype(str)
            ).str.strip()
        else:
            df_lib["condition"] = "library"

    df_lib["_run_id"] = df_lib.get("_library_run_id", "library").astype(str)
    df_lib["_combined_label"] = (
        df_lib["condition"].astype(str) + " @ " + df_lib["_run_id"].astype(str)
    )

    controls: list[str] = []
    if "control_labels" in lib_index.columns:
        for lbl in lib_index["control_labels"].dropna().astype(str).tolist():
            for tok in lbl.split(","):
                tok = tok.strip()
                if tok and tok not in controls:
                    controls.append(tok)

    profiles = _compute_condition_sscores(
        df_lib, morph_cols, "_combined_label",
        control_labels=controls,
        baseline_mode=baseline_mode,
    )
    if profiles.empty:
        return None

    # Split into selected (target) profiles and the run-matched controls.
    gene_set = {g.strip() for g in genes if g and g.strip()}
    is_target = []
    is_control = []
    for label in profiles.index:
        cond = str(label).rsplit(" @ ", 1)[0]
        gene = cond.split()[0] if cond.strip() else ""
        is_target.append(gene in gene_set)
        is_control.append(_match_controls([cond], controls)[0] if controls else False)
    is_target = np.array(is_target)
    is_control = np.array(is_control)

    if not is_target.any():
        return None

    target_profiles = profiles.loc[is_target]
    ctrl_profiles = profiles.loc[is_control]

    # Rank features by spread among targets.
    spread = target_profiles.max(axis=0) - target_profiles.min(axis=0)
    top_feat_names = list(spread.sort_values(ascending=False).head(top_features).index)

    # Heatmap: rows = features (top by spread), columns = profiles.
    # Values are S-scores (already control-anchored). Diverging RdBu so
    # +/- of the baseline read as opposite colours.
    z = target_profiles[top_feat_names].T.values  # features × profiles
    profile_labels = list(target_profiles.index)

    # Symmetric colour limits about 0 so the diverging scale stays centred.
    abs_max = float(np.nanmax(np.abs(z))) if z.size else 1.0
    abs_max = max(abs_max, 1.0)

    # Per-cell hover text including control range for context.
    if not ctrl_profiles.empty:
        ctrl_min = ctrl_profiles[top_feat_names].min(axis=0)
        ctrl_max = ctrl_profiles[top_feat_names].max(axis=0)
    else:
        ctrl_min = pd.Series({f: float("nan") for f in top_feat_names})
        ctrl_max = pd.Series({f: float("nan") for f in top_feat_names})

    customdata = [
        [
            [
                float(ctrl_min[feat]) if not np.isnan(ctrl_min[feat]) else 0.0,
                float(ctrl_max[feat]) if not np.isnan(ctrl_max[feat]) else 0.0,
            ]
            for _ in profile_labels
        ]
        for feat in top_feat_names
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=profile_labels,
            y=top_feat_names,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{x}<br>"
                "S-score: %{z:+.2f}<br>"
                "control range: %{customdata[0]:+.2f} to %{customdata[1]:+.2f}"
                "<extra></extra>"
            ),
            colorbar=dict(title=dict(text="S-score", side="right"), thickness=12),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Per-feature S-score heatmap — {', '.join(sorted(gene_set))} "
                 f"({len(target_profiles)} profile(s), baseline: {baseline_mode})",
            x=0.5, xanchor="center",
        ),
        autosize=True,
        margin=dict(l=200, r=20, t=80, b=140),
        plot_bgcolor="white",
        xaxis=dict(
            title="", tickangle=-30,
            showgrid=False,
        ),
        yaxis=dict(
            title="", autorange="reversed",
            showgrid=False,
        ),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        out_path,
        include_plotlyjs=True,
        full_html=True,
        default_height="100vh",
        config={"responsive": True},
    )
    _make_plot_html_fullheight(out_path)
    return out_path


def render_library_html(
    out_path: "Path",
    *,
    library_dir: "Path | None" = None,
    species: str = "",
    color_by: str = "cluster",
    feature_col: str | None = None,
    highlight_genes: list[str] | None = None,
    baseline_mode: str = "pooled",
    batch_correct: bool = True,
) -> "Path | None":
    """Render an interactive Plotly HTML for the library on its own.

    Useful for the Analysis page's default view: shows each
    (run, condition) S-score profile as a point, with hover info and
    a configurable colouring (``cluster`` / ``run_id`` / ``feature``).
    When ``highlight_genes`` is non-empty, points whose mutant/gene is
    not in the list are dimmed.
    Returns the written path, or ``None`` if the library has no usable
    data for *species*.
    """
    import pandas as pd

    from .feature_library import FeatureLibrary

    out_path = Path(out_path)
    try:
        lib = FeatureLibrary(library_dir)
        df_lib = lib.load_species(species)
        lib_index = lib.list_runs(species=species or None)
    except Exception:  # noqa: BLE001
        return None

    if df_lib.empty:
        return None

    morph_cols = _select_morphology_cols(df_lib)
    if len(morph_cols) < 3:
        return None

    df_lib = df_lib.dropna(subset=morph_cols).copy()
    if "condition" not in df_lib.columns:
        if "well" in df_lib.columns:
            parts = df_lib["well"].astype(str).str.split("__", expand=True)
            atc_lib = parts[0].str.replace("_focused", "", regex=False)
            mutant_lib = parts[2] if 2 in parts.columns else df_lib["well"].astype(str)
            df_lib["condition"] = (
                mutant_lib.astype(str) + " " + atc_lib.astype(str)
            ).str.strip()
        else:
            df_lib["condition"] = "library"

    df_lib["_run_id"] = df_lib.get("_library_run_id", "library").astype(str)
    df_lib["_experiment_type"] = df_lib.get(
        "_library_experiment_type", "knockdown",
    ).astype(str).fillna("knockdown")
    df_lib["_is_current_run"] = False
    df_lib["_combined_label"] = (
        df_lib["condition"].astype(str) + " @ " + df_lib["_run_id"].astype(str)
    )

    # Union all controls registered with library runs.
    controls: list[str] = []
    if "control_labels" in lib_index.columns:
        for lbl in lib_index["control_labels"].dropna().astype(str).tolist():
            for tok in lbl.split(","):
                tok = tok.strip()
                if tok and tok not in controls:
                    controls.append(tok)

    profiles = _compute_condition_sscores(
        df_lib, morph_cols, "_combined_label",
        control_labels=controls,
        baseline_mode=baseline_mode,
    )
    if len(profiles) < 2:
        return None

    meta = _condition_meta_table(
        df_lib, profiles.index, "_combined_label",
        control_labels=controls,
    )
    embedding, lbl_cluster = _embed_profiles(profiles, batch_correct=batch_correct)

    n_runs = len(lib_index)
    title = (
        f"Feature library — {n_runs} run(s), species: {species or 'all'}"
    )
    fig = _plot_condition_plotly(
        profiles, embedding, lbl_cluster, meta, title,
        color_by=color_by, feature_col=feature_col,
        highlight_genes=highlight_genes or [],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        out_path,
        include_plotlyjs=True,
        full_html=True,
        default_height="100vh",
        config={"responsive": True},
    )
    # Patch the generated HTML so html/body fill the viewport — plotly's
    # default_height covers the plot div, but the surrounding body still
    # defaults to ``height: auto`` which leaves whitespace beneath.
    _make_plot_html_fullheight(out_path)
    return out_path


def _render_clustering_figure(
    profiles, meta_rows, *, title, out_dir, base_name, progress_cb=_noop,
    batch_correct: bool = True,
):
    """Render a condition-level clustering figure as both PNG and HTML."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(profiles) < 2:
        progress_cb(0.9, f"qc_plots: {base_name}: <2 conditions, skipping")
        return

    embedding, lbl_cluster = _embed_profiles(profiles, batch_correct=batch_correct)

    # Static PNG (single panel — cluster-colored)
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    _plot_condition_panel_static(
        profiles, embedding, lbl_cluster, meta_rows, title, ax,
    )
    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / f"{base_name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Interactive HTML
    try:
        plotly_fig = _plot_condition_plotly(
            profiles, embedding, lbl_cluster, meta_rows, title,
        )
        html_out = out_dir / f"{base_name}.html"
        plotly_fig.write_html(
            html_out,
            include_plotlyjs=True,
            full_html=True,
            default_height="100vh",
            config={"responsive": True},
        )
        _make_plot_html_fullheight(html_out)
    except Exception as e:  # noqa: BLE001
        progress_cb(0.95, f"qc_plots: {base_name}.html skipped ({e})")


def _morphology_cluster_plots(
    df_current,
    *,
    out_dir,
    palette,
    atc_states,
    library_dir=None,
    species="",
    current_run_id="",
    control_labels: list[str] | None = None,
    batch_correct: bool = True,
    progress_cb=_noop,
):
    """Generate condition-level S-score clustering plots (static PNG + Plotly HTML).

    Two figures are produced when a feature library is available:
    ``morphology_clustering_run.{png,html}`` (current run only) and
    ``morphology_clustering_library.{png,html}`` (current run scattered
    against every registered library run, with each (run, condition) as
    its own point so reproducibility is visible).
    """
    import pandas as pd

    morph_cols = _select_morphology_cols(df_current)
    if len(morph_cols) < 3:
        progress_cb(0.9, "qc_plots: too few morphology columns for clustering")
        return

    progress_cb(0.85, "qc_plots: building clustering figures")

    if "condition" not in df_current.columns:
        df_current = df_current.copy()
        df_current["condition"] = (
            df_current.get("mutant", pd.Series("unknown", index=df_current.index)).astype(str)
            + " "
            + df_current.get("atc", pd.Series("", index=df_current.index)).astype(str)
        ).str.strip()

    df_current_clean = df_current.dropna(subset=morph_cols).copy()
    if len(df_current_clean["condition"].unique()) < 2:
        progress_cb(0.9, "qc_plots: <2 conditions in current run, skipping clustering")
        return

    # --- Solo: condition-level S-scores from the current run only ---
    df_solo = df_current_clean.copy()
    df_solo["_run_id"] = current_run_id or "current"
    df_solo["_experiment_type"] = "knockdown"
    df_solo["_is_current_run"] = True

    profiles_solo = _compute_condition_sscores(
        df_solo, morph_cols, "condition",
        control_labels=control_labels or [],
    )
    meta_solo = _condition_meta_table(
        df_solo, profiles_solo.index, "condition",
        control_labels=control_labels or [],
    )
    _render_clustering_figure(
        profiles_solo, meta_solo,
        title="Morphological clustering — current run",
        out_dir=out_dir, base_name="morphology_clustering_run",
        progress_cb=progress_cb,
        batch_correct=batch_correct,
    )
    progress_cb(0.9, "qc_plots: solo clustering done")

    # --- Library comparison ---
    # ``library_dir=None`` is fine — FeatureLibrary falls back to the
    # default location (``~/.mycoprep/feature_library/``).
    try:
        from .feature_library import FeatureLibrary
        lib = FeatureLibrary(library_dir)
        df_lib = lib.load_species(species)
        lib_index = lib.list_runs(species=species or None)
    except Exception as e:  # noqa: BLE001
        progress_cb(0.92, f"qc_plots: library load failed ({e})")
        return

    if df_lib.empty:
        sp_label = species or "any species"
        progress_cb(
            0.92,
            f"qc_plots: library at {lib.library_dir} has no runs for {sp_label}, skipping comparison plot",
        )
        return

    # Union the current run's controls with each registered run's controls.
    combined_controls: list[str] = list(control_labels or [])
    if "control_labels" in lib_index.columns:
        for lbl in lib_index["control_labels"].dropna().astype(str).tolist():
            for tok in lbl.split(","):
                tok = tok.strip()
                if tok and tok not in combined_controls:
                    combined_controls.append(tok)

    progress_cb(0.92, "qc_plots: building library comparison figure")

    lib_morph_cols = [c for c in morph_cols if c in df_lib.columns]
    if len(lib_morph_cols) < 3:
        return

    # Derive a condition column on the library frame (same convention as solo).
    if "condition" not in df_lib.columns:
        if "well" in df_lib.columns:
            parts = df_lib["well"].astype(str).str.split("__", expand=True)
            atc_lib = parts[0].str.replace("_focused", "", regex=False)
            mutant_lib = parts[2] if 2 in parts.columns else df_lib["well"].astype(str)
            df_lib = df_lib.copy()
            df_lib["condition"] = (mutant_lib.astype(str) + " " + atc_lib.astype(str)).str.strip()
        else:
            df_lib = df_lib.copy()
            df_lib["condition"] = "library"

    df_lib = df_lib.copy()
    df_lib["_run_id"] = df_lib.get("_library_run_id", "library").astype(str)
    df_lib["_experiment_type"] = df_lib.get(
        "_library_experiment_type", "knockdown",
    ).astype(str).fillna("knockdown")
    df_lib["_is_current_run"] = False

    df_current_tagged = df_current_clean.copy()
    df_current_tagged["_run_id"] = current_run_id or "current"
    df_current_tagged["_experiment_type"] = "knockdown"
    df_current_tagged["_is_current_run"] = True

    keep_cols = lib_morph_cols + [
        "condition", "_run_id", "_experiment_type", "_is_current_run",
    ]
    combined = pd.concat(
        [df_current_tagged[keep_cols], df_lib[keep_cols]],
        ignore_index=True,
    ).dropna(subset=lib_morph_cols)

    # Each (run, condition) gets its own profile so reproducibility is
    # visible — same condition imaged in different runs lands as multiple
    # dots, near each other if the experiments agree.
    combined["_combined_label"] = (
        combined["condition"].astype(str) + " @ " + combined["_run_id"].astype(str)
    )

    profiles_lib = _compute_condition_sscores(
        combined, lib_morph_cols, "_combined_label",
        control_labels=combined_controls,
    )
    meta_lib = _condition_meta_table(
        combined, profiles_lib.index, "_combined_label",
        control_labels=combined_controls,
    )

    n_lib_runs = len(lib.list_runs(species=species or None))
    title = (
        f"Morphological clustering — '{current_run_id or 'current'}' vs library "
        f"({n_lib_runs} runs, species: {species or 'all'})"
    )
    _render_clustering_figure(
        profiles_lib, meta_lib,
        title=title,
        out_dir=out_dir, base_name="morphology_clustering_library",
        progress_cb=progress_cb,
        batch_correct=batch_correct,
    )
    progress_cb(0.95, "qc_plots: library clustering done")


# ──────────────────────────────────────────────────────────────────────────────
# CNN Embedding UMAP (for the Analysis panel "CNN embeddings" view)
# ──────────────────────────────────────────────────────────────────────────────


def available_embedding_models(library_dir: "Path | None" = None) -> list[dict]:
    """Enumerate trained CNN embedding models with extracted parquets.

    Returns a list of ``{model_type, path, mtime}`` dicts, one per
    architecture subdirectory under ``<models_dir>/embeddings/`` that has
    a ``cnn_embeddings.parquet``. Sorted by mtime descending (most recent
    first). Used to populate the Analysis panel's model selector.
    """
    from .feature_library import FeatureLibrary
    out: list[dict] = []
    try:
        lib = FeatureLibrary(library_dir)
    except Exception:  # noqa: BLE001
        return out
    emb_dir = lib.models_dir / "embeddings"
    if not emb_dir.exists():
        return out
    flat = emb_dir / "cnn_embeddings.parquet"
    if flat.exists():
        out.append({
            "model_type": "(flat layout)",
            "path": flat,
            "mtime": flat.stat().st_mtime,
        })
    for sub in emb_dir.iterdir():
        if sub.is_dir():
            p = sub / "cnn_embeddings.parquet"
            if p.exists():
                out.append({
                    "model_type": sub.name,
                    "path": p,
                    "mtime": p.stat().st_mtime,
                })
    out.sort(key=lambda d: d["mtime"], reverse=True)
    return out


def render_embeddings_html(
    out_path: "Path",
    *,
    library_dir: "Path | None" = None,
    species: str = "",
    color_by: str = "cluster",
    feature_col: str | None = None,
    highlight_genes: list[str] | None = None,
    batch_correct: bool = True,
    model_type: str = "",
) -> "Path | None":
    """Render UMAP of CNN embeddings as interactive Plotly HTML.

    ``model_type`` selects which trained model's embeddings to load (one
    of the architecture subdirectories under ``<models_dir>/embeddings/``).
    Empty string = use the most recently-modified one.
    """
    import pandas as pd

    from .feature_library import FeatureLibrary

    out_path = Path(out_path)
    try:
        lib = FeatureLibrary(library_dir)
    except Exception:  # noqa: BLE001
        return None

    # Look for embeddings under <models_dir>/embeddings/. New runs write
    # to per-model-type subdirectories (e.g. ``embeddings/resnet18/``,
    # ``embeddings/supcon_resnet18/``) so different architectures coexist;
    # legacy runs wrote a flat layout directly under ``embeddings/``.
    emb_dir = lib.models_dir / "embeddings"
    emb_path: "Path | None" = None
    if model_type:
        # Explicit selection. Try subdirectory first; fall back to flat
        # layout if the user picked the special "(flat layout)" entry.
        cand = emb_dir / model_type / "cnn_embeddings.parquet"
        if cand.exists():
            emb_path = cand
        elif model_type == "(flat layout)":
            cand = emb_dir / "cnn_embeddings.parquet"
            if cand.exists():
                emb_path = cand
    else:
        # Latest available across all subdirectories + flat layout.
        candidates: list["Path"] = []
        if emb_dir.exists():
            flat = emb_dir / "cnn_embeddings.parquet"
            if flat.exists():
                candidates.append(flat)
            for sub in emb_dir.iterdir():
                if sub.is_dir():
                    p = sub / "cnn_embeddings.parquet"
                    if p.exists():
                        candidates.append(p)
        if candidates:
            emb_path = max(candidates, key=lambda p: p.stat().st_mtime)
    if emb_path is None or not emb_path.exists():
        # Fallback: scan individual run feature dirs (very legacy layout).
        idx = lib.list_runs(species=species or None)
        if idx.empty:
            return None
        for _, row in idx.iloc[::-1].iterrows():
            features_path = lib.library_dir / row["features_file"]
            alt_path = features_path.parent / "embeddings" / "cnn_embeddings.parquet"
            if alt_path.exists():
                emb_path = alt_path
                break
        if emb_path is None or not emb_path.exists():
            return None

    df = pd.read_parquet(emb_path)
    if df.empty:
        return None

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if len(emb_cols) < 10:
        return None

    has_multi_runs = (
        "run_id" in df.columns and df["run_id"].nunique() > 1
    )

    # Detect biological controls from the library's per-run control_labels
    # column. NT/NT1/NT2 etc. are independent biological replicates of the
    # same non-targeting reagent — they're kept as separate points (they
    # aren't more similar to themselves across runs than to each other
    # within a run, so collapsing would be misleading), but they all
    # contribute to the per-run NT anchor used below.
    control_genes: set[str] = set()
    try:
        ctrl_idx = lib.list_runs(species=species or None)
        # ``ctrl_idx`` is a DataFrame; `.get(col, [])` on a DataFrame
        # returns the column Series (or default). Don't combine that with
        # `or` — booling a Series raises "truth value is ambiguous".
        ctrl_col = (
            ctrl_idx["control_labels"]
            if "control_labels" in ctrl_idx.columns
            else []
        )
        for raw in ctrl_col:
            for tok in str(raw).split(","):
                tok = tok.strip()
                if tok:
                    control_genes.add(tok)
    except Exception:  # noqa: BLE001
        pass

    # Aggregate to per-(run, condition) profiles FIRST. Harmony is then
    # applied at this profile level rather than per-cell — the convention
    # used in image-based morphological profiling (Nature Comms 2024
    # 41467-024-50613-5 and similar JUMP-Cell-Painting work) and what we
    # already do for the S-score pipeline. Profile-level Harmony is more
    # stable with few batches: the algorithm doesn't have to invent a
    # cell-level subcluster structure that the data may not support.
    if has_multi_runs and "condition_label" in df.columns:
        profiles = (
            df.groupby(["run_id", "condition_label"])[emb_cols]
            .mean()
            .reset_index()
        )
        # For UMAP / colouring we still want a single label string per point.
        profiles["condition_label"] = (
            profiles["condition_label"].astype(str)
            + " ("
            + profiles["run_id"].astype(str)
            + ")"
        )
    elif "condition_label" in df.columns:
        profiles = df.groupby("condition_label")[emb_cols].mean().reset_index()
        profiles["run_id"] = (
            df["run_id"].iloc[0] if "run_id" in df.columns else "unknown"
        )
    else:
        profiles = df[emb_cols].copy()
        profiles["condition_label"] = "unknown"
        profiles["run_id"] = "unknown"

    if len(profiles) < 3:
        return None

    # Parse gene from the original condition_label (strip the trailing
    # "(run_id)" tag we may have added when splitting by run).
    profiles["gene"] = (
        profiles["condition_label"]
        .str.replace(r"\s*\([^)]*\)\s*$", "", regex=True)
        .str.split().str[0]
    )

    is_control = profiles["gene"].isin(control_genes) if control_genes else None

    # ── Pipeline order: NT-anchor → PCA → Harmony → UMAP ──
    # This is the order used in the S-score feature pipeline. Doing
    # NT-anchoring before PCA/Harmony is analogous to how S-scores are
    # already control-normalised at compute time. Harmony then corrects
    # any *residual* per-run technical variation in PCA space (where
    # cluster geometry is well-defined and N >> dim, unlike raw 512-d
    # where Harmony's k-means is unstable). Critical: the previous code
    # ran Harmony BEFORE NT-anchoring, then NT-anchored on top. That
    # subtracted slightly-different per-run NT means from already-aligned
    # data, re-introducing batch differences that UMAP then amplified —
    # symptom: "Harmony separates the batches".

    # 1. NT-anchor (gene-level control normalisation).
    profiles_arr = profiles[emb_cols].values.astype(float).copy()
    can_per_run_anchor = (
        is_control is not None
        and is_control.sum() >= 1
        and "run_id" in profiles.columns
        and profiles["run_id"].nunique() > 1
    )
    if can_per_run_anchor:
        run_anchors = (
            profiles[is_control].groupby("run_id")[emb_cols].mean()
        )
        global_nt = profiles.loc[is_control, emb_cols].mean().values
        for rid, group_idx in profiles.groupby("run_id").indices.items():
            a = (
                run_anchors.loc[rid].values
                if rid in run_anchors.index else global_nt
            )
            profiles_arr[group_idx] -= a
        anchor_label = f"per-run NT-relative ({sorted(control_genes)})"
    elif is_control is not None and is_control.sum() >= 1:
        profiles_arr -= profiles.loc[is_control, emb_cols].mean().values
        anchor_label = f"NT-relative ({sorted(control_genes)})"
    else:
        profiles_arr -= df[emb_cols].mean().values
        anchor_label = "global-mean centered (no controls tagged)"

    profiles[emb_cols] = profiles_arr

    # 2. PCA → 50-d (or fewer if data is small).
    import numpy as np
    emb_matrix = profiles[emb_cols].values.astype(np.float32)

    try:
        from sklearn.decomposition import PCA
        import umap

        n_components_pca = min(50, emb_matrix.shape[1], emb_matrix.shape[0] - 1)
        pca = PCA(n_components=n_components_pca)
        reduced = pca.fit_transform(emb_matrix)

        # 3. Harmony in PCA space (more stable than raw 512-d).
        if batch_correct and has_multi_runs:
            try:
                import harmonypy
                nclust = max(2, min(20, len(profiles) // 4))
                ho = harmonypy.run_harmony(
                    reduced,
                    pd.DataFrame({"run_id": profiles["run_id"].values}),
                    vars_use="run_id",
                    max_iter_harmony=20,
                    nclust=nclust,
                )
                reduced = ho.Z_corr
            except ImportError:
                pass
            except Exception:  # noqa: BLE001
                pass

        # 4. UMAP.
        n_neighbors = min(15, len(profiles) - 1)
        n_neighbors = max(2, n_neighbors)
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42
        )
        coords = reducer.fit_transform(reduced)
    except ImportError:
        # Fallback to PCA 2D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(emb_matrix)

    # Defragment + add coords in one go to avoid pandas' "highly fragmented"
    # warning that fires when a DataFrame gets many sequential `[col] = ...`
    # assignments.
    profiles = profiles.copy().assign(
        umap_x=coords[:, 0], umap_y=coords[:, 1],
    )

    # Optional: overlay morphological feature from library data
    if feature_col and color_by == "feature":
        try:
            lib_features = lib.load_species(species)
            if not lib_features.empty and "well" in lib_features.columns:
                parts = lib_features["well"].astype(str).str.split("__", expand=True)
                atc = parts[0].str.replace("_focused", "", regex=False)
                mutant = parts[2] if 2 in parts.columns else lib_features["well"].astype(str)
                lib_features["_cond"] = (mutant.astype(str) + " " + atc.astype(str)).str.strip()
                if feature_col in lib_features.columns:
                    feat_means = lib_features.groupby("_cond")[feature_col].mean()
                    profiles["_feature_val"] = profiles["condition_label"].map(feat_means)
        except Exception:  # noqa: BLE001
            pass

    # Render with Plotly
    try:
        import plotly.express as px

        if color_by == "feature" and "_feature_val" in profiles.columns:
            fig = px.scatter(
                profiles, x="umap_x", y="umap_y",
                color="_feature_val",
                hover_data=["condition_label", "run_id", "gene"],
                color_continuous_scale="Viridis",
                labels={"_feature_val": feature_col or "feature"},
            )
        elif color_by == "run_id":
            fig = px.scatter(
                profiles, x="umap_x", y="umap_y",
                color="run_id",
                hover_data=["condition_label", "gene"],
            )
        else:
            fig = px.scatter(
                profiles, x="umap_x", y="umap_y",
                color="gene",
                hover_data=["condition_label", "run_id"],
            )

        # Highlight genes
        if highlight_genes:
            hl_set = {g.lower() for g in highlight_genes}
            profiles["_highlighted"] = profiles["gene"].str.lower().isin(hl_set)
            for trace in fig.data:
                if hasattr(trace, "customdata") and trace.customdata is not None:
                    pass  # opacity handled below
            fig.update_traces(
                marker=dict(size=8, opacity=0.3),
                selector=lambda t: True,
            )
            hl_profiles = profiles[profiles["_highlighted"]]
            if not hl_profiles.empty:
                fig.add_scatter(
                    x=hl_profiles["umap_x"], y=hl_profiles["umap_y"],
                    mode="markers",
                    marker=dict(size=12, color="red", opacity=1.0),
                    text=hl_profiles["condition_label"],
                    hoverinfo="text",
                    name="Highlighted",
                )

        harmony_status = ""
        if has_multi_runs:
            harmony_status = (
                " · Harmony" if batch_correct else " · raw (no batch correct)"
            )
        fig.update_layout(
            title=(
                f"CNN Embedding UMAP — {anchor_label}{harmony_status}"
                + (f" · model: {emb_path.parent.name}"
                   if emb_path.parent.name != "embeddings" else "")
            ),
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            template="plotly_white",
            margin=dict(l=40, r=20, t=60, b=40),
        )

        # Match the Feature Profiles renderer: viewport-height default
        # plus fullheight CSS injection.
        fig.write_html(
            str(out_path),
            include_plotlyjs=True,
            default_height="100vh",
        )
        _make_plot_html_fullheight(out_path)
        return out_path
    except ImportError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# CNN Embedding UMAP via Optimal Transport (Sinkhorn divergence)
# ──────────────────────────────────────────────────────────────────────────────


def _sinkhorn_divergence_matrix(
    point_clouds: list["np.ndarray"],
    reg: float = 0.1,
    n_iter_max: int = 2000,
    progress_cb=None,
) -> "np.ndarray":
    """Pairwise Sinkhorn divergence between point clouds.

    Sinkhorn divergence ([Feydy 2019]) is the entropic-regularised OT cost
    debiased to be a proper divergence:

        SD(μ, ν) = OT_ε(μ, ν) − ½·OT_ε(μ, μ) − ½·OT_ε(ν, ν)

    Returns a symmetric (n, n) matrix of divergences. Each cloud is
    treated as uniformly-weighted.

    ``reg`` is interpreted as a **fraction of the typical (median) cost
    matrix value**, not an absolute number. Sinkhorn's regularisation
    has to be calibrated to the cost-matrix scale or it under- or
    over-regularises silently — the original ``ε=0.1`` worked for unit-
    scale embeddings but was ~5000× too small for raw 512-d sqEuclidean
    distances (typical ~500), giving meaningless distances. With this
    rescaling, ``reg=0.05`` means "5% of typical cost" regardless of
    embedding magnitude.
    """
    import numpy as np
    import ot

    n = len(point_clouds)
    if n == 0:
        return np.zeros((0, 0))

    # Estimate the typical cost scale once, from a sample of inter-cloud
    # cost matrices, so the same effective regularisation is used for
    # every pair.
    rng = np.random.default_rng(0)
    sample_pairs = min(20, n * (n - 1) // 2 if n > 1 else 1)
    sample_costs = []
    if n > 1:
        for _ in range(sample_pairs):
            i, j = rng.choice(n, 2, replace=False)
            M = ot.dist(point_clouds[i], point_clouds[j], metric="sqeuclidean")
            sample_costs.append(float(np.median(M)))
    cost_scale = float(np.median(sample_costs)) if sample_costs else 1.0
    if cost_scale <= 0:
        cost_scale = 1.0
    abs_reg = reg * cost_scale

    def _sinkhorn(M: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        # Log-stabilised Sinkhorn — same as ot.sinkhorn2 but in log-space
        # so it stays numerically well-behaved at small reg / large M.
        return float(
            ot.sinkhorn2(
                a, b, M, reg=abs_reg,
                numItermax=n_iter_max,
                method="sinkhorn_log",
            )
        )

    self_terms = []
    for pc in point_clouds:
        a = np.full(len(pc), 1.0 / max(len(pc), 1))
        M = ot.dist(pc, pc, metric="sqeuclidean")
        self_terms.append(_sinkhorn(M, a, a))

    D = np.zeros((n, n), dtype=np.float64)
    total_pairs = n * (n - 1) // 2
    done = 0
    for i in range(n):
        ai = np.full(len(point_clouds[i]), 1.0 / max(len(point_clouds[i]), 1))
        for j in range(i + 1, n):
            bj = np.full(len(point_clouds[j]), 1.0 / max(len(point_clouds[j]), 1))
            M = ot.dist(point_clouds[i], point_clouds[j], metric="sqeuclidean")
            cost = _sinkhorn(M, ai, bj)
            div = cost - 0.5 * (self_terms[i] + self_terms[j])
            D[i, j] = D[j, i] = max(div, 0.0)
            done += 1
            if progress_cb is not None and total_pairs > 0:
                progress_cb(done / total_pairs)
    return D


def render_embeddings_ot_html(
    out_path: "Path",
    *,
    library_dir: "Path | None" = None,
    species: str = "",
    color_by: str = "cluster",
    feature_col: str | None = None,
    highlight_genes: list[str] | None = None,
    batch_correct: bool = True,
    model_type: str = "",
    n_cells_per_condition: int = 400,
    sinkhorn_reg: float = 0.05,
    progress_cb=None,
) -> "Path | None":
    """Render UMAP of CNN embeddings with **Optimal Transport** distance.

    Instead of collapsing each (run, condition) to its mean and comparing
    means, this computes pairwise Sinkhorn divergences between full cell
    point clouds, then UMAP-projects the resulting distance matrix. Two
    failure modes the mean-based view hides:

    - **Heterogeneous knockdowns** — a gene producing 50% normal + 50%
      elongated cells has the same mean as one producing 100% medium-
      elongated; OT separates them.
    - **Variance differences** — a partially-penetrant knockdown vs.
      a tight off-control cluster.

    Embeddings are loaded from the same per-architecture subdirectory as
    :func:`render_embeddings_html`, so this works on autoencoder,
    SupCon, or any future trained model.

    Computational cost: ``n_cells_per_condition`` cells per group, then
    O((n_groups² / 2) × n²) Sinkhorn iterations. With 84 groups × 400
    cells ≈ a few seconds on CPU.
    """
    import pandas as pd

    from .feature_library import FeatureLibrary

    out_path = Path(out_path)
    if progress_cb is None:
        progress_cb = lambda f: None

    try:
        lib = FeatureLibrary(library_dir)
    except Exception:  # noqa: BLE001
        return None

    # Same model-discovery logic as render_embeddings_html.
    emb_dir = lib.models_dir / "embeddings"
    emb_path: "Path | None" = None
    if model_type:
        cand = emb_dir / model_type / "cnn_embeddings.parquet"
        if cand.exists():
            emb_path = cand
        elif model_type == "(flat layout)":
            cand = emb_dir / "cnn_embeddings.parquet"
            if cand.exists():
                emb_path = cand
    else:
        candidates: list["Path"] = []
        if emb_dir.exists():
            flat = emb_dir / "cnn_embeddings.parquet"
            if flat.exists():
                candidates.append(flat)
            for sub in emb_dir.iterdir():
                if sub.is_dir():
                    p = sub / "cnn_embeddings.parquet"
                    if p.exists():
                        candidates.append(p)
        if candidates:
            emb_path = max(candidates, key=lambda p: p.stat().st_mtime)
    if emb_path is None or not emb_path.exists():
        return None

    df = pd.read_parquet(emb_path)
    if df.empty:
        return None

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if len(emb_cols) < 10:
        return None

    has_multi_runs = (
        "run_id" in df.columns and df["run_id"].nunique() > 1
    )

    # Detect controls (same as the mean-based renderer).
    control_genes: set[str] = set()
    try:
        ctrl_idx = lib.list_runs(species=species or None)
        # ``ctrl_idx`` is a DataFrame; `.get(col, [])` on a DataFrame
        # returns the column Series (or default). Don't combine that with
        # `or` — booling a Series raises "truth value is ambiguous".
        ctrl_col = (
            ctrl_idx["control_labels"]
            if "control_labels" in ctrl_idx.columns
            else []
        )
        for raw in ctrl_col:
            for tok in str(raw).split(","):
                tok = tok.strip()
                if tok:
                    control_genes.add(tok)
    except Exception:  # noqa: BLE001
        pass

    # Per-cell preprocessing: optional Harmony, optional per-run NT
    # subtraction. Done at cell level here (not profile) since OT
    # operates on the full cloud.
    import numpy as np

    if batch_correct and has_multi_runs:
        try:
            import harmonypy
            from sklearn.decomposition import PCA
            # Reduce to ~50-d before Harmony (same logic as S-score path).
            n_pca = min(50, len(emb_cols), max(2, len(df) // 10))
            pca = PCA(n_components=n_pca)
            X_pca = pca.fit_transform(df[emb_cols].values.astype(np.float32))
            ho = harmonypy.run_harmony(
                X_pca,
                pd.DataFrame({"run_id": df["run_id"].values}),
                vars_use="run_id",
                max_iter_harmony=20,
                nclust=min(20, max(2, len(df) // 200)),
            )
            # Replace embeddings columns with corrected (and PCA-reduced)
            # representation. OT works in any dim — the reduced form is
            # actually faster.
            emb_cols = [f"hpc_{i}" for i in range(ho.Z_corr.shape[1])]
            df = df.drop(columns=[c for c in df.columns if c.startswith("emb_")])
            for i, c in enumerate(emb_cols):
                df[c] = ho.Z_corr[:, i]
        except ImportError:
            pass
        except Exception:  # noqa: BLE001
            pass

    df["gene"] = df["condition_label"].str.split().str[0]

    # Per-run NT centering at the cell level. Done before OT, so the
    # point clouds are already control-normalised.
    if "run_id" in df.columns and control_genes:
        is_ctrl_cell = df["gene"].isin(control_genes)
        if is_ctrl_cell.any():
            X = df[emb_cols].values.astype(np.float32)
            for rid, idx in df.groupby("run_id").indices.items():
                rid_ctrl_mask = is_ctrl_cell.values[idx]
                if rid_ctrl_mask.any():
                    a = X[idx][rid_ctrl_mask].mean(axis=0)
                    X[idx] = X[idx] - a
            df[emb_cols] = X

    # Group by (run_id, condition_label), sample n cells per group.
    group_cols = (
        ["run_id", "condition_label"] if has_multi_runs else ["condition_label"]
    )
    groups = df.groupby(group_cols)
    rng = np.random.default_rng(42)
    point_clouds: list[np.ndarray] = []
    group_meta: list[dict] = []
    for key, gdf in groups:
        n = min(n_cells_per_condition, len(gdf))
        if n < 5:
            continue  # too few cells; skip
        idx = rng.choice(len(gdf), size=n, replace=False)
        pc = gdf[emb_cols].values[idx].astype(np.float32)
        point_clouds.append(pc)
        if isinstance(key, tuple):
            run_id, cond = key
        else:
            run_id, cond = "unknown", key
        group_meta.append({
            "run_id": str(run_id),
            "condition_label": str(cond),
            "gene": str(cond).split()[0],
            "n_cells_used": n,
        })
    if len(point_clouds) < 3:
        return None

    progress_cb(0.1)

    # Sinkhorn divergence matrix (the expensive bit).
    try:
        D = _sinkhorn_divergence_matrix(
            point_clouds,
            reg=sinkhorn_reg,
            n_iter_max=200,
            progress_cb=lambda f: progress_cb(0.1 + 0.7 * f),
        )
    except ImportError:
        # POT not installed → render an explanatory placeholder.
        msg = (
            "<html><body style='font-family: sans-serif; padding: 40px;'>"
            "<h2>Optimal-Transport view requires <code>POT</code></h2>"
            "<p>Install with: <code>pip install pot</code></p>"
            "</body></html>"
        )
        out_path.write_text(msg, encoding="utf-8")
        return out_path

    progress_cb(0.85)

    # Symmetrise (already symmetric within numerical noise) and convert
    # to a non-negative distance for UMAP. Sinkhorn divergence is squared-
    # distance-like; take sqrt for a proper-distance feel before UMAP.
    np.fill_diagonal(D, 0.0)
    D = np.sqrt(np.maximum(D, 0.0))

    # UMAP from precomputed distance.
    try:
        import umap
        n_neighbors = min(15, len(point_clouds) - 1)
        n_neighbors = max(2, n_neighbors)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="precomputed",
            random_state=42,
            n_jobs=1,
        )
        coords = reducer.fit_transform(D)
    except Exception:  # noqa: BLE001
        # Fallback: classical MDS.
        from sklearn.manifold import MDS
        coords = MDS(
            n_components=2, dissimilarity="precomputed", random_state=42,
        ).fit_transform(D)

    progress_cb(0.95)

    # Build Plotly DataFrame.
    plot_df = pd.DataFrame(group_meta)
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]

    try:
        import plotly.express as px

        if color_by == "run_id":
            fig = px.scatter(
                plot_df, x="x", y="y", color="run_id",
                hover_data=["condition_label", "gene", "n_cells_used"],
            )
        else:
            fig = px.scatter(
                plot_df, x="x", y="y", color="gene",
                hover_data=["condition_label", "run_id", "n_cells_used"],
            )

        if highlight_genes:
            hl_set = {g.lower() for g in highlight_genes}
            mask = plot_df["gene"].str.lower().isin(hl_set)
            fig.update_traces(marker=dict(size=8, opacity=0.3))
            hl = plot_df[mask]
            if not hl.empty:
                fig.add_scatter(
                    x=hl["x"], y=hl["y"], mode="markers",
                    marker=dict(size=12, color="red", opacity=1.0),
                    text=hl["condition_label"], hoverinfo="text",
                    name="Highlighted",
                )

        harmony_status = ""
        if has_multi_runs:
            harmony_status = (
                " · Harmony" if batch_correct else " · raw"
            )
        title = (
            f"CNN Embeddings — Optimal Transport "
            f"(Sinkhorn ε = {sinkhorn_reg * 100:.1f}% of typical cost)"
            f"{harmony_status}"
            f" · {len(point_clouds)} groups × {n_cells_per_condition} cells"
        )
        if emb_path.parent.name != "embeddings":
            title += f" · model: {emb_path.parent.name}"
        fig.update_layout(
            title=title,
            xaxis_title="UMAP 1 (from OT distance)",
            yaxis_title="UMAP 2 (from OT distance)",
            template="plotly_white",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        fig.write_html(
            str(out_path),
            include_plotlyjs=True,
            default_height="100vh",
        )
        _make_plot_html_fullheight(out_path)
        progress_cb(1.0)
        return out_path
    except ImportError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Feature-profile UMAP via Optimal Transport
# ──────────────────────────────────────────────────────────────────────────────


def render_features_ot_html(
    out_path: "Path",
    *,
    library_dir: "Path | None" = None,
    species: str = "",
    color_by: str = "cluster",
    feature_col: str | None = None,
    highlight_genes: list[str] | None = None,
    batch_correct: bool = True,
    n_cells_per_condition: int = 400,
    sinkhorn_reg: float = 0.05,
    progress_cb=None,
) -> "Path | None":
    """OT-based UMAP of S-score morphological profiles.

    Same idea as :func:`render_embeddings_ot_html` but operating on the
    cell-level morphology features (length, width, intensity stats, etc.)
    in ``all_features.parquet`` instead of the CNN embedding parquet.

    For each (run, condition) cell cloud, computes the pairwise Sinkhorn
    divergence and UMAPs the resulting distance matrix. Captures
    distribution shape (heterogeneity, variance) on top of the per-
    feature mean comparison the standard Feature Profiles view does.

    Features are z-scored across all cells before OT so distance isn't
    dominated by whichever feature has the largest scale.
    """
    import pandas as pd

    from .feature_library import FeatureLibrary

    out_path = Path(out_path)
    if progress_cb is None:
        progress_cb = lambda f: None

    try:
        lib = FeatureLibrary(library_dir)
        df_lib = lib.load_species(species)
    except Exception:  # noqa: BLE001
        return None
    if df_lib.empty:
        return None

    morph_cols = _select_morphology_cols(df_lib)
    if len(morph_cols) < 3:
        return None

    # Run / condition / gene metadata. ``feature_library.load_species``
    # adds ``_library_run_id`` per cell which we treat as the canonical
    # batch label. We deliberately don't rename to ``run_id`` because the
    # underlying per-cell features may already have a ``run_id`` column
    # from extract_features (collision → 2D Series → groupby crashes).
    df = df_lib.copy()
    if "_library_run_id" not in df.columns:
        df["_library_run_id"] = "unknown"
    # Drop any conflicting per-cell run_id so groupby has a single source
    # of truth, then alias the library one to ``run_id`` for readability.
    if "run_id" in df.columns:
        df = df.drop(columns=["run_id"])
    df = df.rename(columns={"_library_run_id": "run_id"})

    if "well" not in df.columns:
        return None
    parts = df["well"].astype(str).str.split("__", expand=True)
    atc = parts[0].str.replace("_focused", "", regex=False)
    mutant = parts[2] if 2 in parts.columns else df["well"].astype(str)
    df["condition_label"] = (mutant.astype(str) + " " + atc.astype(str)).str.strip()
    df["gene"] = mutant.astype(str)

    has_multi_runs = df["run_id"].nunique() > 1

    # Detect controls.
    control_genes: set[str] = set()
    try:
        ctrl_idx = lib.list_runs(species=species or None)
        # ``ctrl_idx`` is a DataFrame; `.get(col, [])` on a DataFrame
        # returns the column Series (or default). Don't combine that with
        # `or` — booling a Series raises "truth value is ambiguous".
        ctrl_col = (
            ctrl_idx["control_labels"]
            if "control_labels" in ctrl_idx.columns
            else []
        )
        for raw in ctrl_col:
            for tok in str(raw).split(","):
                tok = tok.strip()
                if tok:
                    control_genes.add(tok)
    except Exception:  # noqa: BLE001
        pass

    import numpy as np

    # Standardise features to unit variance so OT distances aren't
    # dominated by whichever feature happens to have the largest scale
    # (length in µm vs. intensity in raw counts, etc.). Z-scoring puts
    # everything on the same scale, but heavy-tailed distributions (cell
    # area, intensity outliers) leave a few cells at ±20–80 σ which then
    # dominate the OT cost matrix and break Sinkhorn convergence. Clip to
    # ±5 σ after z-scoring — keeps almost all the signal, kills outliers.
    X = df[morph_cols].values.astype(np.float32)
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma[sigma < 1e-8] = 1.0
    X = (X - mu) / sigma
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -5.0, 5.0)
    df[morph_cols] = X

    # Optional Harmony at cell level (keeps things consistent with the
    # CNN-OT path; the feature space is small enough that a PCA reduction
    # isn't necessary first).
    if batch_correct and has_multi_runs:
        try:
            import harmonypy
            ho = harmonypy.run_harmony(
                df[morph_cols].values.astype(np.float32),
                pd.DataFrame({"run_id": df["run_id"].values}),
                vars_use="run_id",
                max_iter_harmony=20,
                nclust=min(20, max(2, len(df) // 200)),
            )
            df[morph_cols] = ho.Z_corr
        except ImportError:
            pass
        except Exception:  # noqa: BLE001
            pass

    # Per-run NT centring at the cell level.
    if control_genes:
        is_ctrl = df["gene"].isin(control_genes)
        if is_ctrl.any():
            X = df[morph_cols].values.astype(np.float32)
            for rid, idx in df.groupby("run_id").indices.items():
                rid_ctrl = is_ctrl.values[idx]
                if rid_ctrl.any():
                    a = X[idx][rid_ctrl].mean(axis=0)
                    X[idx] = X[idx] - a
            df[morph_cols] = X

    # Group + sample.
    group_cols = ["run_id", "condition_label"] if has_multi_runs else ["condition_label"]
    rng = np.random.default_rng(42)
    point_clouds: list[np.ndarray] = []
    group_meta: list[dict] = []
    for key, gdf in df.groupby(group_cols):
        n = min(n_cells_per_condition, len(gdf))
        if n < 5:
            continue
        idx = rng.choice(len(gdf), size=n, replace=False)
        point_clouds.append(gdf[morph_cols].values[idx].astype(np.float32))
        if isinstance(key, tuple):
            run_id, cond = key
        else:
            run_id, cond = "unknown", key
        group_meta.append({
            "run_id": str(run_id),
            "condition_label": str(cond),
            "gene": str(cond).split()[0],
            "n_cells_used": n,
        })
    if len(point_clouds) < 3:
        return None

    progress_cb(0.1)

    try:
        D = _sinkhorn_divergence_matrix(
            point_clouds,
            reg=sinkhorn_reg,
            n_iter_max=200,
            progress_cb=lambda f: progress_cb(0.1 + 0.7 * f),
        )
    except ImportError:
        msg = (
            "<html><body style='font-family: sans-serif; padding: 40px;'>"
            "<h2>Optimal-Transport view requires <code>POT</code></h2>"
            "<p>Install with: <code>pip install pot</code></p>"
            "</body></html>"
        )
        out_path.write_text(msg, encoding="utf-8")
        return out_path

    np.fill_diagonal(D, 0.0)
    D = np.sqrt(np.maximum(D, 0.0))

    progress_cb(0.85)

    try:
        import umap
        n_neighbors = max(2, min(15, len(point_clouds) - 1))
        coords = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
            metric="precomputed", random_state=42, n_jobs=1,
        ).fit_transform(D)
    except Exception:  # noqa: BLE001
        from sklearn.manifold import MDS
        coords = MDS(
            n_components=2, dissimilarity="precomputed", random_state=42,
        ).fit_transform(D)

    plot_df = pd.DataFrame(group_meta)
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]

    try:
        import plotly.express as px
        if color_by == "run_id":
            fig = px.scatter(
                plot_df, x="x", y="y", color="run_id",
                hover_data=["condition_label", "gene", "n_cells_used"],
            )
        else:
            fig = px.scatter(
                plot_df, x="x", y="y", color="gene",
                hover_data=["condition_label", "run_id", "n_cells_used"],
            )

        if highlight_genes:
            hl_set = {g.lower() for g in highlight_genes}
            fig.update_traces(marker=dict(size=8, opacity=0.3))
            hl = plot_df[plot_df["gene"].str.lower().isin(hl_set)]
            if not hl.empty:
                fig.add_scatter(
                    x=hl["x"], y=hl["y"], mode="markers",
                    marker=dict(size=12, color="red", opacity=1.0),
                    text=hl["condition_label"], hoverinfo="text",
                    name="Highlighted",
                )

        harmony_status = ""
        if has_multi_runs:
            harmony_status = " · Harmony" if batch_correct else " · raw"
        fig.update_layout(
            title=(
                f"Feature Profiles — Optimal Transport "
                f"(Sinkhorn ε = {sinkhorn_reg * 100:.1f}% of typical cost)"
                f"{harmony_status} · {len(point_clouds)} groups × "
                f"{n_cells_per_condition} cells · {len(morph_cols)} features"
            ),
            xaxis_title="UMAP 1 (from OT distance)",
            yaxis_title="UMAP 2 (from OT distance)",
            template="plotly_white",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        fig.write_html(
            str(out_path), include_plotlyjs=True, default_height="100vh",
        )
        _make_plot_html_fullheight(out_path)
        progress_cb(1.0)
        return out_path
    except ImportError:
        return None
