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


def _compute_condition_sscores(df, morph_cols, condition_col="condition"):
    """Compute per-condition S-score profiles (mean + CV, Z-scored vs controls)."""
    import numpy as np
    import pandas as pd

    def _cv(s):
        m = s.mean()
        return s.std() / m if m != 0 else 0.0

    means = df.groupby(condition_col)[morph_cols].mean()
    cvs = df.groupby(condition_col)[morph_cols].agg(_cv)
    cvs.columns = [c + "_CV" for c in cvs.columns]

    profiles = pd.concat([means, cvs], axis=1)

    # Z-score vs controls (conditions containing "WT" or "DMSO" or ATc-)
    control_mask = profiles.index.str.contains(
        r"WT|DMSO|wildtype|control", case=False, regex=True
    )
    if control_mask.any():
        ctrl = profiles.loc[control_mask]
        mu = ctrl.mean()
        sigma = ctrl.std().replace(0, 1)
        profiles = (profiles - mu) / sigma
    else:
        from sklearn.preprocessing import StandardScaler
        vals = StandardScaler().fit_transform(profiles.values)
        profiles = pd.DataFrame(vals, index=profiles.index, columns=profiles.columns)

    return profiles


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


def _run_umap_hdbscan(X_scaled, n_neighbors=15, min_dist=0.1,
                       min_cluster_size=15, random_state=42):
    """PCA → UMAP → HDBSCAN.  Returns (embedding_2d, cluster_labels)."""
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.cluster import HDBSCAN

    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required for clustering plots")

    n_samples, n_features = X_scaled.shape
    n_components = min(n_features, n_samples, 50)
    if n_components >= 2:
        pca = PCA(n_components=min(n_components, 0.95), random_state=random_state)
        X_pca = pca.fit_transform(X_scaled)
    else:
        X_pca = X_scaled

    effective_neighbors = min(n_neighbors, n_samples - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, effective_neighbors),
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X_pca)

    effective_min_cluster = min(min_cluster_size, max(2, n_samples // 10))
    clusterer = HDBSCAN(min_cluster_size=effective_min_cluster)
    labels = clusterer.fit_predict(embedding)

    return embedding, labels


def _plot_umap_panels(
    embedding, labels_condition, labels_cluster, palette_condition,
    title_prefix, ax_left, ax_right, highlight_mask=None,
    experiment_types=None,
):
    """Draw a pair of UMAP panels: left=condition-colored, right=cluster-colored."""
    import numpy as np

    # --- Left: condition ---
    conditions = sorted(set(labels_condition))
    for cond in conditions:
        mask = np.array(labels_condition) == cond
        color = palette_condition.get(cond, "#888888")
        s = 8 if highlight_mask is None else np.where(
            highlight_mask & mask, 12, 3
        )[mask]
        alpha = 0.7 if highlight_mask is None else np.where(
            highlight_mask & mask, 0.8, 0.2
        )[mask]
        ax_left.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=s, alpha=alpha, color=color, label=cond, edgecolors="none",
        )
    ax_left.set_title(f"{title_prefix} — by condition", fontsize=10)
    ax_left.set_xlabel("UMAP 1"); ax_left.set_ylabel("UMAP 2")
    ax_left.legend(fontsize=7, frameon=False, markerscale=2, loc="best")
    for sp in ("top", "right"):
        ax_left.spines[sp].set_visible(False)

    # --- Right: cluster ---
    unique_clusters = sorted(set(labels_cluster))
    cmap = __import__("matplotlib").colormaps.get_cmap("tab10")
    for cl in unique_clusters:
        mask = np.array(labels_cluster) == cl
        color = "#cccccc" if cl == -1 else cmap(cl % 10)
        label = "noise" if cl == -1 else f"cluster {cl}"
        s = 8 if highlight_mask is None else np.where(
            highlight_mask & mask, 12, 3
        )[mask]
        alpha = 0.7 if highlight_mask is None else np.where(
            highlight_mask & mask, 0.8, 0.2
        )[mask]
        ax_right.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=s, alpha=alpha, color=color, label=label, edgecolors="none",
        )
    ax_right.set_title(f"{title_prefix} — by cluster", fontsize=10)
    ax_right.set_xlabel("UMAP 1"); ax_right.set_ylabel("UMAP 2")
    ax_right.legend(fontsize=7, frameon=False, markerscale=2, loc="best")
    for sp in ("top", "right"):
        ax_right.spines[sp].set_visible(False)


def _plot_condition_panels(
    profiles, embedding, labels_cluster,
    title_prefix, ax_left, ax_right,
    highlight_mask=None, experiment_types=None,
):
    """Draw condition-level panels with experiment-type markers."""
    import numpy as np

    conds = profiles.index.tolist()
    cmap = __import__("matplotlib").colormaps.get_cmap("tab20")

    markers = {"knockdown": "o", "drug": "^"}
    default_marker = "o"

    # --- Left: colored by condition, shaped by experiment type ---
    for i, cond in enumerate(conds):
        exp_type = experiment_types[i] if experiment_types else "knockdown"
        marker = markers.get(exp_type, default_marker)
        s = 80 if highlight_mask is None or highlight_mask[i] else 30
        alpha = 0.9 if highlight_mask is None or highlight_mask[i] else 0.35
        edgecolor = "black" if highlight_mask is None or highlight_mask[i] else "none"
        ax_left.scatter(
            embedding[i, 0], embedding[i, 1],
            s=s, alpha=alpha, color=cmap(i % 20),
            marker=marker, edgecolors=edgecolor, linewidths=0.8,
        )
        if highlight_mask is None or highlight_mask[i]:
            ax_left.annotate(
                cond, (embedding[i, 0], embedding[i, 1]),
                fontsize=6, alpha=0.8,
                xytext=(4, 4), textcoords="offset points",
            )
    ax_left.set_title(f"{title_prefix} — S-scores by condition", fontsize=10)
    ax_left.set_xlabel("Component 1"); ax_left.set_ylabel("Component 2")
    for sp in ("top", "right"):
        ax_left.spines[sp].set_visible(False)

    # --- Right: colored by cluster ---
    unique_clusters = sorted(set(labels_cluster))
    cmap_cl = __import__("matplotlib").colormaps.get_cmap("tab10")
    for cl in unique_clusters:
        mask = np.array(labels_cluster) == cl
        color = "#cccccc" if cl == -1 else cmap_cl(cl % 10)
        label = "noise" if cl == -1 else f"cluster {cl}"
        idxs = np.where(mask)[0]
        for i in idxs:
            exp_type = experiment_types[i] if experiment_types else "knockdown"
            marker = markers.get(exp_type, default_marker)
            s = 80 if highlight_mask is None or highlight_mask[i] else 30
            ax_right.scatter(
                embedding[i, 0], embedding[i, 1],
                s=s, alpha=0.8, color=color, marker=marker,
                edgecolors="black" if highlight_mask is None or highlight_mask[i] else "none",
                linewidths=0.6,
            )
        ax_right.scatter([], [], color=color, label=label, marker="o")
    ax_right.set_title(f"{title_prefix} — by cluster", fontsize=10)
    ax_right.set_xlabel("Component 1"); ax_right.set_ylabel("Component 2")
    ax_right.legend(fontsize=7, frameon=False, loc="best")
    for sp in ("top", "right"):
        ax_right.spines[sp].set_visible(False)


def _morphology_cluster_plots(
    df_current,
    *,
    out_dir,
    palette,
    atc_states,
    library_dir=None,
    species="",
    current_run_id="",
    progress_cb=_noop,
):
    """Generate solo and library clustering plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

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

    # --- Solo run plot ---
    rng = np.random.RandomState(42)
    df_solo = df_current.dropna(subset=morph_cols).copy()
    if len(df_solo) < 30:
        progress_cb(0.9, "qc_plots: too few cells for clustering")
        return

    df_solo = _subsample_stratified(df_solo, 5000, "condition", rng)
    X_solo = StandardScaler().fit_transform(df_solo[morph_cols].values)
    emb_solo, labels_solo = _run_umap_hdbscan(X_solo)

    conditions_solo = df_solo["condition"].tolist()
    cond_palette = dict(palette)
    for c in set(conditions_solo):
        if c not in cond_palette:
            atc_match = [a for a in atc_states if a in c]
            cond_palette[c] = palette.get(atc_match[0], "#888888") if atc_match else "#888888"

    # Cell-level solo
    fig_solo, axes_solo = plt.subplots(2, 2, figsize=(14, 12))
    _plot_umap_panels(
        emb_solo, conditions_solo, labels_solo, cond_palette,
        "Cell-level", axes_solo[0, 0], axes_solo[0, 1],
    )

    # Condition-level solo
    profiles_solo = _compute_condition_sscores(df_solo, morph_cols, "condition")
    if len(profiles_solo) >= 2:
        X_prof = profiles_solo.values
        n_cond = len(profiles_solo)
        if n_cond >= 6:
            emb_cond, lbl_cond = _run_umap_hdbscan(
                X_prof, n_neighbors=min(5, n_cond - 1),
                min_cluster_size=max(2, n_cond // 5),
            )
        else:
            from sklearn.decomposition import PCA as _PCA
            n_comp = min(2, X_prof.shape[1], n_cond)
            emb_cond = _PCA(n_components=n_comp).fit_transform(X_prof)
            if emb_cond.shape[1] == 1:
                emb_cond = np.column_stack([emb_cond, np.zeros(n_cond)])
            lbl_cond = np.full(n_cond, 0)

        _plot_condition_panels(
            profiles_solo, emb_cond, lbl_cond,
            "Condition-level", axes_solo[1, 0], axes_solo[1, 1],
        )
    else:
        axes_solo[1, 0].text(0.5, 0.5, "Not enough conditions", transform=axes_solo[1, 0].transAxes, ha="center")
        axes_solo[1, 0].axis("off")
        axes_solo[1, 1].axis("off")

    fig_solo.suptitle("Morphological clustering — current run", fontsize=13, y=1.0)
    fig_solo.tight_layout()
    fig_solo.savefig(out_dir / "morphology_clustering_run.png", dpi=160, bbox_inches="tight")
    plt.close(fig_solo)
    progress_cb(0.9, "qc_plots: solo clustering done")

    # --- Library comparison plot ---
    if library_dir is None:
        return

    try:
        from .feature_library import FeatureLibrary
        lib = FeatureLibrary(library_dir)
        df_lib = lib.load_species(species)
    except Exception:
        return

    if df_lib.empty:
        return

    progress_cb(0.92, "qc_plots: building library comparison figure")

    lib_morph_cols = [c for c in morph_cols if c in df_lib.columns]
    if len(lib_morph_cols) < 3:
        return

    # Add condition column to library data
    if "condition" not in df_lib.columns:
        parts = df_lib["well"].astype(str).str.split("__", expand=True) if "well" in df_lib.columns else None
        if parts is not None:
            atc_lib = parts[0].str.replace("_focused", "", regex=False)
            mutant_lib = parts[2] if 2 in parts.columns else df_lib["well"].astype(str)
            df_lib["condition"] = (mutant_lib.astype(str) + " " + atc_lib.astype(str)).str.strip()
        else:
            df_lib["condition"] = "library"

    # Tag source
    df_current_tagged = df_current.copy()
    df_current_tagged["_is_current_run"] = True
    df_lib["_is_current_run"] = False

    combined = pd.concat(
        [df_current_tagged[lib_morph_cols + ["condition", "_is_current_run"]],
         df_lib[lib_morph_cols + ["condition", "_is_current_run"]]],
        ignore_index=True,
    ).dropna(subset=lib_morph_cols)

    if len(combined) < 30:
        return

    # Subsample library, keep all current run
    current_mask = combined["_is_current_run"].values.astype(bool)
    n_current = current_mask.sum()
    n_lib_budget = max(1000, 5000 - n_current)
    lib_part = combined[~current_mask]
    if len(lib_part) > n_lib_budget:
        lib_part = lib_part.sample(n=n_lib_budget, random_state=rng)
    combined = pd.concat([combined[current_mask], lib_part], ignore_index=True)
    current_mask = combined["_is_current_run"].values.astype(bool)

    X_combined = StandardScaler().fit_transform(combined[lib_morph_cols].values)
    emb_lib, labels_lib = _run_umap_hdbscan(X_combined)

    conditions_lib = combined["condition"].tolist()
    for c in set(conditions_lib):
        if c not in cond_palette:
            cond_palette[c] = "#888888"

    fig_lib, axes_lib = plt.subplots(2, 2, figsize=(14, 12))

    # Cell-level with library
    _plot_umap_panels(
        emb_lib, conditions_lib, labels_lib, cond_palette,
        "Cell-level (library)", axes_lib[0, 0], axes_lib[0, 1],
        highlight_mask=current_mask,
    )

    # Condition-level with library
    combined["_exp_type"] = "knockdown"
    if "_library_experiment_type" in df_lib.columns:
        lib_types = df_lib["_library_experiment_type"].values
    else:
        lib_types = None

    profiles_lib = _compute_condition_sscores(combined, lib_morph_cols, "condition")
    if len(profiles_lib) >= 2:
        X_prof_lib = profiles_lib.values
        n_cond_lib = len(profiles_lib)

        # Determine experiment types per condition
        exp_types_per_cond = []
        for cond in profiles_lib.index:
            sub = combined[combined["condition"] == cond]
            if "_library_experiment_type" in sub.columns:
                types = sub["_library_experiment_type"].dropna().unique()
                exp_types_per_cond.append(types[0] if len(types) > 0 else "knockdown")
            else:
                exp_types_per_cond.append("knockdown")

        # Highlight current run conditions
        cond_highlight = []
        for cond in profiles_lib.index:
            sub = combined[combined["condition"] == cond]
            cond_highlight.append(sub["_is_current_run"].any())

        if n_cond_lib >= 6:
            emb_cond_lib, lbl_cond_lib = _run_umap_hdbscan(
                X_prof_lib,
                n_neighbors=min(5, n_cond_lib - 1),
                min_cluster_size=max(2, n_cond_lib // 5),
            )
        else:
            from sklearn.decomposition import PCA as _PCA
            n_comp = min(2, X_prof_lib.shape[1], n_cond_lib)
            emb_cond_lib = _PCA(n_components=n_comp).fit_transform(X_prof_lib)
            if emb_cond_lib.shape[1] == 1:
                emb_cond_lib = np.column_stack([emb_cond_lib, np.zeros(n_cond_lib)])
            lbl_cond_lib = np.full(n_cond_lib, 0)

        _plot_condition_panels(
            profiles_lib, emb_cond_lib, lbl_cond_lib,
            "Condition-level (library)", axes_lib[1, 0], axes_lib[1, 1],
            highlight_mask=cond_highlight,
            experiment_types=exp_types_per_cond,
        )
    else:
        axes_lib[1, 0].text(0.5, 0.5, "Not enough conditions", transform=axes_lib[1, 0].transAxes, ha="center")
        axes_lib[1, 0].axis("off")
        axes_lib[1, 1].axis("off")

    run_label = current_run_id or "current"
    n_lib_runs = len(lib.list_runs(species=species))
    fig_lib.suptitle(
        f"Morphological clustering — '{run_label}' vs library ({n_lib_runs} runs, species: {species or 'all'})",
        fontsize=12, y=1.0,
    )
    fig_lib.tight_layout()
    fig_lib.savefig(out_dir / "morphology_clustering_library.png", dpi=160, bbox_inches="tight")
    plt.close(fig_lib)
    progress_cb(0.95, "qc_plots: library clustering done")
