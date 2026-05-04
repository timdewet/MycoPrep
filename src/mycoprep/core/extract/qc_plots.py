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
):
    """Compute per-condition S-score profiles (mean + CV, Z-scored vs controls).

    Controls are identified explicitly via *control_labels* (e.g.
    ``["NT1", "NT2", "WT"]``). If none of the conditions match any
    control label, the function falls back to ``StandardScaler`` against
    the full dataset — no longer a true S-score, but the resulting
    geometry is still informative.
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
    # n_jobs=1 explicit so seeded reproducibility doesn't trigger a warning
    # about parallelism being disabled.
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, effective_neighbors),
        min_dist=min_dist,
        random_state=random_state,
        n_jobs=1,
    )
    embedding = reducer.fit_transform(X_pca)

    effective_min_cluster = min(min_cluster_size, max(2, n_samples // 10))
    # copy=False explicit to silence the sklearn 1.10 default-change warning;
    # we do not mutate ``embedding`` after fitting.
    clusterer = HDBSCAN(min_cluster_size=effective_min_cluster, copy=False)
    labels = clusterer.fit_predict(embedding)

    return embedding, labels


def _embed_profiles(profiles):
    """2D embedding + cluster labels from a per-condition S-score table.

    Uses UMAP+HDBSCAN when there are at least 6 conditions (UMAP needs
    enough neighbours to behave); falls back to PCA + a single dummy
    cluster otherwise. Returns (embedding[N×2], cluster_labels[N]).
    """
    import numpy as np

    X = profiles.values
    n = len(profiles)
    if n >= 6:
        emb, lbl = _run_umap_hdbscan(
            X,
            n_neighbors=min(5, n - 1),
            min_cluster_size=max(2, n // 5),
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

    Returns a list of dicts in profile order with: condition, run_id,
    experiment_type, n_cells, is_current_run, is_control.
    """
    import pandas as pd

    rows = []
    for label in profile_labels:
        sub = combined_df[combined_df[label_col] == label]
        if sub.empty:
            rows.append({
                "label": label, "condition": label, "run_id": "",
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
        rows.append({
            "label": label, "condition": cond, "run_id": run_id,
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
    top_features_per_point: int = 5,
):
    """Build an interactive Plotly figure: single scatter, configurable colour.

    ``color_by``:
      - ``"cluster"`` (default): HDBSCAN cluster id; controls grey
      - ``"run_id"``: each registered run gets its own colour; controls grey
      - ``"feature"``: continuous viridis gradient over ``feature_col``
        (an S-score column from ``profiles``); controls excluded from the
        gradient and overlaid as grey circles for reference

    Returns a ``plotly.graph_objects.Figure`` ready to ``write_html``.
    """
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()
    hover_text = _build_hover_text(profiles, meta_rows, top_features_per_point)

    n = len(meta_rows)
    sizes = np.array([22 if r["is_current_run"] else 12 for r in meta_rows])
    line_widths = np.array([2 if r["is_current_run"] else 0 for r in meta_rows])
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
                        line=dict(width=line_widths[sel].tolist(), color="black"),
                    ),
                    hovertext=[hover_text[i] for i in sel],
                    hoverinfo="text",
                ))

    elif color_by == "run_id":
        run_ids = [meta_rows[i]["run_id"] or "(unknown)" for i in non_ctrl_idxs]
        unique_runs = sorted(set(run_ids))
        for k, rid in enumerate(unique_runs):
            grp = [non_ctrl_idxs[j] for j, r in enumerate(run_ids) if r == rid]
            color = _CATEGORICAL_PALETTE[k % len(_CATEGORICAL_PALETTE)]
            for exp_type in ("knockdown", "drug"):
                sel = [i for i in grp if exp_types[i] == exp_type]
                if not sel:
                    continue
                symbol = "circle" if exp_type == "knockdown" else "triangle-up"
                fig.add_trace(go.Scatter(
                    x=embedding[sel, 0], y=embedding[sel, 1],
                    mode="markers",
                    name=f"{rid} ({exp_type})" if exp_type == "drug" else rid,
                    legendgroup=f"run_{rid}",
                    showlegend=(exp_type == "knockdown"),
                    marker=dict(
                        size=sizes[sel].tolist(),
                        color=color,
                        symbol=symbol,
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

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        height=560,
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(font=dict(size=10)),
        plot_bgcolor="white",
        xaxis=dict(title="Component 1", showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="Component 2", showgrid=True, gridcolor="#eee"),
    )
    return fig


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


def render_library_html(
    out_path: "Path",
    *,
    library_dir: "Path | None" = None,
    species: str = "",
    color_by: str = "cluster",
    feature_col: str | None = None,
) -> "Path | None":
    """Render an interactive Plotly HTML for the library on its own.

    Useful for the Analysis page's default view: shows each
    (run, condition) S-score profile as a point, with hover info and
    a configurable colouring (``cluster`` / ``run_id`` / ``feature``).
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
    )
    if len(profiles) < 2:
        return None

    meta = _condition_meta_table(
        df_lib, profiles.index, "_combined_label",
        control_labels=controls,
    )
    embedding, lbl_cluster = _embed_profiles(profiles)

    n_runs = len(lib_index)
    title = (
        f"Feature library — {n_runs} run(s), species: {species or 'all'}"
    )
    fig = _plot_condition_plotly(
        profiles, embedding, lbl_cluster, meta, title,
        color_by=color_by, feature_col=feature_col,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
    return out_path


def _render_clustering_figure(
    profiles, meta_rows, *, title, out_dir, base_name, progress_cb=_noop,
):
    """Render a condition-level clustering figure as both PNG and HTML."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(profiles) < 2:
        progress_cb(0.9, f"qc_plots: {base_name}: <2 conditions, skipping")
        return

    embedding, lbl_cluster = _embed_profiles(profiles)

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
        plotly_fig.write_html(
            out_dir / f"{base_name}.html",
            include_plotlyjs=True,
            full_html=True,
        )
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
    )
    progress_cb(0.95, "qc_plots: library clustering done")
