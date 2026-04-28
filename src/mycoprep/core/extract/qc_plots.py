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

    progress_cb(1.0, f"qc_plots: wrote → {out_dir.name}/")
    return out_dir
