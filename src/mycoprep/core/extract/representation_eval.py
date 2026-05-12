"""Score morphology representations against ground-truth gene groupings.

Given a per-condition representation (either a feature matrix or a
distance matrix) and a mapping from gene → group (operon, family,
functional class, …), measures how well biologically related genes
cluster together using two standard representation-learning metrics:

- **kNN same-group accuracy** at k ∈ {1, 3, 5}: for each query, the
  fraction of its k nearest other conditions that share its group
  (self excluded).
- **mean Average Precision (mAP)**: for each query, AP over the full
  ranked list with same-group entries treated as positives; reported
  as the mean across queries.

The module is pure data — no Qt, no plotting, no training. Pairs with
:mod:`representation_generate` (which ensures the source artefacts
exist) and the ``library compare-representations`` CLI command.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────
# Group table loading
# ─────────────────────────────────────────────────────────────────────────

def load_group_table(
    path: Path,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Parse a CSV/TSV grouping table.

    The file must have a ``gene`` column (case-insensitive) plus one or
    more group columns (e.g. ``operon, family, functional_class``).
    Empty cells mean the gene is unmapped under that scheme. Gene names
    are lowercased and whitespace-stripped for matching.

    The legacy single-scheme pathway CSV (``gene,pathway``) is accepted
    unchanged — ``pathway`` is treated as a normal group column.

    Returns:
        ``(grouping_names, mappings)`` where
        ``mappings[scheme][gene_lower] = group_label``. Empty schemes
        (no mapped genes) are dropped.
    """
    sep = "\t" if str(path).lower().endswith((".tsv", ".tab")) else ","
    df = pd.read_csv(path, sep=sep)
    if df.empty:
        raise ValueError(f"Group table {path} is empty")

    # Case-insensitive 'gene' header.
    lc = {c.lower(): c for c in df.columns}
    if "gene" not in lc:
        raise ValueError(
            f"Group table {path} must have a 'gene' column; got {list(df.columns)}"
        )
    df = df.rename(columns={lc["gene"]: "gene"})

    # Columns starting with "_" are treated as metadata (e.g. _n_conditions
    # emitted by `library gene-template`) and never become grouping schemes.
    grouping_cols = [c for c in df.columns if c != "gene" and not c.startswith("_")]
    if not grouping_cols:
        raise ValueError(
            f"Group table {path} must have at least one non-'gene' column"
        )

    df["gene"] = df["gene"].astype(str).str.strip().str.lower()

    mappings: dict[str, dict[str, str]] = {}
    for col in grouping_cols:
        m: dict[str, str] = {}
        for gene, val in zip(df["gene"], df[col]):
            if not gene or gene == "nan":
                continue
            if pd.isna(val):
                continue
            sval = str(val).strip()
            if not sval or sval.lower() == "nan":
                continue
            m[gene] = sval
        if m:
            mappings[col] = m
    return list(mappings.keys()), mappings


# ─────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────

SCOPE_CROSS_GENE = "cross_gene"
SCOPE_REPLICATE = "replicate"
SCOPE_ALL = "all"
_VALID_SCOPES = (SCOPE_CROSS_GENE, SCOPE_REPLICATE, SCOPE_ALL)


@dataclass
class RepresentationMetrics:
    """Result of evaluating one (representation, batch_correct, grouping, species, scope)."""

    representation: str
    batch_correct: bool
    grouping: str
    species: str
    scope: str
    knn_accuracy: dict[int, float]
    map_overall: float
    per_group: pd.DataFrame
    per_query: pd.DataFrame
    n_conditions_evaluated: int
    n_dropped_unmapped: int
    n_dropped_singleton_group: int
    note: str = ""

    def summary_row(self) -> dict:
        row = {
            "species": self.species,
            "representation": self.representation,
            "batch_correct": self.batch_correct,
            "grouping": self.grouping,
            "scope": self.scope,
            "map": self.map_overall,
            "n_conditions": self.n_conditions_evaluated,
            "n_dropped_unmapped": self.n_dropped_unmapped,
            "n_dropped_singleton_group": self.n_dropped_singleton_group,
            "note": self.note,
        }
        for k, v in self.knn_accuracy.items():
            row[f"knn@{k}"] = v
        return row


def _per_query_metrics(
    D: np.ndarray,
    group_labels: Sequence[str],
    k_list: tuple[int, ...],
    *,
    replicate_keys: Optional[Sequence[str]] = None,
    scope: str = SCOPE_CROSS_GENE,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Compute per-row kNN accuracy and Average Precision.

    ``scope`` controls how same-replicate-key rows (same gene by default)
    are treated:

    - ``"cross_gene"`` (default): same-key rows are excluded from the
      ranking entirely. Positives are rows with the same group but a
      different replicate key. Avoids the "trivial replicate matching"
      pitfall where a gene with many replicates inflates kNN/mAP.
    - ``"replicate"``: positives are same-key rows; the supplied
      ``group_labels`` are ignored (caller should pass replicate keys as
      group labels for this case). Measures replicate consistency.
    - ``"all"``: legacy / sanity-check mode. Positives are same-group;
      no key-based masking. Conflates replicate matching with group
      coherence — use only for comparison.

    Returns ``(knn_arrays, ap_array, valid_mask)``. A query is invalid
    if it has zero positives among its candidates — its row is excluded
    from the summary metrics.
    """
    if scope not in _VALID_SCOPES:
        raise ValueError(f"scope must be one of {_VALID_SCOPES}; got {scope!r}")

    n = D.shape[0]
    g = np.asarray(group_labels)
    if replicate_keys is None:
        # No masking: every row has a unique key.
        rk = np.array([f"__row_{i}__" for i in range(n)])
    else:
        rk = np.asarray([str(k) for k in replicate_keys])

    knn_arrays = {k: np.zeros(n, dtype=np.float64) for k in k_list}
    ap_array = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    if n < 2:
        return knn_arrays, ap_array, valid

    for i in range(n):
        d_i = D[i].astype(np.float64, copy=True)
        d_i[i] = np.inf  # self always excluded

        if scope == SCOPE_CROSS_GENE:
            same_key = (rk == rk[i])
            # Drop every same-key row (including self) from the ranking.
            d_i[same_key] = np.inf
            positives_mask = (g == g[i]) & (~same_key)
        elif scope == SCOPE_REPLICATE:
            positives_mask = (rk == rk[i])
            positives_mask[i] = False
        else:  # SCOPE_ALL
            positives_mask = (g == g[i])
            positives_mask[i] = False

        candidate_mask = ~np.isinf(d_i)
        n_candidates = int(candidate_mask.sum())
        if n_candidates < 1:
            continue
        n_positives = int((positives_mask & candidate_mask).sum())
        if n_positives == 0:
            continue
        valid[i] = True

        order = np.argsort(d_i, kind="stable")
        cand_order = order[:n_candidates]
        ranked_pos = positives_mask[cand_order]

        for k in k_list:
            kk = min(k, n_candidates)
            knn_arrays[k][i] = float(ranked_pos[:kk].mean())

        ranks = np.arange(1, n_candidates + 1)
        cum_hits = np.cumsum(ranked_pos)
        precision_at = cum_hits / ranks
        ap_array[i] = float(precision_at[ranked_pos].mean())

    return knn_arrays, ap_array, valid


def evaluate_distance_matrix(
    D: np.ndarray,
    condition_labels: Sequence[str],
    gene_labels: Sequence[str],
    group_map: dict[str, str],
    *,
    representation: str,
    batch_correct: bool,
    grouping: str,
    species: str,
    scope: str = SCOPE_CROSS_GENE,
    replicate_keys: Optional[Sequence[str]] = None,
    k_list: tuple[int, ...] = (1, 3, 5),
) -> RepresentationMetrics:
    """Score a precomputed distance matrix against one grouping.

    ``scope`` selects how same-replicate-key (default: same-gene) rows
    are handled. See :func:`_per_query_metrics` for the semantics.

    When ``scope == "replicate"``, ``group_map`` is ignored and each
    replicate key becomes its own group; rows whose key occurs only once
    are dropped as singletons.
    """
    if scope not in _VALID_SCOPES:
        raise ValueError(f"scope must be one of {_VALID_SCOPES}; got {scope!r}")

    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square; got shape {D.shape}")
    if len(condition_labels) != D.shape[0] or len(gene_labels) != D.shape[0]:
        raise ValueError(
            f"Label length mismatch: D={D.shape[0]}, "
            f"conds={len(condition_labels)}, genes={len(gene_labels)}"
        )

    cond_arr = [str(c) for c in condition_labels]
    gene_arr = [str(g) for g in gene_labels]
    if replicate_keys is None:
        rk_arr_full = [g.strip().lower() for g in gene_arr]
    else:
        if len(replicate_keys) != D.shape[0]:
            raise ValueError("replicate_keys length must match D")
        rk_arr_full = [str(k).strip().lower() for k in replicate_keys]

    # ── Row admission depends on scope ────────────────────────────────
    kept_idx: list[int] = []
    kept_groups: list[str] = []
    n_unmapped = 0

    if scope == SCOPE_REPLICATE:
        # Replicate scope ignores the user-supplied group_map — each
        # replicate key becomes its own "group" and we keep rows whose
        # key occurs at least twice.
        for i, rk in enumerate(rk_arr_full):
            if rk and rk != "nan":
                kept_idx.append(i)
                kept_groups.append(rk)
            else:
                n_unmapped += 1
    else:
        for i, gn in enumerate(gene_arr):
            g_key = gn.strip().lower()
            if g_key in group_map:
                kept_idx.append(i)
                kept_groups.append(group_map[g_key])
            else:
                n_unmapped += 1

    # ── Singleton handling (scope-aware) ──────────────────────────────
    n_singleton = 0
    if scope == SCOPE_CROSS_GENE:
        # A group is non-trivial when it contains ≥2 *distinct* replicate
        # keys — otherwise the only same-group row is a same-key replicate,
        # which we'd mask out, leaving zero positives.
        keys_by_group: dict[str, set[str]] = {}
        for i, gg in zip(kept_idx, kept_groups):
            keys_by_group.setdefault(gg, set()).add(rk_arr_full[i])
        valid_groups = {gg for gg, ks in keys_by_group.items() if len(ks) >= 2}
    else:
        counts = Counter(kept_groups)
        valid_groups = {gg for gg, c in counts.items() if c >= 2}

    keep2_idx: list[int] = []
    keep2_groups: list[str] = []
    for i, gg in zip(kept_idx, kept_groups):
        if gg in valid_groups:
            keep2_idx.append(i)
            keep2_groups.append(gg)
        else:
            n_singleton += 1

    if len(keep2_idx) < 2:
        return RepresentationMetrics(
            representation=representation,
            batch_correct=batch_correct,
            grouping=grouping,
            species=species,
            scope=scope,
            knn_accuracy={k: float("nan") for k in k_list},
            map_overall=float("nan"),
            per_group=pd.DataFrame(),
            per_query=pd.DataFrame(),
            n_conditions_evaluated=0,
            n_dropped_unmapped=n_unmapped,
            n_dropped_singleton_group=n_singleton,
            note="insufficient conditions after dropping unmapped/singletons",
        )

    Dk = D[np.ix_(keep2_idx, keep2_idx)]
    rk_sub = [rk_arr_full[i] for i in keep2_idx]
    knn_arr, ap_arr, valid = _per_query_metrics(
        Dk, keep2_groups, k_list,
        replicate_keys=rk_sub,
        scope=scope,
    )
    n_valid = int(valid.sum())
    if n_valid == 0:
        return RepresentationMetrics(
            representation=representation,
            batch_correct=batch_correct,
            grouping=grouping,
            species=species,
            scope=scope,
            knn_accuracy={k: float("nan") for k in k_list},
            map_overall=float("nan"),
            per_group=pd.DataFrame(),
            per_query=pd.DataFrame(),
            n_conditions_evaluated=0,
            n_dropped_unmapped=n_unmapped,
            n_dropped_singleton_group=n_singleton,
            note="no queries had positives within their candidate set",
        )

    knn_acc = {k: float(knn_arr[k][valid].mean()) for k in k_list}
    map_overall = float(ap_arr[valid].mean())

    per_query = pd.DataFrame(
        {
            "condition": [cond_arr[i] for i in keep2_idx],
            "gene": [gene_arr[i] for i in keep2_idx],
            "replicate_key": rk_sub,
            "group": keep2_groups,
            "ap": ap_arr,
            "valid": valid,
            **{f"knn@{k}": knn_arr[k] for k in k_list},
        }
    )

    valid_query = per_query[per_query["valid"]].copy()
    agg_dict = {"n_conditions": ("condition", "size"), "map": ("ap", "mean")}
    for k in k_list:
        agg_dict[f"knn@{k}"] = (f"knn@{k}", "mean")
    per_group = (
        valid_query.groupby("group").agg(**agg_dict).reset_index()
        if not valid_query.empty
        else pd.DataFrame()
    )

    return RepresentationMetrics(
        representation=representation,
        batch_correct=batch_correct,
        grouping=grouping,
        species=species,
        scope=scope,
        knn_accuracy=knn_acc,
        map_overall=map_overall,
        per_group=per_group,
        per_query=per_query,
        n_conditions_evaluated=n_valid,
        n_dropped_unmapped=n_unmapped,
        n_dropped_singleton_group=n_singleton,
    )


def evaluate_feature_matrix(
    X: np.ndarray,
    condition_labels: Sequence[str],
    gene_labels: Sequence[str],
    group_map: dict[str, str],
    *,
    representation: str,
    batch_correct: bool,
    grouping: str,
    species: str,
    scope: str = SCOPE_CROSS_GENE,
    replicate_keys: Optional[Sequence[str]] = None,
    metric: str = "cosine",
    k_list: tuple[int, ...] = (1, 3, 5),
) -> RepresentationMetrics:
    """Score a per-condition feature matrix by deriving a pairwise distance."""
    from sklearn.metrics import pairwise_distances

    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 2:
        return RepresentationMetrics(
            representation=representation,
            batch_correct=batch_correct,
            grouping=grouping,
            species=species,
            scope=scope,
            knn_accuracy={k: float("nan") for k in k_list},
            map_overall=float("nan"),
            per_group=pd.DataFrame(),
            per_query=pd.DataFrame(),
            n_conditions_evaluated=0,
            n_dropped_unmapped=0,
            n_dropped_singleton_group=0,
            note="fewer than 2 rows in feature matrix",
        )
    D = pairwise_distances(X, metric=metric)
    return evaluate_distance_matrix(
        D, condition_labels, gene_labels, group_map,
        representation=representation,
        batch_correct=batch_correct,
        grouping=grouping,
        species=species,
        scope=scope,
        replicate_keys=replicate_keys,
        k_list=k_list,
    )


# ─────────────────────────────────────────────────────────────────────────
# Source loaders
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class SourceData:
    """Per-condition data for one representation source.

    Either ``X`` (feature matrix) or ``D`` (distance matrix) is populated;
    callers branch on which is non-None.
    """

    name: str                       # canonical source name, e.g. "OT (supcon_resnet18)"
    species: str
    condition_labels: list[str]
    gene_labels: list[str]
    run_ids: list[str]              # parallel to condition_labels; "" if unknown
    X: Optional[np.ndarray] = None
    D: Optional[np.ndarray] = None
    metric: str = "cosine"          # only used when X is set
    condition_keys: list[str] = field(default_factory=list)  # condition_label w/o run_id
    note: str = ""

    def replicate_keys_for(self, mode: str) -> list[str]:
        """Return replicate keys per condition under the requested mode.

        ``mode == "gene"`` (default) returns the gene token; ``"condition"``
        returns the cleaned condition label (run_id stripped). The latter
        treats different reporter backgrounds / treatments of the same gene
        as distinct biological observations.
        """
        if mode == "gene":
            return list(self.gene_labels)
        if mode == "condition":
            if self.condition_keys:
                return list(self.condition_keys)
            # Fall back: strip "@ run_id" suffix from condition_labels.
            return [c.rsplit(" @ ", 1)[0] for c in self.condition_labels]
        raise ValueError(f"replicate-key mode must be 'gene' or 'condition'; got {mode!r}")


def _gene_from_condition_label(cond: str) -> str:
    """Best-effort gene extraction from a condition or well label.

    Handles both label conventions present in MycoPrep:
    - Parsed condition labels like "hadA ATc+" → first whitespace token.
    - Raw well stems "<atc>__<reporter>__<mutant>[__R<n>]" → ``mutant``
      (third underscore token), via ``crops.derive_condition_fields``.
    """
    cond = str(cond).strip()
    if not cond:
        return ""
    if "__" in cond:
        from .crops import derive_condition_fields
        gene = derive_condition_fields(cond).get("gene", "") or ""
        if gene:
            return gene
        # derive_condition_fields returns "" when the mutant token looks
        # like a control hint — fall back to that token so the row still
        # has a usable key for replicate matching.
        parts = cond.split("__")
        if len(parts) > 2:
            return parts[2].replace("_focused", "").replace("_filtered", "")
    return cond.split()[0]


def apply_harmony(
    X: np.ndarray,
    batch_labels: Sequence[str],
) -> tuple[np.ndarray, bool]:
    """Apply Harmony batch correction by run_id.

    Mirrors the configuration used inside ``qc_plots._run_umap_hdbscan``:
    pinned ``nclust = min(max(2, n_batches), 5)``, 20 outer iterations.

    Returns ``(X_corrected, applied)``. ``applied`` is False when
    Harmony was a no-op (single batch, missing dependency, or failure).
    """
    arr = np.asarray(batch_labels)
    unique = [b for b in np.unique(arr) if str(b)]
    if len(unique) < 2:
        return X, False
    try:
        import harmonypy
    except ImportError:
        return X, False
    try:
        ho = harmonypy.run_harmony(
            X,
            pd.DataFrame({"run_id": arr.astype(str)}),
            vars_use="run_id",
            max_iter_harmony=20,
            nclust=min(max(2, len(unique)), 5),
        )
        return np.asarray(ho.Z_corr), True
    except Exception:  # noqa: BLE001
        return X, False


def load_features_mean_source(
    library_dir: Optional[Path],
    species: str,
    *,
    batch_correct: bool,
    baseline_mode: str = "pooled",
) -> Optional[SourceData]:
    """Build the morphology-features mean (S-score) representation.

    Mirrors the path used by ``render_library_html``: per-(run, condition)
    S-score profile via ``_compute_condition_sscores``, then optionally
    Harmony-corrected at the profile level with run_id as the batch.
    """
    from .feature_library import FeatureLibrary
    from .qc_plots import (
        _compute_condition_sscores,
        _extract_run_ids,
        _select_morphology_cols,
    )

    lib = FeatureLibrary(library_dir)
    df_lib = lib.load_species(species)
    if df_lib.empty:
        return None
    morph_cols = _select_morphology_cols(df_lib)
    if len(morph_cols) < 3:
        return None
    df_lib = df_lib.dropna(subset=morph_cols).copy()
    if df_lib.empty:
        return None

    if "_library_run_id" in df_lib.columns:
        df_lib["_run_id"] = df_lib["_library_run_id"].astype(str)
    else:
        df_lib["_run_id"] = "library"

    # Resolve a usable per-condition label. Older library parquets may
    # carry only `well` (raw filename stem); derive condition_label from
    # the well via crops.derive_condition_fields when that's the case.
    if "condition" in df_lib.columns:
        df_lib["condition"] = df_lib["condition"].astype(str)
    elif "condition_label" in df_lib.columns:
        df_lib["condition"] = df_lib["condition_label"].astype(str)
    elif "well" in df_lib.columns:
        from .crops import derive_condition_fields
        df_lib["condition"] = (
            df_lib["well"].astype(str)
            .map(lambda w: derive_condition_fields(w).get("condition_label") or w)
        )
    else:
        return None
    df_lib["_combined_label"] = (
        df_lib["condition"] + " @ " + df_lib["_run_id"]
    )

    # Use the same control_labels pooling approach as render_library_html:
    # collapse control_labels across runs into a deduplicated list. The
    # index carries the controls as a comma-separated string per run.
    idx = lib.list_runs(species=species)
    control_labels_series = idx.get(
        "control_labels", pd.Series([""] * len(idx)),
    ).astype(str)
    pooled_controls = sorted({
        c.strip()
        for s in control_labels_series
        for c in s.split(",")
        if c.strip()
    })

    profiles = _compute_condition_sscores(
        df_lib, morph_cols, "_combined_label",
        control_labels=pooled_controls,
        baseline_mode=baseline_mode,
    )
    if profiles.empty:
        return None

    condition_labels = list(profiles.index.astype(str))
    run_ids = list(_extract_run_ids(profiles.index))
    # Strip the " @ run_id" suffix so gene-derivation works.
    plain_conds = [c.rsplit(" @ ", 1)[0] if " @ " in c else c for c in condition_labels]
    genes = [_gene_from_condition_label(c) for c in plain_conds]

    X = profiles.values.astype(np.float64)
    note = ""
    if batch_correct:
        X, applied = apply_harmony(X, run_ids)
        if not applied:
            note = "Harmony not applied (single batch or unavailable)"

    return SourceData(
        name="features",
        species=species,
        condition_labels=condition_labels,
        gene_labels=genes,
        run_ids=run_ids,
        condition_keys=plain_conds,
        X=X,
        metric="cosine",
        note=note,
    )


def load_cnn_mean_source(
    library_dir: Optional[Path],
    species: str,
    model_type: str,
    *,
    batch_correct: bool,
) -> Optional[SourceData]:
    """Build the per-condition mean CNN-embedding representation.

    Reads ``<library>/models/embeddings/<model_type>/cnn_embeddings.parquet``
    (the canonical path the Analysis panel reads from), aggregates per
    (run, condition), and optionally applies Harmony at the profile level.
    """
    from .feature_library import FeatureLibrary

    lib = FeatureLibrary(library_dir)
    emb_path = lib.models_dir / "embeddings" / model_type / "cnn_embeddings.parquet"
    if not emb_path.exists():
        return None
    df = pd.read_parquet(emb_path)
    if df.empty:
        return None

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if len(emb_cols) < 10:
        return None

    # Restrict to species if the per-cell rows carry that field; otherwise
    # accept all rows (embeddings parquet doesn't reliably carry species).
    # Filter via library index instead: get run_ids belonging to species.
    idx = lib.list_runs(species=species)
    species_runs = set(idx["run_id"].astype(str)) if not idx.empty else set()
    if species_runs and "run_id" in df.columns:
        df = df[df["run_id"].astype(str).isin(species_runs)].copy()
    if df.empty:
        return None

    df["condition_label"] = df["condition_label"].astype(str)
    df["run_id"] = df.get("run_id", "").astype(str)
    df["_combined_label"] = df["condition_label"] + " @ " + df["run_id"]

    profiles = df.groupby("_combined_label")[emb_cols].mean()
    # Take the first non-empty gene per group as that profile's gene.
    if "gene" in df.columns:
        gene_first = (
            df.assign(gene=df["gene"].astype(str))
            .groupby("_combined_label")["gene"]
            .agg(lambda s: next((x for x in s if x and x != "nan"), ""))
        )
        gene_arr = [gene_first.get(c, "") for c in profiles.index]
    else:
        gene_arr = [
            _gene_from_condition_label(c.rsplit(" @ ", 1)[0])
            for c in profiles.index
        ]

    run_ids = [c.rsplit(" @ ", 1)[1] if " @ " in c else "" for c in profiles.index]

    X = profiles.values.astype(np.float64)
    note = ""
    if batch_correct:
        X, applied = apply_harmony(X, run_ids)
        if not applied:
            note = "Harmony not applied (single batch or unavailable)"

    cond_labels = list(profiles.index.astype(str))
    cond_keys = [c.rsplit(" @ ", 1)[0] if " @ " in c else c for c in cond_labels]

    return SourceData(
        name=model_type,
        species=species,
        condition_labels=cond_labels,
        gene_labels=gene_arr,
        run_ids=run_ids,
        condition_keys=cond_keys,
        X=X,
        metric="cosine",
        note=note,
    )


def load_ot_source(
    sidecar_path: Path,
    *,
    name: str,
    species: str,
    batch_correct: bool,
) -> Optional[SourceData]:
    """Wrap a cached OT sidecar parquet as a :class:`SourceData`.

    Caller chooses the sidecar path (canonical for ``batch_correct=True``,
    diagnostic-side for ``batch_correct=False``). ``name`` should already
    describe both the source (model_type or 'features') and that this is
    the OT variant — e.g. ``"OT (supcon_resnet18)"``.
    """
    from .ot_analysis import load_ot_sidecar

    if not Path(sidecar_path).exists():
        return None
    try:
        D, meta = load_ot_sidecar(Path(sidecar_path))
    except Exception:  # noqa: BLE001
        return None
    if D.size == 0 or meta.empty:
        return None

    condition_labels = meta.get("condition_label", pd.Series([], dtype=str)).astype(str).tolist()
    run_ids = meta.get("run_id", pd.Series([""] * len(meta))).astype(str).tolist()
    if "gene" in meta.columns:
        gene_arr = meta["gene"].astype(str).tolist()
        # Fall back to deriving from condition_label when gene is empty/NaN.
        gene_arr = [
            g if g and g.lower() != "nan" else _gene_from_condition_label(c)
            for g, c in zip(gene_arr, condition_labels)
        ]
    else:
        gene_arr = [_gene_from_condition_label(c) for c in condition_labels]

    # OT sidecars store the cleaned condition_label (run_id is a separate
    # column), so the labels themselves are already valid condition keys.
    return SourceData(
        name=name,
        species=species,
        condition_labels=condition_labels,
        gene_labels=gene_arr,
        run_ids=run_ids,
        condition_keys=list(condition_labels),
        D=D,
    )


# ─────────────────────────────────────────────────────────────────────────
# UMAP-of-source helper
# ─────────────────────────────────────────────────────────────────────────

def umap_project(
    source: SourceData,
    *,
    random_state: int = 42,
) -> Optional[np.ndarray]:
    """Project a source to 2D using the same UMAP params qc_plots uses.

    n_neighbors=3, min_dist=0. For distance-matrix sources passes
    ``metric="precomputed"``; for feature-matrix sources passes the
    source's stored metric. Returns ``None`` if UMAP fails or the
    source has fewer than 3 conditions.
    """
    try:
        import umap
    except ImportError:
        return None

    if source.D is not None:
        n = source.D.shape[0]
        if n < 3:
            return None
        try:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=max(2, min(3, n - 1)),
                min_dist=0.0,
                metric="precomputed",
                random_state=random_state,
                n_jobs=1,
            )
            return reducer.fit_transform(source.D)
        except Exception:  # noqa: BLE001
            return None

    if source.X is not None:
        n = source.X.shape[0]
        if n < 3:
            return None
        try:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=max(2, min(3, n - 1)),
                min_dist=0.0,
                metric=source.metric or "cosine",
                random_state=random_state,
                n_jobs=1,
            )
            return reducer.fit_transform(source.X)
        except Exception:  # noqa: BLE001
            return None

    return None


# ─────────────────────────────────────────────────────────────────────────
# Top-level scoring
# ─────────────────────────────────────────────────────────────────────────

def score_source(
    source: SourceData,
    mappings: dict[str, dict[str, str]],
    *,
    batch_correct: bool,
    k_list: tuple[int, ...] = (1, 3, 5),
    include_umap: bool = True,
    replicate_key_mode: str = "gene",
    include_replicate_scope: bool = True,
) -> list[RepresentationMetrics]:
    """Evaluate one source against every grouping.

    Emits per (grouping × source):
    - one ``scope="cross_gene"`` row — positives are same-group rows
      with a *different* replicate key (avoids replicate inflation).
    - one ``scope="replicate"`` row (when ``include_replicate_scope``,
      computed once per source rather than per grouping) — positives are
      same replicate key. Measures within-replicate consistency.

    Adds UMAP-projected variants of every row when ``include_umap``.
    ``replicate_key_mode`` is passed to :meth:`SourceData.replicate_keys_for`.
    """
    rep_keys = source.replicate_keys_for(replicate_key_mode)
    results: list[RepresentationMetrics] = []

    def _score(*, representation: str, scope: str, grouping: str,
               group_map: dict[str, str], X_override: Optional[np.ndarray] = None,
               metric_override: Optional[str] = None):
        if X_override is not None:
            return evaluate_feature_matrix(
                X_override,
                source.condition_labels,
                source.gene_labels,
                group_map,
                representation=representation,
                batch_correct=batch_correct,
                grouping=grouping,
                species=source.species,
                scope=scope,
                replicate_keys=rep_keys,
                metric=metric_override or "euclidean",
                k_list=k_list,
            )
        if source.D is not None:
            return evaluate_distance_matrix(
                source.D,
                source.condition_labels,
                source.gene_labels,
                group_map,
                representation=representation,
                batch_correct=batch_correct,
                grouping=grouping,
                species=source.species,
                scope=scope,
                replicate_keys=rep_keys,
                k_list=k_list,
            )
        return evaluate_feature_matrix(
            source.X,
            source.condition_labels,
            source.gene_labels,
            group_map,
            representation=representation,
            batch_correct=batch_correct,
            grouping=grouping,
            species=source.species,
            scope=scope,
            replicate_keys=rep_keys,
            metric=source.metric or "cosine",
            k_list=k_list,
        )

    # Cross-gene scope per user grouping.
    for grouping, group_map in mappings.items():
        m = _score(
            representation=source.name,
            scope=SCOPE_CROSS_GENE,
            grouping=grouping,
            group_map=group_map,
        )
        if source.note and not m.note:
            m.note = source.note
        results.append(m)

    # Replicate scope — one row per source (the user-supplied grouping
    # is irrelevant here because replicate scope ignores group_map).
    if include_replicate_scope:
        rep_grouping = f"(replicate · {replicate_key_mode})"
        m = _score(
            representation=source.name,
            scope=SCOPE_REPLICATE,
            grouping=rep_grouping,
            group_map={},
        )
        if source.note and not m.note:
            m.note = source.note
        results.append(m)

    if include_umap:
        emb_2d = umap_project(source)
        if emb_2d is not None:
            umap_name = f"UMAP via {source.name}"
            for grouping, group_map in mappings.items():
                m = _score(
                    representation=umap_name,
                    scope=SCOPE_CROSS_GENE,
                    grouping=grouping,
                    group_map=group_map,
                    X_override=emb_2d,
                    metric_override="euclidean",
                )
                results.append(m)
            if include_replicate_scope:
                rep_grouping = f"(replicate · {replicate_key_mode})"
                m = _score(
                    representation=umap_name,
                    scope=SCOPE_REPLICATE,
                    grouping=rep_grouping,
                    group_map={},
                    X_override=emb_2d,
                    metric_override="euclidean",
                )
                results.append(m)
    return results


def metrics_to_summary_df(
    metrics: Sequence[RepresentationMetrics],
) -> pd.DataFrame:
    """Concatenate ``RepresentationMetrics.summary_row()`` into one frame."""
    rows = [m.summary_row() for m in metrics]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["species", "grouping", "map"], ascending=[True, True, False]
        ).reset_index(drop=True)
    return df


def metrics_to_per_query_df(
    metrics: Sequence[RepresentationMetrics],
) -> pd.DataFrame:
    """Long-format per-condition breakdown across all metrics."""
    frames = []
    for m in metrics:
        if m.per_query.empty:
            continue
        df = m.per_query.copy()
        df.insert(0, "species", m.species)
        df.insert(1, "representation", m.representation)
        df.insert(2, "batch_correct", m.batch_correct)
        df.insert(3, "grouping", m.grouping)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def metrics_to_per_group_df(
    metrics: Sequence[RepresentationMetrics],
) -> pd.DataFrame:
    """Long-format per-group breakdown across all metrics."""
    frames = []
    for m in metrics:
        if m.per_group.empty:
            continue
        df = m.per_group.copy()
        df.insert(0, "species", m.species)
        df.insert(1, "representation", m.representation)
        df.insert(2, "batch_correct", m.batch_correct)
        df.insert(3, "grouping", m.grouping)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────
# Top-level orchestration
# ─────────────────────────────────────────────────────────────────────────

def score_all_representations(
    library_dir: Optional[Path],
    species: str,
    group_csv: Path,
    *,
    model_types: Sequence[str] = (
        "resnet18", "lightweight", "supcon_resnet18", "supcon_lightweight",
    ),
    batch_correct_states: tuple[bool, ...] = (True, False),
    include_features: bool = True,
    include_features_ot: bool = True,
    include_embeddings_ot: bool = True,
    include_umap: bool = True,
    replicate_key_mode: str = "gene",
    include_replicate_scope: bool = True,
    epochs: int = 50,
    batch_size: int = 64,
    retrain: bool = False,
    k_list: tuple[int, ...] = (1, 3, 5),
    out_dir: Optional[Path] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> tuple[list[RepresentationMetrics], "GenerationReport"]:
    """Generate every required artefact and score it against every grouping.

    Returns ``(metrics, generation_report)``. Also writes
    ``summary.csv``, ``per_group.csv``, ``per_query.csv``,
    ``summary.png`` and ``manifest.json`` into ``out_dir``.
    """
    from datetime import datetime
    import json

    from .feature_library import FeatureLibrary
    from .representation_generate import (
        GenerationPlan, ensure_all_representations,
        canonical_emb_ot_sidecar, canonical_features_ot_sidecar,
        _bc_off_emb_ot_sidecar, _bc_off_features_ot_sidecar,
    )

    if out_dir is None:
        lib = FeatureLibrary(library_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = lib.library_dir / "representation_eval" / ts
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Generate artefacts ─────────────────────────────────────────
    plan = GenerationPlan(
        species=species,
        model_types=list(model_types),
        batch_correct_states=tuple(batch_correct_states),
        compute_features_ot=bool(include_features_ot),
        compute_embedding_ot=bool(include_embeddings_ot),
        epochs=int(epochs),
        batch_size=int(batch_size),
        retrain=bool(retrain),
    )

    def _gen_cb(f, msg):
        if progress_cb:
            progress_cb(0.0 + f * 0.85, msg)

    gen_report = ensure_all_representations(
        library_dir, plan, out_dir=out_dir, progress_cb=_gen_cb,
    )

    # ── 2. Load groupings ─────────────────────────────────────────────
    grouping_names, mappings = load_group_table(Path(group_csv))
    if not mappings:
        raise ValueError(f"No usable groupings in {group_csv}")

    # ── 3. Score each source × BC state ───────────────────────────────
    all_metrics: list[RepresentationMetrics] = []

    # Mean: morphology features
    if include_features:
        for bc in batch_correct_states:
            src = load_features_mean_source(
                library_dir, species, batch_correct=bc,
            )
            if src is not None:
                src.name = "features"
                all_metrics.extend(
                    score_source(
                        src, mappings,
                        batch_correct=bc,
                        k_list=k_list, include_umap=include_umap,
                        replicate_key_mode=replicate_key_mode,
                        include_replicate_scope=include_replicate_scope,
                    )
                )

    # OT: morphology features
    if include_features_ot:
        for bc in batch_correct_states:
            sidecar = (
                canonical_features_ot_sidecar(library_dir, species)
                if bc else _bc_off_features_ot_sidecar(out_dir, species)
            )
            src = load_ot_source(
                sidecar,
                name="OT (features)",
                species=species,
                batch_correct=bc,
            )
            if src is not None:
                all_metrics.extend(
                    score_source(
                        src, mappings,
                        batch_correct=bc,
                        k_list=k_list, include_umap=include_umap,
                        replicate_key_mode=replicate_key_mode,
                        include_replicate_scope=include_replicate_scope,
                    )
                )

    # CNN: mean + OT per model_type
    for mt in model_types:
        for bc in batch_correct_states:
            # Mean profiles
            src = load_cnn_mean_source(
                library_dir, species, mt, batch_correct=bc,
            )
            if src is not None:
                all_metrics.extend(
                    score_source(
                        src, mappings,
                        batch_correct=bc,
                        k_list=k_list, include_umap=include_umap,
                        replicate_key_mode=replicate_key_mode,
                        include_replicate_scope=include_replicate_scope,
                    )
                )
            # OT
            if include_embeddings_ot:
                from .feature_library import FeatureLibrary
                lib_local = FeatureLibrary(library_dir)
                emb_parquet = (
                    lib_local.models_dir / "embeddings" / mt / "cnn_embeddings.parquet"
                )
                sidecar = (
                    canonical_emb_ot_sidecar(emb_parquet)
                    if bc else _bc_off_emb_ot_sidecar(out_dir, mt)
                )
                ot_src = load_ot_source(
                    sidecar,
                    name=f"OT ({mt})",
                    species=species,
                    batch_correct=bc,
                )
                if ot_src is not None:
                    all_metrics.extend(
                        score_source(
                            ot_src, mappings,
                            batch_correct=bc,
                            k_list=k_list, include_umap=include_umap,
                        )
                    )

    # ── 4. Write outputs ──────────────────────────────────────────────
    summary_df = metrics_to_summary_df(all_metrics)
    per_group_df = metrics_to_per_group_df(all_metrics)
    per_query_df = metrics_to_per_query_df(all_metrics)

    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    if not per_group_df.empty:
        per_group_df.to_csv(out_dir / "per_group.csv", index=False)
    if not per_query_df.empty:
        per_query_df.to_csv(out_dir / "per_query.csv", index=False)

    plot_summary(summary_df, out_dir / "summary.png")

    manifest = {
        "species": species,
        "model_types": list(model_types),
        "batch_correct_states": list(batch_correct_states),
        "groupings": list(mappings.keys()),
        "k_list": list(k_list),
        "include_umap": bool(include_umap),
        "out_dir": str(out_dir),
        "generation": [
            {
                "kind": a.kind,
                "model_type": a.model_type,
                "batch_correct": a.batch_correct,
                "path": str(a.path) if a.path else None,
                "built": a.built,
                "cached": a.cached,
                "note": a.note,
            }
            for a in gen_report.artefacts
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
    )

    if progress_cb:
        progress_cb(1.0, f"Wrote summary to {summary_csv}")

    return all_metrics, gen_report


# ─────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────

def plot_summary(summary_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    """Render a faceted bar chart of mAP per representation × grouping.

    One subplot per grouping (one column per user-supplied grouping plus
    one column for the replicate-scope summary), representations on the
    x axis; bars coloured by ``batch_correct``. Returns the PNG path or
    None if matplotlib is unavailable / the summary is empty.
    """
    if summary_df.empty:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    groupings = list(summary_df["grouping"].unique())
    species_list = list(summary_df["species"].unique())
    n_groupings = len(groupings)
    n_species = len(species_list)
    if n_groupings == 0:
        return None

    fig_height = max(4.0, 3.5 * n_species)
    fig_width = max(7.0, 1.0 + 2.5 * n_groupings)
    fig, axes = plt.subplots(
        n_species, n_groupings,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharey=True,
    )

    for si, sp in enumerate(species_list):
        for gi, gp in enumerate(groupings):
            ax = axes[si][gi]
            sub = summary_df[
                (summary_df["species"] == sp) & (summary_df["grouping"] == gp)
            ]
            if sub.empty:
                ax.set_axis_off()
                continue
            reps = list(sub["representation"].unique())
            x = np.arange(len(reps))
            width = 0.4
            on = sub[sub["batch_correct"]]
            off = sub[~sub["batch_correct"]]
            on_map = on.set_index("representation")["map"].reindex(reps).values
            off_map = off.set_index("representation")["map"].reindex(reps).values

            if not np.isnan(on_map).all():
                ax.bar(x - width / 2, on_map, width, label="Harmony on")
            if not np.isnan(off_map).all():
                ax.bar(x + width / 2, off_map, width, label="Harmony off")
            ax.set_xticks(x)
            ax.set_xticklabels(reps, rotation=45, ha="right", fontsize=7)
            ax.set_ylim(0, 1.0)
            ax.set_title(f"{sp} · {gp}", fontsize=9)
            if gi == 0:
                ax.set_ylabel("mAP")
            if si == 0 and gi == n_groupings - 1:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
