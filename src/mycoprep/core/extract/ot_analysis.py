"""Downstream analysis of cached OT distance matrices.

Operates on the sidecar parquet written by ``qc_plots._save_ot_sidecar``
next to each rendered ``*_ot.html`` view. The sidecar holds a
``(n_groups × n_groups)`` distance matrix plus the parallel group
metadata (run_id, condition_label, gene, n_cells_used, …).

Provides two analyses that the Mtb reference SupCon-OT pipeline runs
after the distance matrix is computed but that aren't in MycoPrep
proper:

- :func:`rank_condition_matches` — for each query condition, return the
  top-K nearest other conditions ordered by Sinkhorn distance. Generic:
  ``query_set``/``target_set`` filters let the same machinery serve
  gene→gene similarity (today, no drugs in the data) or drug→gene
  matches (when drugs land in the H5).
- :func:`permutation_test` — recompute Sinkhorn distances on shuffled
  cell→condition labels (the cell-level point clouds aren't kept on
  disk, so permutation tests need access to the source CNN-embeddings
  parquet). FDR-BH controls the family-wise false-discovery rate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


def load_ot_sidecar(sidecar_path: Path) -> tuple["np.ndarray", "pd.DataFrame"]:
    """Read a cached OT sidecar parquet.

    Returns ``(D, meta_df)`` where ``D`` is the symmetric distance matrix
    and ``meta_df`` carries the per-group metadata in row order.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(sidecar_path)
    d_cols = sorted(
        [c for c in df.columns if c.startswith("d_")],
        key=lambda c: int(c.split("_", 1)[1]),
    )
    if not d_cols:
        raise ValueError(
            f"OT sidecar {sidecar_path} has no d_* columns",
        )
    D = df[d_cols].to_numpy(dtype=np.float64)
    meta_df = df.drop(columns=d_cols).reset_index(drop=True)
    return D, meta_df


def rank_condition_matches(
    sidecar_path: Path,
    *,
    top_k: int = 5,
    query_set: Optional[Sequence[str]] = None,
    target_set: Optional[Sequence[str]] = None,
    self_distance_threshold: float = 1e-9,
) -> "pd.DataFrame":
    """For each query condition, list its top-K nearest matches.

    Returns a long DataFrame with columns
    ``query, query_run_id, rank, match, match_run_id, distance``.

    ``query_set`` / ``target_set`` accept lists of condition labels.
    ``None`` means "all". When the H5 carries ``is_drug`` /
    ``is_control`` flags, callers can use those upstream to construct
    drug-only or gene-only sets — the function itself is label-agnostic.

    Self-pairs (distance ≤ ``self_distance_threshold``) are dropped so
    a query never matches itself.
    """
    import numpy as np
    import pandas as pd

    D, meta = load_ot_sidecar(sidecar_path)
    n = len(meta)
    if n == 0:
        return pd.DataFrame(
            columns=["query", "rank", "match", "distance"],
        )

    cond_labels = meta.get(
        "condition_label", pd.Series([str(i) for i in range(n)]),
    ).astype(str).tolist()
    run_ids = meta.get(
        "run_id", pd.Series([""] * n),
    ).astype(str).tolist()

    if query_set is not None:
        q_set = {str(s) for s in query_set}
        q_indices = [i for i, c in enumerate(cond_labels) if c in q_set]
    else:
        q_indices = list(range(n))
    if target_set is not None:
        t_set = {str(s) for s in target_set}
        t_indices = [i for i, c in enumerate(cond_labels) if c in t_set]
    else:
        t_indices = list(range(n))

    rows: list[dict] = []
    t_arr = np.array(t_indices, dtype=int)
    for qi in q_indices:
        d = D[qi, t_arr]
        # Drop self.
        keep = (t_arr != qi) & (d > self_distance_threshold)
        if not keep.any():
            continue
        d_keep = d[keep]
        targets = t_arr[keep]
        order = np.argsort(d_keep)[:top_k]
        for rank, idx in enumerate(order, start=1):
            tj = int(targets[idx])
            rows.append({
                "query": cond_labels[qi],
                "query_run_id": run_ids[qi],
                "rank": rank,
                "match": cond_labels[tj],
                "match_run_id": run_ids[tj],
                "distance": float(d_keep[idx]),
            })
    return pd.DataFrame(rows)


def permutation_test(
    embeddings_parquet: Path,
    sidecar_path: Path,
    *,
    pairs: Optional[Sequence[tuple[str, str]]] = None,
    top_k_per_query: int = 1,
    n_perm: int = 1000,
    sub_n: int = 200,
    sinkhorn_reg: float = 0.05,
    random_state: int = 42,
    progress_cb=None,
) -> "pd.DataFrame":
    """Permutation-test the top-K matches per query condition.

    For each (query, match) pair the test shuffles cell→condition labels
    at the cell level, redraws ``sub_n`` cells from each (shuffled)
    pseudo-condition, and recomputes the Sinkhorn distance. The
    permutation p-value is ``(count + 1) / (n_perm + 1)`` where
    ``count`` is the number of permutations whose shuffled distance is
    ≤ the observed distance. FDR-BH controls the family.

    Args:
        embeddings_parquet: per-cell parquet written by
            ``autoencoder.extract_embeddings`` — must carry one row per
            cell with ``emb_*`` columns and ``condition_label``.
        sidecar_path: cached OT sidecar from a previous render call.
            Used to identify which (query, match) pairs are worth testing.
        pairs: explicit (query, match) pairs. When None, runs
            ``rank_condition_matches(..., top_k=top_k_per_query)`` to
            pick the top-K matches per query.
        top_k_per_query: when ``pairs`` is None, take this many top
            matches per query.
        n_perm: number of permutations.
        sub_n: cells subsampled per condition per permutation.
        sinkhorn_reg: regularisation for the Sinkhorn solver. Same
            semantics as in ``qc_plots._sinkhorn_divergence_matrix``
            (rescaled internally to a fraction of median cost).

    Returns a DataFrame with one row per pair and columns
    ``query, match, distance, p_perm, q_FDR, significant``.
    """
    import numpy as np
    import pandas as pd

    try:
        import ot
    except ImportError as exc:
        raise ImportError("permutation_test requires POT (pip install pot)") from exc

    df = pd.read_parquet(embeddings_parquet)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        emb_cols = [c for c in df.columns if c.startswith("pcot_")]
    if not emb_cols or "condition_label" not in df.columns:
        raise ValueError(
            "embeddings parquet must have emb_* columns and condition_label",
        )

    rng = np.random.default_rng(random_state)

    if pairs is None:
        ranked = rank_condition_matches(
            sidecar_path, top_k=int(top_k_per_query),
        )
        pairs_local = list(zip(ranked["query"], ranked["match"]))
    else:
        pairs_local = [(str(a), str(b)) for a, b in pairs]
    if not pairs_local:
        return pd.DataFrame(
            columns=["query", "match", "distance", "p_perm", "q_FDR", "significant"],
        )

    # Group cells by condition_label.
    by_cond: dict[str, np.ndarray] = {
        c: g[emb_cols].to_numpy(dtype=np.float32)
        for c, g in df.groupby("condition_label")
    }

    # Pre-compute the typical cost scale once on a representative pair so
    # ``sinkhorn_reg`` interprets as a fraction of cost, not an absolute.
    sample_pair_keys = list(by_cond.keys())[:8]
    sample_cost = 1.0
    if len(sample_pair_keys) >= 2:
        a = by_cond[sample_pair_keys[0]]
        b = by_cond[sample_pair_keys[1]]
        if len(a) > 4 and len(b) > 4:
            M_sample = ot.dist(a[:200], b[:200], metric="sqeuclidean")
            sample_cost = float(np.median(M_sample)) or 1.0
    abs_reg = sinkhorn_reg * sample_cost

    def _wasserstein(pc1: np.ndarray, pc2: np.ndarray) -> float:
        n1, n2 = len(pc1), len(pc2)
        if n1 < 2 or n2 < 2:
            return float("nan")
        a = np.full(n1, 1.0 / n1)
        b = np.full(n2, 1.0 / n2)
        M = ot.dist(pc1, pc2, metric="sqeuclidean")
        try:
            return float(ot.sinkhorn2(
                a, b, M, reg=abs_reg, numItermax=1000,
                method="sinkhorn_log",
            ))
        except Exception:  # noqa: BLE001
            return float("nan")

    def _subsample(pc: np.ndarray) -> np.ndarray:
        if len(pc) <= sub_n:
            return pc
        idx = rng.choice(len(pc), size=sub_n, replace=False)
        return pc[idx]

    # Observed distances.
    observed: list[float] = []
    valid_pairs: list[tuple[str, str]] = []
    for q, m in pairs_local:
        if q not in by_cond or m not in by_cond:
            continue
        d = _wasserstein(_subsample(by_cond[q]), _subsample(by_cond[m]))
        if not np.isnan(d):
            observed.append(d)
            valid_pairs.append((q, m))
    if not valid_pairs:
        return pd.DataFrame(
            columns=["query", "match", "distance", "p_perm", "q_FDR", "significant"],
        )

    # For each pair, run n_perm shuffles. We pool the two conditions' cells,
    # shuffle, split back into the original sizes, and recompute distance.
    p_values: list[float] = []
    for k, (q, m) in enumerate(valid_pairs):
        pc_q = by_cond[q]
        pc_m = by_cond[m]
        pool = np.vstack([pc_q, pc_m])
        n_q = len(pc_q)
        count = 0
        for p in range(n_perm):
            order = rng.permutation(len(pool))
            shuffled = pool[order]
            pseudo_q = _subsample(shuffled[:n_q])
            pseudo_m = _subsample(shuffled[n_q:])
            d = _wasserstein(pseudo_q, pseudo_m)
            if not np.isnan(d) and d <= observed[k]:
                count += 1
        p_values.append((count + 1) / (n_perm + 1))
        if progress_cb is not None:
            progress_cb((k + 1) / len(valid_pairs))

    # Benjamini-Hochberg FDR.
    p_arr = np.array(p_values, dtype=np.float64)
    n = len(p_arr)
    order = np.argsort(p_arr)
    ranked_p = p_arr[order]
    bh = ranked_p * n / (np.arange(n) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    q_arr = np.empty_like(bh)
    q_arr[order] = np.minimum(bh, 1.0)

    return pd.DataFrame({
        "query": [q for q, _ in valid_pairs],
        "match": [m for _, m in valid_pairs],
        "distance": observed,
        "p_perm": p_values,
        "q_FDR": q_arr,
        "significant": q_arr <= 0.05,
    })
