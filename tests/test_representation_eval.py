"""Correctness tests for the representation_eval evaluator.

Runs against synthetic distance matrices with known structure so the
metrics are exactly predictable. No filesystem / no training.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from mycoprep.core.extract.representation_eval import (
    SCOPE_ALL,
    SCOPE_CROSS_GENE,
    SCOPE_REPLICATE,
    RepresentationMetrics,
    evaluate_distance_matrix,
    evaluate_feature_matrix,
    load_group_table,
)


def _perfect_matrix(n_per_group: int, n_groups: int) -> tuple[np.ndarray, list[str], list[str]]:
    """A distance matrix where same-group entries are 0.1 apart and
    different-group entries are 1.0 apart. Trivially separable."""
    n = n_per_group * n_groups
    D = np.full((n, n), 1.0)
    np.fill_diagonal(D, 0.0)
    conds, genes = [], []
    for g in range(n_groups):
        for i in range(n_per_group):
            conds.append(f"g{g}_c{i}")
            genes.append(f"g{g}_gene{i}")
    for g in range(n_groups):
        for i in range(n_per_group):
            for j in range(n_per_group):
                if i == j:
                    continue
                ii = g * n_per_group + i
                jj = g * n_per_group + j
                D[ii, jj] = 0.1
    return D, conds, genes


def test_perfect_separation_gives_map_one():
    D, conds, genes = _perfect_matrix(3, 2)
    # gene -> group: same-prefix genes share group.
    group_map = {f"g{g}_gene{i}": f"group_{g}" for g in range(2) for i in range(3)}

    m = evaluate_distance_matrix(
        D, conds, genes, group_map,
        representation="synthetic", batch_correct=True,
        grouping="perfect", species="test",
    )
    assert m.n_conditions_evaluated == 6, m
    assert math.isclose(m.knn_accuracy[1], 1.0), m.knn_accuracy
    # kNN@5 averages over 5 nearest; with 2 same-group neighbours per query
    # out of 5 others, the best achievable accuracy is 2/5 = 0.4.
    assert math.isclose(m.knn_accuracy[5], 0.4), m.knn_accuracy
    assert math.isclose(m.map_overall, 1.0), m.map_overall
    assert m.n_dropped_unmapped == 0
    assert m.n_dropped_singleton_group == 0


def test_random_labels_drop_map_to_near_chance():
    D, conds, genes = _perfect_matrix(3, 2)
    # Scramble: rotate group assignments so each gene lands in the "wrong"
    # group relative to its distance neighbours.
    rng = np.random.default_rng(0)
    assignments = list(range(6))
    rng.shuffle(assignments)
    group_map = {
        f"g{(i // 3)}_gene{(i % 3)}": f"group_{assignments[i] % 2}"
        for i in range(6)
    }
    m = evaluate_distance_matrix(
        D, conds, genes, group_map,
        representation="synthetic", batch_correct=True,
        grouping="random", species="test",
    )
    # mAP should be well below 1.0 for shuffled labels.
    assert m.map_overall < 0.95, m.map_overall


def test_unmapped_and_singletons_counted():
    D, conds, genes = _perfect_matrix(3, 2)
    # Only map 4 of 6 genes; leave one singleton group.
    group_map = {
        "g0_gene0": "alpha",
        "g0_gene1": "alpha",
        "g1_gene0": "beta",      # singleton if no other beta
        "g0_gene2": "alpha",
    }
    m = evaluate_distance_matrix(
        D, conds, genes, group_map,
        representation="synthetic", batch_correct=True,
        grouping="partial", species="test",
    )
    assert m.n_dropped_unmapped == 2, m   # g1_gene1, g1_gene2 unmapped
    assert m.n_dropped_singleton_group == 1, m  # 'beta' has only one member
    assert m.n_conditions_evaluated == 3


def test_two_groupings_independent():
    D, conds, genes = _perfect_matrix(3, 2)
    perfect_map = {
        f"g{g}_gene{i}": f"perfect_{g}" for g in range(2) for i in range(3)
    }
    random_map = {
        f"g{g}_gene{i}": f"random_{(g * 3 + i) % 2}"
        for g in range(2) for i in range(3)
    }
    m_perfect = evaluate_distance_matrix(
        D, conds, genes, perfect_map,
        representation="x", batch_correct=True,
        grouping="perfect", species="t",
    )
    m_random = evaluate_distance_matrix(
        D, conds, genes, random_map,
        representation="x", batch_correct=True,
        grouping="random", species="t",
    )
    assert math.isclose(m_perfect.map_overall, 1.0)
    assert m_random.map_overall < m_perfect.map_overall


def test_load_group_table(tmp_path: Path):
    csv = tmp_path / "g.csv"
    csv.write_text(
        "gene,operon,family\n"
        "hadA,had,had\n"
        "hadB,had,had\n"
        "inhA,,fas\n"
    )
    names, mappings = load_group_table(csv)
    assert set(names) == {"operon", "family"}
    assert mappings["operon"]["hada"] == "had"
    assert "inha" not in mappings["operon"]   # empty cell -> unmapped
    assert mappings["family"]["inha"] == "fas"


def _replicate_matrix(
    layout: list[tuple[str, str, int]],
    *,
    same_replicate_d: float = 0.05,
    same_group_diff_gene_d: float = 0.5,
    different_group_d: float = 1.0,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Build a distance matrix from ``[(group, gene, n_replicates), ...]``.

    Returns ``(D, condition_labels, gene_labels, group_labels)``.
    """
    conds, genes, groups = [], [], []
    for group, gene, n in layout:
        for r in range(n):
            conds.append(f"{gene}_rep{r}")
            genes.append(gene)
            groups.append(group)
    n_total = len(conds)
    D = np.full((n_total, n_total), different_group_d)
    np.fill_diagonal(D, 0.0)
    for i in range(n_total):
        for j in range(n_total):
            if i == j:
                continue
            if genes[i] == genes[j]:
                D[i, j] = same_replicate_d
            elif groups[i] == groups[j]:
                D[i, j] = same_group_diff_gene_d
    return D, conds, genes, groups


def test_cross_gene_does_not_inflate_for_replicate_heavy_gene():
    # Group A has hadA × 5 replicates, hadB × 3, hadC × 1; group B has xx × 5
    # and yy × 3. With same-gene = 0.05, same-group-diff-gene = 0.5,
    # different-group = 1.0, every same-gene row dominates the top of a
    # query's ranking — so scope='all' gives kNN@5 = 1.0 (replicate
    # matching guarantees all 5 nearest are same-group). Under cross_gene
    # the same-gene rows are masked out, so a hadA query's top 5 candidates
    # are {hadB×3, hadC×1, xx×1} → only 4 of 5 are same-group → kNN@5 = 0.8.
    D, conds, genes, _ = _replicate_matrix(
        [("A", "hadA", 5), ("A", "hadB", 3), ("A", "hadC", 1),
         ("B", "xx", 5), ("B", "yy", 3)],
    )
    group_map = {"hada": "A", "hadb": "A", "hadc": "A", "xx": "B", "yy": "B"}

    m_all = evaluate_distance_matrix(
        D, conds, genes, group_map,
        representation="x", batch_correct=True,
        grouping="operon", species="t", scope=SCOPE_ALL,
    )
    m_cross = evaluate_distance_matrix(
        D, conds, genes, group_map,
        representation="x", batch_correct=True,
        grouping="operon", species="t", scope=SCOPE_CROSS_GENE,
    )

    # scope='all' has replicate matching pinning kNN@5 to 1.0 — every
    # query's top 5 are same-gene or same-group same-cluster rows.
    assert math.isclose(m_all.knn_accuracy[5], 1.0), m_all.knn_accuracy
    assert math.isclose(m_all.map_overall, 1.0), m_all.map_overall
    # scope='cross_gene' strips out same-gene matches. The new top-5 of
    # a hadA query is {3 hadB, 1 hadC, 1 xx} → 4 positives out of 5.
    # Averaged across queries this comes out strictly below 1.0.
    assert m_cross.knn_accuracy[5] < 0.95, m_cross.knn_accuracy
    assert m_cross.knn_accuracy[5] > 0.5, m_cross.knn_accuracy  # not random


def test_replicate_scope_perfect_matching():
    # Two genes × 3 replicates each; same-gene rows are tight, so the
    # replicate-scope mAP should be 1.0 regardless of group structure.
    D, conds, genes, _ = _replicate_matrix(
        [("A", "geneX", 3), ("B", "geneY", 3)],
    )
    # group_map is ignored under scope='replicate' — pass empty.
    m = evaluate_distance_matrix(
        D, conds, genes, {},
        representation="x", batch_correct=True,
        grouping="(replicate)", species="t",
        scope=SCOPE_REPLICATE,
    )
    assert math.isclose(m.map_overall, 1.0), m.map_overall
    assert math.isclose(m.knn_accuracy[1], 1.0)
    assert m.n_conditions_evaluated == 6


def test_cross_gene_singleton_when_only_one_gene_in_group():
    # Group A has 3 replicates of a single gene; group B has 2 of one gene.
    # Under cross_gene, group A has only 1 distinct gene → singleton; same
    # for group B. Nothing left to score.
    D, conds, genes, _ = _replicate_matrix(
        [("A", "geneA", 3), ("B", "geneB", 2)],
    )
    group_map = {"genea": "A", "geneb": "B"}
    m = evaluate_distance_matrix(
        D, conds, genes, group_map,
        representation="x", batch_correct=True,
        grouping="operon", species="t", scope=SCOPE_CROSS_GENE,
    )
    assert m.n_conditions_evaluated == 0, m
    assert m.n_dropped_singleton_group == 5, m


def test_replicate_key_mode_condition_distinguishes_reporters():
    """When the replicate key is the full condition (not just gene),
    different reporter backgrounds of the same gene become distinct
    biological observations rather than replicates."""
    # 4 rows: hadA-Wag31 ×2, hadA-FtsZ ×2. All same gene, different reporters.
    conds = ["hadA Wag31", "hadA Wag31", "hadA FtsZ", "hadA FtsZ"]
    genes = ["hadA"] * 4
    # Tight within reporter (0.05), loose between reporters (0.6).
    D = np.array([
        [0.0, 0.05, 0.6, 0.6],
        [0.05, 0.0, 0.6, 0.6],
        [0.6, 0.6, 0.0, 0.05],
        [0.6, 0.6, 0.05, 0.0],
    ])
    # With replicate_keys = condition_labels: same-condition rows are
    # replicates; different-condition rows are not. Under scope=replicate,
    # this is exactly recovered.
    m_cond = evaluate_distance_matrix(
        D, conds, genes, {},
        representation="x", batch_correct=False,
        grouping="(replicate · condition)", species="t",
        scope=SCOPE_REPLICATE,
        replicate_keys=conds,
    )
    assert math.isclose(m_cond.map_overall, 1.0), m_cond.map_overall

    # With replicate_keys = gene: all rows are the same gene, so the
    # candidate set is empty under cross_gene (no positives possible);
    # under replicate scope, all 4 rows are mutual positives, including
    # the cross-reporter ones at distance 0.6 — so AP < 1.0.
    m_gene = evaluate_distance_matrix(
        D, conds, genes, {},
        representation="x", batch_correct=False,
        grouping="(replicate · gene)", species="t",
        scope=SCOPE_REPLICATE,
        replicate_keys=genes,
    )
    assert math.isclose(m_gene.map_overall, 1.0), m_gene.map_overall
    # All 4 rows share gene; kNN@1 = 1.0 because the closest other row
    # under either reporter is still the same gene.
    assert math.isclose(m_gene.knn_accuracy[1], 1.0)


def test_feature_matrix_cosine_perfect():
    # 6 points in 4-D: two tight orthogonal clusters.
    X = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.01, 0.0, 0.0],
        [0.99, 0.0, 0.01, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.01, 0.0],
        [0.0, 0.99, 0.0, 0.01],
    ])
    conds = [f"c{i}" for i in range(6)]
    genes = [f"gene{i}" for i in range(6)]
    group_map = {
        "gene0": "A", "gene1": "A", "gene2": "A",
        "gene3": "B", "gene4": "B", "gene5": "B",
    }
    m = evaluate_feature_matrix(
        X, conds, genes, group_map,
        representation="feat", batch_correct=False,
        grouping="cluster", species="t",
        metric="cosine",
    )
    assert math.isclose(m.map_overall, 1.0), m
    assert math.isclose(m.knn_accuracy[1], 1.0)


if __name__ == "__main__":
    import sys
    failed = []
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        try:
            sig = fn.__code__.co_varnames[: fn.__code__.co_argcount]
            if "tmp_path" in sig:
                import tempfile
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"OK  {fn.__name__}")
        except AssertionError as exc:
            failed.append((fn.__name__, exc))
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:
            failed.append((fn.__name__, exc))
            print(f"ERR  {fn.__name__}: {exc}")
    if failed:
        sys.exit(1)
    print("\nall tests passed")
