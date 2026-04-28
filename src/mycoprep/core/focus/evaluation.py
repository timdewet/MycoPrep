"""Compare manual ground-truth labels against each focus metric's pick.

Reads ``manual_labels.csv`` (written by ``focuspicker.labeling``) and the
per-scene ``scene<NN>_scores.csv`` files (written by ``focuspicker.review``)
and reports per-metric agreement with the human labels.

Scenes the user marked as having no in-focus slice (``chosen_z == -1``) are
excluded from the per-metric accuracy numbers but counted and reported
separately, since "no metric was right" is itself a useful failure mode to
track.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .focus import METRIC_NAMES
from .labeling import LABELS_FILENAME, NO_FOCUS, load_manual_labels

EVALUATION_FILENAME = "metric_evaluation.csv"
_SCORES_RE = re.compile(r"scene(\d+)_scores\.csv$")


@dataclass
class MetricEval:
    metric: str
    n_evaluated: int
    exact_match: int
    within_1: int
    mean_abs_error: float


def _scores_path_for_scene(review_dir: Path, scene_index: int) -> Path:
    # The review tool writes scene<NN>_scores.csv with two-digit zero-padding,
    # but be tolerant of other widths just in case.
    candidates = [
        review_dir / f"scene{scene_index:02d}_scores.csv",
        review_dir / f"scene{scene_index}_scores.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Fall back to scanning by parsing filenames.
    for path in review_dir.glob("scene*_scores.csv"):
        m = _SCORES_RE.search(path.name)
        if m and int(m.group(1)) == scene_index:
            return path
    raise FileNotFoundError(
        f"no scores CSV for scene {scene_index} in {review_dir}"
    )


def _read_metric_picks(scores_path: Path) -> dict[str, int]:
    """Return ``{metric_name: chosen_z}`` for one scene's scores CSV."""
    picks: dict[str, int] = {}
    with scores_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            z = int(row["z"])
            for metric in METRIC_NAMES:
                col = f"chosen_by_{metric}"
                if col in row and row[col] == "1":
                    picks[metric] = z
    missing = [m for m in METRIC_NAMES if m not in picks]
    if missing:
        raise RuntimeError(
            f"{scores_path.name} is missing chosen_by_ columns for {missing}"
        )
    return picks


def evaluate_metrics(review_dir: Path) -> dict:
    """Score each metric against the manual labels in ``review_dir``.

    Writes ``metric_evaluation.csv`` and prints a ranked summary table.
    Returns a dict with per-metric stats and the no-focus count.
    """
    review_dir = Path(review_dir)
    labels_path = review_dir / LABELS_FILENAME
    if not labels_path.exists():
        raise FileNotFoundError(
            f"no {LABELS_FILENAME} in {review_dir}; run `focuspicker label` first"
        )

    labels = load_manual_labels(labels_path)
    if not labels:
        raise RuntimeError(f"{labels_path} is empty")

    no_focus_scenes = sorted(s for s, z in labels.items() if z == NO_FOCUS)
    eval_scenes = sorted(s for s, z in labels.items() if z != NO_FOCUS)

    # Per-metric tallies.
    per_metric: dict[str, dict] = {
        m: {"errors": [], "exact": 0, "within_1": 0} for m in METRIC_NAMES
    }

    for scene_index in eval_scenes:
        truth = labels[scene_index]
        scores_path = _scores_path_for_scene(review_dir, scene_index)
        picks = _read_metric_picks(scores_path)
        for metric in METRIC_NAMES:
            err = abs(picks[metric] - truth)
            per_metric[metric]["errors"].append(err)
            if err == 0:
                per_metric[metric]["exact"] += 1
            if err <= 1:
                per_metric[metric]["within_1"] += 1

    n = len(eval_scenes)
    results: list[MetricEval] = []
    for metric in METRIC_NAMES:
        errs = per_metric[metric]["errors"]
        mae = sum(errs) / len(errs) if errs else 0.0
        results.append(
            MetricEval(
                metric=metric,
                n_evaluated=n,
                exact_match=per_metric[metric]["exact"],
                within_1=per_metric[metric]["within_1"],
                mean_abs_error=mae,
            )
        )

    # Rank: most exact matches first, then lowest MAE.
    results.sort(key=lambda r: (-r.exact_match, r.mean_abs_error))

    out_path = review_dir / EVALUATION_FILENAME
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["rank", "metric", "n_evaluated", "exact_match", "within_1", "mean_abs_error"]
        )
        for rank, r in enumerate(results, start=1):
            writer.writerow(
                [rank, r.metric, r.n_evaluated, r.exact_match, r.within_1, f"{r.mean_abs_error:.4f}"]
            )

    # Pretty print.
    print(
        f"[focuspicker] evaluated {n} labelled scenes "
        f"({len(no_focus_scenes)} marked no_focus, excluded from accuracy)"
    )
    if no_focus_scenes:
        print(f"  no_focus scenes: {no_focus_scenes}")
    print()
    print(f"  {'rank':>4}  {'metric':<22}  {'exact':>7}  {'within1':>8}  {'MAE':>8}")
    for rank, r in enumerate(results, start=1):
        exact_pct = (r.exact_match / n * 100) if n else 0.0
        within_pct = (r.within_1 / n * 100) if n else 0.0
        print(
            f"  {rank:>4}  {r.metric:<22}  "
            f"{r.exact_match:>3}/{n:<3} ({exact_pct:>3.0f}%)  "
            f"{r.within_1:>3}/{n:<3}  {r.mean_abs_error:>8.3f}"
        )
    print()
    print(f"[focuspicker] wrote {out_path}")

    return {
        "results": results,
        "n_evaluated": n,
        "no_focus_scenes": no_focus_scenes,
        "evaluation_csv": out_path,
    }
