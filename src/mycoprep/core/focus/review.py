"""Review-mode helper: dump a sample of scenes for visual QC of focus picks.

For each sampled scene the reviewer writes:

    scene<NN>_phase_stack.tif                # (Z, Y, X) — every phase slice
    scene<NN>_chosen_<metric>_z<K>.tif       # one (Y, X) per metric — open them
                                             #   together in Fiji to compare
    scene<NN>_scores.csv                     # per-slice scores for every metric

Plus a top-level ``metric_comparison.csv`` with one row per FOV listing which
slice each metric picked, so you can spot disagreements at a glance.

The original CZI is never touched.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

from . import channel_id, focus, io_czi
from .pipeline import Options


@dataclass
class ReviewResult:
    scene_index: int
    chosen_z_per_metric: dict[str, int]
    phase_stack_path: Path
    chosen_slice_paths: dict[str, Path] = field(default_factory=dict)
    scores_csv_path: Path | None = None

    @property
    def chosen_z(self) -> int:  # for backward-compat with old callers/tests
        # Default to whichever metric was passed via opts.metric — falls back to
        # the first metric if not present.
        return next(iter(self.chosen_z_per_metric.values()))


def _write_scene_scores_csv(
    path: Path, scene_index: int, scores: dict[str, np.ndarray], chosen_per_metric: dict[str, int]
) -> None:
    n_z = len(next(iter(scores.values())))
    metric_names = list(scores)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["scene", "z"] + metric_names + [f"chosen_by_{m}" for m in metric_names]
        writer.writerow(header)
        for z in range(n_z):
            row = [scene_index, z]
            row.extend(f"{scores[m][z]:.6g}" for m in metric_names)
            row.extend("1" if chosen_per_metric[m] == z else "0" for m in metric_names)
            writer.writerow(row)


def review_czi(
    czi_path: Path,
    out_dir: Path,
    n_scenes: int = 10,
    opts: Options | None = None,
    seed: int | None = 0,
    crop_fraction: float = 1.0,
    preblur_sigma: float = 0.0,
    scene_indices: list[int] | None = None,
    use_mask: bool = False,
    smooth_z: bool = False,
) -> list[ReviewResult]:
    """Sample scenes from a CZI and dump phase stacks + per-metric chosen slices.

    If ``scene_indices`` is given, those exact scenes are reviewed and ``n_scenes``
    / ``seed`` are ignored. Otherwise ``n_scenes`` are randomly sampled.
    """
    opts = opts or Options(archive_original=False)
    czi_path = Path(czi_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_indices = io_czi.list_scene_indices(czi_path)
    if not all_indices:
        raise RuntimeError(f"no scenes found in {czi_path}")

    if scene_indices is not None:
        missing = [s for s in scene_indices if s not in all_indices]
        if missing:
            raise RuntimeError(
                f"requested scenes not present in {czi_path.name}: {missing}"
            )
        sampled = sorted(set(scene_indices))
    else:
        rng = random.Random(seed)
        k = min(n_scenes, len(all_indices))
        sampled = sorted(rng.sample(all_indices, k))

    # Read all sampled scenes up front so we can decide the phase channel once
    # from the pooled statistics. Per-scene decisions are unreliable when one
    # scene has an artifact or unusually skewed phase field.
    scenes = [io_czi.read_scene(czi_path, idx) for idx in sampled]
    channel_names = scenes[0].channel_names

    if opts.phase_channel is None:
        phase_idx = channel_id.detect_phase_channel_multi(
            [s.array_zcyx for s in scenes]
        )
    else:
        phase_idx = channel_id.resolve_phase_channel(
            scenes[0].array_zcyx, channel_names, opts.phase_channel
        )

    # Print pooled per-channel diagnostics so the decision is auditable.
    pooled = [
        channel_id.channel_stats(s.array_zcyx, channel_names) for s in scenes
    ]
    n_c = len(channel_names)
    print(f"[focuspicker] channels in {czi_path.name} (pooled over {len(scenes)} scenes):")
    print(
        f"  {'idx':>3}  {'name':<16}  {'mean':>10}  {'mean|skew|':>11}  {'near_bg':>8}"
    )
    for c in range(n_c):
        mean_intensity = float(np.mean([p[c].mean for p in pooled]))
        mean_abs_skew = float(np.mean([abs(p[c].skewness) for p in pooled]))
        mean_near_bg = float(np.mean([p[c].near_bg_fraction for p in pooled]))
        marker = "  <- phase" if c == phase_idx else ""
        print(
            f"  {c:>3}  {channel_names[c]:<16}  {mean_intensity:>10.1f}  "
            f"{mean_abs_skew:>11.2f}  {mean_near_bg:>8.2f}{marker}"
        )
    if crop_fraction < 1.0 or preblur_sigma > 0 or use_mask or smooth_z:
        print(
            f"[focuspicker] preprocessing: center crop={crop_fraction:.2f}, "
            f"preblur sigma={preblur_sigma:.2f}, mask={use_mask}, smooth_z={smooth_z}"
        )

    results: list[ReviewResult] = []
    metric_names = list(focus.METRIC_NAMES)
    for scene in scenes:
        scene_index = scene.index
        phase_stack = scene.array_zcyx[:, phase_idx]  # (Z, Y, X)
        scores = focus.score_stack(
            phase_stack,
            crop_fraction=crop_fraction,
            preblur_sigma=preblur_sigma,
            use_mask=use_mask,
            smooth_z=smooth_z,
        )
        chosen_per_metric = {m: focus.pick_best_slice(scores, metric=m) for m in metric_names}

        stack_path = out_dir / f"scene{scene_index:02d}_phase_stack.tif"
        tifffile.imwrite(
            str(stack_path),
            phase_stack,
            photometric="minisblack",
            metadata={"axes": "ZYX"},
            imagej=True,
        )

        chosen_paths: dict[str, Path] = {}
        for metric, z in chosen_per_metric.items():
            chosen_path = out_dir / f"scene{scene_index:02d}_chosen_{metric}_z{z}.tif"
            tifffile.imwrite(
                str(chosen_path),
                phase_stack[z],
                photometric="minisblack",
                metadata={"axes": "YX"},
                imagej=True,
            )
            chosen_paths[metric] = chosen_path

        csv_path = out_dir / f"scene{scene_index:02d}_scores.csv"
        _write_scene_scores_csv(csv_path, scene_index, scores, chosen_per_metric)

        results.append(
            ReviewResult(
                scene_index=scene_index,
                chosen_z_per_metric=chosen_per_metric,
                phase_stack_path=stack_path,
                chosen_slice_paths=chosen_paths,
                scores_csv_path=csv_path,
            )
        )

    # Top-level comparison table: one row per FOV, one column per metric.
    comparison_path = out_dir / "metric_comparison.csv"
    with comparison_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scene"] + list(metric_names) + ["agreement"])
        for r in results:
            picks = [r.chosen_z_per_metric[m] for m in metric_names]
            agreement = "all_agree" if len(set(picks)) == 1 else f"{len(set(picks))}_distinct"
            writer.writerow([r.scene_index, *picks, agreement])

    # Pretty-print the comparison so you can see it without opening the CSV.
    print()
    print(f"[focuspicker] per-FOV slice picks (open scene*_chosen_<metric>_z*.tif in Fiji):")
    header = "  scene  " + "  ".join(f"{m:>20}" for m in metric_names)
    print(header)
    for r in results:
        cells = "  ".join(
            f"{('z=' + str(r.chosen_z_per_metric[m])):>20}" for m in metric_names
        )
        print(f"  {r.scene_index:>5}  {cells}")
    print()
    print(f"[focuspicker] wrote review to {out_dir}")

    return results
