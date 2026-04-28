"""
Segment pre-normalised TIFF hyperstacks with Cellpose-SAM.

Designed for M. smegmatis data stored as individual multi-channel, multi-FOV
TIFF files in a Replica/Plate folder hierarchy.

Input structure:
    input/
    ├── Replica 1/
    │   ├── Plate 1/
    │   │   ├── experiment_001.tif   (N_FOV, C, Y, X)
    │   │   └── experiment_002.tif
    │   ├── Plate 2/
    │   │   └── ...
    │   └── Plate 3/
    │       └── ...
    └── Replica 2/
        └── ...

Output (flat directory):
    output/
    ├── Replica_1__Plate_1__experiment_001.tif
    ├── Replica_1__Plate_1__experiment_002.tif
    ...

Each output TIFF is an ImageJ hyperstack (N_FOV, C+1, Y, X) with a binary
mask appended as the last channel — identical to cellpose_pipeline.py output.

Sample mode (--sample):
    Picks 1 random FOV per input TIFF for training data diversity.
    Output is compatible with label_cells.py for interactive labelling.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from .cellpose_pipeline import segment_phase, add_cell_boundaries, save_hyperstack
from .label_cells import load_hyperstack


def process_tiff_file(tiff_path, model, phase_channel, diameter=None,
                      classify_opts=None, model_type="cpsam", sample=False):
    """
    Segment all (or a sampled) FOVs in a multi-FOV TIFF hyperstack.

    Args:
        tiff_path: Path to input TIFF file
        model: CellposeModel instance
        phase_channel: 0-indexed channel number for phase contrast
        diameter: Expected cell diameter in pixels (None = auto-estimate)
        classify_opts: dict with cell quality filtering options, or None
        model_type: Cellpose model type string
        sample: If True, process only 1 random FOV

    Returns:
        stacked: numpy array (N_processed, C+1, Y, X)
        fov_names: list of FOV identifier strings
        total_cells: total cells segmented
    """
    data, metadata = load_hyperstack(tiff_path)  # (N_FOV, C, Y, X)
    n_fov = data.shape[0]

    if sample:
        indices = [int(np.random.choice(n_fov, 1))]
    else:
        indices = list(range(n_fov))

    fov_arrays = []
    fov_names = []
    cell_counts = []

    for i, fov_idx in enumerate(indices):
        print(f"    FOV {fov_idx+1}/{n_fov}", end=" ... ")

        channels = data[fov_idx]  # (C, Y, X)
        n_channels = channels.shape[0]

        if phase_channel >= n_channels:
            print(f"SKIPPED (phase channel {phase_channel} but only {n_channels} channels)")
            continue

        phase = channels[phase_channel]

        masks = segment_phase(phase, model, diameter=diameter, model_type=model_type)
        n_cells = masks.max()
        print(f"{n_cells} cells", end=" ... ")

        masks = add_cell_boundaries(masks, boundary_width=1)

        # Optional CNN-based quality filtering
        if classify_opts is not None:
            from .cell_quality_classifier import classify_and_filter_mask
            masks, report = classify_and_filter_mask(
                labeled_mask=masks,
                image_channels=channels,
                phase_channel=phase_channel,
                model_path=classify_opts.get("model_path"),
                pixels_per_um=classify_opts.get("pixels_per_um", 13.8767),
                keep_classes=classify_opts.get("keep_classes", ("good",)),
                confidence_threshold=classify_opts.get("confidence_threshold", 0.5),
                use_rules=classify_opts.get("use_rules", True),
                device=classify_opts.get("device"),
                verbose=True,
            )
            n_cells = report["kept"]

        masks_binary = np.where(masks > 0, 255, 0).astype(np.uint16)
        channels_16 = channels.astype(np.uint16)
        composite = np.concatenate([channels_16, masks_binary[np.newaxis]], axis=0)

        fov_arrays.append(composite)
        fov_names.append(f"fov_{fov_idx:03d}")
        cell_counts.append(n_cells)
        print("done")

    if not fov_arrays:
        return None, [], 0

    stacked = np.stack(fov_arrays, axis=0)
    return stacked, fov_names, sum(cell_counts)


def main():
    print("=" * 70)
    print("DEPRECATED: tiff_pipeline.py has been merged into cellpose_pipeline.py")
    print()
    print("Use instead:")
    print("  python cellpose_pipeline.py --format tiff -i <input> -o <output>")
    print()
    print("All tiff_pipeline.py flags are supported. For example:")
    print("  python cellpose_pipeline.py --format tiff --sample -i <input> -o <output>")
    print("=" * 70)
    sys.exit(1)


if __name__ == "__main__":
    main()
