#!/usr/bin/env python3
"""
CRISPRi Library Imaging Pipeline
=================================
Segments microscopy images with Cellpose and saves stacked TIFFs.

Supported input formats (--format):
    czi   Individual CZI files grouped by condition folder.
          Each CZI is one FOV; all CZIs in a folder are stacked into
          one output hyperstack per condition.
    tiff  Multi-FOV TIFF hyperstacks (ImageJ ZCYX format).
          Each file is processed independently.
    auto  (default) Detects format from file extensions in the input
          directory. Errors if both CZI and TIFF are present.

Output channels:
    All original channels (in order) + BinaryMask as final channel.

    For 3-channel CZI (mCherry, eGFP, Phase):
        C0 = mCherry-DnaN, C1 = eGFP-ImuB, C2 = Phase, C3 = BinaryMask

    For 2-channel CZI (Phase, Fluorescence):
        C0 = Phase, C1 = Fluorescence, C2 = BinaryMask

Output format:
    One ImageJ-compatible hyperstack TIFF per condition/file.
    Dimensions: (N_FOV, C, Y, X) — each FOV is a "slice" in the stack.

Requirements:
    conda create --name cellpose python=3.10
    conda activate cellpose
    pip install 'cellpose[gui]' czifile tifffile numpy scikit-image scipy

    # If you have a GPU (recommended for batch processing):
    # pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

Usage (CZI):
    python cellpose_pipeline.py -i /path/to/experiment -o /path/to/output

    Supports arbitrary folder nesting. The script recursively finds all
    directories containing CZI files and creates one output TIFF per directory.

    Example CZI input:
        experiment/
        ├── WT_MMC/
        │   ├── image1.czi
        ├── dnaN_CRISPRi/
        │   ├── image1.czi

Usage (TIFF):
    python cellpose_pipeline.py -i /path/to/tiffs -o /path/to/output --format tiff

    Example TIFF input:
        input/
        ├── Replica 1/
        │   ├── Plate 1/
        │   │   ├── experiment_001.tif   (N_FOV, C, Y, X)

    Output (flat directory):
        output/
        ├── Replica_1__Plate_1__experiment_001.tif

Classification presets:
    python cellpose_pipeline.py -i ... -o ... --classify-preset mtb
    python cellpose_pipeline.py -i ... -o ... --classify /path/to/model.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

# Named presets for species-specific cell quality classification models.
# Use --classify-preset <name> to select one.
CLASSIFY_PRESETS = {
    "mtb": Path(__file__).parent / "models" / "best_model.pth",
    # "msm": Path(__file__).parent / "models" / "msm_best_model.pth",  # TODO: add when trained
}


def read_czi(filepath):
    """
    Read a CZI file and return all channels as a numpy array.
    
    Returns:
        img: numpy array (C, Y, X) — all channels
    """
    import czifile
    
    img = czifile.imread(str(filepath))
    
    # CZI files often have extra singleton dimensions
    # Typical shape: (1, 1, C, 1, Y, X, 1) or similar
    # Squeeze out all singleton dimensions except channels
    img = np.squeeze(img)
    
    # After squeeze, should be (C, Y, X) for a multi-channel single-Z image
    # or (Y, X) if single-channel
    if img.ndim == 2:
        # Single channel — add a channel dimension
        img = img[np.newaxis, :, :]
    
    if img.ndim != 3:
        raise ValueError(
            f"Unexpected shape {img.shape} after squeezing {filepath}. "
            f"Expected 3D array (C, Y, X). Raw shape was {czifile.imread(str(filepath)).shape}. "
            f"You may need to adjust the channel extraction logic."
        )
    
    return img


def segment_phase(phase_img, model, diameter=None, model_type="cpsam",
                  flow_threshold=0.4, cellprob_threshold=0.0, min_size=15):
    """
    Run Cellpose on a phase contrast image.

    Args:
        phase_img: 2D numpy array (Y, X) — phase contrast image
        model: CellposeModel instance
        diameter: Expected cell diameter in pixels (None = auto-estimate)
        model_type: Cellpose model type string (affects input format)
        flow_threshold: max allowed mask quality score; raise to keep more cells
        cellprob_threshold: probability cutoff; lower (negative) to keep more cells
        min_size: smallest mask in pixels

    Returns:
        masks: 2D integer array (Y, X) — 0 = background, 1,2,3... = cell labels
    """
    # cpsam expects 3-channel RGB input; CNN models work with grayscale
    if model_type == "cpsam":
        img_input = np.stack([phase_img, phase_img, phase_img], axis=-1)
    else:
        img_input = phase_img

    masks, flows, styles = model.eval(
        img_input,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )

    return masks


def add_cell_boundaries(masks, boundary_width=1):
    """
    Add background (0) boundaries between touching cells in a labelled mask.
    
    For each pixel, checks whether any of its neighbours have a different
    non-zero label. If so, that pixel is set to 0 (background). This creates
    a 1-pixel gap between touching cells that allows MicrobeJ (and similar
    tools) to identify them as separate objects.
    
    Args:
        masks: 2D integer array (Y, X) — labelled mask from Cellpose
        boundary_width: Number of erosion iterations (default 1)
    
    Returns:
        masks_bounded: 2D integer array with boundaries set to 0
    """
    from scipy import ndimage
    
    masks_bounded = masks.copy()
    
    for _ in range(boundary_width):
        # Find pixels where any neighbour has a different label
        # Check all 4 cardinal neighbours
        padded = np.pad(masks_bounded, 1, mode='constant', constant_values=0)
        
        center = padded[1:-1, 1:-1]
        up     = padded[:-2,  1:-1]
        down   = padded[2:,   1:-1]
        left   = padded[1:-1, :-2]
        right  = padded[1:-1, 2:]
        
        # A pixel is on a boundary if it's non-zero AND any neighbour
        # has a different non-zero label
        is_nonzero = center > 0
        has_different_neighbour = (
            ((up    != center) & (up    > 0)) |
            ((down  != center) & (down  > 0)) |
            ((left  != center) & (left  > 0)) |
            ((right != center) & (right > 0))
        )
        
        boundary_pixels = is_nonzero & has_different_neighbour
        masks_bounded[boundary_pixels] = 0
    
    return masks_bounded


def segment_single_fov(channels, phase_channel, model, diameter=None,
                       classify_opts=None, model_type="cpsam",
                       flow_threshold=0.4, cellprob_threshold=0.0, min_size=15):
    """
    Segment one FOV: run Cellpose, add boundaries, optionally classify,
    and return a composite array with a binary mask appended.

    Args:
        channels: numpy array (C, Y, X) — all channels for this FOV
        phase_channel: 0-indexed channel number for phase contrast
        model: CellposeModel instance
        diameter: Expected cell diameter in pixels (None = auto-estimate)
        classify_opts: dict with cell quality filtering options, or None
        model_type: Cellpose model type string

    Returns:
        composite: numpy array (C+1, Y, X) — original channels + binary mask
        n_cells: number of cells after all filtering
    """
    phase = channels[phase_channel]

    masks = segment_phase(
        phase, model, diameter=diameter, model_type=model_type,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )
    n_cells = masks.max()
    print(f"{n_cells} cells segmented", end=" ... ")

    masks = add_cell_boundaries(masks, boundary_width=1)

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
    mask_channel = masks_binary[np.newaxis, :, :]  # (1, Y, X)
    composite = np.concatenate([channels_16, mask_channel], axis=0)

    return composite, n_cells


def process_condition(condition_dir, model, phase_channel, diameter=None,
                      classify_opts=None, model_type="cpsam"):
    """
    Process all CZI files in a condition folder.

    Args:
        condition_dir: Path to folder containing CZI files
        model: CellposeModel instance
        phase_channel: 0-indexed channel number for phase contrast
        diameter: Expected cell diameter in pixels (None = auto-estimate)
        classify_opts: dict with cell quality filtering options, or None to skip.
            Keys: model_path, pixels_per_um, keep_classes, confidence_threshold,
                  use_rules, device

    Returns:
        stacked: numpy array (N_FOV, C=N+1, Y, X) — all FOVs stacked (original channels + mask)
        filenames: list of processed filenames (for logging)
        total_cells: total number of cells segmented across all FOVs
    """
    czi_files = sorted(condition_dir.glob("*.czi"))

    if not czi_files:
        print(f"  WARNING: No CZI files found in {condition_dir}")
        return None, [], 0

    fov_arrays = []
    filenames = []
    cell_counts = []

    for i, czi_path in enumerate(czi_files):
        print(f"  [{i+1}/{len(czi_files)}] {czi_path.name}", end=" ... ")

        try:
            channels = read_czi(czi_path)
        except Exception as e:
            print(f"SKIPPED (read error: {e})")
            continue

        n_channels = channels.shape[0]

        if phase_channel >= n_channels:
            print(f"SKIPPED (phase channel {phase_channel} but only {n_channels} channels)")
            continue

        composite, n_cells = segment_single_fov(
            channels, phase_channel, model, diameter, classify_opts, model_type,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
        )

        fov_arrays.append(composite)
        filenames.append(czi_path.name)
        cell_counts.append(n_cells)
        print("done")

    if not fov_arrays:
        return None, [], 0

    # Check that all FOVs have the same spatial dimensions
    shapes = [arr.shape for arr in fov_arrays]
    ref_shape = shapes[0]

    mismatched = [f for f, s in zip(filenames, shapes) if s != ref_shape]
    if mismatched:
        print(f"  WARNING: FOVs have inconsistent dimensions.")
        print(f"  Reference shape: {ref_shape}")
        for f, s in zip(filenames, shapes):
            if s != ref_shape:
                print(f"    {f}: {s} — will be skipped")

        # Keep only matching FOVs
        keep = [s == ref_shape for s in shapes]
        fov_arrays = [a for a, k in zip(fov_arrays, keep) if k]
        filenames = [f for f, k in zip(filenames, keep) if k]
        cell_counts = [c for c, k in zip(cell_counts, keep) if k]

    # Stack all FOVs: (N_FOV, C=4, Y, X)
    stacked = np.stack(fov_arrays, axis=0)
    total_cells = sum(cell_counts)

    return stacked, filenames, total_cells


def process_tiff_unit(tiff_path, model, phase_channel, diameter=None,
                      classify_opts=None, model_type="cpsam", sample=False,
                      flow_threshold=0.4, cellprob_threshold=0.0, min_size=15):
    """
    Segment all (or a sampled) FOVs in a multi-FOV TIFF hyperstack.

    Args:
        tiff_path: Path to input TIFF file (ImageJ ZCYX hyperstack)
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
    from .label_cells import load_hyperstack

    data, metadata = load_hyperstack(tiff_path)  # (N_FOV, C, Y, X)
    n_fov = data.shape[0]

    if sample:
        indices = [int(np.random.choice(n_fov, 1))]
    else:
        indices = list(range(n_fov))

    fov_arrays = []
    fov_names = []
    cell_counts = []

    for fov_idx in indices:
        print(f"    FOV {fov_idx+1}/{n_fov}", end=" ... ")

        channels = data[fov_idx]  # (C, Y, X)
        n_channels = channels.shape[0]

        if phase_channel >= n_channels:
            print(f"SKIPPED (phase channel {phase_channel} but only {n_channels} channels)")
            continue

        composite, n_cells = segment_single_fov(
            channels, phase_channel, model, diameter, classify_opts, model_type,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
        )

        fov_arrays.append(composite)
        fov_names.append(f"fov_{fov_idx:03d}")
        cell_counts.append(n_cells)
        print("done")

    if not fov_arrays:
        return None, [], 0

    stacked = np.stack(fov_arrays, axis=0)
    return stacked, fov_names, sum(cell_counts)


def save_hyperstack(stacked, output_path, condition_name, filenames, channel_labels, pixels_per_um=13.8767):
    """
    Save a 4D array as an ImageJ-compatible hyperstack TIFF.
    
    The TIFF is saved with ImageJ metadata so that when opened in Fiji:
    - Each FOV appears as a separate "slice" (Z-plane)
    - Channels are labelled and can be toggled independently
    - The mask channel (last) can be used directly in MicrobeJ
    
    Args:
        stacked: numpy array (N_FOV, C, Y, X)
        output_path: Path to save the TIFF
        condition_name: Name of the condition (for metadata)
        filenames: List of source filenames (stored in description)
        channel_labels: List of channel name strings (including BinaryMask as last)
        pixels_per_um: Pixel scale
    """
    n_fov, n_channels, h, w = stacked.shape
    
    # Build description with source file mapping
    channel_str = ", ".join(channel_labels)
    description_lines = [
        f"Condition: {condition_name}",
        f"Channels: {channel_str}",
        f"FOVs: {n_fov}",
        "",
        "FOV index -> source file:",
    ]
    for i, fname in enumerate(filenames):
        description_lines.append(f"  Z={i}: {fname}")
    
    description = "\n".join(description_lines)
    
    imagej_metadata = {
        'axes': 'ZCYX',
        'channels': n_channels,
        'slices': n_fov,
        'hyperstack': True,
        'mode': 'composite',
        'Labels': channel_labels,
        'unit': 'um',
        'spacing': 1.0 / pixels_per_um if pixels_per_um else 1.0,  # µm/px for downstream readers
    }
    
    # Resolution in pixels per centimetre (TIFF standard unit)
    pixels_per_cm = pixels_per_um * 10000
    
    tifffile.imwrite(
        str(output_path),
        stacked,
        imagej=True,
        metadata=imagej_metadata,
        description=description,
        resolution=(pixels_per_cm, pixels_per_cm),
        resolutionunit=3,  # 3 = centimetre
    )


def main():
    parser = argparse.ArgumentParser(
        description="Segment microscopy images with Cellpose and create stacked TIFFs. "
                    "Supports CZI files (individual FOVs grouped by condition folder) "
                    "and TIFF hyperstacks (multi-FOV files processed independently)."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to experiment folder containing image files (CZI or TIFF)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output folder for stacked TIFFs"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["auto", "czi", "tiff"],
        default="auto",
        help="Input format (default: auto). "
             "'czi': individual CZI files grouped by condition folder. "
             "'tiff': multi-FOV TIFF hyperstacks processed per file. "
             "'auto': detect from file extensions found in input directory."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        default=False,
        help="TIFF mode only: pick 1 random FOV per file (for training data diversity)"
    )
    parser.add_argument(
        "--diameter", "-d",
        type=float,
        default=None,
        help="Expected cell diameter in pixels (default: auto-estimate). "
             "For Msm at 100x/1.4NA with 11µm pixel camera, try ~5 for width."
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU for Cellpose-SAM (recommended for batch processing)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cpsam",
        help="Cellpose model type (default: cpsam). Options: cpsam (Cellpose-SAM), "
             "cyto3 (fast CNN), cyto2. "
             "cpsam is most accurate but benefits greatly from GPU acceleration."
    )
    parser.add_argument(
        "--flow-threshold",
        type=float,
        default=0.4,
        help="Cellpose flow threshold (lower = more aggressive splitting of touching cells)"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=15,
        help="Minimum mask size in pixels (filters debris)"
    )
    parser.add_argument(
        "--pixels-per-um",
        type=float,
        default=13.8767,
        help="Pixel scale in pixels per micron (default: 13.8767 for 100x/1.4NA + Prime 95B)"
    )
    parser.add_argument(
        "--phase-channel",
        type=int,
        default=None,
        help="0-indexed channel number for phase contrast. "
             "Default: last channel for CZI, channel 0 for TIFF. "
             "For 3-channel CZI (mCherry, eGFP, Phase): use 2. "
             "For 2-channel CZI (Phase, Fluorescence): use 0."
    )
    parser.add_argument(
        "--channel-names",
        type=str,
        default=None,
        help="Comma-separated channel names (excluding mask, which is added automatically). "
             "E.g. 'mCherry-DnaN,eGFP-ImuB,Phase' for 3-channel or 'Phase,GFP' for 2-channel. "
             "Default: C1,C2,...,CN"
    )
    parser.add_argument(
        "--classify",
        type=str,
        default=None,
        metavar="MODEL_PATH",
        help="Path to trained cell quality classifier (.pth). "
             "Enables CNN-based filtering of bad cells from masks. "
             "If omitted, only rule-based filtering is available via --filter-rules."
    )
    preset_names = ", ".join(CLASSIFY_PRESETS.keys())
    parser.add_argument(
        "--classify-preset",
        type=str,
        default=None,
        choices=list(CLASSIFY_PRESETS.keys()),
        help=f"Use a named classification model preset ({preset_names}). "
             "Takes precedence over --classify if both are given."
    )
    parser.add_argument(
        "--filter-rules",
        action="store_true",
        default=False,
        help="Apply rule-based cell filtering (edge cells, debris) even without a CNN model. "
             "Automatically enabled when --classify is used."
    )
    parser.add_argument(
        "--keep-classes",
        type=str,
        default="good",
        help="Comma-separated class names to keep when using --classify. "
             "Default: 'good'. Options: good, out_of_focus, clumped, edge_cell, debris"
    )
    parser.add_argument(
        "--classify-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence to trust CNN prediction (default: 0.5)"
    )
    
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve input format ─────────────────────────────────────────
    fmt = args.format
    if fmt == "auto":
        has_czi = any(input_dir.rglob("*.czi"))
        has_tiff = any(input_dir.rglob("*.tif")) or any(input_dir.rglob("*.tiff"))
        if has_czi and has_tiff:
            print("ERROR: Found both CZI and TIFF files in input directory.")
            print("  Please specify --format czi or --format tiff to disambiguate.")
            sys.exit(1)
        elif has_czi:
            fmt = "czi"
        elif has_tiff:
            fmt = "tiff"
        else:
            print(f"ERROR: No CZI or TIFF files found in {input_dir}")
            sys.exit(1)
        print(f"Auto-detected format: {fmt}")

    if args.sample and fmt != "tiff":
        print("ERROR: --sample is only valid with --format tiff")
        sys.exit(1)

    # ── Discover input files ─────────────────────────────────────────
    if fmt == "czi":
        condition_dirs = sorted(set(
            czi.parent for czi in input_dir.rglob("*.czi")
        ))
        if not condition_dirs:
            print(f"ERROR: No CZI files found anywhere in {input_dir}")
            sys.exit(1)
        print(f"Found {len(condition_dirs)} conditions:")
        for d in condition_dirs:
            rel_path = d.relative_to(input_dir)
            n_files = len(list(d.glob("*.czi")))
            print(f"  {rel_path}: {n_files} CZI files")
    else:  # tiff
        tiff_files = sorted(
            list(input_dir.rglob("*.tif")) + list(input_dir.rglob("*.tiff"))
        )
        if not tiff_files:
            print(f"ERROR: No TIFF files found in {input_dir}")
            sys.exit(1)
        print(f"Found {len(tiff_files)} TIFF files")
        if args.sample:
            print("SAMPLE MODE: picking 1 random FOV per TIFF")
    print()

    # ── GPU detection and diagnostics ────────────────────────────────
    import torch
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if args.gpu:
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU: CUDA ({gpu_name})")
        elif has_mps:
            print(f"GPU: Apple Silicon (MPS)")
        else:
            print(f"WARNING: --gpu was passed but no GPU detected (CUDA: {has_cuda}, MPS: {has_mps})")
            print(f"  Cellpose will fall back to CPU — this will be slow for cpsam.")
            print(f"  Ensure PyTorch is installed with GPU support.")
    else:
        if has_cuda or has_mps:
            gpu_type = "CUDA" if has_cuda else "MPS"
            print(f"GPU: not enabled (but {gpu_type} is available — pass --gpu to use it)")
        else:
            print(f"GPU: none detected, using CPU")
    print()

    # Initialise Cellpose model
    print(f"Loading Cellpose model ({args.model})...")
    from cellpose import models
    model = models.CellposeModel(model_type=args.model, gpu=args.gpu)
    print(f"  Model loaded (type: {args.model}, GPU: {args.gpu})")
    print()

    # ── Auto-detect channel count from first input file ──────────────
    if fmt == "czi":
        first_czi = next(input_dir.rglob("*.czi"))
        first_img = read_czi(first_czi)
        n_input_channels = first_img.shape[0]
        print(f"Detected {n_input_channels} channels in CZI files")
    else:
        from .label_cells import load_hyperstack
        first_data, _ = load_hyperstack(tiff_files[0])
        n_input_channels = first_data.shape[1]
        print(f"Detected {n_input_channels} channels in TIFF files")

    # Determine phase channel (default differs by format)
    phase_channel = args.phase_channel
    if phase_channel is None:
        if fmt == "czi":
            phase_channel = n_input_channels - 1  # CZI: last channel
        else:
            phase_channel = 0  # TIFF: first channel
        print(f"  Phase channel: {phase_channel} (auto-detected for {fmt} format)")
    else:
        print(f"  Phase channel: {phase_channel} (user-specified)")

    # Determine channel labels
    if args.channel_names:
        channel_labels = [s.strip() for s in args.channel_names.split(",")]
        if len(channel_labels) != n_input_channels:
            print(f"ERROR: --channel-names has {len(channel_labels)} names "
                  f"but input has {n_input_channels} channels")
            sys.exit(1)
    else:
        channel_labels = [f"C{i+1}" for i in range(n_input_channels)]

    # Append mask label
    channel_labels_with_mask = channel_labels + ["BinaryMask"]
    mask_channel_num = n_input_channels + 1  # 1-indexed for MicrobeJ

    print(f"  Output channels: {', '.join(channel_labels_with_mask)}")
    print(f"  MicrobeJ mask channel: C{mask_channel_num}")
    print()

    # ── Build cell quality classification options ─────────────────────
    # Resolve --classify-preset to a model path (takes precedence over --classify)
    classify_model_path = args.classify
    if args.classify_preset:
        classify_model_path = str(CLASSIFY_PRESETS[args.classify_preset])
        if args.classify:
            print(f"NOTE: --classify-preset '{args.classify_preset}' overrides --classify")

    classify_opts = None
    if classify_model_path or args.filter_rules:
        keep_classes = tuple(s.strip() for s in args.keep_classes.split(","))
        classify_opts = {
            "model_path": classify_model_path,  # None if only using rules
            "pixels_per_um": args.pixels_per_um,
            "keep_classes": keep_classes,
            "confidence_threshold": args.classify_threshold,
            "use_rules": True,  # Always apply rules when filtering is enabled
            "device": "mps" if has_mps else ("cuda" if has_cuda else "cpu"),
        }
        if classify_model_path:
            print(f"Cell quality filter: CNN model + rules")
            print(f"  Model: {classify_model_path}")
            print(f"  Keep classes: {keep_classes}")
            print(f"  Confidence threshold: {args.classify_threshold}")
        else:
            print(f"Cell quality filter: rule-based only (edge cells, debris)")
        print()

    # ── Process inputs ───────────────────────────────────────────────
    summary = []

    if fmt == "czi":
        for condition_dir in condition_dirs:
            # Build output name from the relative path
            # e.g. ATC Strains/cydA/menH -> ATC_Strains__cydA__menH.tif
            rel_path = condition_dir.relative_to(input_dir)
            condition_name = "__".join(rel_path.parts).replace(" ", "_")

            print(f"Processing: {rel_path}")

            stacked, filenames, total_cells = process_condition(
                condition_dir, model, phase_channel, diameter=args.diameter,
                classify_opts=classify_opts, model_type=args.model
            )

            if stacked is None:
                print(f"  No valid FOVs — skipping\n")
                continue

            output_path = output_dir / f"{condition_name}.tif"
            save_hyperstack(stacked, output_path, condition_name,
                            filenames, channel_labels_with_mask, args.pixels_per_um)

            n_fov = stacked.shape[0]
            print(f"  Saved: {output_path}")
            print(f"  {n_fov} FOVs, {total_cells} total cells segmented\n")

            summary.append({
                'condition': condition_name,
                'n_fov': n_fov,
                'total_cells': total_cells,
            })

    else:  # tiff
        for tiff_path in tiff_files:
            rel_path = tiff_path.relative_to(input_dir)
            parts = list(rel_path.parts)
            parts[-1] = Path(parts[-1]).stem
            condition_name = "__".join(parts).replace(" ", "_")

            print(f"Processing: {rel_path}")

            stacked, fov_names, total_cells = process_tiff_unit(
                tiff_path, model, phase_channel, diameter=args.diameter,
                classify_opts=classify_opts, model_type=args.model,
                sample=args.sample,
            )

            if stacked is None:
                print(f"  No valid FOVs — skipping\n")
                continue

            output_path = output_dir / f"{condition_name}.tif"
            save_hyperstack(stacked, output_path, condition_name,
                            fov_names, channel_labels_with_mask, args.pixels_per_um)

            n_fov = stacked.shape[0]
            print(f"  Saved: {output_path}")
            print(f"  {n_fov} FOVs, {total_cells} total cells segmented\n")

            summary.append({
                'condition': condition_name,
                'n_fov': n_fov,
                'total_cells': total_cells,
            })

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Condition':<50} {'FOVs':>6} {'Cells':>8}")
    print("-" * 80)
    for s in summary:
        print(f"{s['condition']:<50} {s['n_fov']:>6} {s['total_cells']:>8}")
    print("-" * 80)
    print(f"Output directory: {output_dir}")
    print()
    print("To open in Fiji:")
    print("  File > Open > select .tif")
    print("  Image > Hyperstacks > Re-order if needed")
    channel_desc = ", ".join(f"C{i+1}={name}" for i, name in enumerate(channel_labels_with_mask))
    print(f"  Channels: {channel_desc}")
    print()
    print("To use mask in MicrobeJ:")
    print(f"  In Bacteria tab, set Channel to {mask_channel_num} (BinaryMask)")
    print("  Set thresholding to 'Default' (any method works on binary)")
    print("  MicrobeJ will detect each white object as a cell")
    if fmt == "tiff" and args.sample:
        print()
        print("Next steps (training workflow):")
        print(f"  1. Label:  python label_cells.py -i {output_dir} -s <labeled_data_path>")
        print(f"  2. Train:  python train_classifier.py --data <labeled_data_path> --output <model_dir> --gpu")


if __name__ == "__main__":
    main()
