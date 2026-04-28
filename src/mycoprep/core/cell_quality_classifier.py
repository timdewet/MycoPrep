#!/usr/bin/env python3
"""
Cell Quality Classifier
=======================
CNN-based classification of segmented cells into quality categories.
Used as a post-segmentation filter to remove bad cells from masks.

Categories:
    0 = good         — well-segmented, in-focus single cell
    1 = out_of_focus  — blurry cell (low edge contrast in phase)
    2 = clumped       — multiple overlapping cells segmented as one
    3 = edge_cell     — cell touching the FOV boundary (incomplete)
    4 = debris        — non-cell object that passed min_size filter

Usage in pipeline:
    from cell_quality_classifier import classify_and_filter_mask
    filtered_mask = classify_and_filter_mask(
        labeled_mask, channels, phase_channel=2, model_path="model.pth"
    )

Requirements:
    pip install torch torchvision numpy scikit-image scipy
"""

import numpy as np
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

# Binary classification: good (0) vs bad (1)
CLASS_NAMES = ["good", "bad"]
NUM_CLASSES = len(CLASS_NAMES)
CROP_SIZE = 64  # Pixels. Resized crop fed to CNN.

# Legacy 5-class names, used to remap old labeled datasets to binary
LEGACY_CLASS_NAMES = ["good", "out_of_focus", "clumped", "edge_cell", "debris"]

# ── Area channel helpers ──────────────────────────────────────────────────────
#
# The area channel encodes absolute cell size as a log-normalised value in
# [0, 1], computed from the true pixel area in the original (un-resized) mask.
# This is shape-independent: a long rod and a round coccus with the same pixel
# count get exactly the same area value.
#
# Normalisation range:  log(MIN_AREA_PX) … log(MAX_AREA_PX)
#   At 13.87 px/µm → 192.5 px²/µm²
#     MIN_AREA_PX = 10   (~0.05 µm² — tiny debris)
#     MAX_AREA_PX = 10000 (~52 µm² — large clump)

_AREA_LOG_MIN = np.log(10.0)
_AREA_LOG_MAX = np.log(10000.0)
_AREA_LOG_RANGE = _AREA_LOG_MAX - _AREA_LOG_MIN


def _normalise_area(area_px):
    """Map raw pixel area → [0, 1] via log scale."""
    return float(np.clip(
        (np.log(max(area_px, 1)) - _AREA_LOG_MIN) / _AREA_LOG_RANGE,
        0.0, 1.0,
    ))


def _append_area_channel_single(crop, true_area_px=None):
    """
    Append a normalised area channel to a single crop.

    Args:
        crop:          (C, H, W) float32 — last channel is the binary mask
        true_area_px:  pixel count of the cell in the original (un-resized) mask.
                       If None, approximates from the resized mask (less accurate).

    Returns:
        (C+1, H, W) float32 with the area channel appended
    """
    if true_area_px is not None:
        area_val = _normalise_area(true_area_px)
    else:
        # Fallback: use resized mask pixel count (shape-dependent approximation)
        area_val = _normalise_area(float(crop[-1].sum()))

    mask = crop[-1:]                            # (1, H, W) — just for shape
    area_ch = np.full_like(mask, area_val)      # (1, H, W)
    return np.concatenate([crop, area_ch], axis=0)


def _append_area_channel_batch(crops, true_areas_px=None):
    """
    Append a normalised area channel to a batch of crops.

    Args:
        crops:          (N, C, H, W) float32 — last channel is the binary mask
        true_areas_px:  (N,) array of pixel counts from original masks.
                        If None, approximates from resized masks.

    Returns:
        (N, C+1, H, W) float32
    """
    n = crops.shape[0]
    mask = crops[:, -1:, :, :]                  # (N, 1, H, W)

    if true_areas_px is not None:
        vals = np.array([_normalise_area(a) for a in true_areas_px],
                        dtype=np.float32).reshape(n, 1, 1, 1)
    else:
        # Fallback: resized mask pixel count
        raw = mask.sum(axis=(2, 3), keepdims=True)  # (N, 1, 1, 1)
        vals = np.clip(
            (np.log(np.maximum(raw, 1)) - _AREA_LOG_MIN) / _AREA_LOG_RANGE,
            0.0, 1.0,
        ).astype(np.float32)

    area_ch = np.broadcast_to(vals, mask.shape).copy()  # (N, 1, H, W)
    return np.concatenate([crops, area_ch], axis=1)


# Default thresholds for rule-based pre-filtering
EDGE_MARGIN = 2        # Pixels from border to count as edge cell
MIN_AREA_UM2 = 0.3     # Minimum cell area in µm² (debris filter)
MAX_AREA_UM2 = 25.0    # Maximum single-cell area in µm² (clump indicator)


# ── Cell crop extraction ──────────────────────────────────────────────────────

def extract_cell_crop(image_channels, labeled_mask, cell_label, pad=10,
                      phase_channel=None):
    """
    Extract a square crop of a single cell from multi-channel image data.

    The crop is taken from the bounding box of the cell (from the labeled mask)
    plus padding, padded to square, then resized to CROP_SIZE × CROP_SIZE.

    If phase_channel is specified, only that channel is used (+ mask), giving a
    2-channel crop that is independent of the number of fluorescence channels.
    This makes the model generalisable across different acquisition setups.

    Args:
        image_channels: (C, Y, X) numpy array — all imaging channels
        labeled_mask:   (Y, X) integer array — 0=background, N=cell label
        cell_label:     integer — which cell to extract
        pad:            pixels of padding around bounding box
        phase_channel:  if set, extract only this channel (+ mask → 2-ch crop).
                        If None, extract all channels (+ mask → C+1-ch crop).

    Returns:
        crop: (2, CROP_SIZE, CROP_SIZE) or (C+1, CROP_SIZE, CROP_SIZE) float32
              Intensities are individually normalised to [0, 1] per channel.
        bbox: (y_min, y_max, x_min, x_max) of the padded bounding box
    """
    from skimage.transform import resize

    h, w = labeled_mask.shape
    cell_pixels = labeled_mask == cell_label

    if not cell_pixels.any():
        return None, None

    ys, xs = np.where(cell_pixels)
    y_min = max(ys.min() - pad, 0)
    y_max = min(ys.max() + pad + 1, h)
    x_min = max(xs.min() - pad, 0)
    x_max = min(xs.max() + pad + 1, w)

    # Crop imaging channel(s)
    if phase_channel is not None:
        # Phase-only mode: single imaging channel → generalisable model
        channel_crops = image_channels[phase_channel:phase_channel + 1,
                                       y_min:y_max, x_min:x_max].astype(np.float32)
    else:
        # All-channels mode
        channel_crops = image_channels[:, y_min:y_max, x_min:x_max].astype(np.float32)

    # Crop mask (binary for this cell only)
    mask_crop = cell_pixels[y_min:y_max, x_min:x_max].astype(np.float32)
    mask_crop = mask_crop[np.newaxis, :, :]  # (1, h_crop, w_crop)

    combined = np.concatenate([channel_crops, mask_crop], axis=0)  # (N_ch+1, h, w)

    # Pad to square
    _, ch, cw = combined.shape
    side = max(ch, cw)
    padded = np.zeros((combined.shape[0], side, side), dtype=np.float32)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    padded[:, y_off:y_off + ch, x_off:x_off + cw] = combined

    # Resize to CROP_SIZE × CROP_SIZE
    resized = np.zeros((padded.shape[0], CROP_SIZE, CROP_SIZE), dtype=np.float32)
    for c in range(padded.shape[0]):
        resized[c] = resize(padded[c], (CROP_SIZE, CROP_SIZE),
                            order=1, preserve_range=True, anti_aliasing=True)

    # Normalise each channel independently to [0, 1]
    for c in range(resized.shape[0] - 1):  # Skip mask channel
        ch_data = resized[c]
        cmin, cmax = ch_data.min(), ch_data.max()
        if cmax > cmin:
            resized[c] = (ch_data - cmin) / (cmax - cmin)
        else:
            resized[c] = 0.0

    # Mask channel: already 0/1, just re-threshold after resize
    resized[-1] = (resized[-1] > 0.5).astype(np.float32)

    return resized, (y_min, y_max, x_min, x_max)


def extract_all_crops(image_channels, labeled_mask, pad=10, phase_channel=None):
    """
    Extract crops for all cells in a labeled mask.

    Args:
        image_channels: (C, Y, X) array
        labeled_mask:   (Y, X) integer array
        pad:            padding around bounding box
        phase_channel:  if set, extract only this channel (+ mask → 2-ch crop)

    Returns:
        crops:  dict mapping cell_label -> crop array
        bboxes: dict mapping cell_label -> (y_min, y_max, x_min, x_max)
    """
    cell_labels = np.unique(labeled_mask)
    cell_labels = cell_labels[cell_labels > 0]

    crops = {}
    bboxes = {}

    for label in cell_labels:
        crop, bbox = extract_cell_crop(image_channels, labeled_mask, label, pad,
                                       phase_channel=phase_channel)
        if crop is not None:
            crops[label] = crop
            bboxes[label] = bbox

    return crops, bboxes


# ── Rule-based pre-filters ────────────────────────────────────────────────────

def detect_edge_cells(labeled_mask, margin=EDGE_MARGIN):
    """
    Find cells whose bounding box touches the image border.

    Returns:
        set of cell labels that are edge cells
    """
    h, w = labeled_mask.shape
    edge_labels = set()

    for label in np.unique(labeled_mask):
        if label == 0:
            continue
        ys, xs = np.where(labeled_mask == label)
        if (ys.min() < margin or ys.max() >= h - margin or
                xs.min() < margin or xs.max() >= w - margin):
            edge_labels.add(label)

    return edge_labels


def detect_debris_by_area(labeled_mask, pixels_per_um=13.8767, min_area_um2=MIN_AREA_UM2):
    """
    Find cells below a minimum area threshold (likely debris).

    Returns:
        set of cell labels classified as debris
    """
    px_area_per_um2 = pixels_per_um ** 2
    min_area_px = min_area_um2 * px_area_per_um2
    debris_labels = set()

    for label in np.unique(labeled_mask):
        if label == 0:
            continue
        area = np.sum(labeled_mask == label)
        if area < min_area_px:
            debris_labels.add(label)

    return debris_labels


def detect_large_clumps(labeled_mask, pixels_per_um=13.8767, max_area_um2=MAX_AREA_UM2):
    """
    Flag cells above a maximum single-cell area as potential clumps.
    These are candidates for CNN review, not automatic rejection.

    Returns:
        set of cell labels that are suspiciously large
    """
    px_area_per_um2 = pixels_per_um ** 2
    max_area_px = max_area_um2 * px_area_per_um2
    large_labels = set()

    for label in np.unique(labeled_mask):
        if label == 0:
            continue
        area = np.sum(labeled_mask == label)
        if area > max_area_px:
            large_labels.add(label)

    return large_labels


# ── CNN model ─────────────────────────────────────────────────────────────────

def _build_model(in_channels, num_classes=NUM_CLASSES):
    """
    Build a lightweight CNN for cell quality classification.

    Architecture:
        4 conv blocks (Conv2d → BatchNorm → ReLU → MaxPool2×2)
        Global Average Pooling → Dropout → FC → num_classes

    With CROP_SIZE=64, after 4 max-pools: 64→32→16→8→4, then GAP → 256-d.
    Total ~400K parameters — trainable on a few hundred labelled cells.

    Args:
        in_channels: number of input channels (imaging channels + 1 mask channel)
        num_classes: number of output classes

    Returns:
        PyTorch nn.Module
    """
    import torch
    import torch.nn as nn

    class ConvBlock(nn.Module):
        def __init__(self, c_in, c_out, kernel=3, pad=1):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel, padding=pad, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        def forward(self, x):
            return self.block(x)

    class CellQualityCNN(nn.Module):
        def __init__(self, c_in, n_cls):
            super().__init__()
            self.features = nn.Sequential(
                ConvBlock(c_in, 32),
                ConvBlock(32, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(256, n_cls),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return CellQualityCNN(in_channels, num_classes)


# ── Classifier wrapper ────────────────────────────────────────────────────────

class CellQualityClassifier:
    """
    Wrapper that loads a trained CNN and classifies cell crops.

    Usage:
        clf = CellQualityClassifier("model.pth", in_channels=4)
        predictions = clf.predict_batch(crops_array)
    """

    def __init__(self, model_path, in_channels, device=None):
        import torch

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model = _build_model(in_channels, NUM_CLASSES)

        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict_single(self, crop):
        """
        Classify a single cell crop.

        Args:
            crop: (C, CROP_SIZE, CROP_SIZE) float32 numpy array

        Returns:
            class_idx: int
            class_name: str
            probabilities: (NUM_CLASSES,) array
        """
        import torch

        x = torch.from_numpy(crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        class_idx = int(probs.argmax())
        return class_idx, CLASS_NAMES[class_idx], probs

    def predict_batch(self, crops, batch_size=64):
        """
        Classify a batch of cell crops.

        Args:
            crops: (N, C, CROP_SIZE, CROP_SIZE) float32 numpy array
            batch_size: processing batch size

        Returns:
            class_indices: (N,) int array
            probabilities: (N, NUM_CLASSES) array
        """
        import torch

        all_probs = []

        for i in range(0, len(crops), batch_size):
            batch = torch.from_numpy(crops[i:i + batch_size]).to(self.device)
            with torch.no_grad():
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

        all_probs = np.concatenate(all_probs, axis=0)
        class_indices = all_probs.argmax(axis=1)

        return class_indices, all_probs


# ── Main filtering API ────────────────────────────────────────────────────────

def classify_and_filter_mask(
    labeled_mask,
    image_channels,
    phase_channel,
    model_path=None,
    pixels_per_um=13.8767,
    keep_classes=("good",),
    confidence_threshold=0.5,
    use_rules=True,
    device=None,
    verbose=True,
):
    """
    Classify each cell in a labeled mask and return a filtered version
    containing only cells that pass quality control.

    This function applies two levels of filtering:

    1. Rule-based (fast, no model needed):
       - Edge cells: bounding box touches image border
       - Debris: area below minimum threshold

    2. CNN-based (requires trained model):
       - Classifies remaining cells into quality categories
       - Keeps only cells whose predicted class is in keep_classes

    If no model_path is provided, only rule-based filtering is applied.

    Args:
        labeled_mask:   (Y, X) integer array from Cellpose (0=bg, N=cell)
        image_channels: (C, Y, X) array — all imaging channels
        phase_channel:  index of the phase contrast channel
        model_path:     path to trained .pth model (None = rules only)
        pixels_per_um:  pixel scale for area calculations
        keep_classes:   tuple of class names to keep (default: only "good")
        confidence_threshold: minimum softmax probability to trust prediction
        use_rules:      whether to apply rule-based pre-filtering
        device:         torch device (None = auto)
        verbose:        print filtering statistics

    Returns:
        filtered_mask:  (Y, X) integer array with rejected cells set to 0
        report:         dict with filtering statistics
    """
    filtered = labeled_mask.copy()
    all_labels = set(np.unique(labeled_mask)) - {0}
    n_total = len(all_labels)

    report = {
        "total_cells": n_total,
        "removed_edge": 0,
        "removed_debris": 0,
        "removed_cnn": 0,
        "kept": 0,
        "details": {},  # label -> (class_name, confidence)
    }

    if n_total == 0:
        return filtered, report

    # ── Step 1: Rule-based filtering ──────────────────────────────────────
    removed = set()

    if use_rules:
        edge_cells = detect_edge_cells(labeled_mask, margin=EDGE_MARGIN)
        for lbl in edge_cells:
            filtered[filtered == lbl] = 0
            report["details"][int(lbl)] = ("edge_cell", 1.0)
        removed |= edge_cells
        report["removed_edge"] = len(edge_cells)

        debris_cells = detect_debris_by_area(
            labeled_mask, pixels_per_um=pixels_per_um, min_area_um2=MIN_AREA_UM2
        )
        # Don't double-count cells already removed as edge
        debris_only = debris_cells - edge_cells
        for lbl in debris_only:
            filtered[filtered == lbl] = 0
            report["details"][int(lbl)] = ("debris", 1.0)
        removed |= debris_only
        report["removed_debris"] = len(debris_only)

    remaining = all_labels - removed

    # ── Step 2: CNN classification ────────────────────────────────────────
    if model_path is not None and remaining:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # 3 channels: phase + mask + area (area is derived from mask)
        in_channels = 3
        clf = CellQualityClassifier(str(model_path), in_channels, device=device)

        # Extract crops for remaining cells (phase + mask + true area)
        labels_list = sorted(remaining)
        crops_list = []
        for lbl in labels_list:
            crop, _ = extract_cell_crop(image_channels, labeled_mask, lbl,
                                        phase_channel=phase_channel)
            if crop is not None:
                # True pixel area from original mask (shape-independent)
                true_area = int(np.sum(labeled_mask == lbl))
                crop = _append_area_channel_single(crop, true_area_px=true_area)
                crops_list.append(crop)
            else:
                crops_list.append(
                    np.zeros((in_channels, CROP_SIZE, CROP_SIZE), dtype=np.float32)
                )

        crops_array = np.stack(crops_list, axis=0)
        class_indices, probabilities = clf.predict_batch(crops_array)

        keep_indices = {CLASS_NAMES.index(c) for c in keep_classes if c in CLASS_NAMES}
        cnn_removed = 0

        for lbl, cls_idx, probs in zip(labels_list, class_indices, probabilities):
            cls_name = CLASS_NAMES[cls_idx]
            confidence = float(probs[cls_idx])
            report["details"][int(lbl)] = (cls_name, confidence)

            # Remove if not in keep_classes OR below confidence threshold
            if cls_idx not in keep_indices or confidence < confidence_threshold:
                filtered[filtered == lbl] = 0
                cnn_removed += 1

        report["removed_cnn"] = cnn_removed

    report["kept"] = len(set(np.unique(filtered)) - {0})

    if verbose:
        print(f"    Cell QC: {n_total} total → {report['kept']} kept "
              f"(edge={report['removed_edge']}, debris={report['removed_debris']}, "
              f"cnn={report['removed_cnn']})")

    return filtered, report


# ── Utility: save/load training crops ─────────────────────────────────────────

def save_crop_dataset(crops_dict, labels_dict, output_dir):
    """
    Save extracted crops and labels to disk for training.

    Directory structure:
        output_dir/
        ├── crops/
        │   ├── cond_fov_001.npy
        │   ├── cond_fov_002.npy
        │   ...
        ├── labels.csv     (filename, label_idx, label_name)
        └── metadata.json  (crop_size, n_channels, class_names)

    Args:
        crops_dict: dict mapping identifier -> (C, H, W) array
        labels_dict: dict mapping identifier -> class index (int)
        output_dir: path to save directory
    """
    import json
    import csv

    output_dir = Path(output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_channels = None

    for ident, crop in crops_dict.items():
        fname = f"{ident}.npy"
        np.save(crops_dir / fname, crop)
        label_idx = labels_dict.get(ident, -1)
        label_name = CLASS_NAMES[label_idx] if 0 <= label_idx < NUM_CLASSES else "unlabeled"
        rows.append([fname, label_idx, label_name])
        if n_channels is None:
            n_channels = crop.shape[0]

    with open(output_dir / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label_idx", "label_name"])
        writer.writerows(rows)

    meta = {
        "crop_size": CROP_SIZE,
        "n_channels": n_channels,
        "class_names": CLASS_NAMES,
        "n_samples": len(rows),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_crop_dataset(dataset_dir, phase_only=False, phase_channel_in_crop=0):
    """
    Load a saved crop dataset.

    metadata.json is optional — if absent, n_channels is inferred from the
    first .npy file in crops/ and class_names defaults to CLASS_NAMES.

    Args:
        dataset_dir:           path to the saved dataset directory
        phase_only:            if True, slice each crop to keep only the phase
                               channel and the mask channel → (N, 2, H, W).
                               This makes the model generalisable across setups
                               with different numbers of fluorescence channels.
        phase_channel_in_crop: index of the phase channel within the saved crop
                               (default 0, since phase is typically the first
                               imaging channel stored). The mask is always the
                               last channel.

    Returns:
        crops:  (N, C, H, W) float32 array  (C=2 if phase_only, else all channels)
        labels: (N,) int array
        meta:   dict with dataset metadata
    """
    import json
    import csv

    dataset_dir = Path(dataset_dir)

    # ── Load or infer metadata ─────────────────────────────────────────────
    meta_path = dataset_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        # Infer n_channels from the first crop file we can find
        crops_dir = dataset_dir / "crops"
        n_channels = None
        if crops_dir.exists():
            npy_files = sorted(crops_dir.glob("*.npy"))
            if npy_files:
                sample = np.load(npy_files[0])
                n_channels = sample.shape[0]
        meta = {
            "crop_size": CROP_SIZE,
            "n_channels": n_channels,
            "class_names": CLASS_NAMES,
            "n_samples": None,  # filled below
        }
        print(
            f"  Note: metadata.json not found in {dataset_dir}.\n"
            f"  Inferred n_channels={n_channels} from crops directory.\n"
            f"  Using default class names: {CLASS_NAMES}"
        )

    # ── Load crops and labels ──────────────────────────────────────────────
    labels_csv = dataset_dir / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"labels.csv not found in {dataset_dir}.\n"
            f"Run label_cells.py first to generate labeled training data."
        )

    crops_list = []
    labels_list = []
    areas_list  = []   # true pixel areas from labels.csv (None if absent)

    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        has_area_col = "area_px" in (reader.fieldnames or [])
        for row in reader:
            label_idx = int(row["label_idx"])
            if label_idx < 0:
                continue  # skip unlabeled
            fname = row["filename"]
            if not fname.endswith(".npy"):
                fname = fname + ".npy"
            crop_path = dataset_dir / "crops" / fname
            if not crop_path.exists():
                print(f"  Warning: crop file not found, skipping: {crop_path.name}")
                continue
            crop = np.load(crop_path)
            crops_list.append(crop)
            labels_list.append(label_idx)
            # Read true area if the column is present and populated
            if has_area_col and row.get("area_px", "").strip():
                areas_list.append(int(row["area_px"]))
            else:
                areas_list.append(None)

    if not crops_list:
        raise ValueError(
            f"No labeled crops found in {dataset_dir}.\n"
            f"Make sure label_cells.py has been run and cells have been labeled."
        )

    crops = np.stack(crops_list, axis=0).astype(np.float32)
    labels = np.array(labels_list, dtype=np.int64)
    # True areas: use array if all present, else None (triggers fallback in helper)
    true_areas = (np.array(areas_list, dtype=np.float32)
                  if all(a is not None for a in areas_list) else None)
    if true_areas is None and any(a is not None for a in areas_list):
        # Partial coverage — fill missing with None (helper will use mask fallback)
        true_areas = np.array(
            [a if a is not None else 0 for a in areas_list], dtype=np.float32
        )

    # ── Remap legacy 5-class labels to binary (good=0, bad=1) ─────────────
    # Old scheme: 0=good, 1=oof, 2=clumped, 3=edge, 4=debris
    # New scheme: 0=good, 1=bad  (anything > 0 becomes 1)
    stored_classes = meta.get("class_names", CLASS_NAMES)
    if len(stored_classes) > 2 or labels.max() > 1:
        n_remapped = int((labels > 0).sum())
        labels = (labels > 0).astype(np.int64)  # 0 stays 0, everything else → 1
        meta["class_names"] = CLASS_NAMES
        print(f"  Remapped legacy {len(stored_classes)}-class labels → binary "
              f"(good/bad). {n_remapped} cells mapped to 'bad'.")

    # ── Phase-only slicing ─────────────────────────────────────────────────
    if phase_only and crops.shape[1] > 2:
        orig_ch = crops.shape[1]
        # Keep only the phase channel and the mask channel (always last)
        crops = crops[:, [phase_channel_in_crop, -1], :, :]
        print(f"  Phase-only mode: sliced crops from {orig_ch} → 2 channels "
              f"(phase idx={phase_channel_in_crop} + mask)")

    # ── Append area channel ───────────────────────────────────────────────
    # Uses true pixel areas from labels.csv where available (shape-independent),
    # falling back to resized mask fill fraction for older datasets.
    crops = _append_area_channel_batch(crops, true_areas_px=true_areas)
    meta["n_channels"] = crops.shape[1]
    src = "true pixel areas from labels.csv" if true_areas is not None else "mask fallback (re-label for true areas)"
    print(f"  Added area channel → {crops.shape[1]} total channels  [{src}]")

    meta["n_samples"] = len(labels)

    return crops, labels, meta
