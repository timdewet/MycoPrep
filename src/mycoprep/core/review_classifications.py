#!/usr/bin/env python3
"""
Active Learning Review Tool
============================
Review and correct CNN classifier predictions on segmented cells.

Instead of labeling cells from scratch (like label_cells.py), this tool
runs a trained model first, shows predictions with confidence scores,
and lets the user accept or correct them. Cells are sorted by confidence
(lowest first) so you review the most uncertain predictions first.

Keyboard shortcuts:
    a = accept model prediction (mark reviewed, keep label)
    f = flip label (toggle good ↔ bad)
    g = force good
    b = force bad
    u = undo (revert to model prediction)
    A = accept all remaining on current page
    → / n = next page
    ← / p = previous page
    s = save
    q = quit + save

Usage:
    python review_classifications.py \\
        --input output_tiffs/ \\
        --model models_mtb/best_model.pth \\
        --save-dir labeled_data/

    # Review only uncertain predictions (confidence < 0.85):
    python review_classifications.py \\
        --input output_tiffs/ \\
        --model models_mtb/best_model.pth \\
        --save-dir labeled_data/ \\
        --mode uncertain --confidence-threshold 0.85

    # Review all predictions, sorted by confidence:
    python review_classifications.py \\
        --input output_tiffs/ \\
        --model models_mtb/best_model.pth \\
        --save-dir labeled_data/ \\
        --mode all
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .cell_quality_classifier import (
    CLASS_NAMES,
    CROP_SIZE,
    CellQualityClassifier,
    _append_area_channel_batch,
)
from .cell_quality_classifier import extract_cell_crop

from .label_cells import (
    _make_contour_thumb,
    _phase_idx_in_crop,
    _save_progress,
    load_hyperstack,
    get_labeled_mask_from_fov,
)

# ── Grid layout ──────────────────────────────────────────────────────────────

GRID_COLS = 8
GRID_ROWS = 6
PAGE_SIZE = GRID_COLS * GRID_ROWS

# ── Colours ──────────────────────────────────────────────────────────────────

COLOR_GOOD       = "#33cc33"
COLOR_BAD        = "#dd2222"
COLOR_UNREVIEWED = "#555555"
COLOR_SELECTED   = "#ffff00"
COLOR_CORRECTED  = "#ff9900"   # orange border for user-corrected cells

PRED_COLORS = {0: COLOR_GOOD, 1: COLOR_BAD}
SELECTED_LW = 4
NORMAL_LW   = 2
CORRECTED_LW = 3


# ── Extract cells and run predictions ────────────────────────────────────────

def extract_and_predict(tiff_files, model_path, phase_channel, mask_channel,
                        save_dir=None, pixels_per_um=13.8767):
    """
    Extract all cells from TIFFs and classify them with the trained model.

    If save_dir contains an existing labels.csv, manual labels are loaded
    and attached to each cell so the UI can show them.

    Returns a list of cell dicts, each with keys:
        cell_id, thumb, crop, area_px, predicted_class, confidence,
        manual_label (int or None), has_manual_label (bool)
    """
    # Extract cells — one random FOV per TIFF for speed
    import random

    print("Extracting cell crops (1 random FOV per TIFF)...")
    ph_in_crop = _phase_idx_in_crop(phase_channel, mask_channel)
    cells = []

    for tiff_path in tiff_files:
        print(f"  Loading {tiff_path.name} ...", end=" ", flush=True)
        data, meta = load_hyperstack(tiff_path)
        n_fov, n_ch = data.shape[0], data.shape[1]
        print(f"shape={data.shape} ({n_fov} FOVs × {n_ch} ch)", end=" ")

        if mask_channel >= n_ch:
            print(f"— skipping (mask_channel={mask_channel} > {n_ch} channels)")
            continue

        fov_idx = random.randint(0, n_fov - 1)
        fov = data[fov_idx]
        labeled_mask, n = get_labeled_mask_from_fov(fov, mask_channel)
        if n == 0:
            print(f"— FOV {fov_idx}: 0 cells")
            continue

        img_channels = np.delete(fov, mask_channel, axis=0)
        count = 0
        for cell_label in range(1, n + 1):
            crop, _ = extract_cell_crop(img_channels, labeled_mask, cell_label,
                                        phase_channel=ph_in_crop)
            if crop is None:
                continue
            area_px = int(np.sum(labeled_mask == cell_label))
            stem = tiff_path.stem
            cell_id = f"{stem}_fov{fov_idx:03d}_cell{cell_label:04d}"
            thumb = _make_contour_thumb(crop, ph_in_crop)
            cells.append({
                "cell_id": cell_id,
                "thumb":   thumb,
                "crop":    crop,
                "area_px": area_px,
            })
            count += 1
        print(f"— FOV {fov_idx}: {count} cells")

    if not cells:
        print("No cells found.")
        return []

    print(f"  {len(cells)} cells extracted total\n")

    # Load model — read in_channels from training config
    model_dir = Path(model_path).parent
    config_path = model_dir / "training_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        in_channels = config["in_channels"]
    else:
        # Default: phase + mask + area = 3
        in_channels = 3

    print(f"Loading model: {model_path}  (in_channels={in_channels})")
    classifier = CellQualityClassifier(model_path, in_channels=in_channels)

    # Stack crops and append area channel for prediction
    crops = np.stack([c["crop"] for c in cells], axis=0)       # (N, C, H, W)
    areas = np.array([c["area_px"] for c in cells], dtype=np.float32)
    crops_with_area = _append_area_channel_batch(crops, true_areas_px=areas)

    print(f"Running batch prediction on {len(cells)} cells...")
    class_indices, probabilities = classifier.predict_batch(crops_with_area)

    # Load existing manual labels if available
    manual_labels: dict[str, int] = {}
    if save_dir is not None:
        labels_file = Path(save_dir) / "labels.csv"
        if labels_file.exists():
            with open(labels_file) as f:
                for row in csv.DictReader(f):
                    cid = row["filename"].replace(".npy", "")
                    manual_labels[cid] = int(row["label_idx"])
            print(f"  Loaded {len(manual_labels)} existing manual labels")

    # Attach predictions to cell dicts
    n_disagree = 0
    for i, cell in enumerate(cells):
        cell["predicted_class"] = int(class_indices[i])
        cell["confidence"] = float(probabilities[i].max())

        # Manual label (if any)
        cid = cell["cell_id"]
        if cid in manual_labels:
            cell["manual_label"] = manual_labels[cid]
            cell["has_manual_label"] = True
            # Use the manual label as the starting final_label
            cell["final_label"] = manual_labels[cid]
            cell["reviewed"] = True  # already human-reviewed
            cell["corrected"] = (manual_labels[cid] != int(class_indices[i]))
            if cell["corrected"]:
                n_disagree += 1
        else:
            cell["manual_label"] = None
            cell["has_manual_label"] = False
            cell["final_label"] = int(class_indices[i])
            cell["reviewed"] = False
            cell["corrected"] = False

    # Sort by confidence ascending (least confident first)
    cells.sort(key=lambda c: c["confidence"])

    pred_good = sum(1 for c in cells if c["predicted_class"] == 0)
    pred_bad  = sum(1 for c in cells if c["predicted_class"] == 1)
    n_manual  = sum(1 for c in cells if c["has_manual_label"])
    print(f"  Predictions: {pred_good} good, {pred_bad} bad")
    if n_manual > 0:
        print(f"  Manual labels found: {n_manual} "
              f"({n_disagree} disagree with model)")
    print(f"  Confidence range: {cells[0]['confidence']:.3f} – "
          f"{cells[-1]['confidence']:.3f}\n")

    return cells


# ── Merge reviewed labels with existing labeled data ─────────────────────────

def merge_and_save(cells, save_dir, crops_dir, merge=True):
    """
    Save reviewed cells to labeled_data/ format, optionally merging with
    existing labels.csv.
    """
    save_dir  = Path(save_dir)
    crops_dir = Path(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Load existing labels if merging
    labels: dict[str, int] = {}
    areas:  dict[str, int] = {}
    labels_file = save_dir / "labels.csv"

    if merge and labels_file.exists():
        with open(labels_file) as f:
            for row in csv.DictReader(f):
                cid = row["filename"].replace(".npy", "")
                labels[cid] = int(row["label_idx"])
                if "area_px" in row and row["area_px"]:
                    areas[cid] = int(row["area_px"])
        n_existing = len(labels)
    else:
        n_existing = 0

    # Add reviewed cells
    n_new = 0
    n_updated = 0
    for cell in cells:
        if not cell["reviewed"]:
            continue
        cid = cell["cell_id"]
        if cid in labels:
            n_updated += 1
        else:
            n_new += 1
        labels[cid] = cell["final_label"]
        areas[cid] = cell["area_px"]

        # Save crop if not already on disk
        npy_path = crops_dir / f"{cid}.npy"
        if not npy_path.exists():
            np.save(npy_path, cell["crop"])

    # Use the existing _save_progress to write CSV + metadata
    _save_progress(save_dir, crops_dir, labels, areas)

    # Print summary
    reviewed = [c for c in cells if c["reviewed"]]
    corrected = [c for c in cells if c["corrected"]]
    good_to_bad = sum(1 for c in corrected if c["predicted_class"] == 0)
    bad_to_good = sum(1 for c in corrected if c["predicted_class"] == 1)

    print("\n" + "=" * 50)
    print("Review Summary:")
    print(f"  Total cells:       {len(cells):,}")
    print(f"  Reviewed:          {len(reviewed):,} ({100*len(reviewed)/max(len(cells),1):.1f}%)")
    if reviewed:
        n_accepted = len(reviewed) - len(corrected)
        print(f"  Accepted (agree):  {n_accepted:,} ({100*n_accepted/len(reviewed):.1f}%)")
        print(f"  Corrected (flip):  {len(corrected):,} ({100*len(corrected)/len(reviewed):.1f}%)")
        if corrected:
            print(f"    good → bad:      {good_to_bad:,}")
            print(f"    bad → good:      {bad_to_good:,}")
    if n_existing > 0:
        print(f"  Merged with {n_existing:,} existing labels "
              f"({n_new} new, {n_updated} updated)")
    print(f"  Saved to: {save_dir / 'labels.csv'}")
    print("=" * 50)


# ── Grid review UI ───────────────────────────────────────────────────────────

def run_review_grid(cells, save_dir, pixels_per_um=13.8767, merge=True):
    """
    Interactive matplotlib grid for reviewing classifier predictions.

    Shows cells sorted by confidence (lowest first). Border colour reflects
    the current label (model prediction or user correction). User can accept,
    flip, or override each prediction.
    """
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    save_dir  = Path(save_dir)
    crops_dir = save_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(cells)
    n_pages = (n_total + PAGE_SIZE - 1) // PAGE_SIZE

    # ── Build figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS,
        figsize=(GRID_COLS * 1.4, GRID_ROWS * 1.4 + 0.8),
        gridspec_kw={"wspace": 0.05, "hspace": 0.35},
    )
    axes_flat = axes.flatten()

    img_handles = []
    border_patches = []

    for ax in axes_flat:
        ax.axis("off")
        im = ax.imshow(np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.float32),
                       interpolation="nearest")
        img_handles.append(im)

        rect = mpatches.FancyBboxPatch(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            boxstyle="square,pad=0",
            linewidth=NORMAL_LW,
            edgecolor=COLOR_UNREVIEWED,
            facecolor="none",
            clip_on=False,
        )
        ax.add_patch(rect)
        border_patches.append(rect)

    status_text = fig.text(
        0.5, 0.01, "",
        ha="center", va="bottom", fontsize=8,
        fontfamily="monospace",
    )
    fig.patch.set_facecolor("#1a1a1a")

    # ── State ────────────────────────────────────────────────────────────
    state = {"page": 0, "selected": None}

    # ── Render current page ──────────────────────────────────────────────
    def render_page():
        page    = state["page"]
        sel_pos = state["selected"]
        start   = page * PAGE_SIZE

        for pos, (ax, im, rect) in enumerate(zip(axes_flat, img_handles,
                                                   border_patches)):
            idx = start + pos
            if idx < n_total:
                cell = cells[idx]
                im.set_data(cell["thumb"])
                ax.set_visible(True)

                final = cell["final_label"]

                # Border colour and width
                if pos == sel_pos:
                    rect.set_edgecolor(COLOR_SELECTED)
                    rect.set_linewidth(SELECTED_LW)
                elif cell["corrected"]:
                    rect.set_edgecolor(COLOR_CORRECTED)
                    rect.set_linewidth(CORRECTED_LW)
                else:
                    rect.set_edgecolor(PRED_COLORS.get(final, COLOR_UNREVIEWED))
                    rect.set_linewidth(NORMAL_LW)

                # Title: class letter, confidence, area, manual label info
                lbl_char = CLASS_NAMES[final][0].upper()
                conf = cell["confidence"]
                pred_char = CLASS_NAMES[cell["predicted_class"]][0].upper()

                area_px = cell.get("area_px")
                if area_px and area_px > 0:
                    area_um2 = area_px / (pixels_per_um ** 2)
                    size_str = f" {area_um2:.1f}µm²"
                else:
                    size_str = ""

                # Show disagreement between manual label and model
                if cell["has_manual_label"]:
                    manual_char = CLASS_NAMES[cell["manual_label"]][0].upper()
                    if cell["manual_label"] != cell["predicted_class"]:
                        # Disagree: show "manual!=model"
                        disagree_str = f" {manual_char}!={pred_char}"
                        title_color = COLOR_CORRECTED  # orange = attention
                    else:
                        disagree_str = " ✓"
                        title_color = PRED_COLORS.get(final, "#cccccc")
                elif conf < 0.7:
                    disagree_str = ""
                    title_color = COLOR_CORRECTED
                else:
                    disagree_str = ""
                    title_color = PRED_COLORS.get(final, "#cccccc")

                reviewed_mark = "✓" if cell["reviewed"] else ""
                corrected_mark = "✎" if cell["corrected"] else ""
                title = f"{lbl_char} {conf:.2f}{size_str}{disagree_str} {corrected_mark}"
                ax.set_title(title, fontsize=7, color=title_color, pad=1)
            else:
                im.set_data(np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.float32))
                rect.set_edgecolor("#222222")
                rect.set_linewidth(NORMAL_LW)
                ax.set_title("", pad=1)
                ax.set_visible(True)

        # Status bar
        n_reviewed  = sum(1 for c in cells if c["reviewed"])
        n_corrected = sum(1 for c in cells if c["corrected"])
        page_end    = min(start + PAGE_SIZE, n_total)

        # Confidence range on this page
        page_cells = cells[start:page_end]
        if page_cells:
            conf_lo = min(c["confidence"] for c in page_cells)
            conf_hi = max(c["confidence"] for c in page_cells)
            conf_str = f"conf {conf_lo:.2f}–{conf_hi:.2f}"
        else:
            conf_str = ""

        status_text.set_text(
            f"Page {page + 1}/{n_pages}  |  "
            f"Cells {start + 1}–{page_end} of {n_total}  |  "
            f"Reviewed: {n_reviewed}  Corrected: {n_corrected}  |  "
            f"{conf_str}  |  "
            f"a=accept f=flip g/b=force u=undo A=accept-page s=save q=quit"
        )
        status_text.set_color("#cccccc")
        fig.canvas.draw_idle()

    # ── Helper: advance to next unreviewed cell ──────────────────────────
    def _advance_from(pos):
        page = state["page"]
        for next_pos in range(pos + 1, PAGE_SIZE):
            next_idx = page * PAGE_SIZE + next_pos
            if next_idx >= n_total:
                break
            if not cells[next_idx]["reviewed"]:
                return next_pos
        return pos  # stay if none found

    # ── Event: key press ─────────────────────────────────────────────────
    def on_key(event):
        page    = state["page"]
        sel_pos = state["selected"]
        key     = event.key

        # Navigation
        if key in ("right", "n"):
            if page + 1 < n_pages:
                state["page"] += 1
                state["selected"] = None
                render_page()
            return
        if key in ("left", "p"):
            if page > 0:
                state["page"] -= 1
                state["selected"] = None
                render_page()
            return

        # Save / quit
        if key == "s":
            merge_and_save(cells, save_dir, crops_dir, merge=merge)
            return
        if key == "q":
            merge_and_save(cells, save_dir, crops_dir, merge=merge)
            plt.close(fig)
            return

        # Accept all remaining on page
        if key == "A":
            start = page * PAGE_SIZE
            accepted = 0
            for pos in range(PAGE_SIZE):
                idx = start + pos
                if idx >= n_total:
                    break
                cell = cells[idx]
                if not cell["reviewed"]:
                    cell["reviewed"] = True
                    accepted += 1
            print(f"  Accepted {accepted} cells on page {page + 1}")
            render_page()
            return

        # Cell-level actions require a selection
        if sel_pos is None:
            return

        idx = page * PAGE_SIZE + sel_pos
        if idx >= n_total:
            return

        cell = cells[idx]

        if key == "a":
            # Accept: keep model prediction
            cell["reviewed"] = True
            cell["final_label"] = cell["predicted_class"]
            cell["corrected"] = False
            print(f"  ✓ {cell['cell_id']} accepted as "
                  f"{CLASS_NAMES[cell['final_label']]} ({cell['confidence']:.2f})")
            state["selected"] = _advance_from(sel_pos)
            render_page()
            return

        if key == "f":
            # Flip: toggle label
            cell["reviewed"] = True
            cell["final_label"] = 1 - cell["final_label"]
            cell["corrected"] = (cell["final_label"] != cell["predicted_class"])
            print(f"  ✎ {cell['cell_id']} flipped to "
                  f"{CLASS_NAMES[cell['final_label']]} "
                  f"(was {CLASS_NAMES[cell['predicted_class']]} "
                  f"{cell['confidence']:.2f})")
            state["selected"] = _advance_from(sel_pos)
            render_page()
            return

        if key == "g":
            cell["reviewed"] = True
            cell["final_label"] = 0
            cell["corrected"] = (cell["predicted_class"] != 0)
            print(f"  → {cell['cell_id']} → good")
            state["selected"] = _advance_from(sel_pos)
            render_page()
            return

        if key == "b":
            cell["reviewed"] = True
            cell["final_label"] = 1
            cell["corrected"] = (cell["predicted_class"] != 1)
            print(f"  → {cell['cell_id']} → bad")
            state["selected"] = _advance_from(sel_pos)
            render_page()
            return

        if key == "u":
            # Undo: revert to model prediction
            cell["final_label"] = cell["predicted_class"]
            cell["reviewed"] = False
            cell["corrected"] = False
            print(f"  ↩ {cell['cell_id']} reverted to "
                  f"{CLASS_NAMES[cell['predicted_class']]}")
            render_page()
            return

    # ── Event: mouse click ───────────────────────────────────────────────
    def on_click(event):
        if event.inaxes is None:
            return
        for pos, ax in enumerate(axes_flat):
            if ax == event.inaxes:
                idx = state["page"] * PAGE_SIZE + pos
                if idx < n_total:
                    state["selected"] = pos
                    render_page()
                return

    # Disconnect default key handler (captures arrow keys for pan/zoom)
    try:
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    except AttributeError:
        pass

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)

    render_page()

    print("=== Active Learning Review ===")
    print("Cells sorted by confidence (lowest first — most likely errors).")
    print("Click a cell to select it, then:")
    print("  a=accept  f=flip  g=good  b=bad  u=undo")
    print("  A=accept all on page")
    print("  n=next page  p=prev page  (arrow keys also work)")
    print("  s=save  q=quit+save\n")

    plt.show(block=True)

    # Save on window close
    merge_and_save(cells, save_dir, crops_dir, merge=merge)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Review and correct CNN classifier predictions on cells.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Directory of pipeline output TIFFs")
    parser.add_argument("--model", "-m", required=True,
                        help="Path to trained .pth model file")
    parser.add_argument("--save-dir", "-s", required=True,
                        help="Output directory for reviewed labels (labeled_data/ format)")
    parser.add_argument("--phase-channel", type=int, default=0,
                        help="Phase contrast channel index (default: 0)")
    parser.add_argument("--mask-channel", type=int, default=None,
                        help="Mask channel index (default: last channel)")
    parser.add_argument("--pixels-per-um", type=float, default=13.8767,
                        help="Pixel scale (default: 13.8767)")
    parser.add_argument("--confidence-threshold", type=float, default=0.85,
                        help="In 'uncertain' mode, only show cells below this "
                             "confidence (default: 0.85)")
    parser.add_argument("--mode", choices=["uncertain", "all"], default="uncertain",
                        help="'uncertain' = only low-confidence cells; "
                             "'all' = everything sorted by confidence (default: uncertain)")
    parser.add_argument("--no-merge", action="store_true",
                        help="Don't merge with existing labels.csv (overwrite instead)")

    args = parser.parse_args()

    # Resolve input TIFFs
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"ERROR: {input_dir} is not a directory")
        sys.exit(1)

    tiff_files = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    if not tiff_files:
        print(f"ERROR: No .tif/.tiff files found in {input_dir}")
        sys.exit(1)
    print(f"Found {len(tiff_files)} TIFF file(s) in {input_dir}\n")

    # Auto-detect mask channel if not specified
    mask_channel = args.mask_channel
    if mask_channel is None:
        # Default to last channel (pipeline convention)
        data, _ = load_hyperstack(tiff_files[0])
        mask_channel = data.shape[1] - 1
        print(f"Auto-detected mask channel: {mask_channel} (last channel)\n")

    # Extract and predict
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    cells = extract_and_predict(
        tiff_files, str(model_path),
        args.phase_channel, mask_channel,
        save_dir=args.save_dir,
        pixels_per_um=args.pixels_per_um,
    )
    if not cells:
        sys.exit(0)

    # Filter by mode
    if args.mode == "uncertain":
        threshold = args.confidence_threshold
        n_before = len(cells)
        cells = [c for c in cells if c["confidence"] < threshold]
        print(f"Uncertain mode: {len(cells)} / {n_before} cells have "
              f"confidence < {threshold}\n")
        if not cells:
            print("All cells are above the confidence threshold. "
                  "Try --mode all or increase --confidence-threshold.")
            sys.exit(0)

    # Launch review UI
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_review_grid(
        cells, save_dir,
        pixels_per_um=args.pixels_per_um,
        merge=not args.no_merge,
    )


if __name__ == "__main__":
    main()
