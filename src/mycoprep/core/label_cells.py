#!/usr/bin/env python3
"""
Cell Quality Labeling Tool
==========================
Interactive tool for labeling segmented cells as good/bad for training
the cell quality CNN classifier.

Three modes:
    1. grid   (default) — Fast matplotlib grid of 64×64 thumbnails. Click to
                          select a cell, press a key to label. Requires only
                          matplotlib + scikit-image.
    2. napari — Full-FOV viewer. Requires napari[all].
    3. montage — Generates contact-sheet PNGs + CSV template for offline labeling.

Workflow (grid mode — recommended):
    1. Run this script pointing at your pipeline output TIFFs
    2. A grid of phase-contrast cell crops appears
    3. Click a crop to select it (yellow border)
    4. Press a key to label:
           g = good
           b = bad
           u = undo
    5. Navigate pages: → / n = next page, ← / p = prev page
    6. s = save,  q = quit and save

Requirements:
    grid/montage: pip install matplotlib scikit-image  (already in cellpose env)
    napari:       pip install 'napari[all]'

Usage:
    python label_cells.py --input output_tiffs/ --save-dir labeled_data/
    python label_cells.py --input output_tiffs/ --save-dir labeled_data/ --mode napari
    python label_cells.py --input output_tiffs/ --save-dir labeled_data/ --mode montage
    python label_cells.py --input output_tiffs/ --save-dir labeled_data/ \\
        --phase-channel 2 --mask-channel 3
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

from .cell_quality_classifier import (
    CLASS_NAMES,
    CROP_SIZE,
    extract_cell_crop,
)

# ── Class colours (matplotlib-compatible) ─────────────────────────────────────
CLASS_COLORS = {
    None: "#555555",   # unlabeled — dark grey
    0:    "#33cc33",   # good — green
    1:    "#dd2222",   # bad — red
}
SELECTED_COLOR  = "#ffff00"   # yellow
SELECTED_LW     = 4
NORMAL_LW       = 2

GRID_COLS = 8
GRID_ROWS = 6
PAGE_SIZE = GRID_COLS * GRID_ROWS


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_hyperstack(tiff_path):
    """
    Load a pipeline output TIFF (ImageJ ZCYX hyperstack).

    tifffile may return the data as a flat (Z*C, Y, X) array rather than
    the expected (Z, C, Y, X), depending on version.  We use the ImageJ
    metadata ('slices' and 'channels') to reshape correctly.

    Returns (N_FOV, C, Y, X) and metadata dict.
    """
    with tifffile.TiffFile(str(tiff_path)) as tif:
        data = tif.asarray()
        metadata = dict(tif.imagej_metadata) if tif.imagej_metadata else {}

    if data.ndim == 4:
        # Already (Z, C, Y, X) — nothing to do
        pass
    elif data.ndim == 3:
        n_ch     = int(metadata.get("channels", 1))
        n_slices = int(metadata.get("slices", 1))

        if n_ch > 1 and n_slices > 1:
            # Flat (Z*C, Y, X) → reshape to (Z, C, Y, X)
            expected_frames = n_slices * n_ch
            if data.shape[0] == expected_frames:
                data = data.reshape(n_slices, n_ch, data.shape[1], data.shape[2])
            else:
                # Fallback: can't reconcile, treat as single-FOV with many channels
                data = data[np.newaxis]
        elif n_ch > 1:
            # Single Z, multiple C: (C, Y, X) → (1, C, Y, X)
            data = data[np.newaxis]
        elif n_slices > 1:
            # Multiple Z, single C: (Z, Y, X) → (Z, 1, Y, X)
            data = data[:, np.newaxis]
        else:
            # Single Z, single C: (1, Y, X) → (1, 1, Y, X)
            data = data[np.newaxis]
    elif data.ndim == 2:
        data = data[np.newaxis, np.newaxis]

    return data, metadata


def get_labeled_mask_from_fov(fov_data, mask_channel):
    """Re-label a binary mask channel with connected components."""
    from skimage.measure import label as sk_label
    binary = fov_data[mask_channel] > 0
    labeled = sk_label(binary, connectivity=1)
    return labeled, int(labeled.max())


def _phase_idx_in_crop(phase_channel, mask_channel):
    """Map original phase channel index to index after mask channel is removed."""
    return phase_channel if phase_channel < mask_channel else phase_channel - 1


def _make_contour_thumb(crop, ph_in_crop, contour_color=(0.0, 1.0, 0.4)):
    """
    Build an RGB thumbnail with the mask contour overlaid on the phase image.

    Args:
        crop:          (C+1, H, W) float32 array — last channel is the binary mask
        ph_in_crop:    index of the phase channel in the crop
        contour_color: RGB tuple [0-1] for the contour line

    Returns:
        thumb_rgb: (H, W, 3) float32 array suitable for imshow with RGB
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    phase = crop[min(ph_in_crop, crop.shape[0] - 2)]  # (H, W) float32 [0,1]
    mask  = crop[-1] > 0.5                              # (H, W) bool

    # Contour = dilated mask minus eroded mask (1-pixel ring)
    dilated = binary_dilation(mask, iterations=1)
    eroded  = binary_erosion(mask, iterations=1)
    contour = dilated & ~eroded  # ring of 1-2 px around cell edge

    # Compose: phase as greyscale background, contour in colour
    rgb = np.stack([phase, phase, phase], axis=-1)  # (H, W, 3)
    for ch_i in range(3):
        rgb[contour, ch_i] = contour_color[ch_i]

    return np.clip(rgb, 0, 1)


def _save_progress(save_dir, crops_dir, labels_dict, areas_dict=None):
    """Write labels.csv (with area_px column) and metadata.json."""
    save_dir = Path(save_dir)
    if areas_dict is None:
        areas_dict = {}
    rows = []
    for filename, label_idx in sorted(labels_dict.items()):
        label_name = (CLASS_NAMES[label_idx]
                      if 0 <= label_idx < len(CLASS_NAMES) else "unknown")
        area_px = areas_dict.get(filename, "")
        rows.append([filename, label_idx, label_name, area_px])

    with open(save_dir / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label_idx", "label_name", "area_px"])
        writer.writerows(rows)

    first_crop = next(Path(crops_dir).glob("*.npy"), None)
    n_ch = int(np.load(first_crop).shape[0]) if first_crop else None
    meta = {
        "crop_size": CROP_SIZE,
        "n_channels": n_ch,
        "class_names": CLASS_NAMES,
        "n_samples": len(rows),
        "n_labeled": sum(1 for _, l in labels_dict.items() if l >= 0),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved {len(rows)} labels → {save_dir / 'labels.csv'}")


# ── Cell extraction: load ALL crops up front ──────────────────────────────────

def extract_all_cells(tiff_files, phase_channel, mask_channel):
    """
    Walk every TIFF and every FOV, extract 64×64 phase-contrast thumbnails
    and the full (C+1)-channel crops for every segmented cell.

    Returns:
        cells: list of dicts with keys:
            cell_id   — unique string identifier
            thumb     — (CROP_SIZE, CROP_SIZE) float32 phase thumbnail [0,1]
            crop_path — Path where the .npy crop is/will be stored
            crop      — (C+1, CROP_SIZE, CROP_SIZE) float32 array
    """
    from skimage.measure import label as sk_label

    cells = []
    ph_in_crop = _phase_idx_in_crop(phase_channel, mask_channel)

    for tiff_path in tiff_files:
        print(f"  Loading {tiff_path.name} ...", end=" ", flush=True)
        data, meta = load_hyperstack(tiff_path)
        n_fov, n_ch_loaded = data.shape[0], data.shape[1]
        print(f"shape={data.shape} ({n_fov} FOVs × {n_ch_loaded} ch)")

        if mask_channel >= n_ch_loaded:
            print(f"    WARNING: mask_channel={mask_channel} but only {n_ch_loaded} "
                  f"channels — skipping this file")
            continue

        total_cells_this_tiff = 0
        for fov_idx in range(n_fov):
            fov = data[fov_idx]
            labeled_mask, n = get_labeled_mask_from_fov(fov, mask_channel)
            if n == 0:
                continue

            img_channels = np.delete(fov, mask_channel, axis=0)

            for cell_label in range(1, n + 1):
                crop, _ = extract_cell_crop(img_channels, labeled_mask, cell_label)
                if crop is None:
                    continue

                # True pixel area from original mask (shape-independent size)
                area_px = int(np.sum(labeled_mask == cell_label))

                stem = tiff_path.stem
                cell_id = f"{stem}_fov{fov_idx:03d}_cell{cell_label:04d}"
                thumb = _make_contour_thumb(crop, ph_in_crop)

                cells.append({
                    "cell_id": cell_id,
                    "thumb":   thumb,       # (H, W, 3) RGB with contour
                    "crop":    crop,
                    "area_px": area_px,     # true pixel count in original mask
                })
                total_cells_this_tiff += 1

        print(f"    → {total_cells_this_tiff} cells extracted")

    return cells


# ── Grid labeling mode ────────────────────────────────────────────────────────

def run_grid_labeling(tiff_files, save_dir, phase_channel, mask_channel,
                      pixels_per_um=13.8767):
    """
    Fast matplotlib grid labeler.

    Pre-extracts all 64×64 thumbnails, then renders a page of GRID_COLS×GRID_ROWS
    crops at a time. All interactivity operates on small thumbnail images —
    no full-FOV rendering.
    """
    import matplotlib
    matplotlib.use("Qt5Agg")          # fast, non-blocking; falls back gracefully
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    save_dir  = Path(save_dir)
    crops_dir = save_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # ── Load existing labels ──────────────────────────────────────────────
    labels_file = save_dir / "labels.csv"
    labels: dict[str, int] = {}          # cell_id -> class_idx
    if labels_file.exists():
        with open(labels_file) as f:
            for row in csv.DictReader(f):
                labels[row["filename"].replace(".npy", "")] = int(row["label_idx"])

    # ── Extract all crops ─────────────────────────────────────────────────
    print("Extracting cell crops (one-time, may take a moment)...")
    cells = extract_all_cells(tiff_files, phase_channel, mask_channel)
    import random
    random.shuffle(cells)  # Mix cells from different images for labelling diversity
    if not cells:
        print("No cells found.")
        return

    n_total = len(cells)
    n_pages = (n_total + PAGE_SIZE - 1) // PAGE_SIZE
    print(f"  {n_total} cells across {n_pages} pages\n")

    # Build areas dict (cell_id → true pixel area from original mask)
    areas: dict[str, int] = {c["cell_id"]: c.get("area_px", 0) for c in cells}

    # Save all crops to disk now (needed for training later)
    for cell in cells:
        npy_path = crops_dir / f"{cell['cell_id']}.npy"
        if not npy_path.exists():
            np.save(npy_path, cell["crop"])

    # ── Build figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS,
        figsize=(GRID_COLS * 1.4, GRID_ROWS * 1.4 + 0.6),
        gridspec_kw={"wspace": 0.05, "hspace": 0.25},
    )
    axes_flat = axes.flatten()

    # Pre-draw all subplots (images never change, only borders do)
    img_handles = []
    border_patches = []

    for ax in axes_flat:
        ax.axis("off")
        im = ax.imshow(np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.float32),
                       interpolation="nearest")
        img_handles.append(im)

        # Border rectangle (full-axes size in axes coords)
        rect = mpatches.FancyBboxPatch(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            boxstyle="square,pad=0",
            linewidth=NORMAL_LW,
            edgecolor=CLASS_COLORS[None],
            facecolor="none",
            clip_on=False,
        )
        ax.add_patch(rect)
        border_patches.append(rect)

    status_text = fig.text(
        0.5, 0.01, "",
        ha="center", va="bottom", fontsize=9,
        fontfamily="monospace",
    )
    fig.patch.set_facecolor("#1a1a1a")

    # ── State ─────────────────────────────────────────────────────────────
    state = {"page": 0, "selected": None}   # selected = index within page (0..PAGE_SIZE-1)

    # ── Helper: render current page ───────────────────────────────────────
    def render_page():
        page    = state["page"]
        sel_pos = state["selected"]
        start   = page * PAGE_SIZE

        for pos, (ax, im, rect) in enumerate(zip(axes_flat, img_handles, border_patches)):
            idx = start + pos
            if idx < n_total:
                cell = cells[idx]
                im.set_data(cell["thumb"])
                ax.set_visible(True)

                cid   = cell["cell_id"]
                label = labels.get(cid, None)

                if pos == sel_pos:
                    rect.set_edgecolor(SELECTED_COLOR)
                    rect.set_linewidth(SELECTED_LW)
                else:
                    color = CLASS_COLORS.get(label, CLASS_COLORS.get(1, "#dd2222"))
                    rect.set_edgecolor(color)
                    rect.set_linewidth(NORMAL_LW)

                # Label indicator + area in px and µm²
                if label is not None:
                    display_label = min(label, 1)
                    lbl_char = CLASS_NAMES[display_label][0].upper()
                    lbl_color = CLASS_COLORS.get(label, CLASS_COLORS[1])
                else:
                    lbl_char = "·"
                    lbl_color = CLASS_COLORS[None]

                area_px = cell.get("area_px")
                if area_px is not None and area_px > 0:
                    area_um2 = area_px / (pixels_per_um ** 2)
                    size_str = f"{area_um2:.1f}µm² ({area_px}px)"
                else:
                    size_str = ""

                title = f"{lbl_char}  {size_str}" if size_str else lbl_char
                ax.set_title(title, fontsize=8, color=lbl_color, pad=1)
            else:
                im.set_data(np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.float32))
                rect.set_edgecolor("#222222")
                ax.set_title("", pad=1)
                ax.set_visible(True)

        n_labeled = sum(1 for v in labels.values() if v >= 0)
        page_end  = min(start + PAGE_SIZE, n_total)
        status_text.set_text(
            f"Page {page + 1}/{n_pages}  |  Cells {start + 1}–{page_end} of {n_total}"
            f"  |  Labeled: {n_labeled}/{n_total}"
            f"  |  g=good  b=bad  u=undo  s=save  q=quit"
        )
        status_text.set_color("#cccccc")
        fig.canvas.draw_idle()

    # ── Event: key press ──────────────────────────────────────────────────
    KEY_TO_CLASS = {"g": 0, "b": 1}

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
            _save_progress(save_dir, crops_dir, labels, areas)
            return
        if key == "q":
            _save_progress(save_dir, crops_dir, labels, areas)
            plt.close(fig)
            return

        # Labeling — requires a selected cell
        if sel_pos is None:
            return

        idx = page * PAGE_SIZE + sel_pos
        if idx >= n_total:
            return

        cell_id = cells[idx]["cell_id"]

        if key == "u":
            if cell_id in labels:
                del labels[cell_id]
                print(f"  Undo: {cell_id}")
                render_page()
            return

        if key in KEY_TO_CLASS:
            class_idx = KEY_TO_CLASS[key]
            labels[cell_id] = class_idx
            print(f"  {cell_id}  →  {CLASS_NAMES[class_idx]}")

            # Advance selection to next unlabeled cell on this page
            for next_pos in range(sel_pos + 1, PAGE_SIZE):
                next_idx = page * PAGE_SIZE + next_pos
                if next_idx >= n_total:
                    break
                if cells[next_idx]["cell_id"] not in labels:
                    state["selected"] = next_pos
                    break
            else:
                state["selected"] = sel_pos  # stay on same cell if none found

            render_page()

    # ── Event: mouse click ────────────────────────────────────────────────
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

    # Disconnect matplotlib's default key handler — it captures arrow keys
    # for pan/zoom, which prevents our page navigation from working.
    try:
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    except AttributeError:
        pass  # some backends don't expose this; arrow keys may not work (use n/p)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)

    render_page()

    print("=== Grid Labeling Tool ===")
    print("Click a cell to select it (yellow border), then press a key:")
    print("  g=good  b=bad  u=undo")
    print("  n=next page  p=prev page  (arrow keys also work)")
    print("  s=save  q=quit+save\n")

    plt.show(block=True)

    # Save on window close
    _save_progress(save_dir, crops_dir, labels, areas)
    n_labeled = sum(1 for v in labels.values() if v >= 0)
    print(f"\nDone! Labeled {n_labeled}/{n_total} cells.")
    print(f"  Crops:  {crops_dir}")
    print(f"  Labels: {labels_file}")


# ── Napari labeling mode ──────────────────────────────────────────────────────

def run_napari_labeling(tiff_files, save_dir, phase_channel, mask_channel):
    """
    Full-FOV napari viewer. Uses a Labels layer (integer array, GPU-accelerated)
    rather than a per-update RGBA overlay, which is much faster.
    """
    try:
        import napari
    except ImportError:
        print(
            "\nERROR: napari is not installed.\n"
            "Install with:  pip install 'napari[all]'\n\n"
            "Or use the default grid mode (no extra dependencies):\n"
            "  python label_cells.py -i <tiffs> -s <save_dir>  (drop --mode napari)"
        )
        sys.exit(1)

    save_dir  = Path(save_dir)
    crops_dir = save_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    labels_file = save_dir / "labels.csv"
    labels: dict[str, int] = {}
    areas:  dict[str, int] = {}          # cell_id → true pixel area
    if labels_file.exists():
        with open(labels_file) as f:
            for row in csv.DictReader(f):
                cid = row["filename"].replace(".npy", "")
                labels[cid] = int(row["label_idx"])
                if "area_px" in row and row["area_px"]:
                    areas[cid] = int(row["area_px"])

    # Class index → napari label colour.
    # napari Labels uses integer values; we map:
    #   0 = background (transparent)
    #   1..2 = class 0..1 (good/bad)
    #   3 = unlabeled (transparent)
    #   SELECTED_LABEL = selected (yellow)
    LABEL_OFFSET   = 1   # class 0 → integer label 1
    SELECTED_LABEL = 6

    def _cell_id(tiff_name, fov_idx, cell_label):
        return f"{Path(tiff_name).stem}_fov{fov_idx:03d}_cell{cell_label:04d}"

    def _build_int_labels(labeled_mask, tiff_name, fov_idx, selected):
        """Map segmentation mask to class-coloured integer array for napari Labels."""
        out = np.zeros_like(labeled_mask, dtype=np.int32)
        for lbl in np.unique(labeled_mask):
            if lbl == 0:
                continue
            cell_id = _cell_id(tiff_name, fov_idx, lbl)
            if lbl == selected:
                val = SELECTED_LABEL
            elif cell_id in labels:
                val = labels[cell_id] + LABEL_OFFSET
            else:
                val = LABEL_OFFSET + len(CLASS_NAMES)  # unlabeled
            out[labeled_mask == lbl] = val
        return out

    # Colour map for napari Labels: index → RGBA
    label_colormap = {
        0: (0, 0, 0, 0),           # background
        1: (0.2, 0.8, 0.2, 0.45),  # good
        2: (0.85, 0.1, 0.1, 0.45), # bad
        3: (0.0, 0.0, 0.0, 0.0),   # unlabeled (transparent — no clutter)
        SELECTED_LABEL: (1.0, 1.0, 0.0, 0.7),   # selected = yellow
    }

    state = {
        "tiff_idx": 0,
        "fov_idx":  0,
        "selected": None,
        "data":     None,
        "lmask":    None,
        "tiff_name": None,
    }

    def _save():
        _save_progress(save_dir, crops_dir, labels, areas)

    def _load_fov(viewer):
        tp = tiff_files[state["tiff_idx"]]
        data, meta = load_hyperstack(tp)
        state.update(data=data, tiff_name=tp.name, selected=None)
        fi = min(state["fov_idx"], data.shape[0] - 1)
        state["fov_idx"] = fi

        fov = data[fi]
        lmask, n = get_labeled_mask_from_fov(fov, mask_channel)
        state["lmask"] = lmask

        viewer.layers.clear()

        phase = fov[phase_channel].astype(np.float32)
        viewer.add_image(phase, name="Phase", colormap="gray", blending="additive")

        n_ch = fov.shape[0]
        cmaps = ["green", "magenta", "cyan", "yellow"]
        ch_names = meta.get("Labels", [f"C{i}" for i in range(n_ch)])
        for ch in range(n_ch):
            if ch in (phase_channel, mask_channel):
                continue
            name = ch_names[ch] if ch < len(ch_names) else f"C{ch}"
            viewer.add_image(fov[ch].astype(np.float32), name=name,
                             colormap=cmaps[ch % len(cmaps)],
                             blending="additive", visible=False, opacity=0.7)

        int_labels = _build_int_labels(lmask, tp.name, fi, state["selected"])
        lyr = viewer.add_labels(int_labels, name="Labels", opacity=1.0)
        lyr.color = label_colormap
        lyr.editable = False

        viewer.title = (
            f"{tp.stem} — FOV {fi+1}/{data.shape[0]} — {n} cells  "
            f"[g]ood [f]ocus [c]lump [e]dge [d]ebris  [n]ext [p]rev"
        )

    def _refresh(viewer):
        int_labels = _build_int_labels(
            state["lmask"], state["tiff_name"], state["fov_idx"], state["selected"]
        )
        for lyr in viewer.layers:
            if lyr.name == "Labels":
                lyr.data = int_labels
                break

    viewer = napari.Viewer()
    _load_fov(viewer)

    @viewer.mouse_drag_callbacks.append
    def on_click(viewer, event):
        if event.type != "mouse_press":
            return
        coords = viewer.cursor.position
        if len(coords) < 2:
            return
        y, x = int(round(coords[-2])), int(round(coords[-1]))
        lmask = state["lmask"]
        if 0 <= y < lmask.shape[0] and 0 <= x < lmask.shape[1]:
            lbl = lmask[y, x]
            if lbl > 0:
                state["selected"] = lbl
                _refresh(viewer)

    def _make_labeler(class_idx):
        def _fn(viewer):
            sel = state["selected"]
            if sel is None:
                return
            cid = _cell_id(state["tiff_name"], state["fov_idx"], sel)
            labels[cid] = class_idx
            # Store true pixel area from original mask
            areas[cid] = int(np.sum(state["lmask"] == sel))
            print(f"  {cid} → {CLASS_NAMES[class_idx]}")
            fov = state["data"][state["fov_idx"]]
            img_ch = np.delete(fov, mask_channel, axis=0)
            crop, _ = extract_cell_crop(img_ch, state["lmask"], sel)
            if crop is not None:
                np.save(crops_dir / f"{cid}.npy", crop)
            _refresh(viewer)
        return _fn

    for key, cls in (("g", 0), ("b", 1)):
        viewer.bind_key(key, _make_labeler(cls), overwrite=True)

    @viewer.bind_key("u")
    def undo(viewer):
        sel = state["selected"]
        if sel is None:
            return
        cid = _cell_id(state["tiff_name"], state["fov_idx"], sel)
        labels.pop(cid, None)
        (crops_dir / f"{cid}.npy").unlink(missing_ok=True)
        _refresh(viewer)

    def _navigate(viewer, direction):
        data = state["data"]
        fi   = state["fov_idx"]
        ti   = state["tiff_idx"]
        if direction == 1:
            if fi + 1 < data.shape[0]:
                state["fov_idx"] = fi + 1
            elif ti + 1 < len(tiff_files):
                state["tiff_idx"] = ti + 1
                state["fov_idx"]  = 0
            else:
                print("  Last FOV.")
                return
        else:
            if fi > 0:
                state["fov_idx"] = fi - 1
            elif ti > 0:
                state["tiff_idx"] = ti - 1
                d2, _ = load_hyperstack(tiff_files[ti - 1])
                state["fov_idx"] = d2.shape[0] - 1
            else:
                print("  First FOV.")
                return
        _save()
        _load_fov(viewer)

    viewer.bind_key("n", lambda v: _navigate(v,  1), overwrite=True)
    viewer.bind_key("p", lambda v: _navigate(v, -1), overwrite=True)
    viewer.bind_key("s", lambda v: _save(),           overwrite=True)

    print("\n=== napari Labeling Tool ===")
    print("Click a cell, then: g=good  b=bad  u=undo")
    print("n=next FOV  p=prev FOV  s=save\n")

    napari.run()
    _save()


# ── Montage labeling mode ─────────────────────────────────────────────────────

def run_montage_labeling(tiff_files, save_dir, phase_channel, mask_channel, cols=10):
    """Generate contact-sheet PNGs + labels.csv template for offline labeling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir   = Path(save_dir)
    crops_dir  = save_dir / "crops"
    montage_dir = save_dir / "montages"
    crops_dir.mkdir(parents=True, exist_ok=True)
    montage_dir.mkdir(parents=True, exist_ok=True)

    ph_in_crop = _phase_idx_in_crop(phase_channel, mask_channel)

    print("Extracting crops...")
    cells = extract_all_cells(tiff_files, phase_channel, mask_channel)
    import random
    random.shuffle(cells)  # Mix cells from different images for labelling diversity
    if not cells:
        print("No cells found.")
        return

    for cell in cells:
        np.save(crops_dir / f"{cell['cell_id']}.npy", cell["crop"])

    n_cells = len(cells)
    rows    = (n_cells + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    if rows == 1:
        axes = axes[np.newaxis, :]
    if cols == 1:
        axes = axes[:, np.newaxis]

    for idx, cell in enumerate(cells):
        r, c = divmod(idx, cols)
        axes[r, c].imshow(cell["thumb"])
        axes[r, c].set_title(str(idx), fontsize=6)
        axes[r, c].axis("off")

    for idx in range(n_cells, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    montage_path = montage_dir / "contact_sheet.png"
    plt.savefig(montage_path, dpi=200, bbox_inches="tight")
    plt.close()

    template_path = save_dir / "labels.csv"
    with open(template_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label_idx", "label_name"])
        for cell in cells:
            writer.writerow([f"{cell['cell_id']}.npy", -1, "unlabeled"])

    n_ch = cells[0]["crop"].shape[0] if cells else 0
    with open(save_dir / "metadata.json", "w") as f:
        json.dump({
            "crop_size": CROP_SIZE, "n_channels": n_ch,
            "class_names": CLASS_NAMES, "n_samples": n_cells, "n_labeled": 0,
        }, f, indent=2)

    print(f"\nExtracted {n_cells} crops.")
    print(f"  Montage:   {montage_path}")
    print(f"  Label CSV: {template_path}")
    print(f"\nEdit {template_path.name}: set label_idx (0=good  1=bad)")
    print("Then run: python train_classifier.py -d labeled_data/ -o models/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Label segmented cells for training the quality classifier."
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Directory containing pipeline output TIFFs")
    parser.add_argument("--save-dir", "-s", required=True,
                        help="Directory to save crops and labels")
    parser.add_argument("--mode", "-m",
                        choices=["grid", "napari", "montage"], default="grid",
                        help="grid (default, fast), napari (full FOV), montage (offline)")
    parser.add_argument("--phase-channel", type=int, default=None,
                        help="Phase contrast channel index (default: second-to-last)")
    parser.add_argument("--mask-channel",  type=int, default=None,
                        help="Binary mask channel index (default: last)")
    parser.add_argument("--pixels-per-um", type=float, default=13.8767,
                        help="Pixel scale for area display in µm² (default: 13.8767)")

    args = parser.parse_args()

    input_dir  = Path(args.input)
    tiff_files = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))

    if not tiff_files:
        print(f"ERROR: No TIFF files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(tiff_files)} TIFF file(s)")
    first_data, first_meta = load_hyperstack(tiff_files[0])
    n_ch = first_data.shape[1]

    mask_channel  = args.mask_channel  if args.mask_channel  is not None else n_ch - 1
    phase_channel = args.phase_channel if args.phase_channel is not None else n_ch - 2

    print(f"Channels: {n_ch} total  |  phase={phase_channel}  mask={mask_channel}")
    if "Labels" in first_meta:
        names = first_meta["Labels"]
        if isinstance(names, (list, tuple)):
            for i, name in enumerate(names):
                tag = (" ← phase" if i == phase_channel
                       else " ← mask" if i == mask_channel else "")
                print(f"  C{i}: {name}{tag}")
    print()

    if args.mode == "grid":
        run_grid_labeling(tiff_files, args.save_dir, phase_channel, mask_channel,
                          pixels_per_um=args.pixels_per_um)
    elif args.mode == "napari":
        run_napari_labeling(tiff_files, args.save_dir, phase_channel, mask_channel)
    else:
        run_montage_labeling(tiff_files, args.save_dir, phase_channel, mask_channel)


if __name__ == "__main__":
    main()
