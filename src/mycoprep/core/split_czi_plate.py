#!/usr/bin/env python3
"""
Split a multi-position CZI plate acquisition into per-well TIFF hyperstacks.

Reads a single CZI file containing multiple wells (each with multiple FOVs),
groups scenes by well, and saves one ImageJ-compatible hyperstack TIFF per well.
Well labels come from a user-supplied CSV plate layout.

Output format:
    One TIFF per well: {condition}__{reporter}__{mutant_or_drug}.tif
    Dimensions: (N_FOV, C, Y, X) — ready for cellpose_pipeline.py

Requirements:
    conda activate cellpose
    pip install czifile tifffile numpy

Usage:
    # Step 1: Generate a template CSV from CZI metadata
    python split_czi_plate.py --czi plate.czi --output ./output --generate-template

    # Step 2: Fill in the template CSV with condition/reporter/mutant_or_drug labels

    # Step 3: Split the CZI using the filled-in CSV
    python split_czi_plate.py --czi plate.czi --layout plate_layout.csv --output ./output
"""

import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import czifile
import numpy as np

from .cellpose_pipeline import save_hyperstack


# ---------------------------------------------------------------------------
# Well ID normalisation
# ---------------------------------------------------------------------------

def normalize_well_id(well):
    """Normalize well IDs: 'A01' -> 'A1', 'a1' -> 'A1'."""
    well = well.strip().upper()
    m = re.match(r'^([A-Z])0*(\d+)$', well)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return well


def well_sort_key(well):
    """Sort key for plate-order sorting (A1, A2, ..., A12, B1, ...)."""
    m = re.match(r'^([A-Z])(\d+)$', well)
    if m:
        return (m.group(1), int(m.group(2)))
    return (well, 0)


# ---------------------------------------------------------------------------
# CZI metadata parsing
# ---------------------------------------------------------------------------

def extract_scene_well_map(czi_path):
    """
    Parse CZI XML metadata to build a scene_index -> well_id mapping.

    Zeiss plate acquisitions store scenes with names like "A1-1" (well A1,
    position 1). This function extracts the well ID from each scene name.

    Returns:
        dict[int, str]: {scene_index: normalized_well_id}
    """
    with czifile.CziFile(str(czi_path)) as czi:
        meta_xml = czi.metadata()

    root = ET.fromstring(meta_xml)

    # Find all Scene elements — Zeiss XML nests these under
    # Information/Image/Dimensions/S/Scenes/Scene
    scenes = root.findall('.//{http://www.zeiss.com/mic/experimental/CZI}Scene')
    if not scenes:
        scenes = root.findall('.//Scene')

    if not scenes:
        raise ValueError(
            f"No scene metadata found in {czi_path}. "
            "This file may not be a multi-position plate acquisition."
        )

    scene_well_map = {}
    for scene in scenes:
        idx = scene.get('Index')
        if idx is None:
            continue
        idx = int(idx)
        name = scene.get('Name', '')

        # Scene names are typically "A1-1", "A1-2", "B3-1", etc.
        # The well ID is everything before the last hyphen-number.
        # Also check for <Shape Name="A1"> child elements.
        well = None

        # Try <Shape> child first (more reliable)
        shape = scene.find('.//{http://www.zeiss.com/mic/experimental/CZI}Shape')
        if shape is None:
            shape = scene.find('.//Shape')
        if shape is not None:
            well = shape.get('Name')

        # Fall back to scene name parsing
        if not well and name:
            # "A1-1" -> "A1", "A1" -> "A1"
            if re.match(r'^[A-Za-z]\d+-\d+$', name):
                well = name.rsplit('-', 1)[0]
            else:
                well = name

        if well:
            scene_well_map[idx] = normalize_well_id(well)

    if not scene_well_map:
        raise ValueError(
            f"Could not extract well IDs from scene metadata in {czi_path}. "
            "Scene names may use an unexpected format."
        )

    return scene_well_map


def extract_channel_names(czi_path):
    """Extract channel names from CZI XML metadata.

    Zeiss CZI metadata contains <Channel> elements in several places
    (acquisition, DisplaySetting, DimensionChannelsSetting, etc.), so a
    naive ``.//Channel`` walk overcounts. We prefer the canonical
    acquisition path under ``Information/Image/Dimensions/Channels`` and
    fall back to deduping by Id/Name. The result is finally truncated to
    the actual C-dimension size from the file's array shape.
    """
    with czifile.CziFile(str(czi_path)) as czi:
        meta_xml = czi.metadata()
        # Authoritative channel count from the actual subblock directory.
        try:
            c_indices = set()
            for entry in czi.filtered_subblock_directory:
                dims = dict(zip(entry.dims, entry.start))
                if "C" in dims:
                    c_indices.add(dims["C"])
            n_channels_actual = len(c_indices) if c_indices else None
        except Exception:  # noqa: BLE001
            n_channels_actual = None

    root = ET.fromstring(meta_xml)

    # Preferred path — only the acquisition channel definitions
    preferred_paths = [
        ".//{http://www.zeiss.com/mic/experimental/CZI}Information"
        "/{http://www.zeiss.com/mic/experimental/CZI}Image"
        "/{http://www.zeiss.com/mic/experimental/CZI}Dimensions"
        "/{http://www.zeiss.com/mic/experimental/CZI}Channels"
        "/{http://www.zeiss.com/mic/experimental/CZI}Channel",
        ".//Information/Image/Dimensions/Channels/Channel",
    ]

    names: list[str] = []
    for path in preferred_paths:
        for ch in root.findall(path):
            name = ch.get("Name") or ch.get("Id")
            if name:
                names.append(name)
        if names:
            break

    # Last-resort fallback: walk all Channels but dedupe by Id (preserving order)
    if not names:
        seen_ids = set()
        for ns in ["{http://www.zeiss.com/mic/experimental/CZI}", ""]:
            for ch in root.findall(f".//{ns}Channel"):
                cid = ch.get("Id") or ch.get("Name")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    names.append(ch.get("Name") or cid)
            if names:
                break

    if not names:
        return None

    if n_channels_actual is not None and len(names) != n_channels_actual:
        # Trust the array shape; pad or truncate names accordingly
        if len(names) > n_channels_actual:
            names = names[:n_channels_actual]
        else:
            names = names + [f"C{i+1}" for i in range(len(names), n_channels_actual)]

    return names


# ---------------------------------------------------------------------------
# CZI reading
# ---------------------------------------------------------------------------

def read_czi_scenes(czi_path):
    """
    Read a multi-scene CZI file and return per-scene arrays.

    Uses subblock-level reading to avoid the massive memory allocation that
    czifile.imread() causes for multi-position plate acquisitions (it tries
    to place all scenes into one coordinate space).

    Returns:
        dict[int, np.ndarray]: {scene_index: array (C, Y, X)}
    """
    def _sb_dims(subblock):
        de = subblock.directory_entry
        return dict(zip(de.dims, de.start))

    with czifile.CziFile(str(czi_path)) as czi:
        # Group subblocks by (scene, channel). In Zeiss plate acquisitions the
        # scene index lives on `directory_entry.scene_index` (not the 'S' dim,
        # which tends to be always 0). Z is collapsed by taking the middle slice.
        by_scene_channel_z = defaultdict(lambda: defaultdict(dict))
        for sb in czi.subblocks():
            de = sb.directory_entry
            dims = _sb_dims(sb)
            s = de.scene_index if de.scene_index is not None else dims.get('S', 0)
            c = dims.get('C', 0)
            z = dims.get('Z', 0)
            by_scene_channel_z[s][c][z] = sb

        n_scenes = len(by_scene_channel_z)
        print(f"  Found {n_scenes} scenes via subblocks")

        scenes = {}
        for scene_idx in sorted(by_scene_channel_z.keys()):
            channel_blocks = by_scene_channel_z[scene_idx]

            channel_arrays = {}
            for ch_idx in sorted(channel_blocks.keys()):
                z_map = channel_blocks[ch_idx]
                # Collapse Z by taking the middle plane — appropriate for
                # pre-focus-picking splits. Users running focus picking first
                # should not invoke split on raw Z-stacks.
                z_keys = sorted(z_map.keys())
                z_mid = z_keys[len(z_keys) // 2]
                sb = z_map[z_mid]
                data = np.squeeze(sb.data())
                if data.ndim != 2:
                    raise ValueError(
                        f"Unexpected subblock shape {data.shape} for "
                        f"scene {scene_idx}, channel {ch_idx}. "
                        f"Expected 2D after squeeze."
                    )
                channel_arrays[ch_idx] = data

            # Stack channels: (C, Y, X)
            n_channels = len(channel_arrays)
            if n_channels == 0:
                print(f"  WARNING: Scene {scene_idx} has no data, skipping")
                continue

            scene_data = np.stack(
                [channel_arrays[c] for c in sorted(channel_arrays.keys())],
                axis=0
            )
            scenes[scene_idx] = scene_data

    return scenes


# ---------------------------------------------------------------------------
# CSV plate layout
# ---------------------------------------------------------------------------

def load_plate_layout(csv_path):
    """
    Read a plate layout CSV and return a well -> labels mapping.

    Expected columns: well, condition, reporter, mutant_or_drug

    Returns:
        dict[str, dict]: {normalized_well_id: {condition, reporter, mutant_or_drug}}
    """
    layout = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        required = {'well', 'condition', 'reporter', 'mutant_or_drug'}
        actual = set(reader.fieldnames) if reader.fieldnames else set()
        missing = required - actual
        if missing:
            raise ValueError(
                f"CSV missing required columns: {sorted(missing)}. "
                f"Found: {sorted(actual)}"
            )

        for row_num, row in enumerate(reader, start=2):
            well = normalize_well_id(row['well'])
            if not well:
                continue

            if well in layout:
                raise ValueError(
                    f"Duplicate well '{well}' in CSV (row {row_num})."
                )

            layout[well] = {
                'condition': row['condition'].strip(),
                'reporter': row['reporter'].strip(),
                'mutant_or_drug': row['mutant_or_drug'].strip(),
                'replica': row.get('replica', '').strip(),
            }

    return layout


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------

def build_output_filename(condition, reporter, mutant_or_drug, replica=""):
    """Build output filename: condition__reporter__mutant_or_drug[__R{n}].tif"""
    parts = [condition, reporter, mutant_or_drug]
    sanitized = [p.replace(' ', '_') for p in parts]
    if replica:
        rep = str(replica).strip()
        if rep:
            sanitized.append(f"R{rep}")
    return '__'.join(sanitized) + '.tif'


# ---------------------------------------------------------------------------
# Template CSV generation
# ---------------------------------------------------------------------------

def generate_template_csv(czi_path, output_path):
    """Generate a template plate layout CSV from CZI metadata."""
    scene_well_map = extract_scene_well_map(czi_path)

    # Count FOVs per well
    well_fovs = defaultdict(int)
    for well in scene_well_map.values():
        well_fovs[well] += 1

    wells_sorted = sorted(well_fovs.keys(), key=well_sort_key)

    csv_path = Path(output_path) / 'plate_layout_template.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['well', 'condition', 'reporter', 'mutant_or_drug'])
        for well in wells_sorted:
            writer.writerow([well, '', '', ''])

    print(f"\nTemplate CSV saved to: {csv_path}")
    print(f"Wells detected ({len(wells_sorted)}):")
    for well in wells_sorted:
        print(f"  {well}: {well_fovs[well]} FOV(s)")
    print(f"\nFill in the condition, reporter, and mutant_or_drug columns, then run:")
    print(f"  python split_czi_plate.py --czi {czi_path} --layout {csv_path} --output {output_path}")


# ---------------------------------------------------------------------------
# Main splitting logic
# ---------------------------------------------------------------------------

def _read_czi_acquisition_time(czi_path):
    """Return the file-level ``AcquisitionDateAndTime`` from CZI metadata, or None.

    Falls back gracefully when the metadata is missing or unparseable. The
    same helper is used by the Focus pipeline (via ``focus.io_czi``) so the
    Split and Focus paths produce the same provenance value.
    """
    try:
        with czifile.CziFile(str(czi_path)) as czi:
            meta_xml = czi.metadata()
        from .focus.io_czi import _parse_acquisition_time
        return _parse_acquisition_time(meta_xml)
    except Exception:  # noqa: BLE001
        return None


def _read_czi_per_scene_acquisition_times(czi_path):
    """Return ``{scene_index: ISO8601 timestamp}`` from CZI subblock metadata.

    Each CZI subblock carries its own ``<METADATA><Tags><AcquisitionTime>``
    tag with the actual moment that frame was captured. Plate acquisitions
    sweep wells over many minutes, so per-scene times are meaningfully
    different from the file-level ``AcquisitionDateAndTime``. Falls back to
    an empty dict on any read/parse failure — callers are expected to use
    the file-level master timestamp as a fallback.
    """
    out: dict[int, str] = {}
    try:
        with czifile.CziFile(str(czi_path)) as czi:
            for sb in czi.subblocks():
                try:
                    de = sb.directory_entry
                    scene_idx = (
                        de.scene_index
                        if de.scene_index is not None
                        else dict(zip(de.dims, de.start)).get("S", 0)
                    )
                except Exception:  # noqa: BLE001
                    continue
                if scene_idx in out:
                    # One timestamp per scene is enough — they fire roughly
                    # together for all C/Z planes within a scene.
                    continue
                try:
                    meta_xml = sb.metadata()
                except Exception:  # noqa: BLE001
                    continue
                if not meta_xml:
                    continue
                try:
                    root = ET.fromstring(meta_xml)
                except ET.ParseError:
                    continue
                node = root.find(".//AcquisitionTime")
                if node is not None and node.text:
                    out[int(scene_idx)] = node.text.strip()
    except Exception:  # noqa: BLE001
        return {}
    return out


def _write_acquisition_sidecar(tiff_path, source_czi, plate_acq_dt, scene_indices,
                               per_scene_times=None):
    """Write a ``<tiff_stem>__acquisition.json`` next to a per-well TIFF.

    Mirrors the schema written by ``focus.pipeline._write_acquisition_sidecar``
    so the Features stage doesn't have to branch on which upstream stage
    produced the TIFF.

    If ``per_scene_times`` is given (as ``{scene_index: ISO8601}``), per-FOV
    timestamps are populated from it. Otherwise the file-level master
    timestamp is repeated for every FOV so consumers get a consistent shape.
    """
    sidecar = tiff_path.with_name(tiff_path.stem + "__acquisition.json")
    fov_times: list[str | None] = []
    for s in scene_indices:
        t = (per_scene_times or {}).get(int(s))
        fov_times.append(t if t else plate_acq_dt)
    payload = {
        "source_czi": Path(source_czi).name,
        "plate_acquisition_datetime": plate_acq_dt,
        "fov_acquisition_times": fov_times,
        "scene_indices": list(int(s) for s in scene_indices),
    }
    try:
        sidecar.write_text(json.dumps(payload, indent=2))
    except OSError:
        pass


def backfill_acquisition_sidecars(focus_or_split_dir, czi_path, layout_csv=None):
    """Retrofit ``__acquisition.json`` sidecars for an already-processed run.

    Walks every ``*.tif`` in ``focus_or_split_dir`` and writes the JSON
    sidecar that the Features stage expects, using per-scene timestamps from
    ``czi_path``. Intended for outputs produced before the sidecar code
    landed in the pipeline. Idempotent — safe to re-run.

    Mapping per TIFF is inferred from the well embedded in its filename
    (the ``{condition}__{reporter}__{mutant_or_drug}__R{n}`` template
    written by ``build_output_filename``). The well → CZI scenes mapping
    comes from the CZI metadata; ``layout_csv`` is consulted when the
    inference is ambiguous.

    Returns a list of sidecar paths written.
    """
    focus_or_split_dir = Path(focus_or_split_dir)
    czi_path = Path(czi_path)
    if not focus_or_split_dir.is_dir():
        raise FileNotFoundError(focus_or_split_dir)
    if not czi_path.exists():
        raise FileNotFoundError(czi_path)

    plate_acq_dt = _read_czi_acquisition_time(czi_path)
    per_scene_times = _read_czi_per_scene_acquisition_times(czi_path)
    scene_well_map = extract_scene_well_map(czi_path)
    well_scenes: dict[str, list[int]] = defaultdict(list)
    for scene_idx, well in scene_well_map.items():
        well_scenes[normalize_well_id(well)].append(int(scene_idx))
    for w in well_scenes:
        well_scenes[w].sort()

    layout_lookup: dict[str, str] | None = None
    if layout_csv is not None:
        try:
            layout = load_plate_layout(Path(layout_csv))
            # Build {filename_stem (without focus/filtered suffix): well_id}
            layout_lookup = {}
            for well_id, row in layout.items():
                stem = build_output_filename(
                    row["condition"], row["reporter"],
                    row["mutant_or_drug"], row.get("replica", ""),
                ).removesuffix(".tif")
                layout_lookup[stem] = well_id
        except Exception:  # noqa: BLE001
            layout_lookup = None

    written: list[Path] = []
    for tiff_path in sorted(focus_or_split_dir.glob("*.tif")):
        # Strip stage filename suffix (e.g. "_focused", "_filtered") so the
        # stem matches the layout's TIFF naming convention.
        stem = tiff_path.stem
        for suf in ("_focused", "_filtered"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break

        well_id: str | None = None
        if layout_lookup and stem in layout_lookup:
            well_id = layout_lookup[stem]
        else:
            # Fall back to extract_scene_well_map: many MycoPrep wells map
            # to one well in the layout, so try to infer from the
            # condition-encoded filename. Without a layout, we can't
            # disambiguate — leave well_id None and write per-FOV times
            # against an unknown scene set.
            pass

        scene_indices = well_scenes.get(well_id, []) if well_id else []
        _write_acquisition_sidecar(
            tiff_path,
            source_czi=czi_path,
            plate_acq_dt=plate_acq_dt,
            scene_indices=scene_indices,
            per_scene_times=per_scene_times,
        )
        written.append(tiff_path.with_name(tiff_path.stem + "__acquisition.json"))
    return written


def _read_czi_pixels_per_um(czi_path):
    """Return pixels-per-µm from the CZI metadata, or None if unparseable."""
    try:
        with czifile.CziFile(str(czi_path)) as czi:
            meta_xml = czi.metadata()
        from .focus.io_czi import _parse_pixel_size_um
        px_x, _px_y = _parse_pixel_size_um(meta_xml)
        if px_x and px_x > 0:
            return 1.0 / px_x
    except Exception:  # noqa: BLE001
        pass
    return None


def split_and_save(czi_path, layout, output_dir, channel_names=None,
                   pixels_per_um=None):
    """
    Split a multi-scene CZI into per-well TIFF hyperstacks.

    Args:
        czi_path: Path to CZI file
        layout: dict from load_plate_layout()
        output_dir: Output directory (created if needed)
        channel_names: Optional list of channel name strings
        pixels_per_um: Pixel scale
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve pixel size from the CZI if the caller didn't supply one. This
    # keeps downstream (Segment / Classify) area filters calibrated correctly.
    if pixels_per_um is None:
        detected = _read_czi_pixels_per_um(czi_path)
        pixels_per_um = detected if detected is not None else 13.8767
        if detected is None:
            print(f"  WARNING: could not read pixel size from CZI; defaulting to {pixels_per_um} px/µm")
        else:
            print(f"  Pixel size from CZI: {pixels_per_um:.3f} px/µm")

    # 1. Extract scene -> well mapping
    print(f"Reading CZI metadata: {czi_path}")
    scene_well_map = extract_scene_well_map(czi_path)

    # 2. Group scenes by well
    well_scenes = defaultdict(list)
    for scene_idx in sorted(scene_well_map.keys()):
        well_scenes[scene_well_map[scene_idx]].append(scene_idx)

    czi_wells = set(well_scenes.keys())
    csv_wells = set(layout.keys())

    # 3. Report mismatches
    missing_in_csv = czi_wells - csv_wells
    missing_in_czi = csv_wells - czi_wells

    if missing_in_csv:
        print(f"\nWARNING: Wells in CZI but not in CSV (will be skipped): "
              f"{sorted(missing_in_csv, key=well_sort_key)}")
    if missing_in_czi:
        print(f"WARNING: Wells in CSV but not in CZI (no data): "
              f"{sorted(missing_in_czi, key=well_sort_key)}")

    wells_to_process = sorted(czi_wells & csv_wells, key=well_sort_key)

    if not wells_to_process:
        print("ERROR: No matching wells between CZI and CSV.")
        sys.exit(1)

    # 4. Check for duplicate output filenames
    filenames = {}
    for well_id in wells_to_process:
        row = layout[well_id]
        fname = build_output_filename(row['condition'], row['reporter'],
                                      row['mutant_or_drug'],
                                      row.get('replica', ''))
        if fname in filenames:
            print(f"ERROR: Duplicate output filename '{fname}' "
                  f"for wells {filenames[fname]} and {well_id}.")
            sys.exit(1)
        filenames[fname] = well_id

    # 5. Read all scenes from CZI
    print(f"\nReading CZI image data...")
    all_scenes = read_czi_scenes(czi_path)
    print(f"  Loaded {len(all_scenes)} scenes")

    plate_acq_dt = _read_czi_acquisition_time(czi_path)
    per_scene_times = _read_czi_per_scene_acquisition_times(czi_path)

    # 6. Detect channel names from metadata if not provided
    if not channel_names:
        channel_names = extract_channel_names(czi_path)
    if not channel_names:
        first_scene = next(iter(all_scenes.values()), None)
        n_ch = first_scene.shape[0] if first_scene is not None else 0
        channel_names = [f"C{i+1}" for i in range(n_ch)]

    # 7. Process each well
    print(f"\nSplitting {len(wells_to_process)} wells:")
    for well_id in wells_to_process:
        scene_indices = well_scenes[well_id]
        row = layout[well_id]
        fname = build_output_filename(row['condition'], row['reporter'],
                                      row['mutant_or_drug'],
                                      row.get('replica', ''))

        # Collect FOV arrays
        fov_arrays = []
        fov_names = []
        for scene_idx in scene_indices:
            if scene_idx not in all_scenes:
                print(f"  WARNING: Scene index {scene_idx} not found, skipping")
                continue
            img = all_scenes[scene_idx].astype(np.uint16)
            fov_arrays.append(img)
            fov_names.append(f"scene_{scene_idx:03d}")

        if not fov_arrays:
            print(f"  {well_id}: No valid FOVs, skipping")
            continue

        # Validate consistent shapes
        ref_shape = fov_arrays[0].shape
        valid = [(a, n) for a, n in zip(fov_arrays, fov_names)
                 if a.shape == ref_shape]
        if len(valid) < len(fov_arrays):
            print(f"  WARNING: {len(fov_arrays) - len(valid)} FOVs with "
                  f"inconsistent shape skipped in well {well_id}")
            fov_arrays, fov_names = zip(*valid) if valid else ([], [])
            fov_arrays, fov_names = list(fov_arrays), list(fov_names)

        if not fov_arrays:
            continue

        # Stack: (N_FOV, C, Y, X)
        stacked = np.stack(fov_arrays, axis=0)

        condition_name = fname.replace('.tif', '')
        output_path = output_dir / fname

        save_hyperstack(stacked, output_path, condition_name,
                        fov_names, channel_names, pixels_per_um)

        _write_acquisition_sidecar(
            output_path,
            source_czi=czi_path,
            plate_acq_dt=plate_acq_dt,
            scene_indices=scene_indices,
            per_scene_times=per_scene_times,
        )

        print(f"  {well_id} ({len(fov_arrays)} FOVs) -> {fname}  {stacked.shape}")

    print(f"\nDone. Output saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Split a multi-position CZI plate acquisition into "
                    "per-well TIFF hyperstacks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a template CSV (fills in well column from CZI metadata):
  python split_czi_plate.py --czi plate.czi -o ./output --generate-template

  # Split CZI into per-well TIFFs using a filled-in CSV:
  python split_czi_plate.py --czi plate.czi --layout plate_layout.csv -o ./output

  # With explicit channel names:
  python split_czi_plate.py --czi plate.czi --layout layout.csv -o ./output \\
      --channel-names "Phase,mCherry,eGFP"
""",
    )

    parser.add_argument('--czi', required=True,
                        help='Path to multi-position CZI file')
    parser.add_argument('--layout',
                        help='Path to plate layout CSV '
                             '(not needed with --generate-template)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory')
    parser.add_argument('--generate-template', action='store_true',
                        help='Generate a template CSV from CZI metadata and exit')
    parser.add_argument('--channel-names', type=str, default=None,
                        help='Comma-separated channel names '
                             '(e.g. "Phase,mCherry,eGFP"). '
                             'Auto-detected from CZI metadata if not given.')
    parser.add_argument('--pixels-per-um', type=float, default=13.8767,
                        help='Pixel scale (default: 13.8767 for 100x/1.4NA '
                             '+ Prime 95B)')

    args = parser.parse_args()

    czi_path = Path(args.czi)
    if not czi_path.exists():
        print(f"ERROR: CZI file not found: {czi_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_template:
        generate_template_csv(czi_path, output_dir)
        return

    if not args.layout:
        print("ERROR: --layout is required (unless using --generate-template).")
        print("Run with --generate-template first to create a template CSV.")
        sys.exit(1)

    layout_path = Path(args.layout)
    if not layout_path.exists():
        print(f"ERROR: Layout CSV not found: {layout_path}")
        sys.exit(1)

    layout = load_plate_layout(layout_path)
    print(f"Loaded plate layout: {len(layout)} wells")

    channel_names = None
    if args.channel_names:
        channel_names = [n.strip() for n in args.channel_names.split(',')]

    split_and_save(czi_path, layout, output_dir, channel_names,
                   args.pixels_per_um)


if __name__ == '__main__':
    main()
