"""Thin wrapper around pylibCZIrw for reading multi-scene Z-stack CZIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

import numpy as np


@dataclass
class Scene:
    index: int
    array_zcyx: np.ndarray
    channel_names: list[str]
    pixel_size_um: tuple[float | None, float | None]  # (X, Y)
    acquisition_time: str | None


def _parse_channel_names(metadata_xml: str, n_channels: int) -> list[str]:
    try:
        root = ET.fromstring(metadata_xml)
    except ET.ParseError:
        return [f"C{i}" for i in range(n_channels)]
    names: list[str] = []
    for ch in root.iter("Channel"):
        name = ch.attrib.get("Name") or ch.findtext("Fluor") or ch.findtext("ShortName")
        if name:
            names.append(name)
    if len(names) < n_channels:
        names.extend(f"C{i}" for i in range(len(names), n_channels))
    return names[:n_channels]


def _parse_pixel_size_um(metadata_xml: str) -> tuple[float | None, float | None]:
    try:
        root = ET.fromstring(metadata_xml)
    except ET.ParseError:
        return (None, None)
    sizes: dict[str, float] = {}
    for dist in root.iter("Distance"):
        axis = dist.attrib.get("Id")
        value_text = dist.findtext("Value")
        if axis and value_text:
            try:
                # CZI distances are in metres; convert to micrometres.
                sizes[axis] = float(value_text) * 1_000_000.0
            except ValueError:
                pass
    return (sizes.get("X"), sizes.get("Y"))


def _parse_acquisition_time(metadata_xml: str) -> str | None:
    try:
        root = ET.fromstring(metadata_xml)
    except ET.ParseError:
        return None
    node = root.find(".//AcquisitionDateAndTime")
    return node.text if node is not None else None


def _parse_scene_wells(metadata_xml: str) -> dict[int, str]:
    """Best-effort mapping of CZI scene index → well name (e.g. ``"A1"``).

    Zeiss stores well info in several places depending on ZEN version and
    plate template. We look in the most common locations and return whichever
    yields well-shaped names. If nothing matches, return an empty dict and
    callers will fall back to per-scene output.
    """
    try:
        root = ET.fromstring(metadata_xml)
    except ET.ParseError:
        return {}

    out: dict[int, str] = {}
    for scene in root.iter("Scene"):
        idx_attr = scene.attrib.get("Index")
        if idx_attr is None:
            continue
        try:
            idx = int(idx_attr)
        except ValueError:
            continue

        # Candidate locations for the well label, in priority order.
        candidates: list[str | None] = [
            scene.findtext("ArrayName"),
            scene.findtext("Shape/Name"),
            scene.findtext("Well/Name"),
            scene.findtext("WellId"),
            scene.findtext("Name"),
            scene.attrib.get("Name"),
        ]
        for cand in candidates:
            if cand and _looks_like_well(cand):
                out[idx] = cand.strip()
                break
    return out


def _looks_like_well(text: str) -> bool:
    """Heuristic: a well label is a short letter+digits token like A1, H12, AB3."""
    import re

    return bool(re.fullmatch(r"[A-Za-z]{1,2}\d{1,3}", text.strip()))


def _read_scene_from_doc(doc, scene_index: int, channel_names, pixel_size, acq_time) -> Scene:
    bbox = doc.total_bounding_box
    z_range = bbox.get("Z", (0, 1))
    c_range = bbox.get("C", (0, 1))
    t_range = bbox.get("T", (0, 1))
    n_z = z_range[1] - z_range[0]
    n_c = c_range[1] - c_range[0]
    if t_range[1] - t_range[0] != 1:
        raise ValueError(
            f"focuspicker expects a singleton T dimension, got T range {t_range}"
        )

    # Single-position CZIs (no plate metadata) have no
    # ``scenes_bounding_rectangle`` entries; pylibCZIrw raises if we pass
    # any scene index in that case. Detect and pass ``scene=None`` so
    # the document is read as a single image.
    has_scenes = bool(doc.scenes_bounding_rectangle)
    scene_arg = scene_index if has_scenes else None

    sample_plane = np.squeeze(
        doc.read(
            plane={"T": t_range[0], "C": c_range[0], "Z": z_range[0]},
            scene=scene_arg,
        )
    )
    y_size, x_size = sample_plane.shape[-2], sample_plane.shape[-1]
    stack = np.empty((n_z, n_c, y_size, x_size), dtype=sample_plane.dtype)
    for z_idx, z in enumerate(range(z_range[0], z_range[1])):
        for c_idx, c in enumerate(range(c_range[0], c_range[1])):
            plane = doc.read(
                plane={"T": t_range[0], "C": c, "Z": z},
                scene=scene_arg,
            )
            stack[z_idx, c_idx] = np.squeeze(plane)
    return Scene(
        index=scene_index,
        array_zcyx=stack,
        channel_names=channel_names,
        pixel_size_um=pixel_size,
        acquisition_time=acq_time,
    )


def _open(czi_path: Path):
    from pylibCZIrw import czi as pyczi  # imported lazily so tests can stub it

    return pyczi.open_czi(str(czi_path))


def _doc_metadata(doc) -> tuple[list[str], tuple[float | None, float | None], str | None]:
    metadata_xml = doc.raw_metadata
    bbox = doc.total_bounding_box
    c_range = bbox.get("C", (0, 1))
    n_c = c_range[1] - c_range[0]
    return (
        _parse_channel_names(metadata_xml, n_c),
        _parse_pixel_size_um(metadata_xml),
        _parse_acquisition_time(metadata_xml),
    )


def list_scene_indices(czi_path: Path) -> list[int]:
    """Return all scene indices present in the CZI, sorted."""
    with _open(czi_path) as doc:
        rects = doc.scenes_bounding_rectangle or {0: None}
        return sorted(rects)


def list_scene_wells(czi_path: Path) -> dict[int, str]:
    """Return ``{scene_index: well_name}`` for any scenes that have well metadata.

    Scenes without parseable well info are simply absent from the dict.
    """
    with _open(czi_path) as doc:
        return _parse_scene_wells(doc.raw_metadata)


def read_scene(czi_path: Path, scene_index: int) -> Scene:
    """Read a single scene by index."""
    with _open(czi_path) as doc:
        names, pixel_size, acq_time = _doc_metadata(doc)
        return _read_scene_from_doc(doc, scene_index, names, pixel_size, acq_time)


def iter_scenes(czi_path: Path) -> Iterator[Scene]:
    """Yield each scene of a CZI as a (Z, C, Y, X) numpy array plus metadata."""
    with _open(czi_path) as doc:
        names, pixel_size, acq_time = _doc_metadata(doc)
        rects = doc.scenes_bounding_rectangle or {0: None}
        for scene_index in sorted(rects):
            yield _read_scene_from_doc(doc, scene_index, names, pixel_size, acq_time)
