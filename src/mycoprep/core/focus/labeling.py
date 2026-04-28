"""Manual ground-truth labelling tool for FocusPicker review directories.

The user runs ``focuspicker review`` to dump per-FOV phase Z-stacks, then runs
this labeller to scrub through each stack and mark the truly best in-focus
slice (or mark the FOV as having no in-focus slice at all). Labels are saved
to ``manual_labels.csv`` in the review directory and can later be compared
against each metric's pick by ``focuspicker.evaluation``.

The UI is intentionally **hint-blind**: it never shows which slice any focus
metric chose, so labels are anchored to the user's eye, not to the algorithm.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

NO_FOCUS = -1
LABELS_FILENAME = "manual_labels.csv"
_SCENE_RE = re.compile(r"scene(\d+)_phase_stack\.tif$")


@dataclass
class _SceneEntry:
    scene_index: int
    stack_path: Path


def _discover_scenes(review_dir: Path) -> list[_SceneEntry]:
    entries: list[_SceneEntry] = []
    for path in review_dir.glob("scene*_phase_stack.tif"):
        m = _SCENE_RE.search(path.name)
        if m:
            entries.append(_SceneEntry(int(m.group(1)), path))
    entries.sort(key=lambda e: e.scene_index)
    return entries


def load_manual_labels(path: Path) -> dict[int, int]:
    """Read ``manual_labels.csv`` and return ``{scene_index: chosen_z}``.

    A ``chosen_z`` of ``-1`` (``NO_FOCUS``) means the user marked the FOV as
    having no in-focus slice.
    """
    labels: dict[int, int] = {}
    if not path.exists():
        return labels
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels[int(row["scene"])] = int(row["chosen_z"])
    return labels


def _save_labels_atomic(path: Path, labels: dict[int, int]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scene", "chosen_z", "note"])
        for scene_index in sorted(labels):
            chosen_z = labels[scene_index]
            note = "no_focus" if chosen_z == NO_FOCUS else ""
            writer.writerow([scene_index, chosen_z, note])
    os.replace(tmp, path)


def label_review_dir(review_dir: Path, resume: bool = True) -> Path:
    """Open the matplotlib labeller for every phase stack in ``review_dir``.

    If ``resume`` is True (the default) and a labels CSV already exists, any
    scene that already has a label is skipped on entry. The user can still
    walk back to it with the ``b`` key and overwrite the label.

    Returns the path to the labels CSV.
    """
    import matplotlib.pyplot as plt

    review_dir = Path(review_dir)
    if not review_dir.is_dir():
        raise NotADirectoryError(f"{review_dir} is not a directory")

    entries = _discover_scenes(review_dir)
    if not entries:
        raise RuntimeError(
            f"no scene*_phase_stack.tif files found in {review_dir}"
        )

    labels_path = review_dir / LABELS_FILENAME
    labels = load_manual_labels(labels_path) if resume else {}

    # Pre-load stacks lazily to keep memory bounded; cache only the current one.
    state = {
        "i": 0,
        "z": 0,
        "stack": None,  # current (Z, Y, X) array
        "vmin": 0.0,
        "vmax": 1.0,
        "quit": False,
    }

    if resume:
        # Advance to the first unlabelled scene.
        while state["i"] < len(entries) and entries[state["i"]].scene_index in labels:
            state["i"] += 1
        if state["i"] >= len(entries):
            print(f"[focuspicker] all {len(entries)} scenes already labelled")
            return labels_path

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(top=0.92, bottom=0.08)
    img_artist = None

    def _load_current() -> None:
        nonlocal img_artist
        entry = entries[state["i"]]
        stack = tifffile.imread(str(entry.stack_path))
        if stack.ndim != 3:
            raise RuntimeError(
                f"expected (Z, Y, X) stack in {entry.stack_path.name}, got shape {stack.shape}"
            )
        state["stack"] = stack
        # Per-stack autoscale; never per-slice (per-slice rescaling fakes
        # sharpness contrast).
        state["vmin"] = float(np.percentile(stack, 1))
        state["vmax"] = float(np.percentile(stack, 99))
        if state["vmax"] <= state["vmin"]:
            state["vmax"] = state["vmin"] + 1.0
        # Default cursor: middle of stack.
        state["z"] = stack.shape[0] // 2
        ax.clear()
        img_artist = ax.imshow(
            stack[state["z"]],
            cmap="gray",
            vmin=state["vmin"],
            vmax=state["vmax"],
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        _refresh_title()

    def _refresh_slice() -> None:
        stack = state["stack"]
        img_artist.set_data(stack[state["z"]])
        _refresh_title()
        fig.canvas.draw_idle()

    def _refresh_title() -> None:
        entry = entries[state["i"]]
        n_z = state["stack"].shape[0]
        ax.set_title(
            f"scene {entry.scene_index}   z={state['z'] + 1}/{n_z}   "
            f"({state['i'] + 1}/{len(entries)})\n"
            "←/→ scrub  ↑/↓ jump  Enter/Space=accept  n=no focus  "
            "s=skip  b=back  q=save & quit"
        )

    def _commit(chosen_z: int) -> None:
        entry = entries[state["i"]]
        labels[entry.scene_index] = chosen_z
        _save_labels_atomic(labels_path, labels)
        _advance()

    def _advance() -> None:
        state["i"] += 1
        if state["i"] >= len(entries):
            state["quit"] = True
            plt.close(fig)
            return
        _load_current()
        fig.canvas.draw_idle()

    def _back() -> None:
        if state["i"] == 0:
            return
        state["i"] -= 1
        _load_current()
        fig.canvas.draw_idle()

    def _on_key(event) -> None:
        if state["stack"] is None:
            return
        n_z = state["stack"].shape[0]
        key = event.key
        if key == "right":
            state["z"] = min(n_z - 1, state["z"] + 1)
            _refresh_slice()
        elif key == "left":
            state["z"] = max(0, state["z"] - 1)
            _refresh_slice()
        elif key == "up":
            state["z"] = min(n_z - 1, state["z"] + 3)
            _refresh_slice()
        elif key == "down":
            state["z"] = max(0, state["z"] - 3)
            _refresh_slice()
        elif key in ("enter", " "):
            _commit(state["z"])
        elif key == "n":
            _commit(NO_FOCUS)
        elif key == "s":
            _advance()
        elif key == "b":
            _back()
        elif key == "q":
            state["quit"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    _load_current()
    plt.show()

    n_focus = sum(1 for v in labels.values() if v != NO_FOCUS)
    n_no_focus = sum(1 for v in labels.values() if v == NO_FOCUS)
    print(
        f"[focuspicker] saved {len(labels)} labels to {labels_path} "
        f"({n_focus} with focus, {n_no_focus} marked no_focus)"
    )
    return labels_path
