"""Shared pytest fixtures for MycoPrep tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic numpy Generator for reproducible synthetic data."""
    return np.random.default_rng(seed=12345)


@pytest.fixture
def labeled_mask_3cells():
    """A 50x60 labeled mask with three non-touching rectangular cells.

    Layout (label : bbox):
        1 : rows 5..15,  cols 5..15   (10x10  = 100 px)
        2 : rows 20..30, cols 20..40  (10x20  = 200 px)
        3 : rows 35..45, cols 45..55  (10x10  = 100 px)
    """
    mask = np.zeros((50, 60), dtype=np.int32)
    mask[5:15, 5:15] = 1
    mask[20:30, 20:40] = 2
    mask[35:45, 45:55] = 3
    return mask


@pytest.fixture
def edge_touching_mask():
    """50x60 mask: label 1 in interior, label 2 touches top edge, label 3 touches right edge."""
    mask = np.zeros((50, 60), dtype=np.int32)
    mask[10:20, 10:20] = 1     # interior
    mask[0:5, 20:30] = 2       # touches top
    mask[30:40, 55:60] = 3     # touches right
    return mask
