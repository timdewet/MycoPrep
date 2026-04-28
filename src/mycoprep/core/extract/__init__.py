"""Per-cell feature extraction (Phase A: skimage.regionprops baseline).

Phase A produces a Parquet table of regionprops-derived morphology + per-channel
intensity columns plus, optionally, single-cell HDF5 crops in the schema
consumed by ``MorphologicalProfiling_Mtb/python/supcon_ot_pipeline.py``.

Phase B (planned) vendors midline-derived columns from MOMIA.
"""

from .api import (
    ExtractOpts,
    consolidate_crops,
    consolidate_features,
    extract_features_tiff,
)
from .qc_plots import make_qc_plots

__all__ = [
    "ExtractOpts",
    "consolidate_crops",
    "consolidate_features",
    "extract_features_tiff",
    "make_qc_plots",
]
