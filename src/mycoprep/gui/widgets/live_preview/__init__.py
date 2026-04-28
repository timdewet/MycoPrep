"""Live preview package — persistent right-column preview of the pipeline.

The package is composed of:

- :mod:`canvas` — a stacked pyqtgraph image view (phase + fluor channels +
  mask boundaries + per-cell text labels).
- :mod:`panel` — the right-column widget hosting the sample/FOV picker,
  the canvas, and the channel/label controls.

Subsequent phases will add :mod:`cache`, :mod:`controller`,
:mod:`worker`, :mod:`focus_single_fov`, and :mod:`features_single_fov`.
"""

from .canvas import MultiOverlayCanvas
from .panel import LivePreviewPanel

__all__ = ["MultiOverlayCanvas", "LivePreviewPanel"]
