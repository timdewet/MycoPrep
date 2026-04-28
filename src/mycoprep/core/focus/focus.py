"""Focus metrics for selecting the sharpest slice of a Z-stack.

Six classical sharpness metrics are computed for every slice. Each one is
restricted to a foreground mask (computed once per stack) so that the score
is not diluted by empty agar; the per-Z score curves are then mildly smoothed
along Z to suppress single-slice noise spikes; and a 7th "ensemble" score is
derived as the mean of per-metric min-max-normalised scores. ``ensemble`` is
the default for slice selection because no single classical metric is
reliable across every FOV — the rank-averaged combination is consistently
better than the best individual metric on real bacteria phase-contrast
images.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, laplace, median_filter, sobel
from skimage.filters import threshold_otsu

METRIC_NAMES = (
    "normalized_variance",
    "brenner",
    "tenengrad",
    "laplacian_variance",
    "sml",
    "vollath_f4",
    "ensemble",
)
DEFAULT_METRIC = "vollath_f4"

# Names of the raw classical metrics (everything except the ensemble).
_RAW_METRICS = tuple(m for m in METRIC_NAMES if m != "ensemble")


def _as_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float64, copy=False)


def _preprocess(image: np.ndarray, crop_fraction: float, preblur_sigma: float) -> np.ndarray:
    img = _as_float(image)
    if 0 < crop_fraction < 1.0:
        h, w = img.shape[-2:]
        new_h = max(1, int(round(h * crop_fraction)))
        new_w = max(1, int(round(w * crop_fraction)))
        y0 = (h - new_h) // 2
        x0 = (w - new_w) // 2
        img = img[..., y0:y0 + new_h, x0:x0 + new_w]
    if preblur_sigma > 0:
        img = gaussian_filter(img, sigma=preblur_sigma)
    return img


def compute_focus_mask(
    stack_zyx: np.ndarray,
    sigma_lo: float = 2.0,
    sigma_hi: float = 16.0,
    dilate_iters: int = 3,
) -> np.ndarray:
    """Build a 2D foreground mask shared across all Z slices of a stack.

    Strategy: bandpass-filter every slice (highlights cell-scale structure
    regardless of phase-contrast polarity), take the maximum absolute response
    across Z (so the mask is the *union* of where structure exists, not biased
    toward whichever slice happens to be sharpest), Otsu-threshold, and
    dilate. If the mask collapses to nothing, fall back to scoring the whole
    frame.
    """
    img = _as_float(stack_zyx)
    lo = gaussian_filter(img, sigma=(0, sigma_lo, sigma_lo))
    hi = gaussian_filter(img, sigma=(0, sigma_hi, sigma_hi))
    response = np.abs(lo - hi).max(axis=0)
    if response.max() <= 0:
        return np.ones(response.shape, dtype=bool)
    try:
        thresh = float(threshold_otsu(response))
    except ValueError:
        return np.ones(response.shape, dtype=bool)
    mask = response > thresh
    if dilate_iters > 0:
        mask = binary_dilation(mask, iterations=dilate_iters)
    if mask.sum() < 100:
        return np.ones(response.shape, dtype=bool)
    return mask


def normalized_variance(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Variance normalised by mean intensity. Robust to overall brightness."""
    img = _as_float(image)
    vals = img[mask] if mask is not None else img
    mean = float(vals.mean())
    if mean <= 0:
        return 0.0
    return float(vals.var() / mean)


def brenner(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Brenner gradient: sum of squared intensity differences at lag 2."""
    img = _as_float(image)
    diff = img[2:, :] - img[:-2, :]
    sq = diff * diff
    if mask is not None:
        return float(sq[mask[1:-1, :]].sum())
    return float(sq.sum())


def tenengrad(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Sum of squared Sobel gradient magnitudes."""
    img = _as_float(image)
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    g2 = gx * gx + gy * gy
    if mask is not None:
        return float(g2[mask].sum())
    return float(g2.sum())


def laplacian_variance(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Variance of the Laplacian. Standard sharpness measure, often best baseline."""
    img = _as_float(image)
    lap = laplace(img)
    if mask is not None:
        vals = lap[mask]
        if vals.size < 2:
            return 0.0
        return float(vals.var())
    return float(lap.var())


def sml(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Sum of Modified Laplacian (Nayar 1994).

    Designed specifically for microscopy / phase contrast. Often the strongest
    classical metric for in-focus selection of cellular images.
    """
    img = _as_float(image)
    ml_x = np.abs(2 * img[:, 1:-1] - img[:, :-2] - img[:, 2:])
    ml_y = np.abs(2 * img[1:-1, :] - img[:-2, :] - img[2:, :])
    if mask is not None:
        return float(ml_x[mask[:, 1:-1]].sum() + ml_y[mask[1:-1, :]].sum())
    return float(ml_x.sum() + ml_y.sum())


def vollath_f4(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Vollath's F4 autocorrelation-based focus measure. Robust to pixel noise."""
    img = _as_float(image)
    p1 = img[:-1, :] * img[1:, :]
    p2 = img[:-2, :] * img[2:, :]
    if mask is not None:
        m1 = mask[:-1, :] & mask[1:, :]
        m2 = mask[:-2, :] & mask[2:, :]
        return float(p1[m1].sum()) - float(p2[m2].sum())
    return float(p1.sum()) - float(p2.sum())


_METRIC_FUNCS = {
    "normalized_variance": normalized_variance,
    "brenner": brenner,
    "tenengrad": tenengrad,
    "laplacian_variance": laplacian_variance,
    "sml": sml,
    "vollath_f4": vollath_f4,
}


def _smooth_along_z(scores: np.ndarray) -> np.ndarray:
    """Convolve a 1D score vector with [1,2,1]/4 with reflective edges."""
    n = len(scores)
    if n < 3:
        return scores
    padded = np.concatenate([[scores[0]], scores, [scores[-1]]])
    kernel = np.array([1.0, 2.0, 1.0]) / 4.0
    return np.convolve(padded, kernel, mode="valid")


def _ensemble_scores(raw: Dict[str, np.ndarray]) -> np.ndarray:
    """Min-max-normalise each raw metric vector along Z, then average."""
    norms = []
    for name in _RAW_METRICS:
        arr = raw[name]
        lo = float(arr.min())
        hi = float(arr.max())
        if hi - lo < 1e-12:
            norms.append(np.zeros_like(arr, dtype=np.float64))
        else:
            norms.append((arr - lo) / (hi - lo))
    return np.mean(np.stack(norms, axis=0), axis=0)


def score_stack(
    stack_zyx: np.ndarray,
    crop_fraction: float = 1.0,
    preblur_sigma: float = 0.0,
    use_mask: bool = True,
    smooth_z: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute every focus metric for every slice of a Z-stack.

    Parameters
    ----------
    stack_zyx : array of shape (Z, Y, X)
    crop_fraction : float in (0, 1], default 1.0
        If <1, score only the central ``crop_fraction`` of each slice.
    preblur_sigma : float, default 0.0
        Optional Gaussian pre-blur sigma applied before scoring.
    use_mask : bool, default True
        If True, restrict scoring to a foreground mask computed once per
        stack. The mask is the same for every Z (computed from the union of
        bandpass responses) so it does not bias toward sharper slices.
    smooth_z : bool, default True
        If True, smooth each metric's per-Z score vector with [1,2,1]/4
        before returning. Suppresses single-slice noise spikes that flip
        argmax between adjacent near-tied slices.

    Returns
    -------
    dict mapping metric name to a 1D array of length Z. Includes an
    ``"ensemble"`` entry which is the mean of per-metric min-max-normalised
    score vectors — the recommended default for slice selection.
    """
    if stack_zyx.ndim != 3:
        raise ValueError(f"expected (Z, Y, X) stack, got shape {stack_zyx.shape}")
    n_z = stack_zyx.shape[0]
    pre = [_preprocess(stack_zyx[z], crop_fraction, preblur_sigma) for z in range(n_z)]
    pre_arr = np.stack(pre, axis=0)
    mask = compute_focus_mask(pre_arr) if use_mask else None
    raw: Dict[str, np.ndarray] = {
        name: np.array([func(plane, mask=mask) for plane in pre], dtype=np.float64)
        for name, func in _METRIC_FUNCS.items()
    }
    if smooth_z:
        raw = {name: _smooth_along_z(arr) for name, arr in raw.items()}
    raw["ensemble"] = _ensemble_scores(raw)
    return raw


def pick_per_pixel_z(
    stack_zyx: np.ndarray,
    sharpness_sigma: float = 2.0,
    smooth_size: int = 15,
) -> np.ndarray:
    """Per-pixel argmax of sharpness across Z (extended depth of field).

    For each pixel, returns the index of the Z slice with the sharpest local
    content. The sharpness measure is a Gaussian-smoothed absolute Laplacian
    (consolidates the per-pixel signal so a true edge always beats sensor
    noise on a flat region). The resulting integer Z-map is median-filtered
    with a ``smooth_size``-pixel window to suppress isolated speckle and
    produce spatially coherent picks across nearby pixels.

    Returns an integer Z-index map of shape ``(Y, X)``.
    """
    if stack_zyx.ndim != 3:
        raise ValueError(f"expected (Z, Y, X) stack, got shape {stack_zyx.shape}")
    img = _as_float(stack_zyx)
    sharpness = np.empty_like(img)
    for z in range(img.shape[0]):
        sharpness[z] = gaussian_filter(np.abs(laplace(img[z])), sigma=sharpness_sigma)
    z_map = sharpness.argmax(axis=0).astype(np.int32)
    if smooth_size and smooth_size > 1:
        z_map = median_filter(z_map, size=smooth_size)
    return z_map


def assemble_edf_plane(stack_zyx: np.ndarray, z_map_yx: np.ndarray) -> np.ndarray:
    """Gather a single (Y, X) plane from a stack using a per-pixel Z map.

    ``stack_zyx`` is one channel's full Z-stack. ``z_map_yx`` is the integer
    Z index to read at each pixel (typically derived from the phase channel
    via :func:`pick_per_pixel_z` so all channels stay spatially coherent).
    """
    if stack_zyx.ndim != 3:
        raise ValueError(f"expected (Z, Y, X) stack, got shape {stack_zyx.shape}")
    if z_map_yx.shape != stack_zyx.shape[1:]:
        raise ValueError(
            f"z_map shape {z_map_yx.shape} does not match stack YX {stack_zyx.shape[1:]}"
        )
    n_z = stack_zyx.shape[0]
    z_clipped = np.clip(z_map_yx, 0, n_z - 1).astype(np.intp)
    plane = np.take_along_axis(stack_zyx, z_clipped[None, :, :], axis=0)[0]
    return plane


def pick_best_slice(scores: Dict[str, np.ndarray], metric: str = DEFAULT_METRIC) -> int:
    """Return the index of the best-focused slice.

    Ties are broken in favour of the slice closest to the middle of the stack.
    """
    if metric not in scores:
        raise KeyError(f"metric {metric!r} not in scores; have {list(scores)}")
    values = scores[metric]
    best = float(values.max())
    candidates = np.flatnonzero(values >= best - 1e-12)
    middle = (len(values) - 1) / 2.0
    return int(candidates[np.argmin(np.abs(candidates - middle))])
