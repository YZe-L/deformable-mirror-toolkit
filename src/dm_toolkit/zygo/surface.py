# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-05

"""Surface statistics: PV/RMS, Zernike-group removal, and slice sampling."""

import numpy as np
from scipy import ndimage

from . import zernike as zern


def pv_rms(hmap, mask):
    """(PV, RMS) in waves over the masked pixels.

    Args:
        hmap: Surface-height map.
        mask: Boolean mask selecting valid samples.
    """
    v = hmap[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 0.0
    return float(v.max() - v.min()), float(v.std())


def remove(waves, mask, circ, groups, fit_result=None, order=6):
    """Subtract the selected surface-removal groups.

    Height map with the selected Removal groups subtracted (piston + more).

    Args:
        waves: Wavefront values in waves.
        mask: Boolean aperture mask.
        circ: Detected aperture `(cx, cy, radius)`.
        groups: Removal-group names from `REMOVAL_GROUPS`.
        fit_result: Fitted Zernike result used for coefficient removal.
        order: Maximum Zernike order retained in the fit.
    """
    if fit_result is None:
        fit_result = zern.fit(waves, mask, circ, order=order)
    idx = []
    for g in groups:
        idx += zern.REMOVAL_GROUPS.get(g, [])
    res = waves.copy()
    if idx:
        model = zern.reconstruct(fit_result["coeffs"], idx, mask, circ,
                                 waves.shape)
        res = res - np.nan_to_num(model)
    res[mask] -= res[mask].mean()
    res[~mask] = np.nan
    pv, rms = pv_rms(res, mask)
    return res, pv, rms


def slice_profile(hmap, p0, p1, n=400):
    """Sample the height map along the segment p0->p1 (pixel coords).

    Returns (distance_px, height_waves). Bilinear interpolation via
    map_coordinates; NaN (outside aperture) stays NaN so the curve breaks.

    Args:
        hmap: Surface-height map.
        p0: First endpoint coordinate.
        p1: Second endpoint coordinate.
        n: Number of requested samples or output points.
    """
    (x0, y0), (x1, y1) = p0, p1
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    vals = ndimage.map_coordinates(np.nan_to_num(hmap, nan=np.nan),
                                   [ys, xs], order=1, mode="constant",
                                   cval=np.nan)
    dist = np.hypot(xs - x0, ys - y0)
    return dist, vals
