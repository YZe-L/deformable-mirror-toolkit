# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-05

"""Aperture geometry: the analysis circle for the Zernike fit."""

import numpy as np


def aperture(mask):
    """(cx, cy, r) for the valid mask: centroid + equivalent-area radius."""
    ys, xs = np.where(mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    r = float(np.sqrt(mask.sum() / np.pi))
    return cx, cy, r


def unit_coords(mask, circ):
    """Return polar coordinates on the masked unit disk.

    Masked-pixel polar coords on the unit disk: (ys, xs, rho, theta, keep).

    keep selects pixels with rho <= 1 (the fit domain). Image y is row index
    (downward); theta uses atan2(y, x) consistently for fit and reconstruction.

    Args:
        mask: Boolean mask selecting valid samples.
        circ: Boolean mask for the circular aperture.
    """
    cx, cy, r = circ
    ys, xs = np.where(mask)
    rho = np.hypot(xs - cx, ys - cy) / r
    theta = np.arctan2(ys - cy, xs - cx)
    keep = rho <= 1.0
    return ys, xs, rho, theta, keep
