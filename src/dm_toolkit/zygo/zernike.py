# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-05

"""ZFR (Zygo "Zernike Fringe") polynomials + least-squares fit."""

import math

import numpy as np

# Fringe order: term index -> (n, m). m>0 => cos(|m|t), m<0 => sin(|m|t).
FRINGE_NM = [
    (0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (2, -2), (3, 1), (3, -1),
    (4, 0), (3, 3), (3, -3), (4, 2), (4, -2), (5, 1), (5, -1), (6, 0),
    (4, 4), (4, -4), (5, 3), (5, -3), (6, 2), (6, -2), (7, 1), (7, -1),
    (8, 0), (5, 5), (5, -5), (6, 4), (6, -4), (7, 3), (7, -3), (8, 2),
    (8, -2), (9, 1), (9, -1), (10, 0),
]

# order (max radial degree) -> number of Fringe terms
ORDER_TERMS = {4: 9, 6: 16, 8: 25, 10: 36}

# Short human names + which term indices each Removal button subtracts.
NAMES = {
    0: "piston", 1: "tilt X", 2: "tilt Y", 3: "focus", 4: "astig 0",
    5: "astig 45", 6: "coma X", 7: "coma Y", 8: "spherical",
}
REMOVAL_GROUPS = {
    "Piston": [0], "Tilt": [1, 2], "Power": [3],
    "Astigmatism": [4, 5], "Coma": [6, 7], "Spherical": [8],
}


# Radial polynomial as (degree, coefficient) pairs, precomputed per term.
def _radial_poly(n, m):
    m = abs(m)
    out = []
    for k in range((n - m) // 2 + 1):
        c = ((-1) ** k * math.factorial(n - k)
             / (math.factorial(k) * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)))
        out.append((n - 2 * k, c))
    return out


_TERMPOLY = [_radial_poly(n, m) for (n, m) in FRINGE_NM]


def _prepare(rho, theta, indices):
    """Precompute rho powers and cos/sin(m*theta) needed by the given terms.

    Powers are built by successive multiply (cheap) instead of per-term `**`;
    reused across terms this is ~5-10x faster than recomputing each polynomial.

    Args:
        rho: Normalised radial coordinate.
        theta: Angular coordinate, in radians.
        indices: Zero-based item indices.
    """
    maxdeg = max(FRINGE_NM[i][0] for i in indices)
    maxm = max(abs(FRINGE_NM[i][1]) for i in indices)
    pows = [np.ones_like(rho)]
    for _ in range(maxdeg):
        pows.append(pows[-1] * rho)
    cos = [None] * (maxm + 1)
    sin = [None] * (maxm + 1)
    for mm in range(maxm + 1):
        a = mm * theta
        cos[mm], sin[mm] = np.cos(a), np.sin(a)
    return pows, cos, sin


def _term_from(idx, pows, cos, sin):
    n, m = FRINGE_NM[idx]
    R = None
    for deg, c in _TERMPOLY[idx]:
        p = c * pows[deg]
        R = p if R is None else R + p
    if m == 0:
        return R
    return R * (cos[m] if m > 0 else sin[-m])


def term(idx, rho, theta):
    """Single Fringe term idx evaluated at (rho, theta).

    Args:
        idx: Zero-based item indices.
        rho: Normalised radial coordinate.
        theta: Angular coordinate, in radians.
    """
    pows, cos, sin = _prepare(rho, theta, [idx])
    return _term_from(idx, pows, cos, sin)


def basis(rho, theta, n_terms):
    """Design matrix (npix, n_terms) of the first n_terms Fringe polynomials.

    Args:
        rho: Normalised radial coordinate.
        theta: Angular coordinate, in radians.
        n_terms: Number of n terms.
    """
    idx = range(n_terms)
    pows, cos, sin = _prepare(rho, theta, idx)
    return np.stack([_term_from(i, pows, cos, sin) for i in idx], axis=1)


def fit_coords(rho, theta, w, order):
    """ZFR least-squares fit given precomputed (rho, theta, w).

    No coord rebuild.

    Args:
        rho: Normalised radial coordinate.
        theta: Angular coordinate, in radians.
        w: Image width or weighting value.
        order: Polynomial, Zernike, or symmetry order.
    """
    n_terms = ORDER_TERMS[order]
    B = basis(rho, theta, n_terms)
    coeffs, *_ = np.linalg.lstsq(B, w, rcond=None)
    return {"coeffs": coeffs, "n_terms": n_terms, "order": order}


def fit(waves, mask, circ, order=6, max_points=40000):
    """Least-squares ZFR fit of the surface (waves) over the aperture.

    Fits on a strided subsample (~max_points) -- the global coefficients are
    unchanged vs the full ~800k points but ~20x faster.

    Args:
        waves: Sequence of wave values.
        mask: Boolean mask selecting valid samples.
        circ: Boolean mask for the circular aperture.
        order: Polynomial, Zernike, or symmetry order.
        max_points: Maximum permitted points.
    """
    from .geometry import unit_coords
    _ys, _xs, rho, theta, keep = unit_coords(mask, circ)
    rk, tk = rho[keep], theta[keep]
    wk = waves[mask][keep]
    step = max(1, len(wk) // max_points)
    out = fit_coords(rk[::step], tk[::step], wk[::step], order)
    out["circ"] = circ
    return out


def eval_terms(coeffs, indices, rho, theta):
    """Sum of the selected terms at (rho, theta) -> 1-D values.

    Args:
        coeffs: Model coefficients.
        indices: Zero-based item indices.
        rho: Normalised radial coordinate.
        theta: Angular coordinate, in radians.
    """
    indices = [i for i in indices if i < len(coeffs)]
    if not indices:
        return np.zeros(len(rho))
    pows, cos, sin = _prepare(rho, theta, indices)
    vals = np.zeros(len(rho))
    for i in indices:
        vals += coeffs[i] * _term_from(i, pows, cos, sin)
    return vals


def map_from_vals(vals, ys, xs, out_shape):
    """Scatter 1-D masked values into a full 2-D map (NaN elsewhere).

    Args:
        vals: Sequence of val values.
        ys: Vertical coordinates.
        xs: Horizontal coordinates.
        out_shape: Requested output array shape.
    """
    out = np.full(out_shape, np.nan)
    out[ys, xs] = vals
    return out


def reconstruct(coeffs, indices, mask, circ, out_shape):
    """Full 2-D map (NaN outside mask) from the selected Fringe terms only.

    Args:
        coeffs: Model coefficients.
        indices: Zero-based item indices.
        mask: Boolean mask selecting valid samples.
        circ: Boolean mask for the circular aperture.
        out_shape: Requested output array shape.
    """
    from .geometry import unit_coords
    ys, xs, rho, theta, _keep = unit_coords(mask, circ)
    return map_from_vals(eval_terms(coeffs, indices, rho, theta), ys, xs,
                         out_shape)


def fit_map(coeffs, mask, circ, out_shape, n_terms=None):
    """The fitted surface (all terms up to n_terms) as a 2-D map.

    Args:
        coeffs: Model coefficients.
        mask: Boolean mask selecting valid samples.
        circ: Boolean mask for the circular aperture.
        out_shape: Requested output array shape.
        n_terms: Number of n terms.
    """
    n = len(coeffs) if n_terms is None else n_terms
    return reconstruct(coeffs, range(n), mask, circ, out_shape)
