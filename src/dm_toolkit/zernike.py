# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-06-19

"""Zernike decomposition of a reconstructed wavefront (hand-rolled)."""

import math

import numpy as np

# Noll j -> human name (the classic low-order aberrations / Zygo Seidel set)
NOLL_NAMES = {
    1: "piston", 2: "tilt X", 3: "tilt Y", 4: "defocus",
    5: "astig 45", 6: "astig 0", 7: "coma Y", 8: "coma X",
    9: "trefoil Y", 10: "trefoil X", 11: "spherical",
    12: "2nd astig 0", 13: "2nd astig 45", 14: "tetrafoil X",
    15: "tetrafoil Y", 16: "2nd coma X", 17: "2nd coma Y",
    22: "2nd spherical",
}


def noll_to_nm(j):
    """Noll index j (1-based) -> (n, m) radial order / azimuthal frequency."""
    if j < 1:
        raise ValueError("Noll index starts at 1")
    n = 0
    while j > n * (n + 1) // 2 + n + 1:
        n += 1
    # Remaining offset inside radial order n.
    p = j - (n * (n + 1) // 2 + 1)
    # M values for this n, ordered as Noll does (sign from j parity)
    ms = [mm for mm in range(-n, n + 1, 2)]
    ms.sort(key=lambda mm: (abs(mm), mm < 0))
    m = ms[p]
    # Noll sign convention: even j -> cosine (m>0), odd j -> sine (m<0)
    if m != 0:
        m = abs(m) if (j % 2 == 0) else -abs(m)
    return n, m


def _radial(n, m, rho):
    """Zernike radial polynomial R_n^|m|(rho).

    Args:
        n: Number of requested samples or output points.
        m: Mode index or model order.
        rho: Normalised radial coordinate.
    """
    m = abs(m)
    R = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        c = ((-1) ** k * math.factorial(n - k)
             / (math.factorial(k) * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)))
        R += c * rho ** (n - 2 * k)
    return R


def zernike_mode(j, rho, theta):
    """Single Noll-indexed, RMS-normalised Zernike mode at (rho, theta).

    Args:
        j: Noll Zernike index.
        rho: Normalised radial coordinate.
        theta: Angular coordinate, in radians.
    """
    n, m = noll_to_nm(j)
    R = _radial(n, m, rho)
    if m > 0:
        return math.sqrt(2 * (n + 1)) * R * np.cos(m * theta)
    if m < 0:
        return math.sqrt(2 * (n + 1)) * R * np.sin(-m * theta)
    return math.sqrt(n + 1) * R


def basis(rho, theta, n_modes):
    """(n_modes, npix) matrix of the first n_modes Noll modes at the points.

    Args:
        rho: Normalised radial coordinate.
        theta: Angular coordinate, in radians.
        n_modes: Number of n modes.
    """
    return np.vstack([zernike_mode(j, rho, theta)
                      for j in range(1, n_modes + 1)])


def unit_coords(mask, circ):
    """Aperture pixels -> unit-disk (rho, theta), y-up.

    Returns rho, theta, ys, xs.

    Args:
        mask: Boolean mask selecting valid samples.
        circ: Boolean mask for the circular aperture.
    """
    cx, cy, r = circ
    ys, xs = np.where(mask)
    X = (xs - cx) / r
    Y = -(ys - cy) / r
    rho = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)
    return rho, theta, ys, xs


def fit(wavefront, mask, circ, n_modes=15):
    """Least-squares Zernike fit of a wavefront map (waves).

    Returns coeffs (waves, Noll 1..n_modes), the smooth fitted map, the
    residual map (input minus fit, both NaN outside the aperture), and
    per-mode RMS plus named-aberration magnitudes.

    Args:
        wavefront: Wavefront map to process.
        mask: Boolean mask selecting valid samples.
        circ: Boolean mask for the circular aperture.
        n_modes: Number of n modes.
    """
    rho, theta, ys, xs = unit_coords(mask, circ)
    keep = rho <= 1.0
    rho, theta, ys, xs = rho[keep], theta[keep], ys[keep], xs[keep]
    Z = basis(rho, theta, n_modes)  # (n_modes, npix)
    vals = wavefront[ys, xs]
    coeffs, *_ = np.linalg.lstsq(Z.T, vals, rcond=None)

    fitvals = Z.T @ coeffs
    fit_map = np.full(wavefront.shape, np.nan)
    res_map = np.full(wavefront.shape, np.nan)
    fit_map[ys, xs] = fitvals
    res_map[ys, xs] = vals - fitvals

    # piston (Noll 1) carries no aberration; report everything above it
    ab = coeffs.copy()
    rms_total = float(np.sqrt(np.sum(ab[1:] ** 2)))
    res_rms = float(np.nanstd(res_map[mask])) if mask.any() else 0.0
    named = named_aberrations(coeffs)
    return dict(coeffs=coeffs, names=[NOLL_NAMES.get(j, f"Z{j}")
                                      for j in range(1, n_modes + 1)],
                fit_map=fit_map, residual=res_map,
                rms_fit_waves=rms_total, rms_residual_waves=res_rms,
                named=named, n_modes=n_modes)


def _pair(coeffs, j_cos, j_sin, order):
    """Magnitude + physical orientation (deg) of a cos/sin Zernike pair.

    cos coeff = j_cos, sin coeff = j_sin; angle = atan2(sin, cos)/order.

    Args:
        coeffs: Model coefficients.
        j_cos: Sequence of j co values.
        j_sin: Noll index of the sine partner.
        order: Polynomial, Zernike, or symmetry order.
    """
    a = coeffs[j_cos - 1] if j_cos <= len(coeffs) else 0.0
    b = coeffs[j_sin - 1] if j_sin <= len(coeffs) else 0.0
    return float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)) / order)


def named_aberrations(coeffs):
    """Return named aberrations.

    Low-order aberrations in waves (magnitude, and angle where it applies).
    """
    def c(j):
        return float(coeffs[j - 1]) if j <= len(coeffs) else 0.0
    astig_mag, astig_ang = _pair(coeffs, 6, 5, 2)
    coma_mag, coma_ang = _pair(coeffs, 8, 7, 1)
    trefoil_mag, trefoil_ang = _pair(coeffs, 10, 9, 3)
    return dict(
        defocus=c(4),
        astigmatism=astig_mag, astigmatism_angle=astig_ang,
        coma=coma_mag, coma_angle=coma_ang,
        trefoil=trefoil_mag, trefoil_angle=trefoil_ang,
        spherical=c(11),
    )


def orthonormality_error(n_modes=15, size=400):
    """Return orthonormality error.

    Self-check: |G - I| where G is the Gram matrix of the modes over the
    unit disk. ~1e-2 (grid-limited) confirms the modes are RMS-orthonormal.

    Args:
        n_modes: Number of n modes.
        size: Requested output size.
    """
    ax = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(ax, ax)
    rho = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)
    disk = rho <= 1.0
    Z = basis(rho[disk], theta[disk], n_modes)  # (n_modes, npix)
    G = (Z @ Z.T) / Z.shape[1]  # Mean product = inner prod.
    return float(np.max(np.abs(G - np.eye(n_modes))))


def cross_check_prysm(n_modes=15, size=256):
    """Return cross check prysm.

    Max abs difference between the hand-rolled basis and prysm's modes on a
    unit disk (scratch + library, then compare). ~1e-12 confirms the radial
    polynomials. Returns None if prysm is unavailable.

    Args:
        n_modes: Number of n modes.
        size: Requested output size.
    """
    try:
        from prysm import polynomials as P
    except Exception:
        return None
    ax = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(ax, ax)
    rho = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)
    disk = rho <= 1.0
    diffs = []
    for j in range(1, n_modes + 1):
        n, m = noll_to_nm(j)
        ours = zernike_mode(j, rho, theta)
        ref = P.zernike_nm(n, m, rho, theta, norm=True)
        a, b = ours[disk], ref[disk]
        diffs.append(min(np.max(np.abs(a - b)), np.max(np.abs(a + b))))
    return float(np.max(diffs))
