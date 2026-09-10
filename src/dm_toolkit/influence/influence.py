# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.1, 2026-07-31

"""Influence matrix and DM eigenmodes from measured surfaces.

Push-pull each actuator about a bias and difference the surfaces; stack the
columns as the influence matrix; remove piston, tip and tilt; differentiate,
because image metrics respond to the wavefront slope; take the SVD of the
gradient matrix, whose right singular vectors are the control vectors and
whose singular values rank the modes (Debarre et al., Opt. Express 15, 8176,
2007). Small singular values amplify noise, so the mode count is truncated
at the measured noise floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Surface height to wavefront phase on reflection at normal incidence: the ray
# traverses the sag twice, so phase = (2*pi/lambda) * 2 * z.
PHASE_PER_NM = 4.0 * np.pi

SCHEME_CENTRAL = "central"
SCHEME_FORWARD = "forward"

# How far a singular value must beat the noise floor to be worth correcting.
KEEP_MARGIN = 2.0


@dataclass
class Pupil:
    """The beam footprint on the mirror, in source-image pixels."""
    cx: float
    cy: float
    r: float

    def as_tuple(self):
        return self.cx, self.cy, self.r


@dataclass
class Grid:
    """Resampling target: a square grid with a unit-radius circle inscribed."""
    n: int

    @property
    def inside(self):
        """Boolean mask of the grid points inside the unit circle."""
        x, y = self.coords
        return (x ** 2 + y ** 2) <= 1.0

    @property
    def coords(self):
        """X and Y grids in pupil-radius units, spanning -1 to +1."""
        ax = np.linspace(-1.0, 1.0, self.n)
        return np.meshgrid(ax, ax)

    @property
    def step(self):
        """Grid spacing in pupil-radius units."""
        return 2.0 / max(self.n - 1, 1)

    @property
    def circ(self):
        """(cx, cy, r) of the inscribed circle in grid pixels."""
        half = (self.n - 1) / 2.0
        return half, half, half


def resample(surf, pupil: Pupil, grid: Grid):
    """Resample one surface onto the pupil grid, in nanometres.

    Uses normalised interpolation, so an invalid pixel reduces the weight
    instead of poisoning the result with NaN. A grid point whose window is
    not almost fully valid is reported invalid.

    Args:
        surf: `io_surface.SurfaceMap` to resample.
        pupil: Beam footprint on that surface, in its own pixels.
        grid: Output grid.

    Returns:
        The height grid with NaN outside, and its boolean validity mask.
    """
    X, Y = grid.coords
    # Grid coordinates are y-up in pupil units; image rows increase downward.
    src_x = (pupil.cx + X * pupil.r).astype(np.float32)
    src_y = (pupil.cy - Y * pupil.r).astype(np.float32)
    valid = np.asarray(surf.mask, bool) & np.isfinite(surf.z_nm)
    z = np.where(valid, surf.z_nm, 0.0).astype(np.float32)
    w = valid.astype(np.float32)
    zi = cv2.remap(z, src_x, src_y, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    wi = cv2.remap(w, src_x, src_y, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    ok = grid.inside & (wi > 0.999)
    out = np.full((grid.n, grid.n), np.nan)
    out[ok] = zi[ok] / wi[ok]
    return out, ok


def remove_piston_tilt(values, X, Y):
    """Subtract the least-squares plane from masked values.

    Fitted over exactly the points supplied, which must be the pupil points and
    nothing else: a plane removed over a different area leaves a residual tilt
    over the one that matters, and no error is raised either way.

    Args:
        values: 1-D heights at the pupil points.
        X: Their x coordinates in pupil-radius units.
        Y: Their y coordinates in pupil-radius units.

    Returns:
        The values with the fitted plane removed, and the (piston, x, y)
        coefficients that were removed.
    """
    A = np.column_stack([np.ones_like(X), X, Y])
    coef, *_ = np.linalg.lstsq(A, values, rcond=None)
    return values - A @ coef, coef


def push_pull(z_plus, z_minus, bit_span):
    """Per-bit influence from a symmetric pair, in the input's units per bit.

    Args:
        z_plus: Surface with the actuator driven above the bias.
        z_minus: Surface with it driven below.
        bit_span: Total bit difference between the two, e.g. 1600 for +/-800.

    Raises:
        ValueError: If `bit_span` is zero.
    """
    if not bit_span:
        raise ValueError("bit_span must be non-zero")
    return (np.asarray(z_plus, float) - np.asarray(z_minus, float)) \
        / float(bit_span)


def slope_from_pairs(halves, amplitudes):
    """Least-squares influence per bit from several symmetric pairs.

    A straight line through the origin fitted to the (amplitude,
    half-difference) points:

        IF = sum(a_j * d_j) / sum(a_j^2)

    Even-order terms have already cancelled inside each half-difference.
    This is the best line over the amplitudes measured, not the tangent at
    the bias.

    Args:
        halves: One (Z_plus - Z_minus) / 2 column per pair, same pupil points.
        amplitudes: The matching positive offsets, in bits.

    Returns:
        The influence column, in the input's units per bit.

    Raises:
        ValueError: If nothing was supplied or every amplitude is zero.
    """
    halves = [np.asarray(h, float) for h in halves]
    amps = [float(a) for a in amplitudes]
    if not halves or len(halves) != len(amps):
        raise ValueError("need one half-difference per amplitude")
    denominator = sum(a * a for a in amps)
    if denominator <= 0:
        raise ValueError("amplitudes must not all be zero")
    return sum(a * h for a, h in zip(amps, halves)) / denominator


def effective_span(amplitudes):
    """Push-pull bit span a multi-pair fit is as quiet as.

    The variance of `slope_from_pairs` is `sigma^2 / (2 * sum(a_j^2))`;
    equating it with a single pair's gives the span that pair would need,
    which keeps the noise floor comparable with the singular values.

    Args:
        amplitudes: The positive offsets of the pairs used, in bits.

    Returns:
        The equivalent span in bits; for a lone pair it is that pair's own span.
    """
    return 2.0 * float(np.sqrt(sum(float(a) ** 2 for a in amplitudes)))


def gradient_matrix(C, grid: Grid, inside, scheme=SCHEME_CENTRAL):
    """Stack the x and y derivatives of every column of C.

    Only differences whose whole stencil lies inside the pupil are kept: a
    difference straddling the edge produces a false slope that dominates the
    SVD. Central differencing halves the noise of forward differencing.
    Every row carries the quadrature weight of its grid cell, so
    `grad(C)^T grad(C)` is the discrete pupil integral and the singular
    values do not depend on the grid size.

    Args:
        C: (npix, n_act) matrix whose columns are maps flattened over `inside`.
        grid: The grid those maps live on.
        inside: Boolean pupil mask on that grid.
        scheme: `SCHEME_CENTRAL` or `SCHEME_FORWARD`.

    Returns:
        The (2 * n_points, n_act) gradient matrix, and the number of gradient
        sample points per direction.
    """
    n_act = C.shape[1]
    maps = np.full((n_act, grid.n, grid.n), np.nan)
    for k in range(n_act):
        maps[k][inside] = C[:, k]
    h = grid.step
    if scheme == SCHEME_FORWARD:
        gx_ok = inside[:, :-1] & inside[:, 1:]
        gy_ok = inside[:-1, :] & inside[1:, :]
        gx = (maps[:, :, 1:] - maps[:, :, :-1]) / h
        gy = (maps[:, 1:, :] - maps[:, :-1, :]) / h
    else:
        gx_ok = inside[:, :-2] & inside[:, 1:-1] & inside[:, 2:]
        gy_ok = inside[:-2, :] & inside[1:-1, :] & inside[2:, :]
        gx = (maps[:, :, 2:] - maps[:, :, :-2]) / (2.0 * h)
        gy = (maps[:, 2:, :] - maps[:, :-2, :]) / (2.0 * h)
    dx = np.stack([g[gx_ok] for g in gx], axis=1)
    dy = np.stack([g[gy_ok] for g in gy], axis=1)
    # sqrt of the cell area, so that squaring the matrix gives the area
    # integral.
    return np.vstack([dx, dy]) * h, int(gx_ok.sum() + gy_ok.sum())


@dataclass
class Modes:
    """The eigen-decomposition of one gradient matrix."""
    s: np.ndarray  # Singular values, descending.
    v: np.ndarray  # (n_act, n_act); column i is the raw control direction.
    ctrl: np.ndarray  # (n_act, n_act); column i produces 1 unit RMS of mode i.
    maps: np.ndarray  # (n_act, n, n) mode surfaces, NaN outside the pupil.
    gram: np.ndarray  # G = grad(C)^T grad(C); its eigenvectors are v.

    @property
    def n(self):
        return len(self.s)

    def condition(self, keep):
        """Condition number of the Gram matrix restricted to `keep` modes.

        This is the quantity Wang & Dong plot in their Fig. 10: it grows with
        the number of modes retained, and a large value is what makes the modal
        estimate fail rather than merely become inaccurate.
        """
        keep = max(1, min(int(keep), self.n))
        beta = self.s[:keep] ** 2
        return float(beta[0] / beta[-1]) if beta[-1] > 0 else float("inf")


def eigenmodes(gC, C, grid: Grid, inside) -> Modes:
    """Eigenmodes of the mirror from its gradient matrix.

    The SVD is taken of the gradient matrix rather than the eigen-decomposition
    of `G = grad(C)^T grad(C)`, because forming `G` squares the condition number
    and the small singular values are exactly the ones being judged. `G` is
    returned anyway: it is the object Ren & Dong's camera-only self-calibration
    measures, so having both lets the two routes be compared directly.

    Args:
        gC: Gradient matrix from `gradient_matrix`.
        C: The influence matrix those gradients came from.
        grid: The grid the maps live on.
        inside: Boolean pupil mask on that grid.
    """
    _u, s, vt = np.linalg.svd(gC, full_matrices=False)
    v = vt.T
    n_act = C.shape[1]
    maps = np.full((n_act, grid.n, grid.n), np.nan)
    ctrl = np.zeros_like(v)
    for i in range(n_act):
        if s[i] <= 0:
            continue
        col = C @ (v[:, i] / s[i])
        rms = float(np.sqrt(np.mean(col ** 2)))
        # Scale each control vector so one unit of its coefficient is one unit
        # RMS of wavefront; the three-point solve is invariant to this.
        scale = 1.0 / rms if rms > 0 else 0.0
        ctrl[:, i] = v[:, i] / s[i] * scale
        maps[i][inside] = col * scale
    return Modes(s=s, v=v, ctrl=ctrl, maps=maps, gram=gC.T @ gC)


def noise_singular_value(pairs, bit_span, grid: Grid, inside,
                         scheme=SCHEME_CENTRAL, remove_plane=False):
    """Singular value a column of pure measurement noise would produce.

    `(z1 - z2) / sqrt(2)` of two repeats is one realisation of a single
    measurement's error; pushed through the same scaling and gradient
    operator, its norm is the floor to truncate the mode count at.

    Args:
        pairs: Sequence of (col1, col2) repeated measurements, each already
            flattened over the pupil points and scaled to the influence
            matrix's own units.
        bit_span: The push-pull bit span the real columns were divided by.
        grid: The grid the maps live on.
        inside: Boolean pupil mask on that grid.
        scheme: Differencing scheme, matched to the real columns.
        remove_plane: Whether to apply the same piston/tip/tilt projection used
            for the influence columns.

    Returns:
        The median singular value over the supplied pairs, or NaN with no pairs.
    """
    vals = []
    X, Y = grid.coords
    for col1, col2 in pairs:
        d = np.asarray(col1, float) - np.asarray(col2, float)
        if not np.isfinite(d).all():
            continue
        # sqrt(2) undoes the differencing of two independent measurements; the
        # second sqrt(2) is the noise of the push-pull difference itself.
        col = d / np.sqrt(2.0) * np.sqrt(2.0) / float(bit_span)
        if remove_plane:
            col, _ = remove_piston_tilt(col, X[inside], Y[inside])
        g, _ = gradient_matrix(col.reshape(-1, 1), grid, inside, scheme)
        vals.append(float(np.linalg.norm(g)))
    return float(np.median(vals)) if vals else float("nan")


def keep_count(s, floor):
    """How many modes stand clear of the noise floor.

    A margin, not bare `s > floor`: at equality the mode's coefficient is all
    noise, and the solve divides by that singular value, so it turns the noise
    into a large confident-looking correction.
    """
    if not np.isfinite(floor) or floor <= 0:
        return int(len(s))
    return int(np.count_nonzero(np.asarray(s, float) > KEEP_MARGIN * floor))


def footprint_weight(column):
    """Per-point weight that follows where one actuator actually acts.

    An actuator moves a patch of the mirror, so a pupil-wide RMS averages
    the signal over points that carry none. Weighting by the column's own
    energy measures the footprint from the data instead.

    Args:
        column: One actuator's fitted influence column.

    Returns:
        Weights summing to one, or None when the column carries no energy.
    """
    w = np.asarray(column, float).ravel() ** 2
    w = np.where(np.isfinite(w), w, 0.0)
    total = float(w.sum())
    return w / total if total > 0 else None


def weighted_rms(values, weight=None):
    """RMS of `values`, over the actuator's footprint when a weight is given.

    Args:
        values: Column or map to summarise.
        weight: Optional per-point weights, as from `footprint_weight`.
    """
    v = np.asarray(values, float).ravel()
    ok = np.isfinite(v)
    if weight is None:
        return float(np.sqrt(np.mean(v[ok] ** 2))) if ok.any() else float("nan")
    w = np.asarray(weight, float).ravel()
    ok &= np.isfinite(w)
    total = float(w[ok].sum())
    if total <= 0:
        return float("nan")
    return float(np.sqrt(float(w[ok] @ (v[ok] ** 2)) / total))


def debiased_rms(values, sigma, weight=None):
    """RMS with the known measurement noise taken back out.

    A measured mean square is the signal's plus the noise's, so an RMS reads
    high, and by more as the signal shrinks; left in, the noisiest pair
    would look like odd-order non-linearity.

    Args:
        values: Column to summarise.
        sigma: Per-point noise of that column, same units.
        weight: Optional footprint weights.

    Returns:
        The noise-corrected RMS, floored at zero.
    """
    rms = weighted_rms(values, weight)
    if not np.isfinite(rms):
        return rms
    return float(np.sqrt(max(rms ** 2 - float(sigma) ** 2, 0.0)))


def shape_difference(a, b, weight=None):
    """RMS difference between two maps after normalising each to unit RMS.

    Decides whether one linear influence function describes the actuator
    over its driven range; a pure gain change reads zero.

    Args:
        a: First column or map.
        b: Second column or map.
        weight: Optional footprint weights. Without them the normalisation is
            pupil-wide and the answer is dominated by the points where neither
            column has any signal; see `footprint_weight`.

    Returns:
        Difference RMS as a fraction of unit RMS, sign-matched so an inverted
        pair does not read as a shape change.
    """
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    w = None
    if weight is not None:
        w = np.asarray(weight, float).ravel()
        ok &= np.isfinite(w)
    if ok.sum() < 4:
        return float("nan")
    a, b = a[ok], b[ok]
    if w is not None:
        w = w[ok]
    ra = weighted_rms(a, w)
    rb = weighted_rms(b, w)
    if not (ra > 0 and rb > 0):
        return float("nan")
    a, b = a / ra, b / rb
    return float(min(weighted_rms(a - b, w), weighted_rms(a + b, w)))


def superposition_error(C, offsets, measured):
    """Relative error of the linear model on one arbitrary command.

    `C @ offsets` is the model's prediction for a command it was not built
    from; everything downstream assumes the columns add.

    Args:
        C: Influence matrix, columns per actuator.
        offsets: Bit offsets from the bias, in the column order of C.
        measured: Measured surface change for that command, same units as C,
            flattened over the same pupil points.

    Returns:
        Residual RMS over measured RMS, and the predicted column.
    """
    predicted = C @ np.asarray(offsets, float)
    measured = np.asarray(measured, float)
    ok = np.isfinite(predicted) & np.isfinite(measured)
    if ok.sum() < 4:
        return float("nan"), predicted
    resid = np.sqrt(np.mean((measured[ok] - predicted[ok]) ** 2))
    scale = np.sqrt(np.mean(measured[ok] ** 2))
    return (float(resid / scale) if scale > 0 else float("nan")), predicted


def zernike_of(map2d, inside, grid: Grid, n_modes=15):
    """Zernike coefficients of one mode map, in the map's own units.

    RMS-normalised Noll modes, so a coefficient is directly the RMS that term
    contributes -- which is what makes the mode table readable as "this
    eigenmode is mostly astigmatism".

    Args:
        map2d: Mode surface with NaN outside the pupil.
        inside: Boolean pupil mask.
        grid: The grid the map lives on.
        n_modes: Number of Noll terms to fit.
    """
    from .. import zernike as Z
    ok = inside & np.isfinite(map2d)
    if ok.sum() < n_modes + 4:
        return None
    return Z.fit(map2d, ok, grid.circ, n_modes=n_modes)
