# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-08

"""Temporal phase-shifting interferometry (PSI) -- the Zygo-style algorithm."""

import numpy as np
import cv2

from .fringes import unwrap2d


# Core solves (all operate on a flat (N, P) frame stack)
def _solve_phase(stack, steps):
    """Least-squares phase given the per-frame steps.

    Model I_k = a + c*cos(d_k) - s*sin(d_k) with c = b*cos(phi), s = b*sin(phi).
    M is shared by every pixel, so one 3x3 inverse solves the whole frame.
    Returns (phi, modulation b, background a), each length P.

    Args:
        stack: Image or wavefront stack.
        steps: Sequence of command increments.
    """
    steps = np.asarray(steps, float)
    M = np.stack([np.ones_like(steps), np.cos(steps), -np.sin(steps)], axis=1)
    Minv = np.linalg.pinv(M)  # (3, N)
    a, c, s = Minv @ stack  # each (P,)
    phi = np.arctan2(s, c)
    b = np.hypot(c, s)
    return phi, b, a


def _solve_steps(stack, phi):
    """Least-squares per-frame steps given the phase (AIA's other half).

    Each frame I_k = u_k + v_k*cos(phi) + w_k*sin(phi); d_k = atan2(-w_k, v_k).
    D is shared by every frame, so one 3x3 inverse solves all N steps.

    Args:
        stack: Image or wavefront stack.
        phi: Optical phase values, in radians.
    """
    D = np.stack([np.ones_like(phi), np.cos(phi), np.sin(phi)], axis=1)
    Dinv = np.linalg.pinv(D)  # (3, P)
    U = Dinv @ stack.T  # (3, N): u, v, w per frame
    return np.arctan2(-U[2], U[1])


def _pca_steps(stack):
    """Seed the steps from the top-2 temporal principal components (Vargas).

    Removing each pixel's temporal mean, the frame-space covariance is rank-2
    and its two eigenvectors span (cos d_k, sin d_k). Good enough to seed AIA
    even when the real steps are non-uniform and far from pi/2.
    """
    J = stack - stack.mean(axis=0, keepdims=True)  # Remove background a(x)
    C = J @ J.T  # (N, N)
    w, V = np.linalg.eigh(C)
    e1, e2 = V[:, -1], V[:, -2]
    return np.arctan2(e2, e1)


def _fix_sign(steps, phi):
    """Make the phase-step trend increase monotonically.

    Make the phase-step trend increasing so the surface sign is deterministic.

    The solve is ambiguous under (d, phi) -> (-d, -phi) (a complex conjugate),
    which flips concave/convex. Anchoring the net step direction removes it; a
    physical `invert` toggle still overrides at the call site if the piezo runs
    the other way.

    Args:
        steps: Sequence of command increments.
        phi: Optical phase values, in radians.
    """
    trend = np.polyfit(np.arange(len(steps)), np.unwrap(steps), 1)[0]
    if trend < 0:
        return -steps, -phi
    return steps, phi


# Phase-extraction methods
def aia(stack, iters=20, tol=1e-6, mask_flat=None, init=None):
    """Advanced Iterative Algorithm: self-calibrating N-frame PSI.

    Alternates phase-given-steps and steps-given-phase to convergence, so the
    steps are recovered from the data (unknown, uniform-per-frame, any value).
    Steps are estimated over `mask_flat` (fringe region) when given; the phase
    is solved everywhere. Returns (phi, modulation, steps).

    Args:
        stack: Image or wavefront stack.
        iters: Number of optimisation or refinement iterations.
        tol: Numerical or exposure tolerance.
        mask_flat: Flattened aperture mask.
        init: Initial parameter vector.
    """
    n = stack.shape[0]
    steps = np.asarray(init, float) if init is not None else _pca_steps(
        stack if mask_flat is None else stack[:, mask_flat])
    steps = steps - steps[0]
    sel = stack if mask_flat is None else stack[:, mask_flat]
    prev = steps
    phi = None
    for _ in range(int(iters)):
        phi, _b, _a = _solve_phase(stack, steps)
        steps = _solve_steps(sel, phi[mask_flat] if mask_flat is not None else phi)
        steps = steps - steps[0]
        if np.max(np.abs(np.angle(np.exp(1j * (steps - prev))))) < tol:
            break
        prev = steps
    phi, b, _a = _solve_phase(stack, steps)
    steps, phi = _fix_sign(steps, phi)
    return phi, b, steps


def lsq(stack, steps):
    """Solve the calibrated phase steps by least squares.

    N-frame least squares with KNOWN (calibrated) steps -- the same phase
    solver Zygo uses with a calibrated PZT. No self-calibration.

    Args:
        stack: Image or wavefront stack.
        steps: Sequence of command increments.
    """
    phi, b, _a = _solve_phase(stack, steps)
    return phi, b, np.asarray(steps, float)


_METHODS = {"aia", "lsq", "auto"}


# Masks
def variance_mask(stack2d, frac=0.10):
    """Rough fringe region from temporal contrast, for step estimation.

    Args:
        stack2d: Flattened image stack.
        frac: Requested fractional level.
    """
    std = stack2d.std(axis=0)
    return std > frac * float(std.max())


def modulation_mask(mod2d, frac=0.10, close=15):
    """Build a valid-data mask from fringe modulation.

    Valid-data mask from fringe modulation (Zygo-style): threshold the
    modulation, keep the largest blob, fill holes. `frac` of the peak.

    Args:
        mod2d: Two-dimensional modulation map.
        frac: Requested fractional level.
        close: Upper boundary of the modulation interval.
    """
    m = (mod2d > frac * float(np.nanmax(mod2d))).astype(np.uint8)
    k = np.ones((close, close), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return m.astype(bool)
    big = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [big], -1, 1, cv2.FILLED)
    return filled.astype(bool)


# Top-level reconstruction
def reconstruct(frames, method="aia", steps=None, mod_frac=0.10, invert=False,
                unwrap=True, aia_iters=20):
    """N interferograms -> wrapped/unwrapped phase, OPD map, mask, steps.

    Args:
        frames: Interferogram stack with shape `(N, H, W)`.
        method: Phase-reconstruction method.
        steps: Known phase steps in radians, or `None` for AIA estimation.
        mod_frac: Modulation threshold as a fraction of the peak.
        invert: Whether to invert the measured sign or image convention.
        unwrap: Whether to spatially unwrap the recovered phase.
        aia_iters: Maximum number of AIA phase-step iterations.

    Returns:
        A mapping containing the unwrapped and wrapped phase, OPD in waves,
        modulation, aperture mask, mean intensity, fitted phase steps, and
        method.
    """
    stack = np.asarray(frames, float)
    if stack.ndim != 3 or stack.shape[0] < 3:
        raise ValueError("need a stack of >=3 frames, shape (N, H, W)")
    n, h, w = stack.shape
    flat = stack.reshape(n, -1)

    if method == "auto":
        method = "lsq" if steps is not None else "aia"

    if method == "aia":
        vmask = variance_mask(stack).reshape(-1)
        phi, mod, est = aia(flat, iters=aia_iters, mask_flat=vmask)
    elif method == "lsq":
        if steps is None:
            raise ValueError("lsq needs known steps")
        phi, mod, est = lsq(flat, steps)
    else:
        raise ValueError(f"unknown method {method!r} (use {_METHODS})")

    phi = phi.reshape(h, w)
    mod = mod.reshape(h, w)
    if invert:
        phi = -phi
    mask = modulation_mask(mod, frac=mod_frac)

    wrapped = np.where(mask, phi, np.nan)
    if unwrap:
        uw = unwrap2d(phi, mask)
        uw = uw - np.nanmean(np.where(mask, uw, np.nan))  # Kill piston offset.
        phase = np.where(mask, uw, np.nan)
    else:
        phase = wrapped

    opd = phase / (2 * np.pi)  # OPD in waves.
    return dict(phase=phase, wrapped=wrapped, opd_waves=opd, modulation=mod,
                mask=mask, intensity=stack.mean(axis=0), steps=est,
                method=method, invert=bool(invert))


def build_measurement(recon, path="psi", wavelength_nm=None, scale=0.5,
                      attrs=None):
    """Wrap a PSI reconstruction as a Zygo-core Measurement.

    fringes = OPD in waves; Measurement.waves = fringes * scale (0.5 for the
    double pass), the .datx convention.

    Args:
        recon: Result of `reconstruct`.
        path: Name recorded on the measurement.
        wavelength_nm: Wavelength, in nanometres.
        scale: Interferometric scale factor.
        attrs: Extra attributes recorded on the measurement.
    """
    from ..zygo.io_datx import Measurement
    fringes = np.where(recon["mask"], recon["opd_waves"], np.nan)
    md = {"measure_mode": "PSI", "system": "Michelson",
          "psi_method": recon.get("method", "?")}
    if attrs:
        md.update(attrs)
    return Measurement(path=str(path), fringes=fringes, mask=recon["mask"],
                       scale=float(scale), wavelength_nm=wavelength_nm,
                       intensity=recon.get("intensity"), lateral_res_m=0.0,
                       attrs=md)


def step_report(steps):
    """Human summary of the estimated steps (deg), for the status chip."""
    d = np.rad2deg(np.diff(np.unwrap(np.asarray(steps, float))))
    if len(d) == 0:
        return "steps: --"
    return (f"steps ~ {np.mean(d):.1f} deg/frame "
            f"(spread {np.std(d):.1f}); n={len(steps)}")
