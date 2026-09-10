# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.2, 2026-08-26

"""Compute encircled-energy and Strehl-ratio curves."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
from scipy.special import j0, j1

from ..beam import beam

_V_R0 = 3.8317059702075125  # First zero of J1 = the Airy dark ring.


def r0_px(wavelength_nm: float, focal_mm: float, aperture_mm: float,
          pixel_um: float) -> float:
    """Airy first-dark-ring radius 1.22*lambda*f/D on the sensor, in pixels.

    Returns 0.0 when any parameter is missing/non-positive (= unknown optics).

    Args:
        wavelength_nm: Wavelength, in nanometres.
        focal_mm: Focal, in millimetres.
        aperture_mm: Aperture, in millimetres.
        pixel_um: Pixel, in micrometres.
    """
    if min(wavelength_nm, focal_mm, aperture_mm, pixel_um) <= 0:
        return 0.0
    lam_um = wavelength_nm * 1e-3
    return float(1.22 * lam_um * (focal_mm / aperture_mm) / pixel_um)


def ideal_ee(x):
    """Ideal Airy encircled-energy fraction at radius x = r / r0."""
    v = _V_R0 * np.asarray(x, float)
    ee = 1.0 - j0(v) ** 2 - j1(v) ** 2
    return np.where(v > 0, ee, 0.0)


def measured_ee(frame, r_grid_px):
    """Return the encircled-energy curve on a pixel-radius grid.

    Args:
        frame: Captured image frame.
        r_grid_px: R grid, in pixels.
    """
    gray = beam.as_gray(frame)
    detection = beam.subtract_background(gray)
    h, w = gray.shape
    peak = beam.brightest_resolved_pixel(detection)
    if peak is None:
        return None
    cx, cy = peak
    k = max(4, min(h, w) // 20)
    corners = np.concatenate([
        gray[:k, :k].ravel(), gray[:k, -k:].ravel(),
        gray[-k:, :k].ravel(), gray[-k:, -k:].ravel()])
    signal = gray - float(np.median(corners))
    Y, X = np.mgrid[0:h, 0:w]
    rr = np.hypot(X - cx, Y - cy).ravel()
    grid = np.asarray(r_grid_px, float)
    if not len(grid) or float(grid[-1]) <= 0:
        return None
    inside = rr <= float(grid[-1])
    radii = rr[inside]
    values = signal.ravel()[inside]
    total = float(values.sum())
    if total <= 0:
        return None
    order = np.argsort(radii)
    cum = np.cumsum(values[order]) / total
    curve = np.interp(grid, radii[order], cum, left=0.0, right=1.0)
    # Finite sampled background can produce tiny downward wiggles. EE is a
    # cumulative quantity, so enforce its physical monotonicity after the
    # signed-noise cancellation rather than clipping individual pixels.
    curve = np.maximum.accumulate(curve)
    curve = np.clip(curve, 0.0, 1.0)
    curve[-1] = 1.0
    return curve


@dataclass
class EEStage:
    """One corrected state measured between `before` and `after`.

    A staged two-mirror run reaches its answer in more than one move, and the
    spot parked on the first mirror's best shape is a real measured state --
    not an interpolation between the two ends. Each such state gets its own
    curve so the figure can say which mirror bought which part of the gain.

    Attributes:
        name: Stage label, e.g. ``"DM9"``.
        ee: Encircled-energy curve on the shared radius grid, or None.
        strehl: EE-Strehl at r0, or None when the optics are unknown.
    """

    name: str
    ee: np.ndarray | None
    strehl: float | None = None


@dataclass
class EECurve:
    """Ideal / before / after EE curves on one shared radius grid."""
    r_px: np.ndarray
    r0_px: float  # 0 = unknown optics (no ideal, no ratio)
    x: np.ndarray | None  # r / r0 ("multiple of diffraction")
    ideal: np.ndarray | None
    before: np.ndarray | None
    after: np.ndarray | None
    strehl_before: float | None = None  # EE_before / EE_ideal at r = r0.
    strehl_after: float | None = None
    # Intermediate states, in the order they were measured. Empty for a
    # single-mirror run, which reaches `after` in one stage.
    stages: tuple[EEStage, ...] = ()


def build_curves(frame_before, frame_after, r0: float,
                 x_max: float = 20.0, n: int = 400,
                 stage_frames=()) -> EECurve | None:
    """EE curves for the before/after frames vs the ideal Airy.

    With r0 > 0 the grid spans [0, x_max*r0] px; otherwise half the frame and
    only the measured curves. Every curve shares the same radius grid.

    Args:
        frame_before: Frame captured before optimisation.
        frame_after: Frame captured after optimisation.
        r0: Diffraction-limited reference radius, in pixels.
        x_max: Maximum horizontal command value.
        n: Number of requested samples or output points.
        stage_frames: Ordered ``(name, frame)`` pairs for the states measured
            between the two ends, e.g. the spot parked on DM9's best shape
            before DM5 was touched.
    """
    staged = [(str(name), frame) for name, frame in (stage_frames or ())
              if frame is not None]
    frames = [f for f in (frame_before, frame_after) if f is not None]
    frames += [frame for _, frame in staged]
    if not frames:
        return None
    if r0 > 0:
        r_max = x_max * r0
    else:
        h, w = beam.as_gray(frames[0]).shape
        r_max = 0.5 * float(min(h, w))
    r_grid = np.linspace(0.0, r_max, int(n))
    before = measured_ee(frame_before, r_grid) if frame_before is not None else None
    after = measured_ee(frame_after, r_grid) if frame_after is not None else None
    stage_ee = [(name, measured_ee(frame, r_grid)) for name, frame in staged]
    x = ideal = sb = sa = None
    strehl = lambda ee: None
    if r0 > 0:
        x = r_grid / r0
        ideal = ideal_ee(x)
        e0 = float(ideal_ee(1.0))  # ~0.838 at the dark ring.
        strehl = lambda ee: (None if ee is None
                             else float(np.interp(r0, r_grid, ee) / e0))
        if before is not None:
            sb = float(np.interp(r0, r_grid, before) / e0)
        if after is not None:
            sa = float(np.interp(r0, r_grid, after) / e0)
    stages = tuple(EEStage(name, ee, strehl(ee)) for name, ee in stage_ee)
    return EECurve(r_grid, float(r0), x, ideal, before, after, sb, sa, stages)


def save_csv(curve: EECurve, path, meta: dict | None = None):
    """Dump the curves to CSV.

    '#' header lines carry the metadata, then one row per radius -- any plotting
    software can reproduce the figure.

    Args:
        curve: Sampled curve data.
        path: Filesystem path used by the operation.
        meta: Metadata associated with the measurement.
    """
    ratio_b = ratio_a = None
    ratio_stage = [None] * len(curve.stages)
    if curve.ideal is not None:
        safe = np.maximum(curve.ideal, 1e-9)
        ok = curve.ideal > 1e-3  # Ratio meaningless near r=0.
        if curve.before is not None:
            ratio_b = np.where(ok, curve.before / safe, np.nan)
        if curve.after is not None:
            ratio_a = np.where(ok, curve.after / safe, np.nan)
        ratio_stage = [None if st.ee is None
                       else np.where(ok, st.ee / safe, np.nan)
                       for st in curve.stages]

    def col(a, i):
        if a is None:
            return ""
        v = float(a[i])
        return f"{v:.6g}" if np.isfinite(v) else ""

    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("# encircled-energy / Strehl curves"
                 " (ee = signed-background-subtracted energy(<r) / energy"
                 " inside the maximum plotted radius)\n")
        fh.write(f"# r0_px={curve.r0_px:.6g}"
                 "  (1.22*lambda*f/D on the sensor; 0 = unknown optics)\n")
        for k, v in (meta or {}).items():
            fh.write(f"# {k}={v}\n")
        if curve.strehl_before is not None:
            fh.write(f"# ee_strehl_at_r0_before={curve.strehl_before:.6g}\n")
        if curve.strehl_after is not None:
            fh.write(f"# ee_strehl_at_r0_after={curve.strehl_after:.6g}\n")
        for st in curve.stages:
            if st.strehl is not None:
                fh.write(f"# ee_strehl_at_r0_{_slug(st.name)}="
                         f"{st.strehl:.6g}\n")
        # The stage columns are APPENDED, never interleaved: a script written
        # against the seven-column file keeps working, and a staged run simply
        # carries more columns after them.
        wr = csv.writer(fh)
        wr.writerow(["r_px", "r_over_r0", "ee_ideal", "ee_before", "ee_after",
                     "ratio_before", "ratio_after"]
                    + [f"ee_{_slug(st.name)}" for st in curve.stages]
                    + [f"ratio_{_slug(st.name)}" for st in curve.stages])
        for i in range(len(curve.r_px)):
            wr.writerow([f"{curve.r_px[i]:.4f}", col(curve.x, i),
                         col(curve.ideal, i), col(curve.before, i),
                         col(curve.after, i), col(ratio_b, i), col(ratio_a, i)]
                        + [col(st.ee, i) for st in curve.stages]
                        + [col(r, i) for r in ratio_stage])


def _slug(name):
    """Column-safe lower-case form of a stage name."""
    return "".join(c if c.isalnum() else "_" for c in str(name)).lower()
