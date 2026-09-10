# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 2.4, 2026-07-15

"""Beam width numerics: D4sigma, 1/e^2 width and the ISO 11146 M^2 fit."""

import numpy as np
import cv2
from scipy.optimize import curve_fit


def as_gray(frame):
    """2-D float view of a frame (averages channels if colour)."""
    a = np.asarray(frame, dtype=float)
    return a.mean(axis=2) if a.ndim == 3 else a


def subtract_background(img, noise_k=2.0):
    """Remove the corner noise floor (mean + noise_k*std) and clip to >=0.

    The second moment is dominated by the wings, so the noise pedestal there
    must be zeroed, not just the mean offset, or it inflates the diameter.

    Args:
        img: Input image.
        noise_k: Noise threshold in estimated standard deviations.
    """
    h, w = img.shape
    k = max(4, min(h, w) // 20)
    corners = np.concatenate([img[:k, :k].ravel(), img[:k, -k:].ravel(),
                              img[-k:, :k].ravel(), img[-k:, -k:].ravel()])
    floor = float(corners.mean()) + noise_k * float(corners.std())
    return np.clip(img - floor, 0.0, None)


def _moments(img):
    """Centroid and axis-aligned 1-sigma widths (px) from marginal sums."""
    total = float(img.sum())
    if total <= 0:
        return None
    x = np.arange(img.shape[1], dtype=float)
    y = np.arange(img.shape[0], dtype=float)
    px = img.sum(axis=0)
    py = img.sum(axis=1)
    cx = float((px * x).sum() / total)
    cy = float((py * y).sum() / total)
    sx = float(np.sqrt(max((px * (x - cx) ** 2).sum() / total, 0.0)))
    sy = float(np.sqrt(max((py * (y - cy) ** 2).sum() / total, 0.0)))
    return cx, cy, sx, sy, total


def _locate(img):
    """Locate.

    Robust spot centre + radius: blur, threshold at 1/e^2 of peak, take the
    centroid and an area-equivalent radius. Seeds the moment iteration so wing
    noise over the full frame can't blow up the first estimate.
    """
    blur = cv2.GaussianBlur(img.astype(np.float32), (0, 0),
                            max(2.0, max(img.shape) / 200.0))
    peak = float(blur.max())
    if peak <= 0:
        return None
    mask = blur >= 0.135 * peak
    n = int(mask.sum())
    if n < 4:
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean()), float(np.sqrt(n / np.pi))


def locate_spot(img):
    """Locate spot.

    Round spot estimate for live previews: centre + area-equivalent radius
    (px) from the 1/e^2 footprint. Cheap and insensitive to fringes/speckle,
    so the overlay reads as a circle. Preview only -- M^2 uses d4sigma().
    """
    return _locate(subtract_background(as_gray(img)))


def brightest_resolved_pixel(img, max_reject=5, iso_ratio=0.5):
    """Coordinates of the brightest non-isolated pixel, or ``None``.

    This is the positional counterpart of :func:`robust_peak`: isolated hot
    pixels are discarded, but the centre is otherwise the actual brightest
    sensor location rather than a centroid or a heavily blurred maximum.

    Args:
        img: Input image.
        max_reject: Maximum number of rejected measurements.
        iso_ratio: Isotropy threshold expressed as an axis ratio.
    """
    work = as_gray(img).copy()
    h, w = work.shape
    for _ in range(max_reject):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        peak = float(work[y, x])
        if peak <= 0.0:
            return None
        neighbours = [float(work[yy, xx])
                      for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                      for yy, xx in ((y + dy, x + dx),)
                      if 0 <= yy < h and 0 <= xx < w]
        if neighbours and max(neighbours) >= iso_ratio * peak:
            return float(x), float(y)
        work[y, x] = 0.0
    return None


def locate_peak_spot(img):
    """Locate peak spot.

    Live-overlay spot location whose centre is the brightest *resolved*
    feature, rather than the footprint centroid returned by :func:`locate_spot`.

    Isolated hot pixels are rejected before the argmax.  The footprint is still
    used for the display radius.  This makes a crosshair follow the actual
    moving bright core while retaining a stable zoom size.
    """
    bg = subtract_background(as_gray(img))
    loc = _locate(bg)
    if loc is None:
        return None
    peak = brightest_resolved_pixel(bg)
    if peak is None:
        return None
    return peak[0], peak[1], float(loc[2])


def robust_peak(img, max_reject=5, iso_ratio=0.5):
    """Return the robust frame peak.

    Frame peak (raw counts) with isolated hot pixels rejected -- for
    auto-exposure and saturation guards, so a single stuck/hot pixel or a stray
    reflection speck can't pin the exposure near full-scale and drive the frame
    dark. A pixel only counts as the peak if its brightest 4-neighbour is at
    least `iso_ratio` of it; a real focused spot always has bright neighbours so
    its true peak is preserved, while a lone spike is dropped and we retry.

    Args:
        img: Input image.
        max_reject: Maximum number of rejected measurements.
        iso_ratio: Isotropy threshold expressed as an axis ratio.
    """
    # Compare in the native dtype; copy only when a pixel is actually rejected.
    a = np.asarray(img)
    if a.ndim == 3:
        a = a.mean(axis=2)  # float64 accumulator, as np.mean always uses.
    work = a
    h, w = work.shape
    for _ in range(max_reject):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        peak = float(work[y, x])
        if peak <= 0.0:
            return 0.0
        nb = 0.0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                nb = max(nb, float(work[yy, xx]))
        if nb >= iso_ratio * peak:
            return peak
        if work is a:
            work = a.copy()  # Never mutate the caller's frame.
        work[y, x] = 0  # Lone hot pixel -> drop and retry.
    return float(a.max())  # Give up: fall back to the raw max.


def d4sigma(img, k=3.0, iters=3):
    """Measure the ISO 11146 D4-sigma beam diameter.

    ISO 11146 D4sigma: background subtract, then iterate the second-moment
    integration aperture until it tracks the spot. k is the aperture size as a
    multiple of the beam diameter (ISO ~3), so the elliptical radius is
    k*2*sigma. Returns centroid, sigmas and diameters dx=dy=4*sigma (px).

    Args:
        img: Input image.
        k: Scale or model coefficient.
        iters: Number of optimisation or refinement iterations.
    """
    img = subtract_background(img)
    loc = _locate(img)
    if loc is None:
        return None
    cx, cy, r0 = loc
    sx, sy = r0, r0  # Seed widths (used if moments fail)
    ax, ay = 2.0 * r0, 2.0 * r0  # Bounded initial aperture.
    total = 0.0
    for _ in range(max(1, iters)):
        # Moments over the elliptical aperture only: slice its bounding box out
        # first so the mask/moment work never touches the rest of the frame.
        x0, x1 = max(0, int(cx - ax)), min(img.shape[1], int(cx + ax) + 2)
        y0, y1 = max(0, int(cy - ay)), min(img.shape[0], int(cy + ay) + 2)
        sub = img[y0:y1, x0:x1]
        if sub.size == 0:
            break
        yy, xx = np.ogrid[:sub.shape[0], :sub.shape[1]]
        mask = (((xx - (cx - x0)) / ax) ** 2
                + ((yy - (cy - y0)) / ay) ** 2 <= 1.0)
        m = _moments(np.where(mask, sub, 0.0))
        if m is None:
            break
        cx, cy, sx, sy, total = m
        cx, cy = cx + x0, cy + y0
        ax, ay = max(2.0 * k * sx, 3.0), max(2.0 * k * sy, 3.0)
    return dict(cx=cx, cy=cy, sx=sx, sy=sy,
                dx=4.0 * sx, dy=4.0 * sy, total=total)


def _gaussian(x, a, x0, w, c):
    return a * np.exp(-2.0 * (x - x0) ** 2 / w ** 2) + c


def _gauss_diameter_1d(profile, sigma_guess):
    """1/e^2 diameter (px) from a Gaussian fit to one marginal profile.

    Args:
        profile: Device or calibration profile.
        sigma_guess: Sequence of sigma gues values.
    """
    x = np.arange(len(profile), dtype=float)
    a0 = float(profile.max() - profile.min())
    if a0 <= 0:
        return np.nan
    p0 = (a0, float(np.argmax(profile)), max(2.0 * sigma_guess, 2.0),
          float(profile.min()))
    try:
        popt, _ = curve_fit(_gaussian, x, profile, p0=p0, maxfev=5000)
    except Exception:
        return np.nan
    return 2.0 * abs(popt[2])


def gaussian_1e2(img, moments):
    """1/e^2 diameters (px) in x and y from Gaussian fits to the marginals.

    Args:
        img: Input image.
        moments: Sequence of moment values.
    """
    img = subtract_background(img)
    return (_gauss_diameter_1d(img.sum(axis=0), moments["sx"]),
            _gauss_diameter_1d(img.sum(axis=1), moments["sy"]))


def _gauss2d(coords, a, x0, y0, wx, wy, c):
    x, y = coords
    return a * np.exp(-2.0 * ((x - x0) ** 2 / wx ** 2
                              + (y - y0) ** 2 / wy ** 2)) + c


def fit_gaussian_2d(img, sat_level=None, max_size=200):
    """Fit gaussian 2d.

    2D Gaussian fit to the spot, excluding clipped pixels so a saturated
    core still yields the true width and peak. Returns amplitude, centre,
    1/e^2 radii wx/wy and offset in full-res px, or None. Runs on a
    downsampled copy (capture/offline only -- too heavy for the live path).

    Args:
        img: Input image.
        sat_level: Sensor saturation level.
        max_size: Maximum processing dimension, in pixels.
    """
    orig = as_gray(img)
    g = subtract_background(orig)
    s = max(1, int(max(g.shape) / max_size))
    small, orig_s = g[::s, ::s], orig[::s, ::s]
    loc = _locate(small)
    if loc is None:
        return None
    cx0, cy0, r0 = loc
    yy, xx = np.mgrid[:small.shape[0], :small.shape[1]].astype(float)
    keep = ((xx - cx0) ** 2 + (yy - cy0) ** 2) <= (3.0 * r0) ** 2
    if sat_level is not None:
        keep &= orig_s < 0.99 * sat_level  # Drop the clipped core.
    if keep.sum() < 12:
        return None
    p0 = (float(small.max()), cx0, cy0, max(r0, 2.0), max(r0, 2.0), 0.0)
    try:
        popt, _ = curve_fit(_gauss2d, (xx[keep], yy[keep]), small[keep],
                            p0=p0, maxfev=8000)
    except Exception:
        return None
    a, x0, y0, wx, wy, c = popt
    return dict(a=float(a), cx=float(x0 * s), cy=float(y0 * s),
                wx=abs(float(wx)) * s, wy=abs(float(wy)) * s, c=float(c))


def reconstruct_display(img, sat_level):
    """Reconstruct display.

    Display-only: fill the clipped core with the fitted Gaussian model so an
    overexposed spot still shows its full intensity shape. Real data elsewhere.

    Args:
        img: Input image.
        sat_level: Sensor saturation level.
    """
    orig = as_gray(img)
    if float(orig.max()) < 0.99 * sat_level:
        return orig
    fit = fit_gaussian_2d(orig, sat_level=sat_level)
    if fit is None:
        return orig
    yy, xx = np.mgrid[:orig.shape[0], :orig.shape[1]].astype(float)
    model = _gauss2d((xx, yy), fit["a"], fit["cx"], fit["cy"],
                     fit["wx"], fit["wy"], fit["c"])
    out = orig.astype(float).copy()
    sat = orig >= 0.99 * sat_level
    out[sat] = np.maximum(out[sat], model[sat])
    return out


def measure_spot(img, k=3.0, iters=3, with_1e2=True, sat_level=None):
    """Full spot measurement: D4sigma (+ optional 1/e^2).

    Diameters in px. When sat_level is given and the spot is clipped, 1/e^2
    comes from a 2D Gaussian fit to the unsaturated wings (saturation-robust);
    D4sigma is left as-is and flagged unreliable.

    Args:
        img: Input image.
        k: Scale or model coefficient.
        iters: Number of optimisation or refinement iterations.
        with_1e2: Whether to 1e2.
        sat_level: Sensor saturation level.
    """
    img = as_gray(img)
    m = d4sigma(img, k=k, iters=iters)
    if m is None:
        return None
    m["saturated"] = sat_level is not None and float(img.max()) >= 0.99 * sat_level
    m["reconstructed"] = False
    if not with_1e2:
        m["dx_1e2"] = m["dy_1e2"] = np.nan
        return m
    if m["saturated"]:
        fit = fit_gaussian_2d(img, sat_level=sat_level)
        if fit is not None:
            m["dx_1e2"], m["dy_1e2"] = 2.0 * fit["wx"], 2.0 * fit["wy"]
            m["reconstructed"] = True
            return m
    m["dx_1e2"], m["dy_1e2"] = gaussian_1e2(img, m)
    return m


def fit_m2(z_mm, d_um, dstd_um, wavelength_nm):
    """ISO 11146 M^2 from a fit of w^2 (radius squared) against z.

    w(z)^2 = A + B z + C z^2 (numpy.polyfit), then z0 = -B/2C,
    w0^2 = A - B^2/4C, M^2 = (pi/lambda) sqrt(AC - B^2/4). The fit
    covariance is propagated to sigma(M^2). Per-point std (if any) weights
    the fit by 1/var(w^2).

    Args:
        z_mm: Z, in millimetres.
        d_um: D, in micrometres.
        dstd_um: Dstd, in micrometres.
        wavelength_nm: Wavelength, in nanometres.
    """
    z = np.asarray(z_mm, float) * 1e-3  # M
    w = (np.asarray(d_um, float) / 2.0) * 1e-6  # Radius, m
    y = w ** 2
    lam = float(wavelength_nm) * 1e-9
    if len(z) < 3:
        return dict(ok=False, reason="need >= 3 points")

    weights = None
    s = np.asarray(dstd_um, float)
    if np.all(np.isfinite(s)) and np.all(s > 0):
        sigma_y = 2.0 * w * (s / 2.0 * 1e-6)  # Var propagation
        weights = 1.0 / np.maximum(sigma_y, 1e-18)

    coeffs, cov = np.polyfit(z, y, 2, w=weights, cov=True)
    C, B, A = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    Q = A * C - B * B / 4.0
    if C <= 0 or Q <= 0:
        return dict(ok=False, reason="not a focus (open-up parabola needed)")

    m2 = np.pi / lam * np.sqrt(Q)
    w0 = np.sqrt(A - B * B / (4.0 * C))  # M
    z0 = -B / (2.0 * C)  # M
    zr = np.pi * w0 ** 2 / (m2 * lam)  # M

    # Error propagation, cov order [C, B, A]
    pre = np.pi / lam / (2.0 * np.sqrt(Q))
    grad = np.array([pre * A, pre * (-B / 2.0), pre * C])  # D/dC, d/dB, d/dA.
    m2_err = float(np.sqrt(max(grad @ cov @ grad, 0.0)))

    model = A + B * z + C * z ** 2
    ss_res = float(np.sum((y - model) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return dict(ok=True, reason="ok", M2=float(m2), M2_err=m2_err,
                w0_um=float(w0 * 1e6), z0_mm=float(z0 * 1e3),
                zR_mm=float(zr * 1e3), r2=r2, A=A, B=B, C=C)


def predicted_waist_um(focal_mm, divergence_mrad):
    """Predict the focused waist in micrometres.

    Sanity-check focused waist w0 = f * |theta| (lens angle->position); the
    divergence sign is only convention.

    Args:
        focal_mm: Focal, in millimetres.
        divergence_mrad: Far-field divergence, in milliradians.
    """
    return float(focal_mm * 1e-3 * abs(divergence_mrad) * 1e-3 * 1e6)


def rayleigh_mm(w0_um, m2, wavelength_nm):
    """Rayleigh range zR = pi w0^2 / (M^2 lambda).

    Args:
        w0_um: W0, in micrometres.
        m2: Beam-propagation quality factor.
        wavelength_nm: Wavelength, in nanometres.
    """
    w0 = w0_um * 1e-6
    return float(np.pi * w0 ** 2 / (m2 * wavelength_nm * 1e-9) * 1e3)


def save_fit_figure(path, points, fit_x, fit_y, meta):
    """Save fit figure.

    Publication-style caustic plot: symmetric +/- beam radius vs z, x and y
    data points with fit envelopes and the waist line. points: list of dicts
    with z_mm, dx_um, dy_um.

    Args:
        path: Filesystem path used by the operation.
        points: Measurement or plot points.
        fit_x: Horizontal coordinates used for fitting.
        fit_y: Vertical coordinates used for fitting.
        meta: Metadata associated with the measurement.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "font.family": "DejaVu Sans",
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.grid": True, "grid.color": "#d9d9d9",
        "grid.linewidth": 0.7, "axes.edgecolor": "#444444", "axes.linewidth": 1.0,
        "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
    })

    z = np.array([p["z_mm"] for p in points], float)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for key, fit, color, label in (("dx_um", fit_x, "#1f6feb", "x"),
                                   ("dy_um", fit_y, "#e8710a", "y")):
        r = np.array([p[key] for p in points], float) / 2.0 / 1000.0  # Mm
        ax.plot(z, r, "o", color=color, ms=4, alpha=0.8)
        ax.plot(z, -r, "o", color=color, ms=4, alpha=0.8,
                label=f"{label} data")
        if fit and fit.get("ok"):
            zz = np.linspace(z.min(), z.max(), 400)
            w = np.sqrt(np.clip(fit["A"] + fit["B"] * zz * 1e-3
                                + fit["C"] * (zz * 1e-3) ** 2, 0, None)) * 1e3
            ax.plot(zz, w, color=color, lw=2.0)
            ax.plot(zz, -w, color=color, lw=2.0,
                    label=rf"{label} fit  $M^2$={fit['M2']:.2f}$\pm${fit['M2_err']:.2f}")
            ax.axvline(fit["z0_mm"], color=color, ls=":", lw=1.0, alpha=0.6)

    ax.axhline(0, color="#888888", lw=0.8)
    ax.set(xlabel="z / mm", ylabel="beam radius / mm",
           title=meta.get("title", "M² caustic (ISO 11146)"))
    ax.legend(loc="upper center", ncol=4)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
