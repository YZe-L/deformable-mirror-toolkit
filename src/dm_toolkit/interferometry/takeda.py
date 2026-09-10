# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 3.1, 2026-07-01

"""Takeda wavefront reconstruction: interferogram -> 2-D wavefront map."""

import numpy as np
import cv2
from scipy.optimize import curve_fit
from scipy import fft as sfft

from .fringes import downsample, detect_aperture, unwrap2d  # noqa: F401

WAVELENGTH_NM = 520.0


def _remove_piston_tilt(wf, mask):
    """Subtract the best-fit plane (alignment tilt), keep real aberration.

    Args:
        wf: Wavefront map to process.
        mask: Boolean mask selecting valid samples.
    """
    ys, xs = np.where(mask)
    A = np.column_stack([np.ones_like(xs, float), xs.astype(float),
                         ys.astype(float)])
    coef, *_ = np.linalg.lstsq(A, wf[ys, xs], rcond=None)
    Y, X = np.mgrid[0:wf.shape[0], 0:wf.shape[1]]
    plane = coef[0] + coef[1] * X + coef[2] * Y
    return wf - plane


def _lobe_radius(absF, kx, ky, d_lobe, n_sigma=2.0):
    """Estimate the carrier side-lobe radius.

    Adaptive band-pass radius (px): the radial energy spread of the carrier
    side-lobe. Broad for curved fringes (wide lobe), tight for clean ones.

    Args:
        absF: Magnitude of the Fourier spectrum.
        kx: Horizontal spatial-frequency coordinates.
        ky: Vertical spatial-frequency coordinates.
        d_lobe: Distance from the spectrum centre to the carrier lobe.
        n_sigma: Number of n sigma.
    """
    h, w = absF.shape
    R = 0.8 * d_lobe  # Search window, clear of DC.
    Y, X = np.ogrid[:h, :w]
    rad = np.sqrt((X - kx) ** 2 + (Y - ky) ** 2)
    win = rad <= R
    e = absF[win] ** 2
    tot = float(e.sum())
    if tot <= 0:
        return 0.30 * d_lobe
    sigma = float(np.sqrt((e * rad[win] ** 2).sum() / tot))
    return n_sigma * sigma


def takeda_wavefront(img, mask, lobe_radius_frac=0.30, apodize=0.04):
    """FFT -> select carrier side-lobe -> phase -> wavefront (waves).

    Returns wavefront/display/wrapped maps, pv/rms, carrier, spectrum + lobe
    info for plotting.

    Args:
        img: Input image.
        mask: Boolean mask selecting valid samples.
        lobe_radius_frac: Lobe radius expressed as a fraction.
        apodize: Whether to apply edge apodisation.
    """
    h, w = img.shape
    soft = (cv2.GaussianBlur(mask.astype(float), (0, 0), apodize * max(h, w))
            if apodize else mask.astype(float))
    work = (img - img[mask].mean()) * soft
    F = np.fft.fftshift(sfft.fft2(work, workers=-1))
    cy, cx = h // 2, w // 2

    # Side-lobe = brightest point away from DC, one half-plane only.
    absF = np.abs(F)
    mag = absF.copy()
    Y, X = np.ogrid[:h, :w]
    dc = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) < 0.015 * min(h, w)
    mag[dc] = 0
    mag[:cy, :] = 0
    ky, kx = np.unravel_index(np.argmax(mag), mag.shape)

    # Carrier quality: a clear tilt carrier is an isolated lobe far from DC; a
    # closed-fringe (ring) interferogram has no carrier, so its strongest peak
    # is diffuse and hugs DC. Flag it so Takeda doesn't fail silently.
    dist_frac = float(np.hypot(kx - cx, ky - cy)) / (0.5 * min(h, w))
    nz = mag[mag > 0]
    bg = float(np.median(nz)) if nz.size else 1.0
    contrast = float(absF[ky, kx] / bg) if bg > 0 else 0.0
    c_level = ("none" if dist_frac < 0.06 or contrast < 5.0
               else "weak" if dist_frac < 0.12 or contrast < 15.0
               else "good")
    carrier_quality = dict(level=c_level, contrast=contrast,
                           dist_frac=dist_frac)

    # Sub-pixel carrier: centroid of |F|^2 around the integer peak.
    win = 4
    ys2, xs2 = slice(ky - win, ky + win + 1), slice(kx - win, kx + win + 1)
    patch = mag[ys2, xs2] ** 2
    gy, gx = np.arange(ys2.start, ys2.stop), np.arange(xs2.start, xs2.stop)
    tot = patch.sum()
    ky_s = (patch.sum(1) * gy).sum() / tot
    kx_s = (patch.sum(0) * gx).sum() / tot
    carrier = (kx_s - cx, ky_s - cy)

    # Band-pass the lobe with a Hann edge, then shift it to DC; a hard disc
    # rings. The radius adapts to the lobe's spread.
    d_lobe = np.hypot(kx - cx, ky - cy)
    rr = (lobe_radius_frac / 0.30) * _lobe_radius(absF, kx, ky, d_lobe)
    rr = float(np.clip(rr, 0.12 * d_lobe, 0.46 * d_lobe))
    d = np.sqrt((X - kx) ** 2 + (Y - ky) ** 2)
    lobe = np.where(d < rr, 0.5 * (1.0 + np.cos(np.pi * d / rr)), 0.0)
    sel = np.roll(np.roll(F * lobe, cy - ky, axis=0), cx - kx, axis=1)

    field = sfft.ifft2(np.fft.ifftshift(sel), workers=-1)
    wrapped = np.angle(field)
    unwrapped = unwrap2d(wrapped, mask)
    wf_waves = _remove_piston_tilt(unwrapped / (2 * np.pi), mask)

    wf_waves = wf_waves - wf_waves[mask].mean()
    pv = float(wf_waves[mask].max() - wf_waves[mask].min())
    rms = float(wf_waves[mask].std())
    disp = np.where(mask, wf_waves, np.nan)
    return dict(wavefront=wf_waves, display=disp,
                wrapped=np.where(mask, wrapped, np.nan),
                pv_waves=pv, rms_waves=rms, carrier=carrier,
                carrier_quality=carrier_quality,
                spectrum=np.log1p(absF), fft_centre=(cx, cy),
                lobe=(kx, ky, rr))


def quality_summary(rms_waves, wavelength_nm=WAVELENGTH_NM):
    """Optical-quality verdict from the wavefront RMS.

    Strehl ~ exp(-(2pi*RMS)^2); Marechal: RMS <= lambda/14 = diff-limited.

    Args:
        rms_waves: Sequence of rms wave values.
        wavelength_nm: Wavelength, in nanometres.
    """
    marechal = 1.0 / 14.0
    strehl = float(np.exp(-(2 * np.pi * rms_waves) ** 2))
    return dict(rms_waves=float(rms_waves),
                rms_nm=float(rms_waves * wavelength_nm),
                surface_rms_waves=float(rms_waves / 2),
                strehl=strehl,
                marechal_limit_waves=marechal,
                diffraction_limited=bool(rms_waves <= marechal))


def _cos2(x, a, L, phi, b0, b1):
    """cos^2 fringe model on a linear background.

    Args:
        x: Input coordinate or scalar value.
        a: Input array or scalar value.
        L: Azimuthal mode order.
        phi: Optical phase values, in radians.
        b0: Lower interval bound.
        b1: Upper interval bound.
    """
    return a * np.cos(np.pi * x / L + phi) ** 2 + b0 + b1 * x


def strip_cos2(img, center, radius, direction, mask=None, wavefront=None,
               max_bend=0.25):
    """Box-projected 1-D fringe profile + cos^2 fit -> true spacing L.

    Averaging region = rectangle inscribed in the aperture.

    Args:
        img: Input image.
        center: Centre coordinate of the analyzed region.
        radius: Aperture or spot radius.
        direction: Sweep or response direction.
        mask: Boolean mask selecting valid samples.
        wavefront: Wavefront map to process.
        max_bend: Maximum permitted bend.
    """
    cx, cy = float(center[0]), float(center[1])
    fx, fy = direction
    n = np.hypot(fx, fy)
    ax, ay = (fx / n, fy / n) if n else (1.0, 0.0)  # Unit along normal.
    px, py = -ay, ax  # Unit along fringes.

    if mask is None:
        Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
    ys, xs = np.where(mask)
    w = img[ys, xs]
    t = (xs - cx) * ax + (ys - cy) * ay
    s = -(xs - cx) * ay + (ys - cy) * ax

    band = 0.10
    if wavefront is not None:
        wv = wavefront[ys, xs]
        ti_all = np.round(t).astype(int)
        ti_all -= ti_all.min()
        for cand in (0.10, 0.05):
            inb = np.abs(s) <= cand * radius
            nb = np.bincount(ti_all[inb])
            sb = np.bincount(ti_all[inb], weights=wv[inb])
            s2b = np.bincount(ti_all[inb], weights=wv[inb] ** 2)
            good = nb > 10
            var = s2b[good] / nb[good] - (sb[good] / nb[good]) ** 2
            std = np.sqrt(np.clip(var, 0, None))
            bend = 3.3 * np.percentile(std, 90)
            band = cand
            if bend < max_bend:
                break

    def project(dth):
        """Bin-average box pixels along the normal rotated by dth (rad)."""
        tt = t[sel] + s[sel] * dth
        ti = np.round(tt).astype(int)
        ti -= ti.min()
        sums = np.bincount(ti, weights=w[sel])
        cnts = np.bincount(ti)
        prof = sums / np.maximum(cnts, 1)
        valid = cnts > 0.5 * cnts.max()
        return np.arange(len(prof))[valid].astype(float), prof[valid]

    S = band * radius
    T = np.sqrt(max(radius ** 2 - S ** 2, 0.0))
    sel = (np.abs(s) <= S) & (np.abs(t) <= T)

    cands = np.linspace(-0.035, 0.035, 15)
    scores = [project(c)[1].std() for c in cands]
    b = int(np.argmax(scores))
    if 0 < b < len(cands) - 1:  # Parabolic refine
        y0, y1, y2 = scores[b - 1], scores[b], scores[b + 1]
        denom = y0 - 2 * y1 + y2
        dth = cands[b] + (0.5 * (y0 - y2) / denom if denom else 0) * \
            (cands[1] - cands[0])
    else:
        dth = cands[b]
    x, prof = project(dth)
    x = x - x[0]

    h_img, w_img = img.shape
    L0 = 1.0 / max(np.hypot(fx / w_img, fy / h_img), 1e-9)

    # Matched-filter pre-fit (precise L + phase start for curve_fit)
    pc = prof - prof.mean()
    Ls = np.linspace(L0 * 0.85, L0 * 1.18, 241)
    C = (pc[None, :] * np.exp(-2j * np.pi * x[None, :] / Ls[:, None])).sum(1)
    k = int(np.argmax(np.abs(C)))
    if 0 < k < len(Ls) - 1:
        y0_, y1_, y2_ = np.abs(C[k - 1]), np.abs(C[k]), np.abs(C[k + 1])
        den = y0_ - 2 * y1_ + y2_
        Lb = Ls[k] + (0.5 * (y0_ - y2_) / den if den else 0) * (Ls[1] - Ls[0])
    else:
        Lb = Ls[k]
    phi0 = 0.5 * np.angle(C[k])

    p0 = [prof.max() - prof.min(), Lb, phi0, prof.min(), 0.0]
    bounds = ([0, Lb * 0.95, phi0 - np.pi / 2, -np.inf, -np.inf],
              [np.inf, Lb * 1.05, phi0 + np.pi / 2, np.inf, np.inf])
    nn = np.hypot(ax + px * dth, ay + py * dth)
    ax, ay = (ax + px * dth) / nn, (ay + py * dth) / nn
    px, py = -ay, ax

    endpts = ((cx - T * ax, cy - T * ay), (cx + T * ax, cy + T * ay))
    corners = [(cx + tt * ax + ss * px, cy + tt * ay + ss * py)
               for tt, ss in ((-T, -S), (T, -S), (T, S), (-T, S))]
    angle = float(np.degrees(np.arctan2(ay, ax)))

    try:
        popt, pcov = curve_fit(_cos2, x, prof, p0=p0, bounds=bounds,
                               maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        return dict(x=x, profile=prof, fit=_cos2(x, *popt), L=popt[1],
                    L_err=perr[1], ok=True, n_px=int(sel.sum()), band=band,
                    p0=endpts[0], p1=endpts[1], corners=corners, angle=angle)
    except Exception as e:
        return dict(x=x, profile=prof, fit=None, ok=False, msg=str(e),
                    n_px=int(sel.sum()), band=band,
                    p0=endpts[0], p1=endpts[1], corners=corners, angle=angle)


def analyse(img, lobe_radius_frac=0.30, shrink=0.8,
            wavelength_nm=WAVELENGTH_NM):
    """Full single-frame analysis -> dict for the six panels + save.

    Args:
        img: Input grayscale image.
        lobe_radius_frac: Fourier band-pass radius as a fraction of the
            spectrum size.
        shrink: Aperture-radius or image-downsampling factor.
        wavelength_nm: Wavelength, in nanometres.
    """
    mask, circ = detect_aperture(img, shrink=shrink)
    res = takeda_wavefront(img, mask, lobe_radius_frac=lobe_radius_frac)
    q = quality_summary(res["rms_waves"], wavelength_nm)
    strip = strip_cos2(img, (circ[0], circ[1]), circ[2], res["carrier"],
                       mask=mask, wavefront=res["wavefront"])
    cut = res["display"][int(round(circ[1])), :]
    return dict(img=img, mask=mask, circ=circ, res=res, strip=strip, q=q,
                cut=cut)


# Colour mapping
def lut_from_mpl(name, n=256):
    """Return lut from mpl.

    matplotlib colormap -> (n, 4) uint8 LUT (for fast numpy mapping in
    the worker thread, no Qt objects).

    Args:
        name: Display or identifier name.
        n: Number of requested samples or output points.
    """
    try:
        import matplotlib
        cmap = matplotlib.colormaps[name].resampled(n)
    except Exception:
        import matplotlib.cm as mcm
        cmap = mcm.get_cmap(name, n)
    return (cmap(np.arange(n)) * 255).astype(np.uint8)


def apply_lut(arr, lut, lo, hi, mask=None):
    """Map a float array to RGBA via LUT; mask=False -> transparent.

    Args:
        arr: Input array.
        lut: Colour lookup table.
        lo: Lower bound.
        hi: Upper bound.
        mask: Boolean mask selecting valid samples.
    """
    span = hi - lo
    norm = (arr - lo) / span if span > 0 else np.zeros_like(arr)
    norm = np.where(np.isfinite(norm), np.clip(norm, 0, 1), 0.0)
    idx = (norm * (len(lut) - 1)).astype(np.uint16)
    rgba = lut[idx]
    if mask is not None:
        rgba = rgba.copy()
        rgba[~mask, 3] = 0
    return rgba


# Saved figure
def _draw_spectrum(ax, res):
    """Spectrum panel: DC + both side-lobes + band-pass circle + arrow.

    Args:
        ax: Matplotlib axes to draw on.
        res: Result record or residual data.
    """
    spec = res["spectrum"]
    cx, cy = res["fft_centre"]
    kx, ky, rr = res["lobe"]
    H, W = spec.shape
    d = np.hypot(kx - cx, ky - cy)
    m = int(max(50, 3.0 * d))
    m = min(m, cx, cy, W - cx - 1, H - cy - 1)
    sub = spec[cy - m:cy + m, cx - m:cx + m]
    ax.imshow(sub, cmap="magma", extent=[cx - m, cx + m, cy + m, cy - m],
              vmin=np.percentile(sub, 75), vmax=np.percentile(sub, 99.9))
    th = np.linspace(0, 2 * np.pi, 120)
    kx2, ky2 = 2 * cx - kx, 2 * cy - ky
    ax.plot(cx, cy, "o", mfc="none", mec="cyan", ms=9, mew=1.5)
    ax.plot(kx + rr * np.cos(th), ky + rr * np.sin(th), "lime", lw=1.4)
    ax.plot(kx2, ky2, "x", color="orange", ms=8, mew=1.6)
    ax.annotate("", xy=(kx, ky), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.6))
    ax.text(cx, cy - m * 0.12, "DC", color="cyan", ha="center", fontsize=9)
    ax.text(kx, ky + m * 0.16, "+1 (selected)", color="lime", ha="center",
            fontsize=8)
    ax.text(kx2, ky2 - m * 0.10, "-1 (conjugate)", color="orange",
            ha="center", fontsize=8)
    ax.set_title("2. FFT spectrum")


def build_figure(fig, result, name, wavelength_nm=WAVELENGTH_NM):
    """Draw the 2x3 wavefront panel onto a matplotlib figure.

    Draw the 2x3 wavefront panel onto a matplotlib figure (same layout
    as the interferometer main_wavefront output).

    Args:
        fig: Matplotlib figure to update.
        result: Result data to format or display.
        name: Display or identifier name.
        wavelength_nm: Wavelength, in nanometres.
    """
    img, (cx, cy, r) = result["img"], result["circ"]
    res, strip, q = result["res"], result["strip"], result["q"]
    ax = fig.subplots(2, 3)

    ax[0, 0].imshow(img, cmap="gray")
    th = np.linspace(0, 2 * np.pi, 200)
    ax[0, 0].plot(cx + r * np.cos(th), cy + r * np.sin(th), "r-", lw=1)
    (px0, py0), (px1, py1) = strip["p0"], strip["p1"]
    ax[0, 0].plot([px0, px1], [py0, py1], "r--", lw=1.3)
    box = strip["corners"] + strip["corners"][:1]
    ax[0, 0].plot([p[0] for p in box], [p[1] for p in box], "g-", lw=1.6)
    ax[0, 0].set_title("1. interferogram + aperture + projection box")

    _draw_spectrum(ax[0, 1], res)

    ax[0, 2].imshow(res["wrapped"], cmap="twilight")
    ax[0, 2].set_title("3. wrapped phase")

    im = ax[1, 0].imshow(res["display"], cmap="jet")
    ax[1, 0].set_title(f"4. wavefront (waves)\nPV={res['pv_waves']:.2f}, "
                       f"RMS={res['rms_waves']:.3f}")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046, label="waves")

    ax[1, 1].plot(strip["x"], strip["profile"], "0.6", lw=1,
                  label=f"projected ({strip['n_px']} px averaged)")
    if strip["ok"]:
        ax[1, 1].plot(strip["x"], strip["fit"], "r--", lw=1.2,
                      label=f"cos^2 fit, L={strip['L']:.1f}px")
    ax[1, 1].set_title(f"1-D projected profile ({strip['angle']:.0f} deg, "
                       f"band +/-{strip['band']:.0%} r)")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_xlabel("distance along fringe normal (px)")
    ax[1, 1].set_ylabel("intensity")

    ax[1, 2].plot(res["display"][int(cy), :])
    ax[1, 2].set_title("wavefront cut")
    ax[1, 2].set_ylabel("waves")
    ax[1, 2].set_xlabel("x (px)")

    for a_ in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0]):
        a_.axis("off")
    dl = "diffraction-limited" if q["diffraction_limited"] \
        else "NOT diff-limited"
    fig.suptitle(f"wavefront -- {name} | PV={res['pv_waves']:.2f}, "
                 f"RMS={res['rms_waves']:.3f} waves | "
                 f"Strehl={q['strehl']:.2f} | {dl}", fontsize=12)


def save_figure(path, result, name, wavelength_nm=WAVELENGTH_NM, dpi=110):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(14, 9))
    build_figure(fig, result, name, wavelength_nm)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
