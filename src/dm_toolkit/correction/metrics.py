# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 4.6, 2026-08-29

"""Turn one camera frame into one higher-is-better score plus diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import cv2
from scipy.special import j1

from ..beam.spot_quality import extract_features
from ..beam import beam
from . import settings as S
from . import ee_curve as EE
from . import fastmath as FM

# Smallest Airy-core radius, in resampled px, that measure() will accept. Below
# this the core is unresolved and every r0-anchored score is nonsense.
_MIN_R0_GRID_PX = 3.0

# r0 = 1.22*lambda*F# while the incoherent MTF dies at f_c = 1/(lambda*F#), so
# the diffraction cutoff is 1.22/r0 cyc/px and a frequency normalised to it is
# m = f * r0 / 1.22. Used by _psd_band to make its band optics-independent.
_R0_CUTOFF = 1.22

# Warning level for descriptors that become unreliable near the noise floor.
MIN_PEAK_SNR = 10.0


@dataclass
class SecondMomentROI:
    """One fixed full-frame aperture used only by the second moment.

    Other metrics keep their adaptive whole-pattern aperture.  Keeping this
    aperture fixed is what makes every second-moment reading belong to the same
    quadratic function of the mirror command.
    """

    cx: float
    cy: float
    radius: float
    margin_pct: float = 0.0

    @classmethod
    def from_reading(cls, reading, margin_pct=30.0):
        """Lock an aperture around a valid reading with percentage headroom.

        Args:
            reading: Spot reading whose adaptive whole-pattern aperture seeds
                the fixed circle.
            margin_pct: Extra radius as a percentage of the detected radius.

        Returns:
            A full-frame fixed second-moment aperture.

        Raises:
            ValueError: If the reading has no finite, positive aperture.
        """
        radius = float(getattr(reading, "photometry_aperture_radius", 0.0))
        cx = float(getattr(reading, "photometry_aperture_cx", float("nan")))
        cy = float(getattr(reading, "photometry_aperture_cy", float("nan")))
        margin = max(0.0, float(margin_pct))
        if not (radius > 0 and np.isfinite(cx) and np.isfinite(cy)):
            raise ValueError("cannot lock a second-moment ROI from this reading")
        return cls(cx=cx, cy=cy, radius=radius * (1.0 + margin / 100.0),
                   margin_pct=margin)

    def contains(self, reading, tolerance_px=1.0):
        """Check whether this circle contains an adaptive spot aperture.

        Args:
            reading: Spot reading whose adaptive aperture is tested.
            tolerance_px: Numerical allowance at the circle boundary.

        Returns:
            True if the entire detected aperture lies inside this fixed ROI.
        """
        return bool(self.containment(reading, tolerance_px)["contains"])

    def containment(self, reading, tolerance_px=1.0):
        """Return the values used by the whole-pattern containment guard.

        Args:
            reading: Spot reading whose adaptive aperture is tested.
            tolerance_px: Numerical allowance at the fixed-circle boundary.

        Returns:
            A dictionary containing the verdict, the centre displacement, the
            adaptive and required radii, the fixed radius, and the positive
            overshoot beyond the tolerated fixed boundary.
        """
        cx = float(getattr(reading, "photometry_aperture_cx", float("nan")))
        cy = float(getattr(reading, "photometry_aperture_cy", float("nan")))
        radius = float(getattr(reading, "photometry_aperture_radius", 0.0))
        valid = bool(radius > 0 and np.isfinite(cx) and np.isfinite(cy))
        distance = (float(np.hypot(cx - self.cx, cy - self.cy))
                    if valid else float("nan"))
        required = distance + radius if valid else float("nan")
        allowed = self.radius + max(0.0, float(tolerance_px))
        contains = bool(valid and required <= allowed)
        overshoot = max(0.0, required - allowed) if valid else float("inf")
        return dict(
            contains=contains,
            center_distance_px=distance,
            adaptive_radius_px=radius,
            required_radius_px=required,
            fixed_radius_px=float(self.radius),
            tolerance_px=max(0.0, float(tolerance_px)),
            overshoot_px=float(overshoot))


@dataclass
class SpotReading:
    """One measured frame: the score plus everything worth logging."""
    score: float  # Primary objective, higher is better.
    pib: float  # Power-in-bucket fraction [0..1]
    peak_norm: float  # Energy-normalised peak (Strehl proxy)
    r_ee80: float  # 80% encircled-energy radius, px.
    d4sigma: float  # D4sigma diameter, px.
    ellipticity: float  # 0 round .. ->1 elongated.
    asymmetry: float  # Coma-like skew
    n_spots: int  # Bright blobs detected (want 1)
    total_energy: float  # Whole-frame energy after background subtract.
    peak_frac: float  # Raw peak / full scale (saturation guard)
    cx: float
    cy: float
    has_signal: bool  # A MEASURABLE spot was found (see peak_snr)
    # Peak height over the background noise sigma. Diagnostic only: a smeared
    # spot has a low peak and must still be scored.
    peak_snr: float = 0.0
    # Aperture energy over the noise of that sum. This is what has_signal
    # gates on: energy survives the mirror spreading the light.
    energy_snr: float = 0.0
    halo_frac: float = 0.0  # Energy outside the core crop (ring/halo proxy)
    ring_contrast: float = 0.0  # Secondary radial peak / core peak.
    aberration: str = "round"  # Dominant aberration fingerprint (diagnostic)
    # DIAGNOSTIC ONLY -- kept for continuity with older logs. Not in the shape
    # gate: its per-ring std/mean explodes on the Airy's dark rings, so it rates
    # a perfect spot worse than an aberrated one. Use azim_m1..m4 instead.
    ring_asym: float = 0.0  # azimuthal (ring) non-uniformity; 0 = symmetric
    # Azimuthal Fourier orders of the intensity, 0 = circular
    # (azimuthal_orders).
    # One per symmetry, so the gate can price each aberration separately.
    azim_m1: float = 0.0  # 1-fold: coma / decentre.
    azim_m2: float = 0.0  # 2-fold: astigmatism away from best focus.
    azim_m3: float = 0.0  # 3-fold: trefoil
    azim_m4: float = 0.0  # 4-fold: the diamond astigmatism makes AT focus.
    far_halo_frac: float = 0.0  # Energy in 10..20 Airy radii / energy <20 r0.
    core_ee_frac: float = 0.0  # Energy inside r0 / energy inside 20 r0.
    sidelobe_frac: float = 0.0  # Energy in 1..5 r0 / energy inside 20 r0.
    cross_sidelobe: float = 0.0  # Worst side peak on the X/Y cuts / main peak.
    cross_asym: float = 0.0  # Mirror asymmetry of the X/Y cuts (0 = even)
    sharpness: float = 0.0  # sum(I^2)/(sum I)^2 in the aperture; centre-free
    # Image PSD over one normalised-frequency annulus, over DC (see _psd_band).
    # NaN without r0: there is then no cutoff to normalise against.
    psd_band: float = float("nan")
    # Centroid-referenced second moment, px^2 at FULL-FRAME scale. Raw and
    # smaller-is-better: the modal solvers read THIS field, never the 0..1
    # score below, because their arithmetic needs the untransformed quadratic.
    second_moment: float = float("nan")
    # Blurred peak / energy inside the 20 r0 aperture (signed baseline): the
    # Strehl proxy. Not whole-frame -- see _aperture.
    peak_whole: float = 0.0
    # Absolute 0..1 quality against the diffraction-limited spot at the same
    # sampling. NaN when the optics are unconfigured.
    strehl: float = float("nan")  # peak_whole / ideal peak_whole.
    ee_strehl: float = float("nan")  # Power-in-bucket / ideal power-in-bucket.
    conc_score: float = float("nan")  # Ideal r_ee80 / r_ee80.
    size_score: float = float("nan")  # Ideal d4sigma / d4sigma.
    sym_score: float = float("nan")  # Shape gate / ideal shape gate.
    sharp_score: float = float("nan")  # Sharpness / ideal sharpness.
    psd_score: float = float("nan")  # psd_band / ideal psd_band.
    # Reciprocal <r^2>, a monotonic primitive for the run-relative 0..1 display
    # score only: the Airy second moment has no aperture-independent value, so
    # this is not a Strehl-like absolute. Modal solvers read `second_moment`.
    second_moment_score: float = float("nan")
    # The adaptive aperture used by every legacy metric. These diagnostics seed
    # a run-wide fixed aperture when second moment is selected.
    photometry_aperture_cx: float = float("nan")
    photometry_aperture_cy: float = float("nan")
    photometry_aperture_radius: float = float("nan")
    # Actual second-moment aperture and its outer-annulus energy diagnostic.
    second_moment_roi_radius: float = float("nan")
    second_moment_edge_frac: float = float("nan")
    second_moment_roi_clipped: bool = False
    # Milliseconds this call spent removing the baseline: the fixed-pattern
    # subtract plus the level/sigma clip. Measured in both modes, so the
    # measured-reference and corner-estimate costs are directly comparable.
    bg_ms: float = 0.0
    # The averaged image this reading came from, attached only for the
    # controllers that need pixels (`wavefront.PseudoWfs`); `measure` leaves
    # it None so readings kept for a whole run stay small.
    frame: object = None


def _blur_sigma(shape):
    """Peak-reading blur.

    Also the reference's blur -- a Strehl-style ratio is only fair when the
    ideal spot is blurred exactly like the measured one.
    """
    return max(1.5, max(shape) / 300.0)


def _corners(gray):
    """Return the flattened frame-corner samples.

    The four corner patches of a frame, flattened -- the fallback background
    sample when no measured reference is available.
    """
    h, w = gray.shape
    k = max(4, min(h, w) // 20)
    return np.concatenate([gray[:k, :k].ravel(), gray[:k, -k:].ravel(),
                           gray[-k:, :k].ravel(), gray[-k:, -k:].ravel()])


def _baseline(gray, backend, ref=None):
    """Estimate the background level and uncertainty.

    (level, sigma, level_err) of the background: the measured reference if
    the caller has one, else this frame's own corners.

    level_err is the standard error of the LEVEL -- the term a run-level
    reference shrinks most, and the one _energy_snr multiplies by the aperture
    pixel count. 1.253 = sqrt(pi/2), a median's s.e. against a mean's.

    Args:
        gray: Grayscale image data.
        backend: Numerical backend to use.
        ref: Reference data for relative measurements.
    """
    if ref is not None:
        return ref
    c = _corners(gray)
    sigma = FM.std(c, backend)
    return (FM.median(c, backend), sigma,
            1.253 * sigma / max(np.sqrt(c.size), 1.0))


def _energy_snr(gray, cx, cy, r_out, backend=FM.BACKEND_NUMPY, ref=None):
    """Measure aperture energy relative to summed background noise.

    Args:
        gray: Grayscale image data.
        cx: Horizontal centre coordinate, in pixels.
        cy: Vertical centre coordinate, in pixels.
        r_out: Outer photometry-aperture radius, in pixels.
        backend: Numerical backend to use.
        ref: Reference data for relative measurements.
    """
    base, sigma, base_err = _baseline(gray, backend, ref)
    x0, x1, y0, y1 = _spot_window(gray.shape, cx, cy, r_out)
    crop = gray[y0:y1, x0:x1] - base
    Y, X = np.ogrid[0:crop.shape[0], 0:crop.shape[1]]
    inside = np.hypot(X - (cx - x0), Y - (cy - y0)) <= r_out
    n_ap = int(inside.sum())
    if n_ap <= 0:
        return 0.0
    energy = float(crop[inside].sum())
    if sigma <= 0:  # Quantised dark: any excess is signal.
        return float("inf") if energy > 0 else 0.0
    noise = np.hypot(sigma * np.sqrt(n_ap), base_err * n_ap)
    return float(energy / max(noise, 1e-9))


def _peak_snr(gray, backend=FM.BACKEND_NUMPY):
    """Measure blurred peak height relative to background noise.

    Stays on its own corners even when a reference exists: the blur changes
    sigma, and this is a diagnostic -- has_signal gates on _energy_snr.

    Args:
        gray: Grayscale image data.
        backend: Numerical backend to use.
    """
    blur = cv2.GaussianBlur(np.asarray(gray, np.float32), (0, 0),
                            _blur_sigma(gray.shape))
    corners = _corners(blur)
    base, sigma = FM.mean(corners, backend), FM.std(corners, backend)
    peak = float(blur.max())
    if sigma <= 0:
        # A perfectly flat background: erring permissive is deliberate, since
        # a false rejection stalls the loop and the energy guard still
        # catches a false accept.
        return 0.0 if peak <= base else float("inf")
    return float((peak - base) / sigma)


def _pattern_footprint(bg, r0=0.0, blur=None):
    """Estimate the centre and containment radius of the full pattern.

    Args:
        bg: Background reference used for correction.
        r0: Diffraction-limited reference radius, in pixels.
        blur: Blur width applied to the simulated or measured image.
    """
    if blur is None:  # Callers that already have one pass it.
        blur = cv2.GaussianBlur(bg.astype(np.float32), (0, 0),
                                _blur_sigma(bg.shape))
    peak = float(blur.max())
    if peak <= 0:
        return None
    mask = blur >= 0.135 * peak
    if int(mask.sum()) < 4:
        return None
    ys, xs = np.nonzero(mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    d = np.hypot(xs - cx, ys - cy)
    r_cover = float(np.percentile(d, 98.0))
    return cx, cy, max(r_cover, 3.0 * r0 if r0 > 0 else 6.0)


def _bucket_center(blur, pcx, pcy, r_search):
    """Brightest point within r_search of the pattern centre.

    The bucket belongs on the pattern's own centre, not on whichever speckle
    is brightest; a small search around it absorbs jitter and the centroid
    pull of coma without letting the bucket run off to a fragment.

    Args:
        blur: Blur width applied to the simulated or measured image.
        pcx: Horizontal photometry-bucket centre, in pixels.
        pcy: Vertical photometry-bucket centre, in pixels.
        r_search: Maximum radius searched for a bucket centre.
    """
    h, w = blur.shape
    lim = int(max(r_search, 1.0))
    ix, iy = int(round(pcx)), int(round(pcy))
    x0, x1 = max(0, ix - lim), min(w, ix + lim + 1)
    y0, y1 = max(0, iy - lim), min(h, iy + lim + 1)
    win = blur[y0:y1, x0:x1]
    if win.size == 0:
        return float(pcx), float(pcy)
    dy, dx = np.unravel_index(int(np.argmax(win)), win.shape)
    return float(x0 + dx), float(y0 + dy)


def _peak_center(bg):
    """Brightest resolved point (bucket centre) plus a blob-counting blur."""
    blur = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), _blur_sigma(bg.shape))
    peak = beam.brightest_resolved_pixel(bg)
    if peak is None:
        y, x = np.unravel_index(int(np.argmax(blur)), blur.shape)
        peak = float(x), float(y)
    return peak[0], peak[1], blur


def _airy(r0_px, size):
    """Diffraction-limited Airy intensity on a size x size grid, r0 in px.

    Args:
        r0_px: R0, in pixels.
        size: Requested output size.
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    rr = np.hypot(x - (size - 1) / 2.0, y - (size - 1) / 2.0)
    v = np.maximum(EE._V_R0 * rr / max(float(r0_px), 1e-9), 1e-9)
    return ((2.0 * j1(v) / v) ** 2).astype(np.float32)


@lru_cache(maxsize=64)
def _ideal_ref(r0_grid, sigma, bucket_grid, symmetry_weight, psd_lo, psd_hi):
    """Measure an ideal spot with the same metrics used for real frames.

    Args:
        r0_grid: Diffraction-limited radius on the simulation grid.
        sigma: Estimated standard deviation or spot width.
        bucket_grid: Boolean mask for the diffraction-limited energy bucket.
        symmetry_weight: Weight assigned to azimuthal symmetry.
        psd_lo: Lower spatial-frequency limit of the reference band.
        psd_hi: Upper spatial-frequency limit of the reference band.
    """
    size = int(np.clip(round(44.0 * r0_grid), 96, 400))
    # Integrate over pixels rather than point-sampling the Airy pattern. A
    # four-times oversampled, area-averaged reference matches the sensor and
    # later resampling, so a measured perfect spot can reach a score of 1.0.
    _SS = 4
    img = cv2.resize(_airy(r0_grid * _SS, size * _SS) * 3000.0, (size, size),
                     interpolation=cv2.INTER_AREA)
    bg = beam.subtract_background(img)
    feats = extract_features(img)
    if not feats.ok:
        return None
    blur = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), sigma)
    c = (size - 1) / 2.0
    # Same aperture rule as a real frame (whole-pattern footprint, floored at
    # 20 r0), so the ratios every absolute metric takes stay valid. For an Airy
    # the footprint is ~1.5 r0, so the floor wins and this is 20 r0 as before.
    fp = _pattern_footprint(bg, r0_grid, blur)
    r_out = max(2.0 * fp[2], 20.0 * r0_grid, 30.0) if fp else 20.0 * r0_grid
    r_out = min(r_out, 0.5 * size)
    crop, rr, inside, ap_energy, ccx, ccy = _aperture(img, c, c, r_out)
    if ap_energy <= 0:
        return None
    r_spot = (max(6.0 * feats.r_ee80, 30.0) if np.isfinite(feats.r_ee80)
              else 0.4 * size)
    cs, ca = cross_profile_metrics(bg, c, c, feats.r_ee80, r0_grid)
    am = azimuthal_orders(bg, c, c, r0_grid)  # Same code path as a real frame,
    ideal = SpotReading(  # So its own biases cancel below.
        0.0, _pib(crop, rr, ap_energy, ccx, ccy, bucket_grid, 1),
        feats.peak_norm, float(feats.r_ee80), float(feats.d4sigma),
        feats.ellipticity, feats.asymmetry, _count_spots(blur), ap_energy,
        0.0, c, c, True,
        halo_frac=feats.halo_frac, ring_contrast=feats.ring_contrast,
        ring_asym=azimuthal_asymmetry(bg, c, c,
                                      r_max=min(0.4 * size, r_spot)),
        azim_m1=am[1], azim_m2=am[2], azim_m3=am[3], azim_m4=am[4],
        cross_sidelobe=cs, cross_asym=ca,
        peak_whole=float(blur.max()) / ap_energy)
    # Per-term deadband for the shape gate: what a perfect spot reads on each
    # symptom, so `shape_quality` charges only the excess.
    return dict(pib=ideal.pib, peak_whole=ideal.peak_whole,
                r_ee80=ideal.r_ee80, d4sigma=ideal.d4sigma,
                # Noiseless reference, so no sigma correction to make.
                sharpness=_sharpness(crop, inside, 0.0),
                psd_band=_psd_band(crop, rr, r_out, r0_grid,
                                   psd_lo, psd_hi, 0.0),
                # The diffraction floor r_diff^2 of the identity in
                # settings.METRIC_SECOND_MOMENT: what a perfect spot still reads in this
                # same aperture, and the level the coarse stage cannot go below.
                second_moment=_second_moment(crop, inside, ccx, ccy),
                shape_floor={a: float(getattr(ideal, a)) for a in
                             ("azim_m1", "azim_m2", "azim_m3", "azim_m4",
                              "ellipticity", "ring_contrast")})


def _absolute(value, ideal, higher_is_better=True):
    """Measured over ideal (or its reciprocal), floored at 0; 1.0 is the limit.

    NaN when the optics are unknown. Deliberately not capped at 1.0: a cap
    removes the gradient near the ideal, and the ideal carries model error.

    Args:
        value: Measured metric value.
        ideal: The same metric for the ideal Airy pattern.
        higher_is_better: Whether a larger metric means a better spot.
    """
    if ideal is None or not (np.isfinite(value) and np.isfinite(ideal)):
        return float("nan")
    if higher_is_better:
        return float(max(value / max(ideal, 1e-12), 0.0))
    return float(max(ideal / max(value, 1e-12), 0.0))


def _spot_window(shape, cx, cy, half):
    """Clamped square crop bounds around (cx, cy): x0, x1, y0, y1.

    Args:
        shape: Requested array or output shape.
        cx: Horizontal centre coordinate, in pixels.
        cy: Vertical centre coordinate, in pixels.
        half: Half-resolution input data.
    """
    h, w = shape
    half = int(max(half, 4))
    x0, x1 = max(0, int(cx) - half), min(w, int(cx) + half + 1)
    y0, y1 = max(0, int(cy) - half), min(h, int(cy) + half + 1)
    return x0, x1, y0, y1


def azimuthal_asymmetry(bg, cx, cy, n_theta=24, r_max=None, r_step=2.0):
    """Measure azimuthal intensity variation around the spot centre.

    Args:
        bg: Background reference used for correction.
        cx: Horizontal centre coordinate, in pixels.
        cy: Vertical centre coordinate, in pixels.
        n_theta: Number of n theta.
        r_max: Maximum radial coordinate or aperture radius.
        r_step: Radial bin width, in pixels.
    """
    h, w = bg.shape
    if r_max is None:
        r_max = 0.4 * min(h, w)
    x0, x1, y0, y1 = _spot_window(bg.shape, cx, cy, r_max)
    crop = bg[y0:y1, x0:x1]
    Y, X = np.ogrid[0:crop.shape[0], 0:crop.shape[1]]
    dx, dy = X - (cx - x0), Y - (cy - y0)
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx) + np.pi  # 0..2pi
    nr = max(2, int(r_max / r_step))
    ri = np.minimum((r * (1.0 / r_step)).astype(np.int64), nr - 1)
    ti = np.minimum((th * (n_theta / (2 * np.pi))).astype(np.int64), n_theta - 1)
    flat = (ri * n_theta + ti).ravel()
    wp = crop.ravel()
    sums = np.bincount(flat, weights=wp, minlength=nr * n_theta).reshape(nr, n_theta)
    cnts = np.bincount(flat, minlength=nr * n_theta).reshape(nr, n_theta)
    mean_bin = sums / np.maximum(cnts, 1)  # Mean intensity per (r, theta)
    ring_mean = mean_bin.mean(axis=1)
    ring_std = mean_bin.std(axis=1)
    cv = ring_std / np.maximum(ring_mean, 1e-9)  # Per-ring azimuthal spread.
    rr = (np.arange(nr) + 0.5) * r_step
    wr = ring_mean * rr  # ~ energy in each annulus.
    return float((cv * wr).sum() / max(wr.sum(), 1e-12))


def azimuthal_orders(bg, cx, cy, r0, m_max=4, r_lo=1.0, r_hi=12.0, n_theta=24):
    """Return azimuthal orders.

    Return normalized azimuthal Fourier amplitudes for the selected annulus.

    Args:
        bg: Background reference used for correction.
        cx: Horizontal centre coordinate, in pixels.
        cy: Vertical centre coordinate, in pixels.
        r0: Diffraction-limited reference radius, in pixels.
        m_max: Highest azimuthal order to evaluate.
        r_lo: Inner radial bound, in pixels.
        r_hi: Outer radial bound, in pixels.
        n_theta: Number of n theta.
    """
    # NaN, not 0.0, when there is nothing to anchor the window on: 0 would read
    # as "measured, perfectly circular". The gate maps NaN to no penalty either
    # way, but the log then says "not measured" instead of "clean".
    zero = {m: float("nan") for m in range(1, m_max + 1)}
    if not (r0 > 0):
        return zero
    r_lo_px, r_hi_px = max(r_lo * r0, 2.0), max(r_hi * r0, 8.0)
    x0, x1, y0, y1 = _spot_window(bg.shape, cx, cy, r_hi_px)
    crop = bg[y0:y1, x0:x1]
    if crop.size == 0:
        return zero
    Y, X = np.ogrid[0:crop.shape[0], 0:crop.shape[1]]
    dx, dy = X - (cx - x0), Y - (cy - y0)
    r, th = np.hypot(dx, dy), np.arctan2(dy, dx) + np.pi
    step = max(1.0, 0.25 * r0)
    nr = max(2, int((r_hi_px - r_lo_px) / step))
    keep = np.broadcast_to((r >= r_lo_px) & (r < r_hi_px), crop.shape)
    ri = np.clip(((r - r_lo_px) / step).astype(np.int64), 0, nr - 1)
    ti = np.minimum((th * (n_theta / (2 * np.pi))).astype(np.int64), n_theta - 1)
    flat = (np.broadcast_to(ri, crop.shape)[keep] * n_theta
            + np.broadcast_to(ti, crop.shape)[keep])
    n = nr * n_theta
    sums = np.bincount(flat, weights=crop[keep], minlength=n).reshape(nr, n_theta)
    cnts = np.bincount(flat, minlength=n).reshape(nr, n_theta)
    # The MEAN per (r, theta) cell, never the sum: square pixels put unequal
    # pixel COUNTS into equal angular bins, and a sum reads that sampling
    # pattern as real structure -- enough to give a perfect Airy a spurious a1.
    occupied = cnts > 0
    mean_bin = np.divide(sums, cnts, out=np.zeros_like(sums), where=occupied)
    ring_n = occupied.sum(axis=1)
    ring_mean = np.divide(mean_bin.sum(axis=1), np.maximum(ring_n, 1),
                          out=np.zeros(nr), where=ring_n > 0)
    # An empty cell takes its ring's mean, so a sampling gap adds no modulation.
    mean_bin = np.where(occupied, mean_bin, ring_mean[:, None])
    rr = r_lo_px + (np.arange(nr) + 0.5) * step
    wr = np.maximum(ring_mean, 0.0) * rr  # ~ energy in each annulus.
    denom = float((wr * ring_mean).sum())
    if denom <= 0:
        return zero
    ang = 2.0 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    out = {}
    for m in range(1, m_max + 1):
        cm = np.abs((mean_bin * np.exp(-1j * m * ang)).sum(axis=1)) / n_theta
        out[m] = float((wr * cm).sum() / denom)
    return out


def _cut_sidelobe_asym(p, c):
    """Return cut sidelobe asym.

    One 1-D cut through the peak: (worst sidelobe / main peak, mirror asym).

    A sidelobe is a RE-RISE outside the main lobe: walk out from the peak until
    the cut first falls below 0.25 of it, then look for a bump in the tail. A
    clean Gaussian decays monotonically, so both sides give 0; a double spot is
    mirror-SYMMETRIC yet re-rises, so the sidelobe term still punishes it --
    symmetry alone can never catch that failure.

    Args:
        p: Probability, parameter vector, or polynomial value.
        c: Class index or coefficient value.
    """
    n = len(p)
    if n < 7 or not (0 <= c < n):
        return 0.0, 0.0
    lo, hi = max(0, c - 2), min(n, c + 3)  # Re-anchor on the local max.
    c = lo + int(np.argmax(p[lo:hi]))
    peak = float(p[c])
    if peak <= 0:
        return 0.0, 0.0
    side = 0.0
    for arm in (p[c::-1], p[c:]):  # Walk left, then right.
        below = np.nonzero(arm < 0.25 * peak)[0]
        if not len(below) or len(arm) - below[0] < 3:
            continue  # Lobe fills the window.
        tail = arm[below[0]:]
        pk = int(np.argmax(tail))  # A rise-then-fall = side peak.
        if pk > 0 and tail[pk] > 1.15 * tail[0]:
            side = max(side, float(tail[pk]) / peak)
    left, right = p[c - 1::-1], p[c + 1:]
    m = min(len(left), len(right))
    if m >= 3:
        l, r = left[:m], right[:m]
        asym = float(np.abs(l - r).sum() / max(float((l + r).sum()), 1e-9))
    else:
        asym = 0.0
    return min(side, 1.0), min(asym, 1.0)


def cross_profile_metrics(bg, cx, cy, r80, r0=0.0):
    """Measure sidelobes and asymmetry from peak-centred cross-sections.

    Args:
        bg: Background reference used for correction.
        cx: Horizontal centre coordinate, in pixels.
        cy: Vertical centre coordinate, in pixels.
        r80: Radius enclosing 80 percent of the energy, in pixels.
        r0: Diffraction-limited reference radius, in pixels.
    """
    h, w = bg.shape
    icx, icy = int(round(cx)), int(round(cy))
    if not (0 <= icx < w and 0 <= icy < h):
        return 0.0, 0.0
    half = int(max(20.0 * r0, 6.0 * r80 if np.isfinite(r80) else 0.0, 20.0))
    x0, x1 = max(0, icx - half), min(w, icx + half + 1)
    y0, y1 = max(0, icy - half), min(h, icy + half + 1)
    row = bg[max(0, icy - 1):icy + 2, x0:x1].mean(axis=0)
    col = bg[y0:y1, max(0, icx - 1):icx + 2].mean(axis=1)
    kern = np.full(3, 1.0 / 3.0, np.float32)
    row = np.convolve(row, kern, mode="same")
    col = np.convolve(col, kern, mode="same")
    s_r, a_r = _cut_sidelobe_asym(row, icx - x0)
    s_c, a_c = _cut_sidelobe_asym(col, icy - y0)
    return max(s_r, s_c), 0.5 * (a_r + a_c)


def shape_quality(r: SpotReading, cfg: S.LoopSettings, floor=None) -> float:
    """Return a bounded quality gate that rejects pathological spot shapes.

    Args:
        r: Radial coordinate or radius.
        cfg: Configuration for the operation.
        floor: Lower floor applied to the calculated value.
    """
    def finite_clip(value, hi=None, floor_v=0.0):
        value = float(value)
        if not np.isfinite(value):
            return 0.0
        value = max(value - float(floor_v), 0.0)  # Per-term deadband
        return float(np.clip(value, 0.0, hi)) if hi is not None else value

    f = floor or {}

    def term(attr, hi=None):
        return finite_clip(getattr(r, attr), hi, f.get(attr, 0.0))

    scale = max(float(cfg.symmetry_weight), 0.0) / 3.0
    penalty = scale * (
        # One term per symmetry, each verified to respond to ITS OWN mode well
        # above its floor: a_1 coma 0.39 vs 0.04, a_3 trefoil 0.22 vs 0.07,
        # a_4 astigmatism 0.21 vs 0.02.
        2.00 * term("azim_m1", 1.0)  # Coma / decentre
        + 2.00 * term("azim_m2", 1.0)  # Astigmatism off best focus.
        + 2.00 * term("azim_m3", 1.0)  # Trefoil
        + 2.00 * term("azim_m4", 1.0)  # Focal astigmatism diamond.
        # Best signal-to-floor of any term measured (23x over 0..3 rms waves):
        # blind at small aberration, but a strong clean signal at large.
        + 1.00 * term("ellipticity", 1.0)
        + 0.75 * term("ring_contrast")  # The only ring/defocus detector.
        # Fragmentation is counted in 2-D, by connected components, not inferred
        # from two 1-D cuts (see the dropped cross_sidelobe below).
        + 0.70 * max(int(r.n_spots) - 1, 0)
    )
    # Keep unstable cross-section descriptors for logging, not optimization.
    return float(np.exp(-min(penalty, 60.0)))


def _aperture(gray, cx, cy, r_out, backend=FM.BACKEND_NUMPY, ref=None):
    """Measure peak-centred photometry inside the outer radius.

    Args:
        gray: Grayscale image data.
        cx: Horizontal centre coordinate, in pixels.
        cy: Vertical centre coordinate, in pixels.
        r_out: Outer photometry-aperture radius, in pixels.
        backend: Numerical backend to use.
        ref: Reference data for relative measurements.
    """
    base = _baseline(gray, backend, ref)[0]
    x0, x1, y0, y1 = _spot_window(gray.shape, cx, cy, r_out)
    crop = gray[y0:y1, x0:x1] - base  # Signed: noise cancels.
    ccx, ccy = cx - x0, cy - y0
    Y, X = np.ogrid[0:crop.shape[0], 0:crop.shape[1]]
    rr = np.hypot(X - ccx, Y - ccy)
    inside = rr <= r_out
    return crop, rr, inside, float(crop[inside].sum()), ccx, ccy


def energy_aperture_px(cfg: S.LoopSettings, shape) -> float:
    """Radius (px) of the fixed aperture an occlusion test should measure in.

    Far wider than the 20 r0 scoring aperture, because energy leaving this
    one is read as a lost beam, so it must hold the worst shape the mirror
    can make. Still bounded, because the baseline error scales with pixels.

    Args:
        cfg: Loop settings supplying the optics.
        shape: Frame shape, (rows, cols).
    """
    r0 = EE.r0_px(cfg.wavelength_nm, cfg.focal_mm, cfg.aperture_mm, cfg.pixel_um)
    side = float(min(shape[0], shape[1]))
    r = 60.0 * r0 if r0 > 0 else 0.25 * side  # Unknown optics: frame fraction.
    return float(min(r, 0.45 * side))


def beam_energy(frame, cx, cy, r_out) -> float:
    """Return baseline-subtracted energy inside the selected radius.

    Args:
        frame: Captured image frame.
        cx: Horizontal centre coordinate, in pixels.
        cy: Vertical centre coordinate, in pixels.
        r_out: Outer photometry-aperture radius, in pixels.
    """
    gray = np.asarray(frame, dtype=np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2, dtype=np.float32)
    _, _, _, energy, _, _ = _aperture(gray, float(cx), float(cy), float(r_out))
    return float(energy)


def _corner_sigma(gray, backend=FM.BACKEND_NUMPY, ref=None):
    """Per-pixel background noise sigma.

    Args:
        gray: Grayscale image data.
        backend: Numerical backend to use.
        ref: Reference data for relative measurements.
    """
    return _baseline(gray, backend, ref)[1]


def _despike(a, factor=3.0, floor_frac=0.05):
    """Replace isolated spikes by their 3x3 median.

    A hot pixel has dark neighbours; a resampled spot core does not, so it
    passes through untouched. Needed because sharpness is quadratic in
    intensity. Both tests are required: the ratio test alone marks every
    positive noise pixel as a spike and turns this into a denoising filter.

    Args:
        a: Baseline-subtracted crop.
        factor: Ratio to the 3x3 median above which a pixel is a spike.
        floor_frac: Fraction of the peak below which no pixel is a spike.
    """
    a = np.asarray(a, np.float32)
    med = cv2.medianBlur(a, 3)
    hot = (a > factor * med) & (a > floor_frac * float(a.max()))
    return np.where(hot, med, a)


def _sharpness(crop, inside, sigma=0.0):
    """Return Muller-Buffington sharpness from baseline-corrected photometry.

    Args:
        crop: Crop applied to the input image.
        inside: Mask selecting samples inside the aperture.
        sigma: Estimated standard deviation or spot width.
    """
    vals = _despike(crop)[inside].astype(np.float64)
    s1 = float(vals.sum())
    if s1 <= 0 or vals.size == 0:
        return 0.0
    raw = float((vals ** 2).sum())
    # Cap noise subtraction at 80% of the raw sum. Without the cap, dim spots
    # collapse when sum(I^2) approaches the noise floor; the biased capped
    # reading remains monotonic enough for the optimizer to climb.
    return float(max(raw - min(vals.size * float(sigma) ** 2, 0.8 * raw), 0.0)
                 / s1 ** 2)


def _second_moment(crop, inside, ccx, ccy):
    """Centroid-referenced second moment over the photometry aperture, in px^2.

    `sum(I r^2) / sum(I)` about the intensity centroid, which is what makes it
    blind to tilt. Linear in intensity, so a signed baseline leaves it unbiased
    and no noise correction is needed; the r^2 weight does amplify any residual
    background, so a positive baseline inflates it without bound.

    The aperture must stay FIXED for the quadratic form to hold -- the caller's
    whole-pattern footprint, never a spot-derived radius that shrinks as the
    correction improves.

    Args:
        crop: Baseline-subtracted photometry crop.
        inside: Mask selecting samples inside the aperture.
        ccx: Horizontal aperture centre within the crop, in pixels.
        ccy: Vertical aperture centre within the crop, in pixels.

    Returns:
        The second moment in px^2 on the crop's own grid, or NaN when the
        aperture holds no positive energy.
    """
    vals = np.asarray(crop, np.float64)[inside]
    total = float(vals.sum())
    if not (total > 0) or vals.size == 0:
        return float("nan")
    Y, X = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    x = X[inside].astype(np.float64) - float(ccx)
    y = Y[inside].astype(np.float64) - float(ccy)
    # Centroid first, in the same aperture, so the moment is about the spot's
    # own centre of mass and tilt drops out.
    cx = float((vals * x).sum()) / total
    cy = float((vals * y).sum()) / total
    m2 = float((vals * ((x - cx) ** 2 + (y - cy) ** 2)).sum()) / total
    return m2 if np.isfinite(m2) and m2 >= 0 else float("nan")


def _psd_band(crop, rr, r_out, r0, m_lo, m_hi, sigma=0.0):
    """Integrate normalized image power over a spatial-frequency annulus.

    Args:
        crop: Crop applied to the input image.
        rr: Radial coordinate grid.
        r_out: Outer photometry-aperture radius, in pixels.
        r0: Diffraction-limited reference radius, in pixels.
        m_lo: Lower normalised spatial-frequency bound.
        m_hi: Upper normalised spatial-frequency bound.
        sigma: Estimated standard deviation or spot width.
    """
    if not (r0 > 0 and m_hi > m_lo >= 0):
        return float("nan")
    a = np.asarray(crop, np.float32)
    rd = np.asarray(rr, np.float32)
    h, wid = a.shape
    # Decimate frequencies above the measured band to reduce FFT cost.
    # The ideal reference uses the same factor to preserve score normalization.
    s = int(min(max(1.0, 0.2 * float(r0) / (float(m_hi) * _R0_CUTOFF)),
                min(h, wid) / 64.0))
    if s > 1:
        a = cv2.resize(a, (wid // s, h // s), interpolation=cv2.INTER_AREA)
        rd = cv2.resize(rd, (wid // s, h // s), interpolation=cv2.INTER_AREA)
        h, wid = a.shape
        r0 = float(r0) / s
        sigma = float(sigma) / s  # Averaging s^2 px divides the noise by s.
    # Radial raised cosine over the photometry aperture, built on the
    # decimated grid (the ratio rd/r_out is scale-free).
    w = (0.5 * (1.0 + np.cos(np.pi * np.clip(rd / max(float(r_out), 1e-9),
                                             0.0, 1.0)))).astype(np.float32)
    a = a * w
    dc = float(a.sum())
    if not (dc > 0):
        return float("nan")
    spec = np.abs(np.fft.rfft2(a)) ** 2
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(wid)[None, :]
    m = np.hypot(fy, fx) * (float(r0) / _R0_CUTOFF)  # -> normalised frequency
    band = (m >= float(m_lo)) & (m < float(m_hi))
    if not band.any():
        # The window cannot resolve the requested band at all. NaN, never 0:
        # 0 would read as "measured, no contrast there" and the optimiser would
        # happily climb the noise instead.
        return float("nan")
    # Rfft2 drops the conjugate half of the plane, so every column except DC
    # (and Nyquist, when the width is even) stands for two bins of the area sum.
    mult = np.full(spec.shape[1], 2.0)
    mult[0] = 1.0
    if wid % 2 == 0:
        mult[-1] = 1.0
    mult = np.broadcast_to(mult, spec.shape)
    total = float((spec * mult)[band].sum())
    n_bins = float(mult[band].sum())
    noise = float(sigma) ** 2 * float((w ** 2).sum()) * n_bins
    return float(max(total - min(noise, 0.8 * total), 0.0)
                 / (float(h) * float(wid) * dc ** 2))


def _diffraction_energy_fractions(crop, rr, inside, total, r0):
    """Photometric core, near-sidelobe and far-halo energy fractions.

    All are normalised inside 20 r0, matching the full EE diagnostic window:
    core = 0..1 r0, sidelobe = 1..5 r0, far halo = 10..20 r0. Keeping the
    annuli separate prevents an improvement far away from compensating a bad
    bright ring immediately beside the core.

    Args:
        crop: Crop applied to the input image.
        rr: Radial coordinate grid.
        inside: Mask selecting samples inside the aperture.
        total: Total item or frame count used for progress reporting.
        r0: Diffraction-limited reference radius, in pixels.
    """
    if r0 <= 0 or total <= 0:
        return 0.0, 0.0, 0.0
    vals, radii = crop[inside], rr[inside]
    core = float(vals[radii <= r0].sum()) / total
    sidelobe = float(vals[(radii > r0) & (radii <= 5.0 * r0)].sum()) / total
    far_halo = float(vals[radii > 10.0 * r0].sum()) / total
    return (float(np.clip(core, 0.0, 1.0)),
            float(np.clip(sidelobe, 0.0, 1.0)),
            float(np.clip(far_halo, 0.0, 1.0)))


def _count_spots(blur, frac=0.35, min_area=6):
    """How many separate bright blobs sit above frac*peak.

    Args:
        blur: Blur width applied to the simulated or measured image.
        frac: Requested fractional level.
        min_area: Minimum permitted area.
    """
    peak = float(blur.max())
    if peak <= 0:
        return 0
    mask = (blur >= frac * peak).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return int(sum(stats[i, cv2.CC_STAT_AREA] >= min_area for i in range(1, n)))


def _pib(crop, rr, total, ccx, ccy, radius, upsample):
    """Return peak-intensity-bucket energy relative to aperture energy.

    Args:
        crop: Crop applied to the input image.
        rr: Radial coordinate grid.
        total: Total item or frame count used for progress reporting.
        ccx: Horizontal candidate centre, in pixels.
        ccy: Vertical candidate centre, in pixels.
        radius: Aperture or spot radius.
        upsample: Interpolation factor for power-in-bucket integration.
    """
    if total <= 0 or radius <= 0:
        return 0.0
    f = max(1, int(upsample))
    if f <= 1:
        return float(np.clip(crop[rr <= radius].sum() / total, 0.0, 1.0))
    # Cubic resize preserves the local mean, so the zoomed sum / f^2 is the
    # same energy on the same signed-baseline scale as `total`
    sub = cv2.resize(crop.astype(np.float32), None, fx=f, fy=f,
                     interpolation=cv2.INTER_CUBIC)
    yy, xx = np.ogrid[:sub.shape[0], :sub.shape[1]]
    cx_up, cy_up = ccx * f + (f - 1) / 2.0, ccy * f + (f - 1) / 2.0
    in_bucket = (xx - cx_up) ** 2 + (yy - cy_up) ** 2 <= (radius * f) ** 2
    return float(np.clip(float(sub[in_bucket].sum()) / (f * f) / total,
                         0.0, 1.0))


def _ring_strength(bg, cx, cy, r80):
    """Height of a secondary maximum past the core, as a fraction of the peak.

    0 when the profile is monotonic. Only the first 4*r80 radii are binned.

    Args:
        bg: Baseline-subtracted image.
        cx: Horizontal centre coordinate, in pixels.
        cy: Vertical centre coordinate, in pixels.
        r80: Radius enclosing 80 percent of the energy, in pixels.
    """
    r_max = max(6.0, 4.0 * r80 if np.isfinite(r80) else 6.0)
    x0, x1, y0, y1 = _spot_window(bg.shape, cx, cy, r_max)
    crop = bg[y0:y1, x0:x1]
    Y, X = np.ogrid[0:crop.shape[0], 0:crop.shape[1]]
    ri = np.hypot(X - (cx - x0), Y - (cy - y0)).astype(int).ravel()
    prof = np.bincount(ri, weights=crop.ravel()) / np.maximum(
        np.bincount(ri), 1)
    prof = prof[:max(6, int(4 * r80))]
    core = float(prof[0]) if len(prof) else 0.0
    if core <= 0 or len(prof) < 6:
        return 0.0
    below = np.where(prof < 0.35 * core)[0]  # First fall past the core.
    if not len(below) or len(prof) - below[0] < 3:
        return 0.0
    tail = prof[below[0]:]
    peak = int(np.argmax(tail))  # A bump (rise then fall) = ring.
    if peak > 0 and tail[peak] > 1.2 * tail[0]:
        return float(tail[peak] / core)
    return 0.0


def classify_aberration(bg, feats, pib, am=None):
    """Classify aberration patterns from intensity symmetry.

    Args:
        bg: Background reference used for correction.
        feats: Extracted image or wavefront features.
        pib: Power-in-bucket metric.
        am: Azimuthal-modulation features.
    """
    if not np.isfinite(feats.r_ee80):
        return "unknown"
    am = am or {}
    present = []
    if am.get(4, 0.0) > 0.12 or feats.ellipticity > 0.30:
        present.append(f"astig {feats.theta_deg:+.0f}deg")
    if am.get(1, 0.0) > 0.12:
        present.append("coma")
    if am.get(3, 0.0) > 0.12:
        present.append("trefoil")
    if _ring_strength(bg, feats.cx, feats.cy, feats.r_ee80) > 0.15:
        present.append("spherical")
    if len(present) >= 2:
        return "mixed: " + " + ".join(p.split()[0] for p in present)
    if present:
        return present[0]
    # Bad+diffuse => a blend.
    return "round" if pib >= 0.55 else "mixed/unclear"


def measure(frame, cfg: S.LoopSettings, bit_depth=None, max_size=600,
            bg_ref=None, exposure_ms=None, n_avg=1,
            second_moment_roi: SecondMomentROI | None = None) -> SpotReading:
    """Measure all configured metrics for one frame.

    Args:
        frame: Input image frame.
        cfg: Configuration for the operation.
        bit_depth: Camera or image bit depth.
        max_size: Maximum length of the resized image side, in pixels.
        bg_ref: Run-wide `BackgroundRef`; it is scaled, never re-estimated,
            if exposure changes.
        exposure_ms: Exposure, in milliseconds.
        n_avg: Number of frames averaged into `frame`.
        second_moment_roi: Optional run-wide full-frame aperture used only for
            the second moment.  All other metrics retain the adaptive aperture.

    Returns:
        A spot reading containing the selected score and all diagnostics.
    """
    backend = FM.resolve(getattr(cfg, "reduction_backend", FM.BACKEND_NUMPY))
    gray_full = np.asarray(frame, dtype=np.float32)  # Float32: half the memory.
    if gray_full.ndim == 3:  # Traffic of beam.as_gray.
        gray_full = gray_full.mean(axis=2, dtype=np.float32)
    # Remove the fixed pattern first, at full resolution, so what follows works
    # against a flat zero baseline. `applied` decides whether the fixed part
    # still has to be counted in the noise (see BackgroundRef.noise_at).
    t_bg = time.perf_counter()
    applied = (bg_ref is not None and bg_ref.pattern is not None
               and bg_ref.pattern.shape == gray_full.shape)
    if applied:
        k = bg_ref.scale_for(exposure_ms)
        gray_full = gray_full - (bg_ref.pattern if k == 1.0
                                 else bg_ref.pattern * k)
    bg_ms = (time.perf_counter() - t_bg) * 1e3
    h, w = gray_full.shape
    long_side = max(h, w)
    r0_full = EE.r0_px(cfg.wavelength_nm, cfg.focal_mm,
                       cfg.aperture_mm, cfg.pixel_um)
    scale = long_side / float(max_size) if long_side > max_size else 1.0
    if r0_full > 0:
        scale = min(scale, r0_full / _MIN_R0_GRID_PX)
    if scale > 1.0:
        small_w = max(1, int(round(w / scale)))
        small_h = max(1, int(round(h / scale)))
        gray = cv2.resize(gray_full, (small_w, small_h),
                          interpolation=cv2.INTER_AREA)
        px_scale = 0.5 * (h / small_h + w / small_w)
    else:
        gray = gray_full
        px_scale = 1.0
    # (level, sigma, level_err) the whole measurement works against
    t_bg = time.perf_counter()
    base_ref = (bg_ref.noise_at(px_scale, applied, exposure_ms, n_avg)
                if bg_ref is not None else None)
    bg = (np.clip(gray - (base_ref[0] + 2.0 * base_ref[1]), 0.0, None)
          if base_ref is not None else beam.subtract_background(gray))
    bg_ms += (time.perf_counter() - t_bg) * 1e3
    total = float(bg.sum())
    maxv = (1 << int(bit_depth)) - 1 if bit_depth else float(gray.max() or 1)
    peak_frac = float(gray.max()) / maxv if maxv else 0.0

    cx, cy, blur = _peak_center(bg)
    r0 = r0_full / px_scale if r0_full > 0 else 0.0

    # Beam presence is checked against a run-level reference in the UI.
    # Per-frame SNR remains diagnostic because poor shapes can have low peaks.
    fp0 = _pattern_footprint(bg, r0, blur)
    e_cx, e_cy = (fp0[0], fp0[1]) if fp0 else (cx, cy)
    e_r = max(2.0 * fp0[2], 20.0 * r0, 30.0) if fp0 else max(20.0 * r0, 30.0)
    e_snr = _energy_snr(gray, e_cx, e_cy, min(e_r, 0.5 * max(gray.shape)),
                        backend, base_ref)
    snr = _peak_snr(gray, backend)
    feats = extract_features(gray)  # Descriptors on small frame.
    if not feats.ok or total <= 0:
        return SpotReading(-np.inf, 0.0, 0.0, float("nan"), float("nan"),
                           float("nan"), float("nan"), 0, total, peak_frac,
                           0.0, 0.0, has_signal=False, peak_snr=snr,
                           energy_snr=e_snr, bg_ms=bg_ms)
    # Ring stats only need the spot neighbourhood, not the whole frame.
    r_spot = (max(6.0 * feats.r_ee80, 30.0) if np.isfinite(feats.r_ee80)
              else 0.4 * min(gray.shape))
    ring_asym = azimuthal_asymmetry(  # 0 = azimuthally symmetric.
        bg, cx, cy, r_max=min(0.4 * min(gray.shape), r_spot))
    # Scope photometry to the whole pattern: centre on the energy centroid
    # and cover the 1/e^2 footprint and the 20-r0 diffraction window.
    fp = fp0  # Already computed for the beam test.
    if fp is not None:
        pcx, pcy, r_cover = fp
        r_out = max(2.0 * r_cover, 20.0 * r0, 30.0)
    else:  # No footprint: legacy behaviour.
        pcx, pcy = cx, cy
        r_cover = float("nan")
        r_out = 20.0 * r0 if r0 > 0 else min(0.4 * min(gray.shape), r_spot)
    r_out = min(r_out, 0.5 * max(gray.shape))  # Keep it on the sensor.
    crop, rr, inside, ap_energy, _pcx_c, _pcy_c = _aperture(gray, pcx, pcy,
                                                            r_out, backend,
                                                            base_ref)
    # The bucket goes where the focus should be, searched only near the pattern
    # centre (3 r0) so it can never jump to a distant speckle
    bcx, bcy = _bucket_center(blur, pcx, pcy, 3.0 * r0 if r0 > 0 else 6.0)
    x0_ap, _, y0_ap, _ = _spot_window(gray.shape, pcx, pcy, r_out)
    ccx, ccy = bcx - x0_ap, bcy - y0_ap
    # Energy-fraction radii about the bucket centre; the aperture stays on
    # the pattern centre so the denominator covers everything scattered.
    Yb, Xb = np.ogrid[0:crop.shape[0], 0:crop.shape[1]]
    rr_b = np.hypot(Xb - ccx, Yb - ccy)
    core_ee, sidelobe, far_halo = _diffraction_energy_fractions(
        crop, rr_b, inside, ap_energy, r0)
    cross_side, cross_asym = cross_profile_metrics(bg, cx, cy, feats.r_ee80, r0)
    # Auto sub-pixel upsampling when the (small-frame) spot is only a few px.
    up = cfg.upsample_factor if (np.isfinite(feats.r_ee80)
                                 and feats.r_ee80 < cfg.upsample_small_px / px_scale) else 1
    # Keep the bucket fixed at the diffraction limit. A spot-derived radius
    # grows with blur and can reward degradation; r0 is instead a property of
    # the optics. Convert the full-frame radius to the resampled grid.
    if cfg.bucket_radius_px > 0:
        radius = cfg.bucket_radius_px / px_scale
    elif r0_full > 0:
        radius = r0_full / px_scale
    else:  # Unknown optics: legacy fallback.
        radius = max(1.5 * feats.r_ee80, 4.0)
    pib = _pib(crop, rr_b, ap_energy, ccx, ccy, radius, up)
    # Centre-free concentration on the SAME signed-baseline aperture, so the
    # whole-pattern scoping that protects pib protects this too.
    px_sigma = _corner_sigma(gray, backend, base_ref)
    sharp = _sharpness(crop, inside, px_sigma)
    # Same aperture and same signed baseline, but read one frequency band at a
    # time instead of summed over all of them (see _psd_band)
    psd = _psd_band(crop, rr, r_out, r0, cfg.psd_m_lo, cfg.psd_m_hi, px_sigma)
    # The second moment gets a separate fixed aperture when supplied: a moving
    # integration domain breaks the exact quadratic form ModalFast relies on.
    moment_radius_grid = r_out
    moment_edge_frac = float("nan")
    moment_roi_clipped = False
    if second_moment_roi is None:
        moment_crop, moment_rr, moment_inside = crop, rr, inside
        moment_ccx, moment_ccy = _pcx_c, _pcy_c
    else:
        moment_cx = float(second_moment_roi.cx) / px_scale
        moment_cy = float(second_moment_roi.cy) / px_scale
        moment_radius_grid = float(second_moment_roi.radius) / px_scale
        moment_crop, moment_rr, moment_inside, _, moment_ccx, moment_ccy = \
            _aperture(gray, moment_cx, moment_cy, moment_radius_grid, backend,
                      base_ref)
        moment_roi_clipped = bool(
            moment_cx - moment_radius_grid < 0
            or moment_cy - moment_radius_grid < 0
            or moment_cx + moment_radius_grid > gray.shape[1] - 1
            or moment_cy + moment_radius_grid > gray.shape[0] - 1)
    moment_grid = _second_moment(moment_crop, moment_inside,
                                 moment_ccx, moment_ccy)
    # Diagnostic only.  The containment guard uses the detected whole-pattern
    # aperture because positive-clipped camera noise can dominate an annulus;
    # this number remains useful in logs for spotting a marginal background.
    moment_positive = np.clip(moment_crop, 0.0, None)
    moment_energy = float(moment_positive[moment_inside].sum())
    if moment_energy > 0:
        edge = moment_inside & (moment_rr >= 0.9 * moment_radius_grid)
        moment_edge_frac = float(moment_positive[edge].sum() / moment_energy)
    n_spots = _count_spots(blur)
    # Evaluate azimuthal symmetry about the D4sigma energy centroid, never the
    # photometric peak. A displaced origin creates artificial one-fold
    # modulation and can mislabel pure astigmatism as coma.
    am = azimuthal_orders(bg, feats.cx, feats.cy, r0)

    label = classify_aberration(bg, feats, pib, am)
    r = SpotReading(0.0, pib, feats.peak_norm,
                    feats.r_ee80 * px_scale, feats.d4sigma * px_scale,
                    feats.ellipticity, feats.asymmetry, n_spots, total,
                    peak_frac, cx * px_scale, cy * px_scale,
                    has_signal=True,
                    peak_snr=snr, energy_snr=e_snr,
                    halo_frac=feats.halo_frac,
                    ring_contrast=feats.ring_contrast,
                    aberration=label,
                    ring_asym=ring_asym,
                    azim_m1=am[1], azim_m2=am[2],
                    azim_m3=am[3], azim_m4=am[4],
                    far_halo_frac=far_halo,
                    core_ee_frac=core_ee,
                    sidelobe_frac=sidelobe,
                    cross_sidelobe=cross_side,
                    cross_asym=cross_asym,
                    sharpness=sharp,
                    psd_band=psd,
                    second_moment=moment_grid * px_scale ** 2,
                    photometry_aperture_cx=pcx * px_scale,
                    photometry_aperture_cy=pcy * px_scale,
                    photometry_aperture_radius=r_out * px_scale,
                    second_moment_roi_radius=moment_radius_grid * px_scale,
                    second_moment_edge_frac=moment_edge_frac,
                    second_moment_roi_clipped=moment_roi_clipped,
                    bg_ms=bg_ms,
                    peak_whole=(float(blur.max()) / ap_energy
                                if ap_energy > 0 else 0.0))
    # An Airy second moment depends on the integration radius, so "ideal /
    # measured" would change scale with the ROI margin. Use the reciprocal raw
    # moment; `norm_score` then reports M_start/(M + M_start).
    if np.isfinite(r.second_moment) and r.second_moment > 0:
        r.second_moment_score = 1.0 / r.second_moment
    # Absolute quality: this reading vs a perfect spot at the same sampling.
    ref = (_ideal_ref(round(r0_full / px_scale, 3),
                      round(_blur_sigma(gray.shape), 3), round(radius, 3),
                      round(float(cfg.symmetry_weight), 3),
                      round(float(cfg.psd_m_lo), 4),
                      round(float(cfg.psd_m_hi), 4))
           if r0_full > 0 else None)
    if ref is not None:
        r.strehl = _absolute(r.peak_whole, ref["peak_whole"])
        r.ee_strehl = _absolute(pib, ref["pib"])
        r.conc_score = _absolute(feats.r_ee80, ref["r_ee80"], False)
        r.size_score = _absolute(feats.d4sigma, ref["d4sigma"], False)
        # No ratio here: the gate's per-term deadband already puts a perfect
        # spot at 1.0, and dividing two exp(-sum)s built on different grids
        # left a diffraction-limited spot reading below an aberrated one.
        r.sym_score = shape_quality(r, cfg, ref["shape_floor"])
        r.sharp_score = _absolute(sharp, ref["sharpness"])
        r.psd_score = _absolute(psd, ref["psd_band"])
    r.score = primary_score(r, cfg)
    return r


def make_norm_ref(r: SpotReading, cfg: S.LoopSettings) -> dict:
    """Seed reference for `norm_score`: the absolute quality of one reading.

    Whichever reading you pass becomes the 0.5 mark. The loop passes the
    first valid frame of the run; the bench passes one image for the whole
    session, so every run's initial score says how it really started.

    Args:
        r: Radial coordinate or radius.
        cfg: Configuration for the operation.
    """
    return dict(q=primary_score(r, cfg), metric=cfg.metric,
                # Diagnostics only; scoring uses `q`
                peak_whole=r.peak_whole, pib=r.pib,
                r=r.r_ee80, d=r.d4sigma)


def norm_score(r: SpotReading, ref: dict, cfg: S.LoopSettings) -> float:
    """Map a metric to a reference-centred score in the range 0 to 1.

    `v / (v + seed)` puts the reference at 0.5 and is monotonic in `v`. The
    seed differs from run to run, so runs are compared on `primary_score`,
    which is referenced to the diffraction limit.

    Args:
        r: The reading to score.
        ref: The seed reading the run started from.
        cfg: Loop settings naming the metric.
    """
    v = primary_score(r, cfg)
    ref = ref or {}
    seed = ref.get("q") if ref.get("metric", cfg.metric) == cfg.metric else None
    if not (np.isfinite(v) and v >= 0 and seed is not None
            and np.isfinite(seed) and v + seed > 0):
        return 0.0
    return float(np.clip(v / (v + seed), 0.0, 1.0))


def contrast_score(x: float, gamma: float = 3.0) -> float:
    """Apply a symmetric display contrast to a reference-centred score.

    Args:
        x: Input coordinate or scalar value.
        gamma: Contrast or nonlinear scaling exponent.
    """
    g = max(float(gamma), 1e-6)
    if not np.isfinite(x):
        return float("nan")
    x = float(np.clip(x, 0.0, 1.0))
    a = x ** g
    b = (1.0 - x) ** g
    return float(a / (a + b)) if a + b > 0 else 0.5


def _legacy_raw(r: SpotReading, cfg: S.LoopSettings) -> float:
    """Objective when the optics are unknown (r0 <= 0).

    Raw and not comparable across runs. Always positive and higher-is-better
    (radius metrics become 1/(1+r)), so the seed map in `norm_score` is
    defined.

    Args:
        r: The reading to score.
        cfg: Loop settings naming the metric.
    """
    def smaller_is_better(v):
        return float(1.0 / (1.0 + v)) if np.isfinite(v) and v >= 0 else 0.0

    if cfg.metric == S.METRIC_PEAK:
        return float(r.peak_whole)
    if cfg.metric == S.METRIC_SHARP:
        # Already a "higher is better" concentration; raw, so not comparable
        # across runs, but the ranking within a run is unaffected.
        return float(r.sharpness)
    if cfg.metric == S.METRIC_PSD:
        # Without r0 there is no cutoff, so psd_band is NaN; sharpness is the
        # same integral over every frequency (Parseval).
        return float(r.sharpness)
    if cfg.metric == S.METRIC_SECOND_MOMENT:
        # The second moment needs no optical constants, and its displayed
        # ruler is deliberately run-relative even when those constants exist.
        return float(r.second_moment_score)
    if cfg.metric == S.METRIC_R_EE80:
        return smaller_is_better(r.r_ee80)
    if cfg.metric == S.METRIC_RMS:
        return smaller_is_better(r.d4sigma)
    return float(r.pib)


def _geomean(parts) -> float:
    """Geometric mean of 0..1 quality factors.

    Used to gate an objective with independent guards. A PRODUCT of the same
    factors collapses to ~0 for any imperfect spot and leaves the optimiser a
    flat, unclimbable landscape; the geometric mean keeps every factor's
    gradient alive while still letting any one bad axis pull the score down.

    The cap at 1.0 stays here even though `_absolute` no longer caps: this is
    a GATE, so one axis reading past the modelled ideal must not be allowed to
    buy back another axis that is genuinely bad.
    """
    parts = np.clip(parts, 1e-6, 1.0)
    return float(np.exp(np.mean(np.log(parts))))


def _astig_gate(base, r: SpotReading, cfg: S.LoopSettings) -> float:
    """Astigmatism gate for the angle-blind metrics; disabled and unused.

    Multiplies the score by `(1 - azim_m2) ** w`, where `azim_m2` is the
    two-fold azimuthal modulation that `psd_band` and `sharpness` cannot see.
    Never applied to `psd_band` itself, and skipped for the modal algorithms,
    so the quadratic form stays intact. Re-enable by restoring the two
    `base = _astig_gate(...)` lines in `primary_score`.

    Args:
        base: The ungated score.
        r: The reading it came from.
        cfg: Loop settings; `astig_weight` is the exponent.

    Returns:
        `base * (1 - azim_m2) ** w`, or `base` when the gate is off, the
        modulation is unmeasurable, or a modal solve is running.
    """
    w = float(getattr(cfg, "astig_weight", 0.0))
    if w <= 0 or cfg.algorithm in S.MODAL_ALGOS or not np.isfinite(r.azim_m2):
        return base
    return base * float(np.clip(1.0 - float(r.azim_m2), 0.0, 1.0)) ** w


def primary_score(r: SpotReading, cfg: S.LoopSettings) -> float:
    """Return the objective value maximized by the optimizer.

    Args:
        r: Radial coordinate or radius.
        cfg: Configuration for the operation.
    """
    if cfg.metric == S.METRIC_PEAK:
        base = r.strehl  # Peak-based Strehl proxy.
    elif cfg.metric == S.METRIC_SHARP:
        # Astigmatism gate disabled; see `_astig_gate`.
        base = r.sharp_score
    elif cfg.metric == S.METRIC_PSD:
        # Astigmatism gate disabled; see `_astig_gate`.
        base = r.psd_score
    elif cfg.metric == S.METRIC_SECOND_MOMENT:
        # Bare: a multiplicative gate would destroy the exact quadratic form
        # this metric exists for, and the solvers read r.second_moment anyway.
        base = r.second_moment_score
    elif cfg.metric == S.METRIC_R_EE80:
        # Gate the size metric with energy concentration and shape quality.
        base = _geomean([r.conc_score, r.ee_strehl, r.sym_score])
    elif cfg.metric == S.METRIC_RMS:
        # D4sigma size, gated the same way as R_EE80: a pure size metric alone
        # rewards a small-but-diffuse/broken spot, so pair it with the energy
        # concentration and the shape gate.
        base = _geomean([r.size_score, r.ee_strehl, r.sym_score])
    else:
        base = r.ee_strehl  # Power-in-bucket at r0 = EE-Strehl.
    if not np.isfinite(base):  # Unknown optics -> no anchor.
        return _legacy_raw(r, cfg)
    if cfg.roundness_weight > 0 and np.isfinite(r.ellipticity):
        base *= max(0.0, 1.0 - r.ellipticity) ** cfg.roundness_weight
    return float(base)
