# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.1, 2026-08-03

"""Score a spot series over candidate PSD bands and recommend one.

The modal solve assumes the reciprocal band metric `G = 1/g` is quadratic in
the aberration (Debarre and Booth, Opt. Express 15, 8176, 2007), which holds
over a range that depends on the band. This module fits `G` against the
drive for every candidate band and reports each band's half width, fit
quality and signal-to-scatter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from ..correction import metrics as M
from ..correction import settings as S

# Default ladder of candidate bands, normalised to the diffraction cutoff.
# Spread from narrow-and-low (largest capture range) to wide-and-high (most
# signal), so one run brackets the trade-off instead of testing one guess.
DEFAULT_BANDS = ((0.02, 0.05),
                 (0.05, 0.10),
                 (0.05, 0.20),
                 (0.05, 0.30),
                 (0.05, 0.60),
                 (0.10, 1.00))

MIN_R2 = 0.80  # Below this the quadratic model does not describe the band.
MIN_SNR = 3.0  # Metric swing must clear the point-to-point scatter this far.
MIN_BINS = 3  # Below this a "band" is a couple of FFT cells, not an annulus.
BIAS_FRACTION = 0.7  # Recommended bias as a fraction of the half width.


@dataclass
class BandResult:
    """One candidate band fitted against the drive axis."""
    m_lo: float
    m_hi: float
    g: np.ndarray  # Raw band value per point, NaN where unmeasurable.
    fit: np.ndarray  # Fitted G = 1/g per point, NaN where not fitted.
    a: float = float("nan")  # G at the vertex (1 / peak g).
    b: float = float("nan")  # Curvature of G in drive units^-2.
    x0: float = float("nan")  # Drive value that minimises G, i.e. best spot.
    half_width: float = float("nan")  # sqrt(a / b), in drive units.
    r2: float = float("nan")  # Quadratic fit quality on G.
    snr: float = float("nan")  # Metric swing over residual scatter.
    bins: int = 0  # Approximate radial FFT bins inside the band.
    note: str = ""

    @property
    def label(self) -> str:
        return f"{self.m_lo:g}-{self.m_hi:g}"

    @property
    def usable(self) -> bool:
        """Whether this band can carry a three-point modal solve.

        The bin count is a gate, not a footnote: a band spanning a couple of
        FFT cells reports those cells' noise, and its near-flat G fits an
        enormous half width that nothing in the data supports.
        """
        return (np.isfinite(self.half_width) and self.b > 0
                and self.r2 >= MIN_R2 and self.snr >= MIN_SNR
                and self.bins >= MIN_BINS)


@dataclass
class ScanResult:
    """Every band fitted for one series, plus the resulting advice."""
    bands: list = field(default_factory=list)
    drive: np.ndarray = field(default_factory=lambda: np.zeros(0))
    drive_label: str = ""
    r0_px: float = 0.0
    best: BandResult | None = None
    bias: float = float("nan")  # Recommended bias amplitude, drive units.
    messages: list = field(default_factory=list)

    @property
    def drive_span(self) -> float:
        return (float(self.drive.max() - self.drive.min())
                if self.drive.size else 0.0)


def load_gray(path):
    """Read a saved spot crop at its native depth as a float array.

    The snapshots are 16-bit PNGs of raw camera counts, so IMREAD_UNCHANGED is
    required: the default 8-bit conversion would quantise away most of the
    dynamic range the band metric integrates over.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"cannot read {path}")
    if img.ndim == 3:
        img = img.mean(axis=2)
    return np.asarray(img, np.float64)


def band_bins(size, r0_px, m_lo, m_hi):
    """Approximate radial FFT bins a square crop resolves inside a band.

    A band thinner than one bin cannot be measured at all, and two or three bins
    make a noisy reading, so this is the sanity check on a very low band. It
    mirrors the decimation rule in `dm_closed_loop.core.metrics._psd_band`,
    which remains the source of truth for the value itself.

    Args:
        size: Side of the analysed crop, in pixels.
        r0_px: Diffraction-limited radius on the sensor, in pixels.
        m_lo: Lower normalised spatial-frequency bound.
        m_hi: Upper normalised spatial-frequency bound.
    """
    if not (r0_px > 0 and m_hi > m_lo >= 0 and size > 0):
        return 0
    s = int(min(max(1.0, 0.2 * r0_px / (m_hi * M._R0_CUTOFF)), size / 64.0))
    n = max(1, int(size) // max(1, s))
    # Bin spacing in normalised frequency on the decimated grid.
    step = (r0_px / s) / M._R0_CUTOFF / n
    return int(max(0.0, (m_hi - m_lo) / step)) if step > 0 else 0


def _quadratic_fit(x, y):
    """Least-squares `y = a + b*(x - x0)^2`, returned as (a, b, x0, r2, model).

    Fitted as a plain polynomial and re-centred afterwards, so it needs no
    starting guess and cannot fail to converge -- which matters because the
    vertex often lies outside the scanned range (the mirror's own aberration
    offsets it), and an iterative fit seeded at the middle would wander.
    Returns b <= 0 unchanged: the caller treats that as "not a bowl", which is
    the honest answer for a band where the model has broken down.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return (float("nan"),) * 4 + (np.full(len(x), np.nan),)
    xs, ys = x[ok], y[ok]
    # Centre and scale x to keep the Vandermonde conditioned on bit values.
    mid, span = float(xs.mean()), float(xs.max() - xs.min()) or 1.0
    u = (xs - mid) / span
    c = np.polyfit(u, ys, 2)
    b = float(c[0]) / span ** 2
    model = np.full(len(x), np.nan)
    model[ok] = np.polyval(c, u)
    resid = float(np.sum((ys - model[ok]) ** 2))
    total = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - resid / total if total > 0 else float("nan")
    if b <= 0:
        return float("nan"), b, float("nan"), r2, model
    x0 = mid - float(c[1]) / (2.0 * float(c[0])) * span
    a = float(np.polyval(c, (x0 - mid) / span))
    return a, b, x0, r2, model


def _fit_band(m_lo, m_hi, g, drive, size, r0_px):
    """Fit one band's reciprocal metric against the drive axis."""
    res = BandResult(m_lo=float(m_lo), m_hi=float(m_hi), g=g,
                     fit=np.full(len(g), np.nan),
                     bins=band_bins(size, r0_px, m_lo, m_hi))
    good = np.isfinite(g) & (g > 0)
    if good.sum() < 4:
        res.note = "not enough valid points (band unresolved at this crop size)"
        return res
    big = np.full(len(g), np.nan)
    big[good] = 1.0 / g[good]  # G = 1/g: quadratic for ANY aberration size.
    a, b, x0, r2, model = _quadratic_fit(np.asarray(drive, float), big)
    res.fit, res.a, res.b, res.x0, res.r2 = model, a, b, x0, r2
    if not (b > 0):
        res.note = "no bowl: 1/g does not curve upward, model broken here"
        return res
    res.half_width = float(np.sqrt(a / b)) if a > 0 else float("nan")
    swing = float(np.nanmax(g[good]) - np.nanmin(g[good]))
    scatter = float(np.sqrt(np.nanmean((big[good] - model[good]) ** 2)))
    # Compare like with like: convert the residual scatter on G back to g at the
    # vertex, where g is largest, so the ratio reads as a fraction of the swing.
    peak = float(np.nanmax(g[good]))
    scatter_g = scatter * peak ** 2 if np.isfinite(scatter) else float("nan")
    res.snr = swing / scatter_g if scatter_g > 0 else float("inf")
    if res.bins < MIN_BINS:
        res.note = (f"only ~{res.bins} radial bins at this crop size: too few "
                    "to read as a band")
    return res


def scan(series, cfg: S.LoopSettings, bands=DEFAULT_BANDS, progress=None):
    """Fit every candidate band against one drive series and recommend one.

    Each image is scored through the same `metrics.measure` the live loop
    uses. The optics in `cfg` must be right: without them there is no
    diffraction cutoff and no normalised frequency.

    Args:
        series: `index.Series` of spots recorded along one drive line.
        cfg: Loop settings supplying the optics; its own band is ignored.
        bands: Candidate `(m_lo, m_hi)` pairs to fit.
        progress: Optional `callable(done, total)` progress hook.

    Returns:
        A `ScanResult`; `best` and `bias` are None/NaN when no band survives.
    """
    out = ScanResult(drive=np.asarray(series.drive, float),
                     drive_label=series.drive_label)
    if not len(series):
        out.messages.append("no images to analyse")
        return out

    frames = [load_gray(p.path) for p in series.points]
    size = min(min(f.shape) for f in frames)
    if len({f.shape for f in frames}) > 1:
        out.messages.append(
            "images are NOT all the same size: a band's frequency bins depend "
            "on the array size, so these values are not comparable. Re-record "
            "with a fixed snapshot crop.")

    out.r0_px = float(EE_r0(cfg))
    if out.r0_px <= 0:
        out.messages.append(
            "optics unknown (wavelength / focal length / aperture / pixel), so "
            "there is no diffraction cutoff to normalise the band against -- "
            "every band reads NaN. Fill the optics in the loop settings.")

    total = len(bands) * len(frames)
    done = 0
    for m_lo, m_hi in bands:
        sub = replace(cfg, psd_m_lo=float(m_lo), psd_m_hi=float(m_hi))
        g = np.empty(len(frames))
        for i, frame in enumerate(frames):
            g[i] = M.measure(frame, sub).psd_band
            done += 1
            if progress is not None:
                progress(done, total)
        out.bands.append(_fit_band(m_lo, m_hi, g, out.drive, size, out.r0_px))

    _recommend(out)
    return out


def EE_r0(cfg: S.LoopSettings) -> float:
    """Diffraction-limited radius on the sensor for these optics, in pixels."""
    from ..correction import ee_curve as EE
    return EE.r0_px(cfg.wavelength_nm, cfg.focal_mm, cfg.aperture_mm,
                    cfg.pixel_um)


def _recommend(out: ScanResult):
    """Pick a band and a bias amplitude, and say why in `out.messages`.

    Preference order follows Debarre section 7.3: a band whose capture range
    covers the aberration actually present is worth more than a band with a
    stronger response, because outside the capture range the three-point solve
    returns a number of the wrong size and sometimes the wrong sign. Among bands
    that reach that far, the strongest response wins.

    "That far" is measured from the CENTRE of the scan, not across it. Both
    `half_width` and the distance to be covered are radii from the vertex, and
    comparing the radius with the full span demanded twice the capture range
    anything needed -- which rejected every band that honestly turned over
    inside the scan and left only the ones too flat to have a vertex at all.
    """
    usable = [b for b in out.bands if b.usable]
    if not usable:
        out.messages.append(
            "no band survived: none of them produced an upward-curving 1/g "
            "with an honest fit over enough radial bins. Either the drive did "
            "not change the aberration enough, or the spot is far outside "
            "every band's capture range -- flatten the mirror first (run one "
            "correction), then re-record.")
        return
    # A vertex outside the scanned range means the series never saw g turn
    # over, so sqrt(a/b) rests on a curvature the data barely constrains: that
    # half width is extrapolated, not measured.
    lo, hi = float(out.drive.min()), float(out.drive.max())
    bracketed = [b for b in usable if lo <= b.x0 <= hi]
    if not bracketed:
        out.messages.append(
            "every band puts its best spot OUTSIDE the range you scanned, so "
            "every half width below is extrapolated. Re-record centred on the "
            "best spot before trusting any of them.")
        bracketed = usable
    reach = 0.5 * out.drive_span  # Centre-to-end: the units half_width is in.
    covering = [b for b in bracketed if b.half_width >= reach]
    pool = covering or bracketed
    out.best = max(pool, key=lambda b: b.snr)
    out.bias = BIAS_FRACTION * out.best.half_width
    if not covering:
        widest = max(bracketed, key=lambda b: b.half_width)
        out.messages.append(
            f"no band's capture range ({widest.half_width:.0f} bit at best) "
            f"reaches the {reach:.0f} bit from the centre of the scan to its "
            "ends, so the ends of the scan are outside the model. Trust the "
            "recommendation for corrections SMALLER than the half width, or "
            "scan a narrower range.")
    if out.best.bins < 6:
        out.messages.append(
            f"the chosen band resolves only ~{out.best.bins} radial bins at "
            "this crop size; a larger snapshot crop would measure it better.")
