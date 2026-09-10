# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-08-21

"""Modal phase retrieval from a single focal-plane spot image.

A scalar Fraunhofer model of the spot is fitted to the measured frame; the
free parameters are a short vector of Noll-indexed Zernike coefficients plus
sub-pixel centre, pupil scale, flux and background. One in-focus frame
leaves one overall sign of the even modes unobservable (fixed by convention,
see `even_sign_flipped`), the astigmatism axis known modulo 90 degrees,
defocus and spherical strongly correlated, and piston and tilt absorbed.
Coefficients are RMS-normalised waves in the Noll ordering of `zernike`;
angles are in image coordinates (x right, y down).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from prysm.propagation import Wavefront
from scipy.optimize import least_squares

from . import efield as EF
from .. import zernike as ZK

# Noll indices fitted by default: defocus through spherical.
DEFAULT_MODES = (4, 5, 6, 7, 8, 9, 10, 11)


@dataclass(frozen=True)
class Optics:
    """The four constants that fix the pupil-to-sensor scale.

    Attributes:
        wavelength_nm: Source wavelength, in nanometres.
        focal_mm: Focal length forming the spot, in millimetres.
        aperture_mm: Nominal pupil diameter, in millimetres.
        pixel_um: Sensor pixel pitch, in micrometres.
    """

    wavelength_nm: float
    focal_mm: float
    aperture_mm: float
    pixel_um: float

    @property
    def r0_px(self):
        """float: Airy radius 1.22*lambda*f/D, in sensor pixels."""
        return (1.22 * self.wavelength_nm * 1e-3 * self.focal_mm
                / (self.aperture_mm * self.pixel_um))

    def valid(self):
        """Whether all four constants are positive and finite."""
        vals = (self.wavelength_nm, self.focal_mm, self.aperture_mm,
                self.pixel_um)
        return all(np.isfinite(v) and v > 0 for v in vals)


@dataclass(frozen=True)
class RetrievalOptions:
    """Knobs of the fit, with defaults chosen for a 5-10 px Airy core.

    Attributes:
        modes: Noll indices to fit.
        roi_px: Side of the square crop, or 0 to size it from the spot so
            the crop holds `roi_energy_frac` of the spot's own energy.
        roi_r80_factor: Crop side as a multiple of the measured 80%
            encircled-energy radius, used when `roi_px` is 0.
        roi_max_px: Ceiling on the derived side, to bound the cost.
        pupil_samples: Samples across the pupil diameter in the model.
        bound_waves: Per-coefficient bound, in RMS waves.
        fit_pupil_scale: Fit a scale on the nominal aperture. Off by
            default: it is degenerate with defocus and spherical.
        efield_rounds: Alternations between the electric-field search and
            the intensity fit. Off by default; see `efield`.
        gs_iterations: Alternating-projection rounds inside the
            Gerchberg-Saxton field estimate.
        multistart: Try sign-flipped restarts when the first fit fits badly.
        residual_ok: Noise-weighted residual below which a fit is trusted
            and no restart is attempted. A simulation criterion; real frames
            sit far above it, so `patience` is what stops the search there.
        patience: Stop escalating after this many consecutive restarts that
            fail to improve the best residual by `improve_frac`. Off by
            default, because an early stop can settle in a worse basin.
        improve_frac: Fractional residual improvement that counts as
            progress.
        coarse_nfev: Evaluation cap for the three-mode coarse stage.
        fit_background: Treat the background pedestal as a free parameter.
            Off by default: the frame outside the spot measures it directly,
            and a free pedestal trades against the aberration amplitude.
        pixel_selection: Fit only the brighter pixels, noise-weighted. Off
            by default: on real frames the answer then depends on the
            selection settings.
        snr_floor: With `pixel_selection`, keep pixels above this many
            background sigma.
        min_pixels_per_unknown: Floor on kept pixels per fitted parameter.
        coarse_bin: Upper bound on the coarse stage's pixel binning, capped
            so the binned sampling stays at or finer than Nyquist. The
            coarse stage keeps the full field of view and bins instead of
            cropping.
        coarse_pupil_samples: Pupil grid for the coarse stage.
        tol: Shared ftol/xtol/gtol for the solver, tighter than the scipy
            default because most of the ROI is background.
        max_nfev: Cap on model evaluations per solver call.
        basis: A `dm_basis.DMBasis` to expand the phase in instead of
            Zernikes, or None. With a basis, `modes` is ignored and the
            reported Zernike coefficients are the projection onto Noll
            1..15.
    """

    modes: tuple = DEFAULT_MODES
    roi_px: int = 0
    roi_r80_factor: float = 4.0
    roi_max_px: int = 384
    pupil_samples: int = 96
    bound_waves: float = 1.5
    fit_pupil_scale: bool = False
    efield_rounds: int = 0
    gs_iterations: int = 25
    multistart: bool = True
    residual_ok: float = 1.5
    patience: int = 99
    improve_frac: float = 0.02
    coarse_nfev: int = 150
    fit_background: bool = False
    pixel_selection: bool = False
    snr_floor: float = 3.0
    min_pixels_per_unknown: int = 40
    coarse_bin: int = 3
    coarse_pupil_samples: int = 48
    tol: float = 1e-13
    max_nfev: int = 400
    # Excluded from eq/hash: it holds arrays, and the dataclass is frozen.
    basis: object = field(default=None, compare=False)


@dataclass
class WavefrontEstimate:
    """One fit's answer, plus everything needed to judge it.

    Attributes:
        coeffs: Noll index -> RMS waves. Even-mode signs are arbitrary.
        named: Orientation-invariant magnitudes and angles, from
            `zernike.named_aberrations`.
        rms_waves: sqrt(sum of squared coefficients); sign-invariant.
        rms_rad: The same figure in radians.
        n_pixels: Pixels that survived the SNR cut and were fitted.
        photon_gain: Variance per unit signal, measured from this frame.
        sigma_waves: Per-mode one-sigma uncertainty from the inverse Fisher
            matrix of the weighted fit. Optimistic under uniform weighting,
            and it excludes model error entirely.
        sigma_rms_waves: The same uncertainty propagated onto `rms_waves`.
        efield_used: True when the electric-field search moved the answer.
        strehl: |<exp(i*2*pi*phi)>|^2 over the pupil, valid at any level.
        strehl_marechal: exp(-(2*pi*rms)^2); meaningless above ~0.15 waves.
        residual: RMS of the noise-weighted model-minus-data over the fitted
            pixels; 1.0 is the photon-noise floor.
        converged: True when the solver stopped on a real criterion and the
            residual is below `RetrievalOptions.residual_ok`.
        reason: Why a fit was refused or judged poor; empty when fine.
        beyond_capture_range: True when the pre-fit estimate from the spot's
            second moment exceeds `CAPTURE_RANGE_WAVES`; the coefficients
            are then a lower bound.
        prior_waves: That pre-fit estimate, in RMS waves.
        sign_resolved: False; one overall even-mode sign is unobservable.
        even_sign_flipped: True when the convention (largest even coefficient
            positive) negated the whole even block.
        pupil_scale: Fitted pupil diameter over the configured aperture.
        centre_px: Fitted spot centre in full-frame pixels.
        shift_um: Fitted spot offset from the ROI centre, in sensor
            micrometres. Stored so `render_model` can reproduce the fitted
            image exactly rather than re-deriving it.
        flux: Fitted total spot energy, in the frame's own units.
        background: Fitted background pedestal, in the frame's own units.
        roi_origin: (x0, y0) of the fitted crop within the full frame.
        roi_px: Side of the fitted crop.
        n_evals: Model evaluations spent.
        ms: Wall-clock cost of the fit.
        pupil_phase: Reconstructed pupil phase in waves, NaN outside the
            pupil, or None when not requested.
    """

    coeffs: dict = field(default_factory=dict)
    named: dict = field(default_factory=dict)
    rms_waves: float = float("nan")
    rms_rad: float = float("nan")
    n_pixels: int = 0
    photon_gain: float = float("nan")
    sigma_waves: dict = field(default_factory=dict)
    sigma_rms_waves: float = float("nan")
    efield_used: bool = False
    strehl: float = float("nan")
    strehl_marechal: float = float("nan")
    residual: float = float("nan")
    converged: bool = False
    reason: str = ""
    beyond_capture_range: bool = False
    prior_waves: float = float("nan")
    sign_resolved: bool = False
    even_sign_flipped: bool = False
    pupil_scale: float = float("nan")
    centre_px: tuple = (float("nan"), float("nan"))
    shift_um: tuple = (float("nan"), float("nan"))
    flux: float = float("nan")
    background: float = float("nan")
    roi_origin: tuple = (0, 0)
    roi_px: int = 0
    n_evals: int = 0
    ms: float = 0.0
    pupil_phase: np.ndarray | None = None
    fit_coeffs: tuple = ()
    modal_amplitudes: dict = field(default_factory=dict)
    modal_sigma: dict = field(default_factory=dict)
    basis_name: str = "zernike"

    def coeff_vector(self, n_modes=11):
        """Coefficients as a dense Noll 1..n_modes array, for `zernike.fit`
        style helpers that index from Noll 1.

        Args:
            n_modes: Highest Noll index in the returned vector.
        """
        out = np.zeros(int(n_modes))
        for j, a in self.coeffs.items():
            if 1 <= j <= n_modes:
                out[j - 1] = a
        return out


class _Forward:
    """Cached Fraunhofer forward model: coefficients in, spot image out.

    The Zernike basis and pupil mask are evaluated once; each call costs one
    matrix-triple-product DFT onto the sensor grid. The DFT form is used
    rather than a padded FFT because it renders straight onto the camera's
    pixel pitch over the ROI only, so cost follows the ROI and not the
    padding an FFT would need to reach the same sampling.
    """

    def __init__(self, optics: Optics, opts: RetrievalOptions):
        self.optics = optics
        self.opts = opts
        self.modes = tuple(int(j) for j in opts.modes)
        n = int(opts.pupil_samples)
        # Unit-disk grid with y increasing DOWN the rows, so the pupil and
        # the camera frame share one coordinate system and a reported angle
        # matches what the operator sees on screen.
        ax = (np.arange(n) - (n - 1) / 2.0) / (n / 2.0)
        x, y = np.meshgrid(ax, ax)
        rho, theta = np.hypot(x, y), np.arctan2(y, x)
        self.mask = rho <= 1.0
        self.dm = getattr(opts, "basis", None)
        if self.dm is None:
            modes_flat = ZK.basis(rho.ravel(), theta.ravel(), max(self.modes))
            self.basis = np.stack(
                [modes_flat[j - 1].reshape(n, n) for j in self.modes])
            self.basis *= self.mask  # Zernikes are defined on the disk only.
        else:
            # The mirror's own eigenmodes: `modes` becomes a plain 1..K label
            # set, and reporting goes through `DMBasis.zernike_matrix`.
            self.dm = self.dm.at_samples(n)
            self.basis = self.dm.maps * self.mask
            self.modes = tuple(range(1, len(self.dm) + 1))
        self.amp = self.mask.astype(float)
        self.wav_um = optics.wavelength_nm / 1000.0
        # Waves of pupil tilt per micrometre of spot displacement. A tilt of
        # (D / 2f / lambda) * x_norm moves the spot one lambda*f/D, which is
        # how the sub-pixel centre is applied -- see `psf`.
        self.tilt_waves_per_um = (optics.aperture_mm * 1e3
                                  / (2.0 * optics.focal_mm
                                     * optics.wavelength_nm))
        self.tilt_x, self.tilt_y = x * self.mask, y * self.mask

    def phase_waves(self, coeffs, shift_um=(0.0, 0.0)):
        """Pupil phase map in waves for one coefficient vector.

        Args:
            coeffs: One amplitude per fitted mode, in RMS waves.
            shift_um: Spot displacement to encode as pupil tilt, in sensor
                micrometres. Omit for the aberration alone.
        """
        phase = np.tensordot(np.asarray(coeffs, float), self.basis, axes=1)
        sx, sy = float(shift_um[0]), float(shift_um[1])
        if sx or sy:
            gain = self.tilt_waves_per_um
            phase = phase + gain * (sx * self.tilt_x + sy * self.tilt_y)
        return phase

    def strehl(self, coeffs):
        """Exact peak-ratio Strehl of one coefficient vector.

        |<exp(i*2*pi*phi)>|^2 over the pupil. Unlike the Maréchal form this
        stays meaningful at the aberration levels an uncorrected deformable
        mirror actually shows.

        Args:
            coeffs: One amplitude per fitted mode, in RMS waves.
        """
        phase = self.phase_waves(coeffs)[self.mask]
        return float(np.abs(np.exp(1j * 2.0 * np.pi * phase).mean()) ** 2)

    def psf(self, coeffs, shift_um, pupil_scale, samples):
        """Unit-energy model spot on the sensor grid.

        Args:
            coeffs: One amplitude per fitted mode, in RMS waves.
            shift_um: Spot centre offset (x, y) from the ROI centre, in
                micrometres on the sensor.
            pupil_scale: Multiplier on the configured aperture diameter.
            samples: Side of the square output grid, in pixels.

        Returns:
            The modelled intensity, normalised to unit sum.
        """
        diameter_mm = self.optics.aperture_mm * float(pupil_scale)
        dx_mm = diameter_mm / self.basis.shape[-1]
        opd_nm = (self.phase_waves(coeffs, shift_um)
                  * self.optics.wavelength_nm)
        wf = Wavefront.from_amp_and_phase(self.amp, opd_nm, self.wav_um,
                                          dx_mm)
        # `shift` stays (0, 0): prysm caches its matrix-DFT on the shift and
        # never evicts. The displacement is exact as a tilt in the pupil.
        focused = wf.focus_fixed_sampling(
            efl=self.optics.focal_mm, dx=self.optics.pixel_um,
            samples=int(samples), shift=(0, 0))
        img = np.asarray(focused.intensity.data, float)
        total = img.sum()
        return img / total if total > 0 else img


def find_spot(frame, roi_px):
    """Locate the spot and return a square crop around it.

    The centre is the intensity-weighted centroid of everything above a
    conservative threshold, which follows a spread or broken spot instead of
    latching onto the brightest pixel.

    Args:
        frame: 2-D intensity image.
        roi_px: Side of the square crop, in pixels.

    Returns:
        A tuple of the crop, the crop origin (x0, y0), and the full-frame
        centre (cx, cy).
    """
    img = np.asarray(frame, float)
    if img.ndim == 3:
        img = img.mean(axis=2)
    base = float(np.median(np.concatenate(
        [img[:8, :].ravel(), img[-8:, :].ravel(),
         img[:, :8].ravel(), img[:, -8:].ravel()])))
    sig = np.clip(img - base, 0.0, None)
    thresh = 0.05 * float(sig.max()) if sig.max() > 0 else 0.0
    weight = np.where(sig >= thresh, sig, 0.0)
    total = weight.sum()
    h, w = img.shape
    if total <= 0:
        cx, cy = w / 2.0, h / 2.0
    else:
        ys, xs = np.mgrid[0:h, 0:w]
        cx = float((weight * xs).sum() / total)
        cy = float((weight * ys).sum() / total)
    half = int(roi_px) // 2
    x0 = int(np.clip(round(cx) - half, 0, max(w - 2 * half, 0)))
    y0 = int(np.clip(round(cy) - half, 0, max(h - 2 * half, 0)))
    crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
    return crop, (x0, y0), (cx, cy)


# Wavefront RMS in waves per sqrt of the excess second moment (in units of
# (lambda*f/D)^2), measured by simulation over 0.1 to 1.0 waves.
MAGNITUDE_PRIOR_K = 0.0906

# Wavefront RMS beyond which this estimator stops recovering the truth,
# measured on simulated spots and consistent with the literature's ~0.5
# waves. More restarts cannot buy past it.
CAPTURE_RANGE_WAVES = 0.45

# Defocus against spherical is the one degeneracy here that is not a sign;
# this is the magnitude error it leaves on simulated cases carrying the
# pair. A fitted Z4 or Z11 below this is not resolved.
RADIAL_DEGENERACY_WAVES = 0.093


def magnitude_prior(crop, optics):
    """Order-of-magnitude wavefront RMS, from the spot's own second moment.

    The centroid-referenced second moment of the far field is the
    diffraction term plus the mean-square wavefront gradient (Soloviev,
    arXiv:1707.08489, Eq. 7), so the excess tracks the aberration. Used to
    size the coarse stage's search seeds.

    Args:
        crop: The fitted region, background included.
        optics: Wavelength, focal length, aperture and pixel pitch.

    Returns:
        An estimate of the wavefront RMS in waves. Meaningless below about
        0.1 waves, where the excess falls into the diffraction term's own
        variation, but nothing needs a large seed there.
    """
    image = np.asarray(crop, float)
    signal = np.clip(image - float(np.median(image)), 0.0, None)
    total = float(signal.sum())
    lambda_f_d = optics.r0_px / 1.22
    if not total > 0 or not lambda_f_d > 0:
        return 0.0
    ys, xs = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    cx = float((signal * xs).sum() / total)
    cy = float((signal * ys).sum() / total)
    moment = float((signal * ((xs - cx) ** 2 + (ys - cy) ** 2)).sum() / total)
    # The diffraction-limited second moment in the same units, subtracted so
    # a near-perfect spot does not claim aberration.
    excess = max(moment / lambda_f_d ** 2 - 6.1, 0.0)
    return float(MAGNITUDE_PRIOR_K * np.sqrt(excess))


def spot_radius(frame, fraction=0.8, window_px=100.0):
    """Radius holding `fraction` of the energy inside a bounded window.

    The bound matters: a diffuse pedestal over the whole frame outweighs the
    spot, so the fraction is referenced to a window around it.

    Args:
        frame: The full frame.
        fraction: Enclosed-energy fraction; 0.8 is the usual beam-size
            convention and the one `metrics.r_ee80` already uses.
        window_px: Radius of the reference window, in pixels.

    Returns:
        The radius in pixels.
    """
    image = np.asarray(frame, float)
    if image.ndim == 3:
        image = image.mean(axis=2)
    signal = np.clip(image - float(np.median(image)), 0.0, None)
    total = float(signal.sum())
    if not total > 0:
        return float(window_px)
    ys, xs = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    cx = float((signal * xs).sum() / total)
    cy = float((signal * ys).sum() / total)
    radius = np.hypot(xs - cx, ys - cy).ravel()
    order = np.argsort(radius)
    cumulative = np.cumsum(signal.ravel()[order])
    inside = radius[order] < window_px
    if not inside.any():
        return float(window_px)
    reference = float(cumulative[inside][-1])
    index = int(np.searchsorted(cumulative, fraction * reference))
    return float(radius[order][min(index, radius.size - 1)])


def converged_side(frame, factor, min_px=64, max_px=384):
    """Crop side at which the retrieved coefficients stop moving.

    `factor` multiplies the measured 80% encircled-energy radius, so the crop
    tracks the spot. The factor was set by grid convergence: coefficients
    settle at four times r80 and do not move thereafter.

    Args:
        frame: The full frame.
        factor: Multiple of r80; 4.0 is the measured value.
        min_px: Floor, so a near-diffraction-limited spot keeps enough
            pixels to constrain eleven parameters.
        max_px: Ceiling, so one pathological frame cannot ask for the whole
            sensor.

    Returns:
        An even side length in pixels.
    """
    radius = spot_radius(frame)
    image = np.asarray(frame, float)
    limit = int(min(max_px, 2 * (min(image.shape[:2]) // 2)))
    side = int(2 * round(float(factor) * radius / 2.0))
    return int(min(max(side, int(min_px)), limit))


def _bin2d(image, factor):
    """Sum square blocks of pixels, as a detector with larger pixels would.

    Args:
        image: 2-D array.
        factor: Block side. 1 returns the input unchanged.

    Returns:
        The binned image, trimmed to a whole number of blocks.
    """
    factor = int(factor)
    if factor <= 1:
        return image
    h = (image.shape[0] // factor) * factor
    w = (image.shape[1] // factor) * factor
    trimmed = image[:h, :w]
    return trimmed.reshape(h // factor, factor,
                           w // factor, factor).sum(axis=(1, 3))


def _measure_background(frame, crop, origin):
    """Pedestal level, from the frame well outside the fitted crop.

    The spot's own wings reach far, so this samples the frame border rather
    than an annulus just outside the ROI. Median, so a stray reflection or a
    hot pixel cannot move it.

    Args:
        frame: The full frame.
        crop: The fitted crop, used only as a fallback.
        origin: The crop's (x0, y0), used only as a fallback.

    Returns:
        The background level in the frame's own units.
    """
    del origin
    image = np.asarray(frame, float)
    if image.ndim == 3:
        image = image.mean(axis=2)
    band = max(8, min(image.shape) // 20)
    if min(image.shape) <= 4 * band:  # A frame no bigger than the spot.
        return float(np.median(crop))
    border = np.concatenate([image[:band, :].ravel(),
                             image[-band:, :].ravel(),
                             image[:, :band].ravel(),
                             image[:, -band:].ravel()])
    return float(np.median(border))


def _photon_gain(crop, background, sigma):
    """Variance per unit signal, measured from the frame itself.

    Photon-transfer method: regress local variance on local mean, with the
    local statistics taken from adjacent-pixel differences. The gain is not
    1 in DN, because the frames are averages and carry a camera gain.

    Args:
        crop: The fitted region.
        background: Measured pedestal, subtracted before regressing.
        sigma: Background noise sigma, the intercept of the same relation.

    Returns:
        Variance per unit signal, clamped to be non-negative. Zero means the
        data show no signal-dependent noise, which leaves a pure read-noise
        model rather than an unusable one.
    """
    image = np.asarray(crop, float)
    signal = np.clip(image - background, 0.0, None)
    # Pair each pixel with its right-hand neighbour.
    left, right = signal[:, :-1].ravel(), signal[:, 1:].ravel()
    mean = 0.5 * (left + right)
    var = 0.5 * (left - right) ** 2  # Unbiased for the per-pixel variance.
    order = np.argsort(mean)
    mean, var = mean[order], var[order]
    n_bins = 24
    edges = np.linspace(0, mean.size, n_bins + 1).astype(int)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo < 32:
            continue
        xs.append(float(np.median(mean[lo:hi])))
        # Median of a chi-square with one degree of freedom is 0.4549 sigma^2.
        ys.append(float(np.median(var[lo:hi])) / 0.4549)
    if len(xs) < 4:
        return 0.0
    xs, ys = np.asarray(xs), np.asarray(ys) - sigma ** 2
    denom = float(xs @ xs)
    return max(float(xs @ ys) / denom, 0.0) if denom > 0 else 0.0


def _noise_sigma(frame, crop=None):
    """Per-pixel background noise, from the frame's outer border.

    Adjacent-pixel differences along the border, reduced by sqrt(2). Uses
    the standard deviation, not a median absolute deviation: averaged
    frames are heavily quantised and a median-based estimator returns 0.

    Args:
        frame: The full frame.
        crop: Fallback region when the frame is too small to have a border.

    Returns:
        The per-pixel sigma, floored above zero so weights stay finite.
    """
    image = np.asarray(frame, float)
    if image.ndim == 3:
        image = image.mean(axis=2)
    band = max(8, min(image.shape) // 20)
    if min(image.shape) > 4 * band:
        strips = [image[:band, :], image[-band:, :],
                  image[:, :band].T, image[:, -band:].T]
    elif crop is not None:
        strips = [np.asarray(crop, float)]
    else:
        strips = [image]
    diffs = np.concatenate([np.diff(s, axis=1).ravel() for s in strips
                            if s.shape[1] > 1])
    sigma = float(np.std(diffs) / np.sqrt(2.0)) if diffs.size else 0.0
    return max(sigma, 1e-6)




def _solve_flux_background(model, data, weight,
                           fixed_background=None):
    """Least-squares flux and background for one model spot.

    Both enter the model linearly, so they are eliminated by variable
    projection from the 2x2 normal equations. The solve must use the same
    pixels and weights as the residual it feeds.

    Args:
        model: Unit-energy model spot over the fitted pixels.
        data: Measured intensity over the same pixels.
        weight: Per-pixel weight over the same pixels, as used by the
            residual.
        fixed_background: Pin the pedestal to this measured level and solve
            only the flux, or None to solve both.

    Returns:
        The (flux, background) pair, with flux clipped to be non-negative.
    """
    w2 = weight * weight
    if fixed_background is not None:
        # One linear unknown: the pedestal is a measurement, not a parameter.
        offset = data - fixed_background
        s_mm = float(w2 @ (model * model))
        if s_mm <= 0:
            return 0.0, float(fixed_background)
        return (max(float(w2 @ (model * offset)) / s_mm, 0.0),
                float(fixed_background))
    s_ww = float(w2.sum())
    s_m = float(w2 @ model)
    s_mm = float(w2 @ (model * model))
    s_d = float(w2 @ data)
    s_md = float(w2 @ (model * data))
    det = s_mm * s_ww - s_m * s_m
    if abs(det) < 1e-30 or s_ww <= 0:
        return 0.0, s_d / s_ww if s_ww > 0 else 0.0
    flux = (s_ww * s_md - s_m * s_d) / det
    back = (s_mm * s_d - s_m * s_md) / det
    return (max(flux, 0.0), back) if flux > 0 else (0.0, s_d / s_ww)


def estimate_wavefront(frame, optics: Optics, options: RetrievalOptions = None,
                       guess=None, want_phase=True) -> WavefrontEstimate:
    """Fit Zernike coefficients to one measured focal-plane spot.

    Args:
        frame: 2-D intensity image holding a single spot. Background need not
            be removed; a pedestal is fitted.
        optics: Wavelength, focal length, aperture and pixel pitch.
        options: Fit configuration; defaults are used when omitted.
        guess: A previous estimate to warm start from, or None for a cold
            fit. Warm starting is what makes consecutive loop points cheap.
        want_phase: Attach the reconstructed pupil phase map to the result.

    Returns:
        The estimate. Check `converged` and `residual` before quoting any
        number from it, and read the module docstring before quoting a sign.
    """
    opts = options or RetrievalOptions()
    started = time.perf_counter()
    if not optics.valid():
        return WavefrontEstimate(reason="optics not configured")

    roi = (int(opts.roi_px) if opts.roi_px > 0
           else converged_side(frame, opts.roi_r80_factor,
                               max_px=opts.roi_max_px))
    crop, origin, centre = find_spot(frame, roi)
    if crop.size == 0 or crop.shape[0] != crop.shape[1]:
        return WavefrontEstimate(reason="frame smaller than the ROI")
    sigma = _noise_sigma(frame, crop)
    measured_background = _measure_background(frame, crop, origin)
    gain = 0.0  # See `_photon_gain`: not measurable from a structured frame.
    peak = float(crop.max() - np.median(crop))
    if not peak > 5.0 * sigma:
        return WavefrontEstimate(reason="no spot above the background noise")

    fwd = _Forward(optics, opts)
    # Unweighted, deliberately: Poisson weighting drowns the core in
    # background pixels and left coma and trefoil unrecovered.
    n_modes = len(fwd.modes)
    evals = [0]

    def unpack(p):
        coeffs = p[:n_modes]
        shift = (p[n_modes], p[n_modes + 1])
        scale = p[n_modes + 2] if opts.fit_pupil_scale else 1.0
        return coeffs, shift, scale

    def select_pixels(pixels, sigma_px):
        """Boolean mask of the pixels worth fitting.

        Keeps everything above `snr_floor` sigma, then relaxes the threshold
        if that left too few to keep the fit comfortably overdetermined.

        Args:
            pixels: The crop under consideration.
            sigma_px: Background noise sigma at this crop's binning.

        Returns:
            A flat boolean mask over the crop.
        """
        signal = pixels.ravel() - np.median(pixels)
        if not opts.pixel_selection:
            return np.ones(signal.size, bool)
        wanted = opts.min_pixels_per_unknown * (n_modes + 2)
        mask = signal > opts.snr_floor * sigma_px
        if mask.sum() < min(wanted, signal.size):
            # Fall back to the brightest `wanted` pixels rather than to the
            # whole crop: the point of the cut is to exclude pure background.
            order = np.argsort(signal)[::-1]
            mask = np.zeros(signal.size, bool)
            mask[order[:min(wanted, signal.size)]] = True
        # Flux and background are projected out over exactly these pixels,
        # so a fixed sample of sub-threshold pixels is kept to pin the
        # pedestal.
        dark = np.flatnonzero(~mask)
        if dark.size:
            take = min(dark.size, max(int(0.25 * mask.sum()), 200))
            mask[dark[np.linspace(0, dark.size - 1, take).astype(int)]] = True
        return mask

    def make_residual(forward, pixels, sigma_px, background_px, gain_px):
        """Residual closure over the fitted pixels of one crop.

        Weighted by the inverse of the expected per-pixel noise, which is
        only safe once the background pixels have been excluded.

        Args:
            forward: The model to evaluate.
            pixels: The crop.
            sigma_px: Background noise sigma at this crop's binning.

        Returns:
            A (residual function, kept-pixel mask) pair.
        """
        flat = pixels.ravel()
        side = pixels.shape[0]
        pinned = None if opts.fit_background else background_px
        mask = select_pixels(pixels, sigma_px)
        kept = flat[mask]
        # Per-pixel inverse noise with the variance measured, var = sigma^2 +
        # gain * signal, so `residual` is a reduced chi-square with floor 1.0.
        variance = sigma_px ** 2 + gain_px * np.clip(kept - background_px,
                                                     0.0, None)
        weight = 1.0 / np.sqrt(np.maximum(variance, 1e-12))
        scaled = kept * weight

        def residual(p):
            evals[0] += 1
            coeffs, shift, scale = unpack(p)
            model = forward.psf(coeffs, shift, scale, side).ravel()[mask]
            flux, back = _solve_flux_background(model, kept, weight,
                                                pinned)
            return (flux * model + back) * weight - scaled

        return residual, mask, kept, weight

    residual, keep, keep_data, keep_weight = make_residual(
        fwd, crop, sigma, measured_background, gain)
    # The coarse stage only has to tell defocus from astigmatism, so it runs
    # binned; binning is bounded by Nyquist for lambda*f/D, never chosen.
    airy_px = optics.r0_px / 1.22  # lambda*f/D in pixels.
    bin_n = max(1, min(int(opts.coarse_bin), int(airy_px / 2.0) or 1))
    coarse_crop = _bin2d(crop, bin_n)
    coarse_fwd = _Forward(
        replace(optics, pixel_um=optics.pixel_um * bin_n),
        replace(opts, pupil_samples=min(opts.coarse_pupil_samples,
                                        opts.pupil_samples)))
    # Binning sums bin^2 pixels: the pedestal scales with the block area, the
    # noise with the block side, and variance-per-signal is unchanged.
    coarse_residual, *_ = make_residual(
        coarse_fwd, coarse_crop, sigma * bin_n,
        measured_background * bin_n * bin_n, gain)
    coarse_peak = float(coarse_crop.max() - np.median(coarse_crop))

    px_um = optics.pixel_um
    lo = [-opts.bound_waves] * n_modes + [-6.0 * px_um, -6.0 * px_um]
    hi = [opts.bound_waves] * n_modes + [6.0 * px_um, 6.0 * px_um]
    # Explicit, not "jac". A coefficient is O(0.1 waves) and a shift is
    # O(20 micrometres), so an unscaled trust region takes a step that is
    # negligible for one and wild for the other, and the solver crawls.
    x_scale = np.array([0.1] * n_modes + [px_um, px_um])
    if opts.fit_pupil_scale:
        lo.append(0.5)
        hi.append(2.0)
        x_scale = np.append(x_scale, 0.1)

    def run(start, free=None, budget=None, coarse=False):
        """Solve from one seed, optionally holding all but `free` at zero.

        Args:
            start: Full parameter vector to start from.
            free: Indices of the coefficients allowed to move, or None for
                all of them. The nuisance terms always move.
            budget: Override for `max_nfev`.
            coarse: Evaluate on the small coarse-stage crop.
        """
        start = np.clip(start, np.asarray(lo) + 1e-9, np.asarray(hi) - 1e-9)
        if free is None:
            lo_r, hi_r, x0 = lo, hi, start
        else:
            # Freezing by bounds, not by a smaller parameter vector: the
            # residual closure then needs no second code path.
            frozen = [i for i in range(n_modes) if i not in free]
            lo_r, hi_r, x0 = list(lo), list(hi), start.copy()
            for i in frozen:
                lo_r[i], hi_r[i], x0[i] = -1e-9, 1e-9, 0.0
        return least_squares(coarse_residual if coarse else residual, x0,
                             bounds=(lo_r, hi_r), method="trf",
                             x_scale=x_scale,
                             max_nfev=budget or opts.max_nfev,
                             ftol=opts.tol, xtol=opts.tol, gtol=opts.tol)

    def norm_residual(res, coarse=False):
        """Weighted residual expressed in units of the background sigma.

        A value near 1 means the model is as close to the data as the noise
        allows; larger means real, unmodelled structure. This is a scale the
        SNR cut does not move, unlike a plain fraction of the peak.
        """
        del coarse
        return float(np.sqrt(np.mean(res.fun ** 2)))

    zero = np.zeros(len(lo))
    if opts.fit_pupil_scale:
        zero[-1] = 1.0

    # Stage 1, coarse: only defocus and the two astigmatism terms move, from
    # several seeds, which avoids the astigmatism local minimum. In an
    # eigenmode basis these are the three largest-singular-value modes.
    low = ([i for i, j in enumerate(fwd.modes) if j in (4, 5, 6)]
           if fwd.dm is None else list(range(min(3, n_modes))))
    # Seed amplitude from the spot itself. Positive only: 4, 5 and 6 are
    # even, so -a and +a render identical spots.
    prior = magnitude_prior(crop, optics)
    seed_amp = float(np.clip(prior, 0.1, opts.bound_waves))
    seeds = [zero]
    # Noll 4, 5 and 6 are all even, so -a and +a render pixel-identical
    # spots and a negative seed re-solves what the positive one covers. An
    # eigenmode has no such parity, so both signs are real starting points.
    signs = (1.0,) if fwd.dm is None else (1.0, -1.0)
    for i in low:
        for sign in signs:
            seed = zero.copy()
            seed[i] = sign * seed_amp
            seeds.append(seed)
    # A wavefront larger than the bound cannot be represented at all, so the
    # bound follows the prior rather than capping it.
    if seed_amp >= opts.bound_waves * 0.9:
        wide = float(seed_amp * 2.0)
        for i in range(n_modes):
            lo[i], hi[i] = -wide, wide

    coarse = None
    if guess is not None and getattr(guess, "coeffs", None):
        # A warm start is an answer, not a probe: try it at full resolution
        # first, and skip the seed sweep entirely when it still fits. This is
        # what makes consecutive points of a closed loop cheap.
        warm = zero.copy()
        for i, j in enumerate(fwd.modes):
            warm[i] = float(guess.coeffs.get(j, 0.0))
        warm_fit = run(warm)
        if norm_residual(warm_fit) <= opts.residual_ok:
            coarse, seeds = warm_fit, []
        else:
            seeds.insert(0, warm)

    candidates = []
    if seeds:
        for seed in seeds:
            out = run(seed, free=low, budget=opts.coarse_nfev, coarse=True)
            candidates.append((norm_residual(out, coarse=True), out.x))
        # Every basin the coarse stage found is kept, best first; a sign flip
        # cannot cross between basins.
        candidates.sort(key=lambda item: item[0])
        coarse = candidates[0][1]
    else:
        coarse = coarse.x

    # Stage 2, full. Seeded from the coarse answer, and -- because the even
    # modes are sign-degenerate, so the landscape has mirrored minima -- also
    # from its sign-flipped variants when the first pass fits badly.
    best = run(coarse)
    best_res = norm_residual(best)

    # Stage 3, escape: the other coarse basins first, then the
    # electric-field search.
    if opts.multistart and best_res > opts.residual_ok:
        retries = [x for _score, x in candidates[1:]]
        retries.append(zero)
        if fwd.dm is None:
            for j in (4, 11):
                if j in fwd.modes:
                    flip = best.x.copy()
                    flip[fwd.modes.index(j)] *= -1.0
                    retries.append(flip)
        else:
            # Same intent, expressed for a basis whose modes are not Noll
            # indices: the mirrored minimum is the twin wavefront, which is
            # a different vector here rather than a pair of sign flips.
            twin = best.x.copy()
            twin[:n_modes] = fwd.dm.twin(twin[:n_modes])
            retries.append(twin)
        stale = 0
        for start in retries:
            out = run(start)
            score = norm_residual(out)
            if score < best_res * (1.0 - opts.improve_frac):
                best, best_res, stale = out, score, 0
            else:
                if score < best_res:
                    best, best_res = out, score
                stale += 1
            if best_res <= opts.residual_ok or stale >= opts.patience:
                break

    used_efield = False
    if opts.efield_rounds > 0 and best_res > opts.residual_ok:
        psf_estimate = EF.measured_psf(crop)
        if psf_estimate is not None:
            for _round in range(int(opts.efield_rounds)):
                amps, shift_now, scale_now = unpack(best.x)
                estimated = EF.gerchberg_saxton(
                    fwd, psf_estimate, amps, shift_now, scale_now,
                    opts.gs_iterations)
                proposal, _score = EF.search(
                    fwd, estimated, amps, shift_now, scale_now,
                    crop.shape[0])
                start = best.x.copy()
                start[:n_modes] = proposal
                out = run(start)
                score = norm_residual(out)
                if score < best_res:
                    best, best_res, used_efield = out, score, True
                if best_res <= opts.residual_ok:
                    break

    coeffs, shift, scale = unpack(best.x)
    sigma_coeff, sigma_rms = _fisher_sigma(best, coeffs, n_modes)
    final_model = fwd.psf(coeffs, shift, scale, crop.shape[0]).ravel()[keep]
    fitted_flux, fitted_background = _solve_flux_background(
        final_model, keep_data, keep_weight,
        None if opts.fit_background else measured_background)
    if fwd.dm is None:
        coeffs, even_flipped = _fix_even_sign(coeffs, fwd.modes)
        dense = _dense(coeffs, fwd.modes)
        report = {j: float(a) for j, a in zip(fwd.modes, coeffs)}
        sigma_report = {j: float(v) for j, v in zip(fwd.modes, sigma_coeff)}
        modal, modal_sigma = {}, {}
        rms = float(np.sqrt(np.sum(np.square(coeffs))))
    else:
        coeffs, dense, even_flipped = _dm_report(fwd, coeffs)
        report = {j: float(dense[j - 1]) for j in range(4, len(dense) + 1)}
        # Modal sigma carried onto the Noll terms, diagonal only, so it is a
        # lower bound like `sigma_rms_waves`.
        zmat = fwd.dm.zernike_matrix()
        sig = np.sqrt(np.square(np.asarray(sigma_coeff, float))
                      @ np.square(zmat))
        sigma_report = {j: float(sig[j - 1]) for j in range(4, len(dense) + 1)}
        modal = {n: float(a) for n, a in zip(fwd.dm.labels, coeffs)}
        modal_sigma = {n: float(v)
                       for n, v in zip(fwd.dm.labels, sigma_coeff)}
        # Phase RMS over the pupil with piston and tilt removed, computed from
        # the map because the eigenmodes are not orthonormal.
        rms = _phase_rms(fwd, coeffs)
    named = ZK.named_aberrations(dense)
    est = WavefrontEstimate(
        coeffs=report,
        named=named,
        rms_waves=rms,
        rms_rad=rms * 2.0 * np.pi,
        n_pixels=int(keep.sum()),
        photon_gain=float(gain),
        sigma_waves=sigma_report,
        fit_coeffs=tuple(float(a) for a in coeffs),
        modal_amplitudes=modal,
        modal_sigma=modal_sigma,
        basis_name=("zernike" if fwd.dm is None
                    else f"dm:{Path(fwd.dm.source).parent.name}"),
        sigma_rms_waves=sigma_rms,
        efield_used=used_efield,
        strehl=fwd.strehl(coeffs),
        strehl_marechal=float(np.exp(-(2.0 * np.pi * rms) ** 2)),
        residual=float(best_res),
        converged=bool(best.status > 0),
        reason=_reason(best.status, prior),
        beyond_capture_range=bool(prior > CAPTURE_RANGE_WAVES),
        prior_waves=float(prior),
        sign_resolved=False,
        even_sign_flipped=even_flipped,
        pupil_scale=float(scale),
        centre_px=(origin[0] + crop.shape[1] / 2.0 + shift[0] / px_um,
                   origin[1] + crop.shape[0] / 2.0 + shift[1] / px_um),
        shift_um=(float(shift[0]), float(shift[1])),
        flux=fitted_flux,
        background=fitted_background,
        roi_origin=(int(origin[0]), int(origin[1])),
        roi_px=int(crop.shape[0]),
        n_evals=int(evals[0]),
        ms=(time.perf_counter() - started) * 1e3)
    if want_phase:
        phase = fwd.phase_waves(coeffs)
        est.pupil_phase = np.where(fwd.mask, phase, np.nan)
    return est


def render_model(est: WavefrontEstimate, optics: Optics,
                 options: RetrievalOptions = None):
    """Re-render the fitted spot of one estimate, in the frame's own units.

    Put the model beside the measured crop to see whether the fit reproduced
    the structure or only its brightness.

    Args:
        est: A converged or unconverged estimate from `estimate_wavefront`.
        optics: The same optical constants the fit used.
        options: The same options the fit used; defaults are used when
            omitted, which is correct only if the fit used defaults too.

    Returns:
        A 2-D array the size of the fitted crop, or None when the estimate
        carries no fit to render.

    Raises:
        ValueError: If the estimate predates the stored nuisance terms.
    """
    if not est.coeffs or not est.roi_px:
        return None
    if not np.isfinite(est.flux):
        raise ValueError("estimate carries no fitted flux to render with")
    opts = options or RetrievalOptions()
    fwd = _Forward(optics, opts)
    # `fit_coeffs` is the vector the solver actually converged on, in the
    # basis it was fitted in. `coeffs` is the Zernike REPORT, which is the
    # same wavefront only when the fit was in Zernikes to begin with.
    amps = (list(est.fit_coeffs) if est.fit_coeffs
            else [est.coeffs.get(j, 0.0) for j in fwd.modes])
    if len(amps) != len(fwd.modes):
        raise ValueError("options do not match the basis this estimate was "
                         f"fitted in ({len(amps)} coefficients, "
                         f"{len(fwd.modes)} modes)")
    scale = est.pupil_scale if np.isfinite(est.pupil_scale) else 1.0
    model = fwd.psf(amps, est.shift_um, scale, est.roi_px)
    return est.flux * model + est.background


def fitted_crop(frame, est: WavefrontEstimate):
    """The exact crop of `frame` that produced `est`, for a like-for-like
    comparison against `render_model`.

    Args:
        frame: The full frame the estimate was made from.
        est: The estimate carrying the crop origin and size.
    """
    x0, y0 = est.roi_origin
    side = est.roi_px
    img = np.asarray(frame, float)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img[y0:y0 + side, x0:x0 + side]


def _even_modes(modes):
    """The subset of `modes` whose Zernikes are even under r -> -r.

    A Zernike of radial order n has parity (-1)^n, so the even block is
    exactly the modes of even radial order.

    Args:
        modes: Noll indices.

    Returns:
        A boolean array, one entry per mode.
    """
    return np.array([ZK.noll_to_nm(j)[0] % 2 == 0 for j in modes])


def _fix_even_sign(coeffs, modes):
    """Resolve the one free even-mode sign by a stated convention.

    The relative signs inside the even block are left alone; the convention
    is that the largest-magnitude even coefficient comes out positive.

    Args:
        coeffs: Fitted coefficients, in fit order.
        modes: The matching Noll indices.

    Returns:
        A (coefficients, flipped) pair.
    """
    values = np.asarray(coeffs, float).copy()
    even = _even_modes(modes)
    if not even.any():
        return values, False
    dominant = int(np.argmax(np.abs(np.where(even, values, 0.0))))
    if values[dominant] >= 0:
        return values, False
    values[even] = -values[even]
    return values, True


def _reason(status, prior):
    """The caveat a caller must read before quoting the coefficients."""
    notes = []
    if status <= 0:
        notes.append("solver hit its evaluation budget without converging")
    if prior > CAPTURE_RANGE_WAVES:
        notes.append(
            f"spot implies about {prior:.2f} waves rms, beyond the measured "
            f"capture range of {CAPTURE_RANGE_WAVES:.2f}; treat the "
            "coefficients as a lower bound, not a measurement")
    return "; ".join(notes)


def _fisher_sigma(result, coeffs, n_modes):
    """Per-mode one-sigma uncertainty from the converged fit.

    The residual is already divided by the expected per-pixel noise, so
    J^T J is the Fisher information matrix and (J^T J)^-1 the parameter
    covariance.

    Args:
        result: The scipy least-squares result at the solution.
        coeffs: The fitted coefficients, for the RMS error propagation.
        n_modes: How many leading parameters are coefficients.

    Returns:
        A (per-mode sigma array, sigma on the wavefront RMS) pair. Both are
        NaN when the Jacobian is singular, which is what a mode the data
        cannot constrain looks like.
    """
    nan = np.full(n_modes, float("nan"))
    jac = getattr(result, "jac", None)
    if jac is None or jac.size == 0:
        return nan, float("nan")
    try:
        covariance = np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        return nan, float("nan")
    variances = np.diag(covariance)[:n_modes]
    if not np.all(np.isfinite(variances)) or np.any(variances < 0):
        return nan, float("nan")
    sigma = np.sqrt(variances)
    # rms = sqrt(sum a^2), so d(rms)/da_j = a_j / rms; the modes are treated
    # as independent, which the off-diagonal covariance says they are not,
    # so this is a lower bound on the RMS uncertainty.
    total = float(np.sqrt(np.sum(np.square(coeffs))))
    if not total > 0:
        return sigma, float("nan")
    weights = np.asarray(coeffs, float) / total
    return sigma, float(np.sqrt(np.sum((weights * sigma) ** 2)))


def _dense(coeffs, modes):
    """Noll 1..max(modes) vector, for `common.zernike.named_aberrations`."""
    out = np.zeros(max(modes))
    for j, a in zip(modes, coeffs):
        out[j - 1] = float(a)
    return out


def _phase_rms(fwd, coeffs):
    """RMS of a fitted pupil phase in waves, piston and tilt removed.

    For an orthonormal Zernike set this equals `sqrt(sum a_j^2)`; for the
    mirror's eigenmodes, which are neither orthonormal on the model pupil
    nor confined to the reported Noll terms, only the map gives the right
    number.

    Args:
        fwd: The forward model whose basis produced `coeffs`.
        coeffs: Fitted amplitudes, in fit order.
    """
    n = fwd.mask.shape[0]
    circ = ((n - 1) / 2.0,) * 3
    phase = fwd.phase_waves(coeffs)
    fit = ZK.fit(np.where(fwd.mask, phase, np.nan), fwd.mask, circ, n_modes=3)
    return float(np.nanstd(fit["residual"][fwd.mask]))


def _dm_report(fwd, coeffs):
    """Modal amplitudes -> (amplitudes, dense Noll vector, flipped).

    Applies the same even-sign convention as the Zernike path by swapping
    to the twin wavefront, so amplitudes, pupil phase and reported Zernikes
    describe one wavefront.

    Args:
        fwd: The forward model carrying the DM basis.
        coeffs: Fitted modal amplitudes.

    Returns:
        A (amplitudes, dense Noll 1..N vector, flipped) triple.
    """
    zmat = fwd.dm.zernike_matrix()
    amps = np.asarray(coeffs, float)
    dense = amps @ zmat
    even = np.array([ZK.noll_to_nm(j)[0] % 2 == 0
                     for j in range(1, len(dense) + 1)])
    even[:3] = False  # Piston and tilt carry no aberration.
    if not even.any():
        return amps, dense, False
    dominant = int(np.argmax(np.abs(np.where(even, dense, 0.0))))
    if dense[dominant] >= 0:
        return amps, dense, False
    amps = fwd.dm.twin(amps)
    return amps, amps @ zmat, True
