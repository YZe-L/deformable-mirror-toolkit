# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-08-21

"""Electric-field-space search, to escape the intensity fit's local minima.

Implements Zingarelli and Cain, Appl. Opt. 52, 7435 (2013): the detector
field is estimated by Gerchberg-Saxton from the measured intensity and
correlated against modelled fields, whose optima sit in different places
from the intensity fit's. The field estimate is accurate only for small
aberration (about 0.16 waves per mode), so `efield_rounds` defaults to 0.
"""

from __future__ import annotations

import numpy as np
from prysm.propagation import Wavefront


def measured_psf(crop):
    """Normalised, non-negative PSF estimate from one measured crop.

    Args:
        crop: Background-inclusive measured intensity.

    Returns:
        The crop with its background level removed, clipped at zero and
        scaled to unit sum, or None when there is no positive signal.
    """
    image = np.clip(np.asarray(crop, float) - float(np.median(crop)),
                    0.0, None)
    total = float(image.sum())
    return image / total if total > 0 else None


def _pupil_wavefront(forward, coeffs, shift_um, pupil_scale):
    """The modelled pupil field as a prysm wavefront."""
    diameter_mm = forward.optics.aperture_mm * float(pupil_scale)
    dx_mm = diameter_mm / forward.basis.shape[-1]
    phase = forward.phase_waves(coeffs, shift_um)
    field = forward.amp * np.exp(1j * 2.0 * np.pi * phase)
    return Wavefront(field.astype(complex), forward.wav_um, dx_mm,
                     space="pupil"), dx_mm


def model_field(forward, coeffs, shift_um, pupil_scale, samples):
    """Modelled complex field in the detector plane.

    This is the same propagation `_Forward.psf` performs, stopped one step
    earlier: the field rather than its squared modulus.

    Args:
        forward: The cached forward model.
        coeffs: One amplitude per fitted mode, in RMS waves.
        shift_um: Spot offset from the ROI centre, in sensor micrometres.
        pupil_scale: Multiplier on the configured aperture diameter.
        samples: Side of the square detector grid, in pixels.

    Returns:
        A complex array of shape (samples, samples).
    """
    wave, _ = _pupil_wavefront(forward, coeffs, shift_um, pupil_scale)
    focused = wave.focus_fixed_sampling(
        efl=forward.optics.focal_mm, dx=forward.optics.pixel_um,
        samples=int(samples), shift=(0, 0))
    return np.asarray(focused.data)


def gerchberg_saxton(forward, psf, coeffs, shift_um, pupil_scale,
                     iterations=25):
    """Estimate the detector-plane field from the measured intensity.

    Alternating projections between the two constraints this problem
    actually has: the measured magnitude in the detector plane, and the
    known aperture amplitude in the pupil.

    Args:
        forward: The cached forward model.
        psf: Measured, unit-sum, non-negative PSF estimate.
        coeffs: Starting coefficients, in RMS waves.
        shift_um: Spot offset from the ROI centre, in sensor micrometres.
        pupil_scale: Multiplier on the configured aperture diameter.
        iterations: Alternating-projection rounds.

    Returns:
        The estimated complex detector field, on the grid of `psf`.
    """
    samples = psf.shape[0]
    magnitude = np.sqrt(psf)
    diameter_mm = forward.optics.aperture_mm * float(pupil_scale)
    dx_mm = diameter_mm / forward.basis.shape[-1]
    pupil_samples = forward.basis.shape[-1]
    phase = forward.phase_waves(coeffs, shift_um)
    pupil = forward.amp * np.exp(1j * 2.0 * np.pi * phase)

    field = None
    for _ in range(int(iterations)):
        wave = Wavefront(pupil.astype(complex), forward.wav_um, dx_mm,
                         space="pupil")
        field = np.asarray(wave.focus_fixed_sampling(
            efl=forward.optics.focal_mm, dx=forward.optics.pixel_um,
            samples=samples, shift=(0, 0)).data)
        # Detector constraint: keep the retrieved phase, take the measured
        # magnitude. The guard keeps a zero-field pixel from producing NaN.
        scale = np.divide(magnitude, np.abs(field),
                          out=np.zeros_like(magnitude),
                          where=np.abs(field) > 0)
        field = field * scale
        back = Wavefront(field, forward.wav_um, forward.optics.pixel_um,
                         space="psf")
        recovered = np.asarray(back.unfocus_fixed_sampling(
            efl=forward.optics.focal_mm, dx=dx_mm, samples=pupil_samples,
            shift=(0, 0)).data)
        # Pupil constraint: the aperture amplitude is known exactly, so only
        # the recovered phase survives. This is also what makes the
        # back-propagation's amplitude error at the pupil edge harmless.
        pupil = forward.amp * np.exp(1j * np.angle(recovered))
    return field


def correlation(estimated, model):
    """Global-phase-invariant correlation between two complex fields.

    The modulus of the normalised inner product (Zingarelli and Cain, Eq.
    29, with the piston offset removed).

    Args:
        estimated: Field estimated from the measurement.
        model: Field predicted by a candidate wavefront.

    Returns:
        A correlation in 0..1; 1 is a perfect match up to piston.
    """
    a, b = estimated.ravel(), model.ravel()
    denom = np.sqrt(float(np.vdot(a, a).real) * float(np.vdot(b, b).real))
    if not denom > 0:
        return 0.0
    return float(abs(np.vdot(a, b)) / denom)


def search(forward, estimated, coeffs, shift_um, pupil_scale, samples,
           step=0.15, min_step=0.02, max_rounds=40):
    """Direct search maximising field correlation, one mode at a time.

    Perturb each coefficient by plus and minus a step, keep whichever
    candidate correlates best, then halve the step and repeat.

    Args:
        forward: The cached forward model.
        estimated: Detector field estimated from the measurement.
        coeffs: Starting coefficients, in RMS waves.
        shift_um: Spot offset from the ROI centre, in sensor micrometres.
        pupil_scale: Multiplier on the configured aperture diameter.
        samples: Side of the square detector grid, in pixels.
        step: Initial coefficient increment, in RMS waves.
        min_step: Increment below which the search stops.
        max_rounds: Cap on sweeps, so a flat landscape still terminates.

    Returns:
        A (coefficients, correlation) pair.
    """
    best = np.array(coeffs, float)
    score = correlation(estimated, model_field(forward, best, shift_um,
                                               pupil_scale, samples))
    rounds = 0
    while step >= min_step and rounds < max_rounds:
        improved = False
        for index in range(best.size):
            for sign in (1.0, -1.0):
                trial = best.copy()
                trial[index] += sign * step
                trial_score = correlation(
                    estimated,
                    model_field(forward, trial, shift_um, pupil_scale,
                                samples))
                if trial_score > score:
                    best, score, improved = trial, trial_score, True
            rounds += 1
            if rounds >= max_rounds:
                break
        if not improved:
            step *= 0.5
    return best, score
