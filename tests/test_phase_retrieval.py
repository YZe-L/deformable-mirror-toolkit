# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.1, 2026-08-22

"""Round-trip, ambiguity and calibration checks for single-frame retrieval.

Written against `unittest`, as the rest of this repository is; there is no
pytest in the project environment.

Every test asserts only on the public output of `estimate_wavefront` for a
given image, never on iteration counts or private structure, so the solver
policy can be retuned without rewriting the suite.

Tolerances are stated in waves and justified per test. Single-mode recovery
gets 0.02 waves, a factor of ten below the aberrations under test and about
twice the worst error measured across the case table; mixtures get 0.05,
since mode cross-talk is real and the assertion is that the fit is usable,
not that it is exact.
"""

from __future__ import annotations

import unittest

import numpy as np

from dm_toolkit import zernike as ZK
from dm_toolkit.phase_retrieval import (Optics, RetrievalOptions, estimate_wavefront,
                               fitted_crop, render_model)
from dm_toolkit.phase_retrieval.estimate import _Forward, _dense

# The bench this project actually runs: 520 nm, f/36, 3.45 um pixels, so the
# Airy radius is 6.6 px and the core is comfortably sampled.
OPTICS = Optics(wavelength_nm=520.0, focal_mm=1800.0, aperture_mm=50.0,
                pixel_um=3.45)
TERMS = ("defocus", "astigmatism", "coma", "trefoil", "spherical")
FLUX = 3e6  # Photons in the modelled spot: a bright, unsaturated exposure.
BACKGROUND = 30.0


def render(coeffs, seed=0, flux=FLUX, shift_um=(1.2, -0.8)):
    """A noisy full-frame image of one aberrated spot.

    The background carries its own Poisson noise rather than being a flat
    pedestal: a noiseless border drives the estimator's noise floor to zero,
    which no detector reaches and which hid a real weighting bug once.

    Args:
        coeffs: Mapping of Noll index to RMS waves.
        seed: Seed for the Poisson noise.
        flux: Total photons in the spot.
        shift_um: Spot offset from the ROI centre, in sensor micrometres.

    Returns:
        A 1080x1440 float frame, as the camera pages deliver.
    """
    forward = _Forward(OPTICS, RetrievalOptions())
    amps = np.array([coeffs.get(j, 0.0) for j in forward.modes])
    rng = np.random.default_rng(seed)
    spot = forward.psf(amps, shift_um, 1.0, 300) * flux + BACKGROUND
    frame = rng.poisson(np.full((1080, 1440), BACKGROUND)).astype(float)
    frame[400:700, 500:800] = rng.poisson(np.clip(spot, 0, None))
    return frame


def truth(coeffs):
    """The named magnitudes a perfect retrieval of `coeffs` would report."""
    forward = _Forward(OPTICS, RetrievalOptions())
    dense = _dense([coeffs.get(j, 0.0) for j in forward.modes], forward.modes)
    named = ZK.named_aberrations(dense)
    named["defocus"] = abs(named["defocus"])
    named["spherical"] = abs(named["spherical"])
    return named


class RoundTrip(unittest.TestCase):
    """Injected coefficients come back at the right magnitude."""

    SINGLE = {"defocus": {4: 0.2}, "astigmatism": {6: 0.25},
              "coma": {8: 0.2}, "trefoil": {9: 0.2}, "spherical": {11: 0.2}}

    def test_single_modes(self):
        for name, coeffs in self.SINGLE.items():
            with self.subTest(mode=name):
                est = estimate_wavefront(render(coeffs), OPTICS)
                want = truth(coeffs)
                self.assertTrue(est.converged, est.reason)
                for term in TERMS:
                    self.assertAlmostEqual(est.named[term], want[term],
                                           delta=0.02, msg=term)
                self.assertAlmostEqual(
                    est.rms_waves,
                    float(np.sqrt(sum(v ** 2 for v in coeffs.values()))),
                    delta=0.02)

    def test_mixture(self):
        """A mixture of five modes is separated to better than 0.05 waves."""
        coeffs = {4: 0.15, 5: 0.1, 6: -0.12, 8: 0.18, 11: 0.1}
        est = estimate_wavefront(render(coeffs), OPTICS)
        want = truth(coeffs)
        self.assertTrue(est.converged, est.reason)
        for term in TERMS:
            self.assertAlmostEqual(est.named[term], want[term], delta=0.05,
                                   msg=term)


class Ambiguities(unittest.TestCase):
    """The documented degeneracies, asserted rather than worked around."""

    def test_even_mode_sign_is_not_observable(self):
        """+a and -a on an even mode are indistinguishable, by construction.

        Smith et al., JOSA A 30, 2002 (2013), Property 3.1: the linear term
        of the PSF is invariant in the even aberrations, so a single
        in-focus frame cannot separate them. The two must therefore report
        the same magnitudes and the same RMS.
        """
        for mode in (4, 6, 11):
            with self.subTest(noll=mode):
                plus = estimate_wavefront(render({mode: 0.22}), OPTICS)
                minus = estimate_wavefront(render({mode: -0.22}), OPTICS)
                self.assertAlmostEqual(plus.rms_waves, minus.rms_waves,
                                       delta=0.02)
                for term in TERMS:
                    self.assertAlmostEqual(plus.named[term],
                                           minus.named[term], delta=0.02,
                                           msg=term)

    def test_relative_even_mode_signs_are_observable(self):
        """Only ONE overall even-mode sign is lost, not one per mode.

        The twin solution is phi(r) -> -phi(-r): it negates the whole even
        block together. Flipping Z11 against Z4 is therefore a physically
        different wavefront and must be recovered as such -- taking each
        even coefficient's magnitude independently would throw away a real
        measurement. Orban de Xivry et al., MNRAS 509, 5701 (2021) Sec. 3.6.
        """
        same = estimate_wavefront(render({4: 0.20, 11: 0.15}), OPTICS)
        opposed = estimate_wavefront(render({4: 0.20, 11: -0.15}), OPTICS)
        self.assertGreater(same.named["spherical"], 0.10)
        self.assertLess(opposed.named["spherical"], -0.10)
        for est in (same, opposed):
            self.assertAlmostEqual(est.named["defocus"], 0.20, delta=0.03)

    def test_the_twin_solution_lands_on_the_convention(self):
        """A globally negated even block reports as its positive twin."""
        plus = estimate_wavefront(render({4: 0.20, 11: 0.15}), OPTICS)
        twin = estimate_wavefront(render({4: -0.20, 11: -0.15}), OPTICS)
        for term in ("defocus", "spherical"):
            self.assertAlmostEqual(plus.named[term], twin.named[term],
                                   delta=0.03, msg=term)

    def test_strehl_ranks_wavefronts_in_order(self):
        """The objective ranks increasing aberration correctly.

        This is the property an optimiser depends on, and it is weaker and
        far more robust than per-coefficient accuracy.
        """
        scores = [estimate_wavefront(render({4: s, 6: s}), OPTICS).strehl
                  for s in (0.05, 0.15, 0.25, 0.35)]
        self.assertEqual(scores, sorted(scores, reverse=True))


class Robustness(unittest.TestCase):
    """Nuisance parameters, refusals, and the two-recording regression."""

    def test_nuisance_terms_do_not_move_the_answer(self):
        """A shifted, dimmer, pedestal-shifted spot fits the same wavefront."""
        coeffs = {6: 0.2, 8: 0.15}
        base = estimate_wavefront(render(coeffs), OPTICS)
        moved = estimate_wavefront(
            render(coeffs, seed=7, flux=FLUX / 4, shift_um=(-9.0, 6.5)),
            OPTICS)
        for term in TERMS:
            self.assertAlmostEqual(base.named[term], moved.named[term],
                                   delta=0.05, msg=term)

    def test_two_recordings_of_one_wavefront_agree(self):
        """Near-identical frames must give near-identical decompositions.

        Regression for a real failure: two correction runs recorded the same
        start state (frames 97.9% cross-correlated) and the retrieval
        reported one as 0.79 waves of astigmatism and the other as 0.48
        waves of defocus. Those are separate basins of one landscape, and
        the coarse stage was choosing between them from a crop too small to
        hold a spot that large. The assertion is on agreement BETWEEN the
        two, not on either being right.
        """
        coeffs = {6: 0.55, 4: 0.2, 9: 0.25}
        first = estimate_wavefront(render(coeffs, seed=1), OPTICS)
        second = estimate_wavefront(render(coeffs, seed=2), OPTICS)
        self.assertAlmostEqual(first.rms_waves, second.rms_waves, delta=0.05)
        for term in TERMS:
            self.assertAlmostEqual(first.named[term], second.named[term],
                                   delta=0.06, msg=term)

    def test_coarse_stage_keeps_the_whole_spot(self):
        """A spread spot is still decomposed correctly, not read from its core.

        The coarse stage bins the full ROI rather than cropping it, so a
        large aberration is not classified from the core alone. Kept inside
        the measured capture range (0.45 waves) so that this tests the
        binning and not the capture range -- `test_beyond_capture_range`
        covers the other side.
        """
        coeffs = {6: 0.35, 9: 0.2}
        est = estimate_wavefront(render(coeffs), OPTICS)
        want = truth(coeffs)
        self.assertAlmostEqual(est.named["astigmatism"], want["astigmatism"],
                               delta=0.08)
        self.assertLess(est.named["defocus"], 0.15)  # Not read as defocus.

    def test_beyond_capture_range_is_announced(self):
        """A wavefront past the capture range is flagged, not quietly wrong.

        Measured limit 0.45 waves rms (`calibration.check_capture_range`),
        consistent with Pellegrino (Rochester 2019) Ch. 3.2. Past it the fit
        lands short -- 0.71 waves came back as 0.37 -- and the solver still
        terminates normally, so a caller checking only `converged` would be
        misled. The guard uses the second-moment prior, which is computed
        from the spot BEFORE fitting and so cannot be fooled by the fit.
        """
        est = estimate_wavefront(render({6: 0.5, 4: 0.4, 8: 0.3}), OPTICS)
        self.assertTrue(est.beyond_capture_range)
        self.assertIn("lower bound", est.reason)
        modest = estimate_wavefront(render({6: 0.25}), OPTICS)
        self.assertFalse(modest.beyond_capture_range)
        self.assertEqual(modest.reason, "")

    def test_refuses_unconfigured_optics(self):
        """No pupil scale means no fit, and the refusal says so."""
        est = estimate_wavefront(render({4: 0.2}),
                                 Optics(520.0, 0.0, 50.0, 3.45))
        self.assertFalse(est.converged)
        self.assertIn("optics", est.reason)

    def test_refuses_a_blank_frame(self):
        """Background only is refused, not fitted to a plausible number."""
        rng = np.random.default_rng(3)
        blank = rng.poisson(np.full((1080, 1440), BACKGROUND)).astype(float)
        est = estimate_wavefront(blank, OPTICS)
        self.assertFalse(est.converged)
        self.assertTrue(est.reason)


class Calibration(unittest.TestCase):
    """The two numbers a reader has to trust: residual and error bar."""

    def test_residual_has_no_absolute_scale(self):
        """The residual grows with flux even when the fit is exact.

        Asserting the documented limitation rather than a threshold. Under
        the default uniform weighting the weights use the background sigma,
        while the core's own shot noise grows as sqrt(flux), so a correct fit
        scores worse at higher flux. This is why `converged` rests on the
        solver's own criterion and why the residual may only be compared
        within one frame. Making it absolute needs a per-pixel variance
        model, hence the detector gain, hence a flat-field calibration this
        project does not yet have.
        """
        faint = estimate_wavefront(render({6: 0.25}, flux=FLUX / 10), OPTICS)
        bright = estimate_wavefront(render({6: 0.25}, flux=FLUX * 10), OPTICS)
        for est in (faint, bright):  # Both fits are correct.
            self.assertAlmostEqual(est.named["astigmatism"], 0.25, delta=0.02)
        self.assertGreater(bright.residual, faint.residual)

    def test_crop_is_sized_from_the_spot(self):
        """A larger aberration must be given a larger crop, automatically.

        The crop side is 4x the measured 80% encircled-energy radius, a
        multiple established by grid convergence rather than chosen, so it
        has to track the spot rather than a fixed pixel count.
        """
        small = estimate_wavefront(render({6: 0.12}), OPTICS)
        large = estimate_wavefront(render({6: 0.12, 4: 0.35, 9: 0.3}), OPTICS)
        self.assertGreater(large.roi_px, small.roi_px)

    def test_error_bars_scale_as_the_weighting_dictates(self):
        """The Fisher bound follows 1/flux, not 1/sqrt(flux), and that is why.

        Asserting the real behaviour rather than the one photon statistics
        would give. The default weights are the inverse BACKGROUND noise,
        which does not grow with the signal, so the Jacobian scales with flux
        and the bound with 1/flux. A correct per-pixel variance model would
        make it 1/sqrt(flux); that needs the detector gain, which needs
        flat-field data this project does not have yet.

        If this test starts failing towards sqrt, someone has added the
        variance model -- update the docstring in `estimate` with it.
        """
        bright = estimate_wavefront(render({6: 0.25}, flux=FLUX * 4), OPTICS)
        faint = estimate_wavefront(render({6: 0.25}), OPTICS)
        self.assertTrue(np.isfinite(bright.sigma_rms_waves))
        self.assertTrue(np.isfinite(faint.sigma_rms_waves))
        ratio = faint.sigma_rms_waves / bright.sigma_rms_waves
        self.assertAlmostEqual(ratio, 4.0, delta=0.6)

    def test_render_model_reproduces_the_fitted_image(self):
        """`render_model` returns the image the fit actually settled on.

        The model-versus-data figure is only an honesty check if it shows
        the fitted model and not a re-derived one, so the estimate carries
        the nuisance terms needed to rebuild it exactly.
        """
        frame = render({6: 0.2, 8: 0.15})
        est = estimate_wavefront(frame, OPTICS)
        model = render_model(est, OPTICS)
        data = fitted_crop(frame, est)
        self.assertEqual(model.shape, data.shape)
        self.assertLess(est.residual, 3.0)
        scatter = float(np.sqrt(np.mean((model - data) ** 2)))
        self.assertLess(scatter, 0.05 * (data.max() - np.median(data)))


class Conventions(unittest.TestCase):
    """A coefficient must mean the same thing on every page of the app."""

    def test_zernike_basis_matches_the_app(self):
        """The model's basis is `common.zernike`'s basis, mode for mode.

        Other pages report Zernikes through `common.zernike`; a different
        ordering or normalisation here would make these numbers
        untabulatable beside the interferometer's.
        """
        forward = _Forward(OPTICS, RetrievalOptions())
        n = forward.basis.shape[-1]
        axis = (np.arange(n) - (n - 1) / 2.0) / (n / 2.0)
        x, y = np.meshgrid(axis, axis)
        rho, theta = np.hypot(x, y), np.arctan2(y, x)
        mask = rho <= 1.0
        for i, j in enumerate(forward.modes):
            with self.subTest(noll=j):
                want = ZK.zernike_mode(j, rho[mask], theta[mask])
                np.testing.assert_allclose(forward.basis[i][mask], want,
                                           atol=1e-9)

    def test_radians_and_waves_agree(self):
        """`rms_rad` is `rms_waves` in the unit the literature quotes."""
        est = estimate_wavefront(render({6: 0.25}), OPTICS)
        self.assertAlmostEqual(est.rms_rad, est.rms_waves * 2 * np.pi,
                               places=9)


if __name__ == "__main__":
    unittest.main()
