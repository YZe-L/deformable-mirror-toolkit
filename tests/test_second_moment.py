# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.2, 2026-08-02

"""Check the property the whole modal solve rests on: the second moment is quadratic.

`test_modal` verifies the INVERSION against a landscape that is quadratic by
construction. This file verifies the premise instead -- that a real diffraction
calculation actually produces such a landscape -- by aberrating a pupil, taking
the far field, and fitting the second moment against the coefficient squared.

If these fail, the modal solvers are inverting a model the optics do not obey,
and no amount of correctness in the arithmetic would save them.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from dm_toolkit.correction import metrics as M
from dm_toolkit.correction import settings as S

N = 256  # Pupil samples across the diameter.
PAD = 4  # Far-field zero padding, so one lambda/D spans PAD pixels.


def _pupil():
    y, x = np.mgrid[-1:1:N * 1j, -1:1:N * 1j]
    r = np.hypot(x, y)
    return r, np.arctan2(y, x), (r <= 1.0).astype(float)


R, TH, PUP = _pupil()

# RMS-normalised Zernikes, so a coefficient IS the wavefront error in rad RMS.
MODES = {
    "astigmatism": np.sqrt(6) * R ** 2 * np.cos(2 * TH),
    "defocus": np.sqrt(3) * (2 * R ** 2 - 1),
    "coma": np.sqrt(8) * (3 * R ** 3 - 2 * R) * np.sin(TH),
}


def far_field(phase):
    """Intensity in the focal plane for one pupil phase, in rad."""
    big = np.zeros((N * PAD, N * PAD), complex)
    big[:N, :N] = PUP * np.exp(1j * phase)
    return np.abs(np.fft.fftshift(np.fft.fft2(big))) ** 2


def moment(psf, radius):
    """Second moment inside a FIXED aperture about the frame centre."""
    c = psf.shape[0] // 2
    h = int(radius)
    crop = psf[c - h:c + h + 1, c - h:c + h + 1]
    yy, xx = np.ogrid[0:crop.shape[0], 0:crop.shape[1]]
    inside = np.hypot(xx - h, yy - h) <= radius
    return M._second_moment(crop, inside, h, h)


class SecondMomentIsQuadraticTest(unittest.TestCase):
    """The second moment must follow `c0 + c1 a^2` over the range a solve probes."""

    AMPLITUDES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    RADIUS = 120  # 30 lambda/D, wide enough that truncation stays small.

    def _fit(self, mode):
        values = np.array([moment(far_field(a * MODES[mode]), self.RADIUS)
                           for a in self.AMPLITUDES])
        design = np.column_stack([np.ones_like(self.AMPLITUDES),
                                  self.AMPLITUDES ** 2])
        coef, *_ = np.linalg.lstsq(design, values, rcond=None)
        residual = values - design @ coef
        return coef, float(np.max(np.abs(residual)) / np.ptp(values))

    def test_each_mode_is_quadratic(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                coef, worst = self._fit(mode)
                self.assertGreater(coef[1], 0, f"{mode}: curvature must be > 0")
                self.assertLess(worst, 0.01,
                                f"{mode}: departs from a parabola by "
                                f"{100 * worst:.2f}% of its swing")

    def test_curvature_ratio_matches_gradient_energy(self):
        """Defocus should read twice astigmatism's curvature.

        The identity behind the second moment makes the quadratic coefficient proportional to
        each mode's mean square wavefront GRADIENT, and for RMS-normalised
        Zernikes of the same radial order that ratio is exactly 2. Getting this
        right is what makes the impact matrix's driven-coordinate Gram
        eigenvalue a usable curvature.
        """
        ratio = self._fit("defocus")[0][1] / self._fit("astigmatism")[0][1]
        self.assertAlmostEqual(ratio, 2.0, delta=0.05)

    def test_truncation_is_what_breaks_it(self):
        """A tighter aperture must degrade the fit, not improve it.

        Recorded because it is the failure mode in practice: the metric's
        aperture has to hold the aberrated spot, and coma -- which throws energy
        furthest -- is the first to suffer.
        """
        values = np.array([moment(far_field(a * MODES["coma"]), 40)
                           for a in self.AMPLITUDES])
        design = np.column_stack([np.ones_like(self.AMPLITUDES),
                                  self.AMPLITUDES ** 2])
        coef, *_ = np.linalg.lstsq(design, values, rcond=None)
        tight = float(np.max(np.abs(values - design @ coef)) / np.ptp(values))
        self.assertGreater(tight, self._fit("coma")[1])


class SecondMomentBasicsTest(unittest.TestCase):
    """Properties the solvers rely on beyond the quadratic form."""

    @staticmethod
    def _gaussian(sigma, shift=0.0, size=400):
        y, x = np.mgrid[0:size, 0:size]
        c = size / 2.0
        return 3000 * np.exp(-(((x - c - shift) ** 2 + (y - c) ** 2)
                               / (2 * sigma ** 2)))

    @staticmethod
    def _whole(img):
        h = img.shape[0] // 2 - 1
        c = img.shape[0] // 2
        crop = img[c - h:c + h, c - h:c + h]
        yy, xx = np.ogrid[0:crop.shape[0], 0:crop.shape[1]]
        inside = np.hypot(xx - h, yy - h) <= h
        return M._second_moment(crop, inside, h, h)

    def test_matches_the_analytic_gaussian_moment(self):
        """A 2-D Gaussian has <r^2> = 2 sigma^2, exactly."""
        for sigma in (4.0, 6.0, 8.0):
            with self.subTest(sigma=sigma):
                self.assertAlmostEqual(self._whole(self._gaussian(sigma)),
                                       2 * sigma ** 2, delta=0.05 * sigma ** 2)

    def test_is_blind_to_translation(self):
        """Referenced to the centroid, so tilt must not register.

        This is what keeps the metric consistent with an impact matrix that had
        piston/tip/tilt removed: a shifted spot is the same spot.
        """
        still = self._whole(self._gaussian(6.0))
        moved = self._whole(self._gaussian(6.0, shift=25.0))
        self.assertAlmostEqual(moved, still, delta=0.02 * still)

    def test_empty_aperture_reads_nan_not_zero(self):
        """Zero would read as a perfect spot and be maximised."""
        blank = np.zeros((64, 64), float)
        inside = np.ones((64, 64), bool)
        self.assertTrue(np.isnan(M._second_moment(blank, inside, 32, 32)))


class SecondMomentScoringTest(unittest.TestCase):
    """The metric has to reach the optimiser the same way the others do."""

    def _cfg(self):
        return S.LoopSettings(metric=S.METRIC_SECOND_MOMENT, wavelength_nm=635.0,
                              focal_mm=100.0, aperture_mm=10.0, pixel_um=3.45)

    def test_measure_reports_it_and_prefers_the_tighter_spot(self):
        cfg = self._cfg()
        tight = M.measure(SecondMomentBasicsTest._gaussian(6.0).astype(np.float32),
                          cfg, bit_depth=12)
        loose = M.measure(SecondMomentBasicsTest._gaussian(12.0).astype(np.float32),
                          cfg, bit_depth=12)
        self.assertLess(tight.second_moment, loose.second_moment)
        self.assertGreater(tight.score, loose.score)

    def test_score_is_ungated(self):
        """primary_score must pass second_moment_score through untouched.

        A multiplicative gate would destroy the exact quadratic form, which is
        the only reason this metric exists.
        """
        cfg = self._cfg()
        r = M.measure(SecondMomentBasicsTest._gaussian(6.0).astype(np.float32), cfg,
                      bit_depth=12)
        self.assertEqual(M.primary_score(r, cfg), r.second_moment_score)

    def test_display_score_is_run_relative_not_aperture_absolute(self):
        """The 0..1 ruler is anchored to the run's fixed-ROI baseline."""
        cfg = self._cfg()
        seed = M.measure(SecondMomentBasicsTest._gaussian(12.0).astype(np.float32),
                         cfg, bit_depth=12)
        tight = M.measure(SecondMomentBasicsTest._gaussian(6.0).astype(np.float32),
                          cfg, bit_depth=12)
        ref = M.make_norm_ref(seed, cfg)
        self.assertAlmostEqual(M.norm_score(seed, ref, cfg), 0.5)
        self.assertAlmostEqual(
            M.norm_score(tight, ref, cfg),
            seed.second_moment / (seed.second_moment + tight.second_moment),
            places=6)

    def test_other_metrics_still_work(self):
        """The new branch must not have displaced an existing objective."""
        img = SecondMomentBasicsTest._gaussian(6.0).astype(np.float32)
        for metric in (S.METRIC_PIB, S.METRIC_PEAK, S.METRIC_SHARP,
                       S.METRIC_R_EE80, S.METRIC_RMS):
            with self.subTest(metric=metric):
                cfg = self._cfg()
                r = M.measure(img, S.LoopSettings(**{**cfg.__dict__,
                                                    "metric": metric}),
                              bit_depth=12)
                self.assertTrue(np.isfinite(r.score))

    def test_fixed_roi_stays_fixed_when_the_spot_size_changes(self):
        """The modal metric must not silently resize its integration domain."""
        cfg = self._cfg()
        seed = M.measure(
            SecondMomentBasicsTest._gaussian(6.0).astype(np.float32), cfg,
            bit_depth=12)
        roi = M.SecondMomentROI.from_reading(seed, margin_pct=30.0)
        tight = M.measure(
            SecondMomentBasicsTest._gaussian(6.0).astype(np.float32), cfg,
            bit_depth=12, second_moment_roi=roi)
        loose = M.measure(
            SecondMomentBasicsTest._gaussian(12.0).astype(np.float32), cfg,
            bit_depth=12, second_moment_roi=roi)
        self.assertAlmostEqual(tight.second_moment_roi_radius, roi.radius)
        self.assertAlmostEqual(loose.second_moment_roi_radius, roi.radius)
        self.assertGreater(loose.second_moment, tight.second_moment)

    def test_roi_margin_is_relative_to_the_detected_aperture(self):
        cfg = self._cfg()
        seed = M.measure(
            SecondMomentBasicsTest._gaussian(6.0).astype(np.float32), cfg,
            bit_depth=12)
        roi = M.SecondMomentROI.from_reading(seed, margin_pct=30.0)
        self.assertAlmostEqual(
            roi.radius, 1.30 * seed.photometry_aperture_radius)
        self.assertEqual(roi.margin_pct, 30.0)

    def test_containment_reports_the_geometric_overshoot(self):
        roi = M.SecondMomentROI(cx=10.0, cy=20.0, radius=30.0)
        reading = SimpleNamespace(
            photometry_aperture_cx=16.0,
            photometry_aperture_cy=28.0,
            photometry_aperture_radius=22.0)
        detail = roi.containment(reading, tolerance_px=1.0)
        self.assertFalse(detail["contains"])
        self.assertAlmostEqual(detail["center_distance_px"], 10.0)
        self.assertAlmostEqual(detail["required_radius_px"], 32.0)
        self.assertAlmostEqual(detail["overshoot_px"], 1.0)


if __name__ == "__main__":
    unittest.main()
