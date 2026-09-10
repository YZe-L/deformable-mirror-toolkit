# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.6, 2026-08-16

"""Drive the modal solvers against a simulated mirror with a known aberration.

The simulation is deliberately the model the solvers assume -- a metric exactly
quadratic in mode coefficients, with no cross terms -- because what is under
test is the INVERSION, not the physics. Whether a real spot obeys that model is
settled separately, by measuring the second moment on simulated point spread functions
(test_second_moment).
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, replace

import numpy as np

from dm_toolkit.correction import modal as MODAL
from dm_toolkit.correction import settings as S
from dm_toolkit.correction.optimizers import Staged, make_optimizer
from dm_toolkit.influence import im_store as ST

CHANNELS = (1, 2, 4, 6, 8)
N_MODES = 3
BIT_PER_RAD = 200.0  # Realistic stroke: one radian is a few hundred bits.
FLOOR = 50.0  # Diffraction floor of the simulated metric.


@dataclass
class FakeReading:
    """Only the fields a modal solver reads."""
    second_moment: float = float("nan")
    psd_band: float = float("nan")


class Mirror:
    """A mirror whose merit is exactly quadratic in its mode coefficients."""

    def __init__(self, aberration, curvature, seed=0):
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(len(CHANNELS), len(CHANNELS))))
        self.q = q[:, :N_MODES]  # Orthonormal mode directions.
        self.ctrl = BIT_PER_RAD * self.q  # Bit offset per rad RMS of a mode.
        self.centre = np.full(len(CHANNELS), 2000.0)
        self.aberration = np.asarray(aberration, float)
        self.curvature = np.asarray(curvature, float)

    def coefficients(self, bits):
        """Mode coefficients present at this command, in rad RMS."""
        offset = np.asarray(bits, float) - self.centre
        return self.aberration + self.q.T @ offset / BIT_PER_RAD

    def second_moment(self, bits):
        a = self.coefficients(bits)
        return FLOOR + float((self.curvature * a ** 2).sum())

    def reading(self, bits):
        m = self.second_moment(bits)
        # The PSD band reads larger for a better spot and it is its RECIPROCAL
        # that is quadratic, so invert the same quadratic to build one.
        return FakeReading(second_moment=m, psd_band=1.0 / m)

    def stored_matrix(self, keep=N_MODES):
        """The calibration the influence-matrix pipeline would have written.

        `StoredMatrix.curvature` combines these singular values with the norm
        of each RMS-scaled ctrl column, exactly as the real pipeline does. All
        fake columns share one norm, so the common factor is absorbed by the
        mode-1 CCD scale anchor and the requested relative curvature remains.
        """
        return ST.StoredMatrix(
            path=None, channels=CHANNELS, ctrl=self.ctrl,
            s=np.sqrt(self.curvature), keep=keep, grid_n=128,
            laser_nm=635.0, saved_utc="", source="simulated")


def settings(algorithm, **kw):
    acts = [S.Actuator(channel=c, start=2000, bit_min=0, bit_max=4095)
            for c in CHANNELS]
    cfg = S.LoopSettings(actuators=acts, algorithm=algorithm,
                         metric=S.MODAL_METRIC.get(algorithm, S.METRIC_SECOND_MOMENT))
    # Defaults a caller may override, rather than pinned arguments that would
    # collide with one.
    defaults = dict(min_step=1, modal_bias_rad=1.0, modal_rounds=2,
                    modal_auto_bias=False)
    return replace(cfg, **{**defaults, **kw})


def run(opt, mirror, budget=400):
    """Drive an optimiser to convergence, returning the bits it settled on."""
    for _ in range(budget):
        cmd = opt.ask()
        bits = np.array([cmd[c] for c in CHANNELS], float)
        reading = mirror.reading(bits)
        opt.observe(reading)
        # Higher-is-better, as the driver always supplies.
        opt.tell(1.0 / (1.0 + reading.second_moment), valid=True)
        if opt.status()["converged"]:
            break
    best = opt.best_command()
    return np.array([best[c] for c in CHANNELS], float)


class ModalSolveTest(unittest.TestCase):
    """Each solver must remove a known aberration it was never told."""

    ABERRATION = np.array([0.8, -1.1, 0.45])
    CURVATURE = np.array([4.0, 2.5, 1.0])

    def _check(self, algorithm, cls, tol):
        mirror = Mirror(self.ABERRATION, self.CURVATURE)
        cfg = settings(algorithm)
        opt = cls(cfg, mirror.stored_matrix())
        residual = mirror.coefficients(run(opt, mirror))
        start = float(np.abs(self.ABERRATION).max())
        worst = float(np.abs(residual).max())
        self.assertLess(
            worst, tol,
            f"{algorithm}: residual {residual} exceeds {tol} rad "
            f"(started at {start:.2f})")

    def test_fit_recovers_the_aberration(self):
        """2N+1 measures every curvature it uses, so it should be tightest."""
        self._check(S.ALGO_MODAL_FIT, MODAL.ModalFit, 0.02)

    def test_fast_recovers_the_aberration(self):
        """N+2 reuses the matrix curvature and self-calibrates the scale."""
        self._check(S.ALGO_MODAL_FAST, MODAL.ModalFast, 0.02)

    def test_psd_recovers_the_aberration(self):
        """Same solve on 1/g rather than the second moment."""
        self._check(S.ALGO_MODAL_PSD, MODAL.ModalPsd, 0.02)

    def test_fast_uses_fewer_measurements_than_fit(self):
        """The point of the N+2 solve is the measurement count."""
        mirror = Mirror(self.ABERRATION, self.CURVATURE)
        counts = {}
        for algorithm, cls in ((S.ALGO_MODAL_FIT, MODAL.ModalFit),
                               (S.ALGO_MODAL_FAST, MODAL.ModalFast)):
            opt = cls(settings(algorithm), mirror.stored_matrix())
            run(opt, mirror)
            counts[algorithm] = opt.status()["iter"]
        self.assertLess(counts[S.ALGO_MODAL_FAST], counts[S.ALGO_MODAL_FIT])

    def test_fast_is_n_plus_two_then_one_verification(self):
        """Cold-start estimation costs N+2; the commanded answer is verified."""
        mirror = Mirror(self.ABERRATION, self.CURVATURE)
        opt = MODAL.ModalFast(
            settings(S.ALGO_MODAL_FAST, modal_rounds=1),
            mirror.stored_matrix())
        for _ in range(N_MODES + 2):
            cmd = opt.ask()
            bits = np.array([cmd[c] for c in CHANNELS], float)
            reading = mirror.reading(bits)
            opt.observe(reading)
            opt.tell(1.0 / (1.0 + reading.second_moment), valid=True)
        self.assertEqual(opt.status()["iter"], N_MODES + 2)
        self.assertEqual(opt.status()["stage"], "verify")
        self.assertFalse(opt.converged)
        # The next reading is not part of the estimate: it independently checks
        # the command the model just predicted.
        cmd = opt.ask()
        bits = np.array([cmd[c] for c in CHANNELS], float)
        reading = mirror.reading(bits)
        opt.observe(reading)
        opt.tell(1.0 / (1.0 + reading.second_moment), valid=True)
        self.assertTrue(opt.converged)
        self.assertEqual(opt.status()["iter"], N_MODES + 3)

    def test_one_round_is_the_global_quadratic_default(self):
        self.assertEqual(S.LoopSettings().modal_rounds, 1)

    def test_each_modal_solver_pins_the_metric_it_was_derived_for(self):
        self.assertEqual(S.MODAL_METRIC[S.ALGO_MODAL_FIT],
                         S.METRIC_SECOND_MOMENT)
        self.assertEqual(S.MODAL_METRIC[S.ALGO_MODAL_FAST],
                         S.METRIC_SECOND_MOMENT)
        self.assertEqual(S.MODAL_METRIC[S.ALGO_MODAL_PSD], S.METRIC_PSD)

    def test_stale_curvature_only_hurts_the_fast_solve(self):
        """A wrong matrix curvature must not corrupt the fitted solve.

        This is the practical difference between the two: the fit divides only
        by numbers it measured this run, so a stale calibration cannot put a
        wrong scale on its answer.
        """
        mirror = Mirror(self.ABERRATION, self.CURVATURE)
        stale = mirror.stored_matrix()
        # Curvature ratios wrong by 3x, mode shapes still right.
        stale = ST.StoredMatrix(**{**stale.__dict__,
                                   "s": stale.s * np.array([1.0, 3.0, 0.5])})
        opt = MODAL.ModalFit(settings(S.ALGO_MODAL_FIT), stale)
        residual = mirror.coefficients(run(opt, mirror))
        self.assertLess(float(np.abs(residual).max()), 0.02)


class ModalGuardTest(unittest.TestCase):
    """The solvers must refuse what they cannot legitimately solve."""

    def test_channel_mismatch_is_refused(self):
        mirror = Mirror([0.5, 0.0, 0.0], [4.0, 2.5, 1.0])
        cfg = settings(S.ALGO_MODAL_FIT)
        cfg = replace(cfg, actuators=[S.Actuator(channel=c) for c in (1, 2, 3)])
        with self.assertRaises(ST.MatrixMismatch):
            MODAL.ModalFit(cfg, mirror.stored_matrix())

    def test_modal_algorithm_without_a_matrix_is_refused(self):
        with self.assertRaises(ValueError):
            make_optimizer(settings(S.ALGO_MODAL_FIT), matrix=None)

    def test_modal_factory_refuses_a_matrix_for_another_ao_wavelength(self):
        mirror = Mirror([0.5, 0.0, 0.0], [4.0, 2.5, 1.0])
        cfg = settings(S.ALGO_MODAL_FIT, wavelength_nm=520.0)
        with self.assertRaisesRegex(ST.MatrixMismatch, "635.*520"):
            make_optimizer(cfg, mirror.stored_matrix())

    def test_keep_limits_how_many_modes_are_driven(self):
        """Modes below the noise floor must be left alone, not divided by."""
        mirror = Mirror([0.5, 0.3, 0.2], [4.0, 2.5, 1.0])
        opt = MODAL.ModalFit(settings(S.ALGO_MODAL_FIT),
                             mirror.stored_matrix(keep=2))
        self.assertEqual(opt.n_modes, 2)


class StoredCurvatureScaleTest(unittest.TestCase):
    """Stored curvature must use the coordinates actually driven by ctrl."""

    def test_rms_rescaling_is_part_of_modal_curvature(self):
        """s^2 alone is only the curvature before ctrl columns are rescaled.

        The impact-matrix pipeline scales every eigenvector independently so
        one coefficient means one radian RMS.  ModalFast probes those scaled
        columns, therefore its curvature is s_i^2 * ||ctrl_i||^2, not s_i^2.
        """
        q = np.eye(3)
        s = np.array([4.0, 2.0, 0.5])
        column_norms = np.array([3.0, 5.0, 11.0])
        matrix = ST.StoredMatrix(
            path=None, channels=(1, 2, 4), ctrl=q * column_norms,
            s=s, keep=3, grid_n=128, laser_nm=635.0,
            saved_utc="", source="simulated")
        np.testing.assert_allclose(
            matrix.curvature, s ** 2 * column_norms ** 2)

    def test_flat_response_leaves_a_mode_alone(self):
        """A non-convex probe triple must yield no correction, not a divide."""
        mirror = Mirror([0.5, 0.0, 0.0], [4.0, 2.5, 1.0])
        opt = MODAL.ModalFit(settings(S.ALGO_MODAL_FIT), mirror.stored_matrix())
        opt._m = {"base": 10.0, (0, +1): 10.0, (0, -1): 10.0,
                  (1, +1): 10.0, (1, -1): 10.0, (2, +1): 10.0, (2, -1): 10.0}
        self.assertTrue(np.all(np.asarray(opt._solve()) == 0.0))
        self.assertTrue(opt.notes)


class StagedTest(unittest.TestCase):
    """The solve must hand over to a search without losing its result."""

    def test_hands_over_and_keeps_improving(self):
        mirror = Mirror([0.8, -1.1, 0.45], [4.0, 2.5, 1.0])
        cfg = settings(S.ALGO_MODAL_FIT,
                       polish_algorithm=S.ALGO_HILL,
                       polish_metric=S.METRIC_PIB)
        opt = make_optimizer(cfg, mirror.stored_matrix())
        self.assertIsInstance(opt, Staged)
        self.assertEqual(opt.metric, S.METRIC_SECOND_MOMENT)
        # Run only until the modal stage converges and hands over.
        for _ in range(400):
            cmd = opt.ask()
            bits = np.array([cmd[c] for c in CHANNELS], float)
            reading = mirror.reading(bits)
            opt.observe(reading)
            opt.tell(1.0 / (1.0 + reading.second_moment), valid=True)
            if opt.handed_over:
                break
        self.assertTrue(opt.handed_over)
        self.assertEqual(opt.metric, S.METRIC_PIB)
        self.assertTrue(opt.status()["stage"].startswith("polish/"))
        # The search must start from where the solve finished, not from scratch.
        handed = np.array([opt.ask()[c] for c in CHANNELS], float)
        self.assertLess(float(np.abs(mirror.coefficients(handed)).max()), 0.05)

    def test_no_polish_returns_the_bare_solver(self):
        mirror = Mirror([0.5, 0.0, 0.0], [4.0, 2.5, 1.0])
        opt = make_optimizer(settings(S.ALGO_MODAL_FIT), mirror.stored_matrix())
        self.assertIsInstance(opt, MODAL.ModalFit)


class SaturatingMirror(Mirror):
    """Quadratic only up to `linear_to` rad; beyond that the metric saturates.

    Stands in for the real failure the calibration exists to catch: an aperture
    that stops holding the aberrated spot, so the second moment climbs less than
    the quadratic model says and a large probe reads a curvature that is not
    there.
    """

    def __init__(self, aberration, curvature, linear_to, seed=0):
        super().__init__(aberration, curvature, seed)
        self.linear_to = float(linear_to)

    def second_moment(self, bits):
        a = self.coefficients(bits)
        capped = np.clip(np.abs(a), None, self.linear_to) * np.sign(a)
        return FLOOR + float((self.curvature * capped ** 2).sum())


class ProbeCalibrationTest(unittest.TestCase):
    """The sweep must choose an amplitude instead of trusting the setting."""

    CURVATURE = np.array([4.0, 2.5, 1.0])

    def _solver(self, mirror, **kw):
        cfg = settings(S.ALGO_MODAL_FIT, modal_auto_bias=True, **kw)
        return MODAL.ModalFit(cfg, mirror.stored_matrix())

    def test_a_clean_mirror_keeps_the_full_amplitude(self):
        """Nothing to fix, so the sweep must not shrink the probe."""
        mirror = Mirror([0.4, -0.3, 0.2], self.CURVATURE)
        opt = self._solver(mirror, modal_bias_rad=1.0)
        run(opt, mirror)
        self.assertAlmostEqual(opt.bias, 1.0, places=6)
        self.assertTrue(all(e["usable"] for e in opt.bias_report
                            if e["residual_frac"] is not None))

    def test_a_saturating_mirror_forces_a_smaller_amplitude(self):
        """The whole point: detect that the large probe left the model."""
        mirror = SaturatingMirror([0.3, -0.2, 0.1], self.CURVATURE,
                                  linear_to=0.6)
        opt = self._solver(mirror, modal_bias_rad=2.0)
        run(opt, mirror)
        self.assertLess(opt.bias, 2.0)
        self.assertIsNotNone(opt.bias_scanned)
        self.assertTrue(any("reduced the amplitude" in n for n in opt.notes))

    def test_the_chosen_amplitude_still_solves(self):
        """A calibrated run must correct at least as well as a lucky one."""
        mirror = SaturatingMirror([0.3, -0.2, 0.1], self.CURVATURE,
                                  linear_to=0.6)
        opt = self._solver(mirror, modal_bias_rad=2.0)
        residual = mirror.coefficients(run(opt, mirror))
        self.assertLess(float(np.abs(residual).max()), 0.05)

    def test_every_candidate_is_recorded(self):
        """The record has to show what was rejected, not just what was kept."""
        mirror = Mirror([0.4, -0.3, 0.2], self.CURVATURE)
        opt = self._solver(mirror, modal_bias_rad=1.0)
        run(opt, mirror)
        fractions = [e["fraction"] for e in opt.bias_report]
        self.assertEqual(fractions, sorted(S.LoopSettings().modal_bias_ladder))
        for entry in opt.bias_report:
            self.assertIn("usable", entry)
            self.assertIn("amplitude_rad", entry)

    def test_three_point_candidates_are_not_judged(self):
        """A parabola through 3 points fits exactly and proves nothing."""
        mirror = Mirror([0.4, -0.3, 0.2], self.CURVATURE)
        opt = self._solver(mirror, modal_bias_rad=1.0)
        run(opt, mirror)
        smallest = opt.bias_report[0]
        self.assertEqual(smallest["points"], 3)
        self.assertFalse(smallest["usable"])
        self.assertIn("too few points", smallest["reason"])

    def test_disabling_it_uses_the_typed_value(self):
        mirror = SaturatingMirror([0.3, -0.2, 0.1], self.CURVATURE,
                                  linear_to=0.6)
        cfg = settings(S.ALGO_MODAL_FIT, modal_auto_bias=False,
                       modal_bias_rad=2.0)
        opt = MODAL.ModalFit(cfg, mirror.stored_matrix())
        run(opt, mirror)
        self.assertAlmostEqual(opt.bias, 2.0, places=6)
        self.assertIsNone(opt.bias_scanned)


class SofteningMirror(Mirror):
    """Quadratic to `knee` rad, then linear with the same slope.

    The five-actuator mirror's measured sweep: the metric kept moving at every
    amplitude -- so the outer rungs carry far more signal than the inner ones
    -- but it stopped being a parabola. `SaturatingMirror` cannot stand in for
    that case: a flat tail gives every rung the same swing, so no amount of
    noise can separate them.
    """

    def __init__(self, aberration, curvature, knee, seed=0):
        super().__init__(aberration, curvature, seed)
        self.knee = float(knee)

    def second_moment(self, bits):
        a = np.abs(self.coefficients(bits))
        k = self.knee
        soft = np.where(a <= k, a ** 2, k ** 2 + 2.0 * k * (a - k))
        return FLOOR + float((self.curvature * soft).sum())


class WeakResponseTest(unittest.TestCase):
    """The sweep must not adopt an amplitude whose response is in the noise.

    The five-actuator bench mirror in the 2026-08-16 dual-mirror run: every
    rung fitted a parabola, the sweep took the smallest, and the probes it
    sized moved the score by about three sigma. Fitting a parabola through
    noise is not a failure the residual test can see -- four scattered points
    admit one, convex half the time -- so signal strength is judged separately.
    """

    # A tenth of the other tests': one radian of mode 1 barely moves the
    # metric, which is what leaves the small rungs inside the noise.
    CURVATURE = np.array([0.4, 0.25, 0.1])

    def _swept(self, noise_hint):
        """Sweep one softening mirror at a stated measurement noise.

        The knee is inside the ladder, so the outer rungs leave the parabola
        while carrying the most signal -- the amplitude then has to be chosen
        by weighing fit against noise, which is the whole point.
        """
        mirror = SofteningMirror([0.4, -0.3, 0.2], self.CURVATURE, knee=0.2)
        cfg = settings(S.ALGO_MODAL_FIT, modal_auto_bias=True,
                       modal_bias_rad=1.0)  # One radian, so bias == fraction.
        opt = MODAL.ModalFit(cfg, mirror.stored_matrix())
        # The loop feeds this in from its split-half estimate; the simulated
        # mirror is noiseless, so the size of the noise is stated instead of
        # sampled -- the rule under test reads the estimate, not the scatter.
        opt.noise_hint = float(noise_hint)
        run(opt, mirror)
        return opt

    def test_every_rung_records_its_signal_to_noise(self):
        opt = self._swept(0.0)
        for entry in opt.bias_report:
            self.assertIn("resolved", entry)
            self.assertGreater(entry["snr"], 0.0)
            self.assertEqual(entry["noise"], opt.bias_noise)

    def test_a_quiet_bench_takes_the_largest_rung_that_fits(self):
        """Unchanged behaviour when every rung is far above the noise."""
        opt = self._swept(0.0)
        self.assertTrue(all(e["resolved"] for e in opt.bias_report))
        self.assertAlmostEqual(opt.bias, 0.5, places=6)

    def test_the_amplitude_is_not_chosen_from_inside_the_noise(self):
        """The fix: the rung that fits best is skipped when it has no signal."""
        opt = self._swept(6e-6)
        fitted = [e for e in opt.bias_report if e["usable"]]
        self.assertTrue(fitted, "expected a rung that still fits a parabola")
        self.assertFalse(any(e["resolved"] for e in fitted),
                         "test noise too small to bite")
        # The rung the old rule would have taken, skipped for one with signal.
        self.assertGreater(opt.bias, max(e["fraction"] for e in fitted))
        self.assertTrue(any("cleared the noise" in n for n in opt.notes))

    def test_a_sweep_entirely_in_the_noise_takes_the_largest_rung(self):
        """Nothing to retreat to, so the only chance is the biggest probe."""
        opt = self._swept(1.2e-5)
        self.assertFalse(any(e["resolved"] for e in opt.bias_report))
        self.assertAlmostEqual(opt.bias, 1.0, places=6)
        self.assertTrue(any("measurement noise" in n for n in opt.notes))


class RailTest(unittest.TestCase):
    """The solve must not strand itself against an actuator limit.

    Observed on the bench: one round jumped several channels to 0 or 4095, the
    score collapsed, and it stayed there for every remaining round -- only the
    handover to the polish search, which restarts from `best_command`, brought
    it back.
    """

    CURVATURE = np.array([4.0, 2.5, 1.0])

    def _solver(self, **kw):
        mirror = Mirror([0.5, -0.4, 0.2], self.CURVATURE)
        cfg = settings(S.ALGO_MODAL_FIT, **kw)
        return MODAL.ModalFit(cfg, mirror.stored_matrix()), mirror

    def test_a_symmetric_probe_at_a_rail_is_reported_unmeasurable(self):
        # Pretending a rail-clipped command was a full modal probe is worse than
        # skipping it: the resulting coefficient belongs to a different shape.
        opt, _ = self._solver()
        opt.x0 = np.array(opt.hi, float)  # Hard against the top limit.
        for mode in range(opt.n_modes):
            opt._realized.clear()
            opt._bits(mode, +1)
            self.assertEqual(opt._effective_amp(mode, +1), 0.0)

    def test_an_overshooting_correction_is_scaled_not_clipped(self):
        # Clipping per actuator leaves a shape that is no longer a combination
        # of the calibrated modes; scaling keeps the direction.
        opt, _ = self._solver()
        step = -(opt.ctrl[:, :opt.n_modes] @ np.full(opt.n_modes, 50.0))
        fit = opt._fit_step(step)
        moved = fit * step
        self.assertLess(fit, 1.0)
        self.assertTrue(np.all(opt.x0 + moved >= opt.lo - 0.5))
        self.assertTrue(np.all(opt.x0 + moved <= opt.hi + 0.5))
        # Parallel to the requested correction, i.e. still the same shape.
        cos = float(moved @ step / (np.linalg.norm(moved)
                                    * np.linalg.norm(step)))
        self.assertAlmostEqual(cos, 1.0, places=9)

    def test_the_working_point_keeps_room_to_probe(self):
        opt, _ = self._solver()
        opt.x0 = np.array(opt.hi, float) - 1.0
        opt._m = {"base": 1.0}
        for k in opt._plan:
            opt._m[k] = 1.0
        opt._apply()
        room = np.minimum(opt.x0 - opt.lo, opt.hi - opt.x0)
        self.assertTrue(np.all(room > 0), f"no headroom left: {opt.x0}")

    def test_a_shallow_parabola_cannot_demand_a_huge_jump(self):
        # Curvature near zero is noise; dividing by it put the vertex tens of
        # probe amplitudes away, which is what reached the rail. Barely convex
        # (1e-6) with a large slope is exactly that case.
        opt, _ = self._solver()
        a = opt._vertex(1.0, 1.5, 0.500001, 0)
        self.assertLessEqual(
            abs(a), MODAL._VERTEX_REACH * opt._symmetric_amp(0) + 1e-9)
        self.assertTrue(any("probe amplitudes out" in n for n in opt.notes))

    def test_a_solve_started_near_a_rail_escapes_it(self):
        # 100 bits of travel left is the worst case `_fit_step` can now leave.
        # It cannot correct such a start in four rounds -- each is bounded by
        # the vertex clamp -- but it must make real progress, not sit there.
        mirror = Mirror([0.5, -0.4, 0.2], self.CURVATURE)
        cfg = settings(S.ALGO_MODAL_FIT, modal_rounds=4)
        opt = MODAL.ModalFit(cfg, mirror.stored_matrix())
        near = np.array(opt.hi, float) - 100.0
        opt.x0 = near.copy()
        opt._pending = opt._quant(opt.x0)
        before = float(np.abs(mirror.coefficients(near)).max())
        after = float(np.abs(mirror.coefficients(run(opt, mirror))).max())
        self.assertLess(after, before / 2.0,
                        f"stuck at the rail: {before:.2f} -> {after:.2f} rad")
        self.assertTrue(any("probe cut" in n for n in opt.notes))


class RetreatTest(unittest.TestCase):
    """A round that made the spot worse must not seed the next one.

    Observed on the bench: the working point walked out to the ends of the
    range and stayed there for whole rounds, because `_apply` moves it to
    whatever the solve computed and the next round fits its parabola about
    that point, however bad it is.
    """

    CURVATURE = np.array([4.0, 2.5, 1.0])

    def _solver(self):
        mirror = Mirror([0.4, -0.3, 0.15], self.CURVATURE)
        cfg = settings(S.ALGO_MODAL_FIT, modal_rounds=3)
        return MODAL.ModalFit(cfg, mirror.stored_matrix()), mirror

    def _at_verify(self, opt, accepted, candidate):
        """Put a solver at the point where a computed correction is judged."""
        opt.accepted_x = opt._quant(np.full(len(CHANNELS), accepted))
        opt.accepted_score = 0.9
        opt.accepted_value = 100.0
        opt.best_x = opt.accepted_x.copy()
        opt.best_score = 0.9
        opt.x0 = np.full(len(CHANNELS), candidate)
        opt._pending = opt._quant(opt.x0)
        opt.stage = "verify"

    def test_a_worse_correction_is_abandoned(self):
        opt, _ = self._solver()
        self._at_verify(opt, 2000.0, 3500.0)
        opt.tell(0.2, valid=True)  # Far below accepted: a real step back.
        np.testing.assert_allclose(opt.x0, opt.accepted_x)
        self.assertEqual(opt.stage, "recenter")
        self.assertTrue(any("worse than the" in n for n in opt.notes))

    def test_a_correction_within_noise_of_accepted_is_kept(self):
        opt, _ = self._solver()
        opt.set_noise(0.01)
        moved = 2100.0
        self._at_verify(opt, 2000.0, moved)
        opt.tell(0.895, valid=True)  # Not an improvement, but not a step back.
        np.testing.assert_allclose(opt.accepted_x,
                                   opt._quant(np.full(len(CHANNELS), moved)))
        self.assertEqual(opt.stage, "probe")

    def test_the_solve_still_converges_with_the_guard(self):
        mirror = Mirror([0.7, -0.5, 0.3], self.CURVATURE)
        cfg = settings(S.ALGO_MODAL_FIT, modal_rounds=3)
        opt = MODAL.ModalFit(cfg, mirror.stored_matrix())
        residual = mirror.coefficients(run(opt, mirror))
        self.assertLess(float(np.abs(residual).max()), 0.05,
                        f"guard broke the solve: {residual}")


class CurvatureGateTest(unittest.TestCase):
    """A curvature no better resolved than the noise must not be divided by.

    `M+ - 2 M0 + M-` is a second difference, so noise on it is sqrt(6) times
    that of one reading. Where the true curvature sits at that level its SIGN
    is a coin toss: non-convex skips the mode, barely-positive sends the vertex
    far outside the probes. Only the second face moves the mirror, which is why
    the runaway showed up in some runs and not others.
    """

    CURVATURE = np.array([4.0, 2.5, 1.0])

    def _solver(self, noise):
        mirror = Mirror([0.5, -0.4, 0.2], self.CURVATURE)
        opt = MODAL.ModalFit(settings(S.ALGO_MODAL_FIT),
                             mirror.stored_matrix())
        opt.value_noise = noise
        return opt

    def test_a_curvature_inside_the_noise_is_refused(self):
        opt = self._solver(1.0)
        # Curvature 2.0, well under 3*sqrt(6)*1.0; slope large, so the old test
        # (curvature > 0) would have returned a huge vertex.
        self.assertEqual(opt._vertex(100.0, 151.0, 51.0, 0), 0.0)
        self.assertTrue(any("not above its own noise" in n for n in opt.notes))

    def test_a_well_resolved_curvature_still_solves(self):
        opt = self._solver(0.01)
        a = opt._vertex(100.0, 151.0, 51.0, 0)
        self.assertNotEqual(a, 0.0)

    def test_a_non_convex_response_is_refused_at_any_floor(self):
        opt = self._solver(0.0)
        self.assertEqual(opt._vertex(100.0, 99.0, 99.0, 0), 0.0)

    def test_the_sweep_records_a_residual_for_every_judged_rung(self):
        # The sweep no longer sets the gate -- see GateFloorTest -- but its
        # residuals are still the record of which amplitude was believable.
        mirror = Mirror([0.5, -0.4, 0.2], self.CURVATURE)
        cfg = settings(S.ALGO_MODAL_FIT, modal_auto_bias=True,
                       modal_bias_rad=1.0)
        opt = MODAL.ModalFit(cfg, mirror.stored_matrix())
        run(opt, mirror)
        self.assertTrue(any(e.get("resid_rms") is not None
                            for e in opt.bias_report))


class QuantizedProbeTest(unittest.TestCase):
    """Modal probes must survive command quantisation as the same mode."""

    CTRL = np.array([
        [-154.34, -18.68, -35.65, -124.15],
        [32.42, 96.91, 337.40, -1498.93],
        [47.71, -97.39, -1116.30, -569.22],
        [27.51, 90.96, -447.40, 1382.72],
        [43.36, -89.49, 1132.88, 427.68],
    ])
    SINGULAR_VALUES = np.array([0.041052, 0.029322, 0.006034, 0.005519])

    def _matrix(self):
        return ST.StoredMatrix(
            path=None, channels=CHANNELS, ctrl=self.CTRL,
            s=self.SINGULAR_VALUES, keep=4, grid_n=128,
            laser_nm=520.0, saved_utc="", source="bench fixture")

    def _solver(self, **kwargs):
        params = dict(min_step=50, modal_bit_step=1, modal_bias_rad=0.25)
        params.update(kwargs)
        cfg = settings(S.ALGO_MODAL_FAST, **params)
        return MODAL.ModalFast(cfg, self._matrix())

    def test_modal_bit_grid_is_independent_of_search_min_step(self):
        opt = self._solver()
        base = np.array(list(opt.ask().values()))
        probe = opt._bits(1, +1)
        self.assertFalse(np.array_equal(probe, base))
        self.assertGreater(opt._effective_amp(1, +1), 0.0)

    def test_fast_solver_equalises_predicted_probe_signal(self):
        opt = self._solver(modal_bias_rad=0.5)
        amplitudes = np.array([opt._requested_amp(i)
                               for i in range(opt.n_modes)])
        curvature = opt.matrix.curvature[:opt.n_modes]
        predicted = curvature * amplitudes ** 2
        np.testing.assert_allclose(predicted, predicted[0], rtol=1e-12)

    def test_roi_miss_retries_only_the_current_probe_at_half_scale(self):
        opt = self._solver(modal_bias_rad=0.5)
        opt.stage = "probe"
        opt._k = 3  # Mode 3 positive probe in the N+2 plan.
        opt._pending = opt._bits(*opt._plan[opt._k])
        before = np.linalg.norm(opt._pending - opt._quant(opt.x0))
        result = opt.recover_roi_miss()
        after = np.linalg.norm(opt._pending - opt._quant(opt.x0))
        self.assertEqual(result["action"], "retry_probe")
        self.assertGreater(after, 0.0)
        self.assertLess(after, before)

    def test_halving_a_negative_probe_remeasures_its_positive_pair(self):
        opt = self._solver(modal_bias_rad=0.5)
        opt.stage = "probe"
        opt._k = 1  # Mode 1 negative; its positive partner was measured first.
        opt._m[(0, +1)] = 123.0
        opt._pending = opt._bits(*opt._plan[opt._k])
        result = opt.recover_roi_miss()
        self.assertEqual(result["action"], "retry_probe")
        self.assertEqual(opt._k, 0)
        self.assertNotIn((0, +1), opt._m)
        np.testing.assert_array_equal(opt._pending, opt._bits(0, +1))

    def test_roi_miss_halves_a_candidate_correction(self):
        opt = self._solver()
        origin = opt.x0.copy()
        step = np.array([200.0, 600.0, 600.0, -550.0, -650.0])
        opt.stage = "base"
        opt._correction_origin = origin.copy()
        opt._correction_step = step.copy()
        opt._correction_scale = 1.0
        opt.x0 = opt._quant(origin + step)
        opt._pending = opt.x0.copy()
        result = opt.recover_roi_miss()
        self.assertEqual(result["action"], "retry_correction")
        np.testing.assert_allclose(opt._pending, opt._quant(origin + 0.5 * step))


def _cliff_matrix(costs):
    """Stored matrix with a singular-value cliff after mode 2.

    Args:
        costs: Bit-per-radian scale of each mode's control column.

    Returns:
        A `StoredMatrix` whose modes are orthogonal but priced differently.
    """
    n = len(CHANNELS)
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    ctrl = q[:, :3] * np.asarray(costs, float)
    return ST.StoredMatrix(
        path=None, channels=CHANNELS, ctrl=ctrl,
        s=np.array([0.04, 0.03, 0.0025]),  # 12x cliff after mode 2.
        keep=3, grid_n=128, laser_nm=635.0, saved_utc="", source="simulated")


class AutoTrustTest(unittest.TestCase):
    """Automatic rounds may narrow a probe, never dissolve one.

    Curvature scales with the square of the probe amplitude while the gate it
    is judged against does not, so an unbounded shrink is self-reinforcing:
    every halving makes the same mode four times harder to resolve, and a mode
    that quantises to nothing is silently dropped from the plan.
    """

    CURVATURE = np.array([4.0, 2.5, 1.0])

    def _solver(self):
        mirror = Mirror([0.5, -0.4, 0.2], self.CURVATURE)
        cfg = settings(S.ALGO_MODAL_FIT, modal_auto_rounds=True,
                       modal_rounds=8, modal_bias_rad=1.0)
        opt = MODAL.ModalFit(cfg, mirror.stored_matrix())
        opt.last_coefficients = [0.3, -0.2, 0.1]
        return opt

    def test_probe_trust_never_quantises_a_mode_out_of_the_plan(self):
        opt = self._solver()
        for _ in range(12):  # Far more shrinks than a run could ask for.
            opt._shrink_auto_trust("simulated non-quiet round")
        for mode in range(opt.n_modes):
            opt._realized.clear()
            self.assertNotEqual(
                opt._effective_amp(mode, +1.0), 0.0,
                f"mode {mode + 1} probe vanished on the command grid")

    def test_an_accepted_round_restores_probe_trust(self):
        opt = self._solver()
        opt._shrink_auto_trust("simulated non-quiet round")
        self.assertLess(float(np.min(opt._trust_scale)), 1.0)
        opt._restore_auto_trust()
        np.testing.assert_allclose(opt._trust_scale, 1.0)

    def test_roi_recovery_and_auto_trust_do_not_share_state(self):
        # Two independent causes; one shared variable would let a footprint
        # problem and a model-mismatch problem multiply into each other.
        opt = self._solver()
        opt._shrink_auto_trust("simulated non-quiet round")
        np.testing.assert_allclose(opt._probe_scale, 1.0)


class ModeResolverTest(unittest.TestCase):
    """The control basis stops where the mirror stops being controllable."""

    def _solver(self, matrix, **kw):
        cfg = settings(S.ALGO_MODAL_FIT, **kw)
        return MODAL.ModalFit(cfg, matrix)

    def test_a_cliff_with_a_control_cost_jump_truncates_the_basis(self):
        opt = self._solver(_cliff_matrix([200.0, 200.0, 6000.0]))
        self.assertEqual(opt.n_modes, 2)
        self.assertIn("cliff", " ".join(opt.mode_resolution["reasons"]))

    def test_a_cliff_without_a_cost_jump_is_retained(self):
        # A bare singular-value step can be a stale scale calibration, which
        # the 2N+1 solve re-measures anyway.
        opt = self._solver(_cliff_matrix([200.0, 200.0, 200.0]))
        self.assertEqual(opt.n_modes, 3)

    def test_an_operator_mode_count_still_wins(self):
        opt = self._solver(_cliff_matrix([200.0, 200.0, 6000.0]),
                           modal_modes=3)
        self.assertEqual(opt.n_modes, 3)
        self.assertEqual(opt.mode_resolution["source"], "operator")

    def test_the_headroom_test_records_the_amplitude_it_judged(self):
        # Headroom is judged against the REQUESTED amplitude, which the probe
        # sweep may later reduce; the record has to say which number was used.
        opt = self._solver(_cliff_matrix([200.0, 200.0, 200.0]),
                           modal_bias_rad=1.0)
        self.assertEqual(opt.mode_resolution["headroom_judged_at_rad"], 1.0)


class SweepRefusalTest(unittest.TestCase):
    """An unidentifiable perturbation must stop the run, not pick a rung."""

    def _mirror(self):
        # Saturates below the smallest guard rung, so no amplitude anywhere in
        # the sweep sees a convex quadratic.
        return SaturatingMirror([0.3, -0.2, 0.1], np.array([4.0, 2.5, 1.0]),
                                linear_to=1e-4)

    def test_a_sweep_that_identifies_nothing_refuses(self):
        mirror = self._mirror()
        cfg = settings(S.ALGO_MODAL_FIT, modal_auto_bias=True,
                       modal_bias_rad=1.0)
        opt = MODAL.ModalFit(cfg, mirror.stored_matrix())
        run(opt, mirror)
        self.assertIsNotNone(opt.status()["failed_reason"])
        self.assertIsNone(opt.bias_scanned)

    def test_a_refused_solve_does_not_hand_over_to_polish(self):
        mirror = self._mirror()
        cfg = settings(S.ALGO_MODAL_FIT, modal_auto_bias=True,
                       modal_bias_rad=1.0, polish_algorithm=S.ALGO_HILL,
                       polish_metric=S.METRIC_PIB)
        staged = make_optimizer(cfg, mirror.stored_matrix())
        run(staged, mirror)
        self.assertFalse(staged.handed_over)
        self.assertIsNotNone(staged.status()["failed_reason"])


class GateFloorTest(unittest.TestCase):
    """The curvature gate must never be able to switch itself off."""

    CURVATURE = np.array([4.0, 2.5, 1.0])

    def test_the_floor_is_strictly_positive_without_a_noise_estimate(self):
        mirror = Mirror([0.5, -0.4, 0.2], self.CURVATURE)
        opt = MODAL.ModalFit(settings(S.ALGO_MODAL_FIT),
                             mirror.stored_matrix())
        opt.set_noise(0.0)  # Nothing measured yet.
        opt._update_value_noise(0.5, 100.0)
        self.assertGreater(opt.value_noise, 0.0)

    def test_the_floor_tracks_the_metric_instead_of_freezing(self):
        mirror = Mirror([0.5, -0.4, 0.2], self.CURVATURE)
        opt = MODAL.ModalFit(settings(S.ALGO_MODAL_FIT),
                             mirror.stored_matrix())
        opt.set_noise(1e-3)
        opt._update_value_noise(0.5, 1000.0)
        far = opt.value_noise
        opt._update_value_noise(0.5, 100.0)
        self.assertLess(opt.value_noise, far)

    def test_a_run_records_one_floor_per_round(self):
        mirror = Mirror([0.5, -0.4, 0.2], self.CURVATURE)
        cfg = settings(S.ALGO_MODAL_FIT, modal_rounds=2)
        opt = MODAL.ModalFit(cfg, mirror.stored_matrix())
        run(opt, mirror)
        self.assertTrue(opt.value_noise_by_round)
        self.assertTrue(all(e["used"] > 0 for e in opt.value_noise_by_round))


class TravelBudgetTest(unittest.TestCase):
    """A probe plan must not walk the mirror further than it can be trusted.

    The nine-actuator bench run that motivated this drove 59835 bits per
    channel because one retained mode cost 5498 bit/rad; the excursion alone
    made the mirror irreproducible at the level of the signal being measured.
    """

    def test_total_commanded_travel_stays_bounded(self):
        mirror = Mirror([0.5, -0.4, 0.2], np.array([4.0, 2.5, 1.0]))
        cfg = settings(S.ALGO_MODAL_FIT, modal_rounds=3)
        opt = MODAL.ModalFit(cfg, mirror.stored_matrix())
        span = float(np.max(opt.hi - opt.lo))
        travel = np.zeros(len(CHANNELS))
        previous = None
        for _ in range(400):
            cmd = opt.ask()
            bits = np.array([cmd[c] for c in CHANNELS], float)
            if previous is not None:
                travel += np.abs(bits - previous)
            previous = bits
            reading = mirror.reading(bits)
            opt.observe(reading)
            opt.tell(1.0 / (1.0 + reading.second_moment), valid=True)
            if opt.status()["converged"]:
                break
        # Three rounds of 2N+1 on three modes is 18 probes plus overhead; even
        # at a full probe each way that is far below ten full sweeps of range.
        self.assertLess(float(np.max(travel)), 10 * span)


if __name__ == "__main__":
    unittest.main()
