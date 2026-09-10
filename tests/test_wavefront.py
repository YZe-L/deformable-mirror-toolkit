# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-08-24

"""Focal-plane wavefront sensing: unit chain, then the closed loop.

The unit chain is checked against `tools/correctability.py`, which does the
same waves/bits arithmetic offline. If the two ever disagree, one of them
is wrong about the factor 2 for reflection or about radians versus waves.

The loop is then driven against a SIMULATED mirror whose influence matrix is
deliberately not the one the controller holds, because that mismatch -- the
matrix being a local slope at the bias while a correction runs ~870 bit -- is
the reason this controller iterates instead of firing once. A test on a
perfect plant would pass without exercising the only hard part.
"""

from __future__ import annotations

import numpy as np
import pytest

from dm_toolkit.phase_retrieval import Optics
from dm_toolkit.correction import settings as S
from dm_toolkit.correction.metrics import SpotReading
from dm_toolkit.correction.wavefront import FIT_MODES, PlantError, PseudoWfs, _Plant

NPIX = 40  # Pupil pixels across the synthetic aperture.


def _synthetic_plant(n_act=9, keep=6, gain=1.0, seed=3):
    """A plant with smooth, overlapping actuator bumps -- a membrane mirror.

    Args:
        n_act: Actuators.
        keep: Eigenmodes the controller may use.
        gain: Scales waves-per-bit, so a controller built at 1.0 and a mirror
            built at 1.2 disagree exactly as a local slope does.
        seed: Actuator placement.

    Returns:
        A `_Plant`.
    """
    from dm_toolkit import zernike as ZK

    rng = np.random.default_rng(seed)
    ax = (np.arange(NPIX) - (NPIX - 1) / 2.0) / (NPIX / 2.0)
    x, y = np.meshgrid(ax, ax)
    inside = np.hypot(x, y) <= 1.0
    xs, ys = x[inside], y[inside]
    centres = rng.uniform(-0.7, 0.7, size=(n_act, 2))
    bumps = np.stack([np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / 0.25)
                      for cx, cy in centres], axis=1)
    bumps -= bumps.mean(axis=0)  # Piston is not observable, as on the bench.
    waves_per_bit = gain * 3e-4 * bumps

    # The same construction the pipeline uses: eigenmodes of the influence
    # matrix, each column rescaled to "bits for one unit of modal amplitude".
    _u, _s, vt = np.linalg.svd(waves_per_bit, full_matrices=False)
    ctrl = vt[:keep].T.copy()
    for i in range(keep):
        column = waves_per_bit @ ctrl[:, i]
        ctrl[:, i] /= np.sqrt(np.mean(column ** 2))

    rho, theta, _ys, _xs = ZK.unit_coords(inside, ((NPIX - 1) / 2.0,) * 3)
    basis = np.vstack([ZK.zernike_mode(j, rho, theta) for j in FIT_MODES])
    return _Plant(waves_per_bit, ctrl, basis, keep, "synthetic")


def test_unit_chain_matches_the_offline_tool():
    """`_Plant` and `correctability.split` must give the same floor."""
    from scipy.optimize import lsq_linear

    plant = _synthetic_plant()
    coeffs = np.array([0.30, 0.55, -0.12, 0.08, -0.05, 0.10, 0.04, 0.06])

    bits, _amps = plant.command(coeffs)
    mine = plant.residual_waves(coeffs, bits)

    # correctability.split's arithmetic, written out so the test does not
    # simply call the code it is checking.
    wavefront = coeffs @ plant.basis
    theirs_amps = lsq_linear(plant.waves_per_bit @ plant.modal,
                             -wavefront, bounds=(-np.inf, np.inf)).x
    theirs = float(np.sqrt(np.mean(
        (wavefront + plant.waves_per_bit @ (plant.modal @ theirs_amps)) ** 2)))
    assert mine == pytest.approx(theirs, rel=1e-9)
    # And it must actually remove something, or the test proves nothing.
    assert mine < 0.8 * float(np.sqrt(np.mean(wavefront ** 2)))


def test_correction_never_leaves_the_controllable_subspace():
    """The commanded wavefront lies in the span of the retained eigenmodes.

    This is the property that distinguishes the controller from a scalar
    search: over the recorded runs the searches raise the uncorrectable floor
    from 0.173 to 0.223 waves, and this is why this one cannot.
    """
    plant = _synthetic_plant()
    coeffs = np.array([0.4, -0.3, 0.2, 0.15, 0.1, -0.2, 0.05, 0.12])
    bits, amps = plant.command(coeffs)
    commanded = plant.waves_per_bit @ bits
    span = plant.waves_per_bit @ plant.modal
    # Projecting onto the span and back must change nothing.
    fit, *_ = np.linalg.lstsq(span, commanded, rcond=None)
    assert np.allclose(span @ fit, commanded, atol=1e-12)
    assert fit == pytest.approx(amps, rel=1e-6, abs=1e-9)


def test_load_refuses_a_matrix_of_the_wrong_size(tmp_path):
    path = tmp_path / "influence_matrix.npz"
    np.savez(path, c_rad_per_bit=np.zeros((10, 4)), ctrl=np.zeros((4, 4)),
             inside=np.ones((4, 4), bool), grid_n=4, channels=np.arange(4))
    with pytest.raises(PlantError, match="4 actuators"):
        _Plant.load(path, [1, 2, 3, 4, 5])


def test_verify_against_tolerates_a_sign_flip_but_not_a_permutation():
    """Singular-vector signs are arbitrary; actuator order is not."""
    plant = _synthetic_plant()
    flipped = plant.modal.copy()
    flipped[:, 1] *= -1.0
    assert "confirmed" in plant.verify_against(flipped)
    permuted = plant.modal.copy()
    permuted[[0, 1]] = permuted[[1, 0]]
    with pytest.raises(PlantError, match="wrong piezos"):
        plant.verify_against(permuted)


def _reachable_truth(plant, amps, stray=0.1, bit_budget=600.0):
    """A wavefront this plant can mostly undo, plus a little it cannot.

    Built FROM the plant rather than drawn at random, because a random Noll
    vector is mostly outside the reach of nine smooth actuators and the test
    would then be measuring the fixture's correctability rather than the
    controller. The `stray` term keeps the floor non-zero, so "lands short"
    stays a meaningful thing to assert.

    The reachable part is then scaled so its ideal correction costs
    `bit_budget` on the busiest actuator. Without that the correction runs
    into the rails, an actuator pins, and the run stalls above the
    unconstrained floor for a reason that has nothing to do with what the
    test is about -- rails have `test_the_correction_stays_inside_every_
    actuator_range` to themselves.

    Args:
        plant: The mirror.
        amps: Modal amplitudes to build the reachable part from.
        stray: Trefoil the six modes cannot reach, as a FRACTION of the
            reachable part -- absolute would dominate whenever `bit_budget`
            shrinks the reachable part, and the fixture's correctability would
            then depend on a number chosen for a different scale.
        bit_budget: Bits the ideal correction may spend on one actuator.

    Returns:
        Noll 4-11 coefficients, in `FIT_MODES` order.
    """
    pupil = plant.waves_per_bit @ (plant.modal @ np.asarray(amps, float))
    got, *_ = np.linalg.lstsq(plant.basis.T, pupil, rcond=None)
    cost = float(np.max(np.abs(plant.command(got)[0])))
    if cost > 0:
        got = got * (bit_budget / cost)
    got[FIT_MODES.index(9)] += stray * float(np.sqrt((got ** 2).sum()))
    return got


# --- The loop ------------------------------------------------------------

class _Bench:
    """A mirror plus a camera the controller can be driven against.

    The controller's plant and the mirror's differ by `gain`, standing in for
    the influence matrix being a local slope. `sign` flips the even block of
    what the sensor reports, standing in for the twin ambiguity a single
    in-focus frame cannot resolve.
    """

    def __init__(self, truth, mirror, sensor, sign=1.0, noise=0.0, seed=0,
                 bias=2000.0):
        self.truth = np.asarray(truth, float)
        self.mirror = mirror
        self.sensor = sensor
        self.sign = float(sign)
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        self.even = np.array([j in (4, 5, 6, 11) for j in FIT_MODES])
        # `truth` is the aberration AT THE BIAS, and the influence matrix is a
        # slope about it, so only the offset from the bias moves the mirror.
        # Counting the bias itself as commanded wavefront would make the
        # starting spot an artefact of where the actuators happen to rest.
        self.bias = np.full(mirror.waves_per_bit.shape[1], float(bias))

    def wavefront(self, bits):
        """Pupil phase in waves with `bits` applied, over the sensor pixels."""
        offset = np.asarray(bits, float) - self.bias
        return self.truth @ self.mirror.basis + self.mirror.waves_per_bit @ offset

    def coeffs(self, bits):
        """What a perfect eight-mode fit would report for that wavefront."""
        got, *_ = np.linalg.lstsq(self.mirror.basis.T, self.wavefront(bits),
                                  rcond=None)
        got = np.where(self.even, self.sign * got, got)
        return got + self.rng.normal(0.0, self.noise, size=got.shape)

    def score(self, bits):
        """Monotone in spot quality: a Marechal Strehl of the true wavefront."""
        rms = float(np.sqrt(np.mean(self.wavefront(bits) ** 2)))
        return float(np.exp(-(2.0 * np.pi * rms) ** 2))


class _FakeEstimate:
    """The fields `PseudoWfs` reads out of a `WavefrontEstimate`."""

    def __init__(self, coeffs, residual=29.0):
        self.coeffs = {j: float(c) for j, c in zip(FIT_MODES, coeffs)}
        self.rms_waves = float(np.sqrt(sum(c * c for c in coeffs)))
        self.residual = float(residual)
        self.beyond_capture_range = False


def _drive(opt, bench, shots=40):
    """Run the ask/observe/tell loop, returning the true RMS at each point."""
    trail = []
    for _ in range(shots):
        bits = np.array([opt.ask()[a.channel] for a in opt.acts], float)
        trail.append(float(np.sqrt(np.mean(bench.wavefront(bits) ** 2))))
        reading = SpotReading(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0,
                              0.0, 0.0, True)
        # Stand in for the retrieval: `observe` is what would call it.
        opt.est = _FakeEstimate(bench.coeffs(bits))
        opt.tell(bench.score(bits), valid=True)
        if opt.converged:
            break
    return trail


def _settings(plant, rounds=6):
    acts = [S.Actuator(channel=i + 1, bit_min=0, bit_max=4095, start=2000)
            for i in range(plant.waves_per_bit.shape[1])]
    return S.LoopSettings(algorithm=S.ALGO_WFS, actuators=acts, min_step=1,
                          move_step=300, noise_threshold=1e-9,
                          wfs_rounds=rounds, wavelength_nm=632.8,
                          focal_mm=100.0, aperture_mm=5.0, pixel_um=3.45)


def test_an_exact_plant_is_corrected_to_its_floor_in_one_round():
    """The control law itself, with nothing else in the way.

    If the matrix is right, sense - project - drive has nothing left to do
    after one command: the remaining wavefront is exactly what six eigenmodes
    cannot reach. This is the test that would catch a wrong factor 2, a
    radians/waves slip, or a transposed matrix, because all three would leave
    a residual far above the floor.
    """
    plant = _synthetic_plant()
    truth = _reachable_truth(plant, [0.5, -0.4, 0.3, 0.2, -0.25, 0.15])
    floor = plant.residual_waves(truth, plant.command(truth)[0])

    opt = PseudoWfs(_settings(plant, rounds=1), plant=plant)
    trail = _drive(opt, _Bench(truth, plant, plant))
    assert trail[0] > 2 * floor          # There was something to correct,
    assert trail[-1] == pytest.approx(floor, abs=1e-4)   # and it is all gone.


def test_a_mismatched_matrix_lands_short_and_iterating_recovers_some_of_it():
    """Why the controller iterates instead of firing once.

    The influence matrix is a local slope at the bias while a real correction
    runs ~870 bit, and full-range superposition error on this mirror is 44%.
    Here the mirror is 25% stiffer than the controller believes.

    The gap does NOT close geometrically, and the test does not pretend it
    does: near the floor the correction is a few bits, so the command grid and
    the accept gate -- not the model error -- decide what happens, and the run
    settles a little above the floor. Measured on this fixture (start 0.081,
    floor 0.037), one round leaves 0.0042 waves of gap and six leave 0.0029.
    """
    controller = _synthetic_plant(gain=1.0)
    mirror = _synthetic_plant(gain=1.25)
    truth = _reachable_truth(controller, [0.5, -0.4, 0.3, 0.2, -0.25, 0.15])
    floor = controller.residual_waves(truth, controller.command(truth)[0])

    ends = {}
    for rounds in (1, 6):
        opt = PseudoWfs(_settings(controller, rounds), plant=controller)
        ends[rounds] = _drive(opt, _Bench(truth, mirror, controller))[-1]
    assert ends[1] - floor > 1e-3        # One round cannot reach the floor,
    assert ends[6] < ends[1]             # and iterating recovers part of it.


def test_the_twin_sign_is_resolved_by_measurement():
    """A sensor reporting the wrong even-block sign must not double the error.

    One in-focus frame cannot choose the sign, so the
    controller spends one round finding out which one improves the spot.
    """
    plant = _synthetic_plant()
    truth = _reachable_truth(plant, [0.4, -0.3, 0.25, 0.15, -0.2, 0.1])
    start = float(np.sqrt(np.mean((truth @ plant.basis) ** 2)))
    floor = plant.residual_waves(truth, plant.command(truth)[0])

    right = PseudoWfs(_settings(plant, rounds=6), plant=plant)
    reference = _drive(right, _Bench(truth, plant, plant))[-1]

    opt = PseudoWfs(_settings(plant, rounds=6), plant=plant)
    trail = _drive(opt, _Bench(truth, plant, plant, sign=-1.0))
    assert opt.sign == -1.0            # It found the twin,
    assert opt.sign_locked             # confirmed it by measurement,
    assert trail[-1] < start           # improved rather than doubling,
    # and ended up where the un-ambiguous run did: the sign cost a round, not
    # the correction. Compared against that run rather than a fixed fraction,
    # because how far either can get is a property of the plant.
    assert trail[-1] < 1.3 * max(reference, floor)


def test_a_collapsed_fit_is_discarded_rather_than_driven():
    """A residual jump means the fit stopped describing the spot."""
    plant = _synthetic_plant()
    bench = _Bench(np.array([0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                   plant, plant)
    opt = PseudoWfs(_settings(plant, rounds=6), plant=plant)

    bits = np.array([opt.ask()[a.channel] for a in opt.acts], float)
    opt.est = _FakeEstimate(bench.coeffs(bits), residual=29.0)
    opt.tell(bench.score(bits), valid=True)      # Seed sets the reference.
    accepted = opt.accepted_x.copy()

    bits = np.array([opt.ask()[a.channel] for a in opt.acts], float)
    # Same spot, but the fit fell apart. Its coefficients are not drivable.
    opt.est = _FakeEstimate(np.full(len(FIT_MODES), 5.0), residual=200.0)
    opt.tell(bench.score(bits) * 0.5, valid=True)
    assert any("discarded without driving" in n for n in opt.notes)
    assert np.array_equal(opt.accepted_x, accepted)


def test_the_correction_stays_inside_every_actuator_range():
    """Scaled as one vector, never clipped: a clipped shape is a new shape."""
    plant = _synthetic_plant()
    cfg = _settings(plant, rounds=4)
    # A pinned actuator leaves almost no room in one direction.
    cfg.actuators[0].bit_max = 2010
    opt = PseudoWfs(cfg, plant=plant)
    bench = _Bench(np.array([0.6, 0.7, -0.3, 0.2, -0.1, 0.25, 0.1, 0.15]),
                   plant, plant)
    _drive(opt, bench)
    assert np.all(opt.accepted_x >= opt.lo) and np.all(opt.accepted_x <= opt.hi)
    assert int(opt.accepted_x[0]) <= 2010


def test_handover_step_leaves_the_polish_room_to_search():
    """A warm-started hill climb must not converge before it has swept.

    It halves its step until `min_step`, so a handover already at that grid
    would report agreement it never tested.
    """
    plant = _synthetic_plant()
    cfg = _settings(plant, rounds=4)
    opt = PseudoWfs(cfg, plant=plant)
    _drive(opt, _Bench(np.array([0.3, 0.3, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
                       plant, plant))
    step = opt.handover_step()
    assert step >= 4 * cfg.min_step      # Two halvings of headroom,
    assert step <= max(cfg.min_step, cfg.move_step)   # but no cold-run leap.
