# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-08-24

"""Focal-plane wavefront sensing as the loop's controller.

The averaged frame of one measurement goes into `phase_retrieval`, the
retrieved Noll coefficients are projected onto the measured eigenmodes, and
the bounded least-squares solution is the command: one measurement per
round. The influence matrix is a local slope, so the controller iterates.
The even block's overall sign is unobservable, so a failed first correction
is retried with the twin sign; a round whose fit residual jumps is dropped.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from .. import zernike as ZK
from ..phase_retrieval import (Optics, RetrievalOptions,
                                        estimate_wavefront)
from ..phase_retrieval.dm_basis import DEFAULT_KEEP
from ..phase_retrieval.estimate import _even_modes
from .optimizers import _Base

# Noll terms the retrieval fits; matches `phase_retrieval.DEFAULT_MODES` and
# `tools/correctability.py`.
FIT_MODES = tuple(range(4, 12))

# A residual this many times the first accepted round's says the fit has
# stopped describing the spot, which makes the coefficients unsafe to drive.
_RESIDUAL_REJECT = 2.0

# Fraction of a rejected round's correction to retry with. Halving is the
# coarsest useful retreat, matching `modal._AUTO_TRUST_SHRINK`: gentler needs
# several failed rounds to matter and the budget is a handful of rounds.
_TRUST_SHRINK = 0.5

# Below this the trust region cannot express a correction the command grid
# would even round to something nonzero, so shrinking again is theatre.
_TRUST_FLOOR = 0.1

# Halvings the polish stage must have left when it takes over. A hill climb
# stops when its step reaches `min_step`, so handing it a step already at that
# grid converges it immediately -- which looks like agreement and is silence.
_POLISH_HALVINGS = 2


class PlantError(ValueError):
    """The influence matrix cannot drive the actuators this loop owns."""


class _Plant:
    """The mirror as the controller sees it: waves in, bits out.

    Attributes:
        waves_per_bit: (npix, n_act) influence matrix over the pupil pixels,
            in waves per command bit, columns in the LOOP's actuator order.
        modal: (n_act, keep) bits per unit amplitude of each retained
            eigenmode.
        basis: (len(FIT_MODES), npix) Noll modes on the same pupil pixels.
        keep: Eigenmodes retained.
        source: The file the matrix came from, for the run record.
    """

    def __init__(self, waves_per_bit, modal, basis, keep, source):
        self.pairing = []  # Filled by `load`; see its note on column order.
        self.waves_per_bit = waves_per_bit
        self.modal = modal
        self.basis = basis
        self.keep = int(keep)
        self.source = str(source)
        # Waves at each pupil pixel per unit amplitude of each eigenmode. The
        # solve's design matrix, formed once: it does not depend on the state.
        self.design = waves_per_bit @ modal

    @classmethod
    def load(cls, path, channels, keep=DEFAULT_KEEP):
        """Read an Impact Matrix `influence_matrix.npz` for these actuators.

        The session file carries the pupil maps the lightweight store lacks.
        Its `channels` are the session's own labels, so columns are paired by
        position; the assumption is recorded in `pairing` and can be checked
        with `verify_against`.

        Args:
            path: An Impact Matrix `influence_matrix.npz`.
            channels: The loop's actuator channels, in the loop's order.
            keep: Eigenmodes retained; see `dm_basis.DEFAULT_KEEP`.

        Returns:
            A `_Plant`.

        Raises:
            PlantError: If the file does not describe this many actuators.
        """
        data = np.load(str(path), allow_pickle=True)
        for key in ("c_rad_per_bit", "ctrl", "inside", "grid_n", "channels"):
            if key not in data:
                raise PlantError(
                    f"{path}: no '{key}' -- re-export it from the Impact "
                    "Matrix page")
        have = [int(c) for c in data["channels"]]
        want = [int(c) for c in channels]
        if len(have) != len(want):
            raise PlantError(
                f"{path} was measured on {len(have)} actuators but this loop "
                f"drives {len(want)}. An eigenmode is defined by all the "
                "actuators measured together, so a subset will not do")
        waves_per_bit = np.asarray(data["c_rad_per_bit"], float)
        n_keep = int(min(int(keep), int(np.asarray(data["ctrl"]).shape[1])))
        modal = np.asarray(data["ctrl"], float)[:, :n_keep]
        # `c_rad_per_bit` carries the factor 2 for reflection and the laser
        # conversion already; waves is the unit the retrieval reports in.
        waves_per_bit = waves_per_bit / (2.0 * np.pi)

        inside = np.asarray(data["inside"], bool)
        grid_n = int(data["grid_n"])
        if waves_per_bit.shape[0] != int(inside.sum()):
            raise PlantError(
                f"{path}: {waves_per_bit.shape[0]} matrix rows for "
                f"{int(inside.sum())} pupil pixels")
        rho, theta, _ys, _xs = ZK.unit_coords(inside, ((grid_n - 1) / 2.0,) * 3)
        basis = np.vstack([ZK.zernike_mode(j, rho, theta) for j in FIT_MODES])
        plant = cls(waves_per_bit, modal, basis, n_keep, path)
        plant.pairing = [{"column": i, "file_label": h, "loop_channel": w}
                         for i, (h, w) in enumerate(zip(have, want))]
        return plant

    def verify_against(self, stored, tolerance=0.02):
        """Confirm the positional pairing against the loop's own calibration.

        The saved calibration uses the loop's channel numbers, so a column-
        for-column match confirms the pairing assumed in `load`. Columns are
        compared up to sign.

        Args:
            stored: A `common.calibration.StoredMatrix` already aligned to the
                loop's actuators, or None to skip.
            tolerance: Relative agreement required, as a fraction.

        Returns:
            A note describing what was checked, for the run record.

        Raises:
            PlantError: If the two files disagree, which means the columns do
                not describe the same actuators and the correction would be
                applied to the wrong piezos.
        """
        if stored is None:
            return ("column order assumed positional; no saved calibration "
                    "was loaded to confirm it against")
        stored = np.asarray(stored, float)
        if stored.shape[0] != self.modal.shape[0]:
            raise PlantError(
                f"{self.source} covers {self.modal.shape[0]} actuators but "
                f"the loaded calibration covers {stored.shape[0]} -- they are "
                "not the same mirror")
        n = min(self.modal.shape[1], stored.shape[1])
        mine, theirs = self.modal[:, :n], np.asarray(stored, float)[:, :n]
        for i in range(n):
            a, b = mine[:, i], theirs[:, i]
            scale = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-12)
            if min(float(np.abs(a - b).max()),
                   float(np.abs(a + b).max())) > tolerance * scale:
                raise PlantError(
                    f"{self.source} disagrees with the loop's saved "
                    f"calibration on eigenmode {i + 1}. The two files do not "
                    "describe the same actuators in the same order, so the "
                    "correction would go to the wrong piezos -- pick the "
                    "session file that belongs to this calibration")
        return (f"column order confirmed against the saved calibration on "
                f"{n} eigenmode(s), to {tolerance:.0%}")

    def wavefront(self, coeffs):
        """Pupil phase in waves from a Noll 4-11 coefficient vector."""
        return np.asarray(coeffs, float) @ self.basis

    def command(self, coeffs, room=None):
        """Bit offsets that best cancel `coeffs`, inside the travel available.

        Travel enters as a constraint on the solve rather than as a scale
        afterwards, so one railed actuator cannot zero the whole correction
        and the answer stays inside the calibrated subspace.

        Args:
            coeffs: Noll 4-11 RMS amplitudes in waves, in `FIT_MODES` order.
            room: `(lower, upper)` bit offsets each actuator has left, or None
                for the unconstrained answer the offline tool computes.

        Returns:
            An (n_act,) array of bit offsets, and the (keep,) modal
            amplitudes behind it.
        """
        target = -self.wavefront(coeffs)
        free = lsq_linear(self.design, target, bounds=(-np.inf, np.inf)).x
        if room is None:
            return self.modal @ free, free
        lower, upper = (np.asarray(v, float) for v in room)
        bits = self.modal @ free
        if np.all(bits >= lower - 1e-9) and np.all(bits <= upper + 1e-9):
            return bits, free  # Already feasible; no need to constrain.
        amps = self._constrained(target, free, lower, upper)
        return self.modal @ amps, amps

    def _constrained(self, target, start, lower, upper):
        """Least squares over modal amplitudes, subject to the bit box.

        Posed on the `keep x keep` Gram matrix. The answer is judged on
        feasibility and improvement, not on the solver's success flag, which
        SLSQP sets spuriously on converged answers here.

        Args:
            target: Wavefront to cancel, over the pupil pixels.
            start: The unconstrained amplitudes, used only for the fallback.
            lower: Per-actuator lower bit offset.
            upper: Per-actuator upper bit offset.

        Returns:
            The (keep,) modal amplitudes.
        """
        from scipy.optimize import minimize

        # ||D a - t||^2 = a'Ga - 2a'b + const, dropped to `keep` dimensions.
        # Normalised by the pixel count so the line search sees an O(1) cost
        # whatever the pupil sampling is.
        n = self.design.shape[0]
        gram = (self.design.T @ self.design) / n
        rhs = (self.design.T @ target) / n

        def cost(a):
            return float(a @ gram @ a - 2.0 * a @ rhs)

        def grad(a):
            return 2.0 * (gram @ a - rhs)

        zero = np.zeros_like(start)
        out = minimize(
            cost, zero, jac=grad, method="SLSQP",
            constraints=[{"type": "ineq",
                          "fun": lambda a: upper - self.modal @ a,
                          "jac": lambda a: -self.modal},
                         {"type": "ineq",
                          "fun": lambda a: self.modal @ a - lower,
                          "jac": lambda a: self.modal}],
            options={"maxiter": 200, "ftol": 1e-12})
        bits = self.modal @ out.x
        feasible = (np.all(bits >= lower - 1e-6)
                    and np.all(bits <= upper + 1e-6))
        if feasible and cost(out.x) < cost(zero):
            return out.x
        # Last resort: the free shape, shortened until it fits. It can reach
        # zero when an actuator is hard against a rail, and that is the honest
        # answer once the constrained solve has also failed to find a move.
        return start * self._room_fraction(self.modal @ start, lower, upper)

    @staticmethod
    def _room_fraction(bits, lower, upper):
        """Largest factor keeping `f * bits` inside the box. The fallback."""
        scale = 1.0
        for b, lo, hi in zip(bits, lower, upper):
            limit = hi if b > 0 else lo
            if b != 0 and abs(b) > abs(limit):
                scale = min(scale, abs(limit) / abs(b))
        return max(scale, 0.0)

    def residual_waves(self, coeffs, bits):
        """RMS in waves left over the pupil after driving `bits`."""
        left = self.wavefront(coeffs) + self.waves_per_bit @ np.asarray(bits,
                                                                        float)
        return float(np.sqrt(np.mean(left ** 2)))


class PseudoWfs(_Base):
    """Retrieve the wavefront from one frame, project it, drive it, repeat.

    One measurement per round. `observe` does the sensing -- it is handed the
    same averaged frame the score was computed from -- and `tell` does the
    deciding, so the expensive fit runs off the GUI thread with the rest of
    the decision.

    The metric is not the objective here; it is the judge. The controller
    minimises the retrieved wavefront, and the scalar score only decides
    whether a round is kept: a projection onto eight modes going down is
    not the same statement as the spot getting better.
    """

    def __init__(self, cfg, matrix=None, plant=None):
        """
        Args:
            cfg: Loop settings; reads the `wfs_*` knobs and the optics block.
            matrix: The loop's `StoredMatrix`, when one is loaded. Used only
                to confirm the session file's column order -- this controller
                takes its physics from the session file, which is the only one
                carrying pupil maps.
            plant: A ready `_Plant`, or None to load `cfg.wfs_influence_npz`.

        Raises:
            PlantError: If no usable influence matrix is available, or the two
                files disagree about the actuators.
            ValueError: If the optics block is not filled in.
        """
        super().__init__(cfg)
        self.optics = Optics(wavelength_nm=cfg.wavelength_nm,
                             focal_mm=cfg.focal_mm,
                             aperture_mm=cfg.aperture_mm,
                             pixel_um=cfg.pixel_um)
        if not self.optics.valid():
            raise ValueError(
                "Focal-plane wavefront sensing needs the optics block "
                "(wavelength, focal length, aperture, pixel pitch) filled in: "
                "they set the pupil-to-sensor scale the fit inverts")
        if plant is None:
            path = str(getattr(cfg, "wfs_influence_npz", "") or "").strip()
            if not path:
                raise PlantError(
                    "Focal-plane wavefront sensing needs an Impact Matrix "
                    "influence_matrix.npz -- the loop's saved calibration "
                    "holds no pupil maps, so it cannot expand a wavefront")
            plant = _Plant.load(path, [int(a.channel) for a in self.acts],
                                keep=int(getattr(cfg, "wfs_keep_modes", 0)
                                         or DEFAULT_KEEP))
        self.plant = plant
        aligned = None
        if matrix is not None:
            try:
                aligned = matrix.align([int(a.channel) for a in self.acts])
            except Exception:  # noqa: BLE001 -- reported as "unconfirmed"
                aligned = None
        self.pairing_note = plant.verify_against(aligned)

        self.rounds = max(1, int(getattr(cfg, "wfs_rounds", 4)))
        self.round = 0
        self.max_step_bit = float(getattr(cfg, "wfs_max_step_bit", 0.0) or 0.0)
        self.opts = RetrievalOptions(
            roi_px=int(getattr(cfg, "wfs_roi_px", 0) or 0))
        # Which coefficients flip together under the unobservable even-block
        # sign. Computed from the fitted modes so it stays right if they move.
        self._even = np.asarray(_even_modes(FIT_MODES), bool)

        self.stage = "seed"
        self.trust = 1.0
        self.sign = 1.0  # Even-block convention in force.
        self.sign_locked = False
        self._sign_tried = False

        # Verified ledger, kept apart from the observation ledger for the same
        # reason `modal._Modal` keeps them apart: a driven round is evidence,
        # not a point the controller has agreed to stand on.
        self.accepted_x = self._quant(self.x)
        self.accepted_score = -np.inf
        self.accepted_est = None
        self._pending = self.accepted_x.copy()

        self.est = None  # Latest fit, whatever became of it.
        self._guess = None  # Warm start for the next fit.
        self._proposed = np.zeros(len(self.acts))  # Last correction formed.
        self._base_residual = float("nan")  # Residual of the first fit kept.
        self.notes = [self.pairing_note]
        self.history = []  # One dict per round, for the run record.
        self.stop_reason = None

    # Sensing
    def observe(self, reading):
        """Fit the wavefront of the frame this reading describes.

        Args:
            reading: The `metrics.SpotReading` the next `tell` describes. Its
                `frame` is the AVERAGED image of the measurement, so the fit
                sees exactly the frames-per-measure the operator set.
        """
        frame = getattr(reading, "frame", None)
        if frame is None:
            self.est = None
            return
        with self.phase("retrieval"):
            self.est = estimate_wavefront(np.asarray(frame, float),
                                          self.optics, self.opts,
                                          guess=self._guess)

    # Deciding
    def tell(self, score, valid=True):
        """Keep or discard the round just measured, and form the next command.

        Args:
            score: The judge's score for the measured point.
            valid: Whether the point is usable at all.
        """
        self.iter += 1
        if not valid:
            return  # Hold the command; the loop re-measures.
        self.last_score = score
        if self.converged:
            self._park_hold(score)
            return
        if self.est is None:
            self._finish("no frame reached the controller; nothing was driven")
            return
        if self.stage == "seed":
            self._accept(score)
            self._base_residual = float(self.est.residual)
            self._advance()
            return
        if self._collapsed():
            # A fit that stopped describing the spot cannot be driven, and it
            # cannot be told apart from a real state change by its number
            # alone. Retreat rather than guess which one it was.
            self.notes.append(
                f"round {self.round}: fit residual {self.est.residual:.0f} "
                f"against {self._base_residual:.0f} at the accepted point; "
                "discarded without driving it")
            self._retreat()
            return
        if score > self.accepted_score + self._noise_level():
            self._accept(score)
            self.sign_locked = True  # This sign produced a real improvement.
            self._advance()
            return
        # No improvement. On the very first correction that is as likely to be
        # the even-block sign as the step size, and the twin costs one round
        # to rule out, so try it before shrinking anything.
        if not self.sign_locked and not self._sign_tried:
            self._sign_tried = True
            self.sign = -self.sign
            self.notes.append(
                f"round {self.round}: no gain; retrying with the twin "
                "even-mode sign, which one in-focus frame cannot choose")
            self._drive_from_accepted()
            return
        self._retreat()

    def _collapsed(self):
        """Whether this fit stopped describing the spot (not merely short)."""
        base = self._base_residual
        return (np.isfinite(base) and base > 0
                and float(self.est.residual) > _RESIDUAL_REJECT * base)

    def _accept(self, score):
        self.accepted_x = self._pending.copy()
        self.accepted_score = float(score)
        self.accepted_est = self.est
        self._guess = self.est  # Warm start: 156 ms instead of 14-22 s.
        if score > self.best_score:
            self.best_score = float(score)
            self.best_x = self._pending.copy()

    def _retreat(self):
        """A round failed: go back to the verified point on a shorter lever."""
        self.trust *= _TRUST_SHRINK
        if self.trust < _TRUST_FLOOR:
            self._finish("corrections stopped paying; trust region exhausted")
            return
        self.notes.append(f"round {self.round}: no gain, trust -> "
                          f"{self.trust:.2f}")
        self._drive_from_accepted()

    def _advance(self):
        """A round was kept: spend the next one on the remaining wavefront.

        A round is a CORRECTION driven, not a measurement taken: the seed
        measures the starting spot and commands nothing, so counting it would
        make `wfs_rounds=1` drive nothing at all.
        """
        if self.round >= self.rounds:
            self._finish(f"completed {self.round} round(s)")
            return
        self.round += 1
        self._drive_from_accepted()

    def _drive_from_accepted(self):
        """Form and stage the command for the wavefront at the kept point."""
        est = self.accepted_est
        coeffs = np.array([est.coeffs.get(j, 0.0) for j in FIT_MODES], float)
        coeffs = np.where(self._even, self.sign * coeffs, coeffs)
        base = self.accepted_x.astype(float)
        bits, amps = self.plant.command(coeffs, room=(self.lo - base,
                                                      self.hi - base))
        self._proposed = bits.copy()

        # The travel is already in the solve; these two only shorten a
        # feasible vector, and shortening keeps it feasible because zero is.
        scale = self.trust
        if self.max_step_bit > 0:
            peak = float(np.max(np.abs(bits))) or 1.0
            scale = min(scale, self.max_step_bit / peak)
        step = bits * scale
        target = self._quant(base + step)

        applied = float(np.max(np.abs(target - self.accepted_x)))
        self.history.append(dict(
            round=self.round, rms_waves=float(est.rms_waves),
            residual=float(est.residual), sign=float(self.sign),
            trust=float(self.trust), scale=float(scale),
            max_bit=applied,
            predicted_waves=self.plant.residual_waves(coeffs, step),
            amplitudes=[float(a) for a in amps],
            beyond_capture_range=bool(est.beyond_capture_range)))
        if applied < max(1, int(self.cfg.min_step)):
            # Rounding to the command grid erased the whole correction: the
            # mirror cannot express what is left, so nothing here can.
            self._finish("remaining correction is below one command step")
            return
        self.stage = "correct"
        self._pending = target

    def _finish(self, reason):
        self.converged = True
        self.stage = "park"
        self.stop_reason = reason
        self._pending = self.accepted_x.copy()
        self.best_x = self.accepted_x.copy()
        self.best_score = max(self.best_score, self.accepted_score)

    # Handover
    def handover_step(self):
        """Probe size the polish search should start from, in bits.

        The correction this controller still wanted when it stopped, in the
        mirror's own bits. Bounded below by `_POLISH_HALVINGS` grid steps and
        above by the cold `move_step`.

        Returns:
            A step in bits.
        """
        grid = max(1, int(self.cfg.min_step))
        want = float(np.max(np.abs(self._proposed))) if len(self._proposed) \
            else 0.0
        return int(min(max(want, (2 ** _POLISH_HALVINGS) * grid),
                       max(grid, int(self.cfg.move_step))))

    def status(self):
        st = super().status()
        est = self.accepted_est
        st.update(stage=self.stage, round=self.round, rounds=self.rounds,
                  trust=self.trust, even_sign=self.sign,
                  sign_locked=self.sign_locked,
                  stop_reason=self.stop_reason,
                  wavefront_waves=(float(est.rms_waves) if est else
                                   float("nan")),
                  fit_residual=(float(est.residual) if est else float("nan")),
                  beyond_capture_range=(bool(est.beyond_capture_range)
                                        if est else False),
                  notes=list(self.notes))
        return st

    def report(self):
        """Everything the run record should carry about this stage."""
        return dict(controller="focal-plane wavefront sensing",
                    influence_matrix=str(self.plant.source),
                    column_pairing=list(self.plant.pairing),
                    pairing_check=self.pairing_note,
                    modes_kept=self.plant.keep,
                    fitted_noll=list(FIT_MODES),
                    roi_px=dict(setting=int(getattr(self.cfg, "wfs_roi_px", 0)),
                                used=int(self.est.roi_px) if self.est else 0,
                                source=("4x the measured r80"
                                        if not getattr(self.cfg, "wfs_roi_px", 0)
                                        else "operator")),
                    rounds_requested=self.rounds, rounds_run=self.round,
                    even_sign=self.sign, sign_locked=self.sign_locked,
                    stop_reason=self.stop_reason,
                    handover_step_bit=self.handover_step(),
                    history=self.history, notes=list(self.notes))
