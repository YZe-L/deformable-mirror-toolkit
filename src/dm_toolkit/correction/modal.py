# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 2.7, 2026-09-01

"""Model-based modal solves, as opposed to the searches in `optimizers`.

Drives the measured eigenmodes and inverts a quadratic model to get the
aberration coefficients in a fixed handful of measurements. Valid only for a
metric that is quadratic in the aberration. Every such metric goes flat near
the diffraction limit, so `optimizers.Staged` hands over to a search.
"""

from __future__ import annotations

import numpy as np

from .optimizers import _Base

# Fraction of a mode's own actuator headroom one probe may spend. The full
# range is the mirror's to use: this only stops a probe from being TRUNCATED,
# which would make it a different shape from the mode that was calibrated.
_PROBE_FILL = 1.0

# Fraction of the usable travel one correction may spend; scaled as one
# vector, never clipped per actuator.
_STEP_FILL = 1.0

# A three-point parabola measured at +/-b says nothing trustworthy about a
# vertex many b away; beyond this the solve is extrapolating, not fitting.
_VERTEX_REACH = 3.0

# Curvature must stand this far above its noise: `M+ - 2 M0 + M-` carries
# sqrt(6) times the noise of one reading.
_CURV_SIGMA = 3.0

# Ratio between consecutive singular values that ends the useful control
# basis; dimensionless, so it transfers across mirrors.
_SINGULAR_CLIFF = 5.0

# A probe realised below half its requested amplitude carries under a quarter
# of the intended curvature signal. Explicit operator mode counts still win.
_MIN_PROBE_FRACTION = 0.5

# Fallback curvature floor, as a fraction of the metric, until a split-half
# estimate exists.
_RELATIVE_METRIC_NOISE = 1e-4

# Metric swing a swept rung must show above the noise before it may set the
# probe amplitude; a convex fit alone is not evidence.
_MIN_PROBE_SNR = 10.0

# Smallest probe, in modal grid steps on the mode's largest actuator: below
# four steps quantisation, not the mirror, decides the curvature.
_MIN_PROBE_GRID_STEPS = 4.0

# A probe that expands beyond the fixed second-moment ROI is retried at half
# amplitude.  Three halvings cover one octave below the smallest cold-start
# rung without letting a bad footprint loop forever.
_ROI_PROBE_RETRIES = 3
_ROI_SCALE = 0.5

# Automatic round selection stops only after two consecutive quiet rounds.
_AUTO_MIN_ROUNDS = 2
_AUTO_CONFIRM_ROUNDS = 2
# A correction is physically small below this fraction of the usable travel.
_AUTO_COMMAND_FRACTION = 0.005

# A solved direction is walked at more than one length: rescaling costs one
# image, whereas a new probe set costs 2N+1 or N+2. The direction, the probes
# and the vertex formula are untouched; only the step length is measured.
_LINE_BACKTRACK = (0.5, 0.25)
# Extend rungs form a curvature-error ladder: a curvature read `e` too high
# makes the step 1/(1-e) too short, and the probe design permits e <= 0.25,
# so the first rung is 1.33x.
_LINE_EXTEND = (4.0 / 3.0, 2.0)

# A mode whose probes returned nothing for two consecutive rounds is
# re-probed every other round while set aside; never dropped, only rationed.
_IDLE_MODE_ROUNDS = 2
_IDLE_MODE_RETRY = 2

# Halving is the coarsest useful retreat; bounded below by
# `_MIN_PROBE_GRID_STEPS` and released again by a verified improvement.
_AUTO_TRUST_SHRINK = 0.5


class _Modal(_Base):
    """Shared driving for the modal solves.

    Subclasses supply which probes to take (`_probe_plan`), how to read the
    quadratic quantity out of a measurement (`_value`), and how to turn the
    probes into coefficients (`_solve`).
    """

    # The mode that carries the whole round's scale and so is never rationed;
    # None when every mode measures its own parabola.
    _scale_mode = None

    def __init__(self, cfg, matrix):
        """
        Args:
            cfg: Loop settings; reads the `modal_*` knobs.
            matrix: A `common.calibration.StoredMatrix` covering exactly the
                actuators in `cfg`.

        Raises:
            MatrixMismatch: If the matrix does not cover these actuators.
        """
        super().__init__(cfg)
        self.matrix = matrix
        # Rows reordered into this loop's actuator order; raises if the sets
        # differ, which is the only way to catch a matrix measured for a
        # different mirror before it drives one.
        self.ctrl = matrix.align([int(a.channel) for a in self.acts])
        self.bias = float(cfg.modal_bias_rad)
        self.mode_resolution = {}
        self.n_modes = self._resolve_mode_count(cfg, matrix)
        self.auto_rounds = bool(getattr(cfg, "modal_auto_rounds", False))
        self.rounds = max(_AUTO_MIN_ROUNDS if self.auto_rounds else 1,
                          int(cfg.modal_rounds))
        self.round = 0
        self.notes = []
        self.clipped = False
        self._capped = set()  # Modes already reported as amplitude-limited.
        # Two independent reasons to narrow a probe, kept apart on purpose: a
        # shared factor would let a camera-footprint problem and a model
        # mismatch multiply into each other and dissolve the probe.
        self._probe_scale = np.ones(self.n_modes, float)  # ROI footprint.
        self._trust_scale = np.ones(self.n_modes, float)  # Auto-round trust.
        self._roi_probe_retries = np.zeros(self.n_modes, int)
        # (mode, signed factor) -> realised coefficient after integer command
        # quantisation. Cleared whenever the working point or amplitude moves.
        self._realized = {}
        self.x0 = np.array(self.best_x, float)  # Working point, in bits.
        self._correction_origin = self.x0.copy()
        self._correction_step = np.zeros_like(self.x0)
        self._correction_scale = 1.0
        # The scale of the step actually BANKED, which is not the last scale
        # tried: a rejected extension leaves `_correction_scale` on the rung it
        # was testing, and the round keeps the shorter step.
        self._kept_scale = 1.0
        # Which half of the scale ladder is being walked, and the score the
        # round started from (see the "verify" branch of `tell`).
        self._line_mode = None
        self._line_previous = -np.inf
        self._line_steps = 0
        self.line_search = []  # One record per round that re-scaled its step.
        self._reading = None
        # The full 2N+1 / N+2 plan, and the subset this round measures. They
        # differ only once a mode has been set aside for returning nothing
        # (see `_update_idle_modes`); the full plan stays the run's budget.
        self._full_plan = self._probe_plan()
        self._plan = list(self._full_plan)
        self._idle_rounds = np.zeros(self.n_modes, int)
        self._idle_since = {}  # Mode -> round it was set aside.
        self.idle_modes = []  # One record per set-aside and per return.
        self._k = 0
        self._m = {}  # Probe key -> raw metric value.
        # A modal probe is evidence about a parabola, not a command the solver
        # has accepted.  Keep the verified centre and the best observation in
        # separate ledgers so extra rounds cannot silently become hill-climb.
        self.accepted_x = self._quant(self.x0)
        self.accepted_score = -np.inf
        self.accepted_value = float("nan")
        self.observed_best_x = self.accepted_x.copy()
        self.observed_best_score = -np.inf
        self.stop_reason = None
        self.failed_reason = None
        self._finish_after_recenter = False
        self._finish_reason_pending = None
        self._recenter_cause = None
        self._last_correction_max_bit = 0.0
        self._auto_quiet_rounds = 0
        self.auto_round_report = []
        # Per-round spot snapshots: the frame measured on the centre the round
        # accepted. `_spot_due` carries the round number across that gap.
        self.rounds_completed = []
        self.spot_round = None
        self._spot_due = None
        self.last_coefficients = []
        self.value_noise_by_round = []
        # Probe-amplitude calibration, run once before the first solve round.
        self.auto_bias = bool(getattr(cfg, "modal_auto_bias", False))
        self.bias_requested = self.bias  # What the operator typed.
        self.bias_report = []  # One entry per candidate, for the run record.
        self.bias_guard_report = []  # Extra measured recovery below ladder.
        ladder = tuple(getattr(cfg, "modal_bias_ladder", ()) or ())
        self._bias_candidates = tuple(sorted({abs(float(v)) for v in ladder
                                              if float(v) != 0.0}))
        self.tolerance = float(getattr(cfg, "modal_bias_tolerance", 0.08))
        self.bias_scanned = None  # Set once the sweep has chosen one.
        self.bias_points = []  # (amplitude_rad, metric) the sweep measured.
        # Metric noise the rung verdicts were judged against. Recorded because
        # "this rung was in the noise" is only checkable against the number
        # that stood for the noise at that moment.
        self.bias_noise = None
        # Raw-metric noise used by the curvature gate. It is recomputed from
        # the accepted centre and the latest split-half score noise each round;
        # zero exists only before the first active-axis baseline is measured.
        self.value_noise = 0.0
        # Three points define a parabola exactly, so the smallest candidate
        # cannot be validated; two inner diagnostic rungs support recovery.
        self._bias_guards = ((0.25 * self._bias_candidates[0],
                              0.5 * self._bias_candidates[0])
                             if self._bias_candidates else ())
        sweep = tuple(sorted(set(self._bias_guards + self._bias_candidates)))
        self._calib = [(0, s * f) for f in sweep for s in (-1.0, 1.0)]
        self.stage = "base"
        self._pending = self._quant(self.x0)

    def _resolve_mode_count(self, cfg, matrix):
        """Resolve a stable, contiguous control basis for any mirror size.

        ``StoredMatrix.keep`` remains the outer noise-floor bound.  In auto
        mode, this resolver additionally stops where a singular-value cliff is
        confirmed by a control-cost jump, or before the first mode that cannot
        realise half of the requested probe at the current working point.  An
        explicit ``modal_modes`` value keeps its existing operator-override
        meaning and is only bounded by the stored usable count.
        """
        stored = int(max(1, min(matrix.keep or matrix.n_modes,
                               matrix.n_modes, self.ctrl.shape[1])))
        singular = np.asarray(matrix.s, float)[:stored]
        suggested = stored
        reasons = []
        cliff = None
        cliff_index = None
        for i in range(1, len(singular)):
            before, after = singular[i - 1], singular[i]
            ratio = (float(before / after)
                     if np.isfinite(before) and np.isfinite(after) and after > 0
                     else float("inf"))
            if ratio >= _SINGULAR_CLIFF:
                cliff_index = i
                cliff = dict(after_mode=i, ratio=ratio,
                             singular_before=float(before),
                             singular_after=float(after))
                break

        centre = np.asarray(self.best_x, float)
        room = np.minimum(centre - self.lo, self.hi - centre)
        headroom = []
        for mode in range(stored):
            column = np.abs(np.asarray(self.ctrl[:, mode], float))
            used = column > 0
            cap = (float(np.min(room[used] / column[used]))
                   if np.any(used) else float("inf"))
            fraction = (cap / self.bias if self.bias > 0 else float("inf"))
            headroom.append(dict(mode=mode + 1, cap_rad=cap,
                                 requested_fraction=fraction,
                                 max_bit_per_rad=(float(np.max(column))
                                                  if column.size else 0.0)))
            if mode < suggested and fraction < _MIN_PROBE_FRACTION:
                suggested = max(1, mode)
                reasons.append(
                    f"mode {mode + 1} has only {fraction:.2f}x requested "
                    "probe headroom")
                break

        # A singular-value cliff alone can be a stale scale calibration; Modal
        # Fit remeasures that scale. It becomes a control-basis cutoff when the
        # corresponding inverse column also becomes abnormally expensive.
        if cliff_index is not None:
            costs = np.max(np.abs(self.ctrl[:, :stored]), axis=0)
            prior = costs[:cliff_index]
            reference = float(np.median(prior[prior > 0])) if np.any(
                prior > 0) else 0.0
            cost_ratio = (float(costs[cliff_index] / reference)
                          if reference > 0 else float("inf"))
            cliff["control_cost_ratio"] = cost_ratio
            if cost_ratio >= _SINGULAR_CLIFF:
                suggested = min(suggested, cliff_index)
                reasons.append(
                    f"singular-value cliff {cliff['ratio']:.2f}x after mode "
                    f"{cliff_index} coincides with a {cost_ratio:.2f}x "
                    "control-cost jump")
            else:
                reasons.append(
                    f"singular-value cliff {cliff['ratio']:.2f}x was retained "
                    f"because control cost changes only {cost_ratio:.2f}x")

        explicit = int(cfg.modal_modes)
        used = min(stored, explicit) if explicit > 0 else suggested
        used = int(max(1, used))
        self.mode_resolution = dict(
            source="operator" if explicit > 0 else "automatic",
            requested=explicit, stored_keep=stored,
            suggested=int(max(1, suggested)), used=used,
            # Headroom is judged at the amplitude the operator asked for; a
            # later sweep can only reduce it.
            headroom_judged_at_rad=float(self.bias),
            singular_cliff=cliff, headroom=headroom,
            reasons=reasons or ["stored usable basis has no stability cliff"])
        return used

    # Subclass hooks
    def _probe_plan(self):
        """Return the (mode, sign) probes to take each round."""
        raise NotImplementedError

    def _solve(self):
        """Return the estimated aberration per mode, in rad RMS."""
        raise NotImplementedError

    def _value(self, reading):
        """Return the raw, quadratic metric value this solve inverts."""
        raise NotImplementedError

    # Driving
    def _quant(self, v):
        """Quantise modal vectors on their own native command grid.

        Args:
            v: One actuator command vector.

        Returns:
            An integer command clipped to the configured actuator limits.
        """
        step = max(1, int(getattr(self.cfg, "modal_bit_step", 1)))
        out = np.round(np.asarray(v, float) / step) * step
        return np.clip(out, self.lo, self.hi).astype(int)

    def observe(self, reading):
        self._reading = reading

    def _requested_amp(self, mode):
        """Return the requested per-mode amplitude before actuator limits.

        Args:
            mode: Zero-based eigenmode index.

        Returns:
            Requested RMS wavefront coefficient in radians.
        """
        return float(self.bias * self._probe_scale[mode]
                     * self._trust_scale[mode])

    def _headroom_amp(self, mode):
        """Return a probe amplitude capped by symmetric actuator headroom.

        The bit cost of one radian differs by an order of magnitude between
        eigenmodes, so one amplitude for all of them either barely moves the
        cheap modes or rails the dear ones -- and a clipped probe is no longer
        the mode that was calibrated.

        Args:
            mode: Zero-based eigenmode index.

        Returns:
            Largest requested amplitude that keeps both probe signs in range.
        """
        c = np.abs(np.asarray(self.ctrl[:, mode], float))
        room = np.minimum(self.x0 - self.lo, self.hi - self.x0)
        usable = c > 0
        requested = self._requested_amp(mode)
        if not np.any(usable):
            return requested
        cap = _PROBE_FILL * float(np.min(room[usable] / c[usable]))
        amplitude = max(0.0, min(requested, cap))
        if cap >= requested:
            return amplitude
        if mode not in self._capped:
            self._capped.add(mode)
            self.notes.append(
                f"mode {mode + 1}: probe cut from {requested:.2f} to "
                f"{amplitude:.2f} rad; at {float(np.max(c)):.0f} bit/rad "
                "there is no more actuator headroom")
        return amplitude

    def _probe_delta(self, mode, factor=1.0):
        """Build one symmetric, quantised modal displacement.

        Args:
            mode: Zero-based eigenmode index.
            factor: Signed fraction of the mode's requested amplitude.

        Returns:
            A tuple ``(delta_bits, effective_rad, shape_error)``. The effective
            coefficient is the least-squares projection of the integer command
            back onto the stored control column.
        """
        factor = float(factor)
        base = self._quant(self.x0)
        c = np.asarray(self.ctrl[:, mode], float)
        requested = abs(factor) * self._headroom_amp(mode)
        step = max(1, int(getattr(self.cfg, "modal_bit_step", 1)))
        delta = np.round(requested * c / step) * step
        # Preserve exact +/- symmetry. Headroom was computed for both signs;
        # this final clamp only covers a half-step round-off at a rail.
        room = np.minimum(base - self.lo, self.hi - base)
        limited = np.clip(delta, -room, room)
        if np.any(np.abs(limited - delta) > 0.5):
            self.clipped = True
        delta = limited
        denom = float(c @ c)
        effective = float(delta @ c / denom) if denom > 0 else 0.0
        model = effective * c
        model_norm = float(np.linalg.norm(model))
        shape_error = (float(np.linalg.norm(delta - model)) / model_norm
                       if model_norm > 0 else float("inf"))
        signed_effective = np.sign(factor) * effective
        self._realized[(int(mode), factor)] = dict(
            requested_rad=np.sign(factor) * requested,
            effective_rad=signed_effective,
            shape_error=shape_error,
            delta_bits=(np.sign(factor) * delta).astype(int).tolist())
        return np.sign(factor) * delta, signed_effective, shape_error

    def _effective_amp(self, mode, factor=1.0):
        """Return the coefficient actually realised by a quantised probe."""
        key = (int(mode), float(factor))
        if key not in self._realized:
            self._probe_delta(mode, factor)
        return float(self._realized[key]["effective_rad"])

    def _symmetric_amp(self, mode):
        """Return the mean realised magnitude of a mode's two probe signs."""
        plus = abs(self._effective_amp(mode, +1.0))
        minus = abs(self._effective_amp(mode, -1.0))
        return 0.5 * (plus + minus)

    def _bits(self, mode, sign):
        """Return one symmetrically quantised modal probe command.

        Args:
            mode: Zero-based eigenmode index.
            sign: Signed amplitude factor; probe plans use +/-1 and the cold
                sweep uses fractional values.

        Returns:
            Quantised actuator vector for the requested probe.
        """
        delta, effective, _ = self._probe_delta(mode, sign)
        if not abs(effective) > 0:
            key = (int(mode), float(sign))
            if key not in self._capped:
                self._capped.add(key)
                self.notes.append(
                    f"mode {mode + 1}: {abs(float(sign)):.2f}x probe vanished "
                    "on the modal command grid; measurement will be ignored")
        return self._quant(self.x0) + delta.astype(int)

    def _record_observation(self, score):
        """Record a measurement without granting it solver authority.

        Args:
            score: Objective score just measured at ``self._pending``.
        """
        self.last_score = score
        if score > self.observed_best_score:
            self.observed_best_score = score
            self.observed_best_x = self._pending.copy()

    def _score_margin(self, reference):
        """Smallest verified centre change treated as real progress."""
        scale = abs(reference) if np.isfinite(reference) else 0.0
        # Measured noise decides once it exists, same rule as the searches
        # (see optimizers._measured); the relative term is only the cold floor.
        return self._measured(3.0, max(0.002 * scale, 1e-9))

    def _auto_command_threshold(self):
        """Mirror-scaled integer correction regarded as physically small."""
        span = np.asarray(self.hi, float) - np.asarray(self.lo, float)
        usable = span[np.isfinite(span) & (span > 0)]
        typical = float(np.median(usable)) if usable.size else 0.0
        grid = float(max(1, int(getattr(self.cfg, "modal_bit_step", 1))))
        return max(grid, _AUTO_COMMAND_FRACTION * typical)

    def _min_trust_scale(self, mode):
        """Smallest trust factor that still leaves a resolvable probe.

        Args:
            mode: Zero-based eigenmode index.

        Returns:
            Trust factor at which the mode's largest actuator component spans
            `_MIN_PROBE_GRID_STEPS` of the modal command grid.
        """
        column = np.abs(np.asarray(self.ctrl[:, mode], float))
        cost = float(np.max(column)) if column.size else 0.0
        if not (cost > 0 and self.bias > 0):
            return 1.0
        grid = float(max(1, int(getattr(self.cfg, "modal_bit_step", 1))))
        floor = _MIN_PROBE_GRID_STEPS * grid / (self.bias * cost)
        return float(min(1.0, floor))

    def _shrink_auto_trust(self, reason):
        """Narrow the next round's probes after a non-improving large step.

        Args:
            reason: Why trust is being reduced, for the run notes.

        Returns:
            One record per mode whose trust changed.
        """
        if not self.auto_rounds or not self.n_modes:
            return []
        coefficients = np.asarray(self.last_coefficients, float)
        active = np.flatnonzero(np.isfinite(coefficients)
                                & (np.abs(coefficients) > 1e-12))
        active = active[active < self.n_modes]
        if not active.size:
            active = np.arange(self.n_modes)
        changed, floored = [], []
        for mode in active:
            before = float(self._trust_scale[mode])
            limit = self._min_trust_scale(mode)
            after = max(before * _AUTO_TRUST_SHRINK, limit)
            if after >= before:
                floored.append(int(mode + 1))
                continue
            self._trust_scale[mode] = after
            changed.append(dict(mode=int(mode + 1), before=before,
                                after=float(after), floor=limit))
        self._realized.clear()
        if changed:
            self.notes.append(
                f"round {self.round}: {reason}; probe trust reduced for "
                f"{len(changed)} active mode(s)")
        if floored:
            self.notes.append(
                f"round {self.round}: {reason}, but mode(s) "
                f"{floored} are already at the smallest resolvable probe; "
                "the model, not the amplitude, is the remaining error")
        return changed

    # Line search along one solved direction.
    def _line_candidate(self, scale):
        """The command this direction reaches at `scale`, or None if it rails.

        Scaled as ONE vector and range-checked as one, exactly like
        `_fit_step`: clipping a single actuator would leave a shape that is no
        longer a combination of the calibrated modes.

        Args:
            scale: Multiple of the already-fitted correction step.

        Returns:
            np.ndarray | None: The quantised command, or None when it would
            leave the actuator range, land where the mirror already is, or
            land back on the round's starting point.
        """
        if not np.any(np.abs(self._correction_step) > 1e-9):
            return None
        raw = self._correction_origin + float(scale) * self._correction_step
        if np.any(raw < self.lo - 0.5) or np.any(raw > self.hi + 0.5):
            return None
        candidate = self._quant(raw)
        if np.array_equal(candidate, self._quant(self.x0)):
            return None
        if np.array_equal(candidate, self._quant(self._correction_origin)):
            return None
        return candidate

    def _line_step(self, scale, why):
        """Drive the same direction at another scale; True when it was queued."""
        if not getattr(self.cfg, "modal_line_search", True):
            return False
        candidate = self._line_candidate(scale)
        if candidate is None:
            return False
        self._correction_scale = float(scale)
        self.x0 = candidate.astype(float)
        self._pending = candidate
        actual = candidate.astype(float) - self._correction_origin
        self._last_correction_max_bit = (float(np.max(np.abs(actual)))
                                         if actual.size else 0.0)
        self._realized.clear()
        self._line_steps += 1
        self.notes.append(
            f"round {self.round}: {why} -- same direction at "
            f"{100 * scale:.0f}% ({self._last_correction_max_bit:.0f} bit), "
            "one measurement instead of a new probe set")
        self.stage = "verify"
        return True

    def _line_backtrack(self):
        """Try the next shorter scale of this direction after a rejection."""
        for scale in _LINE_BACKTRACK:
            if scale < self._correction_scale and self._line_step(
                    scale, "correction overshot"):
                return True
        return False

    def _line_extend(self):
        """Try the next longer scale after a step the mirror did not reject."""
        for scale in _LINE_EXTEND:
            if scale > self._correction_scale and self._line_step(
                    scale, "testing a longer step"):
                return True
        return False

    def _end_round(self, previous, score, value, fresh):
        """Close an accepted round on the best point it measured.

        Args:
            previous: Score the round started from.
            score: Score of the point being kept.
            value: Raw metric at that point.
            fresh: Whether the LAST measurement was taken at that point. A
                round whose final line step overshot ends standing somewhere
                worse, so its centre is re-driven before the next round uses
                it as `M0`.
        """
        self._line_mode = None
        if not fresh:
            # `_last_correction_max_bit` must describe the step kept, or a
            # quiet round can never be recognised.
            kept = np.asarray(self.accepted_x, float) - self._correction_origin
            self._last_correction_max_bit = (float(np.max(np.abs(kept)))
                                             if kept.size else 0.0)
        if self._line_steps:
            # `_kept_scale`, not `_correction_scale`: the latter is still on
            # the rung a rejected extension was testing, so recording it made
            # every overshooting round claim it had kept the longer step.
            self.line_search.append(dict(round=self.round,
                                         extra_measurements=self._line_steps,
                                         scale_kept=self._kept_scale,
                                         scale_last_tried=self._correction_scale))
        self._close_round("accepted", score, fresh)
        auto_stop = self._auto_round_decision(previous, score, "accepted")
        if self.round >= self.rounds:
            self._finish_or_verify_observed(
                ("automatic-round safety maximum completed after "
                 "correction verification" if self.auto_rounds else
                 "configured rounds completed after correction verification"))
            return
        if auto_stop:
            self._finish_or_verify_observed(
                "automatic rounds converged after two verified quiet rounds")
            return
        if fresh:
            self._begin_round(score, value)
            return
        self.x0 = np.asarray(self.accepted_x, float)
        self._pending = self.accepted_x.copy()
        self.stage = "recenter"
        self._recenter_cause = "line_overshoot"
        self._finish_after_recenter = False

    def _restore_auto_trust(self):
        """Return trust to full after a correction the mirror confirmed.

        Trust was withdrawn because a prediction failed. A verified
        improvement says that reason no longer holds, and leaving the probes
        narrow would keep every later curvature needlessly noisy.

        Returns:
            True when any mode's trust was restored.
        """
        if not np.any(self._trust_scale < 1.0):
            return False
        self._trust_scale = np.ones(self.n_modes, float)
        self._realized.clear()
        self.notes.append(
            f"round {self.round}: correction verified; probe trust restored")
        return True

    def _auto_round_decision(self, previous, score, outcome):
        """Record one completed round and return whether auto mode may stop.

        A quiet round needs two independent facts: the freshly measured centre
        did not change beyond score noise, and the correction that physically
        reached the integer command grid was small.  Rejected corrections are
        model-mismatch evidence, never convergence evidence.

        Args:
            previous: Accepted score before this round's correction.
            score: Freshly verified score after it.
            outcome: One of "accepted", "rejected" or "zero_correction".

        Returns:
            True when automatic mode has enough evidence to stop.
        """
        if not self.auto_rounds:
            return False
        margin = self._score_margin(previous)
        gain = (float(score - previous)
                if np.isfinite(score) and np.isfinite(previous)
                else float("nan"))
        command_threshold = self._auto_command_threshold()
        stable_score = bool(np.isfinite(gain) and abs(gain) <= margin)
        small_command = self._last_correction_max_bit <= command_threshold
        quiet = bool(outcome in ("accepted", "zero_correction")
                     and stable_score and small_command)
        # A correction the mirror confirmed retires the reason trust was
        # withdrawn, so the next round measures on full-length levers again.
        if outcome == "accepted" and np.isfinite(gain) and gain > margin:
            self._restore_auto_trust()
        trust_change = []
        if quiet:
            self._auto_quiet_rounds += 1
        else:
            self._auto_quiet_rounds = 0
            if (outcome == "rejected"
                    or (outcome == "accepted" and stable_score
                        and not small_command)):
                trust_change = self._shrink_auto_trust(
                    "rejected correction" if outcome == "rejected" else
                    "large correction produced no score change above noise")
        confirmed = bool(self.round >= _AUTO_MIN_ROUNDS
                         and self._auto_quiet_rounds
                         >= _AUTO_CONFIRM_ROUNDS)
        report = dict(
            round=int(self.round), outcome=str(outcome),
            previous_score=(float(previous) if np.isfinite(previous) else None),
            verified_score=(float(score) if np.isfinite(score) else None),
            score_gain=(gain if np.isfinite(gain) else None),
            score_margin=float(margin), stable_score=stable_score,
            correction_max_bit=float(self._last_correction_max_bit),
            command_threshold_bit=float(command_threshold),
            small_command=bool(small_command), quiet=quiet,
            quiet_confirmations=int(self._auto_quiet_rounds),
            required_confirmations=_AUTO_CONFIRM_ROUNDS,
            minimum_rounds=_AUTO_MIN_ROUNDS, stop_confirmed=confirmed,
            trust_change=trust_change)
        self.auto_round_report.append(report)
        self.notes.append(
            f"round {self.round}: automatic-round check -- score change "
            f"{gain:+.4g} (quiet <= {margin:.4g}), correction "
            f"{self._last_correction_max_bit:.1f} bit "
            f"(small <= {command_threshold:.1f}); quiet confirmation "
            f"{self._auto_quiet_rounds}/{_AUTO_CONFIRM_ROUNDS}")
        return confirmed

    def _metric_noise(self, score, value):
        """Point noise at one working point, in raw quadratic-metric units.

        `metrics.norm_score` is ``v / (v + seed)`` and the modal objectives
        make ``v`` the reciprocal of `_value`, so ``dscore/dvalue`` is
        ``score*(1-score)/value``. Inverting it puts the measured split-half
        score sigma in the same units as ``M0, M+, M-``.

        Args:
            score: Score measured at that point.
            value: Raw quadratic metric at that same point.

        Returns:
            ``(relative_floor, measured)``. The measured term is 0 until the
            loop has fed in a split-half sigma, which is why the floor exists.
        """
        magnitude = abs(float(value)) if np.isfinite(value) else 0.0
        relative = _RELATIVE_METRIC_NOISE * magnitude
        measured = 0.0
        if (self.noise_hint > 0 and np.isfinite(score)
                and np.isfinite(value) and 0.0 < score < 1.0):
            slope = max(float(score) * (1.0 - float(score)), 1e-3)
            measured = self.noise_hint * magnitude / slope
        return relative, measured

    def _update_value_noise(self, score, value):
        """Store the current point noise for this round's curvature gate.

        Recomputed at every accepted centre: sweep fit error is model error,
        not measurement noise, and must not be frozen into later rounds as if
        it were.

        Args:
            score: Verified score at the accepted centre.
            value: Raw quadratic metric at that same centre.
        """
        metric_value = float(value)
        relative, measured = self._metric_noise(score, value)
        self.value_noise = max(relative, measured, np.finfo(float).eps)
        self.value_noise_by_round.append(dict(
            round=int(self.round + 1), score=float(score),
            metric_value=metric_value, relative_floor=float(relative),
            measured_floor=float(measured), used=float(self.value_noise)))

    def _accept_center(self, score, value):
        """Make the current ``x0`` the only command the solver may return."""
        self.accepted_x = self._quant(self.x0)
        # Keep x0 on the integer command the mirror was driven to, so headroom
        # never reasons about travel that does not exist.
        self.x0 = self.accepted_x.astype(float)
        self.accepted_score = float(score)
        self.accepted_value = float(value)
        self.best_x = self.accepted_x.copy()
        self.best_score = float(score)
        self.last_score = float(score)
        if self._spot_due is not None:
            # Every path that closes a round without keeping its last
            # measurement re-drives the accepted centre, and that command
            # arrives here: this is the frame that shows the round's result.
            self.spot_round = int(self._spot_due)
            self._spot_due = None

    def _close_round(self, outcome, score, fresh):
        """Record one finished round and point at the frame that shows it.

        Args:
            outcome: How the round ended (``accepted``, ``rejected``, ...).
            score: Score of the centre the round leaves standing.
            fresh: Whether the measurement just told() was taken at that
                centre. When it was not, the snapshot waits for the recenter
                measurement that re-drives it.
        """
        self.rounds_completed.append(dict(
            round=int(self.round), outcome=str(outcome), score=float(score),
            bits=self._cmd(self.accepted_x)))
        if fresh:
            self.spot_round = int(self.round)
        else:
            self._spot_due = int(self.round)

    def _begin_round(self, score, value):
        """Use one verified active-axis centre as this round's ``M0``."""
        self._update_value_noise(score, value)
        self._m = {"base": float(value)}
        self._realized.clear()
        self._k = 0
        if self.auto_bias and self._calib and self.round == 0:
            self.stage = "calibrate"
            self._pending = self._bits(*self._calib[0])
            return
        self._start_probe_plan()

    def _queue_next_probe(self):
        """Queue the next probe that actually moves on the integer bit grid.

        A railed actuator can reduce a symmetric modal amplitude to zero.  Such
        a point contains no information: measuring it only remeasures M0 while
        pretending to consume one member of the 2N/N+2 plan.  Record it as
        unavailable and advance locally, before a camera exposure is spent.

        Returns:
            True if a moving probe was queued, otherwise False.
        """
        while self._k < len(self._plan):
            key = self._plan[self._k]
            pending = self._bits(*key)
            if abs(self._effective_amp(*key)) > 0:
                self._pending = pending
                return True
            self._m[key] = float("nan")
            self._k += 1
        return False

    def _active_probe_plan(self):
        """This round's probes: the full plan, less the modes being rationed.

        Returns:
            list: ``(mode, signed factor)`` entries to measure this round. A
                set-aside mode reappears every `_IDLE_MODE_RETRY` rounds so it
                can prove it became solvable again.
        """
        if not self._idle_since:
            return list(self._full_plan)
        rnd = self.round + 1
        due = {mode for mode, since in self._idle_since.items()
               if (rnd - since) % _IDLE_MODE_RETRY == 0}
        plan = [key for key in self._full_plan
                if int(key[0]) not in self._idle_since or int(key[0]) in due]
        # Never hand back an empty round. With every mode set aside and none
        # due, an empty plan would end the solve on "no probe fits within the
        # remaining headroom", which is a different fact entirely.
        return plan or list(self._full_plan)

    def _update_idle_modes(self, coefficients):
        """Ration the probes of modes whose solve keeps returning nothing.

        A zero coefficient means every gate in `_vertex` rejected the mode, so
        its probes bought no correction. Only modes actually probed this round
        are judged, and a set-aside mode returns the moment it solves again.

        Args:
            coefficients: This round's solved coefficient per mode.
        """
        rnd = self.round + 1
        for mode in sorted({int(key[0]) for key in self._plan}):
            if mode == self._scale_mode:
                continue
            if not (mode < len(coefficients) and coefficients[mode] == 0.0):
                if self._idle_since.pop(mode, None) is not None:
                    self.idle_modes.append(
                        dict(mode=mode + 1, round=rnd, action="returned"))
                    self.notes.append(
                        f"round {rnd}: mode {mode + 1} solved again on its "
                        "re-probe; back in every round's probe set")
                self._idle_rounds[mode] = 0
                continue
            self._idle_rounds[mode] += 1
            if (self._idle_rounds[mode] >= _IDLE_MODE_ROUNDS
                    and mode not in self._idle_since):
                self._idle_since[mode] = rnd
                self.idle_modes.append(
                    dict(mode=mode + 1, round=rnd, action="set_aside",
                         idle_rounds=int(self._idle_rounds[mode])))
                self.notes.append(
                    f"round {rnd}: mode {mode + 1} returned no correction "
                    f"{int(self._idle_rounds[mode])} rounds running; probing "
                    f"it every {_IDLE_MODE_RETRY} rounds instead of every "
                    "round until it solves again")

    def _start_probe_plan(self):
        """Start a round, or finish if no mode remains physically probeable."""
        self.stage = "probe"
        self._plan = self._active_probe_plan()
        if self._queue_next_probe():
            return
        self.notes.append(
            f"round {self.round + 1}: every modal probe is unchanged on the "
            "integer command grid; retained the verified centre")
        self._finish_or_verify_observed(
            "no modal probe fits within the remaining actuator headroom")

    def _finish(self, reason):
        """Park on the last verified centre with an explicit stopping reason."""
        self.converged = True
        self.stop_reason = str(reason)
        self._finish_reason_pending = None
        self.stage = "park"
        self.x0 = np.asarray(self.accepted_x, float)
        self._pending = self.accepted_x.copy()

    def _finish_or_verify_observed(self, reason):
        """Before parking, replay a materially better measured command once.

        Modal probes are diagnostic samples, so a single high reading must not
        silently become the output.  They are still real mirror commands,
        though: discarding a repeatably superior one is equally wrong.  At a
        genuine terminal condition, re-drive the best observed command and
        adopt it only if the fresh, selection-free reading still beats the
        accepted centre beyond measured noise.
        """
        margin = self._score_margin(self.accepted_score)
        candidate = np.asarray(self.observed_best_x, int)
        better = (np.isfinite(self.observed_best_score)
                  and self.observed_best_score > self.accepted_score + margin)
        distinct = not np.array_equal(candidate, self.accepted_x)
        if better and distinct:
            self.notes.append(
                f"final check: remeasuring observed best "
                f"{self.observed_best_score:.4f} before parking instead of "
                f"discarding it for accepted {self.accepted_score:.4f}")
            self._finish_reason_pending = str(reason)
            self.x0 = candidate.astype(float)
            self._pending = candidate.copy()
            self.stage = "verify_observed"
            return
        self._finish(reason)

    def _fail(self, reason):
        """Refuse an unidentifiable solve instead of inventing a fallback."""
        self.failed_reason = str(reason)
        self.notes.append(self.failed_reason)
        self.converged = True
        self.stage = "failed"
        self.x0 = np.asarray(self.accepted_x, float)
        self._pending = self.accepted_x.copy()

    def _finite(self, *keys):
        """Values for `keys`, or None when any is missing or not finite."""
        out = []
        for key in keys:
            value = self._m.get(key, float("nan"))
            if not np.isfinite(value):
                return None
            out.append(float(value))
        return out

    def _vertex(self, m_base, m_plus, m_minus, mode):
        """Coefficient from a symmetric three-point parabola, or 0.

        `a = b (M+ - M-) / (2 (M+ - 2 M0 + M-))`. A non-positive denominator
        means the response is not convex, so the mode is left alone: dividing
        by a near-zero curvature turns noise into a large false correction.
        """
        curvature = m_plus - 2.0 * m_base + m_minus
        floor = _CURV_SIGMA * np.sqrt(6.0) * self.value_noise
        if not (curvature > max(0.0, floor)):
            self.notes.append(
                f"mode {mode + 1}: curvature {curvature:.4g} is not above its "
                f"own noise ({floor:.4g}); left alone")
            return 0.0
        b = self._symmetric_amp(mode)
        if not (b > 0):
            self.notes.append(
                f"mode {mode + 1}: quantised probe has zero effective "
                "amplitude; left alone")
            return 0.0
        a = b * (m_plus - m_minus) / (2.0 * curvature)
        # A shallow curvature is usually noise, and dividing by it puts the
        # vertex far outside the three points that were actually measured.
        reach = _VERTEX_REACH * b
        if abs(a) > reach:
            self.notes.append(
                f"mode {mode + 1}: vertex {abs(a) / max(b, 1e-9):.1f} probe "
                f"amplitudes out; clamped to {reach:.2f} rad")
            a = float(np.sign(a) * reach)
        return a

    def tell(self, score, valid=True):
        """Advance the solve with one evaluated measurement.

        Args:
            score: Objective score; used only to track the best point, never in
                the solve itself, which reads `_value`.
            valid: Whether the measurement is usable.
        """
        self.iter += 1
        self.spot_round = None  # One-shot: names THIS measurement only.
        if not valid:
            return
        if self.converged:
            self._park_hold(score)
            return
        self._record_observation(score)
        value = self._value(self._reading)
        if self.stage == "verify_observed":
            reason = self._finish_reason_pending or "modal solve complete"
            previous = self.accepted_score
            margin = self._score_margin(previous)
            if score > previous + margin:
                self._accept_center(score, value)
                self.notes.append(
                    f"final check: observed best reproduced at {score:.4f}; "
                    "retained it")
                self._finish(f"{reason}; verified best measured command retained")
            else:
                self.notes.append(
                    f"final check: observed best replayed at {score:.4f}, not "
                    f"above accepted {previous:.4f}; returning to accepted")
                self.x0 = np.asarray(self.accepted_x, float)
                self._pending = self.accepted_x.copy()
                self.stage = "recenter_final"
            return
        if self.stage == "recenter_final":
            reason = self._finish_reason_pending or "modal solve complete"
            self._accept_center(score, value)
            self._finish(f"{reason}; best measured command did not reproduce")
            return
        if self.stage == "verify":
            # `_line_previous` is the score this ROUND started from. It stops
            # being `accepted_score` the moment a line step banks a better
            # centre, and the round is still judged against where it began.
            if self._line_mode is None:
                self._line_previous = self.accepted_score
            previous = self._line_previous
            margin = self._score_margin(previous)

            if self._line_mode == "extend":
                # Judged against the point already banked THIS round: the only
                # question is "is longer better", and the answer may never lose
                # the shorter step that already worked.
                banked = self.accepted_score
                if score > banked + self._score_margin(banked):
                    self._accept_center(score, value)
                    self._kept_scale = self._correction_scale
                    if self.round < self.rounds and self._line_extend():
                        return
                    self._end_round(previous, score, value, fresh=True)
                    return
                self.notes.append(
                    f"round {self.round}: "
                    f"{100 * self._correction_scale:.0f}% overshot "
                    f"({score:.4f} vs {banked:.4f}); keeping the shorter step")
                self._end_round(previous, banked, self.accepted_value,
                                fresh=False)
                return

            if np.isfinite(previous) and score < previous - margin:
                # The direction cost a whole probe set. Walk it back one image
                # at a time before discarding it.
                if self._line_backtrack():
                    self._line_mode = "backtrack"
                    return
                self._auto_round_decision(previous, score, "rejected")
                self._close_round("rejected", self.accepted_score, False)
                self.notes.append(
                    f"round {self.round}: correction was worse than the "
                    "accepted centre at every scale tried; remeasuring the "
                    "accepted command")
                self.x0 = np.asarray(self.accepted_x, float)
                self._pending = self.accepted_x.copy()
                self.stage = "recenter"
                self._recenter_cause = "rejected_correction"
                self._finish_after_recenter = self.round >= self.rounds
                self._line_mode = None
                return
            improved = np.isfinite(previous) and score > previous + margin
            self._accept_center(score, value)
            self._kept_scale = self._correction_scale
            # Extend on "better" and on "no change": a step that did not move
            # the score was too short to show above the noise.
            if self.round < self.rounds and self._line_extend():
                self._line_mode = "extend"
                self.notes.append(
                    f"round {self.round}: the full step "
                    + ("improved the score" if improved else
                       "moved the mirror without moving the score"))
                return
            self._end_round(previous, score, value, fresh=True)
            return
        if self.stage == "recenter":
            previous = self.accepted_score
            cause = self._recenter_cause
            self._recenter_cause = None
            self._accept_center(score, value)
            if self._finish_after_recenter:
                self._finish_or_verify_observed(
                    (("automatic-round safety maximum completed; "
                      "rejected correction retained the verified centre")
                     if self.auto_rounds else
                     "configured rounds completed; rejected correction "
                     "retained the verified centre"))
            elif (cause == "zero_correction"
                  and self._auto_round_decision(
                      previous, score, "zero_correction")):
                self._finish_or_verify_observed(
                    "automatic rounds converged after two verified quiet "
                    "rounds")
            else:
                self._begin_round(score, value)
            return
        if self.stage == "base":
            # Reached once, for the seed. Every later centre arrives through
            # "verify" or "recenter", which do their own accept/reject: there
            # is no earlier accepted centre here to retreat to.
            self._accept_center(score, value)
            self._begin_round(score, value)
            return
        if self.stage == "calibrate":
            key = self._calib[self._k]
            effective = self._effective_amp(*key)
            self._m[key] = value if abs(effective) > 0 else float("nan")
            self._k += 1
            if self._k < len(self._calib):
                self._pending = self._bits(*self._calib[self._k])
                return
            if not self._choose_bias():
                self._fail(
                    "probe calibration was inconclusive: no tested amplitude "
                    "had a reproducible convex quadratic response")
                return
            # Keep only the baseline: the sweep's points were taken at
            # FRACTIONS of the old amplitude, so they do not answer the probes
            # the solve is about to take at the new one.
            self._m = {"base": self._m["base"]}
            self._k = 0
            self._start_probe_plan()
            return
        key = self._plan[self._k]
        effective = self._effective_amp(*key)
        self._m[key] = value if abs(effective) > 0 else float("nan")
        self._k += 1
        if self._queue_next_probe():
            return
        self._apply()

    def _choose_bias(self):
        """Adopt the largest swept amplitude that is quadratic AND resolved.

        Each candidate is judged on every point inside it, which is what
        separates too-small (noise scatters every point) from too-large (only
        the outer points leave the parabola). Two verdicts, not one: `usable`
        says the response fitted a parabola, `resolved` says the metric moved
        far enough above the measurement noise for that fit to mean anything.
        A rung can pass the first on noise alone -- four scattered points admit
        a parabola, convex half the time -- so the amplitude is chosen from
        both. When they disagree the noise wins: a parabola biased by model
        error still points the right way and the next round corrects it, while
        a probe inside the noise sends the solve somewhere random.
        """
        base = self._m.get("base", float("nan"))
        if not np.isfinite(base):
            self.notes.append("probe calibration: no usable active-axis "
                              "baseline")
            return False
        # The sweep drove mode 1. Use the coefficients its INTEGER commands
        # actually produced, not the requested fractions printed on the UI.
        amp0 = self._headroom_amp(0)
        candidates = list(self._bias_candidates)
        measured_fractions = sorted({abs(s) for _, s in self._calib})
        # Keep the readings themselves, not just the verdicts: the verdict says
        # a rung failed, only the points say whether it was noise or curvature.
        self.bias_points = sorted(
            [(0.0, float(base))]
            + [(self._effective_amp(0, sign * f),
                float(self._m[(0, sign * f)]))
               for f in measured_fractions for sign in (-1.0, 1.0)
               if np.isfinite(self._m.get((0, sign * f), float("nan")))])
        # Re-read the noise: the sweep just fed the loop a dozen measurements,
        # so the split-half sigma is current.
        noise = max(self.value_noise, max(self._metric_noise(
            self.accepted_score, base)), np.finfo(float).eps)
        self.bias_noise = float(noise)
        chosen = None
        for f in candidates:
            points = [(0.0, base)]
            for other in candidates:
                if other > f:
                    continue
                for sign in (-1.0, 1.0):
                    v = self._m.get((0, sign * other))
                    if v is not None and np.isfinite(v):
                        points.append((self._effective_amp(0, sign * other), v))
            entry = self._judge_rung(f, points, noise)
            self.bias_report.append(entry)
            if entry["usable"] and entry["resolved"]:
                chosen = f
        if chosen is None and self._any_convex(self.bias_report):
            # Convex somewhere but nowhere both quadratic and resolved: model
            # error. Retreat to the gentlest rung that cleared the noise.
            floor = [e["fraction"] for e in self.bias_report if e["resolved"]]
            if floor:
                chosen = min(floor)
                self.notes.append(
                    "probe calibration: no amplitude both fitted a parabola "
                    "and cleared the noise; took the smallest that cleared it, "
                    f"{chosen * amp0:.2f} rad")
        if chosen is None:
            # The smallest candidate is unjudgeable; test one smaller recovery
            # amplitude on the two inner guard rungs before refusing.
            if len(self._bias_guards) >= 2:
                guard_choice = max(self._bias_guards)
                points = [(0.0, base)]
                for other in self._bias_guards:
                    for sign in (-1.0, 1.0):
                        v = self._m.get((0, sign * other))
                        if v is not None and np.isfinite(v):
                            points.append((self._effective_amp(
                                0, sign * other), v))
                entry = self._judge_rung(guard_choice, points, noise)
                self.bias_guard_report.append(entry)
                if entry["usable"] and entry["resolved"]:
                    chosen = guard_choice
        if chosen is None:
            # Nothing cleared the noise. Positive curvature is still
            # directional evidence, so continue at the largest swept amplitude.
            convex = self._any_convex(self.bias_report
                                      + self.bias_guard_report)
            if convex and candidates:
                chosen = max(e["fraction"] for e in convex)
                self.notes.append(
                    f"probe calibration: no rung's response reached "
                    f"{_MIN_PROBE_SNR:.0f}x the measurement noise; continuing "
                    f"at the largest convex amplitude, {chosen * amp0:.2f} rad, "
                    "which is the only one with a chance of a real signal")
            else:
                # With no positive curvature there is no modal direction to
                # divide by.  Keep refusing that genuinely unidentifiable case.
                self.notes.append(
                    "probe calibration: no amplitude passed the quadratic test "
                    "and no positive curvature was measured; the modal solve "
                    "was not started")
                return False
        self.bias = float(chosen * amp0)
        self.bias_scanned = self.bias
        self._realized.clear()
        if chosen < max(candidates):
            self.notes.append(
                f"probe calibration: reduced the amplitude from "
                f"{self.bias_requested:.2f} to {self.bias:.2f} rad")
        return True

    @staticmethod
    def _any_convex(reports):
        """Rungs that were judged and came back convex.

        Args:
            reports: `bias_report`-shaped entries; unjudged ones are ignored.

        Returns:
            The convex entries, empty when the sweep saw no positive curvature
            anywhere -- the one case with no modal direction to divide by.
        """
        return [e for e in reports
                if e.get("residual_frac") is not None
                and np.isfinite(float(e["residual_frac"]))
                and e.get("curvature") is not None
                and np.isfinite(float(e["curvature"]))
                and float(e["curvature"]) > 0]

    def _judge_rung(self, fraction, points, noise):
        """Test one swept amplitude for a convex quadratic response.

        Only a rung with a point to spare can be judged for fit quality; `snr`
        is measured even for the rungs too short to fit.

        Args:
            fraction: Amplitude fraction this rung represents.
            points: ``(realised_rad, metric)`` pairs inside the rung.
            noise: Measurement noise on one reading, in metric units.

        Returns:
            One `bias_report` entry, always carrying `usable` and `resolved`.
        """
        realised = [abs(u) for u, _ in points if u != 0]
        m = np.array([p[1] for p in points], float)
        swing = float(np.ptp(m)) if m.size else 0.0
        snr = swing / noise if noise > 0 else float("inf")
        entry = dict(fraction=float(fraction),
                     amplitude_rad=(max(realised) if realised else 0.0),
                     points=len(points), swing=swing, noise=float(noise),
                     snr=float(snr),
                     resolved=bool(snr >= _MIN_PROBE_SNR))
        if len(points) < 4:
            entry.update(residual_frac=None, usable=False,
                         reason="too few points to test the fit")
            return entry
        u = np.array([p[0] for p in points])
        design = np.column_stack([np.ones_like(u), u, u ** 2])
        coef, *_ = np.linalg.lstsq(design, m, rcond=None)
        worst = float(np.max(np.abs(m - design @ coef)))
        frac = worst / swing if swing > 0 else float("inf")
        ok = bool(np.isfinite(frac) and frac <= self.tolerance and coef[2] > 0)
        entry.update(residual_frac=round(float(frac), 5),
                     curvature=float(coef[2]), usable=ok,
                     # c0 + c1 u + c2 u^2, so the fit can be redrawn later.
                     fit=[float(c) for c in coef],
                     resid_rms=float(np.sqrt(np.mean(
                         (m - design @ coef) ** 2))),
                     reason=("" if ok and entry["resolved"] else
                             "not convex" if not coef[2] > 0 else
                             "departs from a parabola" if not ok else
                             f"swing {swing:.4g} is only {snr:.1f}x the "
                             f"measurement noise"))
        return entry

    def _probe_reserve(self):
        """Bits per actuator one probe of every solved mode would need.

        `_headroom_amp` asks for ``amp * |ctrl[:, m]|`` bits of symmetric
        room per mode; a destination that leaves less can no longer probe.

        Returns:
            tuple: (all-modes reserve, cheapest-mode reserve), per actuator.
                The second is the weaker promise `_fit_step` falls back on.
        """
        cols = [self._requested_amp(mode)
                * np.abs(np.asarray(self.ctrl[:, mode], float))
                for mode in range(self.n_modes)]
        if not cols:
            zero = np.zeros_like(np.asarray(self.x0, float))
            return zero, zero
        stack = np.vstack(cols)
        return np.max(stack, axis=0), np.min(stack, axis=0)

    def _fit_step(self, step):
        """Largest fraction that stays in range and preserves probe headroom.

        Scaled, not clipped per actuator: clipping one component leaves a shape
        that is no longer a combination of the calibrated modes, which is how a
        solve strands itself on a rail with a spot worse than it started.

        What is reserved at the destination boundary is one PROBE, not one
        modal grid step. Reserving the grid step reserved 1 bit while the next
        round's probes needed ~1700, so a correction was free to land where
        every mode was unprobeable -- and a round that cannot probe solves
        zero, which leaves no command that walks back off the rail. Three
        recorded DM5 runs died exactly there, at 0.06-0.13 absolute against
        0.19 for the same settings that never touched a rail.
        """
        room = np.where(step > 0, self.hi - self.x0, self.x0 - self.lo)
        big = np.abs(step) > 1e-9
        if not np.any(big):
            return 1.0
        grid = float(max(1, int(getattr(self.cfg, "modal_bit_step", 1))))
        full, cheapest = self._probe_reserve()

        def fit_for(reserve):
            left = np.maximum(0.0, room - np.maximum(grid, reserve))
            return float(min(1.0, _STEP_FILL
                             * np.min(left[big] / np.abs(step[big]))))

        fit = fit_for(full)
        if fit > 0.0:
            return fit
        # Holding a probe of every mode would forbid the move outright, so
        # the promise degrades to keeping the cheapest mode probeable.
        fit = fit_for(cheapest)
        self.notes.append(
            f"round {self.round + 1}: cannot hold a probe of every mode and "
            f"still move; reserving the cheapest mode only, step at "
            f"{100 * fit:.0f}%")
        return fit

    def _apply(self):
        """Subtract the solved aberration, then start the next round."""
        # The one expensive shot of a modal round: every other point in the
        # 2N+1 / N+2 sequence only stores a number, so a modal run's decision
        # cost is a spike here and ~0 elsewhere. Timed separately to say so.
        with self.phase("modal_solve"):
            a = np.asarray(self._solve(), float)
        self.last_coefficients = [float(v) for v in a]
        self._update_idle_modes(a)
        # Minus because the solve reports the aberration PRESENT and the
        # correction is its negative. Column i of ctrl is the bit offset for one
        # radian of mode i, so the whole correction is one matrix-vector product.
        step = -(self.ctrl[:, :len(a)] @ a)
        fit = self._fit_step(step)
        if fit < 1.0:
            self.notes.append(f"round {self.round + 1}: correction scaled to "
                              f"{100 * fit:.0f}% to stay in range and retain "
                              "probe headroom")
        self._correction_origin = self._quant(self.x0).astype(float)
        self._correction_step = fit * step
        self._correction_scale = 1.0
        self._kept_scale = 1.0
        self._line_steps = 0
        self._line_mode = None
        candidate = self._quant(
            self._correction_origin + self._correction_step)
        actual_step = candidate.astype(float) - self._correction_origin
        self._last_correction_max_bit = (float(np.max(np.abs(actual_step)))
                                         if actual_step.size else 0.0)
        self.x0 = candidate.astype(float)
        self._realized.clear()
        self.round += 1
        self._m = {}
        grid = max(1, int(getattr(self.cfg, "modal_bit_step", 1)))
        if self._last_correction_max_bit < grid:
            self._close_round("zero_correction", self.accepted_score, False)
            if self.round >= self.rounds:
                self._finish_or_verify_observed(
                    (("automatic-round safety maximum completed; correction "
                      "was below the modal command grid")
                     if self.auto_rounds else
                     "configured rounds completed; correction was below the "
                     "modal command grid"))
            else:
                # Re-measure the accepted centre. Fixed mode spends the exact
                # requested count; automatic mode uses this independent repeat
                # as one half of its two-round quiet confirmation.
                self.notes.append(
                    f"round {self.round}: correction was below the modal "
                    "command grid; remeasuring the accepted centre")
                self.x0 = np.asarray(self.accepted_x, float)
                self._pending = self.accepted_x.copy()
                self.stage = "recenter"
                self._recenter_cause = "zero_correction"
                self._finish_after_recenter = False
            return
        # Every predicted correction is measured before it can become the next
        # centre.  The verified point is either accepted and reused as the next
        # M0, or rejected and explicitly recentered.
        self.stage = "verify"
        self._pending = self._quant(self.x0)

    def _skip_probe(self, key):
        """Mark one unmeasurable probe and advance without fitting it.

        Args:
            key: ``(mode, signed_factor)`` entry currently being measured.

        Returns:
            Recovery action dictionary for the UI event log.
        """
        self._m[key] = float("nan")
        self._k += 1
        if self.stage == "calibrate":
            if self._k < len(self._calib):
                self._pending = self._bits(*self._calib[self._k])
            else:
                if not self._choose_bias():
                    self._fail(
                        "probe calibration was inconclusive after ROI "
                        "recovery: no tested amplitude remained usable")
                    return dict(action="stop_calibration", recovered=False,
                                message=self.failed_reason)
                self._m = {"base": self._m["base"]}
                self._k = 0
                self._start_probe_plan()
            return dict(action="skip_calibration", recovered=True,
                        message="oversized calibration rung skipped")
        if self._k < len(self._plan):
            if not self._queue_next_probe():
                self._apply()
        else:
            self._apply()
        return dict(action="skip_probe", recovered=True,
                    message=f"mode {key[0] + 1} skipped after ROI retries")

    def _restart_mode_probe(self, mode):
        """Restart every probe for one mode after changing its amplitude.

        A measured positive point and a retried negative point at half the
        amplitude are not a symmetric parabola.  Discard any earlier point for
        this mode and start its local probe group again; measurements from
        completed modes remain valid.

        Args:
            mode: Zero-based eigenmode index whose scale changed.

        Returns:
            True when at least one earlier point was discarded.
        """
        first = min(i for i, key in enumerate(self._plan)
                    if int(key[0]) == int(mode))
        restarted = first < self._k
        for key in tuple(self._m):
            if (isinstance(key, tuple) and len(key) == 2
                    and int(key[0]) == int(mode)):
                del self._m[key]
        self._k = first
        if not self._queue_next_probe():
            self._apply()
        return restarted

    def recover_roi_miss(self):
        """Recover when the current spot outgrows the fixed moment ROI.

        The ROI itself remains frozen. Probe points are retried at half their
        per-mode amplitude; a correction is retried at half its step and then
        falls back toward a previously measured command. No metric value from
        the overflowing frame enters the quadratic fit.

        Returns:
            An action dictionary when recovery is possible, otherwise ``None``.
        """
        if self.stage == "calibrate":
            key = self._calib[self._k]
            self.notes.append(
                f"probe calibration: {abs(key[1]):.2f}x rung exceeded the "
                "fixed ROI and was excluded")
            return self._skip_probe(key)
        if self.stage == "probe":
            key = self._plan[self._k]
            mode = int(key[0])
            if self._roi_probe_retries[mode] < _ROI_PROBE_RETRIES:
                before = self._quant(self.x0)
                self._roi_probe_retries[mode] += 1
                self._probe_scale[mode] *= _ROI_SCALE
                self._realized.clear()
                restarted = self._restart_mode_probe(mode)
                candidate = self._pending
                if not np.array_equal(candidate, before):
                    retry_key = self._plan[self._k]
                    used = abs(self._effective_amp(*retry_key))
                    self.notes.append(
                        f"mode {mode + 1}: fixed ROI exceeded; retrying at "
                        f"{used:.3f} realised rad")
                    return dict(
                        action="retry_probe", recovered=True, mode=mode + 1,
                        amplitude_rad=used,
                        message=(f"mode {mode + 1} probe halved and "
                                 + ("both signs re-measured" if restarted else
                                    "retried")))
            self.notes.append(
                f"mode {mode + 1}: fixed ROI still exceeded after "
                f"{int(self._roi_probe_retries[mode])} retries; left alone")
            return self._skip_probe(key)
        if self.stage in ("base", "verify") and np.any(
                np.abs(self._correction_step) > 1e-9):
            self._correction_scale *= _ROI_SCALE
            candidate = self._quant(
                self._correction_origin
                + self._correction_scale * self._correction_step)
            current = self._quant(self.x0)
            if (self._correction_scale >= _ROI_SCALE ** _ROI_PROBE_RETRIES
                    and not np.array_equal(candidate, current)
                    and not np.array_equal(candidate,
                                           self._quant(self._correction_origin))):
                self.x0 = candidate.astype(float)
                self._pending = candidate
                actual = candidate.astype(float) - self._correction_origin
                self._last_correction_max_bit = (
                    float(np.max(np.abs(actual))) if actual.size else 0.0)
                self._realized.clear()
                self.notes.append(
                    f"round {self.round}: correction exceeded the fixed ROI; "
                    f"retrying at {100 * self._correction_scale:.0f}%")
                return dict(
                    action="retry_correction", recovered=True,
                    correction_scale=self._correction_scale,
                    message=("candidate correction halved and re-measured"))
            self.x0 = np.array(self.best_x, float)
            self._pending = self._quant(self.x0)
            self._last_correction_max_bit = 0.0
            self._realized.clear()
            self.notes.append(
                f"round {self.round}: correction could not fit the fixed ROI; "
                "rolled back to the best measured command")
            return dict(action="rollback_correction", recovered=True,
                        message="candidate correction rolled back to best")
        return None

    def status(self):
        st = super().status()
        st.update(round=self.round, rounds=self.rounds, modes=self.n_modes,
                  rounds_completed=list(self.rounds_completed),
                  spot_round=self.spot_round,
                  auto_rounds=self.auto_rounds,
                  auto_min_rounds=_AUTO_MIN_ROUNDS,
                  auto_confirm_rounds=_AUTO_CONFIRM_ROUNDS,
                  auto_quiet_rounds=self._auto_quiet_rounds,
                  auto_command_threshold_bit=self._auto_command_threshold(),
                  auto_round_report=list(self.auto_round_report),
                  clipped=self.clipped, notes=list(self.notes),
                  failed_reason=self.failed_reason,
                  stop_reason=self.stop_reason,
                  mode_resolution=dict(self.mode_resolution),
                  accepted_score=self.accepted_score,
                  accepted_bits=self._cmd(self.accepted_x),
                  observed_best_score=self.observed_best_score,
                  observed_best_bits=self._cmd(self.observed_best_x),
                  value_noise=float(self.value_noise),
                  value_noise_by_round=list(self.value_noise_by_round),
                  last_coefficients=list(self.last_coefficients),
                  bias_guard_report=list(self.bias_guard_report),
                  bias_rad=self.bias, bias_scanned=self.bias_scanned,
                  bias_points=list(self.bias_points),
                  bias_noise=self.bias_noise,
                  probe_amplitudes_rad=[self._requested_amp(i)
                                        for i in range(self.n_modes)],
                  idle_modes=list(self.idle_modes),
                  idle_mode_rounds=_IDLE_MODE_ROUNDS,
                  idle_mode_retry=_IDLE_MODE_RETRY,
                  roi_probe_retries=[int(v) for v in self._roi_probe_retries])
        return st


class ModalFit(_Modal):
    """2N+1: measure a parabola per mode, assuming nothing about its curvature.

    The safe default: every number it divides by was measured this run, so a
    stale matrix can only affect the mode shapes, never the scale.
    """

    def _value(self, reading):
        return float(getattr(reading, "second_moment", float("nan")))

    def _probe_plan(self):
        # Interleaved +/- per mode, so the two probes of one parabola are
        # adjacent in time and slow drift largely cancels.
        return [(i, s) for i in range(self.n_modes) for s in (+1, -1)]

    def _solve(self):
        out = np.zeros(self.n_modes)
        for i in range(self.n_modes):
            got = self._finite("base", (i, +1), (i, -1))
            if got is None:
                self.notes.append(f"mode {i + 1}: unmeasurable, left alone")
                continue
            out[i] = self._vertex(got[0], got[1], got[2], i)
        return out


class ModalFast(_Modal):
    """N+2: reuse driven-coordinate curvature from the impact matrix.

    Knowing the curvature leaves only a linear term to measure, so one probe
    per mode plus a baseline would be N+1. The overall camera/optics scale
    `kappa` is the one factor the matrix cannot supply, so one extra negative
    probe gives mode 1 a full parabola and measures it. Hence N+2.  The stored
    relative curvature is `s_i^2 ||ctrl_i||^2`, not bare `s_i^2`, because the
    loop drives independently RMS-normalised control columns.
    """

    # Mode 1 calibrates kappa for every other mode, so it is probed every
    # round however often it comes back idle.
    _scale_mode = 0

    def _value(self, reading):
        return float(getattr(reading, "second_moment", float("nan")))

    def _probe_plan(self):
        # Mode 1 takes both signs -- it calibrates kappa. The rest take one.
        return [(0, +1), (0, -1)] + [(i, +1) for i in range(1, self.n_modes)]

    def _requested_amp(self, mode):
        """Equalise the matrix-predicted second-moment probe signal.

        Args:
            mode: Zero-based eigenmode index.

        Returns:
            Requested amplitude in radians RMS. Strong-curvature modes use a
            smaller coefficient, so one reference amplitude does not make the
            expensive low-singular-value modes overgrow the camera ROI.
        """
        g = np.asarray(self.matrix.curvature, float)[:self.n_modes]
        if not (g[0] > 0 and g[mode] > 0):
            relative = 1.0
        else:
            relative = float(np.sqrt(g[0] / g[mode]))
        return float(self.bias * relative * self._probe_scale[mode]
                     * self._trust_scale[mode])

    def _solve(self):
        out = np.zeros(self.n_modes)
        g = np.asarray(self.matrix.curvature, float)[:self.n_modes]
        anchor = self._finite("base", (0, +1), (0, -1))
        if anchor is None:
            self.notes.append("mode 1 unmeasurable, so the scale could not be "
                              "calibrated; no correction applied this round")
            return out
        m_base, m_plus, m_minus = anchor
        out[0] = self._vertex(m_base, m_plus, m_minus, 0)
        curvature = m_plus - 2.0 * m_base + m_minus
        if not (curvature > 0) or not (g[0] > 0):
            self.notes.append("scale calibration failed; only mode 1 corrected")
            return out
        # curvature/(2 b^2) is kappa*g_1; dividing by the matrix's own g_1
        # isolates kappa. Each mode is probed at its own amplitude b.
        b0 = self._symmetric_amp(0)
        if not (b0 > 0):
            self.notes.append("mode 1 quantised to zero; scale unavailable")
            return out
        kappa = curvature / (2.0 * b0 ** 2) / g[0]
        for i in range(1, self.n_modes):
            got = self._finite("base", (i, +1))
            if got is None or not (g[i] > 0):
                self.notes.append(f"mode {i + 1}: unmeasurable, left alone")
                continue
            c_i = kappa * g[i]
            b = abs(self._effective_amp(i, +1.0))
            if not (b > 0):
                self.notes.append(f"mode {i + 1}: no headroom to probe, left "
                                  "alone")
                continue
            # M_i - M0 = c_i (2 a b + b^2) -> a = (M_i - M0)/(2 c_i b) - b/2
            out[i] = (got[1] - got[0]) / (2.0 * c_i * b) - b / 2.0
        return out


class ModalPsd(ModalFit):
    """2N+1 on `G = 1/g`, the Debarre/Booth merit function.

    Same machinery as `ModalFit`; only the quantity differs. The PSD band reads
    larger for a better spot and it is its reciprocal that is quadratic
    (Debarre, Booth & Wilson, Opt. Express 15, 8176, 2007, Eq. 30).

    Quadratic only to leading order, over a range set by the band -- run Band
    Scan first.
    """

    def _value(self, reading):
        g = float(getattr(reading, "psd_band", float("nan")))
        # A vanishing band value would send the reciprocal to infinity and
        # poison the parabola. Report it as unmeasurable instead.
        return 1.0 / g if np.isfinite(g) and g > 0 else float("nan")
