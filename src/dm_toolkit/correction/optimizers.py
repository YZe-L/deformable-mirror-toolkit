# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 3.16, 2026-08-25

"""The optimisers, behind one tiny interface."""

from __future__ import annotations

import collections
import dataclasses
import importlib
import threading
import time

import numpy as np

from . import budget as B
from . import settings as S

# Heavy optional packages behind each algorithm's lazy __init__ import.
_HEAVY_IMPORTS = {S.ALGO_CMAES: "cma", S.ALGO_BO: "skopt"}
_warm_started = set()

# Sigmas of measured per-point noise a gain must clear: one false accept per
# ~177-point run gives z = 2.53, k = z*sqrt(2) = 3.6.
_GATE_SIGMA = 3.6
# Used only until the first split-half sigma exists, i.e. after one point. Not
# scaled by the score: a floor that grows with the score is what kept the
# measured value from ever being used.
_COLD_GATE = 0.002


def warm_imports(algos):
    """Load an optimizer's optional dependency in a background thread."""
    names = [_HEAVY_IMPORTS[a] for a in algos
             if a in _HEAVY_IMPORTS and _HEAVY_IMPORTS[a] not in _warm_started]
    if not names:
        return
    _warm_started.update(names)

    def _load():
        for n in names:
            try:
                importlib.import_module(n)
            except Exception:  # noqa: BLE001 -- reported at use
                pass

    threading.Thread(target=_load, daemon=True,
                     name="optimizer-import-warm").start()


def make_optimizer(cfg: S.LoopSettings, matrix=None):
    """Factory: build the optimiser named in the settings.

    Args:
        cfg: Loop settings.
        matrix: Stored impact matrix, required by the modal algorithms and
            ignored by the searches.

    Returns:
        An optimiser, wrapped in `Staged` when a modal solve is configured to
        hand over to a search afterwards.

    Raises:
        ImportError: If CMA-ES or Bayesian is chosen without its package.
        ValueError: If a modal algorithm is chosen with no stored matrix.
        MatrixMismatch: If the matrix has the wrong actuators or AO wavelength.
    """
    if cfg.algorithm in S.MODAL_ALGOS:
        # Imported here, not at module scope: modal.py subclasses _Base from
        # this module, so a top-level import would close the cycle.
        from . import modal as MODAL

        if matrix is None:
            raise ValueError(
                f"{S.ALGO_LABELS.get(cfg.algorithm, cfg.algorithm)} needs a "
                "measured impact matrix; run the influence-matrix pipeline first")
        matrix.require_wavelength(cfg.wavelength_nm)
        builder = {S.ALGO_MODAL_FIT: MODAL.ModalFit,
                   S.ALGO_MODAL_FAST: MODAL.ModalFast,
                   S.ALGO_MODAL_PSD: MODAL.ModalPsd}[cfg.algorithm]
        solver = builder(cfg, matrix)
        if cfg.polish_algorithm and cfg.polish_algorithm != S.POLISH_NONE:
            return Staged(cfg, solver, cfg.polish_algorithm, cfg.polish_metric)
        return solver
    if cfg.algorithm == S.ALGO_WFS:
        # Imported here for the same reason as `modal`: it subclasses _Base.
        from . import wavefront as WFS

        # `matrix` is optional here, unlike for the modal solves: it only
        # confirms the session file's column order (see `_Plant.load`).
        solver = WFS.PseudoWfs(cfg, matrix)
        if cfg.polish_algorithm and cfg.polish_algorithm != S.POLISH_NONE:
            return Staged(cfg, solver, cfg.polish_algorithm, cfg.polish_metric)
        return solver
    if cfg.algorithm == S.ALGO_SPGD:
        return SPGD(cfg)
    if cfg.algorithm == S.ALGO_GENETIC:
        return Genetic(cfg)
    if cfg.algorithm == S.ALGO_CMAES:
        return CMAES(cfg)
    if cfg.algorithm == S.ALGO_BO:
        return BayesOpt(cfg)
    if cfg.algorithm == S.ALGO_SA:
        return SimAnneal(cfg)
    return HillClimb(cfg)


class _Phase:
    """Stopwatch for one named block inside a decision call (see _Base.phase).

    Accumulates rather than overwrites, so a phase entered twice in one call
    (a retry, a per-mode loop) reports the total that call actually spent.
    """

    __slots__ = ("_opt", "_name", "_t0")

    def __init__(self, opt, name):
        self._opt, self._name, self._t0 = opt, name, 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc):
        marks = getattr(self._opt, "_phase_ms", None)
        if marks is None:
            marks = self._opt._phase_ms = {}
        ms = (time.perf_counter() - self._t0) * 1e3
        marks[self._name] = marks.get(self._name, 0.0) + ms
        return False


class _Base:
    def __init__(self, cfg: S.LoopSettings):
        self.cfg = cfg
        self.acts = cfg.actuators
        self.lo = np.array([a.bit_min for a in self.acts], float)
        self.hi = np.array([a.bit_max for a in self.acts], float)
        self.x = np.array([a.start for a in self.acts], float)  # Working point
        # Command grid, per axis. A joint two-mirror run gives each channel the
        # grid its own mirror was tuned with; every other run gets one value
        # repeated, which is exactly the scalar this used to be.
        self._min_step = np.maximum(1, self._per_channel("min_step")
                                    .astype(int)).astype(float)
        self.best_x = self._quant(self.x)
        self.best_score = -np.inf
        self.last_score = -np.inf
        self.converged = False
        self.iter = 0
        self._pending = self._quant(self.x)
        self.noise_hint = 0.0  # Measured per-point score sigma (from tab)
        # Set by `Staged` on a polish stage: a search from a solved point must
        # not read "little gain" as being in the wrong basin.
        self.warm_start = False

    def _per_channel(self, name):
        """One knob as a per-actuator vector.

        Bit-scale knobs (step sizes, dither and mutation widths) mean nothing
        away from the mirror they were tuned on, so a joint run carries one
        value per channel in `cfg.channel_knobs`. Everything else -- and every
        single-mirror run -- reads the scalar field, repeated, so the arithmetic
        below is identical to the scalar form it replaced.

        Args:
            name: Field name of the knob.

        Returns:
            np.ndarray of one value per actuator, in `self.acts` order.
        """
        return np.array([float(self.cfg.knob_for(a.channel, name))
                         for a in self.acts], float)

    def set_noise(self, sigma):
        """Latest measured per-point score noise (std, final-score units).

        Optimisers use it to gate what counts as a real improvement.
        """
        if np.isfinite(sigma):
            self.noise_hint = max(0.0, float(sigma))

    def _measured(self, k, floor):
        """`k` sigma of the MEASURED noise; the constant only before one exists.

        The loop scores each half of every point's frames, so a measured
        sigma exists from the first point; the constant covers only the
        cold start.

        Args:
            k: Multiplier on the measured sigma for this particular decision.
            floor: Cold-start value, used only until the first measurement.

        Returns:
            The gate, in score units.
        """
        if self.noise_hint > 0:
            return float(k) * self.noise_hint
        return float(floor)

    def noise_gate(self):
        """The accept gate this optimizer is using now, for the run record."""
        return float(self._noise_level())

    def _noise_level(self):
        """Default gate for optimizers that do not define their own."""
        return self._measured(_GATE_SIGMA, _COLD_GATE)

    def precision(self):
        """How precise the NEXT point has to be (see `core.budget`).

        Asked once per point, before it is driven, in the same place `ask`
        supplies the command. The default asks for the finest budget on
        offer, which is what the operator typed -- so an optimiser that does
        not override this behaves exactly as it did before measure speed
        became automatic, and only the algorithms that have been given a
        derivation of their own decision contrast go faster.

        Returns:
            A `budget.PrecisionRequest`.
        """
        return B.PrecisionRequest(delta=0.0)

    def observe(self, reading):
        """Hand over the whole measurement, before the score reaches `tell`.

        A no-op for the searches, which only compare scores. The modal solvers
        override it: they need the metric's raw value, not the 0..1 score whose
        normalisation would destroy the quadratic form they invert.

        Args:
            reading: The `metrics.SpotReading` the next `tell` describes.
        """

    @property
    def metric(self):
        """Metric whose score this optimiser is currently being told.

        Fixed for a plain optimiser; the staged one changes it at handover, so
        the caller must read this rather than assume `cfg.metric`.
        """
        return self.cfg.metric

    # Bit-grid helpers
    def _quant(self, v):
        step = self._min_step
        v = np.round(np.asarray(v, float) / step) * step
        return np.clip(v, self.lo, self.hi).astype(int)

    def _cmd(self, v):
        return {a.channel: int(b) for a, b in zip(self.acts, v)}

    # Driver interface
    def ask(self):
        return self._cmd(self._pending)

    def best_command(self):
        return self._cmd(self.best_x)

    def _note_best(self, score):
        self.last_score = score
        if score > self.best_score:
            self.best_score = score
            self.best_x = self._pending.copy()

    def _park_hold(self, score):
        """Converged: sit on the best bits, tracking an HONEST best_score.

        Raises are taken immediately; lower parked readings pull best down
        slowly (EMA). best_score is the max of noisy readings, so without the
        down-correction a lucky spike would sit above every honest re-reading
        and distort the disturbance detector's reference forever.
        """
        self.last_score = score
        if score > self.best_score:
            self.best_score = score
            self.best_x = self._pending.copy()
        else:
            self.best_score += 0.1 * (score - self.best_score)
        self._pending = self.best_x.copy()

    # Decision-cost instrumentation
    def phase(self, name):
        """Time one internal step of a decision call.

        The expensive part of a decision is algorithm-specific and invisible
        from the caller: a Bayesian tell() is a GP refit that grows with the
        history, a genetic tell() is free except at a generation boundary, a
        modal observe() is free except on the shot that runs the solve. Wrap
        those blocks in `with self.phase("gp_fit"):` and the loop writes
        them to decision_timing.csv.

        Args:
            name: Phase label, owned by the algorithm.
        """
        return _Phase(self, name)

    def take_phase_timing(self):
        """Return {phase: ms} recorded since the last call, and clear them."""
        marks = getattr(self, "_phase_ms", None)
        self._phase_ms = {}
        return marks or {}

    def status(self):
        # `gen` is the completed-generation count for GA and CMA-ES, None for
        # the rest.
        return dict(algorithm=self.cfg.algorithm, iter=self.iter,
                    converged=self.converged, best_score=self.best_score,
                    last_score=self.last_score,
                    stage=getattr(self, "stage", "search"),
                    gen=getattr(self, "gen", None),
                    best_bits=self._cmd(self.best_x))


class HillClimb(_Base):
    """Best-anchored coordinate descent with a greedy line-search.

    Every probe is taken from the BEST-known point, never from a free-running
    centre. A probe is adopted only if it beats best by more than the noise
    floor. Once converged it parks quietly on the best bits; disturbance
    recovery is handled by the UI's persistent-drop detector, not by periodic
    pokes that visibly shake the spot.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._move_step = self._per_channel("move_step")
        self.step = np.maximum(self._min_step, self._move_step)
        self.axis = 0
        self.stage = "seed"  # Seed -> minus -> plus -> chase, per axis.
        self._probe = {}
        self._improved = False
        self._dir = 1  # Winning direction during a line-search.
        self._mode_queue = []  # Coupled multi-actuator probes after a stall.
        self._mode_i = 0
        self.rng = np.random.default_rng()
        self._escape_used = False  # One broad non-greedy basin search per run.
        self._escape_queue = []
        self._escape_i = 0
        self._escape_improved = False
        self._escape_ref = np.nan  # best_score when the escape bank started.
        self._verify_retries = 0  # Sweeps retried after a deflated best.
        self._seed_score = np.nan  # First measured score (escape gate)
        # How far recent probes landed from best and how precise they were:
        # together they set the precision this climb needs (see `precision`).
        span = max(6, 2 * len(self.acts))
        self._probe_gap = collections.deque(maxlen=span)
        self._probe_delta = collections.deque(maxlen=span)

    def _axis_vec(self, sign):
        """A probe one `step` along the current axis -- always FROM the best."""
        v = self.best_x.astype(float).copy()
        v[self.axis] += sign * self.step[self.axis]
        return self._quant(v)

    def _noise_level(self):
        """Smallest score gain that counts as real.

        An explicit noise_threshold wins; otherwise gate on the MEASURED
        per-point score sigma fed in via set_noise, at `_GATE_SIGMA`,
        falling back to the `_COLD_GATE` floor until that sigma exists.
        """
        if self.cfg.noise_threshold > 0:
            return float(self.cfg.noise_threshold)
        return self._measured(_GATE_SIGMA, _COLD_GATE)

    def _adopt(self, sign, score):
        """Return adopt.

        A probe beat best -> make it the new best and chase one further out.

        Args:
            sign: Candidate orientation sign.
            score: Objective score for the evaluated point.
        """
        self._verify_retries = 0  # Real progress resets the cap.
        new = self._axis_vec(sign)
        # Pinned at bound.
        if int(new[self.axis]) == int(self.best_x[self.axis]):
            self._advance_axis()
            return
        self.best_x = new  # Record the new best bits.
        self.best_score = score
        self._improved = True
        self._dir = sign
        self.stage = "chase"
        self._pending = self._axis_vec(sign)  # One further from the NEW best.

    def _mode_patterns(self):
        """Return mode patterns.

        A few coupled probes for mirrors whose useful directions are not
        aligned with individual piezos. No influence matrix required.
        """
        n = len(self.acts)
        if n < 2:
            return []
        pats = []
        all_one = np.ones(n, float)
        alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
        pats.extend([all_one, -all_one, alt, -alt])
        return pats

    def _start_mode_escape(self):
        base = self.best_x.astype(float)
        seen = {tuple(self.best_x.tolist())}
        self._mode_queue = []
        for pat in self._mode_patterns():
            v = self._quant(base + self.step * pat)
            key = tuple(v.tolist())
            if key not in seen:
                seen.add(key)
                self._mode_queue.append(v)
        self._mode_i = 0
        if not self._mode_queue:
            return False
        self.stage = "mode"
        self._pending = self._mode_queue[0]
        return True

    def _start_global_escape(self):
        """Build a finite bank of broad probes when greedy local moves stall.

        A coordinate climb normally refuses to walk through a worse first step,
        so an optimum 600+ bits away can be invisible even though the UI looks
        like it is searching. This stage evaluates distant points directly:
        coarse one-axis lines across every actuator range plus joint random
        moves at several radii. It runs once, then hands the best basin back to
        the ordinary fine climb.
        """
        if self._escape_used:
            return False
        if self.warm_start:
            # Polishing a solved point. The bank below drives every actuator
            # to its rails and samples the whole box, which from a good start
            # is the "big jump" it exists to make from a bad one.
            self._escape_used = True
            return False
        # Use escape probes only when relative improvement from the seed is
        # small.
        gain = self.best_score - self._seed_score
        scale = max(abs(self._seed_score), abs(self.best_score), 1e-9)
        if np.isfinite(self._seed_score) and gain > 0.10 * scale:
            self._escape_used = True
            return False
        self._escape_used = True
        base = self.best_x.astype(float).copy()
        seen = {tuple(self.best_x.tolist())}
        queue = []

        # Full-range one-axis scan. The 4*move_step pitch bounds the
        # measurement budget while still reaching rails a greedy +/-move_step
        # never sees; the fine climb that follows covers the gaps anyway.
        pitches = np.maximum(self._min_step, 4 * self._move_step)
        for axis in range(len(self.acts)):
            pitch = float(pitches[axis])
            values = np.arange(self.lo[axis], self.hi[axis] + 0.5 * pitch,
                               pitch, dtype=float)
            values = np.append(values, self.hi[axis])
            for value in values:
                v = base.copy()
                v[axis] = value
                q = self._quant(v)
                key = tuple(q.tolist())
                if key not in seen:
                    seen.add(key)
                    queue.append(q)

        # Rail-corner patterns: coupled mode directions driven full range,
        # for a mirror whose useful shape sits at a corner of the box.
        span = self.hi - self.lo
        for pat in self._mode_patterns():
            q = self._quant(base + span * pat)
            key = tuple(q.tolist())
            if key not in seen:
                seen.add(key)
                queue.append(q)

        # Joint moves cross coupled valleys. Include local/mid/global radii,
        # then a few full-box samples for basins far from the starting shape.
        n = len(self.acts)
        joint_n = max(8, 2 * n)
        # The first two radii are per axis, so a joint run steps each mirror at
        # its own scale along the shared random direction; the third is set by
        # the command box, which both mirrors share.
        radii = (2.0 * self._move_step,
                 4.0 * self._move_step,
                 0.35 * float(np.mean(self.hi - self.lo)))
        for i in range(joint_n):
            direction = self.rng.normal(size=n)
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            v = base + radii[i % len(radii)] * direction
            if i >= joint_n - max(4, n):
                v = self.lo + self.rng.random(n) * (self.hi - self.lo)
            q = self._quant(v)
            key = tuple(q.tolist())
            if key not in seen:
                seen.add(key)
                queue.append(q)

        if not queue:
            return False
        self._escape_queue = queue
        self._escape_i = 0
        self._escape_improved = False
        self._escape_ref = float(self.best_score)
        self.stage = "escape"
        self._pending = queue[0]
        return True

    def _finish_failed_sweep(self):
        """Return finish failed sweep.

        A full sweep failed: re-measure the best point BEFORE shrinking or
        converging. best_score is the max of noisy readings (winner's curse),
        so after enough probes it drifts above the true value and then nothing
        can beat it -- the loop would 'converge' while the spot is still bad.
        The verify reading is selection-free; tell() decides what follows.
        """
        self.stage = "verify"
        self._pending = self.best_x.copy()

    def _sweep_done(self):
        """Honest stall confirmed: escape once, else shrink, else converge."""
        if self._start_global_escape():
            return
        if np.all(self.step <= self._min_step):
            self.converged = True
            self.stage = "park"
            self._pending = self.best_x.copy()
            return
        self.step = np.maximum(self._min_step, self.step // 2)
        self._improved = False
        self.axis, self.stage = 0, "minus"
        self._pending = self._axis_vec(-1)

    def _advance_axis(self):
        """Return advance axis.

        Finish this actuator; on a full fruitless sweep, shrink or converge.
        """
        self.axis += 1
        if self.axis >= len(self.acts):
            self.axis = 0
            if not self._improved:  # A full sweep beat nothing.
                if (np.any(self.step > 2 * self._min_step)
                        and self._start_mode_escape()):
                    return
                self._finish_failed_sweep()
                return
            self._improved = False
        self.stage = "minus"
        self._pending = self._axis_vec(-1)

    def _note_contrast(self, score):
        """Record how far this probe landed from best, and how well it was seen.

        The measurement's own resolution is recovered from the noise the loop
        feeds in: the gate is `_GATE_SIGMA * sigma` and a difference of two
        measurements carries `sqrt(2)` of it, so `sqrt(2) * gate` is what the
        point that produced this score could resolve. It slightly UNDER-states
        a short-settle point, whose split halves share the same step
        shortfall and so cannot see it; that errs towards asking for more
        precision, which is the safe direction.
        """
        if not np.isfinite(score) or not np.isfinite(self.best_score):
            return
        self._probe_gap.append(abs(score - self.best_score))
        self._probe_delta.append(np.sqrt(2.0) * self._noise_level())

    def precision(self):
        """How small a score difference this climb has to resolve now.

        Searching: the contrast recent probes showed, with their noise taken
        out in quadrature. Verifying: the finest budget. Parked: the
        disturbance threshold. The floor is the accept gate.
        """
        gate = self._noise_level()
        if self.converged:
            drop = abs(self.cfg.disturb_drop_frac * self.best_score)
            return B.PrecisionRequest(delta=max(drop, gate))
        if self.stage == "verify":
            return B.PrecisionRequest(delta=0.0, critical=True)
        if not self._probe_gap:
            return B.PrecisionRequest(delta=0.0)  # Seed: no evidence yet.
        gap = float(np.median(self._probe_gap))
        seen = float(np.median(self._probe_delta)) if self._probe_delta else 0.0
        contrast = np.sqrt(max(0.0, gap * gap - seen * seen))
        return B.PrecisionRequest(delta=max(float(contrast), gate))

    def tell(self, score, valid=True):
        """Update the optimizer with an evaluated score.

        Args:
            score: Objective score for the evaluated point.
            valid: Whether the evaluated point is valid.
        """
        self.iter += 1
        if not valid:
            return  # Keep self._pending, tab re-measures.
        if self.stage in ("minus", "plus", "chase", "mode", "escape"):
            self._note_contrast(score)
        self.last_score = score
        if self.converged:  # Hold on best; UI triggers rescan on drift.
            self._park_hold(score)
            return
        if self.stage == "seed":
            self.best_score = score  # Anchor the start point as the first best.
            self._seed_score = score  # Remembered for the escape gate.
            self.stage = "minus"
            self._pending = self._axis_vec(-1)
            return
        if self.stage == "minus":
            self._probe["minus"] = score
            self.stage = "plus"
            self._pending = self._axis_vec(+1)
            return
        noise = self._noise_level()
        if self.stage == "verify":
            spike = self.best_score - score
            self.best_score = score  # Selection-free re-anchor
            if spike > 2.0 * noise and self._verify_retries < 1:
                # The recorded best was clearly a noise spike; retry once at
                # a halved step rather than re-flailing at the same scale.
                self._verify_retries += 1
                self.step = np.maximum(self._min_step, self.step // 2)
                self._improved = False
                self.axis, self.stage = 0, "minus"
                self._pending = self._axis_vec(-1)
            else:
                self._verify_retries = 0
                self._sweep_done()
            return
        if self.stage == "plus":
            m, p = self._probe["minus"], score
            if p > self.best_score + noise and p >= m:
                self._adopt(+1, p)  # + side is a real improvement.
            elif m > self.best_score + noise and m > p:
                self._adopt(-1, m)  # - Side is a real improvement.
            else:
                self._advance_axis()  # Neither beats best -> next axis.
            return
        if self.stage == "mode":
            if score > self.best_score + noise:
                self.best_x = self._pending.copy()
                self.best_score = score
                self._improved = True
                self.axis, self.stage = 0, "minus"
                self._pending = self._axis_vec(-1)
                return
            self._mode_i += 1
            if self._mode_i < len(self._mode_queue):
                self._pending = self._mode_queue[self._mode_i]
            else:
                self._finish_failed_sweep()
            return
        if self.stage == "escape":
            if score > self.best_score + noise:
                self.best_x = self._pending.copy()
                self.best_score = score
                self._escape_improved = True
            self._escape_i += 1
            # A clearly better basin ends the bank early: escape only has to
            # FIND a basin, not rank every distant point, and the fine climb
            # explores it far more cheaply than the remaining probes would.
            clear = self._escape_ref + max(0.05 * abs(self._escape_ref),
                                           3.0 * noise)
            done = (self._escape_i >= len(self._escape_queue)
                    or (self._escape_improved and self.best_score > clear))
            if not done:
                self._pending = self._escape_queue[self._escape_i]
                return
            if self._escape_improved:
                # Refine normally around the best distant basin found.
                self.axis, self.stage = 0, "minus"
                self._improved = False
                self._pending = self._axis_vec(-1)
            else:
                self._finish_failed_sweep()
            return
        # stage == "chase": keep stepping the winning way while it still pays
        if score > self.best_score + noise:
            self._adopt(self._dir, score)
        else:
            self._advance_axis()

    def rescan(self):
        """Return rescan.

        Re-open the step to recover from a disturbance -- but only moderately.
        The optimum was just here, so there is no need to fling actuators to the
        rails with the full initial step; a small step plus the line-search
        still
        chases a large real disturbance while keeping the score from crashing.
        """
        self.step = np.maximum(
            self._min_step, np.minimum(self._move_step, 4 * self._min_step))
        self.converged = False
        self.axis, self.stage = 0, "minus"
        self._improved = False
        self._pending = self._axis_vec(-1)


class SPGD(_Base):
    """Implement stochastic parallel gradient descent for actuator control."""

    _PLATEAU_ITERS = 60  # Gradient steps with no beyond-noise best improvement.
    _STALL_ITERS = 12

    def __init__(self, cfg):
        super().__init__(cfg)
        self.delta = np.maximum(self._min_step,
                                self._per_channel("spgd_perturb"))
        self.gain = self._per_channel("spgd_gain")
        self.rng = np.random.default_rng()
        self._stall = 0
        self._plateau = 0
        self._plateau_best = -np.inf
        self._pair_delta = np.nan
        self._pair_gate = np.nan
        self._effective_delta = np.nan
        self._direction_span = np.ones(len(self.acts), float)
        self.stage = "seed"  # Measure the current shape before dithering.

    def _start_iter(self):
        self.sign = self.rng.choice([-1.0, 1.0], size=len(self.acts))
        self.stage = "plus"
        self._plus = self._quant(self.x + self.delta * self.sign)
        self._minus = self._quant(self.x - self.delta * self.sign)
        self._pending = self._plus.copy()

    def _note_best(self, score):
        """Adopt only improvements distinguishable from measured noise."""
        self.last_score = score
        threshold = self._measured(1.5, 3e-4)
        if not np.isfinite(self.best_score) or score > self.best_score + threshold:
            self.best_score = score
            self.best_x = self._pending.copy()

    def tell(self, score, valid=True):
        """Update the optimizer with an evaluated score.

        Args:
            score: Objective score for the evaluated point.
            valid: Whether the evaluated point is valid.
        """
        self.iter += 1
        if not valid:
            return
        if self.converged:
            self._park_hold(score)
            return
        self._note_best(score)
        if self.stage == "seed":
            self._start_iter()
            return
        if self.stage == "plus":
            self.jp = score
            self.stage = "minus"
            self._pending = self._minus.copy()
            return
        # Minus -> update along the gradient estimate.
        jm = score
        old = self.x.copy()
        self._pair_delta = float(self.jp - jm)
        # Plus/minus scores carry sqrt(2) times the per-point sigma; outside
        # one pair sigma keep the measured magnitude.
        self._pair_gate = self._measured(np.sqrt(2.0), 3e-4)
        self._effective_delta = (self._pair_delta
                                 if abs(self._pair_delta) > self._pair_gate
                                 else 0.0)
        # At a rail the requested +/- probes are asymmetric. Scale each axis by
        # the displacement that actually reached the command grid instead of
        # pretending both probes moved by the full dither.
        self._direction_span = ((self._plus - self._minus)
                                / np.maximum(2.0 * self.delta, 1.0))
        self.x = np.clip(
            self.x + self.gain * self._effective_delta * self._direction_span,
            self.lo, self.hi)
        if np.all(np.abs(self._quant(self.x) - self._quant(old))
                  < self._min_step):
            self._stall += 1
        else:
            self._stall = 0
        # No-progress backstop (3-sigma so a lucky spike cannot keep resetting
        # it): stops an oscillating high-gain run that never stalls.
        thr = self._measured(2.0, 3e-4)
        if self.best_score > self._plateau_best + thr:
            self._plateau_best = self.best_score
            self._plateau = 0
        else:
            self._plateau += 1
        self.converged = (self._stall >= self._STALL_ITERS
                          or self._plateau >= self._PLATEAU_ITERS)
        if self.converged:
            self.stage = "park"
            self._pending = self.best_x.copy()
            return
        self._start_iter()

    def rescan(self):
        self._stall = 0
        self._plateau = 0
        self._plateau_best = -np.inf
        self.converged = False
        self.x = self.best_x.astype(float).copy()
        self._start_iter()

    def status(self):
        """Return SPGD state including the latest noise-gated gradient."""
        out = super().status()
        out.update(
            pair_score_delta=float(self._pair_delta),
            pair_noise_gate=float(self._pair_gate),
            effective_score_delta=float(self._effective_delta),
            direction_span=[float(value) for value in self._direction_span],
            stalled_pairs=int(self._stall),
            plateau_pairs=int(self._plateau),
        )
        return out


class Genetic(_Base):
    """Implement an elitist genetic search with mutation and crossover."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rng = np.random.default_rng()
        self._mutation = self._per_channel("ga_mutation")
        self.pop = self._seed_population()
        self.fit = [None] * len(self.pop)
        self.i = 0
        self.gen = 0
        self._gen_best_x = self._quant(self.x)  # Best elite of the last gen,
        self._gen_best_score = -np.inf  # On fresh (reproducible) score.
        self._pending = self.pop[0]

    def _seed_population(self):
        pop = [self._quant(self.x)]  # Seed with the start point.
        for _ in range(max(2, self.cfg.ga_population) - 1):
            v = self.lo + self.rng.random(len(self.acts)) * (self.hi - self.lo)
            pop.append(self._quant(v))
        return pop

    def tell(self, score, valid=True):
        """Update the optimizer with an evaluated score.

        Args:
            score: Objective score for the evaluated point.
            valid: Whether the evaluated point is valid.
        """
        self.iter += 1
        if not valid:
            return
        if self.converged:
            self._park_hold(score)
            return
        self.last_score = score
        self.fit[self.i] = score
        self.i += 1
        if self.i < len(self.pop):
            self._pending = self.pop[self.i]
            return
        # Take the generation's best on the fresh fitness just measured,
        # before _evolve resets the population.
        fits = [f if f is not None else -np.inf for f in self.fit]
        gbi = int(np.argmax(fits))
        self._gen_best_score = float(fits[gbi])
        self._gen_best_x = self.pop[gbi].copy()
        # Selection + crossover + mutation: a genetic tell() is free for the
        # whole population and then pays here once, so the decision cost is a
        # spike every ga_population points rather than a per-point average.
        with self.phase("ga_evolve"):
            self._evolve()
        self.i = 0
        self.gen += 1
        # Track the best reproducible shape for the live display/best_command.
        if self._gen_best_score > self.best_score:
            self.best_score = self._gen_best_score
            self.best_x = self._gen_best_x.copy()
        if self.gen >= self.cfg.ga_generations:
            # Fixed budget reached: park on the CURRENT generation's best elite
            # (drift-robust, reproducible), never the stale all-time spike
            self.best_x = self._gen_best_x.copy()
            self.best_score = self._gen_best_score
            self.converged = True
        self._pending = self.best_x if self.converged else self.pop[self.i]

    def _evolve(self):
        order = np.argsort([f if f is not None else -np.inf
                            for f in self.fit])[::-1]
        elite_n = max(1, len(self.pop) // 4)
        elites = [self.pop[i] for i in order[:elite_n]]
        new = list(elites)
        while len(new) < len(self.pop):
            a, b = self._tournament(), self._tournament()
            mask = self.rng.random(len(self.acts)) < 0.5
            child = np.where(mask, a, b).astype(float)
            child += self.rng.normal(0.0, self._mutation, len(self.acts))
            new.append(self._quant(child))
        self.pop = new
        self.fit = [None] * len(self.pop)

    def _tournament(self, k=3):
        idx = self.rng.integers(0, len(self.pop), size=k)
        best = max(idx, key=lambda j: self.fit[j] if self.fit[j] is not None
                   else -np.inf)
        return self.pop[best].astype(float)

    def rescan(self):
        self.pop = self._seed_population()
        self.pop[0] = self.best_x
        self.fit = [None] * len(self.pop)
        self.i, self.gen = 0, 0
        self._gen_best_score = -np.inf
        self._gen_best_x = self.best_x.copy()
        self.converged = False
        self._pending = self.pop[0]


class SimAnneal(_Base):
    """Implement simulated annealing for actuator control."""

    _TETHER_GATE = 0.3  # Tether active once T < this fraction of T0.
    _TETHER_N = 6  # Consecutive below-band points before re-anchoring.

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rng = np.random.default_rng()
        self.step0 = np.maximum(self._min_step, self._per_channel("sa_step"))
        self.step = self.step0.copy()
        self.cooling = float(min(max(cfg.sa_cooling, 0.5), 0.9999))
        self.t_frac = float(max(cfg.sa_t0, 1e-4))
        self.T = self.T0 = None  # Armed by the seed measurement.
        self.cur_x = self._quant(self.x)  # Last ACCEPTED point of the walk.
        self.cur_score = -np.inf
        self.stage = "seed"
        self._pending = self.cur_x.copy()
        self._low = 0  # Consecutive points below the band.
        self._polish = []  # Final +/- probes around the best.
        self._polish_i = 0

    def _propose(self):
        """Isotropic ALL-actuator Gaussian step.

        Measured on a connected 5-D two-well (worse basin seeded, 8 runs per
        setting): all-axis escapes up to 6/8 vs 4/8 for one-actuator-at-a-time
        proposals -- the saddle runs along the diagonal, which single-axis
        staircases cross through deeper corner states. So all-axis, not the
        per-segment variant.
        """
        v = self.cur_x.astype(float) + self.rng.normal(
            0.0, np.maximum(self.step, self._min_step), len(self.acts))
        q = self._quant(v)
        if np.array_equal(q, self.cur_x):  # Quantised to a no-move: nudge.
            axis = int(self.rng.integers(len(self.acts)))
            v[axis] += self._min_step[axis] * (1.0 if self.rng.random() < 0.5
                                               else -1.0)
            q = self._quant(v)
        self._pending = q

    def _floor_T(self):
        """Temperature where annealing sinks below the measurement noise."""
        scale = abs(self.cur_score) if np.isfinite(self.cur_score) else 1.0
        return self._measured(2.5, 1e-3 * max(scale, 1e-9))

    def _start_polish(self):
        """Start polish.

        T reached the noise floor: before parking, one greedy +/- min-step
        sweep per axis around the best. The random walk's step never shrinks
        below min_step in EVERY axis at once, so it is too coarse to finish
        the job -- this cheap 2N-point sweep does the fine local refinement.
        """
        self.stage = "polish"
        s = np.maximum(self._min_step, self.step)
        probes = []
        for ax in range(len(self.acts)):
            for sign in (1.0, -1.0):
                v = self.best_x.astype(float).copy()
                v[ax] += sign * s[ax]
                q = self._quant(v)
                if not np.array_equal(q, self.best_x):
                    probes.append(q)
        self._polish, self._polish_i = probes, 0
        if not probes:
            self._finish()
        else:
            self._pending = probes[0]

    def _finish(self):
        self.converged = True
        self.stage = "park"
        self._pending = self.best_x.copy()

    def tell(self, score, valid=True):
        """Update the optimizer with an evaluated score.

        Args:
            score: Objective score for the evaluated point.
            valid: Whether the evaluated point is valid.
        """
        self.iter += 1
        if not valid:
            return
        if self.converged:
            self._park_hold(score)
            return
        self._note_best(score)  # Adopts a polish probe that wins.
        if self.stage == "seed":
            self.cur_score = score
            self.T0 = self.T = self.t_frac * max(abs(score), 1e-6)
            self.stage = "walk"
            self._propose()
            return
        if self.stage == "polish":
            self._polish_i += 1
            if self._polish_i < len(self._polish):
                self._pending = self._polish[self._polish_i]
            else:
                self._finish()
            return
        delta = score - self.cur_score  # Metropolis on the proposal.
        if delta >= 0 or self.rng.random() < np.exp(delta / max(self.T, 1e-12)):
            self.cur_x = self._pending.copy()
            self.cur_score = score
        # Cold-phase best tether (see class docstring): persistent deep drift
        # is re-anchored on the best; transient dips (a crossing in progress)
        # are left alone by the persistence count.
        gap_hit = (self.T < self._TETHER_GATE * self.T0
                   and self.best_score - self.cur_score >
                   max(3.0 * self.T, self._measured(5.0, 0.0)))
        self._low = self._low + 1 if gap_hit else 0
        if self._low >= self._TETHER_N:
            self.cur_x = self.best_x.copy()
            self.cur_score = self.best_score
            self._low = 0
        self.T *= self.cooling
        self.step = np.maximum(self._min_step,
                               self.step0 * (self.T / self.T0) ** 0.5)
        if self.T < self._floor_T():
            self._start_polish()
            return
        self._propose()

    def rescan(self):
        """Disturbance: reheat to a quarter of T0 and walk from the best."""
        self.converged = False
        self.stage = "walk"
        self.cur_x = self.best_x.copy()
        self.cur_score = self.best_score
        if self.T0:
            self.T = 0.25 * self.T0
        self.step = np.maximum(self._min_step, 0.5 * self.step0)
        self._low = 0
        self._propose()


class CMAES(_Base):
    """CMA-ES via the `cma` package.

    Population batches like the GA, but the sampling cloud learns direction,
    size and correlations from the results, so it needs far fewer measurements
    and tolerates noise. Internally continuous; the hardware command is the
    quantised sample and CMA is told the score against that quantised point.
    Converges when its own stop criteria fire or the cloud shrinks below the
    hardware min step.
    """

    _PLATEAU_GENS = 10  # Generations with no beyond-noise best improvement.
    _MIN_GENS = 5  # Never trust the sigma/es.stop early-out before this.
    # Many generations have actually run.

    def __init__(self, cfg):
        super().__init__(cfg)
        import cma  # Lazy: optional dependency.
        self._cma = cma
        n = len(self.acts)
        self.popsize = (cfg.cma_popsize if cfg.cma_popsize > 0
                        else S.default_cma_popsize(n))
        self.sigma0 = np.maximum(self._min_step, self._per_channel("cma_sigma"))
        self._plateau = 0
        self._plateau_best = -np.inf
        self.gen = 0
        self._new_es(self.x, self.sigma0)

    def _new_es(self, centre, widths):
        """Open a cloud whose 1-sigma width is `widths[i]` on axis i.

        `cma` samples axis i at `sigma * CMA_stds[i]`, so only the product is
        defined; anchoring sigma on the mean keeps the multipliers around 1 and
        leaves a single-mirror run with exactly the scalar sigma it had before,
        since every width is then the same number.

        Args:
            centre: Starting mean, in bits.
            widths: Per-axis 1-sigma search width, in bits.
        """
        widths = np.asarray(widths, float)
        self._sigma = float(np.mean(widths))
        self._stds = widths / self._sigma
        self.es = self._cma.CMAEvolutionStrategy(
            np.asarray(centre, float).tolist(), self._sigma,
            {"bounds": [list(self.lo), list(self.hi)],
             "popsize": int(self.popsize), "verbose": -9,
             "CMA_stds": list(self._stds)})
        self.gen = 0
        self._new_batch()

    def _new_batch(self):
        self._batch = self.es.ask()
        self._scores = []
        self._batch_best_score = -np.inf  # Best sample of THIS batch,
        self._batch_best_x = self.best_x.copy()  # On its fresh score.
        self._i = 0
        self._pending = self._quant(self._batch[0])

    def tell(self, score, valid=True):
        """Update the optimizer with an evaluated score.

        Args:
            score: Objective score for the evaluated point.
            valid: Whether the evaluated point is valid.
        """
        self.iter += 1
        if not valid:
            return
        if self.converged:
            self._park_hold(score)
            return
        self.last_score = score
        self._scores.append(-float(score))  # Cma minimises
        # Best sample of this batch on its fresh score and applied bits, so
        # park time can reach it again. Same policy as Genetic.
        if score > self._batch_best_score:
            self._batch_best_score = score
            self._batch_best_x = self._quant(self._batch[self._i])
        self._i += 1
        if self._i < len(self._batch):
            self._pending = self._quant(self._batch[self._i])
            return
        # Covariance update, once per batch: same shape of cost as the GA's
        # evolve step -- free for the batch, then one spike.
        with self.phase("cma_update"):
            self.es.tell(self._batch, self._scores)
        self.gen += 1
        bbs, bbx = self._batch_best_score, self._batch_best_x
        # Add noise-aware plateau convergence because CMA's numerical criteria
        # do not fire on this quantised bench. Converge after `_PLATEAU_GENS`
        # batches without an improvement above max(2*sigma_noise, 1e-4).
        thr = self._measured(2.0, 1e-4)
        if bbs > self._plateau_best + thr:
            self._plateau_best = bbs
            self._plateau = 0
        else:
            self._plateau += 1
        if bbs > self.best_score:
            self.best_score = bbs
            self.best_x = bbx.copy()
        # Ignore CMA-ES stop signals until enough generations have been sampled.
        # The cloud is spent once it can no longer propose a move on ANY
        # axis's own grid: `es.sigma * stds[i]` is the width it still has there.
        collapsed = ((self.es.stop()
                      or np.all(self.es.sigma * self._stds < self._min_step))
                     and self.gen >= self._MIN_GENS)
        if collapsed or self._plateau >= self._PLATEAU_GENS:
            # Park on the reproducible current-state batch-best, never the
            # stale all-time spike -- identical policy to the GA.
            self.best_x = bbx.copy()
            self.best_score = bbs
            self.converged = True
            self._pending = self.best_x.copy()
            return
        self._new_batch()

    def rescan(self):
        """Disturbance: restart a small cloud on the best-known shape."""
        self.converged = False
        self._plateau = 0
        self._plateau_best = -np.inf
        self._new_es(self.best_x,
                     np.maximum(4 * self._min_step, self.sigma0 / 4))


class BayesOpt(_Base):
    """Gaussian-process Bayesian optimisation via scikit-optimize.

    Remembers every measurement, fits a GP with a noise term and asks the point
    with the best expected improvement -- the most sample-efficient global
    search here, built for measurements that cost ~1 s each. Model refit cost
    grows with the point count, so a fixed evaluation budget parks it on the
    best.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._make_opt()

    def _make_opt(self):
        from skopt import Optimizer as SkOptimizer
        from skopt.space import Real
        from skopt.utils import cook_estimator, normalize_dimensions
        space = [Real(float(l), float(h)) for l, h in zip(self.lo, self.hi)]
        # GP with a fitted white-noise term, as gp_minimize cooks it: without
        # it the GP interpolates measurement noise and the search goes blind.
        est = cook_estimator("GP", space=normalize_dimensions(space),
                             noise="gaussian")
        self.skopt = SkOptimizer(
            space, base_estimator=est, acq_func="EI",
            acq_func_kwargs={"xi": float(self.cfg.bo_xi)},
            acq_optimizer="sampling",
            n_initial_points=max(2, self.cfg.bo_init_points),
            initial_point_generator="lhs")
        self._n_told = 0
        self._pending = self._quant(self.x)  # Anchor on the seed shape.

    def tell(self, score, valid=True):
        """Update the optimizer with an evaluated score.

        Args:
            score: Objective score for the evaluated point.
            valid: Whether the evaluated point is valid.
        """
        self.iter += 1
        if not valid:
            return
        if self.converged:
            self._park_hold(score)
            return
        self._note_best(score)
        # Both halves of a Bayesian decision live in tell(), and they grow at
        # different rates -- the refit with the history, the acquisition search
        # with the space. Split so a slow late point says WHICH one it was.
        with self.phase("gp_fit"):
            self.skopt.tell([float(v) for v in self._pending], -float(score))
        self._n_told += 1
        if self._n_told >= max(self.cfg.bo_init_points + 1, self.cfg.bo_budget):
            self.converged = True
            self._pending = self.best_x.copy()
            return
        with self.phase("acq_opt"):
            nxt = self.skopt.ask()
        self._pending = self._quant(np.asarray(nxt, float))

    def rescan(self):
        """Return rescan.

        Disturbance means the objective itself changed, so the old samples
        would mislead the model: restart the GP seeded on the best shape.
        """
        self.converged = False
        self.x = self.best_x.astype(float).copy()
        self._make_opt()


class Staged:
    """Run a modal solve, then hand its result to a search to finish.

    Modal metrics lose sensitivity near the diffraction limit, so the solve gets
    the mirror close and a search finishes on a metric that still has gradient
    there.

    A facade rather than a `_Base` subclass: it owns no search state, and the
    second stage is a normal optimiser that merely starts from a good point.
    The score scale changes at handover, so `metric` lets the driver re-score
    the same frame for whichever stage is live.
    """

    def __init__(self, cfg, first, polish_algorithm, polish_metric):
        """
        Args:
            cfg: Loop settings, the template for the polish stage.
            first: The modal solver to run first.
            polish_algorithm: Any ALGO_* search to run afterwards.
            polish_metric: Metric to score that search on.
        """
        self.cfg = cfg
        self.first = first
        self.polish_algorithm = polish_algorithm
        self.polish_metric = polish_metric
        self.active = first
        self.handed_over = False

    @property
    def metric(self):
        return self.active.metric

    # Forwarded like `best_command`: the driver reads these to decide whether a
    # fresh point beats the incumbent, and the answer must come from the stage
    # that is live, because the score scale changes at handover.
    @property
    def best_score(self):
        return self.active.best_score

    @property
    def best_x(self):
        return self.active.best_x

    def _hand_over(self):
        """Start the search from the bits the solve settled on.

        Two things change besides the starting bits, both because polishing a
        solved point is not the same job as searching from a cold one:

        * The probe size comes from the solver when it can derive one, so the
          search opens at the scale of what is actually LEFT rather than at
          the stride a run that knew nothing would take.
        * `warm_start` tells the search its small early gains are expected,
          so it does not read them as "wrong basin" and leap.
        """
        best = self.first.best_command()
        acts = [dataclasses.replace(a, start=int(best.get(a.channel, a.start)))
                for a in self.cfg.actuators]
        cfg = dataclasses.replace(self.cfg, algorithm=self.polish_algorithm,
                                  metric=self.polish_metric, actuators=acts)
        step = getattr(self.first, "handover_step", None)
        if callable(step):
            cfg = dataclasses.replace(cfg, move_step=int(step()))
        self.active = make_optimizer(cfg)
        self.active.warm_start = True
        self.handed_over = True

    def ask(self):
        return self.active.ask()

    def observe(self, reading):
        self.active.observe(reading)

    def precision(self):
        """The active stage's request: the two stages measure differently."""
        return self.active.precision()

    def tell(self, score, valid=True):
        self.active.tell(score, valid)
        status = self.active.status()
        if (not self.handed_over and status["converged"]
                and not status.get("failed_reason")):
            self._hand_over()

    def take_phase_timing(self):
        return self.active.take_phase_timing()

    def set_noise(self, sigma):
        self.active.set_noise(sigma)

    def noise_gate(self):
        """The active stage's gate, so the record follows the handover."""
        return self.active.noise_gate()

    def best_command(self):
        return self.active.best_command()

    def rescan(self):
        if hasattr(self.active, "rescan"):
            self.active.rescan()

    def recover_roi_miss(self):
        """Forward a fixed-ROI rejection to the active modal stage.

        Returns:
            The modal recovery action, or ``None`` after handover to a search.
        """
        recover = getattr(self.active, "recover_roi_miss", None)
        return recover() if recover is not None else None

    def status(self):
        st = dict(self.active.status())
        st["stage"] = ("polish/" if self.handed_over
                       else "solve/") + str(st.get("stage", ""))
        st["handed_over"] = self.handed_over
        st["metric"] = self.metric
        return st
