# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 2.1, 2026-08-16

"""Scan plan: which knobs to sweep, over what ranges, in what order."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field, asdict

import numpy as np

from ..correction import settings as S

# One control run every this many scan runs, plus one closing the block: the
# grid's middle cell re-measured over time, so membrane drift can be divided
# out of every other cell.
_CONTROL_EVERY = 12


@dataclass
class Knob:
    """One scannable parameter: name maps 1:1 onto a LoopSettings field."""
    name: str
    lo: float
    hi: float
    points: int
    log: bool = False
    integer: bool = True

    def values(self):
        if self.points <= 1 or self.lo == self.hi:
            vals = np.array([self.lo], float)
        elif self.log and self.lo > 0:
            vals = np.geomspace(self.lo, self.hi, self.points)
        else:
            vals = np.linspace(self.lo, self.hi, self.points)
        if self.integer:
            vals = np.unique(np.round(vals).astype(int))
            return [int(v) for v in vals]
        return [float(v) for v in vals]


# Settle time and frames per point are fixed from the sweep measurements
# rather than scanned: (settle_ms, frames) per mirror size. DM9 is the slower
# mirror, so its value is also safe for DM5.
MEASUREMENT_FIXED = {
    5: (300, 8),   # delay 414 ms -> 0.39% (peak) / 0.77% (psd_band)
    9: (500, 8),   # delay 614 ms -> 0.93% (peak) / 0.56% (psd_band)
}
DEFAULT_MEASUREMENT = MEASUREMENT_FIXED[9]


def measurement_for(n_act):
    """Fixed (settle_ms, frames) for a mirror of this actuator count."""
    return MEASUREMENT_FIXED.get(int(n_act), DEFAULT_MEASUREMENT)


# Sweep ranges respect the 50-bit hardware resolution and are sized per
# mirror. A knob that only sets run length is pinned to one value; knobs that
# change behaviour get five levels. Pinned knobs stay as one-point rows.
KNOBS_BY_SIZE = {
    # ---- 5 actuators -------------------------------------------------
    5: {
        S.ALGO_HILL: [Knob("move_step", 50, 800, 5, log=True),
                      Knob("min_step", 50, 350, 5)],
        S.ALGO_SPGD: [Knob("spgd_perturb", 50, 350, 5),
                      # step = gain * dscore; gains under ~1e3 move <50 bits
                      # and stall.
                      Knob("spgd_gain", 300, 30000, 5, log=True,
                           integer=False)],
        # Population from the five-dimensional diversity floor to 32;
        # mutation from the observable 50-bit floor to ~12% of stroke.
        S.ALGO_GENETIC: [Knob("ga_population", 8, 32, 5),
                         Knob("ga_mutation", 50, 500, 5, log=True),
                         Knob("ga_generations", 20, 20, 1)],
        # Popsize brackets CMA's own recommendation for N=5 (4+3lnN ~ 9): from
        # just below it to ~3x, where a bigger generation buys robustness for
        # cost.
        S.ALGO_CMAES: [Knob("cma_sigma", 50, 1000, 5, log=True),
                       Knob("cma_popsize", 6, 26, 5)],
        # init_points ~2..5 x dimension before the GP steers; xi from
        # pure-exploit (0.001) to strong-explore (0.1), log so the middle
        # point sits at the 0.01 default.
        S.ALGO_BO: [Knob("bo_init_points", 10, 26, 5),
                    Knob("bo_xi", 0.001, 0.1, 5, log=True, integer=False),
                    Knob("bo_budget", 120, 120, 1)],
        # t0 from near-greedy (0.02) to wide-roaming (0.3). Cooling is pinned
        # because it is this algorithm's budget knob (0.99 gives ~300 points).
        S.ALGO_SA: [Knob("sa_t0", 0.02, 0.3, 5, log=True, integer=False),
                    Knob("sa_step", 50, 600, 5, log=True),
                    Knob("sa_cooling", 0.99, 0.99, 1, integer=False)],
    },
    # 9 actuators. Ranges follow the measured anchors: one radian of the
    # cheapest eigenmode costs 167 bit, and the largest single-channel
    # excursion of a converged solution sets the top of every bit range.
    9: {
        # move_step is the probe length, min_step the stopping resolution --
        # the two ends of the same time/quality trade.
        S.ALGO_HILL: [Knob("move_step", 50, 800, 5, log=True),
                      Knob("min_step", 50, 350, 5)],
        S.ALGO_SPGD: [Knob("spgd_perturb", 50, 400, 5),
                      Knob("spgd_gain", 300, 30000, 5, log=True,
                           integer=False)],
        # Budget scan: population and mutation pinned, generations scanned.
        S.ALGO_GENETIC: [Knob("ga_generations", 2, 40, 6, log=True),
                         Knob("ga_population", 17, 17, 1),
                         Knob("ga_mutation", 141, 141, 1)],
        # CMA-ES: lambda 8..26 brackets the default 4+3ln(9) = 11; sigma0 by
        # the quarter-of-the-domain rule over about +/-1000 bit.
        S.ALGO_CMAES: [Knob("cma_sigma", 50, 850, 5, log=True),
                       Knob("cma_popsize", 8, 26, 5)],
        # Budget scan: init_points and xi pinned, budget scanned. Past ~200
        # points the GP refit, not the mirror, dominates the cost.
        S.ALGO_BO: [Knob("bo_budget", 20, 200, 6, log=True),
                    Knob("bo_init_points", 18, 18, 1),
                    Knob("bo_xi", 0.001, 0.001, 1, integer=False)],
        # SA: t0 spans near-greedy to accept-almost-anything; cooling is
        # pinned at 0.99 (~300 points) as this algorithm's budget knob.
        S.ALGO_SA: [Knob("sa_t0", 0.02, 0.2, 5, log=True, integer=False),
                    Knob("sa_step", 80, 600, 5, log=True),
                    Knob("sa_cooling", 0.99, 0.99, 1, integer=False)],
    },
}
DEFAULT_KNOBS = KNOBS_BY_SIZE[9]


def default_knobs(algo, n_act):
    """Knob table for one algorithm on a mirror of this actuator count."""
    table = KNOBS_BY_SIZE.get(int(n_act), DEFAULT_KNOBS)
    return table.get(algo, DEFAULT_KNOBS[algo])


# Measured per-point cost: t_point = 66 ms + settle_ms + 49.7 ms x N.
_OVERHEAD_S = 0.066
_FRAME_S = 0.0497
# ETA-only nominal point counts for the self-converging optimizers (the real
# run ends on the optimizer's own converged flag, never a fixed budget).
# 190 is the DM9 median over the hill-climb and CMA-ES runs on record.
_NOMINAL_POINTS = 190


@dataclass
class BenchPlan:
    """Everything one bench session needs; serialised to plan.json."""
    algos: dict = field(default_factory=dict)  # Algo -> list[Knob]
    # Single values: settle and frames are fixed constants (MEASUREMENT_FIXED),
    # so phase B is skipped unless more than one value is given.
    settle_list: list = field(
        default_factory=lambda: [DEFAULT_MEASUREMENT[0]])
    frames_list: list = field(
        default_factory=lambda: [DEFAULT_MEASUREMENT[1]])
    # Run order inside one algorithm is shuffled with this seed, written to
    # plan.json; 0 means a fresh seed is picked at every Start.
    order_seed: int = 0
    control_every: int = _CONTROL_EVERY  # 0 = no control runs.
    # What the loop's compensation switch was set to. The bench itself never
    # compensates, so this is recorded to keep "200 bits" unambiguous.
    compensation: dict = field(default_factory=dict)
    repeats: int = 3  # Each config measured 3x -> median + spread.
                                     # Per cell, so a knob->score trend is real
                                     # enough to explain, not a one-shot
                                     # reading.
    noise_s: float = 60.0
    # Minutes parked at 0 when switching algorithms, so each one opens on a
    # relaxed membrane.
    rest_min: int = 45
    # Generous backstops: a run ends on the optimizer's own converged flag;
    # these only stop a non-converging config from hanging an unattended scan.
    safety_points: int = 100000
    safety_timeout_s: float = 1800.0
    # The loop's actuator table: channel/start/min/max per piezo; the bench
    # resets to each actuator's own start bits.
    actuators: list = field(default_factory=S.default_actuators)
    active_mirror: int = S.DEFAULT_LAYOUT
    inactive_mirror: int = S.DM9
    hold_inactive_shape: bool = False
    # The optimiser never sees these channels; the engine merges them into
    # every hardware setpoint so the other mirror keeps its shape.
    fixed_channels: dict = field(default_factory=lambda: {
        channel: 0 for channel in S.LAYOUTS[S.DM9]
    })
    metric: str = S.METRIC_PEAK
    # Full LoopSettings snapshot at bench start; every run builds on it so
    # bench scoring is identical to the loop's.
    base: dict = field(default_factory=dict)

    @property
    def settle_mid(self):
        s = sorted(self.settle_list)
        return s[len(s) // 2]

    @property
    def frames_mid(self):
        f = sorted(self.frames_list)
        return f[len(f) // 2]

    def to_dict(self):
        d = asdict(self)
        d["algos"] = {a: [asdict(k) for k in ks]
                      for a, ks in self.algos.items()}
        d["actuators"] = [asdict(a) for a in self.actuators]
        d["fixed_channels"] = {
            str(channel): int(bit)
            for channel, bit in self.fixed_channels.items()
        }
        return d


def knob_grid(knobs):
    """All knob-value combinations as dicts (full factorial)."""
    names = [k.name for k in knobs]
    return [dict(zip(names, combo))
            for combo in itertools.product(*(k.values() for k in knobs))]


def phase_a_runs(plan: BenchPlan, algo: str):
    """Knob grid at the fixed measurement setting, shuffled, with controls.

    Deterministic for one plan: the order comes from `plan.order_seed` mixed
    with the algorithm name, so the ETA, the job queue and a later replay all
    build exactly the same list, while two algorithms do not share one
    permutation.

    Args:
        plan: Measurement or optimisation plan.
        algo: Optimisation algorithm identifier.
    """
    grid = knob_grid(plan.algos[algo])
    runs = [dict(algo=algo, phase="knobs", knobs=knobs,
                 settle_ms=plan.settle_mid, frames=plan.frames_mid,
                 repeat=rep)
            for knobs in grid for rep in range(plan.repeats)]
    # Grid order is the one order that guarantees drift lands on a knob.
    random.Random(f"{plan.order_seed}:{algo}").shuffle(runs)
    if not (plan.control_every > 0 and grid):
        return runs
    middle = grid[len(grid) // 2]

    out, taken = [], 0

    def add_control():
        nonlocal taken
        out.append(dict(algo=algo, phase="control", knobs=dict(middle),
                        settle_ms=plan.settle_mid, frames=plan.frames_mid,
                        repeat=taken))
        taken += 1

    for i, spec in enumerate(runs):
        if i % plan.control_every == 0:
            add_control()
        out.append(spec)
    add_control()  # Closes the chart: a drift curve needs both ends.
    return out


def phase_b_runs(plan: BenchPlan, algo: str, best_knobs: dict):
    """settle x frames grid at the winning knob config from phase A.

    Empty when both lists hold a single value: settle and frames are then
    fixed constants (the default -- see MEASUREMENT_FIXED), and a 1x1 "grid"
    would only re-run the phase-A winner under another name.

    Args:
        plan: Measurement or optimisation plan.
        algo: Optimisation algorithm identifier.
        best_knobs: Sequence of best knob values.
    """
    if len(set(plan.settle_list)) < 2 and len(set(plan.frames_list)) < 2:
        return []
    runs = []
    for settle in plan.settle_list:
        for frames in plan.frames_list:
            for rep in range(plan.repeats):
                runs.append(dict(algo=algo, phase="measure",
                                 knobs=dict(best_knobs), settle_ms=settle,
                                 frames=frames, repeat=rep))
    return runs


def pick_best_knobs(rows, algo):
    """Return pick best knobs.

    Winning knob config from phase-A rows: highest median converged score,
    ties broken by shorter median total time. Falls back to best_score when a
    config never converged.

    Args:
        rows: Input table rows.
        algo: Optimisation algorithm identifier.
    """
    groups = {}
    for r in rows:
        if r["algo"] != algo or r["phase"] != "knobs":
            continue
        key = tuple(sorted(r["knobs"].items()))
        groups.setdefault(key, []).append(r)
    if not groups:
        return None
    def rank(item):
        rs = item[1]
        score = float(np.median([r["score_converged"]
                                 if np.isfinite(r["score_converged"])
                                 else r["score_best"] for r in rs]))
        t = float(np.median([r["t_total_s"] for r in rs]))
        return (score, -t)
    best = max(groups.items(), key=rank)
    return dict(best[0])


def build_settings(plan: BenchPlan, spec: dict) -> S.LoopSettings:
    """Build the LoopSettings for one run.

    The plan's base snapshot with only algorithm, knobs, settle and frames
    overridden. Fresh objects every run; the shared table is never mutated.

    Args:
        plan: Measurement or optimisation plan.
        spec: Measurement, plot, or waveform specification.
    """
    if plan.base:
        cfg = S.LoopSettings.from_dict(dict(plan.base))
    else:
        cfg = S.LoopSettings(metric=plan.metric)
    cfg.actuators = [S.Actuator(**asdict(a)) for a in plan.actuators]
    cfg.algorithm = spec["algo"]
    cfg.settle_ms = int(spec["settle_ms"])
    cfg.frames_per_measure = int(spec["frames"])
    for name, value in spec["knobs"].items():
        setattr(cfg, name, type(getattr(cfg, name))(value))
    return cfg


def points_estimate(plan: BenchPlan, algo: str, knobs: dict) -> int:
    """Rough measurement count of one run, for the ETA display only.

    The real run length is set by the optimizer's own convergence, not by this.

    Args:
        plan: Measurement or optimisation plan.
        algo: Optimisation algorithm identifier.
        knobs: Sequence of knob values.
    """
    if algo == S.ALGO_GENETIC:
        # Fixed budget: population x generations evaluations (+ the baseline)
        return int(knobs.get("ga_population", 16)
                   * knobs.get("ga_generations", 12)) + 2
    if algo == S.ALGO_BO:
        return int(knobs.get("bo_budget", 60)) + 2
    if algo == S.ALGO_SA:
        # Geometric cooling: points to fall from T0 to ~5% of it.
        c = min(max(float(knobs.get("sa_cooling", 0.97)), 0.5), 0.9999)
        return int(np.log(0.05) / np.log(c)) + 2
    return _NOMINAL_POINTS


# Bayesian optimisation refits the GP on the whole history every point, so
# its decision cost grows as 1.87 s * (n/165)^2 (measured).
_BO_GP_S, _BO_GP_REF = 1.87, 165.0


def run_seconds(plan: BenchPlan, algo: str, knobs: dict, t_pt: float) -> float:
    """Wall time of one run: measurement, plus any decision cost that shows.

    Args:
        plan: Measurement or optimisation plan.
        algo: Optimisation algorithm identifier.
        knobs: This run's knob values.
        t_pt: Seconds one measured point costs.

    Returns:
        Estimated seconds, including the Start-bit baseline point.
    """
    pts = points_estimate(plan, algo, knobs)
    total = (pts + 1) * t_pt
    if algo == S.ALGO_BO and pts > 0:
        n = np.arange(1, pts + 1, dtype=float)
        total += float(np.sum(_BO_GP_S * (n / _BO_GP_REF) ** 2))
    return total


def estimate(plan: BenchPlan):
    """(total_runs, total_seconds) for the plan preview."""
    total_runs, total_s = 0, 0.0
    for algo in plan.algos:
        a = phase_a_runs(plan, algo)
        mid_knobs = knob_grid(plan.algos[algo])[0]
        b = phase_b_runs(plan, algo, mid_knobs)
        for spec in a + b:
            t_pt = (spec["settle_ms"] / 1e3 + spec["frames"] * _FRAME_S
                    + _OVERHEAD_S)
            # One explicit Start-bit baseline is measured before ask()/tell().
            total_s += run_seconds(plan, algo, spec["knobs"], t_pt)
        total_runs += len(a) + len(b)
        total_s += 2 * plan.noise_s
    # One rest per algorithm SWITCH (not per run)
    total_s += max(len(plan.algos) - 1, 0) * plan.rest_min * 60.0
    return total_runs, total_s
