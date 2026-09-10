# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 2.0, 2026-08-29

"""Plan for the repeatability campaign: the same correction run, many times.

Takes the loop configuration as it stands and decides only how many times
each algorithm is given the same problem, in what order, how long the
membrane rests in between, and whether hysteresis compensation is the thing
under test. A modal solve is always compensated (see `forces_comp`).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..correction import settings as S

# Run order. Interleaved is the default: the membrane creeps between runs, so
# running every algorithm once per round spreads the drift over all of them.
ORDER_INTERLEAVED = "interleaved"
ORDER_BLOCKED = "blocked"
ORDERS = (ORDER_INTERLEAVED, ORDER_BLOCKED)

# Hysteresis compensation. FOLLOW leaves the loop's own switch alone; the
# others make compensation the thing under test. BOTH is a paired design: the
# two arms of one repeat run back-to-back, and the opening arm alternates.
COMP_FOLLOW = "follow"
COMP_OFF = "off"
COMP_ON = "on"
COMP_BOTH = "both"
COMP_MODES = (COMP_FOLLOW, COMP_OFF, COMP_ON, COMP_BOTH)

# A modal solve is always compensated: it commands a fitted vertex in one move
# and never re-measures it, so uncompensated it would measure the driver, not
# the solve. The mode above applies to the searches alone.
def forces_comp(algo):
    """True when this algorithm is compensated whatever the mode says."""
    return algo in S.MODAL_ALGOS

# Display names short enough to sit under a box in a distribution figure.
_SHORT = {S.ALGO_MODAL_FIT: "2N+1 (moment)",
          S.ALGO_MODAL_FAST: "N+2 (moment)",
          S.ALGO_MODAL_PSD: "2N+1 (1/PSD)"}


def short_label(algo):
    """Name short enough for an axis tick or a checkbox."""
    if algo in _SHORT:
        return _SHORT[algo]
    text = S.ALGO_LABELS.get(algo, algo).split(" (")[0]
    return text.split(": ", 1)[-1] if ": " in text else text


# What the session holds still, re-read before each run; a change aborts the
# session. Groups follow LoopSettings.active_dict(): metric, measurement,
# optics, hold and actuators. The `algorithm` group is deliberately free, and
# the `speed_floor_*` fields legitimately differ between algorithms.
_HELD_GROUPS = ("metric", "measurement", "optics", "hold")
_NOT_HELD = ("speed_floor_ms", "speed_floor_frames")
# The two measurement fields kept per mirror: on a staged plan they are
# recorded per stage and the unprefixed name is dropped.
_PER_MIRROR_MEASUREMENT = ("settle_ms", "frames_per_measure")


def fingerprint(snapshot):
    """Flat {name: value} of everything a session must keep still.

    Args:
        snapshot: A `DMLoopTab.build_bench_snapshot()` result.

    Returns:
        dict: Comparable scalars, keyed by a name worth showing an operator.
    """
    cfg = snapshot["settings"]
    stages = snapshot.get("stages") or [(None, cfg)]
    out = {}
    active = cfg.active_dict()
    per_stage = len(stages) > 1
    for group in _HELD_GROUPS:
        for key, value in (active.get(group) or {}).items():
            if key in _NOT_HELD or key == "label":
                continue
            if per_stage and group == "measurement"                     and key in _PER_MIRROR_MEASUREMENT:
                continue
            out[f"{group}.{key}"] = value
    # settle and frames are mirror-local, so a two-mirror plan records each
    # stage's separately; a one-mirror plan keeps the plain names.
    if per_stage:
        for mirror, stage_cfg in stages:
            measurement = stage_cfg.active_dict().get("measurement") or {}
            for key in _PER_MIRROR_MEASUREMENT:
                if key in measurement:
                    out[f"DM{int(mirror)}.measurement.{key}"] = measurement[key]
    # Per channel, not as one list: a diff then names the piezo whose Start
    # bit moved instead of printing both tables and leaving you to spot it.
    for _, stage_cfg in stages:
        for a in stage_cfg.actuators:
            out[f"actuator.ch{a.channel}"] = (int(a.start), int(a.bit_min),
                                              int(a.bit_max))
    # The plan itself. Switching DM9 -> DM5 to the joint search mid-session
    # changes the experiment more than any knob below it, and without this key
    # nothing would notice.
    for key in ("run_plan", "active_mirror", "inactive_mirror",
                "hold_inactive_shape", "adaptive_settling"):
        out[key] = snapshot.get(key)
    # A joint run's search size is not on either mirror, so it would otherwise
    # slip through: population 43 -> 16 between runs is a different experiment
    # and has to abort the session, exactly like a changed settle would.
    for knob, value in (snapshot.get("joint_search") or {}).items():
        out[f"joint.{knob}"] = value
    for c, b in (snapshot.get("fixed_channels") or {}).items():
        out[f"fixed.ch{int(c)}"] = int(b)
    return out


def fingerprint_diff(before, now):
    """Names that changed between two fingerprints, with both values.

    Only what `before` recorded is checked. Normally the two sides come from
    the same function and that is every key; a session started by an older
    version recorded fewer, and checking a narrower set honestly beats
    reporting every key it never had as a change.
    """
    changed = []
    for key in sorted(before):
        was, is_ = before.get(key), now.get(key)
        if was != is_:
            changed.append(f"{key}: {was} -> {is_}")
    return changed


def fingerprint_of_plan(plan: RepeatPlan):
    """What a saved session must still match, from the session itself.

    A resumed session must be the same experiment as the one it joins, so
    the reference is what that session started with. Older sessions without
    a stored fingerprint are recovered from the LoopSettings snapshot.

    Args:
        plan: A plan read back from plan.json.

    Returns:
        dict: The same shape `fingerprint` returns, possibly with fewer keys.
    """
    if plan.fixed:
        return dict(plan.fixed)
    if not plan.base:
        return {}
    try:
        cfg = S.LoopSettings.from_dict(dict(plan.base))
    except (TypeError, ValueError, KeyError):
        return {}
    return fingerprint({"settings": cfg,
                        "active_mirror": plan.active_mirror,
                        # Not recorded by those versions: left out rather than
                        # compared against a value invented here.
                        "run_plan": None,
                        "inactive_mirror": None,
                        "hold_inactive_shape": None,
                        "fixed_channels": {}})


def metric_required(algo):
    """The metric this algorithm forces, or None when it takes any.

    A modal solve fits a parabola to one particular quantity, so the metric
    is pinned when one is selected (S.MODAL_METRIC); mixing it with another
    metric would compare runs scored two ways.
    """
    return S.MODAL_METRIC.get(algo)


def metric_list(plan):
    """The metrics this session runs, in order; never empty.

    An older plan.json has only the scalar `metric`, which is exactly a
    one-metric sweep -- so nothing has to special-case the old shape.
    """
    return tuple(plan.metrics) or (plan.metric,)


def algos_for_metric(plan, metric):
    """The ticked algorithms that can actually be scored on `metric`.

    A modal solve pins its metric (`metric_required`), so it runs only in
    the block whose metric it needs and is absent from the others. An
    algorithm that pins nothing runs in every block.
    """
    return tuple(algo for algo in plan.algos
                 if metric_required(algo) in (None, metric))


@dataclass
class RepeatPlan:
    """Everything one repeatability session needs; saved as plan.json."""

    algos: tuple = ()  # Run order of the ticked algorithms.
    repeats: int = 10  # Runs per algorithm.
    interval_s: float = 60.0  # Hold after each run, mirror already at 0.
    order: str = ORDER_INTERLEAVED
    comp_mode: str = COMP_FOLLOW
    # No time limit lives here, on purpose: a run ends when its algorithm
    # ends. A wall clock would cut runs before convergence and leave no
    # convergence time to record.

    # Free text the operator typed before starting: what this session is
    # testing. Saved into plan.json, because six months later the folder name
    # is a timestamp and nothing else says why the session was run.
    note: str = ""

    # Recorded, never applied: the loop settings at Start, frozen so a resumed
    # session can be checked against the bench it started on.
    fixed: dict = field(default_factory=dict)

    # `metric` is the single-metric case and stays first in `metrics`; a
    # non-empty `metrics` runs each in turn as its own block of repeats.
    metric: str = S.METRIC_PEAK
    metrics: tuple = ()
    # Rest between two metric blocks: changing the objective disturbs more
    # than another repeat of the same one.
    metric_gap_s: float = 1200.0
    # Which mirrors the runs drive, and in what order -- one of settings'
    # PLAN_* keys. `active_mirror` is the single-mirror special case and is
    # None for a plan that drives both; read `mirrors` for the general answer.
    run_plan: str = S.PLAN_DM5
    mirrors: tuple = ()
    active_mirror: int | None = S.DEFAULT_LAYOUT
    compensation: dict = field(default_factory=dict)
    autostop_s: float = 0.0
    base: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["algos"] = list(self.algos)
        d["mirrors"] = list(self.mirrors)
        d["metrics"] = list(self.metrics)
        return d

    def plan_label(self):
        """How this session's mirror plan reads in a log line or a title."""
        return S.PLAN_LABELS.get(self.run_plan, str(self.run_plan))

    def is_staged(self):
        """Whether the runs hand one mirror over to the next."""
        return len(self.mirrors) > 1 and self.run_plan != S.PLAN_JOINT

    @classmethod
    def from_dict(cls, data):
        """Rebuild a plan saved as plan.json, ignoring unknown keys.

        Args:
            data: Parsed plan.json.

        Returns:
            RepeatPlan: With whatever the file could supply; missing fields
                keep their defaults, so an older file still loads.
        """
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in (data or {}).items() if k in known}
        if "algos" in kw:
            kw["algos"] = tuple(kw["algos"] or ())
        if "mirrors" in kw:
            kw["mirrors"] = tuple(int(m) for m in (kw["mirrors"] or ()))
        if "metrics" in kw:
            kw["metrics"] = tuple(kw["metrics"] or ())
        return cls(**kw)


def load_session(session_dir):
    """A saved session: its plan, and the (algo, comp, repeat) already run.

    Args:
        session_dir: A dm_repeat session directory.

    Returns:
        tuple: (plan, done, n_planned) -- `done` is a set of
            (algo, comp, repeat, metric) keys, and
            `n_planned` is what the plan asks for in total. (None, set(), 0)
            when the directory is not a readable session.
    """
    d = Path(session_dir)
    try:
        plan = RepeatPlan.from_dict(
            json.loads((d / "plan.json").read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None, set(), 0
    done = set()
    try:
        with open(d / "runs.csv", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    done.add((row["algo"], row.get("compensation", ""),
                              int(row["repeat"]), row.get("metric", "")))
                except (KeyError, TypeError, ValueError):
                    continue
    except (OSError, ValueError):
        pass
    return plan, done, len(run_order(plan))


def remaining(plan: RepeatPlan, done):
    """The plan's runs that `done` does not already cover, renumbered.

    A repeat is identified by (algorithm, arm, repeat number), not by its
    position in the order: the point of resuming is to end up with the run
    COUNT the plan asked for, whatever order the interruption left behind.

    Args:
        plan: The session plan.
        done: Keys already recorded, from `load_session`.

    Returns:
        list: Run specs, `index` continuing from what was done.
    """
    # The arm is part of the identity only when the algorithm has two of
    # them; the metric only when the session sweeps more than one.
    swept = len(metric_list(plan)) > 1
    keyed = {(algo, comp, rep, met if swept else "")
             for algo, comp, rep, met in done}
    loose = {(algo, rep, met if swept else "") for algo, _c, rep, met in done}
    out = []
    for spec in run_order(plan):
        met = spec.get("metric", plan.metric) if swept else ""
        paired = len(comp_arms(plan, spec["repeat"], spec["algo"])) > 1
        if paired:
            arm = "on" if spec["comp"] else "off"
            seen = (spec["algo"], arm, spec["repeat"], met) in keyed
        else:
            seen = (spec["algo"], spec["repeat"], met) in loose
        if seen:
            continue
        out.append(spec)
    for i, spec in enumerate(out):
        spec["index"] = len(done) + i + 1
    return out


def find_unfinished(out_root):
    """The newest session directory with runs still owed, or None.

    Args:
        out_root: The output directory the session writes under.

    Returns:
        tuple: (path, done_count, planned_count), or None.
    """
    root = Path(out_root) / "dm_repeat"
    try:
        dirs = sorted((p for p in root.iterdir() if p.is_dir()),
                      key=lambda p: p.name)
    except OSError:
        return None
    for d in reversed(dirs):
        plan, done, planned = load_session(d)
        if plan is not None and planned and len(done) < planned:
            return d, len(done), planned
    return None


def run_order(plan: RepeatPlan):
    """The session's runs, in the order they will be started.

    Args:
        plan: The session plan.

    Returns:
        list: One dict per run with `algo`, `repeat`, `comp` (True, False, or
            None to leave the loop's switch alone), `metric` and `index`.
            A multi-metric session is these blocks end to end, never
            interleaved: switching the evaluation function mid-round would put
            two rulers inside one round.
    """
    runs = []
    for metric in metric_list(plan):
        algos = list(algos_for_metric(plan, metric))
        if not algos:
            continue
        block = []
        if plan.order == ORDER_BLOCKED:
            for algo in algos:
                for rep in range(int(plan.repeats)):
                    for comp in comp_arms(plan, rep, algo):
                        block.append(dict(algo=algo, repeat=rep, comp=comp))
        else:
            for rep in range(int(plan.repeats)):
                # Rotate the round: without it the algorithm scheduled first
                # would sit on the freshest membrane in every single round,
                # which is a systematic advantage rather than a random one.
                start = rep % len(algos)
                for algo in algos[start:] + algos[:start]:
                    for comp in comp_arms(plan, rep, algo):
                        block.append(dict(algo=algo, repeat=rep, comp=comp))
        for spec in block:
            spec["metric"] = metric
        runs += block
    for i, spec in enumerate(runs):
        spec["index"] = i + 1
    return runs


def comp_arms(plan: RepeatPlan, rep, algo=None):
    """The compensation states one (algorithm, repeat) is run at.

    Args:
        plan: The session plan.
        rep: Which repeat this is; only used to alternate the pair order.
        algo: The algorithm this repeat runs. A modal solve takes one
            compensated arm whatever the mode is (see `forces_comp`); None
            asks what the mode alone would give, for a summary line.

    Returns:
        tuple: True / False per arm, or (None,) to leave the switch alone.
    """
    if algo is not None and forces_comp(algo):
        return (True,)
    if plan.comp_mode == COMP_BOTH:
        return (False, True) if rep % 2 == 0 else (True, False)
    if plan.comp_mode == COMP_OFF:
        return (False,)
    if plan.comp_mode == COMP_ON:
        return (True,)
    return (None,)


def comp_states(plan: RepeatPlan):
    """The compensation states this plan's runs will really use.

    What the mode is called is no longer enough to know whether the session
    needs a working profile or leaves an uncompensated arm behind: one ticked
    modal solve puts a compensated run into an OFF session. The callers that
    have to refuse a session ask this instead of reading `comp_mode`.

    Returns:
        set: Any of True / False / None (the switch left alone).
    """
    return {spec["comp"] for spec in run_order(plan)}


def estimate_s(plan: RepeatPlan, run_s):
    """Session seconds for a measured (or assumed) per-run duration.

    Args:
        plan: The session plan.
        run_s: Seconds one correction run takes, or NaN when nothing has been
            measured yet.

    Returns:
        float: Total seconds, or NaN while `run_s` is unknown.
    """
    n = len(run_order(plan))
    if n == 0:
        return 0.0
    try:
        per = float(run_s)
    except (TypeError, ValueError):
        return float("nan")
    if not per > 0:
        return float("nan")
    # One inter-BLOCK wait per metric change, the rest at the per-run interval.
    gaps = max(len(metric_list(plan)) - 1, 0)
    return (n * per + max(n - 1 - gaps, 0) * float(plan.interval_s)
            + gaps * float(plan.metric_gap_s))
