# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-08-16

"""How long one measurement is worth spending.

A point costs `FIXED_MS + settle_ms + PER_FRAME_MS * frames`. Two measured
curves drive the choice: `r(t)`, the fraction of a step completed after `t`
ms (a bias), and `sigma(N)`, the Allan deviation of one measurement of `N`
frames (a variance). A budget resolves `delta = sqrt(2) z sigma(N) / r(t)`,
and the ladder is the Pareto front over the measured grid. When even the
finest rung cannot resolve a decision, `Q = sigma^2 * C / r^2` ranks the
frame counts. Both curves are per mirror and come from the sweep records.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

# Measured per-point cost. `bench.sweep_plots` imports these so the figures
# and the running loop cost points identically.
FIXED_MS = 66.0
PER_FRAME_MS = 49.7

PROFILE_PATH = Path(__file__).with_name("dm_loop_profiles.json")
PROFILE_KEY = "speed_profiles"

# Points at the anchor budget before its measured mean replaces the modelled
# cost as the baseline; one unlucky point would misprice the run.
_REFERENCE_POINTS = 5


@dataclass(frozen=True)
class PrecisionRequest:
    """What one decision needs from its measurement.

    Attributes:
        delta: Score difference this decision must resolve. `inf` asks for
            the cheapest rung there is, `0` for the finest.
        bias: Largest residual step-response shortfall the decision
            tolerates; unbounded for a comparison against a fixed anchor,
            none for a curve's vertex.
        var_pred: Variance of the scores this point's neighbours were
            measured just after; zero when every candidate follows the same
            command, non-zero for a population.
        critical: True when the point writes state that later points trust;
            those are measured at the finest budget whatever `delta` says.
    """
    delta: float = 0.0
    bias: float = float("inf")
    var_pred: float = 0.0
    critical: bool = False


@dataclass(frozen=True)
class Budget:
    """One measurement's chosen cost, and what it buys.

    Attributes:
        settle_ms: Wait before the scored frames.
        frames: Frames averaged into the score.
        sigma: Noise of one measurement at this budget.
        delta: Smallest score difference it can resolve.
        bias: Residual shortfall it leaves on a measured gain.
        cost_ms: Modelled per-point cost.
        speedup: How many times cheaper than the finest rung of its ladder.
    """
    settle_ms: int
    frames: int
    sigma: float
    delta: float
    bias: float
    cost_ms: float
    speedup: float = 1.0

    @property
    def tier(self):
        """Stable name of this budget: the budget itself.

        Named rather than numbered because a ladder is rebuilt whenever the
        operator retypes the settle, and a rung called "coarse" would then
        quietly mean something else in the next run's log.
        """
        return f"{self.settle_ms}ms/{self.frames}f"

    def describe(self):
        """One line for the status chip and the Fastest-allowed list."""
        rate = "" if self.speedup < 1.05 else f", {self.speedup:.1f}x faster"
        plural = "" if self.frames == 1 else "s"
        return (f"{self.settle_ms} ms + {self.frames} frame{plural} = "
                f"{self.cost_ms:.0f} ms{rate}")


def cost_ms(settle_ms, frames):
    """Modelled cost of one point."""
    return FIXED_MS + float(settle_ms) + PER_FRAME_MS * int(frames)


class SpeedProfile:
    """One mirror's measured step response and noise curve.

    Args:
        step: {settle_ms: captured fraction of the step}, from the sweep.
        noise: {frames: Allan deviation of the score}, from the sweep.
        noise_sem: {frames: standard error of that deviation across sessions}.
        z: Sigma multiple a difference must clear to be believed. Passed in by
            the caller so this module stays independent of the optimisers; the
            loop supplies its own accept gate, so a budget resolves exactly
            what the optimiser will act on.
        sessions: How many sweep recordings it was fitted from (reporting).
        source: Where it came from (reporting).
    """

    def __init__(self, step, noise, noise_sem, z, sessions=0, source=""):
        self.step = {int(k): float(v) for k, v in step.items() if float(v) > 0}
        self.noise = {int(k): float(v) for k, v in noise.items()
                      if float(v) > 0}
        self.noise_sem = {int(k): float(v) for k, v in noise_sem.items()}
        self.z = float(z)
        self.sessions = int(sessions)
        self.source = str(source)
        if not self.step or not self.noise:
            raise ValueError("speed profile needs both curves")
        self.delays = sorted(self.step)
        self.frames = sorted(self.noise)

    # Curves
    def response(self, settle_ms):
        """Captured fraction `r(t)` at the nearest measured delay.

        Nearest rather than interpolated: between two measured delays the
        shape is unknown, and a budget that claims a response nobody measured
        is exactly the kind of number this module refuses to invent.
        """
        return self.step[min(self.delays,
                             key=lambda t: abs(t - float(settle_ms)))]

    def sigma(self, frames):
        """Measurement noise at the nearest measured frame count."""
        return self.noise[self.nearest_frames(frames)]

    def nearest_frames(self, frames):
        return min(self.frames, key=lambda n: abs(n - int(frames)))

    def resolution(self, settle_ms, frames):
        """Smallest score difference one measurement can resolve."""
        return (math.sqrt(2.0) * self.z * self.sigma(frames)
                / self.response(settle_ms))

    def q(self, settle_ms, frames):
        """Wall-clock cost of one unit of variance: `sigma^2 * C / r^2`."""
        r = self.response(settle_ms)
        return (self.sigma(frames) ** 2 * cost_ms(settle_ms, frames) / (r * r))

    def frames_for(self, settle_ms):
        """The frame count that is not beaten on `Q` at this settle.

        The smallest `N` whose `Q` is within one standard error of the best
        `Q`. Ties lose to the cheaper count, because a shorter point also
        shortens the loop's reaction to a disturbance. Reported by the Rebuild
        summary so a typed frame count can be checked against it.
        """
        best = min(self.frames, key=lambda n: self.q(settle_ms, n))
        q_best = self.q(settle_ms, best)
        sem = self.noise_sem.get(best, 0.0)
        # Q goes as sigma^2, so a relative error on sigma doubles on Q.
        slack = (2.0 * sem / self.noise[best] * q_best) if self.noise[best] \
            else 0.0
        for n in self.frames:
            if self.q(settle_ms, n) <= q_best + slack:
                return n
        return best

    def budget(self, settle_ms, frames, finest_cost=None):
        """Describe one `(settle, frames)` pair."""
        c = cost_ms(settle_ms, frames)
        return Budget(settle_ms=int(settle_ms), frames=int(frames),
                      sigma=self.sigma(frames),
                      delta=self.resolution(settle_ms, frames),
                      bias=1.0 - self.response(settle_ms),
                      cost_ms=c,
                      speedup=1.0 if not finest_cost else finest_cost / c)

    # Ladders
    def ladder(self, settle_ms, frames):
        """Rungs from cheapest to finest, anchored on what the operator typed.

        The finest rung IS `(settle_ms, frames)`: automatic mode may go faster
        than what was asked for, never slower. Everything cheaper than it is
        the Pareto front of the measured grid -- a pair earns a rung only if
        nothing else is both cheaper AND finer. Retyping the anchor rebuilds
        the whole ladder, and each mirror builds its own from its own curves.
        """
        finest = self.budget(settle_ms, frames)
        pairs = sorted(((cost_ms(t, n), t, n)
                        for t in self.delays for n in self.frames
                        if cost_ms(t, n) < finest.cost_ms),
                       key=lambda p: p[0])
        rungs, best = [], math.inf
        for c, t, n in pairs:
            d = self.resolution(t, n)
            if d < best - 1e-15:
                best = d
                rungs.append(self.budget(t, n, finest.cost_ms))
        # A rung no finer than the anchor while costing nearly as much buys
        # nothing; drop it so the Fastest-allowed list stays short and every
        # entry is a real choice.
        rungs = [b for b in rungs if b.delta > finest.delta
                 and b.cost_ms < 0.95 * finest.cost_ms]
        return rungs + [self.budget(settle_ms, frames, finest.cost_ms)]

    def solve(self, request: PrecisionRequest, settle_ms, frames, floor=None):
        """Cheapest rung that meets `request`.

        Args:
            request: What the decision needs.
            settle_ms: Anchor settle -- the finest rung (what the operator
                typed).
            frames: Anchor frame count.
            floor: `(settle_ms, frames)` of the coarsest rung the operator
                allows, or None for the whole ladder.

        Returns:
            A `Budget`; the finest rung when none can resolve the difference.
        """
        rungs = self.ladder(settle_ms, frames)
        if floor is not None:
            floor_cost = cost_ms(*floor)
            kept = [b for b in rungs if b.cost_ms >= floor_cost - 1e-9]
            rungs = kept or rungs[-1:]
        if request.critical:
            return rungs[-1]
        for b in rungs:
            if b.bias <= request.bias \
                    and self._effective(b, request) <= request.delta:
                return b
        return rungs[-1]

    @staticmethod
    def _effective(budget: Budget, request: PrecisionRequest):
        """`delta`, widened by the shortfall a population cannot cancel.

        Every candidate of a population follows a DIFFERENT predecessor, so
        each carries a different fraction of a different score. That behaves
        as extra variance of `(1 - r)^2 * var_pred`, added to the
        measurement's own before the resolution is read off. A climb that
        always probes from its best has `var_pred = 0` and is unaffected, so
        no algorithm needs a hand-set minimum response.
        """
        if request.var_pred <= 0:
            return budget.delta
        extra = (budget.bias ** 2) * float(request.var_pred)
        return budget.delta * math.sqrt(
            1.0 + extra / max(budget.sigma ** 2, 1e-30))


# Stored profiles
def load_profile(layout, z, path=PROFILE_PATH):
    """The stored profile for a mirror, or None when there is none.

    Args:
        layout: Actuator count identifying the mirror (5 or 9).
        z: Sigma multiple, supplied by the caller (see `SpeedProfile`).
        path: Profile file; the default is the shared calibration file.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    entry = (data.get(PROFILE_KEY) or {}).get(str(int(layout)))
    if not entry:
        return None
    try:
        return SpeedProfile(entry["step"], entry["noise"],
                            entry.get("noise_sem", {}), z,
                            sessions=entry.get("sessions", 0),
                            source=entry.get("source", ""))
    except (KeyError, TypeError, ValueError):
        return None


def save_profile(layout, step, noise, noise_sem, sessions, source,
                 path=PROFILE_PATH):
    """Store one mirror's curves beside the other loop calibration.

    Only this mirror's entry is replaced; the file's other keys, and the other
    mirror's profile, are read back and written out unchanged.

    Returns:
        True when it reached disk.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {}
    data.setdefault(PROFILE_KEY, {})[str(int(layout))] = {
        "step": {str(k): round(float(v), 6) for k, v in step.items()},
        "noise": {str(k): float(v) for k, v in noise.items()},
        "noise_sem": {str(k): float(v) for k, v in noise_sem.items()},
        "sessions": int(sessions),
        "source": str(source)}
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        return True
    except (OSError, TypeError, ValueError):
        return False


class TierTally:
    """Points and wall-clock time spent at each rung of one run.

    Once the budget varies the loop pays a different price per point, so the
    run record has to say where the time went. Counts what was actually
    measured, not what was modelled.
    """

    def __init__(self):
        self.rows = {}
        self._finest_cost = None
        self._finest_tier = None

    def set_reference(self, budget: Budget):
        """Name the anchor budget the saving is measured against."""
        self._finest_cost = budget.cost_ms
        self._finest_tier = budget.tier

    def note(self, budget: Budget, elapsed_ms):
        """Add one finished point."""
        row = self.rows.setdefault(
            budget.tier, {"points": 0, "total_ms": 0.0,
                          "settle_ms": budget.settle_ms,
                          "frames": budget.frames, "sigma": budget.sigma,
                          "delta": budget.delta, "cost_ms": budget.cost_ms})
        row["points"] += 1
        if elapsed_ms is not None and math.isfinite(elapsed_ms):
            row["total_ms"] += float(elapsed_ms)

    @property
    def points(self):
        return sum(r["points"] for r in self.rows.values())

    @property
    def total_ms(self):
        return sum(r["total_ms"] for r in self.rows.values())

    def reference_ms(self):
        """What one point at the anchor budget costs, best evidence first.

        A run that took points at the anchor has MEASURED that price on this
        bench today, which beats the model: the model is a fit over earlier
        sessions and this machine may not match it. Only when the anchor was
        never used does the modelled cost stand in.
        """
        row = self.rows.get(self._finest_tier)
        if row and row["points"] >= _REFERENCE_POINTS:
            return row["total_ms"] / row["points"]
        return self._finest_cost

    def saved_ms(self):
        """Time not spent, versus every point at the anchor budget.

        Both sides are wall clock measured the same way, so this cannot be
        inflated by the cost model being optimistic.
        """
        ref = self.reference_ms()
        if ref is None:
            return 0.0
        return max(0.0, ref * self.points - self.total_ms)

    def summary(self):
        """`settle/frames n | ...` counts, cheapest rung first."""
        order = sorted(self.rows, key=lambda k: self.rows[k]["cost_ms"])
        return " | ".join(f"{k} {self.rows[k]['points']}" for k in order)

    def table(self):
        """Rows for `speed_tiers.csv`, cheapest rung first."""
        out = []
        for tier in sorted(self.rows, key=lambda k: self.rows[k]["cost_ms"]):
            row = self.rows[tier]
            n = max(1, row["points"])
            out.append([tier, row["points"], row["settle_ms"], row["frames"],
                        f"{row['total_ms']:.1f}", f"{row['total_ms'] / n:.1f}",
                        f"{row['sigma']:.6g}", f"{row['delta']:.6g}"])
        return out
