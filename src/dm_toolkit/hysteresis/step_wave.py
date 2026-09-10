# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 2.0, 2026-07-23

"""Step / square-wave drive sequences for the hysteresis step-tracking test."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .compensator import HysteresisCompensator

BIT_MIN, BIT_MAX = 0, 4095  # PWM range; the device profile clamps again.


def build_levels(mode, *, baseline, target=None, targets=None, levels=None,
                 repeats=1, centre=None, amplitude=None, points=None):
    """Ordered list of held bit LEVELS for one run.

    single    -> baseline, target, baseline, target, ... (repeats cycles)
    plan      -> for each t in targets: baseline, t ; whole list x repeats
    staircase -> the explicit `levels` list x repeats (baseline framing added)
    sine      -> centre + amplitude*sin(2 pi k / points), k=0..points-1, x
    repeats

    Args:
        mode: Operating or analysis mode.
        baseline: Reference level used to construct the sequence.
        target: Target value for the operation.
        targets: Target displacement or command values.
        levels: Display or quantisation levels.
        repeats: Number of repeated measurements.
        centre: Centre coordinate of the analyzed region.
        amplitude: Command or fitted response amplitude.
        points: Measurement or plot points.
    """
    b = int(baseline)
    reps = max(1, int(repeats))
    if mode == "sine":
        # A sampled sinusoid: consecutive samples near the peaks legitimately
        # round to the same bit, so this branch must NOT be de-duplicated --
        # collapsing them would shorten the crest and distort the period.
        c = int(centre if centre is not None else baseline)
        a = float(amplitude or 0.0)
        n = max(4, int(points or 24))
        seq = []
        for _ in range(reps):
            for k in range(n):
                seq.append(int(round(c + a * math.sin(2.0 * math.pi * k / n))))
        seq.append(c)  # Close the last cycle on centre.
        return _clamp(seq)
    if mode == "single":
        seq = [b]
        for _ in range(reps):
            seq += [int(target), b]
    elif mode == "plan":
        cycle = []
        for t in (targets or []):
            cycle += [b, int(t)]
        cycle += [b]  # return to baseline each cycle
        seq = (cycle * reps) if cycle else [b]
    elif mode == "staircase":
        base = [int(v) for v in (levels or [])]
        # A strictly increasing list is the up ramp and is mirrored back down;
        # a list that already turns around is used verbatim.
        if len(base) > 1 and all(base[i] < base[i + 1]
                                 for i in range(len(base) - 1)):
            base = base + base[-2::-1]
        seq = ([b] + base + [b]) * reps if base else [b]
    else:
        raise ValueError(f"unknown step mode {mode!r}")
    return _dedupe(seq)


def _dedupe(seq):
    """Return dedupe.

    Collapse consecutive equal levels (a repeated bit is just a longer hold,
    so it need not be a separate commanded step).
    """
    out = []
    for v in seq:
        if not out or out[-1] != v:
            out.append(v)
    return out


def _clamp(seq, lo=BIT_MIN, hi=BIT_MAX):
    return [max(lo, min(hi, int(v))) for v in seq]


@dataclass(frozen=True)
class StepPlan:
    profile_id: str
    levels: list  # Nominal bit level per step (what the axis means)
    ideal_nm: list  # Linear target; legacy profile: loading target.
    command_bits: list  # Bits actually commanded this pass (raw or comp)
    nominal_voltage_v: list
    applied_voltage_v: list
    compensated: bool
    n_setup: int = 0  # Leading levels that are the anchor pre-roll.
    zero_bit: int = 0  # Nominal bit whose hold defines the measurement zero.


def _loading_nm(comp, bit):
    """Single-valued loading-branch displacement at `bit` (no memory change).

    Args:
        comp: Compensator or compensation result.
        bit: PWM command bit.
    """
    return comp.plan_bit(int(bit), clamp=True).predicted_displacement_nm


def build_step_plan(profile, levels, *, compensated, anchor=None, home=True):
    """Convert nominal levels into a single-pass drive plan.

    Args:
        profile: Device or calibration profile.
        levels: Display or quantisation levels.
        compensated: Whether the dataset uses compensated commands.
        anchor: Reference command used to establish the displacement zero.
        home: Home the channel before the first level.
    """
    levels = [int(v) for v in levels]
    ref = HysteresisCompensator(profile)  # Fresh model for the reference.
    ref.commit(ref.plan_home())
    linearized = ref.profile.linearized_command is not None

    def _target(bit):
        if linearized:
            return float(ref.nominal_bit_to_displacement(int(bit), clamp=True))
        # Legacy absolute loading-branch displacement at that level.
        return float(_loading_nm(ref, int(bit)))

    if anchor is not None:
        home_bit = int(ref.plan_home().bit)
        pre = ([home_bit] if home else []) + [int(anchor)]
        n_home = 1 if home else 0  # The leading bit-0 reset, if present.
        zero_bit = int(anchor)
    else:
        pre = []
        n_home = 0
        zero_bit = levels[0] if levels else 0

    all_levels = pre + levels
    target_abs = [_target(b) for b in all_levels]

    comp = HysteresisCompensator(profile)
    comp.commit(comp.plan_home())
    curve = comp.voltage_curve
    cmd, applied = [], []
    for k, b in enumerate(all_levels):
        # The bit-0 home is always a RAW reset; the anchor and the whole
        # waveform follow the pass's scheme so the anchor sits on the same law
        # the ideal is drawn against.
        raw_here = k < n_home
        if compensated and not raw_here:
            if linearized:
                r = comp.plan_linearized_bit(b, clamp=True)
            else:
                r = comp.plan_displacement(target_abs[k], clamp=True)
        else:
            r = comp.plan_bit(b, clamp=True)
        comp.commit(r)  # Advance memory for the next step.
        cmd.append(int(r.bit))
        applied.append(float(r.applied_voltage_v))
    # Ideal shown/scored is RELATIVE to the zero bit, because the measured
    # displacement is zeroed there too (the settled anchor hold, or the first
    # recorded frame when there is no anchor).
    base = _target(zero_bit) if all_levels else 0.0
    ideal_rel = [x - base for x in target_abs]
    nominal_v = [float(curve.bit_to_voltage(b, clamp=True)) for b in all_levels]
    prof_id = (profile if isinstance(profile, str)
               else getattr(profile, "device_id", str(profile)))
    return StepPlan(profile_id=str(prof_id), levels=all_levels,
                    ideal_nm=ideal_rel, command_bits=cmd,
                    nominal_voltage_v=nominal_v, applied_voltage_v=applied,
                    compensated=bool(compensated), n_setup=len(pre),
                    zero_bit=int(zero_bit))
