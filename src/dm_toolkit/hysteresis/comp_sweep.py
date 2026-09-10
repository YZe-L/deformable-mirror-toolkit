# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.1, 2026-07-23

"""Hysteresis-compensated drive sequences for loop sweeps."""

from __future__ import annotations

from dataclasses import dataclass

from ..interferometry import loop
from .compensator import HysteresisCompensator
from .device_profile import PROFILE_DIRECTORY, load_device_profile


def available_profiles():
    """List the available device profiles.

    (name, display_name) for every device profile shipped with the app,
    name-sorted. Empty if the devices folder is missing.
    """
    out = []
    for path in sorted(PROFILE_DIRECTORY.glob("*.json")):
        try:
            prof = load_device_profile(path)
            out.append((path.stem, prof.display_name))
        except Exception:  # noqa: BLE001
            out.append((path.stem, path.stem))
    return out


@dataclass(frozen=True)
class CompensatedSweep:
    """A compensated drive plan sharing the raw sweep's nominal bit grid."""

    profile_id: str
    turn: int  # Index of the top (up->down turnaround)
    nominal_bits: list  # The raw bit grid (what the axis means)
    nominal_voltage_v: list  # Driver voltage of each nominal bit (x-axis)
    target_nm: list  # Linear target (legacy: loading reference)
    command_bits: list  # Compensated bits actually sent to the Pi.
    applied_voltage_v: list  # Driver voltage of each command bit.
    predicted_nm: list  # Model displacement the command should give.
    clamped: list  # Per-point: request fell outside model range.


def _loading_reference(comp, up_bits):
    """Build the monotonic loading-branch reference.

    Loading-curve displacement at each up-branch bit, from a homed model
    stepped monotonically up. Single-valued -> used as the target both ways.

    Args:
        comp: Compensator or compensation result.
        up_bits: PWM command bits for up.
    """
    comp.commit(comp.plan_home())
    ref = {}
    for bit in up_bits:
        result = comp.plan_bit(int(bit), clamp=True)
        ref[int(bit)] = comp.commit(result)  # Commit advances the memory.
    return ref


def build_compensated_sweep(profile, lo, hi, step):
    """Build compensated sweep.

    Compensated plan for one range, on the same nominal grid as the raw
    sweep. `profile` is a device name (e.g. 'dm_d') or a DeviceProfile.

    Args:
        profile: Device or calibration profile.
        lo: Lower bound.
        hi: Upper bound.
        step: Increment between consecutive values.
    """
    nominal_bits, turn = loop.bit_sequence(int(lo), int(hi), int(step))
    up_bits = nominal_bits[:turn + 1]

    # Build a single-valued target for the nominal grid. New profiles define a
    # linear nominal-bit displacement coordinate. The loading-curve reference
    # remains only as backward compatibility for older profiles.
    reference_comp = HysteresisCompensator(profile)
    linearized = reference_comp.profile.linearized_command is not None
    if linearized:
        ref = {
            int(bit): reference_comp.nominal_bit_to_displacement(int(bit))
            for bit in nominal_bits
        }
    else:
        ref = _loading_reference(reference_comp, up_bits)

    # Commands on a fresh model, committed in true sweep order (path-dependent)
    comp = HysteresisCompensator(profile)
    comp.commit(comp.plan_home())
    curve = comp.voltage_curve

    command_bits, applied_v, predicted, clamped = [], [], [], []
    for bit in nominal_bits:
        target = ref[int(bit)]
        if linearized:
            result = comp.plan_linearized_bit(int(bit), clamp=True)
        else:
            result = comp.plan_displacement(target, clamp=True)
        comp.commit(result)  # Advance memory for the next.
        command_bits.append(int(result.bit))
        applied_v.append(float(result.applied_voltage_v))
        predicted.append(float(result.predicted_displacement_nm))
        clamped.append(bool(result.clamped))

    nominal_v = [float(curve.bit_to_voltage(int(b), clamp=True))
                 for b in nominal_bits]
    target_nm = [float(ref[int(b)]) for b in nominal_bits]

    prof_id = (profile if isinstance(profile, str)
               else getattr(profile, "device_id", str(profile)))
    return CompensatedSweep(
        profile_id=str(prof_id), turn=turn,
        nominal_bits=[int(b) for b in nominal_bits],
        nominal_voltage_v=nominal_v, target_nm=target_nm,
        command_bits=command_bits, applied_voltage_v=applied_v,
        predicted_nm=predicted, clamped=clamped)
