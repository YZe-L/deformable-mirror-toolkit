# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.2, 2026-07-28

"""Shared auto-exposure policy for the closed loop and the automation bench."""

from __future__ import annotations

# Idle live-preview trip band (raw counts). Over TRIP we duck; we hold until the
# peak falls under RETURN, then climb back to the base.
PREVIEW_TRIP_COUNTS = 1000.0
PREVIEW_DUCK_COUNTS = 850.0  # Land the peak here when ducking (margin)

RETURN_COUNTS = 100.0  # return to base only below this peak
FLOOR_MS = 0.04  # Never duck below this exposure.

# Per-point targeting: the peak is put on TARGET x full scale before a point
# is scored; TARGET_BAND is the deadband that keeps this from chasing noise.
TARGET_BAND = 0.20
MAX_RETARGETS = 2  # Per point, then score what we have.


def exposure_for_target(peak, cur, maxv, target, band=TARGET_BAND):
    """Exposure that puts `peak` on `target * maxv`, or None to hold.

    peak/maxv are raw counts (peak should be a ROBUST peak -- a hot pixel must
    not set the exposure). Returns None inside the deadband, at which point the
    caller scores the frames it already has.

    Args:
        peak: Peak image intensity.
        cur: Current parameter or command value.
        maxv: Maximum finite value used for normalisation.
        target: Target value for the operation.
        band: Target peak-intensity band.
    """
    if not (maxv > 0 and cur > 0 and peak > 0 and 0 < target < 1):
        return None
    frac = peak / maxv
    if abs(frac - target) <= band * target:
        return None
    return max(FLOOR_MS, cur * (target / frac))


def preview_exposure(peak, cur, base, *, trip=PREVIEW_TRIP_COUNTS,
                     duck_to=PREVIEW_DUCK_COUNTS, floor=FLOOR_MS,
                     return_counts=RETURN_COUNTS):
    """Idle-preview exposure to command, or ``None`` to hold.

    Over ``trip`` the exposure ducks so the peak lands near ``duck_to`` and
    holds there until the peak drops under ``return_counts``, then returns
    to ``base``.

    Args:
        peak: Current frame peak, in raw counts.
        cur: Current exposure, in milliseconds.
        base: Fixed base exposure, in milliseconds.
        trip: Peak count above which the exposure ducks.
        duck_to: Peak count the ducked exposure aims for.
        floor: Lowest exposure allowed, in milliseconds.
        return_counts: Peak count below which the base is restored.
    """
    if base <= 0.0:  # No base set yet: leave the camera be.
        return None
    if peak > trip:
        target = cur * (duck_to / max(peak, 1.0))
        return max(floor, min(base, target))
    if cur < base and peak < return_counts:
        return base  # Dim enough: safe to climb back.
    return None  # Hold the (possibly ducked) set-point.


def should_return_to_base(peak, cur, base, *, return_counts=RETURN_COUNTS):
    """Whether a ducked exposure may climb back to the base.

    True only when ducked below the base and the spot has gone dim, so a
    ducked exposure sticks instead of bouncing on every bright/dim point.

    Args:
        peak: Current frame peak, in raw counts.
        cur: Current exposure, in milliseconds.
        base: Fixed base exposure, in milliseconds.
        return_counts: Peak count below which the base is restored.
    """
    return base > 0.0 and cur < base and peak < return_counts
