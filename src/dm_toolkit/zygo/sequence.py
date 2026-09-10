# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 2.0, 2026-07-07

"""Scan plan: an ordered list of DM setpoints to measure."""

import itertools
from dataclasses import dataclass, field

BIT_MIN, BIT_MAX = 0, 4095


def clamp(bit, lo=BIT_MIN, hi=BIT_MAX):
    return int(max(lo, min(hi, round(bit))))


@dataclass
class Setpoint:
    channels: dict  # {channel:int -> bit:int}
    label: str = ""

    def clamped(self, lo=BIT_MIN, hi=BIT_MAX):
        return {int(c): clamp(b, lo, hi) for c, b in self.channels.items()}

    def folder_name(self):
        """Return folder name.

        Mx-side folder name for this point: piezo count + each channel's bit,
        e.g. {1:2000,3:0} -> '2p_c1b2000_c3b0'. Sorted by channel,
        filesystem-safe.
        """
        ch = {int(c): int(b) for c, b in self.channels.items()}
        parts = "_".join(f"c{c}b{ch[c]}" for c in sorted(ch))
        return f"{len(ch)}p_{parts}"


@dataclass
class ScanPlan:
    setpoints: list = field(default_factory=list)
    bias: int = 0  # Rest bit for channels not named in a setpoint.
    bit_min: int = BIT_MIN
    bit_max: int = BIT_MAX

    def __len__(self):
        return len(self.setpoints)

    def full_setpoint(self, sp, channels):
        """Expand one setpoint to every configured channel (unnamed -> bias).

        Args:
            sp: Full deformable-mirror setpoint mapping.
            channels: Actuator channel identifiers.
        """
        out = {c: self.bias for c in channels}
        out.update(sp.clamped(self.bit_min, self.bit_max))
        return out

    # Builders
    @staticmethod
    def single_channel_sweep(channels, values, bias=0, **kw):
        """For each channel, sweep it through `values` (others at bias).

        Args:
            channels: Actuator channel identifiers.
            values: Values processed by the operation.
            bias: Baseline PWM command applied to the actuator.
            **kw: Additional keyword arguments forwarded to the wrapped
                callable.
        """
        pts = []
        for c in channels:
            for v in values:
                pts.append(Setpoint({c: v}, label=f"ch{c}={v}"))
        return ScanPlan(pts, bias=bias, **kw)

    @staticmethod
    def all_together(channels, values, bias=0, **kw):
        """Move every channel to the same value, for each value.

        Args:
            channels: Actuator channel identifiers.
            values: Values processed by the operation.
            bias: Baseline PWM command applied to the actuator.
            **kw: Additional keyword arguments forwarded to the wrapped
                callable.
        """
        pts = [Setpoint({c: v for c in channels}, label=f"all={v}")
               for v in values]
        return ScanPlan(pts, bias=bias, **kw)

    @staticmethod
    def from_rows(rows, channels, bias=0, **kw):
        """rows: list of {channel: bit}. Free-form custom table.

        Args:
            rows: Input table rows.
            channels: Actuator channel identifiers.
            bias: Baseline PWM command applied to the actuator.
            **kw: Additional keyword arguments forwarded to the wrapped
                callable.
        """
        pts = [Setpoint(dict(r), label=", ".join(f"ch{c}={b}"
                                                 for c, b in r.items()))
               for r in rows]
        return ScanPlan(pts, bias=bias, **kw)

    @staticmethod
    def grid(channels, values, bias=0, **kw):
        """Build the full Cartesian scan grid.

        Full Cartesian product: every combination of `values` across every
        channel (all possibilities). N channels x M values -> M**N setpoints.
        Because `values` includes 0, single-piezo sweeps (one channel varied,
        the rest at 0) are naturally contained in the product. Order: last
        channel varies fastest.

        Args:
            channels: Actuator channel identifiers.
            values: Values processed by the operation.
            bias: Baseline PWM command applied to the actuator.
            **kw: Additional keyword arguments forwarded to the wrapped
                callable.
        """
        channels = [int(c) for c in channels]
        values = [clamp(v, kw.get("bit_min", BIT_MIN), kw.get("bit_max", BIT_MAX))
                  for v in values]
        pts = []
        for combo in itertools.product(values, repeat=len(channels)):
            chans = {c: v for c, v in zip(channels, combo)}
            label = ", ".join(f"ch{c}={chans[c]}" for c in channels)
            pts.append(Setpoint(chans, label=label))
        return ScanPlan(pts, bias=bias, **kw)

    @staticmethod
    def grid_count(n_channels, n_values):
        """Return the Cartesian scan-grid size.

        M**N without building the (possibly huge) list -- for a live preview.

        Args:
            n_channels: Number of n channels.
            n_values: Number of n values.
        """
        return int(n_values) ** int(n_channels)

    @staticmethod
    def continuous_ranges(channel_ranges, step, bias=0, **kw):
        """Build synchronized per-channel ramp ranges.

        Ramp several channels together, each over its OWN range, by `step`.

        Args:
            channel_ranges: Sequence of `(channel, low_bit, high_bit)` ranges
                advanced in lockstep.
            step: Increment between consecutive values.
            bias: Baseline PWM command applied to the actuator.
            **kw: Additional keyword arguments forwarded to the wrapped
                callable.
        """
        ramps = {int(c): _ramp(int(lo), int(hi), step)
                 for c, lo, hi in channel_ranges}
        if not ramps:
            return ScanPlan([], bias=bias, **kw)
        n = max(len(v) for v in ramps.values())
        pts = []
        for k in range(n):
            chans = {c: v[min(k, len(v) - 1)] for c, v in ramps.items()}
            label = ", ".join(f"ch{c}={chans[c]}" for c in sorted(chans))
            pts.append(Setpoint(chans, label=label))
        return ScanPlan(pts, bias=bias, **kw)


def _ramp(lo, hi, step):
    """Integer values lo..hi inclusive stepping by |step| (either direction).

    Args:
        lo: Lower bound.
        hi: Upper bound.
        step: Increment between consecutive values.
    """
    step = max(1, abs(int(step)))
    if hi >= lo:
        vals = list(range(lo, hi + 1, step))
    else:
        vals = list(range(lo, hi - 1, -step))
    if not vals:
        vals = [lo]
    if vals[-1] != hi:
        vals.append(hi)
    return vals


def values_by_step(step, lo=BIT_MIN, hi=BIT_MAX):
    """Build an inclusive stepped value sequence.

    Per-piezo value list lo..hi by |step|, always including both ends (so 0
    and the top are present). This is each channel's axis in ScanPlan.grid.

    Args:
        step: Increment between consecutive values.
        lo: Lower bound.
        hi: Upper bound.
    """
    return _ramp(int(lo), int(hi), int(step))


def linspace_bits(lo, hi, n):
    """n integer bit values from lo to hi inclusive.

    Args:
        lo: Lower bound.
        hi: Upper bound.
        n: Number of requested samples or output points.
    """
    if n <= 1:
        return [clamp(lo)]
    step = (hi - lo) / (n - 1)
    return [clamp(lo + i * step) for i in range(n)]
