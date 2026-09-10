# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.1, 2026-07-22

"""Profile-selectable conversion between integer PWM bit and driver voltage."""

from __future__ import annotations

from math import floor, isfinite
from typing import Any

from . import driver_voltage_lut
from .pi_model import _chebyshev


def _number(value: float | int, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _limit(value: float, low: float, high: float, *, clamp: bool, name: str) -> float:
    if low <= value <= high:
        return value
    # Inverse PI arithmetic can land a few ulps beyond an exact endpoint.
    # Treat that round-off as the endpoint, while preserving real range errors.
    tolerance = 1e-12 * max(1.0, abs(low), abs(high))
    if low - tolerance <= value <= high + tolerance:
        return min(max(value, low), high)
    if not clamp:
        raise ValueError(f"{name}={value} outside valid range [{low}, {high}]")
    return min(max(value, low), high)


class LegacyDriverVoltageCurve:
    """Adapter retaining the original shared driver fit for older profiles."""

    def __init__(self, minimum_bit: int, maximum_bit: int):
        self.minimum_bit = int(minimum_bit)
        self.maximum_bit = int(maximum_bit)
        self.minimum_voltage_v = driver_voltage_lut.bit_to_voltage(self.minimum_bit)
        self.maximum_voltage_v = driver_voltage_lut.bit_to_voltage(self.maximum_bit)

    def bit_to_voltage(self, bit_value: float | int, *, clamp: bool = True) -> float:
        """Convert a PWM command bit to drive voltage.

        Args:
            bit_value: PWM command bit.
            clamp: Whether to clamp values to the supported range.
        """
        value = _limit(
            _number(bit_value, "bit_value"),
            self.minimum_bit,
            self.maximum_bit,
            clamp=clamp,
            name="bit_value",
        )
        return driver_voltage_lut.bit_to_voltage(value, clamp=False)

    def voltage_to_bit(self, voltage_v: float | int, *, clamp: bool = True) -> int:
        """Convert drive voltage to a PWM command bit.

        Args:
            voltage_v: Drive voltage, in volts.
            clamp: Whether to clamp values to the supported range.
        """
        target = _limit(
            _number(voltage_v, "voltage_v"),
            self.minimum_voltage_v,
            self.maximum_voltage_v,
            clamp=clamp,
            name="voltage_v",
        )
        # clamp=True: this curve's own minimum can round a hair below the
        # LUT's rail, so a rail voltage is not rejected here.
        result = driver_voltage_lut.voltage_to_bit(target, clamp=True)
        return min(max(result, self.minimum_bit), self.maximum_bit)


class ChebyshevDriverVoltageCurve:
    """Compact monotonic Chebyshev voltage fit stored in a device profile."""

    def __init__(self, config: dict[str, Any]):
        self.minimum_bit = int(config["minimum_bit"])
        self.maximum_bit = int(config["maximum_bit"])
        self.minimum_voltage_v = float(config["minimum_voltage_v"])
        self.maximum_voltage_v = float(config["maximum_voltage_v"])
        self.coefficients = tuple(float(value) for value in config["chebyshev_coefficients"])
        self.raw_zero_v = float(config["raw_zero_v"])
        self.output_scale = float(config["output_scale"])
        if self.minimum_bit >= self.maximum_bit:
            raise ValueError("driver voltage curve requires minimum_bit < maximum_bit")
        if self.minimum_voltage_v >= self.maximum_voltage_v:
            raise ValueError("driver voltage curve requires increasing voltage limits")

        values = [self._curve(float(bit)) for bit in range(self.minimum_bit, self.maximum_bit + 1)]
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("profile driver voltage curve must be strictly increasing")
        if abs(values[0] - self.minimum_voltage_v) > 1e-6:
            raise ValueError("profile driver curve does not match minimum_voltage_v")
        if abs(values[-1] - self.maximum_voltage_v) > 1e-6:
            raise ValueError("profile driver curve does not match maximum_voltage_v")

    def _curve(self, bit_value: float) -> float:
        normalized = 2.0 * (bit_value - self.minimum_bit) / (
            self.maximum_bit - self.minimum_bit
        ) - 1.0
        raw = _chebyshev(self.coefficients, normalized)
        return self.minimum_voltage_v + (raw - self.raw_zero_v) * self.output_scale

    def bit_to_voltage(self, bit_value: float | int, *, clamp: bool = True) -> float:
        """Convert a PWM command bit to drive voltage.

        Args:
            bit_value: PWM command bit.
            clamp: Whether to clamp values to the supported range.
        """
        value = _limit(
            _number(bit_value, "bit_value"),
            self.minimum_bit,
            self.maximum_bit,
            clamp=clamp,
            name="bit_value",
        )
        integer_bit = min(
            max(floor(value + 0.5), self.minimum_bit), self.maximum_bit
        )
        return self._curve(float(integer_bit))

    def voltage_to_bit(self, voltage_v: float | int, *, clamp: bool = True) -> int:
        """Convert drive voltage to a PWM command bit.

        Args:
            voltage_v: Drive voltage, in volts.
            clamp: Whether to clamp values to the supported range.
        """
        target = _limit(
            _number(voltage_v, "voltage_v"),
            self.minimum_voltage_v,
            self.maximum_voltage_v,
            clamp=clamp,
            name="voltage_v",
        )
        if target <= self.minimum_voltage_v:
            return self.minimum_bit
        if target >= self.maximum_voltage_v:
            return self.maximum_bit

        low = float(self.minimum_bit)
        high = float(self.maximum_bit)
        for _ in range(52):
            middle = (low + high) / 2.0
            if self._curve(middle) < target:
                low = middle
            else:
                high = middle
        lower_bit = min(max(floor((low + high) / 2.0), self.minimum_bit), self.maximum_bit)
        upper_bit = min(lower_bit + 1, self.maximum_bit)
        lower_error = abs(self._curve(float(lower_bit)) - target)
        upper_error = abs(self._curve(float(upper_bit)) - target)
        return lower_bit if lower_error <= upper_error else upper_bit


def voltage_curve_from_profile(profile_data: dict[str, Any]):
    """Build a voltage curve from a device profile.

    Build a profile-specific curve, falling back to the original shared fit.
    """
    hardware = profile_data["hardware"]
    config = profile_data.get("driver_voltage_model")
    if config is None:
        return LegacyDriverVoltageCurve(
            minimum_bit=int(hardware["minimum_bit"]),
            maximum_bit=int(hardware["maximum_bit"]),
        )
    if config.get("type") != "chebyshev":
        raise ValueError(f"unsupported driver voltage model {config.get('type')!r}")
    return ChebyshevDriverVoltageCurve(config)
