# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-22

"""Convert 12-bit driver commands to fitted output voltages."""

from __future__ import annotations

from math import exp, floor, isfinite, log1p


SOURCE_WORKBOOK = "6.5测试驱动器输出端口电压特性.xlsx"
SOURCE_SHEET = "电压曲线汇总"

MIN_BIT = 0
MAX_BIT = 4095
SATURATION_BIT = 4090
MIN_VOLTAGE = 1.05685
MAX_VOLTAGE = 150.2

# Revalidate fit error and monotonicity after changing these coefficients.
_START_SLOPE = 0.048536857539800429
_START_KNEE = 24.083446806881689
_START_WIDTH = 3.3867549167711477
_POLYNOMIAL = (
    -45.213682416196917,  # x^2
    -11.133544993726920,  # x^3
    25.733069170901469,  # x^4
    -17.390692759302873,  # x^5
)
_SATURATION_WIDTH = 0.036527059838893761


def _number(value: float | int, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} 必须是有限数值") from exc
    if not isfinite(result):
        raise ValueError(f"{name} 必须是有限数值")
    return result


def _limit(value: float, low: float, high: float, *, clamp: bool, name: str) -> float:
    if low <= value <= high:
        return value
    # The fitted endpoint can differ from the declared limit by a few ulps.
    tolerance = 1e-12 * max(1.0, abs(low), abs(high))
    if low - tolerance <= value <= high + tolerance:
        return min(max(value, low), high)
    if not clamp:
        raise ValueError(f"{name}={value} 超出有效范围 [{low}, {high}]")
    return min(max(value, low), high)


def _softplus(value: float) -> float:
    """Evaluate softplus without numerical overflow."""
    if value > 40.0:
        return value
    if value < -40.0:
        return exp(value)
    return log1p(exp(value))


def _curve(bit_value: float) -> float:
    """Evaluate the fitted voltage at a continuous bit coordinate."""
    normalized = bit_value / MAX_BIT
    raw_voltage = MIN_VOLTAGE + _START_SLOPE * _START_WIDTH * (
        _softplus((bit_value - _START_KNEE) / _START_WIDTH)
        - _softplus(-_START_KNEE / _START_WIDTH)
    )

    for power, coefficient in enumerate(_POLYNOMIAL, start=2):
        raw_voltage += coefficient * normalized**power

    # Smooth the saturation limit to preserve monotonicity.
    return MAX_VOLTAGE - _SATURATION_WIDTH * _softplus(
        (MAX_VOLTAGE - raw_voltage) / _SATURATION_WIDTH
    )


def voltage_limits() -> tuple[float, float]:
    """Return the measured voltage range covered by the model."""
    return MIN_VOLTAGE, MAX_VOLTAGE


def bit_to_voltage(bit_value: float | int, *, clamp: bool = True) -> float:
    """Return the fitted voltage for the nearest integer bit.

    Args:
        bit_value: PWM command bit.
        clamp: Whether to clamp values to the supported range.
    """
    value = _limit(
        _number(bit_value, "bit_value"),
        MIN_BIT,
        MAX_BIT,
        clamp=clamp,
        name="bit_value",
    )
    integer_bit = min(max(floor(value + 0.5), MIN_BIT), MAX_BIT)
    return _curve(float(integer_bit))


def voltage_to_bit(voltage: float | int, *, clamp: bool = True) -> int:
    """Return the integer bit with the smallest fitted voltage error.

    Args:
        voltage: Drive voltage, in volts.
        clamp: Whether to clamp values to the supported range.
    """
    target = _limit(
        _number(voltage, "voltage"),
        MIN_VOLTAGE,
        MAX_VOLTAGE,
        clamp=clamp,
        name="voltage",
    )
    if target <= MIN_VOLTAGE:
        return MIN_BIT
    if target >= MAX_VOLTAGE:
        return SATURATION_BIT

    low = float(MIN_BIT)
    high = float(SATURATION_BIT)
    for _ in range(48):
        middle = (low + high) / 2.0
        if _curve(middle) < target:
            low = middle
        else:
            high = middle

    lower_bit = min(max(floor((low + high) / 2.0), MIN_BIT), SATURATION_BIT)
    upper_bit = min(lower_bit + 1, SATURATION_BIT)
    lower_error = abs(_curve(float(lower_bit)) - target)
    upper_error = abs(_curve(float(upper_bit)) - target)
    return lower_bit if lower_error <= upper_error else upper_bit


def _self_test() -> None:
    """Check boundaries, monotonicity, and inverse consistency."""
    voltages = tuple(_curve(float(bit)) for bit in range(MAX_BIT + 1))
    assert all(a <= b for a, b in zip(voltages, voltages[1:]))
    assert abs(bit_to_voltage(0) - MIN_VOLTAGE) < 1e-12
    assert voltage_to_bit(MIN_VOLTAGE) == MIN_BIT
    assert voltage_to_bit(MAX_VOLTAGE) == SATURATION_BIT
    assert voltage_to_bit(100.45) == 2400

    for bit in (0, 25, 100, 1000, 2400, 3900, 4080):
        recovered = voltage_to_bit(bit_to_voltage(bit))
        assert abs(recovered - bit) <= 1


if __name__ == "__main__":
    _self_test()
    print(f"拟合电压范围: {MIN_VOLTAGE:.6f} V 至 {MAX_VOLTAGE:.6f} V")
    print(f"目标 50.0 V -> bit {voltage_to_bit(50.0)}")
    print(f"bit 2400 -> {bit_to_voltage(2400):.6f} V")
