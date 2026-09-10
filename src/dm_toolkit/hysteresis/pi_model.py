# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.1, 2026-07-23

"""Stateful provisional modified Prandtl-Ishlinskii model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _chebyshev(coefficients: tuple[float, ...], x: float) -> float:
    """Evaluate a Chebyshev series using Clenshaw recurrence.

    Args:
        coefficients: Model coefficients.
        x: Input coordinate or scalar value.
    """
    if not coefficients:
        return 0.0
    b_k1 = 0.0
    b_k2 = 0.0
    for coefficient in reversed(coefficients[1:]):
        b_k = 2.0 * x * b_k1 - b_k2 + coefficient
        b_k2, b_k1 = b_k1, b_k
    return x * b_k1 - b_k2 + coefficients[0]


@dataclass
class PlayOperator:
    threshold_v: float
    weight_nm_per_v: float
    state_v: float = 0.0

    def next_state(self, input_v: float) -> float:
        return max(
            input_v - self.threshold_v,
            min(input_v + self.threshold_v, self.state_v),
        )


@dataclass(frozen=True)
class InverseResult:
    requested_nm: float
    effective_voltage_v: float
    predicted_nm: float
    reachable_min_nm: float
    reachable_max_nm: float
    clamped: bool


class ModifiedPrandtlIshlinskii:
    """Stateful forward and numerical inverse model loaded from a profile."""

    def __init__(self, model_config: dict[str, Any]):
        primary = model_config["primary_response"]
        self.input_min_v = float(model_config["effective_input_min_v"])
        self.input_max_v = float(model_config["effective_input_max_v"])
        self.primary_domain = tuple(float(value) for value in primary["domain_v"])
        self.primary_coefficients = tuple(float(value) for value in primary["chebyshev_coefficients"])
        self.primary_zero_nm = float(primary["raw_zero_nm"])
        self.primary_scale = float(primary["output_scale"])
        self.operators = [
            PlayOperator(
                threshold_v=float(item["threshold_v"]),
                weight_nm_per_v=float(item["weight_nm_per_v"]),
            )
            for item in model_config["play_operators"]
        ]
        self.current_input_v = self.input_min_v

    def reset(self, effective_voltage_v: float = 0.0) -> float:
        """Reset model memory onto the initial monotonic loading branch."""
        input_v = self._bounded_input(effective_voltage_v)
        self.current_input_v = input_v
        for operator in self.operators:
            operator.state_v = max(input_v - operator.threshold_v, 0.0)
        return self.predict(input_v)

    def _bounded_input(self, value: float) -> float:
        value = float(value)
        if not self.input_min_v <= value <= self.input_max_v:
            raise ValueError(
                f"effective voltage {value:.6f} V outside "
                f"[{self.input_min_v:.6f}, {self.input_max_v:.6f}] V"
            )
        return value

    def primary_response(self, effective_voltage_v: float) -> float:
        low, high = self.primary_domain
        x = 2.0 * (effective_voltage_v - low) / (high - low) - 1.0
        raw = _chebyshev(self.primary_coefficients, x)
        return (raw - self.primary_zero_nm) * self.primary_scale

    def _evaluate(self, effective_voltage_v: float, *, commit: bool) -> float:
        """Evaluate the hysteresis model at one effective voltage.

        Args:
            effective_voltage_v: Effective voltage, in volts.
            commit: Whether to update the model's internal state.
        """
        input_v = self._bounded_input(effective_voltage_v)
        output_nm = self.primary_response(input_v)
        next_states: list[float] = []
        for operator in self.operators:
            state = operator.next_state(input_v)
            initial_loading = max(input_v - operator.threshold_v, 0.0)
            output_nm += operator.weight_nm_per_v * (state - initial_loading)
            next_states.append(state)
        if commit:
            self.current_input_v = input_v
            for operator, state in zip(self.operators, next_states):
                operator.state_v = state
        return output_nm

    def predict(self, effective_voltage_v: float) -> float:
        """Predict one transition without changing hysteresis memory."""
        return self._evaluate(effective_voltage_v, commit=False)

    def commit(self, effective_voltage_v: float) -> float:
        """Advance model memory after a command was successfully applied."""
        return self._evaluate(effective_voltage_v, commit=True)

    def inverse(self, target_nm: float, *, clamp: bool = False) -> InverseResult:
        """Invert the next model-state transition.

        Numerically invert the next state transition for the current memory.

        Args:
            target_nm: Target displacement, in nanometres.
            clamp: Whether to clamp values to the supported range.
        """
        requested = float(target_nm)
        low_v = self.input_min_v
        high_v = self.input_max_v
        low_nm = self.predict(low_v)
        high_nm = self.predict(high_v)
        if high_nm < low_nm:
            raise RuntimeError("model is not monotonic for the current state")

        target = requested
        was_clamped = False
        endpoint_tolerance_nm = 1e-9 * max(
            1.0,
            abs(low_nm),
            abs(high_nm),
        )
        if (
            target < low_nm - endpoint_tolerance_nm
            or target > high_nm + endpoint_tolerance_nm
        ):
            if not clamp:
                raise ValueError(
                    f"target {target:.3f} nm is not reachable from current model state; "
                    f"next-command range is {low_nm:.3f}..{high_nm:.3f} nm"
                )
            target = min(max(target, low_nm), high_nm)
            was_clamped = True
        else:
            # Profile endpoints are rounded decimal calibration values, while
            # the Chebyshev evaluation is binary floating point. Treat a
            # sub-micro-nanometre mismatch as the same reachable endpoint.
            target = min(max(target, low_nm), high_nm)

        for _ in range(56):
            middle_v = (low_v + high_v) / 2.0
            if self.predict(middle_v) < target:
                low_v = middle_v
            else:
                high_v = middle_v
        effective_v = (low_v + high_v) / 2.0
        return InverseResult(
            requested_nm=requested,
            effective_voltage_v=effective_v,
            predicted_nm=self.predict(effective_v),
            reachable_min_nm=low_nm,
            reachable_max_nm=high_nm,
            clamped=was_clamped,
        )

    def state(self) -> dict[str, Any]:
        return {
            "effective_voltage_v": self.current_input_v,
            "predicted_displacement_nm": self.predict(self.current_input_v),
            "play_states_v": [operator.state_v for operator in self.operators],
        }
