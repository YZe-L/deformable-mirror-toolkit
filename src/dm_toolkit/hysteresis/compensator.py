# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 2.1, 2026-08-14

"""Combine the Piezo hysteresis model with the driver bit/voltage curve."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .device_profile import DeviceProfile, load_device_profile
from .driver_voltage_curve import voltage_curve_from_profile
from .pi_model import ModifiedPrandtlIshlinskii


@dataclass(frozen=True)
class CommandResult:
    """A planned hardware command. Planning never changes model memory."""

    command_kind: str
    requested_value: float
    requested_unit: str
    bit: int
    target_voltage_v: float
    applied_voltage_v: float
    effective_voltage_v: float
    predicted_displacement_nm: float
    target_displacement_nm: float | None = None
    clamped: bool = False
    reachable_min_nm: float | None = None
    reachable_max_nm: float | None = None


class HysteresisCompensator:
    """Stateful command planner for one device profile.

    Call :meth:`plan_*`, write the returned integer bit to the hardware, and
    only then call :meth:`commit`.  This prevents a failed I2C write from
    advancing the software hysteresis memory.
    """

    def __init__(self, profile: DeviceProfile | str = "piezo_a"):
        self.profile = (
            profile if isinstance(profile, DeviceProfile) else load_device_profile(profile)
        )
        self.model = ModifiedPrandtlIshlinskii(self.profile.model)
        self.voltage_curve = voltage_curve_from_profile(self.profile.data)
        self._origin_v = float(self.profile.model["voltage_origin_v"])
        self._minimum_voltage_v = float(self.profile.limits["minimum_command_voltage_v"])
        self._maximum_voltage_v = float(self.profile.limits["maximum_command_voltage_v"])
        self._minimum_bit = int(self.profile.hardware["minimum_bit"])
        self._maximum_bit = int(self.profile.hardware["maximum_bit"])
        self._linearized_command = self.profile.linearized_command
        self.homed = False
        self.current_bit: int | None = None
        self.current_voltage_v: float | None = None
        self.current_displacement_nm: float | None = None

    @staticmethod
    def _finite(value: float | int, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a finite number") from exc
        if not isfinite(number):
            raise ValueError(f"{name} must be a finite number")
        return number

    def _require_home(self) -> None:
        if not self.homed:
            raise RuntimeError("model state is unknown; apply the 'home' command first")

    def _bounded_voltage(self, value: float, *, clamp: bool) -> tuple[float, bool]:
        if self._minimum_voltage_v <= value <= self._maximum_voltage_v:
            return value, False
        if not clamp:
            raise ValueError(
                f"voltage {value:.6f} V outside calibrated command range "
                f"{self._minimum_voltage_v:.6f}..{self._maximum_voltage_v:.6f} V"
            )
        return min(max(value, self._minimum_voltage_v), self._maximum_voltage_v), True

    def _result_from_voltage(
        self,
        *,
        command_kind: str,
        requested_value: float,
        requested_unit: str,
        target_voltage_v: float,
        target_displacement_nm: float | None = None,
        clamped: bool = False,
        reachable_min_nm: float | None = None,
        reachable_max_nm: float | None = None,
    ) -> CommandResult:
        # The PC copy tolerates floating-point drift at the calibrated rails.
        # The next line still clamps the bit to the profile range, while
        # `clamp=False` would reject values such as 1.0568499999 vs 1.05685.
        """Build a command result from a target voltage.

        Args:
            command_kind: Kind of command requested by the caller.
            requested_value: Original value requested by the caller.
            requested_unit: Unit supplied with the requested command.
            target_voltage_v: Target voltage, in volts.
            target_displacement_nm: Target displacement, in nanometres.
            clamped: Whether the request was clamped to the device range.
            reachable_min_nm: Reachable minimum, in nanometres.
            reachable_max_nm: Reachable maximum, in nanometres.
        """
        bit = self.voltage_curve.voltage_to_bit(target_voltage_v, clamp=True)
        bit = min(max(bit, self._minimum_bit), self._maximum_bit)
        applied_voltage_v = self.voltage_curve.bit_to_voltage(bit, clamp=True)
        effective_voltage_v = min(
            max(applied_voltage_v - self._origin_v, self.model.input_min_v),
            self.model.input_max_v,
        )
        predicted_nm = self.model.predict(effective_voltage_v)
        return CommandResult(
            command_kind=command_kind,
            requested_value=requested_value,
            requested_unit=requested_unit,
            bit=bit,
            target_voltage_v=target_voltage_v,
            applied_voltage_v=applied_voltage_v,
            effective_voltage_v=effective_voltage_v,
            predicted_displacement_nm=predicted_nm,
            target_displacement_nm=target_displacement_nm,
            clamped=clamped,
            reachable_min_nm=reachable_min_nm,
            reachable_max_nm=reachable_max_nm,
        )

    def plan_home(self) -> CommandResult:
        """Return the bit-zero command without changing model memory."""
        applied_voltage_v = self.voltage_curve.bit_to_voltage(self._minimum_bit, clamp=False)
        return CommandResult(
            command_kind="home",
            requested_value=0.0,
            requested_unit="bit",
            bit=self._minimum_bit,
            target_voltage_v=applied_voltage_v,
            applied_voltage_v=applied_voltage_v,
            effective_voltage_v=0.0,
            predicted_displacement_nm=0.0,
            target_displacement_nm=0.0,
        )

    def _plan_target_displacement(
        self,
        target_nm: float,
        *,
        command_kind: str,
        requested_value: float,
        requested_unit: str,
        clamp: bool,
        already_clamped: bool = False,
    ) -> CommandResult:
        """Plan a command for a target displacement.

        Args:
            target_nm: Target displacement, in nanometres.
            command_kind: Kind of command requested by the caller.
            requested_value: Original value requested by the caller.
            requested_unit: Unit supplied with the requested command.
            clamp: Whether to clamp values to the supported range.
            already_clamped: Whether an earlier planning stage already clamped
                the requested value.
        """
        inverse = self.model.inverse(target_nm, clamp=clamp)
        target_voltage_v = self._origin_v + inverse.effective_voltage_v
        return self._result_from_voltage(
            command_kind=command_kind,
            requested_value=requested_value,
            requested_unit=requested_unit,
            target_voltage_v=target_voltage_v,
            target_displacement_nm=target_nm,
            clamped=already_clamped or inverse.clamped,
            reachable_min_nm=inverse.reachable_min_nm,
            reachable_max_nm=inverse.reachable_max_nm,
        )

    def plan_displacement(self, target_nm: float, *, clamp: bool = False) -> CommandResult:
        """Plan a compensated command for a requested relative displacement.

        Args:
            target_nm: Target displacement, in nanometres.
            clamp: Whether to clamp values to the supported range.
        """
        self._require_home()
        requested = self._finite(target_nm, "target_nm")
        return self._plan_target_displacement(
            requested,
            command_kind="displacement",
            requested_value=requested,
            requested_unit="nm",
            clamp=clamp,
        )

    def nominal_bit_to_displacement(
        self,
        nominal_bit: int,
        *,
        clamp: bool = False,
    ) -> float:
        """Map a nominal bit to displacement.

        Map a nominal control bit onto the profile's linear displacement axis.

        Args:
            nominal_bit: Nominal linearised PWM command bit.
            clamp: Whether to clamp values to the supported range.
        """
        config = self._linearized_command
        if config is None:
            raise RuntimeError(
                f"device profile {self.profile.device_id!r} does not define "
                "a linearized nominal-bit command"
            )
        if isinstance(nominal_bit, bool) or not isinstance(nominal_bit, int):
            raise TypeError("nominal_bit must be an integer")

        minimum_bit = int(config["minimum_nominal_bit"])
        maximum_bit = int(config["maximum_nominal_bit"])
        bounded = nominal_bit
        if not minimum_bit <= bounded <= maximum_bit:
            if not clamp:
                raise ValueError(
                    f"nominal bit {bounded} outside linearized range "
                    f"{minimum_bit}..{maximum_bit}"
                )
            bounded = min(max(bounded, minimum_bit), maximum_bit)

        minimum_nm = float(config["minimum_target_displacement_nm"])
        maximum_nm = float(config["maximum_target_displacement_nm"])
        fraction = (bounded - minimum_bit) / (maximum_bit - minimum_bit)
        return minimum_nm + fraction * (maximum_nm - minimum_nm)

    def displacement_to_nominal_bit(
        self,
        displacement_nm: float,
        *,
        clamp: bool = False,
    ) -> int:
        """Map a physical displacement onto the linear nominal-bit axis.

        This is the inverse of :meth:`nominal_bit_to_displacement`.  It does
        not inspect or change PI-model memory: the nominal axis describes the
        requested displacement, while hysteresis memory is used later when
        that nominal bit is planned for the hardware.

        Args:
            displacement_nm: Target physical displacement in nanometres.
            clamp: Whether to clamp a displacement outside the calibrated
                nominal range.

        Returns:
            Nearest integer nominal bit for the requested displacement.
        """
        config = self._linearized_command
        if config is None:
            raise RuntimeError(
                f"device profile {self.profile.device_id!r} does not define "
                "a linearized nominal-bit command"
            )
        requested = self._finite(displacement_nm, "displacement_nm")
        minimum_bit = int(config["minimum_nominal_bit"])
        maximum_bit = int(config["maximum_nominal_bit"])
        minimum_nm = float(config["minimum_target_displacement_nm"])
        maximum_nm = float(config["maximum_target_displacement_nm"])
        bounded = requested
        if not minimum_nm <= bounded <= maximum_nm:
            if not clamp:
                raise ValueError(
                    f"displacement {requested:.3f} nm outside linearized "
                    f"range {minimum_nm:.3f}..{maximum_nm:.3f} nm"
                )
            bounded = min(max(bounded, minimum_nm), maximum_nm)
        fraction = (bounded - minimum_nm) / (maximum_nm - minimum_nm)
        nominal = round(minimum_bit + fraction * (maximum_bit - minimum_bit))
        return int(min(max(nominal, minimum_bit), maximum_bit))

    def raw_bit_to_equivalent_nominal_bit(
        self,
        raw_bit: int,
        *,
        clamp: bool = False,
    ) -> int:
        """Return the nominal bit with the same predicted physical position.

        Planning the raw command uses the current PI memory but does not
        commit it.  The resulting predicted displacement is then expressed on
        the profile's linear nominal axis.  A loop can therefore measure its
        raw Start-bit state first and initialise either a search or a modal
        solve at the physically same state on the compensated axis.

        Args:
            raw_bit: Hardware bit that will be used for the raw Start point.
            clamp: Whether either conversion may clamp to its calibrated
                range.
        """
        command = self.plan_bit(int(raw_bit), clamp=clamp)
        return self.displacement_to_nominal_bit(
            command.predicted_displacement_nm, clamp=clamp)

    def plan_linearized_bit(
        self,
        nominal_bit: int,
        *,
        clamp: bool = False,
    ) -> CommandResult:
        """Plan a linear-displacement, hysteresis-compensated nominal bit.

        Args:
            nominal_bit: Nominal linearised PWM command bit.
            clamp: Whether to clamp values to the supported range.
        """
        self._require_home()
        if isinstance(nominal_bit, bool) or not isinstance(nominal_bit, int):
            raise TypeError("nominal_bit must be an integer")
        config = self._linearized_command
        if config is None:
            self.nominal_bit_to_displacement(nominal_bit, clamp=clamp)
            raise AssertionError("unreachable")

        minimum_bit = int(config["minimum_nominal_bit"])
        maximum_bit = int(config["maximum_nominal_bit"])
        bounded = nominal_bit
        nominal_was_clamped = False
        if not minimum_bit <= bounded <= maximum_bit:
            if not clamp:
                raise ValueError(
                    f"nominal bit {bounded} outside linearized range "
                    f"{minimum_bit}..{maximum_bit}"
                )
            bounded = min(max(bounded, minimum_bit), maximum_bit)
            nominal_was_clamped = True

        target_nm = self.nominal_bit_to_displacement(bounded, clamp=False)
        return self._plan_target_displacement(
            target_nm,
            command_kind="linearized_bit",
            requested_value=float(nominal_bit),
            requested_unit="nominal_bit",
            clamp=clamp,
            already_clamped=nominal_was_clamped,
        )

    def plan_voltage(self, voltage_v: float, *, clamp: bool = False) -> CommandResult:
        """Plan a direct physical-voltage command while tracking PI memory.

        Args:
            voltage_v: Drive voltage, in volts.
            clamp: Whether to clamp values to the supported range.
        """
        self._require_home()
        requested = self._finite(voltage_v, "voltage_v")
        target, was_clamped = self._bounded_voltage(requested, clamp=clamp)
        return self._result_from_voltage(
            command_kind="voltage",
            requested_value=requested,
            requested_unit="V",
            target_voltage_v=target,
            clamped=was_clamped,
        )

    def plan_bit(self, bit: int, *, clamp: bool = False) -> CommandResult:
        """Plan a direct integer-bit command while tracking PI memory.

        Args:
            bit: PWM command bit.
            clamp: Whether to clamp values to the supported range.
        """
        self._require_home()
        if isinstance(bit, bool) or not isinstance(bit, int):
            raise TypeError("bit must be an integer")
        requested = bit
        bounded = requested
        was_clamped = False
        if not self._minimum_bit <= bounded <= self._maximum_bit:
            if not clamp:
                raise ValueError(
                    f"bit {bounded} outside device range "
                    f"{self._minimum_bit}..{self._maximum_bit}"
                )
            bounded = min(max(bounded, self._minimum_bit), self._maximum_bit)
            was_clamped = True
        target_voltage_v = self.voltage_curve.bit_to_voltage(bounded, clamp=False)
        result = self._result_from_voltage(
            command_kind="bit",
            requested_value=float(requested),
            requested_unit="bit",
            target_voltage_v=target_voltage_v,
            clamped=was_clamped,
        )
        # Voltage inversion normally returns the same bit. Preserve the exact
        # requested integer in the saturated high-end region as well.
        if result.bit != bounded:
            applied_voltage_v = self.voltage_curve.bit_to_voltage(bounded, clamp=False)
            effective_voltage_v = min(
                max(applied_voltage_v - self._origin_v, self.model.input_min_v),
                self.model.input_max_v,
            )
            result = CommandResult(
                command_kind=result.command_kind,
                requested_value=result.requested_value,
                requested_unit=result.requested_unit,
                bit=bounded,
                target_voltage_v=target_voltage_v,
                applied_voltage_v=applied_voltage_v,
                effective_voltage_v=effective_voltage_v,
                predicted_displacement_nm=self.model.predict(effective_voltage_v),
                target_displacement_nm=result.target_displacement_nm,
                clamped=result.clamped,
            )
        return result

    def commit(self, command: CommandResult) -> float:
        """Advance model memory after a successful output write."""
        if command.command_kind == "home":
            predicted = self.model.reset(0.0)
            self.homed = True
        else:
            self._require_home()
            predicted = self.model.commit(command.effective_voltage_v)
        self.current_bit = command.bit
        self.current_voltage_v = command.applied_voltage_v
        self.current_displacement_nm = predicted
        return predicted

    def status(self) -> dict[str, object]:
        return {
            "device_id": self.profile.device_id,
            "profile_status": self.profile.status,
            "homed": self.homed,
            "current_bit": self.current_bit,
            "current_voltage_v": self.current_voltage_v,
            "predicted_displacement_nm": self.current_displacement_nm,
            "linearized_command": self._linearized_command,
            "model": self.model.state(),
        }
