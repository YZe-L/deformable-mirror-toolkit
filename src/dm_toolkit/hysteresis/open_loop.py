# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.3, 2026-08-14

"""Open-loop nominal-bit compensation for one driven channel."""

from __future__ import annotations

from pathlib import Path

from .compensator import HysteresisCompensator
from .device_profile import load_device_profile


class OpenLoopChannel:
    """One channel's compensator: nominal bit in, hardware bit out.

    The model is path dependent, so a session runs home -> plan -> write ->
    commit, and :meth:`commit` is called only after the write succeeded.

    Attributes:
        path (Path): Device profile JSON backing this channel.
        compensator (HysteresisCompensator): The stateful planner.
        homed (bool): True once a home command has been committed.
    """

    def __init__(self, profile_path):
        """Load a profile for one channel.

        Args:
            profile_path (str | Path): Device profile JSON.

        Raises:
            ValueError: If the profile has no linearized nominal-bit command,
                which manual bit driving needs.
        """
        self.path = Path(profile_path)
        profile = load_device_profile(self.path)
        if profile.linearized_command is None:
            raise ValueError(
                f"{self.path.name} has no linearized_command section, so it "
                "cannot map a nominal bit onto a compensated bit")
        self.compensator = HysteresisCompensator(profile)
        self.homed = False

    @property
    def name(self):
        """str: Profile file stem, used as the label in the UI and log."""
        return self.path.stem

    def bit_voltage(self, bit):
        """Driver voltage of one hardware bit, from the profile's curve.

        The curve is a property of the driver, so this is meaningful whether
        or not the bit came out of a compensated plan.

        Args:
            bit (int): PWM command bit.

        Returns:
            float: Driver output in volts.
        """
        return float(self.compensator.voltage_curve.bit_to_voltage(int(bit),
                                                                   clamp=True))

    def plan_home(self):
        """Plan the bit-zero home command that starts a session.

        Returns:
            CommandResult: The home command; commit it after the write.
        """
        return self.compensator.plan_home()

    def plan(self, nominal_bit):
        """Plan the compensated command for a nominal bit.

        Args:
            nominal_bit (int): The bit the operator asked for.

        Returns:
            CommandResult: Carries the hardware ``bit`` to send and the
                predicted displacement.

        Raises:
            RuntimeError: If the channel has not been homed yet.
        """
        return self.compensator.plan_linearized_bit(int(nominal_bit),
                                                    clamp=True)

    def plan_raw(self, bit):
        """Plan an UNcompensated bit that still advances model memory.

        The bit goes to the driver exactly as given -- no nominal-to-hardware
        conversion -- while the PI model records the voltage the hardware
        really saw, so later compensated plans start from the true state.

        Args:
            bit (int): Hardware bit to send unchanged.

        Returns:
            CommandResult: ``bit`` equals the requested bit (clamped to the
                device range); commit it after the write.

        Raises:
            RuntimeError: If the channel has not been homed yet.
        """
        return self.compensator.plan_bit(int(bit), clamp=True)

    def equivalent_nominal_bit(self, raw_bit):
        """Nominal bit for the physical state of a proposed raw command.

        The conversion is evaluated against the channel's current hysteresis
        memory and does not advance it.  It intentionally refuses an
        out-of-range conversion instead of silently giving the optimiser a
        different physical seed.
        """
        return self.compensator.raw_bit_to_equivalent_nominal_bit(
            int(raw_bit), clamp=False)

    def commit(self, command):
        """Advance model memory after the hardware write succeeded.

        Args:
            command (CommandResult): The command that was written.

        Returns:
            float: Predicted displacement in nanometres.
        """
        predicted = self.compensator.commit(command)
        if command.command_kind == "home":
            self.homed = True
        return predicted
