# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.1, 2026-07-23

"""Load and validate device-specific hysteresis calibration profiles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


PROFILE_DIRECTORY = Path(__file__).with_name("devices")


@dataclass(frozen=True)
class DeviceProfile:
    """Validated wrapper around a JSON device description."""

    path: Path
    data: dict[str, Any]

    @property
    def device_id(self) -> str:
        return str(self.data["device_id"])

    @property
    def display_name(self) -> str:
        return str(self.data.get("display_name", self.device_id))

    @property
    def status(self) -> str:
        return str(self.data["status"])

    @property
    def hardware(self) -> dict[str, Any]:
        return dict(self.data["hardware"])

    @property
    def model(self) -> dict[str, Any]:
        return dict(self.data["model"])

    @property
    def limits(self) -> dict[str, Any]:
        return dict(self.data["limits"])

    @property
    def calibration(self) -> dict[str, Any]:
        return dict(self.data["calibration"])

    @property
    def linearized_command(self) -> dict[str, Any] | None:
        value = self.data.get("linearized_command")
        return dict(value) if value is not None else None


def _profile_path(name_or_path: str | Path) -> Path:
    candidate = Path(name_or_path)
    if candidate.suffix.lower() == ".json" or candidate.parent != Path("."):
        return candidate.expanduser().resolve()
    return (PROFILE_DIRECTORY / f"{candidate.name}.json").resolve()


def _validate(data: dict[str, Any], path: Path) -> None:
    """Validate a device-profile mapping.

    Args:
        data: Input data used by the operation.
        path: Filesystem path used by the operation.
    """
    required = ("schema_version", "device_id", "status", "hardware", "model", "limits", "calibration")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path}: missing profile keys: {', '.join(missing)}")
    if data["schema_version"] != 1:
        raise ValueError(f"{path}: unsupported schema_version {data['schema_version']!r}")
    if data["model"].get("type") != "modified_prandtl_ishlinskii":
        raise ValueError(f"{path}: unsupported model type {data['model'].get('type')!r}")
    operators = data["model"].get("play_operators", [])
    if not operators:
        raise ValueError(f"{path}: at least one play operator is required")
    thresholds = [float(item["threshold_v"]) for item in operators]
    if thresholds != sorted(thresholds) or thresholds[0] <= 0:
        raise ValueError(f"{path}: play thresholds must be positive and increasing")
    if any(float(item["weight_nm_per_v"]) < 0 for item in operators):
        raise ValueError(f"{path}: provisional inverse requires non-negative play weights")
    voltage_model = data.get("driver_voltage_model")
    if voltage_model is not None:
        required_voltage = (
            "type",
            "minimum_bit",
            "maximum_bit",
            "minimum_voltage_v",
            "maximum_voltage_v",
            "chebyshev_coefficients",
            "raw_zero_v",
            "output_scale",
        )
        missing_voltage = [key for key in required_voltage if key not in voltage_model]
        if missing_voltage:
            raise ValueError(
                f"{path}: missing driver_voltage_model keys: {', '.join(missing_voltage)}"
            )
        if voltage_model["type"] != "chebyshev":
            raise ValueError(f"{path}: unsupported driver voltage model type")

    linearized = data.get("linearized_command")
    if linearized is not None:
        required_linearized = (
            "type",
            "minimum_nominal_bit",
            "maximum_nominal_bit",
            "minimum_target_displacement_nm",
            "maximum_target_displacement_nm",
        )
        missing_linearized = [
            key for key in required_linearized if key not in linearized
        ]
        if missing_linearized:
            raise ValueError(
                f"{path}: missing linearized_command keys: "
                f"{', '.join(missing_linearized)}"
            )
        if linearized["type"] != "linear_nominal_bit_to_displacement":
            raise ValueError(
                f"{path}: unsupported linearized_command type "
                f"{linearized['type']!r}"
            )
        minimum_nominal = int(linearized["minimum_nominal_bit"])
        maximum_nominal = int(linearized["maximum_nominal_bit"])
        hardware_minimum = int(data["hardware"]["minimum_bit"])
        hardware_maximum = int(data["hardware"]["maximum_bit"])
        if minimum_nominal >= maximum_nominal:
            raise ValueError(
                f"{path}: linearized nominal-bit range must be increasing"
            )
        if (
            minimum_nominal < hardware_minimum
            or maximum_nominal > hardware_maximum
        ):
            raise ValueError(
                f"{path}: linearized nominal-bit range must stay inside "
                "the hardware bit range"
            )
        minimum_target = float(linearized["minimum_target_displacement_nm"])
        maximum_target = float(linearized["maximum_target_displacement_nm"])
        if minimum_target >= maximum_target:
            raise ValueError(
                f"{path}: linearized displacement range must be increasing"
            )
        limits = data["limits"]
        if (
            minimum_target < float(limits["minimum_relative_displacement_nm"])
            or maximum_target
            > float(limits["maximum_calibrated_displacement_nm"])
        ):
            raise ValueError(
                f"{path}: linearized displacement range must stay inside "
                "the calibrated displacement limits"
            )


def load_device_profile(name_or_path: str | Path) -> DeviceProfile:
    """Load ``devices/<name>.json`` or an explicit JSON path."""
    path = _profile_path(name_or_path)
    if not path.is_file():
        available = ", ".join(sorted(item.stem for item in PROFILE_DIRECTORY.glob("*.json")))
        raise FileNotFoundError(f"device profile not found: {path}; available: {available or 'none'}")
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: profile root must be a JSON object")
    _validate(data, path)
    return DeviceProfile(path=path, data=data)
