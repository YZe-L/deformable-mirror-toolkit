# SPDX-License-Identifier: GPL-3.0-or-later

"""Validated adaptive-settling descriptors and the sensorless runtime rule.

The runtime deliberately uses only quantities available in a wavefront-sensor-
free loop: the nominal displacement predicted by each channel's hysteresis
profile, whether the planned command clamped, and a calibration residual gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


ADAPTIVE_SCHEMA_VERSION = 1
REQUIRED_TIERS_MS = {"small": 290, "medium": 380, "full": 580}


@dataclass(frozen=True)
class AdaptiveDescriptor:
    """One channel's immutable runtime calibration."""

    path: Path
    channel: int
    mirror_actuators: int
    egate_nm: float
    small_multiplier: float
    medium_multiplier: float
    tiers_ms: dict[str, int]
    calibration_set_id: str
    source_profile_sha256: str
    sha256: str
    data: dict

    @property
    def small_limit_nm(self) -> float:
        return self.small_multiplier * self.egate_nm

    @property
    def medium_limit_nm(self) -> float:
        return self.medium_multiplier * self.egate_nm


@dataclass(frozen=True)
class AdaptiveDecision:
    """Common wait selected for a simultaneous multi-channel command."""

    settle_ms: int
    tier: str
    per_channel_tier: dict[int, str]
    predicted_delta_nm: dict[int, float]
    limiting_channels: tuple[int, ...]
    clamped_channels: tuple[int, ...]
    fallback_reason: str = ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a number") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and greater than zero")
    return number


def validate_descriptor(data: Mapping, path: Path | str = "<memory>") -> None:
    """Raise ``ValueError`` when a descriptor cannot drive the runtime rule."""

    label = str(path)
    if int(data.get("schema_version", -1)) != ADAPTIVE_SCHEMA_VERSION:
        raise ValueError(
            f"{label}: unsupported schema_version {data.get('schema_version')!r}")
    for key in ("channel", "mirror_actuators", "egate_nm", "tiers_ms"):
        if key not in data:
            raise ValueError(f"{label}: missing {key}")
    channel = int(data["channel"])
    mirror = int(data["mirror_actuators"])
    if channel < 1:
        raise ValueError(f"{label}: channel must be positive")
    if mirror not in (5, 9):
        raise ValueError(f"{label}: mirror_actuators must be 5 or 9")
    _positive(data["egate_nm"], f"{label}: egate_nm")
    multipliers = data.get("multipliers") or {}
    small = _positive(multipliers.get("small", 5.0),
                      f"{label}: multipliers.small")
    medium = _positive(multipliers.get("medium", 10.0),
                       f"{label}: multipliers.medium")
    if small >= medium:
        raise ValueError(f"{label}: small multiplier must be below medium")
    try:
        tiers = {str(k): int(v) for k, v in data["tiers_ms"].items()}
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{label}: tiers_ms must map tier names to integers") from error
    if tiers != REQUIRED_TIERS_MS:
        raise ValueError(
            f"{label}: tiers_ms must be exactly {REQUIRED_TIERS_MS}, got {tiers}")


def load_descriptor(path: str | Path) -> AdaptiveDescriptor:
    """Load one channel descriptor and preserve its hash for run provenance."""

    source = Path(path).expanduser().resolve()
    with source.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{source}: descriptor root must be an object")
    validate_descriptor(data, source)
    multipliers = data.get("multipliers") or {}
    return AdaptiveDescriptor(
        path=source,
        channel=int(data["channel"]),
        mirror_actuators=int(data["mirror_actuators"]),
        egate_nm=float(data["egate_nm"]),
        small_multiplier=float(multipliers.get("small", 5.0)),
        medium_multiplier=float(multipliers.get("medium", 10.0)),
        tiers_ms={str(k): int(v) for k, v in data["tiers_ms"].items()},
        calibration_set_id=str(data.get("calibration_set_id") or ""),
        source_profile_sha256=str(data.get("source_profile_sha256") or ""),
        sha256=_sha256(source),
        data=data,
    )


def empirical_quantile_higher(values: Iterable[float], q: float) -> float:
    """Empirical quantile using the conservative ``higher`` convention."""

    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        raise ValueError("at least one finite residual is required")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    index = max(0, min(len(clean) - 1, math.ceil(q * len(clean)) - 1))
    return clean[index]


def quantile_lower_confidence_bound(
    values: Iterable[float], q: float = 0.95, confidence: float = 0.95
) -> tuple[float, int]:
    """Distribution-free one-sided lower confidence bound for a quantile.

    Returns the highest order statistic whose coverage probability is at least
    ``confidence``. A lower bound is intentional here: it avoids inflating the
    runtime gate from a small calibration sample, while the empirical P95 is
    also reported separately for engineering review.
    """

    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        raise ValueError("at least one finite residual is required")
    if not 0.0 < q < 1.0 or not 0.0 < confidence < 1.0:
        raise ValueError("q and confidence must lie strictly in (0, 1)")
    n = len(clean)
    chosen = 1
    for k in range(1, n + 1):
        coverage = sum(
            math.comb(n, j) * q**j * (1.0 - q) ** (n - j)
            for j in range(k, n + 1)
        )
        if coverage + 1e-15 >= confidence:
            chosen = k
        else:
            break
    return clean[chosen - 1], chosen


def quantile_confidence_interval(
    values: Iterable[float], q: float = 0.95, confidence: float = 0.95
) -> tuple[float, float, int, int]:
    """Exact distribution-free two-sided interval for a population quantile."""
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        raise ValueError("at least one finite residual is required")
    n = len(clean)
    tail = (1.0 - confidence) / 2.0
    lower_rank = 1
    for k in range(1, n + 1):
        probability = sum(
            math.comb(n, j) * q**j * (1.0 - q) ** (n - j)
            for j in range(k, n + 1))
        if probability >= 1.0 - tail:
            lower_rank = k
        else:
            break
    upper_rank = n
    for k in range(1, n + 1):
        probability = sum(
            math.comb(n, j) * q**j * (1.0 - q) ** (n - j)
            for j in range(0, k))
        if probability >= 1.0 - tail:
            upper_rank = k
            break
    return (clean[lower_rank - 1], clean[upper_rank - 1],
            lower_rank, upper_rank)


def choose_wait(
    descriptors: Mapping[int, AdaptiveDescriptor],
    previous_nm: Mapping[int, float],
    requested_nm: Mapping[int, float],
    clamped: Iterable[int] = (),
) -> AdaptiveDecision:
    """Select one common wait using the most restrictive active channel."""

    channels = sorted(set(requested_nm))
    missing = [channel for channel in channels if channel not in descriptors]
    if missing:
        raise ValueError("missing adaptive descriptors for " +
                         ", ".join(f"ch{c}" for c in missing))
    absent_previous = [channel for channel in channels
                       if channel not in previous_nm]
    if absent_previous:
        raise ValueError("missing previous displacement for " +
                         ", ".join(f"ch{c}" for c in absent_previous))
    clamped_set = set(int(c) for c in clamped)
    per_channel: dict[int, str] = {}
    deltas: dict[int, float] = {}
    rank = {"small": 0, "medium": 1, "full": 2}
    for channel in channels:
        delta = abs(float(requested_nm[channel]) - float(previous_nm[channel]))
        if not math.isfinite(delta):
            raise ValueError(f"ch{channel} predicted displacement is non-finite")
        deltas[channel] = delta
        descriptor = descriptors[channel]
        if channel in clamped_set:
            tier = "full"
        elif delta <= descriptor.small_limit_nm:
            tier = "small"
        elif delta <= descriptor.medium_limit_nm:
            tier = "medium"
        else:
            tier = "full"
        per_channel[channel] = tier
    if not channels:
        return AdaptiveDecision(REQUIRED_TIERS_MS["full"], "full", {}, {},
                                (), (), "no active channels")
    tier = max(per_channel.values(), key=rank.__getitem__)
    limiting = tuple(c for c in channels if per_channel[c] == tier)
    settle_ms = REQUIRED_TIERS_MS[tier]
    return AdaptiveDecision(
        settle_ms=settle_ms,
        tier=tier,
        per_channel_tier=per_channel,
        predicted_delta_nm=deltas,
        limiting_channels=limiting,
        clamped_channels=tuple(sorted(clamped_set.intersection(channels))),
        fallback_reason="clamped command" if clamped_set.intersection(channels)
        else "",
    )


def descriptor_document(
    *, channel: int, mirror_actuators: int, egate_nm: float,
    calibration_set_id: str, source_profile: str = "",
    source_profile_sha256: str = "", statistics: Mapping | None = None,
) -> dict:
    """Build the stable JSON document written by calibration analysis."""

    return {
        "schema_version": ADAPTIVE_SCHEMA_VERSION,
        "kind": "fringe_studio_adaptive_settling_channel",
        "channel": int(channel),
        "mirror_actuators": int(mirror_actuators),
        "calibration_set_id": str(calibration_set_id),
        "egate_nm": float(egate_nm),
        "multipliers": {"small": 5.0, "medium": 10.0},
        "tiers_ms": dict(REQUIRED_TIERS_MS),
        "decision_rule": (
            "all channels <=5*Egate:{small}ms; all <=10*Egate:{medium}ms; "
            "otherwise:{full}ms; any clamp:{full}ms".format(**REQUIRED_TIERS_MS)
        ),
        "source_profile": str(source_profile),
        "source_profile_sha256": str(source_profile_sha256),
        "statistics": dict(statistics or {}),
    }
