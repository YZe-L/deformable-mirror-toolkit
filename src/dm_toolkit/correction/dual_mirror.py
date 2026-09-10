# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.5, 2026-08-27

"""Compose mirror-specific optimizers behind the normal optimizer interface."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from . import budget as BUD
from . import settings as S


_PLAN_STAGES = {
    S.PLAN_DM5: (S.DM5,),
    S.PLAN_DM9: (S.DM9,),
    S.PLAN_DM5_DM9: (S.DM5, S.DM9),
    S.PLAN_DM9_DM5: (S.DM9, S.DM5),
    # The joint plan touches both mirrors; whether it drives them in one
    # stage is a separate question answered by `is_joint`.
    S.PLAN_JOINT: (S.DM5, S.DM9),
}


def expand_plan(plan: str) -> tuple[int, ...]:
    """Expand a persisted plan identifier into the mirrors it drives.

    For the sequential plans the order is the stage order; for the joint plan
    both mirrors are driven at once and the order is only the canonical one
    used for channel numbering.

    Args:
        plan: One of the ``PLAN_*`` identifiers in :mod:`settings`.

    Returns:
        Ordered mirror identifiers.

    Raises:
        ValueError: If the plan identifier is unknown.
    """
    try:
        return _PLAN_STAGES[str(plan)]
    except KeyError as exc:
        raise ValueError(f"unknown DM run plan: {plan!r}") from exc


def is_joint(plan: str) -> bool:
    """Whether the plan optimises both mirrors as one 14-axis search."""
    return str(plan) == S.PLAN_JOINT


@dataclass(frozen=True)
class MirrorStage:
    """One independently configured mirror stage.

    Attributes:
        name: Stable label written to status and run records.
        mirror: Mirror identifier, currently ``DM5`` or ``DM9``.
        channels: Channels owned by this stage.
        initial_seed: First-entry optimizer bits for these channels. This may
            differ from the raw command when compensation is active.
        optimizer_factory: Builder receiving this stage's current channel seed.
    """

    name: str
    mirror: int
    channels: tuple[int, ...]
    initial_seed: Mapping[int, int]
    optimizer_factory: Callable[[Mapping[int, int]], Any]


def merge_joint(stage_settings, joint_search=None):
    """Fuse both mirrors' settings into one 14-axis configuration.

    Every knob belongs to one group: ``PER_CHANNEL_KNOBS`` keep each
    mirror's value, ``JOINT_MEASUREMENT_KNOBS`` take the larger value, and
    ``JOINT_SEARCH_KNOBS`` come from `joint_search` only.

    Args:
        stage_settings: Ordered ``(mirror, LoopSettings)`` pairs, one per
            mirror, as `_stage_settings` builds them.
        joint_search: ``{knob: value}`` for the 14-axis search, typed on the
            Joint search panel. A knob left out keeps the base settings' value.

    Returns:
        ``(cfg, report)`` -- the merged settings and a record of where every
        value came from, for `run_config.json`.

    Raises:
        ValueError: If fewer than two mirrors are given, if they do not agree
            on one algorithm, or if that algorithm is a model-based solve.
    """
    stages = [(int(mirror), cfg) for mirror, cfg in stage_settings]
    if len(stages) < 2:
        raise ValueError("a joint run needs both mirrors")
    algorithms = {cfg.algorithm for _, cfg in stages}
    if len(algorithms) != 1:
        named = ", ".join(
            f"DM{mirror}={S.ALGO_LABELS.get(cfg.algorithm, cfg.algorithm)}"
            for mirror, cfg in stages)
        raise ValueError(
            "a joint run is ONE search over 14 axes, so both mirrors must be "
            f"set to the same algorithm ({named}). Two different searches "
            "moving at the same time cannot tell which of them a score change "
            "belongs to.")
    algorithm = stages[0][1].algorithm
    if algorithm in S.SOLVER_ALGOS:
        raise ValueError(
            f"{S.ALGO_LABELS.get(algorithm, algorithm)} drives the eigenmodes "
            "of ONE mirror's measured impact matrix, and no 14-channel matrix "
            "exists. Run it per mirror, or pick a search for the joint plan.")

    # Bit-scale knobs: each channel keeps its own mirror's value.
    channel_knobs = {}
    per_channel = {knob: {} for knob in S.PER_CHANNEL_KNOBS}
    for mirror, cfg in stages:
        for actuator in cfg.actuators:
            values = {knob: getattr(cfg, knob) for knob in S.PER_CHANNEL_KNOBS}
            channel_knobs[int(actuator.channel)] = values
            for knob, value in values.items():
                per_channel[knob][str(actuator.channel)] = value

    actuators = [actuator for _, cfg in stages for actuator in cfg.actuators]
    actuators.sort(key=lambda actuator: actuator.channel)

    replacements = {}
    measurement = {}
    for knob in S.JOINT_MEASUREMENT_KNOBS:
        offered = {mirror: getattr(cfg, knob) for mirror, cfg in stages}
        replacements[knob] = max(offered.values())
        measurement[knob] = {
            **{f"DM{mirror}": value for mirror, value in offered.items()},
            "used": replacements[knob],
            "rule": "the slower mirror decides -- one point moves both"}
    # The search's own size, typed for 14 axes. Each mirror's value is recorded
    # beside it as REFERENCE, so the record says what the joint number was
    # chosen against without implying it was derived from them.
    typed = dict(joint_search or {})
    search = {}
    for knob in S.JOINT_SEARCH_KNOBS:
        if knob not in _read_knobs(algorithm):
            continue
        used = typed.get(knob, getattr(stages[0][1], knob))
        replacements[knob] = used
        search[knob] = {
            "used": used,
            # `typed` carries only the knobs this algorithm reads, so this
            # test says whether the operator chose the value for 14 axes.
            "source": ("joint panel, for 14 axes" if knob in typed else
                       "NOT from the joint panel -- fell back to DM"
                       f"{stages[0][0]}'s own value"),
            "reference": {f"DM{mirror}": getattr(cfg, knob)
                          for mirror, cfg in stages}}
    # The scalar `min_step` is the fallback for a channel without an
    # override; the coarsest grid in the run is the safe choice.
    replacements["min_step"] = max(int(cfg.min_step) for _, cfg in stages)
    # The speed ladder is fitted per mirror, so a joint run pays the full
    # typed budget on every point.
    replacements["speed_mode"] = S.SPEED_FIXED
    replacements["speed_floor_ms"] = 0
    replacements["speed_floor_frames"] = 0

    cfg = dataclasses.replace(stages[0][1], actuators=actuators,
                              channel_knobs=channel_knobs, **replacements)
    report = dict(
        algorithm=algorithm,
        algorithm_label=S.ALGO_LABELS.get(algorithm, algorithm),
        channels={f"DM{mirror}": [a.channel for a in cfg_m.actuators]
                  for mirror, cfg_m in stages},
        per_channel={knob: table for knob, table in per_channel.items()
                     if knob in _read_knobs(algorithm)
                     and len(set(table.values())) > 1},
        measurement=measurement,
        search=search,
        min_step_fallback=replacements["min_step"],
        speed_mode_forced=("fixed -- the Auto ladder is fitted per mirror and "
                           "no joint sweep exists"),
    )
    return cfg, report


def _read_knobs(algorithm):
    """Knob names the given algorithm actually reads, plus the shared grid."""
    return set(S.ALGO_PARAMS.get(algorithm, ())) | {"min_step"}


class SequentialMirrors:
    """Run mirror stages in order while emitting one complete DM command.

    The class deliberately implements the existing optimizer protocol. The UI
    and camera state machine therefore see one optimizer even though each stage
    owns a different channel subset and may use a different algorithm. A stage
    is handed over only after its parked best command has been measured once.
    """

    def __init__(self, stages: Iterable[MirrorStage],
                 initial_command: Mapping[int, int]):
        self.stages = tuple(stages)
        if not self.stages:
            raise ValueError("a mirror sequence needs at least one stage")
        self._command = {int(channel): int(bit)
                         for channel, bit in initial_command.items()}
        if not self._command:
            raise ValueError("initial mirror command is empty")
        self._validate_stages()
        self._stage_index = -1
        self._optimizer = None
        self._verifying = False
        self._converged = False
        self._nominal_channels: set[int] = set()
        self._activated_mirrors: set[int] = set()
        self._completed: list[dict[str, Any]] = []
        # The optimizer behind each completed stage, kept because it alone
        # can still say how it sized its probes or why it stopped.
        self._completed_optimizers: list[Any] = []
        self._activate(0)

    def _validate_stages(self) -> None:
        owners: dict[int, int] = {}
        for stage in self.stages:
            channels = {int(channel) for channel in stage.channels}
            if not channels:
                raise ValueError(f"stage {stage.name!r} has no channels")
            missing = channels - self._command.keys()
            if missing:
                raise ValueError(
                    f"stage {stage.name!r} channels missing from initial shape: "
                    f"{sorted(missing)}")
            seed_channels = {int(channel) for channel in stage.initial_seed}
            if seed_channels != channels:
                raise ValueError(
                    f"stage {stage.name!r} seed channels do not match its "
                    f"owned channels")
            overlap = {channel for channel in channels
                       if channel in owners and owners[channel] != stage.mirror}
            if overlap:
                raise ValueError(
                    f"different mirrors overlap on channels {sorted(overlap)}")
            owners.update({channel: stage.mirror for channel in channels})

    @property
    def active_stage(self) -> MirrorStage:
        """Return the stage currently receiving measurements."""
        return self.stages[self._stage_index]

    @property
    def active_optimizer(self):
        """Return the optimizer currently owned by the camera loop."""
        return self._optimizer

    @property
    def completed_optimizers(self) -> tuple[Any, ...]:
        """Return the finished stages' optimizers, in `sequence_completed`
        order.

        Kept out of `status()` on purpose: these are live objects for the run
        record to interrogate, not JSON.
        """
        return tuple(self._completed_optimizers)

    @property
    def nominal_channels(self) -> set[int]:
        """Channels currently expressed on an optimizer's nominal bit axis."""
        return set(self._nominal_channels)

    @property
    def metric(self):
        """Return the active optimizer's metric identifier."""
        return self._optimizer.metric

    @property
    def modal_stage_active(self) -> bool:
        """Whether the active inner optimizer is still in a modal solve."""
        cfg = getattr(self._optimizer, "cfg", None)
        return bool(cfg is not None and cfg.algorithm in S.MODAL_ALGOS
                    and not getattr(self._optimizer, "handed_over", False))

    @property
    def best_score(self):
        """Return the best shared-metric score seen across completed stages."""
        scores = [float(stage["best_score"]) for stage in self._completed]
        scores.append(float(self._optimizer.best_score))
        return max(scores)

    @property
    def last_score(self):
        """Return the active stage's last score."""
        return self._optimizer.last_score

    def _activate(self, index: int) -> None:
        self._stage_index = index
        stage = self.stages[index]
        if stage.mirror in self._activated_mirrors:
            seed = {channel: self._command[channel]
                    for channel in stage.channels}
        else:
            seed = {int(channel): int(bit)
                    for channel, bit in stage.initial_seed.items()}
        self._optimizer = stage.optimizer_factory(seed)
        self._verifying = False
        self._nominal_channels.update(self.active_stage.channels)
        self._activated_mirrors.add(stage.mirror)

    def _merge(self, partial: Mapping[int, int]) -> dict[int, int]:
        allowed = set(self.active_stage.channels)
        unknown = {int(channel) for channel in partial} - allowed
        if unknown:
            raise ValueError(
                f"stage {self.active_stage.name!r} commanded channels it does "
                f"not own: {sorted(unknown)}")
        for channel, bit in partial.items():
            self._command[int(channel)] = int(bit)
        return dict(self._command)

    def ask(self) -> dict[int, int]:
        """Return the next full command, including every held mirror channel."""
        if self._converged:
            return dict(self._command)
        return self._merge(self._optimizer.ask())

    def best_command(self) -> dict[int, int]:
        """Return a full command with the active stage parked on its best."""
        if self._converged:
            return dict(self._command)
        command = dict(self._command)
        command.update({int(channel): int(bit) for channel, bit in
                        self._optimizer.best_command().items()})
        return command

    def observe(self, reading) -> None:
        """Forward a complete spot reading to the active optimizer."""
        self._optimizer.observe(reading)

    def set_noise(self, sigma) -> None:
        """Forward the latest score-noise estimate to the active optimizer."""
        self._optimizer.set_noise(sigma)

    def noise_gate(self) -> float:
        """Accept gate of the active stage, for the run record."""
        gate = getattr(self._optimizer, "noise_gate", None)
        return float(gate()) if gate is not None else float("nan")

    def precision(self) -> BUD.PrecisionRequest:
        """Measurement precision the active stage needs for the next point.

        A stage that has just been handed over is starting its own search, so
        the request is the new stage's, never the finished one's. While the
        plan is verifying a hand-over the point decides which shape the next
        stage inherits, so it is measured as well as the mirror can.
        """
        if self._converged or self._verifying:
            return BUD.PrecisionRequest(delta=0.0, critical=True)
        ask = getattr(self._optimizer, "precision", None)
        return ask() if ask is not None else BUD.PrecisionRequest(delta=0.0)

    def tell(self, score, valid=True) -> None:
        """Commit a score and perform a verified stage handover when ready."""
        if self._converged:
            self._optimizer.tell(score, valid=valid)
            return
        self._optimizer.tell(score, valid=valid)
        if not valid or not self._optimizer.status().get("converged", False):
            return
        if not self._verifying:
            self._verifying = True
            self._merge(self._optimizer.best_command())
            return

        self._merge(self._optimizer.best_command())
        record = {
            "name": self.active_stage.name,
            "mirror": self.active_stage.mirror,
            "best_score": float(self._optimizer.best_score),
            "best_bits": {str(channel): int(self._command[channel])
                          for channel in self.active_stage.channels},
        }
        # A stage appears once in a sequence; `rescan` re-opens the final
        # stage after a disturbance, so update in place rather than append.
        index = next((i for i, done in enumerate(self._completed)
                      if done["name"] == record["name"]), None)
        if index is None:
            self._completed.append(record)
            self._completed_optimizers.append(self._optimizer)
        else:
            self._completed[index] = record
            self._completed_optimizers[index] = self._optimizer
        next_index = self._stage_index + 1
        if next_index < len(self.stages):
            self._activate(next_index)
        else:
            self._converged = True

    def rescan(self) -> None:
        """Re-open the active final stage after a confirmed disturbance."""
        if hasattr(self._optimizer, "rescan"):
            self._optimizer.rescan()
        self._converged = False
        self._verifying = False

    def recover_roi_miss(self):
        """Forward fixed-ROI recovery to an active modal optimizer."""
        recover = getattr(self._optimizer, "recover_roi_miss", None)
        return recover() if recover is not None else None

    def take_phase_timing(self):
        """Return and clear timings owned by the active optimizer."""
        take = getattr(self._optimizer, "take_phase_timing", None)
        return take() if take is not None else {}

    def status(self) -> dict[str, Any]:
        """Return active-stage status plus sequence progress."""
        inner = dict(self._optimizer.status())
        inner_stage = str(inner.get("stage", "search"))
        inner.update(
            algorithm="mirror_sequence",
            inner_algorithm=inner.get("algorithm"),
            stage=f"{self.active_stage.name}: {inner_stage}",
            inner_stage=inner_stage,
            sequence_stage=self.active_stage.name,
            sequence_index=self._stage_index,
            sequence_count=len(self.stages),
            sequence_verifying=self._verifying,
            sequence_completed=list(self._completed),
            converged=self._converged,
            best_score=self.best_score,
            best_bits=self.best_command(),
        )
        return inner
