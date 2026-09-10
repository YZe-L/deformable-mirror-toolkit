# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-08-15

"""Plan for the measurement-parameter sweep (settle time, frames averaged).

Both are swept on one step transient per round, so every point of a round
shares the same mirror motion and camera noise. No optimiser is involved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from ..correction import settings as S
from ..config import OUTPUT_DIR

# Operator input is remembered by the application's UI state; only a sweep
# saved before that mechanism existed is read here, once.
LEGACY_SETTINGS_PATH = OUTPUT_DIR / "dm_sweep" / "dm_sweep.json"

# The nine-element mirror's working shape, used as the opening shape. The
# actuator table decides which channels exist.
DEFAULT_SHAPE = {6: 1743, 7: 2164, 8: 2029, 9: 1802, 10: 1739,
                 11: 1888, 12: 2720, 13: 1884, 14: 1769}

# The three scoring functions, all read off one measure() call.
METRICS = tuple(S.METRICS_OFFERED)

# Both averaging modes from one recording: avg_frame scores the averaged
# frame; avg_score averages the per-frame scores.
MODES = (S.SCORE_AVG_FRAME, S.SCORE_AVG_SCORE)


@dataclass
class SweepPlan:
    """Everything one sweep session needs; serialised to sweep_plan.json."""

    # The shape under test: channel -> bit. Channels come from the loop's
    # actuator table; these bits are only what an unset row opens with.
    shape: dict = field(default_factory=lambda: dict(DEFAULT_SHAPE))
    active_mirror: int = S.DEFAULT_LAYOUT
    inactive_mirror: int = S.DM9
    hold_inactive_shape: bool = False
    fixed_channels: dict = field(default_factory=lambda: {
        channel: 0 for channel in S.LAYOUTS[S.DM9]
    })
    park_bit: int = 2000  # Every channel is driven here between rounds.

    # The two sweeps are independent switches and come out of one recording.
    do_settle: bool = True
    do_frames: bool = True

    # Settle sweep: these delays, each scored over `settle_frames` frames.
    delays_ms: list = field(default_factory=lambda: [55, 112, 169, 258, 350,
                                                     516, 670, 773, 990])
    settle_frames: int = 4

    # Frames sweep: this fixed delay, then cumulative windows frames[i0:i0+N],
    # which is exactly how the loop pays for N frames.
    frames_settle_ms: int = 200
    frames_list: list = field(default_factory=lambda: [1, 2, 4, 8, 12, 16,
                                                       20, 24, 28, 32])

    rounds: int = 10  # Scatter points per sweep point.
    park_s: float = 3.0  # Hold at park_bit before the step.
    rest_s: float = 3.0  # Hold at park_bit after the round's post-processing.
    ref_frames: int = 8  # Frames averaged into each round's park reading.
    window_margin: float = 1.2  # Recording headroom over the computed need.

    # Score noise, recorded ONCE at the end of the session so it can never
    # interleave with -- and disturb -- a settle measurement.
    noise_s: float = 30.0
    noise_chunk_s: float = 1.0  # Capture granularity, bounds peak memory.

    # Full LoopSettings snapshot at sweep start, so the sweep scores exactly
    # like the loop does.
    base: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["shape"] = {str(c): int(b) for c, b in self.shape.items()}
        d["fixed_channels"] = {
            str(channel): int(bit)
            for channel, bit in self.fixed_channels.items()
        }
        return d

    @property
    def channels(self):
        return sorted(int(c) for c in self.shape)

    def park_bits(self):
        """Every channel of the shape driven to the park bit."""
        return {int(c): int(self.park_bit) for c in self.shape}

    def step_bits(self):
        """The shape under test, as channel -> bit."""
        return {int(c): int(b) for c, b in self.shape.items()}

    def window_s(self, frame_s):
        """Recording length one round needs, in seconds.

        The window has to cover whichever ENABLED sweep reaches further into
        the transient: the last delay plus its averaging window, or the frames
        sweep's fixed delay plus its longest window.

        Args:
            frame_s: Measured camera frame period, in seconds.
        """
        frame_s = max(float(frame_s), 1e-4)
        needs = []
        if self.do_settle:
            needs.append(max(self.delays_ms or [0]) / 1e3
                         + self.settle_frames * frame_s)
        if self.do_frames:
            needs.append(self.frames_settle_ms / 1e3
                         + max(self.frames_list or [1]) * frame_s)
        return max(needs or [frame_s]) * float(self.window_margin)

    def resolvable(self, frame_s):
        """Delays that land closer together than two frame periods.

        A delay list finer than the camera can resolve does not fail -- it
        silently returns the same frames for two neighbouring points, which is
        why the achieved delay is recorded beside the nominal one.

        Args:
            frame_s: Measured camera frame period, in seconds.

        Returns:
            list: (previous, current) delay pairs that are too close together.
        """
        gap_ms = 2000.0 * max(float(frame_s), 1e-4)
        bad, ds = [], sorted(self.delays_ms)
        for a, b in zip(ds, ds[1:]):
            if b - a < gap_ms:
                bad.append((a, b))
        return bad


def read_legacy():
    """The pre-ui_state settings file, or {} when there is nothing usable.

    Read once when the UI state holds no sweep yet. A missing or broken
    file simply means the defaults.
    """
    try:
        data = json.loads(LEGACY_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
