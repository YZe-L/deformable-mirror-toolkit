# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.1, 2026-08-14

"""Per-iteration timing recorder for the closed loop.

One point is a relay across the GUI, the Pi link reader, the capture worker
and the scoring pool. `mark()` stores one `perf_counter()` value per
hand-off so `timing.csv` reconstructs the round as a timeline. Only the GUI
thread touches a Timeline; worker stamps travel back in their results.
"""

from __future__ import annotations

import csv
import math
import time

# Hand-offs of one round, in the order they happen. `send` opens the round and
# `ask` closes it: the next command goes out immediately after `ask`, so
# ask - send is the true loop period.
STAMPS = (
    "send",         # Round opens: the next setpoint is known (GUI).
    "wire",         # About to write it to the socket (GUI).
    "sent",         # Socket write returned (GUI).
    "ack_rx",       # T_SETTLED landed in the link reader thread.
    "ack_gui",      # _on_settled actually ran (GUI).
    "settle_end",   # Settle timer fired, capture requested (GUI).
    "cap_req",      # Capture worker accepted the request (worker).
    "frame_first",  # Camera published the first frame of this point.
    "frame_last",   # Camera published the last frame of this point.
    "cap_done",     # Capture worker had all frames (worker).
    "cap_gui",      # _on_capture_done ran (GUI).
    "work",         # Scoring job started on the pool.
    "avg",          # Frame average done (pool).
    "score",        # All measure() calls done (pool).
    "score_gui",    # _finalize_point_done ran (GUI).
    "tell",         # Decision job started on the pool: opt.observe() next.
    "obs_end",      # opt.observe() returned; opt.tell() next (pool).
    "tell_end",     # opt.tell() returned (pool).
    "tell_gui",     # _commit_done ran (GUI).
    "ask",          # opt.ask() returned: next command decided (GUI).
)

# (column, from, to). Every duration is a difference of two stamps above, so a
# reader can always check a number against the timeline it came from.
_SPANS = (
    ("d_cmd_build", "send", "wire"),        # Compensate + 3-D redraw.
    ("d_write", "wire", "sent"),            # The socket write itself.
    ("d_rtt", "wire", "ack_rx"),            # PC -> Pi -> PC, before the GUI hop.
    ("d_ack_hop", "ack_rx", "ack_gui"),     # Qt queue + GUI-thread lag.
    ("d_settle", "ack_gui", "settle_end"),  # The wait actually taken.
    ("d_cap_wait", "cap_req", "frame_first"),   # Waiting for the first frame.
    ("d_cap_frames", "frame_first", "frame_last"),  # n-1 frame periods.
    ("d_capture", "cap_req", "cap_done"),   # Whole capture.
    ("d_cap_hop", "cap_done", "cap_gui"),
    ("d_pool_hop", "cap_gui", "work"),      # Queued behind the thread pool.
    ("d_avg", "work", "avg"),               # Frame averaging.
    ("d_score", "avg", "score"),            # measure(): the scoring maths.
    ("d_score_hop", "score", "score_gui"),
    ("d_decide_hop", "score_gui", "tell"),
    # The three decision steps kept apart: observe() takes the raw metric,
    # tell() updates the learner, ask() chooses the next command.
    ("d_observe", "tell", "obs_end"),
    ("d_tell", "obs_end", "tell_end"),
    ("d_decide", "tell", "tell_end"),       # observe + tell together.
    ("d_tell_hop", "tell_end", "tell_gui"),
    ("d_ask", "tell_gui", "ask"),           # Next command chosen.
    ("d_round", "send", "ask"),             # The loop period.
)

# Which algorithm paid the decision cost and what it scales with (n_told,
# stage / generation). `tier` names the measurement budget, e.g. "570ms/8f".
_CONTEXT = ("algo", "stage", "gen", "converged", "n_iter", "tier")

_HEAD = (["sample", "seq", "iso", "event"]
         + list(_CONTEXT)
         + [f"t_{n}" for n in STAMPS]
         + [c for c, _, _ in _SPANS]
         # d_bg is the background part of d_score; d_bg_all sums it over every
         # measure() call. `d_settle_set` is the wait this point was commanded.
         + ["d_pi_set", "d_link", "d_settle_set", "n_frames", "cam_fps",
            "d_start_to_send", "d_bg", "d_bg_all",
            "adaptive_on", "adaptive_tier"])

_FRAME_HEAD = ["sample", "frame_idx", "t_pub", "t_pick", "d_stale",
               "d_since_prev", "d_cam_read"]

# Long format on purpose: which sub-phases exist depends on the algorithm, and
# a staged run changes algorithm mid-file. A wide table would need columns for
# every algorithm at once and rewrite its header at handover.
_PHASE_HEAD = ["sample", "algo", "stage", "call", "phase", "ms"]


def _ms(t, t0):
    """Milliseconds from the run start, or NaN when the stamp is missing."""
    return float("nan") if t is None or t0 is None else (t - t0) * 1e3


def _fmt(x):
    return "" if x is None or not math.isfinite(x) else f"{x:.3f}"


class Timeline:
    """Stamps for one run, written out one row per measured point.

    Attributes:
        t0: `perf_counter` of Start; every `t_*` column is relative to it.
    """

    def __init__(self):
        self.t0 = None
        self._t = {}
        self._v = {}
        self._frames = []
        self._phases = []
        self._file = self._writer = None
        self._frame_file = self._frame_writer = None
        self._phase_file = self._phase_writer = None

    # Run lifecycle
    def start_run(self, t0=None):
        """Anchor the run clock; call where the Start handler stamps its own."""
        self.t0 = time.perf_counter() if t0 is None else float(t0)
        self.reset()

    def open(self, directory):
        """Open timing.csv and frame_timing.csv in the run's log folder.

        Args:
            directory: Run log folder, already created.

        Returns:
            True when both files opened; False leaves recording off.
        """
        self.close()  # A restarted run must not leak the previous handles.
        try:
            self._file = open(directory / "timing.csv", "w", newline="",
                              encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(_HEAD)
            self._frame_file = open(directory / "frame_timing.csv", "w",
                                    newline="", encoding="utf-8")
            self._frame_writer = csv.writer(self._frame_file)
            self._frame_writer.writerow(_FRAME_HEAD)
            self._phase_file = open(directory / "decision_timing.csv", "w",
                                    newline="", encoding="utf-8")
            self._phase_writer = csv.writer(self._phase_file)
            self._phase_writer.writerow(_PHASE_HEAD)
        except OSError:
            self.close()
            return False
        return True

    def close(self):
        for handle in (self._file, self._frame_file, self._phase_file):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
        self._file = self._writer = None
        self._frame_file = self._frame_writer = None
        self._phase_file = self._phase_writer = None

    @property
    def active(self) -> bool:
        return self._writer is not None

    # Recording
    def reset(self):
        """Drop the current round's stamps, ready for the next `send`."""
        self._t.clear()
        self._v.clear()
        self._frames = []
        self._phases = []

    def mark(self, name, t=None):
        """Stamp one hand-off.

        Args:
            name: One of `STAMPS`.
            t: A `perf_counter` value taken elsewhere (a worker thread), or
                None to stamp now.
        """
        self._t[name] = time.perf_counter() if t is None else float(t)

    def mark_at(self, name, t):
        """Stamp with a `perf_counter` value taken on another thread.

        A None is ignored, so a worker that could not stamp leaves a gap in the
        timeline rather than a stamp of when the GUI got round to reading it.
        """
        if t is not None:
            self._t[name] = float(t)

    def value(self, name, v):
        """Record a non-timestamp scalar (the Pi's set_ms, the settle setting)."""
        self._v[name] = v

    def phases(self, call, marks):
        """Take an optimiser's own breakdown of one decision call.

        Args:
            call: Which call produced them -- "observe", "tell" or "ask".
            marks: {phase name -> milliseconds}, or None when the algorithm
                reports nothing. Recorded verbatim; the algorithm owns the
                names, so a new one needs no change here.
        """
        for name, ms in (marks or {}).items():
            self._phases.append((str(call), str(name), ms))

    def context(self, status):
        """Record which algorithm decided this point, and its own progress.

        Args:
            status: The optimiser's `status()` dict. A staged run reports the
                stage that is actually running, which is the point: the two
                halves of such a run have completely different decision costs.
        """
        status = status or {}
        self._v["algo"] = status.get("algorithm", "")
        self._v["stage"] = status.get("stage", "")
        gen = status.get("gen")
        self._v["gen"] = "" if gen is None else gen
        self._v["converged"] = int(bool(status.get("converged")))
        # What a Bayesian GP refit and a CMA update actually scale with.
        self._v["n_iter"] = status.get("iter", "")

    def frames(self, stamps):
        """Take the capture worker's per-frame stamps.

        Args:
            stamps: List of (t_publish, t_pickup, cam_read_ms) per frame, in
                capture order. Missing camera stamps come through as None.
        """
        self._frames = list(stamps or [])
        pub = [s[0] for s in self._frames if s and s[0] is not None]
        if pub:
            self.mark("frame_first", min(pub))
            self.mark("frame_last", max(pub))

    # Output
    def write(self, sample, iso, event="", seq=""):
        """Append this round's row, then reset for the next one.

        Always resets, even with recording off, so a disabled log cannot leak
        one round's stamps into the next.

        Args:
            sample: Point number this row belongs to.
            iso: Wall-clock stamp of the point, matching loop_log.csv.
            event: Event label of the point, matching loop_log.csv.
            seq: Setpoint number of the round, matching sends.csv. `sample`
                counts SCORED points only, so a round that was driven and then
                skipped shares its neighbour's; seq identifies it uniquely.
        """
        if self._writer is None:
            self.reset()
            return
        t0 = self.t0
        row = [sample, seq, iso, event]
        row += [self._v.get(c, "") for c in _CONTEXT]
        row += [_fmt(_ms(self._t.get(n), t0)) for n in STAMPS]
        for _, a, b in _SPANS:
            ta, tb = self._t.get(a), self._t.get(b)
            row.append(_fmt((tb - ta) * 1e3
                            if ta is not None and tb is not None
                            else float("nan")))
        # The Pi reports how long its own set_pwm calls took, so the rest of the
        # round trip is link + both socket stacks -- the number that moves when
        # the network hiccups rather than the mirror.
        pi_set = float(self._v.get("pi_set_ms", float("nan")))
        ts, ta = self._t.get("wire"), self._t.get("ack_rx")
        rtt = (ta - ts) * 1e3 if ts is not None and ta is not None else float("nan")
        n = len([s for s in self._frames if s])
        span = self._t.get("frame_last"), self._t.get("frame_first")
        fps = float("nan")
        if n > 1 and None not in span and span[0] > span[1]:
            fps = (n - 1) / (span[0] - span[1])
        row += [_fmt(pi_set), _fmt(rtt - pi_set),
                _fmt(self._v.get("settle_set_ms", float("nan"))), n, _fmt(fps),
                _fmt(_ms(self._t.get("sent"), t0)),
                _fmt(float(self._v.get("bg_ms", float("nan")))),
                _fmt(float(self._v.get("bg_ms_all", float("nan")))),
                self._v.get("adaptive_on", ""),
                self._v.get("adaptive_tier", "")]
        self._writer.writerow(row)
        self._file.flush()
        self._write_frames(sample)
        self._write_phases(sample)
        self.reset()

    def _write_phases(self, sample):
        """One row per algorithm sub-phase of this point's decision."""
        if self._phase_writer is None or not self._phases:
            return
        algo = self._v.get("algo", "")
        stage = self._v.get("stage", "")
        for call, name, ms in self._phases:
            self._phase_writer.writerow(
                [sample, algo, stage, call, name,
                 _fmt(float(ms)) if ms is not None else ""])
        self._phase_file.flush()

    def _write_frames(self, sample):
        """One row per captured frame: when it appeared and how stale it was."""
        if self._frame_writer is None or not self._frames:
            return
        t0, prev = self.t0, None
        for i, s in enumerate(self._frames):
            if not s:
                continue
            pub, pick, read_ms = s
            since = ((pub - prev) * 1e3
                     if pub is not None and prev is not None else float("nan"))
            stale = ((pick - pub) * 1e3
                     if pub is not None and pick is not None else float("nan"))
            self._frame_writer.writerow(
                [sample, i, _fmt(_ms(pub, t0)), _fmt(_ms(pick, t0)),
                 _fmt(stale), _fmt(since),
                 _fmt(float("nan") if read_ms is None else float(read_ms))])
            if pub is not None:
                prev = pub
        self._frame_file.flush()
