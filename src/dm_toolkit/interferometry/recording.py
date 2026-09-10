# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.4, 2026-08-13

"""Live interval-recording analysis worker."""

import queue

from PyQt5 import QtCore

from .. import probe
from .tracking import FringeTracker, SurfaceCentreTracker

CSV_HEADER = ("frame,timestamp_iso,elapsed_s,displacement_nm,step_nm,"
              "phase_step_rad,quality\n")


class AnalysisWorker(QtCore.QThread):
    """Represent a queued analysis worker.

    Consumes (index, iso, elapsed, frame) tuples from a queue, computes
    displacement and streams rows to CSV (flushed per row -> crash-safe).
    """

    measured = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(int, str)

    def __init__(self, in_queue, csv_path, wavelength_nm, invert,
                 absolute=False, surface=False, shrink=0.8, parent=None,
                 follow_probe=False):
        """Initialize the AnalysisWorker.

        Args:
            in_queue: Queue that supplies captured frames.
            csv_path: Filesystem path for the csv data.
            wavelength_nm: Wavelength, in nanometres.
            invert: Whether to invert the measured sign or image convention.
            absolute: Report displacement against the first frame.
            surface: Use the carrier-free surface tracker.
            shrink: Image downsampling factor.
            parent: Parent Qt object.
        """
        super().__init__(parent)
        self.q = in_queue
        self.csv_path = str(csv_path)
        # surface=True: carrier-free vortex centre displacement. Otherwise
        # plain wrapped phase: absolute vs the first frame, or a running sum.
        self._surface = bool(surface)
        if surface:
            self.tracker = SurfaceCentreTracker(
                wavelength_nm=wavelength_nm, invert=invert,
                absolute=absolute, shrink=shrink,
                follow_probe=follow_probe)
        else:
            self.tracker = FringeTracker(wavelength_nm=wavelength_nm,
                                         invert=invert, absolute=absolute,
                                         simple=True)
        self._reset_ref = False
        self._stop_req = False
        self.n_done = 0

    def request_reset_reference(self):
        self._reset_ref = True

    def request_stop(self):
        """Request an immediate worker stop.

        Abort: discard the queue backlog (exact frame-limit truncation /
        app close); a normal Stop drains the backlog instead.
        """
        self._stop_req = True

    def _drain_to_latest(self, item):
        """Drain to latest.

        Surface mode: reconstruction is far slower than capture, so process
        only the NEWEST queued frame and drop the stale ones. Keeps the worker
        real-time and the queue near-empty, so Stop is immediate (no giant
        backlog to grind through). Returns (newest_item, saw_sentinel).
        """
        saw = False
        while True:
            try:
                nxt = self.q.get_nowait()
            except queue.Empty:
                break
            if nxt is None:  # Capture stopped
                saw = True
                break
            item = nxt
        return item, saw

    def run(self):
        try:
            fh = open(self.csv_path, "w", encoding="utf-8")
            # Which point on the mirror these heights came from. The plot and
            # this file are fed by ONE tracker, so they cannot disagree -- but
            # a file read months later has to say where it was measured.
            if self._surface:
                fh.write("# probe: %s (normalised aperture coords)\n"
                         % probe.PROBE.label())
            fh.write(CSV_HEADER)
            fh.flush()
        except OSError as e:
            self.failed.emit(f"cannot open CSV: {e}")
            return
        try:
            while True:
                item = self.q.get()
                if item is None or self._stop_req:  # Sentinel / user stop.
                    break
                ended = False
                if self._surface:  # Keep only the newest frame.
                    item, ended = self._drain_to_latest(item)
                idx, iso, elapsed, frame = item
                if self._reset_ref:
                    self.tracker.reset()
                    self._reset_ref = False
                try:
                    m = self.tracker.update(frame)
                except Exception as e:
                    self.failed.emit(f"analysis failed: {e}")
                    break
                m.update(frame_index=idx, timestamp_iso=iso,
                         elapsed_s=elapsed, backlog=self.q.qsize())
                fh.write(f"{idx},{iso},{elapsed:.6f},"
                         f"{m['displacement_nm']:.6f},{m['step_nm']:.6f},"
                         f"{m['phase_step']:.9f},{m['quality']:.6f}\n")
                fh.flush()
                self.n_done += 1
                self.measured.emit(m)
                if ended:  # Newest frame done, capture over.
                    break
        finally:
            fh.close()
        self.finished_ok.emit(self.n_done, self.csv_path)
