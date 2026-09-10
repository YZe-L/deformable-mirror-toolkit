# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 2.1, 2026-07-01

"""Offline MP4 response analysis + live step-watch / setup-time workers."""

import threading
import time
from pathlib import Path

import numpy as np
import cv2
from PyQt5 import QtCore

from . import fringes
from .fringes import downsample
from .tracking import FringeTracker, SurfaceCentreTracker
from .response import StepWatchLogic, analyze_response_trace, \
    monotonic_unwrap, robust_sigma

VIDEO_MAX_SIZE = 640  # Analysis side for video frames.
VIDEO_FPS_FALLBACK = 20.0


def list_videos(folder):
    """MP4/AVI/MOV files sorted numerically by stem (stem = bit value)."""
    vids = [p for p in Path(folder).iterdir()
            if p.suffix.lower() in (".mp4", ".avi", ".mov")]

    def key(p):
        try:
            return (0, float(p.stem))
        except ValueError:
            return (1, p.stem)
    return sorted(vids, key=key)


class VideoResponseWorker(QtCore.QThread):
    """Analyse every video in a folder: displacement trace + response time."""

    progress = QtCore.pyqtSignal(int, int)  # Frames done, total.
    video_done = QtCore.pyqtSignal(dict)
    done = QtCore.pyqtSignal(list)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, folder, wavelength_nm, monotonic=True, parent=None):
        super().__init__(parent)
        self.folder = str(folder)
        self.wavelength_nm = float(wavelength_nm)
        self.monotonic = bool(monotonic)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            vids = list_videos(self.folder)
            if not vids:
                raise RuntimeError("no video files (*.mp4/avi/mov) in folder")
            total = 0
            for p in vids:
                cap = cv2.VideoCapture(str(p))
                total += max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
                cap.release()
            total = max(total, 1)

            results, frames_done = [], 0
            for p in vids:
                res = self._analyse_one(p, total, frames_done)
                if res is None:  # Stopped
                    return
                frames_done += res.pop("_frames")
                results.append(res)
                self.video_done.emit(res)
            self.done.emit(results)
        except Exception as e:
            self.failed.emit(str(e))

    def _analyse_one(self, path, total, frames_done):
        """Analyze one recorded response video.

        Args:
            path: Filesystem path used by the operation.
            total: Total item or frame count used for progress reporting.
            frames_done: Number of frames processed before this video.
        """
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return dict(file=path.name, bit=path.stem, status="open failed",
                        fps=np.nan, n_frames=0, response_s=np.nan,
                        t_start=np.nan, t_settle=np.nan, amplitude_nm=np.nan,
                        t=np.array([]), d=np.array([]), _frames=0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not np.isfinite(fps) or fps <= 1:
            fps = VIDEO_FPS_FALLBACK
        # Absolute phase vs the first frame, not cumulative integration, so a
        # per-frame error stays local. Temporal np.unwrap stitches the trace.
        k = self.wavelength_nm / (4 * np.pi)
        dm = refm = prevm = None
        ph_abs, ph_cum, n = [], [], 0
        status = None
        while True:
            if self._stop:
                cap.release()
                return None
            ok, frame = cap.read()
            if not ok:
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                img = downsample(np.asarray(frame, float), VIDEO_MAX_SIZE)
                if dm is None:
                    dm = fringes.build_demod(img)
                    refm = prevm = fringes.demod(img, dm)[dm["mask"]]
                fm = fringes.demod(img, dm)[dm["mask"]]
                ph_abs.append(float(np.angle(np.mean(fm * np.conj(refm)))))
                ph_cum.append(float(np.angle(np.mean(fm * np.conj(prevm)))))
                prevm = fm
            except Exception as e:
                status = f"demod failed: {e}"
                break
            n += 1
            if n % 20 == 0:
                self.progress.emit(frames_done + n, total)
        cap.release()

        ph_abs = np.asarray(ph_abs)
        flag = None
        if not len(ph_abs):
            d = np.array([])
        elif self.monotonic:
            # One-way step: force a monotonic unwrap (recovers up to lambda/2
            # per frame by adding the overflow in the known direction)
            ph_uw, back = monotonic_unwrap(ph_abs)
            d = ph_uw * k
            if back > 0.08:  # A frame still moved > lambda/2.
                flag = "fast: a fringe was missed - use higher fps"
        else:
            d = np.unwrap(ph_abs) * k
            # Reliability cross-check (general case): absolute vs cumulative
            # plateaus diverge only under genuine aliasing -- threshold-free,
            # does not cry wolf on a clean multi-fringe move.
            d_cum = np.cumsum(ph_cum) * k
            bn = max(3, len(d) // 10)
            if abs(np.median(d[-bn:]) - np.median(d_cum[-bn:])) \
                    > self.wavelength_nm / 8.0:
                flag = "ALIASED - use higher fps"
        if status is None:
            ana = analyze_response_trace(d, fps,
                                         wavelength_nm=self.wavelength_nm,
                                         aliased=np.array([], int))
            if flag and ana["status"] not in ("no motion", "too short"):
                ana["status"] = flag
        else:
            ana = dict(status=status, response_s=np.nan, t_start=np.nan,
                       t_settle=np.nan, rise_s=np.nan, amplitude_nm=np.nan,
                       sigma_step=np.nan, aliased=np.array([], int))
        ana.pop("aliased")  # Not a table/CSV column.
        return dict(file=path.name, bit=path.stem, fps=fps, n_frames=n,
                    t=np.arange(len(d)) / fps, d=d, _frames=n, **ana)


class WatchWorker(QtCore.QThread):
    """Represent a watch worker.

    Live step watch: reference at arm time, displacement at full camera
    rate, StepWatchLogic on top, CSV streamed per sample.
    """

    sample = QtCore.pyqtSignal(dict)  # t, d
    event = QtCore.pyqtSignal(dict)  # Calibrated / motion / step.
    failed = QtCore.pyqtSignal(str)
    stopped = QtCore.pyqtSignal(str)  # Csv path

    def __init__(self, acq_worker, csv_path, wavelength_nm, invert,
                 parent=None):
        """Initialize the WatchWorker.

        Args:
            acq_worker: Camera acquisition worker.
            csv_path: Filesystem path for the csv data.
            wavelength_nm: Wavelength, in nanometres.
            invert: Whether to invert the measured sign or image convention.
            parent: Parent Qt object.
        """
        super().__init__(parent)
        self.acq = acq_worker
        self.csv_path = str(csv_path)
        # absolute = compare every frame to the arm-time reference; simple =
        # plain wrapped phase, no spatial unwrap on the live feed.
        self.tracker = FringeTracker(wavelength_nm=wavelength_nm,
                                     invert=invert, absolute=True, simple=True)
        self.logic = StepWatchLogic()
        self._stop = False
        self._seq = 0
        self._last_alias_warn = -10.0

    def stop(self):
        self._stop = True

    def run(self):
        try:
            fh = open(self.csv_path, "w", encoding="utf-8")
            fh.write("frame,elapsed_s,displacement_nm,state\n")
        except OSError as e:
            self.failed.emit(f"cannot open CSV: {e}")
            return
        t0 = time.perf_counter()
        n = 0
        try:
            while not self._stop:
                seq, frame = self.acq.get_latest(self._seq)
                if frame is None:
                    time.sleep(0.005)
                    continue
                self._seq = seq
                t = time.perf_counter() - t0
                try:
                    m = self.tracker.update(frame)
                except Exception as e:
                    self.failed.emit(f"watch analysis failed: {e}")
                    break
                d = m["displacement_nm"]
                # Motion faster than lambda/4 per frame aliases: surface it.
                if m["near_limit"] and t - self._last_alias_warn > 2.0:
                    self._last_alias_warn = t
                    self.event.emit(dict(type="aliasing", t=t))
                for ev in self.logic.feed(t, d):
                    self.event.emit(ev)
                fh.write(f"{n},{t:.6f},{d:.6f},{self.logic.state}\n")
                fh.flush()
                n += 1
                self.sample.emit(dict(t=t, d=d, state=self.logic.state))
        finally:
            fh.close()
        self.stopped.emit(self.csv_path)


class NoiseWorker(QtCore.QThread):
    """Represent a noise worker.

    Record displacement at full camera rate for a fixed duration, then ship
    the trace for noise-floor analysis (RMS + spectrum). Uses the same tracker
    as the live recording -- straight (FringeTracker) or bent (Surface), so the
    noise method is identical, only the underlying demod differs. Keep the DM
    static for the whole window.
    """

    progress = QtCore.pyqtSignal(float)  # elapsed_s
    done = QtCore.pyqtSignal(dict)  # t (list), d (list)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, acq_worker, wavelength_nm, invert, duration_s,
                 surface=False, shrink=0.8, parent=None):
        """Initialize the NoiseWorker.

        Args:
            acq_worker: Camera acquisition worker.
            wavelength_nm: Wavelength, in nanometres.
            invert: Whether to invert the measured sign or image convention.
            duration_s: Duration, in seconds.
            surface: Use the carrier-free surface tracker.
            shrink: Image downsampling factor.
            parent: Parent Qt object.
        """
        super().__init__(parent)
        self.acq = acq_worker
        self.duration_s = float(duration_s)
        if surface:
            self.tracker = SurfaceCentreTracker(
                wavelength_nm=wavelength_nm, invert=invert, shrink=shrink)
        else:
            self.tracker = FringeTracker(wavelength_nm=wavelength_nm,
                                         invert=invert, absolute=True,
                                         simple=True)
        self._stop = False
        self._seq = 0

    def stop(self):
        self._stop = True

    def run(self):
        t0 = time.perf_counter()
        ts, ds = [], []
        while not self._stop and (time.perf_counter() - t0) < self.duration_s:
            seq, frame = self.acq.get_latest(self._seq)
            if frame is None:
                time.sleep(0.005)
                continue
            self._seq = seq
            t = time.perf_counter() - t0
            try:
                d = self.tracker.update(frame)["displacement_nm"]
            except Exception as e:
                self.failed.emit(f"noise analysis failed: {e}")
                return
            ts.append(t)
            ds.append(d)
            self.progress.emit(t)
        self.done.emit(dict(t=ts, d=ds))


class SetupTimeWorker(QtCore.QThread):
    """Live DM setup-time test.

    Records displacement at full camera rate; on an external trigger (Pi 'Enter'
    -> applies the step) it times the gap to the first rising knee of the
    response, then to the settled plateau. One CSV holds the whole trace (flat
    -> rise); each trigger appends a summary row. The tracker is shared with
    Live recording: straight-fringe Takeda by default, or carrier-free surface
    reconstruction for bent/closed fringes.
    """

    sample = QtCore.pyqtSignal(dict)  # t, d
    armed = QtCore.pyqtSignal(dict)  # Trial, t0  (trigger accepted)
    result = QtCore.pyqtSignal(dict)  # Trial, setup_s, amp_nm, t0, t_knee.
    failed = QtCore.pyqtSignal(str)
    stopped = QtCore.pyqtSignal(str)  # Csv path

    def __init__(self, acq_worker, csv_path, wavelength_nm, invert,
                 surface=False, shrink=0.8,
                 knee_k=6.0, base_n=20, settle_quiet=8, max_settle_s=2.0,
                 parent=None):
        """Initialize the SetupTimeWorker.

        Args:
            acq_worker: Camera acquisition worker.
            csv_path: Filesystem path for the csv data.
            wavelength_nm: Wavelength, in nanometres.
            invert: Whether to invert the measured sign or image convention.
            surface: Use the carrier-free surface tracker.
            shrink: Aperture shrink factor applied before analysis.
            knee_k: Detection threshold in baseline-noise standard deviations.
            base_n: Baseline samples used for the noise estimate.
            settle_quiet: Number of quiet samples required for settling.
            max_settle_s: Maximum settle, in seconds.
            parent: Parent Qt object.
        """
        super().__init__(parent)
        self.acq = acq_worker
        self.csv_path = str(csv_path)
        self.surface = bool(surface)
        if self.surface:
            self.tracker = SurfaceCentreTracker(
                wavelength_nm=wavelength_nm, invert=invert, shrink=shrink)
        else:
            self.tracker = FringeTracker(
                wavelength_nm=wavelength_nm, invert=invert,
                absolute=True, simple=True)
        self.knee_k = float(knee_k)  # knee = > knee_k * baseline noise
        self.base_n = int(base_n)  # Pre-trigger samples for baseline.
        self.settle_quiet = int(settle_quiet)
        self.max_settle_s = float(max_settle_s)
        self._stop = False
        self._seq = 0
        self._lock = threading.Lock()
        self._pending = None  # (trial, bit) from trigger()
        self._buf_d = []  # Recent displacements (baseline)
        self._w = None  # Active watch state, or None.

    def stop(self):
        self._stop = True

    def trigger(self, trial, bit):
        """Called from the GUI when the Pi fires the step.

        Args:
            trial: Trial identifier or trial record.
            bit: PWM command bit.
        """
        with self._lock:
            self._pending = (int(trial), int(bit))

    def run(self):
        try:
            fh = open(self.csv_path, "w", encoding="utf-8")
            fh.write("frame,elapsed_s,displacement_nm,phase,trial\n")
        except OSError as e:
            self.failed.emit(f"cannot open CSV: {e}")
            return
        t0 = time.perf_counter()
        n = 0
        try:
            while not self._stop:
                seq, frame = self.acq.get_latest(self._seq)
                if frame is None:
                    time.sleep(0.005)
                    continue
                self._seq = seq
                t = time.perf_counter() - t0
                try:
                    d = self.tracker.update(frame)["displacement_nm"]
                except Exception as e:
                    self.failed.emit(f"setup-time analysis failed: {e}")
                    break
                self._buf_d.append(d)
                if len(self._buf_d) > 400:
                    self._buf_d.pop(0)

                phase = self._step(t, d)
                fh.write("%d,%.6f,%.6f,%s,%s\n"
                         % (n, t, d, phase, self._w["trial"] if self._w else ""))
                fh.flush()
                n += 1
                self.sample.emit(dict(t=t, d=d))
        finally:
            fh.close()
        self.stopped.emit(self.csv_path)

    def _step(self, t, d):
        """Per-sample state machine; returns the CSV phase tag.

        Args:
            t: Time samples or scalar time.
            d: Displacement or distance samples.
        """
        with self._lock:
            pend, self._pending = self._pending, None
        if pend is not None:  # A new trigger -> arm the watch.
            trial, bit = pend
            base = np.asarray(self._buf_d[-self.base_n:] or [d], float)
            baseline = float(np.median(base))
            sigma = robust_sigma(base - baseline)
            self._w = dict(trial=trial, bit=bit, t0=t, baseline=baseline,
                           thr=max(self.knee_k * sigma, 0.3), knee_t=None,
                           plateau=baseline, quiet=0, prev=d)
            self.armed.emit(dict(trial=trial, t0=t))
            return "trigger"

        w = self._w
        if w is None:
            return "pre"
        if w["knee_t"] is None:  # Still looking for the rising knee.
            if abs(d - w["baseline"]) > w["thr"]:
                w["knee_t"] = t
                return "knee"
            return "watching"
        # Past the knee: track the plateau, finalise on settle or a max window.
        w["plateau"] = d
        if abs(d - w["prev"]) < max(0.5 * w["thr"], 0.15):
            w["quiet"] += 1
        else:
            w["quiet"] = 0
        w["prev"] = d
        if w["quiet"] >= self.settle_quiet or (t - w["knee_t"]) > self.max_settle_s:
            self.result.emit(dict(trial=w["trial"], setup_s=w["knee_t"] - w["t0"],
                                  amp_nm=w["plateau"] - w["baseline"],
                                  t0=w["t0"], t_knee=w["knee_t"]))
            self._w = None
            return "post"
        return "rising"
