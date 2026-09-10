# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.9, 2026-08-03

"""Continuous camera acquisition workers."""

import os
import time
from datetime import datetime

import numpy as np
import cv2
from PyQt5 import QtCore

from . import dll


def save_frame(path, frame, bit_depth=None):
    """Write a frame scaled to the file's full dynamic range.

    Write a frame scaled to the file's full dynamic range (the sensor
    delivers 10/12-bit values inside a uint16 buffer). Shared by snapshots
    and the dual-camera raw recorder.

    Args:
        path: Filesystem path used by the operation.
        frame: Captured image frame.
        bit_depth: Camera or image bit depth.
    """
    out = np.asarray(frame)
    if out.dtype == np.uint8:
        cv2.imwrite(path, out)
        return
    bits = bit_depth
    if not bits:
        mx = int(out.max())
        bits = max(8, int(np.ceil(np.log2(mx + 1)))) if mx else 8
    maxv = (1 << int(bits)) - 1
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg", ".bmp"):  # 8-bit-only formats
        out = np.clip(out.astype(np.float64) * (255.0 / maxv),
                      0, 255).astype(np.uint8)
    else:
        out = np.clip(out.astype(np.float64) * (65535.0 / maxv),
                      0, 65535).astype(np.uint16)
    cv2.imwrite(path, out)


class BaseAcqWorker(QtCore.QThread):
    """Frame acquisition loop with display hand-off and recording tap."""

    opened = QtCore.pyqtSignal(dict)  # Camera info
    error = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False
        self._lock = QtCore.QMutex()
        self._latest = None
        # When the newest frame was published and how long the SDK poll took;
        # consumers timing a measurement need both.
        self._latest_t = 0.0
        self._latest_read_ms = float("nan")
        self._bit_depth = None
        self._seq = 0
        self._snapshot_path = None
        self._rec_queue = None
        self._rec_interval = 1.0
        self._rec_t0 = 0.0
        self._rec_next = 0.0
        self._rec_index = 0
        # Cap the read/copy rate: no consumer is faster than 60 fps (display
        # 30, measure tick 40), but at floor exposures the sensor can push
        # hundreds -- copying each frame then just burns a core all scan long.
        self._pub_min_dt = 1.0 / 60.0
        self._pub_last = 0.0

    # GUI-side API
    def get_latest(self, last_seq, with_time=False):
        """Newest frame if fresher than last_seq, else (last_seq, None).

        Args:
            last_seq: Sequence number the caller already has.
            with_time: Also return (publish stamp, SDK read ms) of that frame,
                read under the same lock so they cannot describe another
                frame. Default off: every existing caller keeps the 2-tuple.
        """
        self._lock.lock()
        try:
            if self._seq == last_seq or self._latest is None:
                return (last_seq, None, None, None) if with_time \
                    else (last_seq, None)
            if with_time:
                return (self._seq, self._latest, self._latest_t,
                        self._latest_read_ms)
            return self._seq, self._latest
        finally:
            self._lock.unlock()

    def request_snapshot(self, path):
        self._lock.lock()
        self._snapshot_path = str(path)
        self._lock.unlock()

    def start_recording(self, out_queue, interval_s):
        """Start publishing camera frames to a recording queue.

        Args:
            out_queue: Queue that receives recorded frames.
            interval_s: Interval, in seconds.
        """
        self._lock.lock()
        self._rec_interval = max(0.001, float(interval_s))
        self._rec_t0 = time.perf_counter()
        self._rec_next = 0.0
        self._rec_index = 0
        self._rec_queue = out_queue
        self._lock.unlock()

    def is_recording(self):
        self._lock.lock()
        try:
            return self._rec_queue is not None
        finally:
            self._lock.unlock()

    def stop_recording(self):
        self._lock.lock()
        q = self._rec_queue
        self._rec_queue = None
        self._lock.unlock()
        if q is not None:
            q.put(None)  # Sentinel -> analysis thread finishes.

    def stop(self):
        self._stop = True

    # Subclass hooks
    def _open(self):
        raise NotImplementedError

    def _read(self):
        """Return a new 2-D numpy frame or None on poll timeout."""
        raise NotImplementedError

    def _close(self):
        pass

    def _apply_pending_settings(self):
        pass

    def _pace_publication(self, published_at, recording):
        """Cap non-recording publication/copy work at 60 fps.

        The clock is recorded after the wait, so the sleep never counts as
        frame spacing. Cameras delivering <=60 fps never wait.

        Args:
            published_at: Timestamp of the previous published frame.
            recording: Whether frame recording is active.
        """
        if recording:
            self._pub_last = published_at
            return
        wait = self._pub_min_dt - (published_at - self._pub_last)
        if wait > 0:
            time.sleep(wait)
            self._pub_last = time.perf_counter()
        else:
            self._pub_last = published_at

    # Main loop
    def run(self):
        try:
            info = self._open()
        except Exception as e:
            self.error.emit(str(e))
            return
        self.opened.emit(info)
        try:
            while not self._stop:
                self._apply_pending_settings()
                # Around _read() only: a successful poll blocks until the
                # sensor hands the frame over, so this is the camera->PC
                # transfer (plus the buffer copy) and nothing of ours.
                t_read = time.perf_counter()
                frame = self._read()
                if frame is None:
                    continue
                now = time.perf_counter()
                snap = None
                self._lock.lock()
                self._latest = frame
                self._latest_t = now
                self._latest_read_ms = (now - t_read) * 1e3
                self._seq += 1
                if self._snapshot_path:
                    snap, self._snapshot_path = self._snapshot_path, None
                q = self._rec_queue
                if q is not None and (now - self._rec_t0) >= self._rec_next:
                    elapsed = now - self._rec_t0
                    self._rec_next = elapsed + self._rec_interval
                    q.put((self._rec_index,
                           datetime.now().isoformat(timespec="milliseconds"),
                           elapsed, frame))
                    self._rec_index += 1
                self._lock.unlock()
                if snap:
                    self._save_snapshot(snap, frame)
                # Pace to ~60 fps; the SDK ring overwrites dropped frames, so
                # sleeping here cannot grow memory. Recording taps are exempt.
                self._pace_publication(now, recording=q is not None)
        finally:
            self.stop_recording()
            self._close()

    def _save_snapshot(self, path, frame):
        try:
            save_frame(path, frame, self._bit_depth)
        except Exception as e:
            self.error.emit(f"snapshot failed: {e}")


class ThorlabsAcqWorker(BaseAcqWorker):
    """Thorlabs scientific camera, continuous software-triggered mode.

    The SDK is owned externally (CameraManager) and shared across all
    cameras -- only one TLCameraSDK may exist per process. This worker
    just opens/arms one camera by serial on the shared sdk."""

    def __init__(self, sdk, serial, exposure_ms=20.0, parent=None):
        """Initialize the ThorlabsAcqWorker.

        Args:
            sdk: Camera SDK module.
            serial: Camera serial number.
            exposure_ms: Exposure, in milliseconds.
            parent: Parent Qt object.
        """
        super().__init__(parent)
        self._pending_exposure_ms = None
        self._exposure_ms = float(exposure_ms)
        self._sdk = sdk
        self._serial = str(serial)
        self._cam = None
        self._info = None  # Cached opened-info for replay.

    def set_exposure_ms(self, ms):
        self._lock.lock()
        self._pending_exposure_ms = float(ms)
        self._lock.unlock()

    def get_exposure_ms(self):
        """Return the active or queued exposure in milliseconds.

        The exposure the CAMERA is actually on, or the value queued for it.

        The single source of truth: ask here instead of caching, so no
        consumer scores against a stale exposure.
        """
        self._lock.lock()
        try:
            # The worker's own record, not an SDK query: this is read on every
            # display refresh and on every measured point, and it must never
            # cost a device round-trip. _apply_pending_settings keeps it true.
            pend = self._pending_exposure_ms
            return float(pend if pend is not None else self._exposure_ms)
        finally:
            self._lock.unlock()

    def replay_opened(self, callback=None):
        """Replay the cached opened.

        Replay the cached opened(info) to ONE late consumer that attached
        after this worker was already opened (shared via the manager). Only
        `callback` is invoked -- never the opened signal, which would re-notify
        every existing consumer. Deferred so the caller's wiring is live first.
        """
        if self._info is not None and callback is not None:
            info = self._info
            QtCore.QTimer.singleShot(0, lambda: callback(info))

    def _open(self):
        cam = self._sdk.open_camera(self._serial)
        cam.frames_per_trigger_zero_for_unlimited = 0  # Continuous
        cam.image_poll_timeout_ms = 200
        cam.exposure_time_us = int(self._exposure_ms * 1000)
        cam.arm(2)
        cam.issue_software_trigger()
        self._cam = cam
        self._bit_depth = cam.bit_depth
        rng = cam.exposure_time_range_us
        pixel_um = getattr(cam, "sensor_pixel_width_um", None)
        self._info = dict(source="thorlabs",
                          model=cam.model, serial=self._serial,
                          bit_depth=cam.bit_depth,
                          width=cam.image_width_pixels,
                          height=cam.image_height_pixels,
                          pixel_um=float(pixel_um) if pixel_um else None,
                          exposure_ms=cam.exposure_time_us / 1000.0,
                          exposure_range_ms=(rng.min / 1000.0, rng.max / 1000.0))
        return self._info

    def _apply_pending_settings(self):
        # Take the request under the lock, but never hold it across the SDK
        # write: get_exposure_ms() is called from the GUI thread every refresh.
        self._lock.lock()
        ms = self._pending_exposure_ms
        self._pending_exposure_ms = None
        self._lock.unlock()
        if ms is not None:
            try:
                self._cam.exposure_time_us = int(ms * 1000)
                self._exposure_ms = float(ms)  # Last known good.
            except Exception as e:
                self.error.emit(f"exposure change failed: {e}")

    def _read(self):
        frame = self._cam.get_pending_frame_or_null()
        if frame is None:
            return None
        return np.copy(frame.image_buffer)  # Buffer invalid after next poll.

    def _close(self):
        # Dispose only our camera; the shared SDK is owned by CameraManager.
        if self._cam is not None:
            self._cam.disarm()
            self._cam.dispose()
            self._cam = None
