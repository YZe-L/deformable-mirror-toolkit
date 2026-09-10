# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.1, 2026-07-12

"""CameraManager -- one shared TLCameraSDK, ref-counted acquisition workers."""

from . import dll  # noqa: F401  -- configures the Thorlabs DLL search path
from .camera import ThorlabsAcqWorker


class CameraManager:
    def __init__(self):
        self._sdk = None
        self._serials = None  # Cached first-discovery list.
        self._workers = {}  # Serial -> ThorlabsAcqWorker
        self._refcount = {}  # Serial -> int

    def _ensure_sdk(self):
        if self._sdk is None:
            from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
            self._sdk = TLCameraSDK()
        return self._sdk

    def list_cameras(self, refresh=False):
        """Serial numbers of connected cameras.

        Cached from the first discovery: an opened camera may drop out of
        discover_available_cameras, so we only re-discover on explicit refresh
        while nothing is open.
        """
        if self._serials is None or (refresh and not self._workers):
            self._serials = list(self._ensure_sdk().discover_available_cameras())
        return list(self._serials)

    def acquire(self, serial, exposure_ms=20.0, on_opened=None, on_error=None):
        """Return a started worker for `serial`, ref-counted.

        Callbacks are connected before the worker starts (fresh) or receive a
        replayed opened(info) (shared reuse), so late consumers still get the
        info.

        Args:
            serial: Camera serial number.
            exposure_ms: Exposure, in milliseconds.
            on_opened: Callback invoked after the resource opens.
            on_error: Callback invoked when the operation fails.
        """
        serial = str(serial)
        worker = self._workers.get(serial)
        if worker is None:
            worker = ThorlabsAcqWorker(self._ensure_sdk(), serial,
                                       exposure_ms=exposure_ms)
            self._workers[serial] = worker
            self._refcount[serial] = 1
            if on_opened is not None:
                worker.opened.connect(on_opened)
            if on_error is not None:
                worker.error.connect(on_error)
            worker.start()
        else:
            self._refcount[serial] += 1
            if on_opened is not None:
                worker.opened.connect(on_opened)
            if on_error is not None:
                worker.error.connect(on_error)
            # Replay the cached info to the new consumer only; re-emitting the
            # opened signal would re-notify every earlier consumer.
            worker.replay_opened(on_opened)
        return worker

    def release(self, serial, on_opened=None, on_error=None):
        """Release one camera-worker reference.

        Drop one reference; disconnect this consumer's callbacks, and stop +
        drop the worker when the last reference goes.

        Args:
            serial: Camera serial number.
            on_opened: Callback invoked after the resource opens.
            on_error: Callback invoked when the operation fails.
        """
        serial = str(serial)
        if serial not in self._refcount:
            return
        worker = self._workers.get(serial)
        if worker is not None:
            for sig, cb in ((worker.opened, on_opened),
                            (worker.error, on_error)):
                if cb is not None:
                    try:
                        sig.disconnect(cb)
                    except (TypeError, RuntimeError):
                        pass
        self._refcount[serial] -= 1
        if self._refcount[serial] <= 0:
            self._workers.pop(serial, None)
            self._refcount.pop(serial, None)
            if worker is not None:
                worker.stop()
                worker.wait(3000)

    def shutdown(self):
        """Stop every worker and dispose the SDK. Call once on app close."""
        for worker in list(self._workers.values()):
            worker.stop()
            worker.wait(3000)
        self._workers.clear()
        self._refcount.clear()
        if self._sdk is not None:
            try:
                self._sdk.dispose()
            finally:
                self._sdk = None
