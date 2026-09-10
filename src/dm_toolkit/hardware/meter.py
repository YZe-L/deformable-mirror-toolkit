# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-06-12

"""TENMA 72-7732A multimeter worker (DC voltage, passive listener)."""

import time
from datetime import datetime

from PyQt5 import QtCore

from .tenma7732a import VID, PID, EXPONENT, baud_feature_report, parse_dcv


class MeterWorker(QtCore.QThread):
    """Opens the meter cable and streams parsed DC-voltage readings."""

    opened = QtCore.pyqtSignal(dict)
    reading = QtCore.pyqtSignal(dict)  # t, iso, volts, overload, dc, name.
    error = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            import hid
        except ImportError as e:
            self.error.emit(f"hidapi not installed: {e}")
            return
        if not hid.enumerate(VID, PID):
            self.error.emit("meter cable not found (USB-HID 1A86:E008) -- "
                            "is the USB cable plugged in?")
            return
        dev = hid.device()
        try:
            dev.open(VID, PID)
            dev.send_feature_report(baud_feature_report())
        except OSError as e:
            self.error.emit(f"cannot open meter: {e}")
            return
        self.opened.emit({})

        buf = bytearray()
        t0 = time.perf_counter()
        try:
            while not self._stop:
                chunk = dev.read(8, timeout_ms=500)
                if not chunk:
                    continue
                n = chunk[0] & 0x0F  # 0xFn: n payload bytes.
                if (chunk[0] & 0xF0) != 0xF0 or n == 0:
                    continue
                buf.extend(b & 0x7F for b in chunk[1:1 + n])  # Strip parity
                while True:
                    cut = buf.find(b"\r\n")
                    if cut < 0:
                        if len(buf) > 64:  # Garbage flood: resync.
                            del buf[:-11]
                        break
                    frame = bytes(buf[:cut])
                    del buf[:cut + 2]
                    if len(frame) != 9:  # Partial frame: drop.
                        continue
                    m = parse_dcv(frame)
                    self.reading.emit(dict(
                        t=time.perf_counter() - t0,
                        iso=datetime.now().isoformat(timespec="milliseconds"),
                        volts=m["volts"],
                        overload=m["overload"],
                        dc=m["func"] in EXPONENT,
                        name=m["name"]))
        finally:
            dev.close()
