# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-06-12

"""TENMA 72-7732A (UNI-T UT71D) protocol over the WCH CH9325 USB-HID cable."""

VID, PID = 0x1A86, 0xE008
BAUDRATE = 2400

# UT71x frame: DDDDD R F S1 S2 \r \n  (11 bytes)
# D: 5 digits, value = DDDDD * 10^exponent   R: range   F: function.
FUNC_DCV, FUNC_DCMV = 1, 3
EXPONENT = {FUNC_DCV: (0, -4, -3, -2, -1, 0, 0, 0),  # 4/40/400/1000 V
            FUNC_DCMV: (-5, 0, 0, 0, 0, 0, 0, 0)}  # 400 mV
FUNC_NAMES = {0: "AC mV", 1: "DC V", 2: "AC V", 3: "DC mV"}


def baud_feature_report():
    """One-time HID feature report configuring the cable's UART."""
    return bytes([0x00, BAUDRATE & 0xFF, (BAUDRATE >> 8) & 0xFF,
                  0x00, 0x00, 0x03])


def parse_dcv(frame):
    """9-byte frame (CRLF stripped) -> dict(volts, overload, func, name).

    volts is None when overloaded or the dial is not on DC voltage.
    """
    func = frame[6] - 0x30
    name = FUNC_NAMES.get(func, f"function {func}")
    if func not in EXPONENT:
        return dict(volts=None, overload=False, func=func, name=name)
    digits = frame[0:5]
    if not digits.isdigit():  # ':'/'<' patterns = OL/UL
        return dict(volts=None, overload=True, func=func, name=name)
    rng = frame[5] - 0x30
    if not 0 <= rng <= 7:
        return dict(volts=None, overload=True, func=func, name=name)
    volts = int(digits) * 10.0 ** EXPONENT[func][rng]
    if frame[8] & 0b100:  # Sign flag
        volts = -volts
    return dict(volts=volts, overload=False, func=func, name=name)
