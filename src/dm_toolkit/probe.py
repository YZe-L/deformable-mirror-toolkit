# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-08-13

"""The one point on the mirror whose height the live readers sample.

Kept in normalised aperture coordinates, u, v in [-1, 1] from the aperture
centre in units of its radius, so it survives downsampling, aperture
re-detection and drift of the mirror in the frame. (0, 0) is the centre.
Only readers created with ``follow_probe=True`` use it.
"""

from PyQt5 import QtCore

CENTRE = (0.0, 0.0)


class ProbePoint(QtCore.QObject):
    """Where to read the height, shared between the pages that opt in.

    The value is one immutable tuple, replaced whole. Worker threads read it
    without a lock: swapping a tuple reference is atomic, so a reader gets
    either the old point or the new one, never half of each.

    Attributes:
        changed (pyqtSignal): (u, v) whenever the point moves, for the views
            that draw a marker on it.
    """

    changed = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._uv = CENTRE

    @property
    def uv(self):
        """tuple[float, float]: The point, in normalised aperture coords."""
        return self._uv

    def set(self, u, v):
        """Move the probe, clamped to the aperture.

        Args:
            u (float): Across, in units of the aperture radius.
            v (float): Down, in units of the aperture radius.

        Returns:
            tuple[float, float]: The point actually adopted.
        """
        u = min(max(float(u), -1.0), 1.0)
        v = min(max(float(v), -1.0), 1.0)
        # A pick just outside the rim is a near miss, not a request to read
        # nothing: pull it back onto the edge instead of dropping the click.
        radius = (u * u + v * v) ** 0.5
        if radius > 1.0:
            u, v = u / radius, v / radius
        if (u, v) != self._uv:
            self._uv = (u, v)
            self.changed.emit(u, v)
        return self._uv

    def reset(self):
        """Put the probe back at the aperture centre."""
        return self.set(*CENTRE)

    @property
    def at_centre(self):
        """bool: Whether the probe is still where it starts."""
        return self._uv == CENTRE

    def label(self):
        """str: The point as the pages show it in a status line."""
        if self.at_centre:
            return "centre"
        return "u %+.2f, v %+.2f" % self._uv


# One shared point for the whole application.
PROBE = ProbePoint()


def to_pixels(uv, circ):
    """Turn a normalised probe point into pixels of one particular image.

    Args:
        uv (tuple): (u, v) in normalised aperture coordinates.
        circ (tuple): That image's aperture, (cx, cy, r).

    Returns:
        tuple[float, float]: (x, y) in pixels.
    """
    u, v = uv
    cx, cy, r = circ
    return cx + float(u) * r, cy + float(v) * r


def from_pixels(x, y, circ):
    """Turn a click on one particular image into a normalised probe point.

    Args:
        x (float): Pixel across.
        y (float): Pixel down.
        circ (tuple): That image's aperture, (cx, cy, r).

    Returns:
        tuple[float, float]: (u, v) in normalised aperture coordinates.
    """
    cx, cy, r = circ
    r = float(r) if r else 1.0
    return (float(x) - cx) / r, (float(y) - cy) / r
