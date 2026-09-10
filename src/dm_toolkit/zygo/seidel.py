# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-05

"""Named aberration magnitudes/angles from the ZFR Fringe coefficients."""

import numpy as np


def _mag_ang(a_cos, a_sin, ang_div=1.0):
    mag = float(np.hypot(a_cos, a_sin))
    ang = float(np.degrees(np.arctan2(a_sin, a_cos)) / ang_div)
    return mag, ang


def seidel(coeffs):
    """dict of named aberrations from Fringe coeffs (waves / degrees).

    Uses the first-order Fringe terms: tilt Z1/Z2, focus Z3, astig Z4/Z5,
    coma Z6/Z7, spherical Z8. Angles are the orientation of each aberration.
    """
    c = np.asarray(coeffs)
    g = lambda i: float(c[i]) if i < len(c) else 0.0
    tilt_mag, tilt_ang = _mag_ang(g(1), g(2))
    ast_mag, ast_ang = _mag_ang(g(4), g(5), ang_div=2.0)  # 2-theta -> /2
    coma_mag, coma_ang = _mag_ang(g(6), g(7))
    return {
        "TiltMag": tilt_mag, "TiltAng": tilt_ang,
        "FocMag": g(3),
        "AstMag": ast_mag, "AstAng": ast_ang,
        "ComMag": coma_mag, "ComAng": coma_ang,
        "SA": g(8),
    }
