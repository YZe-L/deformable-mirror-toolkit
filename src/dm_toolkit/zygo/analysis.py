# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-05

"""Full analysis of one imported Measurement: surface, Zernike and Seidel."""

from dataclasses import dataclass, field

import numpy as np

from . import zernike as zern
from . import surface as surf
from .seidel import seidel
from .geometry import aperture, unit_coords
from .io_datx import Measurement


@dataclass
class Analysis:
    meas: Measurement
    circ: tuple  # (cx, cy, r)
    order: int
    fit: dict  # zern.fit result at `order`
    seidels: dict
    residual_rms_by_order: dict  # {order: rms} for the RMS-vs-order plot
    _raw_map: np.ndarray  # Raw surface (waves), NaN outside.
    _raw_vals: np.ndarray  # 1-D raw values over masked pixels.
    _group_vals: dict  # Group name -> 1-D ZFR contribution.
    _ys: np.ndarray = field(default=None)
    _xs: np.ndarray = field(default=None)
    _rho: np.ndarray = field(default=None)
    _theta: np.ndarray = field(default=None)
    _res_buf: np.ndarray = field(default=None)  # Reused scatter buffer.

    @property
    def waves(self):
        return self.meas.waves

    @property
    def mask(self):
        return self.meas.mask

    # Page 1: live Zernike-group removal (1-D subtraction -> fast)
    def removed(self, groups):
        """(height_map, pv, rms, levels) with the selected groups subtracted.

        Works on the 1-D masked values (precomputed per group), scatters to a
        2-D map once, and derives display levels from the 1-D values -- so a
        toggle is milliseconds.
        """
        vals = self._raw_vals.copy()
        for g in groups:
            gv = self._group_vals.get(g)
            if gv is not None:
                vals -= gv
        vals -= vals.mean()
        # Scatter into the reused buffer (stays NaN outside the mask)
        self._res_buf[self._ys, self._xs] = vals
        lo, hi = float(vals.min()), float(vals.max())  # Full range, like Zygo.
        return (self._res_buf, hi - lo, float(vals.std()), (lo, hi))

    # Page 2: decomposition maps.
    def input_map_raw(self):
        """Raw surface (waves), NaN outside the aperture."""
        return self._raw_map

    def fit_map(self, coeffs=None):
        coeffs = self.fit["coeffs"] if coeffs is None else coeffs
        vals = zern.eval_terms(coeffs, range(len(coeffs)), self._rho, self._theta)
        return zern.map_from_vals(vals, self._ys, self._xs, self.waves.shape)

    def residual_map(self, coeffs=None):
        return self._raw_map - self.fit_map(coeffs)


def _residual_rms_by_order(rk, tk, wk):
    """Residual RMS after a subsampled fit at each order (for the plot).

    Args:
        rk: Radial polynomial samples.
        tk: Tangential polynomial samples.
        wk: Angular weighting samples.
    """
    out = {}
    for o in sorted(zern.ORDER_TERMS):
        B = zern.basis(rk, tk, zern.ORDER_TERMS[o])
        c, *_ = np.linalg.lstsq(B, wk, rcond=None)
        out[o] = float((wk - B @ c).std())
    return out


def analyze(meas: Measurement, order=6) -> Analysis:
    """Run the full analysis at the requested Zernike order.

    Coordinates are computed once and reused for every fit (subsampled) and for
    the group/raw maps -- no repeated arctan2 over ~800k points.

    Args:
        meas: Measurement result or measurement callable.
        order: Polynomial, Zernike, or symmetry order.
    """
    mask = meas.mask
    circ = aperture(mask)
    shape = meas.waves.shape
    ys, xs, rho, theta, keep = unit_coords(mask, circ)
    w_all = meas.waves[mask]

    rk, tk, wk = rho[keep], theta[keep], w_all[keep]
    step = max(1, len(wk) // 40000)
    rs, ts, ws = rk[::step], tk[::step], wk[::step]
    c6 = zern.fit_coords(rs, ts, ws, 6)["coeffs"]
    coeffs = c6 if order == 6 else zern.fit_coords(rs, ts, ws, order)["coeffs"]
    fit = {"coeffs": coeffs, "n_terms": zern.ORDER_TERMS[order],
           "order": order, "circ": circ}

    raw_map = zern.map_from_vals(w_all, ys, xs, shape)
    group_vals = {g: zern.eval_terms(c6, idxs, rho, theta)
                  for g, idxs in zern.REMOVAL_GROUPS.items()}
    rms_by_order = _residual_rms_by_order(rs, ts, ws)

    return Analysis(
        meas=meas, circ=circ, order=order, fit=fit,
        seidels=seidel(coeffs), residual_rms_by_order=rms_by_order,
        _raw_map=raw_map, _raw_vals=w_all, _group_vals=group_vals,
        _ys=ys, _xs=xs, _rho=rho, _theta=theta,
        _res_buf=np.full(shape, np.nan),
    )
