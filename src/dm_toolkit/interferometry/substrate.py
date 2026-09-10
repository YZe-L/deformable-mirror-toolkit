# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-08

"""Substrate = Zygo's "Subtract System Error"."""

import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class Substrate:
    opd_waves: np.ndarray  # System OPD map (waves), NaN outside.
    mask: np.ndarray  # Bool
    wavelength_nm: float | None
    scale: float  # Interferometric scale factor (0.5 double pass)
    meta: dict  # Timestamp, dm_bias, method, steps, note.

    @property
    def shape(self):
        return self.opd_waves.shape

    # Persistence
    def save(self, path):
        """Write a portable .npz (arrays + JSON meta)."""
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        np.savez_compressed(
            path,
            opd_waves=np.where(self.mask, self.opd_waves, np.nan),
            mask=self.mask,
            wavelength_nm=np.array(
                [np.nan if self.wavelength_nm is None else self.wavelength_nm]),
            scale=np.array([self.scale]),
            meta=np.array([json.dumps(self.meta)]),
        )
        return path

    @classmethod
    def load(cls, path):
        d = np.load(str(path), allow_pickle=False)
        wl = float(d["wavelength_nm"][0])
        return cls(
            opd_waves=d["opd_waves"], mask=d["mask"].astype(bool),
            wavelength_nm=None if np.isnan(wl) else wl,
            scale=float(d["scale"][0]),
            meta=json.loads(str(d["meta"][0])),
        )

    @classmethod
    def from_recon(cls, recon, wavelength_nm=None, scale=0.5, dm_bias=None,
                   note=""):
        """Build a substrate from a PSI reconstruction (see psi.reconstruct).

        Args:
            recon: Reconstructed phase or surface map.
            wavelength_nm: Wavelength, in nanometres.
            scale: Scale factor applied to the data.
            dm_bias: Sequence of dm bia values.
            note: Optional explanatory note.
        """
        meta = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "method": recon.get("method", "?"),
                "steps_deg": list(np.rad2deg(np.asarray(recon.get("steps", []),
                                                        float))),
                "dm_bias": dm_bias or {}, "note": note}
        return cls(opd_waves=np.where(recon["mask"], recon["opd_waves"], np.nan),
                   mask=recon["mask"].astype(bool), wavelength_nm=wavelength_nm,
                   scale=float(scale), meta=meta)

    # Application
    def subtract(self, recon, drift_warn_px=3.0):
        """Return a NEW recon dict with the system OPD removed.

        Works on the intersection of the two masks. Fixed-cavity rigs keep the
        beam put, so a straight pixel subtraction is correct (like Zygo). If the
        aperture centroid has drifted more than `drift_warn_px`, a 'drift_px'
        note is attached so the caller can flag a re-reference.

        Args:
            recon: Reconstructed phase or surface map.
            drift_warn_px: Drift warn, in pixels.
        """
        if recon["opd_waves"].shape != self.opd_waves.shape:
            raise ValueError(
                f"substrate shape {self.opd_waves.shape} != measurement "
                f"{recon['opd_waves'].shape}; re-take the substrate at this "
                "camera resolution")
        mask = recon["mask"] & self.mask
        out = dict(recon)
        opd = np.where(mask, recon["opd_waves"] - self.opd_waves, np.nan)
        opd = np.where(mask, opd - np.nanmean(opd[mask]), np.nan)  # Re-piston
        out["opd_waves"] = opd
        out["phase"] = opd * (2 * np.pi)
        out["mask"] = mask
        out["substrate_applied"] = True
        out["drift_px"] = _centroid_drift(recon["mask"], self.mask)
        if out["drift_px"] > drift_warn_px:
            out["substrate_warn"] = (
                f"aperture drifted {out['drift_px']:.1f}px from the substrate "
                "-- consider re-taking it")
        return out


def _centroid_drift(mask_a, mask_b):
    """Pixel distance between two masks' centroids (0 if either is empty).

    Args:
        mask_a: Mask for the first centroid measurement.
        mask_b: Mask for the second centroid measurement.
    """
    ya, xa = np.where(mask_a)
    yb, xb = np.where(mask_b)
    if not len(xa) or not len(xb):
        return 0.0
    return float(np.hypot(xa.mean() - xb.mean(), ya.mean() - yb.mean()))
