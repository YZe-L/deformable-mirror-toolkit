# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-05

"""Read a Zygo .datx (HDF5) measurement into a plain Measurement object."""

from dataclasses import dataclass, field

import numpy as np
import h5py

# Curated attribute suffixes worth surfacing (the datx stores dozens); the
# stored key is like "Data Context.Data Attributes.Acquisition Mode".
_ATTR_KEYS = {
    "Acquisition Mode": "measure_mode",
    "Camera Mode": "camera_mode",
    "Camera Name": "camera",
    "System Type": "system",
    "System Serial Number": "serial",
    "Time Stamp": "timestamp",
    "Subtract System Error": "sys_err_subtracted",
}


@dataclass
class Measurement:
    """One imported .datx: raw surface (fringes) + intensity + metadata."""
    path: str
    fringes: np.ndarray  # Surface in fringes, NaN outside aperture.
    mask: np.ndarray  # Bool, True where valid.
    scale: float  # Interferometric scale factor (~0.5)
    wavelength_nm: float | None  # from the file, None if absent
    intensity: np.ndarray | None  # Raw intensity image (counts) or None.
    lateral_res_m: float  # Metres/pixel, 0.0 if uncalibrated.
    attrs: dict = field(default_factory=dict)

    @property
    def waves(self) -> np.ndarray:
        """Surface height map in waves (fringes * scale), NaN outside."""
        return self.fringes * self.scale

    @property
    def shape(self):
        return self.fringes.shape


def _first_dataset(group):
    """The single per-GUID child dataset under a /Data/<kind> group."""
    for k in group:
        if isinstance(group[k], h5py.Dataset):
            return group[k]
    return None


def _scalar(v):
    """Unwrap a (possibly array-wrapped) HDF5 attribute to a python scalar."""
    a = np.array(v).ravel()
    return a[0] if a.size else None


def read_datx(path: str) -> Measurement:
    """Load a .datx into a Measurement (surface fringes + intensity + attrs)."""
    with h5py.File(path, "r") as f:
        surf = _first_dataset(f["Data/Surface"])
        raw = surf[()].astype(np.float64)
        nodata = float(_scalar(surf.attrs["No Data"]))
        scale = float(_scalar(surf.attrs.get("Interferometric Scale Factor", 0.5)))
        wl = surf.attrs.get("Wavelength")
        wavelength_nm = float(_scalar(wl)) * 1e9 if wl is not None else None

        mask = raw < nodata
        fringes = np.where(mask, raw, np.nan)

        intensity = None
        if "Data/Intensity" in f:
            idat = _first_dataset(f["Data/Intensity"])
            if idat is not None:
                intensity = idat[()].astype(np.float64)
                inod = idat.attrs.get("No Data")
                if inod is not None:
                    intensity = np.where(intensity >= float(_scalar(inod)),
                                         np.nan, intensity)

        attrs, lateral = _read_attributes(f)
    return Measurement(path=str(path), fringes=fringes, mask=mask, scale=scale,
                       wavelength_nm=wavelength_nm, intensity=intensity,
                       lateral_res_m=lateral, attrs=attrs)


def _read_attributes(f):
    """Read attributes.

    Pull the curated metadata + lateral resolution from the Attributes group.
    """
    attrs, lateral = {}, 0.0
    if "Attributes" not in f:
        return attrs, lateral
    for guid in f["Attributes"]:
        node = f["Attributes"][guid]
        for key in node.attrs:
            for suffix, short in _ATTR_KEYS.items():
                if key.endswith(suffix):
                    val = _scalar(node.attrs[key])
                    if isinstance(val, bytes):
                        val = val.decode(errors="ignore")
                    if short == "timestamp":
                        val = _fmt_timestamp(val)
                    attrs[short] = val
            if key.endswith("Lateral Resolution:Value"):
                lateral = float(_scalar(node.attrs[key]) or 0.0)
    return attrs, lateral


def _fmt_timestamp(val):
    """Format a Zygo timestamp as ISO text.

    Zygo stores time as a struct (epoch_seconds, ms, tz, dst) -> ISO string.
    """
    try:
        secs = int(np.asarray(val).item()[0]) if np.asarray(val).dtype.names \
            else int(val)
        import datetime
        return datetime.datetime.fromtimestamp(secs).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(val)


def find_datx(folder: str):
    """List .datx files in a folder (sorted), for the folder picker."""
    from pathlib import Path
    p = Path(folder)
    if not p.is_dir():
        return []
    return sorted(p.glob("*.datx"))
