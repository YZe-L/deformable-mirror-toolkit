# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-08

"""Offline PSI measurement bundle: one .npz that perfectly restores a run."""

import json
from datetime import datetime

import numpy as np

from . import psi


_ARRAYS = ("phase", "wrapped", "opd_waves", "modulation", "mask", "intensity",
           "steps")


def save_bundle(path, frames, recon, meta=None):
    """Write frames + reconstruction + metadata to a .npz. Returns the path.

    Args:
        path: Filesystem path used by the operation.
        frames: Captured image frames.
        recon: Reconstructed phase or surface map.
        meta: Metadata associated with the measurement.
    """
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    frames = np.asarray(frames)
    md = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          "method": recon.get("method", "?"), "invert": recon.get("invert", False),
          "frame_dtype": str(frames.dtype), "n_frames": int(frames.shape[0])}
    if meta:
        md.update(meta)
    store = {f"recon_{k}": np.asarray(recon[k]) for k in _ARRAYS if k in recon}
    np.savez_compressed(path, frames=frames, meta=np.array([json.dumps(md)]),
                        **store)
    return path


def load_bundle(path, recompute=False):
    """Load a bundle -> (frames, recon, meta).

    recompute=False (default) returns the stored reconstruction verbatim.
    recompute=True re-runs psi.reconstruct() on the raw frames with the saved
    method -- use it to re-derive after an algorithm change (identical result on
    the same code).

    Args:
        path: Filesystem path used by the operation.
        recompute: Whether to recompute cached analysis products.
    """
    d = np.load(str(path), allow_pickle=False)
    frames = d["frames"]
    meta = json.loads(str(d["meta"][0]))
    if recompute:
        recon = psi.reconstruct(frames.astype(float),
                                method=meta.get("method", "aia"),
                                invert=bool(meta.get("invert", False)))
    else:
        recon = {k: d[f"recon_{k}"] for k in _ARRAYS if f"recon_{k}" in d}
        recon["mask"] = recon["mask"].astype(bool)
        recon["method"] = meta.get("method", "?")
        recon["invert"] = bool(meta.get("invert", False))
    return frames, recon, meta


def measurement_from_bundle(path, recompute=False):
    """Convenience: load a bundle and return a ready Zygo-core Measurement.

    Args:
        path: Filesystem path used by the operation.
        recompute: Whether to recompute cached analysis products.
    """
    _frames, recon, meta = load_bundle(path, recompute=recompute)
    return psi.build_measurement(
        recon, path=path, wavelength_nm=meta.get("wavelength_nm"),
        scale=float(meta.get("scale", 0.5)), attrs=meta)
