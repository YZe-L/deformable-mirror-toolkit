# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-28

"""Run-level background reference measured outside the spot.

Averaging N frames over the area the spot does not use gives far more
samples than the per-frame corner estimate and removes fixed pattern too.
The level scales with exposure, so `exposure_ms` is recorded and
`scale_for` carries the reference across a change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..beam import beam

EXPOSURE_TOL = 0.20  # Fractional exposure drift still considered valid.


@dataclass
class BackgroundRef:
    """What the sensor reads where the beam is not."""
    level: float  # Counts to subtract (median of the sample)
    sigma: float  # Per-pixel temporal noise (read noise)
    fixed_sigma: float  # Spatial spread left after time-averaging.
    n_frames: int
    n_pixels: int
    exposure_ms: float
    shape: tuple = (0, 0)
    hot_pixels: int = 0  # Pixels above level + 8 sigma in the mean frame.
    spot: tuple = (0.0, 0.0, 0.0)  # Cx, cy, r used to mask the beam out.
    stamp: str = ""
    # Per-pixel mean of the background, full frame. Not serialised (saved
    # alongside as .npy); None when the fixed pattern is not worth removing.
    pattern: np.ndarray | None = field(default=None, repr=False)

    def valid_at(self, exposure_ms, tol=EXPOSURE_TOL) -> bool:
        if not (self.exposure_ms > 0 and exposure_ms > 0):
            return False
        return abs(exposure_ms - self.exposure_ms) / self.exposure_ms <= tol

    @property
    def level_err(self):
        """Standard error of `level`, as a median (the 1.253 = sqrt(pi/2)).

        NOT sigma/sqrt(n_pixels*n_frames): the frames are not independent of
        each other in the fixed pattern, which averages down with PIXELS only
        and stays put however many frames you take. Treating all N*F samples as
        independent understated this by ~3x on measured data.
        """
        n = max(self.n_pixels, 1)
        temporal = self.sigma / np.sqrt(n * max(self.n_frames, 1))
        fixed = self.fixed_sigma / np.sqrt(n)
        return float(1.253 * np.hypot(temporal, fixed))

    def scale_for(self, exposure_ms):
        """Factor carrying the reference to `exposure_ms`.

        Dark current and stray light -- what a background at a working
        exposure is made of -- are both proportional to integration time, so
        the LEVEL scales with it. A camera with a large fixed bias offset would
        not; `level` in the run record shows this, by not falling towards zero
        as the exposure shortens. Read noise does not scale, so `sigma` is left
        alone.
        """
        if not (self.exposure_ms > 0 and exposure_ms and exposure_ms > 0):
            return 1.0
        return float(exposure_ms) / self.exposure_ms

    def noise_at(self, px_scale, pattern_applied, exposure_ms=None, n_avg=1):
        """(level, sigma, level_err) for the frame measure() was handed.

        `sigma` is the single-frame temporal noise; the mean of `n_avg` frames
        and area resampling both reduce it, while the fixed pattern does not
        average down and is added back when the map was not subtracted.

        Args:
            px_scale: Area-resampling factor applied to the frame.
            pattern_applied: Whether the per-pixel map was subtracted.
            exposure_ms: Exposure the frame was taken at, in milliseconds.
            n_avg: Number of frames averaged into the scored frame.
        """
        s = float(px_scale) if px_scale > 1.0 else 1.0
        k = self.scale_for(exposure_ms)
        temporal = self.sigma / np.sqrt(max(int(n_avg), 1))
        sigma = (temporal if pattern_applied
                 else float(np.hypot(temporal, self.fixed_sigma)))
        level = 0.0 if pattern_applied else self.level * k
        return level, float(sigma) / s, self.level_err * k

    def to_dict(self) -> dict:
        return dict(level=round(self.level, 4), sigma=round(self.sigma, 4),
                    fixed_sigma=round(self.fixed_sigma, 4),
                    n_frames=self.n_frames, n_pixels=self.n_pixels,
                    exposure_ms=round(self.exposure_ms, 4),
                    shape=list(self.shape), hot_pixels=self.hot_pixels,
                    spot=[round(v, 2) for v in self.spot], stamp=self.stamp,
                    has_pattern=self.pattern is not None)

    def summary(self) -> str:
        txt = (f"level {self.level:.2f} +/- {self.sigma:.2f} counts "
               f"({self.n_frames} frames, {self.n_pixels / 1e3:.0f}k px, "
               f"{self.exposure_ms:.3f} ms)")
        if self.pattern is not None:
            txt += f", fixed pattern {self.fixed_sigma:.2f}"
        if self.hot_pixels:
            txt += f", {self.hot_pixels} hot px"
        return txt


def measure(frames, spot_margin=4.0, min_pixels=5000, max_level_frac=0.5):
    """Background reference from N frames of the live beam.

    The spot is located on the time-averaged frame and masked out with
    `spot_margin` times its radius; the rest is the sample. Returns None
    when that sample is too small or too bright to be background.

    Args:
        frames: Captured image frames.
        spot_margin: Extra pixels retained around the detected spot.
        min_pixels: Minimum permitted pixels.
        max_level_frac: Maximum permitted level fraction.
    """
    if not frames:
        return None
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 4:  # Colour: same channel mean as capture.
        arr = arr.mean(axis=3, dtype=np.float32)
    n = arr.shape[0]
    mean_img = arr.mean(axis=0, dtype=np.float32)
    # Temporal noise per pixel; one frame cannot show it, so fall back to the
    # spatial spread rather than reporting a confident zero.
    temporal = arr.std(axis=0, dtype=np.float32) if n > 1 else None
    h, w = mean_img.shape

    loc = beam.locate_spot(mean_img)
    yy, xx = np.ogrid[0:h, 0:w]
    if loc is None:  # No beam: the whole frame is sample.
        cx, cy, r = w / 2.0, h / 2.0, 0.0
        outside = np.ones((h, w), bool)
    else:
        cx, cy, r = loc
        outside = ((xx - cx) ** 2 + (yy - cy) ** 2) > (spot_margin * r) ** 2

    sample = outside  # The whole discarded area.
    if int(sample.sum()) < min_pixels:
        return None

    vals = mean_img[sample]
    level, fixed_sigma = float(np.median(vals)), float(vals.std())
    # Contrast guard: a beam that fills the frame defeats locate_spot, and a
    # sample that is not far below the peak is not background.
    if level >= max_level_frac * float(mean_img.max()):
        return None
    sigma = (float(temporal[sample].mean()) if temporal is not None
             else fixed_sigma)
    hot = int((vals > level + 8.0 * max(sigma, 1e-6)).sum())

    # Keep the per-pixel map only when the fixed pattern is worth removing:
    # below the single-frame noise it subtracts nothing and only adds a way
    # for a stale reference to bias a frame.
    pattern = None
    if fixed_sigma > 0.5 * max(sigma, 1e-6):
        pattern = mean_img.copy()
        if loc is not None:  # Inside the beam, subtract the level.
            pattern[~outside] = level  # Only -- the spot is not background.

    return BackgroundRef(
        level=level, sigma=float(sigma), fixed_sigma=fixed_sigma,
        n_frames=n, n_pixels=int(sample.sum()),
        exposure_ms=0.0, shape=(h, w), hot_pixels=hot,
        spot=(float(cx), float(cy), float(r)),
        stamp=datetime.now().isoformat(timespec="seconds"), pattern=pattern)
