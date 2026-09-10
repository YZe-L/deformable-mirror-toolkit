# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-06-12

"""Takeda carrier demodulation: interferogram -> complex fringe field."""

import numpy as np
import cv2
from scipy import fft as sfft

# Optional FFTW backend for all scipy.fft calls (same transforms, faster)
try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(60.0)
    sfft.set_global_backend(pyfftw.interfaces.scipy_fft)
except ImportError:
    pass


def downsample(img, max_size=800):
    """Shrink so the longest side <= max_size (anti-aliased).

    Args:
        img: Input image.
        max_size: Maximum processing dimension, in pixels.
    """
    h, w = img.shape
    s = max_size / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)),
                      interpolation=cv2.INTER_AREA)


def load_frame(path, max_size=800):
    """Image file -> float grayscale, downsampled.

    Args:
        path: Filesystem path used by the operation.
        max_size: Maximum processing dimension, in pixels.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(path)
    return downsample(raw.astype(float), max_size)


def detect_aperture(img, thresh_frac=0.12, shrink=0.8):
    """Find the circular beam -> (bool mask, (cx, cy, r)).

    Args:
        img: Input image.
        thresh_frac: Detection threshold as a fraction of the peak.
        shrink: Image downsampling factor.
    """
    u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Blur wider than the fringe spacing so dark fringes vanish.
    b = cv2.GaussianBlur(u8, (0, 0), max(img.shape) / 20.0)
    th = (b > thresh_frac * b.max()).astype(np.uint8) * 255
    k = np.ones((21, 21), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(th)
    cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)
    M = cv2.moments(filled, binaryImage=True)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    r = np.sqrt((filled > 0).sum() / np.pi) * shrink
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    return mask, (cx, cy, r)


def unwrap2d(phase, mask=None):
    """2-D phase unwrap inside the aperture.

    Args:
        phase: Training, measurement, or optical phase identifier.
        mask: Boolean mask selecting valid samples.
    """
    try:
        from skimage.restoration import unwrap_phase
        if mask is not None:
            ma = np.ma.masked_array(phase, mask=~mask)
            return np.asarray(unwrap_phase(ma).filled(0.0))
        return unwrap_phase(phase)
    except Exception:
        return np.unwrap(np.unwrap(phase, axis=1), axis=0)


def find_lobe(img, mask, soft, lobe_radius_frac=0.30):
    """Carrier side-lobe in the spectrum -> (kx, ky, radius).

    Args:
        img: Input image.
        mask: Boolean mask selecting valid samples.
        soft: Whether to use a soft-edged Fourier mask.
        lobe_radius_frac: Lobe radius expressed as a fraction.
    """
    h, w = img.shape
    work = (img - img[mask].mean()) * soft
    F = np.fft.fftshift(sfft.fft2(work, workers=-1))
    cy, cx = h // 2, w // 2
    mag = np.abs(F)
    Y, X = np.ogrid[:h, :w]
    mag[np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) < 0.015 * min(h, w)] = 0
    mag[:cy, :] = 0  # One half-plane
    ky, kx = np.unravel_index(np.argmax(mag), mag.shape)
    rr = lobe_radius_frac * np.hypot(kx - cx, ky - cy)
    return kx, ky, rr


def build_demod(img0):
    """Fixed carrier filter from frame 0 (same filter for every frame)."""
    mask, _ = detect_aperture(img0)
    h, w = img0.shape
    soft = cv2.GaussianBlur(mask.astype(float), (0, 0), 0.04 * max(h, w))
    kx, ky, rr = find_lobe(img0, mask, soft)
    Y, X = np.ogrid[:h, :w]
    lobe = np.sqrt((X - kx) ** 2 + (Y - ky) ** 2) < rr
    return dict(mask=mask, soft=soft, lobe=lobe)


def demod(img, dm):
    """One frame -> complex fringe field (band-passed carrier).

    Args:
        img: Input image.
        dm: Deformable-mirror phase or command data.
    """
    work = (img - img[dm["mask"]].mean()) * dm["soft"]
    F = np.fft.fftshift(sfft.fft2(work, workers=-1))
    return sfft.ifft2(np.fft.ifftshift(F * dm["lobe"]), workers=-1)
