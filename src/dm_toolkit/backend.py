# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 2.0, 2026-06-29

"""FFT backend: pick GPU (torch/CUDA) or CPU (scipy + pyfftw) per call."""

import numpy as np
from scipy import fft as sfft

# Torch import and CUDA init cost ~3 s, so the GPU probe is lazy and cached.
_torch = None  # The module, once imported.
_cuda = None  # Bool, once probed (None = not yet)

# Auto picks GPU only above this pixel area (small frames: host<->device copy
# dominates, CPU wins)
GPU_AREA_THRESHOLD = 360 * 360


def _probe():
    """Import torch + probe CUDA once. Returns whether CUDA is usable."""
    global _torch, _cuda
    if _cuda is None:
        try:
            import torch
            _torch = torch
            _cuda = bool(torch.cuda.is_available())
        except Exception:  # Torch optional / no CUDA.
            _torch = None
            _cuda = False
    return _cuda


def cuda_available():
    return _probe()


def device_name():
    if _probe():
        try:
            return _torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA"
    return "CPU"


def resolve(backend, n_pixels):
    """'cpu' | 'gpu' | 'auto' -> the device actually used ('cpu' or 'gpu').

    Args:
        backend: Numerical backend to use.
        n_pixels: Number of n pixels.
    """
    if backend == "cpu":
        return "cpu"
    if backend == "gpu":
        return "gpu" if _probe() else "cpu"
    if backend == "auto" and n_pixels >= GPU_AREA_THRESHOLD and _probe():
        return "gpu"
    return "cpu"


def _spiral_kernel_np(h, w):
    """Spiral-phase (vortex) operator exp(i*atan2(v,u)) on the fft2 grid.

    Args:
        h: Image or rectangle height.
        w: Image width or weighting value.
    """
    u = np.fft.fftfreq(w)[None, :]
    v = np.fft.fftfreq(h)[:, None]
    ang = np.arctan2(v, u)
    V = np.exp(1j * ang).astype(np.complex64)
    V[0, 0] = 0.0  # Kill DC (atan2(0,0) is undefined)
    return V


def spiral_transform(fn, device="cpu"):
    """Apply the vortex operator: IFFT( V * FFT(fn) ).

    Returns complex64 numpy. device='gpu' runs the two FFTs on CUDA via torch.

    Args:
        fn: Callable or sampled function.
        device: Computation device to use.
    """
    fn = np.ascontiguousarray(fn, dtype=np.float32)
    h, w = fn.shape
    if device == "gpu" and _probe():
        torch = _torch
        t = torch.from_numpy(fn).cuda()
        F = torch.fft.fft2(t)
        uu = torch.fft.fftfreq(w, device="cuda").unsqueeze(0)
        vv = torch.fft.fftfreq(h, device="cuda").unsqueeze(1)
        V = torch.exp(1j * torch.atan2(vv, uu)).to(torch.complex64)
        V[0, 0] = 0.0
        spt = torch.fft.ifft2(F * V)
        return spt.cpu().numpy().astype(np.complex64)
    F = sfft.fft2(fn, workers=-1)
    spt = sfft.ifft2(F * _spiral_kernel_np(h, w), workers=-1)
    return spt.astype(np.complex64)
