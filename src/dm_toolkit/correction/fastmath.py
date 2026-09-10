# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-28

"""Selectable reduction backend for the frame average and corner statistics.

Both backends return the same numbers. The frame average accumulates in
float32, which is exact for this data; bottleneck is used only where the
array already has a kernel dtype (see _bn_ready).
"""

from __future__ import annotations

import time

import numpy as np

try:
    import bottleneck as bn
except ImportError:  # Optional; BACKEND_BN then falls back.
    bn = None


# Also the combo keys and the value stored in LoopSettings.reduction_backend.
BACKEND_NUMPY = "numpy"  # np.sum(dtype=float32) / n
BACKEND_BN = "bottleneck"

BACKENDS = (BACKEND_NUMPY, BACKEND_BN)

BACKEND_LABELS = {
    BACKEND_NUMPY: "NumPy",
    BACKEND_BN: "Bottleneck (where it is free)",
}

_LEGACY = {"numpy_f32": BACKEND_NUMPY}  # The float64/float32 split, now merged.


def available() -> bool:
    """True when bottleneck is importable, i.e. BACKEND_BN does anything."""
    return bn is not None


def resolve(backend) -> str:
    """Backend name -> one this module implements; unknown/missing -> NumPy."""
    backend = _LEGACY.get(backend, backend)
    if backend == BACKEND_BN and bn is None:
        return BACKEND_NUMPY
    return backend if backend in BACKENDS else BACKEND_NUMPY


# The only dtypes bottleneck ships kernels for; anything else hits bn.slow.
_BN_DTYPES = frozenset((np.dtype(np.float64), np.dtype(np.float32),
                        np.dtype(np.int64), np.dtype(np.int32)))


def _bn_ready(a) -> bool:
    """True when bottleneck can act on `a` as it already is.

    Never casts to get here: the cast costs more than the kernel saves.
    """
    return bn is not None and a.dtype in _BN_DTYPES


# Narrow integer frames, which every mono sensor here delivers. None is in
# _BN_DTYPES, so the fast path never takes work from bottleneck.
_EXACT_SUM_DTYPES = frozenset((np.dtype(np.uint8), np.dtype(np.uint16),
                               np.dtype(np.int8), np.dtype(np.int16)))
_F32_EXACT_INT = 1 << 24  # Largest integer float32 represents exactly.


def _int_frames(frames, first) -> bool:
    """True when `frames` can be summed in float32 with no rounding at all.

    Requires every frame to be a same-shaped array of one narrow integer
    dtype, and the worst-case running total to stay inside float32's exact
    integer range, so in-place accumulation returns the same bits as np.sum.

    Args:
        frames: Captured image frames.
        first: `frames[0]`, already as an array.
    """
    if first.dtype not in _EXACT_SUM_DTYPES:
        return False
    info = np.iinfo(first.dtype)
    largest = max(int(info.max), abs(int(info.min)))  # Signed frames go both ways.
    if len(frames) * largest >= _F32_EXACT_INT:
        return False
    return all(isinstance(f, np.ndarray) and f.dtype == first.dtype
               and f.shape == first.shape for f in frames)


def mean_frames(frames, backend=BACKEND_NUMPY):
    """Average same-shape frames into one image (float32).

    Args:
        frames: Captured image frames.
        backend: Numerical backend to use.
    """
    n = len(frames)
    if n == 1:
        return np.asarray(frames[0], dtype=np.float32)
    first = np.asarray(frames[0])
    if _int_frames(frames, first):
        # Integer frames sum exactly in float32, so accumulating in place
        # matches np.sum without the extra stacked copy.
        acc = first.astype(np.float32)
        for f in frames[1:]:
            acc += f
        acc /= np.float32(n)
        return acc
    arr = np.asarray(frames)
    if resolve(backend) == BACKEND_BN and _bn_ready(arr):  # Colour path only.
        return bn.nanmean(arr, axis=0).astype(np.float32, copy=False)
    # dtype= is the accumulator, not a conversion: no second copy
    return np.sum(arr, axis=0, dtype=np.float32) / np.float32(n)


def median(a, backend=BACKEND_NUMPY):
    """Median of a 1-D array. Value-identical across backends.

    Args:
        a: Input array or scalar value.
        backend: Numerical backend to use.
    """
    if resolve(backend) == BACKEND_BN and _bn_ready(a):
        return float(bn.median(a))
    return float(np.median(a))


def std(a, backend=BACKEND_NUMPY):
    """Return the population standard deviation.

    Population standard deviation (ddof=0, like ndarray.std and bn.nanstd).

    Args:
        a: Input array or scalar value.
        backend: Numerical backend to use.
    """
    if resolve(backend) == BACKEND_BN and _bn_ready(a):
        return float(bn.nanstd(a))
    return float(np.asarray(a).std())


def mean(a, backend=BACKEND_NUMPY):
    """Arithmetic mean of a 1-D array.

    Args:
        a: Input array or scalar value.
        backend: Numerical backend to use.
    """
    if resolve(backend) == BACKEND_BN and _bn_ready(a):
        return float(bn.nanmean(a))
    return float(np.asarray(a).mean())


def benchmark_average(frames, repeats=7):
    """Time mean_frames per backend, in BACKENDS order.

    Each dict holds backend/label/ms/speedup/max_abs_diff/available, both
    compared against BACKEND_NUMPY. `ms` is the best run, not the mean -- the
    tail is scheduler noise.

    Args:
        frames: Captured image frames.
        repeats: Number of repeated measurements.
    """
    ref = None
    base_ms = None
    out = []
    for backend in BACKENDS:
        if backend == BACKEND_BN and bn is None:
            out.append(dict(backend=backend, label=BACKEND_LABELS[backend],
                            ms=float("nan"), speedup=float("nan"),
                            max_abs_diff=float("nan"), available=False))
            continue
        best = float("inf")
        res = None
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            res = mean_frames(frames, backend)
            best = min(best, time.perf_counter() - t0)
        ms = best * 1e3
        if ref is None:
            ref, base_ms = res, ms
            diff = 0.0
        else:
            diff = float(np.max(np.abs(res.astype(np.float64)
                                       - ref.astype(np.float64))))
        out.append(dict(backend=backend, label=BACKEND_LABELS[backend],
                        ms=ms, speedup=(base_ms / ms if ms > 0 else float("nan")),
                        max_abs_diff=diff, available=True))
    return out
