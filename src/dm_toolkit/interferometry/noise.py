# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-06-30

"""Noise-floor spectrum from a live displacement trace."""

import numpy as np

from .response import robust_sigma


NOISE_BANDS = (
    ("drift", 0.01, 0.1),
    ("environment", 0.1, 1.0),
    ("fast", 1.0, None),
)


def _integrated_noise_bands(freq, asd, df, nyquist):
    """Integrate one-sided ASD into useful displacement-noise bands.

    The returned RMS values are in nm. A value is NaN when the record is too
    short to contain a bin in that band, or when Nyquist is below the band.

    Args:
        freq: Frequency samples or requested PWM frequency.
        asd: Amplitude spectral-density samples.
        df: Tabular measurement data.
        nyquist: Nyquist frequency, in hertz.
    """
    bands = []
    for key, lo, requested_hi in NOISE_BANDS:
        hi = nyquist if requested_hi is None else min(requested_hi, nyquist)
        if hi <= lo:
            value = float("nan")
        else:
            # Keep boundary bins in exactly one band.
            mask = (freq >= lo) & ((freq <= hi) if hi == nyquist
                                   else (freq < hi))
            value = (float(np.sqrt(np.sum(asd[mask] ** 2) * df))
                     if np.any(mask) else float("nan"))
        bands.append(dict(key=key, lo=lo, hi=hi, rms=value))

    positive = freq > 0
    spectral_rms = (float(np.sqrt(np.sum(asd[positive] ** 2) * df))
                    if np.any(positive) else float("nan"))
    return bands, spectral_rms


def noise_spectrum(t, d):
    """(t, d) trace -> spectrum, sampling, and integrated-noise metrics.

    amp: one-sided amplitude spectrum (nm) -- height of each frequency.
    asd: amplitude spectral density (nm/sqrt(Hz)) -- the noise floor.
    rms: robust 1-sigma of the raw trace (nm). ptp: peak-to-peak (nm).
    bands: ASD-integrated RMS for 0.01-0.1, 0.1-1, and 1-Nyquist Hz.
    spectral_rms: ASD-integrated RMS over all positive-frequency bins.

    Args:
        t: Time samples or scalar time.
        d: Displacement or distance samples.
    """
    t = np.asarray(t, float)
    d = np.asarray(d, float)
    n = int(len(d))
    empty = dict(freq=np.array([]), amp=np.array([]), asd=np.array([]),
                 fs=0.0, rms=float("nan"), ptp=float("nan"), n=n, df=0.0,
                 bands=[], spectral_rms=float("nan"))
    if n < 8:
        return empty
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return empty
    fs = 1.0 / dt
    # Uniform resample at the median rate, then remove mean + linear drift.
    tu = np.arange(t[0], t[-1], dt)
    du = np.interp(tu, t, d)
    N = int(len(du))
    if N < 8:
        return empty
    idx = np.arange(N)
    du = du - np.polyval(np.polyfit(idx, du, 1), idx)

    win = np.hanning(N)
    X = np.fft.rfft(du * win)
    freq = np.fft.rfftfreq(N, d=dt)
    cg = win.mean()  # Coherent gain -> amplitude scale.
    amp = np.abs(X) * 2.0 / (N * cg)
    nps = float(np.sum(win ** 2))  # Window noise power -> ASD scale.
    asd = np.abs(X) * np.sqrt(2.0 / (fs * nps))
    df = fs / N
    bands, spectral_rms = _integrated_noise_bands(
        freq, asd, df, fs / 2.0)
    return dict(freq=freq, amp=amp, asd=asd, fs=fs,
                rms=float(robust_sigma(d - np.median(d))),
                ptp=float(np.max(d) - np.min(d)), n=n, df=df,
                bands=bands, spectral_rms=spectral_rms)
