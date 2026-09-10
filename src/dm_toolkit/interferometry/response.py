# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 2.0, 2026-06-14

"""Response-time analysis on displacement traces + live step-watch logic."""

import numpy as np


def robust_sigma(x):
    """MAD-based standard deviation (outlier-immune)."""
    x = np.asarray(x, float)
    return 1.4826 * float(np.median(np.abs(x - np.median(x)))) + 1e-12


def monotonic_unwrap(phase, guard=0.4):
    """Unwrap phase under a monotonic one-way motion assumption.

    Args:
        phase: Training, measurement, or optical phase identifier.
        guard: Monotonic-unwrapping tolerance.
    """
    phase = np.asarray(phase, float)
    if len(phase) < 2:
        return phase.copy(), 0.0
    g = guard * np.pi

    def force(sign):
        out = np.empty_like(phase)
        out[0] = phase[0]
        for i in range(1, len(phase)):
            w = (phase[i] - out[i - 1] + np.pi) % (2 * np.pi) - np.pi
            if sign > 0 and w < -g:
                w += 2 * np.pi
            elif sign < 0 and w > g:
                w -= 2 * np.pi
            out[i] = out[i - 1] + w
        return out

    best = None
    for sign in (1.0, -1.0):
        out = force(sign)
        st = np.diff(out)
        against = float(np.abs(st[np.sign(st) != sign]).sum())
        if best is None or against < best[0]:
            best = (against, out, sign)
    against, out, sign = best
    total = float(np.abs(np.diff(out)).sum()) + 1e-9
    return out, against / total


def aliasing_frames(d, wavelength_nm):
    """Frames whose per-frame motion exceeded ~lambda/4 (phase step near pi).

    There the fringe phase aliases (one fringe = lambda/2 = 2*pi wraps), so the
    displacement is undersampled: neither the spike nor the integrated plateau
    can be trusted. Fundamental Nyquist limit -- the only real cure is a higher
    frame rate. We DETECT and flag rather than invent a value, because a step
    beyond lambda/4 has no unique displacement.

    Args:
        d: Displacement or distance samples.
        wavelength_nm: Wavelength, in nanometres.
    """
    limit = 0.83 * (wavelength_nm / 4.0)
    return np.where(np.abs(np.diff(np.asarray(d, float))) > limit)[0] + 1


def analyze_response_trace(d, fps, wavelength_nm=520.0, aliased=None,
                           start_frac=0.1, settle_frac=0.05):
    """Measure onset, rise, and settling times from a displacement trace.

    Args:
        d: Displacement or distance samples.
        fps: Sequence of fp values.
        wavelength_nm: Wavelength, in nanometres.
        aliased: Indices of samples affected by phase aliasing.
        start_frac: Start expressed as a fraction.
        settle_frac: Settle expressed as a fraction.
    """
    d = np.asarray(d, float)
    n = len(d)
    base = dict(response_s=np.nan, t_start=np.nan, t_settle=np.nan,
                rise_s=np.nan, amplitude_nm=np.nan, sigma_step=np.nan,
                aliased=np.array([], int))
    if n < 10:
        return dict(base, status="too short")

    aliased = (aliasing_frames(d, wavelength_nm) if aliased is None
               else np.asarray(aliased, int))
    base["aliased"] = aliased
    bn = max(3, n // 10)
    pre = float(np.median(d[:bn]))
    post = float(np.median(d[-bn:]))
    amp = post - pre
    sigma = robust_sigma(d[:bn] - pre)
    if abs(amp) < max(6.0 * sigma, 0.5):  # No real motion.
        return dict(base, status="no motion", sigma_step=sigma)

    # Onset: first departure from baseline beyond noise / 10% of amplitude,
    # in EITHER direction (an aliasing dip still marks the true start)
    thr = max(6.0 * sigma, start_frac * abs(amp))
    onset = np.where(np.abs(d - pre) > thr)[0]
    start = int(onset[0]) if len(onset) else 0
    # Settle: last time the trace is outside the plateau band.
    tol = max(settle_frac * abs(amp), 6.0 * sigma)
    outside = np.where(np.abs(d - post) > tol)[0]
    settle = min(int(outside[-1]) + 1, n - 1) if len(outside) else start

    # 10-90% rise, on the trace clipped to the physical [pre,post] envelope
    # so aliasing excursions cannot distort it.
    lo, hi = min(pre, post) - 0.1 * abs(amp), max(pre, post) + 0.1 * abs(amp)
    prog = (np.clip(d, lo, hi) - pre) / amp
    p10 = np.where(prog > 0.1)[0]
    p90 = np.where(prog > 0.9)[0]
    rise_s = ((p90[0] - p10[0]) / fps
              if len(p10) and len(p90) and p90[0] >= p10[0] else np.nan)

    if len(aliased):
        status = f"ALIASED {len(aliased)}f - use higher fps"
    else:
        status = "ok" if start > 0 else "moving at t=0"
    return dict(base, status=status,
                response_s=(settle - start) / fps,
                t_start=start / fps, t_settle=settle / fps,
                rise_s=rise_s, amplitude_nm=amp, sigma_step=sigma)


class StepWatchLogic:
    """Online state machine for the live step watch.

    Calibration (n_cal samples) -> noise floor; trigger threshold
    max(trig_k * sigma_d, 0.05 nm) is the minimum observable step.
    Armed: 2 consecutive hot samples -> moving (level beyond min_step, or
    one large increment confirmed by the following samples). quiet_n
    sub-noise increments -> settled: report amplitude + response time and
    re-zero the baseline."""

    def __init__(self, n_cal=40, trig_k=4.0, quiet_n=6):
        """Initialize the StepWatchLogic.

        Args:
            n_cal: Baseline samples used to calibrate the noise.
            trig_k: Trigger threshold in baseline-noise standard deviations.
            quiet_n: Consecutive quiet samples that count as settled.
        """
        self.n_cal = int(n_cal)
        self.trig_k = float(trig_k)
        self.quiet_n = int(quiet_n)
        self.samples = []
        self.state = "calibrating"
        self.baseline = 0.0
        self.sigma_d = None
        self.sigma_step = None
        self.min_step = None
        self._consec = 0
        self._anchor = 0.0
        self._quiet = 0
        self._quiet_t0 = 0.0
        self._t_move = 0.0
        self._prev_d = None

    def feed(self, t, d):
        """Feed one sample -> list of event dicts (may be empty).

        Args:
            t: Time samples or scalar time.
            d: Displacement or distance samples.
        """
        events = []
        self.samples.append(d)
        if self.state == "calibrating":
            if len(self.samples) >= self.n_cal:
                cal = np.asarray(self.samples, float)
                self.baseline = float(np.median(cal))
                self.sigma_d = robust_sigma(cal)
                self.sigma_step = robust_sigma(np.diff(cal))
                self.min_step = max(self.trig_k * self.sigma_d, 0.05)
                self.state = "armed"
                events.append(dict(type="calibrated", t=t,
                                   sigma_d=self.sigma_d,
                                   min_step=self.min_step))
        elif self.state == "armed":
            inc_thr = max(6.0 * self.sigma_step, 0.05)
            if self._consec == 0:
                inc = abs(d - self._prev_d) if self._prev_d is not None \
                    else 0.0
                hot = (abs(d - self.baseline) > self.min_step
                       or inc > inc_thr)
                self._anchor = self._prev_d if self._prev_d is not None \
                    else d
            else:
                hot = (abs(d - self.baseline) > self.min_step
                       or abs(d - self._anchor) > inc_thr)
            if hot:
                self._consec += 1
                if self._consec == 1:
                    self._t_move = t
                if self._consec >= 2:
                    self.state = "moving"
                    self._quiet = 0
                    events.append(dict(type="motion", t=self._t_move))
            else:
                self._consec = 0
        elif self.state == "moving":
            quiet_thr = max(3.0 * self.sigma_step, 0.02)
            if self._prev_d is not None and \
                    abs(d - self._prev_d) < quiet_thr:
                if self._quiet == 0:
                    self._quiet_t0 = t
                self._quiet += 1
                if self._quiet >= self.quiet_n:
                    events.append(dict(
                        type="step", t=t,
                        t_move=self._t_move,
                        response_s=self._quiet_t0 - self._t_move,
                        amplitude_nm=d - self.baseline))
                    self.baseline = d  # Re-zero for the next step.
                    self.state = "armed"
                    self._consec = 0
            else:
                self._quiet = 0
        self._prev_d = d
        return events
