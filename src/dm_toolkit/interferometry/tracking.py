# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.7, 2026-08-13

"""Incremental fringe-phase tracking + aliasing correction + live detector."""

import numpy as np
from scipy import fft as sfft

from . import fringes
from .. import probe
from .loop import LAMBDA_NM


class FringeTracker:
    """Accumulate per-frame phase changes into displacement."""

    def __init__(self, wavelength_nm=LAMBDA_NM, max_size=800, invert=False,
                 absolute=False, simple=False):
        """Initialize the FringeTracker.

        Args:
            wavelength_nm: Wavelength, in nanometres.
            max_size: Maximum processing dimension, in pixels.
            invert: Whether to invert the measured sign or image convention.
            absolute: Report displacement against the first frame.
            simple: Skip the spatial unwrap and keep the plain wrapped step.
        """
        self.wavelength_nm = float(wavelength_nm)
        self.max_size = int(max_size)
        self.sign = -1.0 if invert else 1.0
        self.absolute = bool(absolute)
        self.simple = bool(simple)
        self.reset()

    def reset(self):
        self.dm = None
        self.prev = None  # Full complex field (cropped grid)
        self.ref = None
        self._prev_m = None  # Masked vector of prev.
        self._ref_m = None
        self.cum_phase = 0.0

    @staticmethod
    def _fast_slice(lo, hi, limit):
        """[lo, hi) clipped to [0, limit), grown to an FFT-fast length.

        Args:
            lo: Lower bound.
            hi: Upper bound.
            limit: Maximum number of elements to return.
        """
        lo, hi = max(int(lo), 0), min(int(np.ceil(hi)), limit)
        n = sfft.next_fast_len(hi - lo)
        if n >= limit:
            return slice(0, limit)
        hi = min(lo + n, limit)
        return slice(hi - n, hi)

    def _build(self, img):
        """First frame: crop to the beam, build the carrier filter."""
        h, w = img.shape
        try:
            _, (cx, cy, r) = fringes.detect_aperture(img)
            margin = 3.0 * 0.04 * max(h, w) + 8  # Soft-edge skirt
            self._sy = self._fast_slice(cy - r - margin, cy + r + margin, h)
            self._sx = self._fast_slice(cx - r - margin, cx + r + margin, w)
        except Exception:
            self._sy, self._sx = slice(0, h), slice(0, w)
        img = img[self._sy, self._sx]
        self.dm = fringes.build_demod(img)
        self._mask = self.dm["mask"]
        self._soft32 = self.dm["soft"].astype(np.float32)
        self._lobe_u = np.fft.ifftshift(self.dm["lobe"])  # Shift-free demod
        return img

    def _demod(self, img):
        work = (img - img[self._mask].mean()) * self._soft32
        F = sfft.fft2(work, workers=-1)
        return sfft.ifft2(F * self._lobe_u, workers=-1)

    def _phase_vs_reference(self, field, fm):
        """Return phase vs reference.

        Wrapped phase against the reference, disambiguated with the
        cumulative track (accurate to within half a fringe).

        Args:
            field: Complex optical field.
            fm: Current complex fringe field.
        """
        prod = fm * np.conj(self._ref_m)
        wrapped = float(np.angle(np.mean(prod)))
        if np.std(np.angle(prod * np.complex64(np.exp(-1j * wrapped)))) > 0.8:
            full = fringes.unwrap2d(np.angle(field * np.conj(self.ref)),
                                    self._mask)
            wrapped = float(np.angle(np.exp(1j * full[self._mask].mean())))
        two_pi = 2.0 * np.pi
        return wrapped + two_pi * round((self.cum_phase - wrapped) / two_pi)

    def update(self, frame):
        """frame: 2-D grayscale array -> measurement dict."""
        img = fringes.downsample(np.asarray(frame, np.float32), self.max_size)
        img = self._build(img) if self.dm is None \
            else img[self._sy, self._sx]
        field = self._demod(img)
        if self.prev is None:
            self.prev = self.ref = field
            self._prev_m = self._ref_m = field[self._mask]
            return dict(displacement_nm=0.0, step_nm=0.0, phase_step=0.0,
                        quality=1.0, near_limit=False)

        fm = field[self._mask]
        prod = fm * np.conj(self._prev_m)
        mean_prod = np.mean(prod)
        step = float(np.angle(mean_prod))
        # Near +-pi the difference map wraps into patches: spatial unwrap.
        # Skipped in simple mode (slow, and mis-fires on a low frame rate).
        if not self.simple and \
                np.std(np.angle(prod * np.complex64(np.exp(-1j * step)))) > 0.8:
            full = fringes.unwrap2d(np.angle(field * np.conj(self.prev)),
                                    self._mask)
            step = float(np.angle(np.exp(1j * full[self._mask].mean())))
        quality = float(np.abs(mean_prod) / (np.mean(np.abs(prod)) + 1e-12))

        self.prev = field
        self._prev_m = fm
        self.cum_phase += step
        if self.absolute and self.simple:
            # Absolute vs the first frame, with the 2*pi fringe order taken
            # from the running track. Reliable while per-frame steps stay < pi.
            wrapped = float(np.angle(np.mean(fm * np.conj(self._ref_m))))
            two_pi = 2.0 * np.pi
            phase = wrapped + two_pi * round((self.cum_phase - wrapped) / two_pi)
        elif self.absolute:
            phase = self._phase_vs_reference(field, fm)
        else:
            phase = self.cum_phase
        k = self.wavelength_nm / (4 * np.pi)
        return dict(displacement_nm=self.sign * phase * k,
                    step_nm=self.sign * step * k,
                    phase_step=step,
                    quality=quality,
                    near_limit=abs(step) > 0.83 * np.pi)


# Downsample for the live surface demodulation.
SURFACE_MAX_SIZE = 480


class SurfaceCentreTracker:
    """Track centre displacement in bent or closed fringes."""

    def __init__(self, wavelength_nm=LAMBDA_NM, invert=False, absolute=True,
                 shrink=0.8, max_size=SURFACE_MAX_SIZE, backend="auto",
                 stabilize=True, follow_probe=False):
        """Initialize the SurfaceCentreTracker.

        Args:
            wavelength_nm: Wavelength, in nanometres.
            invert: Whether to invert the measured sign or image convention.
            absolute: Report displacement against the first frame.
            shrink: Aperture shrink factor applied before analysis.
            max_size: Maximum processing dimension, in pixels.
            backend: Numerical backend to use.
            stabilize: Hold the aperture and sign steady between frames.
            follow_probe: Read the shared probe point instead of the aperture
                centre; the point is re-read every frame.
        """
        self.wavelength_nm = float(wavelength_nm)
        self.invert = bool(invert)
        self.shrink = float(shrink)
        self.max_size = int(max_size)
        self.backend = backend
        self.stabilize = bool(stabilize)
        self.follow_probe = bool(follow_probe)
        self._stab = None  # Created lazily (heavy import)
        self.reset()

    def reset(self):
        self.ref_nm = None
        self.prev_nm = None
        if self._stab is not None:
            self._stab.reset()

    def _centre_nm(self, frame):
        # Lazy: heavy deps (backend)
        from . import surface as surfa
        img = fringes.downsample(np.asarray(frame, float), self.max_size)
        # Re-read per frame: the operator can move the point on the Surface
        # tab while this trace is running and the next sample follows it.
        uv = probe.PROBE.uv if self.follow_probe else probe.CENTRE
        if not self.stabilize:
            result = surfa.reconstruct_centre(img, shrink=self.shrink,
                                              invert=self.invert,
                                              backend=self.backend, uv=uv)
        else:
            if self._stab is None:
                self._stab = surfa.SurfaceStabilizer()
            circ = self._stab.aperture(img, self.shrink)
            result = surfa.reconstruct_centre(img, circ=circ,
                                              invert=self.invert,
                                              backend=self.backend, uv=uv)
            if self._stab.is_flipped(result["wavefront"], result["mask"]):
                result = surfa.reconstruct_centre(img, circ=circ,
                                                  invert=not self.invert,
                                                  backend=self.backend, uv=uv)
            self._stab.accept(result["wavefront"], result["mask"])
        # Surface height (double pass): 1 wave OPD = lambda/2 of mirror travel.
        return result["centre_waves"] * self.wavelength_nm / 2.0, \
            result["openness"]

    def update(self, frame):
        centre_nm, openness = self._centre_nm(frame)
        if self.ref_nm is None:
            self.ref_nm = self.prev_nm = centre_nm
            return dict(displacement_nm=0.0, step_nm=0.0, phase_step=0.0,
                        quality=float(openness), near_limit=False)
        step = centre_nm - self.prev_nm
        disp = centre_nm - self.ref_nm
        self.prev_nm = centre_nm
        return dict(displacement_nm=disp, step_nm=step, phase_step=0.0,
                    quality=float(openness), near_limit=False)


# Aliasing correction
def temporal_unwrap_steps(dphi, guard_frac=0.83):
    """Return temporal unwrap steps.

    Near-pi sign ambiguity resolved by sweep continuity: steps with
    |phase| > guard_frac*pi pick the candidate (s or s - 2*pi*sign(s))
    closer to the previous corrected step.
    Returns (corrected array, flipped indices).

    Args:
        dphi: Wrapped phase-step samples, in radians.
        guard_frac: Fractional guard band around the valid region.
    """
    res, changed, prev = [], [], None
    for i, s in enumerate(np.asarray(dphi, float)):
        if prev is not None and abs(s) > guard_frac * np.pi:
            alt = s - 2.0 * np.pi * np.sign(s)
            if abs(alt - prev) < abs(s - prev):
                s = alt
                changed.append(i)
        res.append(s)
        prev = s
    return np.array(res), changed


def correct_aliased_step(s_raw, win, guard_frac=0.83, margin=1.0):
    """Correct aliased step.

    Correct ONE phase step given the corrected steps already on the same
    sweep branch (win): candidate among {s, s±2pi} closest to the median of
    the last 3 if clearly closer; near-pi rule for the first branch steps.
    The window must reset at the sweep turnaround (callers slice it).

    Args:
        s_raw: Uncorrected phase-step samples.
        win: Local window used to detect an aliased step.
        guard_frac: Fractional guard band around the valid region.
        margin: Aliasing decision margin.
    """
    two_pi = 2.0 * np.pi
    if len(win) >= 2:
        exp = float(np.median(win[-3:]))
        best = min((s_raw, s_raw - two_pi, s_raw + two_pi),
                   key=lambda c: abs(c - exp))
        if best != s_raw and abs(best - exp) < abs(s_raw - exp) - margin:
            return best
    elif len(win) and abs(s_raw) > guard_frac * np.pi:
        prev = win[-1]
        alt = s_raw - two_pi * np.sign(s_raw)
        if abs(alt - prev) < abs(s_raw - prev):
            return alt
    return s_raw


def unwrap_sweep_steps(dphi, turn):
    """Unwrap sweep steps.

    Forward pass of correct_aliased_step, branch window resetting at
    step index `turn`. Returns (corrected array, changed indices).

    Args:
        dphi: Wrapped phase-step samples, in radians.
        turn: Turning-point index or sweep direction.
    """
    out, flipped = [], []
    for i, s in enumerate(np.asarray(dphi, float)):
        win = out[turn:] if i >= turn else out
        c = correct_aliased_step(float(s), win)
        if c != s:
            flipped.append(i)
        out.append(c)
    return np.array(out), flipped


# Live detector
FRINGE_SCORE_MIN = 12.0  # FFT carrier peak / spectrum median.
DETECT_MAX_SIZE = 240  # Detector runs on a tiny copy (~2 ms)


def detect_fringes(frame, max_size=DETECT_MAX_SIZE):
    """Cheap beam + fringe-carrier detector for the live overlay.

    Returns full-resolution coordinates; score >> 1 = clear carrier.

    Args:
        frame: Captured image frame.
        max_size: Maximum processing dimension, in pixels.
    """
    img = fringes.downsample(np.asarray(frame, float), max_size)
    scale = frame.shape[0] / img.shape[0]
    try:
        mask, (cx, cy, r) = fringes.detect_aperture(img)
    except Exception:
        return dict(found=False, score=0.0)
    if mask.sum() < 200:  # Beam too small/absent.
        return dict(found=False, score=0.0)

    work = (img - img[mask].mean()) * mask
    F = np.abs(np.fft.fftshift(np.fft.fft2(work)))
    h, w = F.shape
    cyc, cxc = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    F[np.sqrt((X - cxc) ** 2 + (Y - cyc) ** 2) < 0.03 * min(h, w)] = 0
    F[:cyc, :] = 0
    ky, kx = np.unravel_index(int(np.argmax(F)), F.shape)
    half = F[cyc:, :]
    med = float(np.median(half[half > 0])) + 1e-12
    score = float(F[ky, kx]) / med
    dist = float(np.hypot(kx - cxc, ky - cyc))
    # A real carrier sits several cycles from DC; reject low-freq leakage.
    if dist < max(6.0, 0.05 * min(h, w)) or score < FRINGE_SCORE_MIN:
        return dict(found=False, score=score)

    period_px = scale / max(np.hypot((kx - cxc) / w, (ky - cyc) / h), 1e-9)
    return dict(found=True, score=score, period_px=period_px,
                cx=cx * scale, cy=cy * scale, r=r * scale)
