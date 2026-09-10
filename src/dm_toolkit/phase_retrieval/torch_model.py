# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-08-22

"""GPU forward model with analytic gradients, for the retrieval fit.

Same physics as `estimate._Forward` in PyTorch, so the Jacobian comes from
automatic differentiation and the evaluation can run on the GPU. The
propagation is the matrix Fourier transform of Soummer et al., Opt.
Express 15, 15935 (2007), written out so autograd can trace it:

    Q      = (wavelength * focal_length) / (pupil_diameter * output_dx)
    Eout   = exp(-2i pi / N * (1/Q) * outer(Y, V)^T)
    Ein    = exp(-2i pi / N * (1/Q) * outer(X, U)) / (N * Q)
    field  = Eout @ pupil @ Ein

`verify_against_prysm` checks it against the reference implementation.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional at import time
    torch = None


def available(require_cuda=False):
    """Whether this backend can run.

    Args:
        require_cuda: Demand a working GPU rather than accepting the CPU.

    Returns:
        True when torch is importable and, if asked, CUDA is usable.
    """
    if torch is None:
        return False
    return bool(torch.cuda.is_available()) if require_cuda else True


def _fftrange(n, device, dtype):
    """Coordinate grid an FFT would use: -(n//2) .. -(n//2)+n-1."""
    start = -(n // 2)
    return torch.arange(start, start + n, device=device, dtype=dtype)


class TorchForward:
    """Cached Fraunhofer model on one device, differentiable in the modes.

    Attributes:
        device (torch.device): Where the tensors live.
        modes (tuple): Noll indices, in fit order.
        samples (int): Detector grid side, in pixels.
    """

    def __init__(self, optics, modes, basis, amp, tilt_x, tilt_y, samples,
                 device=None, dtype=torch.float32):
        """Build the propagation matrices once.

        Args:
            optics: Wavelength, focal length, aperture and pixel pitch.
            modes: Noll indices, in fit order.
            basis: (n_modes, n, n) Zernike basis, already masked -- taken from
                the numpy model so both backends share one convention.
            amp: (n, n) pupil amplitude.
            tilt_x: (n, n) normalised x coordinate, masked.
            tilt_y: (n, n) normalised y coordinate, masked.
            samples: Detector grid side, in pixels.
            device: Torch device; defaults to CUDA when present.
            dtype: Real dtype; complex uses the matching width.
        """
        if torch is None:
            raise RuntimeError("torch is not installed")
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.optics = optics
        self.modes = tuple(modes)
        self.samples = int(samples)
        self.rdtype = dtype
        self.cdtype = torch.complex64 if dtype == torch.float32 \
            else torch.complex128

        to = dict(device=self.device, dtype=self.rdtype)
        self.basis = torch.as_tensor(np.asarray(basis), **to)
        self.amp = torch.as_tensor(np.asarray(amp), **to)
        self.tilt_x = torch.as_tensor(np.asarray(tilt_x), **to)
        self.tilt_y = torch.as_tensor(np.asarray(tilt_y), **to)
        self.tilt_gain = float(optics.aperture_mm * 1e3
                               / (2.0 * optics.focal_mm
                                  * optics.wavelength_nm))
        self._build_matrices()

    def _build_matrices(self):
        """The two Soummer matrices for the configured sampling."""
        n = int(self.basis.shape[-1])
        diameter_mm = float(self.optics.aperture_mm)
        wavelength_mm = float(self.optics.wavelength_nm) * 1e-6
        focal_mm = float(self.optics.focal_mm)
        output_dx_mm = float(self.optics.pixel_um) * 1e-3
        # Q is the resolution element in units of the output pixel.
        q = (wavelength_mm * focal_mm) / (diameter_mm * output_dx_mm)
        m = 1.0 / q
        idx = _fftrange(n, self.device, self.rdtype)
        out = _fftrange(self.samples, self.device, self.rdtype)
        phase_out = (-2.0 * np.pi / n) * m * torch.outer(idx, out)
        self.e_out = torch.exp(1j * phase_out.to(self.cdtype)).T.contiguous()
        e_in = torch.exp(1j * phase_out.to(self.cdtype))
        self.e_in = (e_in / (n * q)).contiguous()
        self.q = q

    def psf(self, coeffs, shift_um):
        """Unit-sum model spot, differentiable in `coeffs` and `shift_um`.

        Args:
            coeffs: 1-D tensor of amplitudes, in RMS waves.
            shift_um: 1-D tensor of length 2, spot offset in sensor
                micrometres, applied as pupil tilt exactly as the numpy model
                does.

        Returns:
            A (samples, samples) real tensor summing to one.
        """
        phase = torch.tensordot(coeffs, self.basis, dims=1)
        phase = phase + self.tilt_gain * (shift_um[0] * self.tilt_x
                                          + shift_um[1] * self.tilt_y)
        pupil = self.amp * torch.exp(1j * (2.0 * np.pi * phase).to(
            self.cdtype))
        field = self.e_out @ pupil @ self.e_in
        image = field.real ** 2 + field.imag ** 2
        total = image.sum()
        return image / total

    def residual_and_jacobian(self, params, data, weight, mask,
                              fixed_background):
        """Weighted residual and its Jacobian, in one forward/backward pass.

        Args:
            params: 1-D numpy array: coefficients then the two shift terms.
            data: 1-D tensor of measured values over the fitted pixels.
            weight: 1-D tensor of per-pixel weights.
            mask: 1-D bool tensor selecting the fitted pixels, or None for
                all of them.
            fixed_background: Pedestal to pin, or None to solve for it.

        Returns:
            A (residual, jacobian) pair of numpy arrays.
        """
        n_modes = len(self.modes)
        theta = torch.as_tensor(np.asarray(params, float),
                                device=self.device, dtype=self.rdtype)
        theta.requires_grad_(True)

        def evaluate(vec):
            image = self.psf(vec[:n_modes], vec[n_modes:n_modes + 2]).ravel()
            model = image[mask] if mask is not None else image
            flux, back = _solve_linear(model, data, weight, fixed_background)
            return (flux * model + back) * weight - data * weight

        jac = torch.autograd.functional.jacobian(
            evaluate, theta, vectorize=True, strategy="forward-mode")
        with torch.no_grad():
            res = evaluate(theta)
        return (res.detach().cpu().numpy().astype(float),
                jac.detach().cpu().numpy().astype(float))


def _solve_linear(model, data, weight, fixed_background):
    """Flux (and pedestal) by weighted least squares, differentiably.

    Mirrors `estimate._solve_flux_background` so the two backends optimise
    the same function.
    """
    w2 = weight * weight
    if fixed_background is not None:
        offset = data - fixed_background
        denom = torch.dot(w2, model * model)
        flux = torch.dot(w2, model * offset) / torch.clamp(denom, min=1e-30)
        return torch.clamp(flux, min=0.0), fixed_background
    s_ww = w2.sum()
    s_m = torch.dot(w2, model)
    s_mm = torch.dot(w2, model * model)
    s_d = torch.dot(w2, data)
    s_md = torch.dot(w2, model * data)
    det = s_mm * s_ww - s_m * s_m
    det = torch.where(det.abs() < 1e-30, torch.full_like(det, 1e-30), det)
    flux = (s_ww * s_md - s_m * s_d) / det
    back = (s_mm * s_d - s_m * s_md) / det
    return torch.clamp(flux, min=0.0), back


def verify_against_prysm(forward, torch_forward, coeffs, shift_um,
                         samples=None):
    """Largest relative difference between this model and the prysm one.

    Anything above about 1e-6 in single precision means the two are no
    longer the same model.

    Args:
        forward: The numpy `_Forward`.
        torch_forward: The `TorchForward` built from it.
        coeffs: Amplitudes to compare at, in RMS waves.
        shift_um: Spot offset to compare at.
        samples: Detector grid side; defaults to the torch model's.

    Returns:
        The maximum absolute difference divided by the numpy model's peak.
    """
    side = int(samples or torch_forward.samples)
    reference = forward.psf(coeffs, shift_um, 1.0, side)
    with torch.no_grad():
        amps = torch.as_tensor(np.asarray(coeffs, float),
                               device=torch_forward.device,
                               dtype=torch_forward.rdtype)
        offs = torch.as_tensor(np.asarray(shift_um, float),
                               device=torch_forward.device,
                               dtype=torch_forward.rdtype)
        mine = torch_forward.psf(amps, offs).cpu().numpy()
    return float(np.abs(mine - reference).max() / reference.max())
