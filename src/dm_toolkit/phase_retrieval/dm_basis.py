# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-08-22

"""The mirror's own eigenmodes as the phase-retrieval basis.

A low-actuator-count mirror does not make Zernikes: part of every mode it
can drive lies outside the Noll set the retrieval fits, and an intensity
fit absorbs that content into the modes it has. Expanding the phase in the
measured eigenmodes instead uses fewer parameters, makes the fitted
amplitudes the command itself, and leaves what the mirror cannot make in
the residual. The sign twin `phi(r) -> -phi(-r)` remains; `twin` computes
it. Reporting still happens in Zernikes through `zernike_matrix`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import zernike as ZK

# Retained-mode count for the nine-actuator mirror: the singular values
# step by <=1.64x down to mode 6 and by 8.57x after it.
DEFAULT_KEEP = 6

# Noll terms used to report a fitted mode combination; 15 covers through
# tetrafoil, which is what `named_aberrations` expects.
REPORT_TERMS = 15


class DMBasis:
    """The mirror's eigenmodes sampled on the retrieval's pupil grid.

    Attributes:
        maps: (K, n, n) modal shapes in waves per unit modal amplitude,
            masked to the unit disk and normalised to unit RMS over it.
        labels: One name per fitted mode, for figures and CSV headers.
        rotation_deg: Rotation applied to bring the interferometer's frame
            into the camera's. See `orientations`.
        flip: Whether a left-right flip was applied for the same reason.
        source: Where the modes were read from.
    """

    def __init__(self, maps, mask, rotation_deg=0.0, flip=False, source=""):
        self.maps = np.asarray(maps, float)
        self.mask = np.asarray(mask, bool)
        self.rotation_deg = float(rotation_deg)
        self.flip = bool(flip)
        self.source = str(source)
        self.labels = tuple(f"U{i + 1}" for i in range(len(self.maps)))

    def __len__(self):
        return len(self.maps)

    def at_samples(self, samples):
        """The same basis on a different pupil grid.

        The fit runs a coarse stage on a smaller pupil grid than the fine
        one, so a basis handed in at one sampling has to be available at the
        other. The modes are smooth and band-limited by the actuator pitch,
        so resampling them is not an approximation the fit can see.

        Args:
            samples: Pupil samples across the diameter.
        """
        n = int(samples)
        if n == self.maps.shape[-1]:
            return self
        src_n = self.maps.shape[-1]
        maps = np.stack([_resample(m, src_n, n) for m in self.maps])
        ax = (np.arange(n) - (n - 1) / 2.0) / (n / 2.0)
        x, y = np.meshgrid(ax, ax)
        mask = np.hypot(x, y) <= 1.0
        out = np.empty_like(maps)
        for i, m in enumerate(maps):
            v = m[mask]
            out[i] = np.where(mask, (m - v.mean()) / (v.std() or 1.0), 0.0)
        return DMBasis(out, mask, self.rotation_deg, self.flip, self.source)

    def phase(self, amps):
        """Pupil phase in waves for one vector of modal amplitudes."""
        return np.tensordot(np.asarray(amps, float), self.maps, axes=1)

    def zernike_matrix(self, n_terms=REPORT_TERMS):
        """(K, n_terms) matrix mapping modal amplitudes to Noll coefficients.

        Least-squares projection of each mode onto Noll 1..n_terms over the
        pupil, so `amps @ M` is the Zernike description of the fitted phase.
        Reporting only -- the fit itself never uses it.

        Args:
            n_terms: Highest Noll index to report.
        """
        if getattr(self, "_zmat", None) is not None:
            if self._zmat.shape[1] == n_terms:
                return self._zmat
        n = self.maps.shape[-1]
        circ = ((n - 1) / 2.0,) * 3
        rho, theta, ys, xs = ZK.unit_coords(self.mask, circ)
        keep = rho <= 1.0
        basis = ZK.basis(rho[keep], theta[keep], n_terms)
        ys, xs = ys[keep], xs[keep]
        cols = []
        for m in self.maps:
            c, *_ = np.linalg.lstsq(basis.T, m[ys, xs], rcond=None)
            cols.append(c)
        self._zmat = np.vstack(cols)
        return self._zmat

    def twin(self, amps):
        """The other candidate a single in-focus frame cannot rule out.

        The ambiguity is `phi(r) -> -phi(-r)` (Smith et al. 2013, Property
        3.1). In a Zernike basis that is "negate the even radial orders"; in
        an arbitrary basis it is not, so the twin phase is built explicitly
        and projected back onto the modes.

        Args:
            amps: Fitted modal amplitudes.

        Returns:
            The modal amplitudes of the twin, same length.
        """
        phase = self.phase(amps)
        twin = -phase[::-1, ::-1]
        flat = np.stack([m[self.mask] for m in self.maps])
        sol, *_ = np.linalg.lstsq(flat.T, twin[self.mask], rcond=None)
        return sol


def _resample(map2d, grid_n, samples):
    """Bilinear resample of one mode map onto a `samples`-square grid.

    Both grids inscribe the same unit circle -- `impact_matrix.influence.Grid`
    spans -1..+1 over `grid_n` points and `_Forward` spans the same range
    over `pupil_samples` -- so this is a straight coordinate rescale with no
    registration in it.

    Args:
        map2d: One mode surface on the influence-matrix grid.
        grid_n: Side of that grid.
        samples: Side of the retrieval's pupil grid.
    """
    src = np.nan_to_num(np.asarray(map2d, float))
    ax = np.linspace(0.0, grid_n - 1.0, samples)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    y0 = np.clip(np.floor(yy).astype(int), 0, grid_n - 2)
    x0 = np.clip(np.floor(xx).astype(int), 0, grid_n - 2)
    fy, fx = yy - y0, xx - x0
    return ((1 - fy) * (1 - fx) * src[y0, x0]
            + (1 - fy) * fx * src[y0, x0 + 1]
            + fy * (1 - fx) * src[y0 + 1, x0]
            + fy * fx * src[y0 + 1, x0 + 1])


def _orient(maps, rotation_deg, flip):
    """Apply the camera-versus-interferometer orientation to every mode.

    The two instruments look at the mirror down different optical paths, so
    handedness and rotation differ. Both are observable from the spot,
    because rotating a pupil rotates its point-spread function.

    Args:
        maps: (K, n, n) modal shapes.
        rotation_deg: Counter-clockwise rotation, in degrees.
        flip: Mirror the columns before rotating.
    """
    out = np.asarray(maps, float)
    if flip:
        out = out[:, :, ::-1]
    if abs(float(rotation_deg)) < 1e-9:
        return np.ascontiguousarray(out)
    n = out.shape[-1]
    ax = (np.arange(n) - (n - 1) / 2.0)
    x, y = np.meshgrid(ax, ax)
    a = np.deg2rad(float(rotation_deg))
    xs = np.cos(a) * x + np.sin(a) * y + (n - 1) / 2.0
    ys = -np.sin(a) * x + np.cos(a) * y + (n - 1) / 2.0
    x0 = np.clip(np.floor(xs).astype(int), 0, n - 2)
    y0 = np.clip(np.floor(ys).astype(int), 0, n - 2)
    fx, fy = np.clip(xs - x0, 0, 1), np.clip(ys - y0, 0, 1)
    rot = np.empty_like(out)
    for i, m in enumerate(out):
        rot[i] = ((1 - fy) * (1 - fx) * m[y0, x0]
                  + (1 - fy) * fx * m[y0, x0 + 1]
                  + fy * (1 - fx) * m[y0 + 1, x0]
                  + fy * fx * m[y0 + 1, x0 + 1])
    return rot


def from_npz(path, samples, keep=DEFAULT_KEEP, rotation_deg=0.0, flip=False):
    """Build a basis from an Impact Matrix `influence_matrix.npz`.

    The session file carries `mode_maps` (the mirror's eigenmodes as
    surfaces), the pupil mask and the grid size; the lightweight matrix
    store does not.

    Args:
        path: The `influence_matrix.npz` of an Impact Matrix session.
        samples: Pupil samples across the diameter, matching
            `RetrievalOptions.pupil_samples`.
        keep: How many eigenmodes to fit. See `DEFAULT_KEEP`.
        rotation_deg: Orientation correction, see `_orient`.
        flip: Orientation correction, see `_orient`.

    Returns:
        A `DMBasis`.

    Raises:
        KeyError: If the file predates `mode_maps`.
    """
    data = np.load(str(path), allow_pickle=True)
    if "mode_maps" not in data:
        raise KeyError(f"{Path(path).name} carries no mode_maps; re-export "
                       "it with the influence-matrix pipeline")
    grid_n = int(data["grid_n"])
    n_keep = int(min(int(keep), len(data["mode_maps"])))
    maps = np.stack([_resample(data["mode_maps"][i], grid_n, samples)
                     for i in range(n_keep)])
    maps = _orient(maps, rotation_deg, flip)

    ax = (np.arange(samples) - (samples - 1) / 2.0) / (samples / 2.0)
    x, y = np.meshgrid(ax, ax)
    mask = np.hypot(x, y) <= 1.0
    # Unit RMS over the pupil, piston removed. The amplitude unit is then
    # "one wave RMS of this mode", which is the same unit the Zernike path
    # reports in, so the two bases are directly comparable.
    out = np.empty_like(maps)
    for i, m in enumerate(maps):
        v = m[mask]
        out[i] = np.where(mask, (m - v.mean()) / (v.std() or 1.0), 0.0)
    return DMBasis(out, mask, rotation_deg, flip, str(path))


def orientations(step_deg=15.0):
    """The (rotation, flip) grid a calibration scan sweeps.

    One pass of this settles the two orientation unknowns; both are
    observable from the spot alone, so no pupil image is needed.

    Args:
        step_deg: Rotation step, in degrees.
    """
    n = max(1, int(round(360.0 / float(step_deg))))
    return [(i * 360.0 / n, f) for f in (False, True) for i in range(n)]
