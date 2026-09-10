# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.7, 2026-08-15

"""Store and retrieve each mirror's impact matrix as a device calibration.

One file per measurement under `matrices/<mirror>/`, with the mirror named
inside the file as well. Two arrays drive a modal solve: `ctrl`, whose
column i is the bit offset producing one radian RMS of eigenmode i, and
`s`, the singular values. The curvature in the driven coordinates is
`s[i]**2 * ||ctrl[:, i]||**2`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Beside the code, mirroring hyst_comp/devices: one sub-folder per mirror.
MATRIX_DIRECTORY = Path(__file__).with_name("matrices")
# The single file every build before v1.4 wrote and read. Still loaded and still
# listed so an existing calibration keeps working, but never written again.
DEFAULT_NAME = "impact_matrix.npz"
# Where a matrix goes when the operator has not named the mirror. A drawer with
# an unhelpful label still beats overwriting the previous measurement.
UNNAMED = "unnamed"

# Bumped only when a field changes meaning (2 added `mirror`). A newer layout
# is refused rather than half-read.
FORMAT_VERSION = 2


def default_path() -> Path:
    """The pre-v1.4 single-file location, kept only so it still loads."""
    return MATRIX_DIRECTORY / DEFAULT_NAME


def slug(text) -> str:
    """`text` reduced to what is safe in a file name, or "" if nothing is left.

    Args:
        text: Mirror name as the operator typed it.
    """
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z_-]+", "_",
                                     str(text or "").strip())).strip("_")


def matrix_path(mirror, channels, when=None) -> Path:
    """Where a newly measured matrix belongs.

    The stem carries the mirror, the actuator count and the time, so the folder
    is readable without opening anything and a second measurement of the same
    mirror sits beside the first instead of replacing it.

    Args:
        mirror: Mirror name; anything unusable falls back to `UNNAMED`.
        channels: DM channel numbers the matrix covers.
        when: Local timestamp for the name; defaults to now.

    Returns:
        The path to write, whose parent may not exist yet.
    """
    stem = slug(mirror) or UNNAMED
    when = when or datetime.now()
    return (MATRIX_DIRECTORY / stem
            / f"{stem}_{len(list(channels))}ch_{when:%Y%m%d_%H%M%S}.npz")


class MatrixMismatch(ValueError):
    """The stored matrix does not describe the actuators being driven."""


@dataclass(frozen=True)
class StoredMatrix:
    """One saved impact matrix, validated on load."""

    path: Path
    channels: tuple  # DM channel numbers, in the matrix's own column order.
    ctrl: np.ndarray  # (n_act, n_modes): bit offset per rad RMS of each mode.
    s: np.ndarray  # Singular values, descending.
    keep: int  # Modes above the measurement noise floor.
    grid_n: int
    laser_nm: float
    saved_utc: str
    source: str  # Folder the surfaces came from, for traceability.
    mirror: str = ""  # Which mirror this describes; "" in pre-v1.4 files.

    @property
    def n_modes(self) -> int:
        return int(self.ctrl.shape[1])

    @property
    def mirror_label(self) -> str:
        """The mirror name to show, never blank.

        A file from before mirrors were named cannot be attributed by anything
        in it, so it says so rather than borrowing the folder it happens to sit
        in -- being told a matrix is unattributed is what prompts re-measuring
        or renaming it.
        """
        return self.mirror or f"({UNNAMED})"

    @property
    def saved_day(self) -> str:
        return self.saved_utc.split("T")[0] if self.saved_utc else "unknown date"

    def require_wavelength(self, wavelength_nm, relative_tolerance=0.01):
        """Require the matrix to use the closed loop's AO wavelength.

        Zygo's interferometer wavelength converts its fringes to physical
        mirror height.  ``laser_nm`` is different metadata: it converts that
        height to radians in the camera optimization path.  Treating the two
        as interchangeable makes every requested probe amplitude wrong.

        Args:
            wavelength_nm: AO-path wavelength requested by the closed loop.
            relative_tolerance: Largest relative metadata disagreement allowed.

        Returns:
            This matrix, so validation composes with loading.

        Raises:
            MatrixMismatch: If either wavelength is missing or they disagree.
        """
        requested = float(wavelength_nm)
        recorded = float(self.laser_nm)
        if not (np.isfinite(requested) and requested > 0):
            raise MatrixMismatch(
                "closed-loop AO wavelength is not configured; enter the "
                "camera-path laser wavelength before starting a modal solve")
        if not (np.isfinite(recorded) and recorded > 0):
            raise MatrixMismatch(
                "impact matrix has no AO wavelength metadata; regenerate it "
                "with the influence-matrix pipeline")
        scale = max(abs(requested), abs(recorded), 1.0)
        if abs(requested - recorded) > max(0.0, relative_tolerance) * scale:
            raise MatrixMismatch(
                f"impact matrix AO wavelength is {recorded:g} nm but the DM "
                f"Loop is configured for {requested:g} nm; regenerate the "
                "matrix with the camera-path wavelength")
        return self

    @property
    def curvature(self) -> np.ndarray:
        """Per-mode curvature in the RMS-normalised coordinates we drive.

        The SVD gives Gram eigenvalues `s_i^2` for unit-length eigenvectors;
        `influence.eigenmodes` rescales each control column to one radian
        RMS, so the quadratic form picks up `||ctrl_i||^2`. This is what
        lets the N+2 solve skip a parabola per mode.

        Returns:
            One positive driven-coordinate curvature per stored mode.
        """
        ctrl = np.asarray(self.ctrl, float)
        column_norm_sq = np.sum(ctrl ** 2, axis=0)
        return np.asarray(self.s, float) ** 2 * column_norm_sq

    def align(self, channels) -> np.ndarray:
        """Return `ctrl` with its rows in the caller's channel order.

        Args:
            channels: DM channel numbers being driven, in the order the caller's
                command vectors use.

        Returns:
            An (len(channels), n_modes) array of bit offsets per rad RMS.

        Raises:
            MatrixMismatch: If the channel sets differ. Equality is required,
                not containment: an eigenmode is defined by ALL the actuators
                that were measured together, so holding one of them fixed does
                not restrict the mode, it makes it a different shape entirely.
        """
        want = [int(c) for c in channels]
        have = [int(c) for c in self.channels]
        if sorted(want) != sorted(have):
            raise MatrixMismatch(
                f"matrix covers channels {sorted(have)} but the loop drives "
                f"{sorted(want)}; re-measure the impact matrix for this set")
        index = {c: i for i, c in enumerate(have)}
        return np.asarray(self.ctrl, float)[[index[c] for c in want], :]

    def summary(self) -> str:
        """One line for a status label."""
        return (f"{self.mirror_label}: {len(self.channels)} actuators, "
                f"{self.keep} usable mode(s) of {self.n_modes}, measured "
                f"{self.saved_day}")

    def label(self) -> str:
        """One line for a picker entry, short enough for a drop-down.

        A file with no mirror name leads with its own file name. Leading with
        "(unnamed)" instead made every such entry in the list start with the
        same word, so the one thing that told them apart -- the file name --
        sat at the far end of the row where the drop-down clipped it.
        """
        head = self.mirror or self.path.stem
        tail = f"  [{self.path.name}]" if self.mirror else ""
        return (f"{head}  -  {len(self.channels)}ch, "
                f"{self.keep}/{self.n_modes} modes, {self.saved_day}{tail}")


def save_matrix(channels, ctrl, s, keep, *, grid_n=0, laser_nm=float("nan"),
                source="", mirror="", path=None, extras=None) -> Path:
    """Write one impact matrix to the calibration store.

    Args:
        channels: DM channel numbers, matching `ctrl`'s row order.
        ctrl: (n_act, n_modes) bit offset per rad RMS of each mode.
        s: Singular values of the gradient matrix, descending.
        keep: How many modes stand above the measurement noise floor.
        grid_n: Pupil sampling the matrix was computed on.
        laser_nm: Wavelength of the AO path the radians refer to.
        source: Folder the surfaces came from.
        mirror: Which mirror was measured. Recorded in the file and, when
            `path` is left out, used to place it.
        path: Destination; defaults to `matrix_path(mirror, channels)`, which
            never collides with an existing matrix.
        extras: Optional further arrays to carry along for diagnosis.

    Returns:
        The path written.
    """
    path = Path(path) if path is not None else matrix_path(mirror, channels)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        format_version=np.asarray(FORMAT_VERSION),
        channels=np.asarray([int(c) for c in channels], int),
        ctrl=np.asarray(ctrl, float),
        singular_values=np.asarray(s, float),
        keep=np.asarray(int(keep)),
        grid_n=np.asarray(int(grid_n)),
        laser_nm=np.asarray(float(laser_nm)),
        saved_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
        source=np.asarray(str(source)),
        mirror=np.asarray(str(mirror or "")),
    )
    for key, value in (extras or {}).items():
        if key not in payload and value is not None:
            payload[key] = np.asarray(value)
    # Write beside the target and replace, so an interrupted save cannot leave a
    # truncated calibration that later loads as valid-looking numbers.
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Through a file object, not a name: np.savez_compressed appends ".npz" to
    # any path that lacks it, which would silently write past the temporary name
    # and leave the replace below with nothing to rename.
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    tmp.replace(path)
    return path


_REMAP_SUFFIX = re.compile(r"(_ch\d+-\d+)?(_\d{8}_\d{6})?$")


def _remap_stem(path) -> str:
    """The original stem, with any previous remap decoration removed.

    Re-labelling an already re-labelled matrix must not stack suffixes:
    `DM_9_im3_ch6-14` remapped again is `DM_9_im3_ch1-9`, never
    `DM_9_im3_ch6-14_ch1-9`. Old timestamped names are recognised too, so a
    file written by an earlier build folds into the new scheme instead of
    spawning yet another copy.

    Args:
        path: The matrix being re-labelled.
    """
    return _REMAP_SUFFIX.sub("", Path(path).stem, count=1) or Path(path).stem


def remap_matrix(stored, new_channels, *, path=None) -> Path:
    """Re-label one matrix's actuators onto different DM channels.

    For a rewire: the same actuators moved to other header pins, so not one
    number moves, only which channel each row answers to. Only correct when
    new_channels[i] drives the very actuator stored.channels[i] drove; the
    mapping is recorded in the file it writes.

    Args:
        stored: The StoredMatrix to re-label.
        new_channels: The new channel numbers, in the same order as
            `stored.channels`.
        path: Destination; defaults to `<original stem>_ch<lo>-<hi>.npz`
            beside the original, overwritten on a repeat.

    Returns:
        The path written. The original is left alone, so the old wiring's
        calibration survives a rewire that gets undone.

    Raises:
        MatrixMismatch: If the mapping is not one channel per row, or sends
            two actuators to the same channel.
    """
    old = [int(c) for c in stored.channels]
    new = [int(c) for c in new_channels]
    if len(new) != len(old):
        raise MatrixMismatch(
            f"matrix has {len(old)} actuators but {len(new)} channel(s) were "
            f"given to re-label them with; a remap moves wires, it cannot add "
            f"or drop an actuator")
    if len(set(new)) != len(new):
        raise MatrixMismatch(
            f"channels {sorted(c for c in set(new) if new.count(c) > 1)} "
            f"appear twice; two actuators cannot share one channel")
    pairs = ", ".join(f"{o}->{n}" for o, n in zip(old, new) if o != n)
    note = f"remapped {pairs or 'nothing'} from {stored.path.name}"
    if path is None:
        path = stored.path.parent / f"{_remap_stem(stored.path)}_ch{min(new)}-{max(new)}.npz"
    return save_matrix(
        new, stored.ctrl, stored.s, stored.keep,
        grid_n=stored.grid_n, laser_nm=stored.laser_nm,
        # Kept visible in the picker's tooltip: a re-labelled matrix must
        # never be mistaken for a freshly measured one.
        source=f"{stored.source} [{note}]" if stored.source else note,
        mirror=stored.mirror, path=path,
        extras=dict(remapped_from=str(stored.path),
                    remapped_old_channels=np.asarray(old, int),
                    remapped_new_channels=np.asarray(new, int)))


def load_matrix(path=None) -> StoredMatrix | None:
    """Read the stored impact matrix.

    Args:
        path: File to read; defaults to `default_path()`.

    Returns:
        The matrix, or None when no file is present -- a missing calibration is
        the normal state before the first measurement, not an error.

    Raises:
        MatrixMismatch: If the file exists but cannot be trusted: a newer format
            version, or arrays whose shapes contradict each other.
    """
    path = Path(path) if path is not None else default_path()
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        version = int(z["format_version"]) if "format_version" in z else 0
        if version > FORMAT_VERSION:
            raise MatrixMismatch(
                f"{path.name} was written in format {version}, but this build "
                f"reads up to {FORMAT_VERSION}")
        channels = tuple(int(c) for c in z["channels"])
        ctrl = np.asarray(z["ctrl"], float)
        s = np.asarray(z["singular_values"], float)
        keep = int(z["keep"]) if "keep" in z else int(len(s))
        grid_n = int(z["grid_n"]) if "grid_n" in z else 0
        laser_nm = float(z["laser_nm"]) if "laser_nm" in z else float("nan")
        saved = str(z["saved_utc"]) if "saved_utc" in z else ""
        source = str(z["source"]) if "source" in z else ""
        mirror = str(z["mirror"]) if "mirror" in z else ""
    if ctrl.ndim != 2 or ctrl.shape[0] != len(channels):
        raise MatrixMismatch(
            f"{path.name}: ctrl is {ctrl.shape} but there are {len(channels)} "
            "channels")
    if len(s) != ctrl.shape[1]:
        raise MatrixMismatch(
            f"{path.name}: {len(s)} singular values for {ctrl.shape[1]} modes")
    return StoredMatrix(path=path, channels=channels, ctrl=ctrl, s=s,
                        keep=max(0, min(keep, ctrl.shape[1])), grid_n=grid_n,
                        laser_nm=laser_nm, saved_utc=saved, source=source,
                        mirror=mirror)


def list_matrices(directory=None, n_channels=0) -> list:
    """Every readable matrix in the store, for a picker.

    Args:
        directory: Root to search, sub-folders included; defaults to
            `MATRIX_DIRECTORY`.
        n_channels: When positive, keep only matrices covering exactly this many
            actuators. A five-element mirror cannot use a nine-element mirror's
            eigenmodes, so offering them is offering a mistake.

    Returns:
        Matrices grouped by mirror name, newest first within each. Files that do
        not load are skipped rather than raising: one unreadable stray must not
        empty the picker, and `describe` explains any single file on demand.
    """
    directory = Path(directory) if directory is not None else MATRIX_DIRECTORY
    if not directory.is_dir():
        return []
    found = []
    for path in directory.rglob("*.npz"):
        try:
            stored = load_matrix(path)
        except (MatrixMismatch, OSError, ValueError, KeyError):
            continue
        if stored is None:
            continue
        if n_channels and len(stored.channels) != int(n_channels):
            continue
        found.append(stored)
    # Two stable passes: newest first, then gathered by mirror.
    found.sort(key=lambda m: (m.saved_utc, m.path.name), reverse=True)
    found.sort(key=lambda m: m.mirror_label.lower())
    return found


def describe(path=None) -> str:
    """Human-readable state of one stored matrix, for a UI status line."""
    path = Path(path) if path is not None else default_path()
    try:
        stored = load_matrix(path)
    except MatrixMismatch as exc:
        return f"unusable: {exc}"
    if stored is None:
        return f"no matrix at {path.name}"
    return stored.summary()
