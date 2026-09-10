# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.1, 2026-07-30

"""Load a measured mirror surface from what Mx exports, into nanometres.

`.datx` is the HDF5 file Mx writes; wavelength, scale factor, lateral
resolution and validity mask are inside it. `.xyz` is the "Zygo XYZ Data
File - Format 1" text export: fourteen header lines, then one `col row
value` line per pixel with the literal `No Data` where fringes were
unreadable. The unit of the value column is a setting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUFFIXES = (".datx", ".xyz")

# Value-column units an .xyz may be in. Waves is what Mx exports by default;
# the multiplier for it depends on the wavelength, hence the None.
UNIT_WAVES = "waves"
UNIT_NM = "nm"
UNIT_UM = "um"
UNIT_M = "m"
UNIT_NM_PER = {UNIT_WAVES: None, UNIT_NM: 1.0, UNIT_UM: 1e3, UNIT_M: 1e9}

# Channel/bit tag in every spelling that appears in saved folders ("c1b2800",
# "ch1_2800", "channel1_set_2000", "ch 2 2000"). A lookbehind rather than
# \b, because \b does not fire between "_" and "c".
_TAG_CB = re.compile(
    r"(?<![0-9A-Za-z])c(?:hannel|h)?[\s_-]*(\d+)[\s_-]*"
    r"(?:bit|b|set)?[\s_-]*(\d+)(?![0-9])",
    re.IGNORECASE)

_XYZ_MAGIC = "Zygo XYZ Data File"
_XYZ_MAX_HEADER = 60  # Give up looking for the '#' separator beyond this.


@dataclass
class SurfaceMap:
    """One measured surface: heights in nm on a pixel grid, plus its mask."""
    path: Path
    z_nm: np.ndarray  # NaN outside the valid area.
    mask: np.ndarray  # Bool, True where measured.
    lateral_um: float = 0.0  # Micrometres per pixel, 0.0 if unknown.
    wavelength_nm: float = 0.0  # 0.0 if the file did not say.
    bits: dict = field(default_factory=dict)  # {channel: bit} parsed from name.
    source: str = ""
    note: str = ""  # How it was scaled, for the trace and the UI.
    # Whether Mx had already subtracted its system error (the substrate) when
    # the file was written. None for a format that does not record it.
    sys_err_subtracted: bool | None = None

    @property
    def shape(self):
        return self.z_nm.shape

    @property
    def valid_fraction(self) -> float:
        n = self.mask.size
        return float(self.mask.sum()) / n if n else 0.0


def bits_from_name(name: str) -> dict:
    """Parse {channel: bit} out of a file or folder name, empty if none.

    Accepts the compact tag the DM-Zygo scan writes (`c1b2800`) and the spaced
    form seen in hand-named folders (`ch 1 set 2000`, `channel1_set_2000`).
    """
    return {int(c): int(b) for c, b in _TAG_CB.findall(name)}


def bits_for_path(path, levels=2) -> dict:
    """Channel state for a file, searching its name then its parent folders.

    The standalone Mx grabber writes `<setpoint name>/raw/surface.xyz`, so the
    state can sit two directories up from a file whose own name says nothing.
    The nearest name that yields anything wins.

    Args:
        path: Surface file path.
        levels: How many parent directories to try after the file name.
    """
    p = Path(path)
    for name in [p.stem] + [q.name for q in list(p.parents)[:levels]]:
        bits = bits_from_name(name)
        if bits:
            return bits
    return {}


def read_datx(path, wavelength_nm=0.0) -> SurfaceMap:
    """Read a Zygo .datx as a height map in nanometres.

    Height in waves is fringes times the interferometric scale factor, and
    waves become nanometres through the wavelength. A file without a
    wavelength falls back to the argument rather than guessing.

    Args:
        path: Path to the .datx file.
        wavelength_nm: Wavelength to use when the file does not carry one.

    Raises:
        ValueError: If no wavelength is available from either source.
    """
    from ..zygo.io_datx import read_datx as _read
    meas = _read(str(path))
    wl = float(meas.wavelength_nm or wavelength_nm or 0.0)
    if wl <= 0:
        raise ValueError(f"{Path(path).name}: no wavelength in the file and "
                         "none supplied, cannot convert waves to nm")
    z = meas.waves * wl  # Waves of surface height -> nm of surface height.
    return SurfaceMap(path=Path(path), z_nm=z, mask=np.asarray(meas.mask, bool),
                      lateral_um=float(meas.lateral_res_m or 0.0) * 1e6,
                      wavelength_nm=wl,
                      bits=bits_for_path(path),
                      source="datx",
                      note=f"fringes x {meas.scale:g} x {wl:g} nm/wave",
                      sys_err_subtracted=(
                          None if "sys_err_subtracted" not in meas.attrs
                          else bool(meas.attrs["sys_err_subtracted"])))


def xyz_header(path):
    """Read the Zygo XYZ header block without touching the data.

    Stops at the `#` separator, so it reads only a few hundred bytes.

    Args:
        path: Path to the .xyz file.

    Returns:
        A dict with `lines` (the raw header), `n_header` (lines to skip),
        `col0`, `row0`, `n_cols`, `n_rows`, `wavelength_nm` and `scale`; sizes
        are 0 and the wavelength 0.0 for anything the header did not state. A
        file with no header at all (a plain three-column export) reports
        `n_header` 0 and leaves the rest for the data block to supply.
    """
    lines = []
    header = False
    with open(path, "r", encoding="latin-1", errors="replace") as fh:
        for i, raw in enumerate(fh):
            text = raw.strip()
            if text == "#":
                header = True
                break
            lines.append(text)
            if i > _XYZ_MAX_HEADER:
                break
    out = {"lines": lines if header else [],
           "n_header": len(lines) + 1 if header else 0,
           "col0": 0, "row0": 0, "n_cols": 0, "n_rows": 0,
           "wavelength_nm": 0.0, "scale": 0.0}
    if not header:  # Plain x, y, z text: nothing to read but the data.
        return out
    for text in lines:
        nums = _numbers(text)
        # The size line is the sub-array origin in camera coordinates followed
        # by its extent; the data rows use those same coordinates.
        if (not out["n_cols"] and len(nums) == 4
                and all(float(v).is_integer() for v in nums)
                and nums[2] > 1 and nums[3] > 1):
            out["col0"], out["row0"] = int(nums[0]), int(nums[1])
            out["n_cols"], out["n_rows"] = int(nums[2]), int(nums[3])
        # The wavelength is the only field written in metres of visible light.
        for v in nums:
            if 1e-7 < v < 2e-6 and not out["wavelength_nm"]:
                out["wavelength_nm"] = v * 1e9
                # The interferometric scale factor sits just before it.
                idx = nums.index(v)
                if idx > 0 and 0.0 < nums[idx - 1] <= 1.0:
                    out["scale"] = nums[idx - 1]
    return out


def _numbers(text):
    """Every plain number in a header line, quoted strings ignored."""
    out = []
    for token in re.sub(r'"[^"]*"', " ", text).split():
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def _xyz_columns(path, n_header):
    """Read the col/row/value columns of the data block.

    Uses the pandas C parser when available -- a 1200 x 1200 export is 1.44
    million lines, and a pure-Python loop over that takes seconds per file with
    fifty-odd files to read. Four column names are declared so the `No Data`
    token, which splits into two fields, does not change the apparent row width
    partway through the file.

    Args:
        path: Path to the .xyz file.
        n_header: Lines to skip before the data block.
    """
    names = ["col", "row", "value", "_extra"]
    try:
        import pandas as pd
        frame = pd.read_csv(path, sep=r"\s+", skiprows=n_header, header=None,
                            names=names, engine="c", comment="#",
                            dtype=str, na_filter=False)
        col = pd.to_numeric(frame["col"], errors="coerce").to_numpy()
        row = pd.to_numeric(frame["row"], errors="coerce").to_numpy()
        val = pd.to_numeric(frame["value"], errors="coerce").to_numpy()
        return col, row, val
    except ImportError:
        pass
    cols, rows, vals = [], [], []
    with open(path, "r", encoding="latin-1", errors="replace") as fh:
        for _ in range(n_header):
            fh.readline()
        for raw in fh:
            parts = raw.split()
            if len(parts) < 3 or parts[0] == "#":
                continue
            try:
                c, r = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            try:
                v = float(parts[2])
            except ValueError:
                v = np.nan  # "No Data" and anything else unparseable.
            cols.append(c)
            rows.append(r)
            vals.append(v)
    return (np.asarray(cols, np.int64), np.asarray(rows, np.int64),
            np.asarray(vals, float))


def read_xyz(path, unit=UNIT_UM, wavelength_nm=0.0,
             lateral_um=0.0) -> SurfaceMap:
    """Read a Zygo XYZ text export as a height map in nanometres.

    The grid is built from the col/row indices rather than from the line order,
    so a file with gaps or an unexpected raster direction still lands correctly.

    Args:
        path: Path to the .xyz file.
        unit: Unit of the value column, one of the `UNIT_*` constants.
        wavelength_nm: Wavelength for `UNIT_WAVES` when the header lacks one.
        lateral_um: Micrometres per grid step, 0.0 if unknown.

    Raises:
        ValueError: If the header is unreadable, the data block is empty, or
            waves are requested with no wavelength available.
    """
    head = xyz_header(path)
    col, row, val = _xyz_columns(path, head["n_header"])
    ok = np.isfinite(col) & np.isfinite(row)
    if not ok.any():
        raise ValueError(f"{Path(path).name}: no data rows after the header")
    col, row, val = col[ok].astype(np.int64), row[ok].astype(np.int64), val[ok]
    # Camera coordinates -> array indices. The header origin is authoritative;
    # without one, the smallest index present is the only sane assumption.
    col0 = head["col0"] if head["n_cols"] else int(col.min())
    row0 = head["row0"] if head["n_rows"] else int(row.min())
    col, row = col - col0, row - row0
    n_cols = head["n_cols"] or int(col.max()) + 1
    n_rows = head["n_rows"] or int(row.max()) + 1
    inside = (col >= 0) & (col < n_cols) & (row >= 0) & (row < n_rows)
    if not inside.all():
        lost = int((~inside).sum())
        raise ValueError(f"{Path(path).name}: {lost} of {inside.size} rows "
                         f"fall outside the {n_cols} x {n_rows} grid its "
                         "header declares -- file and header disagree")
    grid = np.full((n_rows, n_cols), np.nan)
    grid[row[inside], col[inside]] = val[inside]

    wl = float(head["wavelength_nm"] or wavelength_nm or 0.0)
    per_nm = UNIT_NM_PER.get(unit)
    if per_nm is None:  # Waves: the scale is the wavelength itself.
        if wl <= 0:
            raise ValueError(f"{Path(path).name}: the value column is in waves "
                             "but no wavelength is available")
        per_nm = wl
    return SurfaceMap(path=Path(path), z_nm=grid * per_nm,
                      mask=np.isfinite(grid),
                      lateral_um=float(lateral_um),
                      wavelength_nm=wl,
                      bits=bits_for_path(path),
                      source="xyz",
                      note=f"{unit} x {per_nm:g} nm per unit")


def guess_xyz_unit(path):
    """Guess the value column's unit from its magnitude, for a UI default.

    Waves and micrometres land in the same decade for a mirror a few hundred
    nanometres deep, so the tie-break is micrometres, which Mx writes;
    `xyz_scale_against_datx` settles it when a .datx is at hand.

    Args:
        path: Path to the .xyz file.

    Returns:
        (unit, certain) -- one of the `UNIT_*` constants and whether the
        magnitude alone determined it. `(None, False)` if nothing was sampled.
    """
    head = xyz_header(path)
    values = []
    with open(path, "r", encoding="latin-1", errors="replace") as fh:
        for _ in range(head["n_header"]):
            fh.readline()
        for raw in fh:
            parts = raw.split()
            if len(parts) < 3:
                continue
            try:
                values.append(abs(float(parts[2])))
            except ValueError:
                continue
            if len(values) >= 20000:
                break
    values = [v for v in values if v > 0]
    if not values:
        return None, False
    scale = float(np.percentile(values, 99))
    if scale < 1e-4:
        return UNIT_M, True
    if scale > 20.0:
        return UNIT_NM, True
    return UNIT_UM, False  # Could equally be waves; see the docstring.


def xyz_scale_against_datx(xyz_path, datx_path, wavelength_nm=0.0):
    """Nanometres per .xyz value unit, measured against a matching .datx.

    The .datx is self-describing, so fitting one against the other gives
    the .xyz scale outright. Only meaningful for two saves of the same
    measurement.

    Args:
        xyz_path: The .xyz export.
        datx_path: The .datx of the same measurement.
        wavelength_nm: Fallback wavelength for the .datx.

    Returns:
        (nm_per_unit, correlation) over the pixels valid in both. A correlation
        below about 0.999 means they are not the same measurement, and the scale
        should not be believed.

    Raises:
        ValueError: If the two files have no valid pixels in common.
    """
    dat = read_datx(datx_path, wavelength_nm)
    raw = read_xyz(xyz_path, UNIT_NM, wavelength_nm)  # Unscaled: 1 nm per unit.
    both = dat.mask & raw.mask
    if dat.shape != raw.shape or not both.any():
        raise ValueError("the two files share no valid pixels")
    a, b = raw.z_nm[both], dat.z_nm[both]
    denom = float(np.dot(a, a))
    if denom <= 0:
        raise ValueError("the .xyz values are all zero")
    scale = float(np.dot(a, b) / denom)
    corr = float(np.corrcoef(a, b)[0, 1])
    return scale, corr


def read_surface(path, wavelength_nm=0.0, xyz_unit=UNIT_UM,
                 xyz_lateral_um=0.0) -> SurfaceMap:
    """Read any supported surface file into nanometres.

    Args:
        path: Path to a .datx or .xyz file.
        wavelength_nm: Fallback wavelength when the file lacks one.
        xyz_unit: Unit of an .xyz value column, one of the `UNIT_*` constants.
        xyz_lateral_um: Micrometres per grid step, .xyz only.

    Raises:
        ValueError: If the suffix is not supported.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".datx":
        return read_datx(path, wavelength_nm)
    if suffix == ".xyz":
        return read_xyz(path, xyz_unit, wavelength_nm, xyz_lateral_um)
    raise ValueError(f"unsupported surface file: {path}")


def datx_kind(path):
    """Classify a .datx as a height map, a derived tool output, or unusable.

    Mx writes a .datx for every tool panel (MTF, PSF, PSD, PVr), all with a
    `Data/Surface` group. Two signals tell them apart: the height unit
    (`NoUnits` for MTF, PSF and diffraction maps) and the file name
    (`Surface__<Tool>__<Panel>__<Output>` for tool exports).

    Args:
        path: Path to the .datx file.

    Returns:
        ("height", note) for a usable mirror surface, ("derived", note) for a
        tool output, or ("bad", reason) for a file with no readable height data.
    """
    p = Path(path)
    try:
        import h5py
        with h5py.File(str(p), "r") as f:
            if "Data/Surface" not in f:
                return "bad", "no Data/Surface group"
            group = f["Data/Surface"]
            sets = [group[k] for k in group
                    if isinstance(group[k], h5py.Dataset)]
            if not sets:
                return "bad", "empty Data/Surface group"
            unit = sets[0].attrs.get("Unit")
            unit = unit.decode() if isinstance(unit, bytes) else unit
            unit = str(np.asarray(unit).ravel()[0]) if unit is not None else ""
            shape = sets[0].shape
    except Exception as e:  # noqa: BLE001 -- an unreadable file is just "bad".
        return "bad", f"{type(e).__name__}: {e}"
    if "fringe" not in unit.lower():
        return "derived", f"unit is {unit or 'unset'}, not a height map"
    if p.stem.count("__") >= 2:
        tool = p.stem.split("__")[1] if "__" in p.stem else "?"
        return "derived", f"{tool} tool output, not the measurement"
    return "height", f"{shape[1]} x {shape[0]} fringes"


def describe(path):
    """One-line summary of a surface file, without reading its data block.

    Lets the file table say what each entry is before anything heavy runs, which
    matters because these exports are tens of megabytes each.

    Args:
        path: Path to a .datx or .xyz file.
    """
    p = Path(path)
    if p.suffix.lower() == ".xyz":
        try:
            head = xyz_header(p)
        except (OSError, ValueError) as e:
            return f"unreadable: {e}"
        magic = "Zygo XYZ" if any(_XYZ_MAGIC in ln for ln in head["lines"]) \
            else "XYZ-like"
        wl = head["wavelength_nm"]
        return (f"{magic}  {head['n_cols']} x {head['n_rows']}"
                + (f"  {wl:.1f} nm" if wl else "  wavelength unknown"))
    kind, note = datx_kind(p)
    return f"DATX ({kind}): {note}"


def find_surfaces(folder):
    """Usable surface files in a folder and its immediate subfolders.

    Subfolders are included one level down because the scans write one
    folder per setpoint. When a usable .datx and .xyz share a stem, only the
    .datx is returned.

    Args:
        folder: Directory to search.

    Returns:
        (usable, skipped): the height maps, and (path, reason) for every
        .datx that turned out to be a tool output or unreadable.
    """
    root = Path(folder)
    if not root.is_dir():
        return [], []
    found = [p for p in root.iterdir()
             if p.is_file() and p.suffix.lower() in SUFFIXES]
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        found += [p for p in sorted(sub.iterdir())
                  if p.is_file() and p.suffix.lower() in SUFFIXES]
    candidates, skipped = [], []
    for p in sorted(found, key=lambda q: (q.parent.name, q.name)):
        if p.suffix.lower() == ".datx":
            kind, note = datx_kind(p)
            if kind != "height":
                skipped.append((p, note))
                continue
        candidates.append(p)
    datx_keys = {
        (str(p.parent).casefold(), p.stem.casefold())
        for p in candidates
        if p.suffix.lower() == ".datx"
    }
    keep = [
        p
        for p in candidates
        if not (
            p.suffix.lower() == ".xyz"
            and (str(p.parent).casefold(), p.stem.casefold()) in datx_keys
        )
    ]
    return keep, skipped
