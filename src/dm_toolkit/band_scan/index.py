# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-29

"""Pair the saved spot images of a scan folder with the DM bits behind them."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = (".png", ".tif", ".tiff")

# Filename tag the loop's snapshot writes, e.g. "spot_004_c1b2800_c2b2000.png".
_BIT_TAG = re.compile(r"c(\d+)b(\d+)", re.IGNORECASE)


@dataclass
class Point:
    """One saved spot: the image plus the mirror state it was taken in."""
    path: Path
    index: int
    bits: dict = field(default_factory=dict)  # {channel: bit}
    exposure_ms: float = float("nan")
    saturated: bool = False
    source: str = "filename"  # Where the bits came from.

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class Series:
    """A scan folder read back: its points, its drive axis, and any warnings."""
    folder: Path
    points: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    channels: tuple = ()  # Channels that MOVED across the series.

    def __len__(self):
        return len(self.points)

    @property
    def drive(self):
        """Per-point scalar drive value used as the scan's x axis.

        The band scan needs one number per image to fit against. When several
        channels moved together (the recommended defocus-like ladder) they are
        summed, which is exactly the ladder's own step variable; when one moved,
        it is that channel's bit. Constant channels are excluded so a fixed bias
        cannot swamp the axis.
        """
        out = []
        for p in self.points:
            if self.channels:
                out.append(float(sum(p.bits.get(c, 0) for c in self.channels)))
            else:
                out.append(float(p.index))
        return out

    @property
    def drive_label(self) -> str:
        if not self.channels:
            return "snapshot index"
        if len(self.channels) == 1:
            return f"ch{self.channels[0]} (bit)"
        chans = "+".join(f"ch{c}" for c in self.channels)
        return f"sum of {chans} (bit)"


def _bits_from_name(name: str) -> dict:
    """Parse the cNbM tags the snapshot button puts in a file name."""
    return {int(c): int(b) for c, b in _BIT_TAG.findall(name)}


def _index_from_name(name: str, fallback: int) -> int:
    m = re.search(r"_(\d+)", name)
    return int(m.group(1)) if m else fallback


def _read_sidecar(folder: Path) -> dict:
    """Read scan_index.csv into {file name -> row dict}, empty if absent.

    Rows keyed by file name rather than index so a folder whose images were
    renamed still matches on whatever names survive.
    """
    path = folder / "scan_index.csv"
    if not path.is_file():
        return {}
    rows = {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("file") or "").strip()
                if name:
                    rows[name] = row
    except (OSError, csv.Error):
        return {}
    return rows


def _bits_from_row(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if not key or not key.lower().startswith("ch"):
            continue
        try:
            out[int(key[2:])] = int(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _as_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_series(folder) -> Series:
    """Read a snapshot folder into a Series, sorted by snapshot index.

    The sidecar CSV wins where it covers a file, because it also carries the
    exposure and the saturation flag; the file name is the fallback so a folder
    assembled by hand still works. Both are reported per point so a mismatch is
    visible rather than silent.

    Args:
        folder: Directory holding the spot images and, ideally, scan_index.csv.

    Returns:
        A Series; empty with a warning when the folder holds no images.
    """
    folder = Path(folder)
    series = Series(folder=folder)
    if not folder.is_dir():
        series.warnings.append(f"not a folder: {folder}")
        return series

    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        kinds = "/".join(IMAGE_SUFFIXES)
        series.warnings.append(f"no {kinds} images in {folder}")
        return series

    rows = _read_sidecar(folder)
    if not rows:
        series.warnings.append(
            "no scan_index.csv: bits read from file names, exposure unknown")
    n_saturated = 0
    for i, path in enumerate(files):
        row = rows.get(path.name)
        if row:
            bits = _bits_from_row(row)
            point = Point(path=path,
                          index=int(_as_float(row.get("index"), i)),
                          bits=bits,
                          exposure_ms=_as_float(row.get("exposure_ms")),
                          saturated=bool(_as_float(row.get("saturated"), 0.0)),
                          source="scan_index.csv")
        else:
            point = Point(path=path, index=_index_from_name(path.name, i),
                          bits=_bits_from_name(path.name))
        if not point.bits:
            series.warnings.append(f"{path.name}: no DM bits found")
        n_saturated += bool(point.saturated)
        series.points.append(point)
    series.points.sort(key=lambda p: p.index)

    if n_saturated:
        series.warnings.append(
            f"{n_saturated} image(s) flagged saturated -- a clipped peak "
            "flattens the metric and will bias the fit")
    exposures = {round(p.exposure_ms, 4) for p in series.points
                 if p.exposure_ms == p.exposure_ms}
    if len(exposures) > 1:
        series.warnings.append(
            f"exposure changed across the series ({len(exposures)} values); "
            "the band metric is flux-normalised so this is survivable, but a "
            "constant exposure is the clean way to measure it")
    series.channels = _moving_channels(series.points)
    if not series.channels:
        series.warnings.append(
            "no channel changed across the series: falling back to the "
            "snapshot index as the x axis")
    return series


def _moving_channels(points) -> tuple:
    """Channels whose bit is not the same in every point, in channel order."""
    seen = {}
    for p in points:
        for c, b in p.bits.items():
            seen.setdefault(c, set()).add(b)
    return tuple(sorted(c for c, values in seen.items() if len(values) > 1))
