# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-31

"""Write an exported .npz out again as one plain-text twin.

Values use 17 significant digits, which round-trips float64.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PRECISION = 17  # Significant digits; 17 round-trips float64 exactly.
_WIDTH = 100000  # Effectively no wrapping: one source row stays one text row.


def _block(a) -> str:
    """One array as the text the `value:` line introduces.

    Scalars and vectors go through `repr`, which prints the shortest string
    that reads back as the same double. Matrices go through numpy, whose
    column alignment is what makes them readable at all.
    """
    if a.ndim <= 1:
        return repr(a.tolist())
    return np.array2string(a, separator=", ", max_line_width=_WIDTH,
                           precision=PRECISION, floatmode="unique",
                           threshold=a.size + 1)


def render(payload, source) -> str:
    """The whole text file for one `.npz`.

    Args:
        payload: Mapping of name to array; an open `.npz` works directly, and
            its own insertion order is kept so the text reads in the order the
            arrays were written.
        source: Path of the `.npz` this describes, recorded in the header.

    Returns:
        The file contents.
    """
    names = list(payload.files if hasattr(payload, "files") else payload)
    out = ["# NPZ text export",
           f"# source: {source}",
           f"# arrays: {len(names)}",
           f"# Float precision: {PRECISION} significant digits "
           "(round-trip safe)", ""]
    for name in names:
        a = np.asarray(payload[name])
        out += [f"## {name}", f"shape: {tuple(a.shape)}", f"dtype: {a.dtype}",
                "value:", _block(a), ""]
    return "\n".join(out) + "\n"


def beside(npz_path) -> Path:
    """Write the twin `.txt` next to an existing `.npz` and return its path."""
    npz_path = Path(npz_path)
    path = npz_path.with_suffix(".txt")
    with np.load(npz_path, allow_pickle=False) as z:
        path.write_text(render(z, npz_path), encoding="utf-8")
    return path
