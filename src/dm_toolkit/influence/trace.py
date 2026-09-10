# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.2, 2026-07-31

"""Record a matrix computation the way a worked answer is written out.

Every pipeline stage appends a `Step` with its formula, inputs and outputs,
so the stage that went wrong can be seen without reading the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def fmt_num(v, width=9, sig=3):
    """One number in a fixed-width cell, switching to exponent when needed.

    The small-value threshold is where `sig` decimals would round to zero, not
    a decade below it: in between, a real number renders as 0.000 and a whole
    matrix reads as empty.
    """
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "nan".rjust(width)
    a = abs(float(v))
    if a != 0 and (a >= 10 ** (width - sig - 2) or a < 0.5 * 10 ** -sig):
        return f"{float(v):>{width}.{sig - 1}e}"
    return f"{float(v):>{width}.{sig}f}"


def preview(a, max_rows=6, max_cols=7, width=9, sig=3):
    """Abbreviated bracketed text block for an array of any size.

    Large matrices are shown as a CENTRED window with an ellipsis on each
    skipped side. Centred, not the leading corner: a pupil map is a disc in a
    square grid, so its corners are NaN by construction. 1-D input is shown as
    a single row.

    Args:
        a: Array to render.
        max_rows: Rows shown before rows start being skipped.
        max_cols: Columns shown before columns start being skipped.
        width: Character width of one numeric cell.
        sig: Significant decimals per cell.
    """
    a = np.asarray(a)
    if a.ndim == 0:
        return fmt_num(a.item(), width, sig).strip()
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim > 2:
        return f"<{a.ndim}-D array {a.shape}>"
    rows, cols = a.shape

    def pick(n, keep):
        """Centred index window, plus whether each side had to be cut."""
        if n <= keep:
            return list(range(n)), False, False
        start = (n - keep) // 2
        return list(range(start, start + keep)), start > 0, start + keep < n

    ri, r_head, r_tail = pick(rows, max_rows)
    ci, c_head, c_tail = pick(cols, max_cols)

    def framed(cells, gap):
        """Add the horizontal ellipsis cells around one row of cells."""
        return "  ".join(([gap] if c_head else []) + cells
                         + ([gap] if c_tail else []))

    ellipsis_row = framed(["...".center(width) for _ in ci], "".center(3))
    lines = []
    if r_head:
        lines.append(ellipsis_row)
    for i in ri:
        lines.append(framed([fmt_num(a[i, j], width, sig) for j in ci], " .."))
    if r_tail:
        lines.append(ellipsis_row)
    if len(lines) == 1:
        return f"[ {lines[0]} ]"
    body = [f"| {ln} |" for ln in lines]
    return "\n".join(body)


def stats_line(a, units=""):
    """One-line min / max / RMS summary, ignoring NaN."""
    a = np.asarray(a, float)
    finite = np.isfinite(a)
    if not finite.any():
        return "all NaN"
    v = a[finite]
    u = f" {units}" if units else ""
    return (f"min {fmt_num(v.min()).strip()}{u}   "
            f"max {fmt_num(v.max()).strip()}{u}   "
            f"rms {fmt_num(np.sqrt(np.mean(v ** 2))).strip()}{u}   "
            f"({finite.sum()}/{a.size} finite)")


@dataclass
class MatrixCard:
    """One named array as it is shown in a step."""
    name: str
    shape: tuple
    units: str = ""
    body: str = ""  # Abbreviated numeric block.
    stats: str = ""
    role: str = ""  # What this array MEANS, one line.
    heat: np.ndarray | None = None  # Optional 2-D map for a thumbnail.

    @property
    def shape_text(self) -> str:
        return " x ".join(str(n) for n in self.shape) or "scalar"


# How much a note is asking of the reader. A page where everything is amber
# reads as a page where nothing is: the levels exist so the one note that
# invalidates the result is not sitting in a column of remarks about defaults.
INFO = "info"  # This is how it was configured. Nothing to do.
WARN = "warn"  # The result stands, but it is worth less than it looks.
STOP = "stop"  # Do not use this matrix until it is fixed.


@dataclass
class Note:
    """One remark, and how loudly it should be made."""
    text: str
    level: str = WARN


@dataclass
class Step:
    """One stage of the computation."""
    n: int
    title: str
    formula: str = ""
    why: str = ""
    inputs: list = field(default_factory=list)  # Plain-text input descriptions.
    matrices: list = field(default_factory=list)
    numbers: list = field(default_factory=list)  # (label, value text) pairs.
    notes: list = field(default_factory=list)  # Note objects, see INFO/WARN.

    def note(self, text, level=WARN):
        """Attach one remark at the given level.

        Args:
            text: What to say.
            level: `INFO`, `WARN` or `STOP`.
        """
        self.notes.append(Note(text=text, level=level))
        return self

    def info(self, text):
        """Attach a note that only records how something was configured."""
        return self.note(text, INFO)

    def stop(self, text):
        """Attach a note that says the result must not be used."""
        return self.note(text, STOP)


class Trace:
    """Ordered record of the steps a pipeline took."""

    def __init__(self):
        self.steps = []

    def add(self, title, formula="", why="") -> Step:
        """Start a new step and return it so the caller can fill it in.

        Args:
            title: Short name of the stage.
            formula: The operation applied, written as maths.
            why: One line on why this stage exists at all.
        """
        step = Step(n=len(self.steps) + 1, title=title, formula=formula,
                    why=why)
        self.steps.append(step)
        return step

    @staticmethod
    def matrix(step, name, a, units="", role="", heat=None, **kw) -> MatrixCard:
        """Attach an array to a step, previewed and summarised.

        Args:
            step: Step to attach to.
            name: Symbol the array is called in the formulas.
            a: The array itself.
            units: Physical units of its entries.
            role: One line on what the entries mean.
            heat: Optional 2-D map to draw as a thumbnail.
            **kw: Forwarded to `preview` for per-array formatting.
        """
        a = np.asarray(a)
        card = MatrixCard(name=name, shape=a.shape, units=units,
                          body=preview(a, **kw), stats=stats_line(a, units),
                          role=role, heat=heat)
        step.matrices.append(card)
        return card

    def to_text(self) -> str:
        """The whole trace as plain text, for export next to the results."""
        out = []
        for s in self.steps:
            out.append(f"=== Step {s.n}: {s.title} ===")
            if s.formula:
                out.append(f"    {s.formula}")
            if s.why:
                out.append(f"    why: {s.why}")
            for text in s.inputs:
                out.append(f"    in : {text}")
            for m in s.matrices:
                out.append(f"    out: {m.name}  [{m.shape_text}]"
                           + (f"  {m.units}" if m.units else ""))
                if m.role:
                    out.append(f"         {m.role}")
                for line in m.body.splitlines():
                    out.append(f"         {line}")
                out.append(f"         {m.stats}")
            for label, value in s.numbers:
                out.append(f"    >>  {label}: {value}")
            for note in s.notes:
                mark = {INFO: "-", STOP: "!!"}.get(note.level, "!")
                out.append(f"    {mark:<3} {note.text}")
            out.append("")
        return "\n".join(out)
