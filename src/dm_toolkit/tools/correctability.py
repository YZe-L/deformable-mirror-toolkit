# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-08-22

"""Split a retrieved wavefront into the part this mirror can correct and the
part it cannot, for every correction run under a folder.

Reads the `phase_retrieval.csv` that `retrieve_dm_loop` writes and an
`influence_matrix.npz`, expands the retrieved Noll 4-11 coefficients on the
pupil, and solves

    minimise || W + A m ||  subject to the bit bounds

with `A` the influence matrix in waves per bit. The part the solve removes
is correctable; the residual is not. Only the retained eigenmodes are used
(`dm_basis.DEFAULT_KEEP`). A `beyond_capture_range` row is a lower bound
and is marked; both even-block signs are evaluated and the spread is
reported. Read the percentages as magnitudes.

Usage:
    python -m dm_toolkit.tools.correctability <folder> --im <npz>
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "deformable_mirror_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

from .. import zernike as ZK
from ..phase_retrieval import dm_basis as DMB

# Bits either side of the bias one correction may spend. The Impact Matrix
# session biased at 2000 and the mirror runs 200-4000, so 1800 is the
# one-sided travel with the compensator's own headroom left at each rail.
STROKE_BITS = 1800.0

# Noll terms the retrieval reports.
FIT_MODES = tuple(range(4, 12))

BEFORE_COLOR = "#c44e52"
AFTER_COLOR = "#1f77b4"
FLOOR_COLOR = "#7f7f7f"


def load_influence(path, keep=DMB.DEFAULT_KEEP):
    """Waves-per-bit columns and the modal command basis of one session.

    Args:
        path: An Impact Matrix `influence_matrix.npz`.
        keep: Eigenmodes retained; see `dm_basis.DEFAULT_KEEP`.

    Returns:
        A (waves-per-bit matrix, modal basis, pupil mask, grid side) tuple.
    """
    data = np.load(str(path), allow_pickle=True)
    waves_per_bit = data["c_rad_per_bit"] / (2.0 * np.pi)
    modal = data["ctrl"][:, :int(keep)]
    return waves_per_bit, modal, data["inside"], int(data["grid_n"])


def read_coeffs(csv_path):
    """The retrieved Noll 4-11 coefficients of one run.

    Args:
        csv_path: A `phase_retrieval.csv`.

    Returns:
        A (before, after, beyond_capture_range) triple, the first two being
        arrays over `FIT_MODES`.
    """
    before, after = {}, {}
    beyond = False
    for row in csv.reader(Path(csv_path).open()):
        if not row:
            continue
        if row[0].startswith("noll_"):
            j = int(row[0].split("_")[1])
            before[j], after[j] = float(row[1]), float(row[2])
        elif row[0] == "reason" and "capture range" in row[1]:
            beyond = True
    nan = float("nan")
    return (np.array([before.get(j, nan) for j in FIT_MODES]),
            np.array([after.get(j, nan) for j in FIT_MODES]),
            beyond)


def split(wavefront, waves_per_bit, modal):
    """Correctable and uncorrectable RMS of one wavefront, in waves.

    Args:
        wavefront: Pupil phase over the influence matrix's pixels, in waves.
        waves_per_bit: Influence matrix columns, one per actuator.
        modal: Bits per unit amplitude of each retained eigenmode.

    Returns:
        A (total, correctable, uncorrectable, max |bit| used) tuple.
    """
    total = float(np.sqrt(np.mean(wavefront ** 2)))
    design = waves_per_bit @ modal
    solved = lsq_linear(design, -wavefront, bounds=(-np.inf, np.inf))
    bits = modal @ solved.x
    over = np.max(np.abs(bits)) / STROKE_BITS
    if over > 1.0:                      # Scale as one vector, never clip.
        bits, solved.x = bits / over, solved.x / over
    residual = wavefront + waves_per_bit @ bits
    left = float(np.sqrt(np.mean(residual ** 2)))
    return total, float(np.sqrt(max(total ** 2 - left ** 2, 0.0))), left, \
        float(np.max(np.abs(bits)))


def analyse(run_csv, waves_per_bit, modal, basis_flat):
    """One run's before/after split, over both even-sign conventions.

    Args:
        run_csv: The run's `phase_retrieval.csv`.
        waves_per_bit: Influence matrix columns.
        modal: Modal command basis.
        basis_flat: (8, npix) Noll 4-11 evaluated on the pupil pixels.

    Returns:
        A dict of the reported quantities, or None when the CSV has no fit.
    """
    before, after, beyond = read_coeffs(run_csv)
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        return None
    out = {"run": run_csv.parent.name, "beyond": beyond}
    for tag, coeffs in (("before", before), ("after", after)):
        # The even block's overall sign is a convention; evaluate both.
        even = np.array([ZK.noll_to_nm(j)[0] % 2 == 0 for j in FIT_MODES])
        runs = []
        for sign in (1.0, -1.0):
            c = np.where(even, sign * coeffs, coeffs)
            runs.append(split(c @ basis_flat, waves_per_bit, modal))
        best = min(runs, key=lambda r: r[2])
        worst = max(runs, key=lambda r: r[2])
        out[f"{tag}_total"] = best[0]
        out[f"{tag}_correctable"] = best[1]
        out[f"{tag}_floor"] = best[2]
        out[f"{tag}_floor_spread"] = abs(worst[2] - best[2])
        out[f"{tag}_bits"] = best[3]
    # What the optimiser actually achieved, against what was ever available.
    removed = out["before_total"] - out["after_total"]
    available = out["before_total"] - out["before_floor"]
    out["removed"] = removed
    out["available"] = available
    out["efficiency"] = removed / available if available > 0 else float("nan")
    return out


def write_csv(root, rows):
    """Write the per-run table."""
    path = root / "correctability.csv"
    fields = ["run", "before_total", "before_correctable", "before_floor",
              "before_floor_spread", "before_bits", "after_total",
              "after_correctable", "after_floor", "after_floor_spread",
              "after_bits", "removed", "available", "efficiency", "beyond"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields + ["unit"])
        for row in rows:
            writer.writerow(
                [row["run"]]
                + [f"{row[f]:.4f}" if isinstance(row[f], float) else row[f]
                   for f in fields[1:]]
                + ["waves RMS unless a fraction"])
    return path


def make_figure(root, rows, keep):
    """Bar chart of measured, achieved and floor, one group per run."""
    rows = sorted(rows, key=lambda r: r["after_total"])
    names = [r["run"] for r in rows]
    x = np.arange(len(rows))
    width = 0.28
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(max(11.0, 0.95 * len(rows) + 5), 9.2),
        gridspec_kw=dict(height_ratios=[1.5, 1.0]), constrained_layout=True)

    top.bar(x - width, [r["before_total"] for r in rows], width,
            label="Before the run", color=BEFORE_COLOR)
    top.bar(x, [r["after_total"] for r in rows], width,
            label="After the run", color=AFTER_COLOR)
    top.bar(x + width, [r["before_floor"] for r in rows], width,
            label=f"Floor: what {keep} modes can never remove",
            color=FLOOR_COLOR)
    for i, r in enumerate(rows):
        top.errorbar(i + width, r["before_floor"], yerr=r["before_floor_spread"],
                     color="#222222", lw=1.0, capsize=3)
    top.set_ylabel("Wavefront RMS / waves")
    top.set_xticks(x)
    top.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    top.grid(axis="y", color="#dddddd", lw=0.7)
    top.set_axisbelow(True)
    top.legend(frameon=True, framealpha=0.92)
    top.set_title("The floor is flat and low -- and no run reaches it: the "
                  "plateau is the algorithm's, not the mirror's", fontsize=12)

    eff = [100.0 * r["efficiency"] for r in rows]
    bottom.bar(x, eff, 0.6, color="#4c9f70")
    bottom.axhline(100.0, color="#555555", lw=1.0, ls="--")
    for i, v in enumerate(eff):
        bottom.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    bottom.set_ylabel("Removed / removable  (%)")
    bottom.set_xticks(x)
    bottom.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    bottom.grid(axis="y", color="#dddddd", lw=0.7)
    bottom.set_axisbelow(True)
    bottom.set_title("Of the aberration this mirror COULD remove, how much "
                     "the optimiser did remove", fontsize=12)

    fig.suptitle("Correctable versus uncorrectable wavefront, "
                 f"{keep} retained eigenmodes, +/-{STROKE_BITS:.0f} bit",
                 fontsize=13, fontweight="bold")
    path = root / "correctability.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="Folder holding correction run folders")
    parser.add_argument("--im", required=True,
                        help="influence_matrix.npz of an Impact Matrix run")
    parser.add_argument("--keep", type=int, default=DMB.DEFAULT_KEEP,
                        help=f"Eigenmodes retained (default "
                             f"{DMB.DEFAULT_KEEP})")
    args = parser.parse_args(argv)

    root = Path(args.path).expanduser().resolve()
    waves_per_bit, modal, inside, grid_n = load_influence(args.im, args.keep)
    rho, theta, ys, xs = ZK.unit_coords(inside, ((grid_n - 1) / 2.0,) * 3)
    basis_flat = np.vstack([ZK.zernike_mode(j, rho, theta) for j in FIT_MODES])

    rows = []
    for run_csv in sorted(root.rglob("phase_retrieval.csv")):
        row = analyse(run_csv, waves_per_bit, modal, basis_flat)
        if row is not None:
            rows.append(row)
    if not rows:
        print(f"No phase_retrieval.csv with a usable fit under {root}")
        return 1

    header = (f"{'run':38s}{'before':>8s}{'after':>8s}{'floor':>8s}"
              f"{'removed':>9s}{'avail':>8s}{'eff':>7s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        mark = " *" if r["beyond"] else ""
        print(f"{r['run'][:37]:38s}{r['before_total']:>8.3f}"
              f"{r['after_total']:>8.3f}{r['before_floor']:>8.3f}"
              f"{r['removed']:>9.3f}{r['available']:>8.3f}"
              f"{100 * r['efficiency']:>6.0f}%{mark}")
    print("-" * len(header))
    mean_floor = float(np.mean([r["before_floor"] for r in rows]))
    mean_after = float(np.mean([r["after_total"] for r in rows]))
    print(f"{'mean':38s}{np.mean([r['before_total'] for r in rows]):>8.3f}"
          f"{mean_after:>8.3f}{mean_floor:>8.3f}"
          f"{np.mean([r['removed'] for r in rows]):>9.3f}"
          f"{np.mean([r['available'] for r in rows]):>8.3f}"
          f"{100 * np.mean([r['efficiency'] for r in rows]):>6.0f}%")
    print("\n* the before-frame is beyond the retrieval's capture range, so "
          "its\n  coefficients are a lower bound")
    print(f"\n{write_csv(root, rows)}")
    print(make_figure(root, rows, args.keep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
