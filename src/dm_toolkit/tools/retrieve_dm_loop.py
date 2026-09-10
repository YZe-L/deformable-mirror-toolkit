# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-08-21

"""Retrieve the wavefront of a correction run's before and after spots.

Reads the `spot_before_avg.npy` / `spot_after_avg.npy` pair a run
saves, takes the optical constants from that run's own `run_config.json`,
fits Zernike coefficients to each spot, and writes a comparison figure and a
CSV beside them.

Usage:
    python -m dm_toolkit.tools.retrieve_dm_loop <path> [--roi 192]

`<path>` may be one run folder, or any folder above several: every folder
holding a `spot_before_avg.npy` below it is processed.

One sign of the even block (Z4, Z5, Z6, Z11) is unobservable, so the
magnitude panel shows magnitudes only and the signed panel marks the even
block. See `phase_retrieval.estimate` for the full ambiguity list.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import zernike as ZK
from ..phase_retrieval import (Optics, RetrievalOptions,
                                                  estimate_wavefront,
                                                  find_spot, fitted_crop,
                                                  render_model)
from ..phase_retrieval import dm_basis as DMB
from ..phase_retrieval.estimate import (
    CAPTURE_RANGE_WAVES, RADIAL_DEGENERACY_WAVES)

# The bars, in the order an optics report lists them.
TERMS = (("defocus", "Defocus"), ("astigmatism", "Astigmatism"),
         ("coma", "Coma"), ("trefoil", "Trefoil"),
         ("spherical", "Spherical"))
# Defocus and spherical are m = 0 and have no cos/sin partner, so the
# magnitude panel takes |.|; the signed panel keeps the relative signs.
MAGNITUDE_ONLY = ("defocus", "spherical")
BEFORE_COLOR = "#c44e52"
AFTER_COLOR = "#1f77b4"


def _is_even(j):
    """Whether Noll mode j is even under r -> -r, i.e. of even radial order."""
    return ZK.noll_to_nm(j)[0] % 2 == 0


def _even_anchor(est):
    """The Noll index the even-sign convention pinned positive, or None.

    `estimate._fix_even_sign` makes the largest-magnitude even coefficient
    positive. Which mode that is can differ between the before and after
    frames, and then their even-mode signs are pinned to different anchors
    and must not be compared.
    """
    even = {j: a for j, a in est.coeffs.items() if _is_even(j)}
    return max(even, key=lambda j: abs(even[j])) if even else None


def _anchor_text(est):
    """`Z4` style label for the mode the even-sign convention pinned."""
    anchor = _even_anchor(est)
    return "Z%d" % anchor if anchor else "none"


def find_runs(root: Path):
    """Every run folder at or below `root` holding a before/after spot pair."""
    if (root / "spot_before_avg.npy").is_file():
        return [root]
    return sorted(p.parent for p in root.rglob("spot_before_avg.npy"))


def read_optics(run: Path) -> Optics:
    """Optical constants from a run's `run_config.json`.

    Args:
        run: Run folder.

    Raises:
        FileNotFoundError: If the run has no config.
        KeyError: If the config predates the optics block.
    """
    cfg = json.loads((run / "run_config.json").read_text())
    o = cfg["optics"]
    return Optics(wavelength_nm=float(o["wavelength_nm"]),
                  focal_mm=float(o["focal_mm"]),
                  aperture_mm=float(o["aperture_mm"]),
                  pixel_um=float(o["pixel_um"]))


def _spot_panel(ax, frame, optics, roi_px, title):
    """Draw one log-stretched spot crop with the Airy radius marked."""
    crop, _, _ = find_spot(frame, roi_px)
    base = float(np.median(crop))
    shown = np.clip(crop - base, 1e-3, None)
    ax.imshow(np.log10(shown), cmap="inferno", origin="upper")
    half = crop.shape[0] / 2.0
    ax.add_patch(plt.Circle((half, half), optics.r0_px, fill=False,
                            color="#ffffff", lw=0.9, alpha=0.8))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def _bar_panel(ax, before, after):
    """Draw the before/after named-aberration magnitudes."""
    x = np.arange(len(TERMS))
    width = 0.38

    def magnitudes(est):
        out = []
        for key, _label in TERMS:
            value = est.named.get(key, np.nan)
            out.append(abs(value) if key in MAGNITUDE_ONLY else value)
        return out

    b, a = magnitudes(before), magnitudes(after)
    ax.bar(x - width / 2, b, width, label="Before", color=BEFORE_COLOR)
    ax.bar(x + width / 2, a, width, label="After", color=AFTER_COLOR)
    for xi, (vb, va) in enumerate(zip(b, a)):
        for off, v in ((-width / 2, vb), (width / 2, va)):
            if np.isfinite(v):
                ax.text(xi + off, v, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in TERMS])
    ax.set_ylabel("Magnitude / waves RMS")
    ax.set_xlabel("magnitudes only -- for the signs, read the panel below",
                  fontsize=7.5, color="#555555")
    ax.set_title("Named aberration magnitudes (cos/sin pairs combined)",
                 fontsize=10)
    ax.grid(axis="y", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, framealpha=0.92)


def _zernike_panel(ax, before, after):
    """Draw the per-mode Noll coefficients, signed, with photon-noise bars.

    The named panel above collapses each cos/sin pair into one magnitude,
    which is the right summary but hides which way the astigmatism leans and
    hides the higher orders individually. This one shows the fitted vector as
    it is, and marks the signs that cannot be quoted:

      * The even block is hatched and starred. It shares ONE unobservable
        sign, so the zero line is the convention's zero, not an absolute.
        Signs are real within the block and within the odd block.
      * Z4 and Z11 carry a dashed band at `RADIAL_DEGENERACY_WAVES`. Below
        it the two are not separated at all, so a sign there means nothing
        even relative to the rest of the even block.
    """
    modes = sorted(set(before.coeffs) | set(after.coeffs))
    if not modes:
        ax.axis("off")
        return
    x = np.arange(len(modes))
    width = 0.38
    b = [before.coeffs.get(j, np.nan) for j in modes]
    a = [after.coeffs.get(j, np.nan) for j in modes]
    b_err = [before.sigma_waves.get(j, np.nan) for j in modes]
    a_err = [after.sigma_waves.get(j, np.nan) for j in modes]
    even = [_is_even(j) for j in modes]
    bars_b = ax.bar(x - width / 2, b, width, yerr=b_err, capsize=2,
                    label="Before", color=BEFORE_COLOR,
                    error_kw=dict(lw=0.8, ecolor="#333333"))
    bars_a = ax.bar(x + width / 2, a, width, yerr=a_err, capsize=2,
                    label="After", color=AFTER_COLOR,
                    error_kw=dict(lw=0.8, ecolor="#333333"))
    for bars in (bars_b, bars_a):
        for patch, is_even in zip(bars.patches, even):
            if is_even:
                patch.set_hatch("///")
                patch.set_edgecolor("#ffffff")
    # The radial pair is unresolved below this level, so its sign is empty
    # there too. Drawn per bar rather than across the axes: it is a property
    # of defocus against spherical, not of every mode.
    radial = [i for i, j in enumerate(modes) if j in (4, 11)]
    for i in radial:
        ax.hlines([RADIAL_DEGENERACY_WAVES, -RADIAL_DEGENERACY_WAVES],
                  i - 0.5, i + 0.5, color="#444444", lw=0.9, ls=(0, (4, 3)))
    ax.axhline(0.0, color="#555555", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Z{j}{'*' if e else ''}\n{ZK.NOLL_NAMES.get(j, '')}"
         for j, e in zip(modes, even)], fontsize=7.5)
    ax.set_ylabel("Coefficient / waves RMS")
    ax.set_title("Fitted Zernike coefficients (Noll), error bars = photon "
                 "noise", fontsize=10)
    ax.grid(axis="y", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#bbbbbb", hatch="///",
                         edgecolor="#ffffff"))
    labels.append("* even block -- one shared sign is a convention")
    if radial:
        handles.append(Line2D([0], [0], color="#444444", lw=0.9,
                              ls=(0, (4, 3))))
        labels.append("Z4/Z11 unresolved below %.3f waves"
                      % RADIAL_DEGENERACY_WAVES)
    ax.legend(handles, labels, frameon=True, framealpha=0.92, fontsize=8)


def _phase_panel(ax, est, title):
    """Draw one reconstructed pupil phase map."""
    if est.pupil_phase is None:
        ax.axis("off")
        return
    lim = float(np.nanmax(np.abs(est.pupil_phase))) or 1.0
    im = ax.imshow(est.pupil_phase, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, label="waves")


def make_figure(run: Path, optics, before_frame, after_frame, before, after,
                roi_px, suffix=""):
    """Write the four-panel comparison figure for one run.

    Args:
        run: Run folder the figure is written into.
        optics: Optical constants used for the fits.
        before_frame: Averaged start-of-run spot.
        after_frame: Averaged end-of-run spot.
        before: Estimate for the start spot.
        after: Estimate for the end spot.
        roi_px: ROI side used, for the spot panels.
        suffix: Appended to the file stem, so a run fitted in a second basis
            lands beside the first instead of overwriting it.

    Returns:
        The path written.
    """
    fig = plt.figure(figsize=(14.0, 9.6), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.95])
    _spot_panel(fig.add_subplot(grid[0, 0]), before_frame, optics, roi_px,
                "Before (log)")
    _spot_panel(fig.add_subplot(grid[0, 1]), after_frame, optics, roi_px,
                "After (log)")
    _bar_panel(fig.add_subplot(grid[0, 2:]), before, after)
    _zernike_panel(fig.add_subplot(grid[1, :]), before, after)
    _phase_panel(fig.add_subplot(grid[2, 0]), before, "Before: pupil phase")
    _phase_panel(fig.add_subplot(grid[2, 1]), after, "After: pupil phase")

    text = fig.add_subplot(grid[2, 2:])
    text.axis("off")
    lines = [
        f"{'':20s}{'before':>10s}{'after':>10s}",
        f"{'RMS / waves':20s}{before.rms_waves:>10.3f}{after.rms_waves:>10.3f}",
        f"{'RMS / rad':20s}{before.rms_rad:>10.2f}{after.rms_rad:>10.2f}",
        f"{'Strehl':20s}{before.strehl:>10.4f}{after.strehl:>10.4f}",
        f"{'residual':20s}{before.residual:>10.1f}{after.residual:>10.1f}",
        f"{'spot implies / waves':20s}{before.prior_waves:>10.2f}"
        f"{after.prior_waves:>10.2f}",
        f"{'pixels fitted':20s}{before.n_pixels:>10d}{after.n_pixels:>10d}",
        f"{'fit time / ms':20s}{before.ms:>10.0f}{after.ms:>10.0f}",
        "",
        f"{optics.wavelength_nm:.0f} nm, f/{optics.focal_mm / optics.aperture_mm:.0f}"
        f", {optics.pixel_um:.2f} um/px, Airy r0 = {optics.r0_px:.2f} px",
        "",
        "Coma and trefoil signs are measured. The even block (hatched, *)",
        "shares ONE unobservable sign, pinned by convention on its largest",
        f"mode -- {_anchor_text(before)} before, {_anchor_text(after)} after."
        " Relative signs inside the",
        "block are real, but the two columns are pinned SEPARATELY, so an",
        "even-mode sign change between them is not a measurement.",
        "The residual compares fits of the SAME frame only -- it has no",
        "absolute scale.",
    ]
    for est, tag in ((before, "before"), (after, "after")):
        if est.beyond_capture_range:
            lines.append("")
            lines.append(f"*** {tag.upper()}: {est.prior_waves:.2f} waves is "
                         f"beyond the measured capture")
            lines.append(f"    range of {CAPTURE_RANGE_WAVES:.2f} waves. "
                         "These coefficients are a")
            lines.append("    LOWER BOUND, not a measurement.")
        elif est.reason:
            lines.append(f"WARNING ({tag}): {est.reason}")
    text.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=9,
              va="top", ha="left")

    fig.suptitle(f"Single-frame phase retrieval -- {run.name}"
                 f"  [{before.basis_name} basis]", fontsize=13,
                 fontweight="bold")
    path = run / f"phase_retrieval{suffix}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _radial_profile(image, r0_px):
    """Azimuthally averaged profile about the image centre.

    Args:
        image: 2-D array, background already removed.
        r0_px: Airy radius, used as the x unit.

    Returns:
        A (radius in Airy radii, mean intensity) pair.
    """
    side = image.shape[0]
    yy, xx = np.mgrid[0:side, 0:side]
    r = np.hypot(xx - side / 2.0, yy - side / 2.0).ravel().astype(int)
    counts = np.bincount(r, minlength=side)
    total = np.bincount(r, weights=image.ravel(), minlength=side)
    profile = total / np.maximum(counts, 1)
    radius = np.arange(len(profile)) / (r0_px if r0_px > 0 else 1.0)
    keep = slice(0, side // 2)
    return radius[keep], profile[keep]


def _check_row(axes, frame, est, optics, opts, tag):
    """One row of the model-versus-data check: data, model, difference, cut.

    No residual number can replace this panel. A fit can drive the residual
    down by matching brightness while getting the structure wrong, and the
    only way to see that is to put the two images side by side.
    """
    data = fitted_crop(frame, est)
    model = render_model(est, optics, opts)
    if model is None:
        for ax in axes:
            ax.axis("off")
        return
    base = float(np.median(data))
    data_s, model_s = data - base, model - base
    log_data = np.log10(np.clip(data_s, 1e-2, None))
    log_model = np.log10(np.clip(model_s, 1e-2, None))
    vmin, vmax = float(np.percentile(log_data, 2)), float(log_data.max())

    axes[0].imshow(log_data, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{tag}: measured (log)", fontsize=9)
    axes[1].imshow(log_model, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{tag}: fitted model (log)", fontsize=9)
    diff = model - data
    lim = float(np.percentile(np.abs(diff), 99.5)) or 1.0
    axes[2].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title(f"{tag}: model - measured", fontsize=9)
    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])

    radius, prof_d = _radial_profile(data_s, optics.r0_px)
    _, prof_m = _radial_profile(model_s, optics.r0_px)
    ax = axes[3]
    ax.semilogy(radius, np.clip(prof_d, 1e-2, None), lw=1.7, label="measured")
    ax.semilogy(radius, np.clip(prof_m, 1e-2, None), "--", lw=1.7,
                label="model")
    ax.set_xlabel("radius / Airy r0")
    ax.set_ylabel("mean intensity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"{tag}: RMS {est.rms_waves:.3f} waves, "
                 f"residual {est.residual:.4f}", fontsize=9)


def make_check_figure(run: Path, optics, opts, before_frame, after_frame,
                      before, after, suffix=""):
    """Write the model-versus-data check figure for one run.

    Args:
        run: Run folder the figure is written into.
        optics: Optical constants used for the fits.
        opts: The retrieval options the fits used.
        before_frame: Averaged start-of-run spot.
        after_frame: Averaged end-of-run spot.
        before: Estimate for the start spot.
        after: Estimate for the end spot.
        suffix: Appended to the file stem, see `make_figure`.

    Returns:
        The path written.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.0),
                             constrained_layout=True)
    _check_row(axes[0], before_frame, before, optics, opts, "before")
    _check_row(axes[1], after_frame, after, optics, opts, "after")
    fig.suptitle(f"Does the fitted model reproduce the measured spot?"
                 f"  --  {run.name}  [{before.basis_name} basis]",
                 fontsize=13, fontweight="bold")
    path = run / f"phase_retrieval{suffix}_check.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_csv(run: Path, before, after, suffix=""):
    """Write the per-mode coefficients and summary for one run."""
    path = run / f"phase_retrieval{suffix}.csv"
    modes = sorted(set(before.coeffs) | set(after.coeffs))
    with path.open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["quantity", "before", "after", "unit"])
        for j in modes:
            note = ("waves rms (even block: one shared sign is a convention)"
                    if _is_even(j) else "waves rms (sign measured)")
            w.writerow([f"noll_{j}", f"{before.coeffs.get(j, float('nan')):.6g}",
                        f"{after.coeffs.get(j, float('nan')):.6g}", note])
        w.writerow(["even_sign_anchor", _anchor_text(before),
                    _anchor_text(after),
                    "pinned positive by convention, per column"])
        for key, _label in TERMS:
            # Magnitudes, matching the figure. The signed values are in the
            # noll rows above, where the caveat travels with them.
            b_v = before.named.get(key, float("nan"))
            a_v = after.named.get(key, float("nan"))
            if key in MAGNITUDE_ONLY:
                b_v, a_v = abs(b_v), abs(a_v)
            w.writerow([key, f"{b_v:.6g}", f"{a_v:.6g}", "waves (magnitude)"])
        for key in ("astigmatism_angle", "coma_angle", "trefoil_angle"):
            w.writerow([key, f"{before.named.get(key, float('nan')):.6g}",
                        f"{after.named.get(key, float('nan')):.6g}", "deg"])
        w.writerow(["rms_waves", f"{before.rms_waves:.6g}",
                    f"{after.rms_waves:.6g}", "waves"])
        w.writerow(["strehl_exact", f"{before.strehl:.6g}",
                    f"{after.strehl:.6g}", "-"])
        w.writerow(["strehl_marechal", f"{before.strehl_marechal:.6g}",
                    f"{after.strehl_marechal:.6g}",
                    "invalid above ~0.15 waves rms"])
        w.writerow(["fit_residual", f"{before.residual:.6g}",
                    f"{after.residual:.6g}", "fraction of peak"])
        w.writerow(["converged", before.converged, after.converged, "-"])
        w.writerow(["reason", before.reason, after.reason, "-"])
        w.writerow(["basis", before.basis_name, after.basis_name, "-"])
        # The fitted parameters themselves, when the fit was not in Zernikes.
        # The Noll rows above are then the PROJECTION of this vector, not the
        # quantities the solver varied.
        for name in sorted(set(before.modal_amplitudes)
                           | set(after.modal_amplitudes)):
            nan = float("nan")
            w.writerow([f"mode_{name}",
                        f"{before.modal_amplitudes.get(name, nan):.6g}",
                        f"{after.modal_amplitudes.get(name, nan):.6g}",
                        "modal amplitude, waves rms of that mode"])
    return path


def process(run: Path, roi_px, modes=None, basis=None):
    """Fit both spots of one run and write its figure and CSV.

    Args:
        run: Run folder holding the before/after spot pair.
        roi_px: ROI side, in pixels.
        modes: Noll indices to fit, or None for the default set.
        basis: A `dm_basis.DMBasis` to fit in instead of Zernikes, or None
            for the Zernike default. Both write the same two figures and the
            same CSV; the DM run adds the modal amplitudes to each.

    Returns:
        A (before, after) pair of estimates.
    """
    optics = read_optics(run)
    opts = RetrievalOptions(roi_px=int(roi_px), basis=basis,
                            **({} if modes is None else {"modes": modes}))
    before_frame = np.load(run / "spot_before_avg.npy")
    after_frame = np.load(run / "spot_after_avg.npy")
    before = estimate_wavefront(before_frame, optics, opts)
    after = estimate_wavefront(after_frame, optics, opts)
    # A second basis writes beside the first, never over it: the two answers
    # are the comparison, so losing either one defeats the point.
    suffix = "" if basis is None else "_dm"
    make_figure(run, optics, before_frame, after_frame, before, after, roi_px,
                suffix)
    make_check_figure(run, optics, opts, before_frame, after_frame, before,
                      after, suffix)
    write_csv(run, before, after, suffix)
    return before, after


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="A run folder, or any folder above one")
    parser.add_argument("--roi", type=int, default=192,
                        help="ROI side in pixels (default 192)")
    parser.add_argument("--basis", choices=("zernike", "dm"),
                        default="zernike",
                        help="Phase expansion: Noll Zernikes (default), or "
                             "the mirror's own eigenmodes (see dm_basis)")
    parser.add_argument("--im", default=None,
                        help="influence_matrix.npz of an Impact Matrix "
                             "session; required for --basis dm")
    parser.add_argument("--keep", type=int, default=DMB.DEFAULT_KEEP,
                        help=f"Eigenmodes to fit with --basis dm "
                             f"(default {DMB.DEFAULT_KEEP})")
    parser.add_argument("--rotation", type=float, default=0.0,
                        help="Camera-versus-interferometer rotation in "
                             "degrees, for --basis dm")
    parser.add_argument("--flip", action="store_true",
                        help="Camera-versus-interferometer handedness flip, "
                             "for --basis dm")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many runs (0 = all)")
    args = parser.parse_args(argv)

    basis = None
    if args.basis == "dm":
        if not args.im:
            print("--basis dm needs --im <influence_matrix.npz>")
            return 2
        basis = DMB.from_npz(args.im, samples=RetrievalOptions().pupil_samples,
                             keep=args.keep, rotation_deg=args.rotation,
                             flip=args.flip)
        print(f"basis: {len(basis)} DM eigenmodes from {Path(args.im).parent.name}"
              f", rotation {args.rotation:.0f} deg, flip {args.flip}")

    root = Path(args.path).expanduser().resolve()
    runs = find_runs(root)
    if not runs:
        print(f"No spot_before_avg.npy found under {root}")
        return 1
    found = len(runs)
    if args.limit > 0:
        runs = runs[:args.limit]
    if len(runs) < found:
        print(f"{len(runs)} of {found} run(s) under {root} (--limit)\n")
    else:
        print(f"{len(runs)} run(s) under {root}\n")
    header = (f"{'run':44s}{'rms_before':>11s}{'rms_after':>10s}"
              f"{'resid_b':>9s}{'resid_a':>9s}")
    print(header)
    print("-" * len(header))
    for run in runs:
        try:
            before, after = process(run, args.roi, basis=basis)
        except (OSError, KeyError, ValueError) as err:
            print(f"{run.name[:43]:44s}  skipped: {err}")
            continue
        print(f"{run.name[:43]:44s}{before.rms_waves:>11.3f}"
              f"{after.rms_waves:>10.3f}{before.residual:>9.4f}"
              f"{after.residual:>9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
