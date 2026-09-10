# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.2, 2026-08-11

"""Drive the influence-matrix computation and record every step it takes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import influence as INF
from . import io_surface as IO
from . import trace
from .trace import Trace

ROLE_REF = "ref"  # A repeat of the bias state.
ROLE_PP = "pp"  # One leg of an actuator's push-pull pair.
ROLE_CHECK = "check"  # An arbitrary command, for the superposition test.
ROLES = (ROLE_REF, ROLE_PP, ROLE_CHECK)


@dataclass
class Entry:
    """One measured surface file and what it is for."""
    path: Path
    role: str
    bits: dict = field(default_factory=dict)  # {channel: bit}

    @property
    def name(self):
        return self.path.name


@dataclass
class Job:
    """Everything the pipeline needs besides the files themselves."""
    entries: list
    pupil: INF.Pupil
    grid_n: int = 128
    scheme: str = INF.SCHEME_CENTRAL
    interferometer_nm: float = 632.8  # Only a fallback: the files carry it.
    laser_nm: float = 635.0  # The AO path's own wavelength.
    zernike_terms: int = 15
    xyz_unit: str = IO.UNIT_UM
    xyz_lateral_um: float = 0.0
    remove_piston_tilt: bool = False


@dataclass
class Result:
    """The computed matrices plus the trace that produced them."""
    trace: Trace
    grid: INF.Grid
    inside: np.ndarray | None = None
    channels: tuple = ()
    c_nm: np.ndarray | None = None  # (npix, n_act) nm per bit.
    c_rad: np.ndarray | None = None  # (npix, n_act) rad per bit.
    gradient: np.ndarray | None = None
    modes: INF.Modes | None = None
    noise_sv: float = float("nan")
    keep: int = 0
    zernikes: list = field(default_factory=list)  # Per mode, or None.
    rad_per_bit: float = float("nan")  # For the strongest mode, for Band Scan.
    messages: list = field(default_factory=list)

    @property
    def ok(self):
        return self.modes is not None


def _pair_offsets(offsets):
    """Every symmetric push-pull pair in a ladder, innermost first.

    Each exact +/- pair is one push-pull measurement of the same influence
    function; all are combined into one least-squares column. Only exact
    pairs are returned, because a lopsided pair leaves a quadratic residual
    in the column.

    Args:
        offsets: Bit offsets from the bias, one per measured surface.

    Returns:
        (pairs, unpaired). `pairs` is a list of (negative, positive) tuples
        ordered by amplitude; `unpaired` lists the offsets left over.
    """
    have = {int(o) for o in offsets}
    pairs = [(-a, a) for a in sorted(abs(o) for o in have if o > 0)
             if -a in have]
    matched = {o for pair in pairs for o in pair}
    return pairs, sorted(o for o in have if o and o not in matched)


def run(job: Job, progress=None) -> Result:
    """Compute the influence matrix and eigenmodes, tracing every stage.

    Args:
        job: Files, pupil and options.
        progress: Optional `callable(done, total)` hook.

    Returns:
        A `Result`; `ok` is False and `messages` explains when it stopped early.
    """
    tr = Trace()
    grid = INF.Grid(int(job.grid_n))
    res = Result(trace=tr, grid=grid)

    refs = [e for e in job.entries if e.role == ROLE_REF]
    pps = [e for e in job.entries if e.role == ROLE_PP]
    checks = [e for e in job.entries if e.role == ROLE_CHECK]

    step = tr.add("Inputs",
                  why="Fix what is being measured before any arithmetic, so a "
                      "wrong pupil or a missing bias shows up here rather than "
                      "as a strange mode later.")
    step.inputs = [f"{len(refs)} bias reference(s), {len(pps)} push-pull "
                   f"surface(s), {len(checks)} validation surface(s)",
                   f"pupil: centre ({job.pupil.cx:.1f}, "
                   f"{job.pupil.cy:.1f}) px, radius {job.pupil.r:.1f} px",
                   f"grid: {grid.n} x {grid.n}, "
                   f"{int(grid.inside.sum())} points inside the unit circle",
                   f"differencing: {job.scheme}",
                   f"interferometer: {job.interferometer_nm:g} nm "
                   "(fallback; each file states its own)",
                   f"AO laser: {job.laser_nm:g} nm",
                   ("piston/tip/tilt: fitted and removed here"
                    if job.remove_piston_tilt else
                    "piston/tip/tilt: trusted as already removed in Mx")]
    if not refs:
        step.note("no bias reference: cannot tell which bit is the "
                          "operating point, so nothing can be paired", trace.STOP)
        res.messages.append("add one surface with role 'ref' (all channels at "
                            "the bias, e.g. 2000)")
        return res
    if not pps:
        step.note("no push-pull surfaces", trace.STOP)
        res.messages.append("add the +/- surfaces with role 'pp'")
        return res

    bias = dict(refs[0].bits)
    step.numbers.append(("bias state",
                         ", ".join(f"ch{c}={bias[c]}" for c in sorted(bias))
                         or "unknown (no bits parsed from the file name)"))

    # Step 2: everything onto one grid, in nm.
    total = len(job.entries)
    grids = {}
    step = tr.add("Resample onto the pupil grid",
                  formula="z(x, y) [file pixels]  ->  Z[i, j] [pupil grid], nm",
                  why="Every surface must be sampled on the SAME points before "
                      "they can be differenced, and only the beam footprint "
                      "matters: gradient orthogonality is defined by an "
                      "integral over the pupil, so integrating the wrong "
                      "area gives modes not orthogonal where it counts.")
    notes, sys_err = set(), {}
    for i, e in enumerate(job.entries):
        surf = IO.read_surface(e.path, job.interferometer_nm, job.xyz_unit,
                               job.xyz_lateral_um)
        notes.add(surf.note)
        if surf.sys_err_subtracted is not None:
            sys_err.setdefault(bool(surf.sys_err_subtracted), []).append(e.name)
        z, ok = INF.resample(surf, job.pupil, grid)
        grids[e.path] = (z, ok)
        if progress is not None:
            progress(i + 1, total)
    calibration_entries = refs + pps
    inside = np.logical_and.reduce(
        [grids[e.path][1] for e in calibration_entries]
    )
    res.inside = inside
    npix = int(inside.sum())
    ref_z = grids[refs[0].path][0]
    tr.matrix(step, "Z_ref", np.where(inside, ref_z, np.nan), units="nm",
              role="the bias-state surface, the zero everything is measured "
                   "against", heat=np.where(inside, ref_z, np.nan))
    step.numbers.append(
        ("pupil points valid in every ref + push-pull surface", str(npix))
    )
    step.numbers.append(("height scaling applied", "; ".join(sorted(notes))))
    if sys_err:
        step.numbers.append(
            ("system error (substrate) subtracted in Mx",
             ", ".join(f"{'yes' if k else 'no'} for {len(v)} file(s)"
                       for k, v in sorted(sys_err.items()))))
    if len(sys_err) > 1:
        step.note(
            "these files DISAGREE on whether Mx subtracted the system error, "
            "so "
            "they carry different baselines. Push-pull differencing cancels a "
            "baseline that is the same in both legs -- it cannot cancel one "
            "that "
            "is present in one leg and absent in the other. Re-export the set "
            "with one consistent setting.", trace.STOP)
    if len(notes) > 1:
        step.note(
            "the files were scaled to nanometres by DIFFERENT routes "
            f"({'; '.join(sorted(notes))}). Mixing .datx and .xyz is only safe "
            "once the .xyz unit is right -- waves and micrometres cannot be "
            "told "
            "apart from the numbers alone.", trace.WARN)
    frac = npix / max(int(grid.inside.sum()), 1)
    if frac < 0.98:
        step.note(
            f"only {100 * frac:.1f}% of the pupil is valid in every "
            "calibration file -- "
            "dropouts or a pupil that runs off the measured area. Shrink the "
            "pupil or re-measure; edge points are where false gradients come "
            "from.", trace.WARN)
    if npix < 200:
        res.messages.append("too few valid pupil points to continue")
        return res

    X, Y = grid.coords
    Xin, Yin = X[inside], Y[inside]

    # Step 3: pair the push-pull legs per channel.
    step = tr.add("Pair the push-pull legs",
                  formula="per channel k: (bit_minus, bit_plus) about the "
                          "bias",
                  why="A symmetric pair about the bias cancels the static "
                      "figure and the interferometer's own error, and cancels "
                      "the even-order part of the piezo non-linearity, leaving "
                      "the local slope.")
    per_channel = {}
    for e in pps:
        moved = [c for c, b in e.bits.items() if b != bias.get(c, b)]
        if len(moved) != 1:
            step.note(f"{e.name}: {len(moved)} channels differ from "
                              "the bias, expected exactly 1 -- skipped", trace.WARN)
            continue
        c = moved[0]
        per_channel.setdefault(c, {})[e.bits[c] - bias.get(c, 0)] = e.path
    channels = tuple(sorted(per_channel))
    pairs = {}
    for c in channels:
        found, unpaired = _pair_offsets(list(per_channel[c]))
        if not found:
            step.note(f"ch{c}: no exact +/- pair about the bias -- "
                              "skipped", trace.WARN)
            continue
        pairs[c] = found
        step.inputs.append(
            f"ch{c}: " + ", ".join(f"+/-{hi}" for _lo, hi in found) + " bit")
        if unpaired:
            step.note(
                f"ch{c}: " + ", ".join(f"{o:+d}" for o in unpaired)
                + " bit has no mirror-image partner and was dropped. Only an "
                  "exact +/- pair cancels the even-order non-linearity; "
                  "pairing it with the nearest other amplitude would leave a "
                  "quadratic residual in the column with nothing to flag it.", trace.INFO)
    channels = tuple(c for c in channels if c in pairs)
    res.channels = channels
    if not channels:
        res.messages.append("no channel had a usable +/- pair")
        return res
    named = ", ".join(f"ch{c}" for c in channels)
    step.numbers.append(("actuators with a usable pair",
                         f"{len(channels)}  ({named})"))
    counts = {len(v) for v in pairs.values()}
    step.numbers.append(
        ("symmetric pairs per actuator",
         f"{min(counts)}" if len(counts) == 1
         else f"{min(counts)}..{max(counts)} (they differ)"))

    # Step 4: push-pull to nm per bit.
    step = tr.add("Push-pull to influence functions",
                  formula="d_j = (Z_k(+a_j) - Z_k(-a_j)) / 2,   "
                          "IF_k = sum(a_j d_j) / sum(a_j^2)",
                  why="Each symmetric pair removes everything that did not "
                      "move with actuator k -- which is why the bias reference "
                      "is not needed in the arithmetic here, it cancels. Every "
                      "pair measures the SAME influence function, so all of "
                      "them are fitted at once by a straight line through the "
                      "origin instead of keeping one and discarding the rest. "
                      "The wide pairs dominate because they carry more signal "
                      "against the same noise; that weighting is the "
                      "least-squares algebra, not a choice.")
    c_nm = np.zeros((npix, len(channels)))
    for i, c in enumerate(channels):
        halves, amps = [], []
        for lo, hi in pairs[c]:
            z_lo = grids[per_channel[c][lo]][0][inside]
            z_hi = grids[per_channel[c][hi]][0][inside]
            halves.append((z_hi - z_lo) / 2.0)
            amps.append(hi)
        c_nm[:, i] = INF.slope_from_pairs(halves, amps)
    first = np.full((grid.n, grid.n), np.nan)
    first[inside] = c_nm[:, 0]
    tr.matrix(step, "IF_1", first, units="nm/bit",
              role=f"channel {channels[0]}: how the surface moves per bit",
              heat=first)
    widest = {c: pairs[c][-1][1] for c in channels}
    # RMS over the actuator's own footprint, not peak-to-valley, which is set
    # by whichever two points the noise pushed furthest apart.
    strokes = {c: 2 * widest[c] * INF.weighted_rms(
        c_nm[:, i], INF.footprint_weight(c_nm[:, i]))
        for i, c in enumerate(channels)}
    step.numbers += [
        (f"ch{c} stroke over its widest pair (+/-{widest[c]})",
         f"{strokes[c]:.1f} nm rms over its own footprint")
        for c in channels]
    quiet = {c: INF.effective_span([hi for _lo, hi in pairs[c]])
             for c in channels}
    if max(len(v) for v in pairs.values()) > 1:
        step.numbers.append(
            ("noise gain over using the widest pair alone",
             "  ".join(f"ch{c} {2 * widest[c] / quiet[c]:.2f}x"
                       for c in channels)))

    # Step 5: surface to wavefront phase.
    step = tr.add("Surface height to wavefront phase",
                  formula=f"C = (4*pi / {job.laser_nm:g} nm) * IF"
                          f"   =  {INF.PHASE_PER_NM / job.laser_nm:.5f} "
                          f"rad per nm",
                  why="The light reflects, so it crosses the sag twice: the "
                      "wavefront error is 2 z, and 2*pi/lambda turns that into "
                      "radians. The lambda here is the AO PATH's laser, not "
                      "the "
                      "interferometer's: the interferometer wavelength has "
                      "already done its job converting fringes to nanometres "
                      "of "
                      "mirror shape, and what the loop then sees is that shape "
                      "in radians at its own colour. Getting the factor of 2 "
                      "wrong is a silent 50% scale error on every amplitude "
                      "downstream.")
    c_rad = c_nm * (INF.PHASE_PER_NM / float(job.laser_nm))
    tr.matrix(step, "C", c_rad, units="rad/bit",
              role="influence matrix: one row per pupil point, one column per "
                   "actuator")

    # Step 6: inspect the fitted plane and remove it only when requested.
    title = ("Remove piston, tip and tilt from every column"
             if job.remove_piston_tilt else
             "Inspect piston, tip and tilt without removing them")
    formula = ("C_k <- C_k - (a + b x + c y), fitted on the pupil"
               if job.remove_piston_tilt else
               "fit a + b x + c y for diagnosis; leave C_k unchanged")
    step = tr.add(
        title,
        formula=formula,
        why="Tilt only moves the spot, it does not blur it, yet it carries "
            "large gradient energy. The cited modal methods exclude "
            "piston/tip/tilt. Mx can already remove them, however, and fitting "
            "again on a different pupil changes the stated preprocessing. The "
            "default therefore trusts the exported surfaces; enable the option "
            "only for consistently raw inputs.",
    )
    fitted = np.zeros((3, len(channels)))
    detrended = np.zeros_like(c_rad)
    for i in range(len(channels)):
        detrended[:, i], fitted[:, i] = INF.remove_piston_tilt(
            c_rad[:, i],
            Xin,
            Yin,
        )
    tr.matrix(
        step,
        "plane coefficients",
        fitted,
        units="rad/bit",
        role=("rows: piston, x tilt, y tilt removed from each column"
              if job.remove_piston_tilt else
              "rows: piston, x tilt, y tilt measured but left in each column"),
    )
    if job.remove_piston_tilt:
        c_rad = detrended
    else:
        step.note(
            "No plane was subtracted. This is correct only when every ref, "
            "push-pull and validation surface was exported with the same Mx "
            "piston/tip/tilt removal."
        , trace.INFO)
    residual_energy = np.mean(detrended ** 2, axis=0)
    fitted_tilt = (
        Xin[:, np.newaxis] * fitted[1]
        + Yin[:, np.newaxis] * fitted[2]
    )
    tilt_energy = np.mean(fitted_tilt ** 2, axis=0)
    tilt_fraction = tilt_energy / np.maximum(
        tilt_energy + residual_energy,
        1e-30,
    )
    step.numbers.append(
        ("fitted tilt energy fraction of each column",
         "  ".join(f"ch{c} {100 * f:.0f}%"
                   for c, f in zip(channels, tilt_fraction))))
    c_nm = c_rad / (INF.PHASE_PER_NM / float(job.laser_nm))
    res.c_nm, res.c_rad = c_nm, c_rad

    # Step 7: gradients.
    step = tr.add("Differentiate over the pupil",
                  formula=("grad(C) = [ dC/dx ; dC/dy ]   ("
                           + ("(z[i+1] - z[i-1]) / 2h"
                              if job.scheme == INF.SCHEME_CENTRAL
                              else "z[i+1] - z[i]") + ")"),
                  why="Image-plane metrics respond to the mean square "
                      "wavefront SLOPE, not the wavefront: a ray lands "
                      "displaced in "
                      "proportion to the local gradient. So the orthogonality "
                      "that decouples the modes is orthogonality of the "
                      "derivatives (Debarre Eq. 21 and Eq. 26). Differences "
                      "crossing the pupil edge are dropped -- they would "
                      "subtract a real height from an undefined one and "
                      "dominate the SVD with a false slope.")
    gC, n_pts = INF.gradient_matrix(c_rad, grid, inside, job.scheme)
    res.gradient = gC
    tr.matrix(step, "grad(C)", gC, units="rad/bit per pupil radius",
              role="x derivatives stacked above y derivatives")
    step.numbers += [
        ("gradient sample points", f"{n_pts} ({n_pts // 2} per direction)"),
        ("rows dropped at the pupil edge",
         f"{2 * npix - n_pts} of {2 * npix}")]

    # Step 8: SVD.
    step = tr.add("Singular value decomposition",
                  formula="grad(C) = U_g S V^T      ->     "
                          "modes U = C V S^-1,   controls = V S^-1",
                  why="This rotates the actuator axes onto the principal axes "
                      "of the metric's quadratic form, so the cross terms "
                      "vanish and each mode becomes an independent parabola "
                      "solvable from three measurements. The columns of V are "
                      "also the eigenvectors of the Gram matrix "
                      "G = grad(C)^T grad(C), which is what Ren & Dong's "
                      "camera-only self-calibration measures -- so the two "
                      "routes can be compared column by column.")
    modes = INF.eigenmodes(gC, c_rad, grid, inside)
    res.modes = modes
    tr.matrix(step, "S", modes.s, role="singular values, descending")
    tr.matrix(step, "V", modes.v,
              role="raw control directions; column i drives mode i")
    tr.matrix(step, "ctrl = V/S, RMS-scaled", modes.ctrl, units="bit per rad",
              role="column i, times an amplitude in rad, is the bit offset to "
                   "add to the bias")
    tr.matrix(step, "G = grad(C)^T grad(C)", modes.gram,
              role="Gram matrix; off-diagonal entries are the actuator "
                   "cross-talk the rotation removes")

    # Step 9: noise floor and truncation.
    step = tr.add("Noise floor and how many modes survive",
                  formula="floor = || grad( (Z_ref1 - Z_ref2) / bit span ) ||",
                  why="A singular value divides the estimated coefficient of "
                      "its mode, so a small one turns measurement noise into a "
                      "large wrong correction. Repeats of the same mirror "
                      "state differ only by noise, so pushing that "
                      "difference through the identical operator gives the "
                      "level below which a singular value means nothing. The "
                      "span used is the one a single pair would need to be as "
                      "quiet as the multi-pair fit, so the floor stays "
                      "comparable with the singular values it is judging.")
    span = quiet[channels[0]]
    phase = INF.PHASE_PER_NM / float(job.laser_nm)  # Match C's units.
    ref_pairs = [(grids[a.path][0][inside] * phase,
                  grids[b.path][0][inside] * phase)
                 for a, b in zip(refs, refs[1:])]
    if ref_pairs:
        res.noise_sv = INF.noise_singular_value(
            ref_pairs,
            span,
            grid,
            inside,
            job.scheme,
            remove_plane=job.remove_piston_tilt,
        )
    else:
        res.noise_sv = float("nan")
    res.keep = INF.keep_count(modes.s, res.noise_sv)
    if not ref_pairs:
        step.note(
            "only one bias reference, so there is no noise estimate and no "
            "truncation point. Measure the bias state at least twice (three "
            "times is better) and add them all as 'ref'.", trace.WARN)
        res.keep = len(modes.s)
    step.numbers += [
        ("noise floor singular value",
         f"{res.noise_sv:.4g}" if np.isfinite(res.noise_sv) else "unknown"),
        ("modes clear of the floor (by "
         f"{INF.KEEP_MARGIN:g}x)", f"{res.keep} of {modes.n}"),
        ("condition number keeping that many (Gram, i.e. s^2 ratio)",
         f"{modes.condition(res.keep):.1f}")]
    if res.keep < modes.n:
        step.note(
            f"modes {res.keep + 1}..{modes.n} do not clear the measurement "
            "noise. Correcting them injects noise instead of removing "
            "aberration -- this is the truncation both Chinese papers had to "
            "apply (Wang & Dong Fig. 10).", trace.WARN)

    # Step 10: what the modes are.
    step = tr.add("What each mode is",
                  formula="Zernike fit of U_i over the pupil (RMS-normalised "
                          "Noll terms)",
                  why="Turns the eigenmodes into the language the aberration "
                      "is described in, which is what tells you what this "
                      "mirror physically can and cannot correct.")
    for i in range(modes.n):
        fit = INF.zernike_of(modes.maps[i], inside, grid, job.zernike_terms)
        res.zernikes.append(fit)
        if fit is None or i >= max(res.keep, 1):
            continue
        named = fit["named"]
        parts = sorted(((abs(v), k) for k, v in named.items()
                        if not k.endswith("angle")), reverse=True)[:2]
        step.numbers.append(
            (f"mode {i + 1}  (S = {modes.s[i]:.4g})",
             ", ".join(f"{k} {v:.3f} rad" for v, k in parts)))
    if modes.s[0] > 0:
        # One rad RMS of the strongest mode costs this many bits of drive; the
        # reciprocal is what the Band Scan page wants to print radians.
        bits_per_rad = float(np.max(np.abs(modes.ctrl[:, 0])))
        res.rad_per_bit = (1.0 / bits_per_rad if bits_per_rad > 0
                           else float("nan"))
        step.numbers.append(
            ("mode 1: rad per bit on its largest channel",
             f"{res.rad_per_bit:.6f}  (enter this on the Band Scan page)"))

    # One measurement's own error: two repeats differ by the noise of both,
    # so divide by sqrt(2). Every check below is read against this.
    ref_diffs = [(grids[a.path][0][inside] - grids[b.path][0][inside])
                 / np.sqrt(2.0) for a, b in zip(refs, refs[1:])]
    # The noise floor was measured at channel 0's effective span and scales
    # inversely with it, so each channel is judged at its own.
    noise_by_channel = {c: res.noise_sv * span / quiet[c] for c in channels}
    _check_linearity(tr, job, grids, per_channel, pairs, channels, c_nm,
                     inside, Xin, Yin, ref_diffs)
    _check_channels(tr, grids, per_channel, pairs, channels, c_nm,
                    ref_z[inside], inside, ref_diffs, gC, noise_by_channel)
    _check_superposition(tr, job, grids, checks, bias, channels, c_rad,
                         grids[refs[0].path][0], inside, Xin, Yin, ref_diffs)
    _summarise(tr, res, modes)
    return res


def _weighted_noise(ref_diffs, weight):
    """Repeat-to-repeat noise over one actuator's footprint, in nm.

    Args:
        ref_diffs: One-measurement noise realisations over the pupil points.
        weight: Footprint weights, or None for the pupil-wide figure.

    Returns:
        The median RMS, or NaN when there was only one bias reference.
    """
    vals = [INF.weighted_rms(d, weight) for d in ref_diffs]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def _check_linearity(tr, job, grids, per_channel, pairs, channels, c_nm,
                     inside, Xin, Yin, ref_diffs):
    """Score every amplitude against the fitted column, gain and shape apart."""
    step = tr.add("Linearity check",
                  formula="per pair j:  IF_j = d_j / a_j;   "
                          "gain = rms(IF_j) / rms(IF),   "
                          "shape = difference after scaling both to unit RMS;  "
                          "both weighted by IF^2, both with the noise removed",
                  why="One linear influence function per actuator is only "
                      "valid if the SHAPE does not change with amplitude, and "
                      "the fitted column is only representative if the GAIN "
                      "does not drift either. Dividing the amplitude out "
                      "separates the two: a pure gain change reads zero shape "
                      "difference, so the two numbers point at different "
                      "faults. A gain that slides monotonically with amplitude "
                      "is the piezo's odd-order non-linearity -- the part a "
                      "symmetric pair cannot cancel. Both are measured over "
                      "the actuator's own footprint and shown as a multiple of "
                      "the noise, because a percentage of a pupil the actuator "
                      "barely reaches is not a fault it committed.")
    if max((len(pairs[c]) for c in channels), default=0) < 2:
        step.note(
            "only one amplitude measured per channel, so linearity is "
            "untested. Add a second symmetric pair to the ladder (for example "
            "+/-750 alongside +/-1500) and this check comes free.", trace.WARN)
        return
    for c in channels:
        if len(pairs[c]) < 2:
            continue
        fitted = c_nm[:, channels.index(c)]
        weight = INF.footprint_weight(fitted)
        noise_nm = _weighted_noise(ref_diffs, weight)
        # Noise of each column in its own nm-per-bit units (see
        # influence.effective_span).
        sigma = {} if not np.isfinite(noise_nm) else {
            hi: noise_nm * np.sqrt(2.0) / float(hi - lo)
            for lo, hi in pairs[c]}
        sigma_fit = (noise_nm * np.sqrt(2.0)
                     / INF.effective_span([hi for _lo, hi in pairs[c]])
                     if np.isfinite(noise_nm) else 0.0)
        fitted_rms = max(INF.debiased_rms(fitted, sigma_fit, weight), 1e-30)
        cells, worst, worst_ratio, gains = [], 0.0, float("nan"), []
        for lo, hi in pairs[c]:
            col = INF.push_pull(grids[per_channel[c][hi]][0][inside],
                                grids[per_channel[c][lo]][0][inside], hi - lo)
            if job.remove_piston_tilt:
                col, _ = INF.remove_piston_tilt(col, Xin, Yin)
            d = INF.shape_difference(col, fitted, weight)
            gain = INF.debiased_rms(col, sigma.get(hi, 0.0), weight) \
                / fitted_rms
            # Both normalised columns carry their own noise, so a shape
            # difference can never read below this. Without the ratio a quiet
            # bench and a noisy one get judged by the same percentage.
            expected = (float(np.hypot(sigma.get(hi, 0.0), sigma_fit))
                        / fitted_rms) if np.isfinite(noise_nm) else float("nan")
            ratio = (d / expected if np.isfinite(d) and np.isfinite(expected)
                     and expected > 0 else float("nan"))
            if np.isfinite(d):
                worst = max(worst, d)
            if np.isfinite(ratio):
                worst_ratio = (ratio if not np.isfinite(worst_ratio)
                               else max(worst_ratio, ratio))
            if np.isfinite(gain):
                gains.append(gain)
            cells.append(f"+/-{hi}: gain {gain:.3f}, shape {100 * d:.1f}%"
                         + (f" ({ratio:.1f}x noise)"
                            if np.isfinite(ratio) else ""))
        step.numbers.append((f"ch{c}", "   ".join(cells)))
        real = not np.isfinite(worst_ratio) or worst_ratio > 3.0
        if worst > 0.20 and real:
            step.note(
                f"ch{c}: the shape changes {100 * worst:.0f}% across the "
                "measured amplitudes, well clear of the noise, so one linear "
                "column cannot describe it over this range -- drop the "
                "amplitudes far from the one you will actually drive and "
                "refit.", trace.WARN)
        elif worst > 0.20:
            step.info(
                f"ch{c}: the {100 * worst:.0f}% shape spread is within the "
                "measurement noise, so it is not evidence of non-linearity. "
                "Average more frames per point, or drive wider pairs, before "
                "reading anything into it.")
        if len(gains) > 1 and min(gains) > 0:
            spread = max(gains) / min(gains) - 1.0
            if spread > 0.10:
                step.note(
                    f"ch{c}: the influence per bit changes {100 * spread:.0f}% "
                    "between the smallest and the largest amplitude, with the "
                    "noise already taken out. The column is one straight line "
                    "across all of them, so a command near either end is "
                    "predicted with about that much gain error -- fit the "
                    "matrix over the amplitudes the loop will really use.",
                    trace.WARN)


def _delivered(grids, per_channel, c, column, ref_col, inside):
    """How far actuator `c` actually went at each measured bit offset.

    Projects every single-sided surface onto the fitted column, so the
    answer is in bits of ideal linear drive. Two even nuisance maps (a
    constant drift and an offset-squared term) are fitted alongside and
    discarded; the quadratic one only when the ladder is symmetric, so it
    stays orthogonal to the odd influence column.

    Args:
        grids: {path: (z, ok)} from the resampling step.
        per_channel: {channel: {offset: entry}}.
        c: Channel to report on.
        column: That channel's fitted influence column, over `inside`.
        ref_col: The bias reference over `inside`.
        inside: Boolean pupil mask.

    Returns:
        {offset: delivered bits} and what the fit could not explain, one row
        per measured offset; ({}, None) if the column is empty.
    """
    energy = float(column @ column)
    if energy <= 0:
        return {}, None
    offsets = sorted(per_channel[c])
    stack = np.array([grids[per_channel[c][o]][0][inside] - ref_col
                      for o in offsets])
    o = np.array(offsets, float)
    basis = [np.ones_like(o)]
    if set(offsets) == {-v for v in offsets} and np.any(o):
        basis.append((o / np.max(np.abs(o))) ** 2)
    B = np.column_stack(basis)
    B_pinv = np.linalg.pinv(B)
    amps = stack @ column / energy
    nuisance = np.zeros_like(stack)
    for _ in range(20):
        nuisance = B @ (B_pinv @ (stack - amps[:, np.newaxis] * column))
        amps = (stack - nuisance) @ column / energy
    nuisance = B @ (B_pinv @ (stack - amps[:, np.newaxis] * column))
    resid = stack - nuisance - amps[:, np.newaxis] * column
    return dict(zip(offsets, amps)), resid


def _verdict(asym, floor, inv_snr, resid_frac):
    """One channel's overall grade, and why it got it.

    Args:
        asym: Probe asymmetry, |a(+b) + a(-b)| over the mean probe size, with
            the value common to every channel already taken out.
        floor: Smallest incremental gain along the ramp, as a fraction of the
            channel's own median.
        inv_snr: The measurement noise floor over this column's own norm, both
            in gradient space -- the reciprocal of how clearly the SVD sees
            this actuator at all.
        resid_frac: What one fitted column leaves unexplained across the whole
            ladder, as a fraction of that channel's own stroke.

    Returns:
        (grade, reasons) -- "ok", "watch" or "BAD", and the failing measures.
    """
    # Thresholds are placed where this kind of measurement actually
    # separates; BAD is reserved for a channel standing clear of the rest.
    tests = [
        # A three-point solve divides by (G+ - G-) assuming the two probes are
        # +b and -b. A third of the probe missing biases the estimate; most of
        # it missing means one of the three points never happened.
        (asym, 0.80, 0.30, "probe asymmetry"),
        # A dead zone: the ramp barely moves over part of its range, so no
        # single column and no static warp describes it there.
        (1.0 - floor, 0.50, 0.25, "dead zone"),
        # A column that does not beat the noise contributes a direction the
        # solve fills with noise; judged against the noise, not the others.
        (inv_snr, 1.0 / INF.KEEP_MARGIN, 0.20, "column lost in noise"),
        # Against the channel's own stroke, not the repeat noise: what the
        # column misses is an error in predicting a command.
        (resid_frac, 0.10, 0.05, "one column does not fit the ladder"),
    ]
    bad = [name for v, hi, _lo, name in tests if np.isfinite(v) and v > hi]
    watch = [name for v, hi, lo, name in tests
             if np.isfinite(v) and lo < v <= hi]
    if bad:
        return "BAD", bad
    return ("watch", watch) if watch else ("ok", [])


def _check_channels(tr, grids, per_channel, pairs, channels, c_nm, ref_col,
                    inside, ref_diffs, gC, noise_sv):
    """Grade every actuator on four independent measures, not one number.

    The measures are sensitive to different faults, so all four are shown
    side by side and the grade says which failed. Each is read in bits of
    delivered drive or as a fraction of the actuator's own stroke, never as
    a peak-to-valley and never against the repeat noise.
    """
    step = tr.add(
        "Per-channel verdict",
        formula="a(o) = <Z(o) - Z_ref, IF> / <IF, IF>   [bits delivered];   "
                "residual = Z(o) - Z_ref - a(o) IF - (even-in-o nuisance)",
        why="Projecting each single-sided surface back onto the fitted column "
            "turns the whole ladder into one number per point -- how far the "
            "actuator actually went -- and what that projection cannot explain "
            "is the residual. Everything here is read off that one fit, so the "
            "four measures come from the same data and can be compared "
            "against each other rather than against four different baselines. "
            "A drift between the reference and this block, and the even-order "
            "part of the response that push-pull cancels by construction, are "
            "fitted out first: both are even in the offset where the column is "
            "odd, so removing them cannot touch the amplitudes, and leaving "
            "them in would charge the actuator for shapes the column was never "
            "allowed to contain.")
    measured = {}
    for i, c in enumerate(channels):
        column = c_nm[:, i]
        weight = INF.footprint_weight(column)
        amps, resid = _delivered(grids, per_channel, c, column, ref_col,
                                 inside)
        probe = pairs[c][0][1]  # Innermost pair: the size a probe really is.
        lo, hi = amps.get(-probe), amps.get(probe)
        # Signed, so the part every channel shares can be found and removed
        # below; the grade uses the magnitude of what is left.
        asym = ((hi + lo) / (0.5 * (abs(hi) + abs(lo)))
                if lo is not None and hi is not None and (hi or lo)
                else float("nan"))
        offsets = sorted(amps)
        slopes = [(amps[b] - amps[a]) / (b - a)
                  for a, b in zip(offsets, offsets[1:]) if b > a]
        # The first segment starts at a turning point, where even a healthy
        # piezo lags; it is the rest of the ramp that shows a real dead zone.
        slopes = slopes[1:] or slopes
        floor = (min(slopes) / float(np.median(slopes))
                 if slopes and np.median(slopes) else float("nan"))
        # How far this column stands above the noise in the units the SVD
        # ranks modes by; this decides whether the actuator is in the matrix.
        norm = float(np.linalg.norm(gC[:, i]))
        snr = (norm / noise_sv[c] if np.isfinite(noise_sv[c])
               and noise_sv[c] > 0 else float("nan"))
        if not np.isfinite(snr):
            inv_snr = float("nan")
        else:
            inv_snr = 1.0 / snr if snr > 0 else float("inf")
        # What one column leaves unexplained over every state this channel was
        # measured in, both legs and every amplitude at once -- the direct
        # answer to "does this column describe this actuator".
        noise_nm = _weighted_noise(ref_diffs, weight)
        resid_nm = float("nan")
        if resid is not None:
            # One row per measured offset, and the same footprint under all of
            # them, so every row is judged over the same points.
            spread = (np.broadcast_to(weight, resid.shape)
                      if weight is not None else None)
            resid_nm = INF.weighted_rms(resid, spread)
        ratio = (resid_nm / noise_nm if np.isfinite(noise_nm) and noise_nm > 0
                 else float("nan"))
        # The same stroke Step 4 quotes: this actuator's own motion over its
        # widest pair, over its own footprint.
        stroke_nm = 2 * pairs[c][-1][1] * INF.weighted_rms(column, weight)
        frac = (resid_nm / stroke_nm if np.isfinite(stroke_nm) and stroke_nm > 0
                else float("nan"))
        # Nothing to explain: a residual sitting on the repeat noise is the
        # bench, whatever fraction of the stroke it happens to be.
        if np.isfinite(ratio) and ratio < 3.0:
            frac = 0.0
        measured[c] = dict(asym=asym, floor=floor, snr=snr, inv_snr=inv_snr,
                           resid_nm=resid_nm, ratio=ratio, frac=frac)

    # A probe asymmetry every channel shares is the ladder's approach
    # direction, not one actuator misbehaving; the median removes it first.
    signed = [m["asym"] for m in measured.values() if np.isfinite(m["asym"])]
    common = float(np.median(signed)) if signed else 0.0
    grades = {}
    for c in channels:
        m = measured[c]
        asym = abs(m["asym"] - common)
        grade, reasons = _verdict(asym, m["floor"], m["inv_snr"], m["frac"])
        grades[c] = grade
        step.numbers.append(
            (f"ch{c}",
             f"probe asym {_pct(asym):>7}   min gain {_pct(m['floor']):>7}   "
             f"column {_times(m['snr']):>8} noise   "
             f"residual {m['resid_nm']:5.2f} nm "
             f"({_pct1(m['frac'])} of stroke, {_times(m['ratio'])} noise)"
             f"   -> {grade}"
             + (f"  [{', '.join(reasons)}]" if reasons else "")))
    step.numbers.append(
        ("probe asymmetry common to every channel, removed before grading",
         f"{100 * common:+.0f}% (the ladder's own approach direction)"))
    step.numbers.append(
        ("repeat-to-repeat noise the residual is shown against",
         f"{_weighted_noise(ref_diffs, None):.2f} nm over the whole pupil"
         if ref_diffs else "unknown (one ref only)"))
    worst = [c for c in channels if grades[c] == "BAD"]
    watch = [c for c in channels if grades[c] == "watch"]
    if worst:
        step.note(
            "these actuators fail at least one measure outright: "
            + ", ".join(f"ch{c}" for c in worst)
            + ". The matrix still describes the mirror on average, but a "
              "three-point modal solve probing them will not get the "
              "amplitude it asked for.", trace.WARN)
    if watch:
        step.info("borderline, worth watching rather than acting on: "
                  + ", ".join(f"ch{c}" for c in watch))
    if not worst and not watch:
        step.info("every actuator passes all four measures.")
    return grades


def _pct(v):
    """A ratio as a percentage, or a dash when it could not be measured."""
    return "--" if not np.isfinite(v) else f"{100 * v:.0f}%"


def _pct1(v):
    """A small ratio as a percentage, kept to one decimal.

    A residual worth a few tenths of a percent of the stroke is the answer
    "this column is right"; rounded to whole percent it prints as 0% and reads
    as a number that was never measured.
    """
    return "--" if not np.isfinite(v) else f"{100 * v:.1f}%"


def _times(v):
    """A ratio as a multiple, or a dash when it could not be measured."""
    return "--" if not np.isfinite(v) else f"{v:.1f}x"


def _check_superposition(tr, job, grids, checks, bias, channels, c_rad, ref_z,
                         inside, Xin, Yin, ref_diffs):
    """Predict each validation surface from C and report the residual.

    `as commanded` uses the bits actually sent and is the end-to-end
    open-loop error. `columns refitted` solves for the amplitudes that best
    explain the surface and asks only whether the columns add; a gap between
    the two is a command-scale problem, not a matrix problem.
    """
    step = tr.add("Superposition check",
                  formula="as commanded: rms(measured - C @ (v - bias)) / "
                          "rms(measured);   columns refitted: "
                          "rms(measured - C @ argmin) / rms(measured)",
                  why="The modal solve, the mode ranking and the truncation "
                      "assume the columns simply ADD; they do not assume the "
                      "mirror lands on the bits it was given. Those are two "
                      "different claims and one number cannot test both, so "
                      "the amplitudes are refitted as well: the gap between "
                      "the two is command-scale error, and what survives the "
                      "refit is the only part that says the columns are wrong. "
                      "The relative error also divides by how big the check "
                      "command happened to be, so a small command reads as a "
                      "bad model; the residual against the noise says whether "
                      "there is anything there to explain.")
    if not checks:
        step.note(
            "no validation surfaces, so the linear model is untested. Measure "
            "the mirror under about ten random channel combinations, add them "
            "as 'check', and read the error here -- under 10% means the model "
            "holds.", trace.WARN)
        return
    scale = INF.PHASE_PER_NM / float(job.laser_nm)
    # Same units as the residual: the model can never do better than this, so
    # a residual sitting on it means the model is as right as the bench can
    # show, whatever the percentage says.
    noise_rad = _weighted_noise(ref_diffs, None) * scale
    errors, ratios, fits, delivered = [], [], [], []
    for e in checks:
        offsets = np.array([e.bits.get(c, bias.get(c, 0)) - bias.get(c, 0)
                            for c in channels], float)
        check_rows = grids[e.path][1][inside]
        n_valid = int(check_rows.sum())
        if n_valid < 20:
            step.numbers.append(
                (e.name, f"not comparable ({n_valid} common pupil points)")
            )
            step.note(
                f"{e.name}: too few pixels overlap its own valid mask and the "
                "calibration pupil; it was not used to change the calibration "
                "mask."
            , trace.WARN)
            continue
        check_inside = inside & grids[e.path][1]
        measured = (
            grids[e.path][0][check_inside] - ref_z[check_inside]
        ) * scale
        if job.remove_piston_tilt:
            measured, _ = INF.remove_piston_tilt(
                measured,
                Xin[check_rows],
                Yin[check_rows],
            )
        err, _pred = INF.superposition_error(
            c_rad[check_rows],
            offsets,
            measured,
        )
        errors.append(err)
        # The same surface with the nine amplitudes free. This is the span
        # test: what is left cannot be reached by any command at all, so it is
        # the only part that is the columns' fault.
        A = c_rad[check_rows]
        got, *_ = np.linalg.lstsq(A, measured, rcond=None)
        scale_m = float(np.sqrt(np.mean(measured ** 2)))
        fit_err = (float(np.sqrt(np.mean((measured - A @ got) ** 2)) / scale_m)
                   if scale_m > 0 else float("nan"))
        fits.append(fit_err)
        delivered.append((offsets, got))
        # rms(measured) is what err was divided by, so multiplying it back out
        # recovers the residual itself, which is what the noise can be
        # compared with.
        resid = err * scale_m
        ratio = (resid / noise_rad if np.isfinite(err) and np.isfinite(
            noise_rad) and noise_rad > 0 else float("nan"))
        ratios.append(ratio)
        value = (f"as commanded {100 * err:.1f}%"
                 + (f" ({_times(ratio)} noise)" if np.isfinite(ratio) else "")
                 + f",  columns refitted {100 * fit_err:.1f}%"
                 + f"  over {n_valid} points"
                 if np.isfinite(err) else "not comparable")
        step.numbers.append((e.name, value))
    good = [e for e in errors if np.isfinite(e)]
    if good:
        med = float(np.median(good))
        seen = [r for r in ratios if np.isfinite(r)]
        med_ratio = float(np.median(seen)) if seen else float("nan")
        med_fit = float(np.median([f for f in fits if np.isfinite(f)])) \
            if any(np.isfinite(f) for f in fits) else float("nan")
        step.numbers.append(
            ("median over all checks",
             f"as commanded {100 * med:.1f}%"
             + (f" ({_times(med_ratio)} noise)" if np.isfinite(med_ratio)
                else "")
             + f",  columns refitted {100 * med_fit:.1f}%"))
        _delivered_vs_commanded(step, channels, delivered)
        columns_hold = np.isfinite(med_fit) and med_fit <= 0.10
        # The gate is the GAP between the two numbers, not one threshold on the
        # open-loop error. A 20% miss that refits to 1% is a drive fault worth
        # naming, and the old 30% cliff let it pass in silence.
        if columns_hold and med > 0.10 and med > 3.0 * med_fit:
            step.note(
                f"the columns hold: with the amplitudes refitted only "
                f"{100 * med_fit:.1f}% is left, so the measured surfaces ARE "
                "in the span of the matrix, and the modes, the SVD and the "
                f"truncation are sound. The {100 * med:.0f}% is a command-scale "
                "fault, not a matrix fault -- the mirror does not land on the "
                "bits it is given. Fix it in the drive (hysteresis "
                "compensation, approach direction, a dead leg); the "
                "per-channel gains just above name which channel to fix.",
                trace.WARN)
        elif columns_hold:
            step.info(
                f"the linear model predicts an arbitrary command to "
                f"{100 * med:.1f}% and the columns refit to {100 * med_fit:.1f}%"
                ". Superposition holds and the matrix is usable as it stands.")
        elif med > 0.30 and (not np.isfinite(med_ratio) or med_ratio > 3.0):
            step.note(
                f"the linear model misses {100 * med:.0f}% of the measured "
                f"change and refitting the amplitudes only brings it to "
                f"{100 * med_fit:.0f}%, so the surfaces are not in the span of "
                "the columns. Suspect a push-pull amplitude far from the "
                "working range or a mis-placed pupil -- do not trust the "
                "modes until this comes down.", trace.STOP)
        elif med > 0.30:
            step.note(
                f"the {100 * med:.0f}% is measured against check commands so "
                "small that what the model misses is only the measurement "
                "noise. The model is not shown to be wrong -- it is untested. "
                "Re-run the checks with combinations spanning the range the "
                "loop will drive.", trace.WARN)


def _delivered_vs_commanded(step, channels, delivered):
    """Per-channel drive gain read off the validation surfaces.

    The refit says how far each actuator really went on commands the matrix was
    not built from. Regressing that on what was asked for separates the two
    ways a channel can be mis-driven: a slope away from one is a wrong scale,
    and scatter about the line is the approach-direction hysteresis left over
    after compensation. Neither is visible in the pupil-wide percentage, and
    both name the channel to go and fix.
    """
    if len(delivered) < 3:
        return
    cmd = np.array([d[0] for d in delivered])
    got = np.array([d[1] for d in delivered])
    parts = []
    for i, c in enumerate(channels):
        A = np.column_stack([cmd[:, i], np.ones(len(cmd))])
        coef, *_ = np.linalg.lstsq(A, got[:, i], rcond=None)
        spread = float(np.std(got[:, i] - A @ coef))
        parts.append(f"ch{c} {coef[0]:.2f}x +/-{spread:.0f}")
    step.numbers.append(
        ("delivered per commanded bit, from the refits (slope +/- scatter)",
         "  ".join(parts)))


def _summarise(tr, res: Result, modes: INF.Modes):
    """Close the trace with what to do with the numbers."""
    step = tr.add("Result",
                  why="What the mirror can do, and how to use it.")
    step.numbers += [
        ("actuators measured", str(len(res.channels))),
        ("usable modes after truncation", str(res.keep)),
        ("singular values",
         "  ".join(f"{v:.3g}" for v in modes.s[:min(modes.n, 9)])),
    ]
    step.note(
        "Correcting mode i means adding amplitude a_i (in rad) times column i "
        "of ctrl to the bias bits. The three-point solve is "
        "a_corr = -b (G+ - G-) / (2 G+ - 4 G0 + 2 G-), where G is the "
        "RECIPROCAL of the PSD metric (Debarre Eq. 30 and Eq. 33) or the "
        "masked-detector signal used as it stands.", trace.INFO)
    if res.keep < 3:
        step.note(
            f"only {res.keep} mode(s) stand above the noise. That is a "
            "measurement problem before it is a mirror problem: average more "
            "Zygo frames per point, or drive a larger push-pull amplitude.", trace.WARN)
