# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 2.1, 2026-08-16

"""Figures from a finished or partial dm_sweep directory.

Reads only the CSVs, so a session can be re-plotted at any time:

    python -m dm_toolkit.bench.sweep_plots <dir>
    python -m dm_toolkit.bench.sweep_plots --decide <dm_sweep root> [out dir]

The x axis is logarithmic, the two averaging modes get a panel each, the y
range comes from the data, and raw scores get one panel per metric.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from ..correction import budget as BU

# One colour per scoring function, used by every figure so a colour means the
# same thing across the whole folder.
COLOURS = {"peak": "#3b78e7", "psd_band": "#d93025", "rms": "#188038",
           "pib": "#e37400", "sharpness": "#8430ce",
           "r_ee80": "#c5221f", "second_moment": "#007b83"}
# avg_frame is what the loop actually does, so it leads.
MODE_MARKER = {"avg_frame": "o", "avg_score": "^"}
MODE_LABEL = {"avg_frame": "avg_frame (average frames, then score)",
              "avg_score": "avg_score (score each frame, then average)"}
# Inside a metric's panel the curves are the two modes, so they -- not the
# metric -- need distinguishing colours. The panel title carries the metric
# colour, so the folder's colour code still holds.
MODE_COLOUR = ("#1a1a1a", "#c26c00")

DPI = 200
SCATTER_ALPHA = 0.16


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _colour(metric, i=0):
    return COLOURS.get(metric, f"C{i}")


def load_scores(directory):
    """Rows of sweep_scores.csv, or [] when there are none."""
    return load_csv(directory, "sweep_scores.csv")


def load_csv(directory, name):
    path = Path(directory) / name
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save(fig, directory, name):
    fig.savefig(Path(directory) / "figs" / name, dpi=DPI,
                bbox_inches="tight", facecolor="white")


def _xcol(sweep):
    return "delay_achieved_ms" if sweep == "settle" else "n_frames"


def _keycol(sweep):
    return "delay_nominal_ms" if sweep == "settle" else "n_frames"


class Point(NamedTuple):
    """One sweep point, summarised over the rounds that measured it."""
    key: float      # Nominal value: what identifies the point across rounds.
    x: float        # Mean achieved x of the kept rounds.
    y: float        # Mean score of the kept rounds.
    std: float      # Spread of the kept rounds -- what the error bar shows.
    sem: float      # Uncertainty of the mean; used to judge drift, not drawn.
    n: int          # Rounds kept.
    xs: np.ndarray  # Kept rounds, for the scatter.
    ys: np.ndarray
    xr: np.ndarray  # Rejected rounds, drawn apart so the cut is visible.
    yr: np.ndarray


# Robust rejection: a round is an outlier when it sits more than this many
# robust sigmas from the point's median. MAD rather than std, because the
# outlier being tested for is itself in the std that would test it.
REJECT_SIGMA = 3.0
REJECT_MAX_FRAC = 0.3  # Never throw away more than this share of a point.


def _keep(ys):
    """Mask of the rounds to trust at one sweep point.

    Args:
        ys: Every round's score at this point.

    Returns:
        Boolean mask, all True when the spread is degenerate or the rule would
        cut away more than `REJECT_MAX_FRAC` -- a point that noisy is telling
        us something, and silently dropping a third of it would hide that.
    """
    ys = np.asarray(ys, float)
    if len(ys) < 4:
        return np.ones(len(ys), bool)
    med = float(np.median(ys))
    sigma = 1.4826 * float(np.median(np.abs(ys - med)))
    if not np.isfinite(sigma) or sigma <= 0:
        return np.ones(len(ys), bool)
    keep = np.abs(ys - med) <= REJECT_SIGMA * sigma
    if keep.sum() < (1.0 - REJECT_MAX_FRAC) * len(ys):
        return np.ones(len(ys), bool)
    return keep


def _points(rows, sweep, metric, mode, ycol):
    """Every sweep point of one curve, as `Point` records.

    Binned on the NOMINAL value because that is what identifies a sweep point
    across rounds; the reported x is the mean ACHIEVED value of the bin, which
    is where the measurement really happened.
    """
    xcol, keycol = _xcol(sweep), _keycol(sweep)
    bins = {}
    for r in rows:
        if (r["sweep"] != sweep or r["metric"] != metric
                or r["score_mode"] != mode):
            continue
        key, x, y = _f(r[keycol]), _f(r[xcol]), _f(r[ycol])
        if not (np.isfinite(key) and np.isfinite(x) and np.isfinite(y)):
            continue
        bins.setdefault(key, ([], []))
        bins[key][0].append(x)
        bins[key][1].append(y)
    out = []
    for key in sorted(bins):
        xs, ys = np.array(bins[key][0]), np.array(bins[key][1])
        keep = _keep(ys)
        xk, yk = xs[keep], ys[keep]
        n = len(yk)
        if not n:
            continue
        std = float(np.std(yk, ddof=1)) if n > 1 else 0.0
        out.append(Point(key, float(np.mean(xk)), float(np.mean(yk)), std,
                         std / np.sqrt(n), n, xk, yk, xs[~keep], ys[~keep]))
    return out


def _values_in(rows, column):
    seen = []
    for r in rows:
        if r[column] not in seen:
            seen.append(r[column])
    return seen


def _modes_identical(rows, sweep, ycol, tol=1e-9):
    """True when the averaging modes cannot differ, so one panel will do.

    They are the same operation whenever a single frame is averaged, and
    drawing that twice hides the real curves behind a duplicate.
    """
    modes = _values_in(rows, "score_mode")
    if len(modes) < 2:
        return True
    for metric in _values_in(rows, "metric"):
        ref = None
        for mode in modes:
            got = {p.key: p.y for p in _points(rows, sweep, metric, mode, ycol)}
            if ref is None:
                ref = got
                continue
            keys = set(ref) & set(got)
            if not keys or max(abs(ref[k] - got[k]) for k in keys) > tol:
                return False
    return True


def _fit_settle(x, y):
    """Fit y = a - b*exp(-t/tau) by a coarse tau scan plus a linear solve.

    A grid over tau turns the fit into a linear least squares in (a, b) at
    each tau, which needs no optimiser and cannot wander off to a nonsense
    minimum the way a free three-parameter fit does on ten noisy points.

    Args:
        x: Delay, in milliseconds.
        y: Score.

    Returns:
        (a, b, tau_ms) or None when there is not enough usable data.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4 or np.ptp(x) <= 0:
        return None
    best = None
    for tau in np.geomspace(max(np.ptp(x) / 200.0, 1e-3), np.ptp(x) * 5.0, 300):
        basis = np.column_stack([np.ones_like(x), -np.exp(-x / tau)])
        try:
            coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = float(np.sum((basis @ coef - y) ** 2))
        if best is None or resid < best[0]:
            best = (resid, float(coef[0]), float(coef[1]), float(tau))
    return None if best is None else (best[1], best[2], best[3])


def _fit_frames(n, y):
    """Fit y = a - b/sqrt(N): the shape pure noise averaging would give.

    Linear in (a, b) once 1/sqrt(N) is the regressor, so there is nothing to
    iterate. Departure from this curve is what the extra waiting time bought,
    which is exactly what a cumulative window makes visible.

    Args:
        n: Frames averaged.
        y: Score.

    Returns:
        (a, b) or None when there is not enough usable data.
    """
    n, y = np.asarray(n, float), np.asarray(y, float)
    ok = np.isfinite(n) & np.isfinite(y) & (n > 0)
    n, y = n[ok], y[ok]
    if len(n) < 3:
        return None
    basis = np.column_stack([np.ones_like(n), -1.0 / np.sqrt(n)])
    try:
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return float(coef[0]), float(coef[1])


def _fit_is_useful(x, y, model, curve, err):
    """Whether a fitted curve is worth drawing at all.

    A fit earns its line when it explains materially more than a flat line
    does and stays inside the range of the data it describes.

    Args:
        x: Point positions.
        y: Point means.
        model: The fit evaluated AT the points.
        curve: The fit evaluated across the drawn grid.
        err: Per-point spread, the scale that decides what "flat" means.

    Returns:
        bool: True when the curve should be drawn.
    """
    if model is None or not np.all(np.isfinite(curve)):
        return False
    flat = float(np.sum((y - np.mean(y)) ** 2))
    resid = float(np.sum((model - y) ** 2))
    if flat <= 0:
        return False
    # A real trend cuts the residual substantially AND is bigger than the
    # error bars it has to rise above.
    span = float(np.ptp(y))
    typical = float(np.median(err)) if len(err) else 0.0
    if resid > 0.6 * flat or span < 1.5 * typical:
        return False
    lo, hi = float(np.min(y)), float(np.max(y))
    pad = max(span, typical) * 1.5 + 1e-9
    return bool(np.min(curve) >= lo - pad and np.max(curve) <= hi + pad)


def _draw_curve(ax, rows, sweep, metric, mode, ycol, colour, label):
    """One metric's scatter, means, error bars and fit on `ax`.

    The fit result goes into the curve's own legend entry. Annotating it in
    the right margin instead forces the canvas wide enough to hold the text
    and leaves the plot itself occupying half the figure.

    Returns:
        bool: True when anything was drawn.
    """
    pts = _points(rows, sweep, metric, mode, ycol)
    if not pts:
        return False
    marker = MODE_MARKER.get(mode, "o")
    for p in pts:
        ax.scatter(p.xs, p.ys, s=9, color=colour, alpha=SCATTER_ALPHA,
                   marker=marker, linewidths=0, zorder=2)
        if len(p.xr):  # Rejected rounds stay on the plot, marked as cut.
            ax.scatter(p.xr, p.yr, s=22, color=colour, alpha=0.55, marker="x",
                       linewidths=1.0, zorder=2)
    mx = np.array([p.x for p in pts])
    my = np.array([p.y for p in pts])
    err = np.array([p.std for p in pts])
    n_cut = sum(len(p.xr) for p in pts)
    # Only across the measured span: a model's extrapolation is not data and
    # must never set the y range.
    grid = np.geomspace(max(mx.min(), 1e-6), mx.max(), 400)
    note = ""
    if sweep == "settle":
        fit = _fit_settle(mx, my)
        if fit is not None:
            a, b, tau = fit
            curve = a - b * np.exp(-grid / tau)
            model = a - b * np.exp(-mx / tau)
            t95 = tau * np.log(20.0) if b > 0 else float("nan")
            note = (f"tau {tau:.0f} ms" + (f", 95% @ {t95:.0f} ms"
                                           if np.isfinite(t95) else ""))
        else:
            curve = model = None
    else:
        fit = _fit_frames(mx, my)
        if fit is not None:
            a, b = fit
            curve = a - b / np.sqrt(grid)
            model = a - b / np.sqrt(mx)
            note = f"a {a:.3f}, b {b:+.3f}"
        else:
            curve = model = None
    if curve is not None and _fit_is_useful(mx, my, model, curve, err):
        ax.plot(grid, curve, "-", color=colour, lw=1.4, alpha=0.85, zorder=3)
    else:
        # Saying so is the result. Forcing a curve through flat, noisy points
        # produced an asymptote far outside the data and a line that walked
        # off the bottom of the axes.
        note = "flat within the round-to-round spread -- no fit"
    # The bar is the SPREAD of the rounds, not the uncertainty of their mean:
    # the scatter behind it is what the bar has to describe, and std/sqrt(n)
    # would draw a bar four times smaller than the cloud it sits in.
    if n_cut:
        label += f" [{n_cut} cut]"
    ax.errorbar(mx, my, yerr=err, fmt=marker, color=colour, ms=4.5,
                mew=0.9, mfc="white", lw=0, elinewidth=1.1, capsize=2.5,
                zorder=4, label=f"{label}  ({note})" if note else label)
    return True


def _finish_axes(ax, rows, sweep, ycol, all_y):
    """Log x, data-driven y limits and the axis furniture."""
    ax.set_xscale("log")
    if sweep == "settle":
        ax.set_xlabel("achieved settle delay after the Pi ack (ms)")
        # Plain numbers instead of 10^2. Set here, not after the frames
        # branch below: installing a formatter replaces the fixed labels that
        # branch just chose, which is how 20/24/28/32 came back.
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        ax.set_xlabel("frames averaged (cumulative window)")
        ticks = sorted({int(_f(r["n_frames"])) for r in rows
                        if r["sweep"] == "frames"})
        if ticks:
            # Every value still gets a tick, but only label the ones far
            # enough apart in LOG space to be legible: 20/24/28/32 collide at
            # the right-hand end of a log axis and read as one smear.
            labels, last = [], None
            for i, t in enumerate(ticks):
                keep = (i in (0, len(ticks) - 1) or last is None
                        or np.log10(t / last) > 0.11)
                labels.append(str(t) if keep else "")
                if keep:
                    last = t
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
    if all_y:
        lo, hi = min(all_y), max(all_y)
        pad = max((hi - lo) * 0.08, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
    ax.grid(alpha=0.22, which="major")
    ax.grid(alpha=0.10, which="minor")


def fig_sweep(rows, sweep, directory, ycol="score_norm"):
    """One figure for one sweep: a panel per scoring function.

    One panel per metric, never shared axes; within a panel the two
    averaging modes are the curves.

    Args:
        rows: sweep_scores.csv rows.
        sweep: "settle" or "frames".
        directory: Session folder.
        ycol: "score_norm" or "score_raw".

    Returns:
        int: 1 when a figure was written, else 0.
    """
    sub = [r for r in rows if r["sweep"] == sweep]
    if not sub:
        return 0
    metrics = _values_in(sub, "metric")
    modes = _values_in(sub, "score_mode")
    # Identical modes would draw one curve twice; keep the loop's own.
    collapsed = _modes_identical(sub, sweep, ycol)
    drawn = modes[:1] if collapsed else modes

    fig = Figure(figsize=(5.4 * len(metrics), 5.0), layout="constrained")
    axes = fig.subplots(1, len(metrics), squeeze=False)[0]
    for ax, metric in zip(axes, metrics):
        all_y = []
        for i, mode in enumerate(drawn):
            colour = _colour(metric, i) if len(drawn) == 1 else MODE_COLOUR[i]
            # Short label: the full "average frames, then score" wording is in
            # the suptitle, and in the legend it stretches the box across the
            # panel and sits on top of the data it is describing.
            if not _draw_curve(ax, sub, sweep, metric, mode, ycol, colour,
                               mode):
                continue
            for p in _points(sub, sweep, metric, mode, ycol):
                all_y.extend([p.y - p.std, p.y + p.std])
                all_y.extend(p.ys.tolist())
        _finish_axes(ax, sub, sweep, ycol, all_y)
        ax.set_title(metric, fontsize=11, color=_colour(metric, 0))
        # "best" so the box lands where the data is not; a fixed corner sat on
        # top of the very points it was labelling.
        ax.legend(fontsize=7.5, loc="best", framealpha=0.88)
    axes[0].set_ylabel("norm_score (0.5 = parked state)" if ycol == "score_norm"
                       else "raw score")
    head = ("Settle time vs score" if sweep == "settle"
            else "Frames averaged vs score")
    head += (" -- every point cut from ONE step transient per round\n"
             "dots = the individual rounds, x = rejected as an outlier "
             f"(>{REJECT_SIGMA:g} robust sigma), bar = std of the rest\n"
             "avg_frame = average the frames then score (what the loop "
             "does)   |   avg_score = score each frame then average")
    if collapsed and len(modes) > 1:
        head += ("\navg_frame and avg_score coincide here: with one frame "
                 "averaged they are the same operation")
    fig.suptitle(head, fontsize=10)
    _save(fig, directory, f"{sweep}_vs_{ycol}.png")
    return 1


def fig_drift(directory, rows=None):
    """Parked-state score against round number, judged against the error bars.

    A trend here means the membrane has not relaxed between rounds, so the
    error bars on the sweep figures are absorbing a systematic drift rather
    than measuring repeatability. That makes this the figure to read first,
    and the verdict in the title is the whole point of it.
    """
    park = load_csv(directory, "park_scores.csv")
    if not park:
        return 0
    rows = load_scores(directory) if rows is None else rows
    fig = Figure(figsize=(8.4, 4.6), layout="constrained")
    ax = fig.add_subplot(111)
    verdicts, steps = [], []
    for i, metric in enumerate(_values_in(park, "metric")):
        xs = np.array([_f(r["round"]) for r in park if r["metric"] == metric])
        ys = np.array([_f(r["score_norm"]) for r in park
                       if r["metric"] == metric])
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        if len(xs) < 2:
            continue
        colour = _colour(metric, i)
        ax.plot(xs, ys, "o-", color=colour, ms=4, lw=1.1, alpha=0.9,
                label=metric)
        # Round 0's parked reading IS the session reference, so it is 0.5 by
        # construction, not by measurement. Fitting through it turns the
        # definition into a slope and roughly doubles the reported drift.
        fit_x, fit_y = xs[1:], ys[1:]
        drift = float("nan")
        if len(fit_x) > 2 and np.ptp(fit_x) > 0:
            slope, intercept = np.polyfit(fit_x, fit_y, 1)
            ax.plot(fit_x, slope * fit_x + intercept, ":", color=colour,
                    lw=1.2, alpha=0.8)
            drift = abs(slope) * np.ptp(fit_x)
        if len(ys) > 1:  # The one-off step the first step-and-return costs.
            steps.append((metric, ys[1] - ys[0]))
        # The scale that decides whether it matters: the typical error bar the
        # sweep figures draw. Drift much larger than that is being counted as
        # random scatter when it is a trend.
        sems = [p.sem for mode in _values_in(rows, "score_mode")[:1]
                for p in _points(rows, "settle", metric, mode, "score_norm")]
        sem = float(np.median(sems)) if sems else float("nan")
        if np.isfinite(drift) and np.isfinite(sem) and sem > 0:
            verdicts.append((metric, drift, sem, drift / sem))
    ax.annotate("round 0 is the reference itself\n(0.5 by definition, not "
                "measured -- excluded from the trend)",
                (0, 0.5), fontsize=7.5, color="#666", ha="left", va="bottom",
                xytext=(8, 8), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
    ax.set_xlabel("round")
    ax.set_ylabel("norm_score at the parked state")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, loc="lower right")
    worst = max((v[3] for v in verdicts), default=float("nan"))
    n_rounds = len(_values_in(park, "round"))
    if np.isfinite(worst) and worst > 3.0:
        head = (f"Round-to-round drift at the park bit: NOT SETTLED "
                f"(worst trend is {worst:.0f}x the sweep's own error bar)")
        note = ("The membrane does not return to the same state between "
                f"rounds. Raise Park hold and rerun, or read the sweep as one "
                f"drifting session rather than {n_rounds} repeats.")
    else:
        head = "Round-to-round drift at the park bit: flat enough"
        note = "The trend is within the sweep's own error bars."
    detail = "  |  ".join(f"{m}: trend {d:.4f} vs SEM {s:.4f} ({r:.0f}x)"
                          for m, d, s, r in verdicts)
    if steps:  # Reported apart: a one-off offset is not a running trend.
        detail += ("\nfirst step-and-return offset (round 0 -> 1): "
                   + ", ".join(f"{m} {v:+.4f}" for m, v in steps))
    fig.suptitle(head, fontsize=11.5)
    ax.set_title(note + ("\n" + detail if detail else ""), fontsize=7.5,
                 color="#444")
    _save(fig, directory, "drift_vs_round.png")
    return 1


def _asd(t, s):
    """Amplitude spectral density of an unevenly stamped series.

    Resampled onto the median interval first: the score samples carry real
    timestamps and the chunked recording can drop a frame, so an FFT on the
    raw index would put energy at the wrong frequency.

    Args:
        t: Sample times, in seconds.
        s: Sample values.

    Returns:
        (freq_hz, asd) or (None, None) when the series is too short.
    """
    t, s = np.asarray(t, float), np.asarray(s, float)
    ok = np.isfinite(t) & np.isfinite(s)
    t, s = t[ok], s[ok]
    if len(t) < 16 or np.ptp(t) <= 0:
        return None, None
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return None, None
    grid = np.arange(t[0], t[-1], dt)
    y = np.interp(grid, t, s)
    y = y - y.mean()
    win = np.hanning(len(y))
    # Window power correction, so the level is the signal's, not the window's.
    norm = np.sqrt(np.sum(win ** 2) * dt)
    spec = np.fft.rfft(y * win)
    freq = np.fft.rfftfreq(len(y), dt)
    return freq[1:], np.abs(spec[1:]) / max(norm, 1e-12)


def fig_noise(directory, rows=None):
    """Score noise: the time series, its ASD and what averaging N should buy."""
    noise = load_csv(directory, "noise.csv")
    if not noise:
        return 0
    rows = load_scores(directory) if rows is None else rows
    fig = Figure(figsize=(13.0, 4.4), layout="constrained")
    ax_t, ax_f, ax_n = fig.subplots(1, 3)
    nyquist = float("nan")
    for i, metric in enumerate(_values_in(noise, "metric")):
        colour = _colour(metric, i)
        t = np.array([_f(r["t_s"]) for r in noise if r["metric"] == metric])
        s = np.array([_f(r["score_norm"]) for r in noise
                      if r["metric"] == metric])
        ok = np.isfinite(t) & np.isfinite(s)
        t, s = t[ok], s[ok]
        if len(t) < 4:
            continue
        sigma = float(np.std(s, ddof=1))
        ax_t.plot(t, s, "-", color=colour, lw=0.6, alpha=0.8,
                  label=f"{metric}  sigma {sigma:.5f}")
        freq, asd = _asd(t, s)
        if freq is not None:
            ax_f.loglog(freq, asd, "-", color=colour, lw=0.7, alpha=0.8,
                        label=metric)
            nyquist = float(freq[-1])
        ns = np.arange(1, 33)
        ax_n.plot(ns, sigma / np.sqrt(ns), "-", color=colour, lw=1.4,
                  label=f"{metric}: sigma/sqrt(N)")
        # What the frames sweep actually measured, for comparison: if the two
        # disagree the extra frames are buying settle, not noise reduction.
        pts = _points(rows, "frames", metric, "avg_frame", "score_norm")
        if len(pts) > 1:
            ax_n.plot([p.x for p in pts], [p.std for p in pts], "o--",
                      color=colour, ms=4, lw=1.0, alpha=0.7,
                      label=f"{metric}: measured std")
    ax_t.set_xlabel("t (s)")
    ax_t.set_ylabel("norm_score")
    ax_t.set_title("score at the shape, one sample per frame", fontsize=10)
    ax_t.grid(alpha=0.22)
    ax_t.legend(fontsize=7)
    ax_f.set_xlabel("frequency (Hz)")
    ax_f.set_ylabel("ASD (score / sqrt(Hz))")
    title = "amplitude spectral density"
    if np.isfinite(nyquist):
        title += f"\nNyquist {nyquist:.1f} Hz -- anything faster is aliased"
    ax_f.set_title(title, fontsize=10)
    ax_f.grid(alpha=0.22, which="both")
    ax_f.legend(fontsize=7)
    ax_n.set_xlabel("frames averaged N")
    ax_n.set_ylabel("score sigma")
    ax_n.set_title("white-noise prediction vs what the sweep measured",
                   fontsize=10)
    ax_n.grid(alpha=0.22)
    ax_n.legend(fontsize=7)
    _save(fig, directory, "noise_asd.png")
    return 1


# Decision figures: fix settle_ms, frames_per_measure and score_mode from the
# whole dm_sweep tree (values in bench/plan.py MEASUREMENT_FIXED). Error bars
# are the standard deviation of the points drawn beside them.
DECIDE_COLOUR = {"dm_9": "#b2182b", "dm_5": "#2166ac"}
DECIDE_MARKER = {"dm_9": "o", "dm_5": "s"}
# Per-point cost model, t_point = FIXED + settle_ms + PER_FRAME*N, shared with
# correction.budget so the figures and the running loop agree.
DECIDE_FIXED_MS, DECIDE_PER_FRAME_MS = BU.FIXED_MS, BU.PER_FRAME_MS
DECIDE_CAP_WAIT_MS = 13.9   # settled ack -> first frame, measured
DECIDE_FRAME_MS = 28.72     # camera period, measured
# One 30 s recording drifted hard enough to put its Allan minimum at N=2 while
# the other four agree on 12-16. Named so the exclusion is a stated fact.
DECIDE_SKIP = ("max 128",)


def _decide_sessions(root, mirror):
    """Session folders of one mirror, minus the excluded recordings."""
    out = []
    for path in sorted(Path(root).joinpath(mirror).glob("*")):
        if path.is_dir() and not any(s in path.name for s in DECIDE_SKIP):
            out.append(path)
    return out


def _decide_rounds(root, mirror, sweep, metric, mode="avg_frame"):
    """{(session, round): {key: (achieved_ms, score_norm)}} for one sweep."""
    per = {}
    for path in _decide_sessions(root, mirror):
        for r in load_scores(path):
            if (r["sweep"] != sweep or r["metric"] != metric
                    or r["score_mode"] != mode):
                continue
            key = (_f(r["delay_nominal_ms"]) if sweep == "settle"
                   else int(r["n_frames"]))
            per.setdefault((path.name, int(r["round"])), {})[key] = (
                _f(r["delay_achieved_ms"]), _f(r["score_norm"]))
    return per


def _decide_allan(root, mirror, metric, taus):
    """Overlapping Allan deviation per session, one row each.

    Not the plain standard deviation: the question an optimiser asks is
    whether two measurements taken a second apart differ, and that is what the
    Allan deviation of adjacent block means answers. The plain sd keeps
    falling with N and would say "average forever".
    """
    out = []
    for path in _decide_sessions(root, mirror):
        rows = load_csv(path, "noise.csv")
        x = np.array([_f(r["score_norm"]) for r in rows
                      if r["metric"] == metric])
        if x.size < 4:
            continue
        row = []
        for m in taus:
            c = np.cumsum(np.insert(x, 0, 0.0))
            block = (c[m:] - c[:-m]) / m
            d = block[m:] - block[:-m]
            row.append(np.sqrt(0.5 * np.mean(d ** 2)) if d.size else np.nan)
        out.append(row)
    return np.array(out, float)


def speed_profile(root, mirror, metric="peak"):
    """The two curves the loop's automatic measure speed runs on.

    The quantities `fig_decide_settle` and `fig_decide_frames` draw, as
    numbers. Nothing is fitted or smoothed.

    Args:
        root: dm_sweep root holding dm_5/ and dm_9/.
        mirror: "dm_5" or "dm_9".
        metric: Scoring function; peak is the slowest of the three to settle.

    Returns:
        `(step, noise, noise_sem, sessions, source)`, or None when the tree
        holds no usable recording for this mirror. `step` maps the ACHIEVED
        delay in ms to the captured fraction of the step; `noise` maps frames
        to the Allan deviation of the score; `noise_sem` is that deviation's
        standard error across sessions.
    """
    per = _decide_rounds(root, mirror, "settle", metric)
    taus = [1, 2, 4, 8, 12, 16, 20, 24, 32]
    runs = _decide_allan(root, mirror, metric, taus)
    if not per or not runs.size:
        return None
    keys = sorted({k for m in per.values() for k in m})
    last = keys[-1]
    # The park state is the norm reference, so S(park) = 0.5 exactly and this
    # denominator is a real step size. Pooled over rounds so a noisy per-round
    # denominator cannot inflate the fraction.
    step_size = np.mean([m[last][1] for m in per.values()
                         if last in m]) - 0.5
    step = {}
    for k in keys:
        v = [1 - abs(m[k][1] - m[last][1]) / step_size
             for m in per.values() if k in m and last in m]
        if not v:
            continue
        t = int(round(np.mean([m[k][0] for m in per.values() if k in m])))
        step[t] = float(np.mean(v))
    med = np.median(runs, 0)
    sem = runs.std(0, ddof=1) / np.sqrt(runs.shape[0]) if runs.shape[0] > 1 \
        else np.zeros_like(med)
    noise, noise_sem = {}, {}
    for i, n in enumerate(taus):
        if np.isfinite(med[i]) and med[i] > 0:
            noise[n] = float(med[i])
            noise_sem[n] = float(sem[i]) if np.isfinite(sem[i]) else 0.0
    if not step or not noise:
        return None
    names = [p.name for p in _decide_sessions(root, mirror)]
    return step, noise, noise_sem, runs.shape[0], "; ".join(names)


def _decide_save(fig, out_dir, name):
    fig.savefig(Path(out_dir) / name, dpi=DPI, bbox_inches="tight",
                facecolor="white")
    return name


def fig_decide_settle(root, out_dir, metric="peak"):
    """Why settle_ms is what it is: the reading saturates.

    y is the fraction of the step's own score change the scored window sees,

        f(t) = 1 - |S_r(t) - S_r(inf)| / (S(inf) - 0.5),

    paired inside each round and divided by the pooled step. The park state
    is the norm reference, so S(park) = 0.5 exactly.

    Args:
        root: dm_sweep root holding dm_5/ and dm_9/.
        out_dir: Where to write the PNG.
        metric: Scoring function; peak is the slowest to settle of the three.

    Returns:
        The filename written.
    """
    fig = Figure(figsize=(6.6, 4.7), layout="constrained")
    ax = fig.add_subplot(111)
    for mirror in ("dm_9", "dm_5"):
        per = _decide_rounds(root, mirror, "settle", metric)
        if not per:
            continue
        keys = sorted({k for m in per.values() for k in m})
        last = keys[-1]
        step = np.mean([m[last][1] for m in per.values() if last in m]) - 0.5
        xs, ys, mean, sem, tx = [], [], [], [], []
        for k in keys:
            v = [100 * (1 - abs(m[k][1] - m[last][1]) / step)
                 for m in per.values() if k in m and last in m]
            t = np.mean([m[k][0] for m in per.values() if k in m])
            xs += [t] * len(v)
            ys += v
            tx.append(t)
            mean.append(np.mean(v))
            sem.append(np.std(v, ddof=1))
        colour = DECIDE_COLOUR[mirror]
        ax.plot(xs, ys, ".", ms=2.5, color=colour, alpha=SCATTER_ALPHA * 1.7,
                mec="none", zorder=1)
        ax.errorbar(tx, mean, yerr=sem, fmt="-", marker=DECIDE_MARKER[mirror],
                    ms=5, lw=1.6, capsize=3, elinewidth=1.2, capthick=1.2,
                    ecolor="0.15", color=colour, mfc="white", mew=1.3,
                    zorder=3,
                    label=f"DM{mirror[-1]}  (n = {len(per)} rounds)")
    ax.axhline(99, color="0.35", ls=":", lw=1.1)
    ax.set_xlim(0, 1100)
    ax.set_ylim(20, 103)
    ax.set_xlabel("delay before the scored frames (ms)")
    ax.set_ylabel("captured step response (%)")
    ax.legend(frameon=False, loc="lower right")
    return _decide_save(fig, out_dir, "decide_settle.png")


def fig_decide_frames(root, out_dir, metric="peak",
                      settle=(("dm_9", 500), ("dm_5", 300))):
    """Why frames_per_measure is what it is, in two plain quantities.

    Left: how noisy one measurement is. Right: how many measurements fit in
    a minute. Averaging buys the first only until drift takes over.

    Args:
        root: dm_sweep root holding dm_5/ and dm_9/.
        out_dir: Where to write the PNG.
        metric: Scoring function.
        settle: (mirror, settle_ms) the speed axis is computed at.

    Returns:
        The filename written.
    """
    taus = [1, 2, 4, 8, 12, 16, 20, 24, 32]
    fig = Figure(figsize=(6.8, 4.7), layout="constrained")
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    for mirror, settle_ms in settle:
        runs = 1e4 * _decide_allan(root, mirror, metric, taus)
        if not runs.size:
            continue
        colour = DECIDE_COLOUR[mirror]
        for row in runs:
            ax.plot(taus, row, ".", ms=2.5, color=colour,
                    alpha=SCATTER_ALPHA * 2, mec="none", zorder=1)
        ax.errorbar(taus, np.median(runs, 0),
                    yerr=runs.std(0, ddof=1), fmt="-",
                    marker=DECIDE_MARKER[mirror], ms=5, lw=1.8, capsize=3,
                    elinewidth=1.2, capthick=1.2, ecolor="0.15", color=colour,
                    mfc="white", mew=1.3, zorder=3,
                    label=f"DM{mirror[-1]} noise (left)")
        t_pt = (DECIDE_FIXED_MS + settle_ms
                + DECIDE_PER_FRAME_MS * np.array(taus, float)) / 1e3
        ax2.plot(taus, 60.0 / t_pt, "--", lw=1.4, color=colour, alpha=0.55,
                 zorder=2, label=f"DM{mirror[-1]} speed (right)")
    ax.set_xlim(0, 33)
    ax.set_ylim(0, 11)
    ax2.set_ylim(0, 150)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
    ax.set_xlabel("frames averaged per measurement, $N$")
    ax.set_ylabel("noise of one measurement  (x 1e-4 score units)")
    ax2.set_ylabel("measurements per minute")
    handles = ax.get_legend_handles_labels()
    extra = ax2.get_legend_handles_labels()
    ax.legend(handles[0] + extra[0], handles[1] + extra[1], frameon=False,
              loc="upper center", ncol=2)
    return _decide_save(fig, out_dir, "decide_frames.png")


def fig_decide_mode(root, out_dir, metric="peak"):
    """Why score_mode is avg_frame: the other one keeps a bias.

    y is (score of the averaged frames) - (mean of the per-frame scores), in
    units of that N's measurement noise. Averaging first removes the noise
    before the nonlinear metric sees it, so its bias falls as 1/N. One light
    line per session, because the bias scales with that day's spot noise.

    Args:
        root: dm_sweep root holding dm_5/ and dm_9/.
        out_dir: Where to write the PNG.
        metric: Scoring function.

    Returns:
        The filename written.
    """
    ns = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    fig = Figure(figsize=(6.6, 4.7), layout="constrained")
    ax = fig.add_subplot(111)
    for mirror in ("dm_9", "dm_5"):
        allan = _decide_allan(root, mirror, metric, ns)
        frame = _decide_rounds(root, mirror, "frames", metric, "avg_frame")
        score = _decide_rounds(root, mirror, "frames", metric, "avg_score")
        if not allan.size or not frame:
            continue
        sigma = np.median(allan, 0)
        colour = DECIDE_COLOUR[mirror]
        sessions, curves = sorted({k[0] for k in frame}), []
        for s in sessions:
            row = []
            for i, n in enumerate(ns):
                d = [(frame[k][n][1] - score[k][n][1]) / sigma[i]
                     for k in frame if k[0] == s and n in frame[k]
                     and k in score and n in score[k]]
                row.append(np.mean(d) if d else np.nan)
            curves.append(row)
            ax.plot(ns, row, "-", lw=0.8, color=colour, alpha=0.4, zorder=1)
        curves = np.array(curves, float)
        ax.errorbar(ns, np.nanmedian(curves, 0),
                    yerr=np.nanstd(curves, 0, ddof=1),
                    fmt="-", marker=DECIDE_MARKER[mirror], ms=5, lw=1.6,
                    capsize=3, elinewidth=1.2, capthick=1.2, ecolor="0.15",
                    color=colour, mfc="white", mew=1.3, zorder=3,
                    label=f"DM{mirror[-1]}  ({len(sessions)} sessions, "
                          f"{len(frame)} rounds)")
    ax.axhspan(-1, 1, color="0.86", zorder=0)
    ax.set_xlim(0, 33)
    ax.set_ylim(-3, 18)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
    ax.set_xlabel("frames averaged per measurement, $N$")
    ax.set_ylabel("bias of per-frame scoring  (measurement noises)")
    ax.legend(frameon=False, loc="lower right")
    return _decide_save(fig, out_dir, "decide_mode.png")


def generate_decision(root, out_dir=None):
    """The three figures behind MEASUREMENT_FIXED, from the whole tree.

    Args:
        root: dm_sweep root holding dm_5/ and dm_9/.
        out_dir: Where to write; defaults to `<root>/figs`.

    Returns:
        The filenames written.
    """
    out_dir = Path(out_dir or Path(root) / "figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return [fig_decide_settle(root, out_dir),
            fig_decide_frames(root, out_dir),
            fig_decide_mode(root, out_dir)]


def generate_all(directory):
    """Write every figure the folder has data for.

    Args:
        directory: A dm_sweep session folder.

    Returns:
        int: How many figures were written.
    """
    directory = Path(directory)
    (directory / "figs").mkdir(parents=True, exist_ok=True)
    rows = load_scores(directory)
    n = 0
    for sweep in ("settle", "frames"):
        for ycol in ("score_norm", "score_raw"):
            n += fig_sweep(rows, sweep, directory, ycol)
    n += fig_drift(directory, rows)
    n += fig_noise(directory, rows)
    return n


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--decide":
        written = generate_decision(sys.argv[2],
                                    sys.argv[3] if len(sys.argv) > 3 else None)
        print("\n".join(written))
    else:
        print(generate_all(sys.argv[1]), "figures written")
