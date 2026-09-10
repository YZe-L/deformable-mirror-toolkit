# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 2.7, 2026-08-16

"""Figures from a finished (or partial) bench directory."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# Knob columns per algorithm, in display order (subset of runs.csv columns)
ALGO_KNOBS = {
    "hill_climb": ["move_step", "min_step"],
    "spgd": ["spgd_perturb", "spgd_gain"],
    "genetic": ["ga_population", "ga_generations", "ga_mutation"],
    "cmaes": ["cma_sigma", "cma_popsize"],
    "bayes": ["bo_budget", "bo_init_points", "bo_xi"],
    "anneal": ["sa_t0", "sa_cooling", "sa_step"],
}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_runs(bench_dir):
    path = Path(bench_dir) / "runs.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r = dict(r)
            for k in ("settle_ms", "frames", "repeat", "points", "converged",
                      "t_converge_s", "t_total_s", "score_converged",
                      "score_best", "score_initial", "score_final",
                      "score_gain", "sigma_est", "noise_gate"):
                r[k] = _f(r.get(k))
            rows.append(r)
    return rows


def _score(r):
    """Bench-comparable final score; fall back for pre-feature CSV files."""
    v = _f(r.get("score_final"))
    if np.isfinite(v):
        return v
    converged = _f(r.get("score_converged"))
    return converged if np.isfinite(converged) else _f(r.get("score_best"))


def _initial(r):
    return _f(r.get("score_initial"))


def _gain(r):
    v = _f(r.get("score_gain"))
    if np.isfinite(v):
        return v
    final, initial = _score(r), _initial(r)
    return final - initial if np.isfinite(final) and np.isfinite(initial) \
        else float("nan")


SCORE_VIEWS = (
    ("Final score", _score),
    ("Initial score", _initial),
    ("Gain (final - initial)", _gain),
)


def _save(fig, bench_dir, name):
    out = Path(bench_dir) / "figs" / name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    return out.name


def _pareto(pts):
    """Upper-left frontier of (time, score) points, sorted by time."""
    pts = sorted(pts)
    front, best = [], -np.inf
    for t, s in pts:
        if s > best:
            front.append((t, s))
            best = s
    return front


def _limits(values, default=(0.0, 1.0), symmetric=False):
    """Calculate shared display limits for result maps.

    Args:
        values: Values processed by the operation.
        default: Fallback value used when no finite limit is available.
        symmetric: Make the limits symmetric about zero.
    """
    vals = np.asarray([v for v in values if np.isfinite(v)], float)
    if not len(vals):
        return default
    if symmetric:
        lim = max(float(np.max(np.abs(vals))), 1e-6)
        return -lim, lim
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        pad = max(abs(lo) * 0.05, 0.01)
        lo, hi = lo - pad, hi + pad
    return lo, hi


def _label_picks(front, n=4):
    """At most n frontier points to caption: both ends, then evenly spaced.

    Args:
        front: Indices assigned to the Pareto frontier.
        n: Number of requested samples or output points.
    """
    if len(front) <= n:
        return list(front)
    idx = sorted({0, len(front) - 1}
                 | {round(i * (len(front) - 1) / (n - 1)) for i in range(n)})
    return [front[i] for i in idx][:n]


def _plot_decision(ax, rows, value_fn, title, annotate=False):
    """Plot the optimizer decision trace.

    Args:
        ax: Matplotlib axes to draw on.
        rows: Input table rows.
        value_fn: Callable that extracts the plotted value.
        title: Human-readable plot title.
        annotate: Label the frontier points.
    """
    rs = [r for r in rows if np.isfinite(r["t_total_s"])
          and np.isfinite(value_fn(r))]
    ax.set_xlabel("time to done (s)")
    ax.set_ylabel(title.lower())
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if not rs:
        ax.text(0.5, 0.5, "no initial-score data", ha="center", va="center",
                transform=ax.transAxes)
        return
    pts = [(r["t_total_s"], value_fn(r)) for r in rs]
    ax.scatter(*zip(*pts), s=18, c="#999999", label="runs")
    front = _pareto(pts)
    if front:
        ax.plot(*zip(*front), "o-", color="#d62728", lw=2, ms=6,
                label="frontier")
    handles = None
    if annotate and front:
        # Number the frontier points and list the knobs in the legend; captions
        # pinned to a bunched frontier collide.
        keep = _label_picks(front, n=5)
        handles = []
        for i, (t, s) in enumerate(keep, 1):
            r = next(r for r in rs
                     if r["t_total_s"] == t and value_fn(r) == s)
            knobs = ", ".join(
                f"{k}={r[k]}" for k in ALGO_KNOBS.get(r["algo"], [])
                if r.get(k))
            ax.annotate(str(i), (t, s), fontsize=8, fontweight="bold",
                        color="#d62728", xytext=(5, 4),
                        textcoords="offset points")
            handles.append(Line2D([], [], linestyle="none", marker="$%d$" % i,
                                  color="#d62728", markersize=7,
                                  label=f"{knobs}  ({t:.0f}s, {s:.4f})"))
    ax.legend(handles=handles, fontsize=7, loc="lower right",
              framealpha=0.9) if handles else ax.legend(fontsize=8,
                                                        loc="lower right")


def fig_decision(rows, algo, bench_dir):
    """Create the decision figure.

    Args:
        rows: Input table rows.
        algo: Optimisation algorithm identifier.
        bench_dir: Directory containing the benchmark results.
    """
    # Controls are extra repeats of one middle cell taken to track drift;
    # fig_drift is where they belong.
    rs = [r for r in rows if r["algo"] == algo and r["phase"] != "control"]
    if not rs:
        return None
    fig = Figure(figsize=(15, 4.6), layout="constrained")
    axes = fig.subplots(1, 3)
    for i, (title, value_fn) in enumerate(SCORE_VIEWS):
        _plot_decision(axes[i], rs, value_fn, title, annotate=(i == 0))
    # Each panel gets its own y range: finals and initials differ in scale.
    for ax, fn in zip(axes, (_score, _initial)):
        ax.set_ylim(_limits([fn(r) for r in rs]))
    axes[2].axhline(0.0, color="#555555", ls="--", lw=1)
    fig.suptitle(f"{algo}: decision plots (fixed bench score reference) -- "
                 "captions are a shortlist; full values in runs.csv")
    return _save(fig, bench_dir, f"decision_{algo}.png")


def _heat_data(rs, kx, ky, value_fn):
    """Median value and time matrices on one sweep grid.

    Args:
        rs: Radial sample coordinates.
        kx: Horizontal spatial-frequency coordinates.
        ky: Vertical spatial-frequency coordinates.
        value_fn: Callable that extracts the plotted value.
    """
    xs = sorted({v for r in rs if np.isfinite(v := _f(r.get(kx)))})
    ys = sorted({v for r in rs if np.isfinite(v := _f(r.get(ky)))})
    values = np.full((len(ys), len(xs)), np.nan)
    times = np.full((len(ys), len(xs)), np.nan)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            g = [r for r in rs if _f(r[kx]) == x and _f(r[ky]) == y]
            if g:
                vv = [value_fn(r) for r in g if np.isfinite(value_fn(r))]
                tt = [r["t_total_s"] for r in g
                      if np.isfinite(r["t_total_s"])]
                if vv:
                    values[j, i] = float(np.median(vv))
                if tt:
                    times[j, i] = float(np.median(tt))
    return xs, ys, values, times


def _draw_heat(ax, xs, ys, values, times, kx, ky, title, cmap,
               limits, show_time=False):
    """Draw a heat-map view of the result grid.

    Args:
        ax: Matplotlib axes to draw on.
        xs: Horizontal coordinates.
        ys: Vertical coordinates.
        values: Values processed by the operation.
        times: Sequence of time values.
        kx: Horizontal spatial-frequency coordinates.
        ky: Vertical spatial-frequency coordinates.
        title: Human-readable plot title.
        cmap: Matplotlib colour map.
        limits: Sequence of limit values.
        show_time: Overlay the run time on each cell.
    """
    im = ax.imshow(np.ma.masked_invalid(values), origin="lower", aspect="auto",
                   cmap=cmap, vmin=limits[0], vmax=limits[1])
    for j in range(len(ys)):
        for i in range(len(xs)):
            if np.isfinite(values[j, i]):
                label = f"{values[j, i]:.3f}"
                if show_time and np.isfinite(times[j, i]):
                    label += f"\n{times[j, i]:.0f}s"
                ax.text(i, j, label, ha="center", va="center", fontsize=7,
                        color="white",
                        bbox=dict(facecolor="black", alpha=0.25, pad=1,
                                  edgecolor="none"))
    if not np.isfinite(values).any():
        ax.text(0.5, 0.5, "no initial-score data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xticks(range(len(xs)), [f"{v:g}" for v in xs])
    ax.set_yticks(range(len(ys)), [f"{v:g}" for v in ys])
    ax.set_xlabel(kx)
    ax.set_ylabel(ky)
    ax.set_title(title)
    return im


def _heat_triplet(rs, kx, ky, heading):
    """Draw a three-panel heat-map comparison.

    Args:
        rs: Radial sample coordinates.
        kx: Horizontal spatial-frequency coordinates.
        ky: Vertical spatial-frequency coordinates.
        heading: Heading displayed above the comparison panels.
    """
    data = [_heat_data(rs, kx, ky, fn) for _, fn in SCORE_VIEWS]
    score_lim = _limits(np.concatenate([data[0][2].ravel(),
                                        data[1][2].ravel()]))
    gain_lim = _limits(data[2][2].ravel(), symmetric=True)
    fig = Figure(figsize=(16.5, 4.8), layout="constrained")
    axes = fig.subplots(1, 3)
    for i, (title, _) in enumerate(SCORE_VIEWS):
        xs, ys, values, times = data[i]
        im = _draw_heat(
            axes[i], xs, ys, values, times, kx, ky, title,
            "coolwarm" if i == 2 else "viridis",
            gain_lim if i == 2 else score_lim, show_time=(i == 0))
        fig.colorbar(im, ax=axes[i], label=("score change" if i == 2
                                            else "score"))
    fig.suptitle(heading + " (final cells also show median time)")
    return fig


def _scatter_triplet(rs, knob, heading):
    """Draw a three-panel scatter comparison.

    Args:
        rs: Radial sample coordinates.
        knob: Optimiser hyperparameter being plotted.
        heading: Heading displayed above the comparison panels.
    """
    fig = Figure(figsize=(15, 4.6), layout="constrained")
    axes = fig.subplots(1, 3)
    score_lim = _limits([fn(r) for r in rs for fn in (_score, _initial)])
    gain_lim = _limits([_gain(r) for r in rs], symmetric=True)
    for i, (title, value_fn) in enumerate(SCORE_VIEWS):
        pts = [(v, value_fn(r), r["t_total_s"])
               for r in rs if np.isfinite(v := _f(r.get(knob)))
               and np.isfinite(value_fn(r))]
        ax = axes[i]
        if pts:
            xs, vals, times = map(np.asarray, zip(*pts))
            if i == 0:
                sc = ax.scatter(xs, vals, s=24, c=times, cmap="coolwarm")
                fig.colorbar(sc, ax=ax, label="time (s)")
            elif i == 1:
                ax.scatter(xs, vals, s=24, color="#2ca02c")
            else:
                sc = ax.scatter(xs, vals, s=24, c=vals, cmap="coolwarm",
                                vmin=gain_lim[0], vmax=gain_lim[1])
                fig.colorbar(sc, ax=ax, label="score change")
        else:
            ax.text(0.5, 0.5, "no initial-score data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_xlabel(knob)
        ax.set_ylabel(title.lower())
        ax.set_title(title + (" (color = time)" if i == 0 else ""))
        ax.grid(alpha=0.3)
    axes[0].set_ylim(score_lim)
    axes[1].set_ylim(score_lim)
    axes[2].set_ylim(gain_lim)
    axes[2].axhline(0.0, color="#555555", ls="--", lw=1)
    fig.suptitle(heading)
    return fig


def _heat_plus_marginal(rs, kx, ky, kz, heading):
    """A kx x ky heatmap plus a kz marginal scatter, each in the three views.

    For an algorithm that still varies three knobs after the pinned ones are
    excluded. Two of them make the surface and the third is shown as a
    marginal, because a three-axis grid has no honest flat rendering.

    Args:
        rs: Runs of one algorithm, phase-A only.
        kx: Knob on the heatmap's x axis.
        ky: Knob on the heatmap's y axis.
        kz: Knob shown as a marginal scatter.
        heading: Figure title.
    """
    fig = Figure(figsize=(15, 9), layout="constrained")
    axes = fig.subplots(2, 3)
    data = [_heat_data(rs, kx, ky, fn) for _, fn in SCORE_VIEWS]
    score_lim = _limits([fn(r) for r in rs for fn in (_score, _initial)])
    gain_lim = _limits([_gain(r) for r in rs], symmetric=True)
    for i, (title, value_fn) in enumerate(SCORE_VIEWS):
        xs, ys, values, times = data[i]
        im = _draw_heat(
            axes[0, i], xs, ys, values, times, kx, ky, title,
            "coolwarm" if i == 2 else "viridis",
            gain_lim if i == 2 else score_lim, show_time=(i == 0))
        fig.colorbar(im, ax=axes[0, i],
                     label="score change" if i == 2 else "score")

        pts = [(v, value_fn(r), r["t_total_s"])
               for r in rs if np.isfinite(v := _f(r.get(kz)))
               and np.isfinite(value_fn(r))]
        ax = axes[1, i]
        if pts:
            mx, mv, mt = map(np.asarray, zip(*pts))
            if i == 0:
                sc = ax.scatter(mx, mv, s=22, c=mt, cmap="coolwarm")
                fig.colorbar(sc, ax=ax, label="time (s)")
            elif i == 1:
                ax.scatter(mx, mv, s=22, color="#2ca02c")
            else:
                sc = ax.scatter(mx, mv, s=22, c=mv, cmap="coolwarm",
                                vmin=gain_lim[0], vmax=gain_lim[1])
                fig.colorbar(sc, ax=ax, label="score change")
        ax.set_xlabel(kz)
        ax.set_ylabel(title.lower())
        ax.set_title(f"{kz} marginal: {title.lower()}"
                     + (" (color = time)" if i == 0 else ""))
        ax.grid(alpha=0.3)
    axes[1, 0].set_ylim(score_lim)
    axes[1, 1].set_ylim(score_lim)
    axes[1, 2].set_ylim(gain_lim)
    axes[1, 2].axhline(0.0, color="#555555", ls="--", lw=1)
    fig.suptitle(heading)
    return fig


def fig_knobs(rows, algo, bench_dir):
    """Knob diagnosis for one algorithm (phase-A rows only).

    Args:
        rows: Input table rows.
        algo: Optimisation algorithm identifier.
        bench_dir: Directory containing the benchmark results.
    """
    rs = [r for r in rows if r["algo"] == algo and r["phase"] == "knobs"]
    if not rs:
        return None
    # Axes follow what the data actually varied; a pinned knob stays off them.
    knobs, held = [], []
    for k in ALGO_KNOBS.get(algo, []):
        values = {r.get(k) for r in rs if r.get(k) not in (None, "")}
        if len(values) > 1:
            knobs.append(k)
        elif values:
            held.append(f"{k}={values.pop()}")
    heading = f"{algo}: knob sweep (fixed bench score reference)"
    if held:
        heading += "  |  held: " + ", ".join(held)
    if len(knobs) >= 3:
        fig = _heat_plus_marginal(rs, knobs[0], knobs[1], knobs[2], heading)
    elif len(knobs) == 2:
        fig = _heat_triplet(rs, knobs[0], knobs[1], heading)
    elif knobs:
        fig = _scatter_triplet(rs, knobs[0], heading)
    else:
        return None
    return _save(fig, bench_dir, f"knobs_{algo}.png")


def fig_settle_frames(rows, algo, bench_dir):
    """Create the settle frames figure.

    Args:
        rows: Input table rows.
        algo: Optimisation algorithm identifier.
        bench_dir: Directory containing the benchmark results.
    """
    rs = [r for r in rows if r["algo"] == algo and r["phase"] == "measure"]
    if not rs:
        return None
    fig = _heat_triplet(
        rs, "settle_ms", "frames",
        f"{algo}: settle x frames at best knobs (fixed bench score reference)")
    return _save(fig, bench_dir, f"settle_frames_{algo}.png")


def _load_noise(path):
    if not Path(path).exists():
        return None
    t, s = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            t.append(_f(r["t_s"]))
            s.append(_f(r["score"]))
    return np.array(t), np.array(s)


def _asd(t, s):
    """One-sided amplitude spectral density of the score trace.

    Args:
        t: Time samples or scalar time.
        s: Signal samples or logistic width.
    """
    if len(t) < 8:
        return None
    fs = 1.0 / max(float(np.median(np.diff(t))), 1e-6)
    x = s - np.mean(s)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    asd = np.sqrt(2.0 * np.abs(spec) ** 2 / (fs * len(x)))
    return f[1:], asd[1:], fs / 2.0


def fig_noise(algo, bench_dir):
    """Create the noise figure.

    Args:
        algo: Optimisation algorithm identifier.
        bench_dir: Directory containing the benchmark results.
    """
    nd = Path(bench_dir) / "noise"
    before = _load_noise(nd / f"{algo}_before.csv")
    after = _load_noise(nd / f"{algo}_after.csv")
    if before is None and after is None:
        return None
    fig = Figure(figsize=(10, 4))
    ax1, ax2 = fig.subplots(1, 2)
    for data, label, color in ((before, "before", "#1f77b4"),
                               (after, "after", "#d62728")):
        if data is None:
            continue
        t, s = data
        ax1.plot(t, s, lw=0.8, color=color,
                 label=f"{label} (sigma {np.std(s):.4f})")
        a = _asd(t, s)
        if a:
            f, asd, nyq = a
            ax2.loglog(f, asd, lw=0.8, color=color, label=label)
            ax2.axvline(nyq, color=color, ls="--", lw=0.8, alpha=0.6)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("score")
    ax1.set_title(f"{algo}: score jitter at start bits")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.set_xlabel("frequency (Hz)  (dashed = Nyquist)")
    ax2.set_ylabel("ASD (score/sqrt(Hz))")
    ax2.set_title("noise spectrum")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)
    return _save(fig, bench_dir, f"noise_{algo}.png")


def load_point_curves(bench_dir, rows=None):
    """Running-best score against point count, one curve per run.

    The running best at point p is what a run given a budget of p points
    would have reported, so one long run carries the whole curve. Exact for
    `Genetic` and `BayesOpt`, whose budget knobs are pure stop tests.

    Args:
        bench_dir: A bench session folder.
        rows: runs.csv rows, loaded if omitted.

    Returns:
        {run_id: (algo, numpy array of running-best score by point)}.
    """
    path = Path(bench_dir) / "point_scores.csv"
    if not path.is_file():
        return {}
    rows = load_runs(bench_dir) if rows is None else rows
    algo = {r["run_id"]: r["algo"] for r in rows}
    phase = {r["run_id"]: r.get("phase") for r in rows}
    got = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if phase.get(r["run_id"]) == "control":
                continue
            got.setdefault(r["run_id"], []).append(
                (int(r["point"]), _f(r["score"])))
    out = {}
    for run_id, pairs in got.items():
        pairs.sort()
        scores = np.array([s for _, s in pairs], float)
        if scores.size and np.any(np.isfinite(scores)):
            out[run_id] = (algo.get(run_id, "?"),
                           np.maximum.accumulate(np.nan_to_num(scores,
                                                               nan=-np.inf)))
    return out


def fig_budget(bench_dir, rows=None, algos=None):
    """Score against measurement budget, per algorithm -- the headline plot.

    Curves are the mean over repeats of the running best; the band is the
    standard deviation across repeats.

    Args:
        bench_dir: A bench session folder.
        rows: runs.csv rows, loaded if omitted.
        algos: Restrict to these algorithms; all of them by default.

    Returns:
        The filename written, or None when there is nothing to draw.
    """
    curves = load_point_curves(bench_dir, rows)
    if not curves:
        return None
    by_algo = {}
    for algo, series in curves.values():
        if algos and algo not in algos:
            continue
        by_algo.setdefault(algo, []).append(series)
    if not by_algo:
        return None
    fig = Figure(figsize=(7.2, 5.0), layout="constrained")
    ax = fig.add_subplot(111)
    for i, (algo, series) in enumerate(sorted(by_algo.items())):
        # Repeats of one algorithm can differ in length when the budget knob
        # was itself scanned; average over however many reached each point and
        # stop where fewer than half of them did, so the tail is not one run.
        longest = max(len(s) for s in series)
        mean, sd, x = [], [], []
        for p in range(longest):
            v = [s[p] for s in series if len(s) > p and np.isfinite(s[p])]
            if len(v) < max(2, len(series) / 2):
                break
            x.append(p + 1)
            mean.append(np.mean(v))
            sd.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
        if not x:
            continue
        mean, sd = np.array(mean), np.array(sd)
        colour = f"C{i}"
        ax.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.15,
                        lw=0)
        ax.plot(x, mean, "-", lw=1.8, color=colour,
                label=f"{algo}  ({len(series)} runs)")
    ax.set_xlabel("measurements spent on one command sequence")
    ax.set_ylabel("best score reached so far")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, bench_dir, "budget.png")


def fig_overall(rows, bench_dir):
    """Create the overall figure.

    Args:
        rows: Input table rows.
        bench_dir: Directory containing the benchmark results.
    """
    algos = sorted({r["algo"] for r in rows})
    if not algos:
        return None
    fig = Figure(figsize=(15, 4.8), layout="constrained")
    axes = fig.subplots(1, 3)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for view_i, (title, value_fn) in enumerate(SCORE_VIEWS):
        ax = axes[view_i]
        for i, algo in enumerate(algos):
            rs = [r for r in rows if r["algo"] == algo
                  and np.isfinite(r["t_total_s"])
                  and np.isfinite(value_fn(r))]
            if not rs:
                continue
            pts = [(r["t_total_s"], value_fn(r)) for r in rs]
            ax.scatter(*zip(*pts), s=12, alpha=0.35,
                       color=colors[i % len(colors)])
            front = _pareto(pts)
            if front:
                ax.plot(*zip(*front), "o-", color=colors[i % len(colors)],
                        lw=2, label=algo)
        ax.set_xlabel("time to done (s)")
        ax.set_ylabel(title.lower())
        ax.set_title(title + " (lines = frontiers)")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        elif view_i:
            ax.text(0.5, 0.5, "no initial-score data", ha="center", va="center",
                    transform=ax.transAxes)
    score_lim = _limits([fn(r) for r in rows for fn in (_score, _initial)])
    axes[0].set_ylim(score_lim)
    axes[1].set_ylim(score_lim)
    axes[2].axhline(0.0, color="#555555", ls="--", lw=1)
    fig.suptitle("all algorithms: quality vs time (fixed bench score reference)")
    return _save(fig, bench_dir, "overall.png")


def fig_drift(rows, algo, bench_dir):
    """The bench's own drift, and whether it could still bias the ranking.

    Two tracers: the control runs re-measured every `control_every` runs, and
    every run's `score_initial` at identical start bits. A flat pair means
    the raw ranking can be read as it stands.

    Args:
        rows: Input table rows.
        algo: Optimisation algorithm identifier.
        bench_dir: Directory containing the benchmark results.
    """
    rs = sorted((r for r in rows if r["algo"] == algo),
                key=lambda r: _f(r.get("run_id")))
    if len(rs) < 4:
        return None
    # Elapsed time, not run index: drift is a clock effect and runs differ in
    # length by 5x, so an index axis would distort the slope it is measuring.
    t, ctrl_t, ctrl_v, init_v = [], [], [], []
    total = 0.0
    for r in rs:
        total += max(_f(r.get("t_total_s")), 0.0)
        t.append(total / 3600.0)
        init_v.append(_initial(r))
        if r["phase"] == "control":
            ctrl_t.append(total / 3600.0)
            ctrl_v.append(_score(r))
    fig = Figure(figsize=(11, 4.4), layout="constrained")
    ax, ax2 = fig.subplots(1, 2)
    for axis, xs, ys, title, ylab in (
            (ax, ctrl_t, ctrl_v, "control runs (same knobs, re-measured)",
             "final score of the control cell"),
            (ax2, t, init_v, "every run's start-bit score",
             "score at identical start bits")):
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        good = np.isfinite(xs) & np.isfinite(ys)
        axis.plot(xs[good], ys[good], "o-", ms=4, lw=1.2, color="#2e86ab")
        axis.set_xlabel("elapsed session time (h)")
        axis.set_ylabel(ylab)
        axis.grid(alpha=0.3)
        if good.sum() >= 3:
            slope = np.polyfit(xs[good], ys[good], 1)[0]
            mean = float(np.mean(ys[good]))
            span = float(np.ptp(ys[good]))
            axis.plot(xs[good], np.polyval(
                np.polyfit(xs[good], ys[good], 1), xs[good]), "--",
                color="#d1495b", lw=1.4)
            axis.set_title(
                f"{title}\ntrend {100 * slope / mean:+.2f} %/h, "
                f"range {100 * span / mean:.2f}% of the mean")
        else:
            axis.set_title(title + "\n(too few points to fit)")
    fig.suptitle(f"{algo}: drift control -- flat means the ranking needs no "
                 "correction")
    return _save(fig, bench_dir, f"drift_{algo}.png")


def generate_for_algo(bench_dir, algo, rows=None):
    """Generate every figure for one algorithm from the current runs.csv.

    Called the moment an algorithm finishes, so a scan stopped early still
    has its figures.

    Args:
        bench_dir: Directory containing the benchmark results.
        algo: Optimisation algorithm identifier.
        rows: Input table rows.
    """
    rows = load_runs(bench_dir) if rows is None else rows
    written = []
    for fn in (fig_decision, fig_knobs, fig_settle_frames, fig_drift):
        name = fn(rows, algo, bench_dir)
        if name:
            written.append(name)
    name = fig_noise(algo, bench_dir)
    if name:
        written.append(name)
    return written


def generate_all(bench_dir):
    rows = load_runs(bench_dir)
    written = []
    for algo in sorted({r["algo"] for r in rows}):
        written += generate_for_algo(bench_dir, algo, rows)
    name = fig_overall(rows, bench_dir)
    if name:
        written.append(name)
    name = fig_budget(bench_dir, rows)
    if name:
        written.append(name)
    return written


if __name__ == "__main__":
    for n in generate_all(sys.argv[1]):
        print("figs/" + n)
