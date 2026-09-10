# SPDX-License-Identifier: GPL-3.0-or-later
"""Speed/quality and hyperparameter-combination analysis for a DM bench run.

The normal bench figures answer "what happened in each panel?".  This module
answers the later decision question: which measured parameter combinations are
fast, which are effective, and where does extra runtime stop buying much score?

Run as either::

    python -m dm_toolkit.bench.tradeoff_plots BENCH_DIR
    python tradeoff_plots.py BENCH_DIR
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np
from scipy.optimize import curve_fit

try:  # Package execution.
    from .plots import ALGO_KNOBS, _f, _score, load_runs
except ImportError:  # Direct file execution.
    from plots import ALGO_KNOBS, _f, _score, load_runs


ALGO_COLOURS = {
    "hill_climb": "#e67e22",
    "spgd": "#8c564b",
    "genetic": "#9467bd",
    "cmaes": "#2ca02c",
    "bayes": "#d62728",
    "anneal": "#1f77b4",
}
CHOICE_LABELS = {
    "knee": "knee (balanced)",
    "near_best_fastest": "fastest within 0.01 of best",
    "best_score": "highest observed score",
}


def _usable(rows, algo=None):
    """Non-control runs with finite time and final score."""
    return [r for r in rows
            if r.get("phase") != "control"
            and (algo is None or r.get("algo") == algo)
            and np.isfinite(_f(r.get("t_total_s")))
            and np.isfinite(_score(r))]


def _pareto_rows(rows):
    """Upper-left frontier, retaining the source row for knob captions."""
    out, best = [], -np.inf
    for row in sorted(rows, key=lambda r: (_f(r["t_total_s"]), -_score(r))):
        if _score(row) > best:
            out.append(row)
            best = _score(row)
    return out


def _knee(front):
    """Closest frontier point to the fast/high-score corner.

    Runtime is log-scaled because a 60 -> 600 s increase is a more meaningful
    cost change here than an additive ten seconds at either end.
    """
    if not front:
        return None
    if len(front) < 3:
        return front[-1]
    t = np.log10([_f(r["t_total_s"]) for r in front])
    score = np.asarray([_score(r) for r in front], float)
    tn = (t - t.min()) / max(float(np.ptp(t)), 1e-12)
    sn = (score - score.min()) / max(float(np.ptp(score)), 1e-12)
    return front[int(np.argmin(tn * tn + (1.0 - sn) ** 2))]


def _controls(rows, algo):
    values = [_score(r) for r in rows
              if r.get("algo") == algo and r.get("phase") == "control"
              and np.isfinite(_score(r))]
    sd = float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan")
    return len(values), sd


def _varied_knobs(rows, algo):
    varied, held = [], []
    for knob in ALGO_KNOBS.get(algo, []):
        values = sorted({_f(r.get(knob)) for r in rows
                         if np.isfinite(_f(r.get(knob)))})
        if len(values) > 1:
            varied.append(knob)
        elif values:
            held.append((knob, values[0]))
    return varied, held


def _knob_text(row):
    parts = []
    for knob in ALGO_KNOBS.get(row.get("algo"), []):
        value = row.get(knob)
        if value not in (None, ""):
            parts.append(f"{knob}={float(value):g}")
    return "; ".join(parts)


def recommendations(rows):
    """Three interpretable operating points for every measured algorithm."""
    selected = []
    for algo in sorted({r["algo"] for r in rows}):
        data = _usable(rows, algo)
        if not data:
            continue
        front = _pareto_rows(data)
        top = max(data, key=_score)
        threshold = _score(top) - 0.01
        near = min((r for r in data if _score(r) >= threshold),
                   key=lambda r: _f(r["t_total_s"]))
        picks = (("knee", _knee(front)),
                 ("near_best_fastest", near),
                 ("best_score", top))
        n_control, control_sd = _controls(rows, algo)
        for choice, row in picks:
            selected.append({
                "algo": algo,
                "choice": choice,
                "t_total_s": _f(row["t_total_s"]),
                "score_final": _score(row),
                "score_gain": _f(row.get("score_gain")),
                "measurement_sigma": _f(row.get("sigma_est")),
                "control_repeat_n": n_control,
                "control_repeat_sd": control_sd,
                "parameters": _knob_text(row),
                "run_id": row.get("run_id", ""),
                "row": row,
            })
    return selected


def _sat_model(t, ceiling, amplitude, tau):
    return ceiling - amplitude * np.exp(-t / tau)


def _fit_saturation(front):
    if len(front) < 4:
        return None
    t = np.asarray([_f(r["t_total_s"]) for r in front], float)
    t0 = float(t.min())
    shifted = t - t0
    y = np.asarray([_score(r) for r in front], float)
    try:
        params, _ = curve_fit(
            _sat_model, shifted, y,
            p0=(min(1.0, y.max() + 0.01), max(y.max() - y.min(), 0.02),
                float(np.median(shifted[shifted > 0]))),
            bounds=([y.max(), 0.0, 1.0],
                    [1.05, 1.0, max(float(shifted.max()) * 20.0, 2.0)]),
            maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    predicted = _sat_model(shifted, *params)
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else float("nan")
    return params, t0, r_squared


def fig_speed_quality(rows, selected, output_dir):
    data = _usable(rows)
    if not data:
        return None
    global_front = _pareto_rows(data)
    global_knee = _knee(global_front)
    fit = _fit_saturation(global_front)

    fig = Figure(figsize=(14.5, 5.6), layout="constrained")
    ax, fit_ax = fig.subplots(1, 2)
    label_offsets = {
        "anneal": (6, 10), "bayes": (7, -18), "cmaes": (7, -8),
        "genetic": (7, 6), "hill_climb": (7, 5),
    }
    for algo in sorted({r["algo"] for r in data}):
        group = _usable(rows, algo)
        colour = ALGO_COLOURS.get(algo, "#777777")
        ax.scatter([_f(r["t_total_s"]) for r in group],
                   [_score(r) for r in group], s=24, alpha=0.42,
                   color=colour, label=algo)
        front = _pareto_rows(group)
        ax.plot([_f(r["t_total_s"]) for r in front],
                [_score(r) for r in front], "-", lw=1.4, color=colour)
        balanced = next(x for x in selected
                        if x["algo"] == algo and x["choice"] == "knee")
        yerr = balanced["control_repeat_sd"]
        ax.errorbar(balanced["t_total_s"], balanced["score_final"],
                    yerr=yerr if np.isfinite(yerr) else None, fmt="*",
                    ms=12, capsize=3, color=colour, mec="black", mew=0.5)
        ax.annotate(algo, (balanced["t_total_s"], balanced["score_final"]),
                    xytext=label_offsets.get(algo, (5, 5)),
                    textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("time to completion (s, log scale)")
    ax.set_ylabel("final score")
    ax.set_title("Measured combinations; stars = per-algorithm knee\n"
                 "error bars = SD of repeated middle-control runs")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2, loc="lower right")

    ft = np.asarray([_f(r["t_total_s"]) for r in global_front])
    fy = np.asarray([_score(r) for r in global_front])
    fit_ax.scatter(ft, fy, s=42, color="#222222", label="global Pareto points")
    fit_ax.step(ft, fy, where="post", color="#666666", lw=1.2,
                label="observed best-so-far")
    if fit is not None:
        params, t0, r_squared = fit
        grid = np.geomspace(max(1.0, ft.min()), ft.max(), 300)
        fit_ax.plot(grid, _sat_model(grid - t0, *params),
                    color="#d62728", lw=2.2,
                    label=(f"shifted saturation fit: ceiling={params[0]:.3f}, "
                           f"tau={params[2]:.1f}s, R2={r_squared:.3f}"))
    if global_knee is not None:
        kt, ks = _f(global_knee["t_total_s"]), _score(global_knee)
        fit_ax.plot(kt, ks, "*", ms=15, color="#ffbf00", mec="black",
                    label=f"global knee: {kt:.0f}s, {ks:.3f}")
        fit_ax.annotate(_knob_text(global_knee), (kt, ks), xytext=(8, -24),
                        textcoords="offset points", fontsize=8,
                        arrowprops=dict(arrowstyle="->", lw=0.7))
    fit_ax.set_xscale("log")
    fit_ax.set_xlabel("time to completion (s, log scale)")
    fit_ax.set_ylabel("final score")
    fit_ax.set_title("Global Pareto envelope and diminishing returns")
    fit_ax.grid(alpha=0.3, which="both")
    fit_ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("DM9 optimiser sweep: speed-quality trade-off (settle 570 ms, 8 frames)")
    out = output_dir / "speed_quality_saturation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    return out


def _grid_axis(rows, knob):
    values = sorted({_f(r.get(knob)) for r in rows
                     if np.isfinite(_f(r.get(knob)))})
    lookup = {value: index for index, value in enumerate(values)}
    return values, lookup


def fig_parameter_surfaces(rows, selected, output_dir):
    algos = [a for a in sorted({r["algo"] for r in rows})
             if len(_varied_knobs(_usable(rows, a), a)[0]) >= 2]
    if not algos:
        return None
    fig = Figure(figsize=(18, 10.5))
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.05, top=0.86,
                        wspace=0.18, hspace=0.30)
    axes = list(fig.subplots(2, 3, subplot_kw={"projection": "3d"}).flat)
    for ax, algo in zip(axes, algos):
        data = [r for r in _usable(rows, algo) if r.get("phase") == "knobs"]
        knobs, held = _varied_knobs(data, algo)
        kx, ky = knobs[:2]
        xv, xmap = _grid_axis(data, kx)
        yv, ymap = _grid_axis(data, ky)
        x = np.asarray([xmap[_f(r[kx])] for r in data], float)
        y = np.asarray([ymap[_f(r[ky])] for r in data], float)
        z = np.asarray([_score(r) for r in data], float)
        runtime = np.asarray([_f(r["t_total_s"]) for r in data], float)
        if len(data) >= 3:
            ax.plot_trisurf(x, y, z, color="#b8c6db", alpha=0.28,
                            linewidth=0.25, edgecolor="#777777")
        norm = Normalize(vmin=float(runtime.min()), vmax=float(runtime.max()))
        scatter = ax.scatter(x, y, z, c=runtime, cmap="plasma_r", norm=norm,
                             s=36, depthshade=False)
        balanced = next(s for s in selected
                        if s["algo"] == algo and s["choice"] == "knee")
        br = balanced["row"]
        bx, by, bz = xmap[_f(br[kx])], ymap[_f(br[ky])], _score(br)
        ax.scatter([bx], [by], [bz], marker="*", s=180, color="#00e5ff",
                   edgecolor="black", depthshade=False)
        sd = balanced["control_repeat_sd"]
        if np.isfinite(sd):
            ax.plot([bx, bx], [by, by], [bz - sd, bz + sd], color="black", lw=1)
        ax.set_xticks(range(len(xv)), [f"{v:g}" for v in xv], fontsize=7)
        ax.set_yticks(range(len(yv)), [f"{v:g}" for v in yv], fontsize=7)
        ax.set_xlabel(kx, labelpad=8)
        ax.set_ylabel(ky, labelpad=8)
        ax.set_zlabel("final score", labelpad=6)
        held_text = ", ".join(f"{k}={v:g}" for k, v in held)
        ax.set_title(algo + (f" (held {held_text})" if held_text else "")
                     + "\ncyan star = knee; one run per grid cell", fontsize=10)
        fig.colorbar(scatter, ax=ax, shrink=0.58, pad=0.08, label="time (s)")
    for ax in axes[len(algos):]:
        ax.set_axis_off()
    fig.suptitle("Hyperparameter combinations: height = score, colour = runtime\n"
                 "vertical star bars use repeated middle-control SD "
                 "(algorithm-level, not per-cell uncertainty)")
    out = output_dir / "parameter_surfaces_3d.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    return out


def _write_recommendations(selected, output_dir):
    path = output_dir / "recommended_combinations.csv"
    fields = ["algo", "choice", "t_total_s", "score_final", "score_gain",
              "measurement_sigma", "control_repeat_n", "control_repeat_sd",
              "parameters", "run_id"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fields)
        writer.writeheader()
        for item in selected:
            writer.writerow({k: item[k] for k in fields})
    return path


def _write_summary(rows, selected, output_dir):
    algos = sorted({r["algo"] for r in rows})
    supported = sorted(ALGO_KNOBS)
    lines = [
        "# DM9 speed-quality parameter analysis",
        "",
        "The recorded plan contains **%d measured algorithms**: %s. "
        "The plotting code supports %s too, but it has no rows in this run."
        % (len(algos), ", ".join(algos),
           ", ".join(a for a in supported if a not in algos) or "no others"),
        "",
        "Each parameter-grid cell has one optimisation run. Consequently the "
        "figures do not claim a per-cell standard error. Star error bars are the "
        "sample SD from the four repeated middle-control runs for that algorithm; "
        "they combine optimiser repeatability and session drift. The much smaller "
        "`measurement_sigma` in the CSV is frame/score noise only.",
        "",
        "## Provisional operating points",
        "",
        "| Algorithm | Choice | Time (s) | Score | Control SD | Parameters |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in selected:
        if item["choice"] != "knee":
            continue
        sd = item["control_repeat_sd"]
        lines.append("| %s | %s | %.1f | %.4f | %s | %s |" % (
            item["algo"], CHOICE_LABELS[item["choice"]], item["t_total_s"],
            item["score_final"], f"{sd:.4f}" if np.isfinite(sd) else "n/a",
            item["parameters"]))
    lines += [
        "",
        "The knee is the measured Pareto point closest to the fast/high-score "
        "corner after log-scaling runtime. `recommended_combinations.csv` also "
        "contains the fastest point within 0.01 score of each algorithm's best "
        "and the absolute highest observed point.",
        "",
        "Treat large control SD as a warning that a single winning grid cell may "
        "not repeat. Confirm shortlisted cells with at least 3--5 interleaved "
        "repeats before adopting a final setting.",
        "",
        "The shifted exponential is a descriptive fit to the global Pareto "
        "envelope across different algorithms and parameter combinations. It "
        "shows diminishing returns; it is not a physical convergence model for "
        "any one optimiser.",
    ]
    path = output_dir / "analysis_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate(bench_dir):
    bench_dir = Path(bench_dir)
    output_dir = bench_dir / "figs" / "tradeoff"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_runs(bench_dir)
    selected = recommendations(rows)
    outputs = [
        _write_recommendations(selected, output_dir),
        _write_summary(rows, selected, output_dir),
        fig_speed_quality(rows, selected, output_dir),
        fig_parameter_surfaces(rows, selected, output_dir),
    ]
    return [path for path in outputs if path is not None]


if __name__ == "__main__":
    for written in generate(sys.argv[1]):
        print(written)
