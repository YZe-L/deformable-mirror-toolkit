# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 2.6, 2026-08-29

"""Figures from a repeatability directory: final score, EE-Strehl and time.

Every figure shows all runs, a box for the summary and every run as a
point. A multi-metric session is grouped by metric, because the raw score
scales differ by orders of magnitude; `fig_strehl` compares the blocks.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from ..correction import settings as S
from .repeat_plan import short_label

# Stable colours: an algorithm keeps its colour across both figures and every
# session, so two directories can be laid side by side.
_PALETTE = matplotlib.colormaps["tab10"]
COLORS = {algo: _PALETTE(i % 10)
          for i, algo in enumerate(S.ALGOS_OFFERED)}

# The column `score.png` plots: the run's final score, seed-relative
# (v / (v + seed)), so it is not a comparison between algorithms.
SCORE = "norm_best_score"
# The final score against the diffraction limit. Not comparable between
# metrics, so it stays a summary.csv column and is not plotted.
ABSOLUTE = "score_final"
# The one column that ranks every run on the same physics: encircled energy
# in the diffraction bucket over the ideal. `fig_strehl` plots it.
STREHL = "ee_strehl_after"


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def group_of(row):
    """The (algorithm, arm, metric) a row belongs to.

    `arm` and `metric` are "" unless the session actually varied them, so a
    single-arm single-metric session still labels its boxes with the
    algorithm alone.
    """
    return (row["algo"], row.get("_arm", ""), row.get("_met", ""))


def label(key):
    """Axis label for a group key, or for a bare algorithm name.

    The key is the algorithm plus whatever else separates the box from its
    neighbours -- arm, metric, and for `fig_stages` the mirror -- one per
    line; the empty ones are simply not drawn.
    """
    if isinstance(key, str):
        return short_label(key)
    return "\n".join([short_label(key[0])] + [p for p in key[1:] if p])


def load_runs(repeat_dir):
    """Rows of runs.csv with the numeric columns parsed.

    A session that ran both compensation arms is two experiments per
    algorithm, so the arm becomes part of the group. One that ran a single arm
    is not, and labelling every box "comp off" would only take up room.
    """
    rows = []
    path = Path(repeat_dir) / "runs.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r = dict(r)
            for key in ("run_index", "repeat", "converged", "score_final",
                        "score_raw", "score_seed", "score_gain",
                        "score_final_best_point", "n_points", "n_valid",
                        "elapsed_s", "t_converge_s", "points_to_converge",
                        "t_best_s", "points_to_best",
                        "ee_strehl_before", "ee_strehl_after",
                        "norm_best_score", "n_converged_points"):
                r[key] = _f(r.get(key))
            rows.append(r)
    arms = {str(r.get("compensation") or "") for r in rows}
    arms.discard("")
    split = len(arms) > 1
    # The same test for the metric: a swept session compares rulers, so the
    # ruler has to be on the box. See the module docstring.
    mets = {str(r.get("metric") or "") for r in rows}
    mets.discard("")
    swept = len(mets) > 1
    for r in rows:
        arm = str(r.get("compensation") or "")
        r["_arm"] = f"comp {arm}" if (split and arm) else ""
        met = str(r.get("metric") or "")
        r["_met"] = met if (swept and met) else ""
    return rows


def _groups(rows, key):
    """{(algo, arm): [finite values of `key`]}, in first-seen run order."""
    out = {}
    for r in rows:
        out.setdefault(group_of(r), [])
        v = _f(r.get(key))
        if np.isfinite(v):
            out[group_of(r)].append(v)
    return out


def _converged_counts(rows):
    """{(algo, arm): (converged runs, runs)}."""
    out = {}
    for r in rows:
        key = group_of(r)
        got, total = out.get(key, (0, 0))
        out[key] = (got + int(_f(r.get("converged")) == 1.0), total + 1)
    return out


def _width(groups):
    """Figure width that keeps one box readable, however many there are.

    A three-metric paired session is 25 boxes, so the cap has to clear that
    at roughly an inch a box or the tick labels overprint each other.
    """
    return min(1.9 * max(len(groups), 3) + 3.0, 34.0)


def _save(fig, repeat_dir, name):
    # The session creates figs/; a directory that was moved or rebuilt by hand
    # may not have one, and losing a figure over a missing folder is silly.
    out = Path(repeat_dir) / "figs" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    return out.name


def _box_strip(ax, groups, ylabel, title, note=None):
    """Box plus every individual run, one column per algorithm.

    Args:
        ax: Target axes.
        groups: Mapping of group key to its values.
        ylabel: Y axis label.
        title: Axes title.
        note: Optional `f(key, values)` text drawn above each column.
    """
    names = [a for a, v in groups.items() if v]
    if not names:
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return
    data = [groups[a] for a in names]
    # An extra half step wherever the algorithm changes, so a paired session
    # reads as pairs instead of as one long row of boxes.
    positions, x = [], 1.0
    for i, key in enumerate(names):
        if i:
            prev = names[i - 1]
            # A metric change is a change of ruler, so it takes the wider
            # gap: the blocks either side of it are not on the same scale.
            if len(key) > 2 and key[2] != prev[2]:
                x += 1.4
            elif key[0] != prev[0]:
                x += 0.6
        positions.append(x)
        x += 1.0
    positions = np.asarray(positions, float)
    bp = ax.boxplot(data, positions=positions, widths=0.55, showfliers=False,
                    medianprops=dict(color="#111", lw=1.6),
                    patch_artist=True)
    for patch, key in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS.get(key[0], "#888"))
        patch.set_alpha(0.35)
        if "on" in key[1]:  # The compensated arm of a paired session.
            patch.set_hatch("//")
    rng = np.random.default_rng(0)  # Jitter only; reproducible.
    for pos, key in zip(positions, names):
        vals = np.asarray(groups[key], float)
        jitter = pos + rng.uniform(-0.16, 0.16, size=vals.size)
        ax.plot(jitter, vals, "o", ms=5, mfc=COLORS.get(key[0], "#888"),
                mec="#222", mew=0.6, ls="none", zorder=3)
    ax.set_xticks(positions)
    # `_width` caps the figure, so past a dozen boxes the columns stop getting
    # wider and the long algorithm names start overprinting each other: the
    # tick text has to shrink instead.
    ax.set_xticklabels(
        [f"{label(a)}\nn={len(groups[a])}" for a in names],
        fontsize=9 if len(names) <= 12 else 7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    if note:
        # Along the top, with headroom: a reference line usually sits at the
        # bottom of the axes and text there lands on it.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.14 * (hi - lo))
        for pos, key in zip(positions, names):
            text = note(key, groups[key])
            if text:
                ax.annotate(text, (pos, 0.99),
                            xycoords=("data", "axes fraction"),
                            ha="center", va="top", fontsize=8, color="#444")


def fig_score(rows, repeat_dir):
    """Final score of every run, one box per algorithm."""
    groups = _groups(rows, SCORE)
    fig = Figure(figsize=(_width(groups), 5.4))
    ax = fig.add_subplot(111)

    def note(_key, vals):
        a = np.asarray(vals, float)
        return (f"med {np.median(a):.3f}\nIQR "
                f"{np.percentile(a, 75) - np.percentile(a, 25):.3f}")

    _box_strip(ax, groups, "final score",
               "Final score after optimisation, over identical repeated runs",
               note=note)
    # No 0.5 reference line: every run finishes far above it, so drawing it
    # only stretches the axis down over empty space and squashes the boxes.
    return _save(fig, repeat_dir, "score.png")


def fig_strehl(rows, repeat_dir):
    """EE-Strehl after the run, one box per group; None without optics.

    The figure that answers "which run left the better spot". `score_final`
    cannot: it is each run's own objective, and a session that sweeps metrics
    puts three different quantities on that axis, which is why it is a
    summary.csv column and not a figure. Encircled energy in the diffraction
    bucket is measured the same way for all of them, so this is the figure
    whose boxes may be read ACROSS the metric blocks as well as within one.
    """
    groups = _groups(rows, STREHL)
    if not any(groups.values()):
        return None
    seeds = [v for r in rows
             if np.isfinite(v := _f(r.get("ee_strehl_before")))]
    fig = Figure(figsize=(_width(groups), 5.4))
    ax = fig.add_subplot(111)

    def note(_key, vals):
        a = np.asarray(vals, float)
        return (f"med {np.median(a):.3f}\nIQR "
                f"{np.percentile(a, 75) - np.percentile(a, 25):.3f}")

    _box_strip(ax, groups, "EE-Strehl at r0 (bucket energy / ideal)",
               "What the spot was actually worth, on one ruler for every "
               "metric, over identical repeated runs", note=note)
    lo, hi = ax.get_ylim()
    # The uncorrected mirror, drawn only when it falls inside the axes.
    if seeds and lo <= (seed := float(np.median(seeds))) <= hi:
        ax.axhline(seed, color="#555", ls="--", lw=1.0)
        ax.annotate(f"uncorrected seed, median {seed:.3f}", (0.004, seed),
                    xycoords=("axes fraction", "data"), va="bottom",
                    fontsize=8, color="#555")
    elif seeds:
        ax.set_xlabel(f"uncorrected seed, median "
                      f"{float(np.median(seeds)):.3f} -- below the axes",
                      fontsize=8, color="#555")
    return _save(fig, repeat_dir, "strehl.png")


def fig_time(rows, repeat_dir):
    """Time to converge for every run, one box per algorithm."""
    groups = _groups(rows, "t_converge_s")
    ylabel = "time to converge (s)"
    title = "How long each algorithm took, over identical repeated runs"
    if not any(groups.values()):
        # Nothing converged: a session the safety stop truncated used to draw
        # an empty figure. The one cost it still measured is how long each
        # algorithm took to REACH its best point, and the axis says so.
        groups = _groups(rows, "t_best_s")
        ylabel = "time to the best point (s) -- no run converged"
        title = ("How long each algorithm took to reach its best point. NO "
                 "run in this session converged, so this is not a "
                 "convergence time")
    converged = _converged_counts(rows)
    fig = Figure(figsize=(_width(groups), 5.4))
    ax = fig.add_subplot(111)

    def note(key, _vals):
        got, total = converged.get(key, (0, 0))
        return f"converged\n{got}/{total}" if total else ""

    _box_strip(ax, groups, ylabel, title, note=note)
    return _save(fig, repeat_dir, "time.png")


def _stage_groups(rows, column):
    """{group key + (stage,): [values]} from a ``name=value;...`` column.

    A staged run hands one mirror over to the next, and these columns record
    what the spot was worth at each handover. Grouping them beside the final
    score is the only way a session says which mirror bought which part of the
    gain -- and whether that split is repeatable or a one-run accident.

    Returns an empty mapping when the runs report a single stage, i.e. a
    one-mirror or joint session, which has no split to show.

    Args:
        rows: Parsed runs.csv rows.
        column: ``stage_strehl`` or ``stage_scores``.
    """
    out = {}
    for r in rows:
        key = group_of(r)
        for item in str(r.get(column) or "").split(";"):
            name, _, value = item.partition("=")
            v = _f(value)
            if not name or not np.isfinite(v):
                continue
            out.setdefault(key + (name,), []).append(v)
    return out if len({key[-1] for key in out}) > 1 else {}


def fig_stages(rows, repeat_dir):
    """What each mirror was worth, or None when the runs had one stage.

    Plotted on the EE-Strehl, which is measured against the diffraction limit
    and is therefore comparable across runs -- the same reason the session
    reports `score_final` on the absolute ruler. The optimiser's own
    seed-relative score is the fallback for a session run without optics, and
    the title then says so, because that number ranks each run's starting
    mirror as much as its stages.
    """
    groups = _stage_groups(rows, "stage_strehl")
    ruler = "EE-Strehl at r0"
    if not groups:
        groups = _stage_groups(rows, "stage_scores")
        ruler = "optimiser score (seed-relative -- no optics recorded)"
    if not groups:
        return None
    fig = Figure(figsize=(_width(groups), 5.4))
    ax = fig.add_subplot(111)

    def note(_key, vals):
        return f"med {np.median(np.asarray(vals, float)):.3f}"

    _box_strip(ax, groups, ruler,
               "What each mirror was worth, over identical repeated runs",
               note=note)
    return _save(fig, repeat_dir, "stages.png")


def _stats(values):
    """(median, mean, sd, iqr, min, max) of one group, NaN-safe."""
    a = np.asarray([v for v in values if np.isfinite(v)], float)
    if not a.size:
        nan = float("nan")
        return (nan,) * 6
    return (float(np.median(a)), float(np.mean(a)),
            float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
            float(np.percentile(a, 75) - np.percentile(a, 25)),
            float(a.min()), float(a.max()))


def write_summary(rows, repeat_dir):
    """summary.csv: one line per box, the numbers behind the figures.

    Score and cost side by side. The absolute and raw scores follow at the
    end: not plotted, but recorded, because they are what compares one
    session against another.
    """
    scores = _groups(rows, SCORE)
    absolute = _groups(rows, "score_final")
    strehls = _groups(rows, STREHL)
    seeds = _groups(rows, "ee_strehl_before")
    raws = _groups(rows, "score_raw")
    times = _groups(rows, "t_converge_s")
    points = _groups(rows, "points_to_converge")
    best_times = _groups(rows, "t_best_s")
    best_points = _groups(rows, "points_to_best")
    totals = _groups(rows, "elapsed_s")
    counts = _converged_counts(rows)
    num = lambda x: ("%.6g" % x if np.isfinite(x) else "nan")
    arms = {group_of(r): str(r.get("compensation") or "") for r in rows}
    path = Path(repeat_dir) / "summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["algo", "compensation", "metric", "runs", "converged",
                     "score_median", "score_mean", "score_sd", "score_iqr",
                     "score_min", "score_max",
                     "t_converge_median_s", "t_converge_mean_s",
                     "t_converge_sd_s", "t_converge_iqr_s",
                     "t_converge_min_s", "t_converge_max_s",
                     "points_median", "points_iqr", "run_total_median_s",
                     "absolute_median", "raw_median",
                     # The cross-metric ruler; see `fig_strehl`.
                     "ee_strehl_before_median", "ee_strehl_after_median",
                     # Always present, convergence or not: a truncated session
                     # still has a cost, and these are it.
                     "t_best_median_s", "points_to_best_median"])
        for key, vals in scores.items():
            got, total = counts.get(key, (0, 0))
            s_med, s_mean, s_sd, s_iqr, s_min, s_max = _stats(vals)
            t_med, t_mean, t_sd, t_iqr, t_min, t_max = _stats(
                times.get(key) or [])
            p_med, _, _, p_iqr, _, _ = _stats(points.get(key) or [])
            wr.writerow([
                key[0], arms.get(key, ""), key[2], total, got,
                num(s_med), num(s_mean), num(s_sd), num(s_iqr),
                num(s_min), num(s_max),
                num(t_med), num(t_mean), num(t_sd), num(t_iqr),
                num(t_min), num(t_max),
                num(p_med), num(p_iqr),
                num(_stats(totals.get(key) or [])[0]),
                num(_stats(absolute.get(key) or [])[0]),
                num(_stats(raws.get(key) or [])[0]),
                num(_stats(seeds.get(key) or [])[0]),
                num(_stats(strehls.get(key) or [])[0]),
                num(_stats(best_times.get(key) or [])[0]),
                num(_stats(best_points.get(key) or [])[0])])
    return path.name


def generate(repeat_dir):
    """The two figures plus summary.csv; a failing figure never loses data.

    Args:
        repeat_dir: A repeatability session directory.

    Returns:
        list: File names written.
    """
    rows = load_runs(repeat_dir)
    if not rows:
        return []
    made = []
    for fn in (fig_score, fig_strehl, fig_time, fig_stages):
        try:
            name = fn(rows, repeat_dir)
            if name:  # fig_stages returns None for a one-mirror session.
                made.append(name)
        except Exception as e:  # noqa: BLE001 -- a figure must not lose a run.
            made.append(f"{fn.__name__} failed: {e}")
    try:
        made.append(write_summary(rows, repeat_dir))
    except (OSError, ValueError) as e:
        made.append(f"summary failed: {e}")
    return made
