# SPDX-License-Identifier: GPL-3.0-or-later

"""Analyse adaptive-settling calibration CSVs and write runtime descriptors."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from uuid import uuid4

from .model import (descriptor_document, empirical_quantile_higher,
                    quantile_confidence_interval,
                    quantile_lower_confidence_bound)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(row, key):
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def analyse_rows(rows):
    """Return per-channel residual statistics from formal measurement rows.

    A channel sign is selected from ±1 by whichever gives the smaller median
    absolute transition residual.  Gain is never fitted.  Closure rows and
    clamped transitions remain in the raw file but are not part of ordinary
    residual statistics.
    """
    grouped = {}
    for row in rows:
        if str(row.get("role")) != "transition":
            continue
        if str(row.get("valid", "1")).lower() in ("0", "false", "no"):
            continue
        if str(row.get("clamped", "0")).lower() in ("1", "true", "yes"):
            continue
        channel = int(row["channel"])
        actuated = str(row.get("actuated_channel") or "").strip()
        if actuated and channel != int(actuated):
            continue
        predicted = _number(row, "predicted_delta_nm")
        measured = _number(row, "measured_delta_nm")
        if predicted is None or measured is None:
            continue
        grouped.setdefault(channel, []).append((predicted, measured, row))
    out = {}
    for channel, values in grouped.items():
        candidates = {}
        for sign in (-1, 1):
            residuals = [abs(sign * measured - predicted)
                         for predicted, measured, _ in values]
            candidates[sign] = residuals
        sign = min(candidates, key=lambda candidate:
                   median(candidates[candidate]))
        residuals = candidates[sign]
        empirical = empirical_quantile_higher(residuals, 0.95)
        lower, rank = quantile_lower_confidence_bound(
            residuals, q=0.95, confidence=0.95)
        ci_low, ci_high, ci_low_rank, ci_high_rank = \
            quantile_confidence_interval(residuals, q=0.95, confidence=0.95)
        out[channel] = {
            "n": len(residuals), "sign": sign,
            "median_abs_transition_residual_nm": median(residuals),
            "empirical_p95_abs_transition_residual_nm": empirical,
            "p95_one_sided_95pct_lower_bound_nm": lower,
            "p95_lower_bound_order_rank": rank,
            "p95_distribution_free_95pct_ci_nm": [ci_low, ci_high],
            "p95_ci_order_ranks": [ci_low_rank, ci_high_rank],
            # Runtime gate specified in the design: use the conservative,
            # distribution-free bound and report the empirical P95 beside it.
            "egate_nm": max(lower, 1e-12),
            "residuals_nm": residuals,
        }
    return out


def analyse_session(session_dir: str | Path) -> dict:
    """Reanalyse ``formal_measurements.csv`` and regenerate report/JSON."""
    folder = Path(session_dir).expanduser().resolve()
    raw_path = folder / "formal_measurements.csv"
    with raw_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    stats = analyse_rows(rows)
    derived = _derive_position_errors(rows, stats)
    _write_derived_csv(folder / "derived_residuals.csv", derived)
    for channel, values in stats.items():
        own = [row for row in derived if row["channel"] == channel]
        positions = [abs(row["position_error_nm"]) for row in own
                     if row["role"] == "transition"]
        closures = [abs(row["position_error_nm"]) for row in own
                    if row["role"] == "closure_check"]
        values["empirical_p95_abs_position_error_nm"] = (
            empirical_quantile_higher(positions, 0.95) if positions else None)
        values["closure_abs_error_median_nm"] = (
            median(closures) if closures else None)
        values["closure_abs_error_max_nm"] = max(closures) if closures else None
    try:
        snapshot = json.loads((folder / "config_snapshot.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        snapshot = {}
    mirror = int(snapshot.get("mirror_actuators") or
                 (9 if any(channel > 5 for channel in stats) else 5))
    calibration_set_id = str(snapshot.get("calibration_set_id") or
                             f"adaptive-{datetime.now():%Y%m%d-%H%M%S}-"
                             f"{uuid4().hex[:8]}")
    channels_cfg = {int(item["channel"]): item
                    for item in snapshot.get("channels", [])
                    if isinstance(item, dict) and "channel" in item}
    written = []
    for channel, values in stats.items():
        cfg = channels_cfg.get(channel, {})
        profile = str(cfg.get("compensation_profile") or "")
        profile_hash = ""
        if profile and Path(profile).is_file():
            profile_hash = file_sha256(profile)
        doc = descriptor_document(
            channel=channel, mirror_actuators=mirror,
            egate_nm=values["egate_nm"],
            calibration_set_id=calibration_set_id,
            source_profile=profile, source_profile_sha256=profile_hash,
            statistics={k: v for k, v in values.items()
                        if k != "residuals_nm"})
        path = folder / f"adaptive_settling_ch{channel}.json"
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        written.append(str(path))
    report = {
        "schema_version": 1, "calibration_set_id": calibration_set_id,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "formal_csv": str(raw_path), "mirror_actuators": mirror,
        "channels": {str(k): {name: value for name, value in v.items()
                              if name != "residuals_nm"}
                     for k, v in stats.items()},
        "descriptors": written,
    }
    (folder / "adaptive_settling_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown_report(folder / "adaptive_settling_report.md", report)
    _write_residual_plot(folder / "adaptive_settling_residuals.png", stats)
    manifest = {
        "schema_version": 1, "calibration_set_id": calibration_set_id,
        "mirror_actuators": mirror,
        "channels": {str(channel): {
            "path": str(folder / f"adaptive_settling_ch{channel}.json"),
            "sha256": file_sha256(
                folder / f"adaptive_settling_ch{channel}.json")}
            for channel in stats},
    }
    (folder / "adaptive_settling_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def _derive_position_errors(rows, stats):
    """Build auditable step and window-relative residuals after sign choice."""
    baselines = {}
    closure_baselines = {}
    for row in rows:
        role = str(row.get("role"))
        if role not in ("baseline", "closure_baseline"):
            continue
        if str(row.get("valid", "1")).lower() in ("0", "false", "no"):
            continue
        if str(row.get("clamped", "0")).lower() in ("1", "true", "yes"):
            continue
        channel = int(row["channel"])
        actuated = str(row.get("actuated_channel") or "")
        key = (str(row.get("source") or ""), str(row.get("window") or ""),
               actuated, channel)
        predicted = _number(row, "predicted_position_nm")
        measured = _number(row, "mx_position_nm")
        if predicted is not None and measured is not None:
            target = closure_baselines if role == "closure_baseline" else baselines
            target[key] = (predicted, measured)
    out = []
    for row in rows:
        role = str(row.get("role") or "")
        if role not in ("transition", "closure_check"):
            continue
        if str(row.get("valid", "1")).lower() in ("0", "false", "no"):
            continue
        if str(row.get("clamped", "0")).lower() in ("1", "true", "yes"):
            continue
        channel = int(row["channel"])
        actuated = str(row.get("actuated_channel") or "")
        if actuated and role == "transition" and channel != int(actuated):
            continue
        if channel not in stats:
            continue
        key = (str(row.get("source") or ""), str(row.get("window") or ""),
               actuated, channel)
        baseline = (closure_baselines.get(key) if role == "closure_check"
                    else baselines.get(key))
        predicted = _number(row, "predicted_position_nm")
        measured = _number(row, "mx_position_nm")
        if baseline is None or predicted is None or measured is None:
            continue
        sign = int(stats[channel]["sign"])
        position_error = (sign * (measured - baseline[1])
                          - (predicted - baseline[0]))
        predicted_delta = _number(row, "predicted_delta_nm")
        measured_delta = _number(row, "measured_delta_nm")
        transition_residual = (None if predicted_delta is None
                               or measured_delta is None else
                               sign * measured_delta - predicted_delta)
        out.append({
            "source": key[0], "window": key[1], "role": role,
            "actuated_channel": actuated, "channel": channel,
            "sign": sign, "position_error_nm": position_error,
            "transition_residual_nm": transition_residual,
            "abs_transition_residual_nm": (None if transition_residual is None
                                           else abs(transition_residual)),
        })
    return out


def _write_derived_csv(path, rows):
    fields = ("source", "window", "role", "actuated_channel", "channel",
              "sign", "position_error_nm", "transition_residual_nm",
              "abs_transition_residual_nm")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(path: Path, report: dict) -> None:
    lines = [
        "# Adaptive Settling Calibration Report", "",
        f"Calibration set: `{report['calibration_set_id']}`", "",
        "| Channel | n | Sign | Median residual (nm) | Empirical P95 (nm) | "
        "P95 distribution-free 95% CI (nm) | Runtime Egate: one-sided 95% "
        "P95 lower bound (nm) | Position-error P95 (nm) | Closure median / "
        "max (nm) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for channel, values in report["channels"].items():
        lines.append(
            f"| {channel} | {values['n']} | {values['sign']:+d} | "
            f"{values['median_abs_transition_residual_nm']:.6g} | "
            f"{values['empirical_p95_abs_transition_residual_nm']:.6g} | "
            f"{values['p95_distribution_free_95pct_ci_nm'][0]:.6g}–"
            f"{values['p95_distribution_free_95pct_ci_nm'][1]:.6g} | "
            f"{values['egate_nm']:.6g} | "
            f"{_fmt_optional(values.get('empirical_p95_abs_position_error_nm'))} | "
            f"{_fmt_optional(values.get('closure_abs_error_median_nm'))} / "
            f"{_fmt_optional(values.get('closure_abs_error_max_nm'))} |")
    lines += [
        "", "Clamped, invalid, and closure-check rows are retained in the raw "
        "CSV but excluded from ordinary transition statistics.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_optional(value):
    return "n/a" if value is None else f"{float(value):.6g}"


def _write_residual_plot(path: Path, stats: dict) -> None:
    try:
        from matplotlib.figure import Figure
    except ImportError:
        return
    figure = Figure(figsize=(8, 4.5), tight_layout=True)
    axis = figure.add_subplot(111)
    for channel, values in sorted(stats.items()):
        residuals = sorted(values["residuals_nm"])
        if not residuals:
            continue
        q = [(index + 1) / len(residuals) for index in range(len(residuals))]
        axis.plot(residuals, q, marker=".", ms=3, lw=1, label=f"ch{channel}")
        axis.axvline(values["egate_nm"], alpha=0.25, lw=0.8)
    axis.set_xlabel("Absolute transition residual (nm)")
    axis.set_ylabel("Empirical cumulative probability")
    axis.set_ylim(0, 1.02)
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    figure.savefig(path, dpi=160)
