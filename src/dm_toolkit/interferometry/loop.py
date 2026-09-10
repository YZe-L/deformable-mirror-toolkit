# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-06-12

"""Hysteresis-loop pipeline: drive CSV mapping, branches, metrics, figure."""

import os
import csv
import glob

import numpy as np

LAMBDA_NM = 520.0  # Laser wavelength
STEP_LIMIT = LAMBDA_NM / 4  # Max resolvable motion per frame (phase = pi)

IMG_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


def list_sequence(folder):
    """List sequence.

    Image sequence of a folder: Snapshot_*.jpg preferred, else any
    common image type, name-sorted.
    """
    files = sorted(glob.glob(os.path.join(folder, "Snapshot_*.jpg")))
    if not files:
        files = sorted(p for pat in IMG_PATTERNS
                       for p in glob.glob(os.path.join(folder, pat)))
    return files


def bit_sequence(lo, hi, step):
    """Drive bits for one sweep.

    Drive bits for one sweep: up lo..hi (top included once) then back
    down mirroring the up steps. Returns (sequence, turn index).

    Args:
        lo: Lower bound.
        hi: Upper bound.
        step: Increment between consecutive values.
    """
    up = list(range(lo, hi, step))
    if not up or up[-1] != hi:
        up.append(hi)
    return up + up[:-1][::-1], len(up) - 1


def to_displacement(dphi):
    """Per-step phases -> displacement in nm, up sweep positive."""
    d = np.concatenate([[0.0], np.cumsum(dphi)]) * LAMBDA_NM / (4 * np.pi)
    return -d if d[len(d) // 2] < 0 else d


def steps_from_displacement(d):
    """Displacement (nm) -> per-step phases (rad), inverse of cumsum."""
    return np.diff(np.asarray(d, float)) * (4 * np.pi) / LAMBDA_NM


# Drive csv
def read_csv(path):
    """CSV -> (column names, float matrix); non-numeric cells become nan.

    First row is treated as a header when it does not parse as numbers.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("empty csv")

    def num(c):
        try:
            return float(c)
        except ValueError:
            return np.nan

    width = max(len(r) for r in rows)
    if all(np.isnan(num(c)) for c in rows[0] if c.strip()):
        names = [c.strip() or f"col{i + 1}" for i, c in enumerate(rows[0])]
        body = rows[1:]
    else:
        names = []
        body = rows
    names += [f"col{i + 1}" for i in range(len(names), width)]
    data = np.full((len(body), width), np.nan)
    for i, r in enumerate(body):
        for j, c in enumerate(r):
            data[i, j] = num(c)
    return names, data


def take(data, spec):
    """Take.

    One column segment; spec = (col, row_lo, row_hi), hi exclusive,
    non-numeric rows dropped.

    Args:
        data: Input data used by the operation.
        spec: Measurement, plot, or waveform specification.
    """
    col, lo, hi = spec
    v = data[lo:hi, col]
    return v[np.isfinite(v)]


def full_span(spec, n_rows):
    return spec[1] <= 0 and spec[2] >= n_rows


def drive_from_csv(data, bit_spec, up_spec, dn_spec):
    """Selected segments -> drive dict. specs: (column, row_lo, row_hi).

    up == dn segment        -> 'sequence': rows are frames in sweep order.
    distinct full columns   -> 'levels': one row per bit level (legacy).
    anything else           -> 'segments': explicit up/down frame series
                               (up and down may share one column).

    Args:
        data: Input data used by the operation.
        bit_spec: Column specification for command bits.
        up_spec: Column specification for increasing-branch commands.
        dn_spec: Column specification for decreasing-branch commands.
    """
    bit_spec, up_spec, dn_spec = (tuple(int(x) for x in s)
                                  for s in (bit_spec, up_spec, dn_spec))
    if max(s[0] for s in (bit_spec, up_spec, dn_spec)) >= data.shape[1]:
        raise ValueError(f"csv has only {data.shape[1]} columns")
    bits = take(data, bit_spec)
    vup = take(data, up_spec)
    if not len(bits) or not len(vup):
        raise ValueError("selected columns contain no numeric rows")
    if up_spec == dn_spec:
        m = min(len(bits), len(vup))
        return dict(mode="sequence", bits=bits[:m].astype(int), V=vup[:m])
    vdn = take(data, dn_spec)
    n = len(data)
    if up_spec[0] != dn_spec[0] and all(full_span(s, n) for s in
                                        (bit_spec, up_spec, dn_spec)):
        m = min(len(bits), len(vup), len(vdn))
        return dict(mode="levels", bits=bits[:m].astype(int),
                    vup=vup[:m], vdn=vdn[:m])
    return dict(mode="segments", bits=bits.astype(int), vup=vup, vdn=vdn)


def check_drive(bits, *cols):
    """Report (never fix) voltage entries breaking the local trend.

    Args:
        bits: PWM command samples from the measurement table.
        *cols: Sequence of `(label, values)` column pairs.

    Returns:
        Warning messages for values that violate the drive specification.
    """
    msgs = []
    for label, v in cols:
        for i in range(1, len(v) - 1):
            lo, hi = v[i - 1], v[i + 1]
            margin = 0.5 * abs(hi - lo) + 2.0
            if not (min(lo, hi) - margin <= v[i] <= max(lo, hi) + margin):
                msgs.append(f"{label} @ bit {bits[i]}: {v[i]} "
                            f"(neighbours {lo} / {hi}) -- suspicious")
    return msgs


# Branches
def split_branches(d, bits, vup, vdn):
    """Split branches.

    Branches for the legacy per-level layout, anchored at the
    displacement turnaround (top bit photographed once).

    Args:
        d: Displacement or distance samples.
        bits: PWM command bits.
        vup: Voltages sampled on the increasing branch.
        vdn: Voltages sampled on the decreasing branch.
    """
    t = int(np.argmax(d))
    n = len(d)
    m_up, m_dn = t + 1, n - t - 1
    notes = []
    if m_up != len(bits):
        notes.append(f"up branch {m_up}/{len(bits)} frames")
    if m_dn != len(bits) - 1:
        notes.append(f"down branch {m_dn}/{len(bits) - 1} frames")
    up_V, dn_V = vup[-m_up:], vdn[::-1][1:][:m_dn]
    return dict(t=t, m_up=m_up,
                up_V=up_V, d_up=d[:m_up],
                dn_V=dn_V, d_dn=d[m_up:],
                V_seq=np.concatenate([up_V, dn_V]),
                bit_seq=np.concatenate([bits[-m_up:], bits[::-1][1:][:m_dn]]),
                branch=np.array(["up"] * m_up + ["down"] * m_dn),
                notes=notes)


def split_sequence(d, bits, V):
    """Split sequence.

    Branches when csv rows are frames in sweep order (one V column);
    turnaround = highest bit. Arrays are clipped to the common length.

    Args:
        d: Displacement or distance samples.
        bits: PWM command bits.
        V: Voltage samples.
    """
    n = min(len(d), len(bits))
    notes = []
    if len(bits) != len(d):
        notes.append(f"csv rows {len(bits)} != frames {len(d)} -- "
                     f"clipped to {n}")
    d, bits, V = d[:n], bits[:n], V[:n]
    t = int(np.argmax(bits))
    return dict(t=t, m_up=t + 1,
                up_V=V[:t + 1], d_up=d[:t + 1],
                dn_V=V[t + 1:], d_dn=d[t + 1:],
                V_seq=V, bit_seq=bits,
                branch=np.array(["up"] * (t + 1) + ["down"] * (n - t - 1)),
                notes=notes)


def split_segments(d, bits, vup, vdn):
    """Split segments.

    Branches from explicit up/down voltage series (frames in sweep
    order, possibly cut from one column); turnaround = last up frame.

    Args:
        d: Displacement or distance samples.
        bits: PWM command bits.
        vup: Voltages sampled on the increasing branch.
        vdn: Voltages sampled on the decreasing branch.
    """
    m_up, m_dn = len(vup), len(vdn)
    notes = []
    n = min(len(d), m_up + m_dn)
    if m_up + m_dn != len(d):
        notes.append(f"csv up+down rows {m_up + m_dn} != frames {len(d)} "
                     f"-- clipped to {n}")
    d = d[:n]
    V = np.concatenate([vup, vdn])[:n]
    t = min(m_up, n) - 1
    # Bit rows: full sweep, per-level (mirror the up bits), else pad.
    if len(bits) == m_up + m_dn:
        bit_seq = np.asarray(bits[:n])
    elif len(bits) == m_up:
        bit_seq = np.concatenate([bits, bits[-2::-1][:m_dn]])[:n]
    else:
        bit_seq = np.full(n, -1, dtype=int)
        bit_seq[:min(len(bits), n)] = bits[:min(len(bits), n)]
        notes.append(f"bit rows {len(bits)} match neither the sweep "
                     f"({m_up + m_dn}) nor the up branch ({m_up})")
    return dict(t=t, m_up=t + 1,
                up_V=V[:t + 1], d_up=d[:t + 1],
                dn_V=V[t + 1:], d_dn=d[t + 1:],
                V_seq=V, bit_seq=bit_seq,
                branch=np.array(["up"] * (t + 1) + ["down"] * (n - t - 1)),
                notes=notes)


def hysteresis_metrics(br, dphi):
    """Stroke, remanent, max loop width; flag steps at the lambda/4 limit.

    Args:
        br: Branch or fitted-branch data.
        dphi: Wrapped phase-step samples, in radians.
    """
    steps_nm = np.abs(dphi) * LAMBDA_NM / (4 * np.pi)
    ambiguous = np.where(steps_nm > 0.83 * STEP_LIMIT)[0]
    V_bad = br["V_seq"][ambiguous.min()] if len(ambiguous) else -1.0

    # Both branches on a common voltage grid, widest reliable gap.
    Vg = np.linspace(max(br["up_V"].min(), br["dn_V"].min()),
                     min(br["up_V"].max(), br["dn_V"].max()), 400)
    gap = (np.interp(Vg, br["dn_V"][::-1], br["d_dn"][::-1])
           - np.interp(Vg, br["up_V"], br["d_up"]))
    gi = int(np.argmax(np.abs(np.where(Vg > V_bad, gap, 0))))
    stroke = br["d_up"][-1] - min(br["d_up"].min(), 0)
    return dict(stroke=stroke, max_h=abs(gap[gi]), V_at=Vg[gi],
                remanent=br["d_dn"][-1] - br["d_up"][0],
                ambiguous=ambiguous, V_bad=V_bad, Vg=Vg, gap=gap)


def draw_loop(ax, br, met, full=True):
    """Draw loop.

    Hysteresis loop: raw points in sweep order, closed with a dashed
    remanent segment, max-hysteresis arrow.

    Args:
        ax: Matplotlib axes to draw on.
        br: Branch or fitted-branch data.
        met: Calculated measurement metrics.
        full: Full-resolution input data.
    """
    Vc = np.concatenate([br["V_seq"], br["V_seq"][:1]])
    dc = np.concatenate([br["d_up"], br["d_dn"], br["d_up"][:1]])
    ax.fill(Vc, dc, color="tab:blue", alpha=0.10, lw=0)

    ms, lw = (4.5, 1.8) if full else (3, 1.2)
    ax.plot(br["up_V"], br["d_up"], "o-", ms=ms, lw=lw, color="tab:blue",
            label="up sweep")
    ax.plot(np.concatenate([br["up_V"][-1:], br["dn_V"]]),
            np.concatenate([br["d_up"][-1:], br["d_dn"]]),
            "s-", ms=ms, lw=lw, color="tab:orange", label="down sweep")
    ax.plot([br["dn_V"][-1], br["up_V"][0]], [br["d_dn"][-1], br["d_up"][0]],
            "k--", lw=1.2)

    scale = 30 if full else 16
    for V, d in ((br["up_V"], br["d_up"]), (br["dn_V"], br["d_dn"])):
        for f in (0.3, 0.65):
            i = int(f * (len(V) - 2))
            ax.annotate("", xy=(V[i + 1], d[i + 1]), xytext=(V[i], d[i]),
                        arrowprops=dict(arrowstyle="-|>",
                                        lw=2.5 if full else 1.5,
                                        mutation_scale=scale,
                                        color="tab:blue" if d is br["d_up"]
                                        else "tab:orange"))

    y_up = np.interp(met["V_at"], br["up_V"], br["d_up"])
    y_dn = y_up + np.interp(met["V_at"], met["Vg"], met["gap"])
    ax.annotate("", xy=(met["V_at"], y_dn), xytext=(met["V_at"], y_up),
                arrowprops=dict(arrowstyle="<->", color="crimson",
                                lw=3 if full else 2, mutation_scale=scale))
    ax.annotate(f"max hysteresis\n{met['max_h']:.0f} nm @ {met['V_at']:.0f} V",
                xy=(met["V_at"], (y_up + y_dn) / 2),
                xytext=(met["V_at"] + 16, y_up - met["stroke"] * 0.2),
                fontsize=11 if full else 8, color="crimson",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1))

    if full:
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("piezo voltage (V)")
    ax.set_ylabel("displacement (nm)")
    ax.legend(fontsize=9 if full else 7, loc="upper left")
