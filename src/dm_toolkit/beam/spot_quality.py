# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.1, 2026-07-01

"""Focal-spot quality scoring from a single intensity image."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import cv2

from .beam import as_gray, subtract_background, d4sigma

# Feature-vector field order -- the ML model input contract (keep stable)
FEATURE_NAMES = ("ellipticity", "asymmetry", "halo_frac", "ring_contrast",
                 "r_ee80", "peak_norm", "d4sigma")


# Features
@dataclass
class SpotFeatures:
    """Calibration-free descriptors of one spot (the ML model input)."""
    ellipticity: float  # 0 round .. ->1 elongated  (astigmatism)
    asymmetry: float  # |skewness| about principal axes (coma)
    halo_frac: float  # Energy fraction beyond the core (spherical)
    ring_contrast: float  # First secondary-ring / core peak ratio.
    r_ee80: float  # Radius enclosing 80% energy, px (size)
    peak_norm: float  # Energy-normalised peak (Strehl proxy)
    d4sigma: float  # Mean D4sigma diameter, px (size)
    cx: float = 0.0
    cy: float = 0.0
    theta_deg: float = 0.0  # Astigmatism axis
    ok: bool = True

    def vector(self) -> np.ndarray:
        """Ordered feature vector for a learned model (FEATURE_NAMES order)."""
        return np.array([getattr(self, n) for n in FEATURE_NAMES], float)


def _roi(img, cx, cy, half):
    """Square crop centred on (cx, cy) plus the new local centre.

    Args:
        img: Input image.
        cx: Horizontal centre coordinate, in pixels.
        cy: Vertical centre coordinate, in pixels.
        half: Half-resolution input data.
    """
    h, w = img.shape
    half = int(max(8, half))
    x0, x1 = max(0, int(cx) - half), min(w, int(cx) + half + 1)
    y0, y1 = max(0, int(cy) - half), min(h, int(cy) + half + 1)
    return img[y0:y1, x0:x1], cx - x0, cy - y0


def _moments2d(img):
    """Centroid + 2x2 covariance + standardised 3rd moments of an image.

    Returns cx, cy, (sxx, syy, sxy), (skew_u, skew_v), theta on the principal
    axes (u = major). theta is the major-axis angle (rad).
    """
    total = float(img.sum())
    if total <= 0:
        return None
    Y, X = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(float)
    cx = float((img * X).sum() / total)
    cy = float((img * Y).sum() / total)
    dx, dy = X - cx, Y - cy
    sxx = float((img * dx * dx).sum() / total)
    syy = float((img * dy * dy).sum() / total)
    sxy = float((img * dx * dy).sum() / total)
    theta = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)  # Major-axis angle
    c, s = np.cos(theta), np.sin(theta)
    u, v = dx * c + dy * s, -dx * s + dy * c  # Principal coords
    su = np.sqrt(max((img * u * u).sum() / total, 1e-9))
    sv = np.sqrt(max((img * v * v).sum() / total, 1e-9))
    skew_u = float((img * u ** 3).sum() / total / su ** 3)
    skew_v = float((img * v ** 3).sum() / total / sv ** 3)
    return cx, cy, (sxx, syy, sxy), (skew_u, skew_v), float(theta)


def _radial_profile(img, cx, cy):
    """Azimuthally-averaged intensity I(r) (r in integer px from centre).

    Args:
        img: Input image.
        cx: Horizontal centre coordinate, in pixels.
        cy: Vertical centre coordinate, in pixels.
    """
    Y, X = np.ogrid[0:img.shape[0], 0:img.shape[1]]
    r = np.hypot(X - cx, Y - cy).astype(int)
    sums = np.bincount(r.ravel(), weights=img.ravel())
    cnts = np.bincount(r.ravel())
    return sums / np.maximum(cnts, 1)


def _ring_contrast(prof):
    """Return ring contrast.

    Secondary-ring peak / core peak, measured only beyond the core (after
    the profile first falls below half the central peak). 0 when there is no
    resolvable ring -- stops a tiny near-origin dip reading as a full ring.
    """
    core = float(prof[0]) if len(prof) else 0.0
    if core <= 0 or len(prof) < 5:
        return 0.0
    below = np.where(prof < 0.5 * core)[0]  # End of the core.
    if not len(below) or len(prof) - below[0] < 3:
        return 0.0
    return float(prof[below[0]:].max() / core)


def _ee_radius(img, cx, cy, frac=0.80):
    """Radius (px) enclosing `frac` of the total energy (encircled energy).

    Args:
        img: Input image.
        cx: Horizontal centre coordinate, in pixels.
        cy: Vertical centre coordinate, in pixels.
        frac: Requested fractional level.
    """
    Y, X = np.ogrid[0:img.shape[0], 0:img.shape[1]]
    r = np.hypot(X - cx, Y - cy).ravel()
    w = img.ravel()
    order = np.argsort(r)
    cw = np.cumsum(w[order])
    total = cw[-1] if len(cw) else 0.0
    if total <= 0:
        return float("nan")
    return float(r[order][np.searchsorted(cw, frac * total)])


def extract_features(frame) -> SpotFeatures:
    """All calibration-free spot descriptors from one (gray/colour) frame."""
    img = subtract_background(as_gray(frame))
    m = d4sigma(img)
    if m is None:
        return SpotFeatures(*([float("nan")] * 7), ok=False)
    sig = max(m["sx"], m["sy"])
    crop, cx, cy = _roi(img, m["cx"], m["cy"], half=4.0 * sig)

    mm = _moments2d(crop)
    if mm is None:
        return SpotFeatures(*([float("nan")] * 7), ok=False)
    cxn, cyn, (sxx, syy, sxy), (sk_u, sk_v), theta = mm
    lam1 = 0.5 * (sxx + syy) + np.hypot(0.5 * (sxx - syy), sxy)
    lam2 = 0.5 * (sxx + syy) - np.hypot(0.5 * (sxx - syy), sxy)
    ellip = float(1.0 - np.sqrt(max(lam2, 0.0) / max(lam1, 1e-9)))
    asym = float(np.hypot(sk_u, sk_v))

    prof = _radial_profile(crop, cxn, cyn)
    r80 = _ee_radius(crop, cxn, cyn, 0.80)
    # Energy beyond 1.5x the 80% radius vs total (spherical/defocus halo)
    Y, X = np.ogrid[0:crop.shape[0], 0:crop.shape[1]]
    rr = np.hypot(X - cxn, Y - cyn)
    tot = float(crop.sum())
    halo = float(crop[rr > 1.5 * r80].sum() / tot) if tot > 0 else 0.0

    blur = cv2.GaussianBlur(crop.astype(np.float32), (0, 0), 1.5)
    peak_norm = float(blur.max() / tot) if tot > 0 else 0.0
    return SpotFeatures(
        ellipticity=ellip, asymmetry=asym, halo_frac=halo,
        ring_contrast=_ring_contrast(prof), r_ee80=r80,
        peak_norm=peak_norm, d4sigma=0.5 * (m["dx"] + m["dy"]),
        cx=m["cx"], cy=m["cy"], theta_deg=float(np.degrees(theta)))


# Scoring
@dataclass
class BatchContext:
    """Batch maxima for the relative components (same camera / session)."""
    best_peak_norm: float = 1.0  # Brightest energy-normalised peak.
    best_r_ee80: float = 1.0  # Smallest 80% radius (tightest)

    @classmethod
    def from_features(cls, feats: Sequence[SpotFeatures]) -> "BatchContext":
        ok = [f for f in feats if f.ok and np.isfinite(f.r_ee80)]
        if not ok:
            return cls()
        return cls(best_peak_norm=max(f.peak_norm for f in ok),
                   best_r_ee80=min(f.r_ee80 for f in ok))


@dataclass
class QualityReport:
    """Spot verdict: composite score + components + aberration diagnostic."""
    score: float
    components: dict  # Strehl / concentration / symmetry (0..1)
    aberrations: dict  # Astigmatism / coma / spherical penalties 0..1.
    dominant: str  # Worst aberration
    scale_notes: str
    ok: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def _sat(x, scale):
    """Saturating 0..1 penalty: 0 at x=0, ->1 as x grows (1 - exp(-x/scale)).

    Args:
        x: Input coordinate or scalar value.
        scale: Scale factor applied to the data.
    """
    return float(1.0 - np.exp(-max(x, 0.0) / scale)) if scale > 0 else 0.0


class Scorer(Protocol):
    """Common interface -- swap RuleBasedScorer for a future LearnedScorer."""
    def score(self, f: SpotFeatures, ctx: BatchContext) -> QualityReport: ...


@dataclass
class RuleBasedScorer:
    """Physics-proxy scorer.

    Saturation scales set how harshly each aberration symptom is punished; a
    learned scorer would replace these with weights.
    """
    coma_scale: float = 0.5  # |skewness| at which coma ~63% penalised.
    spherical_scale: float = 0.18  # Halo fraction at which spherical ~63%.

    def score(self, f: SpotFeatures, ctx: BatchContext) -> QualityReport:
        """Calculate the objective score.

        Args:
            f: Extracted feature record or callable.
            ctx: Scoring context.
        """
        if not f.ok:
            return QualityReport(0.0, {}, {}, "n/a", "invalid spot", ok=False)
        # Absolute, calibration-free aberration penalties (0 good .. 1 bad)
        astig = float(np.clip(f.ellipticity, 0.0, 1.0))
        coma = _sat(f.asymmetry, self.coma_scale)
        # Halo fraction is the robust spherical/defocus proxy; a real
        # secondary ring only adds to it.
        spher = _sat(f.halo_frac + 0.10 * f.ring_contrast, self.spherical_scale)
        symmetry = (1 - astig) * (1 - coma) * (1 - spher)
        # Batch-relative quality components.
        strehl = float(np.clip(f.peak_norm / max(ctx.best_peak_norm, 1e-12),
                               0.0, 1.0))
        conc = float(np.clip(ctx.best_r_ee80 / max(f.r_ee80, 1e-9), 0.0, 1.0))
        comps = dict(strehl=strehl, concentration=conc, symmetry=symmetry)
        # Geometric mean -- one bad axis cannot be hidden by another.
        score = float(np.cbrt(max(strehl, 1e-6) * max(conc, 1e-6)
                              * max(symmetry, 1e-6)))
        aberr = dict(astigmatism=astig, coma=coma, spherical=spher)
        dominant = max(aberr, key=aberr.get)
        notes = ("astig/coma/spherical absolute; "
                 "strehl/concentration batch-relative")
        return QualityReport(score, comps, aberr, dominant, notes)


@dataclass
class LearnedScorer:
    """Future hook: a trained model predicting quality from SpotFeatures.

    Same interface as RuleBasedScorer so the call site never changes. Train it
    on logged samples (log_sample) or on simulated PSFs with known Zernike.
    """
    model_path: Path | None = None

    def score(self, f: SpotFeatures, ctx: BatchContext) -> QualityReport:
        raise NotImplementedError(
            "LearnedScorer is a placeholder. Train a model on f.vector() "
            "(FEATURE_NAMES) and map its output into a QualityReport here.")


# Pipeline
def score_batch(frames: Sequence, scorer: Scorer | None = None):
    """Two-pass batch scoring (gather batch maxima, then score each frame).

    Returns list of (SpotFeatures, QualityReport).

    Args:
        frames: Captured image frames.
        scorer: Callable that evaluates one image batch.
    """
    scorer = scorer or RuleBasedScorer()
    feats = [extract_features(fr) for fr in frames]
    ctx = BatchContext.from_features(feats)
    return [(f, scorer.score(f, ctx)) for f in feats]


# Future-ML interfaces
def parse_dm_command_from_filename(name: str):
    """Parse 'a_b_c_d_e.jpg' -> (a, b, c, d, e) DM actuator bits, else None.

    The label hook: attaches the command that produced a spot for supervised
    learning. Returns None when the stem is not all-integer underscore tokens.
    """
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 2 or not all(p.lstrip("-").isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def log_sample(dataset_csv, name, feats: SpotFeatures, report: QualityReport,
               dm_command=None):
    """Append one scored spot to a growing dataset (training-data on-ramp).

    Each row = filename + DM command (if any) + features + score. Writes a
    header on first use. Minimal by design; the data accumulates as you score.

    Args:
        dataset_csv: Filesystem path for the dataset data.
        name: Display or identifier name.
        feats: Extracted image or wavefront features.
        report: Report data to render.
        dm_command: Applied deformable-mirror command.
    """
    dataset_csv = Path(dataset_csv)
    dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = parse_dm_command_from_filename(name) if dm_command is None else dm_command
    cmd_str = "_".join(map(str, cmd)) if cmd else ""
    new = not dataset_csv.exists()
    with open(dataset_csv, "a", encoding="utf-8") as fh:
        if new:
            fh.write("name,dm_command," + ",".join(FEATURE_NAMES)
                     + ",score,dominant\n")
        vals = ",".join(f"{v:.6g}" for v in feats.vector())
        fh.write(f"{name},{cmd_str},{vals},{report.score:.6g},"
                 f"{report.dominant}\n")
