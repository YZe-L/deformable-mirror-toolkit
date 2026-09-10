# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.21, 2026-08-25

"""Closed-loop configuration: actuators, algorithm choice, metric, params."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, fields

from .fastmath import (BACKEND_NUMPY, BACKENDS,  # noqa: F401 (re-export)
                       BACKEND_LABELS as REDUCTION_LABELS)


# Algorithm / metric identifiers (also the combo-box keys in the UI)
ALGO_HILL = "hill_climb"  # Coordinate descent -- best for few actuators.
ALGO_SPGD = "spgd"  # Stochastic parallel gradient -- best for many.
ALGO_GENETIC = "genetic"  # Population search -- global, slow.
ALGO_CMAES = "cmaes"  # CMA-ES -- robust global, learns the landscape.
ALGO_BO = "bayes"  # GP Bayesian optimisation -- fewest measurements.
ALGO_SA = "anneal"  # Simulated annealing -- Metropolis random walk.
# Model-based solves: drive the measured eigenmodes, read a quadratic metric
# and solve for the coefficients, so they need a stored impact matrix.
ALGO_MODAL_FIT = "modal_fit"  # 2N+1: fit one parabola per mode.
ALGO_MODAL_FAST = "modal_fast"  # N+2: reuse the impact matrix's curvature.
ALGO_MODAL_PSD = "modal_psd"  # 2N+1 on 1/PSD, the Debarre merit function.
MODAL_ALGOS = (ALGO_MODAL_FIT, ALGO_MODAL_FAST, ALGO_MODAL_PSD)
# Focal-plane wavefront sensing: fits the wavefront to one averaged frame and
# needs the pupil maps of a full `influence_matrix.npz`.
ALGO_WFS = "wfs"
WFS_ALGOS = (ALGO_WFS,)
# Everything that SOLVES rather than searches, and so may hand a good starting
# point to a polish search.
SOLVER_ALGOS = MODAL_ALGOS + WFS_ALGOS
# A modal solve is a coarse stage: its metric goes flat near the diffraction
# limit, so it can hand over to a search. NONE means stop after the solve.
POLISH_NONE = "none"

# Metric identifiers used by the UI and optimizer.
METRIC_PIB = "pib"  # Power-in-bucket at r0 = EE-Strehl (default)
METRIC_PEAK = "peak"  # Peak / whole-frame energy (Strehl proxy)
METRIC_SHARP = "sharpness"  # sum(I^2)/(sum I)^2 -- centre-free concentration
METRIC_R_EE80 = "r_ee80"  # 80% encircled-energy radius (smaller better)
METRIC_RMS = "rms"  # D4sigma diameter (smaller better)
# Image power spectrum over a normalised-frequency annulus, divided by the DC
# term (the Debarre/Booth merit function); quadratic over low frequencies.
METRIC_PSD = "psd_band"
# Centroid-referenced second moment, <r^2> = sum(I r^2)/sum(I): exactly
# quadratic in the aberration, blind to tilt, flat near the limit.
METRIC_SECOND_MOMENT = "second_moment"

# How the N frames of one measurement become the score the optimiser climbs.
SCORE_AVG_FRAME = "avg_frame"  # Average the frames -> one image -> one score.
SCORE_AVG_SCORE = "avg_score"  # Score each frame -> mean of the scores.
SCORE_MODE_LABELS = {SCORE_AVG_FRAME: "Average frames, then score",
                     SCORE_AVG_SCORE: "Score each frame, then average"}

# How much time one point is allowed to cost (see core.budget).
SPEED_FIXED = "fixed"  # Always the typed settle + frames.
SPEED_AUTO = "auto"  # Cheaper whenever the optimiser can live with it.
SPEED_LABELS = {SPEED_FIXED: "Fixed - always use the values below",
                SPEED_AUTO: "Auto - fast while rough, slow when close"}
# Algorithms that derive how small a difference their next step must
# resolve; extend only together with a `precision()` method.
SPEED_AUTO_ALGOS = (ALGO_HILL,)

# Human labels for the pickers.
ALGO_LABELS = {ALGO_HILL: "Search: hill-climb (few actuators)",
               ALGO_SPGD: "Search: SPGD (many actuators)",
               ALGO_GENETIC: "Search: genetic (global, slow)",
               ALGO_CMAES: "Search: CMA-ES (robust global)",
               ALGO_BO: "Search: Bayesian opt (fewest measurements)",
               ALGO_SA: "Search: simulated annealing (global)",
    ALGO_MODAL_FIT:
                   "Solve: second moment, measure curvature (2N+1, safest)",
    ALGO_MODAL_FAST:
                   "Solve: second moment + matrix curvature (N+2, fastest)",
    ALGO_MODAL_PSD:
                   "Solve: 1/PSD, measure curvature (2N+1, set band)",
    ALGO_WFS:      "Wavefront: focal-plane sensing (1 shot/round)",
}
# Shown next to the algorithm picker, so the choice can be made without reading
# the papers. Each says what it costs and when it is the wrong pick.
ALGO_HELP = {
    ALGO_HILL: "Tries one actuator at a time and keeps what helps. No "
               "calibration needed. Slow but hard to fool.",
    ALGO_SPGD: "Dithers every actuator at once and follows the gradient. "
               "Scales to many actuators; needs a low-noise metric.",
    ALGO_GENETIC: "Population search. Finds a global optimum from a bad start, "
                  "at the price of thousands of measurements.",
    ALGO_CMAES: "Learns the shape of the landscape while searching. Robust, "
                "moderate cost. Needs the `cma` package.",
    ALGO_BO: "Builds a model of the landscape and probes where it expects most "
             "gain. Fewest measurements of any SEARCH. Needs `skopt`.",
    ALGO_SA: "Random walk that accepts occasional worsening to escape local "
             "optima. Global, but needs patience.",
    ALGO_MODAL_FIT:
        "Uses fixed-ROI spot second moment. Drives each measured eigenmode to "
        "+bias and -bias, fits a parabola, "
        "and jumps to its vertex. Assumes nothing beyond the metric being "
        "quadratic -- the curvature is measured, not taken from the matrix. "
        "It is a coarse wavefront-gradient objective, not the final spot-"
        "quality verdict. Start here.",
    ALGO_MODAL_FAST:
        "Also uses fixed-ROI spot second moment, but takes each mode's "
        "curvature from the impact matrix "
        "in the RMS modal coordinates instead of measuring it, so one probe "
        "per mode is enough. The N+2 core estimate is roughly half the cost "
        "of the fit; the optional startup bias scan costs 8 more measurements. "
        "Accuracy rests on the matrix being current for THIS mirror. One "
        "extra anchor probe calibrates the overall camera/optics scale in-place. "
        "Verify or polish on PIB/peak because unmodelled modes can alias into "
        "the second-moment estimate.",
    ALGO_MODAL_PSD:
        "The published Debarre/Booth form: same three-point solve, but on the "
        "reciprocal of the PSD band rather than the second moment. Only "
        "approximately quadratic, so its capture range depends on the band -- "
        "run Band Scan first. Use it to reproduce the paper, not to go faster.",
    ALGO_WFS:
        "Does not read the merit function at all. Fits Noll 4-11 to the "
        "averaged frame of ONE measurement, projects that wavefront onto the "
        "mirror's measured eigenmodes and drives the bounded least-squares "
        "answer -- so a round costs one shot instead of 2N+1, and the command "
        "lies inside the controllable subspace by construction. Needs an "
        "Impact Matrix influence_matrix.npz (the loop's saved calibration "
        "carries no pupil maps) and the optics block. The matrix is a local "
        "slope, so it iterates; the metric you pick is the JUDGE that keeps "
        "or discards each round, not the objective. First fit costs 14-22 s, "
        "later ones 156 ms.",
}
# Every metric reads 0..1 against the diffraction limit (1.0 = perfect).
METRIC_LABELS = {METRIC_PIB: "Power-in-bucket at r0 (EE-Strehl)",
                 METRIC_PEAK: "Normalised peak (Strehl)",
                 METRIC_SHARP: "Sharpness (centre-free concentration)",
                 METRIC_PSD: "Low-frequency PSD band (banded sharpness)",
                 METRIC_SECOND_MOMENT: "Spot second moment <r^2> (exactly quadratic)",
                 METRIC_R_EE80: "Encircled-energy radius",
                 METRIC_RMS: "RMS spot radius"}

# Which knobs each algorithm actually reads, verified against optimizers.py.
# Every other field in LoopSettings is inert for that algorithm.
_MODAL_KNOBS = ("modal_bias_rad", "modal_auto_bias", "modal_bias_ladder",
                "modal_bias_tolerance", "modal_bit_step", "modal_modes",
                "modal_auto_rounds", "modal_rounds", "modal_compensate",
                "modal_line_search")
_WFS_KNOBS = ("wfs_influence_npz", "wfs_keep_modes", "wfs_rounds",
              "wfs_max_step_bit", "wfs_roi_px")
ALGO_PARAMS = {
    ALGO_HILL:    ("move_step", "noise_threshold"),
    ALGO_WFS:     _WFS_KNOBS,
    ALGO_SPGD:    ("spgd_perturb", "spgd_gain"),
    ALGO_GENETIC: ("ga_population", "ga_generations", "ga_mutation"),
    ALGO_CMAES:   ("cma_popsize", "cma_sigma"),
    ALGO_BO:      ("bo_init_points", "bo_budget", "bo_xi"),
    ALGO_SA:      ("sa_t0", "sa_cooling", "sa_step"),
    # All three modal solves read the same knobs: how hard to push each mode,
    # how many modes to trust, and how many times to repeat the solve.
    ALGO_MODAL_FIT:  _MODAL_KNOBS,
    ALGO_MODAL_FAST: _MODAL_KNOBS,
    ALGO_MODAL_PSD:  _MODAL_KNOBS,
}

# The metric each modal solve REQUIRES. Its arithmetic is derived for that one
# quantity, so the UI pins it rather than letting an incompatible pick through:
# a parabola fitted to power-in-bucket would return a confident wrong vertex.
MODAL_METRIC = {ALGO_MODAL_FIT: METRIC_SECOND_MOMENT,
                ALGO_MODAL_FAST: METRIC_SECOND_MOMENT,
                ALGO_MODAL_PSD: METRIC_PSD}

# What the pickers offer. Retired entries stay defined so an old run record
# still loads and replays.
METRICS_OFFERED = (METRIC_PEAK, METRIC_PSD, METRIC_RMS)
_METRICS_RETIRED = (METRIC_PIB, METRIC_SHARP, METRIC_R_EE80,
                    METRIC_SECOND_MOMENT)

# Modal solves: only the 1/PSD form is offered. MODAL_FIT and MODAL_FAST both
# invert the second moment and neither produced a usable correction on this
# bench; they are retired rather than deleted so the comparison can be redone.
MODAL_ALGOS_OFFERED = (ALGO_MODAL_PSD,)
_MODAL_ALGOS_RETIRED = (ALGO_MODAL_FIT, ALGO_MODAL_FAST)

# The searches, in one tuple, so every consumer offers the same list.
SEARCH_ALGOS_OFFERED = (ALGO_HILL, ALGO_GENETIC, ALGO_BO, ALGO_SA)
# Retired from the pickers, not from the code: both still build, run and
# replay. Moving a name into the tuple above puts it back.
_SEARCH_ALGOS_RETIRED = (ALGO_SPGD, ALGO_CMAES)
ALGOS_OFFERED = SEARCH_ALGOS_OFFERED + MODAL_ALGOS_OFFERED + WFS_ALGOS
# Search-command quantisation and stopping grid. Modal vectors use
# ``modal_bit_step`` so their small simultaneous components are not erased.
_ALGO_SHARED = ("min_step",)


def default_min_step(algorithm: str) -> int:
    """Return the command grid appropriate for an algorithm.

    Args:
        algorithm: Algorithm identifier from ``ALGO_*``.

    Returns:
        Grid spacing in command bits.
    """
    return 1 if algorithm == ALGO_SPGD else 50


def default_cma_popsize(n: int) -> int:
    """Return CMA-ES's own default population for `n` search axes.

    The `cma` package's rule, named here because `cma_popsize = 0` means "use
    it" and the panel that offers that 0 has to be able to say what it will
    resolve to for the axis count in play.

    Args:
        n: Number of actuators being searched.

    Returns:
        Samples per generation.
    """
    return 4 + int(3 * math.log(max(1, int(n))))

# Which shaping knobs each metric reads: bucket_radius_px sizes the
# power-in-bucket circle, symmetry_weight drives the shape gate.
METRIC_PARAMS = {
    METRIC_PIB:    ("bucket_radius_px",),
    METRIC_PEAK:   (),
    # Sharpness has no bucket and needs no shape gate: fragmentation already
    # divides it by N, so there is nothing for a gate to add.
    METRIC_SHARP:  (),
    # Same family as sharpness (no bucket, no gate), but the band it integrates
    # over is a real choice, so both edges are part of the record.
    METRIC_PSD:    ("psd_m_lo", "psd_m_hi"),
    # The run locks one aperture from the Start-bit pattern; resizing it
    # between probes would change the objective.
    METRIC_SECOND_MOMENT:    ("second_moment_roi_margin_pct",),
    METRIC_R_EE80: ("bucket_radius_px", "symmetry_weight"),
    METRIC_RMS:    ("bucket_radius_px", "symmetry_weight"),
}
_METRIC_SHARED = ("roundness_weight", "astig_weight", "upsample_small_px",
                  "upsample_factor")

_MEASUREMENT = ("settle_ms", "frames_per_measure", "score_mode",
                "speed_mode", "speed_floor_ms", "speed_floor_frames",
                "exposure_settle_ms", "reduction_backend",
                "use_background", "background_frames",
                "background_refresh_points")
_HOLD = ("disturb_drop_frac", "occlusion_drop_frac")
_OPTICS = ("wavelength_nm", "focal_mm", "aperture_mm", "pixel_um")
_DISPLAY_ONLY = ("score_contrast",)


@dataclass
class Actuator:
    """One piezo: a Pi PWM channel with a bit range and a starting point.

    The loop addresses actuators by `channel` throughout. `name` is derived
    from it rather than stored, so a plot legend or log header can never name a
    channel other than the one the bits were sent to.
    """
    channel: int
    start: int = 2000  # Bit applied at loop start (the search seed)
    bit_min: int = 0
    bit_max: int = 4095

    @property
    def name(self):
        """str: Display label, for plot legends and log headers."""
        return f"ch{self.channel}"


# Physical Pi PWM channels wired to the two mirrors. These ranges are also the
# stable mirror identities used by the dual-mirror coordinator.
DM5 = 5
DM9 = 9
LAYOUTS = {DM5: tuple(range(1, 6)), DM9: tuple(range(6, 15))}
DEFAULT_LAYOUT = 5
DEFAULT_CHANNELS = list(LAYOUTS[DEFAULT_LAYOUT])

# First-release run plans. The expansion function lives in dual_mirror.py so a
# future alternating or joint strategy can reuse the same public plan seam.
PLAN_DM5 = "dm5"
PLAN_DM9 = "dm9"
PLAN_DM5_DM9 = "dm5_then_dm9"
PLAN_DM9_DM5 = "dm9_then_dm5"
# One search over both mirrors at once: 14 axes, one command, one score. The
# sequential plans hand a frozen mirror to the next stage and so cannot reach a
# correction that needs the two mirrors moved together.
PLAN_JOINT = "dm5_dm9_joint"
PLAN_LABELS = {
    PLAN_DM5: "DM5 only",
    PLAN_DM9: "DM9 only",
    PLAN_DM5_DM9: "DM5 -> DM9",
    PLAN_DM9_DM5: "DM9 -> DM5",
    PLAN_JOINT: "DM5 + DM9 together (14 piezos)",
}
# Stage identity of the joint run. Not a member of LAYOUTS: it is not a mirror
# anyone can edit, and `default_actuators`/`_canonical_actuators` must keep
# refusing it. It is the channel COUNT, like DM5 and DM9 are.
DM_JOINT = 14
JOINT_CHANNELS = LAYOUTS[DM5] + LAYOUTS[DM9]

# Joint-run knobs. Bit-scale knobs are a displacement per bit, so each
# channel keeps its own mirror's value (see optimizers._Base._per_channel).
PER_CHANNEL_KNOBS = ("min_step", "move_step", "spgd_perturb", "spgd_gain",
                     "ga_mutation", "cma_sigma", "sa_step")
# Measurement knobs: one point of a joint run moves both mirrors, so the
# larger value is the physical requirement.
JOINT_MEASUREMENT_KNOBS = ("settle_ms", "frames_per_measure")
# Search knobs belong to the 14-axis problem, so the joint run carries its
# own values; the two mirrors' numbers are shown for reference only.
JOINT_SEARCH_KNOBS = ("ga_population", "ga_generations", "cma_popsize",
                      "bo_init_points", "bo_budget", "bo_xi",
                      "sa_t0", "sa_cooling")


def default_actuators(n: int = DEFAULT_LAYOUT) -> list[Actuator]:
    """Return the default actuators for a mirror of `n` elements.

    A size in `LAYOUTS` gets the wiring that mirror is really on; any other
    count extends the five-element pattern two channels at a time, which is a
    guess and is only there so an odd count still produces something usable.
    """
    n = max(1, int(n))
    chans = list(LAYOUTS.get(n, ()))
    if not chans:
        chans = list(DEFAULT_CHANNELS)
        while len(chans) < n:
            chans.append(chans[-1] + 2)
    return [Actuator(channel=c) for c in chans[:n]]


@dataclass
class LoopSettings:
    """Everything a run needs.

    Algorithm-specific fields are ignored by the other algorithms, so one struct
    covers all three.
    """
    actuators: list[Actuator] = field(default_factory=default_actuators)
    # Per-channel overrides of the bit-scale knobs, {channel: {knob: value}};
    # empty for a single-mirror run.
    channel_knobs: dict = field(default_factory=dict)
    algorithm: str = ALGO_HILL
    metric: str = METRIC_PEAK
    min_step: int = 50  # Search quantisation and stopping resolution in bits.

    # measurement (per point)
    settle_ms: int = 80  # Wait after a mirror move before measuring.
    frames_per_measure: int = 4  # Frames medianed into one score.
    # avg_frame: average the frames, score once. avg_score: score each frame
    # and take the mean, at N x the cost.
    score_mode: str = SCORE_AVG_FRAME
    # SPEED_FIXED: every point pays `settle_ms` + `frames_per_measure`.
    # SPEED_AUTO: those become the finest budget (see correction.budget).
    speed_mode: str = SPEED_FIXED
    # Coarsest budget Auto may drop to, as one rung of the mirror's ladder;
    # 0 means the whole ladder.
    speed_floor_ms: int = 0
    speed_floor_frames: int = 0
    # Wait after an exposure change because in-flight frames still use the old
    # setting. This prevents mixed exposures, especially clipped frames after a
    # saturation back-off, from biasing the next averaged score.
    exposure_settle_ms: int = 500
    # Reduction backend for the frame average + measure()'s corner stats (see
    # core.fastmath). Speed only; both backends give the same numbers.
    reduction_backend: str = BACKEND_NUMPY
    # Subtract a measured background instead of re-estimating it from four
    # frame corners every frame (see core.background). Needs a reference taken
    # at the working exposure; without one the per-frame estimate still runs.
    use_background: bool = True
    background_frames: int = 30  # Frames averaged into that reference.
    # Points between mid-run re-measurements of the background; 0 takes it
    # once at Start. Every refresh is logged with its level change.
    background_refresh_points: int = 0

    # Real-time hold: once converged, keep watching and re-correct disturbances.
    # Score drop vs best that triggers re-optimise.
    disturb_drop_frac: float = 0.20
    # Raw-energy drop vs baseline read as occlusion.
    occlusion_drop_frac: float = 0.30

    # Hill-climb / shared search.
    move_step: int = 300  # Initial per-axis probe/move size (bits)
    # Hold if score gain smaller than this (0=auto)
    noise_threshold: float = 0.0

    # SPGD
    spgd_perturb: int = 100  # +/- dither on every actuator each step.
    # Bits moved per unit score difference. dscore is O(1e-3..1e-2) on the 0..1
    # score, so a gain of a few hundred moves <50 bits and stalls at the seed
    # (lazy); ~2000 puts the step in the range that actually searches.
    spgd_gain: float = 2000.0

    # Genetic
    ga_population: int = 16
    ga_generations: int = 12  # Fixed generation budget; the GA stops here.
    ga_mutation: int = 150  # Gaussian mutation sigma (bits)

    # CMA-ES
    cma_popsize: int = 0  # Samples per generation; 0 = auto 4+3*ln(N)
    cma_sigma: int = 300  # Initial search width (bits, ~1 sigma)

    # Bayesian optimisation
    bo_init_points: int = 10  # Space-filling probes before the model steers.
    # Total evaluations before parking on the best. 60 stopped a 5-actuator
    # search after 10 LHS probes + 50 model steps, which measured out as the
    # only algorithm that could not move a badly aberrated spot at all.
    bo_budget: int = 120
    bo_xi: float = 0.01  # Explore-vs-exploit weight of the EI rule.

    # Simulated annealing (Zommer et al., Opt. Lett. 31, 939 (2006))
    sa_t0: float = 0.10  # Start temperature, fraction of |seed score|.
    sa_cooling: float = 0.97  # Geometric cooling per measured point.
    sa_step: int = 300  # Initial random-walk sigma (bits); cools with T.

    # EE / Strehl curve optics: r0 = 1.22*lambda*f/D on the sensor (px).
    # Analysis-only (never steers the loop); any value <= 0 = unknown optics.
    wavelength_nm: float = 632.8
    focal_mm: float = 100.0
    aperture_mm: float = 5.0  # Limiting stop diameter at the focus lens.
    pixel_um: float = 3.45  # Camera pixel pitch.

    # Display-only contrast on the 0.5-anchored score; 1.0 = off. Never
    # touches the optimiser.
    score_contrast: float = 3.0

    # Metric shaping
    bucket_radius_px: float = 0.0  # Power-in-bucket target radius (0 = auto)
    roundness_weight: float = 0.0  # >0 multiplies score by (1-ellipticity)^w.
    # Exponent of the (1 - azim_m2) astigmatism gate for the two angle-blind
    # metrics; never applied inside a modal solve. 0 disables it.
    astig_weight: float = 2.0
    # Strength of the shape gate (metrics.shape_quality) that the two GEOMETRIC
    # metrics multiply in -- a pure size reading otherwise rewards a small-but-
    # broken spot. 0 disables it; the photometric metrics never use it.
    symmetry_weight: float = 3.0
    # Upsample the core when the spot is smaller.
    upsample_small_px: float = 12.0
    upsample_factor: int = 4  # Oversampling factor for tiny spots.
    # PSD frequency limits are normalized to the diffraction cutoff.
    # The default band balances simulated sensitivity and quadratic response.
    psd_m_lo: float = 0.05
    psd_m_hi: float = 0.30
    # A second-moment solve locks the starting spot's whole-pattern aperture
    # for the run, then expands its radius by this percentage. Other metrics
    # remain adaptive. A fixed domain is required for the exact quadratic form.
    second_moment_roi_margin_pct: float = 30.0
    # Modal solve. Reference probe amplitude in rad RMS; too small drowns in
    # noise, too large leaves the quadratic response.
    modal_bias_rad: float = 1.0
    # Modal probes use their own 12-bit grid, because a search's 50-bit grid
    # can quantise weak components away.
    modal_bit_step: int = 1
    # Measure the probe amplitude before solving: the sweep keeps the largest
    # still-quadratic fraction of `modal_bias_rad`.
    modal_auto_bias: bool = True
    # Whether the solve's moves go through the compensator. Separate from the
    # page-wide switch so one matrix can be replayed both ways for comparison.
    modal_compensate: bool = True
    # Fractions of `modal_bias_rad` visited by that sweep, each probed at both
    # signs. Nested, so a candidate can be judged on every point inside it.
    modal_bias_ladder: tuple = (0.25, 0.5, 0.75, 1.0)
    # Largest departure from a parabola, as a fraction of the response's own
    # swing, still counted as quadratic. Simulated far fields sit under 1%; the
    # allowance here is for camera noise, not for model error.
    modal_bias_tolerance: float = 0.08
    # Cap on how many eigenmodes to solve; 0 means every mode the stored
    # matrix rates above its noise floor.
    modal_modes: int = 0
    # False: modal_rounds is an exact operator request. True: it is a hard
    # safety cap and the solver may stop earlier only after two independently
    # verified quiet rounds (score below noise AND integer correction small).
    modal_auto_rounds: bool = False
    # Walk the solved direction at more than one length before spending a
    # new probe set.
    modal_line_search: bool = True
    # Repeats of the whole solve, or the safety cap in automatic mode.
    modal_rounds: int = 1
    # Search algorithm to run after the solve, or POLISH_NONE to stop there.
    # Any of ALGO_* is allowed: the modal stage only supplies a good starting
    # point, and which search finishes best is a property of the bench.
    polish_algorithm: str = POLISH_NONE
    # Metric for that polish stage. The solve's own metric is pinned by
    # MODAL_METRIC, but polish is free -- and should differ, since the whole
    # reason to polish is that the modal metric has gone flat near the limit.
    polish_metric: str = METRIC_PEAK

    # Focal-plane wavefront sensing (ALGO_WFS): the session file with the
    # pupil maps, not the loop's lightweight calibration.
    wfs_influence_npz: str = ""
    # Eigenmodes to correct on. 0 means `dm_basis.DEFAULT_KEEP`, which is
    # where this mirror's singular values fall off a cliff (<=1.64x steps down
    # to mode 6, then 8.57x). Modes past it buy 0.001 waves and amplify noise.
    wfs_keep_modes: int = 0
    # Rounds of sense-project-drive. The influence matrix is a local slope at
    # the bias while a full correction runs ~870 bit, so one shot lands short
    # by design and the loop closes over that error.
    wfs_rounds: int = 4
    # Cap on the largest single-actuator move of one round, in bits; 0 means
    # uncapped.
    wfs_max_step_bit: float = 0.0
    # Retrieval ROI, in pixels. 0 derives it from the spot (4x the measured
    # 80% encircled-energy radius); a fixed value trades convergence for
    # time.
    wfs_roi_px: int = 0

    def to_dict(self) -> dict:
        """Serialize the settings to a dictionary.

        Full serialisation, for anything that must reconstruct these settings
        (see `from_dict`). For a human-readable RECORD of a run, use
        `active_dict` -- this one lists every algorithm's knobs at once.
        """
        d = asdict(self)
        d["actuators"] = [asdict(a) for a in self.actuators]
        return d

    def _channel_knob_record(self, names) -> dict:
        """Per-channel overrides of `names`, keyed by knob, or {} if none."""
        record = {}
        for knob in names:
            per_channel = {str(channel): values[knob]
                           for channel, values in self.channel_knobs.items()
                           if knob in values}
            if per_channel:
                record[f"{knob}_by_channel"] = per_channel
        return record

    def knob_for(self, channel: int, name: str):
        """Return one channel's value of a knob, or the scalar setting.

        Args:
            channel: Pi PWM channel.
            name: Field name of the knob.

        Returns:
            The per-channel override where one exists, else `getattr(self, name)`.
        """
        return self.channel_knobs.get(int(channel), {}).get(
            name, getattr(self, name))

    def active_dict(self) -> dict:
        """Return the settings used by the current run."""
        def grab(names):
            return {n: getattr(self, n) for n in names}

        algo_knobs = ALGO_PARAMS.get(self.algorithm, ())
        metric_knobs = METRIC_PARAMS.get(self.metric, ())
        # A bare modal solve never reads the generic search grid. A staged run
        # does read it after hand-over to its polish search.
        shared_algo = (_ALGO_SHARED if self.algorithm not in MODAL_ALGOS
                       or self.polish_algorithm != POLISH_NONE else ())
        out = {
            "actuators": [asdict(a) for a in self.actuators],
            "algorithm": {"name": self.algorithm,
                          "label": ALGO_LABELS.get(self.algorithm, "?"),
                          **grab(shared_algo), **grab(algo_knobs),
                          # Only the overridden knobs this algorithm reads.
                          **self._channel_knob_record(
                              tuple(shared_algo) + tuple(algo_knobs))},
            "metric": {"name": self.metric,
                       "label": METRIC_LABELS.get(self.metric, "?"),
                       **grab(metric_knobs), **grab(_METRIC_SHARED)},
            "measurement": grab(_MEASUREMENT),
            "hold": grab(_HOLD),
            "optics": grab(_OPTICS),
            "display_only": grab(_DISPLAY_ONLY),
        }
        known = set(_ALGO_SHARED + _METRIC_SHARED + _MEASUREMENT + _HOLD
                    + _OPTICS + _DISPLAY_ONLY + ("actuators", "algorithm",
                                                 "metric", "channel_knobs"))
        for knobs in ALGO_PARAMS.values():
            known |= set(knobs)
        for knobs in METRIC_PARAMS.values():
            known |= set(knobs)
        rest = {f.name: getattr(self, f.name) for f in fields(self)
                if f.name not in known}
        if rest:
            out["unclassified"] = rest
        # Deliberately does not enumerate the omitted knobs.
        out["_note"] = ("Only the parameters this run read. Knobs belonging to "
                        "other algorithms and metrics are omitted, not "
                        "defaulted -- they had no effect on this run.")
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "LoopSettings":
        d = dict(d)
        # JSON has no integer keys, so a saved run_config comes back with the
        # channels as strings; the optimizers index this table by channel.
        if d.get("channel_knobs"):
            d["channel_knobs"] = {
                int(channel): dict(values)
                for channel, values in d["channel_knobs"].items()}
        # Unknown keys are dropped rather than raising: a run_config.json
        # written before `name` became derived still carries it, and an old
        # run must stay replayable.
        act_keys = {f.name for f in fields(Actuator)}
        d["actuators"] = [
            Actuator(**{k: v for k, v in a.items() if k in act_keys})
            for a in d.get("actuators", [])]
        # A retired metric ("peak_sym") must not slide silently into the default
        # branch of primary_score and be replayed as a different objective.
        if "metric" in d and d["metric"] not in METRIC_LABELS:
            d["metric"] = METRIC_PIB
        # An old plan.json, or one naming a backend this machine lacks.
        if "reduction_backend" in d:
            from .fastmath import resolve as _resolve_backend
            d["reduction_backend"] = _resolve_backend(d["reduction_backend"])
        return cls(**d)
