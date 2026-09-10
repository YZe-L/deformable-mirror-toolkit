# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-29

"""Fit a device hysteresis profile from measured drive loops.

Reads measured loop CSVs and produces the ``devices/<id>.json`` profile that
:class:`~.compensator.HysteresisCompensator` consumes.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize

from .device_profile import load_device_profile
from .driver_voltage_curve import LegacyDriverVoltageCurve
from .pi_model import _chebyshev

BIT_COLUMNS = ("bit", "command_bit", "drive_bit")
DISPLACEMENT_COLUMNS = ("displacement_nm", "centre_disp_nm", "disp_nm",
                        "displacement")
LOOP_COLUMNS = ("loop", "range")


def read_loop_csv(path):
    """Read one measured loop file in drive order.

    Args:
        path (str | Path): Measured loop CSV with bit and displacement
            columns.

    Returns:
        dict: ``bits`` (list[int]), ``displacement_nm`` (list[float]),
            ``loops`` (list[int], 1 when the file has no loop/range column),
            ``measurement_source`` and ``source`` (str path), with
            unparsable rows dropped. The fitted response always comes from
            the displacement column, not the raw Mx input column.

    Raises:
        ValueError: If the file has no bit column, no displacement column, or
            fewer than four usable rows.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path.name}: no data rows")
    names = {name.strip().lower(): name for name in rows[0] if name}
    bit_key = next((names[c] for c in BIT_COLUMNS if c in names), None)
    disp_key = next((names[c] for c in DISPLACEMENT_COLUMNS if c in names), None)
    if bit_key is None:
        raise ValueError(f"{path.name}: no bit column "
                         f"(looked for {', '.join(BIT_COLUMNS)})")
    if disp_key is None:
        raise ValueError(f"{path.name}: no displacement column "
                         f"(looked for {', '.join(DISPLACEMENT_COLUMNS)})")
    loop_key = next((names[c] for c in LOOP_COLUMNS if c in names), None)

    bits, disp, loops = [], [], []
    for row in rows:
        try:
            bit = int(round(float(row[bit_key])))
            value = float(row[disp_key])
        except (TypeError, ValueError):
            continue  # Blank displacement cell, header junk.
        if not np.isfinite(value):
            continue
        try:
            loop = int(float(row[loop_key])) if loop_key else 1
        except (TypeError, ValueError):
            loop = 1
        bits.append(bit)
        disp.append(value)
        loops.append(loop)
    if len(bits) < 4:
        raise ValueError(f"{path.name}: only {len(bits)} usable row(s); the "
                         "displacement column looks empty")
    return dict(bits=bits, displacement_nm=disp, loops=loops,
                measurement_source=disp_key, source=str(path))


def _play_states(voltages, threshold):
    """Play-operator state after each input, started on the loading branch.

    Args:
        voltages (np.ndarray): Effective voltage of every point, in order.
        threshold (float): Play threshold in volts.

    Returns:
        np.ndarray: The operator state at each point.
    """
    states = np.empty(voltages.size)
    state = max(float(voltages[0]) - threshold, 0.0)
    for i, value in enumerate(voltages):
        state = max(value - threshold, min(value + threshold, state))
        states[i] = state
    return states


def _design(voltages, v_max, degree, thresholds):
    """Least-squares design matrix for one measurement run.

    Args:
        voltages (np.ndarray): Effective voltage of every point, in order.
        v_max (float): Upper end of the Chebyshev domain, in volts.
        degree (int): Chebyshev degree of the primary response.
        thresholds (np.ndarray): Play thresholds in volts.

    Returns:
        np.ndarray: Columns [primary basis | play features], primary basis
            shifted so the model reads zero at zero volts.
    """
    x = 2.0 * voltages / v_max - 1.0
    x0 = -1.0
    primary = np.stack([np.polynomial.chebyshev.chebval(
        x, np.eye(degree + 1)[k]) - np.polynomial.chebyshev.chebval(
        x0, np.eye(degree + 1)[k]) for k in range(degree + 1)], axis=1)
    play = np.stack([_play_states(voltages, t)
                     - np.maximum(voltages - t, 0.0) for t in thresholds],
                    axis=1)
    return np.concatenate([primary, play], axis=1)


def _solve_monotonic_loading(matrix, target, v_max, degree, thresholds,
                             endpoint_displacement_nm):
    """Solve an endpoint-constrained fit with a strictly increasing loading curve.

    The ordinary least-squares solution can follow small measurement noise in a
    low-slope region and create a local negative slope.  Such a curve has no
    safe inverse.  This constrained least-squares form keeps the PI play
    weights non-negative and imposes positive primary-response increments on a
    dense voltage grid.
    """
    endpoint = float(endpoint_displacement_nm)
    n_primary = degree + 1
    n_weights = thresholds.size
    n_parameters = n_primary + n_weights

    # The primary response alone describes a complete initial loading branch:
    # the play corrections are zero on that branch.  Constrain this response
    # on a denser grid than the subsequent monotonicity validation.
    loading_voltages = np.linspace(0.0, v_max, 2048)
    primary_grid = _design(
        loading_voltages, v_max, degree, thresholds
    )[:, :n_primary]
    loading_increment = np.diff(primary_grid, axis=0)
    monotonic_matrix = np.pad(
        loading_increment, ((0, 0), (0, n_weights))
    )
    endpoint_row = _design(
        np.asarray([v_max]), v_max, degree, thresholds
    )[0, :n_primary]
    equality_matrix = np.zeros((2, n_parameters))
    equality_matrix[0, :n_primary] = endpoint_row
    # The Chebyshev constant is cancelled by the zero-reference transform, so
    # fixing it removes an otherwise unconstrained numerical degree of freedom.
    equality_matrix[1, 0] = 1.0

    lower = np.concatenate([
        np.full(n_primary, -np.inf), np.zeros(n_weights)
    ])
    upper = np.full(n_parameters, np.inf)
    initial = np.zeros(n_parameters)
    # T_1(x) - T_1(-1) rises by two over the domain, yielding a feasible
    # strictly increasing straight line with the requested endpoint.
    initial[1] = endpoint / 2.0
    output_scale = max(abs(endpoint), 1.0)
    point_count = target.size

    def objective(solution):
        residual = (matrix @ solution - target) / output_scale
        return 0.5 * float(residual @ residual) / point_count

    def gradient(solution):
        return matrix.T @ (matrix @ solution - target) / (
            output_scale ** 2 * point_count
        )

    minimum_increment_nm = max(abs(endpoint) * 1e-8, 1e-8)
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=Bounds(lower, upper),
        constraints=(
            LinearConstraint(
                equality_matrix, [endpoint, 0.0], [endpoint, 0.0]
            ),
            LinearConstraint(
                monotonic_matrix, minimum_increment_nm, np.inf
            ),
        ),
        options={"ftol": 1e-12, "maxiter": 3000},
    )
    if not result.success:
        raise RuntimeError(
            "monotonic constrained fit did not converge: " + result.message
        )
    solution = result.x
    residual = matrix @ solution - target
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return solution[:n_primary], solution[n_primary:], rmse


def _detect_broad_dead_zone(runs, v_max):
    """Measure whether an initial loading branch has a broad low-sensitivity zone.

    A normal piezo loading curve may be nonlinear, but it should already have
    traversed a material fraction of its span by one quarter of the drive
    range.  If the measured response at that point is at most 10 percent of
    the complete loading span, a free polynomial fit tends to chase low-level
    noise and lose a unique inverse.  Return diagnostics as well as the
    decision so the generated profile documents which fitter was used.
    """
    diagnostics = {
        "has_broad_dead_zone": False,
        "quarter_drive_response_fraction": None,
        "initial_loading_span_nm": None,
    }
    for voltages, displacement in runs:
        if voltages.size < 4 or voltages[0] > 0.02 * v_max:
            continue
        peak_indices = np.flatnonzero(voltages >= 0.995 * v_max)
        if peak_indices.size == 0:
            continue
        peak_index = int(peak_indices[0])
        branch_v = voltages[:peak_index + 1]
        branch_d = displacement[:peak_index + 1]
        if np.any(np.diff(branch_v) < 0):
            continue
        span = float(branch_d[-1] - branch_d[0])
        if span <= 0.0:
            continue
        response_at_quarter = float(np.interp(0.25 * v_max, branch_v,
                                              branch_d) - branch_d[0])
        fraction = response_at_quarter / span
        diagnostics.update({
            "quarter_drive_response_fraction": fraction,
            "initial_loading_span_nm": span,
            "has_broad_dead_zone": fraction <= 0.10,
        })
        return diagnostics
    return diagnostics


def _solve(runs, v_max, degree, thresholds, endpoint_displacement_nm=None,
           enforce_monotonic_loading=False):
    """Fit primary coefficients and non-negative play weights.

    Args:
        runs (list): (voltages, displacement) array pairs, one per input file.
        v_max (float): Upper end of the Chebyshev domain, in volts.
        degree (int): Chebyshev degree of the primary response.
        thresholds (np.ndarray): Play thresholds in volts.
        endpoint_displacement_nm (float | None): When given, constrain the
            primary loading response at ``v_max`` to this displacement. This
            is the PI-consistent saturated endpoint because every play
            correction is zero after a complete loading excursion.
        enforce_monotonic_loading (bool): Fit the loading curve under strict
            positive-increment constraints, so noisy dead-zone data remains
            safely invertible.

    Returns:
        tuple: Chebyshev coefficients, play weights, and the fit residual RMSE
            in nanometres.
    """
    matrix = np.concatenate([_design(v, v_max, degree, thresholds)
                             for v, _ in runs])
    target = np.concatenate([d for _, d in runs])
    n_primary = degree + 1
    if enforce_monotonic_loading:
        if endpoint_displacement_nm is None:
            raise ValueError(
                "monotonic constrained fitting requires an endpoint "
                "displacement"
            )
        return _solve_monotonic_loading(
            matrix, target, v_max, degree, thresholds,
            endpoint_displacement_nm,
        )
    if endpoint_displacement_nm is None:
        lower = np.concatenate([np.full(n_primary, -np.inf),
                                np.zeros(thresholds.size)])
        upper = np.full(matrix.shape[1], np.inf)
        fit = lsq_linear(matrix, target, bounds=(lower, upper))
        solution = fit.x
    else:
        endpoint = float(endpoint_displacement_nm)
        endpoint_row = _design(
            np.asarray([v_max]), v_max, degree, thresholds
        )[0, :n_primary]
        pivot = int(np.argmax(np.abs(endpoint_row)))
        pivot_scale = float(endpoint_row[pivot])
        if abs(pivot_scale) < 1e-12:
            raise ValueError(
                "Chebyshev basis cannot express the requested endpoint"
            )

        # Eliminate one unbounded primary coefficient from
        # endpoint_row @ coefficients = endpoint. The play-weight columns are
        # left untouched, so their required non-negative bounds are preserved.
        keep = np.asarray([index for index in range(n_primary)
                           if index != pivot], dtype=int)
        pivot_column = matrix[:, [pivot]]
        reduced_primary = (
            matrix[:, keep]
            - pivot_column * (endpoint_row[keep] / pivot_scale)
        )
        reduced_matrix = np.concatenate(
            [reduced_primary, matrix[:, n_primary:]], axis=1
        )
        reduced_target = target - matrix[:, pivot] * endpoint / pivot_scale
        lower = np.concatenate([np.full(keep.size, -np.inf),
                                np.zeros(thresholds.size)])
        upper = np.full(reduced_matrix.shape[1], np.inf)
        fit = lsq_linear(reduced_matrix, reduced_target,
                         bounds=(lower, upper))

        coefficients = np.zeros(n_primary)
        coefficients[keep] = fit.x[:keep.size]
        coefficients[pivot] = (
            endpoint - endpoint_row[keep] @ coefficients[keep]
        ) / pivot_scale
        solution = np.concatenate([coefficients, fit.x[keep.size:]])

    residual = matrix @ solution - target
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return solution[:n_primary], solution[n_primary:], rmse


def fit_profile(datasets, *, device_id, display_name="", description="",
                channel=1, degree=7, n_play=5, minimum_bit=0,
                maximum_bit=4095, settle_seconds=7.0,
                endpoint_displacement_nm=None,
                enforce_monotonic_loading="auto"):
    """Fit a modified Prandtl-Ishlinskii profile from measured loops.

    The driver bit/voltage conversion is the shared ServoPi fit, so the model
    is fitted in exactly the coordinates the compensator later drives it in.

    Args:
        datasets (list[dict]): Runs from :func:`read_loop_csv`; model memory
            restarts at the first point of each run.
        device_id (str): Profile id, also the JSON file stem.
        display_name (str): Human-readable name shown in the app.
        description (str): Free-text note stored in the profile.
        channel (int): PWM channel the data was measured on.
        degree (int): Chebyshev degree of the primary loading response.
        n_play (int): Number of play operators.
        minimum_bit (int): Lowest hardware bit.
        maximum_bit (int): Highest hardware bit.
        settle_seconds (float): Settle time used during the measurement.
        endpoint_displacement_nm (float | None): Exact relative displacement
            required at ``maximum_bit``. Use a stable measured saturated
            endpoint to stop the global least-squares fit pulling the maximum
            command away from its measured displacement.
        enforce_monotonic_loading (bool | "auto"): Apply a dense-grid
            strictly increasing constraint to the primary loading curve.
            ``True`` always applies it and ``False`` never applies it.
            ``"auto"`` (the default) applies it only when the initial loading
            branch has at most 10 percent of its span at one quarter drive.

    Returns:
        tuple: The profile dict ready to save, and a metrics dict with
            ``rmse_nm``, ``max_abs_error_nm``, ``bias_nm``, ``points``,
            ``span_nm``, ``per_loop_rmse_nm`` and ``warnings``.

    Raises:
        ValueError: If the data is empty or the fitted loading curve is not
            monotonic, which the compensator's inverse requires.
    """
    if not datasets:
        raise ValueError("no measurement data given")
    if (endpoint_displacement_nm is not None
            and float(endpoint_displacement_nm) <= 0.0):
        raise ValueError("endpoint displacement must be positive")
    if enforce_monotonic_loading not in (True, False, "auto"):
        raise ValueError(
            "enforce_monotonic_loading must be True, False, or 'auto'"
        )
    curve = LegacyDriverVoltageCurve(minimum_bit, maximum_bit)
    origin_v = curve.bit_to_voltage(minimum_bit)
    v_max = curve.bit_to_voltage(maximum_bit) - origin_v

    runs, loop_tags, warnings = [], [], []
    for data in datasets:
        bits = np.clip(np.asarray(data["bits"], int), minimum_bit, maximum_bit)
        volts = np.array([curve.bit_to_voltage(int(b)) - origin_v
                          for b in bits])
        # The model reads zero at the home bit, so each run is referenced to
        # its own first row -- that row is the home.
        disp = np.asarray(data["displacement_nm"], float)
        runs.append((np.clip(volts, 0.0, v_max), disp - disp[0]))
        name = Path(data["source"]).name
        if int(bits[0]) > minimum_bit:
            warnings.append(f"{name} starts at bit {int(bits[0])}, not "
                            f"{minimum_bit}; the fit assumes the piezo was "
                            "relaxed at the first row")
        loop_tags.extend((Path(data["source"]).stem, lp)
                         for lp in data["loops"])

    dead_zone = _detect_broad_dead_zone(runs, v_max)
    if enforce_monotonic_loading == "auto":
        monotonic_constraint = bool(dead_zone["has_broad_dead_zone"])
        strategy = (
            "monotonic_constrained_dead_zone"
            if monotonic_constraint else "ordinary_least_squares"
        )
    else:
        monotonic_constraint = bool(enforce_monotonic_loading)
        strategy = (
            "monotonic_constrained_forced"
            if monotonic_constraint else "ordinary_least_squares_forced"
        )

    best = None
    # Thresholds are fixed per attempt, so the weights stay a linear fit; a
    # coarse spread search is enough to place them.
    for low_frac in (0.005, 0.01, 0.02):
        for high_frac in (0.2, 0.3, 0.45):
            thresholds = np.geomspace(v_max * low_frac, v_max * high_frac,
                                      max(1, n_play))
            try:
                coefficients, weights, rmse = _solve(
                    runs, v_max, degree, thresholds,
                    endpoint_displacement_nm,
                    monotonic_constraint,
                )
            except RuntimeError:
                continue
            if best is None or rmse < best[0]:
                best = (rmse, thresholds, coefficients, weights)
    if best is None:
        raise ValueError(
            "no monotonic constrained fit converged; reduce the Chebyshev "
            "degree or inspect the calibration data"
        )
    rmse, thresholds, coefficients, weights = best

    raw_zero_nm = float(_chebyshev(tuple(coefficients), -1.0))
    grid = np.linspace(0.0, v_max, 512)
    x = 2.0 * grid / v_max - 1.0
    loading = np.array([_chebyshev(tuple(coefficients), value)
                        for value in x]) - raw_zero_nm
    if np.any(np.diff(loading) <= 0):
        raise ValueError("fitted loading curve is not monotonic, which the "
                         "compensator's inverse needs; try a lower Chebyshev "
                         "degree or add more data. "
                         + " ".join(warnings))
    span_nm = float(loading[-1])
    if (endpoint_displacement_nm is not None
            and not np.isclose(span_nm, float(endpoint_displacement_nm),
                               rtol=0.0, atol=1e-7)):
        raise ValueError("fitted loading curve missed its endpoint constraint")

    predicted = np.concatenate(
        [_design(v, v_max, degree, thresholds)
         @ np.concatenate([coefficients, weights]) for v, _ in runs])
    measured = np.concatenate([d for _, d in runs])
    error = predicted - measured
    per_loop = {}
    for tag in dict.fromkeys(loop_tags):
        mask = np.array([t == tag for t in loop_tags])
        per_loop[f"{tag[0]}_loop_{tag[1]}"] = float(
            np.sqrt(np.mean(error[mask] ** 2)))

    profile = {
        "schema_version": 1,
        "device_id": device_id,
        "display_name": display_name or device_id,
        "status": "fitted_in_app",
        "description": description or (
            f"Fitted from {len(datasets)} measured loop "
            f"file(s) on channel {channel}."),
        "hardware": {
            "driver": "ServoPi PCA9685",
            "i2c_address": 64,
            "pwm_frequency_hz": 1526,
            "default_channel": int(channel),
            "minimum_bit": int(minimum_bit),
            "maximum_bit": int(maximum_bit),
            "default_settle_seconds": float(settle_seconds),
        },
        "model": {
            "type": "modified_prandtl_ishlinskii",
            "input_quantity": "effective_voltage_above_driver_minimum",
            "input_unit": "V",
            "output_quantity": "relative_displacement",
            "output_unit": "nm",
            "voltage_origin_v": float(origin_v),
            "effective_input_min_v": 0.0,
            "effective_input_max_v": float(v_max),
            "primary_response": {
                "type": "chebyshev",
                "domain_v": [0.0, float(v_max)],
                "chebyshev_coefficients": [float(c) for c in coefficients],
                "raw_zero_nm": raw_zero_nm,
                "output_scale": 1.0,
            },
            "play_operators": [
                {"threshold_v": float(t), "weight_nm_per_v": float(w)}
                for t, w in zip(thresholds, weights)
            ],
        },
        "linearized_command": {
            "type": "linear_nominal_bit_to_displacement",
            "minimum_nominal_bit": int(minimum_bit),
            "maximum_nominal_bit": int(maximum_bit),
            "minimum_target_displacement_nm": 0.0,
            "maximum_target_displacement_nm": span_nm,
            "definition": "A nominal bit maps linearly onto displacement; the "
                          "compensator inverts the model to reach it.",
        },
        "limits": {
            "minimum_command_voltage_v": float(origin_v),
            "maximum_command_voltage_v": float(
                curve.bit_to_voltage(maximum_bit)),
            "minimum_relative_displacement_nm": 0.0,
            "maximum_calibrated_displacement_nm": span_nm,
            "reference_definition": "Displacement is zeroed when the home "
                                    "command applies the minimum bit.",
        },
        "calibration": {
            "date": date.today().isoformat(),
            "method": (
                f"Degree-{degree} Chebyshev loading curve plus {n_play} "
                "non-negative play operators, fitted by least squares "
                "over the measured drive order"
                + (", with strictly increasing loading-response constraints"
                   if monotonic_constraint else "")
                + (f", with an exact {float(endpoint_displacement_nm):.9f} nm "
                   "saturated-endpoint constraint."
                   if endpoint_displacement_nm is not None else ".")
            ),
            "driver_voltage_model": "shared ServoPi fit (driver_voltage_lut)",
            "data_files": [d["source"] for d in datasets],
            "displacement_source": sorted({
                d.get("measurement_source", "displacement_nm")
                for d in datasets
            }),
            "displacement_reference": (
                "Each measurement run is zeroed at its first usable row."
            ),
            "endpoint_constraint_nm": (
                float(endpoint_displacement_nm)
                if endpoint_displacement_nm is not None else None
            ),
            "training_points": int(measured.size),
            "training_mae_nm": float(np.mean(np.abs(error))),
            "training_rmse_nm": rmse,
            "training_max_abs_error_nm": float(np.max(np.abs(error))),
            "training_bias_nm": float(np.mean(error)),
            "training_loop_rmse_nm": per_loop,
            "observed_displacement_range_nm": [float(np.min(measured)),
                                               float(np.max(measured))],
            "loading_response_strategy": strategy,
            "dead_zone_detection": dead_zone,
        },
    }
    metrics = {
        "rmse_nm": rmse,
        "mae_nm": float(np.mean(np.abs(error))),
        "max_abs_error_nm": float(np.max(np.abs(error))),
        "bias_nm": float(np.mean(error)),
        "points": int(measured.size),
        "span_nm": span_nm,
        "per_loop_rmse_nm": per_loop,
        "warnings": warnings,
    }
    return profile, metrics


def save_profile(profile, path):
    """Write a fitted profile and prove the compensator can load it.

    Args:
        profile (dict): Result of :func:`fit_profile`.
        path (str | Path): Destination JSON file.

    Returns:
        Path: The written file.

    Raises:
        ValueError: If the written profile fails validation; the file is
            removed so a broken profile is never left behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(profile, stream, indent=2)
        stream.write("\n")
    try:
        load_device_profile(path)
    except Exception as error:
        path.unlink(missing_ok=True)
        raise ValueError(f"fitted profile failed validation: {error}") from error
    return path
