# hysteresis

Version 3.2

Open-loop compensation of piezo hysteresis. A channel's measured loops are
fitted once to a modified Prandtl-Ishlinskii model; at run time the model is
inverted so that a requested nominal code produces the displacement a linear
actuator would have given.

## Files

`pi_model.py` is the model. `PlayOperator` holds one play element with its
threshold and state; `ModifiedPrandtlIshlinskii` combines a Chebyshev primary
response with weighted play operators, advances the state on every applied
voltage, and inverts the model for a target displacement (`InverseResult`).

`device_profile.py` loads a channel's JSON profile from `devices/` and
validates its fields: hardware limits, the driver voltage model, the fitted
coefficients and the linear displacement target.

`driver_voltage_lut.py` converts a 12-bit code to the driver's output
voltage and back from a fitted lookup, with the driver's rails as limits.

`driver_voltage_curve.py` selects the code-to-voltage conversion named in
the profile: the shared lookup (`LegacyDriverVoltageCurve`) or a per-device
Chebyshev fit (`ChebyshevDriverVoltageCurve`).

`compensator.py` joins the two. `HysteresisCompensator` maps nominal code to
target displacement, solves the model for the voltage that reaches it from
the current state, converts to a code and reports clamping in
`CommandResult`. `commit` advances the model state after the write is
acknowledged.

`open_loop.py` is the per-channel object the loop uses. `OpenLoopChannel`
homes the channel, plans a compensated code for each nominal code, plans an
uncompensated code that still advances the model, and predicts the
displacement change that the adaptive wait rule needs.

`fit_profile.py` fits a profile from measured loop CSVs: `read_loop_csv`
reads the drive and displacement columns, `fit_profile` fits the primary
response and non-negative play weights over a grid of thresholds and keeps
the grid with the smallest residual, `save_profile` writes the JSON.

`step_wave.py` builds step, staircase and sine drive sequences for
time-domain tests, compensated or raw (`build_step_plan`).

`comp_sweep.py` builds a compensated up-and-down sweep from a profile
(`build_compensated_sweep`) and lists the profiles available.

## devices/

`DM5/` holds the profiles for channels 1 to 5, `DM9/` for channels 6 to 14.
`dm_d.json` and `piezo_a.json` are single-piezo profiles used by the tests
and as the compensator's default.
