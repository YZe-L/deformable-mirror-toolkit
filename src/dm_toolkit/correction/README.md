# correction

Version 3.2

The sensorless correction core: how a camera frame becomes a score, and how
a score becomes the next mirror command. Nothing here talks to hardware; the
caller captures frames, sends commands and feeds results back through
`ask()` and `tell()`.

## Files

`settings.py` defines `Actuator` (channel, start code, limits) and
`LoopSettings`, the complete run configuration: algorithm, metric, settle
time, frames per measurement, optics, and every algorithm's own parameters.
It also lists which knobs each algorithm actually reads, so a run record
does not carry inert values. `default_actuators` builds the five-channel or
nine-channel layout.

`metrics.py` scores a frame. `measure` locates the pattern, subtracts the
background and returns a `SpotReading` with every metric at once: peak
against the ideal Airy peak, power in the diffraction bucket, PSD band
power, second moment, encircled-energy radius and shape diagnostics.
`primary_score` selects the configured metric, `norm_score` maps it onto the
seed-relative 0 to 1 scale, and `classify_aberration` names the dominant
symptom for the log.

`optimizers.py` holds the search methods behind one interface (`ask`,
`tell`, `best_command`, `converged`, `rescan`): `HillClimb`, `SimAnneal`,
`Genetic`, `BayesOpt`, `SPGD` and `CMAES`. `make_optimizer` builds the one
named in the settings; `Staged` chains a modal solve with a polishing
search. Score gates use the measured per-point noise.

`modal.py` holds the model-based solves. `ModalFit` measures the centre and
a plus and minus probe of every mode and inverts the quadratic (2N+1
measurements); `ModalFast` reuses the stored curvature (N+2); `ModalPsd`
fits the reciprocal band power. All three calibrate the probe amplitude,
walk the solved direction at several lengths and stop after two quiet
rounds.

`wavefront.py` is a controller that skips the score altogether:
`PseudoWfs` retrieves the wavefront from the averaged frame, projects it onto
the mirror modes and commands the bounded least-squares correction.

`dual_mirror.py` drives two mirrors with one optimiser interface, either in
sequence (`SequentialMirrors`) or as one joint search over all channels
(`merge_joint`).

`budget.py` models the cost of a point (fixed time, settle, frames) and
turns a required score resolution into the cheapest settle and frame count
from the measured step-response and noise curves (`SpeedProfile`).

`timeline.py` stamps every hand-off of a point (command sent, acknowledged,
wait done, frames captured, scored, decided) so `timing.csv` reconstructs
the round.

`ee_curve.py` computes encircled-energy curves for the before and after
spots against the ideal Airy pattern and the Strehl-like ratio at the
reference radius.

`background.py` measures a run-level background reference outside the spot
and scales it when the exposure changes.

`autoexposure.py` keeps the peak at a fixed fraction of full scale and
decides when a ducked exposure may return to the base.

`fastmath.py` selects the reduction backend for frame averaging (numpy or
bottleneck); both give the same numbers.

`dm_loop_profiles.json` maps each channel to its hysteresis profile,
adaptive-settling descriptor and control matrix, and stores the measured
speed profiles.
