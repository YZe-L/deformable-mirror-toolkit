# bench

Version 3.2

Plans and figures for the measurement campaigns that characterise the
optimisers and the timing settings. The plans describe what to run and in
what order; the plot modules read the CSVs a session wrote and draw the
figures. The engines that executed the plans were part of the front end and
are not included.

## Files

`plan.py` describes a parameter scan: `Knob` gives one parameter its range
and level count, `default_knobs` the per-algorithm grids, and `BenchPlan`
the phases, repeats, control runs and rest periods. `knob_grid` and
`phase_a_runs` expand the grid into a shuffled run list, `build_settings`
turns one run into `LoopSettings`, and `estimate` predicts the session time
from the measured per-point cost.

`plots.py` draws the scan figures from `runs.csv`: score against knob
values, the settle and frames panels, noise, running best against
measurement budget (`fig_budget`), drift of the control runs, and the
overall Pareto view.

`repeat_plan.py` describes a repeatability session: the same run repeated
many times per algorithm, interleaved or blocked, with hysteresis
compensation either left alone or made the thing under test.
`fingerprint` records what the session must hold constant so a resumed
session can be checked against it.

`repeat_plots.py` draws the distributions of a repeatability session: final
score, encircled-energy ratio, time to converge and the per-mirror split,
one box per algorithm with every run as a point.

`sweep_plan.py` describes the settle-time and frame-count sweep: one step
transient per round, from which every candidate wait and window length is
cut.

`sweep_plots.py` draws the sweep results and the three decision figures
that fix the settle time, the frame count and the averaging order for each
mirror, using the overlapping Allan deviation of the score against frames.
`speed_profile` returns the same curves as numbers for `correction.budget`.

`tradeoff_plots.py` reads a finished scan and plots speed against quality
per parameter combination, with a short list of recommended settings.
