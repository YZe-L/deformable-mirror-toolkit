# interferometry

Version 3.2

Michelson interferometer analysis. One chain turns a stream of carrier
interferograms into a displacement trace of a chosen point on the mirror;
the other turns single frames or phase-shifted stacks into a full surface
map.

## Displacement from carrier fringes

`fringes.py` implements Takeda carrier demodulation. `detect_aperture` finds
the illuminated disk, `find_lobe` locates the carrier sideband in the
spectrum, `build_demod` fixes the band-pass around it, and `demod` returns
the complex fringe field of a frame. `unwrap2d` is the spatial unwrap used
when a step lands near plus or minus pi.

`tracking.py` integrates phase between frames. `FringeTracker` keeps the
sideband and aperture fixed from the first frame and reports displacement in
nanometres, either as a running sum of wrapped steps or as absolute phase
against the first frame. `temporal_unwrap_steps`, `correct_aliased_step` and
`unwrap_sweep_steps` recover fringe order when a step exceeds a quarter
wave; `detect_fringes` is the low-rate fringe detector used to gate
recording. `SurfaceCentreTracker` does the same job on closed fringes
through the carrier-free reconstruction in `surface.py`.

`response.py` analyses a displacement trace after a command step:
`analyze_response_trace` finds onset, settling and the 10 to 90 percent rise
time from pre-step and post-step levels; `monotonic_unwrap` recovers a
one-way step that moved faster than the frame rate; `aliasing_frames` flags
the frames that cannot be recovered. `StepWatchLogic` is the live version
that detects a step and declares it settled.

`noise.py` computes the displacement noise spectrum of a quiet trace.

`loop.py` builds hysteresis loops from a bit sequence and a displacement
trace: `bit_sequence` generates the up-then-down drive, `split_branches`
separates rising and falling branches, `hysteresis_metrics` reports width
and linearity, and `draw_loop` plots them. `read_csv` and `drive_from_csv`
map spreadsheet columns to drive and displacement.

`video.py` applies the trackers offline to MP4 recordings
(`VideoResponseWorker`, one file per command level) and provides the live
workers for the step watch, the noise recording and the settling-time
trigger.

`recording.py` is the long-interval recording worker. It consumes frames
from a queue, tracks displacement and streams one CSV row per frame with an
immediate flush, so a recording that runs for an hour survives a crash.

## Surface maps

`takeda.py` reconstructs a 2-D wavefront map from one carrier interferogram
(`takeda_wavefront`), removes piston and tilt, reports fringe quality
(`quality_summary`) and draws the six-panel figure.

`psi.py` is temporal phase-shifting interferometry. `reconstruct` takes a
stack of frames with known or unknown phase steps; `aia` estimates the steps
from the data, `lsq` solves the phase, and the modulation masks reject
pixels without fringes. `build_measurement` wraps the result as a Zygo-style
`Measurement` so the same analysis code handles both instruments.

`psi_bundle.py` saves a PSI run (frames, steps, settings) as one .npz and
restores it.

`substrate.py` stores a reference surface and subtracts it from later
measurements, the same operation as the Zygo "subtract system error".

`surface.py` reconstructs a surface from closed or bent fringes with the
spiral-phase transform. It is distributed obfuscated; see the root README.
