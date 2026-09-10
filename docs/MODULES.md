# Modules

Version 3.2

One line per file. The version column is the file's own revision number,
carried over from its header. Packages without a number are new to this
release. Each folder also has its own README with a fuller description of
every file in it.

## Top level

| File | Version | Purpose |
| --- | --- | --- |
| `dm_toolkit/__init__.py` | 3.2 | Package marker and version |
| `dm_toolkit/config.py` | 1.2 | Checkout, output and vendor paths; portable settings-file paths |
| `dm_toolkit/zernike.py` | 1.0 | Noll-indexed Zernike fit of a wavefront map and named aberrations |
| `dm_toolkit/backend.py` | 2.0 | FFT backend: GPU (torch) or CPU (scipy, pyFFTW) per call |
| `dm_toolkit/probe.py` | 1.0 | The shared read-out point on the mirror, in aperture coordinates |

## hardware

| File | Version | Purpose |
| --- | --- | --- |
| `camera.py` | 1.9 | Continuous Thorlabs acquisition workers with frame timestamps |
| `manager.py` | 1.1 | One shared camera SDK instance, ref-counted workers per serial |
| `dll.py` | 1.0 | Puts the Thorlabs DLL folder on the search path |
| `pi_link.py` | 1.7 | Discovery, TCP framing and message types shared with the Pi |
| `pi_link_qt.py` | 1.9 | Qt wrapper: connect, multi-channel goto with ack, zero on shutdown |
| `meter.py` | 1.0 | TENMA multimeter worker (DC voltage) |
| `tenma7732a.py` | 1.0 | TENMA 72-7732A protocol over the USB-HID cable |

## interferometry

| File | Version | Purpose |
| --- | --- | --- |
| `fringes.py` | 1.0 | Takeda carrier demodulation: interferogram to complex fringe field |
| `tracking.py` | 1.7 | Frame-to-frame phase tracking with aliasing correction |
| `response.py` | 2.0 | Step-response metrics and the live step-watch logic |
| `noise.py` | 1.0 | Noise-floor spectrum of a displacement trace |
| `loop.py` | 1.0 | Hysteresis loop pipeline: drive mapping, branches, metrics |
| `video.py` | 2.1 | Offline MP4 response analysis and the recording workers |
| `recording.py` | 1.4 | Long interval recording worker (crash-safe CSV) |
| `takeda.py` | 3.1 | Takeda 2-D wavefront reconstruction |
| `psi.py` | 1.0 | Temporal phase-shifting interferometry with self-calibrated steps |
| `psi_bundle.py` | 1.0 | Save and restore a PSI run as one .npz |
| `substrate.py` | 1.0 | Subtract a stored system-error surface |
| `surface.py` | 2.2 | Carrier-free surface reconstruction (obfuscated) |

## hysteresis

| File | Version | Purpose |
| --- | --- | --- |
| `pi_model.py` | 1.1 | Stateful modified Prandtl-Ishlinskii model with play operators |
| `device_profile.py` | 1.1 | Load and validate a channel's calibration profile |
| `driver_voltage_curve.py` | 1.1 | Code to driver-voltage conversion selected by the profile |
| `driver_voltage_lut.py` | 1.0 | Fitted lookup of the driver output voltage |
| `compensator.py` | 2.1 | Inverse model: target displacement to transmitted code |
| `open_loop.py` | 1.3 | Per-channel nominal-code compensation with homing and commit |
| `fit_profile.py` | 1.0 | Fit a profile from measured loop CSVs |
| `step_wave.py` | 2.0 | Step and square-wave drive sequences |
| `comp_sweep.py` | 1.1 | Compensated sweep sequences |
| `devices/` | | Fitted profiles for DM5 (channels 1 to 5) and DM9 (6 to 14) |

## correction

| File | Version | Purpose |
| --- | --- | --- |
| `settings.py` | 1.21 | Actuators, algorithm, metric and every run parameter |
| `metrics.py` | 4.6 | One camera frame to one score plus diagnostics |
| `optimizers.py` | 3.16 | Hill climb, SPGD, genetic, CMA-ES, Bayesian, simulated annealing |
| `modal.py` | 2.7 | 2N+1 and N+2 modal solves on the measured mirror modes |
| `wavefront.py` | 1.0 | Controller driven by single-frame phase retrieval |
| `dual_mirror.py` | 1.5 | Two mirrors behind one optimiser interface |
| `budget.py` | 1.0 | Measurement cost model and the settle/frames ladder |
| `timeline.py` | 1.1 | Per-point timing stamps for every hand-off |
| `ee_curve.py` | 1.2 | Encircled-energy and Strehl curves against the Airy ideal |
| `background.py` | 1.0 | Run-level background reference outside the spot |
| `autoexposure.py` | 1.2 | Exposure policy for the loop and the bench |
| `fastmath.py` | 1.0 | Reduction backend for frame averages |
| `dm_loop_profiles.json` | | Which profile, descriptor and matrix each channel uses |

## adaptive_settling

| File | Version | Purpose |
| --- | --- | --- |
| `model.py` | | Descriptor validation and the wait rule (small, medium, full) |
| `analysis.py` | | Transition residuals from calibration CSVs; writes descriptors |
| `descriptors/` | | Per-channel gate and wait tiers for DM5 and DM9 |

## influence

| File | Version | Purpose |
| --- | --- | --- |
| `io_surface.py` | 1.1 | Read Zygo .datx and .xyz surfaces into nanometres |
| `influence.py` | 1.1 | Influence columns, gradient matrix, SVD modes, noise floor |
| `pipeline.py` | 1.2 | Ordered stages with per-channel checks and superposition test |
| `trace.py` | 1.2 | Worked-answer record of every stage |
| `textdump.py` | 1.0 | Plain-text twin of an exported .npz |
| `session.py` | 1.0 | Remember folder and pupil settings |
| `im_store.py` | 1.7 | Load, save, remap and list control matrices |
| `matrices/` | | Control matrices for DM5 and DM9 |

## zygo

| File | Version | Purpose |
| --- | --- | --- |
| `io_datx.py` | 1.0 | Read a .datx (HDF5) measurement |
| `geometry.py` | 1.0 | Analysis circle and unit coordinates |
| `zernike.py` | 1.0 | Zygo Fringe Zernike polynomials and fit |
| `seidel.py` | 1.0 | Named aberrations from the Fringe coefficients |
| `surface.py` | 1.0 | PV, RMS, term removal and slices |
| `analysis.py` | 1.0 | Complete analysis of one measurement |
| `appx.py` | 1.0 | Unpack an Mx application file to CSV |
| `sequence.py` | 2.0 | Ordered list of mirror setpoints to measure |
| `mx_client.py` | 2.4 | Mx scripting seam with a simulation fallback |
| `mx_agent_link.py` | 1.2 | PC end of the Mx agent link |
| `im_agent_link.py` | 1.0 | PC end of the influence-matrix agent link |
| `metrology.py` | 1.0 | Mx Waves result to surface height |

## beam

| File | Version | Purpose |
| --- | --- | --- |
| `beam.py` | 2.4 | D4sigma, 1/e^2 width and the ISO 11146 M-squared fit |
| `spot_quality.py` | 1.1 | Spot features and the geometric-mean quality score |

## band_scan

| File | Version | Purpose |
| --- | --- | --- |
| `index.py` | 1.0 | Pair saved spot images with the commands behind them |
| `scan.py` | 1.1 | Fit the reciprocal band metric and recommend a band |

## phase_retrieval

| File | Version | Purpose |
| --- | --- | --- |
| `estimate.py` | 1.0 | Fit Zernike coefficients to one focal-plane spot |
| `efield.py` | 1.0 | Electric-field search (Zingarelli and Cain) |
| `torch_model.py` | 1.0 | GPU forward model with autograd Jacobian |
| `dm_basis.py` | 1.0 | Mirror eigenmodes as the retrieval basis |

## bench

| File | Version | Purpose |
| --- | --- | --- |
| `plan.py` | 2.1 | Knob grid, order and cost estimate of a characterisation scan |
| `plots.py` | 2.7 | Figures from a scan directory |
| `repeat_plan.py` | 2.0 | Plan for repeated identical runs |
| `repeat_plots.py` | 2.6 | Figures from a repeatability directory |
| `sweep_plan.py` | 1.0 | Plan for the settle-time and frame-count sweep |
| `sweep_plots.py` | 2.1 | Figures and the Allan-deviation decision plots |
| `tradeoff_plots.py` | | Speed against quality across parameter combinations |

## tools

| File | Version | Purpose |
| --- | --- | --- |
| `retrieve_dm_loop.py` | 1.0 | Phase retrieval on a run's before and after spots |
| `correctability.py` | 1.0 | Split a retrieved wavefront into correctable and residual parts |

## raspberry_pi

See [raspberry_pi/README.md](../raspberry_pi/README.md).

## tests

| File | Covers |
| --- | --- |
| `test_fit_profile.py` | Profile fit from loop CSVs |
| `test_linearized_compensation.py` | Compensated sweeps and step plans |
| `test_im_store.py`, `test_matrix_remap.py` | Control-matrix store |
| `test_phase_retrieval.py`, `test_wavefront.py` | Phase retrieval and its controller |
| `test_adaptive_model.py`, `test_adaptive_analysis.py` | Wait rule and descriptors |
| `test_modal.py`, `test_second_moment.py` | Modal solves and the second-moment metric |
| `test_influence_pipeline.py` | Influence-matrix stages |
| `test_mx_client.py`, `test_mx_agent_link.py`, `test_metrology.py` | Zygo links |
