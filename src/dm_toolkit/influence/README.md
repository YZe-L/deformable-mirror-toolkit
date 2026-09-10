# influence

Version 3.2

Turns measured surface maps into the influence matrix and the mirror's
gradient-orthogonal modes, and stores the resulting control matrix as a
device calibration for the modal solves.

## Files

`io_surface.py` reads what the Zygo software exports into a `SurfaceMap` in
nanometres: `.datx` (HDF5, self-describing) and `.xyz` (text with `No Data`
tokens). `bits_from_name` recovers the channel and code from a file or
folder name, `datx_kind` tells a height map from a tool output such as an
MTF, and `find_surfaces` collects the usable files of a scan folder.

`influence.py` is the numerics. `resample` puts a surface on the pupil grid,
`remove_piston_tilt` subtracts the least-squares plane, `push_pull` and
`slope_from_pairs` build one influence column per actuator from symmetric
pairs, `gradient_matrix` differentiates inside the pupil with quadrature
weights, `eigenmodes` takes the SVD and rescales each control vector to one
radian RMS, and `noise_singular_value` sets the truncation floor from
repeated reference maps. `keep_count`, `footprint_weight` and
`superposition_error` are the checks.

`pipeline.py` runs the stages in order on a folder (`Job` in, `Result` out)
and records every step. It grades each actuator on four measures (probe
asymmetry, incremental gain, signal above noise, residual) and tests the
linear model on surfaces it was not built from.

`trace.py` is the record: each `Step` carries the formula, the inputs and
outputs with shapes and units, and a numeric preview, so the computation can
be read end to end.

`textdump.py` writes a plain-text twin beside every exported .npz.

`session.py` remembers the folder, pupil circle and settings of a computed
matrix so the same dataset reopens with the same geometry.

`im_store.py` is the calibration store the loop reads. `StoredMatrix` holds
the control matrix (bits per radian RMS of each mode), the singular values,
the retained mode count and the wavelength; `curvature` gives the
per-mode curvature in the driven coordinates; `align` reorders rows to the
loop's channels. `save_matrix`, `load_matrix`, `list_matrices` and
`remap_matrix` (for a rewire) manage the files.

## matrices/

The control matrices for the five-actuator mirror (`DM5_im3_ch1-5.npz`) and
the nine-actuator mirror (`unnamed_9ch_20260823_000656.npz`), each with a
text twin. `correction/dm_loop_profiles.json` points at them.
