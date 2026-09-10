# zygo

Version 3.2

Two things: reading and analysing the surface files the Zygo Mx software
writes, and driving that software remotely so a mirror scan can trigger
acquisitions.

## Data files and analysis

`io_datx.py` reads a `.datx` (HDF5) file into a `Measurement`: height in
waves, intensity, the validity mask, wavelength and scale factor.
`find_datx` lists the files of a folder.

`geometry.py` fits the analysis circle to the valid pixels and returns unit
coordinates for the Zernike fit.

`zernike.py` implements the Zygo Fringe Zernike set: `basis`, a least-squares
`fit`, `reconstruct` and `fit_map`. Coefficients are in the same order and
normalisation the Mx software reports.

`seidel.py` converts the Fringe coefficients into named aberration
magnitudes and angles.

`surface.py` computes PV and RMS, removes selected term groups, and samples
slice profiles.

`analysis.py` runs all of the above on one measurement and returns an
`Analysis` with every number and map.

`appx.py` unpacks an Mx application file (`.appx`) into a readable CSV of
its settings.

## Remote control

`sequence.py` builds the ordered list of mirror setpoints a scan visits
(`ScanPlan`, `Setpoint`), including a full grid over channels and single
channel sweeps.

`mx_client.py` wraps the Zygo `zygo` scripting package: connect, acquire,
read a result number, export the surface. It runs on the Mx PC, where that
package is installed under `vendor/zygo`; without the package it falls back
to a simulation. `source_ip` and `probe` sort out which network adapter
reaches the Mx PC.

`mx_agent_link.py` is the PC side of the agent that runs on the Mx PC and
dials out to the PC, so the Mx PC's firewall stays untouched. `MxAgentClient`
offers the same connect, measure, get_result_number and close calls as
`MxClient`, plus `probe_point` (height at one coordinate) and
`set_reference` (re-measure the substrate).

`im_agent_link.py` is the same link used for influence-matrix scans: the
agent saves each surface on the Mx PC and replies with the file name, so no
surface crosses the cable.

`metrology.py` converts a Waves result displayed by Mx into surface height
in nanometres and back, and computes displacement between two readings.
