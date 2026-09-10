# dm_toolkit

Version 3.2

The package root holds the four modules that every sub-package shares. Each
sub-package has its own README with one entry per file.

## Files here

`__init__.py` declares the package and carries `__version__`.

`config.py` defines the paths the rest of the code relies on: `REPO_DIR`
(the checkout root), `OUTPUT_DIR` (where runs write frames, CSVs and
figures) and `VENDOR_DIR` (where the proprietary Thorlabs DLLs and the Zygo
scripting package go). `portable_path` and `resolve_path` store paths in
settings files relative to the checkout, so a settings file written on one
machine still finds its profiles on another.

`zernike.py` is the Zernike toolbox used by every wavefront consumer:
Noll-indexed modes on a unit disk (`zernike_mode`, `basis`), a least-squares
fit of a wavefront map (`fit`), and `named_aberrations`, which turns
coefficients into orientation-free magnitudes and angles (defocus,
astigmatism, coma, trefoil, spherical). `cross_check_prysm` compares the
hand-rolled basis against the prysm library.

`backend.py` picks where a 2-D FFT runs. `resolve` returns the GPU (torch
with CUDA) when it is present and requested, otherwise the CPU path through
scipy or pyFFTW. `spiral_transform` is the spiral-phase operator used by the
carrier-free surface reconstruction. The torch import is lazy because it
costs seconds.

`probe.py` holds the one read-out point on the mirror that the live
displacement readers share. It is stored in normalised aperture coordinates
(u, v in the unit disk) so it survives downsampling and aperture
re-detection. `to_pixels` and `from_pixels` convert for a given aperture
circle.

## Sub-packages

| Folder | What it does |
| --- | --- |
| `hardware/` | Camera, Raspberry Pi link, multimeter |
| `interferometry/` | Michelson fringe analysis, displacement tracking, PSI |
| `hysteresis/` | Piezo hysteresis model, profile fitting, compensation |
| `correction/` | Image scores, optimisers, modal solves, timing |
| `adaptive_settling/` | Scheduled wait from the predicted displacement |
| `influence/` | Influence matrix and mirror modes from surface maps |
| `zygo/` | Zygo data files and remote control of the Mx software |
| `beam/` | Beam width, M-squared, spot features |
| `band_scan/` | Choose the PSD band for the modal solve |
| `phase_retrieval/` | Zernike coefficients from one spot image |
| `bench/` | Plans and figures for measurement campaigns |
| `tools/` | Command-line analysis scripts |
| `pyarmor_runtime_000000/` | Runtime for the obfuscated surface module |
