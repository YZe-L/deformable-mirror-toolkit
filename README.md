# DM Toolkit

Version 3.2

Control, metrology and wavefront-sensorless correction for low-cost
piezoelectric deformable mirrors. The code drives a mirror from a Raspberry
Pi, measures its surface with a Michelson or Zygo interferometer, compensates
piezo hysteresis, and corrects a focal spot from camera images alone with
six search methods and a model-based modal solve.

The Qt front end that the lab used is not part of this release. What is here
is the library it was built on, the Raspberry Pi scripts, the calibration
files for the two mirrors and the tests.

## Install

Python 3.11 on Windows is the tested setup.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[search,speed,hardware,test] 
```

Or `pip install -r requirements.txt`. The Thorlabs camera SDK and the Zygo
scripting package are proprietary and are not included; see
[docs/HARDWARE.md](docs/HARDWARE.md). On the Pi use `requirements-pi.txt`.

Run the tests with:

```bash
pytest
```

## Layout

```
src/dm_toolkit/
    hardware/          camera, Raspberry Pi link, multimeter
    interferometry/    fringe demodulation, displacement tracking, PSI
    hysteresis/        Prandtl-Ishlinskii model, profile fit, compensation
    correction/        scores, optimisers, modal solve, timing, budget
    adaptive_settling/ scheduled-wait rule from predicted displacement
    influence/         influence matrix, mirror modes, control-matrix store
    zygo/              DATX reader, Zernike fit, remote Mx control
    beam/              D4sigma, M-squared, spot features
    band_scan/         PSD band selection from saved spots
    phase_retrieval/   Zernike coefficients from one spot image
    bench/             campaign plans and figures
    tools/             stand-alone commands
raspberry_pi/          scripts that run on the Pi
tests/                 pytest suite (no hardware needed)
docs/                  what every file does, hardware setup
```

[docs/MODULES.md](docs/MODULES.md) lists every file with its version and
purpose, and every folder has a README describing its files in more detail.

## How the pieces connect

A correction run works like this. The PC sends a command vector to the Pi
(`hardware/pi_link_qt.py`), which writes the PWM registers and acknowledges.
With compensation on, `hysteresis/open_loop.py` turns each nominal code into
the transmitted code from the channel's fitted profile, and
`adaptive_settling/model.py` picks the wait from the predicted displacement
change. After the wait the camera frames are averaged and scored by
`correction/metrics.py`. The score goes to the chosen optimiser in
`correction/optimizers.py` (hill climbing, simulated annealing, genetic,
Bayesian, SPGD, CMA-ES) or to the modal solve in `correction/modal.py`, which
needs a control matrix from `influence/`. `correction/timeline.py` stamps
every hand-off so the run can be timed afterwards.

The interferometry side is independent of the loop. `interferometry/fringes.py`
and `tracking.py` turn a Michelson carrier interferogram into a displacement
trace, `response.py` fits the step response, and `hysteresis/fit_profile.py`
fits a compensation profile from measured loops. `influence/pipeline.py`
turns Zygo surface maps into the influence matrix and the mirror modes.

## Obfuscated module

`interferometry/surface.py` (carrier-free surface reconstruction) is shipped
as a PyArmor obfuscated script with its runtime in
`dm_toolkit/pyarmor_runtime_000000/`. It imports and runs like any other
module on 64-bit Windows with Python 3.11. Other platforms need the plain
source, which is not included.

## Code style

Python follows the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html):
Google docstrings, 80 columns, `lower_with_under` names. Comments say why,
not what.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE) and
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
