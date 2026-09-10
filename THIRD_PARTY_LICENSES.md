# Third-party code and licences

This project is released under the GNU General Public License v3.0 or later
(see [LICENSE](LICENSE)). Two runtime dependencies force that choice: PyQt5 is
GPL-3.0 (or commercial) and pyFFTW bundles FFTW3, which is GPL-2.0-or-later.
OpenCV is Apache-2.0, which is compatible with GPLv3 but not GPLv2.

No third-party source code is vendored in this repository.

## Software that is not included

The ServoPi (PCA9685) PWM driver used on the Raspberry Pi is the `ServoPi.py`
module of the
[AB Electronics UK Python Libraries](https://github.com/abelectronicsuk/ABElectronics_Python_Libraries),
MIT licensed. Install it on the Pi as described in
[raspberry_pi/README.md](raspberry_pi/README.md).

The Thorlabs Scientific Camera SDK (`thorlabs_tsi_sdk` and its DLLs) is
governed by the Thorlabs end-user licence and is not redistributed here.
Install it from [ThorCam](https://www.thorlabs.com/software-pages/ThorCam)
and place the DLLs under `vendor/dll/` as described in
[docs/HARDWARE.md](docs/HARDWARE.md).

The Zygo Mx scripting package (`zygo`) ships with the Zygo Mx software and is
not redistributed here. `dm_toolkit/zygo/mx_client.py` looks for it under
`vendor/zygo` and falls back to a simulation when it is absent.

## Obfuscated module

`src/dm_toolkit/interferometry/surface.py` is distributed as a PyArmor
obfuscated script together with its runtime package
`src/dm_toolkit/pyarmor_runtime_000000/`. Both may be redistributed with this
project. The runtime is built for 64-bit Windows and Python 3.11.

## Runtime dependency licences

| Package | Licence |
| --- | --- |
| PyQt5 | GPL-3.0 or commercial |
| pyfftw (bundled FFTW3) | GPL-2.0-or-later |
| numpy, scipy, matplotlib, pandas, h5py, scikit-image | BSD-3-Clause |
| torch, bottleneck, scikit-optimize, cma, pynvml | BSD-3-Clause |
| opencv-python-headless | Apache-2.0 |
| prysm, psutil | MIT |
| RPi.GPIO, smbus2 | MIT |
| hidapi (Python binding) | GPL-3.0 / BSD-3-Clause / HIDAPI (BSD elected) |
