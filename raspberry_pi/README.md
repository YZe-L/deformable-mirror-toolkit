# Raspberry Pi scripts

Version 3.2

The Pi drives the piezo channels through a ServoPi (PCA9685) PWM board at
I2C address 0x40 and 1526 Hz. The PC never talks to the board directly. It
sends 12-bit commands over TCP and the Pi applies them and acknowledges.

## Setup

Enable I2C, then install the Python dependencies:

```bash
pip3 install -r ../requirements-pi.txt
```

The PWM driver is not part of this repository. The scripts do
`from ServoPi import PWM`, which is the `ServoPi.py` module of the
[AB Electronics UK Python Libraries](https://github.com/abelectronicsuk/ABElectronics_Python_Libraries)
(MIT licence). Either install that library on the Pi, or copy its
`ServoPi/ServoPi.py` into this folder. The scripts use `PWM(0x40)`,
`set_pwm_freq`, `output_enable` and `set_pwm(channel, on, off)`.

Copy this folder to the Pi. The scripts import each other by file name, so
keep them together.

## Link

The PC is the server. It broadcasts a UDP beacon on port 65433 and listens on
TCP port 65432. The Pi finds the beacon, dials in and keeps a heartbeat. Every
`goto` carries the full channel vector and a sequence number, and the Pi
replies `settled` with the same number once the PWM registers are written.
`pi_link.py` on the Pi and `dm_toolkit/hardware/pi_link.py` on the PC are the
same protocol file.

## Scripts

| File | Purpose |
| --- | --- |
| `pi_link.py` | Discovery, TCP framing, heartbeat and message types |
| `pi_dm_sequence.py` | Serve multi-channel mirror commands from the PC; zeroes every channel on exit |
| `pi_closed_loop.py` | Serve single-channel commands from the PC |
| `pi_sweep_link.py` | Terminal-driven lock-step sweeps for hysteresis measurements |
| `pi_setup_time.py` | Trigger step transitions for settling-time measurements |

## Run

Multi-channel service for the correction loop and the influence-matrix
scans:

```bash
python3 pi_dm_sequence.py
```

A lock-step hysteresis sweep on one channel:

```bash
python3 pi_sweep_link.py ramp --channel 2 --step 50 --delay 1
```

Add `--dry-run` to any script to print the PWM writes instead of touching the
board, and `--host <ip>` to skip discovery.
