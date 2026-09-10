# Hardware setup

Version 3.2

## Camera

The frame source is a Thorlabs CS165MU (1440 x 1080, 10 bit, 3.45 um
pixels), read through the Thorlabs Scientific Camera SDK.

1. Install ThorCam from Thorlabs and the USB driver it carries.
2. Install the Python package that ships inside the ThorCam SDK archive:

   ```bash
   python -m pip install thorlabs_tsi_camera_python_sdk_package.zip
   ```

3. Copy the SDK's native DLLs to `vendor/dll/64_lib/` (or `32_lib/`).
   `dm_toolkit/hardware/dll.py` adds that folder to the DLL search path
   before the SDK is imported.

`hardware/manager.py` holds one SDK instance and hands out acquisition
workers per camera serial. The SDK is imported lazily, so everything else
works without it.

## Raspberry Pi and mirror

The PC side is `hardware/pi_link.py` (protocol) and `hardware/pi_link_qt.py`
(Qt signals, multi-channel `goto`, shutdown zero). The Pi side is in
`raspberry_pi/`. Both machines must be on the same subnet. Allow inbound TCP 65432
on the PC firewall; the discovery beacon leaves the PC on UDP 65433.

Channel wiring used by the shipped calibration files:

| Mirror | Channels |
| --- | --- |
| DM5 (five actuators) | 1 to 5 |
| DM9 (nine actuators) | 6 to 14 |

Commands are integer codes 0 to 4095 on the PWM duty cycle. The driver
converts them to roughly 1 to 150 V.

## Calibration files shipped with the code

| Path | Contents |
| --- | --- |
| `hysteresis/devices/DM5/*.json`, `DM9/*.json` | Fitted Prandtl-Ishlinskii profile per channel |
| `hysteresis/devices/dm_d.json`, `piezo_a.json` | Single-piezo profiles used by the tests and as defaults |
| `adaptive_settling/descriptors/dm5/`, `dm9/` | Residual gate and wait tiers per channel |
| `influence/matrices/*.npz`, `*.txt` | Control matrices (bits per radian RMS of each mode) |
| `correction/dm_loop_profiles.json` | Which profile, descriptor and matrix each channel uses |

Paths inside `dm_loop_profiles.json` are relative to the checkout, so the
file works on another machine without editing.

## Multimeter

`hardware/tenma7732a.py` reads a TENMA 72-7732A over its USB-HID cable
(`hidapi`). It is used to record the driver voltage while a channel is swept.

## Zygo Mx

Surface maps come from a Zygo Verifire. Two ways to drive it:

`zygo/mx_client.py` wraps the `zygo` scripting package and runs on the Mx PC
itself. Put the package under `vendor/zygo`. Without it the client falls back
to a simulation, which is enough for the tests.

`zygo/mx_agent_link.py` and `zygo/im_agent_link.py` are the PC side of a
small agent that runs on the Mx PC and dials out to the PC, so no inbound
firewall rule is needed on the Mx PC. The agent itself is not part of this
repository.
