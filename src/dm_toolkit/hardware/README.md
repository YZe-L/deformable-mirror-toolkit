# hardware

Version 3.2

Everything that touches a device: the Thorlabs camera, the Raspberry Pi that
drives the mirror, and the TENMA multimeter that records the driver voltage.
The worker classes are Qt threads; the protocol module is plain Python.

## Files

`camera.py` runs continuous acquisition in a background thread.
`ThorlabsAcqWorker` polls the SDK, stamps every frame with the time it became
available and how long the poll took, and publishes the newest frame to
consumers. Publication is capped at 60 fps when nothing is recording.
`get_exposure_ms` is the only source of truth for the exposure; callers ask
rather than cache. `save_frame` writes a frame losslessly.

`manager.py` owns the single `TLCameraSDK` instance and hands out one
acquisition worker per camera serial, reference counted, so two consumers can
share a camera without opening it twice. The SDK is imported lazily; without
it the manager reports that no camera can be opened and everything else still
runs.

`dll.py` adds `vendor/dll/64_lib` (or `32_lib`) to the DLL search path. It
must be imported before `thorlabs_tsi_sdk`; `manager.py` does that.

`pi_link.py` is the wire protocol shared with the Pi, byte for byte the same
file as `raspberry_pi/pi_link.py`. The PC is the server: `Beacon` broadcasts
on UDP 65433, `Listener` accepts the Pi on TCP 65432, and `Link` frames
newline JSON messages with a heartbeat. Message types cover the lock-step
sweep (`at`, `got`, `recorded`), direct commands (`goto`, `settled`), the
settling-time trigger (`trigger`, `result`) and `abort`.

`pi_link_qt.py` wraps the link in Qt signals for a GUI thread.
`PiLinkController` connects, sends multi-channel `goto` commands with
sequence numbers, records the last command accepted (the mirror has no
read-back) and, on shutdown, drives every channel that could be wired to
zero and waits for the acknowledgement. `PiConnectDialog` is the small
connection dialog.

`meter.py` is the multimeter worker. It listens to the TENMA 72-7732A over
its USB-HID cable and emits DC voltage readings.

`tenma7732a.py` holds the cable protocol: the feature report that sets the
baud rate and `parse_dcv`, which decodes one reading.
