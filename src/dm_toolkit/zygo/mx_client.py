# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 2.4, 2026-08-04

"""Thin seam over the Zygo `zygo` scripting API, with a simulation fallback."""

import ipaddress
import json
import shutil
import socket
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from ..config import VENDOR_DIR

MX_PORT = 8733  # Mx Remote Access / scripting web service.
PROBE_TIMEOUT_S = 2.0  # TCP reachability check before the first HTTP call.
# instrument.measure() answers only once Mx has acquired AND analysed, so the
# bound has to cover a slow phase map -- it exists to stop a dead link hanging
# the sweep forever, not to time-limit a real measurement.
REQUEST_TIMEOUT_S = 300.0


def _ensure_zygo_on_path():
    """Put VENDOR_DIR on sys.path so `import zygo` resolves.

    The Zygo scripting package is not part of this repository; copy it
    from the Mx installation to VENDOR_DIR/zygo.
    """
    p = str(VENDOR_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_zygo_on_path()

# Result items to pull after each measurement. Each is (label, Mx result path).
# Paths are placeholders in the form Mx expects (Category / Name); confirm the
# exact strings against the Mx Scripting Guide on the instrument.
DEFAULT_RESULTS = [
    ("PV", ["Surface", "PV"]),
    ("RMS", ["Surface", "RMS"]),
    ("Power", ["Surface", "Power"]),
]

# Per-measurement control-data exports (filename -> Mx control path), saved
# with ui.get_control(path).save_data(file). Confirm the paths on the Mx PC
# (right-click the plot -> Identify); any that error are skipped.
DEFAULT_DATA_CONTROLS = {
    "surface.xyz": ["Analysis", "Surface", "Surface"],
    "zernike_standard.csv": ["Analysis", "Surface", "Zernike"],
}


def local_networks():
    """This PC's IPv4 addresses paired with the subnet each one owns.

    Returns:
        list[tuple[str, ipaddress.IPv4Network]]: (address, subnet) pairs, empty
            when the interface table cannot be read.
    """
    try:
        import psutil
    except ImportError:  # Optional: only the route hint depends on it.
        return []
    found = []
    for addrs in psutil.net_if_addrs().values():
        for addr in addrs:
            if addr.family != socket.AF_INET or not addr.netmask:
                continue
            try:
                net = ipaddress.ip_network(f"{addr.address}/{addr.netmask}",
                                           strict=False)
            except ValueError:
                continue
            found.append((addr.address, net))
    return found


def routes_known():
    """bool: True when the local interface table can be read at all."""
    return bool(local_networks())


def source_ip(host):
    """Local address an Mx call to `host` would leave this PC from, or None.

    Both bench links are direct cables with no gateway, so the answer is the
    adapter whose subnet contains the Zygo PC, picked by longest prefix. A
    UDP-connect is not usable on a multi-homed Windows box.

    Args:
        host (str): Mx hostname or IP; blank means this PC.

    Returns:
        str | None: The source IP, or None when the host is a name, is off
            every local subnet, or the interface table is unreadable.
    """
    if not (host or "").strip():
        return "127.0.0.1"
    try:
        address = ipaddress.ip_address(host.strip())
    except ValueError:  # A name: only DNS could answer, and it may block.
        return None
    if address.is_loopback:
        return "127.0.0.1"
    best, best_len = None, -1
    for local, net in local_networks():
        if address in net and net.prefixlen > best_len:
            best, best_len = local, net.prefixlen
    return best


def probe(host, port=MX_PORT, timeout=PROBE_TIMEOUT_S):
    """Check the Mx port answers before handing over to the zygo package.

    ``connectionmanager.connect`` has no timeout of its own, so an unreachable
    Zygo PC stalls the worker thread with nothing to show for it. A short
    connect first turns that into one sentence naming the actual fault.

    Args:
        host (str): Mx hostname or IP; blank means localhost.
        port (int): Mx web-service port.
        timeout (float): Seconds to wait for the TCP handshake.

    Returns:
        str | None: None when the port answered, else why it did not.
    """
    target = host or "localhost"
    try:
        with socket.create_connection((target, int(port)), timeout):
            return None
    except socket.timeout:
        return (f"no answer from {target}:{port} within {timeout:g}s -- the "
                f"Zygo PC's firewall is dropping the port (add the rule "
                f"below), or the IP is wrong")
    except ConnectionRefusedError:
        return (f"{target} is reachable but refuses port {port} -- Mx is not "
                f"running, or its Remote Access / scripting service is off")
    except socket.gaierror:
        return f"cannot resolve {target!r}"
    except OSError as error:
        return (f"cannot reach {target}:{port} -- {error}. Check the cable and "
                f"that this PC has an address on the same subnet")


class _DirectTransport:
    """urllib stand-in for the zygo package: no proxy, bounded timeout.

    Replaces ``connectionmanager._request`` only, so the rest of the process
    keeps the stock urllib. Both parts matter on this bench:

    * The instrument sits on a private/link-local address. urllib otherwise
      reads the Windows proxy settings, and a VPN or Clash-style proxy does
      not bypass 169.254.* -- every Mx call would be posted to the proxy and
      fail with an error that names the proxy, not Mx.
    * A stock ``urlopen`` waits forever, so a cable pulled mid-sweep would
      freeze the Mx worker thread with no error to report.
    """

    Request = urllib.request.Request

    def __init__(self, timeout=REQUEST_TIMEOUT_S):
        """Build the transport.

        Args:
            timeout (float): Seconds any single Mx call may take.
        """
        self.timeout = float(timeout)
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}))

    def urlopen(self, request, *args, **kwargs):
        """Open one request through the proxy-free opener.

        Args:
            request: urllib Request the zygo package built.
            *args: Passed through to the opener.
            **kwargs: Passed through; a missing timeout gets ours.
        """
        kwargs.setdefault("timeout", self.timeout)
        return self._opener.open(request, *args, **kwargs)


def install_direct_transport(connectionmanager, timeout=REQUEST_TIMEOUT_S):
    """Point the zygo package's one network chokepoint at _DirectTransport.

    Every module in the package sends through ``connectionmanager``, so this
    single swap covers measure, read and save alike.

    Args:
        connectionmanager: The imported zygo.connectionmanager module.
        timeout (float): Seconds any single Mx call may take.

    Returns:
        _DirectTransport: The installed transport.
    """
    transport = getattr(connectionmanager, "_request", None)
    if isinstance(transport, _DirectTransport):
        transport.timeout = float(timeout)
        return transport
    transport = _DirectTransport(timeout)
    connectionmanager._request = transport
    return transport


class MxClient:
    """Connect to Mx (local or remote) and drive measure / save / read."""

    def __init__(self, host=None, app_path=None, sample_datx=None,
                 simulate=False, timeout=REQUEST_TIMEOUT_S):
        """Initialize the MxClient.

        Args:
            host: Remote host name or address.
            app_path: Filesystem path for the app data.
            sample_datx: Reference DATX file used to configure Mx exports.
            simulate: Run against the built-in simulation instead of Mx.
            timeout: Seconds any single Mx call may take.
        """
        self.host = host  # None -> localhost
        self.app_path = app_path  # None -> use whatever app is open.
        self.sample_datx = sample_datx  # Used only in simulation.
        self._force_sim = bool(simulate)  # Explicit offline/dev simulation.
        self.timeout = float(timeout)
        self.connected = False
        self.simulated = False
        self.source_ip = None  # Local IP the Mx traffic actually leaves by.
        self._mx = None
        self._instrument = None
        self._connectionmanager = None

    def connect(self):
        """Connect.

        Connect to Mx, or simulate only when the Zygo package is unavailable.

        By default it uses the application YOU opened in Mx (recommended). Only
        if app_path is given does it open that .appx itself. Warns if no
        application is open, since measure/get_result need one.
        """
        if self._force_sim:  # Explicit offline/dev mode.
            self.simulated = True
            self.connected = True
            return "SIMULATION (forced -- no Mx)"
        try:
            from zygo import mx, instrument  # From the Mx install, see vendor/.
            from zygo import connectionmanager
        except (ImportError, ModuleNotFoundError) as e:
            self.simulated = True
            self.connected = True
            return f"SIMULATION (no zygo package: {e})"

        self._mx, self._instrument = mx, instrument
        self.simulated = False
        target = self.host or "localhost"
        # Two cables leave this PC (Pi and Zygo); record which one carries Mx
        # so a wrong-NIC setup is visible instead of just "connection failed".
        self.source_ip = source_ip(target)
        # No proxy, and a bounded wait -- see _DirectTransport. Installed
        # before the first call, which connectionmanager.connect already makes.
        install_direct_transport(connectionmanager, self.timeout)
        fault = probe(target, MX_PORT)
        if fault is not None:
            self.connected = False
            return f"Mx connection failed -- {fault}"
        try:
            connectionmanager.connect(
                force_if_active=False,
                host=target,
                port=MX_PORT,
            )
        except Exception as e:
            self.connected = False
            return f"Mx connection failed ({target}:{MX_PORT}): {e}"

        self._connectionmanager = connectionmanager
        self.connected = True
        if self.app_path:  # Option B: script opens it.
            if not mx.is_application_open():
                try:
                    mx.open_application(self.app_path)
                except Exception as e:
                    return (f"connected to Mx ({target}) but could not open "
                            f"{self.app_path}: {e}")
            return f"connected ({target}{self._via()}), app ready"
        if not mx.is_application_open():  # Option A: you open it.
            return (f"connected ({target}{self._via()}) but NO app open "
                    "-- open your .appx in Mx first")
        return f"connected to Mx ({target}{self._via()}), app ready"

    def _via(self):
        """str: ``" via <local ip>"`` once a route is known, else ""."""
        return f" via {self.source_ip}" if self.source_ip else ""

    def check_alive(self):
        """Why the link is unusable, or None when Mx still answers.

        Returns:
            str | None: A reason, or None.
        """
        if not self.connected:
            return "Mx is not connected"
        if self.simulated:
            return None
        return probe(self.host or "", MX_PORT)

    def measure(self):
        """Trigger one acquisition + analysis."""
        if self.simulated:
            time.sleep(0.05)
            return
        self._instrument.measure()

    def save_data(self, path):
        """Save the full measurement (.datx). In sim, copy the sample file."""
        path = str(path)
        if self.simulated:
            if self.sample_datx and Path(self.sample_datx).is_file():
                shutil.copyfile(self.sample_datx, path)
            return path
        self._mx.save_data(path)
        return path

    def save_surface_datx(self, path):
        """Save the current processed surface plot as a self-describing DATX.

        Unlike the full-measurement ``mx.save_data`` call, saving the default
        plot control records the data state currently shown by Mx. That is the
        state the impact-matrix reader needs when Mx has already applied masks,
        filtering or fit removal.

        Args:
            path: Destination .datx path on the Mx-visible filesystem.

        Returns:
            The destination path, or ``None`` in simulation without a sample.
        """
        path = str(path)
        if self.simulated:
            if self.sample_datx and Path(self.sample_datx).is_file():
                shutil.copyfile(self.sample_datx, path)
                return path
            return None
        from zygo import ui
        ui.get_control(list(ui.get_default_plot_control_path())).save_data(path)
        return path

    def get_results(self, items=None):
        """dict(label -> value) EXACTLY as Mx displays them (no recompute).

        This is how we avoid the piston/colorbar/Zernike deviations seen when
        recomputing from the .datx: we read Mx's own numbers. items is a list of
        (label, Mx result path); confirm the paths against the Scripting Guide.
        """
        items = items or DEFAULT_RESULTS
        if self.simulated:
            import random
            return {label: round(random.uniform(-1, 5), 3) for label, _ in items}
        try:  # One round-trip for all values.
            paths = [(p, None) for _, p in items]
            vals = self._mx.get_bulk_result_values(paths)
            return {label: v for (label, _), v in zip(items, vals)}
        except Exception:
            out = {}
            for label, path in items:
                try:
                    out[label] = float(self._mx.get_result_number(path))
                except Exception:
                    out[label] = None
            return out

    def get_result_number(self, path, unit=None):
        """One Mx result, converted by Mx into the unit asked for.

        Asking Mx for the unit means the number does not depend on what the Mx
        window happens to display, which is what makes an unattended scan
        reproducible.

        Args:
            path: Mx result path, e.g. ``["Surface", "PV"]``.
            unit: Mx unit name ("Waves", "NanoMeters", ...), or None for the
                unitless read.

        Returns:
            float: The value Mx reports.
        """
        if self.simulated:
            import random
            return round(random.uniform(0.02, 0.4), 6)
        return float(self._mx.get_result_number(list(path), unit))

    def save_plot_pngs(self, controls, prefix):
        """Save each plot control as the PNG Mx renders it.

        Save each plot control as the PNG Mx renders it (surface/fit/residual
        exactly as shown -- zero deviation). controls: {name: control-id}.
        Returns {name: path}. In simulation, writes nothing.

        Args:
            controls: Sequence of control values.
            prefix: Filename or identifier prefix.
        """
        out = {}
        if self.simulated or not controls:
            return out
        for name, control in controls.items():
            try:
                png = self._mx._get_native_image_stream(control)
                path = f"{prefix}_{name}.png"
                with open(path, "wb") as fh:
                    fh.write(png)
                out[name] = path
            except Exception:
                pass
        return out

    def version(self):
        if self.simulated:
            return "simulation"
        try:
            return str(self._mx.get_mx_version())
        except Exception:
            return "unknown"

    def save_signal(self, path):
        """Raw phase-shift signal (SaveSignalData). Best-effort."""
        if self.simulated:
            return None
        self._mx.save_signal_data(str(path))
        return str(path)

    def save_application(self, path):
        """Copy the running .appx (save_application_as). Best-effort."""
        if self.simulated:
            return None
        self._mx.save_application_as(str(path))
        return str(path)

    def save_control(self, control_path, file_path):
        """Export one control's data.

        Export one control's data (surface .xyz, Zernike .csv/.int, …) via
        ui.get_control(path).save_data -- exactly what Mx would write.

        Args:
            control_path: Filesystem path for the control data.
            file_path: Filesystem path for the file data.
        """
        if self.simulated:
            return None
        from zygo import ui
        ui.get_control(control_path).save_data(str(file_path))
        return str(file_path)

    def save_surface_xyz(self, path):
        """Save the surface point cloud (.xyz) from Mx's default plot control.

        Save the surface point cloud (.xyz) from Mx's default plot control --
        the raw companion to the .datx. Server-side write (lands on the Mx PC).
        """
        if self.simulated:
            return None
        from zygo import ui
        ui.get_control(list(ui.get_default_plot_control_path())).save_data(str(path))
        return str(path)

    def log_reports(self):
        """Trigger Mx's configured report.

        Trigger Mx's configured report (Reports Options -> Save Report On).
        Mx has no 'save report to path' call; this is the only hook, and the PDF
        lands in the folder configured IN the app, on the Mx PC. Best-effort.
        """
        if self.simulated:
            return None
        self._mx.log_reports()
        return "log_reports"

    def save_full_dataset(self, out_dir, results=None, setpoint=None,
                          data_controls=None):
        """Write a complete per-measurement bundle into out_dir.

        Write a complete per-measurement bundle into out_dir: processed.datx,
        raw_signal, the .appx, control exports (surface/zernike), plus
        scalar_results.json + manifest.json (always, pure Python). Every Mx call
        is best-effort so a missing control path never aborts the bundle.
        Returns {filename: path} of what was actually written.

        Args:
            out_dir: Destination directory.
            results: Result records to aggregate or display.
            setpoint: Command setpoint associated with the measurement.
            data_controls: Sequence of data control values.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = {}

        def _try(name, fn):
            try:
                p = fn()
                if p:
                    written[name] = str(p)
            except Exception:
                pass

        _try("processed.datx", lambda: self.save_data(str(out / "processed.datx")))
        # .dat is NOT a known signal format (Mx: "does not specify a known
        # signal
        # data file format"); signal data is a .datx container.
        _try("raw_signal.datx", lambda: self.save_signal(out / "raw_signal.datx"))
        _try("application.appx", lambda: self.save_application(out / "application.appx"))
        for fname, ctrl in (data_controls or DEFAULT_DATA_CONTROLS).items():
            _try(fname, lambda f=fname, c=ctrl: self.save_control(c, out / f))
        # Trigger Mx's configured report (writes the .pdf on the Mx PC if
        # Reports
        # Options -> Save Report On is set up). Best-effort; harmless if not.
        _try("log_reports", self.log_reports)

        # Always-writable, hardware-independent records.
        if results:
            p = out / "scalar_results.json"
            p.write_text(json.dumps(results, indent=2), encoding="utf-8")
            written["scalar_results.json"] = str(p)
        manifest = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mx_version": self.version(),
            "simulated": self.simulated,
            "host": self.host or "localhost",
            "setpoint": {str(k): v for k, v in (setpoint or {}).items()},
            "results": results or {},
            "files": sorted(written.keys()),
            "note": ("processed.datx = current data state (Mx SaveData). "
                     "raw_signal via SaveSignalData; surface/zernike via control "
                     "SaveData. See Mx Reference Guide pp.285-286, 443."),
        }
        mp = out / "manifest.json"
        mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        written["manifest.json"] = str(mp)
        return written

    def close(self):
        if self._connectionmanager is not None:
            try:
                self._connectionmanager.terminate()
            except Exception:
                pass
            self._connectionmanager = None
        self.connected = False
