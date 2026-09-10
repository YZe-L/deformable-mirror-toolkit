# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.2, 2026-08-11

"""PC-side end of the Mx agent link, the same shape as the Raspberry Pi's.

A small program on the Mx PC (``mx_agent.py``) dials out to this PC, so no
inbound firewall rule is needed there, and reaches Mx over loopback. This
client is a drop-in for :class:`MxClient` (connect / measure /
get_result_number / close). The protocol is newline-delimited JSON, one
reply per request.
"""

import json
import socket
import time

DEFAULT_PORT = 65433  # 65432 belongs to the Raspberry Pi link.
ACCEPT_TIMEOUT_S = 180.0  # How long "connect" waits for the agent to dial in.
REPLY_TIMEOUT_S = 300.0  # A measure answers only once Mx has analysed.


class AgentError(Exception):
    """The link died, or the agent reported a command it could not run."""


class MxAgentClient:
    """Waits for the Mx PC's agent, then sends it measurement commands."""

    # Which agent this client speaks to. The kit holds two that dial the same
    # port and only one runs at a time, so starting the wrong one has to be
    # named as such instead of failing on the first command sent.
    KIND = "mx"
    OTHER = {"mx": "mx_agent.bat", "im": "im_agent.bat"}
    POINT_AGENT_VERSION = (1, 4)
    BATCH_POINT_AGENT_VERSION = (1, 5)

    def __init__(self, port=DEFAULT_PORT, accept_timeout=ACCEPT_TIMEOUT_S,
                 timeout=REPLY_TIMEOUT_S, simulate=False):
        """Build the client.

        Args:
            port (int): TCP port this app listens on.
            accept_timeout (float): Seconds to wait for the agent to dial in.
            timeout (float): Seconds to wait for one reply.
            simulate (bool): Never used here; kept so the two clients match.
        """
        self.port = int(port)
        self.accept_timeout = float(accept_timeout)
        self.timeout = float(timeout)
        self.simulated = bool(simulate)
        self.connected = False
        self.source_ip = None  # Local address the agent dialled in to.
        self.info = {}  # The agent's hello reply.
        self._sock = None
        self._buf = bytearray()

    # Lifecycle
    def _open(self):
        """Accept the agent, read its hello, and check it is the right one.

        Returns:
            tuple: (peer address, fault). ``fault`` is None when the link is up
                and speaking to the agent this client expects.
        """
        self.close()
        try:
            self._sock = _accept(self.port, self.accept_timeout)
        except AgentError as error:
            return None, f"Mx agent: {error}"
        self.source_ip = self._sock.getsockname()[0]
        peer = self._sock.getpeername()[0]
        self.connected = True
        try:
            self.info = self._ask("hello", timeout=60.0)
        except AgentError as error:
            self.close()
            return peer, (f"agent connected from {peer} but did not answer: "
                          f"{error}")
        # Agents written before the kinds existed only ever spoke the mx set.
        kind = str(self.info.get("kind") or "mx")
        if kind != self.KIND:
            self.close()
            return peer, (f"that is the '{kind}' agent on {peer} -- close it "
                          f"and start {self.OTHER.get(self.KIND, 'the other')} "
                          f"instead")
        # Report the AGENT's simulation, not ours: a made-up number must never
        # reach the CSV labelled as a real measurement.
        self.simulated = bool(self.info.get("simulate"))
        return peer, None

    def connect(self):
        """Listen, wait for the agent, and ask it who it is.

        Returns:
            str: What happened, for a status line.
        """
        peer, fault = self._open()
        if fault is not None:
            return fault
        mx_state = str(self.info.get("mx", "unknown"))
        if "ready" not in mx_state:
            # Keep the link: the operator can open the .appx and press again
            # without the agent having to be restarted.
            return f"agent on {peer} is up, but NO app open -- {mx_state}"
        sim = " (agent SIMULATING)" if self.info.get("simulate") else ""
        agent_version = self.info.get("agent") or "legacy"
        return (f"agent {agent_version} on {peer} ready, "
                f"Mx {self.info.get('mx_version')}, "
                f"reads {self.info.get('result')} in "
                f"{self.info.get('unit')}{sim}")

    def close(self):
        """Say goodbye if we can, then drop the socket."""
        if self._sock is not None:
            try:
                self._ask("bye", timeout=5.0)
            except (AgentError, OSError):
                pass  # Teardown must not raise.
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
        self._buf = bytearray()
        self.connected = False

    def check_alive(self):
        """Why the link is unusable, or None when it is fine.

        Returns:
            str | None: A reason, or None.
        """
        if self._sock is None or not self.connected:
            return "the Mx agent is not connected"
        return None

    # Measurement -- the MxClient interface the worker calls.
    def measure(self):
        """Tell the agent to run one acquisition and analysis on the Mx PC."""
        self._ask("measure")

    def get_result_number(self, path, unit=None):
        """One Mx result, converted by Mx into the unit asked for.

        Args:
            path (list[str]): Mx result path.
            unit (str | None): Mx unit name, or None for the agent's own
                configured unit.

        Returns:
            float: The value Mx reports.
        """
        reply = self._ask("read", result=list(path), unit=unit)
        return float(reply["value"])

    def measure_read(self, path, unit=None):
        """Measure and read in a single round trip.

        Args:
            path (list[str]): Mx result path.
            unit (str | None): Mx unit name.

        Returns:
            float: The value Mx reports.
        """
        reply = self._ask("measure_read", result=list(path), unit=unit)
        return float(reply["value"])

    def probe_point(self, x_mm, y_mm, y_down=False, measure=True, fresh=None):
        """Measure, then read the height of the one pixel at (x, y).

        The agent exports the surface on the Mx PC and picks the row out of
        it; only the number crosses the cable.

        Args:
            x_mm (float): Wanted x, in the millimetres Mx displays.
            y_mm (float): Wanted y, same frame.
            y_down (bool): True when the plot's row 0 is at the largest y.
            measure (bool): Acquire first; False re-reads the surface Mx
                already holds.
            fresh (bool | None): Whether the surface is new since the last
                probe; defaults to `measure`. The agent uses it to tell an
                unchanged export from a dead plot.

        Returns:
            dict: The agent's reply -- `value` in nanometres, plus the pixel
                and coordinates it actually resolved to, for checking against
                Mx's own readout.
        """
        version = self._version_tuple(self.info.get("agent"))
        if version < self.POINT_AGENT_VERSION:
            wanted = ".".join(map(str, self.POINT_AGENT_VERSION))
            found = str(self.info.get("agent") or "unknown")
            raise AgentError(
                f"single-coordinate reads require mx_agent {wanted} or newer "
                f"(connected agent is {found}) -- replace mx_agent.py on the "
                "Mx PC and restart mx_agent.bat")
        return self._ask("probe_point", x=float(x_mm), y=float(y_mm),
                         y_down=bool(y_down), measure=bool(measure),
                         fresh=bool(measure if fresh is None else fresh))

    def probe_points(self, points, y_down=False, measure=True, fresh=None,
                     roi_size=3):
        """Acquire once and return 3x3-median heights for several points."""
        version = self._version_tuple(self.info.get("agent"))
        if version < self.BATCH_POINT_AGENT_VERSION:
            wanted = ".".join(map(str, self.BATCH_POINT_AGENT_VERSION))
            found = str(self.info.get("agent") or "unknown")
            raise AgentError(
                f"multi-coordinate reads require mx_agent {wanted} or newer "
                f"(connected agent is {found}) -- replace mx_agent.py on the "
                "Mx PC and restart mx_agent.bat")
        payload = []
        for index, point in enumerate(points):
            payload.append({
                "id": point.get("id", index),
                "x": float(point["x"]), "y": float(point["y"]),
                "y_down": bool(point.get("y_down", y_down)),
            })
        if not payload:
            raise ValueError("at least one point is required")
        return self._ask(
            "probe_points", points=payload, y_down=bool(y_down),
            measure=bool(measure),
            fresh=bool(measure if fresh is None else fresh),
            roi_size=int(roi_size))

    @staticmethod
    def _version_tuple(value):
        """Comparable numeric prefix of an agent version string."""
        out = []
        for part in str(value or "0").split("."):
            digits = "".join(ch for ch in part if ch.isdigit())
            if not digits:
                break
            out.append(int(digits))
        return tuple((out + [0, 0])[:2])

    def get_reference(self):
        """dict: The substrate Mx currently subtracts, and whether it is on.

        Read-only, and worth doing before a run: subtracting one mirror's
        substrate from another's measurement removes the wrong static figure
        and leaves a residual indistinguishable from a real aberration.
        """
        return self._ask("get_reference")

    def set_reference(self, settle_s=0.0, actuators=None, timeout=None):
        """Re-measure the substrate from the mirror's current state.

        Sent with every actuator at zero. The agent waits, acquires, saves on
        the Mx PC and points Mx at the new file as one order, so a dropped
        cable cannot strand Mx with the subtraction switched off.

        Args:
            settle_s (float): Seconds the agent waits before acquiring, for
                the mirror to finish relaxing after being driven to zero.
            actuators (int | None): Which mirror is on the bench. The Mx PC
                keeps one substrate file per mirror and picks by this, so the
                5-actuator surface can never be written over the 9's.
            timeout (float | None): Reply timeout; the default allows for the
                settle plus an acquisition.

        Returns:
            dict: The agent's reply, including the new substrate's path.
        """
        if timeout is None:
            timeout = self.timeout + float(settle_s)
        return self._ask("set_reference", timeout=timeout,
                         settle_s=float(settle_s),
                         actuators=None if actuators is None
                         else int(actuators))

    def version(self):
        """str: The Mx version the agent reported."""
        return str(self.info.get("mx_version", "unknown"))

    # Protocol
    def _ask(self, cmd, timeout=None, **payload):
        """Send one command and return its reply.

        Args:
            cmd (str): Command name.
            timeout (float | None): Seconds to wait; None uses the default.
            **payload: Command arguments.

        Returns:
            dict: The reply.

        Raises:
            AgentError: If the link died or the agent refused the command.
        """
        if self._sock is None:
            raise AgentError("the Mx agent is not connected")
        request = {"cmd": cmd, "id": int(time.monotonic() * 1000) % 1_000_000}
        request.update({k: v for k, v in payload.items() if v is not None})
        data = json.dumps(request).encode("utf-8") + b"\n"
        try:
            self._sock.settimeout(30.0)
            self._sock.sendall(data)
        except OSError as error:
            self.connected = False
            raise AgentError(f"link lost while sending: {error}") from error
        reply = self._read_reply(self.timeout if timeout is None else timeout)
        if not reply.get("ok"):
            raise AgentError(reply.get("error") or f"{cmd} failed on the Mx PC")
        return reply

    def _read_reply(self, timeout):
        """Read one JSON line back.

        Args:
            timeout (float): Seconds to wait for a whole line.

        Returns:
            dict: The decoded reply.

        Raises:
            AgentError: On timeout, close, or undecodable input.
        """
        deadline = time.monotonic() + timeout
        while True:
            cut = self._buf.find(b"\n")
            if cut >= 0:
                line = bytes(self._buf[:cut])
                del self._buf[:cut + 1]
                try:
                    reply = json.loads(line.decode("utf-8"))
                except ValueError as error:
                    self.connected = False
                    raise AgentError(f"agent sent non-JSON: {error}") from error
                if not isinstance(reply, dict):
                    self.connected = False
                    raise AgentError("agent sent a non-object reply")
                return reply
            left = deadline - time.monotonic()
            if left <= 0:
                raise AgentError(f"no reply from the Mx agent within "
                                 f"{timeout:g}s")
            try:
                self._sock.settimeout(left)
                chunk = self._sock.recv(65536)
            except socket.timeout:
                raise AgentError(f"no reply from the Mx agent within "
                                 f"{timeout:g}s") from None
            except OSError as error:
                self.connected = False
                raise AgentError(f"link lost: {error}") from error
            if not chunk:
                self.connected = False
                raise AgentError("the Mx agent closed the link")
            self._buf += chunk


def _accept(port, timeout):
    """Wait for the agent to dial in.

    Args:
        port (int): TCP port to listen on.
        timeout (float): Seconds to wait.

    Returns:
        socket.socket: The accepted connection.

    Raises:
        AgentError: If nothing connected in time, or the port is unusable.
    """
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(("0.0.0.0", int(port)))
        server.listen(1)
        server.settimeout(timeout)
        sock, _peer = server.accept()
    except socket.timeout:
        raise AgentError(
            f"no agent dialled in on port {port} within {timeout:g}s -- start "
            f"mx_agent.py on the Mx PC") from None
    except OSError as error:
        raise AgentError(f"cannot listen on port {port} -- {error}") from error
    finally:
        server.close()
    sock.settimeout(None)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 10_000, 3_000))  # Windows.
    except (AttributeError, OSError):
        pass  # Keepalive is a bonus, not a requirement.
    return sock


def agent_command(local_ip, port=DEFAULT_PORT):
    """The exact line to type on the Mx PC.

    Args:
        local_ip (str | None): This PC's address on the Mx cable.
        port (int): Port this app listens on.

    Returns:
        str: The command.
    """
    return (f"python mx_agent.py --host {local_ip or '<this PC>'} "
            f"--port {port}")
