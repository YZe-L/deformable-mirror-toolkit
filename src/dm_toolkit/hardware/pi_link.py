# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.7, 2026-07-22

"""Provide the lock-step link between the Raspberry Pi and Windows app.

Windows acts as the server, and the Raspberry Pi acts as the client.
"""

import json
import socket
import struct
import threading
import time
import queue

PROTO = 1
DISCOVERY_PORT = 65433  # UDP beacon port (Windows broadcasts here)
DISCOVERY_MAGIC = "dm_pi_link"

# Message types
T_HELLO = "hello"
T_HELLO_OK = "hello_ok"
T_AT = "at"  # Pi -> Win: moved to a point  {bit, index, dir, peak}
T_GOT = "got"  # Win -> Pi: 'at' received (sent at once, pre-settle) {index}
T_RECORDED = "recorded"  # Win -> Pi: point recorded     {bit, index}
T_GAP = "gap"  # Pi -> Win: entering inter-range rest now  {secs, rng}
# Pi -> Win: just applied a step, start timing  {trial, bit}
T_TRIGGER = "trigger"
T_RESULT = "result"  # Win -> Pi: measured setup time  {trial, setup_s, amp_nm}
T_DONE = "done"  # Pi -> Win: sequence finished.
T_PLAN = "plan"  # Win -> Pi: full sweep bit list to replay {bits, rests}
T_GOTO = "goto"  # Win -> Pi: set DM to a bit now  {bit, seq}
T_SETTLED = "settled"  # Pi -> Win: bit applied + settled  {bit, seq, set_ms}
T_ABORT = "abort"  # Either -> other: stop now      {reason}
T_PING = "ping"
T_PONG = "pong"


class LinkError(Exception):
    pass


class LinkClosed(LinkError):
    pass


class Link:
    """A live connection. Call start() after the hello handshake."""

    def __init__(self, sock, role, name="", hb_interval=2.0, hb_timeout=30.0):
        """Initialize the Link.

        Args:
            sock: Connected network socket.
            role: Communication endpoint role.
            name: Display or identifier name.
            hb_interval: Heartbeat transmission interval, in seconds.
            hb_timeout: Heartbeat timeout, in seconds.
        """
        self.sock = sock
        self.role = role
        self.name = name
        self.peer = None
        self.peer_addr = None  # (ip, port) of the connected peer
        self.hb_interval = hb_interval
        self.hb_timeout = hb_timeout
        self._rx = queue.Queue()  # App messages for the consumer.
        self._tx_lock = threading.Lock()
        self._alive = threading.Event()
        self._alive.set()
        self._last_rx = time.monotonic()
        self._close_reason = None
        self._threads = []

    # Lifecycle
    def start(self):
        self.sock.settimeout(1.0)
        for fn in (self._reader, self._heartbeat):
            t = threading.Thread(target=fn, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    @property
    def alive(self):
        return self._alive.is_set()

    @property
    def close_reason(self):
        return self._close_reason

    def close(self, reason=None, notify=True):
        """Close the communication link.

        Args:
            reason: Reason sent when closing the link.
            notify: Callback used to publish a user-facing notification.
        """
        if not self._alive.is_set():
            return
        if notify:
            try:
                self._send_raw({"t": T_ABORT, "reason": reason or "closed"})
            except OSError:
                pass
        self._shutdown(reason or "closed")

    def _shutdown(self, reason):
        if not self._alive.is_set():
            return
        self._close_reason = reason
        self._alive.clear()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
        self._rx.put(None)  # Wake any blocked get()

    # Sending
    def send(self, t, **fields):
        if not self._alive.is_set():
            raise LinkClosed(self._close_reason or "link closed")
        fields["t"] = t
        self._send_raw(fields)

    def _send_raw(self, obj):
        line = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
        with self._tx_lock:
            self.sock.sendall(line)

    # receiving (app messages only; ping/pong handled internally)
    def get(self, timeout=None):
        """Next app message dict, or None on timeout.

        Raises LinkClosed when the connection is gone.
        """
        try:
            item = self._rx.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is None:
            raise LinkClosed(self._close_reason or "link closed")
        return item

    # Background threads
    def _reader(self):
        buf = b""
        while self._alive.is_set():
            try:
                data = self.sock.recv(4096)
            except socket.timeout:
                if time.monotonic() - self._last_rx > self.hb_timeout:
                    self._shutdown("peer timeout (no heartbeat)")
                    return
                continue
            except OSError:
                self._shutdown("socket error")
                return
            if not data:
                self._shutdown("peer closed")
                return
            self._last_rx = time.monotonic()
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                self._dispatch(line)

    def _dispatch(self, line):
        try:
            msg = json.loads(line.decode("utf-8", "ignore"))
        except (ValueError, TypeError):
            return
        t = msg.get("t")
        if t == T_PING:
            try:
                self._send_raw({"t": T_PONG})
            except OSError:
                pass
            return
        if t == T_PONG:
            return  # Liveness already refreshed above.
        if t == T_ABORT:
            self._shutdown("peer abort: " + str(msg.get("reason", "")))
            return
        self._rx.put(msg)

    def _heartbeat(self):
        while self._alive.is_set():
            time.sleep(self.hb_interval)
            if not self._alive.is_set():
                return
            try:
                self._send_raw({"t": T_PING})
            except OSError:
                self._shutdown("heartbeat send failed")
                return


# Handshake helpers
def _do_hello(sock, role, name, timeout=10.0):
    """Exchange hello/hello_ok, returning the peer's announced name.

    Args:
        sock: Connected network socket.
        role: Communication endpoint role.
        name: Display or identifier name.
        timeout: Maximum wait time, in seconds.
    """
    sock.settimeout(timeout)
    rf = sock.makefile("rb")
    if role == "client":
        _send_line(sock, {"t": T_HELLO, "proto": PROTO, "role": role,
                          "name": name})
        peer = _expect(rf, T_HELLO_OK)
    else:
        peer = _expect(rf, T_HELLO)
        if peer.get("proto") != PROTO:
            _send_line(sock, {"t": T_ABORT, "reason": "proto mismatch"})
            raise LinkError("protocol mismatch: peer proto=%s" % peer.get("proto"))
        _send_line(sock, {"t": T_HELLO_OK, "proto": PROTO, "role": role,
                          "name": name})
    rf.close()
    if peer.get("proto") != PROTO:
        raise LinkError("protocol mismatch")
    return peer.get("name", "")


def _send_line(sock, obj):
    sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode())


def _expect(rf, t):
    line = rf.readline()
    if not line:
        raise LinkClosed("peer closed during handshake")
    msg = json.loads(line.decode("utf-8", "ignore"))
    if msg.get("t") != t:
        raise LinkError("handshake: expected %s, got %s" % (t, msg.get("t")))
    return msg


# client (Pi)
def connect(host, port=65432, name="pi", retries=0, delay=2.0,
            hb_interval=2.0, hb_timeout=30.0):
    """Connect to the Windows server and return a started Link.

    retries<0 means retry forever (survives Windows app restart).

    Args:
        host: Remote host name or address.
        port: TCP port number.
        name: Display or identifier name.
        retries: Sequence of retrie values.
        delay: Delay between operations, in seconds.
        hb_interval: Heartbeat transmission interval, in seconds.
        hb_timeout: Heartbeat timeout, in seconds.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            sock = socket.create_connection((host, port), timeout=10.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            peer = _do_hello(sock, "client", name)
            link = Link(sock, "client", name, hb_interval, hb_timeout)
            link.peer = peer
            return link.start()
        except (OSError, LinkError) as e:
            if retries >= 0 and attempt > retries:
                raise LinkError("cannot connect to %s:%s (%s)" % (host, port, e))
            time.sleep(delay)


# server (Windows)
class Listener:
    """Accept connections one at a time. Reusable across runs."""

    def __init__(self, port=65432, name="win", hb_interval=2.0, hb_timeout=30.0):
        """Initialize the Listener.

        Args:
            port: TCP port number.
            name: Display or identifier name.
            hb_interval: Heartbeat transmission interval, in seconds.
            hb_timeout: Heartbeat timeout, in seconds.
        """
        self.port = port
        self.name = name
        self.hb_interval = hb_interval
        self.hb_timeout = hb_timeout
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("0.0.0.0", port))
        self._srv.listen(1)
        self._srv.settimeout(1.0)

    def accept(self, stop=None):
        """Block until a client connects (or stop() returns True).

        Returns a started Link, or None if stopped.
        """
        while True:
            if stop is not None and stop():
                return None
            try:
                sock, addr = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return None
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                peer = _do_hello(sock, "server", self.name)
            except (OSError, LinkError):
                try:
                    sock.close()
                except OSError:
                    pass
                continue
            link = Link(sock, "server", self.name,
                        self.hb_interval, self.hb_timeout)
            link.peer = peer
            link.peer_addr = addr
            return link.start()

    def close(self):
        try:
            self._srv.close()
        except OSError:
            pass


# Local IP auto-detection (both ends; no `ipconfig`/`hostname -I` needed)
def local_ip():
    """Return local IP.

    Best-guess primary LAN IPv4 of THIS machine, without sending traffic
    (UDP connect just picks the outgoing interface). Falls back to hostname.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # No packet sent for a UDP connect.
        ip = s.getsockname()[0]
    except OSError:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            ip = "127.0.0.1"
    finally:
        s.close()
    return ip


# Our fixed direct-Ethernet subnet (PC 192.168.77.1, Pi 192.168.77.2); a PC
# address in it means a wired link is up, independent of Wi-Fi.
WIRED_PREFIX = "192.168.77."


def all_ipv4():
    """Return all non-loopback IPv4 addresses.

    Every non-loopback IPv4 on THIS machine (best effort, stdlib only),
    primary/default-route address first. local_ip() alone only reports the
    default-route interface (Wi-Fi), which hides a direct-Ethernet address --
    this lets the connect dialog show BOTH the Wi-Fi and the wired IP.
    """
    primary = local_ip()
    ips = [primary] if not primary.startswith("127.") else []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return ips


def wired_ip():
    """Return wired IP.

    This PC's direct-Ethernet IP (192.168.77.x) if the wired link is up,
    else None. Used to advertise the cable path in the connect dialog.
    """
    for ip in all_ipv4():
        if ip.startswith(WIRED_PREFIX):
            return ip
    return None


def _bcast_targets():
    """Return candidate broadcast addresses.

    Broadcast addresses to try: global plus the /24 and /16 directed
    broadcasts derived from our own IP (some APs drop one but pass another).
    """
    targets = ["255.255.255.255"]
    p = local_ip().split(".")
    if len(p) == 4 and p[0] != "127":
        targets.append(".".join(p[:3] + ["255"]))  # x.y.z.255  (/24)
        targets.append(".".join(p[:2] + ["255", "255"]))  # x.y.255.255 (/16)
    return list(dict.fromkeys(targets))


# UDP discovery (so a changed Windows IP does not break the Pi)
class Beacon:
    """Windows side: broadcast our host/port so the Pi can find us."""

    def __init__(self, tcp_port=65432, interval=1.5):
        self.tcp_port = tcp_port
        self.interval = interval
        self._stop = threading.Event()
        self._t = None

    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        payload = json.dumps({"magic": DISCOVERY_MAGIC, "proto": PROTO,
                              "port": self.tcp_port, "host": local_ip()}).encode()
        while not self._stop.is_set():
            for tgt in _bcast_targets():
                try:
                    s.sendto(payload, (tgt, DISCOVERY_PORT))
                except OSError:
                    pass
            self._stop.wait(self.interval)
        s.close()

    def stop(self):
        self._stop.set()


def discover(timeout=10.0):
    """Pi side: listen for the Windows beacon. Returns (host, port) or None."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("", DISCOVERY_PORT))
    except OSError:
        s.close()
        return None
    s.settimeout(timeout)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            try:
                data, addr = s.recvfrom(1024)
            except socket.timeout:
                return None
            try:
                msg = json.loads(data.decode("utf-8", "ignore"))
            except ValueError:
                continue
            if msg.get("magic") == DISCOVERY_MAGIC:
                return addr[0], int(msg.get("port", 65432))
    finally:
        s.close()
    return None
