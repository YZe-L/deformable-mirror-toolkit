# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.9, 2026-08-15

"""Qt wrapper around pi_link for the PC application (Windows = server)."""

import threading
import time

from PyQt5 import QtCore, QtWidgets

from . import pi_link

# Every DM piezo channel that can be wired: 1..5 is the five-element mirror,
# 6..14 the nine-element one. A constant, because this is the last thing that
# runs before the link goes down and must not depend on any page state.
KNOWN_CHANNELS = tuple(range(1, 15))

# Sequence number used only by the shutdown zero, far above anything a
# measurement uses (the loop starts near 0, the bench at 900000).
_ZERO_SEQ = 990000


class PiLinkController(QtCore.QObject):
    # All emitted on the GUI thread (queued from the worker thread)
    connected = QtCore.pyqtSignal(str)  # Peer name
    disconnected = QtCore.pyqtSignal(str)  # Reason
    at = QtCore.pyqtSignal(dict)  # Pi reached a point.
    gap = QtCore.pyqtSignal(dict)  # Pi entered an inter-range rest.
    triggered = QtCore.pyqtSignal(dict)  # Pi fired a step (setup-time test)
    done = QtCore.pyqtSignal(dict)  # Pi finished the sequence.
    settled = QtCore.pyqtSignal(dict)  # Pi applied a closed-loop setpoint.
    status = QtCore.pyqtSignal(str)

    def __init__(self, port=65432, name="dm_toolkit", parent=None):
        """Initialize the PiLinkController.

        Args:
            port: TCP port number.
            name: Display or identifier name.
            parent: Parent Qt object.
        """
        super().__init__(parent)
        self._port = port
        self._name = name
        self._listener = None
        self._beacon = None
        self._link = None
        self._link_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        # Last T_SETTLED recorded WHERE IT ARRIVES (the reader thread), not
        # where it is handled. A GUI-thread timeout can then ask "did the Pi
        # actually ack?" instead of blaming the Pi for our own event-loop lag.
        self._ack_lock = threading.Lock()
        self._last_ack = (-1, 0.0)  # (seq, perf_counter at arrival)
        self._at_lock = threading.Lock()
        self._last_at = (-1, -1, 0.0)  # (index, bit, perf_counter) of T_AT
        self._peer_addr = None  # (ip, port) of the live peer
        # Last multi-channel command accepted by the link. The DM has no
        # read-back, so this is the only record of the real state.
        self._chan_lock = threading.Lock()
        self._last_channels = {}

    @property
    def peer_addr(self):
        """(ip, port) of the connected Pi, or None.

        Lets the UI show WHICH link is live -- a 192.168.77.x ip means the
        direct Ethernet cable, anything else the Wi-Fi/LAN path.
        """
        return self._peer_addr

    # lifecycle (call from GUI thread)
    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def zero_all(self, wait_ms=500):
        """Drive every channel that could be energised to 0 and confirm it.

        The union of `_last_channels` and `KNOWN_CHANNELS` covers every
        channel any caller drove. Unlike a mid-session park this one waits
        for the Pi's ack, because an unconfirmed zero at shutdown means a
        piezo left under drive.

        Args:
            wait_ms: How long to wait for the Pi to acknowledge.

        Returns:
            bool: True when the Pi acknowledged the zero.
        """
        if not self.linked:
            return False
        with self._chan_lock:
            channels = set(self._last_channels)
        channels |= set(KNOWN_CHANNELS)
        zeros = {int(c): 0 for c in sorted(channels)}
        if not self._send(pi_link.T_GOTO, channels={str(c): 0 for c in zeros},
                          seq=_ZERO_SEQ):
            self.status.emit("could not send the shutdown zero -- CHECK THE "
                             "MIRROR, channels may still be energised")
            return False
        with self._chan_lock:
            self._last_channels.update(zeros)
        deadline = time.monotonic() + max(0.0, wait_ms / 1000.0)
        while time.monotonic() < deadline:
            if self.last_ack()[0] == _ZERO_SEQ:
                self.status.emit("all %d channels zeroed and acknowledged"
                                 % len(zeros))
                return True
            time.sleep(0.01)
        # The setpoint is on the wire and TCP will still deliver it; only the
        # confirmation is missing, so say exactly that rather than nothing.
        self.status.emit("shutdown zero sent for %d channels but NOT "
                         "acknowledged within %d ms -- verify the mirror is "
                         "relaxed" % (len(zeros), wait_ms))
        return False

    def stop(self, reason="windows closed"):
        # Before anything is torn down: once _stop is set the reader thread
        # goes away and no ack can arrive, and once the socket closes nothing
        # can be sent at all.
        try:
            self.zero_all()
        except Exception as e:  # noqa: BLE001 (shutdown must always proceed)
            self.status.emit("shutdown zero failed (%s) -- CHECK THE MIRROR"
                             % e)
        self._stop.set()
        with self._link_lock:
            link = self._link
        if link is not None:
            link.close(reason=reason)
        if self._listener is not None:
            self._listener.close()
        if self._beacon is not None:
            self._beacon.stop()
        self._thread = None

    @property
    def linked(self):
        with self._link_lock:
            return self._link is not None and self._link.alive

    def last_ack(self):
        """Return the most recent settled-command acknowledgement.

        (seq, arrival time) of the most recent T_SETTLED, as seen by the
        reader thread. Callers use it to tell a REAL link failure from their
        own GUI-thread stall before declaring the Pi dead.
        """
        with self._ack_lock:
            return self._last_ack

    def last_at(self):
        """Return the most recent sweep-position message.

        (index, bit, arrival time) of the most recent T_AT, stamped in the
        reader thread. Lets a worker THREAD drive the pi_sweep_link 'pc'
        lock-step (wait for the Pi to reach a level, hold+record, then ack via
        send_recorded) without touching the GUI thread.
        """
        with self._at_lock:
            return self._last_at

    def last_channels(self):
        """Return the {channel: bit} state last commanded through this link.

        Accumulated across senders, so a tab can label a measurement with the
        mirror shape another tab set. Empty until the first multi-channel
        command; channels driven by the single-channel `send_goto` are absent
        because that message carries no channel number.
        """
        with self._chan_lock:
            return dict(self._last_channels)

    # sending (call from GUI thread)
    def send_recorded(self, bit, index):
        self._send(pi_link.T_RECORDED, bit=int(bit), index=int(index))

    def send_result(self, trial, setup_s, amp_nm):
        self._send(pi_link.T_RESULT, trial=int(trial),
                   setup_s=float(setup_s), amp_nm=float(amp_nm))

    def send_plan(self, bits, rests=None):
        """Send plan.

        Ship the whole sweep the Pi should replay: a list of integer bits
        (already hysteresis-compensated where the PC decided to) plus an
        optional matching list of pre-point rest seconds. The Pi writes these
        bits verbatim -- it computes nothing. Sent once, before the reference
        handshake, so the Pi drives the existing lock-step over the PC's list.

        Args:
            bits: PWM command bits.
            rests: Rest durations paired with the command sequence.
        """
        payload = [int(b) for b in bits]
        rest = [float(r) for r in (rests or [0.0] * len(payload))]
        return self._send(pi_link.T_PLAN, bits=payload, rests=rest)

    def send_goto(self, bit, seq):
        return self._send(pi_link.T_GOTO, bit=int(bit), seq=int(seq))

    def send_goto_channels(self, channels, seq):
        """Set several DM channels at once.

        Set several DM channels at once: channels={channel:int -> bit:int}.
        Carried in the same T_GOTO message (JSON keys must be strings). The Pi
        applies each pwm.set_pwm(channel, 0, bit) and replies T_SETTLED{seq}.

        Args:
            channels: Actuator channel identifiers.
            seq: Command sequence number or values.
        """
        payload = {str(int(c)): int(b) for c, b in channels.items()}
        ok = self._send(pi_link.T_GOTO, channels=payload, seq=int(seq))
        if ok:
            with self._chan_lock:
                self._last_channels.update({int(c): int(b)
                                            for c, b in channels.items()})
        return ok

    def send_abort(self, reason="windows abort"):
        with self._link_lock:
            link = self._link
        if link is not None:
            link.close(reason=reason)

    def _send(self, t, **kw):
        with self._link_lock:
            link = self._link
        if link is None or not link.alive:
            return False
        try:
            link.send(t, **kw)
            return True
        except pi_link.LinkError:
            return False

    # Worker thread
    def _run(self):
        try:
            self._listener = pi_link.Listener(port=self._port, name=self._name)
        except OSError as e:
            self.status.emit("cannot listen on %d: %s" % (self._port, e))
            return
        self._beacon = pi_link.Beacon(tcp_port=self._port).start()
        self.status.emit("listening on %d, waiting for Pi..." % self._port)
        while not self._stop.is_set():
            link = self._listener.accept(stop=self._stop.is_set)
            if link is None:
                break
            with self._link_lock:
                self._link = link
            self._peer_addr = getattr(link, "peer_addr", None)
            self.connected.emit(link.peer or "pi")
            self._pump(link)
            with self._link_lock:
                self._link = None
            self._peer_addr = None
            if not self._stop.is_set():
                self.disconnected.emit(link.close_reason or "closed")
                self.status.emit("Pi disconnected, listening again...")
        self._beacon.stop()
        self._listener.close()

    def _pump(self, link):
        while not self._stop.is_set():
            try:
                msg = link.get(timeout=0.5)
            except pi_link.LinkClosed:
                return
            if msg is None:
                continue
            t = msg.get("t")
            if t == pi_link.T_AT:
                # Ack receipt immediately (here, in the reader thread, before
                # the GUI hop / settle wait) so the Pi can time pure link
                # latency without the settle wait folded in.
                try:
                    link.send(pi_link.T_GOT, index=msg.get("index"))
                except pi_link.LinkError:
                    pass
                with self._at_lock:
                    self._last_at = (int(msg.get("index", -1)),
                                     int(msg.get("bit", -1)), time.perf_counter())
                self.at.emit(msg)
            elif t == pi_link.T_GAP:
                self.gap.emit(msg)
            elif t == pi_link.T_TRIGGER:
                self.triggered.emit(msg)
            elif t == pi_link.T_DONE:
                self.done.emit(msg)
            elif t == pi_link.T_SETTLED:
                # Stamp it here, before the GUI hop, so a stalled event loop
                # cannot make a healthy link look dead.
                with self._ack_lock:
                    self._last_ack = (int(msg.get("seq", -1)),
                                      time.perf_counter())
                self.settled.emit(msg)


class PiConnectDialog(QtWidgets.QDialog):
    """Start listening and confirm the Pi is linked before a remote run."""

    def __init__(self, controller, parent=None):
        """Initialize the PiConnectDialog.

        Args:
            controller: Controller object wrapped by the Qt adapter.
            parent: Parent Qt object.
        """
        super().__init__(parent)
        self.ctl = controller
        self.setWindowTitle("Connect to Raspberry Pi")
        self.setMinimumWidth(360)
        v = QtWidgets.QVBoxLayout(self)
        my_ip = pi_link.local_ip()
        wired = pi_link.wired_ip()
        port = self.ctl._port
        # Wireless/LAN path (unchanged): auto-discovery, or -H the Wi-Fi IP.
        lines = ["Waiting for the Raspberry Pi to connect.\n",
                 f"This PC (Wi-Fi / LAN):  {my_ip}   (port {port})",
                 "On the Pi, run:",
                 "    python3 pi_sweep_link.py ramp",
                 "If auto-discovery fails, point it here explicitly:",
                 f"    python3 pi_sweep_link.py ramp -H {my_ip}"]
        # Wired path (added): only shown when a direct-Ethernet IP is actually
        # up, so it never confuses a Wi-Fi-only setup.
        if wired:
            lines += [
                "",
                f"Direct Ethernet cable:  {wired}   (port {port})",
                "For the rock-steady wired link, on the Pi run:",
                f"    python3 pi_dm_sequence.py --wired      (uses {wired})",
                f"    python3 pi_sweep_link.py ramp -H {wired}   (other tools)"]
        else:
            lines += [
                "",
                "Direct Ethernet cable: not detected (no 192.168.77.x here). "
                "Plug the cable and set this PC's Ethernet to 192.168.77.1 "
                "to enable the wired link."]
        self.info = QtWidgets.QLabel("\n".join(lines))
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        v.addWidget(self.info)
        self.state = QtWidgets.QLabel("listening...")
        v.addWidget(self.state)
        btns = QtWidgets.QDialogButtonBox()
        self.use_btn = btns.addButton("Use connection",
                                      QtWidgets.QDialogButtonBox.AcceptRole)
        self.use_btn.setEnabled(False)
        btns.addButton(QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

        self.ctl.connected.connect(self._on_connected)
        self.ctl.disconnected.connect(self._on_disconnected)
        self.ctl.status.connect(self.state.setText)
        if self.ctl.linked:
            self._on_connected("pi")
        else:
            self.ctl.start()

    def _on_connected(self, peer):
        addr = self.ctl.peer_addr
        ip = addr[0] if addr else None
        if ip and ip.startswith(pi_link.WIRED_PREFIX):
            how = f" via DIRECT ETHERNET ({ip})"
        elif ip:
            how = f" via Wi-Fi / LAN ({ip})"
        else:
            how = ""
        self.state.setText("connected to '%s'%s -- ready." % (peer, how))
        self.state.setStyleSheet("color:#56d364;")
        self.use_btn.setEnabled(True)

    def _on_disconnected(self, reason):
        self.state.setText("disconnected: %s -- waiting again..." % reason)
        self.state.setStyleSheet("color:#ff6b6b;")
        self.use_btn.setEnabled(False)
