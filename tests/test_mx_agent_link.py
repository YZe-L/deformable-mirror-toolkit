# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the Mx agent link a sweep is driven through."""

from __future__ import annotations

import json
import socket
import threading
import unittest

from dm_toolkit.zygo import mx_agent_link
from dm_toolkit.zygo.mx_agent_link import (AgentError,
                                                                 MxAgentClient)


class FakeAgent:
    """A stand-in for mx_agent.py on the Mx PC: dials in, answers, records."""

    def __init__(self, port, hello=None, fail=None):
        """Prepare an agent that will dial into `port`.

        Args:
            port (int): Port the client listens on.
            hello (dict | None): Extra fields for the hello reply.
            fail (str | None): Command to answer with an error.
        """
        self.port = port
        self.hello = hello or {}
        self.fail = fail
        self.seen = []
        self.sock = None
        self._thread = None

    def start(self):
        """Dial the client in the background."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Close the connection and join the thread."""
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self):
        """Connect, then answer every request until the link closes."""
        for _ in range(80):  # The client may not be listening yet.
            try:
                self.sock = socket.create_connection(("127.0.0.1", self.port),
                                                     0.25)
                break
            except OSError:
                continue
        if self.sock is None:
            return
        buf = bytearray()
        while True:
            try:
                chunk = self.sock.recv(65536)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                cut = buf.find(b"\n")
                line, buf = bytes(buf[:cut]), buf[cut + 1:]
                if not self._answer(json.loads(line.decode())):
                    return

    def _answer(self, request):
        """Reply to one request.

        Args:
            request (dict): What the client sent.

        Returns:
            bool: False when the link should close.
        """
        cmd = request.get("cmd")
        self.seen.append(request)
        if cmd == self.fail:
            reply = {"ok": False, "cmd": cmd, "error": "Mx said no"}
        elif cmd == "hello":
            reply = {"ok": True, "cmd": cmd, "mx": "ready",
                     "kind": "mx", "agent": "1.5", "simulate": False,
                     "mx_version": "7.3", "result": "Analysis/Surface/PV",
                     "unit": "Waves"}
            reply.update(self.hello)
        elif cmd in ("read", "measure_read"):
            reply = {"ok": True, "cmd": cmd, "value": 0.264,
                     "result": "/".join(request.get("result", [])),
                     "unit": request.get("unit")}
        elif cmd == "probe_point":
            reply = {"ok": True, "cmd": cmd, "value": 133.181,
                     "unit": "NanoMeters", "col": 512, "row": 480,
                     "x_actual": 22.68, "y_actual": 28.32,
                     "grid": [924, 924], "nm_per_unit": 1000.0,
                     "y_down": request.get("y_down"), "no_data": False}
        elif cmd == "probe_points":
            reply = {"ok": True, "cmd": cmd, "unit": "NanoMeters",
                     "surface_id": "batch123", "points": [
                         {"id": point.get("id", index), "value": 100.0 + index,
                          "center_value": 99.0 + index, "roi_valid": 9,
                          "no_data": False}
                         for index, point in enumerate(request["points"])]}
        else:
            reply = {"ok": True, "cmd": cmd}
        reply["id"] = request.get("id")
        try:
            self.sock.sendall(json.dumps(reply).encode() + b"\n")
        except OSError:
            return False
        return cmd != "bye"


def _free_port():
    """int: A port nothing is listening on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class ConnectTests(unittest.TestCase):
    """The Mx PC dials in; this app never dials it."""

    def test_a_connected_agent_reports_what_it_will_read(self) -> None:
        port = _free_port()
        agent = FakeAgent(port)
        client = MxAgentClient(port=port, accept_timeout=10.0)
        agent.start()
        try:
            message = client.connect()
            self.assertTrue(client.connected)
            self.assertIn("ready", message)
            self.assertIn("Analysis/Surface/PV", message)
        finally:
            client.close()
            agent.stop()

    def test_no_agent_is_a_plain_sentence_not_a_hang(self) -> None:
        client = MxAgentClient(port=_free_port(), accept_timeout=0.3)
        message = client.connect()
        self.assertFalse(client.connected)
        self.assertIn("no agent dialled in", message)

    def test_an_agent_without_an_open_app_is_not_called_ready(self) -> None:
        port = _free_port()
        agent = FakeAgent(port, hello={"mx": "NO app open -- open your .appx"})
        client = MxAgentClient(port=port, accept_timeout=10.0)
        agent.start()
        try:
            message = client.connect()
            self.assertIn("NO app open", message)
            # The link stays up so the operator can open the .appx and retry
            # without restarting the agent.
            self.assertTrue(client.connected)
        finally:
            client.close()
            agent.stop()


class MeasureTests(unittest.TestCase):
    """One sweep point: measure, then read the number Mx reports."""

    def setUp(self) -> None:
        self.port = _free_port()
        self.agent = FakeAgent(self.port)
        self.client = MxAgentClient(port=self.port, accept_timeout=10.0)
        self.agent.start()
        self.client.connect()

    def tearDown(self) -> None:
        self.client.close()
        self.agent.stop()

    def test_measure_then_read_is_what_the_sweep_sends(self) -> None:
        self.client.measure()
        value = self.client.get_result_number(["Analysis", "Surface", "PV"],
                                              "Waves")

        self.assertAlmostEqual(value, 0.264)
        sent = [msg["cmd"] for msg in self.agent.seen]
        self.assertEqual(sent, ["hello", "measure", "read"])
        read = self.agent.seen[-1]
        self.assertEqual(read["result"], ["Analysis", "Surface", "PV"])
        self.assertEqual(read["unit"], "Waves")

    def test_a_unitless_read_leaves_the_unit_to_the_agent(self) -> None:
        # Omitted, not null: the agent then uses the unit configured next to
        # the application it belongs to.
        self.client.get_result_number(["Analysis", "Surface", "PV"], None)
        self.assertNotIn("unit", self.agent.seen[-1])

    def test_measure_read_does_the_point_in_one_round_trip(self) -> None:
        value = self.client.measure_read(["Analysis", "Surface", "PV"],
                                         "Waves")
        self.assertAlmostEqual(value, 0.264)
        self.assertEqual(self.agent.seen[-1]["cmd"], "measure_read")

    def test_a_live_link_reports_no_fault(self) -> None:
        self.assertIsNone(self.client.check_alive())

    def test_probing_a_point_sends_the_coordinate_and_measures(self) -> None:
        reply = self.client.probe_point(22.68, 28.32)
        sent = self.agent.seen[-1]
        self.assertEqual(sent["cmd"], "probe_point")
        self.assertAlmostEqual(sent["x"], 22.68)
        self.assertAlmostEqual(sent["y"], 28.32)
        self.assertTrue(sent["measure"])  # A sweep point is a fresh acquisition.
        self.assertTrue(sent["fresh"])
        self.assertFalse(sent["y_down"])
        self.assertAlmostEqual(reply["value"], 133.181)

    def test_reading_a_point_again_does_not_re_acquire(self) -> None:
        # "Read without measuring" has to keep meaning that, or the connect
        # dialog's test button quietly moves the mirror's surface on.
        self.client.probe_point(1.0, 2.0, y_down=True, measure=False)
        sent = self.agent.seen[-1]
        self.assertFalse(sent["measure"])
        self.assertFalse(sent["fresh"])
        self.assertTrue(sent["y_down"])

    def test_a_separate_measure_can_mark_the_following_probe_fresh(self) -> None:
        self.client.probe_point(1.0, 2.0, measure=False, fresh=True)

        sent = self.agent.seen[-1]
        self.assertFalse(sent["measure"])
        self.assertTrue(sent["fresh"])

    def test_an_old_agent_is_refused_for_coordinate_reads(self) -> None:
        self.client.info["agent"] = "1.3"

        with self.assertRaises(AgentError) as caught:
            self.client.probe_point(1.0, 2.0)

        self.assertIn("replace mx_agent.py", str(caught.exception))

    def test_batch_probe_sends_all_coordinates_in_one_request(self) -> None:
        reply = self.client.probe_points([
            {"id": 6, "x": 1.25, "y": 2.5, "y_down": True},
            {"id": 7, "x": 3.0, "y": 4.0},
        ])
        sent = self.agent.seen[-1]
        self.assertEqual(sent["cmd"], "probe_points")
        self.assertEqual(len(sent["points"]), 2)
        self.assertEqual(sent["roi_size"], 3)
        self.assertTrue(sent["measure"])
        self.assertEqual([point["id"] for point in reply["points"]], [6, 7])

    def test_agent_1_4_is_refused_for_batch_but_keeps_single_point(self) -> None:
        self.client.info["agent"] = "1.4"
        self.client.probe_point(1.0, 2.0)
        with self.assertRaises(AgentError) as caught:
            self.client.probe_points([{"x": 1.0, "y": 2.0}])
        self.assertIn("1.5", str(caught.exception))


class FailureTests(unittest.TestCase):
    """A failed point must halt the sweep, not poison the next row."""

    def test_an_mx_error_is_raised_with_what_the_agent_said(self) -> None:
        port = _free_port()
        agent = FakeAgent(port, fail="measure")
        client = MxAgentClient(port=port, accept_timeout=10.0)
        agent.start()
        client.connect()
        try:
            with self.assertRaises(AgentError) as caught:
                client.measure()
            self.assertIn("Mx said no", str(caught.exception))
            # The link survives a refused command: only that point is lost.
            self.assertIsNone(client.check_alive())
        finally:
            client.close()
            agent.stop()

    def test_a_dropped_agent_is_noticed_by_check_alive(self) -> None:
        port = _free_port()
        agent = FakeAgent(port)
        client = MxAgentClient(port=port, accept_timeout=10.0)
        agent.start()
        client.connect()
        agent.stop()
        try:
            with self.assertRaises(AgentError):
                client.measure()
            self.assertIsNotNone(client.check_alive())
        finally:
            client.close()

    def test_commands_before_connecting_do_not_crash(self) -> None:
        client = MxAgentClient(port=_free_port())
        with self.assertRaises(AgentError):
            client.measure()
        self.assertEqual(client.check_alive(),
                         "the Mx agent is not connected")


class CommandTextTests(unittest.TestCase):
    """The dialog tells the operator exactly what to type on the Mx PC."""

    def test_the_command_carries_this_pc_and_the_port(self) -> None:
        text = mx_agent_link.agent_command("169.254.15.150", 65433)
        self.assertIn("--host 169.254.15.150", text)
        self.assertIn("--port 65433", text)

    def test_an_unknown_address_stays_a_visible_placeholder(self) -> None:
        self.assertIn("<this PC>", mx_agent_link.agent_command(None))


if __name__ == "__main__":
    unittest.main()
