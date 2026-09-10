# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for compact processed-surface DATX capture."""

from __future__ import annotations

import ipaddress
import socket
import tempfile
import types
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from dm_toolkit.zygo import mx_client
from dm_toolkit.zygo.mx_client import MxClient


class SurfaceDatxTests(unittest.TestCase):
    """Verify compact capture preserves one self-contained DATX file."""

    def test_simulation_copies_the_configured_sample(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sample = root / "sample.datx"
            destination = root / "point.datx"
            sample.write_bytes(b"processed surface")
            client = MxClient(sample_datx=sample, simulate=True)
            client.connect()

            result = client.save_surface_datx(destination)

            self.assertEqual(result, str(destination))
            self.assertEqual(destination.read_bytes(), sample.read_bytes())


class ProbeTests(unittest.TestCase):
    """A dead Mx must be named in one sentence, not waited out."""

    def test_a_listening_port_reports_no_fault(self) -> None:
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            self.assertIsNone(mx_client.probe("127.0.0.1", port))

    def test_a_dead_port_is_reported_with_the_address_that_failed(self) -> None:
        with socket.socket() as spare:  # Bind, then drop it: nothing listens.
            spare.bind(("127.0.0.1", 0))
            port = spare.getsockname()[1]

        fault = mx_client.probe("127.0.0.1", port, timeout=0.3)

        self.assertIsNotNone(fault)
        self.assertIn(f"127.0.0.1:{port}", fault)

    def test_a_refused_port_blames_the_service_not_the_cable(self) -> None:
        # Refused means the machine is up and answered -- so the fault is Mx
        # or its Remote Access service, never the cable or the firewall.
        with mock.patch.object(mx_client.socket, "create_connection",
                               side_effect=ConnectionRefusedError):
            fault = mx_client.probe("169.254.15.190")
        self.assertIn("Remote Access", fault)

    def test_a_dropped_port_blames_the_firewall(self) -> None:
        # Silence means the packets are being dropped, not answered.
        with mock.patch.object(mx_client.socket, "create_connection",
                               side_effect=socket.timeout):
            fault = mx_client.probe("169.254.15.190")
        self.assertIn("firewall", fault)

    def test_source_ip_names_the_local_end_of_the_route(self) -> None:
        self.assertEqual(mx_client.source_ip("127.0.0.1"), "127.0.0.1")


class RouteTests(unittest.TestCase):
    """Which adapter carries Mx decides whether the bench links at all."""

    NETS = [
        ("192.168.77.1", ipaddress.ip_network("192.168.77.0/24")),  # Pi cable
        ("169.254.126.63", ipaddress.ip_network("169.254.0.0/16")),  # APIPA
        ("169.254.15.150", ipaddress.ip_network("169.254.15.0/24")),  # Zygo
    ]

    def test_the_static_subnet_beats_an_autoconfigured_one(self) -> None:
        # Several adapters carry a 169.254/16; only the /24 on the Mx cable
        # is the real route, and Windows picks it by longest prefix.
        with mock.patch.object(mx_client, "local_networks",
                               return_value=self.NETS):
            self.assertEqual(mx_client.source_ip("169.254.15.190"),
                             "169.254.15.150")

    def test_the_pi_subnet_is_still_reported_for_its_own_host(self) -> None:
        with mock.patch.object(mx_client, "local_networks",
                               return_value=self.NETS):
            self.assertEqual(mx_client.source_ip("192.168.77.2"),
                             "192.168.77.1")

    def test_an_off_subnet_address_has_no_local_end(self) -> None:
        with mock.patch.object(mx_client, "local_networks",
                               return_value=self.NETS):
            self.assertIsNone(mx_client.source_ip("8.8.8.8"))

    def test_a_hostname_is_never_resolved(self) -> None:
        # source_ip runs on every keystroke in the dialog; DNS would stall it.
        self.assertIsNone(mx_client.source_ip("zygo-pc"))

    def test_routes_are_unknown_without_an_interface_table(self) -> None:
        with mock.patch.object(mx_client, "local_networks", return_value=[]):
            self.assertFalse(mx_client.routes_known())
            self.assertIsNone(mx_client.source_ip("169.254.15.190"))


class TransportTests(unittest.TestCase):
    """Mx sits on a private address; a system proxy must never see it."""

    def test_the_zygo_package_gets_a_proxy_free_bounded_opener(self) -> None:
        fake = types.SimpleNamespace(_request=urllib.request)
        transport = mx_client.install_direct_transport(fake, timeout=12.0)

        self.assertIs(fake._request, transport)
        self.assertEqual(transport.timeout, 12.0)
        self.assertIs(transport.Request, urllib.request.Request)
        # Stock urllib must be left alone for the rest of the process.
        self.assertIsNot(urllib.request.urlopen, transport.urlopen)

    def test_installing_twice_reuses_the_opener(self) -> None:
        fake = types.SimpleNamespace(_request=urllib.request)
        first = mx_client.install_direct_transport(fake, timeout=5.0)
        second = mx_client.install_direct_transport(fake, timeout=9.0)

        self.assertIs(first, second)
        self.assertEqual(second.timeout, 9.0)

    def test_a_request_without_a_timeout_gets_the_bounded_one(self) -> None:
        transport = mx_client._DirectTransport(timeout=7.0)
        seen = {}
        transport._opener = types.SimpleNamespace(
            open=lambda request, **kw: seen.update(kw))

        transport.urlopen("req")

        self.assertEqual(seen["timeout"], 7.0)


if __name__ == "__main__":
    unittest.main()
