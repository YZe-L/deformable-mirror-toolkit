# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-08-06

"""PC-side end of the impact-matrix agent link.

Same port and newline-JSON protocol as :mod:`mx_agent_link`; this agent
answers with the name of a file it has written on the Mx PC. The measured
surfaces never cross the link.
"""

from . import mx_agent_link
from .mx_agent_link import AgentError  # noqa: F401 -- re-exported for callers.


class ImAgentClient(mx_agent_link.MxAgentClient):
    """Waits for the impact-matrix agent, then asks it to capture points."""

    KIND = "im"

    def connect(self):
        """Listen, wait for the agent, and report what it can do right now.

        Returns:
            str: What happened, for a status line.
        """
        peer, fault = self._open()
        if fault is not None:
            return fault
        mx_state = str(self.info.get("mx", "unknown"))
        if "ready" not in mx_state:
            # Keep the link: the operator can open the .appx and press again
            # without restarting the agent.
            return f"IM agent on {peer} is up, but Mx is not -- {mx_state}"
        sim = " (agent SIMULATING)" if self.info.get("simulate") else ""
        if not self.info.get("plot_ok"):
            # Every processed save would fail, and the raw ones would still
            # succeed -- a whole calibration of files that look fine and carry
            # the alignment tilt. Worth saying loudly, at connect time.
            return (f"IM agent on {peer} ready, but the 3D Surface plot is NOT "
                    f"open ({self.info.get('plot')}) -- open it in Mx, then "
                    f"press Connect again{sim}")
        return (f"IM agent on {peer} ready, Mx {self.info.get('mx_version')}, "
                f"saves from {self.info.get('plot')}{sim}")

    @property
    def plot_ok(self):
        """bool: Whether the agent found the plot that yields processed data."""
        return bool(self.info.get("plot_ok"))

    def session(self, name, channels, bias, note=""):
        """Open a folder on the Mx PC for the points that follow.

        Args:
            name (str): Folder name; blank lets the agent use a timestamp.
            channels (list[int]): Channels this calibration covers.
            bias (int): Bit every channel rests at.
            note (str): Free text kept in the manifest.

        Returns:
            dict: The agent's reply, with ``dir`` and ``name``.
        """
        return self._ask("session", name=str(name or ""),
                         channels=[int(c) for c in channels],
                         bias=int(bias), note=str(note or ""))

    def point(self, bits, bias, actual=None, note="", role=""):
        """Measure one point and have the agent save its two files.

        Args:
            bits (dict): {channel: commanded bit} for EVERY channel -- the
                agent names the file from this and works the role out of it.
            bias (int): Bit the role is judged against.
            actual (dict | None): {channel: bit} actually sent after
                compensation, when it differs. Recorded, never used as the axis.
            note (str): Free text for this point's manifest entry.
            role (str): What the caller thinks the role is. A hint only; the
                agent recomputes it and says so if the two disagree.

        Returns:
            dict: The agent's reply, with ``stem``, ``role``, ``files`` and
                ``sizes``.
        """
        payload = {"bits": {str(int(c)): int(b) for c, b in bits.items()},
                   "bias": int(bias), "note": str(note or ""),
                   "role": str(role or "")}
        if actual:
            payload["actual"] = {str(int(c)): int(b) for c, b in actual.items()}
        return self._ask("im_point", **payload)
