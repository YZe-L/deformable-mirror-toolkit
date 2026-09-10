# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-31

"""Remember the influence-matrix folder and settings between sessions.

Keyed by folder, because the pupil is a property of the measurement.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import im_store as IM_STORE
from ..config import portable_path, resolve_path

PATH = IM_STORE.MATRIX_DIRECTORY.parent / "impact_matrix_sessions.json"
KEEP = 10  # Folders remembered, most recent first.


def load() -> list:
    """Every remembered session, most recent first; empty when there is none.

    Folders come back as absolute paths on THIS machine. Measurement sets
    usually sit under the project's own output folder, so they are stored
    relative to the checkout and re-anchored here; a set kept on another disk
    is stored and returned absolute (see `config.resolve_path`).
    """
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = data.get("recent") if isinstance(data, dict) else None
    return [dict(e, folder=str(resolve_path(e["folder"])))
            for e in (entries or [])
            if isinstance(e, dict) and e.get("folder")]


def latest() -> dict | None:
    """The most recently used session, or None."""
    entries = load()
    return entries[0] if entries else None


def for_folder(folder) -> dict | None:
    """The remembered settings for one folder, or None.

    Matched on the resolved path, so a folder remembered under an old drive
    letter still answers for the same folder here.
    """
    want = Path(folder).resolve()
    for entry in load():
        if Path(entry["folder"]).resolve() == want:
            return entry
    return None


def remember(folder, settings) -> None:
    """Move `folder` to the head of the list, with these settings.

    Never raises: remembering is a convenience, and losing it must not cost a
    measurement.
    """
    want = Path(folder).resolve()
    entries = [e for e in load() if Path(e["folder"]).resolve() != want]
    entries.insert(0, dict(settings, folder=str(want),
                           saved=datetime.now(timezone.utc).isoformat()))
    # Written project-relative where it can be, so the list survives a move to
    # another drive; the in-memory copies above stay absolute.
    stored = [dict(e, folder=portable_path(e["folder"]))
              for e in entries[:KEEP]]
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps({"recent": stored}, indent=2),
                        encoding="utf-8")
    except OSError:
        pass
