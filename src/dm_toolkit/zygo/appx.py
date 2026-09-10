# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-06

"""Unpack a Zygo Mx application (.appx) into a readable settings CSV."""

import csv
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

CSV_HEADER = ["Path", "Name", "Label", "Type", "Property", "Value"]


def _localname(tag):
    """Strip any XML namespace prefix ('{ns}Tag' -> 'Tag')."""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else tag


def _find_app_entry(zf):
    """The settings XML entry: an `Apps/*.app` (fall back to any *.app)."""
    apps = [n for n in zf.namelist() if n.lower().endswith(".app")]
    if not apps:
        raise ValueError("no .app settings entry inside the .appx")
    apps.sort(key=lambda n: (not n.startswith("Apps/"), len(n)))
    return apps[0]


def settings_rows(appx_path):
    """Flatten application settings into rows.

    Flatten the .app XML to rows [Path, Name, Label, Type, Property, Value].

    One row per <Param> under each <Node>; Path is the slash-joined chain of
    ancestor node names/labels, so a control is easy to locate.
    """
    with zipfile.ZipFile(appx_path) as zf:
        root = ET.fromstring(zf.read(_find_app_entry(zf)))

    parent = {child: p for p in root.iter() for child in p}

    def node_id(el):
        a = el.attrib
        return a.get("name") or a.get("label") or _localname(el.tag)

    def path_of(el):
        parts, cur = [], el
        while cur is not None:
            if _localname(cur.tag) == "Node":
                parts.append(node_id(cur))
            cur = parent.get(cur)
        return "/".join(reversed(parts))

    rows = []
    for node in root.iter():
        if _localname(node.tag) != "Node":
            continue
        params = [c for c in node if _localname(c.tag) == "Param"]
        if not params:
            continue
        a = node.attrib
        base = [path_of(node), a.get("name", ""), a.get("label", ""),
                a.get("type", "")]
        for pr in params:
            rows.append(base + [pr.attrib.get("property", ""),
                                 (pr.text or "").strip()])
    return rows


def unpack_appx(appx_path, out_dir=None, extract_gui=False):
    """Write `<stem>_settings.csv` next to the .appx (or in out_dir).

    If extract_gui, also dump Form.xml + Layouts/*.xml into `<stem>_gui/`.
    Returns a summary dict {csv, rows, gui_dir}.

    Args:
        appx_path: Filesystem path for the appx data.
        out_dir: Destination directory.
        extract_gui: Whether to extract gui.
    """
    appx_path = Path(appx_path)
    out_dir = Path(out_dir) if out_dir else appx_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = settings_rows(appx_path)
    csv_path = out_dir / f"{appx_path.stem}_settings.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        w.writerows(rows)

    gui_dir = None
    if extract_gui:
        gui_dir = out_dir / f"{appx_path.stem}_gui"
        gui_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(appx_path) as zf:
            for n in zf.namelist():
                if n == "Form.xml" or n.startswith("Layouts/"):
                    dest = gui_dir / n.replace("/", "_")
                    dest.write_bytes(zf.read(n))
    return {"csv": str(csv_path), "rows": len(rows),
            "gui_dir": str(gui_dir) if gui_dir else None}
