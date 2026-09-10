# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.2

"""Project paths and portable settings-file paths."""

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
# The checkout root (src/dm_toolkit -> ../..). Files inside it -- hysteresis
# profiles, stored matrices -- are saved relative to it, so a settings file
# written on one machine still finds them on another.
REPO_DIR = PKG_DIR.parent.parent
OUTPUT_DIR = REPO_DIR / "output"
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
FRAMES_DIR = OUTPUT_DIR / "frames"
# Proprietary vendor code that is not part of this repository: the Thorlabs
# camera DLLs (vendor/dll) and the Zygo Mx scripting package (vendor/zygo).
VENDOR_DIR = REPO_DIR / "vendor"


def portable_path(path) -> str:
    """A path in the form to write into a settings file.

    Anything inside the checkout is stored relative to it with forward
    slashes; anything outside is stored absolute. A path that is already
    relative is anchored to the checkout, not to the working directory.

    Args:
        path: Path or string to store.

    Returns:
        The text to save.
    """
    path = Path(path)
    resolved = (path if path.is_absolute() else REPO_DIR / path).resolve()
    try:
        return resolved.relative_to(REPO_DIR).as_posix()
    except ValueError:
        return str(resolved)


def resolve_path(text) -> Path:
    """A path read from a settings file, pointed at this machine.

    Relative text is hung off the checkout. Absolute text is returned as is
    when it exists; when it is missing but names this checkout's folder, the
    tail below that folder is re-anchored here.

    Args:
        text: The stored text.

    Returns:
        The path to use. It is not guaranteed to exist.
    """
    path = Path(str(text).replace("\\", "/"))
    if not path.is_absolute():
        return REPO_DIR / path
    if path.exists():
        return path
    parts = path.parts
    if REPO_DIR.name in parts:
        below = len(parts) - 1 - parts[::-1].index(REPO_DIR.name)
        return REPO_DIR.joinpath(*parts[below + 1:])
    return path
