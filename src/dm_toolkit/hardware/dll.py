# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-06-12

"""Thorlabs DLL search path -- import before thorlabs_tsi_sdk."""

import os
import sys

from ..config import VENDOR_DIR

DLL_DIR = VENDOR_DIR / "dll" / ("64_lib" if sys.maxsize > 2 ** 32
                                else "32_lib")

if os.name == "nt" and DLL_DIR.is_dir():
    os.environ["PATH"] = str(DLL_DIR) + os.pathsep + os.environ["PATH"]
    os.add_dll_directory(str(DLL_DIR))
