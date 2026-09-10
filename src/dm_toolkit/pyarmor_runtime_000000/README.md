# pyarmor_runtime_000000

Version 3.2

Runtime package for the one obfuscated module in this release,
`interferometry/surface.py`. That module imports `__pyarmor__` from here at
load time and runs like any other module. The runtime is built for 64-bit
Windows and Python 3.11; on other platforms the surface module cannot be
imported, and everything that does not use it still works.

Do not edit or rename this folder; the obfuscated module looks for it by
name.
