# band_scan

Version 3.2

The modal solve needs a spatial-frequency band in which the reciprocal band
power is quadratic in the aberration. This package chooses that band from a
series of spot images recorded while one actuator was swept.

## Files

`index.py` pairs the saved spot images of a scan folder with the command
behind each one, read from the file name tag (`Series`, `Point`,
`read_series`).

`scan.py` scores every image through the same `correction.metrics.measure`
the loop uses, once per candidate band, then fits the reciprocal band power
against the command with a parabola. For each band `BandResult` reports the
fitted vertex, the half width, the coefficient of determination and the
contrast between the band power swing and the fit scatter. `scan` applies
the screening rules and returns a `ScanResult` with the recommended band and
probe amplitude, or none when no band survives.
