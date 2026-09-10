# tools

Version 3.2

Command-line scripts that post-process saved runs. Run them as modules with
`src` on the path:

```bash
python -m dm_toolkit.tools.retrieve_dm_loop <run folder or parent> [--roi 192]
python -m dm_toolkit.tools.correctability <folder> --im <influence_matrix.npz>
```

## Files

`retrieve_dm_loop.py` finds every run folder holding `spot_before_avg.npy`
and `spot_after_avg.npy`, reads the optics from the run's own
`run_config.json`, fits Zernike coefficients to both spots with
`phase_retrieval`, and writes a four-panel comparison figure, a
model-versus-data check figure and `phase_retrieval.csv` beside them.

`correctability.py` reads that CSV together with an influence-matrix
session file and, for every run, splits the retrieved wavefront into the
part the retained mirror modes could cancel within the available stroke and
the part they could not. It reports how much of the correctable part each
run actually removed, writes a CSV and a figure.
