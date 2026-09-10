# adaptive_settling

Version 3.2

Chooses the wait between a command and the next measurement from what the
loop already knows: the displacement each channel's hysteresis profile
predicts, whether the command clamped, and a per-channel residual gate from
calibration.

## Files

`model.py` defines the descriptor a channel carries (`AdaptiveDescriptor`:
gate in nanometres, multipliers, wait tiers) and validates it on load
(`load_descriptor`). `choose_wait` applies the rule: the largest predicted
displacement change relative to its gate selects the small, medium or full
wait, and any clamp or missing prediction selects the full wait
(`AdaptiveDecision`). The quantile helpers compute the gate from sorted
residuals with an exact binomial bound.

`analysis.py` turns a calibration session into descriptors. `analyse_rows`
computes the signed residual between predicted and measured height for
every transition, `analyse_session` writes one `adaptive_settling_chN.json`
per channel and a report, and `file_sha256` ties each descriptor to the
profile it was measured with.

## descriptors/

`dm5/` and `dm9/` hold the descriptors for channels 1 to 5 and 6 to 14. Each
file records the gate, the tiers, the source profile and the residual
statistics it came from.
