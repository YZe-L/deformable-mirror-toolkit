# beam

Version 3.2

Beam and spot geometry from camera frames, shared by the source
characterisation and the correction scores.

## Files

`beam.py` starts with frame handling (`as_gray`, `subtract_background`,
`locate_spot`, `robust_peak`) and then measures widths: `d4sigma` is the
second-moment diameter with the analysis window re-centred and re-sized
until it settles, `gaussian_1e2` and `fit_gaussian_2d` give the Gaussian
widths, and `measure_spot` bundles them. `fit_m2` fits the ISO 11146
hyperbola to widths measured along the axis and returns the waist, its
position and the beam quality factor for each transverse axis;
`rayleigh_mm` and `predicted_waist_um` derive from it.

`spot_quality.py` extracts the feature vector one spot image gives
(`extract_features`: encircled-energy radius, second moment, ellipticity,
normalised peak, ring strength and more) and scores it. `RuleBasedScorer`
combines the size, energy and shape factors into a geometric-mean score;
`LearnedScorer` is the hook for a trained model. `score_batch` runs a folder
and `log_sample` appends rows to a dataset CSV.
