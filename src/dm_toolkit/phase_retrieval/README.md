# phase_retrieval

Version 3.2

Fits a small set of Zernike coefficients to one focal-plane spot image by
matching a scalar Fraunhofer model to the measured intensity. Used as a
diagnostic of the aberration before and after a correction, and by the
`correction.wavefront` controller.

## Files

`estimate.py` is the fit. `Optics` holds wavelength, focal length, aperture
and pixel pitch; `RetrievalOptions` holds the crop rule, the modes, the
solver tolerances and the optional switches. `estimate_wavefront` crops the
spot (`find_spot`, `converged_side`), seeds the search from the spot's second
moment (`magnitude_prior`), runs a coarse stage on defocus and astigmatism,
then the full fit with restarts, and returns a `WavefrontEstimate` with
coefficients, uncertainties, residual and the flags a caller must read:
`beyond_capture_range`, `converged`, `even_sign_flipped`. `render_model`
draws the fitted spot beside the data.

`efield.py` is the alternative search in the electric field domain
(Zingarelli and Cain): `gerchberg_saxton` estimates the detector field and
`search` maximises the correlation with modelled fields. Off by default; it
helps only close to the diffraction limit.

`torch_model.py` is the same forward model in PyTorch (`TorchForward`), so
the Jacobian comes from automatic differentiation and can run on a GPU.
`verify_against_prysm` checks it against the reference propagation.

`dm_basis.py` lets the fit expand the phase in the mirror's own eigenmodes
instead of Zernikes (`DMBasis`, `from_npz` from an influence-matrix session
file). The fitted amplitudes are then directly the command, and the report
projects them back onto Noll terms.
