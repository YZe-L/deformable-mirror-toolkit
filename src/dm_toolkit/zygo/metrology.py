# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.0, 2026-07-30

"""Convert manually entered Mx height results into mirror displacement."""

from __future__ import annotations


def mx_waves_to_surface_nm(
    waves: float,
    wavelength_nm: float,
    interferometric_scale_factor: float,
) -> float:
    """Convert an Mx Waves result to normal-incidence surface height.

    This is for a Fizeau measurement with one reflection from the test surface.
    Mx has already applied its Interf Scale Factor to the displayed Waves value.
    Inverting that scaling gives ``z = waves * wavelength / (2 * scale)``.

    Args:
        waves: Height result displayed by Mx in Waves.
        wavelength_nm: Interferometer wavelength in nanometres.
        interferometric_scale_factor: Mx Interf Scale Factor used for the
            result.

    Returns:
        Surface height represented by the Mx result, in nanometres.

    Raises:
        ValueError: If the wavelength or scale factor is not positive.
    """
    wavelength = float(wavelength_nm)
    scale = float(interferometric_scale_factor)
    if wavelength <= 0:
        raise ValueError("interferometer wavelength must be positive")
    if scale <= 0:
        raise ValueError("Mx Interf Scale Factor must be positive")
    return float(waves) * wavelength / (2.0 * scale)


def mx_surface_nm_to_waves(
    height_nm: float,
    wavelength_nm: float,
    interferometric_scale_factor: float,
) -> float:
    """Convert normal-incidence surface height to the Mx Waves result.

    Args:
        height_nm: Surface height in nanometres.
        wavelength_nm: Interferometer wavelength in nanometres.
        interferometric_scale_factor: Mx Interf Scale Factor used for the
            result.

    Returns:
        The equivalent Mx result in Waves.

    Raises:
        ValueError: If the wavelength or scale factor is not positive.
    """
    wavelength = float(wavelength_nm)
    scale = float(interferometric_scale_factor)
    if wavelength <= 0:
        raise ValueError("interferometer wavelength must be positive")
    if scale <= 0:
        raise ValueError("Mx Interf Scale Factor must be positive")
    return float(height_nm) * 2.0 * scale / wavelength


def mx_relative_displacement_nm(
    waves: float,
    zero_waves: float,
    wavelength_nm: float,
    interferometric_scale_factor: float,
) -> float:
    """Convert an Mx Waves change from the sweep baseline to nanometres.

    Args:
        waves: Current Mx height result in Waves.
        zero_waves: Mx height result recorded at the first sweep point.
        wavelength_nm: Interferometer wavelength in nanometres.
        interferometric_scale_factor: Mx Interf Scale Factor used for the
            result.

    Returns:
        Surface displacement relative to the first recorded point, in
        nanometres.
    """
    return mx_waves_to_surface_nm(
        float(waves) - float(zero_waves),
        wavelength_nm,
        interferometric_scale_factor,
    )
