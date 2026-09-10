# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the manual Mx Waves conversion."""

from __future__ import annotations

import unittest

from dm_toolkit.zygo import metrology


class MxMetrologyTests(unittest.TestCase):
    """Verify the documented normal-incidence Mx height conversion."""

    def test_surface_scale_converts_one_wave_to_one_wavelength(self) -> None:
        self.assertAlmostEqual(
            metrology.mx_waves_to_surface_nm(1.0, 632.8, 0.5),
            632.8,
        )

    def test_wavefront_scale_converts_one_wave_to_half_wavelength(self) -> None:
        self.assertAlmostEqual(
            metrology.mx_waves_to_surface_nm(1.0, 632.8, 1.0),
            316.4,
        )

    def test_displacement_is_relative_to_first_recorded_height(self) -> None:
        self.assertAlmostEqual(
            metrology.mx_relative_displacement_nm(2.25, 0.25, 632.8, 0.5),
            1265.6,
        )

    def test_surface_nm_converts_back_to_waves(self) -> None:
        self.assertAlmostEqual(
            metrology.mx_surface_nm_to_waves(1265.6, 632.8, 0.5),
            2.0,
        )

    def test_nonpositive_conversion_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            metrology.mx_waves_to_surface_nm(1.0, 0.0, 0.5)
        with self.assertRaises(ValueError):
            metrology.mx_waves_to_surface_nm(1.0, 632.8, 0.0)


if __name__ == "__main__":
    unittest.main()
