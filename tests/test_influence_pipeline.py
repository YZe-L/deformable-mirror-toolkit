# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for impact-matrix file selection and validation masks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dm_toolkit.influence import influence as INF
from dm_toolkit.influence import io_surface as IO
from dm_toolkit.influence import pipeline as P
from dm_toolkit.influence import trace as TR


class VerdictTests(unittest.TestCase):
    """Four measures, so one healthy-looking number cannot hide a fault."""

    # (probe asymmetry, min incremental gain, noise / column norm,
    #  ladder residual / this channel's own stroke)
    HEALTHY = (0.05, 0.95, 0.05, 0.01)

    def test_a_clean_actuator_passes_everything(self) -> None:
        self.assertEqual(P._verdict(*self.HEALTHY), ("ok", []))

    def test_a_dead_zone_is_caught_even_with_a_symmetric_probe(self) -> None:
        # ch6's signature: the pair-averaged numbers look fine because the two
        # halves of every pair cancel, and the ramp is still nearly flat over
        # part of its range.
        grade, reasons = P._verdict(0.05, 0.06, 0.05, 0.01)
        self.assertEqual(grade, "BAD")
        self.assertIn("dead zone", reasons)

    def test_an_asymmetric_probe_is_caught_with_a_perfect_column(self) -> None:
        # ch9's signature: the minus probe barely happens, so a three-point
        # solve fits a parabola through points that are not at +b and -b.
        grade, reasons = P._verdict(1.75, 0.95, 0.05, 0.01)
        self.assertEqual(grade, "BAD")
        self.assertIn("probe asymmetry", reasons)

    def test_a_column_at_the_noise_floor_fails(self) -> None:
        # Nothing else is wrong with it; there is simply not enough of it above
        # the noise for the SVD to place, so the solve would drive noise.
        grade, reasons = P._verdict(0.05, 0.95, 1.0 / 1.5, 0.01)
        self.assertEqual(grade, "BAD")
        self.assertIn("column lost in noise", reasons)

    def test_a_column_clear_of_the_noise_floor_passes(self) -> None:
        # KEEP_MARGIN is the same margin the mode count is truncated at.
        clear = 1.0 / (INF.KEEP_MARGIN * 6)
        self.assertEqual(P._verdict(0.05, 0.95, clear, 0.01), ("ok", []))

    def test_the_ladder_residual_is_judged_against_the_own_stroke(self) -> None:
        # Against the stroke, not the repeat noise: a residual a thousand times
        # the noise is still nothing if the actuator moved a million times it.
        self.assertEqual(P._verdict(0.05, 0.95, 0.05, 0.07)[0], "watch")
        self.assertEqual(P._verdict(0.05, 0.95, 0.05, 0.12)[0], "BAD")
        self.assertEqual(P._verdict(0.05, 0.95, 0.05, 0.02)[0], "ok")

    def test_an_unmeasurable_channel_is_not_graded_as_healthy(self) -> None:
        # A missing pair leaves NaN; that must not read as a passing score.
        grade, reasons = P._verdict(float("nan"), 0.06, 0.05, 0.01)
        self.assertEqual(grade, "BAD")
        self.assertNotIn("probe asymmetry", reasons)


class FootprintTests(unittest.TestCase):
    """Per-actuator scores must be read where the actuator actually acts."""

    @staticmethod
    def _bump(grid, inside, radius=0.25):
        """A localised influence function, flattened over the pupil points."""
        x_coord, y_coord = grid.coords
        r2 = (x_coord**2 + y_coord**2) / radius**2
        return np.exp(-r2)[inside]

    def test_a_shape_score_is_not_swamped_by_the_empty_pupil(self) -> None:
        # Two identical bumps, differing only by noise everywhere. Pupil-wide
        # normalisation reads that noise as a shape change because it divides
        # by an RMS averaged over points the actuator never reaches; the
        # footprint weight reads it where the signal is.
        grid = INF.Grid(64)
        inside = grid.inside
        bump = self._bump(grid, inside)
        rng = np.random.default_rng(0)
        noisy = bump + 0.01 * rng.standard_normal(bump.shape)
        weight = INF.footprint_weight(bump)
        self.assertGreater(INF.shape_difference(noisy, bump), 0.05)
        self.assertLess(INF.shape_difference(noisy, bump, weight), 0.02)

    def test_the_weight_follows_the_column_not_the_pupil(self) -> None:
        grid = INF.Grid(64)
        inside = grid.inside
        weight = INF.footprint_weight(self._bump(grid, inside))
        self.assertAlmostEqual(float(weight.sum()), 1.0, places=12)
        # Concentrated: far fewer points carry the weight than carry the pupil.
        carrying = int((weight > weight.max() * 1e-3).sum())
        self.assertLess(carrying, 0.5 * int(inside.sum()))

    def test_an_empty_column_has_no_footprint(self) -> None:
        self.assertIsNone(INF.footprint_weight(np.zeros(64)))

    def test_rms_reads_high_until_the_noise_is_taken_out(self) -> None:
        # The bias is what makes the narrowest -- noisiest -- push-pull pair
        # report the largest influence per bit, which looks exactly like the
        # odd-order non-linearity the linearity check is there to find.
        rng = np.random.default_rng(1)
        signal = np.full(20000, 1.0)
        noisy = signal + 0.5 * rng.standard_normal(signal.shape)
        # The bias is sqrt(1 + 0.5^2) = 1.118; the tolerance is the sampling
        # error left on one realisation, an order below what is removed.
        self.assertGreater(INF.weighted_rms(noisy), 1.05)
        self.assertAlmostEqual(INF.debiased_rms(noisy, 0.5), 1.0, delta=0.02)

    def test_debiasing_cannot_produce_a_negative_amplitude(self) -> None:
        # A column that is all noise: the honest answer is zero, not a NaN
        # from a negative square root.
        self.assertEqual(INF.debiased_rms(np.zeros(100), 3.0), 0.0)


class NoteLevelTests(unittest.TestCase):
    """A page where every remark is amber reads as one where none of them is."""

    def test_a_note_is_a_warning_unless_it_says_otherwise(self) -> None:
        step = TR.Trace().add("t")
        step.note("something is off")
        self.assertEqual(step.notes[0].level, TR.WARN)

    def test_the_three_levels_are_marked_differently_in_the_text(self) -> None:
        tr = TR.Trace()
        step = tr.add("t")
        step.info("this is how it was configured")
        step.note("the result is worth less than it looks")
        step.stop("do not use this")
        lines = [ln.strip() for ln in tr.to_text().splitlines()
                 if ln.startswith("    ") and ln.strip()[:2] in ("- ", "! ",
                                                                 "!!")]
        self.assertEqual([ln.split()[0] for ln in lines], ["-", "!", "!!"])


class SurfaceSelectionTests(unittest.TestCase):
    """Verify DATX/XYZ twins cannot masquerade as repeated measurements."""

    def test_matching_datx_is_preferred_over_xyz(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            datx = root / "point_c1b1000.datx"
            twin = root / "point_c1b1000.xyz"
            orphan = root / "other_c1b2000.xyz"
            for path in (datx, twin, orphan):
                path.touch()
            with mock.patch.object(
                IO,
                "datx_kind",
                return_value=("height", "test height"),
            ):
                files, skipped = IO.find_surfaces(root)
        self.assertEqual(files, [orphan, datx])
        self.assertEqual(skipped, [])

    def test_unusable_datx_does_not_hide_xyz_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            datx = root / "point.datx"
            xyz = root / "point.xyz"
            datx.touch()
            xyz.touch()
            with mock.patch.object(
                IO,
                "datx_kind",
                return_value=("derived", "not a height"),
            ):
                files, skipped = IO.find_surfaces(root)
        self.assertEqual(files, [xyz])
        self.assertEqual(skipped, [(datx, "not a height")])


class ValidationMaskTests(unittest.TestCase):
    """Ensure validation dropouts cannot change the calibration result."""

    @staticmethod
    def _case(include_check: bool) -> tuple[P.Job, dict[Path, tuple]]:
        grid = INF.Grid(32)
        x_coord, y_coord = grid.coords
        pupil = grid.inside
        bias = {1: 2000, 2: 2000}
        paths = {
            "ref": Path("ref.datx"),
            "ch1_lo": Path("ch1_lo.datx"),
            "ch1_hi": Path("ch1_hi.datx"),
            "ch2_lo": Path("ch2_lo.datx"),
            "ch2_hi": Path("ch2_hi.datx"),
            "check": Path("check.datx"),
        }
        first = x_coord**2 - y_coord**2
        second = 2.0 * x_coord * y_coord
        maps = {
            paths["ref"]: (np.zeros_like(x_coord), pupil.copy()),
            paths["ch1_lo"]: (-100.0 * first, pupil.copy()),
            paths["ch1_hi"]: (100.0 * first, pupil.copy()),
            paths["ch2_lo"]: (-100.0 * second, pupil.copy()),
            paths["ch2_hi"]: (100.0 * second, pupil.copy()),
        }
        entries = [
            P.Entry(paths["ref"], P.ROLE_REF, bias),
            P.Entry(paths["ch1_lo"], P.ROLE_PP, {1: 1900, 2: 2000}),
            P.Entry(paths["ch1_hi"], P.ROLE_PP, {1: 2100, 2: 2000}),
            P.Entry(paths["ch2_lo"], P.ROLE_PP, {1: 2000, 2: 1900}),
            P.Entry(paths["ch2_hi"], P.ROLE_PP, {1: 2000, 2: 2100}),
        ]
        if include_check:
            check_ok = pupil & (x_coord < 0.25)
            maps[paths["check"]] = (
                60.0 * first - 40.0 * second,
                check_ok,
            )
            entries.append(
                P.Entry(
                    paths["check"],
                    P.ROLE_CHECK,
                    {1: 2060, 2: 1960},
                )
            )
        job = P.Job(
            entries=entries,
            pupil=INF.Pupil(16.0, 16.0, 15.0),
            grid_n=32,
            zernike_terms=6,
        )
        return job, maps

    @staticmethod
    def _run(job: P.Job, maps: dict[Path, tuple]) -> P.Result:
        def read_surface(path, *_args):
            z, ok = maps[path]
            return IO.SurfaceMap(
                path=path,
                z_nm=z,
                mask=ok,
                note="synthetic nm",
            )

        def resample(surface, _pupil, _grid):
            return maps[surface.path]

        with (
            mock.patch.object(IO, "read_surface", side_effect=read_surface),
            mock.patch.object(INF, "resample", side_effect=resample),
        ):
            return P.run(job)

    def test_check_dropout_does_not_change_calibration_mask_or_svd(self) -> None:
        base_job, base_maps = self._case(include_check=False)
        check_job, check_maps = self._case(include_check=True)
        base = self._run(base_job, base_maps)
        checked = self._run(check_job, check_maps)
        self.assertTrue(base.ok)
        self.assertTrue(checked.ok)
        np.testing.assert_array_equal(checked.inside, base.inside)
        np.testing.assert_allclose(checked.modes.s, base.modes.s, rtol=1e-12)
        self.assertEqual(int(checked.inside.sum()), int(INF.Grid(32).inside.sum()))

    def test_job_defaults_to_trusting_mx_plane_removal(self) -> None:
        job, _maps = self._case(include_check=False)
        self.assertFalse(job.remove_piston_tilt)


class NoiseProjectionTests(unittest.TestCase):
    """Match noise-floor preprocessing to the selected plane handling."""

    def test_plane_removal_is_optional_for_noise_floor(self) -> None:
        grid = INF.Grid(32)
        inside = grid.inside
        x_coord, _y_coord = grid.coords
        tilt = x_coord[inside]
        pairs = [(tilt, np.zeros_like(tilt))]
        raw = INF.noise_singular_value(
            pairs,
            200,
            grid,
            inside,
            remove_plane=False,
        )
        projected = INF.noise_singular_value(
            pairs,
            200,
            grid,
            inside,
            remove_plane=True,
        )
        self.assertGreater(raw, 1e-6)
        self.assertLess(projected, raw * 1e-10)


if __name__ == "__main__":
    unittest.main()
