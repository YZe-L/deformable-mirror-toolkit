# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 1.3, 2026-08-12

"""Round-trip, place and reject impact matrices in the calibration store."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from dm_toolkit.influence import im_store as ST


class RoundTripTest(unittest.TestCase):
    """What is written must come back unchanged."""

    CHANNELS = (1, 3, 5, 7)

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "im.npz"
        self.ctrl = np.arange(16, dtype=float).reshape(4, 4)
        self.s = np.array([9.0, 4.0, 1.0, 0.1])
        ST.save_matrix(self.CHANNELS, self.ctrl, self.s, keep=3, grid_n=128,
                       laser_nm=635.0, source="folder", mirror="DM9",
                       path=self.path)

    def test_fields_survive(self):
        m = ST.load_matrix(self.path)
        self.assertEqual(m.channels, self.CHANNELS)
        np.testing.assert_allclose(m.ctrl, self.ctrl)
        np.testing.assert_allclose(m.s, self.s)
        self.assertEqual((m.keep, m.grid_n, m.n_modes), (3, 128, 4))
        self.assertEqual(m.source, "folder")
        self.assertEqual(m.mirror, "DM9")

    def test_the_mirror_is_named_in_every_status_string(self):
        """Which mirror a matrix belongs to is the one thing no check catches.

        `align` compares channel numbers, so two nine-element mirrors validate
        against each other. Naming the mirror wherever a matrix is shown is
        what makes the wrong one visible before it is driven.
        """
        m = ST.load_matrix(self.path)
        self.assertIn("DM9", m.summary())
        self.assertIn("DM9", m.label())

    def test_curvature_includes_the_driven_column_scale(self):
        """The N+2 solve operates in rad-RMS `ctrl` coordinates.

        `s^2` belongs to unit actuator eigenvectors.  Rescaling each column to
        one radian RMS must rescale its quadratic curvature by the column norm
        squared as well.
        """
        np.testing.assert_allclose(ST.load_matrix(self.path).curvature,
                                   self.s ** 2 * np.sum(self.ctrl ** 2, axis=0))

    def test_require_wavelength_rejects_a_different_ao_laser(self):
        matrix = ST.load_matrix(self.path)
        with self.assertRaisesRegex(ST.MatrixMismatch, "635.*520"):
            matrix.require_wavelength(520.0)

    def test_require_wavelength_accepts_the_recorded_ao_laser(self):
        matrix = ST.load_matrix(self.path)
        self.assertIs(matrix.require_wavelength(635.0), matrix)

    def test_no_temporary_file_is_left_behind(self):
        """The write is atomic, so an interrupted save cannot half-load."""
        self.assertFalse([p for p in self.dir.iterdir()
                          if p.suffix == ".tmp"])

    def test_missing_file_is_not_an_error(self):
        """No calibration yet is the normal state before the first run."""
        self.assertIsNone(ST.load_matrix(self.dir / "absent.npz"))


class AlignmentTest(unittest.TestCase):
    """Channel order and membership decide whether a matrix may be used."""

    CHANNELS = (1, 3, 5, 7)

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "im.npz"
        self.ctrl = np.arange(16, dtype=float).reshape(4, 4)
        ST.save_matrix(self.CHANNELS, self.ctrl, np.ones(4), keep=4,
                       path=self.path)
        self.m = ST.load_matrix(self.path)

    def test_identity_order_is_unchanged(self):
        np.testing.assert_allclose(self.m.align(self.CHANNELS), self.ctrl)

    def test_rows_follow_the_callers_order(self):
        np.testing.assert_allclose(self.m.align([7, 5, 3, 1]),
                                   self.ctrl[[3, 2, 1, 0], :])

    def test_a_subset_is_refused(self):
        """An eigenmode is defined by every actuator measured together.

        Holding one fixed does not restrict the mode, it makes it a different
        shape -- so a partial match must fail rather than silently drive it.
        """
        with self.assertRaises(ST.MatrixMismatch):
            self.m.align([1, 3, 5])

    def test_an_unknown_channel_is_refused(self):
        with self.assertRaises(ST.MatrixMismatch):
            self.m.align([1, 3, 5, 9])


class PlacementTest(unittest.TestCase):
    """Two mirrors must never land on the same file."""

    def test_the_name_carries_the_mirror_and_the_actuator_count(self):
        path = ST.matrix_path("DM9", range(9),
                              when=datetime(2026, 8, 12, 14, 30, 5))
        # Case is kept, so the folder reads like the hysteresis profiles'
        # devices/DM9 rather than a lower-cased near-match beside it.
        self.assertEqual(path.name, "DM9_9ch_20260812_143005.npz")
        self.assertEqual(path.parent.name, "DM9")

    def test_two_mirrors_go_to_different_folders(self):
        when = datetime(2026, 8, 12, 14, 30, 5)
        five = ST.matrix_path("DM5", range(5), when=when)
        nine = ST.matrix_path("DM9", range(9), when=when)
        self.assertNotEqual(five.parent, nine.parent)
        self.assertNotEqual(five.name, nine.name)

    def test_the_same_mirror_measured_twice_does_not_overwrite(self):
        """The old single well-known name silently destroyed the previous one."""
        first = ST.matrix_path("DM9", range(9),
                               when=datetime(2026, 8, 12, 14, 30, 5))
        second = ST.matrix_path("DM9", range(9),
                                when=datetime(2026, 8, 12, 15, 0, 1))
        self.assertNotEqual(first, second)

    def test_an_unusable_name_falls_back_rather_than_escaping_the_store(self):
        """A typed name reaches the filesystem, so it must not steer the path."""
        for hostile in ("", "   ", "../../etc", "///"):
            path = ST.matrix_path(hostile, range(9))
            self.assertEqual(path.parent.parent, ST.MATRIX_DIRECTORY)

    def test_a_name_keeps_its_readable_characters(self):
        self.assertEqual(ST.slug("DM9 all 2000"), "DM9_all_2000")


class ListingTest(unittest.TestCase):
    """The picker offers what this mirror can actually use, and nothing else."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        for mirror, channels in (("DM9", range(9)), ("DM5", range(5))):
            ST.save_matrix(channels, np.ones((len(list(channels)), 2)),
                           np.ones(2), keep=2, mirror=mirror,
                           path=self.dir / mirror / f"{mirror}.npz")

    def test_sub_folders_are_searched(self):
        self.assertEqual(len(ST.list_matrices(self.dir)), 2)

    def test_a_different_actuator_count_is_not_offered(self):
        """Five-element eigenmodes are not a worse choice for a nine-element
        mirror, they are a meaningless one."""
        found = ST.list_matrices(self.dir, n_channels=9)
        self.assertEqual([m.mirror for m in found], ["DM9"])

    def test_an_unreadable_stray_does_not_empty_the_list(self):
        with open(self.dir / "junk.npz", "wb") as fh:
            fh.write(b"not an npz at all")
        self.assertEqual(len(ST.list_matrices(self.dir)), 2)

    def test_a_missing_store_is_not_an_error(self):
        self.assertEqual(ST.list_matrices(self.dir / "absent"), [])


class LegacyTest(unittest.TestCase):
    """The single file written before v1.4 still loads, and says it is unnamed."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / ST.DEFAULT_NAME
        with open(self.path, "wb") as fh:
            np.savez_compressed(fh, format_version=np.asarray(1),
                                channels=np.asarray([1, 2, 3]),
                                ctrl=np.zeros((3, 2)),
                                singular_values=np.ones(2),
                                keep=np.asarray(2))

    def test_it_still_loads(self):
        self.assertEqual(ST.load_matrix(self.path).channels, (1, 2, 3))

    def test_it_is_shown_as_unattributed(self):
        m = ST.load_matrix(self.path)
        self.assertEqual(m.mirror, "")
        self.assertIn(ST.UNNAMED, m.mirror_label)

    def test_it_is_still_offered_by_the_picker(self):
        self.assertEqual(len(ST.list_matrices(self.dir)), 1)


class CorruptionTest(unittest.TestCase):
    """A file that cannot be trusted must raise, never load partially."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def _write(self, **payload):
        path = self.dir / "bad.npz"
        with open(path, "wb") as fh:
            np.savez_compressed(fh, **payload)
        return path

    def test_a_newer_format_is_refused(self):
        path = self._write(format_version=np.asarray(ST.FORMAT_VERSION + 1),
                           channels=np.asarray([1]), ctrl=np.zeros((1, 1)),
                           singular_values=np.ones(1), keep=np.asarray(1))
        with self.assertRaises(ST.MatrixMismatch):
            ST.load_matrix(path)

    def test_shape_disagreement_is_refused(self):
        """Three channels against a two-row ctrl would misdrive the mirror."""
        path = self._write(format_version=np.asarray(ST.FORMAT_VERSION),
                           channels=np.asarray([1, 2, 3]),
                           ctrl=np.zeros((2, 2)), singular_values=np.ones(2),
                           keep=np.asarray(2))
        with self.assertRaises(ST.MatrixMismatch):
            ST.load_matrix(path)

    def test_describe_reports_rather_than_raises(self):
        """The UI status line must survive a bad file."""
        path = self._write(format_version=np.asarray(ST.FORMAT_VERSION + 1),
                           channels=np.asarray([1]), ctrl=np.zeros((1, 1)),
                           singular_values=np.ones(1), keep=np.asarray(1))
        self.assertIn("unusable", ST.describe(path))


if __name__ == "__main__":
    unittest.main()
