# SPDX-License-Identifier: GPL-3.0-or-later

"""A rewire moves wires, not actuators: the matrix is re-labelled, not redone."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dm_toolkit.influence import im_store as S


class RemapMatrixTests(unittest.TestCase):
    """Re-labelling must move every row's channel and no row's numbers."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.ctrl = np.arange(27, dtype=float).reshape(9, 3)
        self.path = S.save_matrix(
            range(1, 10), self.ctrl, np.array([3.0, 2.0, 1.0]), 2,
            grid_n=64, laser_nm=520.0, source="surfaces/20260812",
            mirror="DM9", path=self.dir / "orig.npz")
        self.stored = S.load_matrix(self.path)

    def remapped(self, new_channels, name="new.npz"):
        """The matrix that comes back after re-labelling onto new_channels."""
        out = S.remap_matrix(self.stored, new_channels,
                             path=self.dir / name)
        return S.load_matrix(out)

    def test_a_block_rewire_keeps_every_number(self) -> None:
        # 1-9 moved to 6-14: same mirror, other header pins.
        got = self.remapped(range(6, 15))
        self.assertEqual(list(got.channels), list(range(6, 15)))
        np.testing.assert_array_equal(got.ctrl, self.ctrl)
        np.testing.assert_array_equal(got.s, self.stored.s)
        self.assertEqual(got.keep, self.stored.keep)
        self.assertEqual(got.laser_nm, self.stored.laser_nm)
        self.assertEqual(got.mirror, self.stored.mirror)

    def test_the_loop_can_then_drive_it(self) -> None:
        # The whole point: align() refused the old file for the new table.
        got = self.remapped(range(6, 15))
        with self.assertRaises(S.MatrixMismatch):
            self.stored.align(range(6, 15))
        np.testing.assert_array_equal(got.align(range(6, 15)), self.ctrl)

    def test_each_row_follows_its_own_actuator(self) -> None:
        # Row i must answer to new_channels[i], whatever order it is asked in.
        got = self.remapped([16, 15, 14, 13, 12, 11, 10, 9, 8], "perm.npz")
        # Ask for them low-to-high: row order must invert with the labels.
        np.testing.assert_array_equal(got.align(range(8, 17)),
                                      self.ctrl[::-1])

    def test_the_file_says_it_was_re_labelled(self) -> None:
        got = self.remapped(range(6, 15))
        self.assertIn("remapped", got.source)
        self.assertIn("orig.npz", got.source)
        with np.load(got.path) as data:
            self.assertEqual(str(data["remapped_from"]), str(self.path))
            np.testing.assert_array_equal(data["remapped_old_channels"],
                                          np.arange(1, 10))
            np.testing.assert_array_equal(data["remapped_new_channels"],
                                          np.arange(6, 15))

    def test_the_original_survives(self) -> None:
        self.remapped(range(6, 15))
        again = S.load_matrix(self.path)
        self.assertEqual(list(again.channels), list(range(1, 10)))

    def test_a_mapping_that_is_not_one_for_one_is_refused(self) -> None:
        with self.assertRaises(S.MatrixMismatch):
            self.remapped(range(6, 12), "short.npz")  # Too few channels.
        with self.assertRaises(S.MatrixMismatch):
            self.remapped([6, 6, 8, 9, 10, 11, 12, 13, 14], "dupe.npz")


if __name__ == "__main__":
    unittest.main()
