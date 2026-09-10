# SPDX-License-Identifier: GPL-3.0-or-later
# Revised by Yuze & Nan, version 2.0, 2026-07-23

from __future__ import annotations

import unittest

from dm_toolkit.hysteresis import comp_sweep, step_wave
from dm_toolkit.hysteresis import HysteresisCompensator


class CameraLinearizedCompensationTests(unittest.TestCase):
    def test_dm_d_profile_exposes_linear_nominal_bit_mapping(self) -> None:
        controller = HysteresisCompensator("dm_d")
        slope = 440.0144 / 4095.0
        for nominal_bit in (0, 500, 1000, 2000, 3000, 4000, 4095):
            self.assertAlmostEqual(
                controller.nominal_bit_to_displacement(nominal_bit),
                slope * nominal_bit,
                places=9,
            )

    def test_surface_sweep_uses_linear_targets_on_both_branches(self) -> None:
        plan = comp_sweep.build_compensated_sweep("dm_d", 0, 4095, 500)
        slope = 440.0144 / 4095.0
        by_nominal: dict[int, list[tuple[int, float, float, bool]]] = {}
        for nominal, target, applied, predicted, clamped in zip(
            plan.nominal_bits,
            plan.target_nm,
            plan.command_bits,
            plan.predicted_nm,
            plan.clamped,
        ):
            self.assertAlmostEqual(target, slope * nominal, places=9)
            if not clamped:
                self.assertLess(abs(predicted - target), 0.11)
            by_nominal.setdefault(nominal, []).append(
                (applied, target, predicted, clamped)
            )

        for nominal_bit in (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000):
            up, down = by_nominal[nominal_bit]
            self.assertNotEqual(up[0], down[0])
            self.assertEqual(up[1], down[1])

        # A large loop cannot physically remove the modeled bit-0 residual.
        self.assertTrue(by_nominal[0][-1][3])
        self.assertEqual(by_nominal[0][-1][0], 0)

    def test_step_plan_uses_same_linear_target_definition(self) -> None:
        levels = [0, 1000, 2000, 3000, 2000, 1000, 0]
        compensated = step_wave.build_step_plan(
            "dm_d",
            levels,
            compensated=True,
        )
        raw = step_wave.build_step_plan("dm_d", levels, compensated=False)
        slope = 440.0144 / 4095.0

        self.assertEqual(raw.command_bits, levels)
        for nominal, target in zip(levels, compensated.ideal_nm):
            self.assertAlmostEqual(target, slope * nominal, places=9)
        self.assertNotEqual(compensated.command_bits[2], levels[2])
        self.assertNotEqual(
            compensated.command_bits[2],
            compensated.command_bits[4],
        )

    def test_sine_levels_keep_every_sample(self) -> None:
        points, cycles = 12, 2
        levels = step_wave.build_levels(
            "sine", baseline=0, centre=2000, amplitude=1000,
            points=points, repeats=cycles,
        )
        # One closing centre sample; crest/trough repeats must NOT be deduped.
        self.assertEqual(len(levels), points * cycles + 1)
        self.assertEqual(levels[0], 2000)
        self.assertEqual(levels[-1], 2000)
        self.assertEqual(max(levels), 3000)
        self.assertEqual(min(levels), 1000)

    def test_sine_amplitude_clamps_to_bit_range(self) -> None:
        levels = step_wave.build_levels(
            "sine", baseline=0, centre=2000, amplitude=3000, points=8,
            repeats=1,
        )
        self.assertEqual(min(levels), 0)
        self.assertLessEqual(max(levels), 4095)

    def test_anchor_preroll_homes_then_anchors_per_pass(self) -> None:
        levels = [2000, 3000, 2000, 1000, 2000]
        comp = step_wave.build_step_plan(
            "dm_d", levels, compensated=True, anchor=2000)
        raw = step_wave.build_step_plan(
            "dm_d", levels, compensated=False, anchor=2000)
        self.assertEqual(comp.n_setup, 2)  # Home bit + anchor bit.
        self.assertEqual(comp.zero_bit, 2000)
        self.assertEqual(comp.levels[: comp.n_setup], [0, 2000])
        # Bit-0 home is a raw reset on both passes.
        self.assertEqual(comp.command_bits[0], 0)
        self.assertEqual(raw.command_bits[0], 0)
        # The anchor follows the pass scheme: raw bit on raw, compensated on
        # comp.
        self.assertEqual(raw.command_bits[1], 2000)
        self.assertNotEqual(comp.command_bits[1], 2000)
        # The compensated anchor equals the compensated bit of the first sine
        # centre (both are nominal 2000 approached from below), so parking and
        # the waveform agree.
        self.assertEqual(comp.command_bits[1], comp.command_bits[2])

    def test_compensated_pass_tracks_the_linear_ideal_zeroed_at_anchor(self):
        """Verify compensated passes against the anchored linear ideal.

        Replaying the COMPENSATED bits through a fresh model must land on the
        ideal to within one bit, with the zero taken straight from the settled
        anchor hold (no model offset) -- matching what the recorder does.
        """
        levels = step_wave.build_levels(
            "sine", baseline=0, centre=2000, amplitude=1000, points=12,
            repeats=2)
        plan = step_wave.build_step_plan(
            "dm_d", levels, compensated=True, anchor=2000)
        model = HysteresisCompensator("dm_d")
        model.commit(model.plan_home())
        one_bit_nm = 440.0144 / 4095.0
        zero = None
        for k, (bit, ideal) in enumerate(zip(plan.command_bits, plan.ideal_nm)):
            predicted = model.commit(model.plan_bit(int(bit), clamp=True))
            if k == plan.n_setup - 1:  # The settled anchor = zero.
                zero = predicted
                continue
            if k < plan.n_setup:
                continue
            # <= 1.5 bit: the anchor and the level each carry one quantisation.
            self.assertLess(abs((predicted - zero) - ideal), 1.5 * one_bit_nm)

    def test_no_anchor_keeps_legacy_behaviour(self) -> None:
        levels = [0, 1000, 2000]
        plan = step_wave.build_step_plan("dm_d", levels, compensated=False)
        self.assertEqual(plan.n_setup, 0)
        self.assertEqual(plan.levels, levels)
        self.assertEqual(plan.ideal_nm[0], 0.0)

    def test_anchor_without_home_skips_bit_zero(self) -> None:
        plan = step_wave.build_step_plan(
            "dm_d", [2000, 3000, 2000], compensated=False, anchor=2000,
            home=False)
        self.assertEqual(plan.n_setup, 1)
        self.assertEqual(plan.levels[0], 2000)
        self.assertEqual(plan.command_bits[0], 2000)

    def test_legacy_profile_keeps_loading_reference_fallback(self) -> None:
        plan = comp_sweep.build_compensated_sweep("piezo_a", 0, 1000, 500)
        self.assertEqual(plan.nominal_bits, [0, 500, 1000, 500, 0])
        self.assertEqual(len(plan.target_nm), len(plan.nominal_bits))
        self.assertGreater(plan.target_nm[2], plan.target_nm[1])


if __name__ == "__main__":
    unittest.main()
