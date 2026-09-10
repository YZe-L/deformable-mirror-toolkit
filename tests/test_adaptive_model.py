import json
import tempfile
import unittest
from pathlib import Path

from dm_toolkit.adaptive_settling.model import (
    REQUIRED_TIERS_MS,
    choose_wait,
    descriptor_document,
    empirical_quantile_higher,
    load_descriptor,
    quantile_confidence_interval,
    quantile_lower_confidence_bound,
)


def _descriptor(tmp_path, channel, egate=2.0, mirror=5):
    path = tmp_path / f"ch{channel}.json"
    path.write_text(json.dumps(descriptor_document(
        channel=channel, mirror_actuators=mirror, egate_nm=egate,
        calibration_set_id="test")), encoding="utf-8")
    return load_descriptor(path)


class AdaptiveModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_runtime_uses_most_restrictive_channel(self):
        descriptors = {1: _descriptor(self.folder, 1),
                       2: _descriptor(self.folder, 2)}
        decision = choose_wait(descriptors, {1: 0.0, 2: 0.0},
                               {1: 9.0, 2: 21.0})
        self.assertEqual(decision.settle_ms, REQUIRED_TIERS_MS["full"])
        self.assertEqual(decision.per_channel_tier, {1: "small", 2: "full"})
        self.assertEqual(decision.limiting_channels, (2,))

    def test_clamp_forces_full_wait(self):
        descriptors = {1: _descriptor(self.folder, 1)}
        decision = choose_wait(descriptors, {1: 0.0}, {1: 0.1}, clamped=[1])
        self.assertEqual(decision.tier, "full")
        self.assertEqual(decision.fallback_reason, "clamped command")

    def test_descriptor_rejects_nonstandard_tiers(self):
        doc = descriptor_document(channel=1, mirror_actuators=5, egate_nm=1.0,
                                  calibration_set_id="test")
        doc["tiers_ms"]["small"] = 300
        path = self.folder / "bad.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "tiers_ms must be exactly"):
            load_descriptor(path)

    def test_empirical_p95_and_distribution_free_lower_bound(self):
        values = list(range(1, 101))
        self.assertEqual(empirical_quantile_higher(values, 0.95), 95)
        lower, rank = quantile_lower_confidence_bound(values)
        self.assertEqual(lower, values[rank - 1])
        self.assertTrue(1 <= rank < 95)
        ci_low, ci_high, low_rank, high_rank = quantile_confidence_interval(values)
        self.assertEqual(ci_low, values[low_rank - 1])
        self.assertEqual(ci_high, values[high_rank - 1])
        self.assertLessEqual(ci_low, ci_high)
