import csv
import json
import tempfile
import unittest
from pathlib import Path

from dm_toolkit.adaptive_settling.analysis import (
    analyse_rows, analyse_session)


class AdaptiveAnalysisTests(unittest.TestCase):
    def test_isolated_rows_only_use_the_actuated_channel(self):
        rows = []
        for value in range(1, 41):
            rows.append({"role": "transition", "channel": "1",
                         "actuated_channel": "1", "valid": "1",
                         "clamped": "0", "predicted_delta_nm": value,
                         "measured_delta_nm": -value + 0.25})
            rows.append({"role": "transition", "channel": "2",
                         "actuated_channel": "1", "valid": "1",
                         "clamped": "0", "predicted_delta_nm": 0,
                         "measured_delta_nm": 1000})
        result = analyse_rows(rows)
        self.assertEqual(set(result), {1})
        self.assertEqual(result[1]["sign"], -1)
        self.assertAlmostEqual(result[1]["egate_nm"], 0.25)

    def test_session_writes_one_descriptor_per_channel(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            fields = ["role", "channel", "actuated_channel", "valid",
                      "clamped", "predicted_delta_nm", "measured_delta_nm"]
            with (folder / "formal_measurements.csv").open(
                    "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for channel in (1, 2):
                    for value in range(1, 41):
                        writer.writerow({"role": "transition", "channel": channel,
                                         "actuated_channel": channel, "valid": 1,
                                         "clamped": 0,
                                         "predicted_delta_nm": value,
                                         "measured_delta_nm": value + 0.5})
            (folder / "config_snapshot.json").write_text(json.dumps({
                "mirror_actuators": 5, "calibration_set_id": "set-a",
                "channels": [{"channel": 1}, {"channel": 2}],
            }), encoding="utf-8")
            report = analyse_session(folder)
            self.assertEqual(report["calibration_set_id"], "set-a")
            self.assertTrue((folder / "adaptive_settling_ch1.json").is_file())
            self.assertTrue((folder / "adaptive_settling_ch2.json").is_file())
            self.assertTrue((folder / "adaptive_settling_report.md").is_file())
