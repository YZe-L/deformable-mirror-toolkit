import csv
from pathlib import Path
import tempfile
import unittest

from dm_toolkit.hysteresis.fit_profile import fit_profile, read_loop_csv
from dm_toolkit.hysteresis.pi_model import ModifiedPrandtlIshlinskii


class FitProfileCsvTests(unittest.TestCase):
    def _write_csv(self, rows):
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "loop.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(temporary.cleanup)
        return path

    def test_uses_displacement_instead_of_raw_mx_input(self):
        rows = [
            {"bit": bit, "displacement_nm": 10 * index,
             "mx_input_value": 100 + index, "mx_input_unit": "nm",
             "range": 2}
            for index, bit in enumerate((0, 100, 200, 300))
        ]

        data = read_loop_csv(self._write_csv(rows))

        self.assertEqual(data["displacement_nm"], [0.0, 10.0, 20.0, 30.0])
        self.assertEqual(data["measurement_source"], "displacement_nm")
        self.assertEqual(data["loops"], [2, 2, 2, 2])

    def test_falls_back_when_mx_input_is_not_nm(self):
        rows = [
            {"bit": bit, "displacement_nm": 20 * index,
             "mx_input_value": 0.1 * index, "mx_input_unit": "lambda"}
            for index, bit in enumerate((0, 100, 200, 300))
        ]

        data = read_loop_csv(self._write_csv(rows))

        self.assertEqual(data["displacement_nm"], [0.0, 20.0, 40.0, 60.0])
        self.assertEqual(data["measurement_source"], "displacement_nm")

    def test_exact_endpoint_constraint_survives_global_least_squares(self):
        endpoint = 4200.0
        dataset = {
            "bits": [0, 1000, 2000, 3000, 4095, 3000, 2000, 1000, 0],
            "displacement_nm": [0, 650, 1250, 1750, endpoint,
                                1850, 1350, 750, 100],
            "loops": [1] * 9,
            "measurement_source": "displacement_nm",
            "source": "synthetic_loop.csv",
        }

        profile, metrics = fit_profile(
            [dataset], device_id="endpoint_test", degree=3, n_play=2,
            endpoint_displacement_nm=endpoint,
        )

        self.assertAlmostEqual(metrics["span_nm"], endpoint, places=7)
        self.assertAlmostEqual(
            profile["linearized_command"]["maximum_target_displacement_nm"],
            endpoint,
            places=7,
        )
        self.assertEqual(
            profile["calibration"]["endpoint_constraint_nm"], endpoint
        )
        self.assertEqual(
            profile["calibration"]["loading_response_strategy"],
            "ordinary_least_squares",
        )

        model = ModifiedPrandtlIshlinskii(profile["model"])
        maximum_input = model.input_max_v
        for history in (
            (0.2, 0.8, 0.4),
            (0.9, 0.1, 0.7, 0.3),
            (0.5, 0.0, 0.95, 0.25),
        ):
            model.reset()
            for fraction in history:
                model.commit(maximum_input * fraction)
            self.assertAlmostEqual(model.commit(maximum_input), endpoint,
                                   places=7)

    def test_monotonic_constraint_handles_noisy_dead_zone(self):
        endpoint = 1000.0
        dataset = {
            "bits": [0, 400, 800, 1200, 1600, 2200, 3000, 4095,
                     3000, 2200, 1600, 1200, 800, 400, 0],
            # The low-bit response is nearly flat and contains a small local
            # measurement reversal.  The fitted loading curve must remain
            # strictly increasing so that the inverse stays well-defined.
            "displacement_nm": [0, 11, 10, 65, 180, 410, 700, endpoint,
                                755, 470, 230, 100, 42, 26, 14],
            "loops": [1] * 15,
            "measurement_source": "displacement_nm",
            "source": "noisy_dead_zone.csv",
        }

        profile, _ = fit_profile(
            [dataset], device_id="dead_zone_test", degree=5, n_play=5,
            endpoint_displacement_nm=endpoint,
            enforce_monotonic_loading="auto",
        )

        self.assertEqual(
            profile["calibration"]["loading_response_strategy"],
            "monotonic_constrained_dead_zone",
        )

        model = ModifiedPrandtlIshlinskii(profile["model"])
        loading = [model.primary_response(
            model.input_max_v * index / 1023
        ) for index in range(1024)]
        self.assertTrue(all(right > left
                            for left, right in zip(loading, loading[1:])))
        self.assertAlmostEqual(loading[-1], endpoint, places=6)


if __name__ == "__main__":
    unittest.main()
