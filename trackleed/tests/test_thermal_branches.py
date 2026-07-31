import unittest

from trackleed.analysis.thermal_branches import (
    classify_thermal_branches,
    normalize_thermal_branch,
)


class ThermalBranchTests(unittest.TestCase):
    def test_peak_and_hold_are_assigned_to_heating(self) -> None:
        branches = classify_thermal_branches(
            frame_temperatures={"1": 20.0, "2": 100.0, "3": 100.0, "4": 30.0},
            frame_orders={"1": 0, "2": 1, "3": 2, "4": 3},
        )

        self.assertEqual(
            branches,
            {"1": "heating", "2": "heating", "3": "heating", "4": "cooling"},
        )

    def test_stored_peak_hold_label_is_normalized(self) -> None:
        self.assertEqual(normalize_thermal_branch("peak_hold"), "heating")
        self.assertEqual(normalize_thermal_branch("Heating/Hold"), "heating")


if __name__ == "__main__":
    unittest.main()
