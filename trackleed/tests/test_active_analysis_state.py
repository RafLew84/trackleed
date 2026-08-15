from __future__ import annotations

import unittest
from typing import cast

from trackleed.logic.active_analysis_state import ActiveAnalysisState
from trackleed.logic.workflows.track_analysis import FitSet
from trackleed.logic.workflows.track_geometry import GeometryResult


class ActiveAnalysisStateTests(unittest.TestCase):
    def test_transitions_keep_selective_invalidation_consistent(self) -> None:
        fit_set = cast(FitSet, object())
        geometry = cast(GeometryResult, object())
        published = ActiveAnalysisState().publish(fit_set, geometry)

        source_changed = published.invalidate_source()
        self.assertIs(source_changed.fit_set, fit_set)
        self.assertIsNone(source_changed.geometry_result)
        self.assertTrue(source_changed.requires_fit_validation)

        geometry_changed = published.invalidate_geometry()
        self.assertIs(geometry_changed.fit_set, fit_set)
        self.assertIsNone(geometry_changed.geometry_result)
        self.assertFalse(geometry_changed.requires_fit_validation)
        self.assertEqual(published.clear(), ActiveAnalysisState())

    def test_invalid_field_combination_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a FitSet"):
            ActiveAnalysisState(requires_fit_validation=True)


if __name__ == "__main__":
    unittest.main()
