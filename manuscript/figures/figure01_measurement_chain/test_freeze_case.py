#!/usr/bin/env python3
"""Contract tests for the representative Figure 1 case."""

from __future__ import annotations

import unittest

import freeze_case


class Figure1RepresentativeCaseTest(unittest.TestCase):
    def test_selection_uses_predeclared_metadata_anchor_and_canonical_spot(self) -> None:
        case = freeze_case.select_representative_case()

        self.assertEqual(case["frame"]["frame_order"], 700)
        self.assertEqual(case["frame"]["frame_id"], 700)
        self.assertEqual(case["frame"]["selection_reason"], "evaluation_range_end")
        self.assertEqual(case["spot_id"], 1)
        self.assertEqual(case["selection_rule"], freeze_case.SELECTION_RULE)

    def test_all_five_proposals_and_common_fits_are_available(self) -> None:
        case = freeze_case.select_representative_case()

        self.assertEqual(
            [row["method"] for row in case["method_outputs"]],
            list(freeze_case.METHOD_ORDER),
        )
        self.assertTrue(
            all(row["proposal_available"] for row in case["method_outputs"])
        )
        self.assertTrue(all(row["fit_valid"] for row in case["method_outputs"]))
        self.assertTrue(
            all(row["rejection_reason"] == "accepted" for row in case["method_outputs"])
        )

    def test_all_methods_share_the_frozen_initialization(self) -> None:
        comparison = freeze_case.select_representative_case()[
            "comparison_contract"
        ]

        self.assertTrue(comparison["same_sequence"])
        self.assertTrue(comparison["same_selected_spot"])
        self.assertTrue(comparison["same_initialization"])
        self.assertEqual(comparison["methods"], list(freeze_case.METHOD_ORDER))
        self.assertEqual(comparison["frame_count"], 701)
        self.assertEqual(comparison["selected_spot_id"], 1)
        self.assertEqual(comparison["seed_frame_order"], 0)
        self.assertEqual(comparison["seed_frame_id"], 0)
        self.assertEqual(
            comparison["point_methods_receive"],
            "Gaussian-refined seed center",
        )
        self.assertEqual(
            comparison["box_and_mask_methods_receive"],
            "48x48 original-pixel bbox centered on the same seed center",
        )
        self.assertAlmostEqual(
            comparison["seed_center_x_original_px"],
            140.795,
        )
        self.assertAlmostEqual(
            comparison["seed_center_y_original_px"],
            712.6157,
        )
        self.assertEqual(
            comparison["shared_bbox_width_original_px"],
            48.0,
        )
        self.assertEqual(
            comparison["shared_bbox_height_original_px"],
            48.0,
        )
        self.assertFalse(comparison["seed_is_ground_truth"])
        self.assertEqual(
            comparison["manual_correction_or_reinitialization"],
            "forbidden",
        )

    def test_targeted_reexport_resolves_native_geometry_gap(self) -> None:
        case = freeze_case.select_representative_case()
        by_method = {
            row["method"]: row for row in case["method_outputs"]
        }

        self.assertTrue(
            by_method["locotrack_base_offline"]["full_native_geometry_archived"]
        )
        self.assertTrue(
            by_method["ostrack_vitb_256_mae_ce"]["full_native_geometry_archived"]
        )
        self.assertTrue(
            by_method["dam4sam_sam21pp_l"]["full_native_geometry_archived"]
        )
        self.assertTrue(
            by_method["lucas_kanade_skip_anchor"][
                "full_native_geometry_archived"
            ]
        )
        self.assertTrue(
            by_method["easyleed_kalman"]["full_native_geometry_archived"]
        )

        geometry = case["native_geometry"]
        self.assertEqual(geometry["status"], "native_geometry_reexported")
        self.assertEqual(geometry["selection"]["target_frame_order"], 700)
        self.assertEqual(geometry["selection"]["spot_id"], 1)
        self.assertEqual(
            set(geometry["methods"]),
            {
                "ostrack_vitb_256_mae_ce",
                "dam4sam_sam21pp_l",
            },
        )
        self.assertEqual(
            geometry["methods"]["ostrack_vitb_256_mae_ce"][
                "geometry_type"
            ],
            "bounding_box",
        )
        self.assertEqual(
            geometry["methods"]["dam4sam_sam21pp_l"]["geometry_type"],
            "binary_mask",
        )
        self.assertEqual(
            by_method["ostrack_vitb_256_mae_ce"][
                "native_geometry_artifact_path"
            ],
            "source_data/representative_native_geometry.json",
        )
        self.assertEqual(
            by_method["dam4sam_sam21pp_l"]["native_geometry_selector"],
            "dam4sam_sam21pp_l",
        )

    def test_committed_source_data_matches_upstream_contract(self) -> None:
        freeze_case.validate()


if __name__ == "__main__":
    unittest.main()
