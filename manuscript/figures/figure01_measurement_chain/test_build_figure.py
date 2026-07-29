#!/usr/bin/env python3
"""Contract tests for publication Figure 1."""

from __future__ import annotations

import unittest

import figure_contract


class Figure1RenderContractTest(unittest.TestCase):
    def test_five_methods_keep_the_planned_representation_order(self) -> None:
        rows = figure_contract.load_method_outputs()

        self.assertEqual(
            [row["method"] for row in rows],
            list(figure_contract.METHOD_ORDER),
        )
        self.assertEqual(
            [row["native_representation"] for row in rows],
            ["point", "bounding box", "mask", "point", "position"],
        )
        self.assertTrue(all(row["proposal_available"] for row in rows))
        self.assertTrue(all(row["fit_valid"] for row in rows))
        self.assertTrue(
            all(row["full_native_geometry_archived"] for row in rows)
        )

    def test_all_proposals_construct_the_same_size_measurement_roi(self) -> None:
        rows = figure_contract.load_method_outputs()

        self.assertTrue(
            all(row["roi_size_px"] == figure_contract.ROI_SIZE_PX for row in rows)
        )
        for row in rows:
            self.assertEqual(
                row["roi_x0_original_px"],
                round(row["proposal_x_original_px"]) - 24,
            )
            self.assertEqual(
                row["roi_y0_original_px"],
                round(row["proposal_y_original_px"]) - 24,
            )

    def test_crop_coordinates_and_empirical_native_geometry_are_consistent(
        self,
    ) -> None:
        crop = figure_contract.load_crop_metadata()
        rows = figure_contract.load_method_outputs()
        by_method = {row["method"]: row for row in rows}
        geometry = figure_contract.load_native_geometry()

        ost_bbox = figure_contract.ostrack_bbox_in_crop(geometry, crop)
        self.assertAlmostEqual(ost_bbox[0], 45.74269556999207)
        self.assertAlmostEqual(ost_bbox[1], 44.7045245170593)
        self.assertAlmostEqual(ost_bbox[2], 40.35474732518196)
        self.assertAlmostEqual(ost_bbox[3], 41.586488485336304)

        mask = figure_contract.load_dam4sam_mask()
        self.assertEqual(mask.shape, (256, 256))
        self.assertEqual(int(mask.sum()), 98)
        self.assertTrue(
            geometry["methods"]["dam4sam_sam21pp_l"][
                "trajectory_replay"
            ]["reduced_center_matches_all_available_frames"]
        )
        self.assertAlmostEqual(
            by_method["ostrack_vitb_256_mae_ce"]["proposal_x_original_px"],
            geometry["methods"]["ostrack_vitb_256_mae_ce"][
                "reduced_center_original_px"
            ][0],
        )

    def test_case_separations_match_the_frozen_descriptive_values(self) -> None:
        contract = figure_contract.load_contract()
        rows = figure_contract.load_method_outputs()

        self.assertAlmostEqual(
            figure_contract.maximum_pairwise_separation(
                rows,
                x_field="proposal_x_original_px",
                y_field="proposal_y_original_px",
            ),
            contract["descriptive_geometry"][
                "maximum_pairwise_native_proposal_separation_px"
            ],
        )
        self.assertAlmostEqual(
            figure_contract.maximum_pairwise_separation(
                rows,
                x_field="gaussian_x_original_px",
                y_field="gaussian_y_original_px",
            ),
            contract["descriptive_geometry"][
                "maximum_pairwise_gaussian_center_separation_px"
            ],
        )

    def test_evidence_ladder_has_four_distinct_roles(self) -> None:
        self.assertEqual(
            [item["regime"] for item in figure_contract.EVIDENCE_ROLES],
            [
                "Cu(111) thermal sequence",
                "Known translations",
                "Ag(111) energy scans",
                "SAM2 family",
            ],
        )
        self.assertEqual(
            [item["role"] for item in figure_contract.EVIDENCE_ROLES],
            [
                "Positive control",
                "Equivariance",
                "Motion + I(V) gaps",
                "Memory mechanism",
            ],
        )


if __name__ == "__main__":
    unittest.main()
