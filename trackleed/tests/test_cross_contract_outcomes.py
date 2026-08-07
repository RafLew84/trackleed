from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trackleed.analysis.cross_contract_outcomes import _cross_contract_distance
from trackleed.publication.cross_contract_outcomes import (
    CrossContractValidationError,
    build_cross_contract_outcomes,
    validate_cross_contract_outcomes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONTRACT = (
    REPO_ROOT / "manuscript/cross_contract/outcomes/analysis_contract.json"
)


def copy_analysis_contract(package: Path) -> None:
    package.mkdir()
    shutil.copyfile(SOURCE_CONTRACT, package / "analysis_contract.json")
    shutil.copyfile(SOURCE_CONTRACT.parent / "README.md", package / "README.md")


class CrossContractOutcomeTests(unittest.TestCase):
    def test_observing_contracts_require_their_own_finite_position(self) -> None:
        with self.assertRaisesRegex(
            CrossContractValidationError,
            "measurable Gaussian record lacks a finite position",
        ):
            _cross_contract_distance(True, False, None, 1.0, None, None)
        with self.assertRaisesRegex(
            CrossContractValidationError,
            "direct native record lacks a finite raw position",
        ):
            _cross_contract_distance(False, True, None, None, None, 1.0)

    def test_validation_fails_closed_when_analysis_package_is_absent(self) -> None:
        with (
            TemporaryDirectory() as temporary,
            self.assertRaisesRegex(
                CrossContractValidationError,
                "cross-contract analysis contract is unavailable",
            ),
        ):
            validate_cross_contract_outcomes(Path(temporary))

    def test_build_retains_every_opportunity_and_reports_four_states(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "outcomes"
            copy_analysis_contract(package)

            report = build_cross_contract_outcomes(REPO_ROOT, package)

            self.assertEqual(report["branch_frame_matrix_count"], 1620)
            self.assertEqual(report["opportunity_count_per_method"], 324)
            self.assertEqual(
                report["measurability_outcomes_by_method"],
                {
                    "dam4sam": {
                        "both_contracts_observe": 262,
                        "gaussian_only": 10,
                        "neither_contract_observes": 9,
                        "viperleed_only": 43,
                    },
                    "easyleed": {
                        "both_contracts_observe": 222,
                        "gaussian_only": 0,
                        "neither_contract_observes": 19,
                        "viperleed_only": 83,
                    },
                    "lk": {
                        "both_contracts_observe": 205,
                        "gaussian_only": 1,
                        "neither_contract_observes": 18,
                        "viperleed_only": 100,
                    },
                    "locotrack": {
                        "both_contracts_observe": 262,
                        "gaussian_only": 10,
                        "neither_contract_observes": 9,
                        "viperleed_only": 43,
                    },
                    "ostrack": {
                        "both_contracts_observe": 259,
                        "gaussian_only": 10,
                        "neither_contract_observes": 9,
                        "viperleed_only": 46,
                    },
                },
            )

    def test_build_reports_contract_specific_gaps_and_joint_recovery(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "outcomes"
            copy_analysis_contract(package)

            report = build_cross_contract_outcomes(REPO_ROOT, package)

            self.assertEqual(
                report["gaussian_gap_event_count_by_method"],
                {
                    "dam4sam": 23,
                    "easyleed": 22,
                    "lk": 20,
                    "locotrack": 25,
                    "ostrack": 24,
                },
            )

    def test_build_separates_fixed_coverage_from_conditional_out_of_branch(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "outcomes"
            copy_analysis_contract(package)

            report = build_cross_contract_outcomes(REPO_ROOT, package)

            self.assertEqual(report["accepted_out_of_branch_event_count"], 124)
            self.assertEqual(
                report["out_of_branch_classifications"],
                {
                    "branch_model_disagreement": 83,
                    "cross_contract_target_inconsistency": 16,
                    "unresolved": 25,
                },
            )

    def test_build_reports_every_predeclared_sensitivity_variant(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "outcomes"
            copy_analysis_contract(package)

            report = build_cross_contract_outcomes(REPO_ROOT, package)

            self.assertEqual(report["sensitivity_variant_count"], 9)
            self.assertEqual(report["sensitivity_result_count"], 45)
            self.assertEqual(
                report["sensitivity_parameters"],
                {
                    "cross_contract_positional_tolerance_px": [0.5, 1.0, 2.0],
                    "native_significance_threshold": [2.0, 2.5, 3.0],
                },
            )
            self.assertEqual(
                report["sensitivity_classification_envelopes_by_method"],
                {
                    "dam4sam": {
                        "branch_model_disagreement": [17, 26],
                        "cross_contract_target_inconsistency": [2, 11],
                        "unresolved": [8, 8],
                    },
                    "easyleed": {
                        "branch_model_disagreement": [4, 4],
                        "cross_contract_target_inconsistency": [0, 0],
                        "unresolved": [0, 0],
                    },
                    "lk": {
                        "branch_model_disagreement": [7, 13],
                        "cross_contract_target_inconsistency": [0, 6],
                        "unresolved": [1, 1],
                    },
                    "locotrack": {
                        "branch_model_disagreement": [17, 26],
                        "cross_contract_target_inconsistency": [2, 11],
                        "unresolved": [8, 8],
                    },
                    "ostrack": {
                        "branch_model_disagreement": [15, 24],
                        "cross_contract_target_inconsistency": [2, 11],
                        "unresolved": [8, 8],
                    },
                },
            )
            self.assertTrue(report["sensitivity_measurability_counts_stable"])

    def test_build_validates_staging_and_is_build_once(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "outcomes"
            copy_analysis_contract(package)

            built = build_cross_contract_outcomes(REPO_ROOT, package)

            self.assertEqual(built["status"], "validated-cross-contract-outcomes")
            self.assertEqual(
                built,
                validate_cross_contract_outcomes(REPO_ROOT, package),
            )
            with self.assertRaisesRegex(
                FileExistsError,
                "Refusing to replace frozen cross-contract analysis",
            ):
                build_cross_contract_outcomes(REPO_ROOT, package)
            self.assertEqual(
                built["method_denominators"],
                {
                    "dam4sam": {
                        "accepted_gaussian_all_frames": 272,
                        "branch_consistent_on_reference_visible": 216,
                        "out_of_branch_all_frames": 36,
                        "out_of_branch_branch_model_disagreement": 23,
                        "out_of_branch_cross_contract_target_inconsistency": 5,
                        "out_of_branch_unresolved": 8,
                        "reference_visible_opportunities": 221,
                    },
                    "easyleed": {
                        "accepted_gaussian_all_frames": 222,
                        "branch_consistent_on_reference_visible": 214,
                        "out_of_branch_all_frames": 4,
                        "out_of_branch_branch_model_disagreement": 4,
                        "out_of_branch_cross_contract_target_inconsistency": 0,
                        "out_of_branch_unresolved": 0,
                        "reference_visible_opportunities": 221,
                    },
                    "lk": {
                        "accepted_gaussian_all_frames": 206,
                        "branch_consistent_on_reference_visible": 188,
                        "out_of_branch_all_frames": 14,
                        "out_of_branch_branch_model_disagreement": 12,
                        "out_of_branch_cross_contract_target_inconsistency": 1,
                        "out_of_branch_unresolved": 1,
                        "reference_visible_opportunities": 221,
                    },
                    "locotrack": {
                        "accepted_gaussian_all_frames": 272,
                        "branch_consistent_on_reference_visible": 217,
                        "out_of_branch_all_frames": 36,
                        "out_of_branch_branch_model_disagreement": 23,
                        "out_of_branch_cross_contract_target_inconsistency": 5,
                        "out_of_branch_unresolved": 8,
                        "reference_visible_opportunities": 221,
                    },
                    "ostrack": {
                        "accepted_gaussian_all_frames": 269,
                        "branch_consistent_on_reference_visible": 216,
                        "out_of_branch_all_frames": 34,
                        "out_of_branch_branch_model_disagreement": 21,
                        "out_of_branch_cross_contract_target_inconsistency": 5,
                        "out_of_branch_unresolved": 8,
                        "reference_visible_opportunities": 221,
                    },
                },
            )

    def test_validation_fails_closed_when_source_data_drifts(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "outcomes"
            copy_analysis_contract(package)
            built = build_cross_contract_outcomes(REPO_ROOT, package)
            with (package / "branch_frame_matrix.csv").open("ab") as handle:
                handle.write(b"tampered")

            with self.assertRaisesRegex(
                CrossContractValidationError,
                "manifested cross-contract output drift: branch_frame_matrix.csv",
            ):
                validate_cross_contract_outcomes(REPO_ROOT, package)
            self.assertEqual(
                built["viperleed_gap_identities"],
                [
                    "viperleed|scan_a|P2_B2|4|4",
                    "viperleed|scan_a|P2_B2|8|8",
                ],
            )
            self.assertEqual(
                built["gap_comparison_by_method"],
                {
                    "dam4sam": {
                        "both_contracts_recover_at_component_exit": 24,
                        "component_count": 24,
                        "exact_boundary_agreement": 1,
                        "gaussian_only": 22,
                        "shared": 1,
                        "viperleed_only": 1,
                    },
                    "easyleed": {
                        "both_contracts_recover_at_component_exit": 23,
                        "component_count": 24,
                        "exact_boundary_agreement": 0,
                        "gaussian_only": 22,
                        "shared": 0,
                        "viperleed_only": 2,
                    },
                    "lk": {
                        "both_contracts_recover_at_component_exit": 21,
                        "component_count": 22,
                        "exact_boundary_agreement": 0,
                        "gaussian_only": 20,
                        "shared": 0,
                        "viperleed_only": 2,
                    },
                    "locotrack": {
                        "both_contracts_recover_at_component_exit": 26,
                        "component_count": 26,
                        "exact_boundary_agreement": 1,
                        "gaussian_only": 24,
                        "shared": 1,
                        "viperleed_only": 1,
                    },
                    "ostrack": {
                        "both_contracts_recover_at_component_exit": 24,
                        "component_count": 25,
                        "exact_boundary_agreement": 1,
                        "gaussian_only": 23,
                        "shared": 1,
                        "viperleed_only": 1,
                    },
                },
            )


if __name__ == "__main__":
    unittest.main()
