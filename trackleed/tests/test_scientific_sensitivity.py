from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.reconstruct_publication import main as reconstruction_main
from trackleed.analysis.scientific_sensitivity import audit_width_trigger
from trackleed.analysis.scientific_tracer_records import (
    CandidateRecord,
    EvaluationRecord,
    read_csv,
)
from trackleed.publication.scientific_sensitivity import run_sensitivity
from trackleed.publication.scientific_panel import run_central_panel


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_PANEL = REPO_ROOT / "manuscript" / "scientific_panel" / "frozen"
MEASUREMENT_CONTRACT = (
    REPO_ROOT / "manuscript" / "scientific_baseline" / "measurement_contract.json"
)


class ScientificSensitivityTests(unittest.TestCase):
    def test_width_trigger_ignores_methods_without_finite_candidates(self) -> None:
        candidate = CandidateRecord(  # type: ignore[typeddict-item]
            record_key="near-boundary",
            case_id="scan_a_pair_1",
            method="easyleed",
            finite_candidate=True,
            sigma_x=0.6,
            sigma_y=4.0,
            opportunity_key="opportunity",
        )
        evaluation = EvaluationRecord(  # type: ignore[typeddict-item]
            record_key="near-boundary",
            full_rejection_causes="",
        )
        contract = json.loads(MEASUREMENT_CONTRACT.read_text(encoding="utf-8"))

        result = audit_width_trigger([candidate], [evaluation], contract)

        self.assertEqual(result.triggered_methods, ("easyleed",))
        self.assertTrue(result.run_width_sensitivity)

    def test_public_runner_writes_validated_sensitivity_assessment(self) -> None:
        with TemporaryDirectory() as temporary:
            publication_output = Path(temporary) / "publication-output"
            output = publication_output / "sensitivity"

            run_central_panel(REPO_ROOT, FROZEN_PANEL.parent, publication_output)
            report = run_sensitivity(REPO_ROOT, FROZEN_PANEL.parent, output)

            self.assertEqual(report.variant_count, 15)
            self.assertEqual(report.unique_roi_candidate_set_count, 3)
            self.assertEqual(report.candidate_count, 4860)
            self.assertEqual(report.evaluation_count, 24300)
            self.assertFalse(report.width_sensitivity_triggered)
            self.assertEqual(report.width_contract_count, 0)
            self.assertTrue(report.central_variant_matches_frozen_record)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "candidate_records.csv",
                    "evaluation_records.csv",
                    "sensitivity_results.csv",
                    "rejection_cause_counts.csv",
                    "gate_margin_audit.csv",
                    "summary.json",
                    "provenance.json",
                    "output_manifest.json",
                    "validation_report.json",
                },
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["variants"]), 15)
            self.assertEqual(
                summary["conditional_width_sensitivity"]["decision"],
                "not_run_no_trigger",
            )
            self.assertEqual(
                summary["conditional_width_sensitivity"]["triggered_methods"],
                [],
            )
            self.assertEqual(
                set(summary["variants"]),
                {
                    "roi_32_proposal_12",
                    "roi_32_proposal_15",
                    "roi_32_proposal_18",
                    "roi_48_proposal_12",
                    "roi_48_proposal_15",
                    "roi_48_proposal_18",
                    "roi_64_proposal_12",
                    "roi_64_proposal_15",
                    "roi_64_proposal_18",
                    "branch_tolerance_12",
                    "branch_tolerance_24",
                    "r_squared_0p80",
                    "r_squared_0p90",
                    "amplitude_multiplier_0p75",
                    "amplitude_multiplier_1p25",
                },
            )
            for variant in summary["variants"].values():
                for method in variant["methods"].values():
                    self.assertEqual(
                        method["physics_baseline_visible_opportunities"], 221
                    )
                    self.assertEqual(
                        method["post_gap_branch_consistent_recovery"][
                            "evaluated_gap_count"
                        ],
                        24,
                    )
                    self.assertEqual(
                        method["geometry_inconsistency_denominator"],
                        method["accepted_measurements_all_frames"],
                    )
                    self.assertEqual(
                        method["pair_residual_denominator"],
                        method["jointly_consistent_pair_frames"],
                    )

            arguments = [
                "--workspace-root",
                str(REPO_ROOT),
                "--panel-package",
                str(FROZEN_PANEL.parent),
                "--output",
                str(publication_output),
                "--validate",
            ]
            self.assertEqual(reconstruction_main(arguments), 0)
            summary["tampered"] = True
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(reconstruction_main(arguments), 2)

    def test_width_trigger_is_audited_per_method_and_records_no_trigger(
        self,
    ) -> None:
        _, candidates = read_csv(
            FROZEN_PANEL / "expected_candidate_records.csv", CandidateRecord
        )
        _, evaluations = read_csv(
            FROZEN_PANEL / "expected_evaluation_records.csv", EvaluationRecord
        )
        contract = json.loads(MEASUREMENT_CONTRACT.read_text(encoding="utf-8"))

        result = audit_width_trigger(candidates, evaluations, contract)

        self.assertFalse(result.run_width_sensitivity)
        self.assertEqual(result.triggered_methods, ())
        self.assertEqual(
            {row["method"] for row in result.method_records},
            {"easyleed", "locotrack", "ostrack", "dam4sam", "lk"},
        )
        for row in result.method_records:
            self.assertEqual(
                row["near_width_boundary_threshold_count"],
                {
                    "dam4sam": 16,
                    "easyleed": 16,
                    "lk": 13,
                    "locotrack": 17,
                    "ostrack": 17,
                }[row["method"]],
            )
            self.assertEqual(row["width_only_rejection_threshold_count"], 12)
            self.assertFalse(row["gate_margin_triggered"])
            self.assertFalse(row["width_only_rejection_triggered"])


if __name__ == "__main__":
    unittest.main()
