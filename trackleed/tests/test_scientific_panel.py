from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from trackleed.publication.scientific_panel import run_central_panel
from trackleed.publication.scientific_tracer import evaluate_candidates


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_PANEL = REPO_ROOT / "manuscript" / "scientific_panel"
METHODS = {"easyleed", "locotrack", "ostrack", "dam4sam", "lk"}
CASES = {
    "scan_a_pair_1",
    "scan_a_pair_2",
    "scan_b_pair_1",
    "scan_b_pair_2",
}


class ScientificPanelTests(unittest.TestCase):
    def test_headless_panel_reproduces_every_method_case_and_fixed_denominator(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "panel-output"

            report = run_central_panel(REPO_ROOT, FROZEN_PANEL, output)

            self.assertEqual(report.candidate_count, 1620)
            self.assertEqual(report.evaluation_count, 1620)
            self.assertEqual(report.case_count, 4)
            self.assertEqual(report.method_count, 5)
            self.assertEqual(report.opportunity_reference_count, 221)
            self.assertEqual(report.gap_reference_count, 24)
            self.assertTrue(report.matches_frozen_central_record)

            with (output / "candidate_records.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                candidates = list(csv.DictReader(handle))
            with (output / "evaluation_records.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                evaluations = list(csv.DictReader(handle))

            self.assertEqual({row["method"] for row in candidates}, METHODS)
            self.assertEqual({row["case_id"] for row in candidates}, CASES)
            self.assertEqual({row["branch"] for row in candidates}, {"B1", "B2"})
            self.assertEqual(
                len({row["image_artifact_id"] for row in candidates}), 81
            )
            self.assertEqual(
                {
                    row["measurement_status"]
                    for row in evaluations
                },
                {
                    "proposal_unavailable",
                    "roi_invalid",
                    "gaussian_rejected",
                    "geometry_inconsistent",
                    "branch_consistent",
                },
            )

            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )["central_result"]
            self.assertEqual(summary["reference_sets"]["opportunities"]["n"], 221)
            self.assertEqual(summary["reference_sets"]["gaps"]["n"], 24)
            for method in METHODS:
                aggregate = summary["methods"][method]
                self.assertEqual(
                    aggregate["physics_baseline_visible_opportunities"], 221
                )
                self.assertEqual(
                    aggregate["post_gap_recovery"]["evaluated_gap_count"], 24
                )
                self.assertEqual(
                    aggregate["geometry_inconsistency_denominator"],
                    aggregate["accepted_measurements_all_frames"],
                )
                self.assertEqual(
                    aggregate["pair_residual_denominator"],
                    aggregate["jointly_consistent_pair_frames"],
                )

            self.assertTrue((output / "case_results.csv").is_file())
            self.assertTrue((output / "aggregate_method_results.csv").is_file())

    def test_pre_fit_failures_and_every_gate_have_distinct_outcomes(self) -> None:
        contract = json.loads(
            (
                REPO_ROOT
                / "manuscript"
                / "scientific_baseline"
                / "measurement_contract.json"
            ).read_text(encoding="utf-8")
        )["central_contract"]
        base: dict[str, object] = {
            "record_key": "candidate",
            "case_id": "scan_a_pair_1",
            "method": "locotrack",
            "frame": 10,
            "energy_ev": 93.5,
            "branch": "B1",
            "physics_baseline_status": "visible_fit",
            "opportunity_key": "opportunity",
            "proposal_available": True,
            "roi_valid": True,
            "solver_status": "completed",
            "finite_candidate": True,
            "proposal_offset_px": 2.0,
            "r_squared": 0.90,
            "amplitude": 0.02,
            "sigma_x": 4.0,
            "sigma_y": 5.0,
            "branch_error_px": 3.0,
        }
        scenarios = {
            "proposal_unavailable": {
                "proposal_available": False,
                "roi_valid": False,
                "solver_status": "not_run",
                "finite_candidate": False,
            },
            "roi_invalid": {"roi_valid": False, "finite_candidate": False},
            "solver_failure": {
                "solver_status": "failed",
                "finite_candidate": False,
            },
            "non_finite_parameters": {"finite_candidate": False},
            "proposal_to_fit": {"proposal_offset_px": 16.0},
            "r_squared": {"r_squared": 0.84},
            "amplitude": {"amplitude": 0.011},
            "width": {"sigma_x": 0.5},
            "physics_branch_consistency": {"branch_error_px": 19.0},
        }

        for cause, changes in scenarios.items():
            with self.subTest(cause=cause):
                candidate = {**base, **changes}
                evaluation = evaluate_candidates([candidate], contract)[0]
                self.assertEqual(evaluation["first_rejection_cause"], cause)
                self.assertEqual(evaluation["full_rejection_causes"], cause)
                if cause == "physics_branch_consistency":
                    self.assertEqual(
                        evaluation["measurement_status"], "geometry_inconsistent"
                    )
                    self.assertTrue(evaluation["accepted_fit"])
                elif cause in {
                    "proposal_unavailable",
                    "roi_invalid",
                    "solver_failure",
                    "non_finite_parameters",
                }:
                    self.assertEqual(evaluation["measurement_status"], cause)
                    self.assertFalse(evaluation["accepted_fit"])
                else:
                    self.assertEqual(
                        evaluation["measurement_status"], "gaussian_rejected"
                    )
                    self.assertFalse(evaluation["accepted_fit"])


if __name__ == "__main__":
    unittest.main()
