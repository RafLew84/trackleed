from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.reconstruct_publication import main as reconstruction_main
from trackleed.analysis.scientific_sensitivity import (
    audit_width_trigger,
    run_width_sensitivity,
)
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
    def test_reconstruction_builds_reader_facing_sensitivity_bundle(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            package = workspace / "panel"
            output = package / "output"
            with (
                patch(
                    "scripts.reconstruct_publication.run_central_panel",
                    return_value=object(),
                ),
                patch(
                    "scripts.reconstruct_publication.run_sensitivity",
                    return_value=object(),
                ),
                patch(
                    "scripts.reconstruct_publication."
                    "build_sensitivity_publication_bundle",
                    return_value=object(),
                ) as build_bundle,
                patch("scripts.reconstruct_publication.asdict", return_value={}),
            ):
                result = reconstruction_main(
                    [
                        "--workspace-root",
                        str(workspace),
                        "--panel-package",
                        str(package),
                        "--output",
                        str(output),
                        "--validate",
                    ]
                )

            self.assertEqual(result, 0)
            build_bundle.assert_called_once_with(
                workspace_root=workspace,
                sensitivity_output_directory=output / "sensitivity",
            )

    def test_reconstruction_validation_does_not_rewrite_publication_bundle(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            package = workspace / "panel"
            output = package / "output"
            output.mkdir(parents=True)
            with (
                patch(
                    "scripts.reconstruct_publication.validate_central_panel_output",
                    return_value=object(),
                ),
                patch(
                    "scripts.reconstruct_publication.validate_sensitivity_output",
                    return_value=object(),
                ),
                patch(
                    "scripts.reconstruct_publication."
                    "validate_sensitivity_publication_bundle",
                    return_value=object(),
                ) as validate_bundle,
                patch(
                    "scripts.reconstruct_publication."
                    "build_sensitivity_publication_bundle",
                    return_value=object(),
                ) as build_bundle,
                patch("scripts.reconstruct_publication.asdict", return_value={}),
            ):
                result = reconstruction_main(
                    [
                        "--workspace-root",
                        str(workspace),
                        "--panel-package",
                        str(package),
                        "--output",
                        str(output),
                        "--validate",
                    ]
                )

            self.assertEqual(result, 0)
            validate_bundle.assert_called_once_with(workspace_root=workspace)
            build_bundle.assert_not_called()

    def test_triggered_width_sweep_reuses_central_and_evaluates_four_additions(
        self,
    ) -> None:
        contract = json.loads(MEASUREMENT_CONTRACT.read_text(encoding="utf-8"))

        with (
            patch(
                "trackleed.analysis.scientific_sensitivity.evaluate_candidates",
                side_effect=lambda _candidates, _contract, *, variant_id: [
                    {"variant_id": variant_id, "record_key": variant_id}
                ],
            ) as evaluate,
            patch(
                "trackleed.analysis.scientific_sensitivity.summarize_variant",
                side_effect=lambda _root, variant, *_args: {"contract": dict(variant)},
            ),
        ):
            evaluations, summaries = run_width_sensitivity(
                REPO_ROOT,
                contract,
                contract["central_contract"],
                [],
                [],
            )

        self.assertEqual(evaluate.call_count, 4)
        self.assertNotIn("width_central", summaries)
        self.assertEqual(
            set(summaries),
            {
                "width_lower_relaxed",
                "width_lower_tight",
                "width_upper_tight",
                "width_upper_relaxed",
            },
        )
        self.assertEqual(
            {row["variant_id"] for row in evaluations},
            set(summaries),
        )

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
            self.assertEqual(
                summary["sensitivity_id"],
                "ag111_five_method_four_case_central_"
                "measurement_window_and_capture_contract_sensitivity",
            )
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

            def resign(changed_artifact: str) -> None:
                manifest_path = output / "output_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["files_sha256"][changed_artifact] = hashlib.sha256(
                    (output / changed_artifact).read_bytes()
                ).hexdigest()
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                validation_path = output / "validation_report.json"
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
                validation["output_manifest_sha256"] = hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest()
                validation_path.write_text(
                    json.dumps(validation, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

            candidate_path = output / "candidate_records.csv"
            with candidate_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = list(rows[0])
            roi_32 = next(row for row in rows if row["roi_side_px"] == "32")
            roi_32["image_sha256"] = "0" * 64
            with candidate_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            resign("candidate_records.csv")
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
