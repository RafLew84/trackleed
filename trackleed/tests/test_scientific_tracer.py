from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from trackleed.publication.scientific_tracer import (
    TracerValidationError,
    evaluate_candidates,
    run_tracer,
    validate_tracer_output,
)
from scripts.run_publication_tracer import main as tracer_main


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_TRACER = REPO_ROOT / "manuscript" / "scientific_tracer"


class ScientificTracerTests(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _snapshot(directory: Path) -> dict[str, bytes]:
        return {
            path.relative_to(directory).as_posix(): path.read_bytes()
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }

    def _refresh_output_integrity(self, output: Path, changed_file: str) -> None:
        changed_path = output / changed_file
        output_manifest_path = output / "output_manifest.json"
        output_manifest = json.loads(
            output_manifest_path.read_text(encoding="utf-8")
        )
        output_manifest["files_sha256"][changed_file] = self._sha256(changed_path)
        self._write_json(output_manifest_path, output_manifest)
        report_path = output / "validation_report.json"
        validation_report = json.loads(report_path.read_text(encoding="utf-8"))
        validation_report["output_manifest_sha256"] = self._sha256(
            output_manifest_path
        )
        self._write_json(report_path, validation_report)

    def test_headless_tracer_builds_candidate_and_evaluation_layers(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "tracer-output"

            report = run_tracer(REPO_ROOT, FROZEN_TRACER, output)

            self.assertEqual(report.candidate_count, 70)
            self.assertEqual(report.evaluation_count, 70)
            self.assertEqual(report.branch_consistent_count, 44)
            self.assertEqual(report.evaluated_gap_count, 5)
            self.assertEqual(report.recovered_gap_count, 5)
            self.assertTrue(report.matches_frozen_central_record)
            with (output / "candidate_records.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                candidates = list(csv.DictReader(handle))
            self.assertEqual(len(candidates), 70)
            required = {
                "record_key",
                "image_source_path",
                "image_sha256",
                "proposal_source_path",
                "proposal_sha256",
                "crop_x0",
                "crop_y0",
                "crop_x1",
                "crop_y1",
                "solver_status",
                "center_x",
                "center_x_error",
                "center_y",
                "center_y_error",
                "amplitude",
                "amplitude_error",
                "sigma_x",
                "sigma_x_error",
                "sigma_y",
                "sigma_y_error",
                "proposal_offset_px",
                "branch_error_px",
                "proposal_to_fit_margin_px",
                "r_squared_margin",
                "amplitude_margin",
                "sigma_x_lower_margin_px",
                "sigma_x_upper_margin_px",
                "sigma_y_lower_margin_px",
                "sigma_y_upper_margin_px",
                "branch_margin_px",
                "missing_state",
            }
            self.assertTrue(required.issubset(candidates[0]))
            completed = [
                row for row in candidates if row["solver_status"] == "completed"
            ]
            self.assertTrue(completed)
            self.assertTrue(all(row["center_x_error"] for row in completed))
            self.assertTrue((output / "evaluation_records.csv").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "provenance.json").is_file())
            self.assertTrue((output / "validation_report.json").is_file())
            self.assertTrue((output / "output_manifest.json").is_file())
            provenance = json.loads(
                (output / "provenance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["execution_device"], "cpu")
            self.assertEqual(
                set(provenance["implementation_sources"]),
                {"fwhm_metrics", "gaussian_fitter", "tracer"},
            )
            self.assertEqual(
                provenance["implementation_sources"]["gaussian_fitter"]["path"],
                "trackleed/analysis/peak_analyzer.py",
            )
            self.assertEqual(
                provenance["runtime"],
                {
                    "numpy": "2.2.5",
                    "python": "3.12.9",
                    "scikit-image": "0.25.2",
                    "scipy": "1.15.2",
                    "tifffile": "2025.3.30",
                },
            )
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["reader_facing"],
                {
                    "conditional_pair_residual_median": "1.983 px",
                    "coverage": "44/44 (100.0%)",
                    "geometry_inconsistency": "9/57 (15.8%)",
                    "post_gap_recovery": "5/5 (100.0%)",
                },
            )

    def test_command_runs_clean_headless_reconstruction_and_validation(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "tracer-output"

            exit_code = tracer_main(
                [
                    "--workspace-root",
                    str(REPO_ROOT),
                    "--tracer-package",
                    str(FROZEN_TRACER),
                    "--output",
                    str(output),
                    "--clean",
                    "--validate",
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(
                (output / "validation_report.json").read_text(encoding="utf-8")
            )
            self.assertTrue(report["matches_frozen_central_record"])
            self.assertTrue(report["fail_closed"])

    def test_threshold_only_evaluation_reuses_candidate_without_copying_fit(self) -> None:
        measurement_contract = json.loads(
            (
                REPO_ROOT
                / "manuscript"
                / "scientific_baseline"
                / "measurement_contract.json"
            ).read_text(encoding="utf-8")
        )["central_contract"]
        candidate = {
            "record_key": "candidate-1",
            "case_id": "scan_a_pair_1",
            "method": "locotrack",
            "frame": 10,
            "energy_ev": 93.5,
            "branch": "B1",
            "physics_baseline_status": "visible_fit",
            "opportunity_key": "opportunity-1",
            "proposal_available": True,
            "roi_valid": True,
            "solver_status": "completed",
            "finite_candidate": True,
            "proposal_offset_px": 2.0,
            "r_squared": 0.87,
            "amplitude": 0.02,
            "sigma_x": 4.0,
            "sigma_y": 5.0,
            "branch_error_px": 3.0,
            "center_x": 123.0,
            "center_y": 456.0,
        }
        before = deepcopy(candidate)

        central = evaluate_candidates([candidate], measurement_contract)[0]
        stricter = deepcopy(measurement_contract)
        stricter["r_squared_min"] = 0.90
        changed = evaluate_candidates(
            [candidate], stricter, variant_id="r_squared_0p90"
        )[0]

        self.assertEqual(candidate, before)
        self.assertEqual(central["measurement_status"], "branch_consistent")
        self.assertEqual(changed["measurement_status"], "gaussian_rejected")
        self.assertEqual(changed["first_rejection_cause"], "r_squared")
        self.assertEqual(changed["full_rejection_causes"], "r_squared")
        self.assertEqual(changed["record_key"], candidate["record_key"])
        self.assertNotIn("center_x", changed)
        self.assertNotIn("center_y", changed)

    def test_altered_input_candidate_or_contract_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid_output = root / "valid-output"
            run_tracer(REPO_ROOT, FROZEN_TRACER, valid_output)

            changed_input = root / "changed-input-package"
            shutil.copytree(FROZEN_TRACER, changed_input)
            frame_map = changed_input / "frozen" / "frame_images.csv"
            frame_map.write_text(
                frame_map.read_text(encoding="utf-8").replace(
                    "4310ed29c5f94db5aaaff2c5ac77457fa6272084a8cf58ba24d3e97142033ef7",
                    "0" * 64,
                    1,
                ),
                encoding="utf-8",
            )
            input_manifest = json.loads(
                (changed_input / "package_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            input_manifest["files_sha256"]["frozen/frame_images.csv"] = (
                self._sha256(frame_map)
            )
            self._write_json(changed_input / "package_manifest.json", input_manifest)
            before = self._snapshot(changed_input)
            with self.assertRaisesRegex(
                TracerValidationError, "Frozen tracer manifest contract changed"
            ):
                validate_tracer_output(REPO_ROOT, changed_input, valid_output)
            self.assertEqual(self._snapshot(changed_input), before)

            changed_candidate = root / "changed-candidate-output"
            shutil.copytree(valid_output, changed_candidate)
            candidate_path = changed_candidate / "candidate_records.csv"
            with candidate_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                candidates = list(reader)
            candidates[0]["center_x_error"] = str(
                float(candidates[0]["center_x_error"]) + 1.0
            )
            with candidate_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(candidates)
            self._refresh_output_integrity(
                changed_candidate,
                "candidate_records.csv",
            )
            before = self._snapshot(changed_candidate)
            with self.assertRaisesRegex(
                TracerValidationError, "Frozen candidate record changed"
            ):
                validate_tracer_output(REPO_ROOT, FROZEN_TRACER, changed_candidate)
            self.assertEqual(self._snapshot(changed_candidate), before)

            changed_evaluation = root / "changed-evaluation-output"
            shutil.copytree(valid_output, changed_evaluation)
            evaluation_path = changed_evaluation / "evaluation_records.csv"
            with evaluation_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                evaluations = list(reader)
            evaluations[0]["classification"] = "tampered"
            with evaluation_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(evaluations)
            self._refresh_output_integrity(
                changed_evaluation,
                "evaluation_records.csv",
            )
            before = self._snapshot(changed_evaluation)
            with self.assertRaisesRegex(
                TracerValidationError, "Frozen evaluation record changed"
            ):
                validate_tracer_output(REPO_ROOT, FROZEN_TRACER, changed_evaluation)
            self.assertEqual(self._snapshot(changed_evaluation), before)

            changed_contract = root / "changed-contract-package"
            shutil.copytree(FROZEN_TRACER, changed_contract)
            contract_path = changed_contract / "frozen" / "tracer_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["expected_counts"]["candidate_records"] = 69
            self._write_json(contract_path, contract)
            contract_manifest = json.loads(
                (changed_contract / "package_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            contract_manifest["files_sha256"]["frozen/tracer_contract.json"] = (
                self._sha256(contract_path)
            )
            self._write_json(
                changed_contract / "package_manifest.json", contract_manifest
            )
            before = self._snapshot(changed_contract)
            with self.assertRaisesRegex(
                TracerValidationError, "Frozen tracer manifest contract changed"
            ):
                validate_tracer_output(REPO_ROOT, changed_contract, valid_output)
            self.assertEqual(self._snapshot(changed_contract), before)


if __name__ == "__main__":
    unittest.main()
