from __future__ import annotations

import csv
from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from trackleed.publication.scientific_baseline import (
    BaselineValidationError,
    build_package,
    validate_package,
)
from scripts.freeze_publication_baseline import main as baseline_cli


METHODS = ("easyleed", "locotrack", "ostrack", "dam4sam", "lk")
CASE_SPECS = (
    ("scan_a_pair_1", "scan_a", "pair_1", 50, 5),
    ("scan_a_pair_2", "scan_a", "pair_2", 55, 7),
    ("scan_b_pair_1", "scan_b", "pair_1", 56, 6),
    ("scan_b_pair_2", "scan_b", "pair_2", 60, 6),
)


class ScientificBaselineContractTests(unittest.TestCase):
    @staticmethod
    def _write_csv(
        path: Path,
        fieldnames: list[str],
        rows: list[dict[str, object]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _create_workspace(self, root: Path) -> Path:
        for series_id in ("scan_a", "scan_b"):
            image_dir = root / "data" / series_id
            image_dir.mkdir(parents=True)
            for frame in range(35):
                (image_dir / f"frame_{frame:03d}.tiff").write_bytes(
                    b"II*\x00frozen-image" + str(frame).encode("ascii")
                )

        cases: list[dict[str, object]] = []
        for case_id, series_id, pair_id, visible_count, gap_count in CASE_SPECS:
            source_dir = root / "sources" / case_id
            baseline_rows = []
            for index in range(70):
                baseline_rows.append(
                    {
                        "frame": index // 2,
                        "energy_ev": 40.0 + index,
                        "branch": "B1" if index % 2 == 0 else "B2",
                        "status": "visible_fit" if index < visible_count else "iv_visibility_gap",
                    }
                )
            baseline_path = source_dir / "scan_measurements.csv"
            self._write_csv(
                baseline_path,
                ["frame", "energy_ev", "branch", "status"],
                baseline_rows,
            )

            gaps = [
                {
                    "branch": "B1" if index % 2 == 0 else "B2",
                    "classification": "iv_visibility_gap",
                    "start_energy_ev": 80.0 + 2 * index,
                    "end_energy_ev": 81.0 + 2 * index,
                    "frame_count": 1,
                }
                for index in range(gap_count)
            ]
            summary_path = source_dir / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "reference_energy_ev": 188.5,
                        "visibility_gaps": gaps,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            proposals: dict[str, str] = {}
            for method in METHODS:
                proposal_path = source_dir / f"{method}_proposals.csv"
                self._write_csv(
                    proposal_path,
                    ["frame", "branch", "native_x", "native_y"],
                    [
                        {
                            "frame": 0,
                            "branch": "B1",
                            "native_x": 10.0,
                            "native_y": 20.0,
                        }
                    ],
                )
                proposals[method] = proposal_path.relative_to(root).as_posix()

            cases.append(
                {
                    "case_id": case_id,
                    "series_id": series_id,
                    "pair_id": pair_id,
                    "raw_images": {
                        "directory": f"data/{series_id}",
                        "pattern": "*.tiff",
                        "expected_count": 35,
                    },
                    "baseline_summary": summary_path.relative_to(root).as_posix(),
                    "baseline_measurements": baseline_path.relative_to(root).as_posix(),
                    "proposals": proposals,
                }
            )

        package_dir = root / "package"
        package_dir.mkdir()
        (package_dir / "source_inventory.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "study": "ag111_principal_stress_test",
                    "cases": cases,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return package_dir

    @staticmethod
    def _snapshot(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    @staticmethod
    def _refresh_package_file_hash(package_dir: Path, relative_path: str) -> None:
        manifest_path = package_dir / "package_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files_sha256"][relative_path] = hashlib.sha256(
            (package_dir / relative_path).read_bytes()
        ).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_build_freezes_all_inputs_and_fixed_opportunities(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            package_dir = self._create_workspace(workspace)

            report = build_package(workspace, package_dir)

            self.assertEqual(report.case_count, 4)
            self.assertEqual(report.method_count, 5)
            self.assertEqual(report.raw_image_count, 70)
            self.assertEqual(report.visible_opportunity_count, 221)
            self.assertEqual(report.evaluable_gap_count, 24)

            input_manifest = json.loads(
                (package_dir / "input_manifest.json").read_text(encoding="utf-8")
            )
            roles = [record["role"] for record in input_manifest["artifacts"]]
            self.assertEqual(roles.count("raw_image"), 70)
            self.assertEqual(roles.count("physics_baseline"), 4)
            self.assertEqual(roles.count("case_contract"), 4)
            self.assertEqual(roles.count("native_proposals"), 20)
            self.assertTrue((package_dir / "opportunities.csv").is_file())
            self.assertTrue((package_dir / "gaps.csv").is_file())
            self.assertTrue((package_dir / "measurement_contract.json").is_file())
            self.assertTrue((package_dir / "package_manifest.json").is_file())
            with (package_dir / "opportunities.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                opportunities = list(csv.DictReader(handle))
            raw_by_id = {
                record["artifact_id"]: record
                for record in input_manifest["artifacts"]
                if record["role"] == "raw_image"
            }
            for opportunity in opportunities:
                image_record = raw_by_id[opportunity["image_artifact_id"]]
                self.assertEqual(
                    opportunity["image_source_path"],
                    image_record["source_path"],
                )
                self.assertEqual(
                    opportunity["image_sha256"],
                    image_record["source_sha256"],
                )

    def test_measurement_contract_freezes_variants_denominators_and_width_semantics(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            package_dir = self._create_workspace(workspace)
            build_package(workspace, package_dir)

            contract = json.loads(
                (package_dir / "measurement_contract.json").read_text(
                    encoding="utf-8"
                )
            )
            central = contract["central_contract"]
            self.assertEqual(
                central,
                {
                    "roi_side_px": 48,
                    "proposal_to_fit_max_px": 15.0,
                    "branch_tolerance_px": 18.0,
                    "r_squared_min": 0.85,
                    "amplitude_reference_min": 0.012,
                    "amplitude_gate_multiplier": 1.0,
                    "center_preference_factor": 0.34,
                    "width_gate": {
                        "sigma_min_px": 0.6,
                        "sigma_max_px": 13.0,
                        "minimum_inclusive": True,
                        "maximum_inclusive": True,
                        "role": "post_fit_publication_acceptance",
                    },
                    "optimizer_hard_bounds": {
                        "sigma_min_px": 0.5,
                        "sigma_max": "roi_axis_length_px",
                        "role": "numerical_solver_constraint",
                        "varied_by_width_sensitivity": False,
                    },
                },
            )

            variants = contract["sensitivity_variants"]
            self.assertEqual(len(variants), 15)
            self.assertEqual(len({variant["variant_id"] for variant in variants}), 15)
            matrix = [
                variant
                for variant in variants
                if variant["family"] == "roi_x_proposal_to_fit"
            ]
            self.assertEqual(
                {
                    (variant["roi_side_px"], variant["proposal_to_fit_max_px"])
                    for variant in matrix
                },
                {
                    (roi, proposal)
                    for roi in (32, 48, 64)
                    for proposal in (12.0, 15.0, 18.0)
                },
            )
            self.assertEqual(
                {
                    variant["variant_id"]
                    for variant in variants
                    if variant["family"] != "roi_x_proposal_to_fit"
                },
                {
                    "branch_tolerance_12",
                    "branch_tolerance_24",
                    "r_squared_0p80",
                    "r_squared_0p90",
                    "amplitude_multiplier_0p75",
                    "amplitude_multiplier_1p25",
                },
            )
            self.assertEqual(
                contract["denominators"]["branch_consistent_coverage"],
                {
                    "kind": "fixed_reference_opportunity_set",
                    "n": 221,
                },
            )
            self.assertEqual(
                contract["denominators"]["post_gap_branch_consistent_recovery"],
                {
                    "kind": "fixed_reference_opportunity_set",
                    "n": 24,
                },
            )
            self.assertEqual(
                contract["comparison_tolerances"],
                {
                    "continuous_values": {"atol": 1e-8, "rtol": 1e-9},
                    "discrete_values": "exact",
                    "reader_facing_values": "exact_after_declared_rounding",
                },
            )
            self.assertEqual(contract["reader_facing_rounding"]["counts"], "exact")
            self.assertEqual(
                contract["reader_facing_rounding"]["pair_distance_residual_px"],
                {"decimal_places": 3},
            )
            finite_candidate = contract["candidate_semantics"]["finite_candidate_fit"]
            self.assertEqual(finite_candidate["evaluated_before"], "acceptance_gates")
            self.assertEqual(
                finite_candidate["all_of"],
                [
                    "proposal_available",
                    "roi_valid",
                    "solver_completed",
                    "finite:center_x",
                    "finite:center_y",
                    "finite:amplitude",
                    "finite:r_squared",
                    "finite:sigma_x",
                    "finite:sigma_y",
                ],
            )
            gate_audit = contract["candidate_semantics"]["gate_margin_audit"]
            self.assertEqual(gate_audit["passing_signed_margin"], ">= 0")
            self.assertEqual(
                gate_audit["width_boundary_distance"],
                "min(abs(sigma - sigma_min), abs(sigma_max - sigma)) "
                "/ (sigma_max - sigma_min)",
            )
            self.assertFalse(
                contract["candidate_semantics"]["width_only_rejection"][
                    "includes_physics_branch_consistency"
                ]
            )

            reference_sets = contract["reference_sets"]
            self.assertEqual(reference_sets["opportunities"]["key_count"], 221)
            self.assertEqual(reference_sets["gaps"]["key_count"], 24)
            self.assertRegex(
                reference_sets["opportunities"]["keys_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertRegex(
                reference_sets["gaps"]["keys_sha256"],
                r"^[0-9a-f]{64}$",
            )
            for variant in variants:
                self.assertEqual(
                    variant["opportunity_set_id"],
                    reference_sets["opportunities"]["set_id"],
                )
                self.assertEqual(
                    variant["gap_set_id"],
                    reference_sets["gaps"]["set_id"],
                )

    def test_build_rejects_any_four_cases_other_than_the_frozen_scan_pair_matrix(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            package_dir = self._create_workspace(workspace)
            inventory_path = package_dir / "source_inventory.json"
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            inventory["cases"][0]["case_id"] = "scan_c_pair_1"
            inventory_path.write_text(
                json.dumps(inventory, indent=2) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "exact frozen case matrix"):
                build_package(workspace, package_dir)

    def test_validation_detects_changed_source_without_mutating_package(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            package_dir = self._create_workspace(workspace)
            build_report = build_package(workspace, package_dir)

            self.assertEqual(validate_package(workspace, package_dir), build_report)
            package_before = self._snapshot(package_dir)
            changed_source = (
                workspace
                / "sources"
                / "scan_a_pair_1"
                / "locotrack_proposals.csv"
            )
            changed_source.write_text(
                changed_source.read_text(encoding="utf-8") + "1,B2,30,40\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                BaselineValidationError,
                "source hash mismatch.*locotrack",
            ):
                validate_package(workspace, package_dir)

            self.assertEqual(self._snapshot(package_dir), package_before)

    def test_validation_rejects_remapped_inventory_even_if_hash_map_is_refreshed(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            package_dir = self._create_workspace(workspace)
            build_package(workspace, package_dir)

            inventory_path = package_dir / "source_inventory.json"
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            inventory["cases"][0]["pair_id"] = "unapproved_pair"
            inventory_path.write_text(
                json.dumps(inventory, indent=2) + "\n",
                encoding="utf-8",
            )
            self._refresh_package_file_hash(package_dir, "source_inventory.json")

            with self.assertRaisesRegex(
                BaselineValidationError,
                "source inventory no longer reproduces the frozen package",
            ):
                validate_package(workspace, package_dir)

    def test_cli_validates_but_refuses_to_rebuild_a_frozen_package(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            package_dir = self._create_workspace(workspace)
            build_package(workspace, package_dir)

            common_args = [
                "--workspace-root",
                str(workspace),
                "--package",
                str(package_dir),
            ]
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(
                    baseline_cli([*common_args, "--validate-only"]),
                    0,
                )
                package_before = self._snapshot(package_dir)
                self.assertEqual(baseline_cli([*common_args, "--build"]), 2)
            self.assertEqual(self._snapshot(package_dir), package_before)
            self.assertIn("Validated scientific-baseline contract", stdout.getvalue())
            self.assertIn("Refusing to replace frozen package", stderr.getvalue())

    def test_frozen_copy_remains_valid_when_its_origin_archive_is_unavailable(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            package_dir = self._create_workspace(workspace)
            report = build_package(workspace, package_dir)
            archived_origin = (
                workspace
                / "sources"
                / "scan_a_pair_1"
                / "locotrack_proposals.csv"
            )
            archived_origin.unlink()

            self.assertEqual(validate_package(workspace, package_dir), report)

    def test_validation_rejects_replaced_opportunity_key_after_hash_refresh(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            package_dir = self._create_workspace(workspace)
            build_package(workspace, package_dir)

            opportunity_path = package_dir / "opportunities.csv"
            with opportunity_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
            rows[0]["opportunity_key"] = "unapproved-opportunity-key"
            self._write_csv(opportunity_path, fieldnames, rows)
            self._refresh_package_file_hash(package_dir, "opportunities.csv")

            with self.assertRaisesRegex(
                BaselineValidationError,
                "opportunity manifest no longer matches frozen physics baseline",
            ):
                validate_package(workspace, package_dir)

    def test_validation_rejects_changed_proposal_schema_without_origin_archive(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            package_dir = self._create_workspace(workspace)
            build_package(workspace, package_dir)
            archived_origin = (
                workspace
                / "sources"
                / "scan_a_pair_1"
                / "locotrack_proposals.csv"
            )
            archived_origin.unlink()

            frozen_path = (
                package_dir
                / "frozen_inputs"
                / "proposals"
                / "scan_a_pair_1"
                / "locotrack.csv"
            )
            self._write_csv(
                frozen_path,
                ["frame", "branch"],
                [{"frame": 0, "branch": "B1"}],
            )
            input_manifest_path = package_dir / "input_manifest.json"
            input_manifest = json.loads(
                input_manifest_path.read_text(encoding="utf-8")
            )
            record = next(
                item
                for item in input_manifest["artifacts"]
                if item["artifact_id"]
                == "native-proposals:scan_a_pair_1:locotrack"
            )
            record["source_sha256"] = hashlib.sha256(
                frozen_path.read_bytes()
            ).hexdigest()
            record["size_bytes"] = frozen_path.stat().st_size
            record["schema"] = {
                "format": "csv",
                "columns": ["frame", "branch"],
                "row_count": 1,
            }
            input_manifest_path.write_text(
                json.dumps(input_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._refresh_package_file_hash(
                package_dir,
                "frozen_inputs/proposals/scan_a_pair_1/locotrack.csv",
            )
            self._refresh_package_file_hash(package_dir, "input_manifest.json")

            with self.assertRaisesRegex(
                BaselineValidationError,
                "native proposal schema changed.*locotrack",
            ):
                validate_package(workspace, package_dir)


if __name__ == "__main__":
    unittest.main()
