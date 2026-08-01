from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from trackleed.publication.scientific_sensitivity_evidence import (
    build_table_s6,
    build_table_s6_artifact,
)
from trackleed.publication.scientific_sensitivity_figure import (
    build_figure_s2_artifact,
)
from trackleed.publication.scientific_claim_adjudication import (
    build_claim_adjudication,
    confirm_claim_adjudication,
)
from trackleed.publication.scientific_sensitivity_bundle import (
    validate_sensitivity_publication_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_ROOT = REPO_ROOT / "manuscript" / "tables" / "table_s6_measurement_sensitivity"
FIGURE_ROOT = (
    REPO_ROOT / "manuscript" / "figures" / "figure_s2_measurement_sensitivity"
)
ADJUDICATION_ROOT = REPO_ROOT / "manuscript" / "claim_adjudication"


class ScientificSensitivityEvidenceTests(unittest.TestCase):
    def test_table_s6_retains_full_precision_and_conditional_denominators(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"

            report = build_table_s6(TABLE_ROOT / "source_data", output)

            self.assertEqual(report.result_count, 375)
            self.assertEqual(report.method_result_count, 75)
            self.assertEqual(report.case_result_count, 300)
            with (output / "table_s6_measurement_sensitivity.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            central_locotrack = next(
                row
                for row in rows
                if row["variant_id"] == "roi_48_proposal_15"
                and row["scope"] == "method"
                and row["method"] == "locotrack"
            )
            self.assertEqual(
                central_locotrack["median_absolute_pair_distance_residual_px"],
                "1.8366454823238882",
            )
            self.assertEqual(central_locotrack["pair_residual_denominator"], "91")
            rendered = (output / "table_s6_measurement_sensitivity.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("**Table S6.", rendered)
            self.assertIn("fixed denominator 221", rendered)
            self.assertIn("fixed denominator 24", rendered)
            self.assertIn("method- and variant-specific conditional n", rendered)
            self.assertNotIn("overall ranking", rendered.lower())

    def test_table_s6_is_deterministic_and_rejects_changed_source_data(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_output = root / "first"
            second_output = root / "second"
            build_table_s6(TABLE_ROOT / "source_data", first_output)
            build_table_s6(TABLE_ROOT / "source_data", second_output)
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in first_output.iterdir()
                },
                {
                    path.name: path.read_bytes()
                    for path in second_output.iterdir()
                },
            )

            changed_source = root / "changed-source"
            shutil.copytree(TABLE_ROOT / "source_data", changed_source)
            results_path = changed_source / "sensitivity_results.csv"
            content = results_path.read_text(encoding="utf-8")
            results_path.write_text(
                content.replace("1.8366454823238882", "9.8366454823238882", 1),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                build_table_s6(changed_source, root / "changed-output")

    def test_table_s6_manifest_covers_sources_code_and_outputs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            report = build_table_s6_artifact(
                REPO_ROOT,
                TABLE_ROOT / "source_data",
                root / "output",
                root / "manifest.json",
            )

            self.assertEqual(report.result_count, 375)
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["table"], "Table S6")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                set(manifest["outputs_sha256"]),
                {
                    "table_s6_measurement_sensitivity.csv",
                    "table_s6_measurement_sensitivity.md",
                },
            )
            self.assertEqual(
                set(manifest["implementation_sha256"]),
                {
                    "trackleed/publication/scientific_sensitivity_evidence.py",
                    "trackleed/publication/scientific_sensitivity_sources.py",
                },
            )

    def test_figure_s2_covers_matrix_ofat_outcomes_and_gate_audit(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            report = build_figure_s2_artifact(
                REPO_ROOT,
                FIGURE_ROOT / "source_data",
                root / "output",
                root / "manifest.json",
            )

            self.assertEqual(report.variant_count, 15)
            self.assertEqual(report.roi_matrix_variant_count, 9)
            self.assertEqual(report.ofat_variant_count, 6)
            self.assertEqual(report.method_count, 5)
            self.assertEqual(report.gate_audit_record_count, 25)
            self.assertFalse(report.width_sensitivity_triggered)
            self.assertEqual(
                {path.name for path in (root / "output").iterdir()},
                {
                    "figure_s2_measurement_sensitivity.png",
                    "figure_s2_measurement_sensitivity.pdf",
                },
            )
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["panel_order"],
                [
                    "coverage",
                    "recovery",
                    "geometry_inconsistency",
                    "conditional_pair_residual",
                    "gate_margin_audit",
                ],
            )
            self.assertEqual(manifest["fixed_denominators"], [221, 24])
            self.assertEqual(
                manifest["conditional_denominators"],
                ["accepted fits", "jointly consistent pair frames"],
            )

    def test_figure_s2_is_deterministic_and_rejects_changed_gate_audit(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("first", "second"):
                build_figure_s2_artifact(
                    REPO_ROOT,
                    FIGURE_ROOT / "source_data",
                    root / name,
                    root / f"{name}.json",
                )
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in (root / "first").iterdir()
                },
                {
                    path.name: path.read_bytes()
                    for path in (root / "second").iterdir()
                },
            )

            changed_source = root / "changed-source"
            shutil.copytree(FIGURE_ROOT / "source_data", changed_source)
            audit_path = changed_source / "gate_margin_audit.csv"
            content = audit_path.read_text(encoding="utf-8")
            audit_path.write_text(
                content.replace("method,,easyleed,313,5,16", "method,,easyleed,313,6,16", 1),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                build_figure_s2_artifact(
                    REPO_ROOT,
                    changed_source,
                    root / "changed-output",
                    root / "changed-manifest.json",
                )

    def test_claim_adjudication_combines_metadata_and_sensitivity_constraints(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "adjudication"

            report = build_claim_adjudication(
                workspace_root=REPO_ROOT,
                contract_path=ADJUDICATION_ROOT / "adjudication_contract.json",
                sensitivity_source_directory=ADJUDICATION_ROOT / "source_data",
                metadata_directory=REPO_ROOT / "manuscript" / "experimental_metadata",
                claim_map_path=REPO_ROOT / "manuscript" / "claim-evidence-map.md",
                output_directory=output,
            )

            self.assertEqual(report.claim_count, 9)
            self.assertEqual(report.preserved_count, 4)
            self.assertEqual(report.narrowed_count, 5)
            self.assertEqual(report.removed_count, 0)
            self.assertEqual(report.author_checkpoint_status, "pending")
            boundary = json.loads(
                (output / "final_claim_boundary.json").read_text(encoding="utf-8")
            )
            by_id = {record["claim_id"]: record for record in boundary["claims"]}
            self.assertEqual(set(by_id), {f"C{index}" for index in range(1, 10)})
            self.assertEqual(by_id["C3"]["decision"], "narrowed")
            self.assertEqual(by_id["C4"]["decision"], "narrowed")
            self.assertTrue(
                all(record["metadata_constraint_ids"] for record in by_id.values())
            )
            self.assertTrue(
                all(record["sensitivity_basis"] for record in by_id.values())
            )
            self.assertFalse(boundary["central_contract_changed"])
            self.assertFalse(boundary["capture_basin_triggered"])
            self.assertEqual(
                boundary["source_links"]["sensitivity_claim_evidence"][
                    "sha256"
                ],
                hashlib.sha256(
                    (output / "sensitivity_claim_evidence.json").read_bytes()
                ).hexdigest(),
            )
            checkpoint = json.loads(
                (output / "author_checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["status"], "pending")
            self.assertEqual(
                checkpoint["boundary_sha256"],
                hashlib.sha256(
                    (output / "final_claim_boundary.json").read_bytes()
                ).hexdigest(),
            )

    def test_claim_adjudication_rejects_a_fabricated_author_confirmation(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "adjudication"
            arguments = {
                "workspace_root": REPO_ROOT,
                "contract_path": ADJUDICATION_ROOT / "adjudication_contract.json",
                "sensitivity_source_directory": ADJUDICATION_ROOT / "source_data",
                "metadata_directory": REPO_ROOT / "manuscript" / "experimental_metadata",
                "claim_map_path": REPO_ROOT / "manuscript" / "claim-evidence-map.md",
                "output_directory": output,
            }
            build_claim_adjudication(**arguments)
            checkpoint_path = output / "author_checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["status"] = "confirmed"
            checkpoint_path.write_text(
                json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Author checkpoint"):
                build_claim_adjudication(**arguments)

    def test_claim_boundary_ignores_non_claim_status_notes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            claim_map = root / "claim-evidence-map.md"
            shutil.copyfile(
                REPO_ROOT / "manuscript" / "claim-evidence-map.md",
                claim_map,
            )
            arguments = {
                "workspace_root": REPO_ROOT,
                "contract_path": ADJUDICATION_ROOT / "adjudication_contract.json",
                "sensitivity_source_directory": ADJUDICATION_ROOT / "source_data",
                "metadata_directory": REPO_ROOT / "manuscript" / "experimental_metadata",
                "claim_map_path": claim_map,
            }
            build_claim_adjudication(
                **arguments,
                output_directory=root / "first",
            )
            claim_map.write_text(
                claim_map.read_text(encoding="utf-8")
                + "\nCheckpoint status note changed.\n",
                encoding="utf-8",
            )
            build_claim_adjudication(
                **arguments,
                output_directory=root / "second",
            )

            self.assertEqual(
                (root / "first" / "final_claim_boundary.json").read_bytes(),
                (root / "second" / "final_claim_boundary.json").read_bytes(),
            )

    def test_author_confirmation_is_complete_and_boundary_bound(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "adjudication"
            arguments = {
                "workspace_root": REPO_ROOT,
                "contract_path": ADJUDICATION_ROOT / "adjudication_contract.json",
                "sensitivity_source_directory": ADJUDICATION_ROOT / "source_data",
                "metadata_directory": REPO_ROOT / "manuscript" / "experimental_metadata",
                "claim_map_path": REPO_ROOT / "manuscript" / "claim-evidence-map.md",
                "output_directory": output,
            }
            build_claim_adjudication(**arguments)

            confirm_claim_adjudication(
                output,
                confirmed_by="publication author via test HITL checkpoint",
                confirmed_date="2026-08-01",
            )
            report = build_claim_adjudication(**arguments)

            self.assertEqual(report.author_checkpoint_status, "confirmed")
            checkpoint = json.loads(
                (output / "author_checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["status"], "confirmed")
            self.assertTrue(all(checkpoint["attestations"].values()))
            self.assertEqual(
                checkpoint["boundary_sha256"],
                hashlib.sha256(
                    (output / "final_claim_boundary.json").read_bytes()
                ).hexdigest(),
            )

    def test_publication_bundle_builds_only_supplementary_sensitivity_items(
        self,
    ) -> None:
        from trackleed.publication.scientific_sensitivity_bundle import (
            build_sensitivity_publication_bundle,
        )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            supplement = root / "supplementary-material.md"
            supplement.write_text("# Supplementary material\n", encoding="utf-8")

            report = build_sensitivity_publication_bundle(
                workspace_root=REPO_ROOT,
                sensitivity_output_directory=(
                    REPO_ROOT
                    / "manuscript"
                    / "scientific_panel"
                    / "output"
                    / "sensitivity"
                ),
                table_root=root / "table-s6",
                figure_root=root / "figure-s2",
                adjudication_root=root / "claim-adjudication",
                supplementary_material_path=supplement,
                adjudication_contract_path=(
                    ADJUDICATION_ROOT / "adjudication_contract.json"
                ),
            )

            self.assertEqual(report.table_s6.result_count, 375)
            self.assertEqual(report.figure_s2.variant_count, 15)
            self.assertEqual(report.claim_adjudication.claim_count, 9)
            rendered = supplement.read_text(encoding="utf-8")
            self.assertEqual(rendered.count("<!-- BEGIN TABLE S6 -->"), 1)
            self.assertEqual(rendered.count("<!-- END TABLE S6 -->"), 1)
            self.assertIn("method- and variant-specific conditional n", rendered)
            figure_two_text = "\n".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in sorted(
                    (
                        REPO_ROOT
                        / "manuscript"
                        / "figures"
                        / "figure02_controlled_equivariance"
                    ).rglob("*")
                )
                if path.is_file() and path.suffix in {".json", ".md", ".py"}
            ).lower()
            self.assertNotIn("measurement-window sensitivity", figure_two_text)

    def test_committed_publication_bundle_validates_without_rewriting(self) -> None:
        artifact_roots = (TABLE_ROOT, FIGURE_ROOT, ADJUDICATION_ROOT)
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for root in artifact_roots
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }

        report = validate_sensitivity_publication_bundle(
            workspace_root=REPO_ROOT
        )

        self.assertEqual(report.table_s6.result_count, 375)
        self.assertEqual(report.figure_s2.variant_count, 15)
        self.assertEqual(
            report.claim_adjudication.author_checkpoint_status,
            "confirmed",
        )
        self.assertEqual(
            before,
            {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for root in artifact_roots
                for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            },
        )


if __name__ == "__main__":
    unittest.main()
