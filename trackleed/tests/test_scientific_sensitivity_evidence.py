from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image

from trackleed.publication.scientific_sensitivity_evidence import (
    build_table_s6,
    build_table_s6_artifact,
)
from trackleed.publication.scientific_sensitivity_figure import (
    build_figure_s2_artifact,
)
from trackleed.publication.scientific_sensitivity_sources import (
    freeze_sensitivity_source_data,
    validate_sensitivity_source_data,
)
from trackleed.publication.scientific_claim_adjudication import (
    build_claim_adjudication,
    confirm_claim_adjudication,
)
from trackleed.publication.scientific_sensitivity_bundle import (
    validate_sensitivity_publication_bundle,
)
from trackleed.publication.scientific_sensitivity_paths import (
    SensitivityPublicationPaths,
    sensitivity_source_root,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = sensitivity_source_root(REPO_ROOT)
TABLE_ROOT = REPO_ROOT / "manuscript" / "tables" / "table_s6_measurement_sensitivity"
FIGURE_ROOT = REPO_ROOT / "manuscript" / "figures" / "figure_s2_measurement_sensitivity"
ADJUDICATION_ROOT = REPO_ROOT / "manuscript" / "claim_adjudication"


class ScientificSensitivityEvidenceTests(unittest.TestCase):
    def test_table_s6_retains_full_precision_and_conditional_denominators(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"

            report = build_table_s6(SOURCE_ROOT, output)

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

    def test_table_s6_emits_compact_view_codebook_and_full_precision_exports(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"

            build_table_s6(SOURCE_ROOT, output)

            with (output / "table_s6_method_aggregate_full_precision.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                aggregate_rows = list(csv.DictReader(handle))
            with (output / "table_s6_measurement_sensitivity.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                canonical_rows = list(csv.DictReader(handle))
            with (output / "table_s6_variant_codebook.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                codebook_rows = list(csv.DictReader(handle))
            self.assertEqual(len(aggregate_rows), 75)
            self.assertEqual(len(canonical_rows), 375)
            self.assertEqual(
                aggregate_rows,
                [row for row in canonical_rows if row["scope"] == "method"],
            )
            self.assertEqual(len(codebook_rows), 15)
            self.assertEqual(
                set(codebook_rows[0]),
                {"variant_id", "reader_code", "meaning", "contract_change"},
            )
            self.assertEqual(
                {row["reader_code"] for row in codebook_rows},
                {
                    "R32-P12",
                    "R32-P15",
                    "R32-P18",
                    "R48-P12",
                    "R48-P15*",
                    "R48-P18",
                    "R64-P12",
                    "R64-P15",
                    "R64-P18",
                    "B12",
                    "B24",
                    "R2-80",
                    "R2-90",
                    "A0.75",
                    "A1.25",
                },
            )
            central_locotrack = next(
                row
                for row in aggregate_rows
                if row["variant_id"] == "roi_48_proposal_15"
                and row["method"] == "locotrack"
            )
            self.assertEqual(
                central_locotrack["median_absolute_pair_distance_residual_px"],
                "1.8366454823238882",
            )

            rendered = (output / "table_s6_measurement_sensitivity.md").read_text(
                encoding="utf-8"
            )
            header = next(line for line in rendered.splitlines() if line.startswith("| Variant"))
            self.assertEqual(header.count("|"), 8)
            self.assertNotIn("| Family |", header)
            self.assertEqual(
                sum(line.startswith("| `") for line in rendered.splitlines()),
                75,
            )
            self.assertIn("| `R48-P15*` | LocoTrack |", rendered)
            self.assertIn("1.837 / 5.973 (`n=91`)", rendered)
            self.assertNotIn("roi_48_proposal_15", rendered)
            self.assertIn("**Variant codebook.**", rendered)
            self.assertIn(
                "`R32-P12`: ROI 32 px; proposal-to-fit 12 px.", rendered
            )
            self.assertIn("`R2-80`: R² minimum 0.80.", rendered)
            self.assertIn(
                '<!-- table-s6-row-style reader_code="R48-P15*" '
                'fill="#EAF2F8" non_color_marker="*" -->',
                rendered,
            )

            workbook_path = output / "table_s6_measurement_sensitivity.xlsx"
            with zipfile.ZipFile(workbook_path) as archive:
                workbook = ET.fromstring(archive.read("xl/workbook.xml"))
                namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                sheet_names = [
                    sheet.attrib["name"]
                    for sheet in workbook.findall("m:sheets/m:sheet", namespace)
                ]
                self.assertEqual(sheet_names, ["method_aggregate", "case_records"])
                method_sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
                case_sheet = ET.fromstring(archive.read("xl/worksheets/sheet2.xml"))
                self.assertEqual(
                    len(method_sheet.findall("m:sheetData/m:row", namespace)), 76
                )
                self.assertEqual(
                    len(case_sheet.findall("m:sheetData/m:row", namespace)), 301
                )
                numeric_cell = method_sheet.find(
                    "m:sheetData/m:row[@r='2']/m:c[@r='H2']", namespace
                )
                self.assertIsNotNone(numeric_cell)
                self.assertNotIn("t", numeric_cell.attrib)
                ordinary_cell = method_sheet.find(
                    "m:sheetData/m:row[@r='2']/m:c[@r='A2']", namespace
                )
                central_cell = method_sheet.find(
                    "m:sheetData/m:row[@r='22']/m:c[@r='A22']", namespace
                )
                self.assertIsNotNone(ordinary_cell)
                self.assertIsNotNone(central_cell)
                self.assertNotEqual(
                    ordinary_cell.attrib.get("s"), central_cell.attrib.get("s")
                )
                self.assertIsNone(method_sheet.find(".//m:f", namespace))
                self.assertIsNone(case_sheet.find(".//m:f", namespace))

    def test_table_s6_is_deterministic_and_rejects_changed_source_data(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_output = root / "first"
            second_output = root / "second"
            build_table_s6(SOURCE_ROOT, first_output)
            build_table_s6(SOURCE_ROOT, second_output)
            self.assertEqual(
                {path.name: path.read_bytes() for path in first_output.iterdir()},
                {path.name: path.read_bytes() for path in second_output.iterdir()},
            )

            changed_source = root / "changed-source"
            shutil.copytree(SOURCE_ROOT, changed_source)
            results_path = changed_source / "sensitivity_results.csv"
            content = results_path.read_text(encoding="utf-8")
            results_path.write_text(
                content.replace("1.8366454823238882", "9.8366454823238882", 1),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                build_table_s6(changed_source, root / "changed-output")

    def test_acceptance_validation_requires_the_authenticated_origin(self) -> None:
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            shutil.copytree(SOURCE_ROOT, source)
            provenance_path = source / "source_provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["sources"]["sensitivity_results.csv"]["origin_sha256"] = "0" * 64
            provenance_path.write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            validate_sensitivity_source_data(source)

            with self.assertRaises(ValueError):
                validate_sensitivity_source_data(
                    source,
                    origin_directory=(
                        REPO_ROOT
                        / "manuscript"
                        / "scientific_panel"
                        / "output"
                        / "sensitivity"
                    ),
                )

    def test_frozen_provenance_uses_portable_canonical_origin_paths(self) -> None:
        provenance = json.loads(
            (SOURCE_ROOT / "source_provenance.json").read_text(encoding="utf-8")
        )

        self.assertEqual(
            {
                record["origin_path"].rsplit("/", 1)[0]
                for record in provenance["sources"].values()
            },
            {"manuscript/scientific_panel/output/sensitivity"},
        )

    def test_source_freeze_rejects_a_relocated_origin(self) -> None:
        with TemporaryDirectory(dir=REPO_ROOT) as temporary:
            relocated_origin = Path(temporary) / "sensitivity"

            with self.assertRaisesRegex(ValueError, "canonical output"):
                freeze_sensitivity_source_data(
                    relocated_origin,
                    Path(temporary) / "frozen",
                    workspace_root=REPO_ROOT,
                )

    def test_table_s6_manifest_covers_sources_code_and_outputs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            report = build_table_s6_artifact(
                REPO_ROOT,
                SOURCE_ROOT,
                root / "output",
                root / "manifest.json",
            )

            self.assertEqual(report.result_count, 375)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["table"], "Table S6")
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["xlsx_generator"]["package"], "XlsxWriter")
            self.assertEqual(
                set(manifest["outputs_sha256"]),
                {
                    "table_s6_method_aggregate_full_precision.csv",
                    "table_s6_measurement_sensitivity.csv",
                    "table_s6_measurement_sensitivity.md",
                    "table_s6_measurement_sensitivity.xlsx",
                    "table_s6_variant_codebook.csv",
                },
            )
            self.assertEqual(
                set(manifest["implementation_sha256"]),
                {
                    "trackleed/publication/scientific_sensitivity_evidence.py",
                    "trackleed/publication/scientific_sensitivity_sources.py",
                },
            )

    def test_committed_table_s6_matches_a_clean_rebuild(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_table_s6_artifact(
                REPO_ROOT,
                SOURCE_ROOT,
                root / "output",
                root / "manifest.json",
            )

            self.assertEqual(
                (root / "manifest.json").read_bytes(),
                (TABLE_ROOT / "manifest.json").read_bytes(),
            )
            rebuilt = {
                path.name: path.read_bytes() for path in (root / "output").iterdir()
            }
            committed = {
                path.name: path.read_bytes()
                for path in (TABLE_ROOT / "output").iterdir()
            }
            self.assertEqual(rebuilt, committed)

    def test_figure_s2_covers_matrix_ofat_outcomes_and_gate_audit(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            report = build_figure_s2_artifact(
                REPO_ROOT,
                SOURCE_ROOT,
                root / "output",
                root / "manifest.json",
            )

            self.assertEqual(report.variant_count, 15)
            self.assertEqual(report.roi_matrix_variant_count, 9)
            self.assertEqual(report.ofat_variant_count, 6)
            self.assertEqual(report.method_count, 5)
            self.assertEqual(report.gate_audit_record_count, 25)
            self.assertEqual(report.gate_margin_summary_record_count, 40)
            self.assertFalse(report.width_sensitivity_triggered)
            self.assertEqual(report.part_count, 2)
            self.assertEqual(report.display_item_count, 1)
            self.assertGreaterEqual(report.minimum_effective_font_size_pt, 6.0)
            self.assertGreaterEqual(report.minimum_heatmap_cell_width_mm, 4.5)
            self.assertGreaterEqual(report.minimum_heatmap_cell_height_mm, 4.5)
            self.assertEqual(
                {path.name for path in (root / "output").iterdir()},
                {
                    "figure_s2_measurement_sensitivity_a-d.png",
                    "figure_s2_measurement_sensitivity_a-d.pdf",
                    "figure_s2_measurement_sensitivity_e-f_continued.png",
                    "figure_s2_measurement_sensitivity_e-f_continued.pdf",
                },
            )
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["display_item_count"], 1)
            self.assertTrue(manifest["continued_figure"])
            self.assertEqual(manifest["placement_width_inches"], 6.25)
            self.assertTrue(manifest["codebook_repeated_on_parts"])
            self.assertEqual(
                [(part["label"], part["panels"]) for part in manifest["parts"]],
                [
                    ("Figure S2a–d", ["a", "b", "c", "d"]),
                    ("Figure S2e–f (continued)", ["e", "f"]),
                ],
            )
            self.assertEqual(
                [part["source_size_inches"] for part in manifest["parts"]],
                [[7.2, 10.0], [7.2, 6.0]],
            )
            self.assertEqual(
                manifest["rendering"]["central_variant"],
                {
                    "id": "roi_48_proposal_15",
                    "label": "R48-P15*",
                    "outline": True,
                },
            )
            self.assertGreaterEqual(
                manifest["rendering"]["minimum_effective_font_size_pt"], 6.0
            )
            self.assertTrue(manifest["rendering"]["fixed_denominators_in_cells"])
            self.assertGreaterEqual(
                manifest["rendering"]["minimum_heatmap_cell_width_mm"], 4.5
            )
            self.assertGreaterEqual(
                manifest["rendering"]["minimum_heatmap_cell_height_mm"], 4.5
            )
            self.assertEqual(
                manifest["panel_order"],
                [
                    "coverage",
                    "recovery",
                    "geometry_inconsistency",
                    "conditional_pair_residual",
                    "signed_gate_margins",
                    "width_trigger_audit",
                ],
            )
            self.assertEqual(manifest["fixed_denominators"], [221, 24])
            self.assertEqual(
                manifest["conditional_denominators"],
                ["accepted fits", "jointly consistent pair frames"],
            )
            caption_source = (FIGURE_ROOT / "README.md").read_text(encoding="utf-8")
            self.assertEqual(caption_source.count("**Caption.**"), 1)
            self.assertIn("`Figure S2 (continued)`", caption_source)
            with Image.open(
                root / "output" / "figure_s2_measurement_sensitivity_a-d.png"
            ) as image:
                self.assertEqual(image.size, (2160, 3000))
            with Image.open(
                root
                / "output"
                / "figure_s2_measurement_sensitivity_e-f_continued.png"
            ) as image:
                self.assertEqual(image.size, (2160, 1800))

    def test_figure_s2_is_deterministic_and_rejects_changed_gate_audit(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("first", "second"):
                build_figure_s2_artifact(
                    REPO_ROOT,
                    SOURCE_ROOT,
                    root / name,
                    root / f"{name}.json",
                )
            self.assertEqual(
                {path.name: path.read_bytes() for path in (root / "first").iterdir()},
                {path.name: path.read_bytes() for path in (root / "second").iterdir()},
            )

            changed_source = root / "changed-source"
            shutil.copytree(SOURCE_ROOT, changed_source)
            audit_path = changed_source / "gate_margin_audit.csv"
            content = audit_path.read_text(encoding="utf-8")
            audit_path.write_text(
                content.replace(
                    "method,,easyleed,313,5,16", "method,,easyleed,313,6,16", 1
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                build_figure_s2_artifact(
                    REPO_ROOT,
                    changed_source,
                    root / "changed-output",
                    root / "changed-manifest.json",
                )

    def test_figure_s2_ignores_ambient_matplotlib_style(self) -> None:
        import matplotlib

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_figure_s2_artifact(
                REPO_ROOT,
                SOURCE_ROOT,
                root / "baseline",
                root / "baseline.json",
            )
            with matplotlib.rc_context(
                {
                    "axes.facecolor": "#ff00ff",
                }
            ):
                build_figure_s2_artifact(
                    REPO_ROOT,
                    SOURCE_ROOT,
                    root / "styled",
                    root / "styled.json",
                )

            self.assertEqual(
                {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (root / "baseline").iterdir()
                },
                {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (root / "styled").iterdir()
                },
            )

    def test_committed_figure_s2_matches_a_clean_rebuild(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_figure_s2_artifact(
                REPO_ROOT,
                SOURCE_ROOT,
                root / "output",
                root / "manifest.json",
            )

            self.assertEqual(
                (root / "manifest.json").read_bytes(),
                (FIGURE_ROOT / "manifest.json").read_bytes(),
            )
            rebuilt = {
                path.name: path.read_bytes() for path in (root / "output").iterdir()
            }
            committed = {
                path.name: path.read_bytes()
                for path in (FIGURE_ROOT / "output").iterdir()
            }
            self.assertEqual(rebuilt, committed)

    def test_claim_adjudication_combines_metadata_and_sensitivity_constraints(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "adjudication"

            report = build_claim_adjudication(
                workspace_root=REPO_ROOT,
                contract_path=ADJUDICATION_ROOT / "adjudication_contract.json",
                sensitivity_source_directory=SOURCE_ROOT,
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
            claim_rows = {
                columns[1].strip(): columns
                for line in (REPO_ROOT / "manuscript" / "claim-evidence-map.md")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.startswith("| C")
                for columns in [line.split("|")]
            }
            for claim_id, claim in by_id.items():
                self.assertEqual(
                    claim_rows[claim_id][2].strip(),
                    claim["final_claim"],
                )
                self.assertEqual(
                    claim_rows[claim_id][-2].strip().lower(),
                    claim["decision"],
                )
            self.assertEqual(
                boundary["source_links"]["sensitivity_claim_evidence"]["sha256"],
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
                "sensitivity_source_directory": SOURCE_ROOT,
                "metadata_directory": REPO_ROOT
                / "manuscript"
                / "experimental_metadata",
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
                "sensitivity_source_directory": SOURCE_ROOT,
                "metadata_directory": REPO_ROOT
                / "manuscript"
                / "experimental_metadata",
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
                "sensitivity_source_directory": SOURCE_ROOT,
                "metadata_directory": REPO_ROOT
                / "manuscript"
                / "experimental_metadata",
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

    def test_confirmed_boundary_mismatch_does_not_rewrite_outputs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "adjudication"
            contract_path = root / "adjudication_contract.json"
            shutil.copyfile(
                ADJUDICATION_ROOT / "adjudication_contract.json",
                contract_path,
            )
            arguments = {
                "workspace_root": REPO_ROOT,
                "contract_path": contract_path,
                "sensitivity_source_directory": SOURCE_ROOT,
                "metadata_directory": REPO_ROOT
                / "manuscript"
                / "experimental_metadata",
                "claim_map_path": REPO_ROOT / "manuscript" / "claim-evidence-map.md",
                "output_directory": output,
            }
            build_claim_adjudication(**arguments)
            confirm_claim_adjudication(
                output,
                confirmed_by="publication author via test HITL checkpoint",
                confirmed_date="2026-08-01",
            )
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["claims"][0]["final_claim"] += " Changed."
            contract_path.write_text(
                json.dumps(contract, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Author checkpoint"):
                build_claim_adjudication(**arguments)

            self.assertEqual(
                before,
                {path.name: path.read_bytes() for path in output.iterdir()},
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
                paths=SensitivityPublicationPaths(
                    source_root=root / "source-data",
                    table_root=root / "table-s6",
                    figure_root=root / "figure-s2",
                    adjudication_root=root / "claim-adjudication",
                    supplementary_material=supplement,
                    adjudication_contract=(
                        ADJUDICATION_ROOT / "adjudication_contract.json"
                    ),
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

    def test_publication_bundle_freezes_one_shared_sensitivity_source_package(
        self,
    ) -> None:
        from trackleed.publication.scientific_sensitivity_bundle import (
            build_sensitivity_publication_bundle,
        )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            supplement = root / "supplementary-material.md"
            supplement.write_text("# Supplementary material\n", encoding="utf-8")
            source_root = root / "source-data"
            table_root = root / "table-s6"
            figure_root = root / "figure-s2"
            adjudication_root = root / "claim-adjudication"
            paths = SensitivityPublicationPaths(
                source_root=source_root,
                table_root=table_root,
                figure_root=figure_root,
                adjudication_root=adjudication_root,
                supplementary_material=supplement,
                adjudication_contract=(
                    ADJUDICATION_ROOT / "adjudication_contract.json"
                ),
            )

            build_sensitivity_publication_bundle(
                workspace_root=REPO_ROOT,
                sensitivity_output_directory=(
                    REPO_ROOT
                    / "manuscript"
                    / "scientific_panel"
                    / "output"
                    / "sensitivity"
                ),
                paths=paths,
            )

            self.assertTrue((source_root / "source_provenance.json").is_file())
            self.assertFalse((table_root / "source_data").exists())
            self.assertFalse((figure_root / "source_data").exists())
            self.assertFalse((adjudication_root / "source_data").exists())
            expected_source_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_root.iterdir()
                if path.is_file()
            }
            for manifest_path in (
                table_root / "manifest.json",
                figure_root / "manifest.json",
            ):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    manifest["source_data_sha256"],
                    expected_source_hashes,
                )
            report = validate_sensitivity_publication_bundle(
                workspace_root=REPO_ROOT,
                sensitivity_output_directory=(
                    REPO_ROOT
                    / "manuscript"
                    / "scientific_panel"
                    / "output"
                    / "sensitivity"
                ),
                paths=paths,
            )
            self.assertEqual(
                report.claim_adjudication.author_checkpoint_status,
                "pending",
            )

    def test_committed_publication_bundle_validates_without_rewriting(self) -> None:
        artifact_roots = (SOURCE_ROOT, TABLE_ROOT, FIGURE_ROOT, ADJUDICATION_ROOT)
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for root in artifact_roots
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }

        report = validate_sensitivity_publication_bundle(workspace_root=REPO_ROOT)

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
