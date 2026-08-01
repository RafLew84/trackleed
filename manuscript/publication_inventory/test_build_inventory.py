"""Contract tests for the final publication-display inventory."""

from __future__ import annotations

import copy
import json
import re
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import build_inventory


INVENTORY_ROOT = Path(__file__).resolve().parent
INVENTORY_PATH = INVENTORY_ROOT / "source_data" / "display_inventory.json"
DISPLAY_INDEX_PATH = INVENTORY_ROOT.parent / "figures-and-tables.md"
SEMANTIC_MANIFEST_PATH = INVENTORY_ROOT / "semantic_manifest.json"
MANUSCRIPT_PATH = INVENTORY_ROOT.parent / "manuscript.md"
SUPPLEMENT_PATH = INVENTORY_ROOT.parent / "supplementary-material.md"


class PublicationInventoryTest(unittest.TestCase):
    def test_inventory_is_the_exact_approved_twenty_item_set(self) -> None:
        inventory = build_inventory.load_inventory(INVENTORY_PATH)
        report = build_inventory.validate_inventory(inventory)

        self.assertEqual(tuple(report["ids"]), build_inventory.EXPECTED_IDS)
        self.assertEqual(report["placement_counts"]["main_figure"], 5)
        self.assertEqual(report["placement_counts"]["main_table"], 2)
        self.assertEqual(report["placement_counts"]["supplementary_figure"], 6)
        self.assertEqual(report["placement_counts"]["supplementary_table"], 7)
        self.assertNotIn("Table 3", report["ids"])
        self.assertTrue(all("caption_title" not in item for item in inventory["items"]))

    def test_every_bundle_has_a_verified_source_to_output_hash_chain(self) -> None:
        inventory = build_inventory.load_inventory(INVENTORY_PATH)
        report = build_inventory.validate_inventory(inventory)

        self.assertEqual(report["bundle_count"], 20)
        self.assertEqual(
            report["source_file_count"],
            report["hashed_source_file_count"],
        )
        self.assertEqual(
            report["output_file_count"],
            report["hashed_output_file_count"],
        )
        self.assertTrue(report["all_builders_present"])
        self.assertTrue(report["all_tests_present"])
        self.assertTrue(report["all_rounding_rules_frozen"])
        self.assertTrue(report["all_denominator_policies_explicit"])

    def test_wrong_title_and_obsolete_placement_are_rejected(self) -> None:
        inventory = build_inventory.load_inventory(INVENTORY_PATH)
        wrong_title = copy.deepcopy(inventory)
        wrong_title["items"][0]["title"] = "Changed title"
        with self.assertRaisesRegex(ValueError, "approved inventory"):
            build_inventory.validate_inventory(wrong_title)

        obsolete = copy.deepcopy(inventory)
        obsolete["items"][6]["id"] = "Table 3"
        with self.assertRaisesRegex(ValueError, "approved inventory"):
            build_inventory.validate_inventory(obsolete)

        alternate_caption = copy.deepcopy(inventory)
        alternate_caption["items"][0]["caption_title"] = "Alternate title"
        with self.assertRaisesRegex(ValueError, "Alternate caption title"):
            build_inventory.validate_inventory(alternate_caption)

        for field, changed_value in (
            ("rounding_rule_ids", ["exact_counts"]),
            ("palette_policy", "Not applicable."),
            (
                "test_paths",
                ["manuscript/publication_inventory/test_build_inventory.py"],
            ),
        ):
            changed_contract = copy.deepcopy(inventory)
            changed_contract["items"][3][field] = changed_value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(
                    ValueError,
                    "approved inventory",
                ),
            ):
                build_inventory.validate_inventory(changed_contract)

    def test_manifest_path_and_digest_fields_are_paired_explicitly(self) -> None:
        frozen_digest = "a" * 64
        origin_digest = "b" * 64

        records = build_inventory._manifest_digest_records(
            {
                "frozen_path": "source_data/frozen.csv",
                "frozen_sha256": frozen_digest,
                "origin_path": "archive/origin.csv",
                "origin_sha256": origin_digest,
            }
        )

        self.assertIn(("source_data/frozen.csv", frozen_digest), records)
        self.assertIn(("archive/origin.csv", origin_digest), records)
        self.assertNotIn(("source_data/frozen.csv", origin_digest), records)
        self.assertNotIn(("archive/origin.csv", frozen_digest), records)

    def test_recovery_displays_use_the_exact_domain_term(self) -> None:
        inventory = build_inventory.load_inventory(INVENTORY_PATH)

        for item in inventory["items"]:
            if item["id"] not in build_inventory.RECOVERY_DISPLAY_IDS:
                continue
            caption = build_inventory._resolve_repo_path(
                item["caption_path"]
            ).read_text(encoding="utf-8")
            self.assertIn(
                build_inventory.RECOVERY_TERM,
                caption.casefold(),
                item["id"],
            )

    def test_main_and_supplement_placements_use_final_cross_references(self) -> None:
        manuscript = MANUSCRIPT_PATH.read_text(encoding="utf-8")
        supplement = SUPPLEMENT_PATH.read_text(encoding="utf-8")

        self.assertNotIn("Table 3", manuscript)
        self.assertNotIn("BEGIN TABLE 3", manuscript)
        self.assertIn("Figure 3; Table S7", manuscript)
        self.assertIn("Figure 4; Table S7", manuscript)
        self.assertIn("Figure S4", manuscript)
        self.assertIn("Figure S6", manuscript)
        self.assertEqual(supplement.count("<!-- BEGIN TABLE S7 -->"), 1)
        self.assertEqual(supplement.count("<!-- END TABLE S7 -->"), 1)
        self.assertGreater(
            supplement.index("<!-- BEGIN TABLE S7 -->"),
            supplement.index("<!-- END TABLE S6 -->"),
        )
        allowed_figures = {f"Figure {number}" for number in range(1, 6)} | {
            f"Figure S{number}" for number in range(1, 7)
        }
        allowed_tables = {"Table 1", "Table 2"} | {
            f"Table S{number}" for number in range(1, 8)
        }
        figure_references = set(re.findall(r"\bFigure (?:S\d+|\d+[a-z]?)", manuscript))
        table_references = set(re.findall(r"\bTable (?:S\d+|\d+)", manuscript))
        self.assertLessEqual(figure_references, allowed_figures)
        self.assertLessEqual(table_references, allowed_tables)

    def test_default_gate_executes_every_declared_bundle_contract(self) -> None:
        inventory = build_inventory.load_inventory(INVENTORY_PATH)
        commands = build_inventory.bundle_test_commands(inventory)
        declared_local_tests = {
            (
                build_inventory._resolve_repo_path(path).parent,
                Path(path).name,
            )
            for item in inventory["items"]
            for path in item["test_paths"]
            if path != "trackleed/tests/test_scientific_sensitivity_evidence.py"
        }
        commanded_local_tests = {
            (cwd, target)
            for cwd, targets in commands
            if cwd != build_inventory.REPO_ROOT
            for target in targets
        }

        self.assertEqual(commanded_local_tests, declared_local_tests)
        self.assertIn(
            (build_inventory.REPO_ROOT, build_inventory.SENSITIVITY_TEST_TARGETS),
            commands,
        )

        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(
            build_inventory.subprocess,
            "run",
            return_value=completed,
        ) as run:
            build_inventory.run_bundle_contract_tests(inventory)
        self.assertEqual(run.call_count, len(commands))

    def test_build_is_deterministic_and_committed_index_is_synchronized(self) -> None:
        inventory = build_inventory.load_inventory(INVENTORY_PATH)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_index = root / "first.md"
            first_manifest = root / "first-manifest.json"
            first_semantic_manifest = root / "first-semantic-manifest.json"
            second_index = root / "second.md"
            second_manifest = root / "second-manifest.json"
            second_semantic_manifest = root / "second-semantic-manifest.json"

            first = build_inventory.build_inventory_artifact(
                inventory,
                output_path=first_index,
                manifest_path=first_manifest,
                semantic_manifest_path=first_semantic_manifest,
            )
            second = build_inventory.build_inventory_artifact(
                inventory,
                output_path=second_index,
                manifest_path=second_manifest,
                semantic_manifest_path=second_semantic_manifest,
            )

            self.assertEqual(first, second)
            self.assertEqual(first_index.read_bytes(), second_index.read_bytes())
            self.assertEqual(first_manifest.read_bytes(), second_manifest.read_bytes())
            self.assertEqual(
                first_semantic_manifest.read_bytes(),
                second_semantic_manifest.read_bytes(),
            )
            self.assertEqual(DISPLAY_INDEX_PATH.read_bytes(), first_index.read_bytes())
            self.assertEqual(
                (INVENTORY_ROOT / "manifest.json").read_bytes(),
                first_manifest.read_bytes(),
            )
            self.assertEqual(
                SEMANTIC_MANIFEST_PATH.read_bytes(),
                first_semantic_manifest.read_bytes(),
            )

            semantic = json.loads(first_semantic_manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                set(semantic["bundles"]), set(build_inventory.EXPECTED_IDS)
            )
            for bundle in semantic["bundles"].values():
                self.assertTrue(bundle["source_data_sha256"])
                self.assertEqual(
                    set(bundle["contracts_sha256"]),
                    {"builder", "caption", "manifests", "validators"},
                )
                self.assertTrue(bundle["contracts_sha256"]["validators"])
                self.assertTrue(
                    all(
                        Path(path).suffix.lower()
                        in build_inventory.SEMANTIC_OUTPUT_SUFFIXES
                        for path in bundle["text_outputs_sha256"]
                    )
                )

            rendered = first_index.read_text(encoding="utf-8")
            headings = rendered.splitlines()
            self.assertEqual(
                sum(
                    line.startswith("### Figure ")
                    and not line.startswith("### Figure S")
                    for line in headings
                ),
                5,
            )
            self.assertEqual(
                sum(
                    line.startswith("### Table ") and not line.startswith("### Table S")
                    for line in headings
                ),
                2,
            )
            self.assertEqual(
                sum(line.startswith("### Figure S") for line in headings),
                6,
            )
            self.assertEqual(
                sum(line.startswith("### Table S") for line in headings),
                7,
            )
            self.assertNotIn("### Table 3", rendered)


if __name__ == "__main__":
    unittest.main()
