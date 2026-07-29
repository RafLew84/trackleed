#!/usr/bin/env python3
"""Contract tests for supplementary Table S2."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import build_table


TABLE_ROOT = Path(__file__).resolve().parent
SUPPLEMENT_PATH = TABLE_ROOT.parents[1] / "supplementary-material.md"


class GatesDenominatorsTableTest(unittest.TestCase):
    def test_gate_contract_preserves_regime_specific_boundaries(self) -> None:
        rows = build_table.load_and_validate_gates()
        rendered = build_table.render_markdown(
            rows,
            build_table.load_and_validate_denominators(),
        )

        self.assertEqual(len(rows), 10)
        self.assertIn("R² ≥ 0.85 and amplitude A ≥ 0.05", rendered)
        self.assertIn("−2 < x_c,y_c < 50 px", rendered)
        self.assertIn("0.6 ≤ σ_x,σ_y ≤ 13 px", rendered)
        self.assertIn("≤ 15 px", rendered)
        self.assertIn("≤ 18 px", rendered)
        self.assertIn(
            "evaluation gate applied after gaussian acceptance",
            rendered.lower(),
        )

    def test_fixed_denominators_are_not_replaced_by_conditional_n(self) -> None:
        rows = build_table.load_and_validate_denominators()
        by_metric = {(row["regime"], row["metric"]): row for row in rows}

        self.assertEqual(
            by_metric[
                (
                    "Independent energy scans",
                    "Branch-consistent measurement coverage",
                )
            ]["denominator_per_method"],
            "221",
        )
        geometry = by_metric[
            (
                "Independent energy scans",
                "Geometry-inconsistent continuation",
            )
        ]
        self.assertEqual(
            geometry["method_specific_n"],
            (
                "LocoTrack=272; OSTrack=269; DAM4SAM=272; "
                "Lucas–Kanade=206; EasyLEED=222"
            ),
        )
        self.assertIn("not the 221", geometry["scope_note"])

    def test_all_predeclared_validation_opportunities_are_recorded(self) -> None:
        rows = build_table.load_and_validate_denominators()
        fixed = {
            (row["regime"], row["metric"]): row["denominator_per_method"]
            for row in rows
            if row["denominator_type"] in {
                "Fixed opportunity",
                "Fixed eligibility window",
                "Audit sample",
                "Separate diagnostic",
            }
        }

        self.assertEqual(fixed[("Controlled translations", "Spot proposal and Gaussian-measurement coverage")], "80")
        self.assertEqual(fixed[("Controlled translations", "Pair coverage and pair-invariance endpoint")], "40")
        self.assertEqual(fixed[("Cu(111) positive control", "Spot proposal and Gaussian-measurement coverage")], "2,804")
        self.assertEqual(fixed[("Independent energy scans", "Post-gap identity recovery")], "24")
        self.assertEqual(fixed[("Controlled SAM2 family", "Branch-consistent measurement coverage")], "182")
        self.assertEqual(fixed[("Restricted external reference", "Branch-image coverage")], "60")
        self.assertEqual(fixed[("Cu(111) frozen visual audit", "Combined overlay review")], "24")

    def test_method_specific_conditional_counts_match_frozen_outputs(self) -> None:
        denominators = build_table.load_and_validate_denominators()
        build_table.validate_upstream_semantics(denominators)

        independent = build_table.read_csv_dicts(
            build_table.UPSTREAM_DIRECTORY / "method_evidence_matrix.csv"
        )
        self.assertEqual(
            [row["scane_accepted_measurements_all_frames"] for row in independent],
            ["272", "269", "272", "206", "222"],
        )
        family = build_table.read_csv_dicts(
            build_table.UPSTREAM_DIRECTORY / "family_aggregate.csv"
        )
        self.assertEqual(
            [row["jointly_consistent_pair_frames"] for row in family],
            ["19", "72", "70"],
        )
        family_by_pair = build_table.read_csv_dicts(
            build_table.UPSTREAM_DIRECTORY / "family_by_pair.csv"
        )
        self.assertEqual(
            [
                row["jointly_consistent_pair_frames"]
                for row in family_by_pair
            ],
            ["10", "38", "36", "9", "34", "34"],
        )

    def test_native_translation_error_has_its_own_conditional_n(self) -> None:
        rows = build_table.load_and_validate_denominators()
        native = next(
            row
            for row in rows
            if row["metric"] == "Native-proposal displacement error"
        )

        self.assertEqual(
            native["method_specific_n"],
            (
                "LocoTrack=80; OSTrack=80; DAM4SAM=80; "
                "Lucas–Kanade=76; EasyLEED=80"
            ),
        )

    def test_pair_2_4_is_a_separate_24_candidate_diagnostic(self) -> None:
        rows = build_table.read_csv_dicts(
            build_table.UPSTREAM_DIRECTORY
            / "pair_2_4_missingness_diagnostic.csv"
        )
        rejected = [row for row in rows if row["refit_accepted"] == "false"]

        self.assertEqual(len(rejected), 24)
        self.assertEqual(rejected[0]["frame_id"], "357")
        self.assertEqual(rejected[-1]["frame_id"], "380")
        self.assertTrue(all(row["tracking_roi_present"] == "true" for row in rejected))
        self.assertTrue(all(row["r_squared_gate"] == "true" for row in rejected))
        self.assertTrue(all(row["amplitude_gate"] == "false" for row in rejected))
        self.assertTrue(all(row["center_gate"] == "true" for row in rejected))
        self.assertTrue(all(row["sigma_gate"] == "true" for row in rejected))

    def test_provenance_is_hash_checked_and_archived(self) -> None:
        provenance = build_table.validate_provenance()

        self.assertEqual(len(provenance["sources"]), 10)
        for source in provenance["sources"]:
            frozen_path = TABLE_ROOT / source["frozen_path"]
            self.assertTrue(frozen_path.is_file())
            self.assertNotIn(".scratch", source["frozen_path"])
            self.assertEqual(
                build_table.sha256_file(frozen_path),
                source["frozen_sha256"],
            )
            if not source.get("origin_required_at_build", True):
                self.assertTrue(source["origin_path"].startswith(".scratch/"))
            if "origin_lf_sha256" in source:
                origin_path = build_table.REPOSITORY_ROOT / source["origin_path"]
                self.assertEqual(
                    build_table.sha256_lf_normalized_text(origin_path),
                    source["origin_lf_sha256"],
                )

    def test_provenance_rejects_malformed_hashes_and_escaping_paths(self) -> None:
        provenance = json.loads(
            build_table.PROVENANCE_PATH.read_text(encoding="utf-8")
        )
        for field, value in (
            ("frozen_sha256", "not-a-hash"),
            ("frozen_path", "../../../../etc/passwd"),
            ("origin_path", "../../../../etc/passwd"),
        ):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(provenance))
                changed["sources"][0][field] = value
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "provenance.json"
                    path.write_text(
                        json.dumps(changed),
                        encoding="utf-8",
                    )
                    with mock.patch.object(
                        build_table,
                        "PROVENANCE_PATH",
                        path,
                    ):
                        with self.assertRaises(ValueError):
                            build_table.validate_provenance()

    def test_only_two_scratch_origins_may_be_optional(self) -> None:
        provenance = json.loads(
            build_table.PROVENANCE_PATH.read_text(encoding="utf-8")
        )
        optional = [
            source
            for source in provenance["sources"]
            if not source.get("origin_required_at_build", True)
        ]
        self.assertEqual(len(optional), 2)
        self.assertTrue(
            all(source["origin_path"].startswith(".scratch/") for source in optional)
        )

        mutations = []
        only_one = json.loads(json.dumps(provenance))
        only_one["sources"][0]["origin_required_at_build"] = True
        mutations.append(only_one)
        committed_optional = json.loads(json.dumps(provenance))
        committed_optional["sources"][2]["origin_required_at_build"] = False
        mutations.append(committed_optional)
        for changed in mutations:
            with tempfile.TemporaryDirectory() as temporary_directory:
                path = Path(temporary_directory) / "provenance.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with mock.patch.object(build_table, "PROVENANCE_PATH", path):
                    with self.assertRaises(ValueError):
                        build_table.validate_provenance()

    def test_supplement_markers_must_be_one_ordered_pair(self) -> None:
        valid = (
            f"{build_table.SUPPLEMENT_BEGIN_MARKER}\ncontent\n"
            f"{build_table.SUPPLEMENT_END_MARKER}"
        )
        self.assertEqual(
            build_table.extract_supplement_block(valid),
            "content\n",
        )

        invalid_records = (
            build_table.SUPPLEMENT_BEGIN_MARKER,
            build_table.SUPPLEMENT_END_MARKER,
            (
                f"{valid}\n"
                f"{build_table.SUPPLEMENT_BEGIN_MARKER}\nduplicate\n"
                f"{build_table.SUPPLEMENT_END_MARKER}"
            ),
            (
                f"{build_table.SUPPLEMENT_END_MARKER}\n"
                f"{build_table.SUPPLEMENT_BEGIN_MARKER}"
            ),
        )
        for invalid in invalid_records:
            with self.subTest(record=invalid):
                with self.assertRaises(ValueError):
                    build_table.extract_supplement_block(invalid)

    def test_rendered_table_has_three_blocks_and_statistical_caveat(self) -> None:
        rendered = build_table.render_markdown(
            build_table.load_and_validate_gates(),
            build_table.load_and_validate_denominators(),
        )

        self.assertIn("**A. Gaussian measurement and evaluation gates**", rendered)
        self.assertIn("**B. Fixed opportunity denominators**", rendered)
        self.assertIn("**C. Method-dependent conditional denominators**", rendered)
        self.assertIn("repeated observations of propagated trajectories", rendered)
        self.assertIn("not statistically independent replicates", rendered)
        self.assertNotIn("26a21", rendered)
        self.assertNotIn("ScanE", rendered)
        self.assertNotIn("C:\\Users", rendered)

    def test_artifact_is_deterministic_and_manifest_covers_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "output"
            manifest_path = root / "manifest.json"

            first = build_table.build_artifact(
                output_directory=output_directory,
                manifest_path=manifest_path,
            )
            first_bytes = {
                path.name: path.read_bytes()
                for path in output_directory.iterdir()
            }
            second = build_table.build_artifact(
                output_directory=output_directory,
                manifest_path=manifest_path,
            )
            second_bytes = {
                path.name: path.read_bytes()
                for path in output_directory.iterdir()
            }

            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(first, second)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["outputs"]), set(first_bytes))

    def test_committed_outputs_and_supplement_are_synchronized(self) -> None:
        output_directory = TABLE_ROOT / "output"
        expected = {
            path.name: path.read_bytes()
            for path in output_directory.iterdir()
            if path.is_file()
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            generated_output = root / "output"
            generated_manifest = root / "manifest.json"
            build_table.build_artifact(
                output_directory=generated_output,
                manifest_path=generated_manifest,
            )
            actual = {
                path.name: path.read_bytes()
                for path in generated_output.iterdir()
            }
            generated_manifest_bytes = generated_manifest.read_bytes()

        self.assertEqual(expected, actual)
        self.assertEqual(
            (TABLE_ROOT / "manifest.json").read_bytes(),
            generated_manifest_bytes,
        )
        markdown = (
            output_directory
            / "table_s2_gaussian_gates_validation_denominators.md"
        ).read_text(encoding="utf-8")
        supplement = SUPPLEMENT_PATH.read_text(encoding="utf-8")
        self.assertEqual(
            build_table.extract_supplement_block(supplement),
            markdown,
        )


if __name__ == "__main__":
    unittest.main()
