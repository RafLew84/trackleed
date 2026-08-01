#!/usr/bin/env python3
"""Behavior tests for the stratified uncertainty-reporting tables."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import build_tables


TABLE_ROOT = Path(__file__).resolve().parent
SUPPLEMENT_PATH = TABLE_ROOT.parents[1] / "supplementary-material.md"


class StratifiedReportingTableTest(unittest.TestCase):
    def test_count_outcome_derives_fraction_and_rate_from_same_counts(
        self,
    ) -> None:
        outcome = build_tables.CountOutcome(3, 4)

        self.assertEqual(outcome.fraction, "3/4")
        self.assertEqual(outcome.rate, 0.75)
        with self.assertRaises(ValueError):
            build_tables.CountOutcome(5, 4)

    def test_energy_scan_table_preserves_each_frozen_series_pair_case(self) -> None:
        rows = build_tables.build_energy_scan_case_table()

        self.assertEqual(len(rows), 20)
        self.assertEqual({row["scan"] for row in rows}, {"Scan A", "Scan B"})
        row = next(
            row
            for row in rows
            if row["scan"] == "Scan A"
            and row["pair"] == "pair 1"
            and row["method"] == "EasyLEED"
        )
        self.assertEqual(row["measurement_coverage"], "42/44")
        self.assertEqual(row["gap_recovery"], "5/5")
        self.assertEqual(row["geometry_inconsistency"], "0/42")
        self.assertEqual(row["conditional_pair_frames"], 15)
        self.assertAlmostEqual(
            row["conditional_median_abs_distance_residual_px"],
            1.5260859052693831,
        )

    def test_leave_one_case_out_recomputes_only_count_based_outcomes(self) -> None:
        rows = build_tables.build_leave_one_case_out_table()

        self.assertEqual(len(rows), 20)
        row = next(
            row
            for row in rows
            if row["omitted_scan"] == "Scan A"
            and row["omitted_pair"] == "pair 1"
            and row["method"] == "EasyLEED"
        )
        self.assertEqual(row["included_case_count"], 3)
        self.assertEqual(row["measurement_coverage"], "172/177")
        self.assertEqual(row["gap_recovery"], "17/19")
        self.assertEqual(row["geometry_inconsistency"], "4/180")
        self.assertNotIn("conditional_residual", row)

    def test_artifact_records_descriptive_policy_and_deterministic_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "output"
            manifest_path = root / "manifest.json"

            build_tables.build_artifact(
                output_directory=output_directory,
                manifest_path=manifest_path,
            )

            self.assertEqual(
                {path.name for path in output_directory.iterdir()},
                {
                    "energy_scan_by_case.csv",
                    "energy_scan_leave_one_case_out.csv",
                    "table_s3_energy_scan_by_case.md",
                },
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["builder_sha256"],
                build_tables.sha256_file(Path(build_tables.__file__)),
            )
            self.assertEqual(
                manifest["uncertainty_policy"]["inference"],
                "descriptive; no confidence intervals or p-values",
            )
            self.assertEqual(
                manifest["uncertainty_policy"]["sensitivity_analysis"],
                (
                    "leave one frozen scan-pair case out; recompute count-based "
                    "outcomes only"
                ),
            )
            self.assertEqual(
                manifest["uncertainty_policy"]["prohibited"],
                [
                    "image-level bootstrap",
                    "binomial confidence intervals over correlated frames or gaps",
                    "reaggregation of case medians as pooled residuals",
                ],
            )
            self.assertEqual(
                manifest["table_status"],
                {"Table S3": "finalized"},
            )
            self.assertEqual(len(manifest["outputs"]), 3)
            for markdown_name in ("table_s3_energy_scan_by_case.md",):
                groups: list[list[str]] = []
                for line in (
                    output_directory / markdown_name
                ).read_text(encoding="utf-8").splitlines():
                    if line.startswith("|"):
                        if not groups or not groups[-1]:
                            groups.append([])
                        groups[-1].append(line)
                    elif groups and groups[-1]:
                        groups.append([])
                groups = [group for group in groups if group]
                self.assertTrue(groups)
                for group in groups:
                    self.assertEqual(
                        {line.count("|") for line in group},
                        {group[0].count("|")},
                    )

    def test_table_s3_render_is_reader_facing_and_defines_each_fraction(
        self,
    ) -> None:
        rendered = build_tables.render_energy_scan_markdown(
            build_tables.build_energy_scan_case_table(),
            build_tables.build_leave_one_case_out_table(),
        )

        self.assertTrue(rendered.startswith("**Table S3."))
        self.assertIn("**A. Outcomes by frozen scan–pair case**", rendered)
        self.assertIn("**B. Leave-one-case-out sensitivity**", rendered)
        self.assertIn(
            "branch-consistent measurements / baseline-visible opportunities",
            rendered,
        )
        self.assertIn(
            "recovered / evaluable natural I(V) gaps",
            rendered,
        )
        self.assertIn("post-gap branch-consistent recovery", rendered)
        self.assertNotIn("identity recovery", rendered)
        self.assertIn(
            (
                "accepted Gaussian fits more than 18 px from the expected "
                "branch / all accepted Gaussian fits"
            ),
            rendered,
        )
        self.assertIn("not a confidence interval", rendered)
        self.assertNotIn("26a21", rendered)
        self.assertNotIn("ScanE", rendered)
        self.assertNotIn("C:\\Users", rendered)

    def test_supplement_markers_must_be_one_ordered_pair_per_table(
        self,
    ) -> None:
        table_contracts = (
            (
                "Table S3",
                build_tables.SUPPLEMENT_S3_BEGIN_MARKER,
                build_tables.SUPPLEMENT_S3_END_MARKER,
                build_tables.extract_supplement_s3_block,
            ),
        )
        for table_label, begin_marker, end_marker, extract in table_contracts:
            valid = f"{begin_marker}\ncontent\n{end_marker}"
            with self.subTest(table=table_label, record="valid"):
                self.assertEqual(extract(valid), "content\n")

            invalid_records = (
                begin_marker,
                end_marker,
                f"{valid}\n{begin_marker}\nduplicate\n{end_marker}",
                f"{end_marker}\n{begin_marker}",
            )
            for invalid in invalid_records:
                with self.subTest(table=table_label, record=invalid):
                    with self.assertRaises(ValueError):
                        extract(invalid)

    def test_provenance_rejects_malformed_hashes_and_escaping_paths(
        self,
    ) -> None:
        provenance = json.loads(
            build_tables.PROVENANCE_PATH.read_text(encoding="utf-8")
        )
        for field, value in (
            ("frozen_sha256", "not-a-hash"),
            ("frozen_path", "../../../../etc/passwd"),
            ("original_path", "../../../../etc/passwd"),
        ):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(provenance))
                changed["sources"][0][field] = value
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "provenance.json"
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with mock.patch.object(
                        build_tables,
                        "PROVENANCE_PATH",
                        path,
                    ):
                        with self.assertRaises(ValueError):
                            build_tables.validate_source_data()

    def test_committed_original_is_required(self) -> None:
        provenance = json.loads(
            build_tables.PROVENANCE_PATH.read_text(encoding="utf-8")
        )
        optional = [
            source
            for source in provenance["sources"]
            if not source.get("original_required_at_build", True)
        ]
        self.assertEqual(optional, [])

        changed = json.loads(json.dumps(provenance))
        changed["sources"][0]["original_required_at_build"] = False
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "provenance.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with mock.patch.object(build_tables, "PROVENANCE_PATH", path):
                with self.assertRaises(ValueError):
                    build_tables.validate_source_data()

    def test_committed_table_s3_and_supplement_are_synchronized(self) -> None:
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
            build_tables.build_artifact(
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
            output_directory / "table_s3_energy_scan_by_case.md"
        ).read_text(encoding="utf-8")
        supplement = SUPPLEMENT_PATH.read_text(encoding="utf-8")
        self.assertEqual(
            build_tables.extract_supplement_s3_block(supplement),
            markdown,
        )

    def test_hash_addressed_text_records_have_stable_line_endings(
        self,
    ) -> None:
        attributes = (TABLE_ROOT / ".gitattributes").read_text(
            encoding="utf-8"
        )

        for extension in ("csv", "json", "md", "py"):
            self.assertIn(f"*.{extension} text eol=lf", attributes)
        self.assertIn("source_data/*.csv binary", attributes)

        for path in (TABLE_ROOT / "output").glob("*.csv"):
            self.assertNotIn(b"\r\n", path.read_bytes())


if __name__ == "__main__":
    unittest.main()
