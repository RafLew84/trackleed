from __future__ import annotations

from pathlib import Path
import csv
import json
import shutil
from tempfile import TemporaryDirectory
import unittest

from trackleed.publication.controlled_translation_evidence import (
    build_trial_outcomes,
    validate_controlled_translation_source_data,
)
from trackleed.publication.controlled_translation_table import (
    build_table_s5_artifact,
)
from trackleed.publication.controlled_translation_figure_s1 import (
    build_figure_s1_artifact,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = (
    REPOSITORY_ROOT / "manuscript" / "source_data" / "controlled_translations"
)
ORIGIN_DIRECTORY = (
    REPOSITORY_ROOT
    / ".scratch"
    / "trackleed-physics-method-paper"
    / "artifacts"
    / "controlled-translation-chain"
)


class ControlledTranslationEvidenceTests(unittest.TestCase):
    def test_figure_s1_contains_every_source_trial_and_diagnostic(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = build_figure_s1_artifact(
                workspace_root=REPOSITORY_ROOT,
                source_directory=SOURCE_DIRECTORY,
                output_directory=root / "output",
                manifest_path=root / "manifest.json",
            )

            self.assertEqual(report.trial_count, 4)
            self.assertEqual(report.panel_count, 12)
            self.assertEqual(report.spot_record_count, 576)
            self.assertEqual(report.pair_record_count, 288)
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["diagnostic_rows"],
                [
                    "full_measurement_paths",
                    "native_vs_refined_displacement",
                    "pair_invariance",
                ],
            )
            self.assertEqual(len(manifest["panel_order"]), 12)
            self.assertEqual(
                set(manifest["outputs_sha256"]),
                {
                    "figure_s1_controlled_translation_trials.pdf",
                    "figure_s1_controlled_translation_trials.png",
                },
            )
            serialized = json.dumps(manifest).lower()
            self.assertIn("not absolute tracking accuracy", serialized)
            self.assertIn("not identity ground truth", serialized)

    def test_table_s5_reports_full_precision_and_explicit_denominators(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = build_table_s5_artifact(
                workspace_root=REPOSITORY_ROOT,
                source_directory=SOURCE_DIRECTORY,
                output_directory=root / "output",
                manifest_path=root / "manifest.json",
            )

            self.assertEqual(report.row_count, 24)
            with (root / "output" / "table_s5_controlled_translation.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
            row = next(
                value
                for value in rows
                if value["source_frame_order"] == "0"
                and value["method"] == "locotrack_base_offline"
            )
            self.assertEqual(row["spot_opportunity_n"], "20")
            self.assertEqual(row["conditional_measurement_error_n"], "20")
            self.assertEqual(row["pair_opportunity_n"], "10")
            self.assertEqual(row["conditional_pair_invariance_n"], "10")
            self.assertEqual(
                row["measurement_error_median_px"],
                "0.0031384264221719837",
            )
            markdown = (
                root / "output" / "table_s5_controlled_translation.md"
            ).read_text(encoding="utf-8")
            self.assertIn("20 spot-step and 10 pair-step", markdown)
            self.assertIn("conditional on accepted Gaussian measurements", markdown)
            self.assertIn("not absolute tracking accuracy", markdown)
            self.assertIn("not identity ground truth", markdown)
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["table"], "Table S5")
            self.assertEqual(len(manifest["outputs_sha256"]), 2)

    def test_source_package_is_authenticated_and_fails_closed(self) -> None:
        report = validate_controlled_translation_source_data(
            SOURCE_DIRECTORY,
            origin_directory=ORIGIN_DIRECTORY,
        )

        self.assertEqual(report["spot_record_count"], 576)
        self.assertEqual(report["pair_record_count"], 288)
        self.assertEqual(report["trial_count"], 4)
        self.assertEqual(report["method_count"], 6)

        with TemporaryDirectory() as temporary:
            changed_source = Path(temporary) / "source"
            shutil.copytree(SOURCE_DIRECTORY, changed_source)
            result_path = changed_source / "equivariance_results.csv"
            result_path.write_text(
                result_path.read_text(encoding="utf-8").replace(
                    "0.0051151386109022506",
                    "9.0051151386109022506",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_controlled_translation_source_data(changed_source)

    def test_trial_outcomes_keep_fixed_and_conditional_denominators(self) -> None:
        rows = build_trial_outcomes(SOURCE_DIRECTORY)

        self.assertEqual(len(rows), 24)
        self.assertEqual(
            {int(row["source_frame_order"]) for row in rows},
            {0, 372, 378, 700},
        )
        self.assertTrue(all(row["spot_opportunity_n"] == 20 for row in rows))
        self.assertTrue(all(row["pair_opportunity_n"] == 10 for row in rows))
        self.assertTrue(
            all(
                row["conditional_measurement_error_n"]
                == row["accepted_measurement_count"]
                for row in rows
            )
        )
        self.assertTrue(
            all(
                row["conditional_pair_invariance_n"]
                == row["valid_pair_count"]
                for row in rows
            )
        )
        low_temperature_locotrack = next(
            row
            for row in rows
            if row["source_frame_order"] == 0
            and row["method"] == "locotrack_base_offline"
        )
        self.assertEqual(low_temperature_locotrack["available_proposal_count"], 20)
        self.assertEqual(low_temperature_locotrack["accepted_measurement_count"], 20)
        self.assertAlmostEqual(
            low_temperature_locotrack["measurement_error_median_px"],
            0.0031384264221719837,
        )
        self.assertEqual(low_temperature_locotrack["valid_pair_count"], 10)
        self.assertAlmostEqual(
            low_temperature_locotrack["maximum_abs_delta_d_over_d_pp"],
            0.00018074382868849865,
        )


if __name__ == "__main__":
    unittest.main()
