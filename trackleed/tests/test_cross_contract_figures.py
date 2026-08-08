from __future__ import annotations

import csv
import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image
from pypdf import PdfReader

from trackleed.publication.cross_contract_figures import (
    CrossContractFigureError,
    build_cross_contract_figures,
    validate_cross_contract_figures,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_SOURCES = REPO_ROOT / "manuscript/cross_contract/figures"


def copy_figure_sources(destination: Path) -> None:
    destination.mkdir()
    for filename in ("README.md", "captions.json", "figure_contract.json"):
        shutil.copyfile(FIGURE_SOURCES / filename, destination / filename)


class CrossContractFigureTests(unittest.TestCase):
    def test_validation_fails_closed_when_figure_package_is_absent(self) -> None:
        with (
            TemporaryDirectory() as temporary,
            self.assertRaisesRegex(
                CrossContractFigureError,
                "cross-contract figure contract is unavailable",
            ),
        ):
            validate_cross_contract_figures(
                Path(temporary), Path(temporary) / "figures"
            )

    def test_builder_creates_full_width_cross_contract_framework(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "figures"
            copy_figure_sources(package)

            report = build_cross_contract_figures(REPO_ROOT, package)
            png = package / "output/figure01_cross_contract_framework.png"
            pdf = package / "output/pdf/figure01_cross_contract_framework.pdf"

            with Image.open(png) as image:
                self.assertEqual(image.size, (2102, 921))
            reader = PdfReader(pdf)
            self.assertEqual(len(reader.pages), 1)
            page = reader.pages[0]
            self.assertAlmostEqual(float(page.mediabox.width), 504.57, places=1)
            self.assertAlmostEqual(float(page.mediabox.height), 221.10, places=1)
            self.assertEqual(report["figures"]["figure01"]["size_mm"], [178.0, 78.0])

    def test_builder_creates_single_column_observability_and_gap_figure(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "figures"
            copy_figure_sources(package)

            report = build_cross_contract_figures(REPO_ROOT, package)
            png = package / "output/figure02_measurability_and_gaps.png"
            pdf = package / "output/pdf/figure02_measurability_and_gaps.pdf"

            with Image.open(png) as image:
                self.assertEqual(image.size, (1039, 1157))
            page = PdfReader(pdf).pages[0]
            self.assertAlmostEqual(float(page.mediabox.width), 249.45, places=1)
            self.assertAlmostEqual(float(page.mediabox.height), 277.80, places=1)
            self.assertEqual(report["figures"]["figure02"]["size_mm"], [88.0, 98.0])

            with (package / "source_data/figure02_measurability.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                measurability = list(csv.DictReader(handle))
            self.assertEqual(len(measurability), 20)
            easyleed = [
                int(row["count"])
                for row in measurability
                if row["method"] == "easyleed"
            ]
            self.assertEqual(easyleed, [222, 0, 83, 19])
            self.assertTrue(
                all(int(row["denominator"]) == 324 for row in measurability)
            )

            with (package / "source_data/figure02_gap_components.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                gaps = list(csv.DictReader(handle))
            totals = {
                category: sum(
                    int(row["component_count"])
                    for row in gaps
                    if row["category"] == category
                )
                for category in ("gaussian_only", "shared", "viperleed_only")
            }
            self.assertEqual(
                totals,
                {"gaussian_only": 111, "shared": 3, "viperleed_only": 7},
            )

    def test_builder_creates_conditional_out_of_branch_classification(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "figures"
            copy_figure_sources(package)

            report = build_cross_contract_figures(REPO_ROOT, package)
            png = package / "output/figure03_out_of_branch_interpretation.png"
            pdf = package / "output/pdf/figure03_out_of_branch_interpretation.pdf"

            with Image.open(png) as image:
                self.assertEqual(image.size, (1039, 1003))
            page = PdfReader(pdf).pages[0]
            self.assertAlmostEqual(float(page.mediabox.width), 249.45, places=1)
            self.assertAlmostEqual(float(page.mediabox.height), 240.94, places=1)
            self.assertEqual(report["figures"]["figure03"]["size_mm"], [88.0, 85.0])

            with (package / "source_data/figure03_out_of_branch.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 15)
            totals = {
                category: sum(
                    int(row["event_count"])
                    for row in rows
                    if row["classification"] == category
                )
                for category in (
                    "branch_model_disagreement",
                    "cross_contract_target_inconsistency",
                    "unresolved",
                )
            }
            self.assertEqual(
                totals,
                {
                    "branch_model_disagreement": 83,
                    "cross_contract_target_inconsistency": 16,
                    "unresolved": 25,
                },
            )
            for method, conditional_denominator, accepted_count in (
                ("easyleed", 4, 222),
                ("locotrack", 36, 272),
                ("ostrack", 34, 269),
                ("dam4sam", 36, 272),
                ("lk", 14, 206),
            ):
                method_rows = [row for row in rows if row["method"] == method]
                self.assertEqual(
                    sum(int(row["event_count"]) for row in method_rows),
                    conditional_denominator,
                )
                self.assertTrue(
                    all(
                        int(row["conditional_denominator"]) == conditional_denominator
                        for row in method_rows
                    )
                )
                self.assertTrue(
                    all(
                        int(row["accepted_gaussian_all_frames"]) == accepted_count
                        for row in method_rows
                    )
                )

    def test_build_is_staged_build_once_and_validates_manifested_outputs(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "figures"
            copy_figure_sources(package)

            built = build_cross_contract_figures(REPO_ROOT, package)

            self.assertEqual(built["status"], "validated-cross-contract-main-figures")
            self.assertEqual(built, validate_cross_contract_figures(REPO_ROOT, package))
            manifest = json.loads(
                (package / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["file_count"], 13)
            with self.assertRaisesRegex(
                FileExistsError,
                "Refusing to replace frozen cross-contract figure package",
            ):
                build_cross_contract_figures(REPO_ROOT, package)

            with (package / "output/figure02_measurability_and_gaps.png").open(
                "ab"
            ) as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(
                CrossContractFigureError,
                "manifested cross-contract figure drift",
            ):
                validate_cross_contract_figures(REPO_ROOT, package)

    def test_build_validates_staging_before_publishing(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "figures"
            copy_figure_sources(package)

            with (
                patch(
                    "trackleed.publication.cross_contract_figures."
                    "_validate_frozen_package",
                    side_effect=CrossContractFigureError("rejected staging"),
                ),
                self.assertRaisesRegex(CrossContractFigureError, "rejected staging"),
            ):
                build_cross_contract_figures(REPO_ROOT, package)

            self.assertFalse((package / "output").exists())
            self.assertFalse((package / "source_data").exists())
            self.assertFalse((package / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
