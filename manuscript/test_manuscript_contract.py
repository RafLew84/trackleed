"""Contract tests for the venue-portable scientific manuscript."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from manuscript.manuscript_contract import validate_manuscript_package


REPO_ROOT = Path(__file__).resolve().parents[1]


class ManuscriptContractTest(unittest.TestCase):
    def test_reader_facing_package_matches_the_frozen_scientific_contract(
        self,
    ) -> None:
        report = validate_manuscript_package(REPO_ROOT / "manuscript")

        self.assertEqual(report["experimental_placeholder_count"], 1)
        self.assertEqual(
            report["authorized_placeholder_kinds"],
            (
                "AUTHOR",
                "EXPERIMENT",
                "RELEASE-AUTHORIZATION",
                "SUBMISSION",
            ),
        )
        self.assertEqual(
            report["claim_ids"],
            tuple(f"C{i}" for i in range(1, 10)),
        )
        self.assertEqual(report["main_section_count"], 7)

    def test_malformed_or_unauthorized_placeholder_is_rejected(self) -> None:
        source_directory = REPO_ROOT / "manuscript"
        manuscript = (source_directory / "manuscript.md").read_text(
            encoding="utf-8"
        )

        for marker in ("TODO[author]", "TODO[AUTHOR_NAME]", "TODO[AUTHOR"):
            with (
                self.subTest(marker=marker),
                tempfile.TemporaryDirectory() as temporary,
            ):
                package = Path(temporary)
                (package / "manuscript.md").write_text(
                    manuscript + "\n" + marker + "\n",
                    encoding="utf-8",
                )
                for filename in (
                    "claim-evidence-map.md",
                    "supplementary-material.md",
                ):
                    (package / filename).write_text(
                        (source_directory / filename).read_text(
                            encoding="utf-8"
                        ),
                        encoding="utf-8",
                    )

                with self.assertRaisesRegex(
                    ValueError,
                    "unauthorized placeholders|malformed placeholder",
                ):
                    validate_manuscript_package(package)


if __name__ == "__main__":
    unittest.main()
