"""Contract tests for the venue-portable scientific manuscript."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import re

from manuscript.manuscript_contract import validate_manuscript_package


REPO_ROOT = Path(__file__).resolve().parents[1]

APPROVED_ABSTRACT = """Quantitative low-energy electron diffraction (LEED) analysis requires persistent association between a measurement region and its target diffraction spot as patterns move and visibility changes. We compared five heterogeneous tracking strategies through a common position-measurement contract: each native output supplied a proposal for a 48 × 48-pixel ROI, and the same two-dimensional Gaussian estimator produced the reported spot position. Robustness was assessed with fixed opportunity sets for coverage and post-gap recovery, accepted-fit denominators for geometry inconsistency, and method-specific surviving \\(n\\) for conditional residuals.

Validation combined known translations, a 701-image Cu(111) positive control, two independent Ag(111) energy scans with natural \\(I(V)\\) gaps, and a controlled SAM2-family memory experiment. Translation errors were 0.0043–0.0265 px with 74/80–80/80 accepted measurements. All methods produced 2,804/2,804 Cu(111) measurements and 701/701 endpoints per primary pair. In the Ag(111) scans, coverage was 188/221–217/221, recovery 17/24–23/24, and conditional median pair-distance residuals 1.737–1.837 px (\\(n=68–91\\)). SAMURAI and DAM4SAM increased coverage over SAM2 but also increased geometry-inconsistent continuation; this mechanism result is not a general ranking.

Across 15 predeclared perturbations of ROI geometry and acceptance gates, absolute outcomes changed, but Lucas–Kanade retained the lowest coverage and the denominator-aware interpretation remained stable. The results assess measurement-chain robustness, not absolute tracking accuracy. Proposal differences were strongly suppressed when the target spot remained inside the common estimator's capture region, making this a supporting mechanism rather than the primary contribution."""


def _section(text: str, start: str, end: str) -> str:
    return text[text.index(start) + len(start) : text.index(end)].strip()


def _normalized(text: str) -> str:
    return re.sub(r"[ \t\r\n]+", " ", text).strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"[^\W_]+(?:[-‐‑‒–—][^\W_]+)*", text))


class ManuscriptContractTest(unittest.TestCase):
    def _write_package(self, package: Path, manuscript: str) -> None:
        source_directory = REPO_ROOT / "manuscript"
        (package / "manuscript.md").write_text(manuscript, encoding="utf-8")
        for filename in ("claim-evidence-map.md", "supplementary-material.md"):
            (package / filename).write_text(
                (source_directory / filename).read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    def test_scientific_reconstruction_contract_is_required(self) -> None:
        source_directory = REPO_ROOT / "manuscript"
        manuscript = (source_directory / "manuscript.md").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            self._write_package(
                package,
                manuscript.replace(
                    "python -m scripts.reconstruct_publication --clean --validate",
                    "a future reconstruction command",
                ),
            )

            with self.assertRaisesRegex(ValueError, "reconstruction command"):
                validate_manuscript_package(package)

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

    def test_approved_content_revision_and_denominator_semantics(self) -> None:
        manuscript = (REPO_ROOT / "manuscript/manuscript.md").read_text(
            encoding="utf-8"
        )
        abstract = _section(manuscript, "## Abstract", "## Keywords")
        self.assertEqual(_normalized(abstract), _normalized(APPROVED_ABSTRACT))
        self.assertEqual(_word_count(abstract), 244)
        self.assertIsNone(
            re.search(r"\b(?:table|figure|supplementary)\b", abstract, re.I)
        )
        self.assertIn(
            "### 1.1 Quantitative LEED requires persistent spot association",
            manuscript,
        )

        objective = _section(
            manuscript,
            "### 1.3 Study objective and contributions",
            "## 2. Measurement chain and validation design",
        )
        contribution_anchors = (
            "A common proposal-to-ROI-to-Gaussian position-measurement contract",
            "A denominator-aware assessment that retains fixed opportunity sets",
            "Controlled and case-based evidence for measurement-chain equivariance",
            "A predeclared 15-variant sensitivity analysis",
            "A mechanistic account of when the common Gaussian estimator suppresses",
        )
        self.assertEqual(
            sorted(objective.index(anchor) for anchor in contribution_anchors),
            [objective.index(anchor) for anchor in contribution_anchors],
        )

        expected_results = (
            "### 4.4 Predeclared measurement-window and capture-contract sensitivity",
            "### 4.5 Controlled SAM2 family exposes a continuity-consistency trade-off",
            "### 4.6 Supporting compatibility and missingness checks",
        )
        for heading in expected_results:
            self.assertIn(heading, manuscript)
        sensitivity = _section(manuscript, expected_results[0], expected_results[1])
        normalized_sensitivity = _normalized(sensitivity)
        for exact_anchor in (
            "15 predeclared measurement-window and capture-contract variants",
            "170–221/221",
            "17–24/24",
            "1/222–69/272",
            "0.45–25.37%",
            "187–286",
            "1.499 px (n=79)–1.939 px (n=99;",
            "54–104",
            "not a parameter optimization or a comparative accuracy ranking",
            "no alternate width contract was generated",
        ):
            self.assertIn(exact_anchor, normalized_sensitivity)

        central_results = _section(manuscript, "### 4.3", expected_results[0])
        normalized_central_results = _normalized(central_results)
        for result in (
            "1.737 px (n=84) for EasyLEED",
            "1.764 px (n=68) for Lucas--Kanade",
            "1.770 px (n=90) for OSTrack",
            "1.788 px (n=91) for DAM4SAM",
            "1.837 px (n=91) for LocoTrack",
        ):
            self.assertIn(result, normalized_central_results)

        self.assertNotIn(
            "The present work does not test other ROI sizes",
            manuscript,
        )
        self.assertIn(
            "Sensitivity thus supports claims about denominator-aware "
            "measurement-chain robustness within the acquisition-closed case "
            "boundary",
            _normalized(manuscript),
        )
        self.assertIn(
            "Conditional residuals must therefore be reported with their "
            "method-specific surviving n, alongside coverage and recovery "
            "evaluated on fixed opportunity sets and the corresponding "
            "geometry-consistency outcomes.",
            _normalized(manuscript),
        )

    def test_malformed_or_unauthorized_placeholder_is_rejected(self) -> None:
        source_directory = REPO_ROOT / "manuscript"
        manuscript = (source_directory / "manuscript.md").read_text(encoding="utf-8")

        for marker in ("TODO[author]", "TODO[AUTHOR_NAME]", "TODO[AUTHOR"):
            with (
                self.subTest(marker=marker),
                tempfile.TemporaryDirectory() as temporary,
            ):
                package = Path(temporary)
                self._write_package(package, manuscript + "\n" + marker + "\n")

                with self.assertRaisesRegex(
                    ValueError,
                    "unauthorized placeholders|malformed placeholder",
                ):
                    validate_manuscript_package(package)

    def test_prohibited_scope_claim_is_rejected(self) -> None:
        source_directory = REPO_ROOT / "manuscript"
        manuscript = (source_directory / "manuscript.md").read_text(
            encoding="utf-8"
        )

        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            self._write_package(
                package,
                manuscript + "\nThis study establishes absolute tracking accuracy.\n",
            )

            with self.assertRaisesRegex(ValueError, "Prohibited positive claim"):
                validate_manuscript_package(package)


if __name__ == "__main__":
    unittest.main()
