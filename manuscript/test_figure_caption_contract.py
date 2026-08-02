#!/usr/bin/env python3
"""Contracts for the approved main-figure captions and numbering."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_ROOT = REPO_ROOT / "manuscript"
INVENTORY_PATH = (
    MANUSCRIPT_ROOT
    / "publication_inventory"
    / "source_data"
    / "display_inventory.json"
)

CAPTION_PATHS = {
    "Figure 1": "manuscript/figures/figure01_measurement_chain/README.md",
    "Figure 4": (
        "manuscript/figures/figure04_controlled_sam2_memory_tradeoff/README.md"
    ),
    "Figure 5": "manuscript/figures/figure05_cu111_physical_endpoint/README.md",
}
APPROVED_NORMALIZED_CAPTION_SHA256 = {
    "Figure 1": "bb420c2a2969abca47bb521de40af9e1f6630670062282cc86ccedb403d40d1f",
    "Figure 4": "638d0c305f94f868c5a4b64ecddac18ddbc2c9a58b2885473edd53a18c0e6f0a",
    "Figure 5": "941d4106c3f16fd439cf30307f9919a4595f537ff1b752cf5c01818ca78d211b",
}


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"[^\W_]+(?:[-‐‑‒–—][^\W_]+)*", text))


def _caption(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"^## Caption(?: draft)?\s*\n(?P<caption>.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"Caption section missing from {path}")
    return match.group("caption").strip()


class FigureCaptionContractTest(unittest.TestCase):
    def test_approved_captions_are_exact_and_keep_declared_lengths(self) -> None:
        for figure, relative_path in CAPTION_PATHS.items():
            with self.subTest(figure=figure):
                caption = _caption(REPO_ROOT / relative_path)
                digest = hashlib.sha256(
                    _normalized(caption).encode("utf-8")
                ).hexdigest()
                self.assertEqual(
                    digest,
                    APPROVED_NORMALIZED_CAPTION_SHA256[figure],
                )
                if figure == "Figure 5":
                    self.assertEqual(_word_count(caption), 189)

    def test_denominator_semantics_and_claim_boundaries_remain_explicit(self) -> None:
        required = {
            "Figure 1": (
                "They retain separate denominators",
                "do not form a composite ranking",
            ),
            "Figure 4": (
                "each with 91 baseline-visible spot opportunities per method",
                "seven and six evaluable gaps",
                "each method's accepted-fit denominator",
                "method-specific \\(n\\)",
                (
                    "One series and two pairs support a mechanism result, "
                    "not a general ranking"
                ),
                "Pair-specific denominators remain separate",
            ),
            "Figure 5": (
                "Every method yields 701/701 endpoints per pair",
                "24 consecutive Gaussian fits fail only the amplitude gate",
                "does not show identity loss or compare the five methods",
                "do not establish crystallographic anisotropy",
            ),
        }

        for figure, anchors in required.items():
            caption = _normalized(
                _caption(REPO_ROOT / CAPTION_PATHS[figure])
            )
            with self.subTest(figure=figure):
                for anchor in anchors:
                    self.assertIn(anchor, caption)

    def test_inventory_and_results_use_the_approved_figure_numbering(self) -> None:
        inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        by_id = {item["id"]: item for item in inventory["items"]}
        expected_artifacts = {
            "Figure 3": "manuscript/figures/figure03_independent_scane_stress",
            "Figure 4": (
                "manuscript/figures/figure04_controlled_sam2_memory_tradeoff"
            ),
            "Figure 5": "manuscript/figures/figure05_cu111_physical_endpoint",
        }
        for figure, artifact_path in expected_artifacts.items():
            self.assertEqual(by_id[figure]["artifact_path"], artifact_path)

        manuscript = (MANUSCRIPT_ROOT / "manuscript.md").read_text(
            encoding="utf-8"
        )
        result_sections = {
            "Figure 5": manuscript[
                manuscript.index("### 4.2") : manuscript.index("### 4.3")
            ],
            "Figure 3": manuscript[
                manuscript.index("### 4.3") : manuscript.index("### 4.4")
            ],
            "Figure 4": manuscript[
                manuscript.index("### 4.5") : manuscript.index("### 4.6")
            ],
        }
        for figure, section in result_sections.items():
            self.assertIn(figure, section)


if __name__ == "__main__":
    unittest.main()
