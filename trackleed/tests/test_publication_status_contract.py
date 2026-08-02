from __future__ import annotations

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "docs" / "publication-status-contract.md"
DOMAIN_ROUTING_PATH = REPO_ROOT / "docs" / "agents" / "domain.md"

EXPECTED_DEFINITIONS = {
    "Content-revision-complete": (
        "all approved content, denominator, sensitivity, display, reconstruction, "
        "rebaseline, and page-level acceptance gates in this record have passed; "
        "this is not `submission-ready`."
    ),
    "Page-level render acceptance": (
        "fail-closed review of every rendered page of every canonical "
        "reader-facing document, with the reviewed page set and acceptance "
        "decision recorded."
    ),
    "Controlled scientific-core rebaseline": (
        "authorized, versioned, predecessor-to-successor fingerprint replacement "
        "with preserved history, component diff, atomic write, and repeatable "
        "clean-checkout validation."
    ),
    "Authorized deferred publication input": (
        "an explicit `AUTHOR`, `EXPERIMENT`, `RELEASE-AUTHORIZATION`, or "
        "`SUBMISSION` placeholder permitted at `content-revision-complete` but not "
        "treated as completed content."
    ),
}


def _normalize_markdown_text(value: str) -> str:
    return " ".join(value.split())


def _read_definitions(document: str) -> dict[str, str]:
    matches = re.finditer(
        r"(?ms)^- \*\*(?P<term>[^*]+)\*\* — (?P<definition>.*?)"
        r"(?=^- \*\*|^## |\Z)",
        document,
    )
    return {
        match.group("term"): _normalize_markdown_text(match.group("definition"))
        for match in matches
    }


class PublicationStatusContractTests(unittest.TestCase):
    def test_canonical_contract_preserves_approved_definitions(self) -> None:
        document = CONTRACT_PATH.read_text(encoding="utf-8")

        self.assertIn("Canonical source: this document", document)
        self.assertEqual(_read_definitions(document), EXPECTED_DEFINITIONS)

    def test_domain_documentation_routes_revision_terms_to_contract(self) -> None:
        routing = DOMAIN_ROUTING_PATH.read_text(encoding="utf-8")

        self.assertIn("publication-status-contract.md", routing)
        for term in EXPECTED_DEFINITIONS:
            self.assertIn(f"`{term}`", routing)


if __name__ == "__main__":
    unittest.main()
