"""Fail-closed validation of the reader-facing manuscript package."""

from __future__ import annotations

import re
from pathlib import Path


FROZEN_TITLE = (
    "Evaluating Heterogeneous LEED Spot-Tracking Methods under a Common "
    "Spot-Position Measurement Contract"
)
MAIN_SECTIONS = (
    "1. Introduction",
    "2. Measurement chain and validation design",
    "3. Tracking strategies and reproducible comparison contract",
    "4. Results",
    "5. Discussion",
    "6. Conclusions",
    "7. Modular end matter",
)
RESULT_SECTIONS = (
    "4.1 Controlled translations isolate measurement-chain behavior",
    "4.2 Cu(111) positive control establishes endpoint agreement",
    (
        "4.3 Independent Ag(111) scans separate availability from "
        "conditional residuals"
    ),
    (
        "4.4 Controlled SAM2 family exposes a continuity-consistency "
        "trade-off"
    ),
    "4.5 Supporting compatibility and missingness checks",
)
METHOD_SECTIONS = (
    "2.1 Tracking proposal and common measurement ROI",
    "2.2 Gaussian-refined position and pairwise observable",
    "2.3 Measurement-chain outcomes",
    "2.4 Evidence roles and independence",
)
AUTHORIZED_PLACEHOLDERS = (
    "AUTHOR",
    "EXPERIMENT",
    "RELEASE-AUTHORIZATION",
    "SUBMISSION",
)
REQUIRED_ANCHORS = (
    "0.0043",
    "0.0265",
    "0.0053",
    "2,804/2,804",
    "701/701",
    "188/221",
    "217/221",
    "17/24",
    "23/24",
    "1.737",
    "1.837",
)
PROHIBITED_POSITIVE_CLAIMS = (
    "establishes absolute tracking accuracy",
    "provides identity ground truth",
    "proves spot identity",
    "universal tracker ranking",
    "best tracker overall",
    "new physical discovery",
    "performs structural i(v) analysis",
)


def _headings(text: str, level: int) -> tuple[str, ...]:
    prefix = "#" * level + " "
    return tuple(
        line[len(prefix) :]
        for line in text.splitlines()
        if line.startswith(prefix) and not line.startswith(prefix + "#")
    )


def _numbered_main_sections(text: str) -> tuple[str, ...]:
    return tuple(
        heading for heading in _headings(text, 2) if re.match(r"^\d+\. ", heading)
    )


def _result_sections(text: str) -> tuple[str, ...]:
    return tuple(
        heading for heading in _headings(text, 3) if heading.startswith("4.")
    )


def _subsections(text: str, section_number: int) -> tuple[str, ...]:
    prefix = f"{section_number}."
    return tuple(
        heading
        for heading in _headings(text, 3)
        if heading.startswith(prefix)
    )


def validate_manuscript_package(manuscript_directory: Path) -> dict[str, object]:
    """Validate structure, terminology, placeholders, and claim boundaries."""

    manuscript = (manuscript_directory / "manuscript.md").read_text(
        encoding="utf-8"
    )
    claim_map = (manuscript_directory / "claim-evidence-map.md").read_text(
        encoding="utf-8"
    )
    supplement = (
        manuscript_directory / "supplementary-material.md"
    ).read_text(encoding="utf-8")
    errors: list[str] = []

    first_line = manuscript.splitlines()[0] if manuscript else ""
    if first_line != f"# {FROZEN_TITLE}":
        errors.append("The manuscript title differs from the frozen title.")

    main_sections = _numbered_main_sections(manuscript)
    if main_sections != MAIN_SECTIONS:
        errors.append("Numbered main sections differ from the approved order.")

    result_sections = _result_sections(manuscript)
    if result_sections != RESULT_SECTIONS:
        errors.append("Result subsections differ from the approved evidence order.")

    method_sections = _subsections(manuscript, 2)
    if method_sections != METHOD_SECTIONS:
        errors.append("Section 2 differs from the approved measurement-chain order.")

    section_three = manuscript[
        manuscript.index(f"## {MAIN_SECTIONS[2]}") : manuscript.index(
            f"## {MAIN_SECTIONS[3]}"
        )
    ]
    expected_table_two_header = (
        "| Method | Native output | Proposal adapter | Native availability | "
        "Shared downstream measurement | Reset policy | Study role |"
    )
    if expected_table_two_header not in section_three:
        errors.append("Table 2 does not expose the approved compact contract.")
    for audit_detail in (
        "Checkpoint or algorithm configuration",
        "upstream revision",
        "Hiera Base+",
        "2.5.2",
        "1.03",
        "41 × 41",
        "256 × 256",
        "SHA-256",
    ):
        if audit_detail in section_three:
            errors.append(
                "Main Section 3 contains an SI-only audit detail: "
                f"{audit_detail}"
            )
    if re.search(r"`[0-9a-f]{7,64}`", section_three):
        errors.append("Main Section 3 contains an SI-only revision or hash.")

    reader_facing = "\n".join((manuscript, claim_map, supplement))
    placeholder_starts = reader_facing.count("TODO[")
    placeholders = tuple(re.findall(r"TODO\[([^\]\r\n]+)\]", reader_facing))
    if len(placeholders) != placeholder_starts:
        errors.append("The reader-facing package contains a malformed placeholder.")
    placeholder_kinds = tuple(sorted(set(placeholders)))
    if placeholder_kinds != AUTHORIZED_PLACEHOLDERS:
        errors.append("The reader-facing package contains unauthorized placeholders.")
    experimental_count = placeholders.count("EXPERIMENT")
    if experimental_count != 1:
        errors.append("Exactly one experimental placeholder is required.")

    if "Control variable/range" in manuscript:
        errors.append(
            "Experimental control ranges must be deferred from the narrative."
        )
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:eV|°C)\b", manuscript):
        errors.append(
            "Reader-facing acquisition values must be deferred to the "
            "experimental-team revision."
        )
    for term in (
        "TODO[TEXT]",
        "TODO[DECISION]",
        "26a21_ScanE",
        "26a24_ScanE",
        ".scratch/",
    ):
        if term in manuscript:
            errors.append(f"Internal or superseded manuscript term remains: {term}")

    for anchor in REQUIRED_ANCHORS:
        if anchor not in manuscript:
            errors.append(f"Required numerical anchor is missing: {anchor}")

    normalized_manuscript = manuscript.casefold()
    for claim in PROHIBITED_POSITIVE_CLAIMS:
        if claim in normalized_manuscript:
            errors.append(f"Prohibited positive claim remains: {claim}")

    exact_term = "post-gap branch-consistent recovery"
    for name, text in (
        ("manuscript", manuscript),
        ("claim map", claim_map),
        ("supplement", supplement),
    ):
        if exact_term not in text.casefold():
            errors.append(f"The {name} does not use the frozen recovery term.")

    normalized_supplement = re.sub(r"\s+", " ", supplement)
    for supplementary_contract_term in (
        "### Native proposal availability and preprocessing contract",
        r"P_{\mathrm{vis}}>0.5",
        r"Gaussian blur with \(\sigma=5\)",
        "white top-hat with a disk of radius 5",
        r"forward--backward closure error \(\leq8\)",
        r"native Gaussian \(R^2\) threshold 0.8",
    ):
        if supplementary_contract_term not in normalized_supplement:
            errors.append(
                "The supplementary native-method contract is incomplete: "
                f"{supplementary_contract_term}"
            )

    claim_ids = tuple(re.findall(r"^\| (C\d+) \|", claim_map, flags=re.MULTILINE))
    expected_claim_ids = tuple(f"C{i}" for i in range(1, 10))
    if claim_ids != expected_claim_ids:
        errors.append("The claim-evidence map must retain claims C1-C9 in order.")

    required_boundary_phrases = (
        "not instrument-traceable absolute tracking accuracy",
        "not external absolute tracking accuracy",
        "not a general ranking",
        "not four independent acquisitions",
    )
    normalized_reader_facing = reader_facing.casefold()
    for phrase in required_boundary_phrases:
        if phrase not in normalized_reader_facing:
            errors.append(f"Required claim boundary is missing: {phrase}")

    if errors:
        raise ValueError("Manuscript contract failed:\n- " + "\n- ".join(errors))

    return {
        "authorized_placeholder_kinds": placeholder_kinds,
        "claim_ids": claim_ids,
        "experimental_placeholder_count": experimental_count,
        "main_section_count": len(main_sections),
        "result_section_count": len(result_sections),
    }


if __name__ == "__main__":
    validate_manuscript_package(Path(__file__).resolve().parent)
