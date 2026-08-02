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
    ("4.3 Independent Ag(111) scans separate availability from conditional residuals"),
    "4.4 Predeclared measurement-window and capture-contract sensitivity",
    ("4.5 Controlled SAM2 family exposes a continuity-consistency trade-off"),
    "4.6 Supporting compatibility and missingness checks",
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
APPROVED_ABSTRACT = r"""Quantitative low-energy electron diffraction (LEED) analysis requires
persistent association between a measurement region and its target
diffraction spot as patterns move and visibility changes. We compared five
heterogeneous tracking strategies through a common position-measurement
contract: each native output supplied a proposal for a 48 × 48-pixel ROI,
and the same two-dimensional Gaussian estimator produced the reported spot
position. Robustness was assessed with fixed opportunity sets for coverage
and post-gap recovery, accepted-fit denominators for geometry inconsistency,
and method-specific surviving \(n\) for conditional residuals.

Validation combined known translations, a 701-image Cu(111) positive
control, two independent Ag(111) energy scans with natural \(I(V)\) gaps,
and a controlled SAM2-family memory experiment. Translation errors were
0.0043–0.0265 px with 74/80–80/80 accepted measurements. All methods
produced 2,804/2,804 Cu(111) measurements and 701/701 endpoints per primary
pair. In the Ag(111) scans, coverage was 188/221–217/221, recovery
17/24–23/24, and conditional median pair-distance residuals 1.737–1.837 px
(\(n=68–91\)). SAMURAI and DAM4SAM increased coverage over SAM2 but also
increased geometry-inconsistent continuation; this mechanism result is not
a general ranking.

Across 15 predeclared perturbations of ROI geometry and acceptance gates,
absolute outcomes changed, but Lucas–Kanade retained the lowest coverage and
the denominator-aware interpretation remained stable. The results assess
measurement-chain robustness, not absolute tracking accuracy. Proposal
differences were strongly suppressed when the target spot remained inside
the common estimator's capture region, making this a supporting mechanism
rather than the primary contribution."""


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _section(text: str, start: str, end: str) -> str:
    return text[text.index(start) + len(start) : text.index(end)]


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
    return tuple(heading for heading in _headings(text, 3) if heading.startswith("4."))


def _subsections(text: str, section_number: int) -> tuple[str, ...]:
    prefix = f"{section_number}."
    return tuple(
        heading for heading in _headings(text, 3) if heading.startswith(prefix)
    )


def validate_manuscript_package(manuscript_directory: Path) -> dict[str, object]:
    """Validate structure, terminology, placeholders, and claim boundaries."""

    manuscript = (manuscript_directory / "manuscript.md").read_text(encoding="utf-8")
    claim_map = (manuscript_directory / "claim-evidence-map.md").read_text(
        encoding="utf-8"
    )
    supplement = (manuscript_directory / "supplementary-material.md").read_text(
        encoding="utf-8"
    )
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

    abstract = _section(manuscript, "## Abstract", "## Keywords")
    if _normalized(abstract) != _normalized(APPROVED_ABSTRACT):
        errors.append("The Abstract differs from the approved 244-word text.")
    abstract_words = re.findall(
        r"[^\W_]+(?:[-–][^\W_]+)*", abstract, flags=re.UNICODE
    )
    if len(abstract_words) != 244:
        errors.append("The approved Abstract must contain exactly 244 words.")
    if re.search(r"\b(?:table|figure|supplementary)\b", abstract, re.IGNORECASE):
        errors.append("The Abstract must not refer to tables, figures, or SI.")
    association_heading = (
        "### 1.1 Quantitative LEED requires persistent spot association"
    )
    if association_heading not in manuscript:
        errors.append("Section 1.1 does not use the approved association framing.")

    normalized_manuscript_text = _normalized(manuscript)
    revised_content_anchors = (
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
        "1.737 px (n=84) for EasyLEED",
        "1.764 px (n=68) for Lucas--Kanade",
        "1.770 px (n=90) for OSTrack",
        "1.788 px (n=91) for DAM4SAM",
        "1.837 px (n=91) for LocoTrack",
        (
            "24 digital image states: four source trials × "
            "(one seed state + five displaced states)"
        ),
        (
            "80 displaced-spot and 40 displaced-pair evaluation opportunities "
            "per method after exclusion of four seed states"
        ),
        (
            "Sensitivity thus supports claims about denominator-aware "
            "measurement-chain robustness within the acquisition-closed case "
            "boundary"
        ),
        (
            "Conditional residuals must therefore be reported with their "
            "method-specific surviving n, alongside coverage and recovery evaluated "
            "on fixed opportunity sets and the corresponding geometry-consistency "
            "outcomes."
        ),
    )
    for anchor in revised_content_anchors:
        if anchor not in normalized_manuscript_text:
            errors.append(f"Approved revised-content anchor is missing: {anchor}")
    if "The present work does not test other ROI sizes" in manuscript:
        errors.append("The superseded ROI-sensitivity limitation remains.")

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
                f"Main Section 3 contains an SI-only audit detail: {audit_detail}"
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

    reconstruction_command = (
        "python -m scripts.reconstruct_publication --clean --validate"
    )
    if reconstruction_command not in manuscript:
        errors.append("The scientific reconstruction command is missing.")

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
