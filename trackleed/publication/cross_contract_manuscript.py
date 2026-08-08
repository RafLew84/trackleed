"""Freeze and validate the focused cross-contract main manuscript."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from trackleed.publication.cross_contract_figures import (
    CrossContractFigureError,
    validate_cross_contract_figures,
)

DEFAULT_PACKAGE_PATH = Path("manuscript/cross_contract/main_manuscript")
EXPECTED_CONTRACT_SHA256 = (
    "a8c281d3679013994eba17341045b97174a12274770af1120e6424a6a0198dae"
)
EXPECTED_MANIFEST_SHA256 = (
    "4d269530a760a2debcd84f588ed1367f0ab1d7998d26724b9a2b7a9bce6bbafe"
)
STATIC_PATHS = ("README.md", "contract.json", "manuscript.md")
GENERATED_PATHS = ("manifest.json", "validation_report.json")
OUTPUT_ROLES = {
    "README.md": "manuscript scope, workflow, and package boundary",
    "contract.json": "frozen manuscript structure, budget, display, and source contract",
    "manuscript.md": "venue-portable reader-facing scientific draft",
}
WORD_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", flags=re.UNICODE)


class CrossContractManuscriptError(RuntimeError):
    """Raised when the cross-contract manuscript package fails validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CrossContractManuscriptError(
            f"cross-contract manuscript source is unavailable: {path}"
        ) from exc
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CrossContractManuscriptError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CrossContractManuscriptError(f"invalid {label}: {path}")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_contract(workspace_root: Path, package: Path) -> dict[str, Any]:
    contract_path = package / "contract.json"
    if not contract_path.is_file():
        raise CrossContractManuscriptError(
            "cross-contract manuscript contract is unavailable"
        )
    if _sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise CrossContractManuscriptError("cross-contract manuscript contract changed")
    contract = _load_json(contract_path, "cross-contract manuscript contract")
    if (
        contract.get("schema_version") != "1.0"
        or contract.get("manuscript_id") != "focused-cross-contract-main-manuscript"
        or contract.get("status") != "frozen-before-main-manuscript-drafting"
        or contract.get("required_main_sections")
        != ["Introduction", "Methods", "Results", "Discussion", "Conclusions"]
    ):
        raise CrossContractManuscriptError(
            "cross-contract manuscript contract identity changed"
        )
    bindings = contract.get("source_bindings")
    if not isinstance(bindings, dict) or len(bindings) != 5:
        raise CrossContractManuscriptError(
            "cross-contract manuscript source bindings changed"
        )
    for binding_id, binding in bindings.items():
        if not isinstance(binding, dict):
            raise CrossContractManuscriptError(
                f"invalid manuscript source binding: {binding_id}"
            )
        relative = binding.get("path")
        expected_sha256 = binding.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_sha256, str):
            raise CrossContractManuscriptError(
                f"invalid manuscript source binding: {binding_id}"
            )
        if _sha256(workspace_root / relative) != expected_sha256:
            raise CrossContractManuscriptError(
                f"cross-contract manuscript source drift: {binding_id}"
            )
    try:
        validate_cross_contract_figures(workspace_root)
    except CrossContractFigureError as exc:
        raise CrossContractManuscriptError(
            f"upstream cross-contract figure package failed validation: {exc}"
        ) from exc
    return contract


def _section(text: str, start: str, end: str | None) -> str:
    try:
        start_index = text.index(start) + len(start)
        end_index = text.index(end, start_index) if end is not None else len(text)
    except ValueError as exc:
        raise CrossContractManuscriptError(
            f"cross-contract manuscript section is unavailable: {start}"
        ) from exc
    return text[start_index:end_index].strip()


def _word_count(text: str) -> int:
    without_figures = re.sub(
        r"^!\[[^\]]*\]\([^\r\n]+\)\s*$", "", text, flags=re.MULTILINE
    )
    without_captions = re.sub(
        r"^\*\*Figure [1-3]\.\*\*[^\r\n]*$",
        "",
        without_figures,
        flags=re.MULTILINE,
    )
    without_headings = re.sub(
        r"^#{1,6}\s+[^\r\n]+$", "", without_captions, flags=re.MULTILINE
    )
    without_citations = re.sub(r"\[@[^\]]+\]", "", without_headings)
    without_code = re.sub(r"`([^`]+)`", r"\1", without_citations)
    return len(WORD_PATTERN.findall(without_code))


def _analyze_manuscript(
    workspace_root: Path,
    package: Path,
    contract: dict[str, Any],
) -> dict[str, object]:
    manuscript_path = package / "manuscript.md"
    try:
        manuscript = manuscript_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CrossContractManuscriptError(
            "cross-contract main manuscript is unavailable"
        ) from exc
    expected_title = f"# {contract['title']}"
    if not manuscript.startswith(expected_title + "\n"):
        raise CrossContractManuscriptError("cross-contract manuscript title changed")

    abstract = _section(manuscript, "## Abstract", "## 1. Introduction")
    abstract_word_count = _word_count(abstract)
    if abstract_word_count > 200:
        raise CrossContractManuscriptError(
            f"cross-contract abstract exceeds 200 words: {abstract_word_count}"
        )
    if re.search(r"\b(?:figure|table|supplementary)\b", abstract, flags=re.IGNORECASE):
        raise CrossContractManuscriptError(
            "cross-contract abstract is not evidence self-contained"
        )

    section_matches = list(
        re.finditer(r"^## ([1-5])\. ([^\r\n]+)$", manuscript, flags=re.MULTILINE)
    )
    main_sections = [match.group(2) for match in section_matches]
    if main_sections != contract["required_main_sections"]:
        raise CrossContractManuscriptError(
            "cross-contract manuscript main-section structure changed"
        )
    main_text = _section(manuscript, "## 1. Introduction", "## References")
    main_text_word_count = _word_count(main_text)
    if not 4200 <= main_text_word_count <= 4600:
        raise CrossContractManuscriptError(
            "cross-contract main-text target changed: "
            f"{main_text_word_count} words, expected 4200-4600"
        )
    results = _section(manuscript, "## 3. Results", "## 4. Discussion")
    conclusions = _section(manuscript, "## 5. Conclusions", "## References")
    if re.search(r"\bsensitivit", abstract, flags=re.IGNORECASE):
        raise CrossContractManuscriptError(
            "claim XC7 appears outside its approved Results/Conclusions sections"
        )
    if "more ViPErLEED-only than Gaussian-only" in conclusions:
        raise CrossContractManuscriptError(
            "claim XC2 appears outside its approved Abstract/Results sections"
        )
    if (
        "sensitivity" not in results.casefold()
        or "sensitivity" not in conclusions.casefold()
    ):
        raise CrossContractManuscriptError(
            "claim XC7 is missing from an approved manuscript section"
        )

    figures = re.findall(
        r"^!\[Figure ([1-3])\]\(([^\r\n]+)\)$", manuscript, flags=re.MULTILINE
    )
    expected_figures = [
        (str(index), str(item["path"]))
        for index, item in enumerate(contract["display_inventory"]["figures"], 1)
    ]
    if figures != expected_figures:
        raise CrossContractManuscriptError(
            "cross-contract manuscript figure inventory changed"
        )
    table_count = len(re.findall(r"^\s*\|.*\|\s*$", manuscript, flags=re.MULTILINE))
    table_count += len(re.findall(r"<table\b", manuscript, flags=re.IGNORECASE))
    if table_count:
        raise CrossContractManuscriptError(
            "cross-contract main manuscript contains a table"
        )

    captions = _load_json(
        workspace_root / "manuscript/cross_contract/figures/captions.json",
        "cross-contract figure captions",
    )
    for index in range(1, 4):
        caption = captions[f"figure0{index}"]
        if manuscript.count(f"**Figure {index}.** {caption}") != 1:
            raise CrossContractManuscriptError(
                f"cross-contract manuscript caption changed: figure{index:02d}"
            )

    placeholders = re.findall(r"TODO\[([^\]\r\n]+)\]", manuscript)
    expected_deferred = contract["authorized_deferred_inputs"]
    expected_categories = [str(item["category"]) for item in expected_deferred]
    if placeholders != expected_categories:
        raise CrossContractManuscriptError(
            "cross-contract manuscript deferred-input inventory changed"
        )
    for item in expected_deferred:
        if manuscript.count(str(item["marker"])) != 1:
            raise CrossContractManuscriptError(
                f"authorized deferred-input marker changed: {item['category']}"
            )
    if manuscript.count("TODO[") != len(placeholders):
        raise CrossContractManuscriptError(
            "cross-contract manuscript contains a malformed placeholder"
        )

    semantic_text = re.sub(r"\s+", " ", re.sub(r"[*_`]", "", manuscript))
    noncanonical_terms = (
        r"\btarget-inconsistenc(?:y|ies)\b",
        r"(?<!Cross-contract )\btarget inconsistenc(?:y|ies)\b",
        r"\bunresolved (?:event|events|class|category)\b",
    )
    for pattern in noncanonical_terms:
        if re.search(pattern, semantic_text, flags=re.IGNORECASE):
            raise CrossContractManuscriptError(
                "cross-contract manuscript uses noncanonical outcome terminology"
            )
    required_terms = (
        "Gaussian-contract measurability",
        "ViPEr-native direct detectability",
        "Gaussian-contract gap",
        "ViPEr-native visibility gap",
        "Cross-contract target inconsistency",
        "Unresolved cross-contract outcome",
        "EasyLEED",
        "LocoTrack",
        "OSTrack",
        "DAM4SAM",
        "Lucas-Kanade",
    )
    for term in required_terms:
        if term not in semantic_text:
            raise CrossContractManuscriptError(
                f"required cross-contract manuscript term is missing: {term}"
            )
    required_anchors = (
        "222/0/83/19",
        "262/10/43/9",
        "259/10/46/9",
        "205/1/100/18",
        "114 Gaussian-contract gaps",
        "121 overlap-connected",
        "118 of 121",
        "124 accepted",
        "83 branch-model",
        "16 Cross-contract target inconsistencies",
        "25 Unresolved cross-contract outcomes",
        "45 method-variant rows",
    )
    for anchor in required_anchors:
        if anchor not in semantic_text:
            raise CrossContractManuscriptError(
                f"required cross-contract numerical anchor is missing: {anchor}"
            )
    prohibited_positive_claims = (
        "ViPErLEED is ground truth",
        "we identify true identity switches",
        "physical disappearance was observed",
        "the best tracker",
        "absolute tracking accuracy was established",
    )
    normalized = manuscript.casefold()
    for claim in prohibited_positive_claims:
        if claim.casefold() in normalized:
            raise CrossContractManuscriptError(
                f"prohibited cross-contract claim remains: {claim}"
            )
    for boundary in (
        "ViPErLEED is therefore neither a sixth equivalent tracker",
        "not a universal tracker ranking",
        "Neither endpoint supplies absolute position or spot-identity ground truth",
        "existing-acquisition-only evidence",
    ):
        if boundary not in semantic_text:
            raise CrossContractManuscriptError(
                f"required cross-contract claim boundary is missing: {boundary}"
            )

    citation_keys = sorted(set(re.findall(r"@([A-Za-z0-9_-]+)", manuscript)))
    allowed_citations = {
        "cho2024locotrack",
        "lucas1981iterative",
        "mayer2012easyleed",
        "pendry1974leed",
        "schmid2025viperleed",
        "videnovic2025dam4sam",
        "ye2022ostrack",
    }
    if not set(citation_keys).issubset(allowed_citations):
        raise CrossContractManuscriptError(
            "cross-contract manuscript contains an uncontracted citation"
        )
    return {
        "abstract_word_count": abstract_word_count,
        "authorized_deferred_inputs": expected_categories,
        "citation_keys": citation_keys,
        "claim_ids": contract["claim_ids"],
        "figure_count": len(figures),
        "main_sections": main_sections,
        "main_text_word_count": main_text_word_count,
        "table_count": table_count,
    }


def _build_manifest(package: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for relative, role in sorted(OUTPUT_ROLES.items()):
        path = package / relative
        if not path.is_file():
            raise CrossContractManuscriptError(
                f"cross-contract manuscript manifest input is unavailable: {relative}"
            )
        files.append(
            {
                "path": relative,
                "role": role,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "file_count": len(files),
        "files": files,
        "manuscript_id": "focused-cross-contract-main-manuscript",
        "schema_version": "1.0",
        "status": "frozen-cross-contract-manuscript-manifest",
    }


def _validation_report(
    workspace_root: Path,
    package: Path,
    contract: dict[str, Any],
) -> dict[str, object]:
    return {
        **_analyze_manuscript(workspace_root, package, contract),
        "claim_boundary_decision_sha256": contract["source_bindings"][
            "author_approved_claim_boundary"
        ]["sha256"],
        "fail_closed": True,
        "main_figure_manifest_sha256": contract["source_bindings"][
            "main_figure_manifest"
        ]["sha256"],
        "manifest_sha256": _sha256(package / "manifest.json"),
        "schema_version": "1.0",
        "scientific_contract_sha256": contract["source_bindings"][
            "scientific_contract"
        ]["sha256"],
        "status": "validated-cross-contract-manuscript",
    }


def _package_file(package: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise CrossContractManuscriptError("manifested manuscript path changed")
    path = (package / value).resolve()
    if not path.is_relative_to(package) or not path.is_file():
        raise CrossContractManuscriptError(
            f"manifested manuscript file is unavailable: {value}"
        )
    return path


def _validate_frozen_package(
    workspace_root: Path,
    package: Path,
) -> dict[str, object]:
    contract = _load_contract(workspace_root, package)
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        raise CrossContractManuscriptError(
            f"cross-contract manuscript manifest is unavailable: {package}"
        )
    if EXPECTED_MANIFEST_SHA256 and _sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise CrossContractManuscriptError("cross-contract manuscript manifest changed")
    manifest = _load_json(manifest_path, "cross-contract manuscript manifest")
    manifested_files = manifest.get("files")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("manuscript_id") != "focused-cross-contract-main-manuscript"
        or manifest.get("status") != "frozen-cross-contract-manuscript-manifest"
        or not isinstance(manifested_files, list)
        or manifest.get("file_count") != len(manifested_files)
    ):
        raise CrossContractManuscriptError(
            "cross-contract manuscript manifest identity changed"
        )
    manifested: set[str] = set()
    for item in manifested_files:
        if not isinstance(item, dict):
            raise CrossContractManuscriptError(
                "cross-contract manuscript manifest record changed"
            )
        relative = item.get("path")
        path = _package_file(package, relative)
        if not isinstance(relative, str) or relative in manifested:
            raise CrossContractManuscriptError(
                "duplicate cross-contract manuscript manifest path"
            )
        if (
            item.get("role") != OUTPUT_ROLES.get(relative)
            or item.get("sha256") != _sha256(path)
            or item.get("size_bytes") != path.stat().st_size
        ):
            raise CrossContractManuscriptError(
                f"manifested cross-contract manuscript drift: {relative}"
            )
        manifested.add(relative)
    if manifested != set(OUTPUT_ROLES):
        raise CrossContractManuscriptError(
            "cross-contract manuscript manifest coverage changed"
        )
    actual = {path.relative_to(package).as_posix() for path in package.rglob("*")}
    expected = set(OUTPUT_ROLES) | {"manifest.json", "validation_report.json"}
    if actual != expected:
        raise CrossContractManuscriptError(
            "cross-contract manuscript package file inventory changed"
        )
    if manifest_path.read_bytes() != _json_bytes(_build_manifest(package)):
        raise CrossContractManuscriptError(
            "cross-contract manuscript manifest no longer reproduces"
        )
    report = _validation_report(workspace_root, package, contract)
    report_path = package / "validation_report.json"
    if not report_path.is_file() or report_path.read_bytes() != _json_bytes(report):
        raise CrossContractManuscriptError(
            "cross-contract manuscript validation report changed"
        )
    return report


def freeze_cross_contract_manuscript(
    workspace_root: Path,
    package_path: Path | None = None,
) -> dict[str, object]:
    """Manifest, validate, and publish the static manuscript package once."""

    workspace_root = workspace_root.resolve()
    package = (
        package_path.resolve()
        if package_path is not None
        else workspace_root / DEFAULT_PACKAGE_PATH
    )
    _load_contract(workspace_root, package)
    if any((package / relative).exists() for relative in GENERATED_PATHS):
        raise FileExistsError(
            f"Refusing to replace frozen cross-contract manuscript: {package}"
        )
    for relative in STATIC_PATHS:
        if not (package / relative).is_file():
            raise CrossContractManuscriptError(
                f"cross-contract manuscript source is unavailable: {relative}"
            )
    with TemporaryDirectory(
        prefix="cross-contract-manuscript-", dir=package.parent
    ) as temporary:
        stage = Path(temporary) / "main_manuscript"
        stage.mkdir()
        for relative in STATIC_PATHS:
            shutil.copyfile(package / relative, stage / relative)
        contract = _load_contract(workspace_root, stage)
        (stage / "manifest.json").write_bytes(_json_bytes(_build_manifest(stage)))
        (stage / "validation_report.json").write_bytes(
            _json_bytes(_validation_report(workspace_root, stage, contract))
        )
        validated = _validate_frozen_package(workspace_root, stage)
        if any(
            _sha256(package / relative) != _sha256(stage / relative)
            for relative in STATIC_PATHS
        ):
            raise CrossContractManuscriptError(
                "cross-contract manuscript sources changed during staged validation"
            )
        for relative in GENERATED_PATHS:
            os.replace(stage / relative, package / relative)
    return validated


def validate_cross_contract_manuscript(
    workspace_root: Path,
    package_path: Path | None = None,
) -> dict[str, object]:
    """Validate the frozen focused venue-portable manuscript draft."""

    workspace_root = workspace_root.resolve()
    package = (
        package_path.resolve()
        if package_path is not None
        else workspace_root / DEFAULT_PACKAGE_PATH
    )
    if not (package / "contract.json").is_file():
        raise CrossContractManuscriptError(
            "cross-contract manuscript contract is unavailable"
        )
    return _validate_frozen_package(workspace_root, package)


__all__ = [
    "CrossContractManuscriptError",
    "freeze_cross_contract_manuscript",
    "validate_cross_contract_manuscript",
]
