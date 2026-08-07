"""Freeze and validate the cross-contract manuscript claim boundary."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from trackleed.publication.cross_contract_outcomes import (
    CrossContractValidationError,
    validate_cross_contract_outcomes,
)

DEFAULT_PACKAGE_PATH = Path("manuscript/cross_contract/claim_boundary")
EXPECTED_DECISION_SHA256 = (
    "0f3e4d727fdf66721fa76924728fcf67ae1443c6c09654395093b20434ed6c18"
)
EXPECTED_MANIFEST_SHA256: str | None = None
GENERATED_FILES = (
    "author_checkpoint.json",
    "manifest.json",
    "validation_report.json",
)


class CrossContractClaimBoundaryError(RuntimeError):
    """Raised when the claim-boundary decision cannot be authenticated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CrossContractClaimBoundaryError(
            f"claim-boundary source is unavailable: {path}"
        ) from exc
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossContractClaimBoundaryError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CrossContractClaimBoundaryError(f"invalid {label}: {path}")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))


def _resolve_package(workspace_root: Path, package_path: Path | None) -> Path:
    return (
        package_path.resolve()
        if package_path is not None
        else workspace_root / DEFAULT_PACKAGE_PATH
    )


def _validate_proposal(
    workspace_root: Path,
    package_path: Path,
) -> tuple[dict[str, Any], str]:
    decision_path = package_path / "decision_record.json"
    if not decision_path.is_file():
        raise CrossContractClaimBoundaryError(
            "cross-contract claim-boundary decision is unavailable"
        )
    decision = _load_json(decision_path, "cross-contract decision record")
    if (
        decision.get("schema_version") != "1.0"
        or decision.get("decision_id") != "cross-contract-thesis-and-claim-boundary"
        or decision.get("status") != "proposed_for_author_checkpoint"
    ):
        raise CrossContractClaimBoundaryError(
            "cross-contract decision identity or status changed"
        )

    bindings = decision.get("source_bindings")
    expected_binding_ids = {
        "cross_contract_outcomes_headline",
        "cross_contract_outcomes_manifest",
        "gap_comparisons",
        "method_summary",
        "scientific_contract",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected_binding_ids:
        raise CrossContractClaimBoundaryError(
            "cross-contract decision source bindings changed"
        )
    for binding_id, binding in bindings.items():
        if not isinstance(binding, dict):
            raise CrossContractClaimBoundaryError(
                f"invalid cross-contract source binding: {binding_id}"
            )
        relative_path = binding.get("path")
        expected_sha256 = binding.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
            raise CrossContractClaimBoundaryError(
                f"invalid cross-contract source binding: {binding_id}"
            )
        source_path = workspace_root / relative_path
        if _sha256(source_path) != expected_sha256:
            raise CrossContractClaimBoundaryError(
                f"cross-contract decision source drift: {binding_id}"
            )

    try:
        outcomes = validate_cross_contract_outcomes(workspace_root)
    except CrossContractValidationError as exc:
        raise CrossContractClaimBoundaryError(
            f"frozen cross-contract outcomes failed validation: {exc}"
        ) from exc
    if outcomes.get("status") != "validated-cross-contract-outcomes":
        raise CrossContractClaimBoundaryError(
            "frozen cross-contract outcomes did not validate"
        )

    thesis = decision.get("thesis_decision")
    if (
        not isinstance(thesis, dict)
        or thesis.get("selected")
        != "combined_continuity_consistency_and_contract_dependent_observability"
        or not isinstance(thesis.get("statement"), str)
        or not thesis["statement"].strip()
        or len(thesis.get("rejected_alternatives", [])) != 2
    ):
        raise CrossContractClaimBoundaryError("selected thesis decision changed")
    title = decision.get("working_title")
    if (
        title != "Continuity is not consistency: Cross-contract observability in LEED "
        "spot measurement chains"
    ):
        raise CrossContractClaimBoundaryError("single working title changed")

    claims = decision.get("claim_evidence_map")
    if not isinstance(claims, list) or [claim.get("claim_id") for claim in claims] != [
        "XC1",
        "XC2",
        "XC3",
        "XC4",
        "XC5",
        "XC6",
        "XC7",
    ]:
        raise CrossContractClaimBoundaryError("claim-evidence inventory changed")
    required_sections = {"abstract", "results", "conclusion"}
    observed_sections: set[str] = set()
    allowed_evidence_classes = {"direct_result", "cross_contract_inference"}
    for claim in claims:
        if not isinstance(claim, dict):
            raise CrossContractClaimBoundaryError("invalid claim-evidence record")
        sections = claim.get("sections")
        if (
            not isinstance(sections, list)
            or not sections
            or not set(sections) <= required_sections
            or claim.get("evidence_class") not in allowed_evidence_classes
        ):
            raise CrossContractClaimBoundaryError("invalid claim-evidence record")
        observed_sections.update(sections)
        for field in (
            "permitted_claim",
            "source",
            "denominator_semantics",
            "required_limitation",
        ):
            if not isinstance(claim.get(field), str) or not claim[field].strip():
                raise CrossContractClaimBoundaryError("invalid claim-evidence record")
        anchors = claim.get("frozen_anchors")
        if not isinstance(anchors, list) or not anchors:
            raise CrossContractClaimBoundaryError("invalid claim-evidence record")
    if observed_sections != required_sections:
        raise CrossContractClaimBoundaryError("claim-evidence section coverage changed")

    rules = decision.get("language_rules")
    expected_concepts = [
        "gaps",
        "recovery",
        "Cross-contract target inconsistency",
        "abstention",
        "possible false persistence",
    ]
    if not isinstance(rules, list) or [rule.get("concept") for rule in rules] != (
        expected_concepts
    ):
        raise CrossContractClaimBoundaryError("claim-language rules changed")
    for rule in rules:
        if not isinstance(rule.get("permitted"), str) or not isinstance(
            rule.get("prohibited"), str
        ):
            raise CrossContractClaimBoundaryError("invalid claim-language rule")

    contribution = decision.get("contribution_boundary")
    if (
        not isinstance(contribution, dict)
        or contribution.get("contribution")
        != "A cross-contract, denominator-aware assessment of failure semantics "
        "in LEED spot-tracking measurement chains."
        or not isinstance(contribution.get("easy_leed_role"), str)
        or not isinstance(contribution.get("viperleed_role"), str)
        or len(contribution.get("not_contributions", [])) != 5
    ):
        raise CrossContractClaimBoundaryError("contribution boundary changed")

    legacy = decision.get("legacy_claim_dispositions")
    if not isinstance(legacy, list) or [
        item.get("legacy_claim_id") for item in legacy
    ] != [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "unqualified_failure_language",
    ]:
        raise CrossContractClaimBoundaryError("legacy claim dispositions changed")
    decision_sha256 = _sha256(decision_path)
    if decision_sha256 != EXPECTED_DECISION_SHA256:
        raise CrossContractClaimBoundaryError(
            "proposed cross-contract decision record drifted"
        )
    return decision, decision_sha256


def preview_cross_contract_claim_boundary(
    workspace_root: Path,
    package_path: Path | None = None,
) -> dict[str, object]:
    """Authenticate and summarize the exact proposal awaiting author approval."""

    workspace_root = workspace_root.resolve()
    package = _resolve_package(workspace_root, package_path)
    decision, decision_sha256 = _validate_proposal(workspace_root, package)
    thesis = decision["thesis_decision"]
    claims = decision["claim_evidence_map"]
    rules = decision["language_rules"]
    return {
        "schema_version": "1.0",
        "status": "awaiting-author-checkpoint",
        "decision_record_sha256": decision_sha256,
        "working_title": decision["working_title"],
        "selected_thesis": thesis["selected"],
        "thesis_statement": thesis["statement"],
        "claim_count": len(claims),
        "language_rule_concepts": [rule["concept"] for rule in rules],
        "legacy_claim_disposition_count": len(decision["legacy_claim_dispositions"]),
        "contribution": decision["contribution_boundary"]["contribution"],
    }


def freeze_cross_contract_claim_boundary(
    workspace_root: Path,
    package_path: Path | None = None,
    *,
    confirmed_by: str,
    confirmed_date: str,
) -> dict[str, object]:
    """Record the explicit author checkpoint and freeze the decision package."""

    if not confirmed_by.strip() or not confirmed_date.strip():
        raise CrossContractClaimBoundaryError(
            "explicit author checkpoint requires confirmed-by and confirmed-date"
        )
    try:
        parsed_date = date.fromisoformat(confirmed_date)
    except ValueError as exc:
        raise CrossContractClaimBoundaryError(
            "confirmed-date must be an ISO calendar date"
        ) from exc
    if parsed_date.isoformat() != confirmed_date:
        raise CrossContractClaimBoundaryError(
            "confirmed-date must be an ISO calendar date"
        )

    workspace_root = workspace_root.resolve()
    package = _resolve_package(workspace_root, package_path)
    for filename in GENERATED_FILES:
        if (package / filename).exists():
            raise CrossContractClaimBoundaryError(
                "cross-contract claim-boundary package is already frozen"
            )
    decision, decision_sha256 = _validate_proposal(workspace_root, package)
    requirements = decision["author_checkpoint_requirements"]
    checkpoint = {
        "attestations": {
            attestation: True for attestation in requirements["required_attestations"]
        },
        "confirmation_statement": requirements["confirmation_statement"],
        "confirmed_by": confirmed_by.strip(),
        "confirmed_date": confirmed_date,
        "decision_record_sha256": decision_sha256,
        "schema_version": "1.0",
        "status": "confirmed",
    }
    report = _stored_validation_report(decision, checkpoint, decision_sha256)

    package.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{package.name}-freeze-", dir=package.parent
    ) as temporary:
        staging = Path(temporary) / package.name
        staging.mkdir()
        shutil.copyfile(package / "README.md", staging / "README.md")
        shutil.copyfile(
            package / "decision_record.json", staging / "decision_record.json"
        )
        _write_json(staging / "author_checkpoint.json", checkpoint)
        _write_json(staging / "validation_report.json", report)
        _write_json(staging / "manifest.json", _manifest(staging, decision_sha256))
        validated = _validate_frozen_package(workspace_root, staging)

        if _sha256(package / "decision_record.json") != decision_sha256:
            raise CrossContractClaimBoundaryError(
                "cross-contract decision changed during checkpoint publication"
            )
        if _sha256(package / "README.md") != _sha256(staging / "README.md"):
            raise CrossContractClaimBoundaryError(
                "cross-contract decision README changed during checkpoint publication"
            )
        for filename in GENERATED_FILES:
            os.replace(staging / filename, package / filename)

    published = validate_cross_contract_claim_boundary(workspace_root, package)
    if published != validated:
        raise CrossContractClaimBoundaryError(
            "published cross-contract claim boundary differs from validated staging"
        )
    return published


def _stored_validation_report(
    decision: dict[str, Any],
    checkpoint: dict[str, Any],
    decision_sha256: str,
) -> dict[str, object]:
    claims = decision["claim_evidence_map"]
    evidence_class_counts = {
        evidence_class: sum(
            claim["evidence_class"] == evidence_class for claim in claims
        )
        for evidence_class in ("direct_result", "cross_contract_inference")
    }
    claims_by_section = {
        section: [claim["claim_id"] for claim in claims if section in claim["sections"]]
        for section in ("abstract", "results", "conclusion")
    }
    return {
        "author_checkpoint": {
            "confirmed_by": checkpoint["confirmed_by"],
            "confirmed_date": checkpoint["confirmed_date"],
        },
        "claim_count": len(claims),
        "claim_ids": [claim["claim_id"] for claim in claims],
        "claims_by_section": claims_by_section,
        "contribution": decision["contribution_boundary"]["contribution"],
        "decision_record_sha256": decision_sha256,
        "evidence_class_counts": evidence_class_counts,
        "fail_closed": True,
        "language_rule_concepts": [
            rule["concept"] for rule in decision["language_rules"]
        ],
        "legacy_claim_dispositions": {
            item["legacy_claim_id"]: item["action"]
            for item in decision["legacy_claim_dispositions"]
        },
        "schema_version": "1.0",
        "selected_thesis": decision["thesis_decision"]["selected"],
        "status": "validated-author-approved-boundary",
        "thesis_statement": decision["thesis_decision"]["statement"],
        "working_title": decision["working_title"],
    }


def _manifest(package: Path, decision_sha256: str) -> dict[str, object]:
    roles = {
        "README.md": "readable_decision",
        "author_checkpoint.json": "author_checkpoint",
        "decision_record.json": "normative_decision",
        "validation_report.json": "validation_report",
    }
    return {
        "decision_record_sha256": decision_sha256,
        "files": {
            filename: {
                "bytes": (package / filename).stat().st_size,
                "role": role,
                "sha256": _sha256(package / filename),
            }
            for filename, role in roles.items()
        },
        "schema_version": "1.0",
        "status": "frozen-author-approved-claim-boundary",
    }


def _validate_frozen_package(
    workspace_root: Path,
    package: Path,
) -> dict[str, object]:
    decision, decision_sha256 = _validate_proposal(workspace_root, package)
    checkpoint = _load_json(package / "author_checkpoint.json", "author checkpoint")
    requirements = decision["author_checkpoint_requirements"]
    required_attestations = requirements["required_attestations"]
    if (
        checkpoint.get("schema_version") != "1.0"
        or checkpoint.get("status") != "confirmed"
        or checkpoint.get("decision_record_sha256") != decision_sha256
        or checkpoint.get("confirmation_statement")
        != requirements["confirmation_statement"]
        or not isinstance(checkpoint.get("confirmed_by"), str)
        or not checkpoint["confirmed_by"].strip()
        or not isinstance(checkpoint.get("confirmed_date"), str)
    ):
        raise CrossContractClaimBoundaryError("author checkpoint changed")
    try:
        confirmed_date = date.fromisoformat(checkpoint["confirmed_date"])
    except ValueError as exc:
        raise CrossContractClaimBoundaryError("author checkpoint date changed") from exc
    if confirmed_date.isoformat() != checkpoint["confirmed_date"]:
        raise CrossContractClaimBoundaryError("author checkpoint date changed")
    if checkpoint.get("attestations") != {
        attestation: True for attestation in required_attestations
    }:
        raise CrossContractClaimBoundaryError("author checkpoint attestations changed")

    manifest_path = package / "manifest.json"
    manifest = _load_json(manifest_path, "claim-boundary manifest")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("status") != "frozen-author-approved-claim-boundary"
        or manifest.get("decision_record_sha256") != decision_sha256
    ):
        raise CrossContractClaimBoundaryError("claim-boundary manifest changed")
    files = manifest.get("files")
    expected_files = {
        "README.md",
        "author_checkpoint.json",
        "decision_record.json",
        "validation_report.json",
    }
    if not isinstance(files, dict) or set(files) != expected_files:
        raise CrossContractClaimBoundaryError(
            "claim-boundary manifest inventory changed"
        )
    actual_files = {path.name for path in package.iterdir() if path.is_file()}
    if actual_files != expected_files | {"manifest.json"}:
        raise CrossContractClaimBoundaryError(
            "claim-boundary package file inventory changed"
        )
    for filename, entry in files.items():
        path = package / filename
        if (
            not isinstance(entry, dict)
            or entry.get("sha256") != _sha256(path)
            or entry.get("bytes") != path.stat().st_size
            or not isinstance(entry.get("role"), str)
        ):
            raise CrossContractClaimBoundaryError(
                f"manifested claim-boundary file drift: {filename}"
            )

    expected_report = _stored_validation_report(decision, checkpoint, decision_sha256)
    if (
        _load_json(
            package / "validation_report.json", "claim-boundary validation report"
        )
        != expected_report
    ):
        raise CrossContractClaimBoundaryError(
            "claim-boundary validation report drifted"
        )
    report = dict(expected_report)
    report["manifest_sha256"] = _sha256(manifest_path)
    return report


def validate_cross_contract_claim_boundary(
    workspace_root: Path,
    package_path: Path | None = None,
) -> dict[str, object]:
    """Validate the frozen, author-approved cross-contract decision package."""

    workspace_root = workspace_root.resolve()
    package = _resolve_package(workspace_root, package_path)
    if not (package / "decision_record.json").is_file():
        raise CrossContractClaimBoundaryError(
            "cross-contract claim-boundary decision is unavailable"
        )
    report = _validate_frozen_package(workspace_root, package)
    if (
        EXPECTED_MANIFEST_SHA256 is not None
        and package == workspace_root / DEFAULT_PACKAGE_PATH
        and report["manifest_sha256"] != EXPECTED_MANIFEST_SHA256
    ):
        raise CrossContractClaimBoundaryError(
            "frozen claim-boundary manifest authentication failed"
        )
    return report
