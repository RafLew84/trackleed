"""Fail-closed assertion of the publication content-revision-complete status."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from manuscript.manuscript_contract import validate_manuscript_package
from manuscript.publication_package.audit_page_renders import (
    AuditError,
    verify_audit_manifest,
)
from trackleed.publication.scientific_core import (
    ScientificCoreValidationError,
    validate_fingerprint,
)


ASSERTED_STATUS = "content-revision-complete"
CONTRACT_STATUS = "approved-content-revision-assertion"
EXPECTED_GATE_IDS = (
    "1-approved-scope-implemented",
    "2-targeted-source-and-display-validation",
    "3-portability-closed-before-rebaseline",
    "4-scientific-acceptance-checkpoint",
    "5-controlled-scientific-core-rebaseline",
    "6-post-rebaseline-scientific-reconstruction",
    "7-canonical-document-build-and-page-level-render-acceptance",
)
AUTHORIZED_PLACEHOLDERS = (
    "AUTHOR",
    "EXPERIMENT",
    "RELEASE-AUTHORIZATION",
    "SUBMISSION",
)
PLACEHOLDER_PATTERN = re.compile(r"TODO\[([^\]\r\n]+)\]")


class ContentRevisionValidationError(ValueError):
    """Raised when the final content-revision assertion is not supported."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContentRevisionValidationError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ContentRevisionValidationError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(workspace_root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContentRevisionValidationError(f"{label} path changed")
    path = (workspace_root / value).resolve()
    if not path.is_relative_to(workspace_root) or not path.is_file():
        raise ContentRevisionValidationError(f"{label} is unavailable: {value}")
    return path


def _relative(workspace_root: Path, path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(workspace_root):
        raise ContentRevisionValidationError(f"Path escaped workspace: {path}")
    return resolved.relative_to(workspace_root).as_posix()


def _validate_hash_record(
    workspace_root: Path,
    record: object,
    label: str,
) -> dict[str, str]:
    if not isinstance(record, dict):
        raise ContentRevisionValidationError(f"{label} record changed")
    path = _resolve(workspace_root, record.get("path"), label)
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ContentRevisionValidationError(f"{label} hash changed")
    current = _sha256(path)
    if current != expected:
        raise ContentRevisionValidationError(
            f"Evidence hash drift: {_relative(workspace_root, path)}"
        )
    result = {
        "path": _relative(workspace_root, path),
        "sha256": current,
    }
    role = record.get("role")
    if role is not None:
        if not isinstance(role, str) or not role.strip():
            raise ContentRevisionValidationError(f"{label} role changed")
        result["role"] = role
    return result


def inventory_placeholders(
    workspace_root: Path,
    source_paths: list[str],
    allowed_categories: list[str],
) -> list[dict[str, object]]:
    """List every placeholder occurrence and reject malformed or unknown tokens."""

    workspace_root = workspace_root.resolve()
    if (
        not allowed_categories
        or len(allowed_categories) != len(set(allowed_categories))
        or any(not isinstance(category, str) for category in allowed_categories)
    ):
        raise ContentRevisionValidationError(
            "Authorized placeholder category set changed"
        )
    allowed = set(allowed_categories)
    records: list[dict[str, object]] = []
    for relative_path in source_paths:
        path = _resolve(workspace_root, relative_path, "placeholder source")
        text = path.read_text(encoding="utf-8")
        matches = list(PLACEHOLDER_PATTERN.finditer(text))
        if text.count("TODO[") != len(matches):
            raise ContentRevisionValidationError(
                f"Malformed placeholder: {_relative(workspace_root, path)}"
            )
        for match in matches:
            category = match.group(1)
            if category not in allowed:
                raise ContentRevisionValidationError(
                    "Uncategorized placeholder: "
                    f"TODO[{category}] in {_relative(workspace_root, path)}"
                )
            line = text.count("\n", 0, match.start()) + 1
            line_start = text.rfind("\n", 0, match.start()) + 1
            records.append(
                {
                    "category": category,
                    "column": match.start() - line_start + 1,
                    "line": line,
                    "path": _relative(workspace_root, path),
                    "token": match.group(0),
                }
            )
    return records


def validate_page_level_acceptance(
    workspace_root: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Verify the accepted per-page manifest and return its stable summary."""

    workspace_root = workspace_root.resolve()
    try:
        manifest = verify_audit_manifest(workspace_root, manifest_path.resolve())
    except (AuditError, OSError, json.JSONDecodeError) as exc:
        raise ContentRevisionValidationError(
            f"Page-level render acceptance failed: {exc}"
        ) from exc
    renditions = manifest["renditions"]
    return {
        "final_decision": manifest["final_decision"],
        "manifest_path": _relative(workspace_root, manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "rendition_count": manifest["rendition_count"],
        "renditions": {
            name: {
                "document_path": rendition["document_path"],
                "document_sha256": rendition["document_sha256"],
                "page_count": rendition["page_count"],
            }
            for name, rendition in sorted(renditions.items())
        },
        "reviewed_page_count": manifest["reviewed_page_count"],
        "reviewer_identity": manifest["reviewer_identity"],
        "status": manifest["status"],
    }


def _validate_contract_shape(contract: dict[str, Any]) -> None:
    if (
        contract.get("schema_version") != "1.0"
        or contract.get("status") != CONTRACT_STATUS
        or contract.get("asserted_status") != ASSERTED_STATUS
    ):
        raise ContentRevisionValidationError("Content-revision contract changed")
    if contract.get("submission_ready") is not False:
        raise ContentRevisionValidationError(
            "content-revision-complete must not assert submission-ready"
        )
    assertion_date = contract.get("assertion_date")
    if not isinstance(assertion_date, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", assertion_date
    ) is None:
        raise ContentRevisionValidationError("Assertion date changed")
    gates = contract.get("gates")
    if not isinstance(gates, list) or tuple(
        gate.get("id") if isinstance(gate, dict) else None for gate in gates
    ) != EXPECTED_GATE_IDS:
        raise ContentRevisionValidationError("Normative gate set changed")
    if tuple(contract.get("authorized_placeholder_categories", ())) != (
        AUTHORIZED_PLACEHOLDERS
    ):
        raise ContentRevisionValidationError(
            "Authorized placeholder category set changed"
        )


def _validate_resolved_tickets(
    workspace_root: Path,
    ticket_paths: object,
) -> list[str]:
    if not isinstance(ticket_paths, list) or len(ticket_paths) != 13:
        raise ContentRevisionValidationError("Required predecessor ticket set changed")
    resolved: list[str] = []
    for value in ticket_paths:
        path = _resolve(workspace_root, value, "predecessor ticket")
        if "Status: `resolved`" not in path.read_text(encoding="utf-8"):
            raise ContentRevisionValidationError(
                f"Predecessor ticket is not resolved: {_relative(workspace_root, path)}"
            )
        resolved.append(_relative(workspace_root, path))
    return resolved


def validate_content_revision(
    workspace_root: Path,
    *,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Validate all seven gates and return a deterministic final status report."""

    workspace_root = workspace_root.resolve()
    contract_path = (
        contract_path.resolve()
        if contract_path is not None
        else workspace_root / "manuscript/content_revision/contract.json"
    )
    contract = _load_json(contract_path, "content-revision contract")
    _validate_contract_shape(contract)

    status_contract = _validate_hash_record(
        workspace_root,
        contract.get("status_contract"),
        "publication status contract",
    )
    gate_reports: list[dict[str, object]] = []
    evidence_by_path: dict[str, dict[str, str]] = {}
    for gate in contract["gates"]:
        evidence = gate.get("evidence")
        if (
            not isinstance(gate.get("title"), str)
            or not isinstance(evidence, list)
            or not evidence
        ):
            raise ContentRevisionValidationError(
                f"Normative gate evidence changed: {gate.get('id')}"
            )
        verified = [
            _validate_hash_record(
                workspace_root,
                item,
                f"{gate['id']} evidence",
            )
            for item in evidence
        ]
        for item in verified:
            evidence_by_path[item["path"]] = item
        gate_reports.append(
            {
                "evidence": verified,
                "id": gate["id"],
                "status": "passed",
                "title": gate["title"],
            }
        )

    implementation_records = contract.get("assertion_implementation")
    if not isinstance(implementation_records, list) or len(
        implementation_records
    ) != 3:
        raise ContentRevisionValidationError("Assertion implementation set changed")
    assertion_implementation = [
        _validate_hash_record(
            workspace_root,
            item,
            "assertion implementation",
        )
        for item in implementation_records
    ]
    for item in assertion_implementation:
        evidence_by_path[item["path"]] = item

    resolved_tickets = _validate_resolved_tickets(
        workspace_root,
        contract.get("required_resolved_tickets"),
    )

    placeholder_sources = contract.get("placeholder_source_paths")
    if not isinstance(placeholder_sources, list) or any(
        not isinstance(value, str) for value in placeholder_sources
    ):
        raise ContentRevisionValidationError("Placeholder source set changed")
    placeholders = inventory_placeholders(
        workspace_root,
        placeholder_sources,
        contract["authorized_placeholder_categories"],
    )
    placeholder_counts = Counter(item["category"] for item in placeholders)
    expected_counts = contract.get("expected_placeholder_counts")
    if expected_counts != dict(placeholder_counts):
        raise ContentRevisionValidationError(
            "Authorized placeholder inventory changed: "
            f"expected={expected_counts}, actual={dict(placeholder_counts)}"
        )

    try:
        manuscript_report = validate_manuscript_package(
            workspace_root / "manuscript"
        )
    except (OSError, ValueError) as exc:
        raise ContentRevisionValidationError(
            f"Missing or invalid scientific content: {exc}"
        ) from exc

    successor = contract.get("successor_fingerprint")
    if not isinstance(successor, dict):
        raise ContentRevisionValidationError("Successor fingerprint record changed")
    fingerprint_path = _resolve(
        workspace_root,
        successor.get("path"),
        "successor fingerprint",
    )
    fingerprint_contract_path = _resolve(
        workspace_root,
        successor.get("contract_path"),
        "scientific-core fingerprint contract",
    )
    try:
        fingerprint_report = validate_fingerprint(
            workspace_root,
            fingerprint_contract_path,
            fingerprint_path,
        )
    except (OSError, ScientificCoreValidationError, ValueError) as exc:
        raise ContentRevisionValidationError(
            f"Successor fingerprint validation failed: {exc}"
        ) from exc
    if (
        fingerprint_report.fingerprint_sha256
        != successor.get("fingerprint_sha256")
        or _sha256(fingerprint_path) != successor.get("file_sha256")
    ):
        raise ContentRevisionValidationError("Successor fingerprint changed")
    replacement_path = _resolve(
        workspace_root,
        successor.get("replacement_record_path"),
        "replacement record",
    )
    replacement_relative = _relative(workspace_root, replacement_path)
    if replacement_relative not in evidence_by_path:
        raise ContentRevisionValidationError("Replacement record evidence changed")
    replacement = _load_json(replacement_path, "replacement record")
    predecessor_fingerprint = successor.get("predecessor_fingerprint_sha256")
    if (
        replacement.get("status") != "controlled-scientific-core-rebaseline"
        or replacement.get("successor_fingerprint_sha256")
        != fingerprint_report.fingerprint_sha256
        or replacement.get("predecessor_fingerprint_sha256")
        != predecessor_fingerprint
        or replacement.get("component_summary", {}).get("total")
        != fingerprint_report.component_count
    ):
        raise ContentRevisionValidationError(
            "Predecessor-to-successor replacement link changed"
        )

    page_audit_record = contract.get("page_level_audit")
    verified_page_audit = _validate_hash_record(
        workspace_root,
        page_audit_record,
        "page-level audit manifest",
    )
    page_audit = validate_page_level_acceptance(
        workspace_root,
        workspace_root / verified_page_audit["path"],
    )

    validations = contract.get("documented_validations")
    if not isinstance(validations, list) or not validations:
        raise ContentRevisionValidationError("Documented validation set changed")
    documented_validations: list[dict[str, str]] = []
    for validation in validations:
        if not isinstance(validation, dict):
            raise ContentRevisionValidationError("Documented validation changed")
        evidence_path = validation.get("evidence_path")
        if (
            not isinstance(evidence_path, str)
            or evidence_path not in evidence_by_path
            or not isinstance(validation.get("command"), str)
            or not isinstance(validation.get("scope"), str)
        ):
            raise ContentRevisionValidationError("Documented validation changed")
        documented_validations.append(
            {
                "command": validation["command"],
                "evidence_path": evidence_path,
                "evidence_sha256": evidence_by_path[evidence_path]["sha256"],
                "scope": validation["scope"],
            }
        )

    reconstruction = contract.get("scientific_reconstruction")
    if not isinstance(reconstruction, dict):
        raise ContentRevisionValidationError("Scientific reconstruction record changed")
    reconstruction_evidence = reconstruction.get("evidence_path")
    if reconstruction_evidence not in evidence_by_path:
        raise ContentRevisionValidationError("Scientific reconstruction evidence changed")
    if (
        reconstruction.get("clean_windows_run_count") != 2
        or not isinstance(reconstruction.get("report_sha256"), str)
        or len(reconstruction["report_sha256"]) != 64
    ):
        raise ContentRevisionValidationError("Scientific reconstruction record changed")
    reconstruction_path = workspace_root / reconstruction_evidence
    reconstruction_text = reconstruction_path.read_text(encoding="utf-8")
    if (
        reconstruction["report_sha256"] not in reconstruction_text
        or reconstruction.get("command") not in reconstruction_text
    ):
        raise ContentRevisionValidationError(
            "Scientific reconstruction evidence does not bind the command and report"
        )

    remaining_work = contract.get("submission_ready_remaining_work")
    if not isinstance(remaining_work, list) or not remaining_work or any(
        not isinstance(item, str) or not item.strip() for item in remaining_work
    ):
        raise ContentRevisionValidationError("Submission-ready boundary changed")

    placeholder_summary = {
        category: placeholder_counts[category]
        for category in AUTHORIZED_PLACEHOLDERS
    }
    placeholder_summary["total"] = len(placeholders)
    return {
        "assertion_contract": {
            "path": _relative(workspace_root, contract_path),
            "sha256": _sha256(contract_path),
        },
        "assertion_date": contract.get("assertion_date"),
        "assertion_implementation": assertion_implementation,
        "documented_validations": documented_validations,
        "gates": gate_reports,
        "page_level_audit": page_audit,
        "placeholder_inventory": placeholders,
        "placeholder_inventory_scope": {
            "authoritative_sources": placeholder_sources,
            "derived_reader_documents": sorted(
                rendition["document_path"]
                for rendition in page_audit["renditions"].values()
            ),
            "derived_render_coverage": "page-level-render-accepted",
        },
        "placeholder_summary": placeholder_summary,
        "resolved_predecessor_tickets": resolved_tickets,
        "schema_version": "1.0",
        "scientific_content_validation": {
            "claim_count": len(manuscript_report["claim_ids"]),
            "status": "passed",
        },
        "scientific_reconstruction": {
            **reconstruction,
            "evidence_sha256": evidence_by_path[reconstruction_evidence]["sha256"],
            "status": "passed",
        },
        "status": ASSERTED_STATUS,
        "status_boundary": {
            "canonical_contract": status_contract,
            "later_status": "submission-ready",
        },
        "submission_ready": False,
        "submission_ready_remaining_work": remaining_work,
        "successor_fingerprint": {
            "component_count": fingerprint_report.component_count,
            "file_sha256": _sha256(fingerprint_path),
            "fingerprint_sha256": fingerprint_report.fingerprint_sha256,
            "path": _relative(workspace_root, fingerprint_path),
            "predecessor_fingerprint_sha256": predecessor_fingerprint,
            "replacement_record_path": replacement_relative,
            "replacement_record_sha256": _sha256(replacement_path),
            "status": fingerprint_report.status,
        },
    }


__all__ = [
    "ContentRevisionValidationError",
    "inventory_placeholders",
    "validate_content_revision",
    "validate_page_level_acceptance",
]
