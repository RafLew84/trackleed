"""Non-mutating integration gate before scientific-core rebaseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from manuscript.manuscript_contract import validate_manuscript_package
from manuscript.publication_package.build_publication_package import (
    verify_manifest as verify_document_manifest,
)
from trackleed.publication.scientific_core import (
    ScientificCoreMismatchError,
    validate_fingerprint,
)
from trackleed.publication.scientific_reconstruction import (
    validate_reader_facing_artifacts,
)


class PreRebaselineValidationError(ValueError):
    """Raised when the approved pre-rebaseline state is not reproduced."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreRebaselineValidationError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PreRebaselineValidationError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_diff(
    workspace_root: Path,
    fingerprint: dict[str, Any],
) -> list[dict[str, Any]]:
    components = fingerprint.get("components")
    if not isinstance(components, list):
        raise PreRebaselineValidationError("Frozen fingerprint components changed")
    report: list[dict[str, Any]] = []
    for component in components:
        if not isinstance(component, dict):
            raise PreRebaselineValidationError("Frozen fingerprint component changed")
        component_id = component.get("id")
        relative_path = component.get("path")
        roles = component.get("roles")
        frozen_sha256 = component.get("sha256")
        if (
            not isinstance(component_id, str)
            or not isinstance(relative_path, str)
            or not isinstance(roles, list)
            or not isinstance(frozen_sha256, str)
        ):
            raise PreRebaselineValidationError(
                "Frozen fingerprint component identity changed"
            )
        path = (workspace_root / relative_path).resolve()
        if not path.is_relative_to(workspace_root) or not path.is_file():
            raise PreRebaselineValidationError(
                f"Fingerprint component is unavailable: {relative_path}"
            )
        current_sha256 = _sha256(path)
        report.append(
            {
                "current_sha256": current_sha256,
                "frozen_sha256": frozen_sha256,
                "id": component_id,
                "path": relative_path,
                "roles": roles,
                "status": (
                    "unchanged" if current_sha256 == frozen_sha256 else "changed"
                ),
            }
        )
    return report


def validate_predecessor_integrity(
    fingerprint: dict[str, Any],
    contract_path: Path,
) -> None:
    body_keys = (
        "schema_version",
        "status",
        "contract_sha256",
        "required_roles",
        "components",
    )
    if any(key not in fingerprint for key in body_keys):
        raise PreRebaselineValidationError("Predecessor fingerprint integrity changed")
    body = {key: fingerprint[key] for key in body_keys}
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    recorded = fingerprint.get("fingerprint_sha256")
    if (
        hashlib.sha256(canonical).hexdigest() != recorded
        or fingerprint["contract_sha256"] != _sha256(contract_path)
    ):
        raise PreRebaselineValidationError("Predecessor fingerprint integrity changed")


def validate_pre_rebaseline(
    workspace_root: Path,
    *,
    expectation_path: Path | None = None,
    fingerprint_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the reviewed revision while requiring predecessor rejection."""

    workspace_root = workspace_root.resolve()
    package = workspace_root / "manuscript" / "scientific_core"
    contract_path = package / "fingerprint_contract.json"
    fingerprint_path = (
        fingerprint_path.resolve()
        if fingerprint_path is not None
        else package / "fingerprint.json"
    )
    expectation_path = (
        expectation_path.resolve()
        if expectation_path is not None
        else package / "pre_rebaseline_contract.json"
    )
    expectation = _load_json(expectation_path, "pre-rebaseline contract")
    fingerprint = _load_json(fingerprint_path, "scientific-core fingerprint")
    validate_predecessor_integrity(fingerprint, contract_path)

    if (
        expectation.get("schema_version") != "1.0"
        or expectation.get("status") != "approved-pre-rebaseline"
    ):
        raise PreRebaselineValidationError("Pre-rebaseline contract changed")
    expected_predecessor = expectation.get(
        "expected_predecessor_fingerprint_sha256"
    )
    if fingerprint.get("fingerprint_sha256") != expected_predecessor:
        raise PreRebaselineValidationError("Predecessor fingerprint changed")

    components = _component_diff(workspace_root, fingerprint)
    expected_count = expectation.get("expected_component_count")
    if len(components) != expected_count:
        raise PreRebaselineValidationError(
            f"Expected {expected_count} fingerprint components, found {len(components)}"
        )
    expected_changed = expectation.get("expected_changed_component_ids")
    if (
        not isinstance(expected_changed, list)
        or len(expected_changed) != len(set(expected_changed))
        or any(not isinstance(item, str) for item in expected_changed)
    ):
        raise PreRebaselineValidationError("Expected component diff changed")
    actual_changed = {
        component["id"]
        for component in components
        if component["status"] == "changed"
    }
    if actual_changed != set(expected_changed):
        missing = sorted(set(expected_changed) - actual_changed)
        unexpected = sorted(actual_changed - set(expected_changed))
        raise PreRebaselineValidationError(
            "Scientific-core component diff changed: "
            f"missing={missing}, unexpected={unexpected}"
        )

    required_review_roles = expectation.get("required_review_roles")
    covered_roles = {
        role for component in components for role in component["roles"]
    }
    if (
        not isinstance(required_review_roles, list)
        or not set(required_review_roles) <= covered_roles
    ):
        raise PreRebaselineValidationError(
            "Required prose/display/claim/limitation/denominator review coverage changed"
        )
    review_scope = {
        role: [
            component["id"]
            for component in components
            if role in component["roles"]
        ]
        for role in required_review_roles
    }

    manuscript_report = validate_manuscript_package(workspace_root / "manuscript")
    reader_report = validate_reader_facing_artifacts(workspace_root)
    document_manifest = verify_document_manifest()
    validations = {
        "canonical_document_manifest": {
            "artifact_count": sum(
                2 for _document in document_manifest["artifacts"].values()
            ),
            "status": "passed",
        },
        "manuscript_contract": {
            "claim_count": len(manuscript_report["claim_ids"]),
            "status": "passed",
        },
        "reader_facing_inventory": {
            "bundle_count": reader_report.bundle_count,
            "output_file_count": reader_report.output_file_count,
            "source_file_count": reader_report.source_file_count,
            "status": "passed",
        },
    }

    fingerprint_before = fingerprint_path.read_bytes()
    try:
        validate_fingerprint(workspace_root, contract_path, fingerprint_path)
    except ScientificCoreMismatchError:
        pass
    else:
        raise PreRebaselineValidationError(
            "Old fingerprint unexpectedly passed; component coverage is incomplete"
        )
    fingerprint_after = fingerprint_path.read_bytes()
    if fingerprint_after != fingerprint_before:
        raise PreRebaselineValidationError(
            "Pre-rebaseline validation modified the predecessor fingerprint"
        )

    changed_count = len(actual_changed)
    return {
        "component_summary": {
            "changed": changed_count,
            "total": len(components),
            "unchanged": len(components) - changed_count,
        },
        "components": components,
        "mutation_policy": {
            "fingerprint_replaced": False,
            "manual_fingerprint_edit_allowed": False,
            "validation_only": True,
        },
        "predecessor": {
            "bytes_preserved": True,
            "file_sha256": _sha256(fingerprint_path),
            "fingerprint_sha256": expected_predecessor,
            "path": (
                fingerprint_path.relative_to(workspace_root).as_posix()
                if fingerprint_path.is_relative_to(workspace_root)
                else str(fingerprint_path)
            ),
            "validation_result": "expected-rejection",
        },
        "pre_rebaseline_contract": {
            "path": (
                expectation_path.relative_to(workspace_root).as_posix()
                if expectation_path.is_relative_to(workspace_root)
                else str(expectation_path)
            ),
            "predecessor_component_source_commit": expectation.get(
                "predecessor_component_source_commit"
            ),
            "sha256": _sha256(expectation_path),
        },
        "required_review_roles": required_review_roles,
        "review_scope": review_scope,
        "schema_version": "1.0",
        "status": "expected-predecessor-rejection",
        "validations": validations,
    }


__all__ = [
    "PreRebaselineValidationError",
    "validate_predecessor_integrity",
    "validate_pre_rebaseline",
]
