"""Authorized predecessor-to-successor scientific-core replacement."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid
from typing import Any

from trackleed.publication.pre_rebaseline import (
    PreRebaselineValidationError,
    validate_predecessor_integrity,
)
from trackleed.publication.scientific_core import freeze_fingerprint


EXPECTED_COMPONENT_COUNT = 23
SCHEMA_VERSION = "1.0"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class ScientificCoreReplacementError(ValueError):
    """Raised when controlled fingerprint replacement is not authorized."""


@dataclass(frozen=True)
class ScientificCoreReplacementReport:
    status: str
    predecessor_fingerprint_sha256: str
    successor_fingerprint_sha256: str
    component_count: int
    history_path: str
    replacement_record_path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScientificCoreReplacementError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ScientificCoreReplacementError(f"{label} must be a JSON object")
    return value


def _resolve_inside(workspace_root: Path, value: Path | str, label: str) -> Path:
    candidate = Path(value)
    path = (
        candidate.resolve()
        if candidate.is_absolute()
        else (workspace_root / candidate).resolve()
    )
    if path == workspace_root or not path.is_relative_to(workspace_root):
        raise ScientificCoreReplacementError(f"{label} escapes the workspace")
    return path


def _relative(workspace_root: Path, path: Path) -> str:
    return path.relative_to(workspace_root).as_posix()


def _git(
    workspace_root: Path,
    *arguments: str,
    accepted_codes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ScientificCoreReplacementError("Git is unavailable") from exc
    if completed.returncode not in accepted_codes:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ScientificCoreReplacementError(
            f"Git validation failed: {detail or arguments[0]}"
        )
    return completed


def _require_tracked(workspace_root: Path, paths: list[Path]) -> None:
    for path in paths:
        _git(
            workspace_root,
            "ls-files",
            "--error-unmatch",
            "--",
            _relative(workspace_root, path),
        )


def _require_commit(workspace_root: Path, implementation_commit: str) -> None:
    if _GIT_COMMIT.fullmatch(implementation_commit) is None:
        raise ScientificCoreReplacementError(
            "Implementation commit must be an exact 40-character SHA-1"
        )
    _git(workspace_root, "cat-file", "-e", f"{implementation_commit}^{{commit}}")
    completed = _git(
        workspace_root,
        "merge-base",
        "--is-ancestor",
        implementation_commit,
        "HEAD",
        accepted_codes=(0, 1),
    )
    if completed.returncode != 0:
        raise ScientificCoreReplacementError(
            "Implementation commit is not an ancestor of HEAD"
        )


def _require_clean_scientific_inputs(
    workspace_root: Path,
    component_paths: list[Path],
    required_paths: list[Path],
) -> None:
    pathspecs = {
        _relative(workspace_root, path) for path in component_paths + required_paths
    }
    pathspecs.add("manuscript")
    pathspecs.add("trackleed/io")
    completed = _git(
        workspace_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *sorted(pathspecs),
    )
    if completed.stdout:
        dirty = sorted(
            path
            for entry in completed.stdout.split("\0")
            if entry
            for path in (entry[3:].strip().replace("\\", "/"),)
            if not path.startswith("manuscript/review/")
            and not path.startswith("manuscript/scientific_core/history/")
        )
        if not dirty:
            return
        raise ScientificCoreReplacementError(
            f"Scientific inputs are dirty or untracked: {dirty}"
        )


def _validate_authorization(
    workspace_root: Path,
    authorization_path: Path,
    decision_path: Path,
    expected_predecessor: str,
    implementation_commit: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    authorization = _load_json(authorization_path, "replacement authorization")
    decision = authorization.get("decision_record")
    reviewed = authorization.get("reviewed_component_diff")
    expected_changed = authorization.get("expected_changed_component_ids")
    if (
        authorization.get("schema_version") != SCHEMA_VERSION
        or authorization.get("status") != "authorized"
        or authorization.get("expected_component_count") != EXPECTED_COMPONENT_COUNT
        or authorization.get("expected_predecessor_fingerprint_sha256")
        != expected_predecessor
        or authorization.get("implementation_commit") != implementation_commit
        or not isinstance(authorization.get("reason"), str)
        or not authorization["reason"].strip()
        or not isinstance(decision, dict)
        or not isinstance(reviewed, dict)
        or not isinstance(expected_changed, list)
        or len(expected_changed) != len(set(expected_changed))
        or any(not isinstance(item, str) for item in expected_changed)
    ):
        raise ScientificCoreReplacementError("Replacement authorization changed")
    declared_decision = _resolve_inside(
        workspace_root,
        str(decision.get("path", "")),
        "Decision record",
    )
    if declared_decision != decision_path or decision.get("sha256") != _sha256(
        decision_path
    ):
        raise ScientificCoreReplacementError("Decision record is not authorized")
    report_path = _resolve_inside(
        workspace_root,
        str(reviewed.get("path", "")),
        "Reviewed component diff",
    )
    if reviewed.get("sha256") != _sha256(report_path):
        raise ScientificCoreReplacementError("Reviewed component diff changed")
    decision_text = decision_path.read_text(encoding="utf-8")
    required_decision_values = (
        "Status: `accepted`",
        "Decision: `accepted`",
        implementation_commit,
        expected_predecessor,
        str(reviewed["sha256"]),
    )
    if any(value not in decision_text for value in required_decision_values):
        raise ScientificCoreReplacementError("Decision record is incomplete")
    return authorization, report_path, _load_json(report_path, "reviewed component diff")


def _validate_component_diff(
    workspace_root: Path,
    contract: dict[str, Any],
    predecessor: dict[str, Any],
    reviewed: dict[str, Any],
    authorization: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[Path]]:
    contract_components = contract.get("components")
    predecessor_components = predecessor.get("components")
    reviewed_components = reviewed.get("components")
    if not all(
        isinstance(items, list) and len(items) == EXPECTED_COMPONENT_COUNT
        for items in (
            contract_components,
            predecessor_components,
            reviewed_components,
        )
    ):
        raise ScientificCoreReplacementError(
            "Replacement requires complete coverage of 23 components"
        )
    contract_by_id = {
        item.get("id"): item for item in contract_components if isinstance(item, dict)
    }
    predecessor_by_id = {
        item.get("id"): item
        for item in predecessor_components
        if isinstance(item, dict)
    }
    reviewed_by_id = {
        item.get("id"): item for item in reviewed_components if isinstance(item, dict)
    }
    if not (
        len(contract_by_id)
        == len(predecessor_by_id)
        == len(reviewed_by_id)
        == EXPECTED_COMPONENT_COUNT
        and set(contract_by_id) == set(predecessor_by_id) == set(reviewed_by_id)
    ):
        raise ScientificCoreReplacementError(
            "Replacement component identities are incomplete"
        )

    components: list[dict[str, Any]] = []
    component_paths: list[Path] = []
    for contract_component in contract_components:
        component_id = contract_component["id"]
        predecessor_component = predecessor_by_id[component_id]
        reviewed_component = reviewed_by_id[component_id]
        relative_path = contract_component.get("path")
        roles = contract_component.get("roles")
        if (
            not isinstance(relative_path, str)
            or not isinstance(roles, list)
            or predecessor_component.get("path") != relative_path
            or predecessor_component.get("roles") != roles
            or reviewed_component.get("path") != relative_path
            or reviewed_component.get("roles") != roles
            or reviewed_component.get("frozen_sha256")
            != predecessor_component.get("sha256")
        ):
            raise ScientificCoreReplacementError(
                f"Component contract changed: {component_id}"
            )
        path = _resolve_inside(workspace_root, relative_path, "Component")
        current_sha256 = _sha256(path)
        if reviewed_component.get("current_sha256") != current_sha256:
            raise ScientificCoreReplacementError(
                f"Unapproved scientific component: {component_id}"
            )
        status = (
            "unchanged"
            if current_sha256 == predecessor_component["sha256"]
            else "changed"
        )
        if reviewed_component.get("status") != status:
            raise ScientificCoreReplacementError(
                f"Reviewed component status changed: {component_id}"
            )
        components.append(
            {
                "id": component_id,
                "path": relative_path,
                "roles": roles,
                "predecessor_sha256": predecessor_component["sha256"],
                "successor_sha256": current_sha256,
                "status": status,
            }
        )
        component_paths.append(path)
    changed_ids = {
        component["id"] for component in components if component["status"] == "changed"
    }
    if changed_ids != set(authorization["expected_changed_component_ids"]):
        raise ScientificCoreReplacementError("Authorized component diff changed")
    summary = reviewed.get("component_summary")
    if summary != {
        "changed": len(changed_ids),
        "total": EXPECTED_COMPONENT_COUNT,
        "unchanged": EXPECTED_COMPONENT_COUNT - len(changed_ids),
    }:
        raise ScientificCoreReplacementError("Reviewed component summary changed")
    if reviewed.get("status") != "expected-predecessor-rejection":
        raise ScientificCoreReplacementError("Reviewed predecessor result changed")
    return components, component_paths


def _write_temp(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _atomic_create_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ScientificCoreReplacementError(
                f"Existing scientific-core history differs: {path}"
            )
        return
    temporary = _write_temp(path, content)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ScientificCoreReplacementError(
                f"Existing scientific-core history differs: {path}"
            )
    finally:
        temporary.unlink(missing_ok=True)


def replace_fingerprint(
    workspace_root: Path,
    *,
    expected_predecessor: str,
    decision_record: Path,
    implementation_commit: str,
    authorization: Path,
    check_only: bool = False,
) -> ScientificCoreReplacementReport:
    """Perform one authorized, atomic scientific-core fingerprint replacement."""

    workspace_root = workspace_root.resolve()
    if _SHA256.fullmatch(expected_predecessor) is None:
        raise ScientificCoreReplacementError(
            "Expected predecessor must be an exact lowercase SHA-256"
        )
    decision_path = _resolve_inside(workspace_root, decision_record, "Decision record")
    authorization_path = _resolve_inside(
        workspace_root, authorization, "Replacement authorization"
    )
    package = workspace_root / "manuscript/scientific_core"
    if authorization_path != package / "replacement_authorization.json":
        raise ScientificCoreReplacementError(
            "Command requires the canonical replacement authorization"
        )
    contract_path = package / "fingerprint_contract.json"
    fingerprint_path = package / "fingerprint.json"
    predecessor_bytes = fingerprint_path.read_bytes()
    predecessor = _load_json(fingerprint_path, "predecessor fingerprint")
    contract = _load_json(contract_path, "fingerprint contract")
    try:
        validate_predecessor_integrity(predecessor, contract_path)
    except PreRebaselineValidationError as exc:
        raise ScientificCoreReplacementError(
            "Expected predecessor integrity changed"
        ) from exc
    if predecessor.get("fingerprint_sha256") != expected_predecessor:
        raise ScientificCoreReplacementError("Expected predecessor does not match")

    authorization_value, report_path, reviewed = _validate_authorization(
        workspace_root,
        authorization_path,
        decision_path,
        expected_predecessor,
        implementation_commit,
    )
    components, component_paths = _validate_component_diff(
        workspace_root,
        contract,
        predecessor,
        reviewed,
        authorization_value,
    )
    _require_commit(workspace_root, implementation_commit)
    required_paths = [
        contract_path,
        fingerprint_path,
        report_path,
        decision_path,
        authorization_path,
    ]
    _require_tracked(workspace_root, component_paths + required_paths)
    _require_clean_scientific_inputs(
        workspace_root,
        component_paths,
        required_paths,
    )

    successor_temporary = package / f".fingerprint.{uuid.uuid4().hex}.tmp"
    try:
        successor_report = freeze_fingerprint(
            workspace_root,
            contract_path,
            successor_temporary,
        )
        successor_bytes = successor_temporary.read_bytes()
        successor = json.loads(successor_bytes)
        successor_by_id = {
            item["id"]: item for item in successor.get("components", [])
        }
        if (
            successor_report.component_count != EXPECTED_COMPONENT_COUNT
            or any(
                successor_by_id.get(item["id"], {}).get("sha256")
                != item["successor_sha256"]
                for item in components
            )
        ):
            raise ScientificCoreReplacementError(
                "Generated successor does not match reviewed component diff"
            )

        history_root = package / "history"
        predecessor_path = (
            history_root / "fingerprints" / f"{expected_predecessor}.json"
        )
        replacement_path = (
            history_root
            / "replacements"
            / f"{successor_report.fingerprint_sha256}.json"
        )
        replacement_value = {
            "authorization": {
                "path": _relative(workspace_root, authorization_path),
                "sha256": _sha256(authorization_path),
            },
            "component_summary": reviewed["component_summary"],
            "components": components,
            "decision_record": {
                "path": _relative(workspace_root, decision_path),
                "sha256": _sha256(decision_path),
            },
            "implementation_commit": implementation_commit,
            "predecessor_fingerprint_sha256": expected_predecessor,
            "reason": authorization_value["reason"],
            "reviewed_component_diff": {
                "path": _relative(workspace_root, report_path),
                "sha256": _sha256(report_path),
            },
            "runtime": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
            },
            "schema_version": SCHEMA_VERSION,
            "status": "controlled-scientific-core-rebaseline",
            "successor_fingerprint_sha256": successor_report.fingerprint_sha256,
        }
        replacement_bytes = (
            json.dumps(replacement_value, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

        if fingerprint_path.read_bytes() != predecessor_bytes:
            raise ScientificCoreReplacementError(
                "Predecessor changed during replacement validation"
            )
        if not check_only:
            _atomic_create_or_verify(predecessor_path, predecessor_bytes)
            _atomic_create_or_verify(replacement_path, replacement_bytes)
            os.replace(successor_temporary, fingerprint_path)
    finally:
        successor_temporary.unlink(missing_ok=True)

    return ScientificCoreReplacementReport(
        status="authorized" if check_only else "replaced",
        predecessor_fingerprint_sha256=expected_predecessor,
        successor_fingerprint_sha256=successor_report.fingerprint_sha256,
        component_count=successor_report.component_count,
        history_path=_relative(workspace_root, predecessor_path),
        replacement_record_path=_relative(workspace_root, replacement_path),
    )


__all__ = [
    "ScientificCoreReplacementError",
    "ScientificCoreReplacementReport",
    "replace_fingerprint",
]
