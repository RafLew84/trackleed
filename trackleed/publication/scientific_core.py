"""Fail-closed scientific-core fingerprint contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
FROZEN_CONTRACT_STATUS = "frozen"
COMPLETE_STATUS = "scientific-baseline-complete"


class ScientificCoreValidationError(ValueError):
    """Raised when the frozen scientific core no longer validates."""


class ScientificCoreMismatchError(ScientificCoreValidationError):
    """Raised when current scientific-core content differs from its baseline."""


@dataclass(frozen=True)
class ScientificCoreReport:
    status: str
    fingerprint_sha256: str
    component_count: int
    covered_roles: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ScientificCoreValidationError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ScientificCoreValidationError(f"{label} must be a JSON object")
    return value


def _resolve_component(workspace_root: Path, relative_path: str) -> Path:
    path = (workspace_root / relative_path).resolve()
    if path == workspace_root or not path.is_relative_to(workspace_root):
        raise ScientificCoreValidationError(
            f"Scientific-core component escapes the workspace: {relative_path}"
        )
    if not path.is_file():
        raise ScientificCoreValidationError(
            f"Scientific-core component is unavailable: {relative_path}"
        )
    return path


def _contract_components(
    workspace_root: Path,
    contract: dict[str, Any],
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    required_roles = contract.get("required_roles")
    components = contract.get("components")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("status") != FROZEN_CONTRACT_STATUS
        or not isinstance(required_roles, list)
        or not required_roles
        or any(not isinstance(role, str) or not role for role in required_roles)
        or len(set(required_roles)) != len(required_roles)
        or not isinstance(components, list)
        or not components
    ):
        raise ScientificCoreValidationError(
            "Scientific-core fingerprint contract changed"
        )

    allowed_roles = set(required_roles)
    component_ids: set[str] = set()
    component_paths: set[str] = set()
    covered_roles: set[str] = set()
    resolved: list[dict[str, object]] = []
    for component in components:
        if not isinstance(component, dict) or set(component) != {
            "id",
            "path",
            "roles",
        }:
            raise ScientificCoreValidationError(
                "Scientific-core component contract changed"
            )
        component_id = component["id"]
        relative_path = component["path"]
        roles = component["roles"]
        if (
            not isinstance(component_id, str)
            or not component_id
            or component_id in component_ids
            or not isinstance(relative_path, str)
            or not relative_path
            or relative_path in component_paths
            or not isinstance(roles, list)
            or not roles
            or any(not isinstance(role, str) for role in roles)
            or not set(roles) <= allowed_roles
        ):
            raise ScientificCoreValidationError(
                "Scientific-core component identity or roles changed"
            )
        path = _resolve_component(workspace_root, relative_path)
        component_ids.add(component_id)
        component_paths.add(relative_path)
        covered_roles.update(roles)
        resolved.append(
            {
                "id": component_id,
                "path": relative_path,
                "roles": roles,
                "sha256": _sha256(path),
            }
        )
    if covered_roles != allowed_roles:
        missing = sorted(allowed_roles - covered_roles)
        raise ScientificCoreValidationError(
            f"Scientific-core roles are not fully covered: {missing}"
        )
    return tuple(required_roles), resolved


def _fingerprint_value(
    workspace_root: Path,
    contract_path: Path,
) -> dict[str, object]:
    contract = _load_json(contract_path, "fingerprint contract")
    required_roles, components = _contract_components(workspace_root, contract)
    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": COMPLETE_STATUS,
        "contract_sha256": _sha256(contract_path),
        "required_roles": list(required_roles),
        "components": components,
    }
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        **body,
        "fingerprint_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _report(fingerprint: dict[str, object]) -> ScientificCoreReport:
    components = fingerprint["components"]
    roles = fingerprint["required_roles"]
    if not isinstance(components, list) or not isinstance(roles, list):
        raise ScientificCoreValidationError("Scientific-core fingerprint changed")
    return ScientificCoreReport(
        status=str(fingerprint["status"]),
        fingerprint_sha256=str(fingerprint["fingerprint_sha256"]),
        component_count=len(components),
        covered_roles=tuple(str(role) for role in roles),
    )


def freeze_fingerprint(
    workspace_root: Path,
    contract_path: Path,
    fingerprint_path: Path,
) -> ScientificCoreReport:
    """Freeze a new fingerprint, refusing to replace an existing baseline."""

    workspace_root = workspace_root.resolve()
    contract_path = contract_path.resolve()
    fingerprint_path = fingerprint_path.resolve()
    if fingerprint_path.exists():
        raise FileExistsError(
            f"Refusing to replace scientific-core fingerprint: {fingerprint_path}"
        )
    fingerprint = _fingerprint_value(workspace_root, contract_path)
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_path.write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return _report(fingerprint)


def validate_fingerprint(
    workspace_root: Path,
    contract_path: Path,
    fingerprint_path: Path,
) -> ScientificCoreReport:
    """Validate the frozen fingerprint without modifying any baseline file."""

    workspace_root = workspace_root.resolve()
    contract_path = contract_path.resolve()
    fingerprint_path = fingerprint_path.resolve()
    frozen = _load_json(fingerprint_path, "scientific-core fingerprint")
    expected = _fingerprint_value(workspace_root, contract_path)
    if frozen != expected:
        raise ScientificCoreMismatchError(
            "Scientific-core fingerprint differs from the frozen baseline"
        )
    return _report(frozen)


__all__ = [
    "ScientificCoreMismatchError",
    "ScientificCoreReport",
    "ScientificCoreValidationError",
    "freeze_fingerprint",
    "validate_fingerprint",
]
