"""Final claim-boundary adjudication from metadata and sensitivity evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from trackleed.analysis.experimental_metadata import validate_metadata_package
from trackleed.analysis.scientific_sensitivity_records import SensitivityResultRecord
from trackleed.analysis.scientific_tracer_records import read_csv
from trackleed.publication.scientific_sensitivity_sources import (
    read_json,
    sha256_file,
    validate_sensitivity_source_data,
)


CLAIM_IDS = tuple(f"C{index}" for index in range(1, 10))
DECISIONS = {"preserved", "narrowed", "removed"}
IMPLEMENTATION_PATHS = (
    Path("trackleed/publication/scientific_claim_adjudication.py"),
    Path("trackleed/publication/scientific_sensitivity_sources.py"),
)
SENSITIVITY_EVIDENCE_ARTIFACT_PATH = (
    "manuscript/claim_adjudication/output/sensitivity_claim_evidence.json"
)


@dataclass(frozen=True)
class ClaimAdjudicationReport:
    claim_count: int
    preserved_count: int
    narrowed_count: int
    removed_count: int
    author_checkpoint_status: str


def _claim_ids_from_map(claim_map_path: Path) -> set[str]:
    text = claim_map_path.read_text(encoding="utf-8")
    return set(re.findall(r"^\| (C\d+) \|", text, flags=re.MULTILINE))


def _claim_inventory_sha256(claim_map_path: Path) -> str:
    claim_rows = [
        line.rstrip()
        for line in claim_map_path.read_text(encoding="utf-8").splitlines()
        if re.match(r"^\| C\d+ \|", line)
    ]
    return hashlib.sha256(("\n".join(claim_rows) + "\n").encode()).hexdigest()


def _sensitivity_claim_evidence(source_directory: Path) -> dict[str, Any]:
    _, results = read_csv(
        source_directory / "sensitivity_results.csv",
        SensitivityResultRecord,
    )
    method_rows = [row for row in results if row["scope"] == "method"]
    variants = sorted({row["variant_id"] for row in method_rows})
    if len(method_rows) != 75 or len(variants) != 15:
        raise ValueError("Claim adjudication sensitivity matrix changed")
    coverage = [row["branch_consistent_measurements"] for row in method_rows]
    recovery = [row["branch_consistent_recovered_gap_count"] for row in method_rows]
    inconsistency = [row["geometry_inconsistent_fraction"] for row in method_rows]
    residual = [
        row["median_absolute_pair_distance_residual_px"] for row in method_rows
    ]
    central = [
        row for row in method_rows if row["variant_id"] == "roi_48_proposal_15"
    ]
    by_variant = {
        variant: [row for row in method_rows if row["variant_id"] == variant]
        for variant in variants
    }
    lk_lowest = all(
        min(rows, key=lambda row: row["branch_consistent_measurements"])["method"]
        == "lk"
        for rows in by_variant.values()
    )
    summary = read_json(source_directory / "sensitivity_summary.json")
    width = summary.get("conditional_width_sensitivity")
    if not isinstance(width, dict):
        raise ValueError("Claim adjudication width decision changed")
    evidence = {
        "variant_count": len(variants),
        "method_count": 5,
        "fixed_coverage_denominator": 221,
        "fixed_recovery_denominator": 24,
        "coverage_numerator_envelope": [min(coverage), max(coverage)],
        "recovery_numerator_envelope": [min(recovery), max(recovery)],
        "geometry_inconsistency_fraction_envelope": [
            min(inconsistency),
            max(inconsistency),
        ],
        "conditional_residual_median_px_envelope": [min(residual), max(residual)],
        "central_conditional_residual_median_px_range": [
            min(row["median_absolute_pair_distance_residual_px"] for row in central),
            max(row["median_absolute_pair_distance_residual_px"] for row in central),
        ],
        "lucas_kanade_lowest_coverage_in_every_variant": lk_lowest,
        "width_sensitivity_decision": width.get("decision"),
        "width_triggered_methods": width.get("triggered_methods"),
        "central_contract_changed": width.get("central_contract_changed"),
        "automatic_configuration_selection": width.get(
            "automatic_configuration_selection"
        ),
    }
    expected = {
        "coverage_numerator_envelope": [170, 221],
        "recovery_numerator_envelope": [17, 24],
        "central_conditional_residual_median_px_range": [
            1.7367102643215162,
            1.8366454823238882,
        ],
        "conditional_residual_median_px_envelope": [
            1.4988461207997261,
            1.9386438331731597,
        ],
    }
    if (
        any(evidence[key] != value for key, value in expected.items())
        or evidence["variant_count"] != 15
        or evidence["lucas_kanade_lowest_coverage_in_every_variant"] is not True
        or evidence["width_sensitivity_decision"] != "not_run_no_trigger"
        or evidence["width_triggered_methods"] != []
        or evidence["central_contract_changed"] is not False
        or evidence["automatic_configuration_selection"] is not False
    ):
        raise ValueError("Sensitivity-derived claim anchors changed")
    return evidence


def _validate_contract(
    contract: dict[str, Any],
    claim_ids: set[str],
    metadata_constraint_ids: set[str],
) -> list[dict[str, Any]]:
    claims = contract.get("claims")
    if (
        contract.get("schema_version") != 1
        or contract.get("status") != "proposed_for_author_checkpoint"
        or not isinstance(contract.get("global_boundary"), str)
        or not isinstance(claims, list)
        or len(claims) != len(CLAIM_IDS)
        or claim_ids != set(CLAIM_IDS)
    ):
        raise ValueError("Claim-adjudication contract changed")
    by_id: dict[str, dict[str, Any]] = {}
    required_keys = {
        "claim_id",
        "decision",
        "final_claim",
        "metadata_constraint_ids",
        "rationale",
        "sensitivity_basis",
    }
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != required_keys:
            raise ValueError("Claim-adjudication record changed")
        claim_id = str(claim["claim_id"])
        if claim_id in by_id or claim_id not in CLAIM_IDS:
            raise ValueError("Claim-adjudication identifier changed")
        if claim["decision"] not in DECISIONS:
            raise ValueError(f"Claim decision changed: {claim_id}")
        constraints = claim["metadata_constraint_ids"]
        basis = claim["sensitivity_basis"]
        if (
            not isinstance(claim["final_claim"], str)
            or not claim["final_claim"].strip()
            or not isinstance(claim["rationale"], str)
            or not claim["rationale"].strip()
            or not isinstance(constraints, list)
            or not constraints
            or not set(constraints).issubset(metadata_constraint_ids)
            or not isinstance(basis, list)
            or not basis
        ):
            raise ValueError(f"Claim boundary is incomplete: {claim_id}")
        by_id[claim_id] = claim
    if set(by_id) != set(CLAIM_IDS):
        raise ValueError("Claim-adjudication inventory changed")
    return [by_id[claim_id] for claim_id in CLAIM_IDS]


def _pending_checkpoint(boundary_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "pending",
        "boundary_sha256": boundary_sha256,
        "confirmed_by": None,
        "confirmed_date": None,
        "confirmation_statement": (
            "Author confirmation is required for the proposed final scientific "
            "claim boundary."
        ),
        "attestations": {
            "all_claim_decisions_reviewed": False,
            "metadata_constraints_accepted": False,
            "sensitivity_constraints_accepted": False,
        },
    }


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    boundary_sha256: str,
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "boundary_sha256",
        "confirmed_by",
        "confirmed_date",
        "confirmation_statement",
        "attestations",
    }
    attestations = checkpoint.get("attestations")
    expected_attestations = {
        "all_claim_decisions_reviewed",
        "metadata_constraints_accepted",
        "sensitivity_constraints_accepted",
    }
    if (
        set(checkpoint) != expected_keys
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("boundary_sha256") != boundary_sha256
        or checkpoint.get("status") not in {"pending", "confirmed"}
        or not isinstance(checkpoint.get("confirmation_statement"), str)
        or not checkpoint["confirmation_statement"].strip()
        or not isinstance(attestations, dict)
        or set(attestations) != expected_attestations
    ):
        raise ValueError("Author checkpoint does not bind the current boundary")
    if checkpoint["status"] == "pending":
        if (
            checkpoint["confirmed_by"] is not None
            or checkpoint["confirmed_date"] is not None
            or any(attestations.values())
        ):
            raise ValueError("Author checkpoint has an invalid pending state")
        return
    if (
        not isinstance(checkpoint["confirmed_by"], str)
        or not checkpoint["confirmed_by"].strip()
        or not isinstance(checkpoint["confirmed_date"], str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", checkpoint["confirmed_date"])
        is None
        or any(value is not True for value in attestations.values())
    ):
        raise ValueError("Author checkpoint confirmation is incomplete")


def validate_author_checkpoint(
    checkpoint: dict[str, Any],
    boundary_sha256: str,
) -> None:
    """Validate that a human checkpoint is complete and boundary-bound."""

    _validate_checkpoint(checkpoint, boundary_sha256)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_path(path: Path, workspace_root: Path) -> str:
    try:
        return path.relative_to(workspace_root).as_posix()
    except ValueError:
        return path.as_posix()


def build_claim_adjudication(
    *,
    workspace_root: Path,
    contract_path: Path,
    sensitivity_source_directory: Path,
    metadata_directory: Path,
    claim_map_path: Path,
    output_directory: Path,
) -> ClaimAdjudicationReport:
    """Build and validate the proposed final scientific claim boundary."""

    workspace_root = workspace_root.resolve()
    contract_path = contract_path.resolve()
    sensitivity_source_directory = sensitivity_source_directory.resolve()
    metadata_directory = metadata_directory.resolve()
    claim_map_path = claim_map_path.resolve()
    output_directory = output_directory.resolve()
    validate_sensitivity_source_data(sensitivity_source_directory)
    validate_metadata_package(
        workspace_root,
        metadata_directory,
        require_author_confirmation=True,
    )
    metadata_constraints = read_json(
        metadata_directory / "metadata_claim_constraints.json"
    )
    metadata_constraint_ids = {
        str(record["constraint_id"])
        for record in metadata_constraints["constraints"]
    }
    contract = read_json(contract_path)
    claims = _validate_contract(
        contract,
        _claim_ids_from_map(claim_map_path),
        metadata_constraint_ids,
    )
    sensitivity_evidence = _sensitivity_claim_evidence(
        sensitivity_source_directory
    )
    decisions = Counter(str(claim["decision"]) for claim in claims)
    output_directory.mkdir(parents=True, exist_ok=True)
    evidence_path = output_directory / "sensitivity_claim_evidence.json"
    boundary_path = output_directory / "final_claim_boundary.json"
    checkpoint_path = output_directory / "author_checkpoint.json"
    _write_json(evidence_path, sensitivity_evidence)
    boundary = {
        "schema_version": 1,
        "scope": "final_scientific_claim_boundary",
        "global_boundary": contract["global_boundary"],
        "claims": claims,
        "decision_counts": {
            decision: decisions[decision] for decision in sorted(DECISIONS)
        },
        "central_contract_changed": False,
        "capture_basin_triggered": False,
        "automatic_configuration_selection": False,
        "source_links": {
            "claim_evidence_map": {
                "path": _source_path(claim_map_path, workspace_root),
                "claim_inventory_sha256": _claim_inventory_sha256(
                    claim_map_path
                ),
            },
            "metadata_claim_constraints": {
                "path": _source_path(
                    metadata_directory / "metadata_claim_constraints.json",
                    workspace_root,
                ),
                "sha256": sha256_file(
                    metadata_directory / "metadata_claim_constraints.json"
                ),
            },
            "sensitivity_claim_evidence": {
                "path": SENSITIVITY_EVIDENCE_ARTIFACT_PATH,
                "sha256": sha256_file(evidence_path),
            },
        },
    }
    _write_json(boundary_path, boundary)
    boundary_sha256 = sha256_file(boundary_path)
    if checkpoint_path.is_file():
        checkpoint = read_json(checkpoint_path)
        recorded_sha256 = str(checkpoint.get("boundary_sha256"))
        _validate_checkpoint(checkpoint, recorded_sha256)
        if (
            checkpoint["status"] == "pending"
            and recorded_sha256 != boundary_sha256
        ):
            checkpoint = _pending_checkpoint(boundary_sha256)
            _write_json(checkpoint_path, checkpoint)
        else:
            _validate_checkpoint(checkpoint, boundary_sha256)
    else:
        checkpoint = _pending_checkpoint(boundary_sha256)
        _validate_checkpoint(checkpoint, boundary_sha256)
        _write_json(checkpoint_path, checkpoint)

    manifest = {
        "schema_version": 1,
        "status": "pending_author_checkpoint"
        if checkpoint["status"] == "pending"
        else "completed",
        "contract_sha256": sha256_file(contract_path),
        "implementation_sha256": {
            path.as_posix(): sha256_file(workspace_root / path)
            for path in IMPLEMENTATION_PATHS
        },
        "outputs_sha256": {
            path.name: sha256_file(path)
            for path in (evidence_path, boundary_path, checkpoint_path)
        },
        "claim_count": len(claims),
        "decision_counts": boundary["decision_counts"],
        "author_checkpoint_status": checkpoint["status"],
    }
    _write_json(output_directory / "manifest.json", manifest)
    return ClaimAdjudicationReport(
        claim_count=len(claims),
        preserved_count=decisions["preserved"],
        narrowed_count=decisions["narrowed"],
        removed_count=decisions["removed"],
        author_checkpoint_status=str(checkpoint["status"]),
    )


def confirm_claim_adjudication(
    output_directory: Path,
    *,
    confirmed_by: str,
    confirmed_date: str,
) -> None:
    """Record an explicit human approval for the generated claim boundary."""

    output_directory = output_directory.resolve()
    boundary_path = output_directory / "final_claim_boundary.json"
    checkpoint_path = output_directory / "author_checkpoint.json"
    manifest_path = output_directory / "manifest.json"
    boundary_sha256 = sha256_file(boundary_path)
    checkpoint = read_json(checkpoint_path)
    _validate_checkpoint(checkpoint, boundary_sha256)
    if checkpoint["status"] != "pending":
        raise ValueError("Author checkpoint is not pending")
    confirmed = {
        **checkpoint,
        "status": "confirmed",
        "confirmed_by": confirmed_by,
        "confirmed_date": confirmed_date,
        "confirmation_statement": (
            "The publication author approves all C1-C9 decisions, the metadata "
            "constraints, the sensitivity constraints, and the final scientific "
            "claim boundary bound by this SHA-256."
        ),
        "attestations": {
            "all_claim_decisions_reviewed": True,
            "metadata_constraints_accepted": True,
            "sensitivity_constraints_accepted": True,
        },
    }
    _validate_checkpoint(confirmed, boundary_sha256)
    _write_json(checkpoint_path, confirmed)
    manifest = read_json(manifest_path)
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, dict) or "author_checkpoint.json" not in outputs:
        raise ValueError("Claim-adjudication manifest changed")
    manifest["status"] = "completed"
    manifest["author_checkpoint_status"] = "confirmed"
    outputs["author_checkpoint.json"] = sha256_file(checkpoint_path)
    _write_json(manifest_path, manifest)


__all__ = [
    "ClaimAdjudicationReport",
    "build_claim_adjudication",
    "confirm_claim_adjudication",
    "validate_author_checkpoint",
]
