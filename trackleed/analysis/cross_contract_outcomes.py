"""Build and validate the principal-scan cross-contract outcome analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from trackleed.analysis.viperleed_principal import (
    ViperleedPrincipalValidationError,
    validate_principal_viperleed_scans,
)

DEFAULT_PACKAGE_PATH = Path("manuscript/cross_contract/outcomes")
EXPECTED_CONTRACT_SHA256 = (
    "6e87d92f48554412f90ae8bd90ec3a81a5116a9ca872f5d1bdb33129498d0c8b"
)
EXPECTED_MANIFEST_SHA256 = (
    "c48f462f8d5c98b638a5acacc6a003bfc2bbc59fe8061051fba44545ff329456"
)
METHODS = ("easyleed", "locotrack", "ostrack", "dam4sam", "lk")
BRANCH_MAPPING = {
    ("pair_1", "B1"): "P1_B1",
    ("pair_1", "B2"): "P1_B2",
    ("pair_2", "B1"): "P2_B1",
    ("pair_2", "B2"): "P2_B2",
}
OUTCOME_LABELS = (
    "both_contracts_observe",
    "gaussian_only",
    "viperleed_only",
    "neither_contract_observes",
)
MATRIX_FIELDS = (
    "record_key",
    "opportunity_key",
    "method",
    "case_id",
    "scan_id",
    "pair_id",
    "gaussian_branch",
    "branch_id",
    "source_frame",
    "energy_ev",
    "physics_baseline_status",
    "accepted_fit",
    "gaussian_contract_measurable",
    "measurement_status",
    "gaussian_x",
    "gaussian_y",
    "native_significance",
    "native_dx_raw",
    "native_dy_raw",
    "native_raw_x",
    "native_raw_y",
    "viper_native_direct_detectability",
    "measurability_outcome",
    "accepted_out_of_branch",
    "nominal_cross_contract_distance_px",
    "nominal_out_of_branch_classification",
)
GAP_FIELDS = (
    "gap_id",
    "contract",
    "method",
    "scan_id",
    "branch_id",
    "start_frame",
    "end_frame",
    "start_energy_ev",
    "end_energy_ev",
    "gap_frame_count",
    "recovery_frame",
    "recovery_energy_ev",
)
GAP_COMPARISON_FIELDS = (
    "comparison_id",
    "method",
    "scan_id",
    "branch_id",
    "component_start_frame",
    "component_end_frame",
    "gaussian_gap_ids",
    "viperleed_gap_ids",
    "comparison_category",
    "exact_boundary_agreement",
    "component_exit_frame",
    "component_exit_energy_ev",
    "gaussian_observes_at_component_exit",
    "viperleed_observes_at_component_exit",
    "both_contracts_recover_at_component_exit",
)
METHOD_SUMMARY_FIELDS = (
    "method",
    "opportunity_count",
    "both_contracts_observe",
    "gaussian_only",
    "viperleed_only",
    "neither_contract_observes",
    "reference_visible_opportunities",
    "branch_consistent_on_reference_visible",
    "branch_consistent_measurement_coverage",
    "accepted_gaussian_all_frames",
    "out_of_branch_all_frames",
    "all_frame_conditional_out_of_branch_fraction",
    "out_of_branch_branch_model_disagreement",
    "out_of_branch_cross_contract_target_inconsistency",
    "out_of_branch_unresolved",
)
OUT_OF_BRANCH_FIELDS = (
    "event_id",
    "record_key",
    "opportunity_key",
    "method",
    "scan_id",
    "branch_id",
    "source_frame",
    "energy_ev",
    "gaussian_x",
    "gaussian_y",
    "viper_native_direct_detectability",
    "native_raw_x",
    "native_raw_y",
    "cross_contract_distance_px",
    "nominal_positional_tolerance_px",
    "classification",
)
SENSITIVITY_FIELDS = (
    "variant_id",
    "native_significance_threshold",
    "cross_contract_positional_tolerance_px",
    "method",
    "opportunity_count",
    "both_contracts_observe",
    "gaussian_only",
    "viperleed_only",
    "neither_contract_observes",
    "gaussian_gap_event_count",
    "viperleed_gap_event_count",
    "gap_comparison_component_count",
    "gap_comparison_shared",
    "gap_comparison_gaussian_only",
    "gap_comparison_viperleed_only",
    "gap_exact_boundary_agreement",
    "both_contracts_recover_at_component_exit",
    "reference_visible_opportunities",
    "branch_consistent_on_reference_visible",
    "accepted_gaussian_all_frames",
    "out_of_branch_all_frames",
    "out_of_branch_branch_model_disagreement",
    "out_of_branch_cross_contract_target_inconsistency",
    "out_of_branch_unresolved",
)
OUTPUT_ROLES = {
    "README.md": "analysis scope, denominator semantics and validation guide",
    "analysis_contract.json": "predeclared cross-contract analysis contract",
    "branch_frame_matrix.csv": ("complete method-by-opportunity cross-contract matrix"),
    "gap_comparisons.csv": ("contract-specific gap overlap and recovery comparisons"),
    "gaussian_gap_events.csv": "bounded Gaussian-contract gap events",
    "headline_summary.json": "headline counts and explicit denominator summary",
    "method_summary.csv": ("method outcomes with fixed and conditional denominators"),
    "out_of_branch_events.csv": (
        "accepted out-of-branch cross-contract classifications"
    ),
    "sensitivity_results.csv": "complete predeclared sensitivity grid",
    "viperleed_gap_events.csv": "bounded ViPEr-native visibility-gap events",
}
GENERATED_PATHS = (
    "branch_frame_matrix.csv",
    "gap_comparisons.csv",
    "gaussian_gap_events.csv",
    "headline_summary.json",
    "method_summary.csv",
    "out_of_branch_events.csv",
    "sensitivity_results.csv",
    "viperleed_gap_events.csv",
    "manifest.json",
    "validation_report.json",
)


class CrossContractValidationError(ValueError):
    """Raised when the frozen cross-contract analysis fails closed."""


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
        raise CrossContractValidationError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CrossContractValidationError(f"{label} must be a JSON object")
    return value


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise CrossContractValidationError(f"Cannot read {label}: {path}") from exc


def _strict_bool(value: object, label: str) -> bool:
    text = str(value).strip().lower()
    if text not in {"true", "false"}:
        raise CrossContractValidationError(f"Invalid boolean for {label}")
    return text == "true"


def _optional_float(value: object, label: str) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        number = float(str(value))
    except ValueError as exc:
        raise CrossContractValidationError(
            f"Invalid numeric value for {label}"
        ) from exc
    return number if math.isfinite(number) else None


def _required_float(value: object, label: str) -> float:
    number = _optional_float(value, label)
    if number is None:
        raise CrossContractValidationError(f"Missing finite value for {label}")
    return number


def _required_int(value: object, label: str) -> int:
    number = _required_float(value, label)
    integer = int(number)
    if float(integer) != number:
        raise CrossContractValidationError(f"Non-integral value for {label}")
    return integer


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def _csv_bytes(
    rows: list[dict[str, object]],
    fields: tuple[str, ...],
) -> bytes:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _format_value(row[field]) for field in fields})
    return output.getvalue().encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _workspace_file(
    workspace_root: Path,
    record: object,
    label: str,
) -> Path:
    if not isinstance(record, dict):
        raise CrossContractValidationError(f"{label} binding changed")
    relative = record.get("path")
    if not isinstance(relative, str) or not relative:
        raise CrossContractValidationError(f"{label} path changed")
    path = (workspace_root / relative).resolve()
    if not path.is_relative_to(workspace_root) or not path.is_file():
        raise CrossContractValidationError(f"{label} is unavailable: {relative}")
    if _sha256(path) != record.get("sha256"):
        raise CrossContractValidationError(f"{label} hash drift")
    return path


def _load_contract(
    workspace_root: Path,
    package_root: Path,
) -> dict[str, Any]:
    path = package_root / "analysis_contract.json"
    if not path.is_file():
        raise CrossContractValidationError(
            f"cross-contract analysis contract is unavailable: {package_root}"
        )
    if _sha256(path) != EXPECTED_CONTRACT_SHA256:
        raise CrossContractValidationError("cross-contract analysis contract changed")
    contract = _load_json(path, "cross-contract analysis contract")
    if (
        contract.get("schema_version") != "1.0"
        or contract.get("analysis_id") != "principal-cross-contract-outcomes"
        or contract.get("status")
        != "predeclared-before-cross-contract-outcome-computation"
    ):
        raise CrossContractValidationError("cross-contract analysis identity changed")
    bindings = contract.get("input_bindings")
    if not isinstance(bindings, dict):
        raise CrossContractValidationError("cross-contract input bindings changed")
    paths = {
        name: _workspace_file(workspace_root, record, name.replace("_", " "))
        for name, record in bindings.items()
    }
    panel_manifest = _load_json(paths["gaussian_panel_manifest"], "panel manifest")
    panel_hashes = panel_manifest.get("files_sha256")
    if (
        panel_manifest.get("immutable") is not True
        or not isinstance(panel_hashes, dict)
        or panel_hashes.get("frozen/panel_contract.json")
        != bindings["gaussian_panel_contract"]["sha256"]
        or panel_hashes.get("frozen/expected_candidate_records.csv")
        != bindings["gaussian_candidate_records"]["sha256"]
        or panel_hashes.get("frozen/expected_evaluation_records.csv")
        != bindings["gaussian_evaluation_records"]["sha256"]
    ):
        raise CrossContractValidationError("Gaussian panel authentication changed")
    try:
        validate_principal_viperleed_scans(workspace_root)
    except ViperleedPrincipalValidationError as exc:
        raise CrossContractValidationError(
            "principal native package validation failed"
        ) from exc
    return contract


def _indexed_rows(
    rows: list[dict[str, str]],
    key_field: str,
    label: str,
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get(key_field, "")
        if not key or key in indexed:
            raise CrossContractValidationError(f"duplicate or missing {label} key")
        indexed[key] = row
    return indexed


def _measurability_outcome(gaussian: bool, native: bool) -> str:
    if gaussian and native:
        return "both_contracts_observe"
    if gaussian:
        return "gaussian_only"
    if native:
        return "viperleed_only"
    return "neither_contract_observes"


def _native_direct_detectability(
    significance: float | None,
    dx_raw: float | None,
    dy_raw: float | None,
    threshold: float,
) -> bool:
    return bool(
        significance is not None
        and significance >= threshold
        and dx_raw is not None
        and dy_raw is not None
    )


def _cross_contract_distance(
    gaussian_measurable: bool,
    native_direct: bool,
    gaussian_x: float | None,
    gaussian_y: float | None,
    native_x: float | None,
    native_y: float | None,
) -> float | None:
    if gaussian_measurable and (gaussian_x is None or gaussian_y is None):
        raise CrossContractValidationError(
            "measurable Gaussian record lacks a finite position"
        )
    if native_direct and (native_x is None or native_y is None):
        raise CrossContractValidationError(
            "direct native record lacks a finite raw position"
        )
    if not gaussian_measurable or not native_direct:
        return None
    assert gaussian_x is not None and gaussian_y is not None
    assert native_x is not None and native_y is not None
    return math.hypot(gaussian_x - native_x, gaussian_y - native_y)


def _out_of_branch_classification(
    accepted_out_of_branch: bool,
    native_direct: bool,
    distance: float | None,
    positional_tolerance: float,
) -> str | None:
    if not accepted_out_of_branch:
        return None
    if not native_direct:
        return "unresolved"
    if distance is None:
        raise CrossContractValidationError(
            "direct out-of-branch event lacks cross-contract distance"
        )
    if distance <= positional_tolerance:
        return "branch_model_disagreement"
    return "cross_contract_target_inconsistency"


def _build_matrix(
    workspace_root: Path,
    contract: Mapping[str, Any],
) -> list[dict[str, object]]:
    bindings = contract["input_bindings"]
    candidate_path = workspace_root / bindings["gaussian_candidate_records"]["path"]
    evaluation_path = workspace_root / bindings["gaussian_evaluation_records"]["path"]
    native_path = workspace_root / bindings["principal_native_records"]["path"]
    candidates = _read_csv(candidate_path, "Gaussian candidate records")
    evaluations = _read_csv(evaluation_path, "Gaussian evaluation records")
    native_rows = _read_csv(native_path, "principal native records")
    if len(candidates) != 1620 or len(evaluations) != 1620 or len(native_rows) != 324:
        raise CrossContractValidationError("cross-contract input row count changed")
    candidates_by_key = _indexed_rows(candidates, "record_key", "candidate")
    evaluations_by_key = _indexed_rows(evaluations, "record_key", "evaluation")
    if set(candidates_by_key) != set(evaluations_by_key):
        raise CrossContractValidationError("Gaussian candidate/evaluation keys changed")
    native_by_key: dict[tuple[str, int, str], dict[str, str]] = {}
    for native_row in native_rows:
        key = (
            native_row.get("scan_id", ""),
            _required_int(native_row.get("source_frame"), "native source frame"),
            native_row.get("branch_id", ""),
        )
        if not key[0] or not key[2] or key in native_by_key:
            raise CrossContractValidationError(
                "principal native opportunity keys changed"
            )
        native_by_key[key] = native_row

    matrix: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, str]] = set()
    for record_key in sorted(evaluations_by_key):
        evaluation = evaluations_by_key[record_key]
        candidate = candidates_by_key[record_key]
        method = evaluation.get("method", "")
        scan_id = candidate.get("series_id", "")
        pair_id = candidate.get("pair_id", "")
        gaussian_branch = evaluation.get("branch", "")
        frame = _required_int(evaluation.get("frame"), "Gaussian frame")
        branch_id = BRANCH_MAPPING.get((pair_id, gaussian_branch))
        if method not in METHODS or not scan_id or branch_id is None:
            raise CrossContractValidationError("Gaussian opportunity identity changed")
        matrix_key = (method, scan_id, frame, branch_id)
        if matrix_key in seen:
            raise CrossContractValidationError("duplicate matrix opportunity")
        seen.add(matrix_key)
        matched_native = native_by_key.get((scan_id, frame, branch_id))
        if matched_native is None:
            raise CrossContractValidationError("native matrix opportunity is missing")
        gaussian_energy = _required_float(
            evaluation.get("energy_ev"), "Gaussian energy"
        )
        native_energy = _required_float(
            matched_native.get("energy_ev"), "native energy"
        )
        if abs(gaussian_energy - native_energy) > 1e-6:
            raise CrossContractValidationError("cross-contract energy mapping changed")
        accepted = _strict_bool(evaluation.get("accepted_fit"), "accepted_fit")
        gaussian = _strict_bool(
            evaluation.get("measurement_valid"), "measurement_valid"
        )
        if accepted != gaussian:
            raise CrossContractValidationError(
                "Gaussian measurability/accepted-fit semantics changed"
            )
        significance = _optional_float(
            matched_native.get("significance"), "significance"
        )
        native_dx = _optional_float(matched_native.get("dx_raw"), "native dx_raw")
        native_dy = _optional_float(matched_native.get("dy_raw"), "native dy_raw")
        native_direct = _native_direct_detectability(
            significance,
            native_dx,
            native_dy,
            2.5,
        )
        if native_direct != _strict_bool(
            matched_native.get("viper_native_direct_detectability"),
            "stored native detectability",
        ):
            raise CrossContractValidationError(
                "native direct-detectability semantics changed"
            )
        gaussian_x = _optional_float(candidate.get("center_x"), "Gaussian x")
        gaussian_y = _optional_float(candidate.get("center_y"), "Gaussian y")
        native_x = _optional_float(matched_native.get("raw_x"), "native raw x")
        native_y = _optional_float(matched_native.get("raw_y"), "native raw y")
        measurement_status = evaluation.get("measurement_status", "")
        out_of_branch = accepted and measurement_status == "geometry_inconsistent"
        distance = _cross_contract_distance(
            gaussian,
            native_direct,
            gaussian_x,
            gaussian_y,
            native_x,
            native_y,
        )
        classification = _out_of_branch_classification(
            out_of_branch,
            native_direct,
            distance,
            1.0,
        )
        matrix.append(
            {
                "record_key": record_key,
                "opportunity_key": f"{scan_id}|{frame}|{native_energy:.10g}|{branch_id}",
                "method": method,
                "case_id": evaluation.get("case_id", ""),
                "scan_id": scan_id,
                "pair_id": pair_id,
                "gaussian_branch": gaussian_branch,
                "branch_id": branch_id,
                "source_frame": frame,
                "energy_ev": native_energy,
                "physics_baseline_status": evaluation.get(
                    "physics_baseline_status", ""
                ),
                "accepted_fit": accepted,
                "gaussian_contract_measurable": gaussian,
                "measurement_status": measurement_status,
                "gaussian_x": gaussian_x,
                "gaussian_y": gaussian_y,
                "native_significance": significance,
                "native_dx_raw": native_dx,
                "native_dy_raw": native_dy,
                "native_raw_x": native_x,
                "native_raw_y": native_y,
                "viper_native_direct_detectability": native_direct,
                "measurability_outcome": _measurability_outcome(
                    gaussian, native_direct
                ),
                "accepted_out_of_branch": out_of_branch,
                "nominal_cross_contract_distance_px": distance,
                "nominal_out_of_branch_classification": classification,
            }
        )
    if len(matrix) != len(METHODS) * 324:
        raise CrossContractValidationError("branch-frame matrix denominator changed")
    return matrix


def _bounded_gap_events(
    rows: list[dict[str, object]],
    *,
    contract_id: str,
    method: str,
    state_field: str,
) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: _required_int(row["source_frame"], "matrix source frame"),
    )
    frames = [
        _required_int(row["source_frame"], "matrix source frame") for row in ordered
    ]
    if frames != list(range(len(ordered))):
        raise CrossContractValidationError("gap frame sequence changed")
    events: list[dict[str, object]] = []
    index = 1
    while index < len(ordered) - 1:
        if not bool(ordered[index][state_field]):
            start_index = index
            while index + 1 < len(ordered) and not bool(
                ordered[index + 1][state_field]
            ):
                index += 1
            end_index = index
            if (
                bool(ordered[start_index - 1][state_field])
                and end_index + 1 < len(ordered)
                and bool(ordered[end_index + 1][state_field])
            ):
                start = ordered[start_index]
                end = ordered[end_index]
                recovery = ordered[end_index + 1]
                scan_id = str(start["scan_id"])
                branch_id = str(start["branch_id"])
                start_frame = _required_int(start["source_frame"], "gap start frame")
                end_frame = _required_int(end["source_frame"], "gap end frame")
                identity_parts = [contract_id]
                if method:
                    identity_parts.append(method)
                identity_parts.extend(
                    [scan_id, branch_id, str(start_frame), str(end_frame)]
                )
                events.append(
                    {
                        "gap_id": "|".join(identity_parts),
                        "contract": contract_id,
                        "method": method,
                        "scan_id": scan_id,
                        "branch_id": branch_id,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "start_energy_ev": _required_float(
                            start["energy_ev"], "gap start energy"
                        ),
                        "end_energy_ev": _required_float(
                            end["energy_ev"], "gap end energy"
                        ),
                        "gap_frame_count": end_index - start_index + 1,
                        "recovery_frame": _required_int(
                            recovery["source_frame"], "gap recovery frame"
                        ),
                        "recovery_energy_ev": _required_float(
                            recovery["energy_ev"], "gap recovery energy"
                        ),
                    }
                )
        index += 1
    return events


def _build_gap_events(
    matrix: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    gaussian_events: list[dict[str, object]] = []
    viperleed_events: list[dict[str, object]] = []
    scans_and_branches = sorted(
        {(str(row["scan_id"]), str(row["branch_id"])) for row in matrix}
    )
    for method in METHODS:
        for scan_id, branch_id in scans_and_branches:
            rows = [
                row
                for row in matrix
                if row["method"] == method
                and row["scan_id"] == scan_id
                and row["branch_id"] == branch_id
            ]
            gaussian_events.extend(
                _bounded_gap_events(
                    rows,
                    contract_id="gaussian",
                    method=method,
                    state_field="gaussian_contract_measurable",
                )
            )
    native_method = METHODS[0]
    for scan_id, branch_id in scans_and_branches:
        rows = [
            row
            for row in matrix
            if row["method"] == native_method
            and row["scan_id"] == scan_id
            and row["branch_id"] == branch_id
        ]
        viperleed_events.extend(
            _bounded_gap_events(
                rows,
                contract_id="viperleed",
                method="",
                state_field="viper_native_direct_detectability",
            )
        )
    return gaussian_events, viperleed_events


def _build_gap_comparisons(
    matrix: list[dict[str, object]],
    gaussian_events: list[dict[str, object]],
    viperleed_events: list[dict[str, object]],
) -> list[dict[str, object]]:
    matrix_by_key = {
        (
            str(row["method"]),
            str(row["scan_id"]),
            str(row["branch_id"]),
            _required_int(row["source_frame"], "matrix source frame"),
        ): row
        for row in matrix
    }
    scans_and_branches = sorted(
        {(str(row["scan_id"]), str(row["branch_id"])) for row in matrix}
    )
    comparisons: list[dict[str, object]] = []
    for method in METHODS:
        for scan_id, branch_id in scans_and_branches:
            intervals = [
                event
                for event in gaussian_events
                if event["method"] == method
                and event["scan_id"] == scan_id
                and event["branch_id"] == branch_id
            ] + [
                event
                for event in viperleed_events
                if event["scan_id"] == scan_id and event["branch_id"] == branch_id
            ]
            ordered = sorted(
                intervals,
                key=lambda event: (
                    _required_int(event["start_frame"], "gap start frame"),
                    _required_int(event["end_frame"], "gap end frame"),
                    str(event["contract"]),
                ),
            )
            components: list[list[dict[str, object]]] = []
            for interval in ordered:
                if components and _required_int(
                    interval["start_frame"], "gap start frame"
                ) <= max(
                    _required_int(item["end_frame"], "gap end frame")
                    for item in components[-1]
                ):
                    components[-1].append(interval)
                else:
                    components.append([interval])
            for component in components:
                gaussian = [
                    event for event in component if event["contract"] == "gaussian"
                ]
                viperleed = [
                    event for event in component if event["contract"] == "viperleed"
                ]
                category = (
                    "shared"
                    if gaussian and viperleed
                    else "gaussian_only"
                    if gaussian
                    else "viperleed_only"
                )
                start = min(
                    _required_int(event["start_frame"], "gap start frame")
                    for event in component
                )
                end = max(
                    _required_int(event["end_frame"], "gap end frame")
                    for event in component
                )
                exact = bool(
                    len(gaussian) == 1
                    and len(viperleed) == 1
                    and _required_int(
                        gaussian[0]["start_frame"], "Gaussian gap start frame"
                    )
                    == _required_int(
                        viperleed[0]["start_frame"], "ViPErLEED gap start frame"
                    )
                    and _required_int(
                        gaussian[0]["end_frame"], "Gaussian gap end frame"
                    )
                    == _required_int(
                        viperleed[0]["end_frame"], "ViPErLEED gap end frame"
                    )
                )
                exit_frame = end + 1
                exit_row = matrix_by_key.get((method, scan_id, branch_id, exit_frame))
                gaussian_exit = bool(
                    exit_row is not None
                    and exit_row["gaussian_contract_measurable"] is True
                )
                viperleed_exit = bool(
                    exit_row is not None
                    and exit_row["viper_native_direct_detectability"] is True
                )
                comparisons.append(
                    {
                        "comparison_id": (
                            f"{method}|{scan_id}|{branch_id}|{start}|{end}"
                        ),
                        "method": method,
                        "scan_id": scan_id,
                        "branch_id": branch_id,
                        "component_start_frame": start,
                        "component_end_frame": end,
                        "gaussian_gap_ids": ";".join(
                            str(event["gap_id"]) for event in gaussian
                        ),
                        "viperleed_gap_ids": ";".join(
                            str(event["gap_id"]) for event in viperleed
                        ),
                        "comparison_category": category,
                        "exact_boundary_agreement": exact,
                        "component_exit_frame": exit_frame,
                        "component_exit_energy_ev": (
                            _required_float(
                                exit_row["energy_ev"], "component exit energy"
                            )
                            if exit_row is not None
                            else None
                        ),
                        "gaussian_observes_at_component_exit": gaussian_exit,
                        "viperleed_observes_at_component_exit": viperleed_exit,
                        "both_contracts_recover_at_component_exit": (
                            gaussian_exit and viperleed_exit
                        ),
                    }
                )
    return comparisons


def _build_method_summaries(
    matrix: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summaries: list[dict[str, object]] = []
    out_of_branch_events: list[dict[str, object]] = []
    for method in METHODS:
        rows = [row for row in matrix if row["method"] == method]
        reference_visible = [
            row for row in rows if row["physics_baseline_status"] == "visible_fit"
        ]
        branch_consistent = [
            row
            for row in reference_visible
            if row["measurement_status"] == "branch_consistent"
        ]
        accepted = [row for row in rows if row["accepted_fit"] is True]
        out_of_branch = [row for row in rows if row["accepted_out_of_branch"] is True]
        classifications = {
            label: sum(
                int(row["nominal_out_of_branch_classification"] == label)
                for row in out_of_branch
            )
            for label in (
                "branch_model_disagreement",
                "cross_contract_target_inconsistency",
                "unresolved",
            )
        }
        if len(reference_visible) != 221:
            raise CrossContractValidationError(
                "reference-visible opportunity denominator changed"
            )
        if sum(classifications.values()) != len(out_of_branch):
            raise CrossContractValidationError(
                "out-of-branch classification denominator changed"
            )
        outcome_counts = {
            label: sum(int(row["measurability_outcome"] == label) for row in rows)
            for label in OUTCOME_LABELS
        }
        summaries.append(
            {
                "method": method,
                "opportunity_count": len(rows),
                **outcome_counts,
                "reference_visible_opportunities": len(reference_visible),
                "branch_consistent_on_reference_visible": len(branch_consistent),
                "branch_consistent_measurement_coverage": (
                    len(branch_consistent) / len(reference_visible)
                ),
                "accepted_gaussian_all_frames": len(accepted),
                "out_of_branch_all_frames": len(out_of_branch),
                "all_frame_conditional_out_of_branch_fraction": (
                    len(out_of_branch) / len(accepted)
                ),
                "out_of_branch_branch_model_disagreement": classifications[
                    "branch_model_disagreement"
                ],
                "out_of_branch_cross_contract_target_inconsistency": (
                    classifications["cross_contract_target_inconsistency"]
                ),
                "out_of_branch_unresolved": classifications["unresolved"],
            }
        )
        for row in out_of_branch:
            classification = row["nominal_out_of_branch_classification"]
            if classification is None:
                raise CrossContractValidationError(
                    "accepted out-of-branch event is unclassified"
                )
            out_of_branch_events.append(
                {
                    "event_id": f"out-of-branch|{row['record_key']}",
                    "record_key": row["record_key"],
                    "opportunity_key": row["opportunity_key"],
                    "method": method,
                    "scan_id": row["scan_id"],
                    "branch_id": row["branch_id"],
                    "source_frame": row["source_frame"],
                    "energy_ev": row["energy_ev"],
                    "gaussian_x": row["gaussian_x"],
                    "gaussian_y": row["gaussian_y"],
                    "viper_native_direct_detectability": row[
                        "viper_native_direct_detectability"
                    ],
                    "native_raw_x": row["native_raw_x"],
                    "native_raw_y": row["native_raw_y"],
                    "cross_contract_distance_px": row[
                        "nominal_cross_contract_distance_px"
                    ],
                    "nominal_positional_tolerance_px": 1.0,
                    "classification": classification,
                }
            )
    return summaries, out_of_branch_events


def _variant_matrix(
    matrix: list[dict[str, object]],
    significance_threshold: float,
    positional_tolerance: float,
) -> list[dict[str, object]]:
    variant: list[dict[str, object]] = []
    for source in matrix:
        row = dict(source)
        significance = _optional_float(
            row["native_significance"], "variant native significance"
        )
        native_dx = _optional_float(row["native_dx_raw"], "variant native dx")
        native_dy = _optional_float(row["native_dy_raw"], "variant native dy")
        native_direct = _native_direct_detectability(
            significance,
            native_dx,
            native_dy,
            significance_threshold,
        )
        gaussian = row["gaussian_contract_measurable"] is True
        distance = _cross_contract_distance(
            gaussian,
            native_direct,
            _optional_float(row["gaussian_x"], "variant Gaussian x"),
            _optional_float(row["gaussian_y"], "variant Gaussian y"),
            _optional_float(row["native_raw_x"], "variant native x"),
            _optional_float(row["native_raw_y"], "variant native y"),
        )
        classification = _out_of_branch_classification(
            row["accepted_out_of_branch"] is True,
            native_direct,
            distance,
            positional_tolerance,
        )
        row.update(
            {
                "viper_native_direct_detectability": native_direct,
                "measurability_outcome": _measurability_outcome(
                    gaussian, native_direct
                ),
                "nominal_cross_contract_distance_px": distance,
                "nominal_out_of_branch_classification": classification,
            }
        )
        variant.append(row)
    return variant


def _variant_id(significance_threshold: float, positional_tolerance: float) -> str:
    significance = format(significance_threshold, "g").replace(".", "p")
    tolerance = format(positional_tolerance, "g").replace(".", "p")
    return f"sig_{significance}_tol_{tolerance}"


def _build_sensitivity_rows(
    matrix: list[dict[str, object]],
    contract: Mapping[str, Any],
) -> list[dict[str, object]]:
    sensitivity = contract.get("sensitivity_contract")
    if not isinstance(sensitivity, dict):
        raise CrossContractValidationError("sensitivity contract changed")
    thresholds = sensitivity.get("native_significance_threshold")
    tolerances = sensitivity.get("cross_contract_positional_tolerance_px")
    if (
        thresholds != [2.0, 2.5, 3.0]
        or tolerances != [0.5, 1.0, 2.0]
        or sensitivity.get("no_method_ranking_or_variant_selection") is not True
        or sensitivity.get("report_all_cartesian_combinations") is not True
    ):
        raise CrossContractValidationError("sensitivity grid changed")
    rows: list[dict[str, object]] = []
    for threshold_value in thresholds:
        threshold = float(threshold_value)
        for tolerance_value in tolerances:
            tolerance = float(tolerance_value)
            variant_matrix = _variant_matrix(matrix, threshold, tolerance)
            gaussian_gaps, viperleed_gaps = _build_gap_events(variant_matrix)
            comparisons = _build_gap_comparisons(
                variant_matrix,
                gaussian_gaps,
                viperleed_gaps,
            )
            method_summaries, _ = _build_method_summaries(variant_matrix)
            for method in METHODS:
                method_summary = next(
                    row for row in method_summaries if row["method"] == method
                )
                method_comparisons = [
                    row for row in comparisons if row["method"] == method
                ]
                rows.append(
                    {
                        "variant_id": _variant_id(threshold, tolerance),
                        "native_significance_threshold": threshold,
                        "cross_contract_positional_tolerance_px": tolerance,
                        "method": method,
                        "opportunity_count": method_summary["opportunity_count"],
                        "both_contracts_observe": method_summary[
                            "both_contracts_observe"
                        ],
                        "gaussian_only": method_summary["gaussian_only"],
                        "viperleed_only": method_summary["viperleed_only"],
                        "neither_contract_observes": method_summary[
                            "neither_contract_observes"
                        ],
                        "gaussian_gap_event_count": sum(
                            int(event["method"] == method) for event in gaussian_gaps
                        ),
                        "viperleed_gap_event_count": len(viperleed_gaps),
                        "gap_comparison_component_count": len(method_comparisons),
                        "gap_comparison_shared": sum(
                            int(row["comparison_category"] == "shared")
                            for row in method_comparisons
                        ),
                        "gap_comparison_gaussian_only": sum(
                            int(row["comparison_category"] == "gaussian_only")
                            for row in method_comparisons
                        ),
                        "gap_comparison_viperleed_only": sum(
                            int(row["comparison_category"] == "viperleed_only")
                            for row in method_comparisons
                        ),
                        "gap_exact_boundary_agreement": sum(
                            int(row["exact_boundary_agreement"] is True)
                            for row in method_comparisons
                        ),
                        "both_contracts_recover_at_component_exit": sum(
                            int(row["both_contracts_recover_at_component_exit"] is True)
                            for row in method_comparisons
                        ),
                        "reference_visible_opportunities": method_summary[
                            "reference_visible_opportunities"
                        ],
                        "branch_consistent_on_reference_visible": (
                            method_summary["branch_consistent_on_reference_visible"]
                        ),
                        "accepted_gaussian_all_frames": method_summary[
                            "accepted_gaussian_all_frames"
                        ],
                        "out_of_branch_all_frames": method_summary[
                            "out_of_branch_all_frames"
                        ],
                        "out_of_branch_branch_model_disagreement": (
                            method_summary["out_of_branch_branch_model_disagreement"]
                        ),
                        "out_of_branch_cross_contract_target_inconsistency": (
                            method_summary[
                                "out_of_branch_cross_contract_target_inconsistency"
                            ]
                        ),
                        "out_of_branch_unresolved": method_summary[
                            "out_of_branch_unresolved"
                        ],
                    }
                )
    if len(rows) != 45:
        raise CrossContractValidationError("sensitivity result denominator changed")
    return rows


def _analysis_report(
    matrix: list[dict[str, object]],
    gaussian_events: list[dict[str, object]],
    viperleed_events: list[dict[str, object]],
    comparisons: list[dict[str, object]],
    method_summaries: list[dict[str, object]],
    out_of_branch_events: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
) -> dict[str, object]:
    by_method: dict[str, dict[str, int]] = {}
    gap_counts: dict[str, int] = {}
    comparison_by_method: dict[str, dict[str, int]] = {}
    denominators_by_method: dict[str, dict[str, int]] = {}
    for method in METHODS:
        rows = [row for row in matrix if row["method"] == method]
        by_method[method] = {
            outcome: sum(int(row["measurability_outcome"] == outcome) for row in rows)
            for outcome in OUTCOME_LABELS
        }
        gap_counts[method] = sum(
            int(event["method"] == method) for event in gaussian_events
        )
        method_comparisons = [row for row in comparisons if row["method"] == method]
        comparison_by_method[method] = {
            "both_contracts_recover_at_component_exit": sum(
                int(row["both_contracts_recover_at_component_exit"] is True)
                for row in method_comparisons
            ),
            "component_count": len(method_comparisons),
            "exact_boundary_agreement": sum(
                int(row["exact_boundary_agreement"] is True)
                for row in method_comparisons
            ),
            "gaussian_only": sum(
                int(row["comparison_category"] == "gaussian_only")
                for row in method_comparisons
            ),
            "shared": sum(
                int(row["comparison_category"] == "shared")
                for row in method_comparisons
            ),
            "viperleed_only": sum(
                int(row["comparison_category"] == "viperleed_only")
                for row in method_comparisons
            ),
        }
        method_summary = next(
            row for row in method_summaries if row["method"] == method
        )
        denominators_by_method[method] = {
            "accepted_gaussian_all_frames": _required_int(
                method_summary["accepted_gaussian_all_frames"],
                "accepted Gaussian denominator",
            ),
            "branch_consistent_on_reference_visible": _required_int(
                method_summary["branch_consistent_on_reference_visible"],
                "branch-consistent numerator",
            ),
            "out_of_branch_all_frames": _required_int(
                method_summary["out_of_branch_all_frames"],
                "out-of-branch numerator",
            ),
            "out_of_branch_branch_model_disagreement": _required_int(
                method_summary["out_of_branch_branch_model_disagreement"],
                "branch-model-disagreement numerator",
            ),
            "out_of_branch_cross_contract_target_inconsistency": _required_int(
                method_summary["out_of_branch_cross_contract_target_inconsistency"],
                "cross-contract-target-inconsistency numerator",
            ),
            "out_of_branch_unresolved": _required_int(
                method_summary["out_of_branch_unresolved"],
                "unresolved numerator",
            ),
            "reference_visible_opportunities": _required_int(
                method_summary["reference_visible_opportunities"],
                "reference-visible denominator",
            ),
        }
    out_of_branch_classifications = {
        label: sum(int(row["classification"] == label) for row in out_of_branch_events)
        for label in (
            "branch_model_disagreement",
            "cross_contract_target_inconsistency",
            "unresolved",
        )
    }
    sensitivity_envelopes: dict[str, dict[str, list[int]]] = {}
    measurability_counts_stable = True
    for method in METHODS:
        rows = [row for row in sensitivity_rows if row["method"] == method]
        sensitivity_envelopes[method] = {}
        for output_name, field in (
            ("branch_model_disagreement", "out_of_branch_branch_model_disagreement"),
            (
                "cross_contract_target_inconsistency",
                "out_of_branch_cross_contract_target_inconsistency",
            ),
            ("unresolved", "out_of_branch_unresolved"),
        ):
            values = [_required_int(row[field], field) for row in rows]
            sensitivity_envelopes[method][output_name] = [min(values), max(values)]
        measurability_tuples = {
            tuple(
                _required_int(row[field], field)
                for field in (
                    "both_contracts_observe",
                    "gaussian_only",
                    "viperleed_only",
                    "neither_contract_observes",
                )
            )
            for row in rows
        }
        measurability_counts_stable = (
            measurability_counts_stable and len(measurability_tuples) == 1
        )
    return {
        "accepted_out_of_branch_event_count": len(out_of_branch_events),
        "branch_frame_matrix_count": len(matrix),
        "gap_comparison_by_method": comparison_by_method,
        "gaussian_gap_event_count": len(gaussian_events),
        "gaussian_gap_event_count_by_method": gap_counts,
        "measurability_outcomes_by_method": by_method,
        "method_denominators": denominators_by_method,
        "opportunity_count_per_method": 324,
        "out_of_branch_classifications": out_of_branch_classifications,
        "schema_version": "1.0",
        "sensitivity_classification_envelopes_by_method": sensitivity_envelopes,
        "sensitivity_measurability_counts_stable": measurability_counts_stable,
        "sensitivity_parameters": {
            "cross_contract_positional_tolerance_px": [0.5, 1.0, 2.0],
            "native_significance_threshold": [2.0, 2.5, 3.0],
        },
        "sensitivity_result_count": len(sensitivity_rows),
        "sensitivity_variant_count": 9,
        "status": "built-cross-contract-outcomes",
        "viperleed_gap_event_count": len(viperleed_events),
        "viperleed_gap_identities": [
            str(event["gap_id"]) for event in viperleed_events
        ],
    }


def _render_analysis(
    workspace_root: Path,
    package_root: Path,
) -> tuple[dict[str, bytes], dict[str, object]]:
    contract = _load_contract(workspace_root, package_root)
    matrix = _build_matrix(workspace_root, contract)
    gaussian_events, viperleed_events = _build_gap_events(matrix)
    comparisons = _build_gap_comparisons(
        matrix,
        gaussian_events,
        viperleed_events,
    )
    method_summaries, out_of_branch_events = _build_method_summaries(matrix)
    sensitivity_rows = _build_sensitivity_rows(matrix, contract)
    report = _analysis_report(
        matrix,
        gaussian_events,
        viperleed_events,
        comparisons,
        method_summaries,
        out_of_branch_events,
        sensitivity_rows,
    )
    artifacts = {
        "branch_frame_matrix.csv": _csv_bytes(matrix, MATRIX_FIELDS),
        "gap_comparisons.csv": _csv_bytes(comparisons, GAP_COMPARISON_FIELDS),
        "gaussian_gap_events.csv": _csv_bytes(gaussian_events, GAP_FIELDS),
        "headline_summary.json": _json_bytes(report),
        "method_summary.csv": _csv_bytes(method_summaries, METHOD_SUMMARY_FIELDS),
        "out_of_branch_events.csv": _csv_bytes(
            out_of_branch_events, OUT_OF_BRANCH_FIELDS
        ),
        "sensitivity_results.csv": _csv_bytes(sensitivity_rows, SENSITIVITY_FIELDS),
        "viperleed_gap_events.csv": _csv_bytes(viperleed_events, GAP_FIELDS),
    }
    return artifacts, report


def _build_manifest(package_root: Path) -> dict[str, object]:
    files = []
    for relative, role in sorted(OUTPUT_ROLES.items()):
        path = package_root / relative
        if not path.is_file():
            raise CrossContractValidationError(
                f"cross-contract manifest input is unavailable: {relative}"
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
        "analysis_id": "principal-cross-contract-outcomes",
        "file_count": len(files),
        "files": files,
        "schema_version": "1.0",
        "status": "frozen-cross-contract-outcome-manifest",
    }


def _validation_report(
    package_root: Path,
    analysis_report: Mapping[str, object],
) -> dict[str, object]:
    return {
        **analysis_report,
        "analysis_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "fail_closed": True,
        "manifest_sha256": _sha256(package_root / "manifest.json"),
        "status": "validated-cross-contract-outcomes",
    }


def _package_path(package_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise CrossContractValidationError("manifested output path changed")
    path = (package_root / value).resolve()
    if not path.is_relative_to(package_root) or not path.is_file():
        raise CrossContractValidationError(f"manifested output is unavailable: {value}")
    return path


def _validate_frozen_analysis(
    workspace_root: Path,
    package_root: Path,
) -> dict[str, object]:
    _load_contract(workspace_root, package_root)
    manifest_path = package_root / "manifest.json"
    if not manifest_path.is_file():
        raise CrossContractValidationError(
            f"cross-contract analysis manifest is unavailable: {package_root}"
        )
    if _sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise CrossContractValidationError("cross-contract analysis manifest changed")
    manifest = _load_json(manifest_path, "cross-contract analysis manifest")
    manifested_files = manifest.get("files")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("analysis_id") != "principal-cross-contract-outcomes"
        or manifest.get("status") != "frozen-cross-contract-outcome-manifest"
        or not isinstance(manifested_files, list)
        or manifest.get("file_count") != len(manifested_files)
    ):
        raise CrossContractValidationError(
            "cross-contract analysis manifest identity changed"
        )
    manifested: set[str] = set()
    for item in manifested_files:
        if not isinstance(item, dict):
            raise CrossContractValidationError("cross-contract manifest record changed")
        relative = item.get("path")
        path = _package_path(package_root, relative)
        if not isinstance(relative, str) or relative in manifested:
            raise CrossContractValidationError("duplicate cross-contract manifest path")
        if (
            item.get("role") != OUTPUT_ROLES.get(relative)
            or _sha256(path) != item.get("sha256")
            or path.stat().st_size != item.get("size_bytes")
        ):
            raise CrossContractValidationError(
                f"manifested cross-contract output drift: {relative}"
            )
        manifested.add(relative)
    if manifested != set(OUTPUT_ROLES):
        raise CrossContractValidationError("cross-contract manifest coverage changed")
    actual_names = {path.name for path in package_root.iterdir() if path.is_file()}
    if actual_names != set(OUTPUT_ROLES) | {"manifest.json", "validation_report.json"}:
        raise CrossContractValidationError(
            "cross-contract package file inventory changed"
        )
    expected_artifacts, analysis_report = _render_analysis(workspace_root, package_root)
    for relative, expected in expected_artifacts.items():
        if (package_root / relative).read_bytes() != expected:
            raise CrossContractValidationError(
                f"cross-contract output no longer reproduces: {relative}"
            )
    if manifest_path.read_bytes() != _json_bytes(_build_manifest(package_root)):
        raise CrossContractValidationError(
            "cross-contract analysis manifest no longer reproduces"
        )
    expected_headline = {
        "accepted_out_of_branch_event_count": 124,
        "branch_frame_matrix_count": 1620,
        "gaussian_gap_event_count": 114,
        "opportunity_count_per_method": 324,
        "sensitivity_result_count": 45,
        "viperleed_gap_event_count": 2,
    }
    if any(
        analysis_report.get(key) != value for key, value in expected_headline.items()
    ):
        raise CrossContractValidationError(
            "cross-contract headline count or denominator changed"
        )
    report = _validation_report(package_root, analysis_report)
    report_path = package_root / "validation_report.json"
    if not report_path.is_file() or report_path.read_bytes() != _json_bytes(report):
        raise CrossContractValidationError("cross-contract validation report changed")
    return report


def build_cross_contract_outcomes(
    workspace_root: Path,
    package_root: Path | None = None,
) -> dict[str, object]:
    """Build deterministic cross-contract source data from frozen inputs."""

    workspace_root = workspace_root.resolve()
    package_root = (
        package_root.resolve()
        if package_root is not None
        else workspace_root / DEFAULT_PACKAGE_PATH
    )
    _load_contract(workspace_root, package_root)
    if not (package_root / "README.md").is_file():
        raise CrossContractValidationError(
            f"cross-contract analysis README is unavailable: {package_root}"
        )
    if any((package_root / relative).exists() for relative in GENERATED_PATHS):
        raise FileExistsError(
            f"Refusing to replace frozen cross-contract analysis: {package_root}"
        )
    with TemporaryDirectory(
        prefix="cross-contract-outcomes-",
        dir=package_root.parent,
    ) as temporary:
        stage = Path(temporary) / "outcomes"
        stage.mkdir()
        shutil.copyfile(
            package_root / "analysis_contract.json",
            stage / "analysis_contract.json",
        )
        shutil.copyfile(package_root / "README.md", stage / "README.md")
        artifacts, analysis_report = _render_analysis(workspace_root, stage)
        for relative, content in artifacts.items():
            (stage / relative).write_bytes(content)
        (stage / "manifest.json").write_bytes(_json_bytes(_build_manifest(stage)))
        (stage / "validation_report.json").write_bytes(
            _json_bytes(_validation_report(stage, analysis_report))
        )
        validated = _validate_frozen_analysis(workspace_root, stage)
        if _sha256(package_root / "analysis_contract.json") != _sha256(
            stage / "analysis_contract.json"
        ) or _sha256(package_root / "README.md") != _sha256(stage / "README.md"):
            raise CrossContractValidationError(
                "cross-contract package changed during staged validation"
            )
        for relative in GENERATED_PATHS:
            os.replace(stage / relative, package_root / relative)
    return validated


def validate_cross_contract_outcomes(
    workspace_root: Path,
    package_root: Path | None = None,
) -> dict[str, object]:
    """Validate an existing frozen cross-contract analysis package."""

    workspace_root = workspace_root.resolve()
    package_root = (
        package_root.resolve()
        if package_root is not None
        else workspace_root / DEFAULT_PACKAGE_PATH
    )
    return _validate_frozen_analysis(workspace_root, package_root)


__all__ = [
    "CrossContractValidationError",
    "build_cross_contract_outcomes",
    "validate_cross_contract_outcomes",
]
