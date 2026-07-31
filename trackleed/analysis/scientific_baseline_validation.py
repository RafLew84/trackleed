"""Fail-closed validation of a frozen scientific-baseline package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trackleed.analysis.scientific_baseline_common import (
    EXPECTED_CASE_COUNT,
    EXPECTED_EVALUABLE_GAPS,
    EXPECTED_VISIBLE_OPPORTUNITIES,
    GAP_FIELDS,
    METHODS,
    OPPORTUNITY_FIELDS,
    BaselineValidationError,
    PackageReport,
    README_TEMPLATE,
    SCHEMA_VERSION,
    STUDY,
    _gap_rows,
    _keys_sha256,
    _load_inventory,
    _opportunity_rows,
    _read_csv,
    _schema,
    _sha256,
    _source_location,
)
from trackleed.analysis.scientific_baseline_contract import measurement_contract


def _validation_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BaselineValidationError(f"cannot read frozen JSON: {path}") from exc


def _validation_location(workspace_root: Path, relative_path: str) -> Path:
    try:
        return _source_location(workspace_root, relative_path)
    except ValueError as exc:
        raise BaselineValidationError(
            f"invalid frozen source path: {relative_path}"
        ) from exc


def _validate_inventory_bindings(
    workspace_root: Path,
    inventory: dict[str, Any],
    records_by_id: dict[str, dict[str, object]],
) -> None:
    expected_raw_cases: dict[str, set[str]] = {}
    for case in inventory["cases"]:
        case_id = str(case["case_id"])
        series_id = str(case["series_id"])
        expected_bindings = {
            f"case-contract:{case_id}": (
                "case_contract",
                str(case["baseline_summary"]),
                f"frozen_inputs/cases/{case_id}/summary.json",
                None,
            ),
            f"physics-baseline:{case_id}": (
                "physics_baseline",
                str(case["baseline_measurements"]),
                f"frozen_inputs/cases/{case_id}/scan_measurements.csv",
                None,
            ),
        }
        for method in METHODS:
            proposal_path = str(case["proposals"][method])
            suffix = Path(proposal_path).suffix.lower()
            expected_bindings[f"native-proposals:{case_id}:{method}"] = (
                "native_proposals",
                proposal_path,
                f"frozen_inputs/proposals/{case_id}/{method}{suffix}",
                method,
            )

        for artifact_id, (
            role,
            configured_source,
            package_path,
            method,
        ) in expected_bindings.items():
            record = records_by_id.get(artifact_id)
            source = _validation_location(workspace_root, configured_source)
            normalized_source = source.relative_to(workspace_root).as_posix()
            if record is None or any(
                (
                    record.get("role") != role,
                    record.get("source_path") != normalized_source,
                    record.get("package_path") != package_path,
                    record.get("case_id") != case_id,
                    record.get("pair_id") != str(case["pair_id"]),
                    record.get("series_id") != series_id,
                    record.get("method") != method if method is not None else False,
                )
            ):
                raise BaselineValidationError(
                    "source inventory no longer reproduces the frozen package: "
                    f"{artifact_id}"
                )

        raw_spec = case["raw_images"]
        raw_directory = Path(str(raw_spec["directory"]))
        resolved_directory = (workspace_root / raw_directory).resolve()
        if (
            raw_directory.is_absolute()
            or not resolved_directory.is_relative_to(workspace_root)
        ):
            raise BaselineValidationError(
                "source inventory no longer reproduces the frozen package: "
                f"invalid raw-image directory for {case_id}"
            )
        raw_paths = sorted(resolved_directory.glob(str(raw_spec["pattern"])))
        if len(raw_paths) != int(raw_spec["expected_count"]):
            raise BaselineValidationError(
                "source inventory no longer reproduces the frozen package: "
                f"raw-image count for {case_id}"
            )
        for path in raw_paths:
            artifact_id = f"raw-image:{series_id}:{path.name}"
            expected_raw_cases.setdefault(artifact_id, set()).add(case_id)
            record = records_by_id.get(artifact_id)
            normalized_source = path.relative_to(workspace_root).as_posix()
            if (
                record is None
                or record.get("role") != "raw_image"
                or record.get("source_path") != normalized_source
                or record.get("series_id") != series_id
            ):
                raise BaselineValidationError(
                    "source inventory no longer reproduces the frozen package: "
                    f"{artifact_id}"
                )

    for artifact_id, expected_case_ids in expected_raw_cases.items():
        actual_case_ids = records_by_id[artifact_id].get("case_ids")
        if not isinstance(actual_case_ids, list) or set(actual_case_ids) != expected_case_ids:
            raise BaselineValidationError(
                "source inventory no longer reproduces the frozen package: "
                f"{artifact_id} case bindings"
            )


def _validate_frozen_input_schemas(
    package_dir: Path,
    inventory: dict[str, Any],
    records_by_id: dict[str, dict[str, object]],
) -> None:
    required_baseline_columns = {"frame", "energy_ev", "branch", "status"}
    required_csv_proposal_columns = {
        "frame",
        "branch",
        "native_x",
        "native_y",
    }
    required_npz_arrays = {
        "tracks_xy": ("float32", lambda n: [2, n, 2]),
        "visible_mask": ("bool", lambda n: [2, n]),
        "mask_areas": ("float32", lambda n: [2, n]),
        "mask_bboxes_xyxy": ("float32", lambda n: [2, n, 4]),
        "mask_scores": ("float32", lambda n: [2, n]),
        "mask_component_counts": ("int32", lambda n: [2, n]),
    }

    for case in inventory["cases"]:
        case_id = str(case["case_id"])
        frame_count = int(case["raw_images"]["expected_count"])
        case_root = package_dir / "frozen_inputs" / "cases" / case_id
        summary = _validation_json(case_root / "summary.json")
        if (
            not isinstance(summary, dict)
            or "reference_energy_ev" not in summary
            or not isinstance(summary.get("visibility_gaps"), list)
        ):
            raise BaselineValidationError(
                f"case contract schema changed for {case_id}"
            )
        thresholds = summary.get("thresholds")
        if thresholds is not None:
            expected_thresholds = {
                "roi_size_px": 48,
                "r_squared_min": 0.85,
                "amplitude_min": 0.012,
                "center_tolerance_px": 15.0,
                "branch_tolerance_px": 18.0,
            }
            if not isinstance(thresholds, dict) or any(
                thresholds.get(key) != value
                for key, value in expected_thresholds.items()
            ):
                raise BaselineValidationError(
                    f"central case thresholds changed for {case_id}"
                )

        baseline_record = records_by_id[f"physics-baseline:{case_id}"]
        baseline_schema = _schema(case_root / "scan_measurements.csv")
        if baseline_schema != baseline_record.get("schema"):
            raise BaselineValidationError(
                f"physics baseline schema record changed for {case_id}"
            )
        if (
            baseline_schema.get("format") != "csv"
            or not required_baseline_columns.issubset(
                set(baseline_schema.get("columns", []))
            )
            or baseline_schema.get("row_count") != 2 * frame_count
        ):
            raise BaselineValidationError(
                f"physics baseline schema changed for {case_id}"
            )

        for method in METHODS:
            artifact_id = f"native-proposals:{case_id}:{method}"
            record = records_by_id[artifact_id]
            frozen_path = package_dir / str(record["package_path"])
            schema = _schema(frozen_path)
            if schema != record.get("schema"):
                raise BaselineValidationError(
                    f"native proposal schema record changed for {case_id}/{method}"
                )
            if schema.get("format") == "csv":
                if not required_csv_proposal_columns.issubset(
                    set(schema.get("columns", []))
                ):
                    raise BaselineValidationError(
                        f"native proposal schema changed for {case_id}/{method}"
                    )
                continue
            if schema.get("format") != "npz":
                raise BaselineValidationError(
                    f"native proposal schema changed for {case_id}/{method}"
                )
            arrays = schema.get("arrays")
            if not isinstance(arrays, dict) or set(arrays) != set(required_npz_arrays):
                raise BaselineValidationError(
                    f"native proposal schema changed for {case_id}/{method}"
                )
            for name, (dtype, shape_for_count) in required_npz_arrays.items():
                if arrays.get(name) != {
                    "dtype": dtype,
                    "shape": shape_for_count(frame_count),
                }:
                    raise BaselineValidationError(
                        f"native proposal schema changed for {case_id}/{method}"
                    )


def validate_package(workspace_root: Path, package_dir: Path) -> PackageReport:
    """Validate a frozen package and its linked sources without modifying them."""

    workspace_root = workspace_root.resolve()
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "package_manifest.json"
    manifest = _validation_json(manifest_path)
    if not isinstance(manifest, dict):
        raise BaselineValidationError("package manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise BaselineValidationError("package manifest schema version changed")
    if manifest.get("study") != STUDY or manifest.get("immutable") is not True:
        raise BaselineValidationError("package manifest scientific identity changed")

    expected_files = manifest.get("files_sha256")
    if not isinstance(expected_files, dict):
        raise BaselineValidationError("package manifest has no file hash map")
    actual_files = {
        path.relative_to(package_dir).as_posix(): path
        for path in sorted(package_dir.rglob("*"))
        if path.is_file() and path.name != "package_manifest.json"
    }
    if set(actual_files) != set(expected_files):
        missing = sorted(set(expected_files) - set(actual_files))
        unexpected = sorted(set(actual_files) - set(expected_files))
        raise BaselineValidationError(
            f"package inventory changed; missing={missing}, unexpected={unexpected}"
        )
    for relative_path, path in actual_files.items():
        if _sha256(path) != expected_files[relative_path]:
            raise BaselineValidationError(
                f"package hash mismatch: {relative_path}"
            )

    input_manifest = _validation_json(package_dir / "input_manifest.json")
    if (
        not isinstance(input_manifest, dict)
        or input_manifest.get("schema_version") != SCHEMA_VERSION
        or input_manifest.get("study") != STUDY
        or not isinstance(input_manifest.get("artifacts"), list)
    ):
        raise BaselineValidationError("input manifest schema changed")

    artifact_ids: set[str] = set()
    records_by_id: dict[str, dict[str, object]] = {}
    roles: list[str] = []
    methods: set[str] = set()
    cases: set[str] = set()
    raw_images = 0
    for record in input_manifest["artifacts"]:
        if not isinstance(record, dict):
            raise BaselineValidationError("input artifact record is not an object")
        artifact_id = str(record.get("artifact_id", ""))
        if not artifact_id or artifact_id in artifact_ids:
            raise BaselineValidationError(
                f"duplicate or empty input artifact ID: {artifact_id!r}"
            )
        artifact_ids.add(artifact_id)
        records_by_id[artifact_id] = record
        role = str(record.get("role", ""))
        roles.append(role)
        if role == "raw_image":
            raw_images += 1
        if record.get("method") is not None:
            methods.add(str(record["method"]))
        if record.get("case_id") is not None:
            cases.add(str(record["case_id"]))
        if isinstance(record.get("case_ids"), list):
            cases.update(str(value) for value in record["case_ids"])

        package_path = record.get("package_path")
        source_path = str(record.get("source_path", ""))
        source = _validation_location(workspace_root, source_path)
        if source.is_file():
            if _sha256(source) != record.get("source_sha256"):
                raise BaselineValidationError(
                    f"source hash mismatch for {artifact_id}: {source_path}"
                )
            if source.stat().st_size != record.get("size_bytes"):
                raise BaselineValidationError(
                    f"source size mismatch for {artifact_id}: {source_path}"
                )
            if _schema(source) != record.get("schema"):
                raise BaselineValidationError(
                    f"source schema mismatch for {artifact_id}: {source_path}"
                )
        elif package_path is None:
            raise BaselineValidationError(
                f"cannot resolve frozen source: {source_path}"
            )
        if package_path is not None:
            frozen_copy = package_dir / str(package_path)
            if not frozen_copy.is_file():
                raise BaselineValidationError(
                    f"frozen input copy is missing for {artifact_id}"
                )
            if _sha256(frozen_copy) != record.get("source_sha256"):
                raise BaselineValidationError(
                    f"frozen input copy differs from source for {artifact_id}"
                )

    try:
        inventory = _load_inventory(package_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise BaselineValidationError(
            "source inventory no longer reproduces the frozen package"
        ) from exc
    _validate_inventory_bindings(workspace_root, inventory, records_by_id)
    _validate_frozen_input_schemas(package_dir, inventory, records_by_id)

    if methods != set(METHODS):
        raise BaselineValidationError(
            f"method inventory changed: {sorted(methods)}"
        )
    if len(cases) != EXPECTED_CASE_COUNT:
        raise BaselineValidationError(f"case inventory changed: {sorted(cases)}")
    if roles.count("physics_baseline") != EXPECTED_CASE_COUNT:
        raise BaselineValidationError("physics-baseline inventory changed")
    if roles.count("case_contract") != EXPECTED_CASE_COUNT:
        raise BaselineValidationError("case-contract inventory changed")
    if roles.count("native_proposals") != EXPECTED_CASE_COUNT * len(METHODS):
        raise BaselineValidationError("native-proposal inventory changed")

    contract = _validation_json(package_dir / "measurement_contract.json")
    try:
        expected_readme = README_TEMPLATE.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise BaselineValidationError("cannot read baseline README template") from exc
    try:
        actual_readme = (package_dir / "README.md").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise BaselineValidationError("cannot read frozen baseline README") from exc
    if actual_readme != expected_readme:
        raise BaselineValidationError("baseline README changed")

    opportunity_columns, opportunities = _read_csv(
        package_dir / "opportunities.csv"
    )
    gap_columns, gaps = _read_csv(package_dir / "gaps.csv")
    if opportunity_columns != OPPORTUNITY_FIELDS:
        raise BaselineValidationError("opportunity schema changed")
    if gap_columns != GAP_FIELDS:
        raise BaselineValidationError("gap schema changed")
    opportunity_keys = [row["opportunity_key"] for row in opportunities]
    gap_keys = [row["gap_key"] for row in gaps]
    if (
        len(opportunities) != EXPECTED_VISIBLE_OPPORTUNITIES
        or len(set(opportunity_keys)) != EXPECTED_VISIBLE_OPPORTUNITIES
    ):
        raise BaselineValidationError("fixed 221 opportunity keys changed")
    if (
        len(gaps) != EXPECTED_EVALUABLE_GAPS
        or len(set(gap_keys)) != EXPECTED_EVALUABLE_GAPS
    ):
        raise BaselineValidationError("fixed 24 gap keys changed")
    expected_opportunities: list[dict[str, object]] = []
    expected_gaps: list[dict[str, object]] = []
    for case in inventory["cases"]:
        case_id = str(case["case_id"])
        series_id = str(case["series_id"])
        raw_spec = case["raw_images"]
        raw_directory = Path(str(raw_spec["directory"]))
        raw_paths = sorted(
            (workspace_root / raw_directory).resolve().glob(
                str(raw_spec["pattern"])
            )
        )
        image_records = [
            records_by_id[f"raw-image:{series_id}:{path.name}"]
            for path in raw_paths
        ]
        expected_opportunities.extend(
            _opportunity_rows(
                case,
                package_dir
                / "frozen_inputs"
                / "cases"
                / case_id
                / "scan_measurements.csv",
                image_records,
            )
        )
        expected_gaps.extend(
            _gap_rows(
                case,
                package_dir / "frozen_inputs" / "cases" / case_id / "summary.json",
            )
        )
    expected_opportunity_rows = [
        {field: str(row[field]) for field in OPPORTUNITY_FIELDS}
        for row in sorted(
            expected_opportunities,
            key=lambda row: str(row["opportunity_key"]),
        )
    ]
    expected_gap_rows = [
        {field: str(row[field]) for field in GAP_FIELDS}
        for row in sorted(expected_gaps, key=lambda row: str(row["gap_key"]))
    ]
    if opportunities != expected_opportunity_rows:
        raise BaselineValidationError(
            "opportunity manifest no longer matches frozen physics baseline"
        )
    if gaps != expected_gap_rows:
        raise BaselineValidationError(
            "gap manifest no longer matches frozen physics baseline"
        )
    expected_contract = measurement_contract(
        opportunity_keys_sha256=_keys_sha256(opportunity_keys),
        gap_keys_sha256=_keys_sha256(gap_keys),
    )
    if contract != expected_contract:
        raise BaselineValidationError("measurement contract changed")

    report = PackageReport(
        case_count=len(cases),
        method_count=len(methods),
        raw_image_count=raw_images,
        visible_opportunity_count=len(opportunities),
        evaluable_gap_count=len(gaps),
    )
    expected_summary = {
        "case_count": report.case_count,
        "method_count": report.method_count,
        "raw_image_count": report.raw_image_count,
        "visible_opportunity_count": report.visible_opportunity_count,
        "evaluable_gap_count": report.evaluable_gap_count,
    }
    if manifest.get("summary") != expected_summary:
        raise BaselineValidationError("package summary changed")
    return report
