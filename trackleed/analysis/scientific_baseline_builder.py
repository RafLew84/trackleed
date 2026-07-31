"""Build a new immutable scientific-baseline package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trackleed.analysis.scientific_baseline_common import (
    EXPECTED_CASE_COUNT,
    EXPECTED_EVALUABLE_GAPS,
    EXPECTED_VISIBLE_OPPORTUNITIES,
    GAP_FIELDS,
    GENERATED_NAMES,
    METHODS,
    OPPORTUNITY_FIELDS,
    PackageReport,
    README_TEMPLATE,
    SCHEMA_VERSION,
    STUDY,
    _artifact_record,
    _copy_frozen_input,
    _gap_rows,
    _keys_sha256,
    _load_inventory,
    _opportunity_rows,
    _resolve_source,
    _sha256,
    _write_csv,
    _write_json,
)
from trackleed.analysis.scientific_baseline_contract import measurement_contract
from trackleed.analysis.scientific_baseline_validation import validate_package


def _package_manifest(package_dir: Path, report: PackageReport) -> dict[str, object]:
    files = {}
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or path.name == "package_manifest.json":
            continue
        files[path.relative_to(package_dir).as_posix()] = _sha256(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY,
        "immutable": True,
        "summary": {
            "case_count": report.case_count,
            "method_count": report.method_count,
            "raw_image_count": report.raw_image_count,
            "visible_opportunity_count": report.visible_opportunity_count,
            "evaluable_gap_count": report.evaluable_gap_count,
        },
        "files_sha256": files,
    }


def build_package(workspace_root: Path, package_dir: Path) -> PackageReport:
    """Create a new frozen package from its authored source inventory."""

    workspace_root = workspace_root.resolve()
    package_dir = package_dir.resolve()
    existing_generated = [
        name for name in GENERATED_NAMES if (package_dir / name).exists()
    ]
    if existing_generated:
        raise FileExistsError(
            "Refusing to replace frozen package content: "
            + ", ".join(sorted(existing_generated))
        )

    inventory = _load_inventory(package_dir)
    cases: list[dict[str, Any]] = inventory["cases"]
    case_ids = [str(case.get("case_id", "")) for case in cases]
    if len(set(case_ids)) != EXPECTED_CASE_COUNT or any(not value for value in case_ids):
        raise ValueError("Case IDs must be non-empty and unique")

    artifacts: list[dict[str, object]] = []
    opportunities: list[dict[str, object]] = []
    gaps: list[dict[str, object]] = []
    raw_records: dict[Path, tuple[dict[str, object], list[str]]] = {}

    for case in cases:
        case_id = str(case["case_id"])
        series_id = str(case["series_id"])
        pair_id = str(case["pair_id"])
        if not series_id or not pair_id:
            raise ValueError(f"Missing series/pair identity for {case_id}")

        raw_spec = case["raw_images"]
        raw_directory = Path(str(raw_spec["directory"]))
        if raw_directory.is_absolute():
            raise ValueError(f"Raw-image directory must be relative: {raw_directory}")
        resolved_raw_dir = (workspace_root / raw_directory).resolve()
        if not resolved_raw_dir.is_relative_to(workspace_root):
            raise ValueError(f"Raw-image directory escapes the workspace: {raw_directory}")
        raw_paths = sorted(resolved_raw_dir.glob(str(raw_spec["pattern"])))
        if len(raw_paths) != int(raw_spec["expected_count"]):
            raise ValueError(
                f"{case_id}: expected {raw_spec['expected_count']} raw images, "
                f"found {len(raw_paths)}"
            )
        case_image_records: list[dict[str, object]] = []
        for source in raw_paths:
            if not source.is_file():
                raise FileNotFoundError(source)
            if source not in raw_records:
                case_bindings = [case_id]
                raw_records[source] = (
                    _artifact_record(
                        artifact_id=f"raw-image:{series_id}:{source.name}",
                        role="raw_image",
                        source=source,
                        workspace_root=workspace_root,
                        case_ids=case_bindings,
                        series_id=series_id,
                    ),
                    case_bindings,
                )
            else:
                raw_records[source][1].append(case_id)
            case_image_records.append(raw_records[source][0])

        summary_source = _resolve_source(
            workspace_root, str(case["baseline_summary"])
        )
        summary_package_path = f"frozen_inputs/cases/{case_id}/summary.json"
        _copy_frozen_input(summary_source, package_dir, summary_package_path)
        artifacts.append(
            _artifact_record(
                artifact_id=f"case-contract:{case_id}",
                role="case_contract",
                source=summary_source,
                workspace_root=workspace_root,
                package_path=summary_package_path,
                case_id=case_id,
                pair_id=pair_id,
                series_id=series_id,
            )
        )
        gaps.extend(_gap_rows(case, summary_source))

        baseline_source = _resolve_source(
            workspace_root, str(case["baseline_measurements"])
        )
        baseline_package_path = (
            f"frozen_inputs/cases/{case_id}/scan_measurements.csv"
        )
        _copy_frozen_input(baseline_source, package_dir, baseline_package_path)
        artifacts.append(
            _artifact_record(
                artifact_id=f"physics-baseline:{case_id}",
                role="physics_baseline",
                source=baseline_source,
                workspace_root=workspace_root,
                package_path=baseline_package_path,
                case_id=case_id,
                pair_id=pair_id,
                series_id=series_id,
            )
        )
        opportunities.extend(
            _opportunity_rows(case, baseline_source, case_image_records)
        )

        proposals = case.get("proposals")
        if not isinstance(proposals, dict) or set(proposals) != set(METHODS):
            raise ValueError(
                f"{case_id}: proposals must contain exactly {list(METHODS)}"
            )
        for method in METHODS:
            proposal_source = _resolve_source(
                workspace_root, str(proposals[method])
            )
            proposal_package_path = (
                f"frozen_inputs/proposals/{case_id}/{method}"
                f"{proposal_source.suffix.lower()}"
            )
            _copy_frozen_input(
                proposal_source,
                package_dir,
                proposal_package_path,
            )
            artifacts.append(
                _artifact_record(
                    artifact_id=f"native-proposals:{case_id}:{method}",
                    role="native_proposals",
                    source=proposal_source,
                    workspace_root=workspace_root,
                    package_path=proposal_package_path,
                    case_id=case_id,
                    method=method,
                    pair_id=pair_id,
                    series_id=series_id,
                )
            )

    artifacts.extend(record for record, _case_ids in raw_records.values())
    artifacts.sort(key=lambda record: str(record["artifact_id"]))
    opportunities.sort(key=lambda row: str(row["opportunity_key"]))
    gaps.sort(key=lambda row: str(row["gap_key"]))

    if len({row["opportunity_key"] for row in opportunities}) != len(opportunities):
        raise ValueError("Opportunity keys are not unique")
    if len({row["gap_key"] for row in gaps}) != len(gaps):
        raise ValueError("Gap keys are not unique")
    if len(opportunities) != EXPECTED_VISIBLE_OPPORTUNITIES:
        raise ValueError(
            f"Expected {EXPECTED_VISIBLE_OPPORTUNITIES} visible opportunities, "
            f"found {len(opportunities)}"
        )
    if len(gaps) != EXPECTED_EVALUABLE_GAPS:
        raise ValueError(
            f"Expected {EXPECTED_EVALUABLE_GAPS} evaluable gaps, found {len(gaps)}"
        )

    report = PackageReport(
        case_count=len(cases),
        method_count=len(METHODS),
        raw_image_count=len(raw_records),
        visible_opportunity_count=len(opportunities),
        evaluable_gap_count=len(gaps),
    )
    _write_json(
        package_dir / "input_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "study": STUDY,
            "artifacts": artifacts,
        },
    )
    _write_csv(
        package_dir / "opportunities.csv",
        OPPORTUNITY_FIELDS,
        opportunities,
    )
    _write_csv(
        package_dir / "gaps.csv",
        GAP_FIELDS,
        gaps,
    )
    _write_json(
        package_dir / "measurement_contract.json",
        measurement_contract(
            opportunity_keys_sha256=_keys_sha256(
                [str(row["opportunity_key"]) for row in opportunities]
            ),
            gap_keys_sha256=_keys_sha256(
                [str(row["gap_key"]) for row in gaps]
            ),
        ),
    )
    (package_dir / "README.md").write_text(
        README_TEMPLATE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_json(
        package_dir / "package_manifest.json",
        _package_manifest(package_dir, report),
    )
    return validate_package(workspace_root, package_dir)
