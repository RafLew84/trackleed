"""Fail-closed orchestration for explicitly local reference validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from trackleed.validation.reference_analysis import (
        ReferenceAcceptanceCriteria,
        ReferenceSeriesSpec,
    )


class ReferenceValidationError(RuntimeError):
    """A local reference input or acceptance check is invalid."""


@dataclass(frozen=True)
class ReferenceValidationConfig:
    """Explicit paths that define one local-only reference run."""

    repository_root: Path
    manifest_path: Path
    report_path: Path


@dataclass(frozen=True)
class ReferenceValidationReport:
    """Summary returned only after every reference check passes."""

    manifest_path: Path
    report_path: Path
    golden_frame_count: int
    series_count: int
    benchmark_report_path: Path


@dataclass(frozen=True)
class _ReferenceInputs:
    root: Path
    four_track_session: Path
    twenty_track_session: Path
    series: tuple[ReferenceSeriesSpec, ...]


def validate_reference_workflow(
    config: ReferenceValidationConfig,
) -> ReferenceValidationReport:
    """Validate the explicit local manifest and all declared reference inputs."""
    manifest = _read_manifest(config.manifest_path)
    approval = manifest.get("golden_approval")
    if not isinstance(approval, dict) or approval.get("status") != "manually_approved":
        raise ReferenceValidationError(
            "golden_approval.status must be 'manually_approved'"
        )
    criteria = _parse_criteria(approval)
    inputs = _reference_inputs(manifest, config.manifest_path, criteria)
    golden_frame_count = sum(len(item.golden_frame_ids) for item in inputs.series)
    golden_keys = [
        (item.session_path.resolve(), frame_id)
        for item in inputs.series
        for frame_id in item.golden_frame_ids
    ]
    if len(golden_keys) != len(set(golden_keys)):
        raise ReferenceValidationError(
            "Every approved golden selection must be a distinct session/frame pair"
        )
    if not 25 <= golden_frame_count <= 35:
        raise ReferenceValidationError(
            "Reference validation requires 25 to 35 manually approved golden frames; "
            f"found {golden_frame_count}"
        )
    if approval.get("frame_count") != golden_frame_count:
        raise ReferenceValidationError(
            "golden approval frame_count must match the declared golden frames: "
            f"{approval.get('frame_count')!r} != {golden_frame_count}"
        )
    _validate_report_path(config)
    benchmark_path = config.report_path.with_name(
        f"{config.report_path.stem}-benchmark.json"
    )
    from trackleed.analysis.current_analysis_benchmark import (
        BenchmarkCase,
        BenchmarkConfig,
        run_benchmark,
    )
    from trackleed.validation.reference_analysis import (
        ReferenceSeriesError,
        analyze_reference_series,
    )
    try:
        series_reports = [
            analyze_reference_series(item)
            for item in inputs.series
        ]
        benchmark_report = run_benchmark(
            BenchmarkConfig(
                cases=(
                    BenchmarkCase("4-track", inputs.four_track_session, 4),
                    BenchmarkCase("20-track", inputs.twenty_track_session, 20),
                ),
                output_path=benchmark_path,
            )
        )
    except (OSError, ValueError, RuntimeError, ReferenceSeriesError) as cause:
        raise ReferenceValidationError(str(cause)) from cause
    payload = _report_payload(
        config,
        inputs,
        series_reports,
        benchmark_report,
        golden_frame_count,
    )
    _write_report(config.report_path, payload)
    return ReferenceValidationReport(
        manifest_path=config.manifest_path.resolve(),
        report_path=config.report_path.resolve(),
        golden_frame_count=golden_frame_count,
        series_count=len(inputs.series),
        benchmark_report_path=benchmark_path.resolve(),
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReferenceValidationError(f"Reference manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as cause:
        raise ReferenceValidationError(f"Cannot read reference manifest: {cause}") from cause
    if not isinstance(payload, dict):
        raise ReferenceValidationError("Reference manifest root must be an object")
    if payload.get("schema_version") != 1:
        raise ReferenceValidationError("Reference manifest schema_version must be 1")
    return payload


def _reference_inputs(
    manifest: dict[str, Any],
    manifest_path: Path,
    criteria: ReferenceAcceptanceCriteria,
) -> _ReferenceInputs:
    root_value = manifest.get("reference_root")
    if not isinstance(root_value, str) or not root_value.strip():
        raise ReferenceValidationError("reference_root must be an explicit local path")
    root = (manifest_path.parent / root_value).resolve()
    if not root.is_dir():
        raise ReferenceValidationError(f"Reference root does not exist: {root}")
    benchmark = manifest.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ReferenceValidationError("benchmark must be an object")
    raw_paths: dict[str, str] = {}
    for key in ("four_track_session", "twenty_track_session"):
        value = benchmark.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ReferenceValidationError(f"benchmark.{key} is required")
        raw_paths[key] = value
    paths = {
        key: _resolve_local(root, value, f"benchmark.{key}")
        for key, value in raw_paths.items()
    }
    series = manifest.get("series")
    if not isinstance(series, list):
        raise ReferenceValidationError("series must be an array")
    if not all(isinstance(item, dict) for item in series):
        raise ReferenceValidationError("Every reference series must be an object")
    parsed_series = tuple(_parse_series(root, item, criteria) for item in series)
    names = [item.name for item in parsed_series]
    if len(names) != len(set(names)):
        raise ReferenceValidationError("Reference series names must be unique")
    return _ReferenceInputs(
        root,
        paths["four_track_session"],
        paths["twenty_track_session"],
        parsed_series,
    )


def _resolve_local(root: Path, value: str, label: str) -> Path:
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root):
        raise ReferenceValidationError(f"{label} must stay inside reference_root")
    if not candidate.is_file():
        raise ReferenceValidationError(f"{label} does not exist: {candidate}")
    return candidate


def _parse_series(
    root: Path,
    payload: dict[str, Any],
    criteria: ReferenceAcceptanceCriteria,
) -> ReferenceSeriesSpec:
    from trackleed.validation.reference_analysis import ReferenceSeriesSpec

    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ReferenceValidationError("Every reference series needs a name")
    session = payload.get("session")
    analysis_manifest = payload.get("analysis_manifest")
    if not isinstance(session, str) or not session.strip():
        raise ReferenceValidationError(f"{name}.session is required")
    if not isinstance(analysis_manifest, str) or not analysis_manifest.strip():
        raise ReferenceValidationError(f"{name}.analysis_manifest is required")
    raw_ids = payload.get("golden_frame_ids")
    if not isinstance(raw_ids, list) or not all(
        isinstance(frame_id, int) and not isinstance(frame_id, bool)
        for frame_id in raw_ids
    ):
        raise ReferenceValidationError(f"{name}.golden_frame_ids must be integers")
    frame_ids = tuple(raw_ids)
    if len(frame_ids) != len(set(frame_ids)):
        raise ReferenceValidationError(f"{name}.golden_frame_ids must be unique")
    return ReferenceSeriesSpec(
        name=name,
        session_path=_resolve_local(root, session, f"{name}.session"),
        analysis_manifest_path=_resolve_local(
            root,
            analysis_manifest,
            f"{name}.analysis_manifest",
        ),
        golden_frame_ids=frame_ids,
        criteria=criteria,
    )


def _parse_criteria(approval: dict[str, Any]) -> ReferenceAcceptanceCriteria:
    from trackleed.validation.reference_analysis import ReferenceAcceptanceCriteria

    payload = approval.get("criteria")
    if not isinstance(payload, dict):
        raise ReferenceValidationError("golden_approval.criteria must be an object")
    integer_fields = (
        "minimum_method_available_frames",
        "minimum_temporal_pairs",
    )
    float_fields = (
        "max_center_error_px",
        "max_temporal_jump_px",
        "max_spot_removal_shift_px",
        "max_ransac_seed_spread_px",
    )
    if any(
        not isinstance(payload.get(name), int)
        or isinstance(payload.get(name), bool)
        or int(payload[name]) < 1
        for name in integer_fields
    ):
        raise ReferenceValidationError(
            "Golden acceptance minimum counts must be positive integers"
        )
    try:
        thresholds = {name: float(payload[name]) for name in float_fields}
    except (KeyError, TypeError, ValueError) as cause:
        raise ReferenceValidationError(
            "Golden acceptance thresholds must be finite non-negative numbers"
        ) from cause
    if any(not isfinite(value) or value < 0.0 for value in thresholds.values()):
        raise ReferenceValidationError(
            "Golden acceptance thresholds must be finite non-negative numbers"
        )
    return ReferenceAcceptanceCriteria(
        *(int(payload[name]) for name in integer_fields),
        *(thresholds[name] for name in float_fields),
    )


def _validate_report_path(config: ReferenceValidationConfig) -> None:
    repository_root = config.repository_root.resolve()
    scratch_root = repository_root / ".scratch"
    report_path = config.report_path.resolve()
    if not report_path.is_relative_to(scratch_root):
        raise ReferenceValidationError(
            "Reference report must be an ignored local path inside .scratch"
        )
    if report_path == config.manifest_path.resolve():
        raise ReferenceValidationError("Reference report must not overwrite its manifest")


def _report_payload(
    config: ReferenceValidationConfig,
    inputs: _ReferenceInputs,
    series_reports: list[dict[str, Any]],
    benchmark_report: dict[str, Any],
    golden_frame_count: int,
) -> dict[str, Any]:
    from trackleed.analysis.current_analysis_acceptance import sha256_files

    declared_inputs = (
        config.manifest_path,
        inputs.four_track_session,
        inputs.twenty_track_session,
        *(item.session_path for item in inputs.series),
        *(item.analysis_manifest_path for item in inputs.series),
    )
    repository_root = config.repository_root.resolve()
    validation_code = (
        *repository_root.joinpath("trackleed").rglob("*.py"),
        repository_root / "scripts" / "validate_current_workflow.py",
        repository_root / "scripts" / "benchmark_current_analysis.py",
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(config.manifest_path.resolve()),
        "manifest_sha256": sha256_files((config.manifest_path,)),
        "declared_inputs_sha256": sha256_files(declared_inputs),
        "code_sha256": sha256_files(validation_code, repository_root),
        "benchmark_code_sha256": benchmark_report["code_sha256"],
        "environment": benchmark_report["environment"],
        "golden_approval": {
            "status": "manually_approved",
            "frame_count": golden_frame_count,
        },
        "series": series_reports,
        "benchmark": benchmark_report,
        "recovery_default_promoted": False,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
