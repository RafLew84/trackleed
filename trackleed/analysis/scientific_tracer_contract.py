"""Contract, authentication and input-integrity validation for the tracer."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from pathlib import Path
import platform
from typing import Any, Mapping

from trackleed.analysis.scientific_baseline_validation import validate_package
from trackleed.analysis.scientific_tracer_common import (
    BASELINE_PACKAGE,
    CASE_MEASUREMENTS_PATH,
    CASE_SUMMARY_PATH,
    MEASUREMENT_CONTRACT_PATH,
    OPPORTUNITIES_PATH,
    PROPOSAL_PATH,
    PUBLISHED_CASE_RESULTS_PATH,
    SCHEMA_VERSION,
    SchemaCompatibility,
    TRACER_AUTHENTICATION_PATH,
    TRACER_ID,
    TracerValidationError,
    read_json,
    relative_file,
    sha256,
)
from trackleed.analysis.scientific_tracer_records import (
    CANDIDATE_FIELDS,
    EVALUATION_FIELDS,
    read_raw_csv,
)


@dataclass(frozen=True)
class PreparedTracer:
    workspace_root: Path
    tracer_dir: Path
    output_dir: Path
    contract: dict[str, Any]
    runtime: dict[str, str]


def _authentication_path(workspace_root: Path) -> Path:
    return workspace_root / TRACER_AUTHENTICATION_PATH


def validate_tracer_package(
    workspace_root: Path,
    tracer_dir: Path,
    *,
    authentication_path: Path | None = None,
    schema_compatibility: SchemaCompatibility = SchemaCompatibility.EXACT,
) -> dict[str, Any]:
    """Authenticate and validate the immutable tracer package."""

    manifest_path = tracer_dir / "package_manifest.json"
    authentication = read_json(
        authentication_path or _authentication_path(workspace_root)
    )
    if (
        authentication.get("schema_version") != SCHEMA_VERSION
        or authentication.get("tracer_id") != TRACER_ID
        or authentication.get("package_manifest_sha256") != sha256(manifest_path)
    ):
        raise TracerValidationError("Frozen tracer manifest contract changed")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("tracer_id") != TRACER_ID
        or manifest.get("immutable") is not True
    ):
        raise TracerValidationError("Frozen tracer package identity changed")
    expected = manifest.get("files_sha256")
    if not isinstance(expected, dict):
        raise TracerValidationError("Frozen tracer manifest has no file hash map")
    actual = {
        path.relative_to(tracer_dir).as_posix(): path
        for path in sorted(tracer_dir.rglob("*"))
        if path.is_file()
        and path.name != "package_manifest.json"
        and "output" not in path.relative_to(tracer_dir).parts
    }
    if set(actual) != set(expected):
        raise TracerValidationError("Frozen tracer package inventory changed")
    for relative, path in actual.items():
        if sha256(path) != expected[relative]:
            raise TracerValidationError(
                f"Frozen tracer package hash mismatch: {relative}"
            )
    contract = read_json(tracer_dir / "frozen" / "tracer_contract.json")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("tracer_id") != TRACER_ID
        or contract.get("case_id") != "scan_a_pair_1"
        or contract.get("method") != "locotrack"
    ):
        raise TracerValidationError("Frozen tracer contract identity changed")
    schemas = contract.get("expected_output_schemas")
    if not isinstance(schemas, dict):
        raise TracerValidationError("Frozen tracer output schemas are missing")
    schema_contracts = {
        "candidate_records.csv": (
            "candidate_records",
            CANDIDATE_FIELDS,
            "Frozen candidate schema changed",
        ),
        "evaluation_records.csv": (
            "evaluation_records",
            EVALUATION_FIELDS,
            "Frozen evaluation schema changed",
        ),
    }
    for output_name, (expected_name, current_fields, error) in schema_contracts.items():
        declared_fields = schemas.get(output_name)
        expected_files = contract.get("expected_files")
        if not isinstance(expected_files, dict) or not isinstance(
            declared_fields, list
        ):
            raise TracerValidationError(error)
        frozen_fields, _ = read_raw_csv(
            tracer_dir / str(expected_files.get(expected_name, ""))
        )
        if frozen_fields != declared_fields:
            raise TracerValidationError(error)
        if (
            schema_compatibility.requires_current_schema
            and declared_fields != list(current_fields)
        ):
            raise TracerValidationError(error)
    _validate_contract_semantics(contract)
    return contract


def _validate_contract_semantics(contract: Mapping[str, Any]) -> None:
    count_names = {
        "branch_consistent_measurements",
        "candidate_records",
        "evaluated_gaps",
        "evaluation_records",
        "geometry_inconsistent_accepted_fits",
        "jointly_consistent_pair_frames",
        "recovered_gaps",
    }
    counts = contract.get("expected_counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != count_names
        or any(not isinstance(value, int) or value < 0 for value in counts.values())
    ):
        raise TracerValidationError("Frozen tracer expected counts are invalid")
    comparison = contract.get("comparison_tolerances")
    continuous = (
        comparison.get("continuous_values")
        if isinstance(comparison, dict)
        else None
    )
    if (
        not isinstance(continuous, dict)
        or set(continuous) != {"atol", "rtol"}
        or any(float(value) < 0 for value in continuous.values())
        or comparison.get("discrete_values") != "exact"
        or comparison.get("reader_facing_values")
        != "exact_after_declared_rounding"
    ):
        raise TracerValidationError(
            "Frozen tracer comparison tolerances are invalid"
        )
    compatibility = contract.get("estimator_compatibility")
    theta_bounds = (
        compatibility.get("optimizer_theta_bounds_rad")
        if isinstance(compatibility, dict)
        else None
    )
    if (
        not isinstance(theta_bounds, list)
        or len(theta_bounds) != 2
        or not float(theta_bounds[0]) < float(theta_bounds[1])
        or not isinstance(compatibility.get("image_loading"), str)
        or not isinstance(compatibility.get("mode"), str)
    ):
        raise TracerValidationError(
            "Frozen tracer estimator compatibility is invalid"
        )
    rounding = contract.get("reader_facing_rounding")
    if (
        not isinstance(rounding, dict)
        or rounding.get("counts") != "exact"
        or not isinstance(rounding.get("pair_distance_residual_px"), dict)
        or not isinstance(rounding.get("rates_percent"), dict)
        or not isinstance(
            rounding["pair_distance_residual_px"].get("decimal_places"), int
        )
        or not isinstance(rounding["rates_percent"].get("decimal_places"), int)
    ):
        raise TracerValidationError(
            "Frozen tracer reader-facing rounding is invalid"
        )
    execution = contract.get("execution_contract")
    runtime = execution.get("runtime") if isinstance(execution, dict) else None
    if (
        not isinstance(runtime, dict)
        or execution.get("device") != "cpu"
        or set(runtime) != {"numpy", "python", "scikit-image", "scipy", "tifffile"}
        or any(not isinstance(value, str) or not value for value in runtime.values())
    ):
        raise TracerValidationError("Frozen tracer execution contract is invalid")
    expected_paths = {
        "case_measurements": CASE_MEASUREMENTS_PATH.as_posix(),
        "case_summary": CASE_SUMMARY_PATH.as_posix(),
        "frame_images": "frozen/frame_images.csv",
        "gaps": "manuscript/scientific_baseline/gaps.csv",
        "measurement_contract": MEASUREMENT_CONTRACT_PATH.as_posix(),
        "opportunities": OPPORTUNITIES_PATH.as_posix(),
        "proposals": PROPOSAL_PATH.as_posix(),
        "published_case_results": PUBLISHED_CASE_RESULTS_PATH.as_posix(),
    }
    inputs = contract.get("frozen_inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(expected_paths):
        raise TracerValidationError("Frozen tracer input set changed")
    for name, path in expected_paths.items():
        if not isinstance(inputs[name], dict) or inputs[name].get("path") != path:
            raise TracerValidationError(f"Frozen tracer input path changed: {name}")


def runtime_fingerprint() -> dict[str, str]:
    return {
        "numpy": distribution_version("numpy"),
        "python": platform.python_version(),
        "scikit-image": distribution_version("scikit-image"),
        "scipy": distribution_version("scipy"),
        "tifffile": distribution_version("tifffile"),
    }


def validate_runtime(contract: Mapping[str, Any]) -> dict[str, str]:
    runtime = runtime_fingerprint()
    expected = contract["execution_contract"]["runtime"]
    if runtime != expected:
        raise TracerValidationError(
            f"Tracer runtime differs from frozen CPU environment: {runtime!r}"
        )
    return runtime


def validate_input_hashes(
    workspace_root: Path,
    tracer_dir: Path,
    contract: Mapping[str, Any],
) -> None:
    inputs = contract.get("frozen_inputs")
    if not isinstance(inputs, dict):
        raise TracerValidationError("Frozen tracer inputs are missing")
    for name, record in inputs.items():
        if not isinstance(record, dict):
            raise TracerValidationError(f"Invalid frozen tracer input: {name}")
        path_text = str(record.get("path", ""))
        base = tracer_dir if path_text.startswith("frozen/") else workspace_root
        path = relative_file(base, path_text)
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise TracerValidationError(
                f"Frozen tracer input hash mismatch: {name}"
            )


def prepare_tracer(
    workspace_root: Path,
    tracer_dir: Path,
    output_dir: Path,
    *,
    require_absent_output: bool = False,
    authentication_path: Path | None = None,
    schema_compatibility: SchemaCompatibility = SchemaCompatibility.EXACT,
) -> PreparedTracer:
    """Resolve paths and enforce all pre-reconstruction contracts."""

    workspace_root = workspace_root.resolve()
    tracer_dir = tracer_dir.resolve()
    output_dir = output_dir.resolve()
    if require_absent_output and output_dir.exists():
        raise FileExistsError(f"Refusing to replace tracer output: {output_dir}")
    validate_package(workspace_root, workspace_root / BASELINE_PACKAGE)
    contract = validate_tracer_package(
        workspace_root,
        tracer_dir,
        authentication_path=authentication_path,
        schema_compatibility=schema_compatibility,
    )
    runtime = validate_runtime(contract)
    validate_input_hashes(workspace_root, tracer_dir, contract)
    return PreparedTracer(
        workspace_root=workspace_root,
        tracer_dir=tracer_dir,
        output_dir=output_dir,
        contract=contract,
        runtime=runtime,
    )


__all__ = [
    "PreparedTracer",
    "prepare_tracer",
    "runtime_fingerprint",
    "validate_input_hashes",
    "validate_runtime",
    "validate_tracer_package",
]
