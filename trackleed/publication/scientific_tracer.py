"""Publication-owned orchestration for the frozen scientific tracer."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from trackleed.analysis.scientific_tracer_common import (
    SCHEMA_VERSION,
    SchemaCompatibility,
    TRACER_ID,
    TracerReport,
    TracerValidationError,
    implementation_sources,
    sha256,
    write_json,
)
from trackleed.analysis.scientific_tracer_contract import prepare_tracer
from trackleed.analysis.scientific_tracer_evaluation import evaluate_candidates
from trackleed.analysis.scientific_tracer_reconstruction import (
    reader_facing,
    reconstruct_tracer_artifacts,
)
from trackleed.analysis.scientific_tracer_records import (
    CANDIDATE_FIELDS,
    EVALUATION_FIELDS,
    CandidateRecord,
    EvaluationRecord,
    GapRecord,
    PairRecord,
    write_csv,
)
from trackleed.analysis.scientific_tracer_validation import (
    VALIDATION_CHECKS,
    output_manifest,
    tracer_report,
    validate_central_result,
    validate_complete_output,
)


def validate_tracer_output(
    workspace_root: Path,
    tracer_dir: Path,
    output_dir: Path,
) -> TracerReport:
    """Validate existing tracer artifacts without rewriting any file."""

    prepared = prepare_tracer(workspace_root, tracer_dir, output_dir)
    return validate_complete_output(
        prepared.workspace_root,
        prepared.tracer_dir,
        prepared.output_dir,
        prepared.contract,
        prepared.runtime,
    )


def _run_tracer(
    workspace_root: Path,
    tracer_dir: Path,
    output_dir: Path,
    *,
    authentication_path: Path | None = None,
    schema_compatibility: SchemaCompatibility = SchemaCompatibility.EXACT,
) -> TracerReport:
    prepared = prepare_tracer(
        workspace_root,
        tracer_dir,
        output_dir,
        require_absent_output=True,
        authentication_path=authentication_path,
        schema_compatibility=schema_compatibility,
    )
    artifacts = reconstruct_tracer_artifacts(
        prepared.workspace_root,
        prepared.tracer_dir,
        prepared.contract,
    )
    validate_central_result(
        prepared.tracer_dir,
        prepared.contract,
        artifacts,
        prepared.workspace_root,
        schema_compatibility=schema_compatibility,
    )

    prepared.output_dir.mkdir(parents=True)
    write_csv(
        prepared.output_dir / "candidate_records.csv",
        CandidateRecord,
        artifacts.candidates,
    )
    write_csv(
        prepared.output_dir / "evaluation_records.csv",
        EvaluationRecord,
        artifacts.evaluations,
    )
    write_csv(
        prepared.output_dir / "pair_records.csv",
        PairRecord,
        artifacts.pairs,
    )
    write_csv(
        prepared.output_dir / "gap_records.csv",
        GapRecord,
        artifacts.gaps,
    )
    write_json(
        prepared.output_dir / "summary.json",
        {
            "central_result": artifacts.summary,
            "reader_facing": reader_facing(
                artifacts.summary,
                prepared.contract["reader_facing_rounding"],
            ),
        },
    )
    write_json(
        prepared.output_dir / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "tracer_id": TRACER_ID,
            "execution_device": "cpu",
            "runtime": prepared.runtime,
            "inputs": prepared.contract["frozen_inputs"],
            "estimator_compatibility": prepared.contract[
                "estimator_compatibility"
            ],
            "implementation_sources": implementation_sources(
                prepared.workspace_root
            ),
        },
    )
    write_json(
        prepared.output_dir / "output_manifest.json",
        output_manifest(prepared.output_dir),
    )
    report = tracer_report(artifacts)
    write_json(
        prepared.output_dir / "validation_report.json",
        {
            **asdict(report),
            "checks": VALIDATION_CHECKS,
            "fail_closed": True,
            "output_manifest_sha256": sha256(
                prepared.output_dir / "output_manifest.json"
            ),
            "schema_version": SCHEMA_VERSION,
            "tracer_id": TRACER_ID,
        },
    )
    return validate_complete_output(
        prepared.workspace_root,
        prepared.tracer_dir,
        prepared.output_dir,
        prepared.contract,
        prepared.runtime,
        schema_compatibility=schema_compatibility,
    )


def run_tracer(
    workspace_root: Path,
    tracer_dir: Path,
    output_dir: Path,
) -> TracerReport:
    """Reconstruct and validate the frozen tracer without tracker inference."""

    return _run_tracer(workspace_root, tracer_dir, output_dir)


def stage_tracer_refreeze(
    workspace_root: Path,
    tracer_dir: Path,
    output_dir: Path,
    authentication_path: Path,
) -> TracerReport:
    """Build current-schema output from an authenticated older tracer schema."""

    return _run_tracer(
        workspace_root,
        tracer_dir,
        output_dir,
        authentication_path=authentication_path,
        schema_compatibility=SchemaCompatibility.RETAINED_FIELDS,
    )


__all__ = [
    "CANDIDATE_FIELDS",
    "EVALUATION_FIELDS",
    "CandidateRecord",
    "EvaluationRecord",
    "TracerReport",
    "TracerValidationError",
    "evaluate_candidates",
    "run_tracer",
    "stage_tracer_refreeze",
    "validate_tracer_output",
]
