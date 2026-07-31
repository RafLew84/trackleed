"""Publication-owned interface for the complete Ag(111) central panel."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from trackleed.analysis.scientific_panel_common import (
    PANEL_ID,
    SCHEMA_VERSION,
    PanelReport,
    TracerValidationError,
    implementation_sources,
    sha256,
    write_json,
)
from trackleed.analysis.scientific_panel_contract import prepare_panel
from trackleed.analysis.scientific_panel_reconstruction import (
    reader_facing_panel,
    reconstruct_panel_artifacts,
)
from trackleed.analysis.scientific_panel_records import (
    AggregateReportRecord,
    CaseReportRecord,
    PanelGapRecord,
    PanelPairRecord,
)
from trackleed.analysis.scientific_panel_validation import (
    VALIDATION_CHECKS,
    panel_output_manifest,
    panel_report,
    validate_complete_panel_output,
    validate_panel_result,
)
from trackleed.analysis.scientific_tracer_records import (
    CandidateRecord,
    EvaluationRecord,
    write_csv,
)


def validate_central_panel_output(
    workspace_root: Path,
    panel_dir: Path,
    output_dir: Path,
) -> PanelReport:
    """Validate existing panel artifacts without rewriting any file."""

    prepared = prepare_panel(workspace_root, panel_dir, output_dir)
    return validate_complete_panel_output(
        prepared.workspace_root,
        prepared.panel_dir,
        prepared.output_dir,
        prepared.contract,
        prepared.runtime,
    )


def run_central_panel(
    workspace_root: Path,
    panel_dir: Path,
    output_dir: Path,
) -> PanelReport:
    """Reconstruct and validate all five methods without tracker inference."""

    prepared = prepare_panel(
        workspace_root,
        panel_dir,
        output_dir,
        require_absent_output=True,
    )
    artifacts = reconstruct_panel_artifacts(
        prepared.workspace_root, prepared.contract
    )
    validate_panel_result(
        prepared.workspace_root,
        prepared.panel_dir,
        prepared.contract,
        artifacts,
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
        PanelPairRecord,
        artifacts.pairs,
    )
    write_csv(
        prepared.output_dir / "gap_records.csv",
        PanelGapRecord,
        artifacts.gaps,
    )
    write_csv(
        prepared.output_dir / "case_results.csv",
        CaseReportRecord,
        artifacts.case_reports,
    )
    write_csv(
        prepared.output_dir / "aggregate_method_results.csv",
        AggregateReportRecord,
        artifacts.aggregate_reports,
    )
    write_json(
        prepared.output_dir / "summary.json",
        {
            "central_result": artifacts.summary,
            "reader_facing": reader_facing_panel(
                artifacts.summary,
                prepared.contract["reader_facing_rounding"],
            ),
        },
    )
    write_json(
        prepared.output_dir / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "tracer_id": PANEL_ID,
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
        panel_output_manifest(prepared.output_dir),
    )
    report = panel_report(artifacts)
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
            "tracer_id": PANEL_ID,
        },
    )
    return validate_complete_panel_output(
        prepared.workspace_root,
        prepared.panel_dir,
        prepared.output_dir,
        prepared.contract,
        prepared.runtime,
    )


__all__ = [
    "PanelReport",
    "TracerValidationError",
    "run_central_panel",
    "validate_central_panel_output",
]
