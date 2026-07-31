"""Publication-owned interface for the Ag(111) sensitivity assessment."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from trackleed.analysis.scientific_panel_common import (
    SCHEMA_VERSION,
    sha256,
    write_json,
)
from trackleed.analysis.scientific_sensitivity import (
    SensitivityArtifacts,
    reconstruct_sensitivity,
)
from trackleed.analysis.scientific_sensitivity_common import (
    SENSITIVITY_ID,
    SensitivityReport,
    implementation_sources,
    sensitivity_output_manifest,
)
from trackleed.analysis.scientific_sensitivity_records import (
    RejectionCauseCountRecord,
    SensitivityResultRecord,
    WidthTriggerRecord,
)
from trackleed.analysis.scientific_sensitivity_reporting import (
    rejection_cause_count_records,
    sensitivity_result_records,
)
from trackleed.analysis.scientific_sensitivity_validation import (
    validate_sensitivity_artifacts,
    validate_sensitivity_output,
)
from trackleed.analysis.scientific_tracer_contract import runtime_fingerprint
from trackleed.analysis.scientific_tracer_records import (
    CandidateRecord,
    EvaluationRecord,
    write_csv,
)


def _report(artifacts: SensitivityArtifacts) -> SensitivityReport:
    return SensitivityReport(
        variant_count=len(artifacts.variant_summaries),
        unique_roi_candidate_set_count=len(
            {row["roi_side_px"] for row in artifacts.candidates}
        ),
        candidate_count=len(artifacts.candidates),
        evaluation_count=len(artifacts.evaluations),
        width_sensitivity_triggered=(artifacts.width_trigger.run_width_sensitivity),
        width_contract_count=len(artifacts.width_summaries),
        central_variant_matches_frozen_record=(
            artifacts.central_variant_matches_frozen_record
        ),
    )


def _summary_document(artifacts: SensitivityArtifacts) -> dict[str, Any]:
    audit = artifacts.width_trigger
    return {
        "schema_version": SCHEMA_VERSION,
        "sensitivity_id": SENSITIVITY_ID,
        "denominator_policy": (
            "fixed reference opportunity sets with explicit variant-specific "
            "denominators for conditional outcomes"
        ),
        "variants": artifacts.variant_summaries,
        "conditional_width_sensitivity": {
            "decision": (
                "run_all_width_contracts"
                if audit.run_width_sensitivity
                else "not_run_no_trigger"
            ),
            "triggered_methods": list(audit.triggered_methods),
            "method_audit": list(audit.method_records),
            "case_diagnostics": list(audit.case_records),
            "width_contracts": artifacts.width_summaries,
            "central_contract_changed": False,
            "optimizer_hard_bounds_changed": False,
            "automatic_configuration_selection": False,
        },
    }


def run_sensitivity(
    workspace_root: Path,
    panel_dir: Path,
    output_dir: Path,
) -> SensitivityReport:
    """Run and validate the complete predeclared Ag(111) sensitivity."""

    workspace_root = workspace_root.resolve()
    panel_dir = panel_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace sensitivity output: {output_dir}")
    artifacts = reconstruct_sensitivity(workspace_root, panel_dir)
    validate_sensitivity_artifacts(artifacts)
    report = _report(artifacts)
    output_dir.mkdir(parents=True)
    write_csv(
        output_dir / "candidate_records.csv",
        CandidateRecord,
        artifacts.candidates,
    )
    write_csv(
        output_dir / "evaluation_records.csv",
        EvaluationRecord,
        artifacts.evaluations,
    )
    write_csv(
        output_dir / "sensitivity_results.csv",
        SensitivityResultRecord,
        sensitivity_result_records(
            artifacts.variant_summaries,
            artifacts.width_summaries,
        ),
    )
    write_csv(
        output_dir / "rejection_cause_counts.csv",
        RejectionCauseCountRecord,
        rejection_cause_count_records(
            artifacts.variant_summaries,
            artifacts.width_summaries,
        ),
    )
    write_csv(
        output_dir / "gate_margin_audit.csv",
        WidthTriggerRecord,
        [
            *artifacts.width_trigger.method_records,
            *artifacts.width_trigger.case_records,
        ],
    )
    write_json(output_dir / "summary.json", _summary_document(artifacts))
    write_json(
        output_dir / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "sensitivity_id": SENSITIVITY_ID,
            "execution_device": "cpu",
            "runtime": runtime_fingerprint(),
            "baseline_package_manifest_sha256": sha256(
                workspace_root
                / "manuscript"
                / "scientific_baseline"
                / "package_manifest.json"
            ),
            "central_panel_package_manifest_sha256": sha256(
                panel_dir / "package_manifest.json"
            ),
            "implementation_sources": implementation_sources(workspace_root),
            "tracker_inference_run": False,
            "central_contract_changed": False,
            "automatic_configuration_selection": False,
        },
    )
    write_json(
        output_dir / "output_manifest.json",
        sensitivity_output_manifest(output_dir),
    )
    write_json(
        output_dir / "validation_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "sensitivity_id": SENSITIVITY_ID,
            "report": asdict(report),
            "checks": {
                "authenticated_inputs": "passed",
                "central_variant_record_match": "passed",
                "complete_predeclared_variants": "passed",
                "fixed_and_conditional_denominators": "passed",
                "gate_margin_audit": "passed",
                "width_trigger_policy": "passed",
                "no_automatic_configuration_selection": "passed",
            },
            "fail_closed": True,
            "output_manifest_sha256": sha256(output_dir / "output_manifest.json"),
        },
    )
    return validate_sensitivity_output(workspace_root, panel_dir, output_dir)


__all__ = [
    "SensitivityReport",
    "run_sensitivity",
    "validate_sensitivity_output",
]
