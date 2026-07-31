"""Measurement-window and capture-contract sensitivity for Ag(111)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trackleed.analysis.scientific_panel_candidates import (
    reconstruct_panel_candidates,
)
from trackleed.analysis.scientific_panel_common import (
    GAPS_PATH,
    MEASUREMENT_CONTRACT_PATH,
    TracerValidationError,
    read_json,
)
from trackleed.analysis.scientific_panel_contract import prepare_panel
from trackleed.analysis.scientific_sensitivity_audit import (
    WidthTriggerAudit,
    audit_width_trigger,
)
from trackleed.analysis.scientific_sensitivity_reporting import (
    summarize_variant,
)
from trackleed.analysis.scientific_tracer_evaluation import evaluate_candidates
from trackleed.analysis.scientific_tracer_records import (
    CandidateRecord,
    EvaluationRecord,
    read_csv,
    read_raw_csv,
)


@dataclass(frozen=True)
class SensitivityArtifacts:
    candidates: list[CandidateRecord]
    evaluations: list[EvaluationRecord]
    variant_summaries: dict[str, dict[str, Any]]
    width_evaluations: list[EvaluationRecord]
    width_summaries: dict[str, dict[str, Any]]
    width_trigger: WidthTriggerAudit
    central_variant_matches_frozen_record: bool


def _complete_variants(
    measurement_contract: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    variants = measurement_contract.get("sensitivity_variants")
    central = measurement_contract.get("central_contract")
    if not isinstance(variants, list) or len(variants) != 15:
        raise TracerValidationError("Sensitivity variant contract changed")
    if not isinstance(central, dict):
        raise TracerValidationError("Central measurement contract changed")
    complete: list[dict[str, Any]] = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise TracerValidationError("Sensitivity variant is invalid")
        complete.append({**central, **variant})
    return central, complete


def _candidate_sets(
    workspace_root: Path,
    panel_contract: dict[str, Any],
    central_candidates: list[CandidateRecord],
    variants: list[dict[str, Any]],
) -> dict[int, list[CandidateRecord]]:
    variant_by_roi = {
        int(variant["roi_side_px"]): variant
        for variant in variants
        if variant["family"] == "roi_x_proposal_to_fit"
        and float(variant["proposal_to_fit_max_px"]) == 15.0
    }
    if set(variant_by_roi) != {32, 48, 64}:
        raise TracerValidationError("Sensitivity ROI contract changed")
    result = {48: central_candidates}
    for roi_side in (32, 64):
        result[roi_side] = reconstruct_panel_candidates(
            workspace_root,
            panel_contract,
            measurement_contract=variant_by_roi[roi_side],
        )
    return result


def _width_variant(
    central: dict[str, Any],
    measurement_contract: dict[str, Any],
    width_value: dict[str, Any],
) -> dict[str, Any]:
    return {
        **central,
        **width_value,
        "variant_id": str(width_value["contract_id"]),
        "family": "conditional_width_sensitivity",
        "is_central": False,
        "width_gate": {
            **central["width_gate"],
            **{key: width_value[key] for key in ("sigma_min_px", "sigma_max_px")},
        },
        "opportunity_set_id": measurement_contract["reference_sets"]["opportunities"][
            "set_id"
        ],
        "gap_set_id": measurement_contract["reference_sets"]["gaps"]["set_id"],
        "amplitude_threshold_min": (
            float(central["amplitude_reference_min"])
            * float(central["amplitude_gate_multiplier"])
        ),
    }


def _run_width_sensitivity(
    workspace_root: Path,
    measurement_contract: dict[str, Any],
    central: dict[str, Any],
    central_candidates: list[CandidateRecord],
    central_summary: dict[str, Any],
    frozen_gaps: list[dict[str, str]],
) -> tuple[list[EvaluationRecord], dict[str, dict[str, Any]]]:
    evaluations: list[EvaluationRecord] = []
    summaries: dict[str, dict[str, Any]] = {}
    width_contract = measurement_contract["conditional_width_sensitivity"]
    for width_value in width_contract["contracts"]:
        variant = _width_variant(central, measurement_contract, width_value)
        width_id = str(variant["variant_id"])
        if width_id == "width_central":
            summary = deepcopy(central_summary)
            summary["contract"] = {**summary["contract"], **variant}
            summaries[width_id] = summary
            continue
        width_evaluations = evaluate_candidates(
            central_candidates,
            variant,
            variant_id=width_id,
        )
        summaries[width_id] = summarize_variant(
            workspace_root,
            variant,
            central_candidates,
            width_evaluations,
            frozen_gaps,
        )
        evaluations.extend(width_evaluations)
    return evaluations, summaries


def reconstruct_sensitivity(
    workspace_root: Path,
    panel_dir: Path,
) -> SensitivityArtifacts:
    """Run all predeclared sensitivity variants from authenticated inputs."""

    prepared = prepare_panel(workspace_root, panel_dir, panel_dir / "output")
    measurement_contract = read_json(
        prepared.workspace_root / MEASUREMENT_CONTRACT_PATH
    )
    central_contract, variants = _complete_variants(measurement_contract)
    expected = prepared.contract["expected_files"]
    _, central_candidates = read_csv(
        prepared.panel_dir / expected["candidate_records"], CandidateRecord
    )
    _, central_evaluations = read_csv(
        prepared.panel_dir / expected["evaluation_records"], EvaluationRecord
    )
    candidate_sets = _candidate_sets(
        prepared.workspace_root,
        prepared.contract,
        central_candidates,
        variants,
    )
    _, frozen_gaps = read_raw_csv(prepared.workspace_root / GAPS_PATH)

    all_evaluations: list[EvaluationRecord] = []
    summaries: dict[str, dict[str, Any]] = {}
    central_matches = False
    for variant in variants:
        variant_id = str(variant["variant_id"])
        candidates = candidate_sets[int(variant["roi_side_px"])]
        evaluations = evaluate_candidates(candidates, variant, variant_id=variant_id)
        if bool(variant["is_central"]):
            central_matches = evaluations == central_evaluations
        summaries[variant_id] = summarize_variant(
            prepared.workspace_root,
            variant,
            candidates,
            evaluations,
            frozen_gaps,
        )
        all_evaluations.extend(evaluations)
    if not central_matches:
        raise TracerValidationError(
            "Central sensitivity variant differs from frozen panel"
        )

    width_trigger = audit_width_trigger(
        central_candidates,
        central_evaluations,
        measurement_contract,
    )
    width_evaluations: list[EvaluationRecord] = []
    width_summaries: dict[str, dict[str, Any]] = {}
    if width_trigger.run_width_sensitivity:
        width_evaluations, width_summaries = _run_width_sensitivity(
            prepared.workspace_root,
            measurement_contract,
            central_contract,
            central_candidates,
            summaries["roi_48_proposal_15"],
            frozen_gaps,
        )
    return SensitivityArtifacts(
        candidates=[
            row for roi_side in (32, 48, 64) for row in candidate_sets[roi_side]
        ],
        evaluations=all_evaluations,
        variant_summaries=summaries,
        width_evaluations=width_evaluations,
        width_summaries=width_summaries,
        width_trigger=width_trigger,
        central_variant_matches_frozen_record=central_matches,
    )


__all__ = [
    "SensitivityArtifacts",
    "WidthTriggerAudit",
    "audit_width_trigger",
    "reconstruct_sensitivity",
]
