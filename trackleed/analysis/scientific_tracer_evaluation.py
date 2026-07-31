"""Threshold-only classification of stored scientific tracer candidates."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

from trackleed.analysis.scientific_tracer_records import (
    MARGIN_FIELDS,
    EvaluationRecord,
    MeasurementStatus,
    RejectionCause,
)


def signed_gate_margins(
    candidate: Mapping[str, object],
    measurement_contract: Mapping[str, Any],
) -> dict[str, float | None]:
    margins: dict[str, float | None] = {field: None for field in MARGIN_FIELDS}
    if not bool(candidate["finite_candidate"]):
        return margins
    proposal_max = float(measurement_contract["proposal_to_fit_max_px"])
    r_squared_min = float(measurement_contract["r_squared_min"])
    amplitude_min = float(
        measurement_contract["amplitude_reference_min"]
    ) * float(measurement_contract["amplitude_gate_multiplier"])
    sigma_min = float(measurement_contract["width_gate"]["sigma_min_px"])
    sigma_max = float(measurement_contract["width_gate"]["sigma_max_px"])
    branch_tolerance = float(measurement_contract["branch_tolerance_px"])
    margins.update(
        {
            "proposal_to_fit_margin_px": proposal_max
            - float(candidate["proposal_offset_px"]),
            "r_squared_margin": float(candidate["r_squared"]) - r_squared_min,
            "amplitude_margin": float(candidate["amplitude"]) - amplitude_min,
            "sigma_x_lower_margin_px": float(candidate["sigma_x"]) - sigma_min,
            "sigma_x_upper_margin_px": sigma_max - float(candidate["sigma_x"]),
            "sigma_y_lower_margin_px": float(candidate["sigma_y"]) - sigma_min,
            "sigma_y_upper_margin_px": sigma_max - float(candidate["sigma_y"]),
            "branch_margin_px": branch_tolerance
            - float(candidate["branch_error_px"]),
        }
    )
    return margins


def evaluate_candidates(
    candidates: Sequence[Mapping[str, object]],
    measurement_contract: Mapping[str, Any],
    *,
    variant_id: str = "roi_48_proposal_15",
) -> list[EvaluationRecord]:
    """Classify stored candidate fits without rerunning or copying fit values."""

    rows: list[EvaluationRecord] = []
    for candidate in candidates:
        margins = signed_gate_margins(candidate, measurement_contract)
        finite_candidate = bool(candidate["finite_candidate"])
        proposal_to_fit_failed = finite_candidate and (
            float(margins["proposal_to_fit_margin_px"]) < 0
        )
        r_squared_failed = finite_candidate and (
            float(margins["r_squared_margin"]) < 0
        )
        amplitude_failed = finite_candidate and (
            float(margins["amplitude_margin"]) < 0
        )
        width_failed = finite_candidate and any(
            float(margins[name]) < 0
            for name in (
                "sigma_x_lower_margin_px",
                "sigma_x_upper_margin_px",
                "sigma_y_lower_margin_px",
                "sigma_y_upper_margin_px",
            )
        )
        branch_failed = finite_candidate and (
            float(margins["branch_margin_px"]) < 0
        )
        causes: list[RejectionCause] = []
        if not bool(candidate["proposal_available"]):
            causes.append("proposal_unavailable")
        elif not bool(candidate["roi_valid"]):
            causes.append("roi_invalid")
        elif candidate["solver_status"] != "completed":
            causes.append("solver_failure")
        elif not finite_candidate:
            causes.append("non_finite_parameters")
        else:
            if proposal_to_fit_failed:
                causes.append("proposal_to_fit")
            if r_squared_failed:
                causes.append("r_squared")
            if amplitude_failed:
                causes.append("amplitude")
            if width_failed:
                causes.append("width")
            if branch_failed:
                causes.append("physics_branch_consistency")

        gaussian_causes = [
            cause for cause in causes if cause != "physics_branch_consistency"
        ]
        accepted_fit = not gaussian_causes
        if gaussian_causes:
            if gaussian_causes[0] in {
                "proposal_unavailable",
                "roi_invalid",
                "solver_failure",
                "non_finite_parameters",
            }:
                measurement_status: MeasurementStatus = gaussian_causes[0]
            else:
                measurement_status = "gaussian_rejected"
        elif "physics_branch_consistency" in causes:
            measurement_status = "geometry_inconsistent"
        else:
            measurement_status = "branch_consistent"

        legacy_reasons: list[str] = []
        legacy_missing_reasons = {
            "proposal_unavailable": "",
            "roi_invalid": "roi_outside_image",
            "solver_failure": "gaussian_fit_failed",
            "non_finite_parameters": "non_finite_parameters",
        }
        missing_cause = causes[0] if causes else None
        if missing_cause in legacy_missing_reasons:
            legacy_fit_reason = legacy_missing_reasons[missing_cause]
        elif finite_candidate:
            if r_squared_failed:
                legacy_reasons.append("low_r2")
            if amplitude_failed:
                legacy_reasons.append("low_amplitude")
            if width_failed:
                legacy_reasons.append("invalid_sigma")
            if proposal_to_fit_failed:
                legacy_reasons.append("center_outside_convergence_zone")
            legacy_fit_reason = (
                "accepted" if not legacy_reasons else "+".join(legacy_reasons)
            )
        else:
            legacy_fit_reason = "non_finite_parameters"
        rows.append(
            EvaluationRecord(
                record_key=str(candidate["record_key"]),
                case_id=str(candidate["case_id"]),
                method=str(candidate["method"]),
                frame=int(candidate["frame"]),
                energy_ev=float(candidate["energy_ev"]),
                branch=str(candidate["branch"]),
                physics_baseline_status=str(
                    candidate["physics_baseline_status"]
                ),
                opportunity_key=cast(str | None, candidate["opportunity_key"]),
                variant_id=variant_id,
                accepted_fit=accepted_fit,
                measurement_valid=accepted_fit,
                measurement_status=measurement_status,
                classification=measurement_status,
                first_rejection_cause=causes[0] if causes else None,
                full_rejection_causes="+".join(causes),
                legacy_fit_reason=legacy_fit_reason,
                **margins,
            )
        )
    return rows


__all__ = ["evaluate_candidates", "signed_gate_margins"]
