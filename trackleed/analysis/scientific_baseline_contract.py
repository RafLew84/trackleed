"""Canonical measurement contract for the Ag(111) publication baseline."""

from __future__ import annotations


SCHEMA_VERSION = 1
STUDY = "ag111_principal_stress_test"
METHODS = ("easyleed", "locotrack", "ostrack", "dam4sam", "lk")
EXPECTED_CASE_COUNT = 4
EXPECTED_VISIBLE_OPPORTUNITIES = 221
EXPECTED_EVALUABLE_GAPS = 24
EXPECTED_CASE_MATRIX = frozenset(
    {
        ("scan_a_pair_1", "scan_a", "pair_1"),
        ("scan_a_pair_2", "scan_a", "pair_2"),
        ("scan_b_pair_1", "scan_b", "pair_1"),
        ("scan_b_pair_2", "scan_b", "pair_2"),
    }
)


def measurement_contract(
    *,
    opportunity_keys_sha256: str,
    gap_keys_sha256: str,
) -> dict[str, object]:
    """Return the canonical contract and its mandatory sensitivity variants."""

    central = {
        "roi_side_px": 48,
        "proposal_to_fit_max_px": 15.0,
        "branch_tolerance_px": 18.0,
        "r_squared_min": 0.85,
        "amplitude_reference_min": 0.012,
        "amplitude_gate_multiplier": 1.0,
        "center_preference_factor": 0.34,
        "width_gate": {
            "sigma_min_px": 0.6,
            "sigma_max_px": 13.0,
            "minimum_inclusive": True,
            "maximum_inclusive": True,
            "role": "post_fit_publication_acceptance",
        },
        "optimizer_hard_bounds": {
            "sigma_min_px": 0.5,
            "sigma_max": "roi_axis_length_px",
            "role": "numerical_solver_constraint",
            "varied_by_width_sensitivity": False,
        },
    }

    def variant(
        variant_id: str,
        family: str,
        *,
        roi_side_px: int = 48,
        proposal_to_fit_max_px: float = 15.0,
        branch_tolerance_px: float = 18.0,
        r_squared_min: float = 0.85,
        amplitude_gate_multiplier: float = 1.0,
    ) -> dict[str, object]:
        center_radius = roi_side_px * 0.34
        return {
            "variant_id": variant_id,
            "family": family,
            "is_central": (
                roi_side_px == 48
                and proposal_to_fit_max_px == 15.0
                and branch_tolerance_px == 18.0
                and r_squared_min == 0.85
                and amplitude_gate_multiplier == 1.0
            ),
            "roi_side_px": roi_side_px,
            "proposal_to_fit_max_px": proposal_to_fit_max_px,
            "branch_tolerance_px": branch_tolerance_px,
            "r_squared_min": r_squared_min,
            "amplitude_gate_multiplier": amplitude_gate_multiplier,
            "amplitude_threshold_min": 0.012 * amplitude_gate_multiplier,
            "center_preference_factor": 0.34,
            "initial_search_radius_px": center_radius,
            "center_bound_radius_px": center_radius,
            "width_gate": central["width_gate"],
            "optimizer_hard_bounds": central["optimizer_hard_bounds"],
            "opportunity_set_id": "baseline_visible_opportunities_v1",
            "gap_set_id": "evaluable_iv_gaps_v1",
        }

    variants = [
        variant(
            f"roi_{roi}_proposal_{int(proposal)}",
            "roi_x_proposal_to_fit",
            roi_side_px=roi,
            proposal_to_fit_max_px=proposal,
        )
        for roi in (32, 48, 64)
        for proposal in (12.0, 15.0, 18.0)
    ]
    variants.extend(
        (
            variant(
                "branch_tolerance_12",
                "branch_tolerance_ofat",
                branch_tolerance_px=12.0,
            ),
            variant(
                "branch_tolerance_24",
                "branch_tolerance_ofat",
                branch_tolerance_px=24.0,
            ),
            variant(
                "r_squared_0p80",
                "r_squared_ofat",
                r_squared_min=0.8,
            ),
            variant(
                "r_squared_0p90",
                "r_squared_ofat",
                r_squared_min=0.9,
            ),
            variant(
                "amplitude_multiplier_0p75",
                "amplitude_ofat",
                amplitude_gate_multiplier=0.75,
            ),
            variant(
                "amplitude_multiplier_1p25",
                "amplitude_ofat",
                amplitude_gate_multiplier=1.25,
            ),
        )
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY,
        "central_contract": central,
        "sensitivity_variants": variants,
        "reference_sets": {
            "opportunities": {
                "set_id": "baseline_visible_opportunities_v1",
                "manifest": "opportunities.csv",
                "key_field": "opportunity_key",
                "key_count": EXPECTED_VISIBLE_OPPORTUNITIES,
                "keys_sha256": opportunity_keys_sha256,
            },
            "gaps": {
                "set_id": "evaluable_iv_gaps_v1",
                "manifest": "gaps.csv",
                "key_field": "gap_key",
                "key_count": EXPECTED_EVALUABLE_GAPS,
                "keys_sha256": gap_keys_sha256,
            },
        },
        "candidate_semantics": {
            "finite_candidate_fit": {
                "evaluated_before": "acceptance_gates",
                "all_of": [
                    "proposal_available",
                    "roi_valid",
                    "solver_completed",
                    "finite:center_x",
                    "finite:center_y",
                    "finite:amplitude",
                    "finite:r_squared",
                    "finite:sigma_x",
                    "finite:sigma_y",
                ],
            },
            "gate_margin_audit": {
                "passing_signed_margin": ">= 0",
                "inclusive_bounds": True,
                "width_boundary_distance": (
                    "min(abs(sigma - sigma_min), abs(sigma_max - sigma)) "
                    "/ (sigma_max - sigma_min)"
                ),
                "near_width_boundary_rule": (
                    "either sigma_x or sigma_y has width_boundary_distance <= 0.05"
                ),
                "trigger_count_rule": "count >= ceil(0.05 * finite_candidate_fit_n)",
                "trigger_scope": "computed separately per method on the central variant",
            },
            "rejection_reporting": {
                "first_reason_order": [
                    "proposal_unavailable",
                    "roi_invalid",
                    "solver_failure",
                    "non_finite_parameters",
                    "proposal_to_fit",
                    "r_squared",
                    "amplitude",
                    "width",
                    "physics_branch_consistency",
                ],
                "full_reasons": "retain every failed gate in evaluation order",
            },
            "width_only_rejection": {
                "definition": (
                    "width gate fails while proposal, ROI, solver, finite-parameter, "
                    "proposal-to-fit, r_squared and amplitude requirements pass"
                ),
                "includes_physics_branch_consistency": False,
                "trigger_count": 12,
                "reference_opportunity_count": EXPECTED_VISIBLE_OPPORTUNITIES,
            },
        },
        "denominators": {
            "branch_consistent_coverage": {
                "kind": "fixed_reference_opportunity_set",
                "n": EXPECTED_VISIBLE_OPPORTUNITIES,
            },
            "post_gap_branch_consistent_recovery": {
                "kind": "fixed_reference_opportunity_set",
                "n": EXPECTED_EVALUABLE_GAPS,
            },
            "geometry_inconsistent_continuation": {
                "kind": "variant_and_method_specific",
                "population": "all_accepted_fits_in_all_visibility_regimes",
            },
            "conditional_pair_distance_residual": {
                "kind": "variant_and_method_specific",
                "population": "jointly_branch_consistent_pair_frames",
                "report_n": True,
            },
        },
        "denominator_language": (
            "fixed reference opportunity sets with explicit variant-specific "
            "denominators for conditional outcomes"
        ),
        "comparison_tolerances": {
            "continuous_values": {"atol": 1e-8, "rtol": 1e-9},
            "discrete_values": "exact",
            "reader_facing_values": "exact_after_declared_rounding",
        },
        "reader_facing_rounding": {
            "source_data": "full_precision",
            "counts": "exact",
            "rates_percent": {"decimal_places": 1},
            "pair_distance_residual_px": {"decimal_places": 3},
            "translation_error_px": {"decimal_places": 4},
            "relative_pair_change_percent": {"decimal_places": 4},
        },
        "conditional_width_sensitivity": {
            "trigger": {
                "gate_margin": {
                    "minimum_fraction": 0.05,
                    "normalized_distance_to_bound_max": 0.05,
                    "population": "finite_candidate_fits_per_method",
                    "count_rule": "ceil(minimum_fraction * N)",
                },
                "width_only_rejection": {
                    "minimum_count": 12,
                    "reference_opportunity_count": EXPECTED_VISIBLE_OPPORTUNITIES,
                    "excludes": "physics_based_branch_consistency",
                },
                "scope": (
                    "if either condition is met by any method, run every width "
                    "contract for all five methods"
                ),
            },
            "contracts": [
                {
                    "contract_id": "width_lower_relaxed",
                    "sigma_min_px": 0.5,
                    "sigma_max_px": 13.0,
                },
                {
                    "contract_id": "width_central",
                    "sigma_min_px": 0.6,
                    "sigma_max_px": 13.0,
                },
                {
                    "contract_id": "width_lower_tight",
                    "sigma_min_px": 0.75,
                    "sigma_max_px": 13.0,
                },
                {
                    "contract_id": "width_upper_tight",
                    "sigma_min_px": 0.6,
                    "sigma_max_px": 10.0,
                },
                {
                    "contract_id": "width_upper_relaxed",
                    "sigma_min_px": 0.6,
                    "sigma_max_px": 16.0,
                },
            ],
            "bounds_inclusive": True,
            "optimizer_hard_bounds_unchanged": True,
        },
    }
