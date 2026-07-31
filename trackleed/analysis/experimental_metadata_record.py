"""Author the evidence-backed experimental-metadata record before HITL review."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


DATA_REGIMES = (
    "manuscript/tables/table01_data_evidence_roles/source_data/data_regimes.csv"
)
CU_MANIFEST = (
    "manuscript/tables/table01_data_evidence_roles/source_data/"
    "upstream_contracts/cu111_positive_control_manifest.json"
)
INDEPENDENT_SCAN_SUMMARY = (
    "manuscript/tables/table01_data_evidence_roles/source_data/"
    "upstream_contracts/independent_energy_scan_summary.json"
)
MEMORY_SCAN_SUMMARY = (
    "manuscript/tables/table01_data_evidence_roles/source_data/"
    "upstream_contracts/memory_analysis_scan_summary.json"
)
EXTERNAL_REFERENCE_SUMMARY = (
    "manuscript/tables/table01_data_evidence_roles/source_data/"
    "upstream_contracts/external_reference_summary.json"
)
TEMPERATURE_PROFILE = (
    "manuscript/figures/figure05_cu111_physical_endpoint/"
    "source_data/temperature_profile.csv"
)
CYCLE_LANDMARKS = (
    "manuscript/figures/figure05_cu111_physical_endpoint/"
    "source_data/cycle_landmarks.csv"
)
TIFF_INVENTORY = "manuscript/experimental_metadata/tiff_metadata_inventory.json"
MANUSCRIPT_DRAFT = "manuscript/manuscript.md"
SCAN_A_SIDECAR = "data/26a21_ScanE/Ag111-3G_E0400.txt"
MEMORY_SIDECAR = "data/26a24_ScanE/Ag111-3G_E0400.txt"
SCAN_A_MEASUREMENTS = (
    "manuscript/scientific_baseline/frozen_inputs/cases/"
    "scan_a_pair_1/scan_measurements.csv"
)
SCAN_B_MEASUREMENTS = (
    "manuscript/scientific_baseline/frozen_inputs/cases/"
    "scan_b_pair_1/scan_measurements.csv"
)


def _sha256(workspace_root: Path, relative_path: str) -> str:
    return hashlib.sha256((workspace_root / relative_path).read_bytes()).hexdigest()


def _provenance(
    workspace_root: Path,
    source_path: str,
    locator: str,
    *,
    basis: str = "frozen_first_party_record",
) -> dict[str, str]:
    return {
        "source_path": source_path,
        "source_sha256": _sha256(workspace_root, source_path),
        "locator": locator,
        "basis": basis,
    }


def _recorded(
    workspace_root: Path,
    field_id: str,
    category: str,
    scope: list[str],
    value: object,
    provenance: list[tuple[str, str]],
    *,
    unit: str | None = None,
    note: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "field_id": field_id,
        "category": category,
        "scope": scope,
        "status": "recorded",
        "value": value,
        "unit": unit,
        "reason": None,
        "provenance": [
            _provenance(workspace_root, path, locator)
            for path, locator in provenance
        ],
        "search_scope": [],
    }
    if note is not None:
        record["note"] = note
    return record


def _unavailable(
    field_id: str,
    category: str,
    scope: list[str],
    reason: str,
    search_scope: list[str],
) -> dict[str, object]:
    return {
        "field_id": field_id,
        "category": category,
        "scope": scope,
        "status": "not available",
        "value": None,
        "unit": None,
        "reason": reason,
        "provenance": [],
        "search_scope": search_scope,
    }


def initial_metadata_record(workspace_root: Path) -> dict[str, object]:
    """Return the conservative, source-linked record for author review."""

    workspace_root = workspace_root.resolve()
    all_series = [
        "cu111_thermal_cycle",
        "ag111_scan_a",
        "ag111_scan_b",
        "ag111_memory_analysis",
    ]
    common_search = [
        MANUSCRIPT_DRAFT,
        DATA_REGIMES,
        CU_MANIFEST,
        INDEPENDENT_SCAN_SUMMARY,
        MEMORY_SCAN_SUMMARY,
        TIFF_INVENTORY,
    ]
    apparatus_reason = (
        "No authoritative apparatus value was found in the audited acquisition, "
        "session, frozen-source or manuscript records."
    )
    imaging_reason = (
        "No authoritative semantic field was found in the audited TIFF tags, "
        "session schema, frozen manifests or manuscript records."
    )
    preparation_reason = (
        "No authoritative preparation or exposure record was found in the "
        "audited local sources."
    )
    fields: list[dict[str, object]] = [
        _recorded(
            workspace_root,
            "measurement_technique",
            "apparatus",
            all_series,
            "low-energy electron diffraction (LEED) diffraction imaging",
            [(DATA_REGIMES, "scientific_series_label and evidence_role columns")],
        ),
        _unavailable(
            "leed_instrument",
            "apparatus",
            all_series,
            apparatus_reason,
            common_search,
        ),
        _unavailable(
            "electron_source",
            "apparatus",
            all_series,
            apparatus_reason,
            common_search,
        ),
        _unavailable(
            "incidence_geometry",
            "apparatus",
            all_series,
            apparatus_reason,
            common_search,
        ),
        _unavailable(
            "sample_manipulator",
            "apparatus",
            all_series,
            apparatus_reason,
            common_search,
        ),
        *[
            _unavailable(
                field_id,
                "apparatus",
                all_series,
                apparatus_reason,
                common_search,
            )
            for field_id in (
                "working_distance",
                "leed_screen",
                "instrument_settings",
            )
        ],
        _unavailable(
            "detector_type",
            "detector_camera_imaging",
            all_series,
            imaging_reason,
            common_search,
        ),
        *[
            _unavailable(
                field_id,
                "detector_camera_imaging",
                all_series,
                imaging_reason,
                common_search,
            )
            for field_id in (
                "detector_camera_manufacturer",
                "sensor_type",
                "imaging_optics",
            )
        ],
        _unavailable(
            "camera_model",
            "detector_camera_imaging",
            all_series,
            imaging_reason,
            common_search,
        ),
        _recorded(
            workspace_root,
            "image_dimensions",
            "detector_camera_imaging",
            all_series,
            {
                "width_px": 1000,
                "height_px": 1000,
                "verified_file_count": 851,
                "series_file_counts": {
                    "cu111_thermal_cycle": 701,
                    "ag111_scan_a": 35,
                    "ag111_scan_b": 46,
                    "ag111_memory_analysis": 69,
                },
            },
            [(TIFF_INVENTORY, "$.series[*].files[*].image_width/image_length")],
            unit="pixel",
        ),
        _recorded(
            workspace_root,
            "pixel_bit_depth",
            "detector_camera_imaging",
            all_series,
            {
                "bits_per_sample": 8,
                "samples_per_pixel": 1,
                "mode": "L",
                "compression_tag": 1,
                "encoding": "single-sample grayscale uncompressed TIFF",
            },
            [
                (
                    TIFF_INVENTORY,
                    "$.series[*].files[*].bits_per_sample/samples_per_pixel/mode/compression",
                )
            ],
        ),
        _recorded(
            workspace_root,
            "tiff_writer_software",
            "detector_camera_imaging",
            all_series,
            "National Instruments IMAQ",
            [(TIFF_INVENTORY, "$.series[*].files[*].software")],
            note=(
                "This is the embedded TIFF writer string; it is not a camera or "
                "detector model."
            ),
        ),
        _unavailable(
            "physical_detector_calibration",
            "detector_camera_imaging",
            all_series,
            "The audited TIFFs and source records contain pixel coordinates but no authoritative physical detector, angular or reciprocal-space calibration.",
            common_search,
        ),
        _unavailable(
            "exposure_time",
            "detector_camera_imaging",
            all_series,
            imaging_reason,
            [*common_search, SCAN_A_SIDECAR, MEMORY_SIDECAR],
        ),
        _unavailable(
            "gain",
            "detector_camera_imaging",
            all_series,
            imaging_reason,
            common_search,
        ),
        _unavailable(
            "binning",
            "detector_camera_imaging",
            all_series,
            imaging_reason,
            common_search,
        ),
        _unavailable(
            "frame_averaging",
            "detector_camera_imaging",
            all_series,
            imaging_reason,
            common_search,
        ),
        _unavailable(
            "illumination_or_beam_current",
            "detector_camera_imaging",
            all_series,
            imaging_reason,
            common_search,
        ),
        _unavailable(
            "detector_response",
            "detector_camera_imaging",
            all_series,
            imaging_reason,
            common_search,
        ),
        _unavailable(
            "acquisition_timestamps",
            "detector_camera_imaging",
            all_series,
            (
                "Date-time-shaped sidecar tokens have no authoritative local "
                "schema, and Scan B has no sidecar; no semantic acquisition "
                "timestamps are admitted."
            ),
            [*common_search, SCAN_A_SIDECAR, MEMORY_SIDECAR],
        ),
        _unavailable(
            "acquisition_preprocessing",
            "detector_camera_imaging",
            all_series,
            imaging_reason,
            common_search,
        ),
        _recorded(
            workspace_root,
            "cu111_sample_identity",
            "sample_preparation_exposure",
            ["cu111_thermal_cycle", "controlled_translations"],
            {"material": "Cu", "nominal_surface_orientation": "(111)"},
            [(DATA_REGIMES, "Cu(111) thermal-cycle positive control row")],
        ),
        _recorded(
            workspace_root,
            "ag111_sample_identity",
            "sample_preparation_exposure",
            ["ag111_scan_a", "ag111_scan_b", "ag111_memory_analysis"],
            {"material": "Ag", "nominal_surface_orientation": "(111)"},
            [(DATA_REGIMES, "Ag(111) energy-scan rows")],
        ),
        *[
            _unavailable(
                field_id,
                "sample_preparation_exposure",
                all_series if field_id == "surface_exposure" else scope,
                preparation_reason,
                common_search,
            )
            for field_id, scope in (
                ("cu111_cleaning", ["cu111_thermal_cycle"]),
                ("ag111_cleaning", ["ag111_scan_a", "ag111_scan_b", "ag111_memory_analysis"]),
                ("cu111_annealing", ["cu111_thermal_cycle"]),
                ("ag111_annealing", ["ag111_scan_a", "ag111_scan_b", "ag111_memory_analysis"]),
                ("surface_exposure", all_series),
            )
        ],
        *[
            _unavailable(
                field_id,
                "sample_preparation_exposure",
                all_series,
                preparation_reason,
                common_search,
            )
            for field_id in (
                "crystal_source_purity",
                "surface_miscut",
                "sample_mounting",
                "thermocouple_attachment",
                "surface_cleanliness_verification",
            )
        ],
        _unavailable(
            "uhv_status_conditions",
            "uhv_conditions",
            all_series,
            (
                "No authoritative record establishes whether or under which "
                "UHV conditions the reported acquisitions were performed; UHV "
                "is not inferred from use of LEED."
            ),
            common_search,
        ),
        _unavailable(
            "base_pressure",
            "uhv_conditions",
            all_series,
            "No base-pressure or gauge record was found in the audited local sources; UHV is not inferred from use of LEED.",
            common_search,
        ),
        _unavailable(
            "acquisition_pressure",
            "uhv_conditions",
            all_series,
            "No pressure-during-acquisition record was found in the audited local sources.",
            common_search,
        ),
        _unavailable(
            "pressure_gauge",
            "uhv_conditions",
            all_series,
            "No pressure-gauge manufacturer, model or calibration record was found in the audited local sources.",
            common_search,
        ),
        _unavailable(
            "process_gases",
            "uhv_conditions",
            all_series,
            "No process-gas, purity or residual-gas record was found in the audited local sources.",
            common_search,
        ),
        _recorded(
            workspace_root,
            "cu111_energy_program",
            "energy_programs",
            ["cu111_thermal_cycle", "controlled_translations"],
            {"constant_energy_ev": 56.0, "acquired_frame_count": 701},
            [
                (CU_MANIFEST, "$.series.energy_ev and $.series.frame_count"),
                (DATA_REGIMES, "Cu(111) thermal-cycle positive control row"),
            ],
            unit="eV",
        ),
        _recorded(
            workspace_root,
            "ag111_scan_a_energy_program",
            "energy_programs",
            ["ag111_scan_a"],
            {
                "minimum_energy_ev": 40.0,
                "maximum_energy_ev": 220.5,
                "acquired_frame_count": 35,
                "analysis_reference_energy_ev": 188.5,
                "exact_schedule_recorded": True,
            },
            [
                (INDEPENDENT_SCAN_SUMMARY, "$.baseline_cases[?series=26a21].energy_range_ev/reference_energy_ev"),
                (SCAN_A_MEASUREMENTS, "header and ordered frame,energy_ev rows"),
                (DATA_REGIMES, "Ag(111) Scans A and B row"),
            ],
            unit="eV",
        ),
        _recorded(
            workspace_root,
            "ag111_scan_b_energy_program",
            "energy_programs",
            ["ag111_scan_b"],
            {
                "minimum_energy_ev": 40.0,
                "maximum_energy_ev": 220.4,
                "acquired_frame_count": 46,
                "analysis_reference_energy_ev": 188.5,
                "exact_schedule_recorded": True,
            },
            [
                (INDEPENDENT_SCAN_SUMMARY, "$.baseline_cases[?series=26a21b].energy_range_ev/reference_energy_ev"),
                (SCAN_B_MEASUREMENTS, "header and ordered frame,energy_ev rows"),
                (DATA_REGIMES, "Ag(111) Scans A and B row"),
            ],
            unit="eV",
        ),
        _recorded(
            workspace_root,
            "ag111_memory_energy_program",
            "energy_programs",
            ["ag111_memory_analysis"],
            {
                "minimum_energy_ev": 45.0,
                "maximum_energy_ev": 220.2,
                "acquired_frame_count": 69,
                "analysis_reference_energy_ev": 187.9,
            },
            [
                (MEMORY_SCAN_SUMMARY, "$.energy_range_ev/frame_count/reference_energy_ev"),
                (DATA_REGIMES, "Ag(111) memory-analysis scan row"),
            ],
            unit="eV",
        ),
        *[
            _unavailable(
                field_id,
                "energy_programs",
                all_series,
                reason,
                [
                    *common_search,
                    SCAN_A_MEASUREMENTS,
                    SCAN_B_MEASUREMENTS,
                    SCAN_A_SIDECAR,
                    MEMORY_SIDECAR,
                ],
            )
            for field_id, reason in (
                (
                    "energy_step_timing_dwell",
                    "Frozen records contain frame order and energy but no authoritative energy-step timing or dwell field.",
                ),
                (
                    "energy_sweep_control_program",
                    "No authoritative sweep-control program or scan-rate record was found in the audited local sources.",
                ),
                (
                    "energy_calibration_uncertainty",
                    "No authoritative energy calibration or uncertainty record was found in the audited local sources.",
                ),
            )
        ],
        _recorded(
            workspace_root,
            "external_reference_energy_program",
            "energy_programs",
            ["restricted_external_reference"],
            {
                "minimum_energy_ev": 187.9,
                "maximum_energy_ev": 220.2,
                "frame_count": 15,
                "relationship": "subset of ag111_memory_analysis",
            },
            [
                (EXTERNAL_REFERENCE_SUMMARY, "$.energy_range_ev/frame_count"),
                (DATA_REGIMES, "Restricted high-energy external reference row"),
            ],
            unit="eV",
        ),
        _recorded(
            workspace_root,
            "cu111_temperature_program",
            "temperature_programs",
            ["cu111_thermal_cycle", "controlled_translations"],
            {
                "frame_count": 701,
                "initial_temperature_c": 10.0,
                "maximum_temperature_c": 800.0,
                "plateau_frame_orders": [372, 373, 374, 375, 376],
                "selected_post_plateau_frame_order": 378,
                "selected_post_plateau_temperature_c": 797.0,
                "final_frame_order": 700,
                "final_temperature_c": 92.0,
                "full_temperature_by_frame_recorded": True,
            },
            [
                (TEMPERATURE_PROFILE, "header and rows 2-702"),
                (CYCLE_LANDMARKS, "all landmark rows"),
                (DATA_REGIMES, "Cu(111) thermal-cycle positive control row"),
            ],
            unit="degree_Celsius",
        ),
        _unavailable(
            "cu111_temperature_sensor",
            "temperature_programs",
            ["cu111_thermal_cycle"],
            "No sensor type, location or thermal-contact record was found in the audited local sources.",
            common_search,
        ),
        _unavailable(
            "cu111_temperature_calibration",
            "temperature_programs",
            ["cu111_thermal_cycle"],
            "No temperature calibration or uncertainty record was found in the audited local sources.",
            common_search,
        ),
        _unavailable(
            "cu111_acquisition_timing",
            "temperature_programs",
            ["cu111_thermal_cycle"],
            "The frozen Cu record contains temperature by frame order but no authoritative elapsed-time or timestamp field.",
            common_search,
        ),
        _unavailable(
            "ag111_independent_scan_temperature",
            "temperature_programs",
            ["ag111_scan_a", "ag111_scan_b"],
            "No authoritative sample-temperature field occurs in the independent energy-scan records; filename tokens are not admitted as evidence.",
            [*common_search, SCAN_A_SIDECAR],
        ),
        _unavailable(
            "ag111_memory_scan_temperature",
            "temperature_programs",
            ["ag111_memory_analysis"],
            "No authoritative sample-temperature field occurs in the memory-analysis records; filename tokens are not admitted as evidence.",
            [*common_search, MEMORY_SIDECAR],
        ),
        _unavailable(
            "cu111_heating_rate",
            "heating_cooling_rates",
            ["cu111_thermal_cycle"],
            "A heating rate cannot be calculated from temperature versus frame order without authoritative acquisition timing.",
            [*common_search, TEMPERATURE_PROFILE],
        ),
        _unavailable(
            "cu111_cooling_rate",
            "heating_cooling_rates",
            ["cu111_thermal_cycle"],
            "A cooling rate cannot be calculated from temperature versus frame order without authoritative acquisition timing.",
            [*common_search, TEMPERATURE_PROFILE],
        ),
    ]
    return {
        "schema_version": 1,
        "study": "trackleed_publication_scientific_core",
        "record_status": "awaiting_author_confirmation",
        "status_definitions": {
            "recorded": "An authoritative source records the value.",
            "not recorded": "The author confirms the value was not recorded.",
            "not available": "No authoritative record is available in the audited sources.",
        },
        "audit_rules": {
            "filename_only_inference_forbidden": True,
            "unlabeled_sidecar_columns_interpreted": False,
            "absence_from_repository_defaults_to": "not available",
            "not_recorded_requires_author_confirmation": True,
        },
        "fields": fields,
    }


def metadata_claim_constraints() -> dict[str, Any]:
    """Return metadata-derived constraints, not the final claim boundary."""

    return {
        "schema_version": 1,
        "scope": "metadata_derived_claim_constraints_only",
        "final_claim_boundary": False,
        "constraints": [
            {
                "constraint_id": "no_instrument_traceable_absolute_geometry",
                "field_ids": [
                    "leed_instrument",
                    "incidence_geometry",
                    "physical_detector_calibration",
                ],
                "claim_impact": "Keep geometric outcomes in pixels or dimensionless relative quantities; do not claim instrument-traceable absolute position, reciprocal-space calibration, or instrumental uncertainty.",
            },
            {
                "constraint_id": "no_absolute_intensity_interpretation",
                "field_ids": [
                    "detector_type",
                    "camera_model",
                    "exposure_time",
                    "gain",
                    "binning",
                    "frame_averaging",
                    "illumination_or_beam_current",
                    "detector_response",
                ],
                "claim_impact": "Do not interpret Gaussian amplitude or fitted volume as calibrated absolute LEED intensity or compare it as a calibrated intensity scale across acquisitions.",
            },
            {
                "constraint_id": "no_preparation_state_attribution",
                "field_ids": [
                    "cu111_cleaning",
                    "ag111_cleaning",
                    "cu111_annealing",
                    "ag111_annealing",
                    "surface_exposure",
                    "surface_cleanliness_verification",
                    "uhv_status_conditions",
                    "base_pressure",
                    "acquisition_pressure",
                    "pressure_gauge",
                    "process_gases",
                ],
                "claim_impact": "Do not attribute changes to a specified preparation state, contamination level, adsorbate exposure, or pressure history.",
            },
            {
                "constraint_id": "no_temperature_kinetics_or_calibrated_transition",
                "field_ids": [
                    "cu111_temperature_sensor",
                    "cu111_temperature_calibration",
                    "cu111_acquisition_timing",
                    "cu111_heating_rate",
                    "cu111_cooling_rate",
                ],
                "claim_impact": "Use the Cu series as a recorded temperature-indexed positive control; do not claim kinetics, a calibrated thermal-expansion coefficient, a phase-transition temperature, or thermal-hysteresis attribution.",
            },
            {
                "constraint_id": "no_ag_scan_kinetics_or_calibrated_iv",
                "field_ids": [
                    "ag111_independent_scan_temperature",
                    "ag111_memory_scan_temperature",
                    "energy_step_timing_dwell",
                    "energy_sweep_control_program",
                    "energy_calibration_uncertainty",
                ],
                "claim_impact": "Treat Ag evidence as case-based behavior over recorded energy sequences; do not claim time-dependent scan kinetics, temperature-dependent Ag behavior, or a calibrated structural LEED I(V) interpretation.",
            },
            {
                "constraint_id": "apparatus_diagnostics_not_component_attribution",
                "field_ids": [
                    "leed_instrument",
                    "detector_type",
                    "incidence_geometry",
                ],
                "claim_impact": "Midpoint drift and apparatus-shadow behavior may remain diagnostics but must not be attributed to a specific physical component or generalized to other LEED systems.",
            },
            {
                "constraint_id": "ag_series_not_controlled_population_replicates",
                "field_ids": [
                    "base_pressure",
                    "acquisition_pressure",
                    "exposure_time",
                    "gain",
                    "illumination_or_beam_current",
                    "ag111_cleaning",
                ],
                "claim_impact": "Treat Scans A and B as two separately acquired case series under the tested measurement contract, not controlled population replicates.",
            },
            {
                "constraint_id": "imaq_writer_not_camera_identity",
                "field_ids": ["tiff_writer_software", "camera_model"],
                "claim_impact": "The National Instruments IMAQ tag may identify the TIFF writer only; it cannot identify the camera or detector model.",
            },
        ],
    }
