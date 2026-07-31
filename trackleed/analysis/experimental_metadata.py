"""Validate source-linked experimental metadata for the publication core."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


SCHEMA_VERSION = 1
STUDY = "trackleed_publication_scientific_core"
ALLOWED_STATUSES = {"recorded", "not recorded", "not available"}
ALLOWED_PROVENANCE_BASES = {
    "direct_record",
    "frozen_first_party_record",
    "author_confirmation",
}
REQUIRED_FIELDS = {
    "apparatus": {
        "measurement_technique",
        "leed_instrument",
        "electron_source",
        "incidence_geometry",
        "sample_manipulator",
        "working_distance",
        "leed_screen",
        "instrument_settings",
    },
    "detector_camera_imaging": {
        "detector_type",
        "camera_model",
        "detector_camera_manufacturer",
        "sensor_type",
        "imaging_optics",
        "image_dimensions",
        "pixel_bit_depth",
        "tiff_writer_software",
        "physical_detector_calibration",
        "exposure_time",
        "gain",
        "binning",
        "frame_averaging",
        "illumination_or_beam_current",
        "detector_response",
        "acquisition_timestamps",
        "acquisition_preprocessing",
    },
    "sample_preparation_exposure": {
        "cu111_sample_identity",
        "ag111_sample_identity",
        "cu111_cleaning",
        "ag111_cleaning",
        "cu111_annealing",
        "ag111_annealing",
        "surface_exposure",
        "crystal_source_purity",
        "surface_miscut",
        "sample_mounting",
        "thermocouple_attachment",
        "surface_cleanliness_verification",
    },
    "uhv_conditions": {
        "uhv_status_conditions",
        "base_pressure",
        "acquisition_pressure",
        "pressure_gauge",
        "process_gases",
    },
    "energy_programs": {
        "cu111_energy_program",
        "ag111_scan_a_energy_program",
        "ag111_scan_b_energy_program",
        "ag111_memory_energy_program",
        "external_reference_energy_program",
        "energy_step_timing_dwell",
        "energy_sweep_control_program",
        "energy_calibration_uncertainty",
    },
    "temperature_programs": {
        "cu111_temperature_program",
        "cu111_temperature_sensor",
        "cu111_temperature_calibration",
        "cu111_acquisition_timing",
        "ag111_independent_scan_temperature",
        "ag111_memory_scan_temperature",
    },
    "heating_cooling_rates": {
        "cu111_heating_rate",
        "cu111_cooling_rate",
    },
}


class MetadataValidationError(ValueError):
    """The metadata package violates its frozen source or schema contract."""


@dataclass(frozen=True)
class MetadataValidationReport:
    field_count: int
    recorded_count: int
    not_recorded_count: int
    not_available_count: int
    constraint_count: int
    author_checkpoint_status: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataValidationError(f"Cannot read JSON: {path}") from exc


def _source_path(workspace_root: Path, relative_path: str) -> Path:
    requested = Path(relative_path)
    if requested.is_absolute():
        raise MetadataValidationError(f"Source path must be relative: {relative_path}")
    workspace = workspace_root.resolve()
    resolved = (workspace / requested).resolve()
    if not resolved.is_relative_to(workspace):
        raise MetadataValidationError(f"Source path escapes workspace: {relative_path}")
    return resolved


def _scalar_tiff_tag(value: object) -> object:
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return value


def _validate_tiff_inventory(workspace_root: Path, inventory_path: Path) -> None:
    inventory = _read_json(inventory_path)
    if (
        not isinstance(inventory, dict)
        or inventory.get("schema_version") != SCHEMA_VERSION
        or not isinstance(inventory.get("series"), list)
    ):
        raise MetadataValidationError("TIFF metadata inventory schema changed")
    seen_paths: set[str] = set()
    for series in inventory["series"]:
        if not isinstance(series, dict) or not isinstance(series.get("files"), list):
            raise MetadataValidationError("TIFF series record is invalid")
        files = series["files"]
        if len(files) != series.get("expected_count"):
            raise MetadataValidationError("TIFF series count changed")
        for record in files:
            if not isinstance(record, dict):
                raise MetadataValidationError("TIFF source record is invalid")
            relative_path = str(record.get("source_path", ""))
            if not relative_path or relative_path in seen_paths:
                raise MetadataValidationError("Duplicate or empty TIFF source path")
            seen_paths.add(relative_path)
            source = _source_path(workspace_root, relative_path)
            if not source.is_file() or _sha256(source) != record.get("source_sha256"):
                raise MetadataValidationError(
                    f"TIFF source hash mismatch: {relative_path}"
                )
            try:
                with Image.open(source) as image:
                    actual = {
                        "image_width": image.width,
                        "image_length": image.height,
                        "mode": image.mode,
                        "bits_per_sample": _scalar_tiff_tag(
                            image.tag_v2.get(258, 8 if image.mode == "L" else None)
                        ),
                        "samples_per_pixel": _scalar_tiff_tag(
                            image.tag_v2.get(277, 1)
                        ),
                        "compression": _scalar_tiff_tag(image.tag_v2.get(259, 1)),
                        "software": _scalar_tiff_tag(image.tag_v2.get(305)),
                    }
            except (OSError, ValueError) as exc:
                raise MetadataValidationError(
                    f"Cannot read TIFF metadata: {relative_path}"
                ) from exc
            expected = {key: record.get(key) for key in actual}
            if actual != expected:
                raise MetadataValidationError(
                    f"TIFF embedded metadata changed: {relative_path}"
                )


def validate_metadata_package(
    workspace_root: Path,
    package_dir: Path,
    *,
    require_author_confirmation: bool = False,
) -> MetadataValidationReport:
    """Validate the package without updating records, hashes, or checkpoint state."""

    workspace_root = workspace_root.resolve()
    package_dir = package_dir.resolve()
    manifest = _read_json(package_dir / "package_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise MetadataValidationError("Package manifest schema changed")
    expected_files = manifest.get("files_sha256")
    if not isinstance(expected_files, dict):
        raise MetadataValidationError("Package manifest has no file hash map")
    actual_files = {
        path.name: path
        for path in package_dir.iterdir()
        if path.is_file() and path.name != "package_manifest.json"
    }
    if set(actual_files) != set(expected_files):
        raise MetadataValidationError("Metadata package file inventory changed")
    for name, path in actual_files.items():
        if _sha256(path) != expected_files[name]:
            raise MetadataValidationError(f"Metadata package hash mismatch: {name}")
    inventory_path = package_dir / "tiff_metadata_inventory.json"
    if inventory_path.is_file():
        _validate_tiff_inventory(workspace_root, inventory_path)

    record_path = package_dir / "experimental_metadata.json"
    constraints_path = package_dir / "metadata_claim_constraints.json"
    record = _read_json(record_path)
    constraints = _read_json(constraints_path)
    checkpoint = _read_json(package_dir / "author_checkpoint.json")
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("study") != STUDY
        or not isinstance(record.get("fields"), list)
    ):
        raise MetadataValidationError("Experimental metadata schema changed")

    required_by_id = {
        field_id: category
        for category, field_ids in REQUIRED_FIELDS.items()
        for field_id in field_ids
    }
    records_by_id: dict[str, dict[str, Any]] = {}
    status_counts = {status: 0 for status in ALLOWED_STATUSES}
    not_recorded_ids: set[str] = set()
    for field in record["fields"]:
        if not isinstance(field, dict):
            raise MetadataValidationError("Metadata field is not an object")
        field_id = str(field.get("field_id", ""))
        if field_id not in required_by_id or field_id in records_by_id:
            raise MetadataValidationError(f"Unknown or duplicate field: {field_id!r}")
        if field.get("category") != required_by_id[field_id]:
            raise MetadataValidationError(f"Wrong category for field: {field_id}")
        status = str(field.get("status", ""))
        if status not in ALLOWED_STATUSES:
            raise MetadataValidationError(f"Invalid status for field: {field_id}")
        status_counts[status] += 1
        records_by_id[field_id] = field
        if status == "recorded":
            if field.get("value") is None or field.get("reason") is not None:
                raise MetadataValidationError(f"Recorded field has no value: {field_id}")
            provenance = field.get("provenance")
            if not isinstance(provenance, list) or not provenance:
                raise MetadataValidationError(
                    f"Recorded field has no provenance: {field_id}"
                )
            for source in provenance:
                if not isinstance(source, dict):
                    raise MetadataValidationError(f"Invalid provenance: {field_id}")
                if source.get("basis") not in ALLOWED_PROVENANCE_BASES:
                    raise MetadataValidationError(
                        f"Recorded field uses unsupported provenance basis: {field_id}"
                    )
                if not str(source.get("locator", "")):
                    raise MetadataValidationError(
                        f"Recorded field has no source locator: {field_id}"
                    )
                source_path = _source_path(
                    workspace_root, str(source.get("source_path", ""))
                )
                if not source_path.is_file():
                    raise MetadataValidationError(
                        f"Provenance source is unavailable: {field_id}"
                    )
                if _sha256(source_path) != source.get("source_sha256"):
                    raise MetadataValidationError(
                        f"Provenance source hash mismatch: {field_id}"
                    )
        else:
            if field.get("value") is not None or not str(field.get("reason", "")):
                raise MetadataValidationError(
                    f"Missing metadata status has an unsupported value: {field_id}"
                )
            if not isinstance(field.get("search_scope"), list) or not field[
                "search_scope"
            ]:
                raise MetadataValidationError(f"Missing search scope: {field_id}")
            if status == "not recorded":
                not_recorded_ids.add(field_id)

    if set(records_by_id) != set(required_by_id):
        raise MetadataValidationError("Required experimental metadata fields changed")

    if (
        not isinstance(constraints, dict)
        or constraints.get("schema_version") != SCHEMA_VERSION
        or constraints.get("scope") != "metadata_derived_claim_constraints_only"
        or constraints.get("final_claim_boundary") is not False
        or not isinstance(constraints.get("constraints"), list)
    ):
        raise MetadataValidationError("Metadata claim-constraint schema changed")
    for constraint in constraints["constraints"]:
        if not isinstance(constraint, dict) or not set(
            constraint.get("field_ids", [])
        ).issubset(records_by_id):
            raise MetadataValidationError("Claim constraint references an unknown field")

    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1:
        raise MetadataValidationError("Author checkpoint schema changed")
    checkpoint_status = str(checkpoint.get("status", ""))
    if checkpoint_status not in {"pending", "confirmed"}:
        raise MetadataValidationError("Author checkpoint status changed")
    if checkpoint.get("record_sha256") != _sha256(record_path) or checkpoint.get(
        "constraints_sha256"
    ) != _sha256(constraints_path):
        raise MetadataValidationError("Author checkpoint does not bind current records")
    if checkpoint_status == "confirmed":
        attestations = checkpoint.get("attestations")
        if (
            record.get("record_status") != "author_confirmed"
            or not str(checkpoint.get("confirmed_by", ""))
            or not str(checkpoint.get("confirmed_date", ""))
            or not str(checkpoint.get("confirmation_statement", ""))
            or not isinstance(attestations, dict)
            or attestations
            != {
                "explicit_gaps_reviewed": True,
                "no_unsupported_inference": True,
                "recovered_values_reviewed": True,
            }
        ):
            raise MetadataValidationError(
                "Confirmed author checkpoint is incomplete"
            )
    confirmed_not_recorded = checkpoint.get("confirmed_not_recorded_field_ids", [])
    if not_recorded_ids and (
        checkpoint_status != "confirmed"
        or not isinstance(confirmed_not_recorded, list)
        or set(confirmed_not_recorded) != not_recorded_ids
    ):
        raise MetadataValidationError(
            "Fields marked not recorded require author confirmation provenance"
        )
    if require_author_confirmation and checkpoint_status != "confirmed":
        raise MetadataValidationError("Author confirmation is required")

    return MetadataValidationReport(
        field_count=len(records_by_id),
        recorded_count=status_counts["recorded"],
        not_recorded_count=status_counts["not recorded"],
        not_available_count=status_counts["not available"],
        constraint_count=len(constraints["constraints"]),
        author_checkpoint_status=checkpoint_status,
    )
