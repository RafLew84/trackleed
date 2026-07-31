from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image, TiffImagePlugin

from trackleed.analysis.experimental_metadata import (
    MetadataValidationError,
    validate_metadata_package,
)


REQUIRED_FIELDS = {
    "apparatus": (
        "measurement_technique",
        "leed_instrument",
        "electron_source",
        "incidence_geometry",
        "sample_manipulator",
        "working_distance",
        "leed_screen",
        "instrument_settings",
    ),
    "detector_camera_imaging": (
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
    ),
    "sample_preparation_exposure": (
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
    ),
    "uhv_conditions": (
        "uhv_status_conditions",
        "base_pressure",
        "acquisition_pressure",
        "pressure_gauge",
        "process_gases",
    ),
    "energy_programs": (
        "cu111_energy_program",
        "ag111_scan_a_energy_program",
        "ag111_scan_b_energy_program",
        "ag111_memory_energy_program",
        "external_reference_energy_program",
        "energy_step_timing_dwell",
        "energy_sweep_control_program",
        "energy_calibration_uncertainty",
    ),
    "temperature_programs": (
        "cu111_temperature_program",
        "cu111_temperature_sensor",
        "cu111_temperature_calibration",
        "cu111_acquisition_timing",
        "ag111_independent_scan_temperature",
        "ag111_memory_scan_temperature",
    ),
    "heating_cooling_rates": (
        "cu111_heating_rate",
        "cu111_cooling_rate",
    ),
}


class ExperimentalMetadataContractTests(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _create_pending_package(self, workspace: Path) -> Path:
        source = workspace / "primary_source.json"
        self._write_json(source, {"energy_ev": 56.0})
        fields = []
        for category, field_ids in REQUIRED_FIELDS.items():
            for field_id in field_ids:
                fields.append(
                    {
                        "field_id": field_id,
                        "category": category,
                        "scope": ["publication_scientific_core"],
                        "status": "not available",
                        "value": None,
                        "unit": None,
                        "reason": "No authoritative record is available in the audited sources.",
                        "provenance": [],
                        "search_scope": ["primary_source.json"],
                    }
                )
        energy = next(
            field for field in fields if field["field_id"] == "cu111_energy_program"
        )
        energy.update(
            {
                "status": "recorded",
                "value": {"constant_energy_ev": 56.0},
                "reason": None,
                "provenance": [
                    {
                        "source_path": "primary_source.json",
                        "source_sha256": self._sha256(source),
                        "locator": "$.energy_ev",
                        "basis": "direct_record",
                    }
                ],
            }
        )

        package = workspace / "metadata_package"
        package.mkdir()
        (package / "README.md").write_text("# Metadata package\n", encoding="utf-8")
        record_path = package / "experimental_metadata.json"
        constraints_path = package / "metadata_claim_constraints.json"
        checkpoint_path = package / "author_checkpoint.json"
        self._write_json(
            record_path,
            {
                "schema_version": 1,
                "study": "trackleed_publication_scientific_core",
                "record_status": "awaiting_author_confirmation",
                "status_definitions": {
                    "recorded": "An authoritative source records the value.",
                    "not recorded": "The author confirms the value was not recorded.",
                    "not available": "No authoritative record is available in the audited sources.",
                },
                "fields": fields,
            },
        )
        self._write_json(
            constraints_path,
            {
                "schema_version": 1,
                "scope": "metadata_derived_claim_constraints_only",
                "final_claim_boundary": False,
                "constraints": [
                    {
                        "constraint_id": "instrument_scope",
                        "field_ids": ["leed_instrument", "detector_type"],
                        "claim_impact": "Do not generalize results to a specified instrument class.",
                    }
                ],
            },
        )
        self._write_json(
            checkpoint_path,
            {
                "schema_version": 1,
                "status": "pending",
                "record_sha256": self._sha256(record_path),
                "constraints_sha256": self._sha256(constraints_path),
                "confirmed_by": None,
                "confirmed_date": None,
                "confirmation_statement": None,
            },
        )
        self._write_json(
            package / "package_manifest.json",
            {
                "schema_version": 1,
                "files_sha256": {
                    path.name: self._sha256(path)
                    for path in (
                        package / "README.md",
                        record_path,
                        constraints_path,
                        checkpoint_path,
                    )
                },
            },
        )
        return package

    def test_pending_record_covers_every_required_field_and_validates(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            package = self._create_pending_package(workspace)

            report = validate_metadata_package(workspace, package)

            self.assertEqual(report.field_count, 58)
            self.assertEqual(report.recorded_count, 1)
            self.assertEqual(report.not_available_count, 57)
            self.assertEqual(report.not_recorded_count, 0)
            self.assertEqual(report.constraint_count, 1)
            self.assertEqual(report.author_checkpoint_status, "pending")

    def test_recorded_value_cannot_use_filename_inference_as_provenance(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            package = self._create_pending_package(workspace)
            record_path = package / "experimental_metadata.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            energy = next(
                field
                for field in record["fields"]
                if field["field_id"] == "cu111_energy_program"
            )
            energy["provenance"][0]["basis"] = "filename_inference"
            self._write_json(record_path, record)
            checkpoint_path = package / "author_checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["record_sha256"] = self._sha256(record_path)
            self._write_json(checkpoint_path, checkpoint)
            manifest_path = package / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files_sha256"]["experimental_metadata.json"] = self._sha256(
                record_path
            )
            manifest["files_sha256"]["author_checkpoint.json"] = self._sha256(
                checkpoint_path
            )
            self._write_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                MetadataValidationError,
                "unsupported provenance basis",
            ):
                validate_metadata_package(workspace, package)

    def test_not_recorded_requires_author_confirmation_provenance(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            package = self._create_pending_package(workspace)
            record_path = package / "experimental_metadata.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            field = next(
                item
                for item in record["fields"]
                if item["field_id"] == "exposure_time"
            )
            field["status"] = "not recorded"
            self._write_json(record_path, record)
            checkpoint_path = package / "author_checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["record_sha256"] = self._sha256(record_path)
            self._write_json(checkpoint_path, checkpoint)
            manifest_path = package / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files_sha256"]["experimental_metadata.json"] = self._sha256(
                record_path
            )
            manifest["files_sha256"]["author_checkpoint.json"] = self._sha256(
                checkpoint_path
            )
            self._write_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                MetadataValidationError,
                "not recorded.*author confirmation",
            ):
                validate_metadata_package(workspace, package)

    def test_changed_provenance_source_fails_without_rewriting_package(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            package = self._create_pending_package(workspace)
            before = {
                path.name: path.read_bytes()
                for path in package.iterdir()
                if path.is_file()
            }
            self._write_json(workspace / "primary_source.json", {"energy_ev": 57.0})

            with self.assertRaisesRegex(
                MetadataValidationError,
                "Provenance source hash mismatch",
            ):
                validate_metadata_package(workspace, package)

            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in package.iterdir()
                    if path.is_file()
                },
                before,
            )

    def test_confirmed_checkpoint_requires_a_complete_author_attestation(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            package = self._create_pending_package(workspace)
            checkpoint_path = package / "author_checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["status"] = "confirmed"
            self._write_json(checkpoint_path, checkpoint)
            manifest_path = package / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files_sha256"]["author_checkpoint.json"] = self._sha256(
                checkpoint_path
            )
            self._write_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                MetadataValidationError,
                "Confirmed author checkpoint is incomplete",
            ):
                validate_metadata_package(
                    workspace,
                    package,
                    require_author_confirmation=True,
                )

    def test_complete_author_attestation_closes_the_checkpoint(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            package = self._create_pending_package(workspace)
            record_path = package / "experimental_metadata.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["record_status"] = "author_confirmed"
            self._write_json(record_path, record)
            checkpoint_path = package / "author_checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint.update(
                {
                    "status": "confirmed",
                    "record_sha256": self._sha256(record_path),
                    "confirmed_by": "publication author",
                    "confirmed_date": "2026-07-31",
                    "confirmation_statement": (
                        "The recovered values, explicit gaps and absence of "
                        "unsupported inference are confirmed."
                    ),
                    "attestations": {
                        "explicit_gaps_reviewed": True,
                        "no_unsupported_inference": True,
                        "recovered_values_reviewed": True,
                    },
                    "confirmed_not_recorded_field_ids": [],
                }
            )
            self._write_json(checkpoint_path, checkpoint)
            manifest_path = package / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files_sha256"]["experimental_metadata.json"] = self._sha256(
                record_path
            )
            manifest["files_sha256"]["author_checkpoint.json"] = self._sha256(
                checkpoint_path
            )
            self._write_json(manifest_path, manifest)

            report = validate_metadata_package(
                workspace,
                package,
                require_author_confirmation=True,
            )

            self.assertEqual(report.author_checkpoint_status, "confirmed")

    def test_tiff_inventory_binds_embedded_image_metadata_to_raw_sources(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            raw_path = workspace / "raw" / "frame.tiff"
            raw_path.parent.mkdir()
            tiff_info = TiffImagePlugin.ImageFileDirectory_v2()
            tiff_info[305] = "Unit Test Writer"
            Image.new("L", (3, 2), color=7).save(raw_path, tiffinfo=tiff_info)
            package = self._create_pending_package(workspace)
            inventory_path = package / "tiff_metadata_inventory.json"
            self._write_json(
                inventory_path,
                {
                    "schema_version": 1,
                    "series": [
                        {
                            "series_id": "test_series",
                            "expected_count": 1,
                            "files": [
                                {
                                    "source_path": "raw/frame.tiff",
                                    "source_sha256": self._sha256(raw_path),
                                    "image_width": 3,
                                    "image_length": 2,
                                    "mode": "L",
                                    "bits_per_sample": 8,
                                    "samples_per_pixel": 1,
                                    "compression": 1,
                                    "software": "Unit Test Writer",
                                }
                            ],
                        }
                    ],
                },
            )
            manifest_path = package / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files_sha256"][inventory_path.name] = self._sha256(
                inventory_path
            )
            self._write_json(manifest_path, manifest)

            validate_metadata_package(workspace, package)
            Image.new("L", (4, 2), color=7).save(raw_path)

            with self.assertRaisesRegex(
                MetadataValidationError,
                "TIFF source hash mismatch",
            ):
                validate_metadata_package(workspace, package)
