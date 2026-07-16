from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from trackleed.io.publication_exporter import create_publication_export


class PublicationExporterTests(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)

    @staticmethod
    def _snapshot_files(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def test_complete_export_copy_retains_data_and_marks_stored_fwhm(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_root = workspace / "data"
            export_dir = source_root / "sample" / "eksport"
            csv_path = export_dir / "full_analysis_per_spot.csv"
            original_columns = [
                "track_id",
                "frame_id",
                "temperature",
                "amplitude",
                "fwhm_x",
                "fwhm_x_error",
                "fwhm_y",
                "fwhm_y_error",
            ]
            self._write_csv(
                csv_path,
                original_columns,
                [
                    {
                        "track_id": 4,
                        "frame_id": 12,
                        "temperature": 350,
                        "amplitude": 0.75,
                        "fwhm_x": 10.0,
                        "fwhm_x_error": 0.2,
                        "fwhm_y": 14.0,
                        "fwhm_y_error": 0.3,
                    }
                ],
            )
            unrelated_bytes = b"publication-sidecar\x00\xff"
            (export_dir / "notes.bin").write_bytes(unrelated_bytes)
            source_before = self._snapshot_files(source_root)
            destination_root = workspace / "publication_exports"

            result = create_publication_export(source_root, destination_root)

            copied_csv = destination_root / "sample" / "eksport" / csv_path.name
            columns, rows = self._read_csv(copied_csv)
            self.assertEqual(columns[: len(original_columns)], original_columns)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["amplitude"], "0.75")
            self.assertEqual(rows[0]["fwhm_x"], "10.0")
            self.assertEqual(rows[0]["fwhm_y"], "14.0")
            self.assertEqual(rows[0]["fwhm_source"], "stored")
            self.assertEqual(rows[0]["fwhm_mean"], "12.0")
            self.assertEqual(
                (destination_root / "sample" / "eksport" / "notes.bin").read_bytes(),
                unrelated_bytes,
            )
            self.assertEqual(self._snapshot_files(source_root), source_before)
            self.assertEqual(result.destination_root, destination_root.resolve())
            self.assertFalse(result.reused)

    def test_sigma_rows_gain_fwhm_while_unrecoverable_rows_remain(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_root = workspace / "data"
            csv_path = source_root / "sample" / "results" / "full_analysis_per_spot.csv"
            original_columns = [
                "track_id",
                "frame_id",
                "sigma_x",
                "sigma_x_error",
                "sigma_y",
                "sigma_y_error",
                "comment",
            ]
            self._write_csv(
                csv_path,
                original_columns,
                [
                    {
                        "track_id": 1,
                        "frame_id": 7,
                        "sigma_x": 2.0,
                        "sigma_x_error": 0.1,
                        "sigma_y": 3.0,
                        "sigma_y_error": 0.2,
                        "comment": "recoverable",
                    },
                    {
                        "track_id": 2,
                        "frame_id": 7,
                        "sigma_x": "",
                        "sigma_x_error": "",
                        "sigma_y": "",
                        "sigma_y_error": "",
                        "comment": "keep me",
                    },
                ],
            )
            destination_root = workspace / "publication_exports"

            create_publication_export(source_root, destination_root)

            copied_csv = destination_root / "sample" / "results" / csv_path.name
            columns, rows = self._read_csv(copied_csv)
            factor = 2.0 * math.sqrt(2.0 * math.log(2.0))
            self.assertEqual(columns[: len(original_columns)], original_columns)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["fwhm_source"], "calculated_from_sigma")
            self.assertAlmostEqual(float(rows[0]["fwhm_x"]), 2.0 * factor)
            self.assertAlmostEqual(float(rows[0]["fwhm_y"]), 3.0 * factor)
            self.assertAlmostEqual(float(rows[0]["fwhm_x_error"]), 0.1 * factor)
            self.assertAlmostEqual(float(rows[0]["fwhm_y_error"]), 0.2 * factor)
            self.assertEqual(rows[1]["comment"], "keep me")
            self.assertEqual(rows[1]["fwhm_source"], "missing")
            for column in (
                "fwhm_x",
                "fwhm_x_error",
                "fwhm_y",
                "fwhm_y_error",
                "fwhm_mean",
                "fwhm_geometric",
            ):
                self.assertEqual(rows[1][column], "")

    def test_positions_session_enriches_legacy_csv_and_copied_session(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_root = workspace / "data"
            sample_dir = source_root / "sample"
            csv_path = sample_dir / "eksport" / "positions_analysis_results.csv"
            self._write_csv(
                csv_path,
                ["frame_id", "group_id", "track_id", "x", "y"],
                [
                    {"frame_id": 19, "group_id": 1, "track_id": 1, "x": 10, "y": 20},
                    {"frame_id": 19, "group_id": 1, "track_id": 2, "x": 30, "y": 40},
                ],
            )
            session_path = sample_dir / "analysis.positions.json"
            session_path.write_text(
                json.dumps(
                    {
                        "type": "positions_analysis_session",
                        "positions_analysis_results": {
                            "fit_points_by_frame": {
                                "19": {
                                    "1": {
                                        "fit_ok": True,
                                        "quality_status": "ok",
                                        "fit_results": {
                                            "sigma_x": 4.0,
                                            "sigma_y": 5.0,
                                            "fwhm_x": 9.42,
                                            "fwhm_y": 11.78,
                                            "fwhm_x_error": 0.4,
                                            "fwhm_y_error": 0.5,
                                        },
                                    },
                                    "2": {
                                        "fit_ok": False,
                                        "quality_status": "failed",
                                        "fit_results": {
                                            "sigma_x": 99.0,
                                            "sigma_y": 98.0,
                                            "fwhm_x": 97.0,
                                            "fwhm_y": 96.0,
                                        },
                                    },
                                }
                            }
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            source_before = self._snapshot_files(source_root)
            destination_root = workspace / "publication_exports"

            create_publication_export(source_root, destination_root)

            columns, rows = self._read_csv(
                destination_root / "sample" / "eksport" / "positions_analysis_results.csv"
            )
            self.assertEqual(len(rows), 2)
            self.assertNotIn("sigma_x", columns)
            self.assertNotIn("sigma_y", columns)
            self.assertEqual(rows[0]["fwhm_x"], "9.42")
            self.assertEqual(rows[0]["fwhm_y"], "11.78")
            self.assertEqual(rows[0]["fwhm_source"], "stored")
            self.assertEqual(rows[1]["fwhm_source"], "missing")
            copied_session = json.loads(
                (destination_root / "sample" / "analysis.positions.json").read_text(
                    encoding="utf-8"
                )
            )
            copied_fit_results = copied_session["positions_analysis_results"][
                "fit_points_by_frame"
            ]["19"]["1"]["fit_results"]
            self.assertEqual(copied_fit_results["fwhm_x"], 9.42)
            self.assertEqual(copied_fit_results["fwhm_y"], 11.78)
            self.assertEqual(copied_fit_results["fwhm_source"], "stored")
            self.assertEqual(copied_fit_results["fwhm_mean"], 10.6)
            failed_fit_results = copied_session["positions_analysis_results"][
                "fit_points_by_frame"
            ]["19"]["2"]["fit_results"]
            self.assertEqual(failed_fit_results["sigma_x"], 99.0)
            self.assertEqual(failed_fit_results["sigma_y"], 98.0)
            self.assertIsNone(failed_fit_results["fwhm_x"])
            self.assertIsNone(failed_fit_results["fwhm_y"])
            self.assertIsNone(failed_fit_results["fwhm_x_error"])
            self.assertIsNone(failed_fit_results["fwhm_y_error"])
            self.assertEqual(failed_fit_results["fwhm_source"], "missing")
            self.assertEqual(self._snapshot_files(source_root), source_before)

    def test_lts_analysis_results_are_enriched_without_needing_a_csv(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_root = workspace / "data"
            session_path = source_root / "sample" / "analysis.lts.json"
            session_path.parent.mkdir(parents=True)
            session_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "metadata_masks": [{"x": 1, "y": 2, "w": 3, "h": 4}],
                        "tracks": [
                            {
                                "track_id": 8,
                                "analysis_results": {
                                    "3": {
                                        "sigma_x": 2.5,
                                        "sigma_y": 3.5,
                                        "sigma_x_error": 0.1,
                                        "sigma_y_error": 0.2,
                                    }
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            destination_root = workspace / "publication_exports"

            create_publication_export(source_root, destination_root)

            copied_session = json.loads(
                (destination_root / "sample" / "analysis.lts.json").read_text(
                    encoding="utf-8"
                )
            )
            result = copied_session["tracks"][0]["analysis_results"]["3"]
            factor = 2.0 * math.sqrt(2.0 * math.log(2.0))
            self.assertEqual(copied_session["metadata_masks"], [{"x": 1, "y": 2, "w": 3, "h": 4}])
            self.assertAlmostEqual(result["fwhm_x"], 2.5 * factor)
            self.assertAlmostEqual(result["fwhm_y"], 3.5 * factor)
            self.assertAlmostEqual(result["fwhm_x_error"], 0.1 * factor)
            self.assertAlmostEqual(result["fwhm_y_error"], 0.2 * factor)
            self.assertEqual(result["fwhm_source"], "calculated_from_sigma")
            manifest = json.loads(
                (destination_root / "export_manifest.json").read_text(encoding="utf-8")
            )
            session_entry = next(
                entry
                for entry in manifest["files"]
                if entry["relative_source_path"] == "sample/analysis.lts.json"
            )
            self.assertEqual(session_entry["row_count"], 1)
            self.assertEqual(
                session_entry["fwhm_provenance"],
                {"stored": 0, "calculated_from_sigma": 1, "missing": 0},
            )
            self.assertEqual(session_entry["missing_fwhm_count"], 0)

    def test_per_frame_summary_and_manifest_report_provenance(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_root = workspace / "data"
            csv_path = source_root / "sample" / "eksport" / "full_analysis_per_spot.csv"
            self._write_csv(
                csv_path,
                [
                    "track_id",
                    "frame_id",
                    "temperature",
                    "energy",
                    "coverage",
                    "fwhm_x",
                    "fwhm_y",
                ],
                [
                    {
                        "track_id": 1,
                        "frame_id": 5,
                        "temperature": 300,
                        "energy": 110,
                        "coverage": 0,
                        "fwhm_x": 10,
                        "fwhm_y": 14,
                    },
                    {
                        "track_id": 2,
                        "frame_id": 5,
                        "temperature": 300,
                        "energy": 110,
                        "coverage": 0,
                        "fwhm_x": "",
                        "fwhm_y": "",
                    },
                ],
            )
            source_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            destination_root = workspace / "publication_exports"

            create_publication_export(source_root, destination_root)

            _, summary_rows = self._read_csv(
                destination_root / "sample" / "eksport" / "fwhm_per_frame.csv"
            )
            self.assertEqual(len(summary_rows), 1)
            self.assertEqual(summary_rows[0]["frame_id"], "5")
            self.assertEqual(summary_rows[0]["temperature"], "300")
            self.assertEqual(summary_rows[0]["fwhm_count"], "1")
            self.assertEqual(summary_rows[0]["fwhm_missing_count"], "1")
            self.assertEqual(summary_rows[0]["fwhm_x_mean"], "10.0")
            self.assertEqual(summary_rows[0]["fwhm_y_mean"], "14.0")
            self.assertEqual(summary_rows[0]["fwhm_mean"], "12.0")
            manifest = json.loads(
                (destination_root / "export_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["version"], 1)
            self.assertEqual(manifest["source_root"], str(source_root.resolve()))
            self.assertEqual(len(manifest["input_fingerprint"]), 64)
            csv_entry = next(
                entry
                for entry in manifest["files"]
                if entry["relative_source_path"] == "sample/eksport/full_analysis_per_spot.csv"
            )
            self.assertEqual(csv_entry["source_sha256"], source_hash)
            self.assertEqual(csv_entry["row_count"], 2)
            self.assertEqual(
                csv_entry["fwhm_provenance"],
                {"stored": 1, "calculated_from_sigma": 0, "missing": 1},
            )
            self.assertEqual(csv_entry["missing_fwhm_count"], 1)

    def test_existing_output_is_reused_only_for_unchanged_inputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_root = workspace / "data"
            csv_path = source_root / "sample" / "eksport" / "full_analysis_per_spot.csv"
            self._write_csv(
                csv_path,
                ["track_id", "frame_id", "fwhm_x", "fwhm_y"],
                [{"track_id": 1, "frame_id": 1, "fwhm_x": 8, "fwhm_y": 10}],
            )
            destination_root = workspace / "publication_exports"
            create_publication_export(source_root, destination_root)
            destination_before = self._snapshot_files(destination_root)

            reused = create_publication_export(source_root, destination_root)

            self.assertTrue(reused.reused)
            self.assertEqual(self._snapshot_files(destination_root), destination_before)

            self._write_csv(
                csv_path,
                ["track_id", "frame_id", "fwhm_x", "fwhm_y"],
                [{"track_id": 1, "frame_id": 1, "fwhm_x": 9, "fwhm_y": 11}],
            )
            with self.assertRaises(FileExistsError):
                create_publication_export(source_root, destination_root)
            self.assertEqual(self._snapshot_files(destination_root), destination_before)

    def test_existing_output_with_modified_copy_is_not_reused(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_root = workspace / "data"
            export_dir = source_root / "sample" / "eksport"
            csv_path = export_dir / "full_analysis_per_spot.csv"
            self._write_csv(
                csv_path,
                ["track_id", "frame_id", "fwhm_x", "fwhm_y"],
                [{"track_id": 1, "frame_id": 1, "fwhm_x": 8, "fwhm_y": 10}],
            )
            (export_dir / "notes.bin").write_bytes(b"original")
            destination_root = workspace / "publication_exports"
            create_publication_export(source_root, destination_root)
            copied_note = destination_root / "sample" / "eksport" / "notes.bin"
            copied_note.write_bytes(b"modified")

            with self.assertRaises(FileExistsError):
                create_publication_export(source_root, destination_root)

            self.assertEqual(copied_note.read_bytes(), b"modified")

    def test_failed_fit_does_not_republish_stale_width_values(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_root = workspace / "data"
            csv_path = source_root / "sample" / "eksport" / "positions_analysis_results.csv"
            self._write_csv(
                csv_path,
                [
                    "frame_id",
                    "track_id",
                    "fit_ok",
                    "quality_status",
                    "sigma_x",
                    "sigma_y",
                    "fwhm_x",
                    "fwhm_y",
                ],
                [
                    {
                        "frame_id": 1,
                        "track_id": 1,
                        "fit_ok": False,
                        "quality_status": "failed",
                        "sigma_x": 4,
                        "sigma_y": 5,
                        "fwhm_x": 9.42,
                        "fwhm_y": 11.78,
                    }
                ],
            )
            destination_root = workspace / "publication_exports"

            create_publication_export(source_root, destination_root)

            _, rows = self._read_csv(
                destination_root / "sample" / "eksport" / csv_path.name
            )
            self.assertEqual(rows[0]["fit_ok"], "False")
            self.assertEqual(rows[0]["quality_status"], "failed")
            self.assertEqual(rows[0]["fwhm_source"], "missing")
            self.assertEqual(rows[0]["sigma_x"], "4")
            self.assertEqual(rows[0]["sigma_y"], "5")
            for column in (
                "fwhm_x",
                "fwhm_x_error",
                "fwhm_y",
                "fwhm_y_error",
                "fwhm_mean",
                "fwhm_geometric",
            ):
                self.assertEqual(rows[0][column], "")

    def test_destination_inside_source_tree_is_rejected_without_writes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "data"
            csv_path = source_root / "sample" / "eksport" / "full_analysis_per_spot.csv"
            self._write_csv(
                csv_path,
                ["track_id", "frame_id", "fwhm_x", "fwhm_y"],
                [{"track_id": 1, "frame_id": 1, "fwhm_x": 8, "fwhm_y": 10}],
            )
            source_before = self._snapshot_files(source_root)
            invalid_destination = source_root / "publication_exports"

            with self.assertRaises(ValueError):
                create_publication_export(source_root, invalid_destination)

            self.assertFalse(invalid_destination.exists())
            self.assertEqual(self._snapshot_files(source_root), source_before)

    def test_standalone_export_csv_does_not_copy_neighboring_raw_tiffs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_root = workspace / "data"
            sample_dir = source_root / "sample"
            csv_path = sample_dir / "positions_analysis_results.csv"
            self._write_csv(
                csv_path,
                ["frame_id", "track_id", "fwhm_x", "fwhm_y"],
                [{"frame_id": 1, "track_id": 1, "fwhm_x": 8, "fwhm_y": 10}],
            )
            raw_tiff = sample_dir / "Ru0001G_0001_T0034_E0077_C0000.tiff"
            raw_tiff.write_bytes(b"raw-image-data")
            (sample_dir / "pattern_center_data.csv").write_bytes(
                b"frame_id,center_x,center_y\n1,100,120\n"
            )
            auxiliary_export = sample_dir / "analysis_assets" / "track_legend.csv"
            auxiliary_export.parent.mkdir()
            auxiliary_export.write_bytes(b"track_id,label\n1,A\n")
            visualization = sample_dir / "track_visualizations" / "frame_0001.png"
            visualization.parent.mkdir()
            visualization.write_bytes(b"publication-visualization")
            session_path = sample_dir / "analysis.lts.json"
            session_path.write_text(json.dumps({"tracks": []}), encoding="utf-8")
            destination_root = workspace / "publication_exports"

            create_publication_export(source_root, destination_root)

            self.assertTrue((destination_root / "sample" / csv_path.name).is_file())
            self.assertTrue((destination_root / "sample" / session_path.name).is_file())
            self.assertEqual(
                (
                    destination_root
                    / "sample"
                    / "track_visualizations"
                    / visualization.name
                ).read_bytes(),
                b"publication-visualization",
            )
            self.assertEqual(
                (destination_root / "sample" / "pattern_center_data.csv").read_bytes(),
                b"frame_id,center_x,center_y\n1,100,120\n",
            )
            self.assertEqual(
                (
                    destination_root
                    / "sample"
                    / "analysis_assets"
                    / "track_legend.csv"
                ).read_bytes(),
                b"track_id,label\n1,A\n",
            )
            self.assertFalse((destination_root / "sample" / raw_tiff.name).exists())

    def test_named_export_directory_is_copied_without_a_recognized_csv(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_root = workspace / "data"
            export_dir = source_root / "sample" / "eksport"
            export_dir.mkdir(parents=True)
            (export_dir / "summary.csv").write_bytes(b"frame,value\n1,2\n")
            (export_dir / "notes.bin").write_bytes(b"complete-export")
            destination_root = workspace / "publication_exports"

            create_publication_export(source_root, destination_root)

            self.assertEqual(
                (destination_root / "sample" / "eksport" / "summary.csv").read_bytes(),
                b"frame,value\n1,2\n",
            )
            self.assertEqual(
                (destination_root / "sample" / "eksport" / "notes.bin").read_bytes(),
                b"complete-export",
            )

    def test_generated_per_frame_file_never_overwrites_existing_exports(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_root = workspace / "data"
            export_dir = source_root / "sample" / "eksport"
            csv_path = export_dir / "full_analysis_per_spot.csv"
            self._write_csv(
                csv_path,
                ["frame_id", "track_id", "fwhm_x", "fwhm_y"],
                [{"frame_id": 1, "track_id": 1, "fwhm_x": 8, "fwhm_y": 10}],
            )
            (export_dir / "fwhm_per_frame.csv").write_bytes(b"original-default")
            (export_dir / "fwhm_per_frame_full_analysis_per_spot.csv").write_bytes(
                b"original-stem"
            )
            destination_root = workspace / "publication_exports"

            create_publication_export(source_root, destination_root)

            copied_export = destination_root / "sample" / "eksport"
            self.assertEqual(
                (copied_export / "fwhm_per_frame.csv").read_bytes(),
                b"original-default",
            )
            self.assertEqual(
                (
                    copied_export / "fwhm_per_frame_full_analysis_per_spot.csv"
                ).read_bytes(),
                b"original-stem",
            )
            generated_path = (
                copied_export / "fwhm_per_frame_full_analysis_per_spot_2.csv"
            )
            self.assertTrue(generated_path.is_file())
            manifest = json.loads(
                (destination_root / "export_manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn(
                "sample/eksport/fwhm_per_frame_full_analysis_per_spot_2.csv",
                manifest["generated_files"],
            )

    def test_conflicting_sibling_sessions_do_not_guess_fwhm_source(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_root = workspace / "data"
            sample_dir = source_root / "sample"
            csv_path = sample_dir / "eksport" / "positions_analysis_results.csv"
            self._write_csv(
                csv_path,
                ["frame_id", "track_id", "x", "y"],
                [{"frame_id": 1, "track_id": 1, "x": 20, "y": 30}],
            )

            def session_payload(fwhm_x: float, fwhm_y: float) -> dict:
                return {
                    "positions_analysis_results": {
                        "fit_points_by_frame": {
                            "1": {
                                "1": {
                                    "fit_ok": True,
                                    "fit_results": {
                                        "fwhm_x": fwhm_x,
                                        "fwhm_y": fwhm_y,
                                    },
                                }
                            }
                        }
                    }
                }

            (sample_dir / "first.positions.json").write_text(
                json.dumps(session_payload(8, 10)), encoding="utf-8"
            )
            (sample_dir / "second.positions.json").write_text(
                json.dumps(session_payload(18, 20)), encoding="utf-8"
            )
            destination_root = workspace / "publication_exports"

            create_publication_export(source_root, destination_root)

            _, rows = self._read_csv(
                destination_root / "sample" / "eksport" / csv_path.name
            )
            self.assertEqual(rows[0]["fwhm_source"], "missing")
            self.assertEqual(rows[0]["fwhm_x"], "")
            self.assertEqual(rows[0]["fwhm_y"], "")

    def test_sibling_session_with_more_matching_keys_is_still_ambiguous(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_root = workspace / "data"
            sample_dir = source_root / "sample"
            csv_path = sample_dir / "eksport" / "positions_analysis_results.csv"
            self._write_csv(
                csv_path,
                ["frame_id", "track_id", "x", "y"],
                [
                    {"frame_id": 1, "track_id": 1, "x": 20, "y": 30},
                    {"frame_id": 1, "track_id": 2, "x": 40, "y": 50},
                ],
            )

            def session_payload(track_ids: list[int], offset: float) -> dict:
                return {
                    "positions_analysis_results": {
                        "fit_points_by_frame": {
                            "1": {
                                str(track_id): {
                                    "fit_ok": True,
                                    "fit_results": {
                                        "fwhm_x": offset + track_id,
                                        "fwhm_y": offset + track_id + 2,
                                    },
                                }
                                for track_id in track_ids
                            }
                        }
                    }
                }

            (sample_dir / "short.positions.json").write_text(
                json.dumps(session_payload([1], 10)), encoding="utf-8"
            )
            (sample_dir / "long.positions.json").write_text(
                json.dumps(session_payload([1, 2], 100)), encoding="utf-8"
            )
            destination_root = workspace / "publication_exports"

            create_publication_export(source_root, destination_root)

            _, rows = self._read_csv(
                destination_root / "sample" / "eksport" / csv_path.name
            )
            self.assertEqual([row["fwhm_source"] for row in rows], ["missing", "missing"])
            self.assertEqual([row["fwhm_x"] for row in rows], ["", ""])

    def test_partial_fwhm_record_is_published_as_missing_with_empty_widths(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_root = workspace / "data"
            csv_path = source_root / "sample" / "eksport" / "full_analysis_per_spot.csv"
            self._write_csv(
                csv_path,
                ["frame_id", "track_id", "fwhm_x", "fwhm_y"],
                [{"frame_id": 1, "track_id": 1, "fwhm_x": 8, "fwhm_y": ""}],
            )
            destination_root = workspace / "publication_exports"

            create_publication_export(source_root, destination_root)

            _, rows = self._read_csv(
                destination_root / "sample" / "eksport" / csv_path.name
            )
            self.assertEqual(rows[0]["fwhm_source"], "missing")
            self.assertEqual(rows[0]["fwhm_x"], "")
            self.assertEqual(rows[0]["fwhm_y"], "")


if __name__ == "__main__":
    unittest.main()
