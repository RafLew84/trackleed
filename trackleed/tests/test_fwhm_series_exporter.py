import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts.export_fwhm_series import main as export_fwhm_series_main
from trackleed.io.fwhm_series_exporter import export_fwhm_series
from trackleed.io.fwhm_series_plots import write_fwhm_plots


class FwhmSeriesExporterTests(unittest.TestCase):
    def test_export_canonicalizes_equivalent_axis_orderings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_csv = root / "full_analysis_per_spot.csv"
            with source_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "track_id",
                        "frame_id",
                        "fwhm_x",
                        "fwhm_x_error",
                        "fwhm_y",
                        "fwhm_y_error",
                        "r_squared",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "track_id": 1,
                            "frame_id": 1,
                            "fwhm_x": 8.0,
                            "fwhm_x_error": 0.2,
                            "fwhm_y": 10.0,
                            "fwhm_y_error": 0.3,
                            "r_squared": 0.95,
                        },
                        {
                            "track_id": 1,
                            "frame_id": 2,
                            "fwhm_x": 10.0,
                            "fwhm_x_error": 0.3,
                            "fwhm_y": 8.0,
                            "fwhm_y_error": 0.2,
                            "r_squared": 0.95,
                        },
                    ]
                )
            source_before = source_csv.read_bytes()

            output_dir = root / "fwhm_export"
            export_fwhm_series(source_csv, output_dir)

            with (output_dir / "full_analysis_per_spot_fwhm.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(source_csv.read_bytes(), source_before)

        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertAlmostEqual(float(row["fwhm_minor"]), 8.0)
            self.assertAlmostEqual(float(row["fwhm_major"]), 10.0)
            self.assertAlmostEqual(
                float(row["fwhm_effective"]),
                math.sqrt(80.0),
            )
            self.assertAlmostEqual(float(row["fwhm_anisotropy"]), 1.25)
            self.assertEqual(row["fwhm_valid"], "true")
            self.assertEqual(row["fwhm_quality_reason"], "ok")

    def test_per_frame_export_excludes_low_quality_fits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_csv = root / "full_analysis_per_spot.csv"
            with source_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "track_id",
                        "frame_id",
                        "temperature",
                        "fwhm_x",
                        "fwhm_y",
                        "r_squared",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "track_id": 1,
                            "frame_id": 1,
                            "temperature": 300,
                            "fwhm_x": 8,
                            "fwhm_y": 10,
                            "r_squared": 0.95,
                        },
                        {
                            "track_id": 2,
                            "frame_id": 1,
                            "temperature": 300,
                            "fwhm_x": 20,
                            "fwhm_y": 20,
                            "r_squared": 0.60,
                        },
                        {
                            "track_id": 1,
                            "frame_id": 2,
                            "temperature": 350,
                            "fwhm_x": 30,
                            "fwhm_y": 30,
                            "r_squared": 0.70,
                        },
                    ]
                )

            output_dir = root / "fwhm_export"
            export_fwhm_series(source_csv, output_dir)

            with (output_dir / "fwhm_per_frame.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["fwhm_valid_count"], "1")
        self.assertEqual(rows[0]["fwhm_invalid_count"], "1")
        self.assertAlmostEqual(
            float(rows[0]["fwhm_effective_mean"]),
            math.sqrt(80.0),
        )
        self.assertEqual(rows[1]["fwhm_valid_count"], "0")
        self.assertEqual(rows[1]["fwhm_invalid_count"], "1")
        self.assertEqual(rows[1]["fwhm_effective_mean"], "")

    def test_export_creates_readable_fwhm_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_csv = root / "full_analysis_per_spot.csv"
            with source_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "track_id",
                        "frame_id",
                        "temperature",
                        "fwhm_x",
                        "fwhm_y",
                        "r_squared",
                    ],
                )
                writer.writeheader()
                for frame_id in range(1, 5):
                    writer.writerow(
                        {
                            "track_id": 1,
                            "frame_id": frame_id,
                            "temperature": 300 + frame_id,
                            "fwhm_x": 8 + frame_id,
                            "fwhm_y": 10 + frame_id,
                            "r_squared": 0.95 if frame_id != 3 else 0.70,
                        }
                    )

            output_dir = root / "fwhm_export"
            export_fwhm_series(source_csv, output_dir)

            plot_names = (
                "fwhm_effective_by_track.png",
                "fwhm_x_y_by_track.png",
                "fwhm_anisotropy_by_track.png",
            )
            for plot_name in plot_names:
                with self.subTest(plot_name=plot_name):
                    plot_path = output_dir / plot_name
                    self.assertTrue(plot_path.is_file())
                    with Image.open(plot_path) as image:
                        self.assertEqual(image.format, "PNG")
                        self.assertGreaterEqual(image.width, 800)
                        self.assertGreaterEqual(image.height, 500)

    def test_x_y_plot_uses_raw_fit_axes_and_marks_quality_gaps(self) -> None:
        rows = [
            {
                "track_id": "1",
                "frame_id": "1",
                "fwhm_x": "8",
                "fwhm_y": "10",
                "fwhm_effective": "9",
                "fwhm_anisotropy": "1.25",
                "fwhm_valid": "true",
            },
            {
                "track_id": "1",
                "frame_id": "2",
                "fwhm_x": "30",
                "fwhm_y": "40",
                "fwhm_effective": "35",
                "fwhm_anisotropy": "1.33",
                "fwhm_valid": "false",
            },
        ]

        with patch(
            "trackleed.io.fwhm_series_plots._render_line_plot"
        ) as render:
            write_fwhm_plots(Path("unused"), rows)

        x_y_call = render.call_args_list[1]
        self.assertEqual(
            x_y_call.kwargs["title"],
            "Raw Gaussian fit-axis FWHM by track (X/Y labels may swap)",
        )
        x_series, y_series = x_y_call.kwargs["series"]
        self.assertEqual(x_series.label, "Track 1 X")
        self.assertEqual(y_series.label, "Track 1 Y")
        self.assertEqual(x_series.y_values[0], 8.0)
        self.assertEqual(y_series.y_values[0], 10.0)
        self.assertTrue(math.isnan(x_series.y_values[1]))
        self.assertTrue(math.isnan(y_series.y_values[1]))

    def test_export_reports_quality_reasons_per_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_csv = root / "full_analysis_per_spot.csv"
            with source_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "track_id",
                        "frame_id",
                        "fwhm_x",
                        "fwhm_y",
                        "r_squared",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "track_id": 1,
                            "frame_id": 1,
                            "fwhm_x": 8,
                            "fwhm_y": 10,
                            "r_squared": 0.95,
                        },
                        {
                            "track_id": 1,
                            "frame_id": 2,
                            "fwhm_x": 20,
                            "fwhm_y": 20,
                            "r_squared": 0.60,
                        },
                    ]
                )

            output_dir = root / "fwhm_export"
            export_fwhm_series(source_csv, output_dir)

            with (output_dir / "fwhm_quality_report.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(
            rows,
            [
                {"track_id": "1", "quality_reason": "ok", "count": "1"},
                {
                    "track_id": "1",
                    "quality_reason": "r_squared_below_0.85",
                    "count": "1",
                },
            ],
        )

    def test_cli_creates_a_non_destructive_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_csv = root / "source.csv"
            with source_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "track_id",
                        "frame_id",
                        "fwhm_x",
                        "fwhm_y",
                        "r_squared",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "track_id": 1,
                        "frame_id": 1,
                        "fwhm_x": 8,
                        "fwhm_y": 10,
                        "r_squared": 0.95,
                    }
                )
            output_dir = root / "result"

            exit_code = export_fwhm_series_main(
                [
                    "--source-csv",
                    str(source_csv),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(
                (output_dir / "full_analysis_per_spot_fwhm.csv").is_file()
            )
            with self.assertRaises(FileExistsError):
                export_fwhm_series(source_csv, output_dir)

    def test_high_r_squared_temporal_width_outlier_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_csv = root / "source.csv"
            with source_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "track_id",
                        "frame_id",
                        "fwhm_x",
                        "fwhm_y",
                        "r_squared",
                    ],
                )
                writer.writeheader()
                for frame_id in range(1, 11):
                    writer.writerow(
                        {
                            "track_id": 9,
                            "frame_id": frame_id,
                            "fwhm_x": 8.0,
                            "fwhm_y": 10.0,
                            "r_squared": 0.95,
                        }
                    )
                writer.writerow(
                    {
                        "track_id": 9,
                        "frame_id": 11,
                        "fwhm_x": 40.0,
                        "fwhm_y": 40.0,
                        "r_squared": 0.95,
                    }
                )

            output_dir = root / "result"
            export_fwhm_series(source_csv, output_dir)
            with (output_dir / "full_analysis_per_spot_fwhm.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[-1]["fwhm_valid"], "false")
        self.assertEqual(
            rows[-1]["fwhm_quality_reason"],
            "temporal_fwhm_outlier",
        )


if __name__ == "__main__":
    unittest.main()
