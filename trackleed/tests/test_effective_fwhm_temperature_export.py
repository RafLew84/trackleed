import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from trackleed.io.effective_fwhm_temperature_export import (
    export_effective_fwhm_temperature,
)


class EffectiveFwhmTemperatureExportTests(unittest.TestCase):
    def test_export_writes_branch_plots_and_axis_free_source_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fit_csv = root / "full_analysis_per_spot.csv"
            series_csv = root / "legacy_plot_data.csv"
            self._write_csv(
                fit_csv,
                [
                    {
                        "track_id": 1,
                        "frame_id": 10,
                        "fwhm_minor": 8,
                        "fwhm_minor_error": 0.1,
                        "fwhm_major": 18,
                        "fwhm_major_error": 0.2,
                        "fwhm_effective": 12,
                        "fwhm_effective_error": 0.08,
                    },
                    {
                        "track_id": 1,
                        "frame_id": 11,
                        "fwhm_minor": 9,
                        "fwhm_minor_error": 0.1,
                        "fwhm_major": 16,
                        "fwhm_major_error": 0.2,
                        "fwhm_effective": 12,
                        "fwhm_effective_error": 0.08,
                    },
                    {
                        "track_id": 1,
                        "frame_id": 12,
                        "fwhm_minor": 8.1,
                        "fwhm_minor_error": 0.1,
                        "fwhm_major": 17.7777777778,
                        "fwhm_major_error": 0.2,
                        "fwhm_effective": 12,
                        "fwhm_effective_error": 0.08,
                    },
                    {
                        "track_id": 1,
                        "frame_id": 13,
                        "fwhm_minor": "",
                        "fwhm_minor_error": "",
                        "fwhm_major": "",
                        "fwhm_major_error": "",
                        "fwhm_effective": "",
                        "fwhm_effective_error": "",
                    },
                ],
            )
            self._write_csv(
                series_csv,
                [
                    self._series_row(10, 0, 20, "heating"),
                    self._series_row(11, 1, 100, "peak_hold"),
                    self._series_row(12, 2, 40, "cooling"),
                    {
                        **self._series_row(13, 3, 30, "cooling"),
                        "plotted": "False",
                        "quality_note": "missing",
                        "missing_reason": "missing_fit",
                    },
                ],
            )

            result = export_effective_fwhm_temperature(
                fit_csv=fit_csv,
                series_csv=series_csv,
                output_dir=root / "export",
            )

            self.assertEqual(
                {path.name for path in result.plot_paths},
                {
                    "fwhm_vs_temperature_heating.png",
                    "fwhm_vs_temperature_heating.pdf",
                    "fwhm_vs_temperature_heating.svg",
                    "fwhm_vs_temperature_cooling.png",
                    "fwhm_vs_temperature_cooling.pdf",
                    "fwhm_vs_temperature_cooling.svg",
                },
            )
            with Image.open(root / "export/fwhm_vs_temperature_heating.png") as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreaterEqual(image.width, 1200)

            with result.data_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = reader.fieldnames or []

            self.assertNotIn("fwhm_x_px", fieldnames)
            self.assertNotIn("fwhm_y_px", fieldnames)
            self.assertEqual(
                [
                    "track_id",
                    "frame_id",
                    "frame_order",
                    "temperature_C",
                    "energy_eV",
                    "coverage",
                    "thermal_branch",
                    "plotted",
                    "quality_note",
                    "missing_reason",
                    "fwhm_minor_px",
                    "fwhm_minor_error_px",
                    "fwhm_major_px",
                    "fwhm_major_error_px",
                    "fwhm_px",
                    "fwhm_error_px",
                    "r_squared",
                ],
                fieldnames,
            )
            self.assertEqual(rows[1]["thermal_branch"], "heating")
            self.assertEqual(float(rows[0]["fwhm_px"]), 12.0)
            self.assertEqual(rows[-1]["fwhm_px"], "")
            self.assertEqual(rows[-1]["plotted"], "False")

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["fwhm_definition"], "sqrt(fwhm_minor * fwhm_major)")
            self.assertEqual(manifest["branches"], {"cooling": 2, "heating": 2})
            self.assertNotIn("series_source_csv", manifest)
            self.assertNotIn("effective", " ".join(path.name for path in result.plot_paths))

    def test_export_rejects_effective_width_inconsistent_with_geometric_mean(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fit_csv = root / "fits.csv"
            series_csv = root / "series.csv"
            self._write_csv(
                fit_csv,
                [
                    {
                        "track_id": 1,
                        "frame_id": 1,
                        "fwhm_minor": 4,
                        "fwhm_minor_error": 0.1,
                        "fwhm_major": 16,
                        "fwhm_major_error": 0.2,
                        "fwhm_effective": 10,
                        "fwhm_effective_error": 0.1,
                    }
                ],
            )
            self._write_csv(
                series_csv,
                [self._series_row(1, 0, 20, "heating")],
            )

            with self.assertRaisesRegex(ValueError, "geometric mean"):
                export_effective_fwhm_temperature(
                    fit_csv=fit_csv,
                    series_csv=series_csv,
                    output_dir=root / "export",
                )

    def test_missing_branch_is_derived_from_temperature_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fit_csv = root / "fits.csv"
            series_csv = root / "series.csv"
            fit_rows = []
            series_rows = []
            for frame_id, temperature in enumerate((20, 50, 100), start=1):
                for track_id in (1, 2):
                    fit_rows.append(
                        {
                            "track_id": track_id,
                            "frame_id": frame_id,
                            "fwhm_minor": 8,
                            "fwhm_minor_error": 0.1,
                            "fwhm_major": 18,
                            "fwhm_major_error": 0.2,
                            "fwhm_effective": math.sqrt(8 * 18),
                            "fwhm_effective_error": 0.1,
                        }
                    )
                    row = self._series_row(
                        frame_id,
                        frame_id - 1,
                        temperature,
                        "",
                    )
                    row["track_id"] = track_id
                    row.pop("thermal_branch")
                    row.pop("frame_order")
                    series_rows.append(row)
            self._write_csv(fit_csv, fit_rows)
            self._write_csv(series_csv, series_rows)

            result = export_effective_fwhm_temperature(
                fit_csv=fit_csv,
                series_csv=series_csv,
                output_dir=root / "export",
            )

            self.assertEqual(
                {path.name for path in result.plot_paths},
                {
                    "fwhm_vs_temperature_heating.png",
                    "fwhm_vs_temperature_heating.pdf",
                    "fwhm_vs_temperature_heating.svg",
                },
            )
            with result.data_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["thermal_branch"] for row in rows}, {"heating"})

    @staticmethod
    def _series_row(
        frame_id: int,
        frame_order: int,
        temperature: int,
        branch: str,
    ) -> dict[str, object]:
        return {
            "track_id": 1,
            "frame_id": frame_id,
            "frame_order": frame_order,
            "temperature_C": temperature,
            "energy_eV": 100,
            "coverage": 0,
            "thermal_branch": branch,
            "plotted": "True",
            "quality_note": "accepted",
            "missing_reason": "",
            "fwhm_x_px": 12,
            "fwhm_y_px": 12,
            "r_squared": 0.99,
        }

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        fieldnames = list(rows[0])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
