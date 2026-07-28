import csv
import json
from pathlib import Path
import tempfile
import unittest

from trackleed.io.fwhm_temperature_quality_plot import (
    QualityFilteredFwhmPlotResult,
    QualityFwhmPlotResult,
    export_quality_filtered_fwhm_temperature_plot,
    export_quality_marked_fwhm_temperature_plot,
)


class FwhmTemperatureQualityPlotTest(unittest.TestCase):
    def test_filtered_result_name_remains_compatible(self):
        self.assertIs(QualityFilteredFwhmPlotResult, QualityFwhmPlotResult)

    def test_export_preserves_sources_and_records_quality_mask(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fwhm_path = root / "fwhm.csv"
            amplitude_path = root / "amplitude.csv"
            output_stem = root / "filtered"

            self._write_csv(
                fwhm_path,
                [
                    {
                        "track_id": "1",
                        "frame_id": "0",
                        "frame_order": "0",
                        "temperature_C": "20",
                        "thermal_branch": "heating",
                        "plotted": "True",
                        "quality_note": "accepted",
                        "missing_reason": "",
                        "fwhm_x_px": "8.0",
                        "fwhm_x_error_px": "0.1",
                        "fwhm_y_px": "9.0",
                        "fwhm_y_error_px": "0.1",
                        "r_squared": "0.95",
                    },
                    {
                        "track_id": "1",
                        "frame_id": "1",
                        "frame_order": "1",
                        "temperature_C": "21",
                        "thermal_branch": "heating",
                        "plotted": "True",
                        "quality_note": "accepted",
                        "missing_reason": "",
                        "fwhm_x_px": "8.2",
                        "fwhm_x_error_px": "0.1",
                        "fwhm_y_px": "9.2",
                        "fwhm_y_error_px": "0.1",
                        "r_squared": "0.95",
                    },
                ],
            )
            self._write_csv(
                amplitude_path,
                [
                    {
                        "track_id": "1",
                        "frame_id": "0",
                        "plotted": "True",
                        "amplitude_au": "0.20",
                        "amplitude_error_au": "0.01",
                        "r_squared": "0.95",
                    },
                    {
                        "track_id": "1",
                        "frame_id": "1",
                        "plotted": "True",
                        "amplitude_au": "0.08",
                        "amplitude_error_au": "0.01",
                        "r_squared": "0.95",
                    },
                ],
            )
            fwhm_before = fwhm_path.read_bytes()
            amplitude_before = amplitude_path.read_bytes()

            result = export_quality_filtered_fwhm_temperature_plot(
                fwhm_path,
                amplitude_path,
                output_stem,
                min_amplitude=0.10,
                min_r_squared=0.90,
            )

            self.assertEqual(fwhm_path.read_bytes(), fwhm_before)
            self.assertEqual(amplitude_path.read_bytes(), amplitude_before)
            self.assertTrue(result.plot_path.is_file())
            self.assertTrue(result.data_path.is_file())
            self.assertTrue(result.manifest_path.is_file())
            self.assertEqual(result.accepted_count, 1)
            self.assertEqual(result.rejected_count, 1)

            with result.data_path.open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["quality_valid"], "True")
            self.assertEqual(rows[0]["quality_reason"], "accepted")
            self.assertEqual(rows[1]["quality_valid"], "False")
            self.assertEqual(rows[1]["quality_reason"], "amplitude_below_0.1")

    def test_marked_export_keeps_rejected_values_as_gray_markers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fwhm_path = root / "fwhm.csv"
            amplitude_path = root / "amplitude.csv"
            output_stem = root / "marked"

            self._write_csv(
                fwhm_path,
                [
                    {
                        "track_id": "1",
                        "frame_id": "0",
                        "frame_order": "0",
                        "temperature_C": "20",
                        "thermal_branch": "heating",
                        "plotted": "True",
                        "quality_note": "accepted",
                        "missing_reason": "",
                        "fwhm_x_px": "8.0",
                        "fwhm_x_error_px": "0.1",
                        "fwhm_y_px": "9.0",
                        "fwhm_y_error_px": "0.1",
                        "r_squared": "0.95",
                    },
                    {
                        "track_id": "1",
                        "frame_id": "1",
                        "frame_order": "1",
                        "temperature_C": "21",
                        "thermal_branch": "heating",
                        "plotted": "True",
                        "quality_note": "accepted",
                        "missing_reason": "",
                        "fwhm_x_px": "12.0",
                        "fwhm_x_error_px": "0.2",
                        "fwhm_y_px": "13.0",
                        "fwhm_y_error_px": "0.2",
                        "r_squared": "0.95",
                    },
                ],
            )
            self._write_csv(
                amplitude_path,
                [
                    {
                        "track_id": "1",
                        "frame_id": "0",
                        "plotted": "True",
                        "amplitude_au": "0.20",
                        "amplitude_error_au": "0.01",
                        "r_squared": "0.95",
                    },
                    {
                        "track_id": "1",
                        "frame_id": "1",
                        "plotted": "True",
                        "amplitude_au": "0.08",
                        "amplitude_error_au": "0.01",
                        "r_squared": "0.95",
                    },
                ],
            )
            fwhm_before = fwhm_path.read_bytes()
            amplitude_before = amplitude_path.read_bytes()

            result = export_quality_marked_fwhm_temperature_plot(
                fwhm_path,
                amplitude_path,
                output_stem,
                min_amplitude=0.10,
                min_r_squared=0.90,
            )

            self.assertEqual(fwhm_path.read_bytes(), fwhm_before)
            self.assertEqual(amplitude_path.read_bytes(), amplitude_before)
            self.assertEqual(result.accepted_count, 1)
            self.assertEqual(result.rejected_count, 1)

            with result.data_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["quality_valid"], "True")
            self.assertEqual(rows[1]["quality_valid"], "False")
            self.assertEqual(rows[1]["quality_reason"], "amplitude_below_0.1")

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["quality_filter"]["rejected_values_are_plotted_as"],
                "gray markers",
            )
            self.assertFalse(manifest["quality_filter"]["interpolation"])

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
