import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from trackleed.io.temperature_series_plots import (
    export_temperature_series_plots,
)


class TemperatureSeriesPlotTests(unittest.TestCase):
    def test_export_writes_fwhm_and_amplitude_plots_against_temperature(
        self,
    ) -> None:
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
                        "fwhm_effective",
                        "fwhm_profile_x",
                        "fwhm_profile_y",
                        "amplitude",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "track_id": 1,
                            "frame_id": 1,
                            "temperature": 25,
                            "fwhm_effective": 9,
                            "fwhm_profile_x": 8,
                            "fwhm_profile_y": 10,
                            "amplitude": 0.8,
                        },
                        {
                            "track_id": 1,
                            "frame_id": 2,
                            "temperature": 100,
                            "fwhm_effective": 8.5,
                            "fwhm_profile_x": 7.5,
                            "fwhm_profile_y": 9.5,
                            "amplitude": 0.7,
                        },
                        {
                            "track_id": 2,
                            "frame_id": 1,
                            "temperature": 25,
                            "fwhm_effective": 10,
                            "fwhm_profile_x": 9,
                            "fwhm_profile_y": 11,
                            "amplitude": 0.6,
                        },
                    ]
                )

            result = export_temperature_series_plots(
                source_csv,
                root / "temperature_plots",
            )

            self.assertEqual(
                {path.name for path in result.plot_paths},
                {
                    "fwhm_effective_vs_temperature_by_track.png",
                    "fwhm_profile_x_y_vs_temperature_by_track.png",
                    "amplitude_vs_temperature_by_track.png",
                },
            )
            for plot_path in result.plot_paths:
                with Image.open(plot_path) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreaterEqual(image.width, 800)
                    self.assertGreaterEqual(image.height, 500)

            manifest = json.loads(
                result.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["x_axis"], "temperature")
            self.assertEqual(manifest["track_count"], 2)
            self.assertEqual(manifest["frame_count"], 2)
            self.assertEqual(manifest["row_count"], 3)


if __name__ == "__main__":
    unittest.main()
