from __future__ import annotations

import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.io.results_exporter import export_full_results
from trackleed.io.session import deserialize_session
from trackleed.logic import track_analysis_compatibility
from trackleed.logic.active_analysis_state import (
    AnalysisResultStatus,
    AnalysisSourceStatus,
)


def _csv_bytes(*lines: str) -> bytes:
    return (os.linesep.join(lines) + os.linesep).encode("utf-8")


SHARED_FIT = {
    "center_x_error": 0.23456,
    "global_center_y": 14.34567,
    "center_y_error": 0.45678,
    "radius_error": 0.11119,
    "volume_error": 1.23456,
    "amplitude_error": 0.12345,
    "sigma_x": 2.34567,
    "sigma_x_error": 0.22229,
    "sigma_y": 3.45678,
    "sigma_y_error": 0.33339,
    "theta": 0.45678,
    "theta_error": 0.04449,
    "offset": 0.56789,
    "offset_error": 0.05559,
    "fwhm_x": 5.67891,
    "fwhm_x_error": 0.66669,
    "fwhm_y": 6.78912,
    "fwhm_y_error": 0.77779,
    "fwhm_profile_x": 5.11115,
    "fwhm_profile_x_error": 0.11115,
    "fwhm_profile_y": 6.11115,
    "fwhm_profile_y_error": 0.21115,
    "fwhm_minor": 5.33335,
    "fwhm_minor_error": 0.31115,
    "fwhm_major": 6.33335,
    "fwhm_major_error": 0.41115,
    "fwhm_effective": 5.77775,
    "fwhm_effective_error": 0.51115,
    "fwhm_anisotropy": 1.22225,
    "fwhm_anisotropy_error": 0.01115,
    "major_axis_angle_deg": 26.12345,
    "major_axis_angle_deg_error": 0.98765,
    "theta_major": 0.45678,
    "theta_major_error": 0.04449,
    "asymmetry": 0.67891,
    "r_squared": 0.98765,
}


def _flat_export_fixture(root: Path) -> tuple[list[Track], dict[int, LeedFile]]:
    source_dir = root / "sources"
    source_dir.mkdir()
    files: dict[int, LeedFile] = {}
    spots: dict[int, SpotROI] = {}
    for frame_id in range(1, 7):
        image_path = source_dir / f"source_{frame_id:04d}.png"
        pixels: np.ndarray = (
            np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) + frame_id
        )
        Image.fromarray(pixels).save(image_path)
        spot = SpotROI(
            7,
            QRect(8 + frame_id * 2, 10, 8, 8),
            12.25 + frame_id * 2,
            14.5,
            1.0,
            "tracking",
        )
        spots[frame_id] = spot
        frame = LeedFile(
            image_path,
            image_path.name,
            False,
            "Cu",
            111,
            "G",
            300.125 + frame_id,
            80.5 + frame_id,
            frame_id,
            0.125,
            [spot],
        )
        if frame_id in (1, 6):
            frame.pattern_center = {
                "center_x": 20.125 + frame_id,
                "center_y": 21.25,
                "error_x": 0.125,
                "error_y": 0.25,
            }
            frame.post_analysis_results = {
                "avg_radius": 12.34567 + frame_id,
                "avg_radius_error": 0.45678,
            }
        files[frame_id] = frame
    track = Track(7, spots)
    track.analysis_results = {
        1: {
            **SHARED_FIT,
            "global_center_x": 13.12345,
            "radius": 9.87654,
            "volume": 123.45678,
            "amplitude": 4.56789,
        },
        2: None,
        6: {
            **SHARED_FIT,
            "global_center_x": 18.12345,
            "radius": 14.87654,
            "volume": 128.45678,
            "amplitude": 9.56789,
        },
    }
    return [track], files


def _legacy_session_payload(source_path: Path) -> dict[str, object]:
    spot = {
        "id": 7,
        "roi": {"x": 8, "y": 9, "w": 10, "h": 11},
        "center_x": 13.0,
        "center_y": 14.0,
        "intensity": 2.0,
        "source": "tracking",
    }
    return {
        "version": 1,
        "data_files": [
            {
                "filepath": str(source_path),
                "filename": source_path.name,
                "is_background": False,
                "element": "Cu",
                "surface": 111,
                "mode": "G",
                "temperature": 300.0,
                "energy": 80.0,
                "id": 1,
                "coverage": None,
                "pattern_center": {"center_x": 20.0, "center_y": 21.0},
                "post_analysis_results": {"average_radius": 12.5},
                "spots": [spot],
            }
        ],
        "background_file": None,
        "metadata_masks": [],
        "tracks": [
            {
                "track_id": 7,
                "spots_by_image_id": {"1": spot},
                "analysis_results": {
                    "1": {"radius": 12.5, "amplitude": 2.25},
                },
            }
        ],
        "spot_settings": {},
        "analysis_circle": None,
        "analysis_ellipse": None,
    }


def _geometry_publications(root: Path):
    tracks = [Track(track_id, {}) for track_id in range(1, 5)]
    files: dict[int, LeedFile] = {}
    yy, xx = np.indices((64, 64), dtype=np.float64)
    offsets = ((-8.0, 0.0), (0.0, 8.0), (8.0, 0.0), (0.0, -8.0))
    for frame_id, center in ((1, (28.0, 29.0)), (2, (32.0, 31.0))):
        pixels: np.ndarray = np.full((64, 64), 0.03, dtype=np.float64)
        spots = []
        for track, (dx, dy) in zip(tracks, offsets, strict=True):
            spot_x, spot_y = center[0] + dx, center[1] + dy
            pixels += np.exp(-((xx - spot_x) ** 2 + (yy - spot_y) ** 2) / 8.0)
            spot = SpotROI(
                track.track_id,
                QRect(int(spot_x) - 5, int(spot_y) - 5, 11, 11),
                spot_x,
                spot_y,
                1.0,
                "tracking",
            )
            track.spots_by_image_id[frame_id] = spot
            spots.append(spot)
        image_path = root / f"geometry_{frame_id:04d}.png"
        Image.fromarray(np.asarray(pixels * 100, dtype=np.uint8)).save(image_path)
        files[frame_id] = LeedFile(
            image_path,
            image_path.name,
            False,
            "Cu",
            111,
            "G",
            300.0 + frame_id,
            80.0,
            frame_id,
            None,
            spots,
        )
    per_frame_publication = track_analysis_compatibility.run_full_analysis_publication(
        tracks,
        {},
        files,
        None,
        lambda _progress: None,
        center_mode="per_frame",
    )
    assert per_frame_publication is not None
    global_median_publication = (
        track_analysis_compatibility.run_geometry_analysis_publication(
            tracks,
            files,
            per_frame_publication.fit_set,
            lambda _progress: None,
            center_mode="global_median",
            previous_geometry_result=per_frame_publication.geometry_result,
        )
    )
    assert global_median_publication is not None
    return per_frame_publication, global_median_publication


class AnalysisExportCompatibilityTests(unittest.TestCase):
    def test_flat_view_preserves_csv_bytes_and_png_meaning(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "export"
            output_dir.mkdir()
            tracks, files = _flat_export_fixture(root)

            export_full_results(tracks, files, str(output_dir))

            expected_full = _csv_bytes(
                "track_id,frame_id,temperature,energy,coverage,global_center_x,center_x_error,global_center_y,center_y_error,radius,radius_error,volume,volume_error,amplitude,amplitude_error,sigma_x,sigma_x_error,sigma_y,sigma_y_error,theta,theta_error,offset,offset_error,fwhm_x,fwhm_x_error,fwhm_y,fwhm_y_error,fwhm_profile_x,fwhm_profile_x_error,fwhm_profile_y,fwhm_profile_y_error,fwhm_minor,fwhm_minor_error,fwhm_major,fwhm_major_error,fwhm_effective,fwhm_effective_error,fwhm_anisotropy,fwhm_anisotropy_error,major_axis_angle_deg,major_axis_angle_deg_error,theta_major,theta_major_error,asymmetry,r_squared",
                "7,1,301.1250,81.5000,0.1250,13.1235,0.2346,14.3457,0.4568,9.8765,0.1112,123.4568,1.2346,4.5679,0.1235,2.3457,0.2223,3.4568,0.3334,0.4568,0.0445,0.5679,0.0556,5.6789,0.6667,6.7891,0.7778,5.1112,0.1111,6.1112,0.2112,5.3334,0.3111,6.3334,0.4112,5.7778,0.5111,1.2223,0.0112,26.1234,0.9877,0.4568,0.0445,0.6789,0.9877",
                "7,6,306.1250,86.5000,0.1250,18.1234,0.2346,14.3457,0.4568,14.8765,0.1112,128.4568,1.2346,9.5679,0.1235,2.3457,0.2223,3.4568,0.3334,0.4568,0.0445,0.5679,0.0556,5.6789,0.6667,6.7891,0.7778,5.1112,0.1111,6.1112,0.2112,5.3334,0.3111,6.3334,0.4112,5.7778,0.5111,1.2223,0.0112,26.1234,0.9877,0.4568,0.0445,0.6789,0.9877",
            )
            expected_csv = {
                "full_analysis_per_spot.csv": expected_full,
                "pattern_center_data.csv": _csv_bytes(
                    "frame_id,temperature,energy,coverage,center_x,center_y,error_x,error_y",
                    "1,301.1250,81.5000,0.1250,21.1250,21.2500,0.1250,0.2500",
                    "6,306.1250,86.5000,0.1250,26.1250,21.2500,0.1250,0.2500",
                ),
                "average_radius_per_frame.csv": _csv_bytes(
                    "frame_id,temperature,energy,coverage,avg_radius,avg_radius_error",
                    "1,301.1250,81.5000,0.1250,13.3457,0.4568",
                    "6,306.1250,86.5000,0.1250,18.3457,0.4568",
                ),
                "track_legend.csv": _csv_bytes(
                    "track_id,start_frame_id,initial_x_pos,initial_y_pos",
                    "7,1,13,13",
                ),
            }
            for filename, expected in expected_csv.items():
                self.assertEqual((output_dir / filename).read_bytes(), expected)

            images_dir = output_dir / "track_visualizations"
            self.assertEqual(
                sorted(path.name for path in images_dir.iterdir()),
                ["frame_0001.png", "frame_0006.png"],
            )
            masks = {}
            for image_path in sorted(images_dir.iterdir()):
                with Image.open(image_path) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (32, 32))
                    pixels = np.asarray(image)
                    purple_overlay = (
                        (pixels[:, :, 0] > 100)
                        & (pixels[:, :, 1] < 50)
                        & (pixels[:, :, 2] > 100)
                    )
                    masks[image_path.name] = purple_overlay
            first_mask = masks["frame_0001.png"]
            sixth_mask = masks["frame_0006.png"]
            self.assertTrue(first_mask[10, 13])
            self.assertFalse(first_mask[13, 17])
            self.assertTrue(sixth_mask[10, 23])
            self.assertTrue(sixth_mask[13, 17])

    def test_legacy_missing_source_exports_saved_flat_values(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "export"
            output_dir.mkdir()
            missing_path = root / "missing-frame.tif"
            loaded = deserialize_session(
                _legacy_session_payload(missing_path),
                root / "legacy.lts.json",
            )
            self.assertIs(
                loaded.analysis_state.result_status,
                AnalysisResultStatus.LEGACY_UNVERIFIED,
            )
            self.assertIs(
                loaded.analysis_state.sources[0].status,
                AnalysisSourceStatus.SOURCE_UNAVAILABLE,
            )

            with patch("trackleed.io.results_exporter.load_tiff", return_value=None):
                export_full_results(
                    loaded.tracks,
                    {frame.id: frame for frame in loaded.files},
                    str(output_dir),
                )

            self.assertEqual(
                (output_dir / "full_analysis_per_spot.csv").read_bytes(),
                _csv_bytes(
                    "track_id,frame_id,temperature,energy,coverage,radius,amplitude",
                    "7,1,300.0000,80.0000,,12.5000,2.2500",
                ),
            )
            self.assertEqual(
                (output_dir / "pattern_center_data.csv").read_bytes(),
                _csv_bytes(
                    "frame_id,temperature,energy,coverage,center_x,center_y",
                    "1,300.0000,80.0000,,20.0000,21.0000",
                ),
            )
            self.assertEqual(
                (output_dir / "average_radius_per_frame.csv").read_bytes(),
                _csv_bytes(
                    "frame_id,temperature,energy,coverage,average_radius",
                    "1,300.0000,80.0000,,12.5000",
                ),
            )
            self.assertEqual(list((output_dir / "track_visualizations").iterdir()), [])

    def test_geometry_change_preserves_exported_gaussian_values(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            per_frame_dir = root / "per-frame"
            global_median_dir = root / "global-median"
            per_frame_dir.mkdir()
            global_median_dir.mkdir()
            per_frame, global_median = _geometry_publications(root)
            self.assertIs(global_median.fit_set, per_frame.fit_set)
            self.assertEqual(
                global_median.fit_set.observations,
                per_frame.fit_set.observations,
            )
            export_full_results(
                per_frame.tracks,
                dict(per_frame.files),
                str(per_frame_dir),
            )
            export_full_results(
                global_median.tracks,
                dict(global_median.files),
                str(global_median_dir),
            )

            def read_rows(directory: Path) -> list[dict[str, str]]:
                with (directory / "full_analysis_per_spot.csv").open(
                    encoding="utf-8",
                    newline="",
                ) as handle:
                    return list(csv.DictReader(handle))

            per_frame_rows = read_rows(per_frame_dir)
            global_median_rows = read_rows(global_median_dir)
            gaussian_fields = set(per_frame_rows[0]) - {"radius", "radius_error"}
            self.assertEqual(
                [
                    {key: row[key] for key in gaussian_fields}
                    for row in per_frame_rows
                ],
                [
                    {key: row[key] for key in gaussian_fields}
                    for row in global_median_rows
                ],
            )
            self.assertNotEqual(
                [row["radius"] for row in per_frame_rows],
                [row["radius"] for row in global_median_rows],
            )
            self.assertNotEqual(
                (per_frame_dir / "pattern_center_data.csv").read_bytes(),
                (global_median_dir / "pattern_center_data.csv").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
