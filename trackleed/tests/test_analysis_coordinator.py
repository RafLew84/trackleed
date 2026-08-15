from __future__ import annotations

import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Event
import types
import unittest

import numpy as np


try:
    from PyQt6.QtCore import QRect
except ModuleNotFoundError:
    pyqt6_module = types.ModuleType("PyQt6")
    qtcore_module = types.ModuleType("PyQt6.QtCore")

    class QRect:  # type: ignore[no-redef]
        def __init__(self, x: int, y: int, width: int, height: int) -> None:
            self._x = x
            self._y = y
            self._width = width
            self._height = height

        def x(self) -> int:
            return self._x

        def y(self) -> int:
            return self._y

        def width(self) -> int:
            return self._width

        def height(self) -> int:
            return self._height

    qtcore_module.QRect = QRect
    pyqt6_module.QtCore = qtcore_module
    sys.modules["PyQt6"] = pyqt6_module
    sys.modules["PyQt6.QtCore"] = qtcore_module

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.io import results_exporter
from trackleed.logic import analysis_coordinator
from trackleed.logic import track_analysis_compatibility
from trackleed.logic.workflows.track_analysis import FitExecutionError


def _leed_file(frame_id: int) -> LeedFile:
    return LeedFile(
        filepath=Path(f"frame_{frame_id:04d}.tif"),
        filename=f"frame_{frame_id:04d}.tif",
        is_background=False,
        element="Au",
        surface=111,
        mode="G",
        temperature=300.0 + frame_id,
        energy=122.0,
        id=frame_id,
        coverage=0.0,
    )


class AnalysisCoordinatorStateTests(unittest.TestCase):
    def test_worker_result_is_not_published_until_gui_adapter_applies_it(self) -> None:
        leed_file = _leed_file(1)
        track = Track(
            track_id=1,
            spots_by_image_id={
                1: SpotROI(
                    id=1,
                    roi_rect=QRect(4, 4, 23, 23),
                    center_x=15.0,
                    center_y=15.0,
                    intensity=1.0,
                    source="test",
                )
            },
        )
        track.analysis_results = {99: {"volume": 123.0}}
        yy, xx = np.indices((31, 31), dtype=np.float64)
        pixels = 0.05 + np.exp(-((xx - 15.0) ** 2 + (yy - 15.0) ** 2) / 18.0)
        original_load_tiff = track_analysis_compatibility.load_tiff
        track_analysis_compatibility.load_tiff = lambda _filepath: pixels
        try:
            publication = track_analysis_compatibility.run_full_analysis_publication(
                [track], {}, {1: leed_file}, None, lambda _progress: None
            )
        finally:
            track_analysis_compatibility.load_tiff = original_load_tiff

        self.assertIsNotNone(publication)
        self.assertEqual(track.analysis_results, {99: {"volume": 123.0}})
        track_analysis_compatibility.publish_analysis(
            publication,
            [track],
            {1: leed_file},
        )
        self.assertEqual(set(track.analysis_results), {1})

    def test_cancelled_cold_run_preserves_previous_complete_publication(self) -> None:
        all_files_map = {1: _leed_file(1), 2: _leed_file(2)}
        for frame_id, leed_file in all_files_map.items():
            leed_file.pattern_center = {
                "center_x": 100.0 + frame_id,
                "center_y": 200.0 + frame_id,
            }
            leed_file.post_analysis_results = {"avg_radius": 50.0 + frame_id}
        track = Track(
            track_id=1,
            spots_by_image_id={
                frame_id: SpotROI(
                    id=frame_id,
                    roi_rect=QRect(4, 4, 23, 23),
                    center_x=15.0,
                    center_y=15.0,
                    intensity=1.0,
                    source="test",
                )
                for frame_id in all_files_map
            },
        )
        track.analysis_results = {99: {"volume": 123.0}}
        expected_analysis = {99: {"volume": 123.0}}
        expected_frames = {
            frame_id: (
                dict(leed_file.pattern_center),
                dict(leed_file.post_analysis_results),
            )
            for frame_id, leed_file in all_files_map.items()
        }
        cancel_event = Event()

        def cancel_after_first_frame(_progress: int) -> None:
            cancel_event.set()

        yy, xx = np.indices((31, 31), dtype=np.float64)
        pixels = 0.05 + np.exp(-((xx - 15.0) ** 2 + (yy - 15.0) ** 2) / 18.0)
        original_load_tiff = track_analysis_compatibility.load_tiff
        track_analysis_compatibility.load_tiff = lambda _filepath: pixels
        try:
            result = analysis_coordinator.run_full_analysis(
                tracks_to_analyze=[track],
                analysis_config={},
                all_files_map=all_files_map,
                background_file=None,
                progress_callback=cancel_after_first_frame,
                cancel_event=cancel_event,
            )
        finally:
            track_analysis_compatibility.load_tiff = original_load_tiff

        self.assertIsNone(result)
        self.assertEqual(track.analysis_results, expected_analysis)
        for frame_id, leed_file in all_files_map.items():
            expected_center, expected_post = expected_frames[frame_id]
            self.assertEqual(leed_file.pattern_center, expected_center)
            self.assertEqual(leed_file.post_analysis_results, expected_post)

    def test_source_error_preserves_previous_complete_publication(self) -> None:
        processed_file = _leed_file(1)
        processed_file.pattern_center = {"center_x": 10.0, "center_y": 20.0}
        processed_file.post_analysis_results = {
            "avg_radius": 278.0,
            "avg_radius_error": 0.2,
        }

        untouched_file = _leed_file(2)
        untouched_file.pattern_center = {"center_x": 30.0, "center_y": 40.0}
        untouched_file.post_analysis_results = {
            "avg_radius": 199.0,
            "avg_radius_error": 0.1,
        }

        track = Track(
            track_id=1,
            spots_by_image_id={
                1: SpotROI(
                    id=1,
                    roi_rect=QRect(0, 0, 8, 8),
                    center_x=4.0,
                    center_y=4.0,
                    intensity=1.0,
                    source="test",
                )
            },
        )
        track.analysis_results = {1: {"volume": 42.0}}

        original_load_tiff = track_analysis_compatibility.load_tiff
        track_analysis_compatibility.load_tiff = lambda _filepath: None
        try:
            with self.assertRaisesRegex(FitExecutionError, "Could not decode"):
                analysis_coordinator.run_full_analysis(
                    tracks_to_analyze=[track],
                    analysis_config={},
                    all_files_map={1: processed_file, 2: untouched_file},
                    background_file=None,
                    progress_callback=lambda _progress: None,
                )
        finally:
            track_analysis_compatibility.load_tiff = original_load_tiff

        self.assertEqual(track.analysis_results, {1: {"volume": 42.0}})
        self.assertEqual(
            processed_file.pattern_center,
            {"center_x": 10.0, "center_y": 20.0},
        )
        self.assertEqual(
            processed_file.post_analysis_results,
            {"avg_radius": 278.0, "avg_radius_error": 0.2},
        )
        self.assertEqual(untouched_file.pattern_center, {"center_x": 30.0, "center_y": 40.0})
        self.assertEqual(
            untouched_file.post_analysis_results,
            {"avg_radius": 199.0, "avg_radius_error": 0.1},
        )

    def test_update_average_radius_clears_stale_average_when_too_few_valid_radii(self) -> None:
        leed_file = _leed_file(1)
        leed_file.post_analysis_results = {
            "avg_radius": 278.0,
            "avg_radius_error": 0.2,
            "other_metric": 7.0,
        }
        valid_track = Track(track_id=1, spots_by_image_id={})
        valid_track.analysis_results = {1: {"radius": 201.0}}
        missing_track = Track(track_id=2, spots_by_image_id={})
        missing_track.analysis_results = {1: {"volume": 10.0}}
        nan_track = Track(track_id=3, spots_by_image_id={})
        nan_track.analysis_results = {1: {"radius": float("nan")}}

        analysis_coordinator._update_average_radius(
            all_files_map={1: leed_file},
            tracks_to_analyze=[valid_track, missing_track, nan_track],
            frame_id=1,
        )

        self.assertNotIn("avg_radius", leed_file.post_analysis_results)
        self.assertNotIn("avg_radius_error", leed_file.post_analysis_results)
        self.assertEqual(leed_file.post_analysis_results, {"other_metric": 7.0})

    def test_update_average_radius_replaces_stale_average_with_current_values(self) -> None:
        leed_file = _leed_file(1)
        leed_file.post_analysis_results = {
            "avg_radius": 278.0,
            "avg_radius_error": 0.2,
        }
        first_track = Track(track_id=1, spots_by_image_id={})
        first_track.analysis_results = {1: {"radius": 199.0}}
        second_track = Track(track_id=2, spots_by_image_id={})
        second_track.analysis_results = {1: {"radius": 203.0}}

        analysis_coordinator._update_average_radius(
            all_files_map={1: leed_file},
            tracks_to_analyze=[first_track, second_track],
            frame_id=1,
        )

        self.assertAlmostEqual(leed_file.post_analysis_results["avg_radius"], 201.0)
        self.assertAlmostEqual(leed_file.post_analysis_results["avg_radius_error"], 1.41421356237)

    def test_circle_fit_bbox_fallback_requires_at_least_three_points(self) -> None:
        center = analysis_coordinator._estimate_center_for_frame(
            [
                {"center_x": 70.0, "center_y": 50.0},
                {"center_x": 65.0, "center_y": 62.0},
            ],
            "circle_fit_bbox_fallback",
        )

        self.assertIsNone(center)

    def test_circle_fit_bbox_fallback_accepts_non_symmetric_arc_points(self) -> None:
        center = analysis_coordinator._estimate_center_for_frame(
            [
                {"center_x": 75.0, "center_y": 50.0},
                {"center_x": 65.0, "center_y": 70.0},
                {"center_x": 50.0, "center_y": 75.0},
            ],
            "circle_fit_bbox_fallback",
        )

        self.assertIsNotNone(center)
        self.assertAlmostEqual(center["center_x"], 50.0, places=6)
        self.assertAlmostEqual(center["center_y"], 50.0, places=6)
        self.assertAlmostEqual(center["circle_radius"], 25.0, places=6)

    def test_run_full_analysis_uses_bbox_centers_when_gaussian_fails_for_circle_fit_fallback(self) -> None:
        leed_file = _leed_file(1)
        tracks = [
            Track(
                track_id=1,
                spots_by_image_id={
                    1: SpotROI(
                        id=1,
                        roi_rect=QRect(71, 46, 8, 8),
                        center_x=75.0,
                        center_y=50.0,
                        intensity=1.0,
                        source="test",
                    )
                },
            ),
            Track(
                track_id=2,
                spots_by_image_id={
                    1: SpotROI(
                        id=2,
                        roi_rect=QRect(61, 66, 8, 8),
                        center_x=65.0,
                        center_y=70.0,
                        intensity=1.0,
                        source="test",
                    )
                },
            ),
            Track(
                track_id=3,
                spots_by_image_id={
                    1: SpotROI(
                        id=3,
                        roi_rect=QRect(46, 71, 8, 8),
                        center_x=50.0,
                        center_y=75.0,
                        intensity=1.0,
                        source="test",
                    )
                },
            ),
        ]

        original_load_tiff = track_analysis_compatibility.load_tiff
        track_analysis_compatibility.load_tiff = lambda _filepath: np.ones((100, 100), dtype=np.float32)
        try:
            result = analysis_coordinator.run_full_analysis(
                tracks_to_analyze=tracks,
                analysis_config={},
                all_files_map={1: leed_file},
                background_file=None,
                progress_callback=lambda _progress: None,
                center_mode="circle_fit_bbox_fallback",
            )
        finally:
            track_analysis_compatibility.load_tiff = original_load_tiff

        self.assertEqual(result, tracks)
        self.assertIsNotNone(leed_file.pattern_center)
        self.assertAlmostEqual(leed_file.pattern_center["center_x"], 50.0, places=6)
        self.assertAlmostEqual(leed_file.pattern_center["center_y"], 50.0, places=6)
        self.assertAlmostEqual(leed_file.pattern_center["circle_radius"], 25.0, places=6)

    def test_set_switch_average_radius_export_does_not_include_inherited_values(self) -> None:
        all_files_map = {1: _leed_file(1), 2: _leed_file(2)}

        set1_track_a = Track(track_id=1, spots_by_image_id={})
        set1_track_a.analysis_results = {
            1: {"radius": 276.0},
            2: {"radius": 278.0},
        }
        set1_track_b = Track(track_id=2, spots_by_image_id={})
        set1_track_b.analysis_results = {
            1: {"radius": 280.0},
            2: {"radius": 282.0},
        }
        for frame_id in all_files_map:
            analysis_coordinator._update_average_radius(
                all_files_map=all_files_map,
                tracks_to_analyze=[set1_track_a, set1_track_b],
                frame_id=frame_id,
            )

        self.assertAlmostEqual(all_files_map[1].post_analysis_results["avg_radius"], 278.0)
        self.assertAlmostEqual(all_files_map[2].post_analysis_results["avg_radius"], 280.0)

        analysis_coordinator._reset_per_frame_analysis_state(all_files_map, [1, 2])

        set2_track_a = Track(track_id=1, spots_by_image_id={})
        set2_track_a.analysis_results = {
            1: {"radius": 201.0},
            2: {"radius": 199.0},
        }
        set2_track_b = Track(track_id=2, spots_by_image_id={})
        set2_track_b.analysis_results = {
            1: {"volume": 10.0},
            2: {"radius": 203.0},
        }
        for frame_id in all_files_map:
            analysis_coordinator._update_average_radius(
                all_files_map=all_files_map,
                tracks_to_analyze=[set2_track_a, set2_track_b],
                frame_id=frame_id,
            )

        with TemporaryDirectory() as temp_dir:
            results_exporter._save_average_radius_csv(all_files_map, temp_dir)
            output_path = Path(temp_dir) / "average_radius_per_frame.csv"
            self.assertTrue(output_path.exists())
            with output_path.open(newline="", encoding="utf-8") as output_file:
                rows = list(csv.DictReader(output_file))

        self.assertEqual([row["frame_id"] for row in rows], ["2"])
        self.assertEqual(rows[0]["avg_radius"], "201.0000")
        exported_radii = {row["avg_radius"] for row in rows}
        self.assertNotIn("278.0000", exported_radii)
        self.assertNotIn("280.0000", exported_radii)


if __name__ == "__main__":
    unittest.main()
