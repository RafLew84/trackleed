from __future__ import annotations

import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from PyQt6.QtCore import QRect

from trackleed.analysis.tracking import track_spots_hungarian_directional
from trackleed.core.data_models import LeedFile, SpotROI
from trackleed.gui.detection_linking_adapter import (
    DetectionLinkingDialogSettings,
    DetectionLinkingSettingsBinding,
    build_detection_linking_request,
    run_detection_linking_job,
)
from trackleed.gui.tracking_publisher import project_tracking_result, publish_tracks
from trackleed.logic.active_analysis_state import ActiveAnalysisState
from trackleed.logic.workflows.detection_linking import (
    DirectionalLinkingConfig,
    RunContext,
    HungarianLinkingConfig,
    default_detection_linking_catalog,
    run_detection_linking,
)
from trackleed.logic.workflows.seeded_tracking import (
    SeedRect,
    TrackObservation,
    TrackedSpotIdentity,
    TrackingResult,
)


def _frame(frame_id: int, spots: list[SpotROI]) -> LeedFile:
    return LeedFile(
        Path(f"frame-{frame_id}.tif"),
        f"frame-{frame_id}.tif",
        False,
        "Cu",
        111,
        "G",
        300.0,
        80.0,
        frame_id,
        None,
        spots,
    )


def _spot(spot_id: int, x: float, y: float) -> SpotROI:
    return SpotROI(
        spot_id,
        QRect(round(x) - 4, round(y) - 4, 8, 8),
        x,
        y,
        10.0,
        "auto",
    )


def _publication_target() -> SimpleNamespace:
    return SimpleNamespace(
        tracking_results=[],
        _active_analysis_state=ActiveAnalysisState(),
    )


class DetectionLinkingAdapterTests(unittest.TestCase):
    def test_request_snapshots_independent_detections(self) -> None:
        first_spot = _spot(11, 10.0, 12.0)
        files = (_frame(1, [first_spot]), _frame(2, [_spot(21, 11.0, 13.0)]))

        request = build_detection_linking_request(
            files,
            HungarianLinkingConfig(max_distance_px=7.0),
        )
        first_spot.center_x = 999.0
        files[0].spots.clear()

        self.assertEqual(request.method_id, "hungarian")
        self.assertEqual(request.frames[0].spots[0].center_x, 10.0)
        self.assertEqual(request.config.max_distance_px, 7.0)

    def test_directional_request_and_settings_bind_by_stable_method_id(self) -> None:
        files = (_frame(1, [_spot(11, 10.0, 12.0)]), _frame(2, [_spot(21, 11.0, 13.0)]))
        config = DirectionalLinkingConfig(
            center_mode="manual",
            manual_center=(20.0, 20.0),
            direction_mode="outward",
        )
        binding = DetectionLinkingSettingsBinding()

        request = build_detection_linking_request(
            files,
            config,
            method_id="directional",
        )
        binding.accept(
            DetectionLinkingDialogSettings(
                hungarian=binding.dialog_settings.hungarian,
                directional=config,
            )
        )

        self.assertEqual(request.method_id, "directional")
        self.assertIs(request.config, config)
        self.assertIs(binding.dialog_settings.directional, config)

    def test_directional_legacy_shim_delegates_to_catalog_workflow(self) -> None:
        files = (_frame(1, [_spot(11, 70.0, 50.0)]), _frame(2, [_spot(21, 75.0, 50.0)]))

        tracks = track_spots_hungarian_directional(
            {1: files[0], 2: files[1]},
            [1, 2],
            {
                "center_mode": "manual",
                "manual_center": (50.0, 50.0),
                "direction_mode": "outward",
                "dynamic_max_distance": False,
            },
        )

        self.assertEqual(tuple(tracks[0].spots_by_image_id), (1, 2))
        self.assertEqual(tracks[0].spots_by_image_id[2].center_x, 75.0)
        self.assertEqual(
            tracks[0].spots_by_image_id[1].source,
            "detection_linking:directional",
        )
        self.assertEqual(tracks[0].spots_by_image_id[1].detection_source, "auto")

    def test_cancelled_worker_job_returns_no_partial_result(self) -> None:
        files = (_frame(1, [_spot(11, 10.0, 12.0)]), _frame(2, [_spot(21, 11.0, 13.0)]))
        request = build_detection_linking_request(files, HungarianLinkingConfig())
        cancelled = Event()
        cancelled.set()

        result = run_detection_linking_job(
            request,
            cancelled,
            progress_callback=lambda _value: None,
        )

        self.assertIsNone(result)

    def test_published_tracks_cannot_be_relinked_as_independent_detections(self) -> None:
        files = (
            _frame(1, [_spot(11, 10.0, 12.0)]),
            _frame(2, [_spot(21, 11.0, 13.0)]),
        )
        request = build_detection_linking_request(files, HungarianLinkingConfig())
        result = run_detection_linking(request, RunContext())

        target = _publication_target()
        publish_tracks(target, files, project_tracking_result(result))
        relink_request = build_detection_linking_request(
            files,
            HungarianLinkingConfig(),
        )
        report = default_detection_linking_catalog().preflight(relink_request)

        self.assertEqual(
            tuple(issue.code for issue in report.issues),
            ("independent_detections_required",),
        )
        self.assertEqual(files[0].spots[0].detection_source, "auto")

    def test_publication_is_atomic_unaliased_and_invalidates_geometry(self) -> None:
        original_detection = _spot(11, 10.0, 12.0)
        files = (_frame(1, [original_detection]), _frame(2, [_spot(21, 11.0, 13.0)]))
        for frame in files:
            frame.pattern_center = {"center_x": 50.0, "center_y": 51.0}
            frame.post_analysis_results = {
                "avg_radius": 25.0,
                "avg_radius_error": 0.5,
                "unrelated": "keep",
            }
        result = TrackingResult(
            (
                TrackedSpotIdentity(
                    1,
                    (
                        TrackObservation(1, SeedRect(6, 8, 8, 8), 10.0, 12.0),
                        TrackObservation(2, SeedRect(7, 9, 8, 8), 11.0, 13.0),
                    ),
                ),
            )
        )

        target = _publication_target()
        publish_tracks(target, files, project_tracking_result(result))
        tracks = target.tracking_results

        self.assertIsNot(tracks[0].spots_by_image_id[1], original_detection)
        self.assertIsNot(tracks[0].spots_by_image_id[1], files[0].spots[0])
        self.assertIsNone(files[0].pattern_center)
        self.assertEqual(files[0].post_analysis_results, {"unrelated": "keep"})


if __name__ == "__main__":
    unittest.main()
