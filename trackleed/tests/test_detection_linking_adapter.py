from __future__ import annotations

import unittest
from pathlib import Path
from threading import Event

from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI
from trackleed.gui.detection_linking_adapter import (
    build_detection_linking_request,
    publish_detection_linking_result,
    run_detection_linking_job,
)
from trackleed.logic.workflows.detection_linking import (
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

        publish_detection_linking_result(files, result)
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

        tracks = publish_detection_linking_result(files, result)

        self.assertIsNot(tracks[0].spots_by_image_id[1], original_detection)
        self.assertIsNot(tracks[0].spots_by_image_id[1], files[0].spots[0])
        self.assertIsNone(files[0].pattern_center)
        self.assertEqual(files[0].post_analysis_results, {"unrelated": "keep"})


if __name__ == "__main__":
    unittest.main()
