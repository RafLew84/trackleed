from __future__ import annotations

import unittest
import inspect

from trackleed.analysis import detection_linking_runtime
from trackleed.logic.workflows import detection_linking

from trackleed.logic.workflows.detection_linking import (
    DetectionLinkingCancelled,
    DetectionLinkingConfig,
    DetectionLinkingFamily,
    DetectionLinkingRequest,
    DetectionLinkingPreflightError,
    RunContext,
    default_detection_linking_catalog,
    run_detection_linking,
)
from trackleed.logic.workflows.seeded_tracking import SeedRect
from trackleed.logic.workflows.spot_detection import (
    DetectedSpot,
    FrameDetections,
    RectSnapshot,
)


def _spot(
    spot_id: int,
    x: float,
    y: float,
    *,
    source: str = "auto",
) -> DetectedSpot:
    return DetectedSpot(
        spot_id,
        RectSnapshot(round(x) - 4, round(y) - 4, 8, 8),
        x,
        y,
        intensity=10.0,
        source=source,
    )


class DetectionLinkingWorkflowTests(unittest.TestCase):
    def test_headless_contract_and_runtime_have_no_gui_model_dependency(self) -> None:
        source = inspect.getsource(detection_linking) + inspect.getsource(
            detection_linking_runtime
        )

        for forbidden in ("PyQt", "trackleed.gui", "LeedFile", "SpotROI", "QRect"):
            self.assertNotIn(forbidden, source)

    def test_hungarian_catalog_method_links_independent_detections(self) -> None:
        catalog = default_detection_linking_catalog()
        method = catalog.method("hungarian")
        request = DetectionLinkingRequest(
            method_id="hungarian",
            frames=(
                FrameDetections(10, (_spot(101, 10.0, 10.0), _spot(102, 30.0, 30.0))),
                FrameDetections(20, (_spot(201, 12.0, 11.0), _spot(202, 29.0, 32.0))),
            ),
            config=DetectionLinkingConfig(
                max_distance_px=5.0,
                max_gap=0,
                min_track_length=2,
            ),
        )

        result = run_detection_linking(request, RunContext(catalog=catalog))

        self.assertEqual(method.family, DetectionLinkingFamily.ASSIGNMENT)
        self.assertTrue(method.capabilities.consumes_detections)
        self.assertIsInstance(result.tracks[0].observations[0].roi, SeedRect)
        self.assertEqual(
            result.tracks[0].observations[0].source,
            "detection_linking:hungarian",
        )
        self.assertEqual(
            dict(result.tracks[0].observations[0].diagnostics)["detection_source"],
            "auto",
        )
        self.assertEqual(
            tuple(
                tuple(
                    (observation.frame_id, observation.center_x, observation.center_y)
                    for observation in track.observations
                )
                for track in result.tracks
            ),
            (
                ((10, 10.0, 10.0), (20, 12.0, 11.0)),
                ((10, 30.0, 30.0), (20, 29.0, 32.0)),
            ),
        )

    def test_missing_detections_is_structured_preflight_failure(self) -> None:
        request = DetectionLinkingRequest(
            "hungarian",
            (FrameDetections(10, ()), FrameDetections(20, ())),
            DetectionLinkingConfig(),
        )
        catalog = default_detection_linking_catalog()

        report = catalog.preflight(request)

        self.assertEqual(
            tuple(issue.code for issue in report.issues),
            ("detections_required",),
        )
        with self.assertRaises(DetectionLinkingPreflightError):
            run_detection_linking(request, RunContext(catalog=catalog))

    def test_manual_seeds_are_rejected_as_non_independent_detections(self) -> None:
        request = DetectionLinkingRequest(
            "hungarian",
            (
                FrameDetections(10, (_spot(1, 10.0, 10.0, source="manual_point"),)),
                FrameDetections(20, (_spot(2, 11.0, 11.0),)),
            ),
            DetectionLinkingConfig(),
        )

        report = default_detection_linking_catalog().preflight(request)

        self.assertEqual(
            tuple(issue.code for issue in report.issues),
            ("independent_detections_required",),
        )

    def test_wrongly_typed_config_fails_closed(self) -> None:
        request = DetectionLinkingRequest(
            "hungarian",
            (
                FrameDetections(10, (_spot(1, 10.0, 10.0),)),
                FrameDetections(20, (_spot(2, 11.0, 11.0),)),
            ),
            DetectionLinkingConfig(max_distance_px="invalid"),  # type: ignore[arg-type]
        )

        report = default_detection_linking_catalog().preflight(request)

        self.assertEqual(
            tuple(issue.code for issue in report.issues),
            ("invalid_config",),
        )

    def test_cancellation_after_progress_does_not_return_partial_tracks(self) -> None:
        request = DetectionLinkingRequest(
            "hungarian",
            (
                FrameDetections(10, (_spot(1, 10.0, 10.0),)),
                FrameDetections(20, (_spot(2, 11.0, 11.0),)),
                FrameDetections(30, (_spot(3, 12.0, 12.0),)),
            ),
            DetectionLinkingConfig(),
        )
        progress = []
        context = RunContext(
            report_progress=progress.append,
            is_cancelled=lambda: bool(progress),
        )

        with self.assertRaises(DetectionLinkingCancelled):
            run_detection_linking(request, context)

        self.assertEqual(tuple(item.frame_id for item in progress), (10,))

    def test_max_gap_links_across_one_empty_detection_frame(self) -> None:
        request = DetectionLinkingRequest(
            "hungarian",
            (
                FrameDetections(10, (_spot(1, 10.0, 10.0),)),
                FrameDetections(20, ()),
                FrameDetections(30, (_spot(3, 12.0, 12.0),)),
            ),
            DetectionLinkingConfig(max_gap=1),
        )

        result = run_detection_linking(request, RunContext())

        self.assertEqual(
            tuple(item.frame_id for item in result.tracks[0].observations),
            (10, 30),
        )


if __name__ == "__main__":
    unittest.main()
