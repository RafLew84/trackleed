import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.gui.spot_detection_adapter import (
    build_detection_request,
    publish_detection_result,
    run_spot_detection_job,
)
from trackleed.logic.workflows.spot_detection import (
    ClassicDetectionConfig,
    DetectedSpot,
    DetectionFrame,
    DetectionRequest,
    DetectionResult,
    FrameDetections,
    RectSnapshot,
)


def _leed_file(frame_id: int) -> LeedFile:
    return LeedFile(
        filepath=Path(f"frame-{frame_id}.tif"),
        filename=f"frame-{frame_id}.tif",
        is_background=False,
        element="Cu",
        surface=111,
        mode="G",
        temperature=300.0,
        energy=80.0,
        id=frame_id,
        coverage=None,
    )


def _spot(spot_id: int, x: int) -> SpotROI:
    return SpotROI(spot_id, QRect(x, 2, 8, 8), float(x + 4), 6.0, 1.0, "auto")


class SpotDetectionGuiAdapterTests(unittest.TestCase):
    def test_request_builder_captures_immutable_values_from_gui_models(self) -> None:
        leed_file = _leed_file(4)
        mask = QRect(1, 2, 3, 4)
        settings = {
            "backend": "Classic",
            "classic": {
                "blur_sigma": 3.0,
                "wth_radius": 4,
                "min_distance": 5,
                "threshold_rel": 0.2,
                "roi_size": 16,
            },
        }

        request = build_detection_request(
            files=(leed_file,),
            spot_settings=settings,
            background_file=None,
            metadata_masks=(mask,),
            analysis_circle=(10, 11, 9),
            analysis_ellipse=None,
        )
        leed_file.filepath = Path("changed.tif")
        settings["classic"]["roi_size"] = 64
        mask.setWidth(99)

        self.assertEqual(request.method_id, "classic")
        self.assertEqual(request.frames[0].source_path, Path("frame-4.tif"))
        self.assertEqual(request.frames[0].frame_id, 4)
        self.assertIsInstance(request.config, ClassicDetectionConfig)
        self.assertEqual(request.config.roi_size, 16)
        self.assertEqual(request.analysis_region.metadata_masks[0].width, 3)
        self.assertIsNotNone(request.analysis_region.circle)
        self.assertEqual(request.analysis_region.circle.radius, 9)

    def test_publish_replaces_spots_without_touching_tracks_or_aliasing(self) -> None:
        first = _leed_file(1)
        second = _leed_file(2)
        tracked_spot = _spot(8, 10)
        first.spots = [tracked_spot]
        second.spots = [_spot(9, 20)]
        track = Track(track_id=3, spots_by_image_id={1: tracked_spot})
        result = DetectionResult(
            frames=(
                FrameDetections(
                    frame_id=1,
                    spots=(
                        DetectedSpot(
                            id=1,
                            roi=RectSnapshot(3, 4, 8, 8),
                            center_x=7.0,
                            center_y=8.0,
                        ),
                    ),
                ),
                FrameDetections(frame_id=2, spots=()),
            )
        )

        publish_detection_result((first, second), result)

        self.assertEqual(len(first.spots), 1)
        self.assertEqual(second.spots, [])
        self.assertEqual(track.spots_by_image_id, {1: tracked_spot})
        self.assertIsNot(first.spots[0], tracked_spot)
        self.assertIsNot(first.spots[0], track.spots_by_image_id[1])

    def test_publish_rejects_incomplete_result_before_mutation(self) -> None:
        first = _leed_file(1)
        second = _leed_file(2)
        first.spots = [_spot(1, 10)]
        second.spots = [_spot(2, 20)]
        original = (first.spots, second.spots)
        incomplete = DetectionResult(frames=(FrameDetections(frame_id=1, spots=()),))

        with self.assertRaises(ValueError):
            publish_detection_result((first, second), incomplete)

        self.assertIs(first.spots, original[0])
        self.assertIs(second.spots, original[1])

    def test_worker_job_builds_run_context_for_public_workflow(self) -> None:
        image = np.zeros((32, 32), dtype=np.uint8)
        image[14:18, 14:18] = 255
        progress: list[int] = []
        request = DetectionRequest(
            method_id="classic",
            frames=(DetectionFrame(5, Path("frame.tif")),),
            config=ClassicDetectionConfig(
                blur_sigma=1.0,
                wth_radius=2,
                min_distance=3,
                threshold_rel=0.05,
                roi_size=8,
            ),
        )

        result = run_spot_detection_job(
            request,
            lambda _path: image,
            SimpleNamespace(is_set=lambda: False),
            progress_callback=progress.append,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.frames[0].frame_id, 5)
        self.assertEqual(progress, [100])
