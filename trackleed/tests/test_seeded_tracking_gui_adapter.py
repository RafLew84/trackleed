import unittest
from pathlib import Path
from threading import Event

import numpy as np

from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.gui.seeded_tracking_adapter import (
    SeededTrackingDialogSettings,
    SeededTrackingSettingsBinding,
    build_seeded_tracking_request,
    clone_tracks_for_review,
    publish_edited_tracks,
    publish_tracking_result,
    run_seeded_tracking_job,
)
from trackleed.logic.workflows.seeded_tracking import (
    OpticalFlowConfig,
    SeedRect,
    TrackObservation,
    TrackedSpotIdentity,
    TrackingResult,
)


def _leed_file(frame_id: int, spots: list[SpotROI]) -> LeedFile:
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
        spots=spots,
    )


def _spot(spot_id: int, x: int, y: int) -> SpotROI:
    return SpotROI(
        spot_id,
        QRect(x - 4, y - 4, 8, 8),
        float(x),
        float(y),
        12.0,
        "manual",
    )


class SeededTrackingGuiAdapterTests(unittest.TestCase):
    def test_settings_binding_changes_only_after_accept(self) -> None:
        binding = SeededTrackingSettingsBinding(roi_size=18)
        candidate = SeededTrackingDialogSettings.defaults(roi_size=40)

        self.assertEqual(binding.dialog_settings.optical_flow.roi_size, 18)
        binding.accept(candidate)

        self.assertIs(binding.dialog_settings, candidate)

    def test_request_snapshots_only_reference_frame_seeds(self) -> None:
        seed = _spot(7, 20, 21)
        reference = _leed_file(1, [seed])
        unrelated_detection = _spot(99, 40, 41)
        second = _leed_file(2, [unrelated_detection])

        request = build_seeded_tracking_request(
            files=(reference, second),
            reference_file=reference,
            method_id="optical_flow",
            config=OpticalFlowConfig(roi_size=18),
            background_file=None,
        )
        reference.filepath = Path("changed.tif")
        seed.center_x = 999.0
        seed.roi_rect.setWidth(99)
        second.spots.clear()

        self.assertEqual(
            tuple(frame.source_path for frame in request.frames),
            (Path("frame-1.tif"), Path("frame-2.tif")),
        )
        self.assertEqual(request.reference_frame_id, 1)
        self.assertEqual(tuple(item.seed_id for item in request.seeds), (7,))
        self.assertEqual(request.seeds[0].center_x, 20.0)
        self.assertEqual(request.seeds[0].roi.width, 8)
        self.assertEqual(request.config.roi_size, 18)

    def test_publication_does_not_alias_detections_and_track_observations(self) -> None:
        original_detection = _spot(7, 20, 21)
        first = _leed_file(1, [original_detection])
        second = _leed_file(2, [_spot(99, 40, 41)])
        result = TrackingResult(
            (
                TrackedSpotIdentity(
                    7,
                    (
                        TrackObservation(
                            1,
                            SeedRect(16, 17, 8, 8),
                            20.0,
                            21.0,
                            source="seed",
                        ),
                        TrackObservation(
                            2,
                            SeedRect(18, 18, 8, 8),
                            22.0,
                            22.0,
                            source="optical_flow",
                        ),
                    ),
                ),
            )
        )

        tracks = publish_tracking_result((first, second), result)

        track_seed = tracks[0].spots_by_image_id[1]
        self.assertIsNot(track_seed, original_detection)
        self.assertIsNot(track_seed, first.spots[0])
        self.assertIsNot(tracks[0].spots_by_image_id[2], second.spots[0])
        self.assertEqual(second.spots[0].center_x, 22.0)
        first.spots[0].center_x = 999.0
        self.assertEqual(track_seed.center_x, 20.0)

    def test_worker_job_turns_cancellation_into_no_result(self) -> None:
        reference = _leed_file(1, [_spot(7, 20, 21)])
        second = _leed_file(2, [])
        request = build_seeded_tracking_request(
            files=(reference, second),
            reference_file=reference,
            method_id="optical_flow",
            config=OpticalFlowConfig(),
            background_file=None,
        )
        cancelled = Event()
        cancelled.set()

        result = run_seeded_tracking_job(
            request,
            lambda _path: np.zeros((32, 32), dtype=np.uint8),
            cancelled,
            progress_callback=lambda _value: None,
        )

        self.assertIsNone(result)

    def test_review_and_accepted_publication_use_independent_spot_copies(self) -> None:
        first = _leed_file(1, [_spot(99, 1, 1)])
        second = _leed_file(2, [_spot(99, 2, 2)])
        raw = [
            Track(
                7,
                {
                    1: _spot(7, 20, 21),
                    2: _spot(7, 22, 23),
                },
            )
        ]

        review = clone_tracks_for_review(raw)
        review[0].spots_by_image_id[1].center_x = 88.0

        self.assertEqual(raw[0].spots_by_image_id[1].center_x, 20.0)
        active = publish_edited_tracks((first, second), review)
        review[0].spots_by_image_id[1].center_x = 99.0
        first.spots[0].center_x = 77.0

        self.assertEqual(active[0].spots_by_image_id[1].center_x, 88.0)
        self.assertIsNot(active[0].spots_by_image_id[1], first.spots[0])
        self.assertEqual(second.spots[0].center_x, 22.0)


if __name__ == "__main__":
    unittest.main()
