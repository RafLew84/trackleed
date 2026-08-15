from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QRect

from trackleed.analysis.seeded_point_tracking_compatibility import (
    run_legacy_point_tracker,
)
from trackleed.core.data_models import SpotROI, Track
from trackleed.logic.workflows.point_tracking import TapirPointConfig
from trackleed.logic.workflows.seeded_tracking import (
    RunContext,
    SeededTrackingCancelled,
    SeededTrackingFrame,
    SeededTrackingRequest,
    SeedRect,
    TrackingSeed,
)


def _request() -> SeededTrackingRequest:
    return SeededTrackingRequest(
        method_id="tapir_offline",
        frames=(
            SeededTrackingFrame(1, Path("one.tif")),
            SeededTrackingFrame(2, Path("two.tif")),
        ),
        reference_frame_id=1,
        seeds=(
            TrackingSeed(
                7,
                20.0,
                21.0,
                SeedRect(16, 17, 8, 8),
                intensity=12.0,
                source="manual",
            ),
        ),
        config=TapirPointConfig(
            models_dir=Path("models/tapir"),
            checkpoint_path=Path("models/tapir/model.pt"),
            roi_size=24,
        ),
    )


class SeededPointTrackingCompatibilityTests(unittest.TestCase):
    def test_adapter_converts_neutral_request_and_result_at_legacy_boundary(
        self,
    ) -> None:
        captured: dict[str, object] = {}
        progress = []

        def _runner(
            files_by_id,
            frame_ids,
            reference_frame_id,
            params,
            progress_callback=None,
            cancel_event=None,
        ):
            captured.update(
                files_by_id=files_by_id,
                frame_ids=frame_ids,
                reference_frame_id=reference_frame_id,
                params=params,
                cancel_event=cancel_event,
            )
            captured["loaded_image"] = files_by_id[1]._seeded_image_reader(
                files_by_id[1].filepath
            )
            progress_callback(50)
            seed = files_by_id[1].spots[0]
            seed.center_x = 999.0
            return [
                Track(
                    7,
                    {
                        1: SpotROI(
                            7,
                            QRect(16, 17, 8, 8),
                            20.0,
                            21.0,
                            12.0,
                            "manual",
                        ),
                        2: SpotROI(
                            7,
                            QRect(18, 18, 24, 24),
                            30.0,
                            31.0,
                            10.0,
                            "tapir_offline",
                        ),
                    },
                )
            ]

        request = _request()
        expected_image = np.full((3, 4), 17, dtype=np.uint8)
        result = run_legacy_point_tracker(
            "tapir_offline",
            request,
            RunContext(
                image_reader=lambda _path: expected_image,
                report_progress=progress.append,
            ),
            runner=_runner,
        )

        self.assertEqual(captured["frame_ids"], [1, 2])
        self.assertEqual(captured["reference_frame_id"], 1)
        self.assertEqual(captured["files_by_id"][1].filepath, Path("one.tif"))
        self.assertEqual(captured["params"]["tapir_models_dir"], "models/tapir")
        self.assertEqual(captured["params"]["roi_size"], 24)
        self.assertIs(captured["loaded_image"], expected_image)
        self.assertEqual(request.seeds[0].center_x, 20.0)
        self.assertEqual(progress[0].completed, 50)
        self.assertEqual(progress[0].total, 100)
        self.assertEqual(
            tuple(item.frame_id for item in result.tracks[0].observations),
            (1, 2),
        )
        self.assertEqual(result.tracks[0].observations[-1].source, "tapir_offline")

    def test_legacy_none_result_is_reported_as_cancellation_only_when_cancelled(
        self,
    ) -> None:
        with self.assertRaises(SeededTrackingCancelled):
            run_legacy_point_tracker(
                "tapir_offline",
                _request(),
                RunContext(
                    image_reader=lambda _path: None,
                    is_cancelled=lambda: True,
                ),
                runner=lambda *_args, **_kwargs: None,
            )


if __name__ == "__main__":
    unittest.main()
