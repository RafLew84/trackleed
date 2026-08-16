from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from trackleed.core.data_models import Track
from trackleed.gui.seeded_tracking_legacy_models import (
    LegacySeededFamily,
    legacy_spot,
    run_legacy_seeded,
)
from trackleed.logic.workflows.point_tracking import TapirPointConfig
from trackleed.logic.workflows.seeded_tracking import (
    RunContext,
    SeededTrackingCancelled,
    SeededRetrackRequest,
    SeededTrackingFrame,
    SeededTrackingRequest,
    SeedRect,
    TrackedSpotIdentity,
    TrackingSeed,
    TrackObservation,
)


def _request() -> SeededTrackingRequest:
    return SeededTrackingRequest(
        "tapir_offline",
        (
            SeededTrackingFrame(1, Path("one.tif")),
            SeededTrackingFrame(2, Path("two.tif")),
        ),
        1,
        (TrackingSeed(7, 20.0, 21.0, SeedRect(16, 17, 8, 8)),),
        TapirPointConfig(),
    )


class SeededTrackingLegacyExecutionTests(unittest.TestCase):
    def test_shared_execution_cancels_before_preparation_or_dispatch(self) -> None:
        family = LegacySeededFamily(
            "Point",
            lambda _method_id, _config: self.fail(
                "Cancellation must precede parameter mapping"
            ),
            lambda _seed: self.fail("Cancellation must precede seed conversion"),
            lambda _method_id, *, retrack: self.fail(
                "Cancellation must precede runner resolution"
            ),
        )

        with self.assertRaises(SeededTrackingCancelled):
            run_legacy_seeded(
                "tapir_offline",
                _request(),
                RunContext(
                    image_reader=lambda _path: self.fail(
                        "Cancellation must precede image access"
                    ),
                    is_cancelled=lambda: True,
                ),
                family,
            )

    def test_shared_execution_prepares_full_run_and_converts_result(self) -> None:
        captured: dict[str, object] = {}
        progress = []
        cancelled = False
        expected_image = np.full((3, 4), 17, dtype=np.uint8)

        def runner(
            files,
            frame_ids,
            reference_id,
            params,
            progress_callback=None,
            cancel_event=None,
            **_kwargs,
        ):
            captured.update(
                files=files,
                frame_ids=frame_ids,
                reference_id=reference_id,
                params=params,
                loaded_image=files[1]._seeded_image_reader(files[1].filepath),
                cancel_event=cancel_event,
            )
            progress_callback(50)
            return [Track(7, {1: files[1].spots[0]})]

        def resolve_runner(method_id: str, *, retrack: bool):
            captured.update(method_id=method_id, retrack=retrack)
            return runner

        family = LegacySeededFamily(
            "Point",
            lambda _method_id, _config: {"roi_size": 24},
            lambda seed: legacy_spot(
                seed.seed_id,
                seed.roi,
                seed.center_x,
                seed.center_y,
                seed.intensity,
                seed.source,
            ),
            resolve_runner,
        )

        result = run_legacy_seeded(
            "tapir_offline",
            _request(),
            RunContext(
                image_reader=lambda _path: expected_image,
                report_progress=progress.append,
                is_cancelled=lambda: cancelled,
            ),
            family,
        )

        self.assertEqual(captured["method_id"], "tapir_offline")
        self.assertFalse(captured["retrack"])
        self.assertEqual(captured["frame_ids"], [1, 2])
        self.assertEqual(captured["reference_id"], 1)
        self.assertEqual(captured["params"], {"roi_size": 24})
        self.assertIs(captured["loaded_image"], expected_image)
        self.assertFalse(captured["cancel_event"].is_set())
        cancelled = True
        self.assertTrue(captured["cancel_event"].is_set())
        self.assertEqual(progress[0].completed, 50)
        self.assertEqual(progress[0].total, 100)
        self.assertEqual(result.tracks[0].observations[0].roi, SeedRect(16, 17, 8, 8))

    def test_shared_execution_selects_retrack_and_converts_one_track(self) -> None:
        captured: dict[str, object] = {}
        cancelled = False
        full_request = _request()
        request = SeededRetrackRequest(
            full_request.method_id,
            full_request.frames,
            TrackedSpotIdentity(
                7,
                (TrackObservation(1, SeedRect(16, 17, 8, 8), 20.0, 21.0),),
            ),
            1,
            full_request.config,
        )

        def runner(
            track,
            start_frame_id,
            files,
            frame_ids,
            params,
            cancel_event=None,
            **_kwargs,
        ):
            captured.update(
                track=track,
                start_frame_id=start_frame_id,
                files=files,
                frame_ids=frame_ids,
                params=params,
                cancel_event=cancel_event,
            )
            return track

        def resolve_runner(_method_id: str, *, retrack: bool):
            captured["retrack"] = retrack
            return runner

        family = LegacySeededFamily(
            "Point",
            lambda _method_id, _config: {"roi_size": 24},
            lambda _seed: self.fail("Re-track must not rebuild full-run seeds"),
            resolve_runner,
        )

        result = run_legacy_seeded(
            "tapir_offline",
            request,
            RunContext(
                image_reader=lambda _path: np.zeros((2, 2)),
                is_cancelled=lambda: cancelled,
            ),
            family,
        )

        self.assertTrue(captured["retrack"])
        self.assertEqual(captured["start_frame_id"], 1)
        self.assertEqual(captured["frame_ids"], [1, 2])
        self.assertFalse(captured["cancel_event"].is_set())
        cancelled = True
        self.assertTrue(captured["cancel_event"].is_set())
        self.assertEqual(result.tracks, (request.track,))

    def test_shared_execution_never_treats_none_as_success(self) -> None:
        cancelled = False

        def runner(*_args, **_kwargs):
            nonlocal cancelled
            cancelled = True
            return None

        family = LegacySeededFamily(
            "Point",
            lambda _method_id, _config: {},
            lambda seed: legacy_spot(
                seed.seed_id,
                seed.roi,
                seed.center_x,
                seed.center_y,
                seed.intensity,
                seed.source,
            ),
            lambda _method_id, *, retrack: runner,
        )

        with self.assertRaises(SeededTrackingCancelled):
            run_legacy_seeded(
                "tapir_offline",
                _request(),
                RunContext(
                    image_reader=lambda _path: np.zeros((2, 2)),
                    is_cancelled=lambda: cancelled,
                ),
                family,
            )

        with self.assertRaisesRegex(RuntimeError, "returned no tracking result"):
            run_legacy_seeded(
                "tapir_offline",
                _request(),
                RunContext(image_reader=lambda _path: np.zeros((2, 2))),
                family,
                runner=lambda *_args, **_kwargs: None,
            )

    def test_shared_execution_rejects_valid_result_after_cancellation(self) -> None:
        full_request = _request()
        retrack_request = SeededRetrackRequest(
            full_request.method_id,
            full_request.frames,
            TrackedSpotIdentity(
                7,
                (TrackObservation(1, SeedRect(16, 17, 8, 8), 20.0, 21.0),),
            ),
            1,
            full_request.config,
        )
        raw_track = Track(
            7,
            {
                1: legacy_spot(
                    7,
                    SeedRect(16, 17, 8, 8),
                    20.0,
                    21.0,
                    0.0,
                    "seed",
                )
            },
        )
        family = LegacySeededFamily(
            "Legacy",
            lambda _method_id, _config: {},
            lambda seed: legacy_spot(
                seed.seed_id,
                seed.roi,
                seed.center_x,
                seed.center_y,
                seed.intensity,
                seed.source,
            ),
            lambda _method_id, *, retrack: self.fail("Explicit runner expected"),
        )
        for request, raw_result in (
            (full_request, [raw_track]),
            (retrack_request, raw_track),
        ):
            with self.subTest(operation=type(request).__name__):
                cancelled = False

                def runner(*_args, **_kwargs):
                    nonlocal cancelled
                    cancelled = True
                    return raw_result

                with self.assertRaises(SeededTrackingCancelled):
                    run_legacy_seeded(
                        "tapir_offline",
                        request,
                        RunContext(
                            image_reader=lambda _path: np.zeros((2, 2)),
                            is_cancelled=lambda: cancelled,
                        ),
                        family,
                        runner=runner,
                    )


if __name__ == "__main__":
    unittest.main()
