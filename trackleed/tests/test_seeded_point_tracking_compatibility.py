from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import QRect

from trackleed.analysis.seeded_point_tracking_runtime import (
    default_point_tracker_configs,
    point_tracking_definitions,
)
from trackleed.core.data_models import SpotROI, Track
from trackleed.gui.seeded_point_tracking_compatibility import (
    legacy_point_tracker_params,
    point_tracker_legacy_bindings,
    run_legacy_point_retracker,
    run_legacy_point_tracker,
)
from trackleed.logic.workflows.point_tracking import TapirPointConfig
from trackleed.logic.workflows.seeded_tracking import (
    RunContext,
    SeededRetrackRequest,
    SeededTrackingCancelled,
    SeededTrackingFrame,
    SeededTrackingRequest,
    SeedRect,
    TrackedSpotIdentity,
    TrackingSeed,
    TrackObservation,
    run_seeded_retrack,
    run_seeded_tracking,
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
    def test_public_workflows_resolve_every_canonical_legacy_adapter(self) -> None:
        calls: list[str] = []
        functions = {}
        for binding in point_tracker_legacy_bindings():
            functions[binding.tracking_runner_name] = (
                lambda *_args, _name=binding.tracking_runner_name, **_kwargs: (
                    calls.append(_name) or []
                )
            )
            functions[binding.retrack_runner_name] = (
                lambda track, *_args, _name=binding.retrack_runner_name, **_kwargs: (
                    calls.append(_name) or track
                )
            )
        context = RunContext(
            image_reader=lambda _path: np.zeros((2, 2)),
            point_tracker_preflight=lambda _method_id, _config: (),
            point_tracker_executor=run_legacy_point_tracker,
            point_tracker_retrack_executor=run_legacy_point_retracker,
        )
        base_request = _request()
        with patch(
            "trackleed.gui.seeded_point_tracking_compatibility.import_module",
            return_value=SimpleNamespace(**functions),
        ):
            for method_id, config in default_point_tracker_configs():
                request = replace(base_request, method_id=method_id, config=config)
                run_seeded_tracking(request, context)
                run_seeded_retrack(
                    SeededRetrackRequest(
                        method_id,
                        request.frames,
                        TrackedSpotIdentity(
                            7,
                            (
                                TrackObservation(
                                    1,
                                    SeedRect(16, 17, 8, 8),
                                    20.0,
                                    21.0,
                                ),
                            ),
                        ),
                        1,
                        config,
                    ),
                    context,
                )

        expected = [
            runner_name
            for binding in point_tracker_legacy_bindings()
            for runner_name in (
                binding.tracking_runner_name,
                binding.retrack_runner_name,
            )
        ]
        self.assertEqual(calls, expected)

    def test_canonical_definitions_cover_every_legacy_runner_mapping(self) -> None:
        definitions = {item.method_id: item for item in point_tracking_definitions()}
        for binding in point_tracker_legacy_bindings():
            self.assertIn(binding.method_id, definitions)
            self.assertTrue(binding.tracking_runner_name.startswith("track_spots_"))
            self.assertTrue(
                binding.retrack_runner_name.startswith("retrack_single_")
            )
            params = legacy_point_tracker_params(
                binding.method_id,
                definitions[binding.method_id].default_config,
            )
            self.assertIn(binding.models_param_name, params)

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

    def test_retrack_adapter_converts_neutral_track_at_gui_boundary(self) -> None:
        request = _request()
        retrack_request = SeededRetrackRequest(
            method_id=request.method_id,
            frames=request.frames,
            track=TrackedSpotIdentity(
                7,
                (
                    TrackObservation(
                        1,
                        SeedRect(16, 17, 8, 8),
                        20.0,
                        21.0,
                        12.0,
                        "manual",
                    ),
                ),
            ),
            start_frame_id=1,
            config=request.config,
        )

        result = run_legacy_point_retracker(
            "tapir_offline",
            retrack_request,
            RunContext(image_reader=lambda _path: np.zeros((2, 2))),
            runner=lambda track, *_args, **_kwargs: track,
        )

        self.assertEqual(result.tracks, (retrack_request.track,))


if __name__ == "__main__":
    unittest.main()
