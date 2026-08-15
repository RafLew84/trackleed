from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from trackleed.analysis.seeded_point_tracking_runtime import (
    default_point_tracker_configs,
    point_tracking_definitions,
)
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

if __name__ == "__main__":
    unittest.main()
