from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from trackleed.analysis.seeded_mask_tracking_runtime import (
    default_mask_tracker_configs,
)

from trackleed.gui.seeded_mask_tracking_compatibility import (
    legacy_mask_tracker_params,
    mask_review_spec_for_source,
    mask_tracker_legacy_bindings,
    run_legacy_mask_retracker,
    run_legacy_mask_tracker,
)
from trackleed.logic.workflows.mask_tracking import Dam4SamMaskConfig, MaskTrackerConfig
from trackleed.logic.workflows.seeded_tracking import (
    RunContext,
    SeededRetrackRequest,
    SeededTrackingFrame,
    SeededTrackingRequest,
    SeedRect,
    SeedRepresentation,
    TrackedSpotIdentity,
    TrackingSeed,
    TrackObservation,
    run_seeded_retrack,
    run_seeded_tracking,
)


def _request(method_id: str, config: MaskTrackerConfig) -> SeededTrackingRequest:
    return SeededTrackingRequest(
        method_id,
        (
            SeededTrackingFrame(1, Path("one.tif")),
            SeededTrackingFrame(2, Path("two.tif")),
        ),
        1,
        (
            TrackingSeed(
                7,
                20.0,
                21.0,
                SeedRect(16, 17, 8, 8),
                polygon_seed_points=((16.0, 17.0), (24.0, 17.0), (20.0, 25.0)),
            ),
        ),
        config,
        seed_representation=SeedRepresentation.MASK,
    )


class SeededMaskTrackingCompatibilityTests(unittest.TestCase):
    def test_mask_source_resolves_to_headless_review_method(self) -> None:
        spec = mask_review_spec_for_source("  SAM2_OFFLINE  ")

        self.assertIsNotNone(spec)
        self.assertEqual(spec.tracker_id, "sam2_offline")

    def test_params_preserve_mask_config_and_legacy_aliases(self) -> None:
        config = Dam4SamMaskConfig(
            models_dir=Path("models/dam4sam"),
            repo_path=Path("repos/dam4sam"),
            checkpoint_path=Path("models/dam4sam/model.pt"),
            tracker_name="sam21pp-B",
        )

        params = legacy_mask_tracker_params("dam4sam_offline", config)

        self.assertEqual(params["repo_path"], "repos/dam4sam")
        self.assertEqual(params["dam4sam_repo_path"], "repos/dam4sam")
        self.assertEqual(params["dam4sam_models_dir"], "models/dam4sam")
        self.assertEqual(params["dam4sam_tracker_name"], "sam21pp-B")
        self.assertEqual(params["checkpoint_path"], "models/dam4sam/model.pt")

    def test_public_workflows_resolve_every_legacy_mask_runner(self) -> None:
        calls: list[str] = []
        functions = {}
        for binding in mask_tracker_legacy_bindings():
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
            mask_tracker_preflight=lambda _method_id, _config: (),
            mask_tracker_executor=run_legacy_mask_tracker,
            mask_tracker_retrack_executor=run_legacy_mask_retracker,
        )
        base_request = _request("cutie_offline", default_mask_tracker_configs()[0][1])
        with patch(
            "trackleed.gui.seeded_mask_tracking_compatibility.import_module",
            return_value=SimpleNamespace(**functions),
        ):
            for method_id, config in default_mask_tracker_configs():
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
                                    polygon_seed_points=request.seeds[0].polygon_seed_points,
                                ),
                            ),
                        ),
                        1,
                        config,
                        SeedRepresentation.MASK,
                    ),
                    context,
                )

        self.assertEqual(
            calls,
            [
                name
                for binding in mask_tracker_legacy_bindings()
                for name in (
                    binding.tracking_runner_name,
                    binding.retrack_runner_name,
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
