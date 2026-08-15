from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import QRect

from trackleed.analysis.seeded_bbox_tracking_runtime import (
    default_bbox_tracker_configs,
)
from trackleed.core.data_models import SpotROI, Track
from trackleed.gui.seeded_bbox_tracking_compatibility import (
    bbox_review_spec_for_source,
    bbox_tracker_gui_spec,
    bbox_tracker_legacy_bindings,
    legacy_bbox_tracker_params,
    run_legacy_bbox_retracker,
    run_legacy_bbox_tracker,
)
from trackleed.logic.workflows.bbox_tracking import TompBoundingBoxConfig
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


def _request() -> SeededTrackingRequest:
    return SeededTrackingRequest(
        "tomp_offline",
        (
            SeededTrackingFrame(1, Path("one.tif")),
            SeededTrackingFrame(2, Path("two.tif")),
        ),
        1,
        (TrackingSeed(7, 20.0, 21.0, SeedRect(16, 17, 8, 8)),),
        TompBoundingBoxConfig(roi_size=24),
        seed_representation=SeedRepresentation.BOUNDING_BOX,
    )


class SeededBboxTrackingCompatibilityTests(unittest.TestCase):
    def test_gui_spec_exposes_subprocess_capability_from_definition(self) -> None:
        functions = SimpleNamespace(
            probe_ostrack_environment=lambda **_kwargs: None,
            format_ostrack_environment_report=lambda _status: "",
            track_spots_ostrack_offline=lambda *_args, **_kwargs: [],
            retrack_single_spot_ostrack_offline=lambda track, *_args, **_kwargs: track,
        )

        with patch(
            "trackleed.gui.seeded_tracking_legacy_models.import_module",
            return_value=functions,
        ):
            spec = bbox_tracker_gui_spec("ostrack_offline")

        self.assertTrue(spec.supports_subprocess)
        self.assertEqual(spec.family, "bbox")

    def test_public_workflows_resolve_every_legacy_bbox_runner(self) -> None:
        calls: list[str] = []
        functions = {}
        for binding in bbox_tracker_legacy_bindings():
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
            bbox_tracker_preflight=lambda _method_id, _config: (),
            bbox_tracker_executor=run_legacy_bbox_tracker,
            bbox_tracker_retrack_executor=run_legacy_bbox_retracker,
        )
        base_request = _request()
        with patch(
            "trackleed.gui.seeded_bbox_tracking_compatibility.import_module",
            return_value=SimpleNamespace(**functions),
        ):
            for method_id, config in default_bbox_tracker_configs():
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
                        SeedRepresentation.BOUNDING_BOX,
                    ),
                    context,
                )

        self.assertEqual(
            calls,
            [
                name
                for binding in bbox_tracker_legacy_bindings()
                for name in (
                    binding.tracking_runner_name,
                    binding.retrack_runner_name,
                )
            ],
        )

    def test_tamos_source_resolves_to_canonical_tomp_method(self) -> None:
        spec = bbox_review_spec_for_source("tamos_offline")

        self.assertIsNotNone(spec)
        self.assertEqual(spec.tracker_id, "tomp_offline")

    def test_params_preserve_bbox_config_and_legacy_aliases(self) -> None:
        config = TompBoundingBoxConfig(
            repo_path=Path("repos/pytracking"),
            checkpoint_path=Path("models/tomp.pth.tar"),
        )

        params = legacy_bbox_tracker_params("tomp_offline", config)

        self.assertEqual(params["repo_path"], "repos/pytracking")
        self.assertEqual(params["tomp_repo_path"], "repos/pytracking")
        self.assertEqual(params["checkpoint_path"], "models/tomp.pth.tar")

    def test_adapter_preserves_bbox_geometry_at_legacy_boundary(self) -> None:
        def runner(
            _files_by_id,
            _frame_ids,
            _reference_frame_id,
            _params,
            **_kwargs,
        ):
            return [
                Track(
                    7,
                    {
                        2: SpotROI(
                            7,
                            QRect(8, 9, 30, 24),
                            23.0,
                            21.0,
                            5.0,
                            "tomp_offline",
                        )
                    },
                )
            ]

        result = run_legacy_bbox_tracker(
            "tomp_offline",
            _request(),
            RunContext(image_reader=lambda _path: np.zeros((2, 2))),
            runner=runner,
        )

        observation = result.tracks[0].observations[0]
        self.assertEqual(observation.roi, SeedRect(8, 9, 30, 24))
        self.assertEqual(observation.source, "tomp_offline")


if __name__ == "__main__":
    unittest.main()
