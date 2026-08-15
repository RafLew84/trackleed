from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from trackleed.analysis.seeded_point_tracking_runtime import (
    default_point_tracker_configs,
    point_tracking_definitions,
)
from trackleed.logic.workflows.point_tracking import (
    LocotrackPointConfig,
    PointTrackerConfig,
    TapirPointConfig,
)
from trackleed.logic.workflows.seeded_tracking import (
    RunContext,
    SeededRetrackRequest,
    SeededTrackingFrame,
    SeededTrackingPreflightError,
    SeededTrackingRequest,
    SeedRect,
    TrackedSpotIdentity,
    TrackingFamily,
    TrackingPreflightIssue,
    TrackingResult,
    TrackingSeed,
    TrackObservation,
    default_seeded_tracking_catalog,
    run_seeded_retrack,
    run_seeded_tracking,
)


def _point_request(
    method_id: str,
    config: PointTrackerConfig,
) -> SeededTrackingRequest:
    return SeededTrackingRequest(
        method_id=method_id,
        frames=(
            SeededTrackingFrame(1, Path("one.tif")),
            SeededTrackingFrame(2, Path("two.tif")),
        ),
        reference_frame_id=1,
        seeds=(TrackingSeed(7, 20.0, 21.0, SeedRect(16, 17, 8, 8)),),
        config=config,
    )


def _tapir_request(config: TapirPointConfig | None = None) -> SeededTrackingRequest:
    return _point_request("tapir_offline", config or TapirPointConfig())


class SeededPointTrackingWorkflowTests(unittest.TestCase):
    def test_default_catalog_exposes_every_stable_point_tracker(self) -> None:
        catalog = default_seeded_tracking_catalog()
        point_methods = tuple(
            method
            for method in catalog.methods
            if method.family is TrackingFamily.POINT
        )

        self.assertEqual(
            tuple(method.method_id for method in point_methods),
            tuple(definition.method_id for definition in point_tracking_definitions()),
        )
        self.assertEqual(
            tuple(method.display_name for method in point_methods),
            (
                "Offline TAPIR",
                "BootsTAPIR Online / Causal",
                "Offline LocoTrack",
                "Offline CoTracker3",
                "Offline TAPNext",
                "Offline Track-On-R",
                "Offline PIPs++",
                "Offline MFTIQ",
            ),
        )
        self.assertTrue(
            all(method.capabilities.supports_retrack for method in point_methods)
        )
        self.assertTrue(
            all(method.retrack_runtime is not None for method in point_methods)
        )

    def test_every_point_method_dispatches_execution_and_retrack_by_stable_id(
        self,
    ) -> None:
        calls: list[tuple[str, str]] = []

        def _execute(method_id, _request, _context):
            calls.append(("run", method_id))
            return TrackingResult(())

        def _retrack(method_id, request, _context):
            calls.append(("retrack", method_id))
            return TrackingResult((request.track,))

        for method_id, config in default_point_tracker_configs():
            request = _point_request(method_id, config)
            context = RunContext(
                image_reader=lambda _path: None,
                point_tracker_preflight=lambda _method_id, _config: (),
                point_tracker_executor=_execute,
                point_tracker_retrack_executor=_retrack,
            )
            run_seeded_tracking(request, context)
            run_seeded_retrack(
                SeededRetrackRequest(
                    method_id=method_id,
                    frames=request.frames,
                    track=TrackedSpotIdentity(
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
                    start_frame_id=1,
                    config=config,
                ),
                context,
            )

        expected_ids = tuple(
            definition.method_id for definition in point_tracking_definitions()
        )
        self.assertEqual(
            calls,
            [
                item
                for method_id in expected_ids
                for item in (("run", method_id), ("retrack", method_id))
            ],
        )

    def test_point_workflow_fails_closed_without_execution_adapter(self) -> None:
        with self.assertRaises(SeededTrackingPreflightError) as caught:
            run_seeded_tracking(
                _tapir_request(),
                RunContext(
                    image_reader=lambda _path: None,
                    point_tracker_preflight=lambda _method_id, _config: (),
                ),
            )

        self.assertIn(
            "executor_unavailable",
            {issue.code for issue in caught.exception.report.issues},
        )

    def test_point_config_is_an_immutable_snapshot(self) -> None:
        config = TapirPointConfig(checkpoint_path=Path("tapir.pt"), roi_size=24)

        with self.assertRaises(FrozenInstanceError):
            config.roi_size = 32  # type: ignore[misc]

        self.assertEqual(config.checkpoint_path, Path("tapir.pt"))
        self.assertEqual(config.roi_size, 24)

    def test_public_workflow_repeats_preflight_and_dispatches_point_runtime(
        self,
    ) -> None:
        calls: list[str] = []

        def _preflight(method_id, config):
            calls.append(f"preflight:{method_id}:{config.roi_size}")
            return ()

        def _execute(method_id, request, _context):
            calls.append(f"run:{method_id}:{request.reference_frame_id}")
            seed = request.seeds[0]
            return TrackingResult(
                (
                    TrackedSpotIdentity(
                        seed.seed_id,
                        (
                            TrackObservation(
                                request.reference_frame_id,
                                seed.roi,
                                seed.center_x,
                                seed.center_y,
                                source=method_id,
                            ),
                        ),
                    ),
                )
            )

        result = run_seeded_tracking(
            _tapir_request(TapirPointConfig(roi_size=24)),
            RunContext(
                image_reader=lambda _path: None,
                point_tracker_preflight=_preflight,
                point_tracker_executor=_execute,
            ),
        )

        self.assertEqual(calls, ["preflight:tapir_offline:24", "run:tapir_offline:1"])
        self.assertEqual(result.tracks[0].observations[0].source, "tapir_offline")

    def test_structured_unavailability_prevents_point_execution(self) -> None:
        executed = False

        def _execute(_method_id, _request, _context):
            nonlocal executed
            executed = True
            return TrackingResult(())

        with self.assertRaises(SeededTrackingPreflightError) as caught:
            run_seeded_tracking(
                _tapir_request(),
                RunContext(
                    image_reader=lambda _path: None,
                    point_tracker_preflight=lambda _method_id, _config: (
                        TrackingPreflightIssue(
                            "checkpoint_unavailable",
                            "The configured checkpoint is unavailable.",
                        ),
                    ),
                    point_tracker_executor=_execute,
                ),
            )

        self.assertFalse(executed)
        self.assertEqual(
            tuple(issue.code for issue in caught.exception.report.issues),
            ("checkpoint_unavailable",),
        )

    def test_point_preflight_rejects_invalid_numeric_configuration(self) -> None:
        report = default_seeded_tracking_catalog().preflight(
            _tapir_request(TapirPointConfig(roi_size=0))
        )

        self.assertIn("invalid_config", {issue.code for issue in report.issues})

    def test_missing_checkpoint_is_a_controlled_preflight_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = default_seeded_tracking_catalog().preflight(
                _tapir_request(
                    TapirPointConfig(
                        models_dir=root / "models",
                        checkpoint_path=root / "missing.pt",
                    )
                )
            )

        self.assertFalse(report.ready)
        self.assertIn(
            "checkpoint_unavailable",
            {issue.code for issue in report.issues},
        )

    def test_method_specific_invalid_config_is_rejected_before_probe(self) -> None:
        report = default_seeded_tracking_catalog().preflight(
            _tapir_request(TapirPointConfig(query_chunk_size=0))
        )

        self.assertIn("invalid_config", {issue.code for issue in report.issues})

    def test_missing_subprocess_resources_are_all_structured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = LocotrackPointConfig(
                models_dir=root / "models",
                repo_path=root / "repo",
                checkpoint_path=root / "missing.ckpt",
                python_executable=root / "missing-python",
            )
            report = default_seeded_tracking_catalog().preflight(
                _point_request("locotrack_offline", config)
            )

        issue_codes = {issue.code for issue in report.issues}
        self.assertTrue(
            {
                "checkpoint_unavailable",
                "repository_unavailable",
                "interpreter_unavailable",
            }.issubset(issue_codes)
        )

    def test_existing_but_unlaunchable_interpreter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            invalid_interpreter = root / "not-python"
            invalid_interpreter.write_text("not an executable", encoding="utf-8")
            config = LocotrackPointConfig(
                models_dir=root / "models",
                repo_path=root / "repo",
                checkpoint_path=root / "missing.ckpt",
                python_executable=invalid_interpreter,
            )
            report = default_seeded_tracking_catalog().preflight(
                _point_request("locotrack_offline", config)
            )

        self.assertIn(
            "interpreter_unavailable",
            {issue.code for issue in report.issues},
        )


if __name__ == "__main__":
    unittest.main()
