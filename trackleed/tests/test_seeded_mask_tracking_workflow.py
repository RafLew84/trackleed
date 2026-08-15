from __future__ import annotations

import unittest
import inspect
import subprocess
import tempfile
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trackleed.analysis import seeded_mask_tracking_runtime as mask_runtime
from trackleed.analysis.seeded_mask_tracking_runtime import (
    default_mask_tracker_configs,
    mask_tracking_definitions,
)
from trackleed.logic.workflows.mask_tracking import (
    CutieMaskConfig,
    MaskTrackerConfig,
    Xmem2MaskConfig,
)

from trackleed.logic.workflows.seeded_tracking import (
    RunContext,
    SeededRetrackRequest,
    SeededTrackingFrame,
    SeededTrackingRequest,
    SeedRect,
    SeedRepresentation,
    SeededTrackingPreflightError,
    SeededTrackingCancelled,
    TrackingSeed,
    TrackingFamily,
    TrackingResult,
    TrackedSpotIdentity,
    TrackObservation,
    default_seeded_tracking_catalog,
    run_seeded_retrack,
    run_seeded_tracking,
)


def _request(method_id: str, config: MaskTrackerConfig) -> SeededTrackingRequest:
    return SeededTrackingRequest(
        method_id=method_id,
        frames=(
            SeededTrackingFrame(1, Path("one.tif")),
            SeededTrackingFrame(2, Path("two.tif")),
        ),
        reference_frame_id=1,
        seeds=(TrackingSeed(7, 20.0, 21.0, SeedRect(16, 17, 8, 8)),),
        config=config,
        seed_representation=SeedRepresentation.MASK,
    )


class SeededMaskTrackingWorkflowTests(unittest.TestCase):
    def test_catalog_exposes_stable_mask_methods_and_capabilities(self) -> None:
        methods = tuple(
            method
            for method in default_seeded_tracking_catalog().methods
            if method.family is TrackingFamily.MASK
        )
        self.assertEqual(
            tuple((item.method_id, item.display_name) for item in methods),
            (
                ("cutie_offline", "Offline Cutie"),
                ("xmem2_offline", "Offline XMem2"),
                ("sam2_offline", "Offline SAM2"),
                ("dam4sam_offline", "Offline DAM4SAM"),
                ("samurai_offline", "Offline SAMURAI"),
                ("deva_offline", "Offline DEVA"),
            ),
        )
        self.assertTrue(all(item.capabilities.supports_retrack for item in methods))
        self.assertTrue(
            all(
                item.capabilities.seed_representation is SeedRepresentation.MASK
                for item in methods
            )
        )

    def test_mask_runtime_has_no_gui_or_widget_dependency(self) -> None:
        source = inspect.getsource(mask_runtime)

        self.assertNotIn("PyQt", source)
        self.assertNotIn("trackleed.gui", source)
        self.assertNotIn("widget", source.lower())

    def test_every_mask_method_dispatches_run_and_retrack_by_stable_id(self) -> None:
        calls: list[tuple[str, str]] = []

        def execute(method_id, _request, _context):
            calls.append(("run", method_id))
            return TrackingResult(())

        def retrack(method_id, request, _context):
            calls.append(("retrack", method_id))
            return TrackingResult((request.track,))

        for method_id, config in default_mask_tracker_configs():
            request = _request(method_id, config)
            request = replace(
                request,
                seeds=(
                    replace(
                        request.seeds[0],
                        polygon_seed_points=(
                            (16.0, 17.0),
                            (24.0, 17.0),
                            (20.0, 25.0),
                        ),
                    ),
                ),
            )
            context = RunContext(
                image_reader=lambda _path: None,
                mask_tracker_preflight=lambda _method_id, _config: (),
                mask_tracker_executor=execute,
                mask_tracker_retrack_executor=retrack,
            )
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

        expected_ids = tuple(item.method_id for item in mask_tracking_definitions())
        self.assertEqual(
            calls,
            [
                call
                for method_id in expected_ids
                for call in (("run", method_id), ("retrack", method_id))
            ],
        )

    def test_missing_optional_resources_are_structured_unavailability(self) -> None:
        status = SimpleNamespace(
            repo_path=None,
            checkpoint_exists=False,
            is_ready=False,
        )
        backend = SimpleNamespace(probe_cutie_environment=lambda **_kwargs: status)
        config = CutieMaskConfig(
            repo_path=Path("missing-repo"),
            checkpoint_path=Path("missing-checkpoint.pth"),
            python_executable=Path("missing-python"),
        )

        with patch.object(mask_runtime, "import_module", return_value=backend):
            report = default_seeded_tracking_catalog().preflight(
                _request("cutie_offline", config)
            )

        self.assertFalse(report.ready)
        self.assertTrue(
            {
                "repository_unavailable",
                "checkpoint_unavailable",
                "interpreter_unavailable",
            }.issubset({issue.code for issue in report.issues})
        )

    def test_workflow_repeats_preflight_and_fails_closed(self) -> None:
        executed = False

        def execute(*_args):
            nonlocal executed
            executed = True
            return TrackingResult(())

        with self.assertRaises(SeededTrackingPreflightError) as caught:
            run_seeded_tracking(
                _request("cutie_offline", CutieMaskConfig()),
                RunContext(
                    image_reader=lambda _path: None,
                    mask_tracker_preflight=lambda _method_id, _config: (
                        mask_runtime.TrackingPreflightIssue(
                            "checkpoint_unavailable",
                            "The configured checkpoint is unavailable.",
                        ),
                    ),
                    mask_tracker_executor=execute,
                ),
            )

        self.assertFalse(executed)
        self.assertEqual(
            tuple(issue.code for issue in caught.exception.report.issues),
            ("checkpoint_unavailable",),
        )

    def test_invalid_mask_config_is_rejected_before_probe(self) -> None:
        config = CutieMaskConfig(window_size=0)

        with patch.object(
            mask_runtime,
            "import_module",
            side_effect=AssertionError("invalid config must not probe"),
        ):
            report = default_seeded_tracking_catalog().preflight(
                _request("cutie_offline", config)
            )

        self.assertEqual(
            tuple(issue.code for issue in report.issues),
            ("invalid_config",),
        )

    def test_xmem2_requires_polygon_seed_before_dispatch(self) -> None:
        executed = False

        def execute(*_args):
            nonlocal executed
            executed = True
            return TrackingResult(())

        with self.assertRaises(SeededTrackingPreflightError) as caught:
            run_seeded_tracking(
                _request("xmem2_offline", Xmem2MaskConfig()),
                RunContext(
                    image_reader=lambda _path: None,
                    mask_tracker_preflight=lambda _method_id, _config: (),
                    mask_tracker_executor=execute,
                ),
            )

        self.assertFalse(executed)
        self.assertIn(
            "polygon_seed_required",
            {issue.code for issue in caught.exception.report.issues},
        )

    def test_xmem2_retrack_requires_polygon_seed_before_dispatch(self) -> None:
        config = Xmem2MaskConfig()
        request = _request("xmem2_offline", config)
        retrack_request = SeededRetrackRequest(
            "xmem2_offline",
            request.frames,
            TrackedSpotIdentity(
                7,
                (TrackObservation(1, SeedRect(16, 17, 8, 8), 20.0, 21.0),),
            ),
            1,
            config,
            SeedRepresentation.MASK,
        )

        with self.assertRaises(SeededTrackingPreflightError) as caught:
            run_seeded_retrack(
                retrack_request,
                RunContext(
                    image_reader=lambda _path: None,
                    mask_tracker_preflight=lambda _method_id, _config: (),
                    mask_tracker_retrack_executor=lambda *_args: TrackingResult(()),
                ),
            )

        self.assertIn(
            "polygon_seed_required",
            {issue.code for issue in caught.exception.report.issues},
        )

    def test_subprocess_readiness_uses_configured_interpreter(self) -> None:
        status = SimpleNamespace(
            repo_path=Path("repo"),
            checkpoint_exists=True,
            is_ready=False,
        )
        backend = SimpleNamespace(probe_cutie_environment=lambda **_kwargs: status)
        catalog = default_seeded_tracking_catalog()
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = Path(temp_dir) / "python.exe"
            executable.write_text("stub", encoding="utf-8")
            config = CutieMaskConfig(
                repo_path=Path("repo"),
                checkpoint_path=Path("cutie.pth"),
                python_executable=executable,
            )
            with (
                patch.object(mask_runtime, "import_module", return_value=backend),
                patch.object(
                    mask_runtime.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0),
                ) as run_probe,
            ):
                report = catalog.preflight(
                    _request("cutie_offline", config)
                )

        self.assertTrue(report.ready)
        self.assertEqual(run_probe.call_args.args[0][0], str(executable))

    def test_cancellation_prevents_mask_dispatch(self) -> None:
        with self.assertRaises(SeededTrackingCancelled):
            run_seeded_tracking(
                _request("cutie_offline", CutieMaskConfig()),
                RunContext(
                    image_reader=lambda _path: None,
                    is_cancelled=lambda: True,
                    mask_tracker_preflight=lambda _method_id, _config: (),
                    mask_tracker_executor=lambda *_args: TrackingResult(()),
                ),
            )

    def test_mask_config_is_an_immutable_snapshot(self) -> None:
        config = CutieMaskConfig(roi_size=24)

        with self.assertRaises(FrozenInstanceError):
            config.roi_size = 32  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
