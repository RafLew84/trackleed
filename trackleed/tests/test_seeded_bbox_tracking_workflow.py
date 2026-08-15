from __future__ import annotations

import inspect
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trackleed.analysis import seeded_bbox_tracking_runtime as bbox_runtime
from trackleed.analysis.seeded_bbox_tracking_runtime import (
    bbox_tracking_definitions,
    default_bbox_tracker_configs,
)
from trackleed.logic.workflows.bbox_tracking import (
    BboxBackendMode,
    OstrackBoundingBoxConfig,
    TompBoundingBoxConfig,
    TompModelVariant,
)
from trackleed.logic.workflows.seeded_tracking import (
    RunContext,
    SeededRetrackRequest,
    SeededTrackingCancelled,
    SeededTrackingFrame,
    SeededTrackingPreflightError,
    SeededTrackingRequest,
    SeedRect,
    SeedRepresentation,
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


def _request(method_id, config) -> SeededTrackingRequest:
    return SeededTrackingRequest(
        method_id=method_id,
        frames=(
            SeededTrackingFrame(1, Path("one.tif")),
            SeededTrackingFrame(2, Path("two.tif")),
        ),
        reference_frame_id=1,
        seeds=(TrackingSeed(7, 20.0, 21.0, SeedRect(16, 17, 8, 8)),),
        config=config,
        seed_representation=SeedRepresentation.BOUNDING_BOX,
    )


def _retrack_request(method_id, config) -> SeededRetrackRequest:
    return SeededRetrackRequest(
        method_id,
        _request(method_id, config).frames,
        TrackedSpotIdentity(
            7,
            (TrackObservation(1, SeedRect(16, 17, 8, 8), 20.0, 21.0),),
        ),
        1,
        config,
        SeedRepresentation.BOUNDING_BOX,
    )


class SeededBboxTrackingWorkflowTests(unittest.TestCase):
    def test_catalog_exposes_stable_bbox_methods_and_capabilities(self) -> None:
        methods = tuple(
            method
            for method in default_seeded_tracking_catalog().methods
            if method.family is TrackingFamily.BOUNDING_BOX
        )

        self.assertEqual(
            tuple((item.method_id, item.display_name) for item in methods),
            (
                ("tomp_offline", "Offline ToMP / TaMOs"),
                ("ostrack_offline", "Offline OSTrack"),
            ),
        )
        self.assertTrue(all(item.capabilities.supports_retrack for item in methods))
        self.assertTrue(
            all(
                item.capabilities.seed_representation
                is SeedRepresentation.BOUNDING_BOX
                for item in methods
            )
        )

    def test_bbox_runtime_has_no_gui_or_widget_dependency(self) -> None:
        source = inspect.getsource(bbox_runtime)

        self.assertNotIn("PyQt", source)
        self.assertNotIn("trackleed.gui", source)
        self.assertNotIn("widget", source.lower())

    def test_every_bbox_method_dispatches_run_and_retrack_by_stable_id(self) -> None:
        calls: list[tuple[str, str]] = []

        def execute(method_id, _request, _context):
            calls.append(("run", method_id))
            return TrackingResult(())

        def retrack(method_id, request, _context):
            calls.append(("retrack", method_id))
            return TrackingResult((request.track,))

        for method_id, config in default_bbox_tracker_configs():
            context = RunContext(
                image_reader=lambda _path: None,
                bbox_tracker_preflight=lambda _method_id, _config: (),
                bbox_tracker_executor=execute,
                bbox_tracker_retrack_executor=retrack,
            )
            run_seeded_tracking(_request(method_id, config), context)
            run_seeded_retrack(_retrack_request(method_id, config), context)

        expected_ids = tuple(item.method_id for item in bbox_tracking_definitions())
        self.assertEqual(
            calls,
            [
                call
                for method_id in expected_ids
                for call in (("run", method_id), ("retrack", method_id))
            ],
        )

    def test_workflow_repeats_preflight_and_fails_closed(self) -> None:
        executed = False

        def execute(_method_id, _request, _context):
            nonlocal executed
            executed = True
            return TrackingResult(())

        with self.assertRaises(SeededTrackingPreflightError) as caught:
            run_seeded_tracking(
                _request("ostrack_offline", OstrackBoundingBoxConfig()),
                RunContext(
                    image_reader=lambda _path: None,
                    bbox_tracker_preflight=lambda _method_id, _config: (
                        TrackingPreflightIssue(
                            "checkpoint_unavailable",
                            "The configured checkpoint is unavailable.",
                        ),
                    ),
                    bbox_tracker_executor=execute,
                ),
            )

        self.assertFalse(executed)
        self.assertEqual(
            tuple(issue.code for issue in caught.exception.report.issues),
            ("checkpoint_unavailable",),
        )

    def test_missing_optional_resources_are_structured_unavailability(self) -> None:
        status = SimpleNamespace(
            repo_path=None,
            tomp_checkpoint_exists=False,
            tamos_checkpoint_exists=False,
            tomp_ready=False,
            tamos_ready=False,
        )
        backend = SimpleNamespace(probe_tomp_environment=lambda **_kwargs: status)
        config = TompBoundingBoxConfig(
            repo_path=Path("missing-repo"),
            checkpoint_path=Path("missing-checkpoint.pth"),
            python_executable=Path("missing-python"),
        )

        with patch.object(bbox_runtime, "import_module", return_value=backend):
            report = default_seeded_tracking_catalog().preflight(
                _request("tomp_offline", config)
            )

        self.assertFalse(report.ready)
        self.assertTrue(
            {
                "repository_unavailable",
                "checkpoint_unavailable",
                "interpreter_unavailable",
            }.issubset({issue.code for issue in report.issues})
        )

    def test_tomp_subprocess_readiness_uses_configured_interpreter(self) -> None:
        status = SimpleNamespace(
            repo_path=Path("repo"),
            tomp_checkpoint_exists=True,
            tamos_checkpoint_exists=False,
            tomp_ready=False,
            tamos_ready=False,
        )
        backend = SimpleNamespace(probe_tomp_environment=lambda **_kwargs: status)
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = Path(temp_dir) / "python"
            executable.write_text("stub", encoding="utf-8")
            config = TompBoundingBoxConfig(
                repo_path=Path("repo"),
                checkpoint_path=Path("tomp.pth.tar"),
                python_executable=executable,
            )
            with (
                patch.object(bbox_runtime, "import_module", return_value=backend),
                patch.object(
                    bbox_runtime.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0),
                ) as run_probe,
            ):
                report = default_seeded_tracking_catalog().preflight(
                    _request("tomp_offline", config)
                )

        self.assertTrue(report.ready)
        self.assertEqual(run_probe.call_args.args[0][0], str(executable))

    def test_tomp_subprocess_defaults_to_current_interpreter(self) -> None:
        status = SimpleNamespace(
            repo_path=Path("repo"),
            tomp_checkpoint_exists=True,
            tamos_checkpoint_exists=False,
            tomp_ready=False,
            tamos_ready=False,
        )
        backend = SimpleNamespace(probe_tomp_environment=lambda **_kwargs: status)
        config = TompBoundingBoxConfig(
            repo_path=Path("repo"),
            checkpoint_path=Path("tomp.pth.tar"),
        )

        with (
            patch.object(bbox_runtime, "import_module", return_value=backend),
            patch.object(
                bbox_runtime.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run_probe,
        ):
            report = default_seeded_tracking_catalog().preflight(
                _request("tomp_offline", config)
            )

        self.assertTrue(report.ready)
        self.assertEqual(run_probe.call_args.args[0][0], sys.executable)

    def test_invalid_bbox_config_is_rejected_before_probe(self) -> None:
        report = default_seeded_tracking_catalog().preflight(
            _request(
                "ostrack_offline",
                OstrackBoundingBoxConfig(window_size=0),
            )
        )

        self.assertIn("invalid_config", {issue.code for issue in report.issues})

    def test_unknown_tomp_variant_and_malformed_center_are_rejected(self) -> None:
        invalid_variant = TompBoundingBoxConfig(
            model_variant="not-a-real-variant"  # type: ignore[arg-type]
        )
        invalid_center = OstrackBoundingBoxConfig(
            manual_center=(1.0,)  # type: ignore[arg-type]
        )

        variant_report = default_seeded_tracking_catalog().preflight(
            _request("tomp_offline", invalid_variant)
        )
        center_report = default_seeded_tracking_catalog().preflight(
            _request("ostrack_offline", invalid_center)
        )

        self.assertIn("invalid_config", {i.code for i in variant_report.issues})
        self.assertIn("invalid_config", {i.code for i in center_report.issues})

    def test_tomp_variants_are_typed_stable_values(self) -> None:
        self.assertEqual(
            tuple(item.value for item in TompModelVariant),
            ("tomp50", "tomp101", "tamos_resnet50", "tamos_swin_base"),
        )

    def test_ostrack_direct_probe_uses_current_interpreter(self) -> None:
        captured: dict[str, object] = {}
        status = SimpleNamespace(
            repo_path=Path("repo"),
            checkpoint_exists=True,
            config_exists=True,
            is_ready=True,
        )

        def probe(**kwargs):
            captured.update(kwargs)
            return status

        backend = SimpleNamespace(probe_ostrack_environment=probe)
        config = OstrackBoundingBoxConfig(
            backend_mode=BboxBackendMode.DIRECT,
            python_executable=Path("unrelated-subprocess-python"),
        )

        with patch.object(bbox_runtime, "import_module", return_value=backend):
            report = default_seeded_tracking_catalog().preflight(
                _request("ostrack_offline", config)
            )

        self.assertTrue(report.ready)
        self.assertIsNone(captured["python_executable"])

    def test_bbox_retrack_rejects_point_seed_representation(self) -> None:
        request = _retrack_request(
            "ostrack_offline",
            OstrackBoundingBoxConfig(),
        )
        request = request.__class__(
            request.method_id,
            request.frames,
            request.track,
            request.start_frame_id,
            request.config,
            SeedRepresentation.POINT,
        )

        with self.assertRaises(SeededTrackingPreflightError) as caught:
            run_seeded_retrack(
                request,
                RunContext(
                    image_reader=lambda _path: None,
                    bbox_tracker_preflight=lambda _method_id, _config: (),
                    bbox_tracker_retrack_executor=(
                        lambda _method_id, _request, _context: TrackingResult(())
                    ),
                ),
            )

        self.assertIn(
            "unsupported_seed_representation",
            {issue.code for issue in caught.exception.report.issues},
        )

    def test_bbox_config_is_an_immutable_snapshot(self) -> None:
        config = OstrackBoundingBoxConfig(roi_size=24)

        with self.assertRaises(FrozenInstanceError):
            config.roi_size = 32  # type: ignore[misc]

    def test_cancellation_prevents_bbox_dispatch(self) -> None:
        with self.assertRaises(SeededTrackingCancelled):
            run_seeded_tracking(
                _request("ostrack_offline", OstrackBoundingBoxConfig()),
                RunContext(
                    image_reader=lambda _path: None,
                    is_cancelled=lambda: True,
                    bbox_tracker_preflight=lambda _method_id, _config: (),
                    bbox_tracker_executor=lambda *_args: TrackingResult(()),
                ),
            )


if __name__ == "__main__":
    unittest.main()
