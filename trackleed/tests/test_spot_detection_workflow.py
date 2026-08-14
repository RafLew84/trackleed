import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import numpy as np

from trackleed.analysis import spot_detection_runtime
from trackleed.logic.workflows.spot_detection import (
    AnalysisRegionSnapshot,
    ClassicDetectionConfig,
    DetectedSpot,
    DetectionCapabilities,
    DetectionFrame,
    DetectionMethod,
    DetectionRequest,
    DetectionRuntimeInput,
    PreflightIssue,
    RectSnapshot,
    RunContext,
    SpotDetectionCancelled,
    SpotDetectionCatalog,
    SpotDetectionExecutionError,
    SpotDetectionFamily,
    SpotDetectionPreflightError,
    YoloDetectionConfig,
    default_spot_detection_catalog,
    run_spot_detection,
)


class _ReadyYoloRuntime:
    def preflight(self, config: object) -> tuple[PreflightIssue, ...]:
        assert isinstance(config, YoloDetectionConfig)
        return ()

    def detect(
        self,
        runtime_input: DetectionRuntimeInput,
        config: object,
    ) -> tuple[DetectedSpot, ...]:
        assert runtime_input.image.shape == (8, 8)
        assert isinstance(config, YoloDetectionConfig)
        return (
            DetectedSpot(
                id=1,
                roi=RectSnapshot(1, 2, 4, 4),
                center_x=3.0,
                center_y=4.0,
                source="yolo",
            ),
        )


class _RecordingClassicRuntime:
    def __init__(
        self,
        events: list[tuple[str, object]],
        fail_on_value: int | None = None,
    ) -> None:
        self.events = events
        self.fail_on_value = fail_on_value

    def preflight(self, config: object) -> tuple[PreflightIssue, ...]:
        assert isinstance(config, ClassicDetectionConfig)
        return ()

    def detect(
        self,
        runtime_input: DetectionRuntimeInput,
        config: object,
    ) -> tuple[DetectedSpot, ...]:
        value = int(runtime_input.image[0, 0])
        self.events.append(("detect", value))
        if value == self.fail_on_value:
            raise RuntimeError("detector failed")
        return ()


def _recording_context(
    events: list[tuple[str, object]],
    *,
    is_cancelled: Callable[[], bool] | None = None,
    fail_on_value: int | None = None,
) -> RunContext:
    def read_image(path: Path) -> np.ndarray:
        value = int(path.stem)
        events.append(("read", value))
        return np.full((2, 2), value, dtype=np.uint8)

    method = DetectionMethod(
        method_id="classic",
        display_name="Classic",
        family=SpotDetectionFamily.CLASSICAL,
        capabilities=DetectionCapabilities(True, True, False),
        runtime=_RecordingClassicRuntime(events, fail_on_value),
    )
    return RunContext(
        image_reader=read_image,
        report_progress=lambda progress: events.append(
            ("progress", progress.completed)
        ),
        is_cancelled=is_cancelled or (lambda: False),
        detection_catalog=SpotDetectionCatalog((method,)),
    )


def _unexpected_image_read(_path: Path) -> np.ndarray:
    raise AssertionError("Preflight must run before image loading")


class SpotDetectionWorkflowTests(unittest.TestCase):
    def test_classic_detection_uses_public_headless_workflow(self) -> None:
        image = np.zeros((64, 64), dtype=np.uint8)
        image[30:35, 30:35] = 255
        events: list[tuple[str, object]] = []

        def read_image(path: Path) -> np.ndarray:
            events.append(("read", path))
            return image

        request = DetectionRequest(
            method_id="classic",
            frames=(DetectionFrame(frame_id=7, source_path=Path("frame.tif")),),
            config=ClassicDetectionConfig(
                blur_sigma=1.0,
                wth_radius=2,
                min_distance=3,
                threshold_rel=0.05,
                roi_size=8,
            ),
        )
        context = RunContext(
            image_reader=read_image,
            report_progress=lambda progress: events.append(
                ("progress", (progress.completed, progress.total))
            ),
        )

        result = run_spot_detection(request, context)

        self.assertEqual([frame.frame_id for frame in result.frames], [7])
        self.assertTrue(result.frames[0].spots)
        nearest = min(
            result.frames[0].spots,
            key=lambda spot: (spot.center_x - 32) ** 2 + (spot.center_y - 32) ** 2,
        )
        self.assertLessEqual(abs(nearest.center_x - 32), 2)
        self.assertLessEqual(abs(nearest.center_y - 32), 2)
        self.assertEqual(
            events,
            [("read", Path("frame.tif")), ("progress", (1, 1))],
        )

    def test_default_catalog_exposes_stable_ids_families_and_capabilities(self) -> None:
        catalog = default_spot_detection_catalog()

        self.assertEqual(
            tuple(method.method_id for method in catalog.methods),
            ("classic", "yolo"),
        )
        self.assertEqual(
            catalog.method("classic").capabilities,
            DetectionCapabilities(True, True, False),
        )
        self.assertIs(
            catalog.method("classic").family,
            SpotDetectionFamily.CLASSICAL,
        )
        self.assertIs(catalog.method("yolo").family, SpotDetectionFamily.NEURAL)
        self.assertTrue(catalog.method("yolo").capabilities.requires_model)

    def test_invalid_classic_config_fails_preflight_before_image_read(self) -> None:
        request = DetectionRequest(
            method_id="classic",
            frames=(DetectionFrame(1, Path("frame.tif")),),
            config=ClassicDetectionConfig(wth_radius=0),
        )

        with self.assertRaises(SpotDetectionPreflightError) as raised:
            run_spot_detection(request, RunContext(_unexpected_image_read))

        self.assertEqual(
            {issue.code for issue in raised.exception.report.issues},
            {"invalid_config"},
        )

    def test_classic_background_subtraction_preserves_metadata_mask(self) -> None:
        image = np.zeros((64, 64), dtype=np.uint8)
        image[30:35, 30:35] = 255
        background = np.ones((64, 64), dtype=np.uint8)
        request = DetectionRequest(
            method_id="classic",
            frames=(DetectionFrame(1, Path("frame.tif")),),
            config=ClassicDetectionConfig(
                blur_sigma=1.0,
                wth_radius=2,
                min_distance=3,
                threshold_rel=0.05,
                roi_size=8,
            ),
            background_path=Path("background.tif"),
            analysis_region=AnalysisRegionSnapshot(
                metadata_masks=(RectSnapshot(28, 28, 10, 10),)
            ),
        )

        result = run_spot_detection(
            request,
            RunContext(
                image_reader=lambda path: (
                    background if path.name == "background.tif" else image
                )
            ),
        )

        self.assertEqual(result.frames[0].spots, ())

    def test_preflight_rejects_empty_or_duplicate_frame_identity(self) -> None:
        empty = DetectionRequest("classic", (), ClassicDetectionConfig())
        duplicate = DetectionRequest(
            "classic",
            (
                DetectionFrame(1, Path("one.tif")),
                DetectionFrame(1, Path("two.tif")),
            ),
            ClassicDetectionConfig(),
        )

        for request, expected_code in (
            (empty, "frames_required"),
            (duplicate, "duplicate_frame_id"),
        ):
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(SpotDetectionPreflightError) as raised:
                    run_spot_detection(request, RunContext(_unexpected_image_read))
                self.assertIn(
                    expected_code,
                    {issue.code for issue in raised.exception.report.issues},
                )

    def test_preflight_rejects_yolo_background_mode_without_background(self) -> None:
        request = DetectionRequest(
            method_id="yolo",
            frames=(DetectionFrame(1, Path("frame.tif")),),
            config=YoloDetectionConfig(
                model_path=Path("missing-model.pt"),
                confidence=1.5,
                subtract_background=True,
            ),
        )

        with self.assertRaises(SpotDetectionPreflightError) as raised:
            run_spot_detection(request, RunContext(_unexpected_image_read))

        issue_codes = {issue.code for issue in raised.exception.report.issues}
        self.assertIn("invalid_config", issue_codes)
        self.assertIn("background_required", issue_codes)

    def test_unavailable_yolo_fails_preflight_before_image_read(self) -> None:
        request = DetectionRequest(
            method_id="yolo",
            frames=(DetectionFrame(1, Path("frame.tif")),),
            config=YoloDetectionConfig(model_path=Path("missing-model.pt")),
        )

        with self.assertRaises(SpotDetectionPreflightError) as raised:
            run_spot_detection(request, RunContext(_unexpected_image_read))

        self.assertIn(
            "model_unavailable",
            {issue.code for issue in raised.exception.report.issues},
        )

    def test_yolo_preflight_is_ready_when_runtime_and_checkpoint_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pt"
            checkpoint.touch()
            with patch.object(
                spot_detection_runtime.importlib.util,
                "find_spec",
                return_value=object(),
            ):
                request = DetectionRequest(
                    method_id="yolo",
                    frames=(DetectionFrame(1, Path("frame.tif")),),
                    config=YoloDetectionConfig(model_path=checkpoint),
                )
                report = default_spot_detection_catalog().preflight(request)

        self.assertTrue(report.ready)

    def test_ready_yolo_runs_through_the_same_public_interface(self) -> None:
        yolo_method = DetectionMethod(
            method_id="yolo",
            display_name="YOLO",
            family=SpotDetectionFamily.NEURAL,
            capabilities=DetectionCapabilities(True, True, True),
            runtime=_ReadyYoloRuntime(),
        )
        request = DetectionRequest(
            method_id="yolo",
            frames=(DetectionFrame(3, Path("frame.tif")),),
            config=YoloDetectionConfig(model_path=Path("model.pt")),
        )

        result = run_spot_detection(
            request,
            RunContext(
                image_reader=lambda _path: np.zeros((8, 8), dtype=np.uint8),
                detection_catalog=SpotDetectionCatalog((yolo_method,)),
            ),
        )

        self.assertEqual(result.frames[0].spots[0].source, "yolo")

    def test_series_is_read_and_processed_one_frame_at_a_time(self) -> None:
        events: list[tuple[str, object]] = []
        request = DetectionRequest(
            method_id="classic",
            frames=tuple(
                DetectionFrame(frame_id=value, source_path=Path(f"{value}.tif"))
                for value in (1, 2, 3)
            ),
            config=ClassicDetectionConfig(),
        )

        result = run_spot_detection(request, _recording_context(events))

        self.assertEqual(tuple(frame.frame_id for frame in result.frames), (1, 2, 3))
        self.assertEqual(
            events,
            [
                ("read", 1),
                ("detect", 1),
                ("progress", 1),
                ("read", 2),
                ("detect", 2),
                ("progress", 2),
                ("read", 3),
                ("detect", 3),
                ("progress", 3),
            ],
        )

    def test_cancellation_does_not_return_partial_batch(self) -> None:
        events: list[tuple[str, object]] = []
        request = DetectionRequest(
            method_id="classic",
            frames=(
                DetectionFrame(1, Path("1.tif")),
                DetectionFrame(2, Path("2.tif")),
            ),
            config=ClassicDetectionConfig(),
        )
        context = _recording_context(
            events,
            is_cancelled=lambda: ("progress", 1) in events,
        )

        with self.assertRaises(SpotDetectionCancelled):
            run_spot_detection(request, context)

        self.assertEqual(events, [("read", 1), ("detect", 1), ("progress", 1)])

    def test_cancellation_during_last_detector_call_discards_batch(self) -> None:
        events: list[tuple[str, object]] = []
        request = DetectionRequest(
            method_id="classic",
            frames=(DetectionFrame(1, Path("1.tif")),),
            config=ClassicDetectionConfig(),
        )
        context = _recording_context(
            events,
            is_cancelled=lambda: ("detect", 1) in events,
        )

        with self.assertRaises(SpotDetectionCancelled):
            run_spot_detection(request, context)

        self.assertEqual(events, [("read", 1), ("detect", 1)])

    def test_runtime_failure_does_not_return_partial_batch(self) -> None:
        events: list[tuple[str, object]] = []
        request = DetectionRequest(
            method_id="classic",
            frames=(
                DetectionFrame(1, Path("1.tif")),
                DetectionFrame(2, Path("2.tif")),
            ),
            config=ClassicDetectionConfig(),
        )

        with self.assertRaises(SpotDetectionExecutionError) as raised:
            run_spot_detection(
                request,
                _recording_context(events, fail_on_value=2),
            )

        self.assertEqual(raised.exception.frame_id, 2)
        self.assertEqual(events[-1], ("detect", 2))
