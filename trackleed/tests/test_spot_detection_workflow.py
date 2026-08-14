from pathlib import Path

import numpy as np

from trackleed.analysis import spot_detection_runtime
from trackleed.logic.workflows.spot_detection import (
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
    SpotDetectionPreflightError,
    YoloDetectionConfig,
    default_spot_detection_catalog,
    run_spot_detection,
)


def test_classic_detection_uses_public_headless_workflow() -> None:
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

    assert [frame.frame_id for frame in result.frames] == [7]
    assert result.frames[0].spots
    nearest = min(
        result.frames[0].spots,
        key=lambda spot: (spot.center_x - 32) ** 2 + (spot.center_y - 32) ** 2,
    )
    assert abs(nearest.center_x - 32) <= 2
    assert abs(nearest.center_y - 32) <= 2
    assert events == [
        ("read", Path("frame.tif")),
        ("progress", (1, 1)),
    ]


def test_default_catalog_exposes_stable_classic_and_yolo_capabilities() -> None:
    catalog = default_spot_detection_catalog()

    assert tuple(method.method_id for method in catalog.methods) == ("classic", "yolo")
    assert catalog.method("classic").capabilities == DetectionCapabilities(
        supports_background=True,
        supports_metadata_masks=True,
        requires_model=False,
    )
    assert catalog.method("yolo").capabilities.requires_model is True


def test_invalid_classic_config_fails_structural_preflight_before_image_read() -> None:
    request = DetectionRequest(
        method_id="classic",
        frames=(DetectionFrame(1, Path("frame.tif")),),
        config=ClassicDetectionConfig(wth_radius=0),
    )

    try:
        run_spot_detection(
            request,
            RunContext(image_reader=lambda _path: (_ for _ in ()).throw(AssertionError())),
        )
    except SpotDetectionPreflightError as exc:
        assert {issue.code for issue in exc.report.issues} == {"invalid_config"}
    else:
        raise AssertionError("Invalid Classic settings must fail before image loading")


def test_unavailable_yolo_fails_preflight_before_reading_images() -> None:
    request = DetectionRequest(
        method_id="yolo",
        frames=(DetectionFrame(1, Path("frame.tif")),),
        config=YoloDetectionConfig(model_path=Path("missing-model.pt")),
    )

    try:
        run_spot_detection(
            request,
            RunContext(image_reader=lambda _path: (_ for _ in ()).throw(AssertionError())),
        )
    except SpotDetectionPreflightError as exc:
        assert "model_unavailable" in {issue.code for issue in exc.report.issues}
    else:
        raise AssertionError("Unavailable YOLO must fail before image loading")


def test_yolo_preflight_is_ready_for_available_runtime_and_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        spot_detection_runtime.importlib.util,
        "find_spec",
        lambda _module_name: object(),
    )

    report = default_spot_detection_catalog().preflight(
        "yolo",
        YoloDetectionConfig(model_path=checkpoint),
    )

    assert report.ready


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


def test_ready_yolo_runs_through_the_same_public_interface() -> None:
    yolo_method = DetectionMethod(
        method_id="yolo",
        display_name="YOLO",
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

    assert result.frames[0].spots[0].source == "yolo"


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
    is_cancelled=lambda: False,
    fail_on_value: int | None = None,
) -> RunContext:
    def read_image(path: Path) -> np.ndarray:
        value = int(path.stem)
        events.append(("read", value))
        return np.full((2, 2), value, dtype=np.uint8)

    method = DetectionMethod(
        method_id="classic",
        display_name="Classic",
        capabilities=DetectionCapabilities(True, True, False),
        runtime=_RecordingClassicRuntime(events, fail_on_value),
    )
    return RunContext(
        image_reader=read_image,
        report_progress=lambda progress: events.append(
            ("progress", progress.completed)
        ),
        is_cancelled=is_cancelled,
        detection_catalog=SpotDetectionCatalog((method,)),
    )


def test_series_is_read_and_processed_one_frame_at_a_time() -> None:
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

    assert tuple(frame.frame_id for frame in result.frames) == (1, 2, 3)
    assert events == [
        ("read", 1),
        ("detect", 1),
        ("progress", 1),
        ("read", 2),
        ("detect", 2),
        ("progress", 2),
        ("read", 3),
        ("detect", 3),
        ("progress", 3),
    ]


def test_cancellation_does_not_return_a_partial_batch() -> None:
    events: list[tuple[str, object]] = []
    request = DetectionRequest(
        method_id="classic",
        frames=(DetectionFrame(1, Path("1.tif")), DetectionFrame(2, Path("2.tif"))),
        config=ClassicDetectionConfig(),
    )
    context = _recording_context(
        events,
        is_cancelled=lambda: ("progress", 1) in events,
    )

    try:
        run_spot_detection(request, context)
    except SpotDetectionCancelled:
        pass
    else:
        raise AssertionError("A cancelled batch must not return partial detections")

    assert events == [("read", 1), ("detect", 1), ("progress", 1)]


def test_cancellation_during_last_detector_call_discards_the_batch() -> None:
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

    try:
        run_spot_detection(request, context)
    except SpotDetectionCancelled:
        pass
    else:
        raise AssertionError("Cancellation during inference must discard the batch")

    assert events == [("read", 1), ("detect", 1)]


def test_runtime_failure_does_not_return_a_partial_batch() -> None:
    events: list[tuple[str, object]] = []
    request = DetectionRequest(
        method_id="classic",
        frames=(DetectionFrame(1, Path("1.tif")), DetectionFrame(2, Path("2.tif"))),
        config=ClassicDetectionConfig(),
    )

    try:
        run_spot_detection(
            request,
            _recording_context(events, fail_on_value=2),
        )
    except SpotDetectionExecutionError as exc:
        assert exc.frame_id == 2
    else:
        raise AssertionError("A failed batch must not return partial detections")

    assert events[-1] == ("detect", 2)
