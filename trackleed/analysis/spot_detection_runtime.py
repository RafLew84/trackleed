"""Runtime adapters for the headless spot-detection workflow."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.morphology import disk, white_tophat
from skimage.util import img_as_float

from trackleed.logic.workflows.spot_detection import (
    ClassicDetectionConfig,
    DetectedSpot,
    DetectionConfig,
    DetectionRuntimeInput,
    PreflightIssue,
    RectSnapshot,
    YoloDetectionConfig,
)


class ClassicDetectionRuntime:
    def preflight(self, config: DetectionConfig) -> tuple[PreflightIssue, ...]:
        if not isinstance(config, ClassicDetectionConfig):
            return (
                PreflightIssue(
                    "invalid_config",
                    "Classic detection requires ClassicDetectionConfig.",
                ),
            )
        if (
            config.blur_sigma < 0
            or config.wth_radius < 1
            or config.min_distance < 1
            or not 0 <= config.threshold_rel <= 1
            or config.roi_size < 1
        ):
            return (
                PreflightIssue(
                    "invalid_config",
                    "Classic detection settings are outside their valid ranges.",
                ),
            )
        return ()

    def detect(
        self,
        runtime_input: DetectionRuntimeInput,
        config: DetectionConfig,
    ) -> tuple[DetectedSpot, ...]:
        if not isinstance(config, ClassicDetectionConfig):
            raise TypeError("Classic detection requires ClassicDetectionConfig")
        main_image = _apply_analysis_regions(runtime_input.image, runtime_input)
        background = runtime_input.background_image
        if background is not None:
            background = _apply_analysis_regions(background, runtime_input)
        return _detect_classic(main_image, background, runtime_input, config)


class YoloDetectionRuntime:
    def preflight(self, config: DetectionConfig) -> tuple[PreflightIssue, ...]:
        if not isinstance(config, YoloDetectionConfig):
            return (
                PreflightIssue(
                    "invalid_config",
                    "YOLO detection requires YoloDetectionConfig.",
                ),
            )
        issues: list[PreflightIssue] = []
        if config.model_path is None:
            issues.append(PreflightIssue("model_required", "Select a YOLO model."))
        elif not config.model_path.is_file():
            issues.append(
                PreflightIssue(
                    "model_unavailable",
                    f"YOLO model file not found: {config.model_path}",
                )
            )
        if importlib.util.find_spec("ultralytics") is None:
            issues.append(
                PreflightIssue(
                    "backend_unavailable",
                    "Ultralytics YOLO is not installed.",
                )
            )
        if importlib.util.find_spec("torch") is None:
            issues.append(
                PreflightIssue("backend_unavailable", "PyTorch is not installed.")
            )
        return tuple(issues)

    def detect(
        self,
        runtime_input: DetectionRuntimeInput,
        config: DetectionConfig,
    ) -> tuple[DetectedSpot, ...]:
        if not isinstance(config, YoloDetectionConfig):
            raise TypeError("YOLO detection requires YoloDetectionConfig")
        image = _apply_analysis_regions(runtime_input.image, runtime_input)
        background = runtime_input.background_image
        if config.subtract_background:
            if background is None:
                raise ValueError(
                    "Background file not loaded. Load a background image before subtracting."
                )
            background = _apply_analysis_regions(background, runtime_input)
            image = _subtract_background(image, background, runtime_input.metadata_masks)
        return _run_legacy_yolo(image, runtime_input, config)


def _detect_classic(
    main_image: np.ndarray,
    background_image: np.ndarray | None,
    runtime_input: DetectionRuntimeInput,
    config: ClassicDetectionConfig,
) -> tuple[DetectedSpot, ...]:
    main_float = img_as_float(main_image.copy())
    _zero_rectangles(main_float, runtime_input.metadata_masks)
    subtracted_image = main_float
    if background_image is not None:
        bg_float = img_as_float(background_image.copy())
        if bg_float.ndim == 3:
            bg_float = rgb2gray(bg_float)
        _zero_rectangles(bg_float, runtime_input.metadata_masks)
        main_values = main_float[main_float > 0]
        bg_values = bg_float[bg_float > 0]
        main_mean = float(main_values.mean()) if main_values.size else 0.0
        bg_mean = float(bg_values.mean()) if bg_values.size else 0.0
        if bg_mean > 1e-6:
            subtracted_image = main_float - bg_float * (main_mean / bg_mean)

    residual = gaussian_filter(subtracted_image, sigma=config.blur_sigma)
    enhanced = white_tophat(
        subtracted_image - residual,
        footprint=disk(config.wth_radius),
    )
    coordinates = peak_local_max(
        enhanced,
        min_distance=config.min_distance,
        threshold_rel=config.threshold_rel,
        exclude_border=True,
    )
    roi_half = config.roi_size // 2
    return tuple(
        DetectedSpot(
            id=index,
            roi=RectSnapshot(
                int(x - roi_half),
                int(y - roi_half),
                config.roi_size,
                config.roi_size,
            ),
            center_x=float(x),
            center_y=float(y),
            intensity=float(enhanced[y, x]),
            source="auto",
        )
        for index, (y, x) in enumerate(coordinates, start=1)
    )


def _apply_analysis_regions(
    image: np.ndarray,
    runtime_input: DetectionRuntimeInput,
) -> np.ndarray:
    circle = runtime_input.circle_region
    ellipse = runtime_input.ellipse_region
    if circle is None and ellipse is None:
        return image
    output = image.copy()
    height, width = output.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    if circle is not None:
        inside = (
            (xx - circle.center_x) ** 2 + (yy - circle.center_y) ** 2
            <= circle.radius**2
        )
        output[~inside] = 0
    if ellipse is not None and ellipse.radius_x > 0 and ellipse.radius_y > 0:
        theta = np.deg2rad(ellipse.angle_degrees)
        dx = xx - ellipse.center_x
        dy = yy - ellipse.center_y
        xr = np.cos(theta) * dx + np.sin(theta) * dy
        yr = -np.sin(theta) * dx + np.cos(theta) * dy
        inside = (
            xr**2 / ellipse.radius_x**2 + yr**2 / ellipse.radius_y**2 <= 1.0
        )
        output[~inside] = 0
    return output


def _subtract_background(
    image: np.ndarray,
    background: np.ndarray,
    masks: tuple[RectSnapshot, ...],
) -> np.ndarray:
    main_float = img_as_float(image)
    bg_float = rgb2gray(background) if background.ndim == 3 else img_as_float(background)
    main_for_mean = main_float.copy()
    bg_for_mean = bg_float.copy()
    _zero_rectangles(main_for_mean, masks)
    _zero_rectangles(bg_for_mean, masks)
    main_values = main_for_mean[main_for_mean > 0]
    bg_values = bg_for_mean[bg_for_mean > 0]
    main_mean = float(main_values.mean()) if main_values.size else 0.0
    bg_mean = float(bg_values.mean()) if bg_values.size else 0.0
    scale = main_mean / bg_mean if bg_mean > 1e-9 else 1.0
    return main_float - bg_float * scale


def _zero_rectangles(
    image: np.ndarray,
    masks: tuple[RectSnapshot, ...],
) -> None:
    for rect in masks:
        image[
            rect.y : rect.y + rect.height,
            rect.x : rect.x + rect.width,
        ] = 0


def _run_legacy_yolo(
    image: np.ndarray,
    runtime_input: DetectionRuntimeInput,
    config: YoloDetectionConfig,
) -> tuple[DetectedSpot, ...]:
    from PyQt6.QtCore import QRect

    from trackleed.analysis.yolo_detector import YOLOSettings, detect_spots_with_yolo

    masks = [QRect(mask.x, mask.y, mask.width, mask.height) for mask in runtime_input.metadata_masks]
    settings = YOLOSettings(
        model_path=str(config.model_path) if config.model_path is not None else None,
        confidence=config.confidence,
        iou=config.iou,
        max_det=config.max_detections,
        bbox_margin_px=config.bbox_margin_px,
        device=config.device,
    )
    return tuple(_from_legacy_spot(spot) for spot in detect_spots_with_yolo(image, masks, settings))


def _from_legacy_spot(spot: Any) -> DetectedSpot:
    rect = spot.roi_rect
    polygon = tuple(
        (float(point[0]), float(point[1]))
        for point in getattr(spot, "polygon_seed_points", ())
    )
    return DetectedSpot(
        id=int(spot.id),
        roi=RectSnapshot(rect.x(), rect.y(), rect.width(), rect.height()),
        center_x=float(spot.center_x),
        center_y=float(spot.center_y),
        intensity=float(getattr(spot, "intensity", 0.0)),
        source=str(getattr(spot, "source", "yolo")),
        polygon_seed_points=polygon,
    )
