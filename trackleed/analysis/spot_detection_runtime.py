"""Runtime adapters for the headless spot-detection workflow."""

from __future__ import annotations

import importlib.util
from pathlib import Path
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
        if (
            not 0 <= config.confidence <= 1
            or not 0 <= config.iou <= 1
            or config.max_detections < 1
            or not config.device.strip()
        ):
            issues.append(
                PreflightIssue(
                    "invalid_config",
                    "YOLO detection settings are outside their valid ranges.",
                )
            )
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
            image = _subtract_background(
                image,
                background,
                runtime_input.analysis_region.metadata_masks,
            )
        return _detect_yolo(
            image,
            runtime_input.analysis_region.metadata_masks,
            config,
        )


def _detect_classic(
    main_image: np.ndarray,
    background_image: np.ndarray | None,
    runtime_input: DetectionRuntimeInput,
    config: ClassicDetectionConfig,
) -> tuple[DetectedSpot, ...]:
    main_float = img_as_float(main_image.copy())
    _zero_rectangles(main_float, runtime_input.analysis_region.metadata_masks)
    subtracted_image = main_float
    if background_image is not None:
        subtracted_image = _subtract_background(
            main_image,
            background_image,
            runtime_input.analysis_region.metadata_masks,
        )

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
    circle = runtime_input.analysis_region.circle
    ellipse = runtime_input.analysis_region.ellipse
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
    main_float = img_as_float(image).copy()
    converted_background = (
        rgb2gray(background) if background.ndim == 3 else img_as_float(background)
    )
    bg_float = converted_background.copy()
    _zero_rectangles(main_float, masks)
    _zero_rectangles(bg_float, masks)
    main_values = main_float[main_float > 0]
    bg_values = bg_float[bg_float > 0]
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


_YOLO_MODEL_CACHE: dict[str, Any] = {}


def _detect_yolo(
    image: np.ndarray,
    masks: tuple[RectSnapshot, ...],
    config: YoloDetectionConfig,
) -> tuple[DetectedSpot, ...]:
    if config.model_path is None:
        raise RuntimeError("No YOLO model selected")
    model = _load_yolo_model(config.model_path, config.device)
    if image.ndim == 2:
        inference_image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        inference_image = np.repeat(image, 3, axis=2)
    else:
        inference_image = image
    if masks:
        inference_image = inference_image.copy()
        _zero_rectangles(inference_image, masks)
    try:
        results = model.predict(
            source=inference_image,
            conf=config.confidence,
            iou=config.iou,
            max_det=config.max_detections,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(f"YOLO inference failed: {exc}") from exc
    if not results:
        return ()
    boxes = results[0].boxes if hasattr(results[0], "boxes") else ()
    if boxes is None:
        return ()
    spots: list[DetectedSpot] = []
    image_height, image_width = inference_image.shape[:2]
    for box in boxes:
        try:
            xyxy = box.xyxy.cpu().numpy().reshape(-1)
            score = (
                float(box.conf.cpu().item())
                if getattr(box, "conf", None) is not None
                else 0.0
            )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            continue
        x0, y0, x1, y1 = (float(value) for value in xyxy.tolist())
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        if any(_contains(mask, center_x, center_y) for mask in masks):
            continue
        spots.append(
            DetectedSpot(
                id=len(spots) + 1,
                roi=_rect_from_xyxy(
                    x0,
                    y0,
                    x1,
                    y1,
                    config.bbox_margin_px,
                    image_width,
                    image_height,
                ),
                center_x=center_x,
                center_y=center_y,
                intensity=score,
                source="yolo",
            )
        )
    return tuple(spots)


def _load_yolo_model(model_path: Path, device: str) -> Any:
    key = str(model_path.resolve())
    if key in _YOLO_MODEL_CACHE:
        return _YOLO_MODEL_CACHE[key]
    if not model_path.is_file():
        raise RuntimeError(f"Model file not found: {model_path}")
    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Ultralytics YOLO and PyTorch are required") from exc
    try:
        model = YOLO(str(model_path))
        selected_device: int | str = device.lower()
        if selected_device in {"auto", "gpu"}:
            selected_device = 0 if torch.cuda.is_available() else "cpu"
        elif selected_device != "cpu":
            selected_device = device
        model.to(selected_device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load YOLO model: {exc}") from exc
    _YOLO_MODEL_CACHE[key] = model
    return model


def _contains(rect: RectSnapshot, x: float, y: float) -> bool:
    return (
        rect.x <= x < rect.x + rect.width
        and rect.y <= y < rect.y + rect.height
    )


def _rect_from_xyxy(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    margin_px: int,
    image_width: int,
    image_height: int,
    min_size: int = 4,
) -> RectSnapshot:
    width = max(x1 - x0 + 2.0 * margin_px, min(min_size, image_width))
    height = max(y1 - y0 + 2.0 * margin_px, min(min_size, image_height))
    width = min(width, image_width)
    height = min(height, image_height)
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    left = min(max(0.0, center_x - width / 2.0), max(0.0, image_width - round(width)))
    top = min(max(0.0, center_y - height / 2.0), max(0.0, image_height - round(height)))
    return RectSnapshot(
        round(left),
        round(top),
        max(1, round(width)),
        max(1, round(height)),
    )
