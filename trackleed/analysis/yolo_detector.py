"""YOLO-based spot detector helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from PyQt6.QtCore import QRect

from trackleed.core.data_models import SpotROI


class YOLODetectionError(RuntimeError):
    """Raised when YOLO inference fails."""


@dataclass
class YOLOSettings:
    model_path: str | None = None
    confidence: float = 0.25
    iou: float = 0.45
    max_det: int = 300
    bbox_margin_px: int = 2
    device: str = "auto"


_model_cache = {}


def detect_spots_with_yolo(
    image: np.ndarray,
    metadata_mask: List[QRect] | None,
    settings: YOLOSettings,
) -> List[SpotROI]:
    if settings.model_path is None:
        raise YOLODetectionError("No YOLO model selected.")

    model = _load_model(settings.model_path, device=settings.device)

    # YOLO expects 3-channel inputs; stack grayscale if needed
    if image.ndim == 2:
        img_rgb = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        img_rgb = np.repeat(image, 3, axis=2)
    else:
        img_rgb = image

    if metadata_mask:
        img_rgb = img_rgb.copy()
        for rect in metadata_mask:
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(img_rgb.shape[1], x + w)
            y1 = min(img_rgb.shape[0], y + h)
            img_rgb[y0:y1, x0:x1] = 0

    try:
        results = model.predict(
            source=img_rgb,
            conf=settings.confidence,
            iou=settings.iou,
            max_det=settings.max_det,
            verbose=False,
        )
    except Exception as exc:
        raise YOLODetectionError(f"YOLO inference failed: {exc}") from exc

    if not results:
        return []
    boxes = results[0].boxes if hasattr(results[0], "boxes") else []
    if boxes is None or len(boxes) == 0:
        return []

    margin = max(0, int(settings.bbox_margin_px))
    rois: List[SpotROI] = []
    for idx, box in enumerate(boxes):
        try:
            xyxy = box.xyxy.cpu().numpy().reshape(-1)
            score = float(box.conf.cpu().item()) if getattr(box, "conf", None) is not None else 0.0
        except Exception:
            continue
        x0, y0, x1, y1 = xyxy.tolist()
        # Skip boxes whose center lies inside metadata masks
        if metadata_mask:
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            skip = False
            for rect in metadata_mask:
                if rect.contains(int(cx), int(cy)):
                    skip = True
                    break
            if skip:
                continue
        rect = QRect(int(x0 - margin), int(y0 - margin), int((x1 - x0) + 2 * margin), int((y1 - y0) + 2 * margin))
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        rois.append(
            SpotROI(
                id=len(rois) + 1,
                roi_rect=rect,
                center_x=float(cx),
                center_y=float(cy),
                intensity=score,
                source="yolo",
            )
        )
    return rois


def _load_model(path_str: str, device: str = "auto"):
    path = Path(path_str)
    key = str(path.resolve())
    if key in _model_cache:
        return _model_cache[key]
    if not path.exists():
        raise YOLODetectionError(f"Model file not found: {path}")
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise YOLODetectionError("Ultralytics YOLO is not installed. Please `pip install ultralytics`.") from exc
    try:
        model = YOLO(str(path))
        dev = device.lower() if isinstance(device, str) else "auto"
        if dev == "auto":
            dev = 0 if torch.cuda.is_available() else "cpu"
        elif dev == "gpu":
            dev = 0 if torch.cuda.is_available() else "cpu"
        elif dev == "cpu":
            dev = "cpu"
        model.to(dev)
    except Exception as exc:
        hint = ""
        if "OrderedDict" in str(exc):
            hint = " (wygląda na state_dict zamiast pełnego checkpointu Ultralytics; użyj best.pt z runs/detect/.../weights)"
        raise YOLODetectionError(f"Failed to load YOLO model: {exc}{hint}") from exc
    _model_cache[key] = model
    return model
