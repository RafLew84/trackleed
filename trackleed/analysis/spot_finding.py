"""Compatibility adapter for the legacy Classic spot-finding API."""


import numpy as np
from PyQt6.QtCore import QRect

from trackleed.analysis.spot_detection_runtime import ClassicDetectionRuntime
from trackleed.core.data_models import SpotROI
from trackleed.logic.workflows.spot_detection import (
    ClassicDetectionConfig,
    DetectionRuntimeInput,
    RectSnapshot,
)


def find_spots(
    main_image: np.ndarray,
    background_image: np.ndarray | None,
    mask_list: list[QRect],
    params: dict,
) -> list[SpotROI]:
    """Run Classic detection and translate its neutral result to ``SpotROI``."""
    config = ClassicDetectionConfig(
        blur_sigma=float(params.get("blur_sigma", 20.0)),
        wth_radius=int(params.get("wth_radius", 8)),
        min_distance=int(params.get("min_distance", 10)),
        threshold_rel=float(params.get("threshold_rel", 0.1)),
        roi_size=int(params.get("roi_size", 32)),
    )
    runtime_input = DetectionRuntimeInput(
        image=main_image,
        background_image=background_image,
        metadata_masks=tuple(
            RectSnapshot(rect.x(), rect.y(), rect.width(), rect.height())
            for rect in mask_list
        ),
        circle_region=None,
        ellipse_region=None,
    )
    detected = ClassicDetectionRuntime().detect(runtime_input, config)
    spots = [
        SpotROI(
            id=spot.id,
            roi_rect=QRect(
                spot.roi.x,
                spot.roi.y,
                spot.roi.width,
                spot.roi.height,
            ),
            center_x=spot.center_x,
            center_y=spot.center_y,
            intensity=spot.intensity,
            source=spot.source,
            polygon_seed_points=[
                (float(point[0]), float(point[1]))
                for point in spot.polygon_seed_points
            ],
        )
        for spot in detected
    ]
    print(f"Found {len(spots)} spots.")
    return spots
