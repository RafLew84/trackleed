"""Compatibility adapter for the legacy Classic spot-finding API."""


import numpy as np
from PyQt6.QtCore import QRect

from trackleed.analysis.spot_detection_runtime import ClassicDetectionRuntime
from trackleed.core.data_models import SpotROI
from trackleed.core.spot_roi_adapter import spot_roi_from_detection
from trackleed.logic.workflows.spot_detection import (
    AnalysisRegionSnapshot,
    DetectionRuntimeInput,
    RectSnapshot,
    classic_config_from_mapping,
)


def find_spots(
    main_image: np.ndarray,
    background_image: np.ndarray | None,
    mask_list: list[QRect],
    params: dict,
) -> list[SpotROI]:
    """Run Classic detection and translate its neutral result to ``SpotROI``."""
    config = classic_config_from_mapping(params)
    runtime_input = DetectionRuntimeInput(
        image=main_image,
        background_image=background_image,
        analysis_region=AnalysisRegionSnapshot(
            metadata_masks=tuple(
                RectSnapshot(rect.x(), rect.y(), rect.width(), rect.height())
                for rect in mask_list
            )
        ),
    )
    detected = ClassicDetectionRuntime().detect(runtime_input, config)
    spots = [spot_roi_from_detection(spot) for spot in detected]
    print(f"Found {len(spots)} spots.")
    return spots
