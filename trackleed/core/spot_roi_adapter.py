"""Compatibility conversion between neutral detections and legacy ROIs."""

from PyQt6.QtCore import QRect

from trackleed.core.data_models import SpotROI
from trackleed.logic.workflows.spot_detection import DetectedSpot


def spot_roi_from_detection(spot: DetectedSpot) -> SpotROI:
    """Translate one neutral detection into a fresh mutable compatibility ROI."""
    return SpotROI(
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
