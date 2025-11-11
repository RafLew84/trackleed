# stal/core/data_models.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict
from PyQt6.QtCore import QRect

@dataclass
class LeedFile:
    """
    A data class to hold information about a single LEED file.
    This class stores the metadata parsed from the filename.
    The actual image data is handled by a cached loader function.
    """
    # --- File-related metadata ---
    filepath: Path
    filename: str
    is_background: bool

    # --- Experimental parameters parsed from filename ---
    element: str
    surface: int
    mode: str  # e.g., 'G' for grayscale, 'RGB' for color
    temperature: float
    energy: float

    # --- Optional parameters (may not exist for all files) ---
    id: Optional[int]
    coverage: Optional[float]

    # --- Analysis Results ---
    spots: List['SpotROI'] = field(default_factory=list, repr=False)
    pattern_center: Optional[Dict[str, float]] = None

@dataclass
class SpotROI:
    """
    Represents a single detected spot and its Region of Interest.
    """
    id: int                     # A unique identifier for the spot within the image
    roi_rect: QRect             # The bounding box of the ROI
    center_x: float             # Precise X coordinate of the peak
    center_y: float             # Precise Y coordinate of the peak
    intensity: float            # Max intensity value at the peak
    source: str                 # How the spot was created: 'auto' or 'manual'

@dataclass
class Track:
    """Represents a single track of a spot over multiple frames."""
    track_id: int  # Unique ID for the entire track across all frames
    # A dictionary mapping an image ID to the SpotROI object in that frame
    spots_by_image_id: Dict[int, SpotROI]