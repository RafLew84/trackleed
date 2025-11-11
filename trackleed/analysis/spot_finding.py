# stal/analysis/spot_finding.py

from typing import List, Optional

import numpy as np
from PyQt6.QtCore import QRect
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.morphology import white_tophat, disk
from skimage.util import img_as_float

from trackleed.core.data_models import SpotROI


def find_spots(
    main_image: np.ndarray,
    background_image: Optional[np.ndarray],
    mask_list: List[QRect],
    params: dict,
) -> List[SpotROI]:
    """
    Analyzes a LEED image to find bright spots using a multi-step pipeline.

    The pipeline includes:
    1. Background normalization and subtraction.
    2. Residual background estimation and subtraction.
    3. White Top-Hat morphological filtering to enhance spots.
    4. Peak detection on the processed image.

    Args:
        main_image: The main LEED image as a NumPy array.
        background_image: The background image as a NumPy array, or None.
        mask_list: A list of QRects defining areas to be ignored.
        params: A dictionary of processing parameters, e.g.,
                {'blur_sigma': 20.0, 'wth_radius': 8, 'roi_size': 32, ...}

    Returns:
        A list of SpotROI objects representing the found spots.
    """
    # --- 1. Preparation and Masking ---
    # Work on float copies to prevent data type issues and modifying originals
    main_float = img_as_float(main_image.copy())

    # Apply metadata mask
    for rect in mask_list:
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        main_float[y : y + h, x : x + w] = 0

    # --- 2. Background Subtraction ---
    subtracted_image = main_float
    if background_image is not None:
        bg_float = img_as_float(background_image.copy())
        # Convert background to grayscale if it's color
        if bg_float.ndim == 3:
            bg_float = rgb2gray(bg_float)

        # Apply mask to background as well
        for rect in mask_list:
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
            bg_float[y : y + h, x : x + w] = 0

        # Normalize background intensity
        # Use mean of non-masked areas to calculate normalization factor
        main_mean = main_float[main_float > 0].mean()
        bg_mean = bg_float[bg_float > 0].mean()

        if bg_mean > 1e-6:  # Avoid division by zero
            norm_factor = main_mean / bg_mean
            normalized_bg = bg_float * norm_factor
            # Initial subtraction
            subtracted_image = main_float - normalized_bg

    # --- 3. Residual Background Subtraction ---
    blur_sigma = params.get("blur_sigma", 20.0)
    # Estimate residual background by heavily blurring the image
    residual_bg = gaussian_filter(subtracted_image, sigma=blur_sigma)
    # Final subtraction
    processed_image = subtracted_image - residual_bg

    # --- 4. White Top-Hat for Spot Enhancement ---
    wth_radius = params.get("wth_radius", 8)
    footprint = disk(wth_radius)
    # The final image for peak finding
    final_image = white_tophat(processed_image, footprint=footprint)

    # --- 5. Peak Finding ---
    # Get parameters from the params dict, with default values
    min_dist = params.get("min_distance", 10)
    thresh_rel = params.get("threshold_rel", 0.1)

    # Find coordinates of local maxima
    coordinates = peak_local_max(
        final_image,
        min_distance=min_dist,
        threshold_rel=thresh_rel,
        exclude_border=True,
    )

    # --- 6. Create SpotROI objects from coordinates ---
    found_spots: List[SpotROI] = []
    roi_size = params.get("roi_size", 32)
    roi_half = roi_size // 2

    for i, (y, x) in enumerate(coordinates):
        roi_rect = QRect(int(x - roi_half), int(y - roi_half), roi_size, roi_size)
        spot = SpotROI(
            id=i + 1,
            roi_rect=roi_rect,
            center_y=float(y),
            center_x=float(x),
            intensity=final_image[y, x],
            source="auto",
        )
        found_spots.append(spot)

    print(f"Found {len(found_spots)} spots.")
    return found_spots
