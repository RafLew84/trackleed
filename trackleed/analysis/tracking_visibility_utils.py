from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from trackleed.core.data_models import SpotROI, Track
from trackleed.analysis.tracking_result_builders import set_spot_attributes


def append_visible_track_spot(
    track: Track,
    *,
    frame_id: int,
    spot: SpotROI,
) -> np.ndarray:
    track.spots_by_image_id[int(frame_id)] = spot
    return np.array(
        [float(spot.center_x), float(spot.center_y)],
        dtype=np.float32,
    )


def append_invisible_track_frame(
    track: Track,
    *,
    frame_id: int,
    forced_spot: SpotROI | None,
    keep_invisible_as_gaps: bool,
    visibility_attribute_name: str,
    extra_attributes: Mapping[str, Any] | None = None,
) -> bool:
    if keep_invisible_as_gaps or forced_spot is None:
        return False

    forced_spot.intensity = 0.0
    attributes: Dict[str, Any] = {
        str(visibility_attribute_name): False,
    }
    if extra_attributes:
        attributes.update(extra_attributes)
    set_spot_attributes(forced_spot, attributes)
    track.spots_by_image_id[int(frame_id)] = forced_spot
    return True


def commit_tracking_visibility_state(
    state: Dict[str, Any],
    *,
    last_valid_xy: np.ndarray | None,
    invisible_streak: int,
    terminated: bool,
    extra_state: Mapping[str, Any] | None = None,
) -> None:
    state["last_valid_xy"] = last_valid_xy
    state["invisible_streak"] = int(invisible_streak)
    state["terminated"] = bool(terminated)
    if extra_state:
        state.update(extra_state)
