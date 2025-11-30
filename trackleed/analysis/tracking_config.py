"""
Default parameter definitions for directional and hybrid tracking modes.
These defaults are intended to be consumed by GUI/state persistence and the
tracking routines (Hungarian + directional gating, YOLO+OF hybrid).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class DirectionalHybridTrackingParams:
    # Center handling
    center_mode: str = "auto"  # "auto" = image center / pattern center; "manual" uses manual_center
    manual_center: Optional[Tuple[float, float]] = None  # (cx, cy) in pixels when center_mode == "manual"

    # Direction gating
    direction_mode: str = "auto"  # "inward", "outward", or "auto" (median radial sign)
    lateral_angle_tolerance_deg: float = 20.0  # allowable side drift relative to the radial vector
    ring_bin_px: float = 25.0  # bin size for assigning spots to concentric rings
    allow_direction_flip: bool = True  # enable global flip when the majority radial sign changes

    # Distance and assignment gating
    base_max_distance_px: float = 20.0  # fallback cap if dynamic is disabled
    dynamic_max_distance: bool = True  # adjust max distance from recent median step
    dynamic_margin_px: float = 5.0  # extra slack added to median step when dynamic gating is on
    use_ring_penalty: bool = True  # penalize jumps between distant rings

    # Track lifecycle
    max_gap: int = 2  # existing gap handling (frames)
    max_misses: int = 3  # tolerated consecutive misses before killing a track
    min_track_length: int = 2  # discard tracks shorter than this
    allow_new_tracks_any_frame: bool = True  # allow births mid-sequence

    # Hybrid mode
    hybrid_mode: bool = False  # enable YOLO+OF hybrid pipeline
    propagate_with_flow: bool = True  # use LK/RAFT-lite between YOLO refreshes
    yolo_refresh_interval: int = 5  # run YOLO every N frames in hybrid mode


def default_directional_hybrid_params(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return a dict with default directional/hybrid tracking parameters, optionally
    patched with user-provided overrides.
    """
    cfg = asdict(DirectionalHybridTrackingParams())
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = value
    return cfg
