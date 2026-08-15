"""Immutable configuration contracts for point-family seeded tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

InferenceResolution: TypeAlias = int | tuple[int, int] | None

POINT_TRACKER_METHOD_IDS = (
    "tapir_offline",
    "tapir_online",
    "locotrack_offline",
    "cotracker3_offline",
    "tapnext_offline",
    "trackonr_offline",
    "pipspp_offline",
    "mftiq_offline",
)


@dataclass(frozen=True)
class PointTrackerConfig:
    models_dir: Path
    repo_path: Path | None = None
    checkpoint_path: Path | None = None
    python_executable: Path | None = None
    backend_mode: str = "direct"
    device: str = "auto"
    inference_resolution: InferenceResolution = 256
    roi_size: int = 32
    window_size: int = 64
    window_overlap: int = 8
    keep_invisible_points_as_gaps: bool = True
    max_step_distance_px: float = 0.0
    max_reacquisition_gap: int = 0
    terminate_after_invisible_frames: int = 0
    center_mode: str = "auto"
    manual_center: tuple[float, float] = (0.0, 0.0)
    direction_mode: str = "none"
    lateral_angle_tolerance_deg: float = 20.0
    max_radial_jump_px: float = 0.0
    recenter_mode: str = "centroid"
    recenter_patch_multiplier: float = 1.25
    centroid_rel_threshold: float = 0.35
    centroid_max_shift_ratio: float = 0.45
    subtract_background: bool = False


@dataclass(frozen=True)
class TapirPointConfig(PointTrackerConfig):
    models_dir: Path = Path("tracking_models/tapir")
    runtime_mode: str = "offline"
    query_chunk_size: int = 64
    use_visibility_gating: bool = True
    visibility_threshold: float = 0.5
    use_uncertainty_gating: bool = True
    expected_dist_threshold: float = 0.5


@dataclass(frozen=True)
class LocotrackPointConfig(PointTrackerConfig):
    models_dir: Path = Path("tracking_models/locotrack")
    backend_mode: str = "subprocess"
    model_size: str = "auto"
    query_chunk_size: int = 64
    subprocess_timeout_sec: float = 1800.0
    use_visibility_gating: bool = True
    visibility_threshold: float = 0.5
    use_uncertainty_gating: bool = True
    expected_dist_threshold: float = 0.5


@dataclass(frozen=True)
class CotrackerPointConfig(PointTrackerConfig):
    models_dir: Path = Path("tracking_models/cotracker3")
    window_len: int = 60
    use_visibility_gating: bool = True


@dataclass(frozen=True)
class TapnextPointConfig(PointTrackerConfig):
    models_dir: Path = Path("tracking_models/tapnext")
    autocast: bool = False
    use_visibility_gating: bool = True
    visibility_threshold: float = 0.5
    use_certainty_gating: bool = False
    certainty_threshold: float = 0.5


@dataclass(frozen=True)
class TrackonrPointConfig(PointTrackerConfig):
    models_dir: Path = Path("tracking_models/trackonr")
    backend_mode: str = "subprocess"
    inference_resolution: InferenceResolution = (384, 512)
    support_grid_size: int = 20
    subprocess_timeout_sec: float = 1800.0
    use_visibility_gating: bool = True
    visibility_threshold: float = 0.8


@dataclass(frozen=True)
class PipsppPointConfig(PointTrackerConfig):
    models_dir: Path = Path("tracking_models/pipspp")
    backend_mode: str = "subprocess"
    inference_resolution: InferenceResolution = (256, 384)
    iters: int = 4
    beautify: bool = True
    use_inbounds_visibility: bool = True
    subprocess_timeout_sec: float = 1800.0


@dataclass(frozen=True)
class MftiqPointConfig(PointTrackerConfig):
    models_dir: Path = Path("tracking_models/mftiq")
    backend_mode: str = "subprocess"
    inference_resolution: InferenceResolution = None
    config_path: Path | None = None
    occlusion_threshold: float = 0.5
    subprocess_timeout_sec: float = 1800.0


def default_point_tracker_configs(
    *,
    roi_size: int = 32,
) -> tuple[tuple[str, PointTrackerConfig], ...]:
    return (
        ("tapir_offline", TapirPointConfig(roi_size=roi_size)),
        (
            "tapir_online",
            TapirPointConfig(roi_size=roi_size, runtime_mode="online"),
        ),
        ("locotrack_offline", LocotrackPointConfig(roi_size=roi_size)),
        ("cotracker3_offline", CotrackerPointConfig(roi_size=roi_size)),
        ("tapnext_offline", TapnextPointConfig(roi_size=roi_size)),
        ("trackonr_offline", TrackonrPointConfig(roi_size=roi_size)),
        ("pipspp_offline", PipsppPointConfig(roi_size=roi_size)),
        ("mftiq_offline", MftiqPointConfig(roi_size=roi_size)),
    )
