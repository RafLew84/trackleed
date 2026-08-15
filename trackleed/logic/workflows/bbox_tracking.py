"""Immutable configuration contracts for bounding-box seeded tracking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeAlias

InferenceResolution: TypeAlias = int | tuple[int, int] | None


class BboxBackendMode(str, Enum):
    DIRECT = "direct"
    SUBPROCESS = "subprocess"


class BboxDeviceMode(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class RepresentativePointMode(str, Enum):
    BBOX_CENTER = "bbox_center"
    PEAK_IN_BBOX = "peak_in_bbox"
    GAUSSIAN_IN_BBOX = "gaussian_in_bbox"


class BboxCenterMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"


class BboxDirectionMode(str, Enum):
    NONE = "none"
    AUTO = "auto"
    INWARD = "inward"
    OUTWARD = "outward"


class TompModelVariant(str, Enum):
    TOMP50 = "tomp50"
    TOMP101 = "tomp101"
    TAMOS_RESNET50 = "tamos_resnet50"
    TAMOS_SWIN_BASE = "tamos_swin_base"


@dataclass(frozen=True)
class BoundingBoxTrackerConfig:
    repo_path: Path | None = None
    checkpoint_path: Path | None = None
    python_executable: Path | None = None
    backend_mode: BboxBackendMode = BboxBackendMode.SUBPROCESS
    device: BboxDeviceMode = BboxDeviceMode.AUTO
    representative_point_mode: RepresentativePointMode = (
        RepresentativePointMode.BBOX_CENTER
    )
    inference_resolution: InferenceResolution = None
    roi_size: int = 32
    window_size: int = 64
    window_overlap: int = 8
    initial_bbox_padding_px: float = 0.0
    subprocess_timeout_sec: float = 1800.0
    use_presence_score_gating: bool = False
    presence_threshold: float = 0.0
    keep_invisible_points_as_gaps: bool = True
    max_step_distance_px: float = 0.0
    max_reacquisition_gap: int = 0
    terminate_after_invisible_frames: int = 0
    center_mode: BboxCenterMode = BboxCenterMode.AUTO
    manual_center: tuple[float, float] = (0.0, 0.0)
    direction_mode: BboxDirectionMode = BboxDirectionMode.NONE
    lateral_angle_tolerance_deg: float = 20.0
    max_radial_jump_px: float = 0.0


@dataclass(frozen=True)
class OstrackBoundingBoxConfig(BoundingBoxTrackerConfig):
    models_dir: Path = Path("tracking_models/ostrack")
    model_variant: str = "vitb_256_mae_ce_32x4_ep300"


@dataclass(frozen=True)
class TompBoundingBoxConfig(BoundingBoxTrackerConfig):
    tomp_models_dir: Path = Path("tracking_models/tomp")
    tamos_models_dir: Path = Path("tracking_models/tamos")
    model_variant: TompModelVariant = TompModelVariant.TOMP50
    representative_point_mode: RepresentativePointMode = (
        RepresentativePointMode.PEAK_IN_BBOX
    )
