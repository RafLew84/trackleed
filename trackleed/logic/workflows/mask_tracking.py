"""Immutable configuration contracts for mask-family seeded tracking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeAlias

InferenceResolution: TypeAlias = int | tuple[int, int] | None


class MaskBackendMode(str, Enum):
    DIRECT = "direct"
    SUBPROCESS = "subprocess"


class MaskDeviceMode(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class MaskRepresentativePointMode(str, Enum):
    MASK_CENTROID = "mask_centroid"
    PEAK_IN_MASK = "peak_in_mask"
    BBOX_CENTER = "bbox_center"
    GAUSSIAN_IN_MASK = "gaussian_in_mask"
    PEAK_IN_BBOX = "peak_in_bbox"
    GAUSSIAN_IN_BBOX = "gaussian_in_bbox"


class InitialMaskMode(str, Enum):
    THRESHOLD = "threshold"
    FULL_BBOX = "full_bbox"
    DISK = "disk"


class Sam2PromptMode(str, Enum):
    POINT = "point"
    BBOX = "bbox"
    POINT_AND_BBOX = "point+bbox"


class MaskCenterMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"


class MaskDirectionMode(str, Enum):
    NONE = "none"
    AUTO = "auto"
    INWARD = "inward"
    OUTWARD = "outward"


@dataclass(frozen=True)
class MaskTrackerConfig:
    models_dir: Path
    repo_path: Path | None = None
    checkpoint_path: Path | None = None
    python_executable: Path | None = None
    backend_mode: MaskBackendMode = MaskBackendMode.SUBPROCESS
    device: MaskDeviceMode = MaskDeviceMode.AUTO
    representative_point_mode: MaskRepresentativePointMode = (
        MaskRepresentativePointMode.MASK_CENTROID
    )
    roi_size: int = 32
    window_size: int = 64
    window_overlap: int = 8
    initial_bbox_padding_px: float = 12.0
    subprocess_timeout_sec: float = 1800.0
    keep_invisible_points_as_gaps: bool = True
    max_step_distance_px: float = 0.0
    max_reacquisition_gap: int = 0
    terminate_after_invisible_frames: int = 0
    max_mask_area_ratio_change: float = 0.0
    max_bbox_aspect_ratio_change: float = 0.0
    max_mask_components: int = 0
    max_bbox_center_jump_px: float = 0.0
    center_mode: MaskCenterMode = MaskCenterMode.AUTO
    manual_center: tuple[float, float] = (0.0, 0.0)
    direction_mode: MaskDirectionMode = MaskDirectionMode.NONE
    lateral_angle_tolerance_deg: float = 20.0
    max_radial_jump_px: float = 0.0


@dataclass(frozen=True)
class CutieMaskConfig(MaskTrackerConfig):
    models_dir: Path = Path("tracking_models/cutie")
    inference_resolution: InferenceResolution = None
    initial_mask_mode: InitialMaskMode = InitialMaskMode.THRESHOLD
    initial_mask_threshold_rel: float = 0.7
    initial_mask_disk_radius_ratio: float = 0.35
    mem_every: int | None = None


@dataclass(frozen=True)
class Xmem2MaskConfig(MaskTrackerConfig):
    models_dir: Path = Path("tracking_models/xmem2")
    inference_resolution: InferenceResolution = None
    initial_mask_mode: InitialMaskMode = InitialMaskMode.THRESHOLD
    initial_mask_threshold_rel: float = 0.7
    initial_mask_disk_radius_ratio: float = 0.35
    mem_every: int | None = None


@dataclass(frozen=True)
class Sam2MaskConfig(MaskTrackerConfig):
    models_dir: Path = Path("tracking_models/sam2")
    config_path: Path | None = None
    prompt_mode: Sam2PromptMode = Sam2PromptMode.POINT_AND_BBOX
    inference_resolution: InferenceResolution = None
    mask_threshold: float = 0.5
    min_mask_area: float = 1.0
    max_mask_area: float | None = None
    use_vos_optimized: bool = False
    apply_postprocessing: bool = True
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = False
    async_loading_frames: bool = False


@dataclass(frozen=True)
class Dam4SamMaskConfig(MaskTrackerConfig):
    models_dir: Path = Path("tracking_models/dam4sam")
    config_path: Path | None = None
    tracker_name: str = "sam21pp-L"
    inference_resolution: InferenceResolution = None


@dataclass(frozen=True)
class SamuraiMaskConfig(MaskTrackerConfig):
    models_dir: Path = Path("tracking_models/samurai")
    config_path: Path | None = None
    model_variant: str = "sam2.1_hiera_base_plus"
    inference_resolution: InferenceResolution = None


@dataclass(frozen=True)
class DevaMaskConfig(MaskTrackerConfig):
    models_dir: Path = Path("tracking_models/deva")
    inference_size: int | None = 480
    chunk_size: int = 4
    amp: bool = True
    initial_bbox_padding_px: float = 0.0
    initial_point_radius_px: float = 4.0
