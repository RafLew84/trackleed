"""XMem2 / XMem++ environment diagnostics and smoke-test helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Optional
import importlib
import json
import re
import shutil
import subprocess
import sys

import numpy as np


DEFAULT_XMEM2_MODELS_DIR = Path("tracking_models") / "xmem2"
DEFAULT_XMEM2_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "xmem2",
    Path(__file__).resolve().parents[2] / "xmem2",
)
DEFAULT_XMEM2_PRIMARY_CHECKPOINT = "XMem.pth"
DEFAULT_XMEM2_SECONDARY_CHECKPOINT = "XMem-s012.pth"
DEFAULT_XMEM2_EXAMPLE_FRAMES = Path("example_videos") / "caps" / "JPEGImages"
DEFAULT_XMEM2_EXAMPLE_MASKS = Path("example_videos") / "caps" / "Annotations"
DEFAULT_XMEM2_EXAMPLE_SEQUENCE_NAME = "caps"
DEFAULT_XMEM2_EXAMPLE_FRAME_LIMIT = 16
DEFAULT_XMEM2_SUBPROCESS_SCRIPT = Path("scripts") / "run_xmem2_subprocess.py"
CHECKPOINT_PRIORITY = (
    DEFAULT_XMEM2_PRIMARY_CHECKPOINT,
    DEFAULT_XMEM2_SECONDARY_CHECKPOINT,
)
XMEM2_IMPORT_ROOTS = (
    "inference",
    "model",
    "util",
    "dataset",
    "XMem_utilities",
)


@dataclass
class XMem2EnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    available_checkpoints: list[Path] = field(default_factory=list)
    selected_checkpoint: Optional[Path] = None
    checkpoint_exists: bool = False
    example_frames_dir: Optional[Path] = None
    example_masks_dir: Optional[Path] = None
    example_inputs_ready: bool = False
    torch_importable: bool = False
    torch_version: Optional[str] = None
    cuda_available: bool = False
    torchvision_importable: bool = False
    cv2_importable: bool = False
    pillow_importable: bool = False
    pandas_importable: bool = False
    tqdm_importable: bool = False
    progressbar_importable: bool = False
    xmem_inference_importable: bool = False
    xmem_network_importable: bool = False
    xmem_config_importable: bool = False
    xmem_run_on_video_path: Optional[str] = None
    xmem_network_path: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return (
            self.repo_path is not None
            and self.checkpoint_exists
            and self.torch_importable
            and self.cuda_available
            and self.torchvision_importable
            and self.cv2_importable
            and self.pillow_importable
            and self.pandas_importable
            and self.tqdm_importable
            and self.progressbar_importable
            and self.xmem_inference_importable
            and self.xmem_network_importable
            and self.xmem_config_importable
        )


@dataclass
class XMem2SmokeTestResult:
    checkpoint_name: str
    example_sequence_name: str
    frame_count: int
    annotation_frame_count: int
    output_mask_count: int
    output_overlay_count: int
    first_output_mask_shape: tuple[int, ...]
    first_output_mask_area: int
    stats_shape: tuple[int, ...]
    frame_limit: int


@dataclass
class XMem2InitialMaskModeResult:
    mode: str
    area: int
    bbox_xyxy: tuple[int, int, int, int] | None
    centroid_xy: tuple[float, float] | None
    reference_iou: float
    jitter_mean_iou: float
    jitter_min_iou: float
    coverage_in_roi: float


@dataclass
class XMem2InitialMaskSampleResult:
    sample_name: str
    image_shape: tuple[int, int]
    roi_rect_xywh: tuple[int, int, int, int]
    reference_area: int
    frame_name: str
    mode_results: list[XMem2InitialMaskModeResult]


@dataclass
class XMem2InitialMaskProbeResult:
    samples: list[XMem2InitialMaskSampleResult]
    evaluated_modes: list[str]
    recommended_mode: str
    threshold_rel: float
    disk_radius_ratio: float
    bbox_padding_ratio: float
    jitter_px: int
    jitter_scale: float
    notes: list[str] = field(default_factory=list)


@dataclass
class XMem2RuntimeSettings:
    checkpoint_path: str | Path
    repo_path: str | Path | None = None
    python_executable: str | Path | None = None
    device: str = "auto"
    backend_mode: str = "subprocess"
    representative_point_mode: str = "mask_centroid"
    inference_resolution: int | tuple[int, int] | None = None
    initial_bbox_padding_px: float = 12.0
    initial_mask_mode: str = "threshold"
    initial_mask_threshold_rel: float = 0.7
    initial_mask_disk_radius_ratio: float = 0.35
    mem_every: int | None = None
    subprocess_timeout_sec: float = 1800.0


@dataclass
class XMem2OfflineResult:
    tracks_xy: np.ndarray
    visible_mask: np.ndarray
    mask_areas: np.ndarray
    mask_bboxes_xyxy: np.ndarray
    mask_scores: np.ndarray
    inference_resolution: tuple[int, int]
    original_resolution: tuple[int, int]
    checkpoint_name: str
    representative_point_mode: str
    initial_mask_mode: str
    backend_mode: str


def discover_xmem2_checkpoints(
    models_dir: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_XMEM2_MODELS_DIR
    discovered: dict[str, Path] = {}

    for candidate in _discover_checkpoints_in_directory(directory):
        discovered[str(candidate.resolve())] = candidate.resolve()

    repo_root = _resolve_xmem2_repo_path(repo_path)
    if repo_root is not None:
        for candidate in _discover_checkpoints_in_directory(repo_root / "saves"):
            discovered[str(candidate.resolve())] = candidate.resolve()

    return sorted(discovered.values(), key=lambda path: path.name.lower())


def select_default_xmem2_checkpoint(checkpoints: list[Path]) -> Optional[Path]:
    if not checkpoints:
        return None
    by_name = {path.name: path for path in checkpoints}
    for preferred_name in CHECKPOINT_PRIORITY:
        if preferred_name in by_name:
            return by_name[preferred_name]
    return checkpoints[0]


def probe_xmem2_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> XMem2EnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_XMEM2_MODELS_DIR
    directory = directory.resolve()
    resolved_repo_path = _resolve_xmem2_repo_path(repo_path)
    checkpoints = discover_xmem2_checkpoints(directory, resolved_repo_path)
    selected_checkpoint = _resolve_checkpoint_path(
        checkpoint_path,
        checkpoints,
        directory,
        resolved_repo_path,
    )
    example_frames_dir, example_masks_dir = _resolve_xmem2_example_inputs(resolved_repo_path)

    status = XMem2EnvironmentStatus(
        models_dir=directory,
        repo_path=resolved_repo_path,
        available_checkpoints=checkpoints,
        selected_checkpoint=selected_checkpoint,
        checkpoint_exists=bool(selected_checkpoint and selected_checkpoint.exists()),
        example_frames_dir=example_frames_dir,
        example_masks_dir=example_masks_dir,
        example_inputs_ready=bool(
            example_frames_dir
            and example_frames_dir.exists()
            and example_masks_dir
            and example_masks_dir.exists()
        ),
    )

    if not directory.exists():
        status.warnings.append(
            f"Checkpoint directory does not exist yet: {directory}. "
            "This is fine if you still keep XMem weights inside the local repo/saves directory."
        )

    if selected_checkpoint is None:
        status.warnings.append(
            "No XMem checkpoint detected in tracking_models/xmem2 or in repo/saves."
        )
        status.hints.append(
            f"Add `{DEFAULT_XMEM2_PRIMARY_CHECKPOINT}` before running XMem2 diagnostics."
        )
    elif checkpoint_path and not status.checkpoint_exists:
        status.errors.append(f"Selected checkpoint not found: {selected_checkpoint}")

    if resolved_repo_path is None:
        status.errors.append(
            "Could not resolve the local XMem2 repo. Clone `XMem2` as `xmem2` next to the "
            "project or pass --repo-path."
        )
        status.hints.append(
            "Expected repo root containing `README.md`, `process_video.py`, "
            "`inference/run_on_video.py`, and `model/network.py`."
        )

    if not status.example_inputs_ready:
        status.warnings.append(
            "Default XMem2 example inputs are missing under `example_videos/caps`. "
            "The optional smoke test will need a valid repo checkout."
        )

    if status.selected_checkpoint is not None and status.selected_checkpoint.name != DEFAULT_XMEM2_PRIMARY_CHECKPOINT:
        status.warnings.append(
            f"Using fallback checkpoint `{status.selected_checkpoint.name}`. "
            f"Upstream examples default to `{DEFAULT_XMEM2_PRIMARY_CHECKPOINT}`."
        )

    try:
        torch = importlib.import_module("torch")
        status.torch_importable = True
        status.torch_version = getattr(torch, "__version__", None)
        try:
            status.cuda_available = bool(torch.cuda.is_available())
        except Exception:
            status.cuda_available = False
    except Exception as exc:
        status.errors.append(f"PyTorch import failed: {exc}")
        status.hints.append(
            "Install the XMem2 runtime torch stack in the dedicated `xmem2` env."
        )

    try:
        importlib.import_module("torchvision")
        status.torchvision_importable = True
    except Exception as exc:
        status.errors.append(f"torchvision import failed: {exc}")

    try:
        importlib.import_module("cv2")
        status.cv2_importable = True
    except Exception as exc:
        status.errors.append(f"OpenCV import failed: {exc}")

    try:
        importlib.import_module("PIL.Image")
        status.pillow_importable = True
    except Exception as exc:
        status.errors.append(f"Pillow import failed: {exc}")

    try:
        importlib.import_module("pandas")
        status.pandas_importable = True
    except Exception as exc:
        status.errors.append(f"pandas import failed: {exc}")

    try:
        importlib.import_module("tqdm")
        status.tqdm_importable = True
    except Exception as exc:
        status.errors.append(f"tqdm import failed: {exc}")

    try:
        importlib.import_module("progressbar")
        status.progressbar_importable = True
    except Exception as exc:
        status.errors.append(f"progressbar2 import failed: {exc}")

    with _xmem2_import_context(resolved_repo_path):
        try:
            run_on_video_module = importlib.import_module("inference.run_on_video")
            status.xmem_inference_importable = True
            module_file = getattr(run_on_video_module, "__file__", None)
            if module_file is not None:
                status.xmem_run_on_video_path = str(Path(module_file).resolve())
        except Exception as exc:
            status.errors.append(f"XMem inference import failed: {exc}")

        try:
            network_module = importlib.import_module("model.network")
            status.xmem_network_importable = True
            module_file = getattr(network_module, "__file__", None)
            if module_file is not None:
                status.xmem_network_path = str(Path(module_file).resolve())
        except Exception as exc:
            status.errors.append(f"XMem network import failed: {exc}")

        try:
            importlib.import_module("util.configuration")
            status.xmem_config_importable = True
        except Exception as exc:
            status.errors.append(f"XMem configuration import failed: {exc}")

    if status.torch_importable and not status.cuda_available:
        status.warnings.append(
            "PyTorch CUDA is not available. XMem2 may still run on CPU, but even short sequences will be very slow."
        )

    if sys.platform.startswith("win"):
        status.warnings.append(
            "Upstream XMem2 was developed primarily around Linux-style workflows. Native Windows can still work if the inference smoke test passes."
        )

    return status


def format_xmem2_environment_report(status: XMem2EnvironmentStatus) -> str:
    selected_checkpoint = (
        str(status.selected_checkpoint) if status.selected_checkpoint is not None else "None"
    )
    available = ", ".join(path.name for path in status.available_checkpoints) or "none"
    lines = [
        "XMem2 / XMem++ environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Selected checkpoint: {selected_checkpoint}",
        f"- Available checkpoints: {available}",
        f"- Example frames: {status.example_frames_dir or 'None'}",
        f"- Example masks: {status.example_masks_dir or 'None'}",
        f"- PyTorch import: {'ok' if status.torch_importable else 'failed'}",
        f"- PyTorch version: {status.torch_version or 'unknown'}",
        f"- CUDA available: {'yes' if status.cuda_available else 'no'}",
        f"- torchvision import: {'ok' if status.torchvision_importable else 'failed'}",
        f"- OpenCV import: {'ok' if status.cv2_importable else 'failed'}",
        f"- Pillow import: {'ok' if status.pillow_importable else 'failed'}",
        f"- pandas import: {'ok' if status.pandas_importable else 'failed'}",
        f"- tqdm import: {'ok' if status.tqdm_importable else 'failed'}",
        f"- progressbar2 import: {'ok' if status.progressbar_importable else 'failed'}",
        f"- XMem inference import: {'ok' if status.xmem_inference_importable else 'failed'}",
        f"- XMem network import: {'ok' if status.xmem_network_importable else 'failed'}",
        f"- XMem configuration import: {'ok' if status.xmem_config_importable else 'failed'}",
    ]
    if status.xmem_run_on_video_path:
        lines.append(f"- XMem run_on_video module: {status.xmem_run_on_video_path}")
    if status.xmem_network_path:
        lines.append(f"- XMem network module: {status.xmem_network_path}")
    if status.errors:
        lines.append("- Errors:")
        lines.extend(f"  - {item}" for item in status.errors)
    if status.warnings:
        lines.append("- Warnings:")
        lines.extend(f"  - {item}" for item in status.warnings)
    if status.hints:
        lines.append("- Hints:")
        lines.extend(f"  - {item}" for item in status.hints)
    return "\n".join(lines)


def run_xmem2_smoke_test(
    *,
    checkpoint_path: str | Path,
    repo_path: str | Path | None = None,
    frame_limit: int = DEFAULT_XMEM2_EXAMPLE_FRAME_LIMIT,
) -> XMem2SmokeTestResult:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"XMem checkpoint not found: {checkpoint}")

    resolved_repo_path = _resolve_xmem2_repo_path(repo_path)
    if resolved_repo_path is None:
        raise RuntimeError("Could not resolve the XMem2 repo for the smoke test.")

    example_frames_dir, example_masks_dir = _resolve_xmem2_example_inputs(resolved_repo_path)
    if example_frames_dir is None or example_masks_dir is None:
        raise RuntimeError("Could not resolve the default XMem2 example inputs.")

    frame_paths = sorted(example_frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No example XMem2 frames found in: {example_frames_dir}")

    frame_limit = max(1, int(frame_limit))
    selected_frame_paths = frame_paths[:frame_limit]
    selected_frame_names = {path.name for path in selected_frame_paths}

    mask_paths = sorted(example_masks_dir.glob("*.png"))
    selected_mask_paths = [
        mask_path
        for mask_path in mask_paths
        if _mask_matches_selected_frames(mask_path, selected_frame_names)
    ]
    if not selected_mask_paths:
        raise RuntimeError(
            "No example masks matched the selected smoke-test frame range."
        )

    with TemporaryDirectory(prefix="trackleed_xmem2_smoke_") as temp_dir:
        temp_root = Path(temp_dir)
        temp_frames_dir = temp_root / "JPEGImages"
        temp_masks_dir = temp_root / "Annotations"
        temp_output_dir = temp_root / "output"
        temp_frames_dir.mkdir(parents=True, exist_ok=True)
        temp_masks_dir.mkdir(parents=True, exist_ok=True)

        for frame_path in selected_frame_paths:
            shutil.copy2(frame_path, temp_frames_dir / frame_path.name)
        for mask_path in selected_mask_paths:
            shutil.copy2(mask_path, temp_masks_dir / mask_path.name)

        frame_indices = [
            _extract_first_int(mask_path.stem)
            for mask_path in selected_mask_paths
        ]
        frame_indices = [idx for idx in frame_indices if idx is not None]
        if not frame_indices:
            raise RuntimeError("Could not parse any annotation frame indices for the XMem2 smoke test.")

        with _xmem2_import_context(resolved_repo_path):
            run_on_video = importlib.import_module("inference.run_on_video").run_on_video
            stats = run_on_video(
                str(temp_frames_dir),
                str(temp_masks_dir),
                str(temp_output_dir),
                frames_with_masks=sorted(frame_indices),
                print_progress=False,
                save_overlay=False,
                overwrite_config={"model": str(checkpoint)},
            )

        output_masks_dir = temp_output_dir / "masks"
        output_overlays_dir = temp_output_dir / "overlay"
        output_mask_paths = sorted(output_masks_dir.glob("*.png"))
        output_overlay_paths = sorted(output_overlays_dir.glob("*.png"))
        if not output_mask_paths:
            raise RuntimeError("XMem2 smoke test finished but did not save any predicted masks.")

        pil_image_module = importlib.import_module("PIL.Image")
        first_output_mask = np.array(pil_image_module.open(str(output_mask_paths[0])))

    return XMem2SmokeTestResult(
        checkpoint_name=checkpoint.name,
        example_sequence_name=DEFAULT_XMEM2_EXAMPLE_SEQUENCE_NAME,
        frame_count=len(selected_frame_paths),
        annotation_frame_count=len(selected_mask_paths),
        output_mask_count=len(output_mask_paths),
        output_overlay_count=len(output_overlay_paths),
        first_output_mask_shape=tuple(int(dim) for dim in first_output_mask.shape),
        first_output_mask_area=int(np.count_nonzero(first_output_mask)),
        stats_shape=tuple(int(dim) for dim in getattr(stats, "shape", (0, 0))),
        frame_limit=frame_limit,
    )


def build_xmem2_initial_mask(
    frame: np.ndarray,
    roi_rect_xywh: tuple[int, int, int, int],
    *,
    mode: str = "threshold",
    threshold_rel: float = 0.7,
    disk_radius_ratio: float = 0.35,
) -> np.ndarray:
    gray = _frame_to_normalized_gray(frame)
    image_h, image_w = gray.shape[:2]
    x, y, width, height = _clamp_roi_rect_xywh(roi_rect_xywh, (image_h, image_w))
    if width <= 0 or height <= 0:
        return np.zeros((image_h, image_w), dtype=bool)

    normalized_mode = _normalize_xmem2_mask_builder_mode(mode)
    mask = np.zeros((image_h, image_w), dtype=bool)

    if normalized_mode == "full_bbox":
        mask[y : y + height, x : x + width] = True
        return mask

    center_x = x + (width - 1) / 2.0
    center_y = y + (height - 1) / 2.0

    if normalized_mode == "disk":
        radius = max(1.0, min(width, height) * float(max(0.05, disk_radius_ratio)))
        yy, xx = np.ogrid[:image_h, :image_w]
        mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2) <= radius**2
        roi_mask = np.zeros_like(mask)
        roi_mask[y : y + height, x : x + width] = True
        return mask & roi_mask

    roi = gray[y : y + height, x : x + width]
    if roi.size == 0:
        return mask

    roi_min = float(np.min(roi))
    roi_max = float(np.max(roi))
    if roi_max <= roi_min:
        local_mask = np.zeros_like(roi, dtype=bool)
        max_index = int(np.argmax(roi))
        max_row, max_col = np.unravel_index(max_index, roi.shape)
        local_mask[max_row, max_col] = True
    else:
        threshold = roi_min + (roi_max - roi_min) * float(np.clip(threshold_rel, 0.0, 1.0))
        local_mask = roi >= threshold
        if not np.any(local_mask):
            local_mask = roi == roi_max

    selected_local_mask = _select_component_near_center(
        local_mask,
        center_xy=((width - 1) / 2.0, (height - 1) / 2.0),
    )
    if selected_local_mask is None or not np.any(selected_local_mask):
        selected_local_mask = local_mask

    mask[y : y + height, x : x + width] = selected_local_mask
    return mask


def probe_xmem2_initial_mask_builder(
    *,
    repo_path: str | Path | None = None,
    modes: list[str] | tuple[str, ...] | None = None,
    threshold_rel: float = 0.7,
    disk_radius_ratio: float = 0.35,
    bbox_padding_ratio: float = 0.15,
    jitter_px: int = 4,
    jitter_scale: float = 0.12,
) -> XMem2InitialMaskProbeResult:
    resolved_repo_path = _resolve_xmem2_repo_path(repo_path)
    if resolved_repo_path is None:
        raise RuntimeError("Could not resolve the XMem2 repo for the mask builder probe.")

    example_frames_dir, example_masks_dir = _resolve_xmem2_example_inputs(resolved_repo_path)
    if example_frames_dir is None or example_masks_dir is None:
        raise RuntimeError("Could not resolve the default XMem2 example inputs.")

    normalized_modes = [
        _normalize_xmem2_mask_builder_mode(mode)
        for mode in (modes or ("threshold", "full_bbox", "disk"))
    ]
    deduped_modes: list[str] = []
    for mode in normalized_modes:
        if mode not in deduped_modes:
            deduped_modes.append(mode)

    Image = importlib.import_module("PIL.Image")
    mask_paths = sorted(example_masks_dir.glob("*.png"))
    if not mask_paths:
        raise RuntimeError(f"No example masks found in: {example_masks_dir}")

    sample_results: list[XMem2InitialMaskSampleResult] = []
    aggregate_scores: dict[str, list[float]] = {mode: [] for mode in deduped_modes}

    for mask_path in mask_paths:
        frame_name = f"{mask_path.stem}.jpg"
        image_path = example_frames_dir / frame_name
        if not image_path.exists():
            continue

        image_np = np.array(Image.open(str(image_path)).convert("RGB"))
        reference_mask = np.array(Image.open(str(mask_path))) > 0
        if reference_mask.ndim != 2 or not np.any(reference_mask):
            continue

        image_h, image_w = reference_mask.shape
        roi_rect_xywh = _expanded_roi_rect_from_mask(
            reference_mask,
            bbox_padding_ratio=float(max(0.0, bbox_padding_ratio)),
            image_shape=(image_h, image_w),
        )
        jittered_rois = _generate_jittered_roi_rects(
            roi_rect_xywh,
            (image_h, image_w),
            jitter_px=int(max(0, jitter_px)),
            jitter_scale=float(max(0.0, jitter_scale)),
        )

        mode_results: list[XMem2InitialMaskModeResult] = []
        for mode in deduped_modes:
            base_mask = build_xmem2_initial_mask(
                image_np,
                roi_rect_xywh,
                mode=mode,
                threshold_rel=threshold_rel,
                disk_radius_ratio=disk_radius_ratio,
            )
            jitter_ious: list[float] = []
            for jittered_roi in jittered_rois:
                jitter_mask = build_xmem2_initial_mask(
                    image_np,
                    jittered_roi,
                    mode=mode,
                    threshold_rel=threshold_rel,
                    disk_radius_ratio=disk_radius_ratio,
                )
                jitter_ious.append(_mask_iou(base_mask, jitter_mask))

            reference_iou = _mask_iou(base_mask, reference_mask)
            jitter_mean_iou = float(np.mean(jitter_ious)) if jitter_ious else 1.0
            jitter_min_iou = float(np.min(jitter_ious)) if jitter_ious else 1.0
            area = int(np.count_nonzero(base_mask))
            roi_area = max(1, int(roi_rect_xywh[2] * roi_rect_xywh[3]))
            combined_score = (reference_iou * 0.7) + (jitter_mean_iou * 0.3)
            aggregate_scores[mode].append(combined_score)
            mode_results.append(
                XMem2InitialMaskModeResult(
                    mode=mode,
                    area=area,
                    bbox_xyxy=_mask_bbox_xyxy(base_mask),
                    centroid_xy=_mask_centroid_xy(base_mask),
                    reference_iou=reference_iou,
                    jitter_mean_iou=jitter_mean_iou,
                    jitter_min_iou=jitter_min_iou,
                    coverage_in_roi=float(area / roi_area),
                )
            )

        sample_results.append(
            XMem2InitialMaskSampleResult(
                sample_name=mask_path.stem,
                image_shape=(image_h, image_w),
                roi_rect_xywh=roi_rect_xywh,
                reference_area=int(np.count_nonzero(reference_mask)),
                frame_name=frame_name,
                mode_results=mode_results,
            )
        )

    if not sample_results:
        raise RuntimeError("Could not build any valid XMem2 mask-builder probe samples.")

    recommended_mode = "threshold"
    best_score = -1.0
    for mode, values in aggregate_scores.items():
        if not values:
            continue
        score = float(np.mean(values))
        if score > best_score:
            best_score = score
            recommended_mode = mode

    notes = [
        "Score laczy zgodnosc z referencyjna maska oraz odpornosc na niewielkie przesuniecia bboxa.",
        "Probe korzysta z upstreamowego przykladu `caps`, wiec wynik trzeba traktowac jako wstepny przed testem na danych LEED.",
        "Dla MVP warto preferowac builder stabilny na jitter bboxa, nawet jesli nie daje idealnie najwiekszej IoU na pojedynczej klatce.",
    ]
    return XMem2InitialMaskProbeResult(
        samples=sample_results,
        evaluated_modes=deduped_modes,
        recommended_mode=recommended_mode,
        threshold_rel=float(threshold_rel),
        disk_radius_ratio=float(disk_radius_ratio),
        bbox_padding_ratio=float(bbox_padding_ratio),
        jitter_px=int(jitter_px),
        jitter_scale=float(jitter_scale),
        notes=notes,
    )


def format_xmem2_initial_mask_probe_report(result: XMem2InitialMaskProbeResult) -> str:
    lines = [
        "XMem2 initial mask builder probe",
        f"- Recommended mode: {result.recommended_mode}",
        f"- Evaluated modes: {', '.join(result.evaluated_modes)}",
        f"- Threshold rel: {result.threshold_rel:.3f}",
        f"- Disk radius ratio: {result.disk_radius_ratio:.3f}",
        f"- BBox padding ratio: {result.bbox_padding_ratio:.3f}",
        f"- Jitter px: {result.jitter_px}",
        f"- Jitter scale: {result.jitter_scale:.3f}",
    ]
    for sample in result.samples:
        x, y, w, h = sample.roi_rect_xywh
        lines.append(f"- Sample: {sample.sample_name}")
        lines.append(f"  - Frame: {sample.frame_name}")
        lines.append(f"  - Image shape: {sample.image_shape}")
        lines.append(f"  - ROI rect xywh: ({x}, {y}, {w}, {h})")
        lines.append(f"  - Reference area: {sample.reference_area}")
        for mode_result in sample.mode_results:
            bbox_text = (
                str(mode_result.bbox_xyxy)
                if mode_result.bbox_xyxy is not None
                else "None"
            )
            centroid_text = (
                f"({mode_result.centroid_xy[0]:.2f}, {mode_result.centroid_xy[1]:.2f})"
                if mode_result.centroid_xy is not None
                else "None"
            )
            lines.append(f"  - Mode `{mode_result.mode}`:")
            lines.append(f"    - area: {mode_result.area}")
            lines.append(f"    - bbox xyxy: {bbox_text}")
            lines.append(f"    - centroid xy: {centroid_text}")
            lines.append(f"    - reference IoU: {mode_result.reference_iou:.4f}")
            lines.append(f"    - jitter mean IoU: {mode_result.jitter_mean_iou:.4f}")
            lines.append(f"    - jitter min IoU: {mode_result.jitter_min_iou:.4f}")
            lines.append(f"    - coverage in ROI: {mode_result.coverage_in_roi:.4f}")
    if result.notes:
        lines.append("- Notes:")
        lines.extend(f"  - {note}" for note in result.notes)
    return "\n".join(lines)


def run_xmem2_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: XMem2RuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
    query_object_ids: np.ndarray | None = None,
) -> XMem2OfflineResult:
    frames_np = np.asarray(frames)
    if frames_np.ndim not in (3, 4):
        raise ValueError(f"Expected frames with ndim 3 or 4, got shape {frames_np.shape}.")
    if frames_np.size == 0:
        raise ValueError("No frames provided for XMem2 inference.")

    query_np = np.asarray(query_points_tyx, dtype=np.float32)
    if query_np.ndim != 2 or query_np.shape[1] != 3:
        raise ValueError("query_points_tyx must have shape [num_queries, 3] in [t, y, x] format.")
    if query_np.shape[0] == 0:
        raise ValueError("No query points provided for XMem2 inference.")

    boxes_np = None
    if query_boxes_xyxy is not None:
        boxes_np = np.asarray(query_boxes_xyxy, dtype=np.float32)
        if boxes_np.shape != (query_np.shape[0], 4):
            raise ValueError(
                f"query_boxes_xyxy must have shape {(query_np.shape[0], 4)}, got {boxes_np.shape}."
            )
    object_ids_np = _normalize_xmem2_query_object_ids(query_object_ids, query_np.shape[0])

    backend_mode = str(settings.backend_mode or "subprocess").strip().lower()
    if backend_mode == "auto":
        backend_mode = "subprocess"
    if backend_mode == "direct":
        return _run_xmem2_direct_offline(
            frames_np,
            query_np,
            settings,
            query_boxes_xyxy=boxes_np,
            query_object_ids=object_ids_np,
        )
    if backend_mode != "subprocess":
        raise ValueError(
            f"Unsupported XMem2 backend mode: {settings.backend_mode}. "
            "Expected `subprocess` or `direct`."
        )
    return _run_xmem2_subprocess_offline(
        frames_np,
        query_np,
        settings,
        query_boxes_xyxy=boxes_np,
        query_object_ids=object_ids_np,
    )


def _run_xmem2_direct_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: XMem2RuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
    query_object_ids: np.ndarray | None = None,
) -> XMem2OfflineResult:
    resolved_repo_path = _resolve_xmem2_repo_path(settings.repo_path)
    if resolved_repo_path is None:
        raise FileNotFoundError(
            "Could not resolve the XMem2 repo. Pass repo_path pointing at the local `xmem2` clone."
        )

    checkpoint_file = _resolve_existing_checkpoint_file(Path(settings.checkpoint_path))
    representative_point_mode = _normalize_representative_point_mode(
        settings.representative_point_mode
    )
    initial_mask_mode = _normalize_xmem2_mask_builder_mode(settings.initial_mask_mode)
    device_name = str(settings.device or "auto").strip().lower()
    if device_name not in {"", "auto", "cuda", "gpu", "cuda:0"}:
        raise ValueError(
            f"Unsupported XMem2 device setting: {settings.device}. "
            "Etap 3 currently supports only auto/cuda."
        )

    prepared_frames, original_resolution = _prepare_frames_rgb(frames)
    inference_resolution = _resolve_xmem2_inference_resolution(
        settings.inference_resolution,
        original_resolution,
    )
    resized_frames = _resize_video_numpy(prepared_frames, inference_resolution)
    gray_frames = _video_to_gray_float(resized_frames)
    scaled_queries = _scale_query_points_tyx(
        query_points_tyx,
        source_resolution=original_resolution,
        target_resolution=inference_resolution,
    )
    object_ids = _normalize_xmem2_query_object_ids(query_object_ids, scaled_queries.shape[0])
    unique_object_ids = list(dict.fromkeys(int(v) for v in object_ids.tolist()))
    object_id_to_class_id = {
        int(object_id): int(object_index + 1)
        for object_index, object_id in enumerate(unique_object_ids)
    }
    class_ids = np.asarray(
        [object_id_to_class_id[int(object_id)] for object_id in object_ids.tolist()],
        dtype=np.int32,
    )
    scaled_boxes = _build_xmem2_prompt_boxes_xyxy(
        scaled_queries,
        resolution=inference_resolution,
        initial_bbox_padding_px=float(settings.initial_bbox_padding_px),
        provided_boxes_xyxy=(
            _scale_boxes_xyxy(
                query_boxes_xyxy,
                source_resolution=original_resolution,
                target_resolution=inference_resolution,
            )
            if query_boxes_xyxy is not None
            else None
        ),
    )
    seed_frames = _extract_query_seed_frames(scaled_queries, total_frames=resized_frames.shape[0])

    num_objects = int(len(unique_object_ids))
    num_frames = int(resized_frames.shape[0])
    tracks_xy_resized = np.full((num_objects, num_frames, 2), np.nan, dtype=np.float32)
    visible_mask = np.zeros((num_objects, num_frames), dtype=bool)
    mask_areas = np.zeros((num_objects, num_frames), dtype=np.float32)
    mask_bboxes_xyxy_resized = np.full((num_objects, num_frames, 4), np.nan, dtype=np.float32)
    mask_scores = np.zeros((num_objects, num_frames), dtype=np.float32)

    with TemporaryDirectory(prefix="trackleed_xmem2_runtime_") as temp_dir:
        temp_root = Path(temp_dir)
        frames_dir = temp_root / "JPEGImages"
        masks_dir = temp_root / "Annotations"
        output_dir = temp_root / "output"
        frames_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        _write_xmem2_video_frames(frames_dir, resized_frames)
        used_seed_frames = _write_xmem2_seed_masks(
            masks_dir,
            frames=resized_frames,
            boxes_xyxy=scaled_boxes,
            seed_frames=seed_frames,
            class_ids=class_ids,
            initial_mask_mode=initial_mask_mode,
            threshold_rel=float(settings.initial_mask_threshold_rel),
            disk_radius_ratio=float(settings.initial_mask_disk_radius_ratio),
        )
        if not used_seed_frames:
            raise RuntimeError("Could not generate any valid XMem2 seed masks from the provided queries.")

        with _xmem2_import_context(resolved_repo_path):
            run_on_video = importlib.import_module("inference.run_on_video").run_on_video
            run_on_video(
                str(frames_dir),
                str(masks_dir),
                str(output_dir),
                frames_with_masks=sorted(used_seed_frames),
                print_progress=False,
                save_overlay=False,
                overwrite_config={
                    "model": str(checkpoint_file),
                    "size": -1,
                    "save_masks": True,
                    "num_objects": int(num_objects),
                    **(
                        {"mem_every": int(settings.mem_every)}
                        if settings.mem_every is not None
                        else {}
                    ),
                },
            )

        output_masks_dir = output_dir / "masks"
        output_mask_paths = sorted(output_masks_dir.glob("*.png"))
        if len(output_mask_paths) != num_frames:
            raise RuntimeError(
                f"Expected {num_frames} XMem2 output masks, got {len(output_mask_paths)}."
            )

        Image = importlib.import_module("PIL.Image")
        area_scale = (
            float(original_resolution[0]) / float(inference_resolution[0])
        ) * (
            float(original_resolution[1]) / float(inference_resolution[1])
        )
        for frame_index, mask_path in enumerate(output_mask_paths):
            class_mask = np.array(Image.open(str(mask_path)), dtype=np.int32)
            if class_mask.ndim == 3:
                class_mask = class_mask[..., 0]
            _record_xmem2_frame_outputs(
                class_mask=class_mask,
                frame_gray=gray_frames[frame_index],
                frame_index=frame_index,
                tracks_xy=tracks_xy_resized,
                visible_mask=visible_mask,
                mask_areas=mask_areas,
                mask_bboxes_xyxy=mask_bboxes_xyxy_resized,
                mask_scores=mask_scores,
                representative_point_mode=representative_point_mode,
                num_objects=num_objects,
                area_scale=area_scale,
            )

    object_seed_frames = np.full(num_objects, num_frames, dtype=np.int32)
    for ref_index, seed_frame in enumerate(seed_frames.tolist()):
        object_index = int(class_ids[ref_index] - 1)
        object_seed_frames[object_index] = min(object_seed_frames[object_index], int(seed_frame))

    for object_index, seed_frame in enumerate(object_seed_frames.tolist()):
        if seed_frame <= 0 or seed_frame >= num_frames:
            continue
        visible_mask[object_index, :seed_frame] = False
        tracks_xy_resized[object_index, :seed_frame, :] = np.nan
        mask_areas[object_index, :seed_frame] = 0.0
        mask_bboxes_xyxy_resized[object_index, :seed_frame, :] = np.nan
        mask_scores[object_index, :seed_frame] = 0.0

    tracks_xy = _scale_tracks_xy(
        tracks_xy_resized,
        source_resolution=inference_resolution,
        target_resolution=original_resolution,
    )
    mask_bboxes_xyxy = _scale_boxes_xyxy(
        mask_bboxes_xyxy_resized,
        source_resolution=inference_resolution,
        target_resolution=original_resolution,
    )
    return XMem2OfflineResult(
        tracks_xy=tracks_xy,
        visible_mask=visible_mask,
        mask_areas=mask_areas,
        mask_bboxes_xyxy=(
            mask_bboxes_xyxy if mask_bboxes_xyxy is not None else mask_bboxes_xyxy_resized
        ),
        mask_scores=mask_scores,
        inference_resolution=inference_resolution,
        original_resolution=original_resolution,
        checkpoint_name=checkpoint_file.name,
        representative_point_mode=representative_point_mode,
        initial_mask_mode=initial_mask_mode,
        backend_mode="direct",
    )


def _run_xmem2_subprocess_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: XMem2RuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
    query_object_ids: np.ndarray | None = None,
) -> XMem2OfflineResult:
    python_executable = _resolve_xmem2_python_executable(settings.python_executable)
    script_path = (Path(__file__).resolve().parents[2] / DEFAULT_XMEM2_SUBPROCESS_SCRIPT).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"XMem2 subprocess script not found: {script_path}")

    with TemporaryDirectory(prefix="trackleed_xmem2_") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "xmem2_input.npz"
        output_path = temp_root / "xmem2_output.npz"
        np.savez_compressed(
            input_path,
            frames=np.asarray(frames),
            query_points_tyx=np.asarray(query_points_tyx, dtype=np.float32),
            query_boxes_xyxy=(
                np.asarray(query_boxes_xyxy, dtype=np.float32)
                if query_boxes_xyxy is not None
                else np.zeros((0, 4), dtype=np.float32)
            ),
            has_query_boxes=np.asarray(query_boxes_xyxy is not None, dtype=bool),
            query_object_ids=(
                np.asarray(query_object_ids, dtype=np.int32)
                if query_object_ids is not None
                else np.zeros((0,), dtype=np.int32)
            ),
            has_query_object_ids=np.asarray(query_object_ids is not None, dtype=bool),
        )

        command = [
            str(python_executable),
            str(script_path),
            "--input-npz",
            str(input_path),
            "--output-npz",
            str(output_path),
            "--checkpoint",
            str(settings.checkpoint_path),
            "--device",
            str(settings.device),
            "--representative-point-mode",
            str(settings.representative_point_mode),
            "--initial-bbox-padding-px",
            str(float(settings.initial_bbox_padding_px)),
            "--initial-mask-mode",
            str(settings.initial_mask_mode),
            "--initial-mask-threshold-rel",
            str(float(settings.initial_mask_threshold_rel)),
            "--initial-mask-disk-radius-ratio",
            str(float(settings.initial_mask_disk_radius_ratio)),
            "--subprocess-timeout-sec",
            str(float(settings.subprocess_timeout_sec)),
        ]
        if settings.repo_path is not None:
            command.extend(["--repo-path", str(settings.repo_path)])
        if settings.inference_resolution is not None:
            if isinstance(settings.inference_resolution, int):
                resolution = (
                    int(settings.inference_resolution),
                    int(settings.inference_resolution),
                )
            else:
                resolution = (
                    int(settings.inference_resolution[0]),
                    int(settings.inference_resolution[1]),
                )
            command.extend(
                [
                    "--height",
                    str(int(resolution[0])),
                    "--width",
                    str(int(resolution[1])),
                ]
            )
        if settings.mem_every is not None:
            command.extend(["--mem-every", str(int(settings.mem_every))])

        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            check=False,
            timeout=float(settings.subprocess_timeout_sec),
        )
        if completed.returncode != 0:
            payload = {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
            raise RuntimeError(
                "XMem2 subprocess inference failed:\n"
                + json.dumps(payload, indent=2, ensure_ascii=False)
            )
        if not output_path.exists():
            raise FileNotFoundError(f"XMem2 subprocess did not create output: {output_path}")

        with np.load(output_path, allow_pickle=False) as data:
            return XMem2OfflineResult(
                tracks_xy=np.asarray(data["tracks_xy"], dtype=np.float32),
                visible_mask=np.asarray(data["visible_mask"], dtype=bool),
                mask_areas=np.asarray(data["mask_areas"], dtype=np.float32),
                mask_bboxes_xyxy=np.asarray(data["mask_bboxes_xyxy"], dtype=np.float32),
                mask_scores=np.asarray(data["mask_scores"], dtype=np.float32),
                inference_resolution=tuple(int(v) for v in data["inference_resolution"].tolist()),
                original_resolution=tuple(int(v) for v in data["original_resolution"].tolist()),
                checkpoint_name=str(data["checkpoint_name"].tolist()),
                representative_point_mode=str(data["representative_point_mode"].tolist()),
                initial_mask_mode=str(data["initial_mask_mode"].tolist()),
                backend_mode="subprocess",
            )


def _discover_checkpoints_in_directory(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.pth"), key=lambda path: path.name.lower())


def _resolve_xmem2_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidates: list[Path] = []
    if repo_path is not None:
        candidates.append(Path(repo_path))
    candidates.extend(DEFAULT_XMEM2_REPO_CANDIDATES)

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if _looks_like_xmem2_repo(resolved):
            return resolved
    return None


def _looks_like_xmem2_repo(path: Path) -> bool:
    return (
        path.exists()
        and (path / "README.md").exists()
        and (path / "process_video.py").exists()
        and (path / "inference" / "run_on_video.py").exists()
        and (path / "model" / "network.py").exists()
    )


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoints: list[Path],
    models_dir: Path,
    repo_path: Optional[Path],
) -> Optional[Path]:
    if checkpoint_path is None:
        return select_default_xmem2_checkpoint(checkpoints)

    candidate = Path(checkpoint_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    by_name = {path.name: path for path in checkpoints}
    if candidate.name in by_name:
        return by_name[candidate.name]

    candidate_paths = [(models_dir / candidate).resolve()]
    if repo_path is not None:
        candidate_paths.append((repo_path / candidate).resolve())
        candidate_paths.append((repo_path / "saves" / candidate.name).resolve())
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    return candidate_paths[0]


def _resolve_xmem2_example_inputs(
    repo_path: Optional[Path],
) -> tuple[Optional[Path], Optional[Path]]:
    if repo_path is None:
        return None, None
    frames_dir = (repo_path / DEFAULT_XMEM2_EXAMPLE_FRAMES).resolve()
    masks_dir = (repo_path / DEFAULT_XMEM2_EXAMPLE_MASKS).resolve()
    return frames_dir, masks_dir


def _mask_matches_selected_frames(mask_path: Path, selected_frame_names: set[str]) -> bool:
    frame_name = f"{mask_path.stem}.jpg"
    return frame_name in selected_frame_names


def _extract_first_int(text: str) -> Optional[int]:
    match = re.search(r"\d+", text)
    if match is None:
        return None
    return int(match.group())


def _normalize_xmem2_mask_builder_mode(mode: str) -> str:
    normalized = str(mode or "threshold").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "threshold": "threshold",
        "threshold_mask_inside_bbox": "threshold",
        "threshold_inside_bbox": "threshold",
        "full_bbox": "full_bbox",
        "bbox": "full_bbox",
        "rect": "full_bbox",
        "box": "full_bbox",
        "disk": "disk",
        "disk_mask": "disk",
        "circle": "disk",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported XMem2 mask builder mode: {mode}. Expected threshold/full_bbox/disk."
        )
    return aliases[normalized]


def _frame_to_normalized_gray(frame: np.ndarray) -> np.ndarray:
    frame_np = np.asarray(frame)
    if frame_np.ndim == 3:
        if frame_np.shape[2] >= 3:
            rgb = frame_np[..., :3].astype(np.float32)
            gray = (
                0.2989 * rgb[..., 0]
                + 0.5870 * rgb[..., 1]
                + 0.1140 * rgb[..., 2]
            )
        else:
            gray = frame_np[..., 0].astype(np.float32)
    else:
        gray = frame_np.astype(np.float32)
    min_val = float(np.min(gray))
    max_val = float(np.max(gray))
    if max_val > min_val:
        gray = (gray - min_val) / (max_val - min_val)
    else:
        gray = np.zeros_like(gray, dtype=np.float32)
    return gray


def _clamp_roi_rect_xywh(
    roi_rect_xywh: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    x, y, width, height = (int(round(v)) for v in roi_rect_xywh)
    width = max(1, width)
    height = max(1, height)
    x = min(max(0, x), max(0, image_w - 1))
    y = min(max(0, y), max(0, image_h - 1))
    width = min(width, image_w - x)
    height = min(height, image_h - y)
    return x, y, max(1, width), max(1, height)


def _select_component_near_center(
    mask: np.ndarray,
    *,
    center_xy: tuple[float, float],
) -> np.ndarray | None:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2 or not np.any(mask_bool):
        return None

    visited = np.zeros_like(mask_bool, dtype=bool)
    best_component: np.ndarray | None = None
    best_distance = float("inf")
    rounded_center = (int(round(center_xy[0])), int(round(center_xy[1])))
    center_x = float(center_xy[0])
    center_y = float(center_xy[1])

    for row, col in np.argwhere(mask_bool):
        row_i = int(row)
        col_i = int(col)
        if visited[row_i, col_i]:
            continue
        component_pixels = _collect_connected_component(mask_bool, visited, row_i, col_i)
        contains_center = any(
            pixel_row == rounded_center[1] and pixel_col == rounded_center[0]
            for pixel_row, pixel_col in component_pixels
        )
        if contains_center:
            selected = np.zeros_like(mask_bool, dtype=bool)
            for pixel_row, pixel_col in component_pixels:
                selected[pixel_row, pixel_col] = True
            return selected
        coords = np.asarray(component_pixels, dtype=np.float32)
        comp_center_x = float(np.mean(coords[:, 1]))
        comp_center_y = float(np.mean(coords[:, 0]))
        distance = (comp_center_x - center_x) ** 2 + (comp_center_y - center_y) ** 2
        if distance < best_distance:
            best_distance = distance
            best_component = np.zeros_like(mask_bool, dtype=bool)
            for pixel_row, pixel_col in component_pixels:
                best_component[pixel_row, pixel_col] = True
    return best_component


def _collect_connected_component(
    mask: np.ndarray,
    visited: np.ndarray,
    start_row: int,
    start_col: int,
) -> list[tuple[int, int]]:
    queue = [(int(start_row), int(start_col))]
    visited[start_row, start_col] = True
    pixels: list[tuple[int, int]] = []
    height, width = mask.shape

    while queue:
        row, col = queue.pop()
        pixels.append((row, col))
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row = row + delta_row
            next_col = col + delta_col
            if (
                0 <= next_row < height
                and 0 <= next_col < width
                and mask[next_row, next_col]
                and not visited[next_row, next_col]
            ):
                visited[next_row, next_col] = True
                queue.append((next_row, next_col))
    return pixels


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a_bool = np.asarray(mask_a, dtype=bool)
    mask_b_bool = np.asarray(mask_b, dtype=bool)
    union = np.count_nonzero(mask_a_bool | mask_b_bool)
    if union == 0:
        return 1.0
    intersection = np.count_nonzero(mask_a_bool & mask_b_bool)
    return float(intersection / union)


def _mask_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0:
        return None
    y0 = int(np.min(coords[:, 0]))
    x0 = int(np.min(coords[:, 1]))
    y1 = int(np.max(coords[:, 0])) + 1
    x1 = int(np.max(coords[:, 1])) + 1
    return (x0, y0, x1, y1)


def _mask_centroid_xy(mask: np.ndarray) -> tuple[float, float] | None:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0:
        return None
    centroid_y = float(np.mean(coords[:, 0]))
    centroid_x = float(np.mean(coords[:, 1]))
    return (centroid_x, centroid_y)


def _expanded_roi_rect_from_mask(
    mask: np.ndarray,
    *,
    bbox_padding_ratio: float,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    bbox_xyxy = _mask_bbox_xyxy(mask)
    if bbox_xyxy is None:
        image_h, image_w = image_shape
        return (0, 0, max(1, image_w), max(1, image_h))
    x0, y0, x1, y1 = bbox_xyxy
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    pad_x = max(2, int(round(width * bbox_padding_ratio)))
    pad_y = max(2, int(round(height * bbox_padding_ratio)))
    roi = (
        x0 - pad_x,
        y0 - pad_y,
        width + 2 * pad_x,
        height + 2 * pad_y,
    )
    return _clamp_roi_rect_xywh(roi, image_shape)


def _generate_jittered_roi_rects(
    base_roi: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    *,
    jitter_px: int,
    jitter_scale: float,
) -> list[tuple[int, int, int, int]]:
    x, y, width, height = base_roi
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    shift_x = max(1, jitter_px, int(round(width * jitter_scale)))
    shift_y = max(1, jitter_px, int(round(height * jitter_scale)))
    scale_dx = max(1, int(round(width * jitter_scale)))
    scale_dy = max(1, int(round(height * jitter_scale)))

    candidates = [
        base_roi,
        (x - shift_x, y, width, height),
        (x + shift_x, y, width, height),
        (x, y - shift_y, width, height),
        (x, y + shift_y, width, height),
        (
            int(round(center_x - (width + scale_dx) / 2.0)),
            int(round(center_y - (height + scale_dy) / 2.0)),
            width + scale_dx,
            height + scale_dy,
        ),
        (
            int(round(center_x - max(1, width - scale_dx) / 2.0)),
            int(round(center_y - max(1, height - scale_dy) / 2.0)),
            max(1, width - scale_dx),
            max(1, height - scale_dy),
        ),
    ]
    unique: list[tuple[int, int, int, int]] = []
    for candidate in candidates:
        clamped = _clamp_roi_rect_xywh(candidate, image_shape)
        if clamped not in unique:
            unique.append(clamped)
    return unique


def _resolve_existing_checkpoint_file(checkpoint_path: Path) -> Path:
    resolved = checkpoint_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"XMem checkpoint not found: {resolved}")
    return resolved


def _normalize_xmem2_query_object_ids(
    query_object_ids: np.ndarray | None,
    num_references: int,
) -> np.ndarray:
    if query_object_ids is None:
        return np.arange(1, int(num_references) + 1, dtype=np.int32)

    object_ids = np.asarray(query_object_ids, dtype=np.int32).reshape(-1)
    if object_ids.shape != (int(num_references),):
        raise ValueError(
            f"query_object_ids must have shape {(int(num_references),)}, got {object_ids.shape}."
        )
    if np.any(object_ids <= 0):
        raise ValueError(
            f"query_object_ids must contain strictly positive integers, got {object_ids.tolist()}."
        )
    return object_ids.astype(np.int32, copy=False)


def _resolve_xmem2_python_executable(
    python_executable: str | Path | None,
) -> Path:
    if python_executable is not None:
        return Path(python_executable).expanduser().resolve()
    return Path(sys.executable).resolve()


def _normalize_representative_point_mode(mode: str) -> str:
    normalized = str(mode or "mask_centroid").strip().lower().replace(" ", "_")
    aliases = {
        "centroid": "mask_centroid",
        "mask_centroid": "mask_centroid",
        "maskcenter": "mask_centroid",
        "peak": "peak_in_mask",
        "peak_in_mask": "peak_in_mask",
        "peakinmask": "peak_in_mask",
        "bbox_center": "bbox_center",
        "bboxcenter": "bbox_center",
        "gaussian": "gaussian_in_mask",
        "2d_gauss": "gaussian_in_mask",
        "gaussian_in_mask": "gaussian_in_mask",
    }
    if normalized in aliases:
        return aliases[normalized]
    raise ValueError(f"Unsupported XMem2 representative point mode: {mode}")


def _prepare_frames_rgb(frames: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    frames_np = np.asarray(frames)
    if frames_np.ndim == 3:
        frames_np = np.stack([frames_np] * 3, axis=-1)
    elif frames_np.ndim == 4 and frames_np.shape[-1] == 1:
        frames_np = np.repeat(frames_np, 3, axis=-1)
    elif frames_np.ndim == 4 and frames_np.shape[-1] >= 3:
        frames_np = frames_np[..., :3]
    else:
        raise ValueError(f"Unsupported XMem2 frame shape: {frames_np.shape}")

    if frames_np.dtype != np.uint8:
        frames_np = _normalize_to_uint8(frames_np)

    original_resolution = (int(frames_np.shape[1]), int(frames_np.shape[2]))
    return frames_np, original_resolution


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.uint8)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros_like(arr, dtype=np.uint8)
    min_val = float(np.nanmin(arr[finite_mask]))
    max_val = float(np.nanmax(arr[finite_mask]))
    if max_val <= min_val:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - min_val) / (max_val - min_val)
    scaled = np.clip(scaled * 255.0, 0.0, 255.0)
    return scaled.astype(np.uint8)


def _resolve_xmem2_inference_resolution(
    resolution: int | tuple[int, int] | None,
    default_resolution: tuple[int, int],
) -> tuple[int, int]:
    default_h, default_w = int(default_resolution[0]), int(default_resolution[1])
    if resolution is None:
        return (default_h, default_w)
    if isinstance(resolution, int):
        target_short = int(resolution)
        if target_short <= 0:
            raise ValueError(f"Invalid XMem2 inference resolution: {resolution}")
        if min(default_h, default_w) == target_short:
            return (default_h, default_w)
        scale = float(target_short) / float(min(default_h, default_w))
        return (
            max(1, int(round(default_h * scale))),
            max(1, int(round(default_w * scale))),
        )
    if len(resolution) != 2:
        raise ValueError(f"Invalid XMem2 resolution: {resolution}")
    return (int(resolution[0]), int(resolution[1]))


def _resize_video_numpy(frames: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target[0]), int(target[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid XMem2 inference resolution: {(target_h, target_w)}")
    if frames.shape[1] == target_h and frames.shape[2] == target_w:
        return frames

    pil_image_module = importlib.import_module("PIL.Image")
    resized_frames = []
    for frame in frames:
        resized = pil_image_module.fromarray(frame).resize(
            (target_w, target_h),
            resample=pil_image_module.Resampling.BILINEAR,
        )
        resized_frames.append(np.asarray(resized, dtype=np.uint8))
    return np.stack(resized_frames, axis=0)


def _video_to_gray_float(frames: np.ndarray) -> np.ndarray:
    frames_np = np.asarray(frames, dtype=np.float32)
    if frames_np.ndim != 4 or frames_np.shape[-1] < 3:
        raise ValueError(f"Expected RGB frames for grayscale conversion, got {frames_np.shape}.")
    gray = (
        0.2989 * frames_np[..., 0]
        + 0.5870 * frames_np[..., 1]
        + 0.1140 * frames_np[..., 2]
    )
    return gray.astype(np.float32, copy=False)


def _scale_query_points_tyx(
    query_points_tyx: np.ndarray,
    source_resolution: tuple[int, int],
    target_resolution: tuple[int, int],
) -> np.ndarray:
    src_h, src_w = source_resolution
    dst_h, dst_w = target_resolution
    scaled = np.array(query_points_tyx, dtype=np.float32, copy=True)
    if src_h <= 0 or src_w <= 0:
        return scaled
    scaled[:, 1] = scaled[:, 1] * (float(dst_h) / float(src_h))
    scaled[:, 2] = scaled[:, 2] * (float(dst_w) / float(src_w))
    return scaled


def _scale_tracks_xy(
    tracks_xy: np.ndarray,
    source_resolution: tuple[int, int],
    target_resolution: tuple[int, int],
) -> np.ndarray:
    src_h, src_w = source_resolution
    dst_h, dst_w = target_resolution
    scaled = np.array(tracks_xy, dtype=np.float32, copy=True)
    if src_h <= 0 or src_w <= 0:
        return scaled
    scaled[..., 0] = scaled[..., 0] * (float(dst_w) / float(src_w))
    scaled[..., 1] = scaled[..., 1] * (float(dst_h) / float(src_h))
    return scaled


def _scale_boxes_xyxy(
    boxes_xyxy: np.ndarray | None,
    source_resolution: tuple[int, int],
    target_resolution: tuple[int, int],
) -> np.ndarray | None:
    if boxes_xyxy is None:
        return None
    src_h, src_w = source_resolution
    dst_h, dst_w = target_resolution
    scaled = np.array(boxes_xyxy, dtype=np.float32, copy=True)
    if src_h <= 0 or src_w <= 0:
        return scaled
    scaled[..., 0] = scaled[..., 0] * (float(dst_w) / float(src_w))
    scaled[..., 2] = scaled[..., 2] * (float(dst_w) / float(src_w))
    scaled[..., 1] = scaled[..., 1] * (float(dst_h) / float(src_h))
    scaled[..., 3] = scaled[..., 3] * (float(dst_h) / float(src_h))
    return scaled


def _build_xmem2_prompt_boxes_xyxy(
    query_points_tyx: np.ndarray,
    *,
    resolution: tuple[int, int],
    initial_bbox_padding_px: float,
    provided_boxes_xyxy: np.ndarray | None,
) -> np.ndarray:
    height, width = int(resolution[0]), int(resolution[1])
    if provided_boxes_xyxy is not None:
        boxes = np.asarray(provided_boxes_xyxy, dtype=np.float32, copy=True)
    else:
        padding = max(1.0, float(initial_bbox_padding_px))
        boxes = np.zeros((query_points_tyx.shape[0], 4), dtype=np.float32)
        boxes[:, 0] = query_points_tyx[:, 2] - padding
        boxes[:, 1] = query_points_tyx[:, 1] - padding
        boxes[:, 2] = query_points_tyx[:, 2] + padding
        boxes[:, 3] = query_points_tyx[:, 1] + padding

    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, max(0.0, float(width - 1)))
    boxes[:, 2] = np.clip(boxes[:, 2], 0.0, float(width))
    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, max(0.0, float(height - 1)))
    boxes[:, 3] = np.clip(boxes[:, 3], 0.0, float(height))

    invalid = boxes[:, 2] <= boxes[:, 0]
    boxes[invalid, 2] = np.minimum(float(width), boxes[invalid, 0] + 1.0)
    invalid = boxes[:, 3] <= boxes[:, 1]
    boxes[invalid, 3] = np.minimum(float(height), boxes[invalid, 1] + 1.0)
    return boxes.astype(np.float32, copy=False)


def _extract_query_seed_frames(query_points_tyx: np.ndarray, total_frames: int) -> np.ndarray:
    seed_frames = np.rint(np.asarray(query_points_tyx[:, 0], dtype=np.float32)).astype(np.int32)
    if np.any(seed_frames < 0) or np.any(seed_frames >= int(total_frames)):
        raise ValueError(
            f"XMem2 query seed frames must be in [0, {int(total_frames) - 1}], got {seed_frames.tolist()}."
        )
    return seed_frames


def _write_xmem2_video_frames(frames_dir: Path, frames: np.ndarray) -> None:
    Image = importlib.import_module("PIL.Image")
    for frame_index, frame_rgb in enumerate(frames):
        frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"
        Image.fromarray(frame_rgb).save(str(frame_path), quality=95)


def _write_xmem2_seed_masks(
    masks_dir: Path,
    *,
    frames: np.ndarray,
    boxes_xyxy: np.ndarray,
    seed_frames: np.ndarray,
    class_ids: np.ndarray,
    initial_mask_mode: str,
    threshold_rel: float,
    disk_radius_ratio: float,
) -> list[int]:
    Image = importlib.import_module("PIL.Image")
    used_seed_frames: list[int] = []
    num_queries = int(boxes_xyxy.shape[0])

    for frame_index in sorted(set(int(idx) for idx in seed_frames.tolist())):
        frame_rgb = frames[frame_index]
        idx_mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
        has_any_object = False
        for query_index in range(num_queries):
            if int(seed_frames[query_index]) != frame_index:
                continue
            bbox = np.asarray(boxes_xyxy[query_index], dtype=np.float32)
            roi_rect_xywh = (
                int(np.floor(float(bbox[0]))),
                int(np.floor(float(bbox[1]))),
                int(max(1.0, np.ceil(float(bbox[2]) - float(bbox[0])))),
                int(max(1.0, np.ceil(float(bbox[3]) - float(bbox[1])))),
            )
            seed_mask = build_xmem2_initial_mask(
                frame_rgb,
                roi_rect_xywh,
                mode=initial_mask_mode,
                threshold_rel=threshold_rel,
                disk_radius_ratio=disk_radius_ratio,
            )
            if not np.any(seed_mask):
                seed_mask = build_xmem2_initial_mask(
                    frame_rgb,
                    roi_rect_xywh,
                    mode="full_bbox",
                    threshold_rel=threshold_rel,
                    disk_radius_ratio=disk_radius_ratio,
                )
            free_mask = seed_mask & (idx_mask == 0)
            if not np.any(free_mask):
                x, y, width_xy, height_xy = _clamp_roi_rect_xywh(roi_rect_xywh, frame_rgb.shape[:2])
                center_x = min(frame_rgb.shape[1] - 1, max(0, int(round(x + (width_xy - 1) / 2.0))))
                center_y = min(frame_rgb.shape[0] - 1, max(0, int(round(y + (height_xy - 1) / 2.0))))
                free_mask = np.zeros_like(seed_mask, dtype=bool)
                free_mask[center_y, center_x] = True

            object_id = int(class_ids[query_index])
            idx_mask[free_mask] = object_id
            has_any_object = True

        if not has_any_object:
            continue
        used_seed_frames.append(frame_index)
        mask_path = masks_dir / f"frame_{frame_index:06d}.png"
        Image.fromarray(idx_mask, mode="P").save(str(mask_path))

    return used_seed_frames


def _record_xmem2_frame_outputs(
    *,
    class_mask: np.ndarray,
    frame_gray: np.ndarray,
    frame_index: int,
    tracks_xy: np.ndarray,
    visible_mask: np.ndarray,
    mask_areas: np.ndarray,
    mask_bboxes_xyxy: np.ndarray,
    mask_scores: np.ndarray,
    representative_point_mode: str,
    num_objects: int,
    area_scale: float,
) -> None:
    class_mask_np = np.asarray(class_mask, dtype=np.int32)
    for query_index in range(num_objects):
        object_id = int(query_index + 1)
        obj_mask = class_mask_np == object_id
        if not np.any(obj_mask):
            continue
        bbox_xyxy = _mask_bbox_xyxy(obj_mask)
        representative_point = _mask_to_representative_point(
            obj_mask,
            frame_gray=frame_gray,
            mode=representative_point_mode,
        )
        if bbox_xyxy is not None:
            mask_bboxes_xyxy[query_index, frame_index, :] = np.asarray(
                bbox_xyxy,
                dtype=np.float32,
            )
        mask_areas[query_index, frame_index] = float(np.count_nonzero(obj_mask) * area_scale)
        mask_scores[query_index, frame_index] = 1.0
        if representative_point is None:
            continue
        tracks_xy[query_index, frame_index, 0] = float(representative_point[0])
        tracks_xy[query_index, frame_index, 1] = float(representative_point[1])
        visible_mask[query_index, frame_index] = True


def _mask_to_representative_point(
    mask_bool: np.ndarray,
    *,
    frame_gray: np.ndarray,
    mode: str,
) -> tuple[float, float] | None:
    bbox = _mask_bbox_xyxy(mask_bool)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox

    if mode == "bbox_center":
        return (float((x0 + x1) * 0.5), float((y0 + y1) * 0.5))

    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0 or ys.size == 0:
        return None

    if mode == "mask_centroid":
        return (float(np.mean(xs)), float(np.mean(ys)))

    if mode == "peak_in_mask":
        return _peak_in_mask(mask_bool, frame_gray)

    if mode == "gaussian_in_mask":
        gaussian_point = _gaussian_center_in_mask(mask_bool, frame_gray, bbox)
        if gaussian_point is not None:
            return gaussian_point
        return _peak_in_mask(mask_bool, frame_gray)

    raise ValueError(f"Unsupported XMem2 representative point mode: {mode}")


def _peak_in_mask(mask_bool: np.ndarray, frame_gray: np.ndarray) -> tuple[float, float] | None:
    masked = np.where(mask_bool, np.asarray(frame_gray, dtype=np.float32), -np.inf)
    try:
        peak_index = int(np.nanargmax(masked))
    except ValueError:
        return None
    peak_y, peak_x = np.unravel_index(peak_index, masked.shape)
    if not np.isfinite(masked[peak_y, peak_x]):
        return None
    return (float(peak_x), float(peak_y))


def _gaussian_center_in_mask(
    mask_bool: np.ndarray,
    frame_gray: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
) -> tuple[float, float] | None:
    try:
        from trackleed.analysis.peak_analyzer import fit_spot_to_gaussian
    except Exception:
        return None

    x0, y0, x1, y1 = bbox_xyxy
    x0_i = max(0, int(np.floor(x0)))
    y0_i = max(0, int(np.floor(y0)))
    x1_i = max(x0_i + 1, int(np.ceil(x1)))
    y1_i = max(y0_i + 1, int(np.ceil(y1)))

    roi_mask = mask_bool[y0_i:y1_i, x0_i:x1_i]
    roi_gray = np.asarray(frame_gray, dtype=np.float32)[y0_i:y1_i, x0_i:x1_i]
    if roi_mask.size == 0 or roi_gray.size == 0 or not np.any(roi_mask):
        return None

    masked_roi = np.where(roi_mask, roi_gray, 0.0)
    fit = fit_spot_to_gaussian(masked_roi)
    if not fit:
        return None

    center_x = float(fit.get("center_x", np.nan))
    center_y = float(fit.get("center_y", np.nan))
    if not np.isfinite(center_x) or not np.isfinite(center_y):
        return None
    return (float(x0_i + center_x), float(y0_i + center_y))


def _normalize_path_key(path_like: str | Path | None) -> str:
    if path_like is None:
        return ""
    try:
        return str(Path(path_like).resolve()).lower()
    except Exception:
        return str(path_like).lower()


@contextmanager
def _xmem2_import_context(repo_path: Optional[Path]) -> Iterator[None]:
    if repo_path is None:
        yield
        return

    resolved_repo_path = repo_path.resolve()
    repo_root_str = str(resolved_repo_path)
    repo_root_key = _normalize_path_key(resolved_repo_path)
    repo_parent_key = _normalize_path_key(resolved_repo_path.parent)
    original_sys_path = list(sys.path)

    filtered_sys_path: list[str] = []
    for entry in original_sys_path:
        entry_key = _normalize_path_key(entry)
        if not entry_key:
            filtered_sys_path.append(entry)
            continue
        if entry_key in {repo_root_key, repo_parent_key}:
            continue
        filtered_sys_path.append(entry)

    original_modules: dict[str, Any] = {}
    for module_name, module in list(sys.modules.items()):
        top_level = module_name.split(".", 1)[0]
        if top_level in XMEM2_IMPORT_ROOTS:
            original_modules[module_name] = module

    sys.path[:] = [repo_root_str, *filtered_sys_path]
    for module_name in list(sys.modules):
        top_level = module_name.split(".", 1)[0]
        if top_level in XMEM2_IMPORT_ROOTS:
            sys.modules.pop(module_name, None)
    try:
        yield
    finally:
        for module_name in list(sys.modules):
            top_level = module_name.split(".", 1)[0]
            if top_level in XMEM2_IMPORT_ROOTS:
                sys.modules.pop(module_name, None)
        sys.modules.update(original_modules)
        sys.path[:] = original_sys_path
