"""SAM 2 environment diagnostics and smoke-test helpers."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Any
import importlib
import json
import os
import subprocess
import sys

import cv2
import numpy as np


DEFAULT_SAM2_MODELS_DIR = Path("tracking_models") / "sam2"
DEFAULT_SAM2_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "sam2",
    Path(__file__).resolve().parents[2] / "sam2",
)
DEFAULT_SAM2_SUBPROCESS_SCRIPT = Path("scripts") / "run_sam2_subprocess.py"
CHECKPOINT_PRIORITY = (
    "sam2.1_hiera_small.pt",
    "sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt",
    "sam2_hiera_small.pt",
    "sam2_hiera_tiny.pt",
    "sam2_hiera_base_plus.pt",
    "sam2_hiera_large.pt",
)
CHECKPOINT_TO_CONFIG = {
    "sam2.1_hiera_tiny.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2_hiera_tiny.pt": "configs/sam2/sam2_hiera_t.yaml",
    "sam2_hiera_small.pt": "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_base_plus.pt": "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_large.pt": "configs/sam2/sam2_hiera_l.yaml",
}

_CONFIG_ROOT_CANDIDATES = (
    Path("."),
    Path("sam2"),
)


@dataclass
class Sam2EnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    available_checkpoints: list[Path] = field(default_factory=list)
    selected_checkpoint: Optional[Path] = None
    checkpoint_exists: bool = False
    config_path: Optional[Path] = None
    config_exists: bool = False
    config_identifier: Optional[str] = None
    torch_importable: bool = False
    torch_version: Optional[str] = None
    cuda_available: bool = False
    sam2_importable: bool = False
    sam2_module_path: Optional[str] = None
    build_sam_importable: bool = False
    image_predictor_importable: bool = False
    video_predictor_importable: bool = False
    cuda_extension_importable: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return (
            self.repo_path is not None
            and self.checkpoint_exists
            and self.config_exists
            and self.config_identifier is not None
            and self.torch_importable
            and self.cuda_available
            and self.sam2_importable
            and self.build_sam_importable
            and self.image_predictor_importable
            and self.video_predictor_importable
        )


@dataclass
class Sam2SmokeTestResult:
    image_masks_shape: tuple[int, ...]
    image_ious_shape: tuple[int, ...]
    video_prompt_masks_shape: tuple[int, ...]
    video_propagated_masks_shape: tuple[int, ...]
    propagated_frame_count: int
    first_image_mask_area: int
    first_video_mask_area: int
    device: str
    checkpoint_name: str
    config_name: str
    cuda_extension_importable: bool


@dataclass
class Sam2RuntimeSettings:
    checkpoint_path: str | Path
    repo_path: str | Path | None = None
    python_executable: str | Path | None = None
    config_path: str | Path | None = None
    device: str = "auto"
    backend_mode: str = "direct"
    prompt_mode: str = "point+bbox"
    representative_point_mode: str = "mask_centroid"
    inference_resolution: int | tuple[int, int] | None = None
    initial_bbox_padding_px: float = 12.0
    mask_threshold: float = 0.5
    min_mask_area: float = 1.0
    max_mask_area: float | None = None
    use_vos_optimized: bool = False
    apply_postprocessing: bool = True
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = False
    async_loading_frames: bool = False
    subprocess_timeout_sec: float = 1800.0


@dataclass
class Sam2OfflineResult:
    tracks_xy: np.ndarray
    visible_mask: np.ndarray
    mask_areas: np.ndarray
    mask_bboxes_xyxy: np.ndarray
    mask_scores: np.ndarray
    mask_component_counts: np.ndarray
    inference_resolution: tuple[int, int]
    original_resolution: tuple[int, int]
    checkpoint_name: str
    config_name: str
    prompt_mode: str
    representative_point_mode: str
    use_vos_optimized: bool
    backend_mode: str


def discover_sam2_checkpoints(
    models_dir: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_SAM2_MODELS_DIR
    discovered: dict[str, Path] = {}

    for candidate in _discover_checkpoints_in_directory(directory):
        discovered[str(candidate.resolve())] = candidate.resolve()

    repo_root = _resolve_sam2_repo_path(repo_path)
    if repo_root is not None:
        for candidate in _discover_checkpoints_in_directory(repo_root / "checkpoints"):
            discovered[str(candidate.resolve())] = candidate.resolve()

    return sorted(discovered.values(), key=lambda path: path.name.lower())


def select_default_sam2_checkpoint(checkpoints: list[Path]) -> Optional[Path]:
    if not checkpoints:
        return None
    by_name = {path.name: path for path in checkpoints}
    for preferred_name in CHECKPOINT_PRIORITY:
        if preferred_name in by_name:
            return by_name[preferred_name]
    return checkpoints[0]


def probe_sam2_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Sam2EnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_SAM2_MODELS_DIR
    directory = directory.resolve()
    resolved_repo_path = _resolve_sam2_repo_path(repo_path)
    checkpoints = discover_sam2_checkpoints(directory, resolved_repo_path)
    selected_checkpoint = _resolve_checkpoint_path(
        checkpoint_path,
        checkpoints,
        directory,
        resolved_repo_path,
    )
    resolved_config_path = _resolve_config_path(config_path, resolved_repo_path, selected_checkpoint)
    config_identifier = _resolve_config_identifier(config_path, resolved_repo_path, selected_checkpoint)

    status = Sam2EnvironmentStatus(
        models_dir=directory,
        repo_path=resolved_repo_path,
        available_checkpoints=checkpoints,
        selected_checkpoint=selected_checkpoint,
        checkpoint_exists=bool(selected_checkpoint and selected_checkpoint.exists()),
        config_path=resolved_config_path,
        config_exists=bool(resolved_config_path and resolved_config_path.exists()),
        config_identifier=config_identifier,
    )

    if not directory.exists():
        status.warnings.append(
            f"Checkpoint directory does not exist yet: {directory}. "
            "This is fine if you keep SAM 2 checkpoints inside the local repo/checkpoints directory."
        )

    if selected_checkpoint is None:
        status.warnings.append(
            "No SAM 2 checkpoint detected in tracking_models/sam2 or in repo/checkpoints."
        )
    elif checkpoint_path and not status.checkpoint_exists:
        status.errors.append(f"Selected checkpoint not found: {selected_checkpoint}")

    if resolved_repo_path is None:
        status.errors.append(
            "Could not resolve the local SAM 2 repo. Clone `sam2` next to the project or pass --repo-path."
        )
        status.hints.append(
            "Expected repo root containing `setup.py`, `sam2/build_sam.py`, and `sam2/sam2_video_predictor.py`."
        )

    if resolved_config_path is None or not status.config_exists:
        status.errors.append(
            "Could not resolve the SAM 2 config matching the selected checkpoint."
        )
        status.hints.append(
            "For `sam2.1_hiera_small.pt`, use `configs/sam2.1/sam2.1_hiera_s.yaml`."
        )

    if config_identifier is None:
        status.errors.append(
            "Could not derive a Hydra config identifier for SAM 2."
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
            "Install `torch>=2.5.1` and matching `torchvision` in the dedicated `sam2` env."
        )

    with _sam2_import_context(resolved_repo_path):
        try:
            sam2_module = importlib.import_module("sam2")
            status.sam2_importable = True
            module_file = getattr(sam2_module, "__file__", None)
            if module_file:
                status.sam2_module_path = str(Path(module_file).resolve())
            else:
                module_paths = getattr(sam2_module, "__path__", None)
                if module_paths:
                    status.sam2_module_path = str(Path(next(iter(module_paths))).resolve())
        except Exception as exc:
            status.errors.append(f"SAM 2 import failed: {exc}")
            status.hints.append("Install the local repo with `pip install -e \".[notebooks]\"`.")

        try:
            importlib.import_module("sam2.build_sam")
            status.build_sam_importable = True
        except Exception as exc:
            status.errors.append(f"SAM 2 build_sam import failed: {exc}")
            _append_sam2_import_hints(status, exc)

        try:
            importlib.import_module("sam2.sam2_image_predictor")
            status.image_predictor_importable = True
        except Exception as exc:
            status.errors.append(f"SAM 2 image predictor import failed: {exc}")
            _append_sam2_import_hints(status, exc)

        try:
            importlib.import_module("sam2.sam2_video_predictor")
            status.video_predictor_importable = True
        except Exception as exc:
            status.errors.append(f"SAM 2 video predictor import failed: {exc}")
            _append_sam2_import_hints(status, exc)

        try:
            importlib.import_module("sam2._C")
            status.cuda_extension_importable = True
        except Exception:
            status.warnings.append(
                "SAM 2 custom CUDA extension is not available. This is OK; mask post-processing may be skipped."
            )

    if status.torch_importable and not status.cuda_available:
        status.warnings.append(
            "PyTorch CUDA is not available. SAM 2 may still run on CPU, but LEED sequences will be very slow."
        )

    if sys.platform.startswith("win"):
        status.warnings.append(
            "Upstream SAM 2 recommends WSL/Ubuntu on Windows. Native Windows can still work if the predictor smoke test passes."
        )

    return status


def format_sam2_environment_report(status: Sam2EnvironmentStatus) -> str:
    selected_checkpoint = (
        str(status.selected_checkpoint) if status.selected_checkpoint is not None else "None"
    )
    available = ", ".join(path.name for path in status.available_checkpoints) or "none"
    lines = [
        "SAM 2 environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Selected checkpoint: {selected_checkpoint}",
        f"- Available checkpoints: {available}",
        f"- Config path: {status.config_path or 'None'}",
        f"- Config identifier: {status.config_identifier or 'None'}",
        f"- PyTorch import: {'ok' if status.torch_importable else 'failed'}",
        f"- PyTorch version: {status.torch_version or 'unknown'}",
        f"- CUDA available: {'yes' if status.cuda_available else 'no'}",
        f"- SAM 2 import: {'ok' if status.sam2_importable else 'failed'}",
        f"- build_sam import: {'ok' if status.build_sam_importable else 'failed'}",
        f"- Image predictor import: {'ok' if status.image_predictor_importable else 'failed'}",
        f"- Video predictor import: {'ok' if status.video_predictor_importable else 'failed'}",
        f"- CUDA extension: {'ok' if status.cuda_extension_importable else 'missing'}",
    ]
    if status.sam2_module_path:
        lines.append(f"- SAM 2 module: {status.sam2_module_path}")
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


def run_sam2_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: Sam2RuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
) -> Sam2OfflineResult:
    """
    Run SAM 2 video propagation on a sequence of frames and sparse seed points.

    Supports both direct import and subprocess execution modes.
    """
    frames_np = np.asarray(frames)
    if frames_np.ndim not in (3, 4):
        raise ValueError(f"Expected frames with ndim 3 or 4, got shape {frames_np.shape}.")
    if frames_np.size == 0:
        raise ValueError("No frames provided for SAM 2 inference.")

    query_np = np.asarray(query_points_tyx, dtype=np.float32)
    if query_np.ndim != 2 or query_np.shape[1] != 3:
        raise ValueError("query_points_tyx must have shape [num_queries, 3] in [t, y, x] format.")
    if query_np.shape[0] == 0:
        raise ValueError("No query points provided for SAM 2 inference.")

    boxes_np = None
    if query_boxes_xyxy is not None:
        boxes_np = np.asarray(query_boxes_xyxy, dtype=np.float32)
        if boxes_np.shape != (query_np.shape[0], 4):
            raise ValueError(
                f"query_boxes_xyxy must have shape {(query_np.shape[0], 4)}, got {boxes_np.shape}."
            )

    backend_mode = str(settings.backend_mode or "direct").strip().lower()
    if backend_mode == "auto":
        backend_mode = "subprocess"

    if backend_mode == "direct":
        return _run_sam2_direct_offline(
            frames_np,
            query_np,
            settings,
            query_boxes_xyxy=boxes_np,
        )
    if backend_mode != "subprocess":
        raise ValueError(
            f"Unsupported SAM 2 backend mode: {settings.backend_mode}. "
            "Expected `subprocess` or `direct`."
        )
    return _run_sam2_subprocess_offline(
        frames_np,
        query_np,
        settings,
        query_boxes_xyxy=boxes_np,
    )


def _run_sam2_direct_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: Sam2RuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
) -> Sam2OfflineResult:
    """
    Run SAM 2 video propagation on a sequence of frames and sparse seed points.

    Args:
        frames: [T, H, W] grayscale or [T, H, W, C] input frames.
        query_points_tyx: [Q, 3] in [t, y, x] using original frame coordinates.
        settings: SAM 2 runtime settings.
        query_boxes_xyxy: Optional [Q, 4] init boxes in [x0, y0, x1, y1].

    Returns:
        Sam2OfflineResult containing representative points and mask summaries.
    """
    frames_np = np.asarray(frames)
    query_np = np.asarray(query_points_tyx, dtype=np.float32)
    boxes_np = (
        None if query_boxes_xyxy is None else np.asarray(query_boxes_xyxy, dtype=np.float32)
    )

    mask_threshold = float(settings.mask_threshold)
    if not 0.0 <= mask_threshold <= 1.0:
        raise ValueError(f"SAM 2 mask_threshold must be in [0, 1], got {mask_threshold}.")

    resolved_repo_path = _resolve_sam2_repo_path(settings.repo_path)
    if resolved_repo_path is None:
        raise FileNotFoundError(
            "Could not resolve the SAM 2 repo. Pass repo_path pointing at the local `sam2` clone."
        )

    torch = importlib.import_module("torch")
    with _sam2_import_context(resolved_repo_path):
        build_sam_module = importlib.import_module("sam2.build_sam")
    build_sam2_video_predictor = getattr(build_sam_module, "build_sam2_video_predictor")
    resolved_device = _resolve_torch_device(torch, settings.device)

    checkpoint_file = _resolve_existing_checkpoint_file(Path(settings.checkpoint_path))
    config_identifier = _resolve_config_identifier(
        settings.config_path,
        resolved_repo_path,
        checkpoint_file,
    )
    resolved_config_path = _resolve_config_path(
        settings.config_path,
        resolved_repo_path,
        checkpoint_file,
    )
    if config_identifier is None or resolved_config_path is None or not resolved_config_path.exists():
        raise FileNotFoundError(
            f"SAM 2 config not found: {resolved_config_path or settings.config_path or 'None'}"
        )

    prepared_frames, original_resolution = _prepare_frames_rgb(frames_np)
    inference_resolution = _normalize_optional_resolution(
        settings.inference_resolution,
        original_resolution,
    )
    resized_frames = _resize_video_numpy(prepared_frames, inference_resolution)
    gray_frames = _video_to_gray_float(resized_frames)
    scaled_queries = _scale_query_points_tyx(
        query_np,
        source_resolution=original_resolution,
        target_resolution=inference_resolution,
    )
    scaled_boxes = _build_sam2_prompt_boxes_xyxy(
        scaled_queries,
        resolution=inference_resolution,
        initial_bbox_padding_px=float(settings.initial_bbox_padding_px),
        provided_boxes_xyxy=(
            _scale_boxes_xyxy(
                boxes_np,
                source_resolution=original_resolution,
                target_resolution=inference_resolution,
            )
            if boxes_np is not None
            else None
        ),
    )
    seed_frames = _extract_query_seed_frames(scaled_queries, total_frames=resized_frames.shape[0])
    prompt_mode = _normalize_prompt_mode(settings.prompt_mode)
    representative_point_mode = _normalize_representative_point_mode(
        settings.representative_point_mode
    )

    num_queries = int(scaled_queries.shape[0])
    num_frames = int(resized_frames.shape[0])
    tracks_xy_resized = np.full((num_queries, num_frames, 2), np.nan, dtype=np.float32)
    visible_mask = np.zeros((num_queries, num_frames), dtype=bool)
    mask_areas = np.zeros((num_queries, num_frames), dtype=np.float32)
    mask_bboxes_xyxy_resized = np.full((num_queries, num_frames, 4), np.nan, dtype=np.float32)
    mask_scores = np.zeros((num_queries, num_frames), dtype=np.float32)
    mask_component_counts = np.zeros((num_queries, num_frames), dtype=np.int32)

    video_predictor = build_sam2_video_predictor(
        config_identifier,
        str(checkpoint_file),
        device=resolved_device,
        vos_optimized=bool(settings.use_vos_optimized),
        apply_postprocessing=bool(settings.apply_postprocessing),
    )

    _configure_cuda_runtime(torch, resolved_device)
    autocast_context = _make_autocast_context(torch, resolved_device)
    with TemporaryDirectory(prefix="trackleed_sam2_") as temp_dir:
        temp_root = Path(temp_dir)
        for seed_frame in np.unique(seed_frames).tolist():
            query_indices = np.where(seed_frames == seed_frame)[0]
            local_frames = resized_frames[seed_frame:]
            local_grays = gray_frames[seed_frame:]
            if local_frames.shape[0] <= 0:
                continue

            video_dir = temp_root / f"seed_{int(seed_frame):05d}"
            _write_video_sequence(video_dir, local_frames)
            state = video_predictor.init_state(
                str(video_dir),
                offload_video_to_cpu=bool(settings.offload_video_to_cpu),
                offload_state_to_cpu=bool(settings.offload_state_to_cpu),
                async_loading_frames=bool(settings.async_loading_frames),
            )

            obj_id_to_query_idx: dict[int, int] = {}
            for query_index in query_indices.tolist():
                point_xy = np.asarray(
                    [[scaled_queries[query_index, 2], scaled_queries[query_index, 1]]],
                    dtype=np.float32,
                )
                labels = np.asarray([1], dtype=np.int32)
                box_xyxy = scaled_boxes[query_index]
                obj_id = int(query_index) + 1
                obj_id_to_query_idx[obj_id] = int(query_index)
                video_predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=point_xy if prompt_mode in {"point", "point+bbox"} else None,
                    labels=labels if prompt_mode in {"point", "point+bbox"} else None,
                    box=box_xyxy if prompt_mode in {"bbox", "point+bbox"} else None,
                )

            with torch.inference_mode():
                with autocast_context:
                    for local_frame_idx, obj_ids, video_res_masks in video_predictor.propagate_in_video(
                        state,
                        start_frame_idx=0,
                        max_frame_num_to_track=int(local_frames.shape[0]),
                    ):
                        global_frame_idx = int(seed_frame + local_frame_idx)
                        frame_gray = local_grays[int(local_frame_idx)]
                        _record_sam2_frame_outputs(
                            video_res_masks=video_res_masks,
                            obj_ids=obj_ids,
                            obj_id_to_query_idx=obj_id_to_query_idx,
                            frame_gray=frame_gray,
                            frame_index=global_frame_idx,
                            tracks_xy=tracks_xy_resized,
                            visible_mask=visible_mask,
                            mask_areas=mask_areas,
                            mask_bboxes_xyxy=mask_bboxes_xyxy_resized,
                            mask_scores=mask_scores,
                            mask_component_counts=mask_component_counts,
                            representative_point_mode=representative_point_mode,
                            mask_threshold=mask_threshold,
                            min_mask_area=float(settings.min_mask_area),
                            max_mask_area=(
                                None
                                if settings.max_mask_area is None
                                else float(settings.max_mask_area)
                            ),
                            inference_resolution=inference_resolution,
                            original_resolution=original_resolution,
                        )

            if hasattr(video_predictor, "reset_state"):
                video_predictor.reset_state(state)

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

    return Sam2OfflineResult(
        tracks_xy=tracks_xy,
        visible_mask=visible_mask,
        mask_areas=mask_areas,
        mask_bboxes_xyxy=mask_bboxes_xyxy,
        mask_scores=mask_scores,
        mask_component_counts=mask_component_counts,
        inference_resolution=inference_resolution,
        original_resolution=original_resolution,
        checkpoint_name=checkpoint_file.name,
        config_name=resolved_config_path.name,
        prompt_mode=prompt_mode,
        representative_point_mode=representative_point_mode,
        use_vos_optimized=bool(settings.use_vos_optimized),
        backend_mode="direct",
    )


def _run_sam2_subprocess_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: Sam2RuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
) -> Sam2OfflineResult:
    python_executable = _resolve_sam2_python_executable(settings.python_executable)
    script_path = (Path(__file__).resolve().parents[2] / DEFAULT_SAM2_SUBPROCESS_SCRIPT).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"SAM 2 subprocess script not found: {script_path}")

    with TemporaryDirectory(prefix="trackleed_sam2_") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "sam2_input.npz"
        output_path = temp_root / "sam2_output.npz"
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
            "--prompt-mode",
            str(settings.prompt_mode),
            "--representative-point-mode",
            str(settings.representative_point_mode),
            "--initial-bbox-padding-px",
            str(float(settings.initial_bbox_padding_px)),
            "--mask-threshold",
            str(float(settings.mask_threshold)),
            "--min-mask-area",
            str(float(settings.min_mask_area)),
            "--subprocess-timeout-sec",
            str(float(settings.subprocess_timeout_sec)),
        ]
        if settings.repo_path is not None:
            command.extend(["--repo-path", str(settings.repo_path)])
        if settings.config_path is not None:
            command.extend(["--config", str(settings.config_path)])
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
            command.extend(["--height", str(int(resolution[0])), "--width", str(int(resolution[1]))])
        if settings.max_mask_area is not None:
            command.extend(["--max-mask-area", str(float(settings.max_mask_area))])
        if settings.use_vos_optimized:
            command.append("--use-vos-optimized")
        if settings.apply_postprocessing:
            command.append("--apply-postprocessing")
        else:
            command.append("--disable-postprocessing")
        if settings.offload_video_to_cpu:
            command.append("--offload-video-to-cpu")
        if settings.offload_state_to_cpu:
            command.append("--offload-state-to-cpu")
        if settings.async_loading_frames:
            command.append("--async-loading-frames")

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
                "SAM 2 subprocess inference failed:\n"
                + json.dumps(payload, indent=2, ensure_ascii=False)
            )

        if not output_path.exists():
            raise FileNotFoundError(f"SAM 2 subprocess did not create output: {output_path}")

        with np.load(output_path, allow_pickle=False) as data:
            return Sam2OfflineResult(
                tracks_xy=np.asarray(data["tracks_xy"], dtype=np.float32),
                visible_mask=np.asarray(data["visible_mask"], dtype=bool),
                mask_areas=np.asarray(data["mask_areas"], dtype=np.float32),
                mask_bboxes_xyxy=np.asarray(data["mask_bboxes_xyxy"], dtype=np.float32),
                mask_scores=np.asarray(data["mask_scores"], dtype=np.float32),
                mask_component_counts=np.asarray(data["mask_component_counts"], dtype=np.int32),
                inference_resolution=tuple(int(v) for v in data["inference_resolution"].tolist()),
                original_resolution=tuple(int(v) for v in data["original_resolution"].tolist()),
                checkpoint_name=str(data["checkpoint_name"].tolist()),
                config_name=str(data["config_name"].tolist()),
                prompt_mode=str(data["prompt_mode"].tolist()),
                representative_point_mode=str(data["representative_point_mode"].tolist()),
                use_vos_optimized=bool(data["use_vos_optimized"].tolist()),
                backend_mode="subprocess",
            )


def run_sam2_smoke_test(
    *,
    checkpoint_path: str | Path,
    repo_path: str | Path | None = None,
    config_path: str | Path | None = None,
    device: str = "auto",
) -> Sam2SmokeTestResult:
    resolved_repo_path = _resolve_sam2_repo_path(repo_path)
    torch = importlib.import_module("torch")
    with _sam2_import_context(resolved_repo_path):
        build_sam_module = importlib.import_module("sam2.build_sam")
        image_predictor_module = importlib.import_module("sam2.sam2_image_predictor")
        importlib.import_module("sam2.sam2_video_predictor")

    build_sam2 = getattr(build_sam_module, "build_sam2")
    build_sam2_video_predictor = getattr(build_sam_module, "build_sam2_video_predictor")
    SAM2ImagePredictor = getattr(image_predictor_module, "SAM2ImagePredictor")

    resolved_device = _resolve_torch_device(torch, device)
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"SAM 2 checkpoint not found: {resolved_checkpoint}")

    config_identifier = _resolve_config_identifier(config_path, resolved_repo_path, resolved_checkpoint)
    config_resolved = _resolve_config_path(config_path, resolved_repo_path, resolved_checkpoint)
    if config_identifier is None or config_resolved is None:
        raise FileNotFoundError("Could not resolve the SAM 2 config for the smoke test.")

    image = _build_dummy_sam2_frame(width=64, height=64, shift_x=0)
    image_point = np.asarray([[24.0, 24.0]], dtype=np.float32)
    image_labels = np.asarray([1], dtype=np.int32)
    image_box = np.asarray([14.0, 14.0, 38.0, 38.0], dtype=np.float32)

    autocast_context = _make_autocast_context(torch, resolved_device)
    with torch.inference_mode():
        with autocast_context:
            image_model = build_sam2(config_identifier, str(resolved_checkpoint), device=resolved_device)
            image_predictor = SAM2ImagePredictor(image_model)
            image_predictor.set_image(image)
            image_masks, image_ious, _ = image_predictor.predict(
                point_coords=image_point,
                point_labels=image_labels,
                box=image_box,
                multimask_output=False,
            )

    image_masks_np = np.asarray(image_masks)
    image_ious_np = np.asarray(image_ious)

    with TemporaryDirectory(prefix="trackleed_sam2_smoke_") as temp_dir:
        temp_root = Path(temp_dir)
        _write_dummy_video_sequence(temp_root)

        with torch.inference_mode():
            with autocast_context:
                video_predictor = build_sam2_video_predictor(
                    config_identifier,
                    str(resolved_checkpoint),
                    device=resolved_device,
                )
                state = video_predictor.init_state(str(temp_root))
                _, _, prompt_masks = video_predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=1,
                    points=image_point,
                    labels=image_labels,
                    box=image_box,
                )

                propagated_masks = None
                propagated_frame_count = 0
                for _, _, masks in video_predictor.propagate_in_video(
                    state,
                    start_frame_idx=0,
                    max_frame_num_to_track=2,
                ):
                    propagated_masks = masks
                    propagated_frame_count += 1

    prompt_masks_np = prompt_masks.detach().cpu().numpy()
    propagated_masks_np = (
        propagated_masks.detach().cpu().numpy()
        if propagated_masks is not None
        else np.zeros((0,), dtype=np.float32)
    )

    return Sam2SmokeTestResult(
        image_masks_shape=tuple(int(dim) for dim in image_masks_np.shape),
        image_ious_shape=tuple(int(dim) for dim in image_ious_np.shape),
        video_prompt_masks_shape=tuple(int(dim) for dim in prompt_masks_np.shape),
        video_propagated_masks_shape=tuple(int(dim) for dim in propagated_masks_np.shape),
        propagated_frame_count=int(propagated_frame_count),
        first_image_mask_area=int(np.count_nonzero(image_masks_np[0] > 0)) if image_masks_np.size else 0,
        first_video_mask_area=int(np.count_nonzero(prompt_masks_np[0] > 0)) if prompt_masks_np.size else 0,
        device=str(resolved_device),
        checkpoint_name=resolved_checkpoint.name,
        config_name=config_resolved.name,
        cuda_extension_importable=_probe_cuda_extension_available(),
    )


def _discover_checkpoints_in_directory(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.pt"), key=lambda path: path.name.lower())


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoints: list[Path],
    models_dir: Path,
    repo_path: Optional[Path],
) -> Optional[Path]:
    if checkpoint_path is None:
        return select_default_sam2_checkpoint(checkpoints)

    candidate = Path(checkpoint_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    available_by_name = {path.name: path for path in checkpoints}
    if candidate.name in available_by_name:
        return available_by_name[candidate.name]

    candidate_paths = [(models_dir / candidate).resolve()]
    if repo_path is not None:
        candidate_paths.append((repo_path / candidate).resolve())
        candidate_paths.append((repo_path / "checkpoints" / candidate.name).resolve())
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    return candidate_paths[0]


def _resolve_existing_checkpoint_file(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM 2 checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix.lower() != ".pt":
        raise FileNotFoundError(f"SAM 2 checkpoint is not a .pt file: {checkpoint_path}")
    return checkpoint_path


def _resolve_sam2_python_executable(
    python_executable: str | Path | None,
) -> Path:
    if python_executable is not None:
        return Path(python_executable).expanduser().resolve()
    return Path(sys.executable).resolve()


def _resolve_config_identifier(
    config_path: str | Path | None,
    repo_path: Optional[Path],
    checkpoint_path: str | Path | None,
) -> Optional[str]:
    if config_path is not None:
        config_candidate = Path(config_path).expanduser()
        if config_candidate.is_absolute():
            if repo_path is None:
                return None
            resolved_candidate = config_candidate.resolve()
            resolved_repo = repo_path.resolve()
            for base_root in _CONFIG_ROOT_CANDIDATES:
                try:
                    relative_path = resolved_candidate.relative_to((resolved_repo / base_root).resolve())
                except ValueError:
                    continue
                normalized = relative_path.as_posix()
                if normalized.startswith("configs/"):
                    return normalized
            return None
        normalized = config_candidate.as_posix()
        if normalized.startswith("sam2/configs/"):
            return normalized[len("sam2/") :]
        return normalized

    if checkpoint_path is None:
        return None
    return CHECKPOINT_TO_CONFIG.get(Path(checkpoint_path).name)


def _resolve_config_path(
    config_path: str | Path | None,
    repo_path: Optional[Path],
    checkpoint_path: str | Path | None,
) -> Optional[Path]:
    config_identifier = _resolve_config_identifier(config_path, repo_path, checkpoint_path)
    if config_identifier is None or repo_path is None:
        return None

    normalized_identifier = config_identifier.replace("\\", "/")
    if normalized_identifier.startswith("sam2/configs/"):
        normalized_identifier = normalized_identifier[len("sam2/") :]
    relative_identifier = Path(normalized_identifier)

    candidate_paths: list[Path] = []
    for base_root in _CONFIG_ROOT_CANDIDATES:
        candidate_paths.append((repo_path / base_root / relative_identifier).resolve())

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    return candidate_paths[-1]


def _resolve_sam2_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidate_paths: list[Path] = []
    if repo_path is not None:
        candidate_paths.append(Path(repo_path).expanduser())
    else:
        candidate_paths.extend(DEFAULT_SAM2_REPO_CANDIDATES)

    for candidate in candidate_paths:
        normalized = _normalize_repo_candidate(candidate)
        if normalized is not None:
            return normalized
    return None


def _normalize_repo_candidate(candidate: Path) -> Optional[Path]:
    candidate = candidate.resolve()
    if (candidate / "sam2" / "build_sam.py").exists() and (candidate / "setup.py").exists():
        return candidate
    return None


def _resolve_torch_device(torch_module, requested_device: str):
    device = str(requested_device or "auto").strip().lower()
    if device in {"auto", "gpu"}:
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if device == "cuda" and not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    return torch_module.device(device)


def _normalize_path_key(path_value: str | os.PathLike[str] | None) -> str:
    if path_value is None:
        return ""
    return os.path.normcase(os.path.normpath(str(path_value)))


@contextmanager
def _sam2_import_context(repo_path: Optional[Path]):
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

    sys.path[:] = [repo_root_str, *filtered_sys_path]
    for module_name in list(sys.modules):
        if module_name == "sam2" or module_name.startswith("sam2."):
            sys.modules.pop(module_name, None)
    try:
        yield
    finally:
        sys.path[:] = original_sys_path


def _make_autocast_context(torch_module, device) -> Any:
    if str(device).startswith("cuda"):
        return torch_module.autocast("cuda", dtype=torch_module.bfloat16)
    return nullcontext()


def _configure_cuda_runtime(torch_module, device) -> None:
    if not str(device).startswith("cuda") or not getattr(torch_module, "cuda", None):
        return
    try:
        index = getattr(device, "index", None)
        if index is not None:
            torch_module.cuda.set_device(index)
        major = torch_module.cuda.get_device_properties(device).major
        if major >= 8:
            torch_module.backends.cuda.matmul.allow_tf32 = True
            torch_module.backends.cudnn.allow_tf32 = True
    except Exception:
        return


def _prepare_frames_rgb(frames: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    frames_np = np.asarray(frames)
    if frames_np.ndim == 3:
        frames_np = np.stack([frames_np] * 3, axis=-1)
    elif frames_np.ndim == 4 and frames_np.shape[-1] == 1:
        frames_np = np.repeat(frames_np, 3, axis=-1)
    elif frames_np.ndim == 4 and frames_np.shape[-1] >= 3:
        frames_np = frames_np[..., :3]
    else:
        raise ValueError(f"Unsupported SAM 2 frame shape: {frames_np.shape}")

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


def _normalize_optional_resolution(
    resolution: int | tuple[int, int] | None,
    default_resolution: tuple[int, int],
) -> tuple[int, int]:
    if resolution is None:
        return tuple(int(v) for v in default_resolution)
    if isinstance(resolution, int):
        return (int(resolution), int(resolution))
    if len(resolution) != 2:
        raise ValueError(f"Invalid SAM 2 resolution: {resolution}")
    return (int(resolution[0]), int(resolution[1]))


def _resize_video_numpy(frames: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target[0]), int(target[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid SAM 2 inference resolution: {(target_h, target_w)}")
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


def _extract_query_seed_frames(query_points_tyx: np.ndarray, total_frames: int) -> np.ndarray:
    seed_frames = np.rint(np.asarray(query_points_tyx[:, 0], dtype=np.float32)).astype(np.int32)
    if np.any(seed_frames < 0) or np.any(seed_frames >= int(total_frames)):
        raise ValueError(
            f"SAM 2 query seed frames must be in [0, {int(total_frames) - 1}], got {seed_frames.tolist()}."
        )
    return seed_frames


def _normalize_prompt_mode(prompt_mode: str) -> str:
    mode = str(prompt_mode or "point+bbox").strip().lower().replace("_", "").replace("-", "")
    if mode in {"pointbbox", "point+box", "pointbox", "point+bbox"}:
        return "point+bbox"
    if mode in {"bbox", "box", "bboxonly", "boxonly"}:
        return "bbox"
    if mode in {"point", "pointonly"}:
        return "point"
    raise ValueError(f"Unsupported SAM 2 prompt mode: {prompt_mode}")


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
    raise ValueError(f"Unsupported SAM 2 representative point mode: {mode}")


def _build_sam2_prompt_boxes_xyxy(
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


def _build_dummy_sam2_frame(
    *,
    width: int,
    height: int,
    shift_x: int,
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = 14 + int(shift_x)
    y0 = 14
    x1 = min(width, x0 + 20)
    y1 = min(height, y0 + 20)
    image[y0:y1, x0:x1, :] = 255
    return image


def _write_dummy_video_sequence(output_dir: Path) -> None:
    pil_image_module = importlib.import_module("PIL.Image")
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, shift_x in enumerate((0, 2)):
        frame = _build_dummy_sam2_frame(width=64, height=64, shift_x=shift_x)
        pil_image_module.fromarray(frame).save(output_dir / f"{idx:05d}.jpg", format="JPEG")


def _write_video_sequence(output_dir: Path, frames: np.ndarray) -> None:
    pil_image_module = importlib.import_module("PIL.Image")
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(np.asarray(frames)):
        pil_image_module.fromarray(frame).save(output_dir / f"{idx:05d}.jpg", format="JPEG")


def _record_sam2_frame_outputs(
    *,
    video_res_masks,
    obj_ids,
    obj_id_to_query_idx: dict[int, int],
    frame_gray: np.ndarray,
    frame_index: int,
    tracks_xy: np.ndarray,
    visible_mask: np.ndarray,
    mask_areas: np.ndarray,
    mask_bboxes_xyxy: np.ndarray,
    mask_scores: np.ndarray,
    mask_component_counts: np.ndarray,
    representative_point_mode: str,
    mask_threshold: float,
    min_mask_area: float,
    max_mask_area: float | None,
    inference_resolution: tuple[int, int],
    original_resolution: tuple[int, int],
) -> None:
    masks_np = video_res_masks.detach().cpu().numpy()
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    elif masks_np.ndim != 3:
        raise RuntimeError(f"Unexpected SAM 2 mask tensor shape: {masks_np.shape}")

    area_scale = (
        float(original_resolution[0]) / float(inference_resolution[0])
    ) * (
        float(original_resolution[1]) / float(inference_resolution[1])
    )

    for obj_offset, obj_id in enumerate(obj_ids):
        query_index = obj_id_to_query_idx.get(int(obj_id))
        if query_index is None:
            continue

        mask_logits = np.asarray(masks_np[obj_offset], dtype=np.float32)
        mask_bool = _mask_logits_to_bool_mask(mask_logits, threshold=float(mask_threshold))
        area_native = float(np.count_nonzero(mask_bool))
        area_estimate = area_native * area_scale
        bbox_xyxy = _mask_to_bbox_xyxy(mask_bool)
        score = _mask_score_from_logits(mask_logits, mask_bool)
        component_count = _mask_component_count(mask_bool)

        mask_areas[query_index, frame_index] = float(area_estimate)
        mask_scores[query_index, frame_index] = float(score)
        mask_component_counts[query_index, frame_index] = int(component_count)
        if bbox_xyxy is not None:
            mask_bboxes_xyxy[query_index, frame_index, :] = np.asarray(bbox_xyxy, dtype=np.float32)

        is_valid = area_native > 0
        is_valid = is_valid and area_estimate >= float(min_mask_area)
        if max_mask_area is not None:
            is_valid = is_valid and area_estimate <= float(max_mask_area)
        if not is_valid:
            continue

        representative_point = _mask_to_representative_point(
            mask_bool,
            frame_gray=frame_gray,
            mode=representative_point_mode,
        )
        if representative_point is None:
            continue

        x, y = representative_point
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        tracks_xy[query_index, frame_index, 0] = float(x)
        tracks_xy[query_index, frame_index, 1] = float(y)
        visible_mask[query_index, frame_index] = True


def _mask_logits_to_bool_mask(mask_logits: np.ndarray, threshold: float) -> np.ndarray:
    probabilities = _sigmoid(mask_logits)
    return probabilities >= float(threshold)


def _mask_score_from_logits(mask_logits: np.ndarray, mask_bool: np.ndarray) -> float:
    if not np.any(mask_bool):
        return 0.0
    probabilities = _sigmoid(mask_logits)
    return float(np.mean(probabilities[mask_bool]))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _mask_to_bbox_xyxy(mask_bool: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return None
    x0 = float(np.min(xs))
    y0 = float(np.min(ys))
    x1 = float(np.max(xs) + 1)
    y1 = float(np.max(ys) + 1)
    return (x0, y0, x1, y1)


def _mask_component_count(mask_bool: np.ndarray) -> int:
    mask_u8 = np.asarray(mask_bool, dtype=np.uint8)
    if mask_u8.size == 0 or not np.any(mask_u8):
        return 0
    num_labels, _ = cv2.connectedComponents(mask_u8, connectivity=8)
    return max(0, int(num_labels) - 1)


def _mask_to_representative_point(
    mask_bool: np.ndarray,
    *,
    frame_gray: np.ndarray,
    mode: str,
) -> tuple[float, float] | None:
    bbox = _mask_to_bbox_xyxy(mask_bool)
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

    raise ValueError(f"Unsupported SAM 2 representative point mode: {mode}")


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
    bbox_xyxy: tuple[float, float, float, float],
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


def _probe_cuda_extension_available() -> bool:
    try:
        importlib.import_module("sam2._C")
        return True
    except Exception:
        return False


def _append_sam2_import_hints(status: Sam2EnvironmentStatus, exc: Exception) -> None:
    message = str(exc).lower()
    if "hydra" in message or "omegaconf" in message:
        status.hints.append("Reinstall the repo via `pip install -e \".[notebooks]\"`.")
    elif "cannot import name '_c' from 'sam2'" in message:
        status.hints.append(
            "The custom CUDA extension is missing. This is usually OK on Windows if you installed with `SAM2_BUILD_CUDA=0`."
        )
    elif "missingconfigexception" in message:
        status.hints.append(
            "Install the repo with `pip install -e .` so the Hydra config package is discoverable."
        )
    else:
        status.hints.append(
            "Verify that the dedicated `sam2` env has the local repo installed and uses `torch>=2.5.1`."
        )
