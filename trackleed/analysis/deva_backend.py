"""DEVA seeded VOS backend helpers.

This module intentionally keeps DEVA isolated behind a subprocess-friendly
contract. Trackleed can prepare frames and seed masks in its main environment,
then execute the actual DEVA evaluation path in the dedicated `deva` Conda env.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Iterator, Optional

import numpy as np


DEFAULT_DEVA_MODELS_DIR = Path("tracking_models") / "deva"
DEFAULT_DEVA_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "deva",
    Path(__file__).resolve().parents[3] / "Tracking-Anything-with-DEVA",
    Path(__file__).resolve().parents[2] / "deva",
)
DEFAULT_DEVA_SUBPROCESS_SCRIPT = Path("scripts") / "run_deva_subprocess.py"
DEFAULT_DEVA_CHECKPOINT_NAME = "DEVA-propagation.pth"
DEFAULT_DEVA_SEQUENCE_NAME = "trackleed_deva"
DEFAULT_DEVA_INFERENCE_SIZE = 480
DEFAULT_DEVA_CHUNK_SIZE = 4


@dataclass
class DevaEnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    available_checkpoints: list[Path] = field(default_factory=list)
    selected_checkpoint: Optional[Path] = None
    checkpoint_exists: bool = False
    checkpoint_size_bytes: Optional[int] = None
    example_vos_path: Optional[Path] = None
    example_vos_ready: bool = False
    torch_importable: bool = False
    torch_version: Optional[str] = None
    torch_cuda_version: Optional[str] = None
    cuda_available: bool = False
    torchvision_importable: bool = False
    pillow_importable: bool = False
    numpy_importable: bool = False
    tqdm_importable: bool = False
    deva_importable: bool = False
    deva_module_path: Optional[str] = None
    deva_network_importable: bool = False
    deva_inference_core_importable: bool = False
    deva_general_vos_dataset_importable: bool = False
    groundingdino_importable: bool = False
    segment_anything_importable: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return (
            self.repo_path is not None
            and self.checkpoint_exists
            and self.example_vos_ready
            and self.torch_importable
            and self.cuda_available
            and self.torchvision_importable
            and self.pillow_importable
            and self.numpy_importable
            and self.tqdm_importable
            and self.deva_importable
            and self.deva_network_importable
            and self.deva_inference_core_importable
            and self.deva_general_vos_dataset_importable
        )


@dataclass
class DevaRuntimeSettings:
    checkpoint_path: str | Path
    repo_path: str | Path | None = None
    python_executable: str | Path | None = None
    device: str = "auto"
    backend_mode: str = "subprocess"
    representative_point_mode: str = "mask_centroid"
    inference_size: int | None = DEFAULT_DEVA_INFERENCE_SIZE
    initial_bbox_padding_px: float = 0.0
    initial_point_radius_px: float = 4.0
    amp: bool = True
    chunk_size: int = DEFAULT_DEVA_CHUNK_SIZE
    subprocess_timeout_sec: float = 1800.0


@dataclass
class DevaOfflineResult:
    masks: np.ndarray
    tracks_xy: np.ndarray
    visible_mask: np.ndarray
    mask_areas: np.ndarray
    mask_bboxes_xyxy: np.ndarray
    mask_scores: np.ndarray
    inference_size: int
    original_resolution: tuple[int, int]
    checkpoint_name: str
    representative_point_mode: str
    initial_mask_mode: str
    backend_mode: str


def discover_deva_checkpoints(
    models_dir: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_DEVA_MODELS_DIR
    discovered: dict[str, Path] = {}
    for candidate in _discover_checkpoints_in_directory(directory):
        discovered[str(candidate.resolve())] = candidate.resolve()

    return sorted(discovered.values(), key=lambda path: path.name.lower())


def select_default_deva_checkpoint(checkpoints: list[Path]) -> Optional[Path]:
    if not checkpoints:
        return None
    by_name = {path.name: path for path in checkpoints}
    if DEFAULT_DEVA_CHECKPOINT_NAME in by_name:
        return by_name[DEFAULT_DEVA_CHECKPOINT_NAME]
    return checkpoints[0]


def probe_deva_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> DevaEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_DEVA_MODELS_DIR
    directory = directory.resolve()
    resolved_repo_path = _resolve_deva_repo_path(repo_path)
    checkpoints = discover_deva_checkpoints(directory, resolved_repo_path)
    selected_checkpoint = _resolve_checkpoint_path(
        checkpoint_path,
        checkpoints,
        directory,
        resolved_repo_path,
    )
    example_vos_path = _resolve_example_vos_path(resolved_repo_path)

    status = DevaEnvironmentStatus(
        models_dir=directory,
        repo_path=resolved_repo_path,
        available_checkpoints=checkpoints,
        selected_checkpoint=selected_checkpoint,
        checkpoint_exists=bool(selected_checkpoint and selected_checkpoint.exists()),
        checkpoint_size_bytes=(
            selected_checkpoint.stat().st_size
            if selected_checkpoint is not None and selected_checkpoint.exists()
            else None
        ),
        example_vos_path=example_vos_path,
        example_vos_ready=bool(example_vos_path and _is_generic_vos_dataset(example_vos_path)),
    )

    if repo_path and resolved_repo_path is None:
        status.errors.append(f"Could not resolve DEVA repo from: {repo_path}")
    elif resolved_repo_path is None:
        status.errors.append(
            "Could not resolve DEVA repo. Pass a path pointing at the local DEVA clone."
        )

    if not directory.exists():
        status.warnings.append(
            f"Checkpoint directory does not exist yet: {directory}. "
            "Place DEVA checkpoints inside tracking_models/deva."
        )

    if selected_checkpoint is None:
        status.errors.append(
            "No DEVA checkpoint detected in tracking_models/deva."
        )
    elif checkpoint_path and not status.checkpoint_exists:
        status.errors.append(f"Selected checkpoint not found: {selected_checkpoint}")

    if example_vos_path is None:
        status.warnings.append("DEVA example/vos dataset was not found.")
    elif not status.example_vos_ready:
        status.warnings.append(f"DEVA example/vos dataset is incomplete: {example_vos_path}")

    try:
        torch = importlib.import_module("torch")
        status.torch_importable = True
        status.torch_version = getattr(torch, "__version__", None)
        status.torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        status.cuda_available = bool(torch.cuda.is_available())
        if not status.cuda_available:
            status.errors.append(
                "CUDA is not available. The official DEVA VOS evaluation path calls CUDA directly."
            )
    except Exception as error:
        status.errors.append(f"torch import failed: {error}")

    status.torchvision_importable = _try_import("torchvision", status.errors, "torchvision")
    status.pillow_importable = _try_import("PIL.Image", status.errors, "Pillow")
    status.numpy_importable = _try_import("numpy", status.errors, "NumPy")
    status.tqdm_importable = _try_import("tqdm", status.errors, "tqdm")

    with _deva_import_context(resolved_repo_path):
        deva_module = _try_import_module("deva")
        status.deva_importable = deva_module is not None
        if deva_module is not None:
            status.deva_module_path = getattr(deva_module, "__file__", None)
        else:
            status.errors.append("DEVA import failed: No module named 'deva'")

        status.deva_network_importable = _try_import(
            "deva.model.network",
            status.errors,
            "DEVA model network",
        )
        status.deva_inference_core_importable = _try_import(
            "deva.inference.inference_core",
            status.errors,
            "DEVA inference core",
        )
        status.deva_general_vos_dataset_importable = _try_import(
            "deva.inference.data.vos_test_datasets",
            status.errors,
            "DEVA VOS dataset loader",
        )
        status.groundingdino_importable = _try_import(
            "groundingdino.util.inference",
            [],
            "GroundingDINO",
        )
        status.segment_anything_importable = _try_import(
            "segment_anything",
            [],
            "Segment Anything",
        )

    if not status.groundingdino_importable:
        status.warnings.append(
            "GroundingDINO is not importable. This is OK for seeded VOS tracking, "
            "but text-prompted/automatic DEVA demos will need it."
        )
    if not status.segment_anything_importable:
        status.warnings.append(
            "segment_anything is not importable. This is OK for seeded VOS tracking, "
            "but automatic/text-prompted DEVA demos will need it."
        )

    if not status.is_ready:
        status.hints.append(
            "Run DEVA from the dedicated `deva` Conda env and set the subprocess Python executable."
        )
        status.hints.append(
            "Trackleed's seeded DEVA path requires DEVA core and DEVA-propagation.pth; "
            "GroundingDINO/SAM stay optional."
        )

    return status


def format_deva_environment_report(status: DevaEnvironmentStatus) -> str:
    lines = [
        "DEVA environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Selected checkpoint: {status.selected_checkpoint or 'None'}",
        f"- Selected checkpoint exists: {'yes' if status.checkpoint_exists else 'no'}",
        f"- Selected checkpoint size MB: {_format_size_mb(status.checkpoint_size_bytes)}",
        f"- Available checkpoints: {_format_checkpoint_paths(status.available_checkpoints)}",
        f"- Example VOS path: {status.example_vos_path or 'None'}",
        f"- Example VOS ready: {'yes' if status.example_vos_ready else 'no'}",
        f"- PyTorch import: {'ok' if status.torch_importable else 'failed'}",
        f"- PyTorch version: {status.torch_version or 'unknown'}",
        f"- PyTorch CUDA version: {status.torch_cuda_version or 'unknown'}",
        f"- CUDA available: {'yes' if status.cuda_available else 'no'}",
        f"- torchvision import: {'ok' if status.torchvision_importable else 'failed'}",
        f"- Pillow import: {'ok' if status.pillow_importable else 'failed'}",
        f"- NumPy import: {'ok' if status.numpy_importable else 'failed'}",
        f"- tqdm import: {'ok' if status.tqdm_importable else 'failed'}",
        f"- DEVA import: {'ok' if status.deva_importable else 'failed'}",
        f"- DEVA module: {status.deva_module_path or 'unknown'}",
        f"- DEVA network import: {'ok' if status.deva_network_importable else 'failed'}",
        f"- DEVA inference core import: {'ok' if status.deva_inference_core_importable else 'failed'}",
        "- DEVA generic VOS dataset import: "
        + ("ok" if status.deva_general_vos_dataset_importable else "failed"),
        f"- GroundingDINO import: {'ok' if status.groundingdino_importable else 'missing'}",
        f"- segment_anything import: {'ok' if status.segment_anything_importable else 'missing'}",
    ]

    if status.errors:
        lines.append("- Errors:")
        lines.extend(f"  - {error}" for error in status.errors)
    if status.warnings:
        lines.append("- Warnings:")
        lines.extend(f"  - {warning}" for warning in status.warnings)
    if status.hints:
        lines.append("- Hints:")
        lines.extend(f"  - {hint}" for hint in status.hints)
    return "\n".join(lines)


def run_deva_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray | None,
    settings: DevaRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
    query_object_ids: np.ndarray | None = None,
    query_initial_masks: np.ndarray | None = None,
) -> DevaOfflineResult:
    """Run DEVA on a short seeded VOS sequence.

    The MVP contract supports one seed frame at local frame 0. A seed can be
    provided as full-frame boolean masks, bbox masks, or point disks.
    """

    frames_rgb, original_resolution = _prepare_frames_rgb(frames)
    num_queries = _infer_num_queries(
        query_points_tyx=query_points_tyx,
        query_boxes_xyxy=query_boxes_xyxy,
        query_initial_masks=query_initial_masks,
    )
    if num_queries <= 0:
        raise ValueError("DEVA requires at least one query point, bbox, or initial mask.")
    if num_queries > 255:
        raise ValueError("DEVA seed masks are PNG label masks and support at most 255 objects.")

    query_points_np = _normalize_query_points_tyx(query_points_tyx, num_queries)
    boxes_np = _normalize_query_boxes_xyxy(query_boxes_xyxy, num_queries)
    masks_np = _normalize_query_initial_masks(
        query_initial_masks,
        num_queries=num_queries,
        frame_shape=original_resolution,
    )
    object_ids_np = _normalize_query_object_ids(query_object_ids, num_queries)
    representative_point_mode = _normalize_representative_point_mode(
        settings.representative_point_mode
    )
    initial_label_mask, initial_mask_mode = _build_deva_initial_label_mask(
        frame_shape=original_resolution,
        query_points_tyx=query_points_np,
        query_boxes_xyxy=boxes_np,
        query_initial_masks=masks_np,
        object_ids=object_ids_np,
        initial_bbox_padding_px=float(settings.initial_bbox_padding_px),
        initial_point_radius_px=float(settings.initial_point_radius_px),
    )

    backend_mode = str(settings.backend_mode or "subprocess").strip().lower()
    if backend_mode == "auto":
        backend_mode = "subprocess"
    if backend_mode == "direct":
        return _run_deva_eval_vos_offline(
            frames_rgb,
            initial_label_mask,
            settings,
            representative_point_mode=representative_point_mode,
            initial_mask_mode=initial_mask_mode,
            backend_mode="direct",
        )
    if backend_mode != "subprocess":
        raise ValueError(
            f"Unsupported DEVA backend mode: {settings.backend_mode}. "
            "Expected `subprocess` or `direct`."
        )
    return _run_deva_subprocess_offline(
        frames_rgb,
        query_points_np,
        settings,
        query_boxes_xyxy=boxes_np,
        query_object_ids=object_ids_np,
        query_initial_masks=masks_np,
    )


def _run_deva_eval_vos_offline(
    frames: np.ndarray,
    initial_label_mask: np.ndarray,
    settings: DevaRuntimeSettings,
    *,
    representative_point_mode: str,
    initial_mask_mode: str,
    backend_mode: str,
) -> DevaOfflineResult:
    resolved_repo_path = _resolve_deva_repo_path(settings.repo_path)
    if resolved_repo_path is None:
        raise FileNotFoundError(
            "Could not resolve the DEVA repo. Pass repo_path pointing at the local DEVA clone."
        )
    checkpoint_file = _resolve_existing_checkpoint_file(
        _resolve_checkpoint_path(
            settings.checkpoint_path,
            discover_deva_checkpoints(DEFAULT_DEVA_MODELS_DIR, resolved_repo_path),
            DEFAULT_DEVA_MODELS_DIR,
            resolved_repo_path,
        )
    )
    _validate_deva_device(settings.device)
    inference_size = _normalize_inference_size(settings.inference_size)
    chunk_size = int(max(1, settings.chunk_size))

    with TemporaryDirectory(prefix="trackleed_deva_runtime_") as temp_dir:
        temp_root = Path(temp_dir)
        dataset_root = temp_root / "dataset"
        output_dir = temp_root / "output"
        _write_deva_generic_vos_dataset(
            dataset_root,
            frames,
            initial_label_mask,
            sequence_name=DEFAULT_DEVA_SEQUENCE_NAME,
        )
        command = [
            sys.executable,
            str(resolved_repo_path / "evaluation" / "eval_vos.py"),
            "--dataset",
            "G",
            "--generic_path",
            str(dataset_root),
            "--model",
            str(checkpoint_file),
            "--output",
            str(output_dir),
            "--save_all",
            "--size",
            str(inference_size),
            "--chunk_size",
            str(chunk_size),
        ]
        if bool(settings.amp):
            command.append("--amp")

        completed = subprocess.run(
            command,
            cwd=str(resolved_repo_path),
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
                "DEVA eval_vos inference failed:\n"
                + json.dumps(payload, indent=2, ensure_ascii=False)
            )

        class_masks = _read_deva_output_label_masks(
            output_dir / DEFAULT_DEVA_SEQUENCE_NAME,
            expected_frame_count=int(frames.shape[0]),
        )

    return _build_deva_result_from_class_masks(
        class_masks,
        frames,
        checkpoint_name=checkpoint_file.name,
        representative_point_mode=representative_point_mode,
        initial_mask_mode=initial_mask_mode,
        inference_size=inference_size,
        backend_mode=backend_mode,
    )


def _run_deva_subprocess_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray | None,
    settings: DevaRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
    query_object_ids: np.ndarray | None = None,
    query_initial_masks: np.ndarray | None = None,
) -> DevaOfflineResult:
    python_executable = _resolve_deva_python_executable(settings.python_executable)
    script_path = (Path(__file__).resolve().parents[2] / DEFAULT_DEVA_SUBPROCESS_SCRIPT).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"DEVA subprocess script not found: {script_path}")

    with TemporaryDirectory(prefix="trackleed_deva_") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "deva_input.npz"
        output_path = temp_root / "deva_output.npz"
        np.savez_compressed(
            input_path,
            frames=np.asarray(frames),
            query_points_tyx=(
                np.asarray(query_points_tyx, dtype=np.float32)
                if query_points_tyx is not None
                else np.zeros((0, 3), dtype=np.float32)
            ),
            has_query_points=np.asarray(query_points_tyx is not None, dtype=bool),
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
            query_initial_masks=(
                np.asarray(query_initial_masks, dtype=bool)
                if query_initial_masks is not None
                else np.zeros((0, frames.shape[1], frames.shape[2]), dtype=bool)
            ),
            has_query_initial_masks=np.asarray(query_initial_masks is not None, dtype=bool),
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
            "--initial-point-radius-px",
            str(float(settings.initial_point_radius_px)),
            "--chunk-size",
            str(int(settings.chunk_size)),
            "--subprocess-timeout-sec",
            str(float(settings.subprocess_timeout_sec)),
        ]
        if settings.repo_path is not None:
            command.extend(["--repo-path", str(settings.repo_path)])
        command.extend(["--inference-size", str(_normalize_inference_size(settings.inference_size))])
        if not bool(settings.amp):
            command.append("--no-amp")

        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            check=False,
            timeout=float(settings.subprocess_timeout_sec),
            env=_build_windows_conda_subprocess_env(python_executable),
        )
        if completed.returncode != 0:
            payload = {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
            raise RuntimeError(
                "DEVA subprocess inference failed:\n"
                + json.dumps(payload, indent=2, ensure_ascii=False)
            )
        if not output_path.exists():
            raise FileNotFoundError(f"DEVA subprocess did not create output: {output_path}")

        with np.load(output_path, allow_pickle=False) as data:
            return DevaOfflineResult(
                masks=np.asarray(data["masks"], dtype=bool),
                tracks_xy=np.asarray(data["tracks_xy"], dtype=np.float32),
                visible_mask=np.asarray(data["visible_mask"], dtype=bool),
                mask_areas=np.asarray(data["mask_areas"], dtype=np.float32),
                mask_bboxes_xyxy=np.asarray(data["mask_bboxes_xyxy"], dtype=np.float32),
                mask_scores=np.asarray(data["mask_scores"], dtype=np.float32),
                inference_size=int(data["inference_size"].tolist()),
                original_resolution=tuple(int(v) for v in data["original_resolution"].tolist()),
                checkpoint_name=str(data["checkpoint_name"].tolist()),
                representative_point_mode=str(data["representative_point_mode"].tolist()),
                initial_mask_mode=str(data["initial_mask_mode"].tolist()),
                backend_mode="subprocess",
            )


def _build_deva_result_from_class_masks(
    class_masks: np.ndarray,
    frames: np.ndarray,
    *,
    checkpoint_name: str,
    representative_point_mode: str,
    initial_mask_mode: str,
    inference_size: int,
    backend_mode: str,
) -> DevaOfflineResult:
    class_masks_np = np.asarray(class_masks, dtype=np.int32)
    if class_masks_np.ndim != 3:
        raise ValueError(f"Expected DEVA class masks with shape [T,H,W], got {class_masks_np.shape}.")
    prepared_frames, original_resolution = _prepare_frames_rgb(frames)
    if class_masks_np.shape[0] != prepared_frames.shape[0]:
        raise ValueError(
            f"Expected {prepared_frames.shape[0]} DEVA masks, got {class_masks_np.shape[0]}."
        )
    if class_masks_np.shape[1:3] != prepared_frames.shape[1:3]:
        raise ValueError(
            "DEVA output mask shape does not match original frame shape: "
            f"{class_masks_np.shape[1:3]} vs {prepared_frames.shape[1:3]}."
        )

    labels = sorted(int(label) for label in np.unique(class_masks_np) if int(label) > 0)
    num_objects = len(labels)
    num_frames, height, width = class_masks_np.shape
    masks = np.zeros((num_objects, num_frames, height, width), dtype=bool)
    tracks_xy = np.full((num_objects, num_frames, 2), np.nan, dtype=np.float32)
    visible_mask = np.zeros((num_objects, num_frames), dtype=bool)
    mask_areas = np.zeros((num_objects, num_frames), dtype=np.float32)
    mask_bboxes_xyxy = np.full((num_objects, num_frames, 4), np.nan, dtype=np.float32)
    mask_scores = np.zeros((num_objects, num_frames), dtype=np.float32)
    gray_frames = _video_to_gray_float(prepared_frames)

    for object_index, label in enumerate(labels):
        for frame_index in range(num_frames):
            mask_bool = class_masks_np[frame_index] == label
            if not np.any(mask_bool):
                continue
            masks[object_index, frame_index] = mask_bool
            visible_mask[object_index, frame_index] = True
            area = int(np.count_nonzero(mask_bool))
            mask_areas[object_index, frame_index] = float(area)
            mask_scores[object_index, frame_index] = 1.0
            bbox = _mask_bbox_xyxy(mask_bool)
            if bbox is not None:
                mask_bboxes_xyxy[object_index, frame_index] = np.asarray(bbox, dtype=np.float32)
            point = _mask_to_representative_point(
                mask_bool,
                frame_gray=gray_frames[frame_index],
                mode=representative_point_mode,
            )
            if point is not None:
                tracks_xy[object_index, frame_index, 0] = float(point[0])
                tracks_xy[object_index, frame_index, 1] = float(point[1])

    return DevaOfflineResult(
        masks=masks,
        tracks_xy=tracks_xy,
        visible_mask=visible_mask,
        mask_areas=mask_areas,
        mask_bboxes_xyxy=mask_bboxes_xyxy,
        mask_scores=mask_scores,
        inference_size=int(inference_size),
        original_resolution=original_resolution,
        checkpoint_name=str(checkpoint_name),
        representative_point_mode=representative_point_mode,
        initial_mask_mode=initial_mask_mode,
        backend_mode=str(backend_mode),
    )


def _build_deva_initial_label_mask(
    *,
    frame_shape: tuple[int, int],
    query_points_tyx: np.ndarray | None,
    query_boxes_xyxy: np.ndarray | None,
    query_initial_masks: np.ndarray | None,
    object_ids: np.ndarray,
    initial_bbox_padding_px: float,
    initial_point_radius_px: float,
) -> tuple[np.ndarray, str]:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid DEVA frame shape: {frame_shape}")

    label_mask = np.zeros((height, width), dtype=np.uint8)
    class_ids = np.arange(1, len(object_ids) + 1, dtype=np.uint8)

    if query_initial_masks is not None:
        for class_id, mask in zip(class_ids.tolist(), query_initial_masks):
            mask_bool = np.asarray(mask, dtype=bool)
            label_mask[mask_bool & (label_mask == 0)] = int(class_id)
        if not np.any(label_mask):
            raise ValueError("DEVA initial masks are empty.")
        return label_mask, "initial_masks"

    if query_boxes_xyxy is not None:
        for class_id, box in zip(class_ids.tolist(), query_boxes_xyxy):
            x0, y0, x1, y1 = _clip_box_xyxy(
                box,
                frame_shape=(height, width),
                padding_px=float(initial_bbox_padding_px),
            )
            if x1 <= x0 or y1 <= y0:
                continue
            region = label_mask[y0:y1, x0:x1]
            region[region == 0] = int(class_id)
        if not np.any(label_mask):
            raise ValueError("DEVA could not build any non-empty bbox seed masks.")
        return label_mask, "bbox"

    if query_points_tyx is None:
        raise ValueError("DEVA requires query points, query boxes, or query initial masks.")
    seed_frames = np.rint(query_points_tyx[:, 0]).astype(np.int32)
    if np.any(seed_frames != 0):
        raise ValueError("DEVA MVP supports only seed prompts on local frame 0.")
    radius = max(1.0, float(initial_point_radius_px))
    yy, xx = np.ogrid[:height, :width]
    for class_id, query in zip(class_ids.tolist(), query_points_tyx):
        center_y = float(query[1])
        center_x = float(query[2])
        mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2) <= radius**2
        label_mask[mask & (label_mask == 0)] = int(class_id)
    if not np.any(label_mask):
        raise ValueError("DEVA could not build any non-empty point seed masks.")
    return label_mask, "point_disk"


def _write_deva_generic_vos_dataset(
    dataset_root: Path,
    frames: np.ndarray,
    initial_label_mask: np.ndarray,
    *,
    sequence_name: str = DEFAULT_DEVA_SEQUENCE_NAME,
) -> None:
    Image = _import_pil_image()
    frames_np, original_resolution = _prepare_frames_rgb(frames)
    if tuple(initial_label_mask.shape) != tuple(original_resolution):
        raise ValueError(
            f"Initial label mask shape {initial_label_mask.shape} does not match frames {original_resolution}."
        )
    image_dir = dataset_root / "JPEGImages" / sequence_name
    mask_dir = dataset_root / "Annotations" / sequence_name
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, frame in enumerate(frames_np):
        image_path = image_dir / f"{frame_index:08d}.jpg"
        Image.fromarray(frame).save(str(image_path), quality=95)

    mask = np.asarray(initial_label_mask, dtype=np.uint8)
    Image.fromarray(mask, mode="P").save(str(mask_dir / "00000000.png"))


def _read_deva_output_label_masks(output_sequence_dir: Path, *, expected_frame_count: int) -> np.ndarray:
    Image = _import_pil_image()
    mask_paths = sorted(output_sequence_dir.glob("*.png"))
    if len(mask_paths) != int(expected_frame_count):
        raise RuntimeError(
            f"Expected {expected_frame_count} DEVA output masks in {output_sequence_dir}, "
            f"got {len(mask_paths)}."
        )
    masks = []
    for mask_path in mask_paths:
        arr = np.asarray(Image.open(str(mask_path)), dtype=np.int32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        masks.append(arr)
    return np.stack(masks, axis=0)


def _infer_num_queries(
    *,
    query_points_tyx: np.ndarray | None,
    query_boxes_xyxy: np.ndarray | None,
    query_initial_masks: np.ndarray | None,
) -> int:
    candidates: list[int] = []
    if query_points_tyx is not None:
        query_np = np.asarray(query_points_tyx)
        if query_np.ndim != 2 or query_np.shape[1] != 3:
            raise ValueError("query_points_tyx must have shape [num_queries, 3] in [t, y, x] format.")
        candidates.append(int(query_np.shape[0]))
    if query_boxes_xyxy is not None:
        boxes_np = np.asarray(query_boxes_xyxy)
        if boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
            raise ValueError("query_boxes_xyxy must have shape [num_queries, 4].")
        candidates.append(int(boxes_np.shape[0]))
    if query_initial_masks is not None:
        masks_np = np.asarray(query_initial_masks)
        if masks_np.ndim != 3:
            raise ValueError("query_initial_masks must have shape [num_queries, height, width].")
        candidates.append(int(masks_np.shape[0]))
    if not candidates:
        return 0
    if len(set(candidates)) != 1:
        raise ValueError(f"DEVA query inputs disagree on object count: {candidates}.")
    return candidates[0]


def _normalize_query_points_tyx(
    query_points_tyx: np.ndarray | None,
    num_queries: int,
) -> np.ndarray | None:
    if query_points_tyx is None:
        return None
    query_np = np.asarray(query_points_tyx, dtype=np.float32)
    if query_np.shape != (int(num_queries), 3):
        raise ValueError(f"query_points_tyx must have shape {(num_queries, 3)}, got {query_np.shape}.")
    return query_np


def _normalize_query_boxes_xyxy(
    query_boxes_xyxy: np.ndarray | None,
    num_queries: int,
) -> np.ndarray | None:
    if query_boxes_xyxy is None:
        return None
    boxes_np = np.asarray(query_boxes_xyxy, dtype=np.float32)
    if boxes_np.shape != (int(num_queries), 4):
        raise ValueError(f"query_boxes_xyxy must have shape {(num_queries, 4)}, got {boxes_np.shape}.")
    return boxes_np


def _normalize_query_initial_masks(
    query_initial_masks: np.ndarray | None,
    *,
    num_queries: int,
    frame_shape: tuple[int, int],
) -> np.ndarray | None:
    if query_initial_masks is None:
        return None
    masks_np = np.asarray(query_initial_masks, dtype=bool)
    expected_shape = (int(num_queries), int(frame_shape[0]), int(frame_shape[1]))
    if masks_np.shape != expected_shape:
        raise ValueError(f"query_initial_masks must have shape {expected_shape}, got {masks_np.shape}.")
    return masks_np


def _normalize_query_object_ids(query_object_ids: np.ndarray | None, num_queries: int) -> np.ndarray:
    if query_object_ids is None:
        return np.arange(1, int(num_queries) + 1, dtype=np.int32)
    object_ids = np.asarray(query_object_ids, dtype=np.int32)
    if object_ids.shape != (int(num_queries),):
        raise ValueError(f"query_object_ids must have shape {(num_queries,)}, got {object_ids.shape}.")
    if len(set(int(v) for v in object_ids.tolist())) != int(num_queries):
        raise ValueError("DEVA query_object_ids must be unique.")
    return object_ids


def _prepare_frames_rgb(frames: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    frames_np = np.asarray(frames)
    if frames_np.ndim == 3:
        frames_np = np.stack([frames_np] * 3, axis=-1)
    elif frames_np.ndim == 4 and frames_np.shape[-1] == 1:
        frames_np = np.repeat(frames_np, 3, axis=-1)
    elif frames_np.ndim == 4 and frames_np.shape[-1] >= 3:
        frames_np = frames_np[..., :3]
    else:
        raise ValueError(f"Unsupported DEVA frame shape: {frames_np.shape}")
    if frames_np.shape[0] == 0:
        raise ValueError("No frames provided for DEVA inference.")

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


def _video_to_gray_float(frames: np.ndarray) -> np.ndarray:
    frames_np = np.asarray(frames, dtype=np.float32)
    return (
        0.2989 * frames_np[..., 0]
        + 0.5870 * frames_np[..., 1]
        + 0.1140 * frames_np[..., 2]
    ).astype(np.float32, copy=False)


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
    if mode == "mask_centroid":
        ys, xs = np.nonzero(mask_bool)
        if xs.size == 0 or ys.size == 0:
            return None
        return (float(np.mean(xs)), float(np.mean(ys)))
    if mode == "peak_in_mask":
        masked = np.where(mask_bool, np.asarray(frame_gray, dtype=np.float32), -np.inf)
        try:
            peak_index = int(np.nanargmax(masked))
        except ValueError:
            return None
        peak_y, peak_x = np.unravel_index(peak_index, masked.shape)
        return (float(peak_x), float(peak_y))
    raise ValueError(f"Unsupported DEVA representative point mode: {mode}")


def _mask_bbox_xyxy(mask_bool: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0 or ys.size == 0:
        return None
    return (
        float(np.min(xs)),
        float(np.min(ys)),
        float(np.max(xs) + 1),
        float(np.max(ys) + 1),
    )


def _clip_box_xyxy(
    box_xyxy: np.ndarray,
    *,
    frame_shape: tuple[int, int],
    padding_px: float,
) -> tuple[int, int, int, int]:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    pad = max(0.0, float(padding_px))
    x0 -= pad
    y0 -= pad
    x1 += pad
    y1 += pad
    left = max(0, min(width, int(np.floor(x0))))
    top = max(0, min(height, int(np.floor(y0))))
    right = max(0, min(width, int(np.ceil(x1))))
    bottom = max(0, min(height, int(np.ceil(y1))))
    return left, top, right, bottom


def _normalize_representative_point_mode(mode: str) -> str:
    normalized = str(mode or "mask_centroid").strip().lower().replace(" ", "_")
    aliases = {
        "centroid": "mask_centroid",
        "mask_centroid": "mask_centroid",
        "maskcenter": "mask_centroid",
        "bbox_center": "bbox_center",
        "bboxcenter": "bbox_center",
        "peak": "peak_in_mask",
        "peak_in_mask": "peak_in_mask",
        "peakinmask": "peak_in_mask",
    }
    if normalized in aliases:
        return aliases[normalized]
    raise ValueError(f"Unsupported DEVA representative point mode: {mode}")


def _normalize_inference_size(inference_size: int | None) -> int:
    if inference_size is None:
        return -1
    size = int(inference_size)
    if size == -1:
        return -1
    if size <= 0:
        raise ValueError(f"Invalid DEVA inference size: {inference_size}")
    return size


def _validate_deva_device(device: str) -> None:
    normalized = str(device or "auto").strip().lower()
    if normalized in {"", "auto", "cuda", "gpu", "cuda:0"}:
        return
    raise ValueError(
        f"Unsupported DEVA device setting: {device}. "
        "The official DEVA eval_vos path is CUDA-only in this integration."
    )


def _resolve_deva_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidates: list[Path] = []
    if repo_path is not None:
        candidates.append(Path(repo_path).expanduser())
    candidates.extend(DEFAULT_DEVA_REPO_CANDIDATES)
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "deva").is_dir() and (resolved / "evaluation" / "eval_vos.py").is_file():
            return resolved
    return None


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoints: list[Path],
    models_dir: Path,
    repo_path: Optional[Path],
) -> Optional[Path]:
    if checkpoint_path is None:
        return select_default_deva_checkpoint(checkpoints)
    candidate = Path(checkpoint_path).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(Path(models_dir) / candidate)
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0].resolve()


def _resolve_existing_checkpoint_file(checkpoint_path: Path | None) -> Path:
    if checkpoint_path is None:
        raise FileNotFoundError("No DEVA checkpoint selected.")
    resolved = Path(checkpoint_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"DEVA checkpoint not found: {resolved}")
    return resolved


def _discover_checkpoints_in_directory(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".pth", ".pt"}
        ],
        key=lambda path: path.name.lower(),
    )


def _resolve_deva_python_executable(python_executable: str | Path | None) -> Path:
    if python_executable:
        return Path(python_executable).expanduser().resolve()
    return Path(sys.executable).resolve()


def _build_windows_conda_subprocess_env(python_executable: Path) -> dict[str, str]:
    env = os.environ.copy()
    if os.name != "nt":
        return env

    exe_path = Path(python_executable).expanduser().resolve()
    env_root = exe_path.parent
    dll_dirs = [
        env_root,
        env_root / "Library" / "bin",
        env_root / "DLLs",
        env_root / "Scripts",
        env_root / "Lib" / "site-packages" / "torch" / "lib",
    ]
    existing = [str(path) for path in dll_dirs if path.exists()]
    current_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(existing + ([current_path] if current_path else []))
    env.setdefault("CONDA_PREFIX", str(env_root))
    env.setdefault("CONDA_DEFAULT_ENV", env_root.name)
    return env


def _resolve_example_vos_path(repo_path: Path | None) -> Optional[Path]:
    if repo_path is None:
        return None
    return repo_path / "example" / "vos"


def _is_generic_vos_dataset(path: str | Path | None) -> bool:
    if path is None:
        return False
    root = Path(path)
    return (
        (root / "JPEGImages").is_dir()
        and (root / "Annotations").is_dir()
        and any((root / "JPEGImages").glob("*"))
        and any((root / "Annotations").glob("*"))
    )


@contextmanager
def _deva_import_context(repo_path: Path | None) -> Iterator[None]:
    added_paths: list[str] = []
    if repo_path is not None:
        repo_str = str(repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
            added_paths.append(repo_str)
    try:
        yield
    finally:
        for added_path in added_paths:
            try:
                sys.path.remove(added_path)
            except ValueError:
                pass


def _try_import_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _try_import(module_name: str, errors: list[str], label: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception as error:
        errors.append(f"{label} import failed: {error}")
        return False


def _format_size_mb(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown"
    return f"{float(size_bytes) / (1024.0 * 1024.0):.1f}"


def _format_checkpoint_paths(paths: list[Path]) -> str:
    return ", ".join(str(path) for path in paths) or "none"


def _import_pil_image():
    from PIL import Image

    return Image
