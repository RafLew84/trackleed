"""Cutie environment diagnostics, probes, and offline inference helpers."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Optional
import importlib
import json
import subprocess
import sys

import numpy as np


DEFAULT_CUTIE_MODELS_DIR = Path("tracking_models") / "cutie"
DEFAULT_CUTIE_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "cutie",
    Path(__file__).resolve().parents[2] / "cutie",
)
DEFAULT_CUTIE_PRIMARY_CHECKPOINT = "cutie-base-mega.pth"
DEFAULT_CUTIE_INTERACTIVE_CHECKPOINT = "coco_lvis_h18_itermask.pth"
DEFAULT_CUTIE_EXAMPLE_SEQUENCE = Path("examples") / "images" / "bike"
DEFAULT_CUTIE_EXAMPLE_MASK = Path("examples") / "masks" / "bike" / "00000.png"
DEFAULT_CUTIE_PROBE_SAMPLES = (
    (
        "bike",
        Path("examples") / "images" / "bike" / "00000.jpg",
        Path("examples") / "masks" / "bike" / "00000.png",
    ),
    (
        "judo",
        Path("examples") / "images" / "judo" / "00000.jpg",
        Path("examples") / "masks" / "judo" / "00000.png",
    ),
)
DEFAULT_CUTIE_SUBPROCESS_SCRIPT = Path("scripts") / "run_cutie_subprocess.py"
CHECKPOINT_PRIORITY = (
    DEFAULT_CUTIE_PRIMARY_CHECKPOINT,
    "cutie-base.pth",
    "cutie-small.pth",
    DEFAULT_CUTIE_INTERACTIVE_CHECKPOINT,
)


@dataclass
class CutieEnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    available_checkpoints: list[Path] = field(default_factory=list)
    selected_checkpoint: Optional[Path] = None
    checkpoint_exists: bool = False
    interactive_checkpoint: Optional[Path] = None
    interactive_checkpoint_exists: bool = False
    example_frames_dir: Optional[Path] = None
    example_mask_path: Optional[Path] = None
    example_inputs_ready: bool = False
    torch_importable: bool = False
    torch_version: Optional[str] = None
    cuda_available: bool = False
    torchvision_importable: bool = False
    hydra_importable: bool = False
    omegaconf_importable: bool = False
    pillow_importable: bool = False
    cutie_importable: bool = False
    cutie_module_path: Optional[str] = None
    cutie_model_importable: bool = False
    inference_core_importable: bool = False
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
            and self.hydra_importable
            and self.omegaconf_importable
            and self.pillow_importable
            and self.cutie_importable
            and self.cutie_model_importable
            and self.inference_core_importable
        )


@dataclass
class CutieSmokeTestResult:
    first_mask_shape: tuple[int, ...]
    second_mask_shape: tuple[int, ...]
    first_mask_area: int
    second_mask_area: int
    object_ids: list[int]
    frame_count: int
    device: str
    checkpoint_name: str
    example_sequence_name: str
    max_internal_size: int


@dataclass
class CutieInitialMaskModeResult:
    mode: str
    area: int
    bbox_xyxy: tuple[int, int, int, int] | None
    centroid_xy: tuple[float, float] | None
    reference_iou: float
    jitter_mean_iou: float
    jitter_min_iou: float
    coverage_in_roi: float


@dataclass
class CutieInitialMaskSampleResult:
    sample_name: str
    image_shape: tuple[int, int]
    roi_rect_xywh: tuple[int, int, int, int]
    reference_area: int
    mode_results: list[CutieInitialMaskModeResult]


@dataclass
class CutieInitialMaskProbeResult:
    samples: list[CutieInitialMaskSampleResult]
    evaluated_modes: list[str]
    recommended_mode: str
    threshold_rel: float
    disk_radius_ratio: float
    bbox_padding_ratio: float
    jitter_px: int
    jitter_scale: float
    notes: list[str] = field(default_factory=list)


@dataclass
class CutieRuntimeSettings:
    checkpoint_path: str | Path
    repo_path: str | Path | None = None
    python_executable: str | Path | None = None
    device: str = "auto"
    backend_mode: str = "direct"
    representative_point_mode: str = "mask_centroid"
    inference_resolution: int | tuple[int, int] | None = None
    initial_bbox_padding_px: float = 12.0
    initial_mask_mode: str = "threshold"
    initial_mask_threshold_rel: float = 0.7
    initial_mask_disk_radius_ratio: float = 0.35
    mem_every: int | None = None
    subprocess_timeout_sec: float = 1800.0


@dataclass
class CutieOfflineResult:
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


def discover_cutie_checkpoints(
    models_dir: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_CUTIE_MODELS_DIR
    discovered: dict[str, Path] = {}

    for candidate in _discover_checkpoints_in_directory(directory):
        discovered[str(candidate.resolve())] = candidate.resolve()

    repo_root = _resolve_cutie_repo_path(repo_path)
    if repo_root is not None:
        for candidate in _discover_checkpoints_in_directory(repo_root / "weights"):
            discovered[str(candidate.resolve())] = candidate.resolve()

    return sorted(discovered.values(), key=lambda path: path.name.lower())


def select_default_cutie_checkpoint(checkpoints: list[Path]) -> Optional[Path]:
    if not checkpoints:
        return None
    by_name = {path.name: path for path in checkpoints}
    for preferred_name in CHECKPOINT_PRIORITY:
        if preferred_name in by_name:
            return by_name[preferred_name]
    return checkpoints[0]


def probe_cutie_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> CutieEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_CUTIE_MODELS_DIR
    directory = directory.resolve()
    resolved_repo_path = _resolve_cutie_repo_path(repo_path)
    checkpoints = discover_cutie_checkpoints(directory, resolved_repo_path)
    selected_checkpoint = _resolve_checkpoint_path(
        checkpoint_path,
        checkpoints,
        directory,
        resolved_repo_path,
    )
    interactive_checkpoint = _find_checkpoint_by_name(
        checkpoints,
        DEFAULT_CUTIE_INTERACTIVE_CHECKPOINT,
    )
    example_frames_dir, example_mask_path = _resolve_cutie_example_inputs(resolved_repo_path)

    status = CutieEnvironmentStatus(
        models_dir=directory,
        repo_path=resolved_repo_path,
        available_checkpoints=checkpoints,
        selected_checkpoint=selected_checkpoint,
        checkpoint_exists=bool(selected_checkpoint and selected_checkpoint.exists()),
        interactive_checkpoint=interactive_checkpoint,
        interactive_checkpoint_exists=bool(
            interactive_checkpoint and interactive_checkpoint.exists()
        ),
        example_frames_dir=example_frames_dir,
        example_mask_path=example_mask_path,
        example_inputs_ready=bool(
            example_frames_dir
            and example_frames_dir.exists()
            and example_mask_path
            and example_mask_path.exists()
        ),
    )

    if not directory.exists():
        status.warnings.append(
            f"Checkpoint directory does not exist yet: {directory}. "
            "This is fine if you still keep Cutie weights inside the local repo/weights directory."
        )

    if selected_checkpoint is None:
        status.warnings.append(
            "No Cutie checkpoint detected in tracking_models/cutie or in repo/weights."
        )
        status.hints.append(
            f"Add `{DEFAULT_CUTIE_PRIMARY_CHECKPOINT}` before running Cutie diagnostics."
        )
    elif checkpoint_path and not status.checkpoint_exists:
        status.errors.append(f"Selected checkpoint not found: {selected_checkpoint}")

    if resolved_repo_path is None:
        status.errors.append(
            "Could not resolve the local Cutie repo. Clone `Cutie` as `cutie` next to the "
            "project or pass --repo-path."
        )
        status.hints.append(
            "Expected repo root containing `pyproject.toml`, `scripting_demo.py`, and "
            "`cutie/inference/inference_core.py`."
        )

    if not status.example_inputs_ready:
        status.warnings.append(
            "Default Cutie example inputs are missing under `examples/images/bike` and "
            "`examples/masks/bike/00000.png`. The optional smoke test will need a valid repo checkout."
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
            "Install the Cutie runtime torch stack in the dedicated `cutie` env."
        )

    try:
        importlib.import_module("torchvision")
        status.torchvision_importable = True
    except Exception as exc:
        status.errors.append(f"torchvision import failed: {exc}")

    try:
        importlib.import_module("hydra")
        status.hydra_importable = True
    except Exception as exc:
        status.errors.append(f"hydra import failed: {exc}")
        status.hints.append("Install Cutie with `pip install -e .` inside the dedicated env.")

    try:
        importlib.import_module("omegaconf")
        status.omegaconf_importable = True
    except Exception as exc:
        status.errors.append(f"omegaconf import failed: {exc}")

    try:
        importlib.import_module("PIL.Image")
        status.pillow_importable = True
    except Exception as exc:
        status.errors.append(f"Pillow import failed: {exc}")

    with _cutie_import_context(resolved_repo_path):
        try:
            cutie_module = importlib.import_module("cutie")
            module_file = getattr(cutie_module, "__file__", None)
            status.cutie_importable = True
            if module_file is not None:
                status.cutie_module_path = str(Path(module_file).resolve())
            else:
                namespace_paths: list[str] = []
                for entry in getattr(cutie_module, "__path__", []):
                    try:
                        namespace_paths.append(str(Path(entry).resolve()))
                    except Exception:
                        namespace_paths.append(str(entry))
                if not namespace_paths:
                    raise RuntimeError(
                        "Cutie import resolved without a file or package path. "
                        "This usually means the installation is incomplete."
                    )
                status.cutie_module_path = "; ".join(namespace_paths)
        except Exception as exc:
            status.errors.append(f"Cutie import failed: {exc}")
            status.hints.append("Install the local repo with `pip install -e .` in the `cutie` env.")

        try:
            importlib.import_module("cutie.model.cutie")
            status.cutie_model_importable = True
        except Exception as exc:
            status.errors.append(f"Cutie model import failed: {exc}")

        try:
            importlib.import_module("cutie.inference.inference_core")
            status.inference_core_importable = True
        except Exception as exc:
            status.errors.append(f"Cutie inference core import failed: {exc}")

    if status.torch_importable and not status.cuda_available:
        status.warnings.append(
            "PyTorch CUDA is not available. Cutie may still run on CPU, but longer sequences will be very slow."
        )

    if not status.interactive_checkpoint_exists:
        status.warnings.append(
            "Interactive Cutie checkpoint `coco_lvis_h18_itermask.pth` was not found. "
            "This is optional for Trackleed MVP."
        )

    if sys.platform.startswith("win"):
        status.warnings.append(
            "Upstream Cutie was tested on Ubuntu. Native Windows can still work if the scripting smoke test passes."
        )

    return status


def format_cutie_environment_report(status: CutieEnvironmentStatus) -> str:
    selected_checkpoint = (
        str(status.selected_checkpoint) if status.selected_checkpoint is not None else "None"
    )
    available = ", ".join(path.name for path in status.available_checkpoints) or "none"
    interactive_checkpoint = (
        str(status.interactive_checkpoint) if status.interactive_checkpoint is not None else "None"
    )
    lines = [
        "Cutie environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Selected checkpoint: {selected_checkpoint}",
        f"- Available checkpoints: {available}",
        f"- Interactive checkpoint: {interactive_checkpoint}",
        f"- Example frames: {status.example_frames_dir or 'None'}",
        f"- Example mask: {status.example_mask_path or 'None'}",
        f"- PyTorch import: {'ok' if status.torch_importable else 'failed'}",
        f"- PyTorch version: {status.torch_version or 'unknown'}",
        f"- CUDA available: {'yes' if status.cuda_available else 'no'}",
        f"- torchvision import: {'ok' if status.torchvision_importable else 'failed'}",
        f"- hydra import: {'ok' if status.hydra_importable else 'failed'}",
        f"- omegaconf import: {'ok' if status.omegaconf_importable else 'failed'}",
        f"- Pillow import: {'ok' if status.pillow_importable else 'failed'}",
        f"- Cutie import: {'ok' if status.cutie_importable else 'failed'}",
        f"- Cutie model import: {'ok' if status.cutie_model_importable else 'failed'}",
        f"- Inference core import: {'ok' if status.inference_core_importable else 'failed'}",
    ]
    if status.cutie_module_path:
        lines.append(f"- Cutie module: {status.cutie_module_path}")
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


def run_cutie_smoke_test(
    *,
    checkpoint_path: str | Path,
    repo_path: str | Path | None = None,
    device: str = "auto",
    max_internal_size: int = 256,
) -> CutieSmokeTestResult:
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Cutie checkpoint not found: {checkpoint}")

    resolved_repo_path = _resolve_cutie_repo_path(repo_path)
    if resolved_repo_path is None:
        raise RuntimeError("Could not resolve the Cutie repo for the smoke test.")

    example_frames_dir, example_mask_path = _resolve_cutie_example_inputs(resolved_repo_path)
    if example_frames_dir is None or example_mask_path is None:
        raise RuntimeError("Could not resolve the default Cutie example inputs.")

    image_paths = sorted(example_frames_dir.glob("*.jpg"))
    if len(image_paths) < 2:
        raise RuntimeError(
            f"Need at least two example frames for the Cutie smoke test, found {len(image_paths)}."
        )

    with _cutie_import_context(resolved_repo_path):
        torch = importlib.import_module("torch")
        Image = importlib.import_module("PIL.Image")
        to_tensor = importlib.import_module("torchvision.transforms.functional").to_tensor
        model, cfg, _, _, inference_core_module = _load_cutie_runtime(
            checkpoint,
            repo_path=resolved_repo_path,
            torch_module=torch,
            device=_resolve_torch_device(torch, device),
            max_internal_size=int(max_internal_size),
            mem_every=None,
        )

        resolved_device = _resolve_torch_device(torch, device)
        processor = inference_core_module.InferenceCore(model, cfg=cfg)
        processor.max_internal_size = int(max_internal_size)

        mask_image = Image.open(str(example_mask_path))
        if getattr(mask_image, "mode", None) not in ("L", "P"):
            raise RuntimeError(
                "Cutie smoke test expects an indexed or grayscale first-frame mask."
            )
        mask_np = np.array(mask_image)
        object_ids = np.unique(mask_np)
        object_ids = object_ids[object_ids != 0].astype(int).tolist()
        if not object_ids:
            raise RuntimeError("Cutie smoke test example mask does not contain any objects.")
        mask_tensor = torch.from_numpy(mask_np).to(resolved_device)

        output_masks: list[np.ndarray] = []
        for frame_index, image_path in enumerate(image_paths[:2]):
            image = Image.open(str(image_path)).convert("RGB")
            image_tensor = to_tensor(image).to(resolved_device).float()
            if frame_index == 0:
                output_prob = processor.step(image_tensor, mask_tensor, objects=object_ids)
            else:
                output_prob = processor.step(image_tensor, end=True)
            output_mask = processor.output_prob_to_mask(output_prob)
            output_masks.append(output_mask.detach().cpu().numpy().astype(np.uint8))

    first_mask = output_masks[0]
    second_mask = output_masks[1]
    return CutieSmokeTestResult(
        first_mask_shape=tuple(first_mask.shape),
        second_mask_shape=tuple(second_mask.shape),
        first_mask_area=int(np.count_nonzero(first_mask)),
        second_mask_area=int(np.count_nonzero(second_mask)),
        object_ids=object_ids,
        frame_count=2,
        device=str(resolved_device),
        checkpoint_name=checkpoint.name,
        example_sequence_name=example_frames_dir.name,
        max_internal_size=int(max_internal_size),
    )


def run_cutie_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: CutieRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
    query_initial_masks: np.ndarray | None = None,
) -> CutieOfflineResult:
    frames_np = np.asarray(frames)
    if frames_np.ndim not in (3, 4):
        raise ValueError(f"Expected frames with ndim 3 or 4, got shape {frames_np.shape}.")
    if frames_np.size == 0:
        raise ValueError("No frames provided for Cutie inference.")

    query_np = np.asarray(query_points_tyx, dtype=np.float32)
    if query_np.ndim != 2 or query_np.shape[1] != 3:
        raise ValueError("query_points_tyx must have shape [num_queries, 3] in [t, y, x] format.")
    if query_np.shape[0] == 0:
        raise ValueError("No query points provided for Cutie inference.")

    boxes_np = None
    if query_boxes_xyxy is not None:
        boxes_np = np.asarray(query_boxes_xyxy, dtype=np.float32)
        if boxes_np.shape != (query_np.shape[0], 4):
            raise ValueError(
                f"query_boxes_xyxy must have shape {(query_np.shape[0], 4)}, got {boxes_np.shape}."
            )
    query_initial_masks_np = _normalize_cutie_query_initial_masks(
        query_initial_masks,
        num_queries=query_np.shape[0],
        frame_shape=frames_np.shape[1:3],
    )

    backend_mode = str(settings.backend_mode or "direct").strip().lower()
    if backend_mode == "auto":
        backend_mode = "direct"
    if backend_mode == "direct":
        return _run_cutie_direct_offline(
            frames_np,
            query_np,
            settings,
            query_boxes_xyxy=boxes_np,
            query_initial_masks=query_initial_masks_np,
        )
    if backend_mode != "subprocess":
        raise ValueError(
            f"Unsupported Cutie backend mode: {settings.backend_mode}. "
            "Expected `direct` or `subprocess`."
        )
    return _run_cutie_subprocess_offline(
        frames_np,
        query_np,
        settings,
        query_boxes_xyxy=boxes_np,
        query_initial_masks=query_initial_masks_np,
    )


def _run_cutie_direct_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: CutieRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
    query_initial_masks: np.ndarray | None = None,
) -> CutieOfflineResult:
    resolved_repo_path = _resolve_cutie_repo_path(settings.repo_path)
    if resolved_repo_path is None:
        raise FileNotFoundError(
            "Could not resolve the Cutie repo. Pass repo_path pointing at the local `cutie` clone."
        )

    checkpoint_file = _resolve_existing_checkpoint_file(Path(settings.checkpoint_path))
    representative_point_mode = _normalize_representative_point_mode(
        settings.representative_point_mode
    )
    initial_mask_mode = _normalize_cutie_mask_builder_mode(settings.initial_mask_mode)
    effective_initial_mask_mode = (
        "polygon" if query_initial_masks is not None else initial_mask_mode
    )

    prepared_frames, original_resolution = _prepare_frames_rgb(frames)
    inference_resolution = _normalize_optional_resolution(
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
    scaled_boxes = _build_cutie_prompt_boxes_xyxy(
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
    scaled_initial_masks = _resize_boolean_mask_stack_numpy(
        query_initial_masks,
        inference_resolution,
    )
    seed_frames = _extract_query_seed_frames(scaled_queries, total_frames=resized_frames.shape[0])

    num_queries = int(scaled_queries.shape[0])
    num_frames = int(resized_frames.shape[0])
    tracks_xy_resized = np.full((num_queries, num_frames, 2), np.nan, dtype=np.float32)
    visible_mask = np.zeros((num_queries, num_frames), dtype=bool)
    mask_areas = np.zeros((num_queries, num_frames), dtype=np.float32)
    mask_bboxes_xyxy_resized = np.full((num_queries, num_frames, 4), np.nan, dtype=np.float32)
    mask_scores = np.zeros((num_queries, num_frames), dtype=np.float32)

    with _cutie_import_context(resolved_repo_path):
        torch = importlib.import_module("torch")
        to_tensor = importlib.import_module("torchvision.transforms.functional").to_tensor
        resolved_device = _resolve_torch_device(torch, settings.device)
        model, cfg, _, _, inference_core_module = _load_cutie_runtime(
            checkpoint_file,
            repo_path=resolved_repo_path,
            torch_module=torch,
            device=resolved_device,
            max_internal_size=-1,
            mem_every=settings.mem_every,
        )
        processor = inference_core_module.InferenceCore(model, cfg=cfg)
        processor.max_internal_size = -1

        with torch.inference_mode():
            with _make_autocast_context(torch, resolved_device):
                for frame_index in range(num_frames):
                    frame_rgb = resized_frames[frame_index]
                    image_tensor = to_tensor(frame_rgb).to(resolved_device).float()
                    seed_indices = np.where(seed_frames == frame_index)[0].tolist()
                    seed_mask_tensor, seed_object_ids = _build_cutie_seed_mask_tensor(
                        frame_rgb,
                        scaled_boxes,
                        seed_indices,
                        query_initial_masks=scaled_initial_masks,
                        device=resolved_device,
                        initial_mask_mode=initial_mask_mode,
                        threshold_rel=float(settings.initial_mask_threshold_rel),
                        disk_radius_ratio=float(settings.initial_mask_disk_radius_ratio),
                        torch_module=torch,
                    )

                    if seed_mask_tensor is not None and seed_object_ids:
                        output_prob = processor.step(
                            image_tensor,
                            seed_mask_tensor,
                            objects=seed_object_ids,
                            end=(frame_index == num_frames - 1),
                        )
                    else:
                        output_prob = processor.step(
                            image_tensor,
                            end=(frame_index == num_frames - 1),
                        )

                    class_mask = processor.output_prob_to_mask(output_prob)
                    _record_cutie_frame_outputs(
                        class_mask=class_mask,
                        output_prob=output_prob,
                        object_manager=processor.object_manager,
                        frame_gray=gray_frames[frame_index],
                        frame_index=frame_index,
                        tracks_xy=tracks_xy_resized,
                        visible_mask=visible_mask,
                        mask_areas=mask_areas,
                        mask_bboxes_xyxy=mask_bboxes_xyxy_resized,
                        mask_scores=mask_scores,
                        representative_point_mode=representative_point_mode,
                        inference_resolution=inference_resolution,
                        original_resolution=original_resolution,
                    )

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
    return CutieOfflineResult(
        tracks_xy=tracks_xy,
        visible_mask=visible_mask,
        mask_areas=mask_areas,
        mask_bboxes_xyxy=mask_bboxes_xyxy
        if mask_bboxes_xyxy is not None
        else mask_bboxes_xyxy_resized,
        mask_scores=mask_scores,
        inference_resolution=inference_resolution,
        original_resolution=original_resolution,
        checkpoint_name=checkpoint_file.name,
        representative_point_mode=representative_point_mode,
        initial_mask_mode=effective_initial_mask_mode,
        backend_mode="direct",
    )


def _run_cutie_subprocess_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: CutieRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
    query_initial_masks: np.ndarray | None = None,
) -> CutieOfflineResult:
    python_executable = _resolve_cutie_python_executable(settings.python_executable)
    script_path = (Path(__file__).resolve().parents[2] / DEFAULT_CUTIE_SUBPROCESS_SCRIPT).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Cutie subprocess script not found: {script_path}")

    with TemporaryDirectory(prefix="trackleed_cutie_") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "cutie_input.npz"
        output_path = temp_root / "cutie_output.npz"
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
            command.extend(["--height", str(int(resolution[0])), "--width", str(int(resolution[1]))])
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
                "Cutie subprocess inference failed:\n"
                + json.dumps(payload, indent=2, ensure_ascii=False)
            )
        if not output_path.exists():
            raise RuntimeError("Cutie subprocess finished without creating the output .npz file.")

        with np.load(output_path, allow_pickle=False) as payload:
            return CutieOfflineResult(
                tracks_xy=np.asarray(payload["tracks_xy"], dtype=np.float32),
                visible_mask=np.asarray(payload["visible_mask"], dtype=bool),
                mask_areas=np.asarray(payload["mask_areas"], dtype=np.float32),
                mask_bboxes_xyxy=np.asarray(payload["mask_bboxes_xyxy"], dtype=np.float32),
                mask_scores=np.asarray(payload["mask_scores"], dtype=np.float32),
                inference_resolution=tuple(
                    int(v) for v in np.asarray(payload["inference_resolution"], dtype=np.int32).tolist()
                ),
                original_resolution=tuple(
                    int(v) for v in np.asarray(payload["original_resolution"], dtype=np.int32).tolist()
                ),
                checkpoint_name=str(payload["checkpoint_name"].item()),
                representative_point_mode=str(payload["representative_point_mode"].item()),
                initial_mask_mode=str(payload["initial_mask_mode"].item()),
                backend_mode="subprocess",
            )


def build_cutie_initial_mask(
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

    normalized_mode = _normalize_cutie_mask_builder_mode(mode)
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


def probe_cutie_initial_mask_builder(
    *,
    repo_path: str | Path | None = None,
    modes: list[str] | tuple[str, ...] | None = None,
    threshold_rel: float = 0.7,
    disk_radius_ratio: float = 0.35,
    bbox_padding_ratio: float = 0.15,
    jitter_px: int = 4,
    jitter_scale: float = 0.12,
) -> CutieInitialMaskProbeResult:
    resolved_repo_path = _resolve_cutie_repo_path(repo_path)
    if resolved_repo_path is None:
        raise RuntimeError("Could not resolve the Cutie repo for the mask builder probe.")

    normalized_modes = [
        _normalize_cutie_mask_builder_mode(mode)
        for mode in (modes or ("threshold", "full_bbox", "disk"))
    ]
    deduped_modes: list[str] = []
    for mode in normalized_modes:
        if mode not in deduped_modes:
            deduped_modes.append(mode)

    Image = importlib.import_module("PIL.Image")
    sample_results: list[CutieInitialMaskSampleResult] = []
    aggregate_scores = {mode: [] for mode in deduped_modes}

    for sample_name, image_rel, mask_rel in DEFAULT_CUTIE_PROBE_SAMPLES:
        image_path = (resolved_repo_path / image_rel).resolve()
        mask_path = (resolved_repo_path / mask_rel).resolve()
        if not image_path.exists() or not mask_path.exists():
            raise RuntimeError(
                f"Cutie probe sample is missing: image={image_path}, mask={mask_path}"
            )

        image_np = np.array(Image.open(str(image_path)).convert("RGB"))
        reference_mask = np.array(Image.open(str(mask_path))) > 0
        if reference_mask.ndim != 2 or not np.any(reference_mask):
            raise RuntimeError(f"Cutie probe reference mask is empty: {mask_path}")

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

        mode_results: list[CutieInitialMaskModeResult] = []
        for mode in deduped_modes:
            base_mask = build_cutie_initial_mask(
                image_np,
                roi_rect_xywh,
                mode=mode,
                threshold_rel=threshold_rel,
                disk_radius_ratio=disk_radius_ratio,
            )
            jitter_ious: list[float] = []
            for jittered_roi in jittered_rois:
                jitter_mask = build_cutie_initial_mask(
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
                CutieInitialMaskModeResult(
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
            CutieInitialMaskSampleResult(
                sample_name=sample_name,
                image_shape=(image_h, image_w),
                roi_rect_xywh=roi_rect_xywh,
                reference_area=int(np.count_nonzero(reference_mask)),
                mode_results=mode_results,
            )
        )

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
        "Score laczy zgodnosc z referencyjna maska pierwszej klatki oraz odpornosc na niewielkie przesuniecia bboxa.",
        "Domyslny seed dla MVP powinien preferowac wysoki reference IoU, ale bez nadmiernej wrazliwosci na jitter ROI.",
        "Probe korzysta z upstreamowych przykladow `bike` i `judo`, wiec wynik trzeba traktowac jako wstepny przed testem na sekwencjach LEED.",
    ]
    return CutieInitialMaskProbeResult(
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


def format_cutie_initial_mask_probe_report(result: CutieInitialMaskProbeResult) -> str:
    lines = [
        "Cutie initial mask builder probe",
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


def _discover_checkpoints_in_directory(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [path.resolve() for path in directory.glob("*.pth") if path.is_file()]


def _resolve_cutie_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidates: list[Path] = []
    if repo_path is not None:
        candidates.append(Path(repo_path))
    candidates.extend(DEFAULT_CUTIE_REPO_CANDIDATES)

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if _looks_like_cutie_repo(resolved):
            return resolved
    return None


def _looks_like_cutie_repo(path: Path) -> bool:
    return (
        path.exists()
        and (path / "pyproject.toml").exists()
        and (path / "scripting_demo.py").exists()
        and (path / "cutie" / "inference" / "inference_core.py").exists()
    )


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    discovered: list[Path],
    models_dir: Path,
    repo_path: Optional[Path],
) -> Optional[Path]:
    if checkpoint_path is None:
        return select_default_cutie_checkpoint(discovered)

    candidate = Path(checkpoint_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    candidate_name = candidate.name
    by_name = {path.name: path for path in discovered}
    if candidate_name in by_name:
        return by_name[candidate_name]

    for base_dir in (models_dir, repo_path / "weights" if repo_path is not None else None):
        if base_dir is None:
            continue
        expanded = (base_dir / candidate).resolve()
        if expanded.exists():
            return expanded

    return candidate.resolve() if candidate.is_absolute() else (models_dir / candidate).resolve()


def _find_checkpoint_by_name(checkpoints: list[Path], checkpoint_name: str) -> Optional[Path]:
    for checkpoint in checkpoints:
        if checkpoint.name == checkpoint_name:
            return checkpoint
    return None


def _resolve_cutie_example_inputs(
    repo_path: Optional[Path],
) -> tuple[Optional[Path], Optional[Path]]:
    if repo_path is None:
        return None, None
    frames_dir = (repo_path / DEFAULT_CUTIE_EXAMPLE_SEQUENCE).resolve()
    mask_path = (repo_path / DEFAULT_CUTIE_EXAMPLE_MASK).resolve()
    return frames_dir, mask_path


def _load_cutie_runtime(
    checkpoint_path: Path,
    *,
    repo_path: Path,
    torch_module: Any,
    device: Any,
    max_internal_size: int,
    mem_every: int | None,
) -> tuple[Any, Any, Any, Any, Any]:
    hydra_module = importlib.import_module("hydra")
    hydra_global = importlib.import_module("hydra.core.global_hydra")
    omegaconf = importlib.import_module("omegaconf")
    cutie_model_module = importlib.import_module("cutie.model.cutie")
    inference_core_module = importlib.import_module("cutie.inference.inference_core")
    args_utils_module = importlib.import_module("cutie.inference.utils.args_utils")

    hydra_global.GlobalHydra.instance().clear()
    with hydra_module.initialize_config_dir(
        version_base="1.3.2",
        config_dir=str((repo_path / "cutie" / "config").resolve()),
        job_name="trackleed_cutie_runtime",
    ):
        cfg = hydra_module.compose(config_name="eval_config")
    with omegaconf.open_dict(cfg):
        cfg["weights"] = str(checkpoint_path)
        cfg["dataset"] = "generic"
        cfg["amp"] = False
        cfg["flip_aug"] = False
        cfg["max_internal_size"] = int(max_internal_size)
        if mem_every is not None:
            cfg["mem_every"] = int(mem_every)
    args_utils_module.get_dataset_cfg(cfg)

    model = cutie_model_module.CUTIE(cfg).to(device).eval()
    model_weights = torch_module.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    model.load_weights(model_weights)
    return model, cfg, omegaconf, cutie_model_module, inference_core_module


def _normalize_cutie_mask_builder_mode(mode: str) -> str:
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
            f"Unsupported Cutie mask builder mode: {mode}. Expected threshold/full_bbox/disk."
        )
    return aliases[normalized]


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
    raise ValueError(f"Unsupported Cutie representative point mode: {mode}")


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


def _prepare_frames_rgb(frames: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    frames_np = np.asarray(frames)
    if frames_np.ndim == 3:
        frames_np = np.stack([frames_np] * 3, axis=-1)
    elif frames_np.ndim == 4 and frames_np.shape[-1] == 1:
        frames_np = np.repeat(frames_np, 3, axis=-1)
    elif frames_np.ndim == 4 and frames_np.shape[-1] >= 3:
        frames_np = frames_np[..., :3]
    else:
        raise ValueError(f"Unsupported Cutie frame shape: {frames_np.shape}")

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
        raise ValueError(f"Invalid Cutie resolution: {resolution}")
    return (int(resolution[0]), int(resolution[1]))


def _resize_video_numpy(frames: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target[0]), int(target[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid Cutie inference resolution: {(target_h, target_w)}")
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


def _normalize_cutie_query_initial_masks(
    query_initial_masks: np.ndarray | None,
    *,
    num_queries: int,
    frame_shape: tuple[int, ...],
) -> np.ndarray | None:
    if query_initial_masks is None:
        return None
    masks = np.asarray(query_initial_masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(
            f"query_initial_masks must have shape [{num_queries}, H, W], got {masks.shape}."
        )
    if masks.shape[0] != int(num_queries):
        raise ValueError(
            f"query_initial_masks must have {int(num_queries)} entries, got {masks.shape[0]}."
        )
    frame_h = int(frame_shape[0])
    frame_w = int(frame_shape[1])
    if masks.shape[1:] != (frame_h, frame_w):
        raise ValueError(
            "query_initial_masks must match the input frame resolution. "
            f"Expected {(frame_h, frame_w)}, got {masks.shape[1:]}."
        )
    empty_indices = [int(idx) for idx in np.where(~np.any(masks.reshape(masks.shape[0], -1), axis=1))[0].tolist()]
    if empty_indices:
        raise ValueError(
            "query_initial_masks contains empty masks for queries: "
            + ", ".join(str(idx) for idx in empty_indices)
        )
    return masks


def _resize_boolean_mask_stack_numpy(
    mask_stack: np.ndarray | None,
    target_resolution: tuple[int, int],
) -> np.ndarray | None:
    if mask_stack is None:
        return None
    masks = np.asarray(mask_stack, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"mask_stack must have shape [N, H, W], got {masks.shape}.")
    target_h, target_w = int(target_resolution[0]), int(target_resolution[1])
    if masks.shape[1:] == (target_h, target_w):
        return masks.astype(bool, copy=True)

    image_module = importlib.import_module("PIL.Image")
    resized = np.zeros((masks.shape[0], target_h, target_w), dtype=bool)
    for mask_index in range(masks.shape[0]):
        raster = image_module.fromarray(masks[mask_index].astype(np.uint8) * 255, mode="L")
        resized_mask = raster.resize((target_w, target_h), resample=image_module.Resampling.NEAREST)
        resized_mask_np = np.asarray(resized_mask, dtype=np.uint8) > 0
        if not np.any(resized_mask_np) and np.any(masks[mask_index]):
            ys, xs = np.nonzero(masks[mask_index])
            centroid_x = float(np.mean(xs))
            centroid_y = float(np.mean(ys))
            src_h, src_w = masks.shape[1], masks.shape[2]
            mapped_x = int(np.clip(round(centroid_x * float(target_w) / float(src_w)), 0, max(0, target_w - 1)))
            mapped_y = int(np.clip(round(centroid_y * float(target_h) / float(src_h)), 0, max(0, target_h - 1)))
            resized_mask_np[mapped_y, mapped_x] = True
        resized[mask_index] = resized_mask_np
    return resized


def _extract_query_seed_frames(query_points_tyx: np.ndarray, total_frames: int) -> np.ndarray:
    seed_frames = np.rint(np.asarray(query_points_tyx[:, 0], dtype=np.float32)).astype(np.int32)
    if np.any(seed_frames < 0) or np.any(seed_frames >= int(total_frames)):
        raise ValueError(
            f"Cutie query seed frames must be in [0, {int(total_frames) - 1}], got {seed_frames.tolist()}."
        )
    return seed_frames


def _build_cutie_prompt_boxes_xyxy(
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


def _build_cutie_seed_mask_tensor(
    frame_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    seed_indices: list[int],
    *,
    query_initial_masks: np.ndarray | None,
    device: Any,
    initial_mask_mode: str,
    threshold_rel: float,
    disk_radius_ratio: float,
    torch_module: Any,
) -> tuple[Any | None, list[int]]:
    if not seed_indices:
        return None, []

    height, width = frame_rgb.shape[:2]
    idx_mask = np.zeros((height, width), dtype=np.int64)
    object_ids: list[int] = []
    for query_index in seed_indices:
        if query_initial_masks is not None:
            seed_mask = np.asarray(query_initial_masks[query_index], dtype=bool)
            if seed_mask.shape != (height, width):
                raise ValueError(
                    "Cutie query_initial_masks must match the inference frame resolution. "
                    f"Expected {(height, width)}, got {seed_mask.shape}."
                )
            bbox_tuple = _mask_bbox_xyxy(seed_mask)
            if bbox_tuple is None:
                raise ValueError(
                    f"Cutie initial mask for query {query_index} is empty after preprocessing."
                )
            bbox = np.asarray(bbox_tuple, dtype=np.float32)
            roi_rect_xywh = (
                int(np.floor(float(bbox[0]))),
                int(np.floor(float(bbox[1]))),
                int(max(1.0, np.ceil(float(bbox[2]) - float(bbox[0])))),
                int(max(1.0, np.ceil(float(bbox[3]) - float(bbox[1])))),
            )
        else:
            bbox = np.asarray(boxes_xyxy[query_index], dtype=np.float32)
            roi_rect_xywh = (
                int(np.floor(float(bbox[0]))),
                int(np.floor(float(bbox[1]))),
                int(max(1.0, np.ceil(float(bbox[2]) - float(bbox[0])))),
                int(max(1.0, np.ceil(float(bbox[3]) - float(bbox[1])))),
            )
            seed_mask = build_cutie_initial_mask(
                frame_rgb,
                roi_rect_xywh,
                mode=initial_mask_mode,
                threshold_rel=threshold_rel,
                disk_radius_ratio=disk_radius_ratio,
            )
            if not np.any(seed_mask):
                seed_mask = build_cutie_initial_mask(
                    frame_rgb,
                    roi_rect_xywh,
                    mode="full_bbox",
                    threshold_rel=threshold_rel,
                    disk_radius_ratio=disk_radius_ratio,
                )

        free_mask = seed_mask & (idx_mask == 0)
        if not np.any(free_mask):
            x, y, width_xy, height_xy = _clamp_roi_rect_xywh(roi_rect_xywh, (height, width))
            center_x = min(width - 1, max(0, int(round(x + (width_xy - 1) / 2.0))))
            center_y = min(height - 1, max(0, int(round(y + (height_xy - 1) / 2.0))))
            free_mask = np.zeros_like(seed_mask, dtype=bool)
            free_mask[center_y, center_x] = True

        object_id = int(query_index + 1)
        idx_mask[free_mask] = object_id
        object_ids.append(object_id)

    if not object_ids:
        return None, []
    return torch_module.from_numpy(idx_mask).to(device), object_ids


def _record_cutie_frame_outputs(
    *,
    class_mask,
    output_prob,
    object_manager,
    frame_gray: np.ndarray,
    frame_index: int,
    tracks_xy: np.ndarray,
    visible_mask: np.ndarray,
    mask_areas: np.ndarray,
    mask_bboxes_xyxy: np.ndarray,
    mask_scores: np.ndarray,
    representative_point_mode: str,
    inference_resolution: tuple[int, int],
    original_resolution: tuple[int, int],
) -> None:
    class_mask_np = class_mask.detach().cpu().numpy().astype(np.int32)
    output_prob_np = output_prob.detach().cpu().numpy().astype(np.float32)
    area_scale = (
        float(original_resolution[0]) / float(inference_resolution[0])
    ) * (
        float(original_resolution[1]) / float(inference_resolution[1])
    )

    for tmp_id, obj in object_manager.tmp_id_to_obj.items():
        query_index = int(obj.id) - 1
        if query_index < 0 or query_index >= tracks_xy.shape[0]:
            continue
        obj_mask = class_mask_np == int(obj.id)
        if not np.any(obj_mask):
            continue

        area_native = float(np.count_nonzero(obj_mask))
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
        mask_areas[query_index, frame_index] = float(area_native * area_scale)

        if 0 <= int(tmp_id) < int(output_prob_np.shape[0]):
            prob_map = output_prob_np[int(tmp_id)]
            mask_scores[query_index, frame_index] = float(np.mean(prob_map[obj_mask]))

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

    raise ValueError(f"Unsupported Cutie representative point mode: {mode}")


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


def _resolve_existing_checkpoint_file(checkpoint_path: Path) -> Path:
    resolved = checkpoint_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Cutie checkpoint not found: {resolved}")
    return resolved


def _resolve_cutie_python_executable(
    python_executable: str | Path | None,
) -> Path:
    if python_executable is not None:
        return Path(python_executable).expanduser().resolve()
    return Path(sys.executable).resolve()


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
        (int(round(center_x - (width + scale_dx) / 2.0)), int(round(center_y - (height + scale_dy) / 2.0)), width + scale_dx, height + scale_dy),
        (int(round(center_x - max(1, width - scale_dx) / 2.0)), int(round(center_y - max(1, height - scale_dy) / 2.0)), max(1, width - scale_dx), max(1, height - scale_dy)),
    ]
    unique: list[tuple[int, int, int, int]] = []
    for candidate in candidates:
        clamped = _clamp_roi_rect_xywh(candidate, image_shape)
        if clamped not in unique:
            unique.append(clamped)
    return unique


def _resolve_torch_device(torch_module: object, device: str) -> object:
    device_name = str(device or "auto").strip().lower()
    if device_name in {"", "auto", "gpu"}:
        if bool(torch_module.cuda.is_available()):  # type: ignore[attr-defined]
            return torch_module.device("cuda")  # type: ignore[attr-defined]
        return torch_module.device("cpu")  # type: ignore[attr-defined]
    if device_name == "cuda" and not bool(torch_module.cuda.is_available()):  # type: ignore[attr-defined]
        return torch_module.device("cpu")  # type: ignore[attr-defined]
    return torch_module.device(device_name)  # type: ignore[attr-defined]


def _make_autocast_context(torch_module: Any, device: Any) -> Any:
    if str(device).startswith("cuda"):
        return torch_module.autocast("cuda", dtype=torch_module.bfloat16)
    return nullcontext()


@contextmanager
def _cutie_import_context(repo_path: Optional[Path]) -> Iterator[None]:
    if repo_path is None:
        yield
        return

    repo_root = repo_path.resolve()
    repo_parent = str(repo_root.parent)
    repo_root_str = str(repo_root)
    removed_entries: list[tuple[int, str]] = []
    for index in reversed(range(len(sys.path))):
        entry = sys.path[index]
        try:
            if str(Path(entry).resolve()) == repo_parent:
                removed_entries.append((index, entry))
                del sys.path[index]
        except Exception:
            continue

    inserted_root = False
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
        inserted_root = True

    try:
        yield
    finally:
        if inserted_root:
            try:
                sys.path.remove(repo_root_str)
            except ValueError:
                pass
        for index, entry in sorted(removed_entries, key=lambda item: item[0]):
            sys.path.insert(index, entry)
