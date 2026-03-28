"""TAPIR environment diagnostics, model loading and offline inference helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import importlib
import inspect

import numpy as np


DEFAULT_TAPIR_MODELS_DIR = Path("tracking_models") / "tapir"
OFFLINE_CHECKPOINT_PRIORITY = (
    "bootstapir_checkpoint_v2.pt",
    "tapir_checkpoint_panning.pt",
)
ONLINE_CHECKPOINT_NAMES = {
    "causal_bootstapir_checkpoint.pt",
    "causal_tapir_checkpoint.pt",
}


@dataclass
class TapirEnvironmentStatus:
    models_dir: Path
    available_checkpoints: list[Path] = field(default_factory=list)
    selected_checkpoint: Optional[Path] = None
    checkpoint_exists: bool = False
    torch_importable: bool = False
    torch_version: Optional[str] = None
    cuda_available: bool = False
    tapir_importable: bool = False
    tapir_module_path: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.torch_importable and self.tapir_importable and self.checkpoint_exists


@dataclass
class TapirRuntimeSettings:
    checkpoint_path: str | Path
    device: str = "auto"
    inference_resolution: int = 256
    query_chunk_size: int = 64
    use_visibility_gating: bool = True
    visibility_threshold: float = 0.5
    use_uncertainty_gating: bool = True
    expected_dist_threshold: float = 0.5


@dataclass
class TapirOfflineResult:
    tracks_xy: np.ndarray
    visible_mask: np.ndarray
    occlusion_logits: np.ndarray
    expected_dist_logits: np.ndarray
    inference_resolution: tuple[int, int]
    original_resolution: tuple[int, int]


_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def discover_tapir_checkpoints(models_dir: str | Path | None = None) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_TAPIR_MODELS_DIR
    if not directory.exists():
        return []
    return sorted(directory.glob("*.pt"), key=lambda path: path.name.lower())


def select_default_tapir_checkpoint(checkpoints: list[Path]) -> Optional[Path]:
    if not checkpoints:
        return None
    by_name = {path.name: path for path in checkpoints}
    for preferred_name in OFFLINE_CHECKPOINT_PRIORITY:
        if preferred_name in by_name:
            return by_name[preferred_name]
    return checkpoints[0]


def probe_tapir_environment(
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> TapirEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_TAPIR_MODELS_DIR
    directory = directory.resolve()
    checkpoints = discover_tapir_checkpoints(directory)
    selected_checkpoint = _resolve_checkpoint_path(checkpoint_path, checkpoints, directory)

    status = TapirEnvironmentStatus(
        models_dir=directory,
        available_checkpoints=checkpoints,
        selected_checkpoint=selected_checkpoint,
        checkpoint_exists=bool(selected_checkpoint and selected_checkpoint.exists()),
    )

    if not directory.exists():
        status.warnings.append(f"Checkpoint directory does not exist: {directory}")

    if checkpoint_path and not status.checkpoint_exists:
        status.errors.append(f"Selected checkpoint not found: {selected_checkpoint}")
    elif selected_checkpoint is None:
        status.warnings.append(
            "No TAPIR checkpoint detected in tracking_models/tapir. "
            "Add an offline .pt checkpoint before running TAPIR."
        )
    elif selected_checkpoint.name in ONLINE_CHECKPOINT_NAMES:
        status.warnings.append(
            f"Selected checkpoint looks causal/online: {selected_checkpoint.name}. "
            "For stage 1 and offline tracking, prefer BootsTAPIR or TAPIR offline checkpoints."
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
        status.hints.append("Install PyTorch in the active environment, e.g. `pip install torch torchvision`.")

    try:
        tapir_module = importlib.import_module("tapnet.torch.tapir_model")
        status.tapir_importable = True
        status.tapir_module_path = str(Path(inspect.getfile(tapir_module)).resolve())
    except Exception as exc:
        status.errors.append(f"TAPIR import failed: {exc}")
        _append_tapir_import_hints(status, exc)

    if status.tapir_importable and status.torch_importable and not status.cuda_available:
        status.warnings.append(
            "PyTorch CUDA is not available. TAPIR can still run on CPU, but longer LEED sequences will be slow."
        )

    return status


def run_tapir_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: TapirRuntimeSettings,
) -> TapirOfflineResult:
    """
    Run offline PyTorch TAPIR on a sequence of frames.

    Args:
        frames: [T, H, W] grayscale or [T, H, W, C] uint8/float frames.
        query_points_tyx: [Q, 3] points in [t, y, x] in the original frame grid.
        settings: TAPIR runtime and checkpoint settings.

    Returns:
        TapirOfflineResult with tracks in original image coordinates.
    """
    if frames is None or np.asarray(frames).size == 0:
        raise ValueError("No frames provided for TAPIR inference.")
    frames_np = np.asarray(frames)
    if frames_np.ndim not in (3, 4):
        raise ValueError(f"Expected frames with ndim 3 or 4, got shape {frames_np.shape}.")

    query_np = np.asarray(query_points_tyx, dtype=np.float32)
    if query_np.ndim != 2 or query_np.shape[1] != 3:
        raise ValueError(
            "query_points_tyx must have shape [num_queries, 3] in [t, y, x] format."
        )

    torch = importlib.import_module("torch")

    device = _resolve_torch_device(torch, settings.device)
    model = _load_tapir_model(torch, settings, device)

    prepared_frames, original_resolution = _prepare_frames_rgb(frames_np)
    resized_frames = _resize_video_numpy(prepared_frames, settings.inference_resolution)
    resized_resolution = (int(resized_frames.shape[1]), int(resized_frames.shape[2]))
    scaled_query_points = _scale_query_points_tyx(
        query_np,
        source_resolution=original_resolution,
        target_resolution=resized_resolution,
    )

    frames_tensor = torch.tensor(resized_frames, dtype=torch.float32, device=device)
    query_tensor = torch.tensor(scaled_query_points, dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(
            _preprocess_frames_tensor(frames_tensor)[None],
            query_tensor[None],
            query_chunk_size=int(max(1, settings.query_chunk_size)),
        )

    tracks = outputs["tracks"][0].detach().cpu().numpy()
    occlusion = outputs["occlusion"][0].detach().cpu().numpy()
    expected_dist = outputs["expected_dist"][0].detach().cpu().numpy()

    visible_mask = _postprocess_visibility(
        occlusion_logits=occlusion,
        expected_dist_logits=expected_dist,
        use_visibility_gating=bool(settings.use_visibility_gating),
        visibility_threshold=float(settings.visibility_threshold),
        use_uncertainty_gating=bool(settings.use_uncertainty_gating),
        expected_dist_threshold=float(settings.expected_dist_threshold),
    )

    tracks_xy = _scale_tracks_xy(
        tracks,
        source_resolution=resized_resolution,
        target_resolution=original_resolution,
    )

    return TapirOfflineResult(
        tracks_xy=tracks_xy,
        visible_mask=visible_mask,
        occlusion_logits=occlusion,
        expected_dist_logits=expected_dist,
        inference_resolution=resized_resolution,
        original_resolution=original_resolution,
    )


def format_tapir_environment_report(status: TapirEnvironmentStatus) -> str:
    selected_checkpoint = (
        str(status.selected_checkpoint) if status.selected_checkpoint is not None else "None"
    )
    available = ", ".join(path.name for path in status.available_checkpoints) or "none"
    lines = [
        "TAPIR environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Models dir: {status.models_dir}",
        f"- Selected checkpoint: {selected_checkpoint}",
        f"- Available checkpoints: {available}",
        f"- PyTorch import: {'ok' if status.torch_importable else 'failed'}",
        f"- PyTorch version: {status.torch_version or 'unknown'}",
        f"- CUDA available: {'yes' if status.cuda_available else 'no'}",
        f"- TAPIR import: {'ok' if status.tapir_importable else 'failed'}",
    ]
    if status.tapir_module_path:
        lines.append(f"- TAPIR module: {status.tapir_module_path}")
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


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoints: list[Path],
    models_dir: Path,
) -> Optional[Path]:
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = models_dir / path
        return path.resolve()
    selected = select_default_tapir_checkpoint(checkpoints)
    return selected.resolve() if selected is not None else None


def _append_tapir_import_hints(status: TapirEnvironmentStatus, exc: Exception) -> None:
    message = str(exc)
    if "No module named 'tapnet'" in message:
        status.hints.append(
            "Install tapnet in the active environment, e.g. from a local clone via `pip install -e . --no-deps`."
        )
        status.hints.append(
            "Then install the PyTorch-side runtime packages, at minimum: `pip install torch torchvision dm-tree einshape typing_extensions`."
        )
        return
    if "chex" in message or "jax" in message or "dm-haiku" in message:
        status.hints.append(
            "Your tapnet install is trying to import the JAX stack. Use the PyTorch import path "
            "`tapnet.torch.tapir_model` and make sure the package resolves cleanly in the active env."
        )
        status.hints.append(
            "If your local tapnet clone imports JAX modules from `tapnet/__init__.py`, patch that file "
            "or install the full JAX dependency set."
        )
        return
    status.hints.append(
        "Verify that `python -c \"from tapnet.torch import tapir_model\"` succeeds in the same environment as Trackleed."
    )


def _resolve_torch_device(torch_module: Any, requested_device: str) -> Any:
    dev = (requested_device or "auto").lower()
    if dev in {"auto", "gpu", "cuda"}:
        if bool(torch_module.cuda.is_available()):
            return torch_module.device("cuda")
        return torch_module.device("cpu")
    return torch_module.device("cpu")


def _load_tapir_model(torch_module: Any, settings: TapirRuntimeSettings, device: Any) -> Any:
    checkpoint_path = Path(settings.checkpoint_path).resolve()
    cache_key = (str(checkpoint_path), str(device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"TAPIR checkpoint not found: {checkpoint_path}")

    tapir_model_module = importlib.import_module("tapnet.torch.tapir_model")
    model = tapir_model_module.TAPIR(pyramid_level=1)
    state_dict = torch_module.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model = model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _prepare_frames_rgb(frames: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    frames_np = np.asarray(frames)
    if frames_np.ndim == 3:
        frames_np = np.stack([frames_np] * 3, axis=-1)
    elif frames_np.ndim == 4 and frames_np.shape[-1] == 1:
        frames_np = np.repeat(frames_np, 3, axis=-1)
    elif frames_np.ndim == 4 and frames_np.shape[-1] >= 3:
        frames_np = frames_np[..., :3]

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


def _resize_video_numpy(frames: np.ndarray, target: int | tuple[int, int]) -> np.ndarray:
    cv2 = importlib.import_module("cv2")
    if isinstance(target, int):
        target_h = int(target)
        target_w = int(target)
    else:
        target_h = int(target[0])
        target_w = int(target[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid TAPIR inference resolution: {(target_h, target_w)}")

    if frames.shape[1] == target_h and frames.shape[2] == target_w:
        return frames

    resized = [
        cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]
    return np.stack(resized, axis=0)


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


def _preprocess_frames_tensor(frames_tensor: Any) -> Any:
    return frames_tensor / 255.0 * 2.0 - 1.0


def _postprocess_visibility(
    occlusion_logits: np.ndarray,
    expected_dist_logits: np.ndarray,
    use_visibility_gating: bool,
    visibility_threshold: float,
    use_uncertainty_gating: bool,
    expected_dist_threshold: float,
) -> np.ndarray:
    torch = importlib.import_module("torch")
    occl = torch.tensor(occlusion_logits, dtype=torch.float32)
    expd = torch.tensor(expected_dist_logits, dtype=torch.float32)
    visible_prob = 1.0 - torch.sigmoid(occl)
    confidence = visible_prob.clone()
    if use_uncertainty_gating:
        confidence = confidence * (1.0 - torch.sigmoid(expd))

    mask = torch.ones_like(confidence, dtype=torch.bool)
    if use_visibility_gating:
        mask = mask & (visible_prob >= float(visibility_threshold))
    if use_uncertainty_gating:
        mask = mask & (1.0 - torch.sigmoid(expd) >= float(expected_dist_threshold))
    return mask.detach().cpu().numpy().astype(bool)
