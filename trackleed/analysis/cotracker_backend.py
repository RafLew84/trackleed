"""CoTracker3 environment diagnostics, model loading and offline inference helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import importlib
import inspect
import sys

import numpy as np


DEFAULT_COTRACKER_MODELS_DIR = Path("tracking_models") / "cotracker3"
DEFAULT_COTRACKER_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "co-tracker",
    Path(__file__).resolve().parents[2] / "co-tracker",
)
OFFLINE_CHECKPOINT_PRIORITY = (
    "scaled_offline.pth",
    "baseline_offline.pth",
    "scaled_online.pth",
    "baseline_online.pth",
)
ONLINE_CHECKPOINT_NAMES = {
    "scaled_online.pth",
    "baseline_online.pth",
}
_MODEL_CACHE: dict[tuple[str, str, str, int], Any] = {}


@dataclass
class CotrackerEnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    available_checkpoints: list[Path] = field(default_factory=list)
    selected_checkpoint: Optional[Path] = None
    checkpoint_exists: bool = False
    torch_importable: bool = False
    torch_version: Optional[str] = None
    torchvision_importable: bool = False
    cuda_available: bool = False
    cotracker_importable: bool = False
    cotracker_module_path: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return (
            self.checkpoint_exists
            and self.torch_importable
            and self.torchvision_importable
            and self.cotracker_importable
        )


@dataclass
class CotrackerSmokeTestResult:
    tracks_shape: tuple[int, ...]
    visibility_shape: tuple[int, ...]
    first_track_point: list[float]
    visibility_dtype: str
    device: str
    checkpoint_name: str


@dataclass
class CotrackerRuntimeSettings:
    checkpoint_path: str | Path
    repo_path: str | Path | None = None
    device: str = "auto"
    inference_resolution: int = 256
    window_len: int = 60
    use_visibility_gating: bool = True


@dataclass
class CotrackerOfflineResult:
    tracks_xy: np.ndarray
    visible_mask: np.ndarray
    inference_resolution: tuple[int, int]
    original_resolution: tuple[int, int]
    checkpoint_name: str


def discover_cotracker_checkpoints(models_dir: str | Path | None = None) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_COTRACKER_MODELS_DIR
    if not directory.exists():
        return []
    return sorted(directory.glob("*.pth"), key=lambda path: path.name.lower())


def select_default_cotracker_checkpoint(checkpoints: list[Path]) -> Optional[Path]:
    if not checkpoints:
        return None
    by_name = {path.name: path for path in checkpoints}
    for preferred_name in OFFLINE_CHECKPOINT_PRIORITY:
        if preferred_name in by_name:
            return by_name[preferred_name]
    return checkpoints[0]


def probe_cotracker_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> CotrackerEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_COTRACKER_MODELS_DIR
    directory = directory.resolve()
    checkpoints = discover_cotracker_checkpoints(directory)
    selected_checkpoint = _resolve_checkpoint_path(checkpoint_path, checkpoints, directory)
    resolved_repo_path = _resolve_cotracker_repo_path(repo_path)

    status = CotrackerEnvironmentStatus(
        models_dir=directory,
        repo_path=resolved_repo_path,
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
            "No CoTracker checkpoint detected in tracking_models/cotracker3. "
            "Add a .pth checkpoint before running CoTracker3."
        )
    elif selected_checkpoint.name in ONLINE_CHECKPOINT_NAMES:
        status.warnings.append(
            f"Selected checkpoint looks online/sliding-window: {selected_checkpoint.name}. "
            "For stage 1 and offline tracking, prefer scaled_offline.pth or baseline_offline.pth."
        )

    if resolved_repo_path is None:
        status.warnings.append(
            "Could not resolve a local `co-tracker` repo. "
            "This is optional if the `cotracker` package is already installed in the active environment."
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
        status.hints.append("Install PyTorch in the active environment.")

    try:
        importlib.import_module("torchvision")
        status.torchvision_importable = True
    except Exception as exc:
        status.errors.append(f"TorchVision import failed: {exc}")
        status.hints.append("Install TorchVision in the active environment.")

    try:
        cotracker_module = _import_cotracker_predictor_module(resolved_repo_path)
        status.cotracker_importable = True
        status.cotracker_module_path = str(Path(inspect.getfile(cotracker_module)).resolve())
    except Exception as exc:
        status.errors.append(f"CoTracker import failed: {exc}")
        _append_cotracker_import_hints(status, exc)

    if status.cotracker_importable and status.torch_importable and not status.cuda_available:
        status.warnings.append(
            "PyTorch CUDA is not available. CoTracker3 can still run on CPU, "
            "but longer LEED sequences will be slow."
        )

    return status


def format_cotracker_environment_report(status: CotrackerEnvironmentStatus) -> str:
    selected_checkpoint = (
        str(status.selected_checkpoint) if status.selected_checkpoint is not None else "None"
    )
    available = ", ".join(path.name for path in status.available_checkpoints) or "none"
    lines = [
        "CoTracker3 environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Selected checkpoint: {selected_checkpoint}",
        f"- Available checkpoints: {available}",
        f"- PyTorch import: {'ok' if status.torch_importable else 'failed'}",
        f"- PyTorch version: {status.torch_version or 'unknown'}",
        f"- TorchVision import: {'ok' if status.torchvision_importable else 'failed'}",
        f"- CUDA available: {'yes' if status.cuda_available else 'no'}",
        f"- CoTracker import: {'ok' if status.cotracker_importable else 'failed'}",
    ]
    if status.cotracker_module_path:
        lines.append(f"- CoTracker module: {status.cotracker_module_path}")
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


def run_cotracker3_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: CotrackerRuntimeSettings,
) -> CotrackerOfflineResult:
    """
    Run offline CoTracker3 on a sequence of frames and sparse query points.

    Args:
        frames: [T, H, W] grayscale or [T, H, W, C] uint8/float frames.
        query_points_tyx: [Q, 3] points in [t, y, x] in the original frame grid.
        settings: CoTracker3 runtime settings including checkpoint and repo path.

    Returns:
        CotrackerOfflineResult with tracks in original image coordinates.
    """
    if frames is None or np.asarray(frames).size == 0:
        raise ValueError("No frames provided for CoTracker3 inference.")
    frames_np = np.asarray(frames)
    if frames_np.ndim not in (3, 4):
        raise ValueError(f"Expected frames with ndim 3 or 4, got shape {frames_np.shape}.")

    query_np = np.asarray(query_points_tyx, dtype=np.float32)
    if query_np.ndim != 2 or query_np.shape[1] != 3:
        raise ValueError(
            "query_points_tyx must have shape [num_queries, 3] in [t, y, x] format."
        )

    torch = importlib.import_module("torch")
    resolved_repo_path = _resolve_cotracker_repo_path(settings.repo_path)
    resolved_device = _resolve_torch_device(torch, settings.device)
    predictor = _load_cotracker_predictor(torch, settings, resolved_repo_path, resolved_device)

    prepared_frames, original_resolution = _prepare_frames_rgb(frames_np)
    resized_frames = _resize_video_numpy(prepared_frames, settings.inference_resolution)
    resized_resolution = (int(resized_frames.shape[1]), int(resized_frames.shape[2]))

    scaled_query_points = _scale_query_points_tyx(
        query_np,
        source_resolution=original_resolution,
        target_resolution=resized_resolution,
    )
    predictor_query_points = _convert_query_points_tyx_to_txy(scaled_query_points)

    video_tensor = (
        torch.tensor(resized_frames, dtype=torch.float32, device=resolved_device)
        .permute(0, 3, 1, 2)[None]
    )
    query_tensor = torch.tensor(
        predictor_query_points,
        dtype=torch.float32,
        device=resolved_device,
    )[None]

    with torch.no_grad():
        pred_tracks, pred_visibility = predictor(video_tensor, queries=query_tensor)

    tracks_txy = pred_tracks.detach().to(dtype=torch.float32).cpu().numpy()[0]
    visibility_txq = pred_visibility.detach().cpu().numpy()[0]

    tracks_xy = np.transpose(tracks_txy, (1, 0, 2))
    visible_mask = np.transpose(visibility_txq, (1, 0)).astype(bool, copy=False)
    if not bool(settings.use_visibility_gating):
        visible_mask = np.ones_like(visible_mask, dtype=bool)

    tracks_xy = _scale_tracks_xy(
        tracks_xy,
        source_resolution=resized_resolution,
        target_resolution=original_resolution,
    )

    return CotrackerOfflineResult(
        tracks_xy=tracks_xy,
        visible_mask=visible_mask,
        inference_resolution=resized_resolution,
        original_resolution=original_resolution,
        checkpoint_name=Path(settings.checkpoint_path).name,
    )


def run_cotracker_smoke_test(
    *,
    checkpoint_path: str | Path,
    repo_path: str | Path | None = None,
    device: str = "auto",
) -> CotrackerSmokeTestResult:
    torch = importlib.import_module("torch")
    predictor_module = _import_cotracker_predictor_module(
        _resolve_cotracker_repo_path(repo_path)
    )
    CoTrackerPredictor = getattr(predictor_module, "CoTrackerPredictor")

    resolved_device = _resolve_torch_device(torch, device)
    predictor = CoTrackerPredictor(
        checkpoint=str(Path(checkpoint_path).resolve()),
        offline=True,
        v2=False,
        window_len=60,
    ).to(resolved_device)
    predictor.eval()

    video = torch.zeros((1, 4, 3, 64, 64), dtype=torch.float32, device=resolved_device)
    queries = torch.tensor(
        [[[0.0, 32.0, 32.0]]],  # (t, x, y)
        dtype=torch.float32,
        device=resolved_device,
    )

    with torch.no_grad():
        pred_tracks, pred_visibility = predictor(video, queries=queries)

    tracks = pred_tracks.detach().to(dtype=torch.float32).cpu().numpy()
    visibility = pred_visibility.detach().cpu().numpy()

    return CotrackerSmokeTestResult(
        tracks_shape=tuple(int(dim) for dim in tracks.shape),
        visibility_shape=tuple(int(dim) for dim in visibility.shape),
        first_track_point=[float(value) for value in tracks[0, 0, 0].tolist()],
        visibility_dtype=str(pred_visibility.dtype),
        device=resolved_device,
        checkpoint_name=Path(checkpoint_path).name,
    )


def _load_cotracker_predictor(
    torch_module: Any,
    settings: CotrackerRuntimeSettings,
    repo_root: Path | None,
    device: str,
) -> Any:
    checkpoint_path = Path(settings.checkpoint_path).resolve()
    cache_key = (
        str(repo_root) if repo_root is not None else "<installed>",
        str(checkpoint_path),
        str(device),
        int(max(4, settings.window_len)),
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CoTracker3 checkpoint not found: {checkpoint_path}")

    predictor_module = _import_cotracker_predictor_module(repo_root)
    CoTrackerPredictor = getattr(predictor_module, "CoTrackerPredictor")
    predictor = CoTrackerPredictor(
        checkpoint=str(checkpoint_path),
        offline=True,
        v2=False,
        window_len=int(max(4, settings.window_len)),
    )
    predictor = predictor.to(device)
    predictor = predictor.eval()
    _MODEL_CACHE[cache_key] = predictor
    return predictor


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoints: list[Path],
    models_dir: Path,
) -> Optional[Path]:
    if checkpoint_path is None:
        return select_default_cotracker_checkpoint(checkpoints)

    candidate = Path(checkpoint_path)
    if not candidate.is_absolute():
        candidate = (models_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _resolve_cotracker_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidate_paths: list[Path] = []
    if repo_path is not None:
        candidate_paths.append(Path(repo_path).expanduser())
    else:
        candidate_paths.extend(DEFAULT_COTRACKER_REPO_CANDIDATES)

    for candidate in candidate_paths:
        normalized = _normalize_cotracker_repo_candidate(candidate)
        if normalized is not None:
            return normalized
    return None


def _normalize_cotracker_repo_candidate(candidate: Path) -> Optional[Path]:
    candidate = candidate.resolve()
    if (candidate / "cotracker" / "predictor.py").exists():
        return candidate
    if (candidate / "predictor.py").exists() and candidate.name == "cotracker":
        return candidate.parent.resolve()
    return None


def _import_cotracker_predictor_module(repo_root: Path | None):
    if repo_root is not None:
        repo_root_str = str(repo_root.resolve())
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
    return importlib.import_module("cotracker.predictor")


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
        raise ValueError(f"Invalid CoTracker3 inference resolution: {(target_h, target_w)}")

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


def _convert_query_points_tyx_to_txy(query_points_tyx: np.ndarray) -> np.ndarray:
    query_points_txy = np.array(query_points_tyx, dtype=np.float32, copy=True)
    query_points_txy[:, 1] = query_points_tyx[:, 2]
    query_points_txy[:, 2] = query_points_tyx[:, 1]
    return query_points_txy


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


def _append_cotracker_import_hints(status: CotrackerEnvironmentStatus, exc: Exception) -> None:
    message = str(exc).lower()
    if "no module named 'cotracker'" in message:
        status.hints.append(
            "Install CoTracker into the active environment, e.g. `pip install -e <path-to-co-tracker>`."
        )
    elif "torchvision" in message:
        status.hints.append("Install TorchVision in the active environment.")
    else:
        status.hints.append(
            "Verify that the local `co-tracker` repo is available or that the `cotracker` package "
            "is installed in the active environment."
        )


def _resolve_torch_device(torch_module: Any, device: str | None) -> str:
    normalized = str(device or "auto").strip().lower()
    if normalized in {"gpu", "cuda"}:
        return "cuda" if bool(torch_module.cuda.is_available()) else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda") and not bool(torch_module.cuda.is_available()):
        return "cpu"
    if normalized == "auto":
        try:
            return "cuda" if bool(torch_module.cuda.is_available()) else "cpu"
        except Exception:
            return "cpu"
    return normalized
