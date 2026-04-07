"""TAPNext environment diagnostics and smoke-test helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import importlib
import inspect
import sys

import numpy as np


DEFAULT_TAPNEXT_MODELS_DIR = Path("tracking_models") / "tapnext"
DEFAULT_TAPNET_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "tapnet",
    Path(__file__).resolve().parents[2] / "tapnet",
)
CHECKPOINT_PRIORITY = (
    "tapnextpp_ckpt.pt",
    "bootstapnext_ckpt.npz",
    "tapnext_ckpt.npz",
)
DEFAULT_IMAGE_SIZE = (256, 256)
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


@dataclass
class TapnextEnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    available_checkpoints: list[Path] = field(default_factory=list)
    selected_checkpoint: Optional[Path] = None
    checkpoint_exists: bool = False
    torch_importable: bool = False
    torch_version: Optional[str] = None
    torchvision_importable: bool = False
    cuda_available: bool = False
    tapnext_importable: bool = False
    tapnext_utils_importable: bool = False
    tapnext_module_path: Optional[str] = None
    tapnext_utils_module_path: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return (
            self.checkpoint_exists
            and self.torch_importable
            and self.torchvision_importable
            and self.tapnext_importable
            and self.tapnext_utils_importable
        )


@dataclass
class TapnextSmokeTestResult:
    tracks_shape: tuple[int, ...]
    track_logits_shape: tuple[int, ...]
    visible_logits_shape: tuple[int, ...]
    certainty_shape: tuple[int, ...]
    first_track_point: list[float]
    device: str
    checkpoint_name: str
    checkpoint_variant: str


@dataclass
class TapnextRuntimeSettings:
    checkpoint_path: str | Path
    repo_path: str | Path | None = None
    device: str = "auto"
    inference_resolution: int | tuple[int, int] = 256
    use_visibility_gating: bool = True
    visibility_threshold: float = 0.5
    use_certainty_gating: bool = False
    certainty_threshold: float = 0.5
    autocast: bool = False


@dataclass
class TapnextOfflineResult:
    tracks_xy: np.ndarray
    visible_mask: np.ndarray
    visible_logits: np.ndarray
    visibility_scores: np.ndarray
    track_logits: np.ndarray
    certainty_scores: np.ndarray
    combined_scores: np.ndarray
    inference_resolution: tuple[int, int]
    original_resolution: tuple[int, int]
    checkpoint_name: str
    checkpoint_variant: str


def discover_tapnext_checkpoints(models_dir: str | Path | None = None) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_TAPNEXT_MODELS_DIR
    if not directory.exists():
        return []
    checkpoints = []
    for pattern in ("*.pt", "*.npz"):
        checkpoints.extend(directory.glob(pattern))
    return sorted(set(checkpoints), key=lambda path: path.name.lower())


def select_default_tapnext_checkpoint(checkpoints: list[Path]) -> Optional[Path]:
    if not checkpoints:
        return None
    by_name = {path.name: path for path in checkpoints}
    for preferred_name in CHECKPOINT_PRIORITY:
        if preferred_name in by_name:
            return by_name[preferred_name]
    return checkpoints[0]


def probe_tapnext_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> TapnextEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_TAPNEXT_MODELS_DIR
    directory = directory.resolve()
    checkpoints = discover_tapnext_checkpoints(directory)
    selected_checkpoint = _resolve_checkpoint_path(checkpoint_path, checkpoints, directory)
    resolved_repo_path = _resolve_tapnet_repo_path(repo_path)

    status = TapnextEnvironmentStatus(
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
            "No TAPNext checkpoint detected in tracking_models/tapnext. "
            "Add tapnextpp_ckpt.pt or a TAPNext .npz checkpoint before running diagnostics."
        )

    if resolved_repo_path is None:
        status.warnings.append(
            "Could not resolve a local `tapnet` repo. "
            "This is optional if the `tapnet` package is already installed in the active environment."
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
        tapnext_module = _import_tapnext_module(resolved_repo_path)
        status.tapnext_importable = True
        status.tapnext_module_path = str(Path(inspect.getfile(tapnext_module)).resolve())
    except Exception as exc:
        status.errors.append(f"TAPNext import failed: {exc}")
        _append_tapnext_import_hints(status, exc)

    try:
        tapnext_utils_module = _import_tapnext_utils_module(resolved_repo_path)
        status.tapnext_utils_importable = True
        status.tapnext_utils_module_path = str(Path(inspect.getfile(tapnext_utils_module)).resolve())
    except Exception as exc:
        status.errors.append(f"TAPNext utils import failed: {exc}")
        _append_tapnext_import_hints(status, exc)

    if status.tapnext_importable and status.torch_importable and not status.cuda_available:
        status.warnings.append(
            "PyTorch CUDA is not available. TAPNext can still run on CPU, "
            "but longer LEED sequences will be slow."
        )

    return status


def format_tapnext_environment_report(status: TapnextEnvironmentStatus) -> str:
    selected_checkpoint = (
        str(status.selected_checkpoint) if status.selected_checkpoint is not None else "None"
    )
    available = ", ".join(path.name for path in status.available_checkpoints) or "none"
    lines = [
        "TAPNext environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Selected checkpoint: {selected_checkpoint}",
        f"- Available checkpoints: {available}",
        f"- PyTorch import: {'ok' if status.torch_importable else 'failed'}",
        f"- PyTorch version: {status.torch_version or 'unknown'}",
        f"- TorchVision import: {'ok' if status.torchvision_importable else 'failed'}",
        f"- CUDA available: {'yes' if status.cuda_available else 'no'}",
        f"- TAPNext import: {'ok' if status.tapnext_importable else 'failed'}",
        f"- TAPNext utils import: {'ok' if status.tapnext_utils_importable else 'failed'}",
    ]
    if status.tapnext_module_path:
        lines.append(f"- TAPNext module: {status.tapnext_module_path}")
    if status.tapnext_utils_module_path:
        lines.append(f"- TAPNext utils module: {status.tapnext_utils_module_path}")
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


def run_tapnext_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: TapnextRuntimeSettings,
) -> TapnextOfflineResult:
    """
    Run TAPNext sequentially on a full sequence and sparse query points.

    Args:
        frames: [T, H, W] grayscale or [T, H, W, C] uint8/float frames.
        query_points_tyx: [Q, 3] points in [t, y, x] in the original frame grid.
        settings: TAPNext runtime settings including checkpoint and gating.

    Returns:
        TapnextOfflineResult with tracks in original image coordinates.
    """
    if frames is None or np.asarray(frames).size == 0:
        raise ValueError("No frames provided for TAPNext inference.")
    frames_np = np.asarray(frames)
    if frames_np.ndim not in (3, 4):
        raise ValueError(f"Expected frames with ndim 3 or 4, got shape {frames_np.shape}.")

    query_np = np.asarray(query_points_tyx, dtype=np.float32)
    if query_np.ndim != 2 or query_np.shape[1] != 3:
        raise ValueError(
            "query_points_tyx must have shape [num_queries, 3] in [t, y, x] format."
        )

    torch = importlib.import_module("torch")
    resolved_device = _resolve_torch_device(torch, settings.device)
    resolved_repo_path = _resolve_tapnet_repo_path(settings.repo_path)
    tapnext_module = _import_tapnext_module(resolved_repo_path)
    tapnext_utils = _import_tapnext_utils_module(resolved_repo_path)

    model = _load_tapnext_model(
        tapnext_module=tapnext_module,
        tapnext_utils_module=tapnext_utils,
        checkpoint_path=settings.checkpoint_path,
        device=resolved_device,
        torch_module=torch,
    )

    prepared_frames, original_resolution = _prepare_frames_rgb(frames_np)
    resized_frames = _resize_video_numpy(prepared_frames, settings.inference_resolution)
    resized_resolution = (int(resized_frames.shape[1]), int(resized_frames.shape[2]))
    scaled_query_points = _scale_query_points_tyx(
        query_np,
        source_resolution=original_resolution,
        target_resolution=resized_resolution,
    )

    video_tensor = torch.tensor(
        resized_frames,
        dtype=torch.float32,
        device=resolved_device,
    )[None]
    query_tensor = torch.tensor(
        scaled_query_points,
        dtype=torch.float32,
        device=resolved_device,
    )[None]

    pred_tracks = []
    pred_track_logits = []
    pred_visible_logits = []
    state = None

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if bool(getattr(settings, "autocast", False)) and str(resolved_device).startswith("cuda")
        else _nullcontext()
    )

    with torch.no_grad():
        with autocast_context:
            for frame_idx in range(video_tensor.shape[1]):
                frame_video = video_tensor[:, frame_idx : frame_idx + 1]
                if frame_idx == 0:
                    tracks, track_logits, visible_logits, state = model(
                        video=frame_video,
                        query_points=query_tensor,
                    )
                else:
                    tracks, track_logits, visible_logits, state = model(
                        video=frame_video,
                        state=state,
                    )
                pred_tracks.append(tracks.detach().to(dtype=torch.float32).cpu())
                pred_track_logits.append(track_logits.detach().to(dtype=torch.float32).cpu())
                pred_visible_logits.append(visible_logits.detach().to(dtype=torch.float32).cpu())

    tracks = torch.cat(pred_tracks, dim=1).transpose(1, 2)
    track_logits = torch.cat(pred_track_logits, dim=1).transpose(1, 2)
    visible_logits = torch.cat(pred_visible_logits, dim=1).transpose(1, 2)
    certainty = tapnext_utils.tracker_certainty(tracks, track_logits).to(dtype=torch.float32)
    visibility_scores = torch.sigmoid(visible_logits)
    combined_scores = visibility_scores * certainty

    visible_mask = _postprocess_visibility(
        visibility_scores=visibility_scores[0, ..., 0].cpu().numpy(),
        certainty_scores=certainty[0, ..., 0].cpu().numpy(),
        use_visibility_gating=bool(settings.use_visibility_gating),
        visibility_threshold=float(settings.visibility_threshold),
        use_certainty_gating=bool(settings.use_certainty_gating),
        certainty_threshold=float(settings.certainty_threshold),
    )

    tracks_xy = _scale_tracks_xy(
        tracks[0].cpu().numpy(),
        source_resolution=resized_resolution,
        target_resolution=original_resolution,
    )

    return TapnextOfflineResult(
        tracks_xy=tracks_xy,
        visible_mask=visible_mask,
        visible_logits=visible_logits[0, ..., 0].cpu().numpy(),
        visibility_scores=visibility_scores[0, ..., 0].cpu().numpy(),
        track_logits=track_logits[0].cpu().numpy(),
        certainty_scores=certainty[0, ..., 0].cpu().numpy(),
        combined_scores=combined_scores[0, ..., 0].cpu().numpy(),
        inference_resolution=resized_resolution,
        original_resolution=original_resolution,
        checkpoint_name=Path(settings.checkpoint_path).name,
        checkpoint_variant=_infer_tapnext_variant(Path(settings.checkpoint_path)),
    )


def run_tapnext_smoke_test(
    *,
    checkpoint_path: str | Path,
    repo_path: str | Path | None = None,
    device: str = "auto",
) -> TapnextSmokeTestResult:
    torch = importlib.import_module("torch")
    resolved_device = _resolve_torch_device(torch, device)
    resolved_repo_path = _resolve_tapnet_repo_path(repo_path)
    tapnext_module = _import_tapnext_module(resolved_repo_path)
    tapnext_utils = _import_tapnext_utils_module(resolved_repo_path)

    model = _load_tapnext_model(
        tapnext_module=tapnext_module,
        tapnext_utils_module=tapnext_utils,
        checkpoint_path=checkpoint_path,
        device=resolved_device,
        torch_module=torch,
    )

    video = torch.zeros((1, 4, DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 3), device=resolved_device)
    query_points = torch.tensor([[[0.0, 128.0, 128.0]]], device=resolved_device)

    pred_tracks = []
    pred_track_logits = []
    pred_visible_logits = []
    state = None

    with torch.no_grad():
        for frame_idx in range(video.shape[1]):
            frame_video = video[:, frame_idx : frame_idx + 1]
            if frame_idx == 0:
                tracks, track_logits, visible_logits, state = model(
                    video=frame_video,
                    query_points=query_points,
                )
            else:
                tracks, track_logits, visible_logits, state = model(
                    video=frame_video,
                    state=state,
                )
            pred_tracks.append(tracks.detach().cpu())
            pred_track_logits.append(track_logits.detach().cpu())
            pred_visible_logits.append(visible_logits.detach().cpu())

    tracks = torch.cat(pred_tracks, dim=1).transpose(1, 2)
    track_logits = torch.cat(pred_track_logits, dim=1).transpose(1, 2)
    visible_logits = torch.cat(pred_visible_logits, dim=1).transpose(1, 2)
    certainty = tapnext_utils.tracker_certainty(tracks, track_logits).detach().cpu()

    return TapnextSmokeTestResult(
        tracks_shape=tuple(int(v) for v in tracks.shape),
        track_logits_shape=tuple(int(v) for v in track_logits.shape),
        visible_logits_shape=tuple(int(v) for v in visible_logits.shape),
        certainty_shape=tuple(int(v) for v in certainty.shape),
        first_track_point=[float(v) for v in tracks[0, 0, 0].tolist()],
        device=str(resolved_device),
        checkpoint_name=Path(checkpoint_path).name,
        checkpoint_variant=_infer_tapnext_variant(Path(checkpoint_path)),
    )


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    available_checkpoints: list[Path],
    models_dir: Path,
) -> Optional[Path]:
    if checkpoint_path:
        candidate = Path(checkpoint_path)
        if not candidate.is_absolute():
            candidate = models_dir / candidate
        return candidate.resolve()
    return select_default_tapnext_checkpoint(available_checkpoints)


def _resolve_tapnet_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidates = []
    if repo_path:
        candidates.append(Path(repo_path))
    candidates.extend(DEFAULT_TAPNET_REPO_CANDIDATES)
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "tapnet").exists():
            return candidate
    return None


def _ensure_repo_parent_on_sys_path(repo_path: Optional[Path]) -> None:
    if repo_path is None:
        return
    parent = str(repo_path.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)


def _import_tapnext_module(repo_path: Optional[Path]):
    _ensure_repo_parent_on_sys_path(repo_path)
    return importlib.import_module("tapnet.tapnext.tapnext_torch")


def _import_tapnext_utils_module(repo_path: Optional[Path]):
    _ensure_repo_parent_on_sys_path(repo_path)
    return importlib.import_module("tapnet.tapnext.tapnext_torch_utils")


def _append_tapnext_import_hints(status: TapnextEnvironmentStatus, exc: Exception) -> None:
    message = str(exc).lower()
    if "no module named 'tapnet'" in message:
        status.hints.append(
            "Install the local tapnet clone or pass --repo-path pointing at the tapnet repo root."
        )
    if "no module named 'einops'" in message:
        status.hints.append("Install `einops` in the active environment.")
    if "no module named 'torchvision'" in message:
        status.hints.append("Install `torchvision` in the active environment.")


def _resolve_torch_device(torch_module, requested_device: str):
    device = str(requested_device or "auto").strip().lower()
    if device in {"auto", "gpu"}:
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if device == "cuda" and not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    return torch_module.device(device)


def _load_tapnext_model(
    *,
    tapnext_module: Any,
    tapnext_utils_module: Any,
    checkpoint_path: str | Path,
    device,
    torch_module,
):
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"TAPNext checkpoint not found: {checkpoint}")

    cache_key = (
        str(checkpoint),
        str(device),
        str(Path(inspect.getfile(tapnext_module)).resolve()),
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model = tapnext_module.TAPNext(image_size=DEFAULT_IMAGE_SIZE)
    suffix = checkpoint.suffix.lower()

    if suffix == ".pt":
        ckpt = torch_module.load(str(checkpoint), map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = {
            key.replace("tapnext.", "", 1): value for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict)
    elif suffix == ".npz":
        model = tapnext_utils_module.restore_model_from_jax_checkpoint(model, str(checkpoint))
    else:
        raise ValueError(f"Unsupported TAPNext checkpoint format: {checkpoint.suffix}")

    model.to(device)
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _infer_tapnext_variant(checkpoint_path: Path) -> str:
    name = checkpoint_path.name.lower()
    if "tapnextpp" in name:
        return "tapnextpp"
    if "bootstapnext" in name:
        return "bootstapnext"
    if "tapnext" in name:
        return "tapnext"
    return "unknown"


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
    import cv2

    if isinstance(target, int):
        target_h = target_w = int(target)
    else:
        target_h, target_w = int(target[0]), int(target[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid TAPNext inference resolution: {(target_h, target_w)}")

    if frames.shape[1] == target_h and frames.shape[2] == target_w:
        return frames

    resized = [
        cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
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
    scale_y = dst_h / max(1, src_h)
    scale_x = dst_w / max(1, src_w)

    scaled = np.array(query_points_tyx, dtype=np.float32, copy=True)
    scaled[:, 1] *= scale_y
    scaled[:, 2] *= scale_x
    return scaled


def _scale_tracks_xy(
    tracks_xy: np.ndarray,
    source_resolution: tuple[int, int],
    target_resolution: tuple[int, int],
) -> np.ndarray:
    src_h, src_w = source_resolution
    dst_h, dst_w = target_resolution
    scale_x = dst_w / max(1, src_w)
    scale_y = dst_h / max(1, src_h)

    scaled = np.array(tracks_xy, dtype=np.float32, copy=True)
    scaled[..., 0] *= scale_x
    scaled[..., 1] *= scale_y
    return scaled


def _postprocess_visibility(
    *,
    visibility_scores: np.ndarray,
    certainty_scores: np.ndarray,
    use_visibility_gating: bool,
    visibility_threshold: float,
    use_certainty_gating: bool,
    certainty_threshold: float,
) -> np.ndarray:
    visible = np.ones_like(visibility_scores, dtype=bool)
    if use_visibility_gating:
        visible &= visibility_scores >= float(visibility_threshold)
    if use_certainty_gating:
        visible &= certainty_scores >= float(certainty_threshold)
    return visible


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
