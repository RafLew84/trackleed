"""Track-On-R environment diagnostics and smoke-test helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import importlib
import inspect
import sys

import numpy as np


DEFAULT_TRACKONR_MODELS_DIR = Path("tracking_models") / "trackonr"
DEFAULT_TRACKONR_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "track_on",
    Path(__file__).resolve().parents[2] / "track_on",
)
CHECKPOINT_PRIORITY = (
    "track_on_r.pt",
    "track_on_r.pth",
    "trackonr_kubric_real_world.pth",
    "trackonr_kubric_real_world.pt",
)
DINO_CACHE_DIRNAME = "models--facebook--dinov3-vits16plus-pretrain-lvd1689m"
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


@dataclass
class TrackonrEnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    available_checkpoints: list[Path] = field(default_factory=list)
    selected_checkpoint: Optional[Path] = None
    checkpoint_exists: bool = False
    torch_importable: bool = False
    torch_version: Optional[str] = None
    torchvision_importable: bool = False
    cuda_available: bool = False
    mmcv_importable: bool = False
    mmcv_version: Optional[str] = None
    mmengine_importable: bool = False
    mmengine_version: Optional[str] = None
    huggingface_hub_importable: bool = False
    huggingface_hub_version: Optional[str] = None
    huggingface_token_present: bool = False
    dinov3_cache_present: bool = False
    trackon_predictor_importable: bool = False
    trackon_predictor_module_path: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return (
            self.repo_path is not None
            and self.checkpoint_exists
            and self.torch_importable
            and self.torchvision_importable
            and self.mmcv_importable
            and self.mmengine_importable
            and self.trackon_predictor_importable
        )


@dataclass
class TrackonrSmokeTestResult:
    tracks_shape: tuple[int, ...]
    visibility_shape: tuple[int, ...]
    first_track_point: list[float]
    visibility_dtype: str
    device: str
    checkpoint_name: str


@dataclass
class TrackonrRuntimeSettings:
    checkpoint_path: str | Path
    repo_path: str | Path | None = None
    device: str = "auto"
    inference_resolution: int | tuple[int, int] = (384, 512)
    support_grid_size: int = 20
    use_visibility_gating: bool = True
    visibility_threshold: float = 0.8


@dataclass
class TrackonrOfflineResult:
    tracks_xy: np.ndarray
    visible_mask: np.ndarray
    inference_resolution: tuple[int, int]
    original_resolution: tuple[int, int]
    checkpoint_name: str
    support_grid_size: int


def discover_trackonr_checkpoints(models_dir: str | Path | None = None) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_TRACKONR_MODELS_DIR
    if not directory.exists():
        return []
    checkpoints = []
    for pattern in ("*.pt", "*.pth"):
        checkpoints.extend(directory.glob(pattern))
    return sorted(set(checkpoints), key=lambda path: path.name.lower())


def select_default_trackonr_checkpoint(checkpoints: list[Path]) -> Optional[Path]:
    if not checkpoints:
        return None
    by_name = {path.name: path for path in checkpoints}
    for preferred_name in CHECKPOINT_PRIORITY:
        if preferred_name in by_name:
            return by_name[preferred_name]
    return checkpoints[0]


def probe_trackonr_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> TrackonrEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_TRACKONR_MODELS_DIR
    directory = directory.resolve()
    checkpoints = discover_trackonr_checkpoints(directory)
    selected_checkpoint = _resolve_checkpoint_path(checkpoint_path, checkpoints, directory)
    resolved_repo_path = _resolve_trackonr_repo_path(repo_path)

    status = TrackonrEnvironmentStatus(
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
            "No Track-On-R checkpoint detected in tracking_models/trackonr. "
            "Add a .pt or .pth checkpoint before running diagnostics."
        )

    if resolved_repo_path is None:
        status.errors.append(
            "Could not resolve the local Track-On repository. "
            "Clone `track_on` next to the project or pass --repo-path."
        )
        status.hints.append(
            "Expected repo root containing `model/trackon_predictor.py` and `utils/train_utils.py`."
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
            "Install PyTorch in the active environment, e.g. "
            "`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`."
        )

    try:
        importlib.import_module("torchvision")
        status.torchvision_importable = True
    except Exception as exc:
        status.errors.append(f"TorchVision import failed: {exc}")
        status.hints.append("Install TorchVision in the active environment.")

    try:
        mmcv = importlib.import_module("mmcv")
        status.mmcv_importable = True
        status.mmcv_version = getattr(mmcv, "__version__", None)
    except Exception as exc:
        status.errors.append(f"MMCV import failed: {exc}")
        status.hints.append("Install `mmcv==2.2.0` in the active environment.")

    try:
        mmengine = importlib.import_module("mmengine")
        status.mmengine_importable = True
        status.mmengine_version = getattr(mmengine, "__version__", None)
    except Exception as exc:
        status.errors.append(f"MMEngine import failed: {exc}")
        status.hints.append("Install `mmengine` in the active environment.")

    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
        status.huggingface_hub_importable = True
        status.huggingface_hub_version = getattr(huggingface_hub, "__version__", None)
        token = None
        get_token = getattr(huggingface_hub, "get_token", None)
        if callable(get_token):
            token = get_token()
        else:
            hf_folder = getattr(huggingface_hub, "HfFolder", None)
            if hf_folder is not None and hasattr(hf_folder, "get_token"):
                token = hf_folder.get_token()
        status.huggingface_token_present = bool(token)
    except Exception as exc:
        status.warnings.append(f"huggingface_hub import failed: {exc}")
        status.hints.append(
            "Install `huggingface_hub` and run `hf auth login` to allow DINOv3 download."
        )

    status.dinov3_cache_present = _resolve_dinov3_cache_dir().exists()
    if not status.dinov3_cache_present and not status.huggingface_token_present:
        status.warnings.append(
            "DINOv3 cache not detected and no Hugging Face token found. "
            "The first model load may fail until you run `hf auth login`."
        )
    elif not status.dinov3_cache_present and status.huggingface_token_present:
        status.warnings.append(
            "DINOv3 cache not detected yet. The first Track-On-R load will download the backbone."
        )

    if resolved_repo_path is not None:
        try:
            predictor_module = _import_trackon_predictor_module(resolved_repo_path)
            status.trackon_predictor_importable = True
            status.trackon_predictor_module_path = str(
                Path(inspect.getfile(predictor_module)).resolve()
            )
        except Exception as exc:
            status.errors.append(f"Track-On-R Predictor import failed: {exc}")
            _append_trackonr_import_hints(status, exc)

    if status.trackon_predictor_importable and status.torch_importable and not status.cuda_available:
        status.warnings.append(
            "PyTorch CUDA is not available. Track-On-R can still run on CPU, "
            "but longer LEED sequences will be slow."
        )

    return status


def format_trackonr_environment_report(status: TrackonrEnvironmentStatus) -> str:
    selected_checkpoint = (
        str(status.selected_checkpoint) if status.selected_checkpoint is not None else "None"
    )
    available = ", ".join(path.name for path in status.available_checkpoints) or "none"
    lines = [
        "Track-On-R environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Selected checkpoint: {selected_checkpoint}",
        f"- Available checkpoints: {available}",
        f"- PyTorch import: {'ok' if status.torch_importable else 'failed'}",
        f"- PyTorch version: {status.torch_version or 'unknown'}",
        f"- TorchVision import: {'ok' if status.torchvision_importable else 'failed'}",
        f"- CUDA available: {'yes' if status.cuda_available else 'no'}",
        f"- MMCV import: {'ok' if status.mmcv_importable else 'failed'}",
        f"- MMCV version: {status.mmcv_version or 'unknown'}",
        f"- MMEngine import: {'ok' if status.mmengine_importable else 'failed'}",
        f"- MMEngine version: {status.mmengine_version or 'unknown'}",
        f"- huggingface_hub import: {'ok' if status.huggingface_hub_importable else 'failed'}",
        f"- Hugging Face token: {'present' if status.huggingface_token_present else 'missing'}",
        f"- DINOv3 cache: {'present' if status.dinov3_cache_present else 'missing'}",
        f"- Predictor import: {'ok' if status.trackon_predictor_importable else 'failed'}",
    ]
    if status.trackon_predictor_module_path:
        lines.append(f"- Predictor module: {status.trackon_predictor_module_path}")
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


def run_trackonr_smoke_test(
    *,
    checkpoint_path: str | Path,
    repo_path: str | Path | None = None,
    device: str = "auto",
) -> TrackonrSmokeTestResult:
    torch = importlib.import_module("torch")
    resolved_device = _resolve_torch_device(torch, device)
    resolved_repo_path = _resolve_trackonr_repo_path(repo_path)
    predictor = _load_trackonr_predictor(
        torch_module=torch,
        checkpoint_path=checkpoint_path,
        repo_root=resolved_repo_path,
        device=resolved_device,
    )

    video = torch.zeros((1, 4, 3, 64, 64), dtype=torch.float32, device=resolved_device)
    queries = torch.tensor(
        [[[0.0, 32.0, 32.0]]],  # (t, x, y)
        dtype=torch.float32,
        device=resolved_device,
    )

    with torch.no_grad():
        pred_tracks, pred_visibility = predictor(video, queries)

    tracks = pred_tracks.detach().to(dtype=torch.float32).cpu().numpy()
    visibility = pred_visibility.detach().cpu().numpy()

    return TrackonrSmokeTestResult(
        tracks_shape=tuple(int(dim) for dim in tracks.shape),
        visibility_shape=tuple(int(dim) for dim in visibility.shape),
        first_track_point=[float(value) for value in tracks[0, 0, 0].tolist()],
        visibility_dtype=str(pred_visibility.dtype),
        device=str(resolved_device),
        checkpoint_name=Path(checkpoint_path).name,
    )


def run_trackonr_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: TrackonrRuntimeSettings,
) -> TrackonrOfflineResult:
    """
    Run Track-On-R on a full sequence of LEED frames and sparse query points.

    Args:
        frames: [T, H, W] grayscale or [T, H, W, C] uint8/float frames.
        query_points_tyx: [Q, 3] points in [t, y, x] in the original frame grid.
        settings: Track-On-R runtime settings including checkpoint and repo path.

    Returns:
        TrackonrOfflineResult with tracks in original image coordinates.
    """
    if frames is None or np.asarray(frames).size == 0:
        raise ValueError("No frames provided for Track-On-R inference.")
    frames_np = np.asarray(frames)
    if frames_np.ndim not in (3, 4):
        raise ValueError(f"Expected frames with ndim 3 or 4, got shape {frames_np.shape}.")

    query_np = np.asarray(query_points_tyx, dtype=np.float32)
    if query_np.ndim != 2 or query_np.shape[1] != 3:
        raise ValueError(
            "query_points_tyx must have shape [num_queries, 3] in [t, y, x] format."
        )
    if query_np.shape[0] == 0:
        raise ValueError("No query points provided for Track-On-R inference.")

    torch = importlib.import_module("torch")
    resolved_device = _resolve_torch_device(torch, settings.device)
    resolved_repo_path = _resolve_trackonr_repo_path(settings.repo_path)

    prepared_frames, original_resolution = _prepare_frames_rgb(frames_np)
    query_points_txy = _convert_query_points_tyx_to_txy(query_np)

    predictor = _load_trackonr_predictor(
        torch_module=torch,
        checkpoint_path=settings.checkpoint_path,
        repo_root=resolved_repo_path,
        device=resolved_device,
        inference_resolution=settings.inference_resolution,
        visibility_threshold=float(settings.visibility_threshold),
        support_grid_size=int(max(0, settings.support_grid_size)),
    )

    video_tensor = (
        torch.tensor(prepared_frames, dtype=torch.float32, device=resolved_device)
        .permute(0, 3, 1, 2)[None]
    )
    query_tensor = torch.tensor(
        query_points_txy,
        dtype=torch.float32,
        device=resolved_device,
    )[None]

    with torch.no_grad():
        pred_tracks, pred_visibility = predictor(video_tensor, query_tensor)

    tracks_txy = pred_tracks.detach().to(dtype=torch.float32).cpu().numpy()[0]
    visibility_txq = pred_visibility.detach().cpu().numpy()[0]

    tracks_xy = np.transpose(tracks_txy, (1, 0, 2))
    visible_mask = np.transpose(visibility_txq, (1, 0)).astype(bool, copy=False)
    if not bool(settings.use_visibility_gating):
        visible_mask = np.ones_like(visible_mask, dtype=bool)

    return TrackonrOfflineResult(
        tracks_xy=tracks_xy,
        visible_mask=visible_mask,
        inference_resolution=_normalize_trackonr_resolution(settings.inference_resolution),
        original_resolution=original_resolution,
        checkpoint_name=Path(settings.checkpoint_path).name,
        support_grid_size=int(max(0, settings.support_grid_size)),
    )


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoints: list[Path],
    models_dir: Path,
) -> Optional[Path]:
    if checkpoint_path is None:
        return select_default_trackonr_checkpoint(checkpoints)

    candidate = Path(checkpoint_path)
    if not candidate.is_absolute():
        candidate = (models_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _resolve_trackonr_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidate_paths: list[Path] = []
    if repo_path is not None:
        candidate_paths.append(Path(repo_path).expanduser())
    else:
        candidate_paths.extend(DEFAULT_TRACKONR_REPO_CANDIDATES)

    for candidate in candidate_paths:
        normalized = _normalize_trackonr_repo_candidate(candidate)
        if normalized is not None:
            return normalized
    return None


def _normalize_trackonr_repo_candidate(candidate: Path) -> Optional[Path]:
    candidate = candidate.resolve()
    if (candidate / "model" / "trackon_predictor.py").exists():
        return candidate
    return None


def _ensure_repo_root_on_sys_path(repo_root: Optional[Path]) -> None:
    if repo_root is None:
        return
    repo_root_str = str(repo_root.resolve())
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _import_trackon_predictor_module(repo_root: Optional[Path]):
    _ensure_repo_root_on_sys_path(repo_root)
    return importlib.import_module("model.trackon_predictor")


def _resolve_dinov3_cache_dir() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub" / DINO_CACHE_DIRNAME


def _append_trackonr_import_hints(status: TrackonrEnvironmentStatus, exc: Exception) -> None:
    message = str(exc).lower()
    if "no module named 'model'" in message or "no module named 'utils'" in message:
        status.hints.append(
            "Pass --repo-path pointing at the local `track_on` repo root."
        )
    if "no module named 'mmcv'" in message:
        status.hints.append("Install `mmcv==2.2.0` in the active environment.")
    if "no module named 'mmengine'" in message:
        status.hints.append("Install `mmengine` in the active environment.")
    if "no module named 'transformers'" in message:
        status.hints.append("Install dependencies from `track_on/requirements.txt`.")
    if "no module named 'timm'" in message:
        status.hints.append("Install dependencies from `track_on/requirements.txt`.")
    if "no module named 'huggingface_hub'" in message:
        status.hints.append("Install `huggingface_hub` and run `hf auth login`.")
    if not status.hints:
        status.hints.append(
            "Verify that the local `track_on` repo is available and that its runtime dependencies "
            "from `requirements.txt` are installed."
        )


def _resolve_torch_device(torch_module, requested_device: str):
    device = str(requested_device or "auto").strip().lower()
    if device in {"auto", "gpu"}:
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if device == "cuda" and not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    return torch_module.device(device)


def _load_trackonr_predictor(
    *,
    torch_module,
    checkpoint_path: str | Path,
    repo_root: Optional[Path],
    device,
    inference_resolution: int | tuple[int, int] = (384, 512),
    visibility_threshold: float = 0.8,
    support_grid_size: int = 20,
):
    if repo_root is None:
        raise FileNotFoundError(
            "Could not resolve the Track-On repo. Pass repo_path pointing at the local `track_on` clone."
        )

    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Track-On-R checkpoint not found: {checkpoint}")

    predictor_module = _import_trackon_predictor_module(repo_root)
    cache_key = (
        str(repo_root.resolve()),
        str(checkpoint),
        str(device),
        str(_normalize_trackonr_resolution(inference_resolution)),
        str(float(visibility_threshold)),
        str(int(max(0, support_grid_size))),
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    Predictor = getattr(predictor_module, "Predictor")
    model_args = _build_trackonr_model_args(
        inference_resolution=inference_resolution,
        visibility_threshold=visibility_threshold,
    )
    predictor = Predictor(
        model_args=model_args,
        checkpoint_path=str(checkpoint),
        support_grid_size=int(max(0, support_grid_size)),
    ).to(device)
    predictor.eval()
    _MODEL_CACHE[cache_key] = predictor
    return predictor


def _build_trackonr_model_args(
    *,
    inference_resolution: int | tuple[int, int],
    visibility_threshold: float,
):
    height, width = _normalize_trackonr_resolution(inference_resolution)

    class Args:
        def __init__(self):
            self.input_size = [height, width]
            self.M = 24
            self.D = 256
            self.K = 16
            self.decoder_layer_num = 3
            self.predicton_head_layer_num = 3
            self.rerank_layer_num = 3
            self.vit_backbone = "dinov3_s_plus"
            self.vit_upsample_factor = 1.143
            self.grad_checkpoint = True
            self.M_i = 72
            self.delta_v = float(visibility_threshold)

    return Args()


def _normalize_trackonr_resolution(
    target: int | tuple[int, int],
) -> tuple[int, int]:
    if isinstance(target, int):
        height = int(target)
        width = int(target)
    else:
        height = int(target[0])
        width = int(target[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid Track-On-R inference resolution: {(height, width)}")
    return height, width


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


def _convert_query_points_tyx_to_txy(query_points_tyx: np.ndarray) -> np.ndarray:
    query_points_txy = np.array(query_points_tyx, dtype=np.float32, copy=True)
    query_points_txy[:, 1] = query_points_tyx[:, 2]
    query_points_txy[:, 2] = query_points_tyx[:, 1]
    return query_points_txy
