"""PIPs++ environment diagnostics and smoke-test helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import importlib
import inspect
import re
import sys

import numpy as np


DEFAULT_PIPSPP_MODELS_DIR = Path("tracking_models") / "pipspp"
DEFAULT_PIPSPP_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "pipspp",
    Path(__file__).resolve().parents[2] / "pipspp",
)
CHECKPOINT_PRIORITY = (
    "reference_model",
    "model-000200000.pth",
)
CHECKPOINT_FILE_PATTERN = "model-*.pth"
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


@dataclass
class PipsppEnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    available_checkpoints: list[Path] = field(default_factory=list)
    selected_checkpoint: Optional[Path] = None
    selected_checkpoint_file: Optional[Path] = None
    checkpoint_exists: bool = False
    torch_importable: bool = False
    torch_version: Optional[str] = None
    cuda_available: bool = False
    einops_importable: bool = False
    cv2_importable: bool = False
    pips_model_importable: bool = False
    saverloader_importable: bool = False
    pips_module_path: Optional[str] = None
    saverloader_module_path: Optional[str] = None
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
            and self.pips_model_importable
            and self.saverloader_importable
        )


@dataclass
class PipsppSmokeTestResult:
    tracks_shape: tuple[int, ...]
    num_prediction_stages: int
    first_track_point: list[float]
    device: str
    checkpoint_name: str


@dataclass
class PipsppRuntimeSettings:
    checkpoint_path: str | Path
    repo_path: str | Path | None = None
    device: str = "auto"
    inference_resolution: int | tuple[int, int] = (256, 384)
    iters: int = 4
    beautify: bool = True
    use_inbounds_visibility: bool = True


@dataclass
class PipsppOfflineResult:
    tracks_xy: np.ndarray
    visible_mask: np.ndarray
    inference_resolution: tuple[int, int]
    original_resolution: tuple[int, int]
    checkpoint_name: str
    num_prediction_stages: int


def discover_pipspp_checkpoints(
    models_dir: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_PIPSPP_MODELS_DIR
    discovered: dict[str, Path] = {}

    for candidate in _discover_pipspp_checkpoints_in_directory(directory):
        discovered[str(candidate.resolve())] = candidate.resolve()

    repo_root = _resolve_pipspp_repo_path(repo_path)
    if repo_root is not None:
        repo_reference_model = repo_root / "reference_model"
        if _contains_pipspp_checkpoint_files(repo_reference_model):
            discovered[str(repo_reference_model.resolve())] = repo_reference_model.resolve()

    return sorted(discovered.values(), key=lambda path: path.name.lower())


def select_default_pipspp_checkpoint(checkpoints: list[Path]) -> Optional[Path]:
    if not checkpoints:
        return None
    by_name = {path.name: path for path in checkpoints}
    for preferred_name in CHECKPOINT_PRIORITY:
        if preferred_name in by_name:
            return by_name[preferred_name]
    dirs = [path for path in checkpoints if path.is_dir()]
    if dirs:
        return dirs[0]
    return checkpoints[0]


def probe_pipspp_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> PipsppEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_PIPSPP_MODELS_DIR
    directory = directory.resolve()
    resolved_repo_path = _resolve_pipspp_repo_path(repo_path)
    checkpoints = discover_pipspp_checkpoints(directory, resolved_repo_path)
    selected_checkpoint = _resolve_checkpoint_path(
        checkpoint_path,
        checkpoints,
        directory,
        resolved_repo_path,
    )
    selected_checkpoint_file = _try_resolve_checkpoint_file(selected_checkpoint)

    status = PipsppEnvironmentStatus(
        models_dir=directory,
        repo_path=resolved_repo_path,
        available_checkpoints=checkpoints,
        selected_checkpoint=selected_checkpoint,
        selected_checkpoint_file=selected_checkpoint_file,
        checkpoint_exists=bool(selected_checkpoint_file and selected_checkpoint_file.exists()),
    )

    if not directory.exists():
        status.warnings.append(
            f"Checkpoint directory does not exist yet: {directory}. "
            "This is fine if you keep the reference checkpoint inside the local PIPs++ repo."
        )

    if checkpoint_path and not status.checkpoint_exists:
        status.errors.append(f"Selected checkpoint not found: {selected_checkpoint}")
    elif selected_checkpoint is None:
        status.warnings.append(
            "No PIPs++ checkpoint detected in tracking_models/pipspp or in repo/reference_model. "
            "Add a checkpoint directory such as `reference_model/` before running diagnostics."
        )

    if resolved_repo_path is None:
        status.errors.append(
            "Could not resolve the local PIPs++ repo. Clone `pips2` as `pipspp` next to the project "
            "or pass --repo-path."
        )
        status.hints.append(
            "Expected repo root containing `nets/pips2.py` and `saverloader.py`."
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
        status.hints.append("Install PyTorch in the active environment used for PIPs++.")

    try:
        importlib.import_module("einops")
        status.einops_importable = True
    except Exception as exc:
        status.errors.append(f"einops import failed: {exc}")
        status.hints.append("Install dependencies from `pipspp/requirements.txt`.")

    try:
        importlib.import_module("cv2")
        status.cv2_importable = True
    except Exception as exc:
        status.warnings.append(f"OpenCV import failed: {exc}")
        status.hints.append("Install dependencies from `pipspp/requirements.txt`.")

    if resolved_repo_path is not None:
        try:
            pips_module = _import_pipspp_module(resolved_repo_path, "nets.pips2")
            status.pips_model_importable = True
            status.pips_module_path = str(Path(inspect.getfile(pips_module)).resolve())
        except Exception as exc:
            status.errors.append(f"PIPs++ model import failed: {exc}")
            _append_pipspp_import_hints(status, exc)

        try:
            saverloader_module = _import_pipspp_module(resolved_repo_path, "saverloader")
            status.saverloader_importable = True
            status.saverloader_module_path = str(
                Path(inspect.getfile(saverloader_module)).resolve()
            )
        except Exception as exc:
            status.errors.append(f"PIPs++ saverloader import failed: {exc}")
            _append_pipspp_import_hints(status, exc)

    if status.torch_importable and not status.cuda_available:
        status.warnings.append(
            "Current upstream PIPs++ uses CUDA-specific tensors in `nets/pips2.py`; "
            "CPU runtime is not supported without patching the local repo."
        )

    if status.selected_checkpoint is not None and status.selected_checkpoint.is_file():
        status.warnings.append(
            "Selected checkpoint is a single .pth file. This is supported by Trackleed diagnostics, "
            "but the upstream repo usually expects a directory such as `reference_model/`."
        )

    return status


def format_pipspp_environment_report(status: PipsppEnvironmentStatus) -> str:
    selected_checkpoint = (
        str(status.selected_checkpoint) if status.selected_checkpoint is not None else "None"
    )
    selected_checkpoint_file = (
        str(status.selected_checkpoint_file)
        if status.selected_checkpoint_file is not None
        else "None"
    )
    available = ", ".join(path.name for path in status.available_checkpoints) or "none"
    lines = [
        "PIPs++ environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Selected checkpoint ref: {selected_checkpoint}",
        f"- Selected checkpoint file: {selected_checkpoint_file}",
        f"- Available checkpoints: {available}",
        f"- PyTorch import: {'ok' if status.torch_importable else 'failed'}",
        f"- PyTorch version: {status.torch_version or 'unknown'}",
        f"- CUDA available: {'yes' if status.cuda_available else 'no'}",
        f"- einops import: {'ok' if status.einops_importable else 'failed'}",
        f"- OpenCV import: {'ok' if status.cv2_importable else 'failed'}",
        f"- PIPs++ model import: {'ok' if status.pips_model_importable else 'failed'}",
        f"- saverloader import: {'ok' if status.saverloader_importable else 'failed'}",
    ]
    if status.pips_module_path:
        lines.append(f"- PIPs++ module: {status.pips_module_path}")
    if status.saverloader_module_path:
        lines.append(f"- saverloader module: {status.saverloader_module_path}")
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


def run_pipspp_smoke_test(
    *,
    checkpoint_path: str | Path,
    repo_path: str | Path | None = None,
    device: str = "auto",
) -> PipsppSmokeTestResult:
    torch = importlib.import_module("torch")
    resolved_device = _resolve_torch_device(torch, device)
    if resolved_device.type != "cuda":
        raise RuntimeError(
            "Current upstream PIPs++ smoke test requires CUDA because `nets/pips2.py` "
            "creates CUDA tensors inside `Pips.forward()`."
        )

    resolved_repo_path = _resolve_pipspp_repo_path(repo_path)
    checkpoint_file = _resolve_checkpoint_file(Path(checkpoint_path))
    model = _load_pipspp_model(
        torch_module=torch,
        checkpoint_path=checkpoint_file,
        repo_root=resolved_repo_path,
        device=resolved_device,
    )

    rgbs = torch.zeros((1, 8, 3, 256, 384), dtype=torch.float32, device=resolved_device)
    trajs = torch.zeros((1, 8, 16, 2), dtype=torch.float32, device=resolved_device)

    with torch.no_grad():
        pred_stages, _, _, _ = model(trajs, rgbs, iters=4, feat_init=None, beautify=True)

    final_pred = pred_stages[-1].detach().to(dtype=torch.float32).cpu().numpy()
    return PipsppSmokeTestResult(
        tracks_shape=tuple(int(dim) for dim in final_pred.shape),
        num_prediction_stages=len(pred_stages),
        first_track_point=[float(value) for value in final_pred[0, 0, 0].tolist()],
        device=str(resolved_device),
        checkpoint_name=checkpoint_file.name,
    )


def run_pipspp_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: PipsppRuntimeSettings,
) -> PipsppOfflineResult:
    """
    Run offline PIPs++ on a sequence of frames and sparse query points.

    Args:
        frames: [T, H, W] grayscale or [T, H, W, C] uint8/float frames.
        query_points_tyx: [Q, 3] points in [t, y, x] in the original frame grid.
        settings: PIPs++ runtime settings including checkpoint and repo path.

    Returns:
        PipsppOfflineResult with tracks in original image coordinates.
    """
    if frames is None or np.asarray(frames).size == 0:
        raise ValueError("No frames provided for PIPs++ inference.")
    frames_np = np.asarray(frames)
    if frames_np.ndim not in (3, 4):
        raise ValueError(f"Expected frames with ndim 3 or 4, got shape {frames_np.shape}.")

    query_np = np.asarray(query_points_tyx, dtype=np.float32)
    if query_np.ndim != 2 or query_np.shape[1] != 3:
        raise ValueError(
            "query_points_tyx must have shape [num_queries, 3] in [t, y, x] format."
        )
    if query_np.shape[0] == 0:
        raise ValueError("No query points provided for PIPs++ inference.")

    torch = importlib.import_module("torch")
    resolved_device = _resolve_torch_device(torch, settings.device)
    if resolved_device.type != "cuda":
        raise RuntimeError(
            "Current upstream PIPs++ runtime requires CUDA because `nets/pips2.py` "
            "creates CUDA tensors inside `Pips.forward()`."
        )

    prepared_frames, original_resolution = _prepare_frames_rgb(frames_np)
    inference_resolution = _normalize_pipspp_resolution(settings.inference_resolution)
    resized_frames = _resize_video_numpy(prepared_frames, inference_resolution)
    scaled_query_points = _scale_query_points_tyx(
        query_np,
        source_resolution=original_resolution,
        target_resolution=inference_resolution,
    )
    seed_frames = _extract_query_seed_frames(scaled_query_points, total_frames=resized_frames.shape[0])

    model = _load_pipspp_model(
        torch_module=torch,
        checkpoint_path=settings.checkpoint_path,
        repo_root=_resolve_pipspp_repo_path(settings.repo_path),
        device=resolved_device,
    )

    num_queries = int(scaled_query_points.shape[0])
    num_frames = int(resized_frames.shape[0])
    tracks_xy_resized = np.full((num_queries, num_frames, 2), np.nan, dtype=np.float32)
    visible_mask = np.zeros((num_queries, num_frames), dtype=bool)
    num_prediction_stages = 0

    unique_seed_frames = np.unique(seed_frames)
    for seed_frame in unique_seed_frames.tolist():
        query_indices = np.where(seed_frames == seed_frame)[0]
        local_frames = resized_frames[seed_frame:]
        local_len = int(local_frames.shape[0])
        if local_len <= 0:
            continue

        local_video = (
            torch.tensor(local_frames, dtype=torch.float32, device=resolved_device)
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
        )
        seed_xy = np.stack(
            [
                scaled_query_points[query_indices, 2],
                scaled_query_points[query_indices, 1],
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        trajs_e0 = (
            torch.tensor(seed_xy, dtype=torch.float32, device=resolved_device)
            .unsqueeze(0)
            .unsqueeze(1)
            .repeat(1, local_len, 1, 1)
        )

        with torch.no_grad():
            pred_stages, _, _, _ = model(
                trajs_e0,
                local_video,
                iters=int(max(1, settings.iters)),
                feat_init=None,
                beautify=bool(settings.beautify),
            )

        num_prediction_stages = max(num_prediction_stages, len(pred_stages))
        local_tracks = pred_stages[-1].detach().to(dtype=torch.float32).cpu().numpy()[0]
        local_tracks = np.transpose(local_tracks, (1, 0, 2))
        tracks_xy_resized[query_indices, seed_frame:, :] = local_tracks

        if bool(settings.use_inbounds_visibility):
            local_visible = _build_inbounds_visibility_mask(
                local_tracks,
                frame_resolution=inference_resolution,
            )
        else:
            local_visible = np.ones(local_tracks.shape[:2], dtype=bool)
        visible_mask[query_indices, seed_frame:] = local_visible

    tracks_xy = _scale_tracks_xy(
        tracks_xy_resized,
        source_resolution=inference_resolution,
        target_resolution=original_resolution,
    )
    return PipsppOfflineResult(
        tracks_xy=tracks_xy,
        visible_mask=visible_mask,
        inference_resolution=inference_resolution,
        original_resolution=original_resolution,
        checkpoint_name=_resolve_checkpoint_file(Path(settings.checkpoint_path)).name,
        num_prediction_stages=int(num_prediction_stages),
    )


def _discover_pipspp_checkpoints_in_directory(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []

    refs: list[Path] = []
    for child in sorted(directory.iterdir(), key=lambda path: path.name.lower()):
        if child.is_dir() and _contains_pipspp_checkpoint_files(child):
            refs.append(child.resolve())
        elif child.is_file() and child.suffix.lower() == ".pth":
            refs.append(child.resolve())

    return refs


def _contains_pipspp_checkpoint_files(directory: Path) -> bool:
    return directory.exists() and directory.is_dir() and any(directory.glob(CHECKPOINT_FILE_PATTERN))


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoints: list[Path],
    models_dir: Path,
    repo_path: Optional[Path],
) -> Optional[Path]:
    if checkpoint_path is None:
        return select_default_pipspp_checkpoint(checkpoints)

    candidate = Path(checkpoint_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    available_by_name = {path.name: path for path in checkpoints}
    if candidate.name in available_by_name:
        return available_by_name[candidate.name]

    candidate_paths = [(models_dir / candidate).resolve()]
    if repo_path is not None:
        candidate_paths.append((repo_path / candidate).resolve())
        candidate_paths.append((repo_path / "reference_model" / candidate).resolve())
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    return candidate_paths[0]


def _try_resolve_checkpoint_file(checkpoint_path: Optional[Path]) -> Optional[Path]:
    if checkpoint_path is None:
        return None
    try:
        return _resolve_checkpoint_file(checkpoint_path)
    except Exception:
        return None


def _resolve_checkpoint_file(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if checkpoint_path.is_file():
        if checkpoint_path.suffix.lower() != ".pth":
            raise FileNotFoundError(f"PIPs++ checkpoint is not a .pth file: {checkpoint_path}")
        return checkpoint_path

    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        raise FileNotFoundError(f"PIPs++ checkpoint path not found: {checkpoint_path}")

    checkpoint_files = sorted(
        checkpoint_path.glob(CHECKPOINT_FILE_PATTERN),
        key=_checkpoint_sort_key,
    )
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No `{CHECKPOINT_FILE_PATTERN}` files found under: {checkpoint_path}"
        )
    return checkpoint_files[-1].resolve()


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"-(\d+)\.pth$", path.name)
    if match:
        return (int(match.group(1)), path.name.lower())
    return (0, path.name.lower())


def _resolve_pipspp_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidate_paths: list[Path] = []
    if repo_path is not None:
        candidate_paths.append(Path(repo_path).expanduser())
    else:
        candidate_paths.extend(DEFAULT_PIPSPP_REPO_CANDIDATES)

    for candidate in candidate_paths:
        normalized = _normalize_pipspp_repo_candidate(candidate)
        if normalized is not None:
            return normalized
    return None


def _normalize_pipspp_repo_candidate(candidate: Path) -> Optional[Path]:
    candidate = candidate.resolve()
    if (candidate / "nets" / "pips2.py").exists() and (candidate / "saverloader.py").exists():
        return candidate
    return None


def _ensure_repo_root_on_sys_path(repo_root: Optional[Path]) -> None:
    if repo_root is None:
        return
    repo_root_str = str(repo_root.resolve())
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _import_pipspp_module(repo_root: Optional[Path], module_name: str):
    _ensure_repo_root_on_sys_path(repo_root)
    return importlib.import_module(module_name)


def _append_pipspp_import_hints(status: PipsppEnvironmentStatus, exc: Exception) -> None:
    message = str(exc).lower()
    if "no module named 'nets'" in message or "no module named 'saverloader'" in message:
        status.hints.append("Pass --repo-path pointing at the local `pipspp` repo root.")
    if "no module named 'einops'" in message:
        status.hints.append("Install dependencies from `pipspp/requirements.txt`.")
    if "no module named 'cv2'" in message:
        status.hints.append("Install dependencies from `pipspp/requirements.txt`.")
    if "no module named 'utils'" in message:
        status.hints.append(
            "Verify that the local `pipspp` repo is on sys.path and not just `reference_model/`."
        )
    if not status.hints:
        status.hints.append(
            "Verify that the local `pipspp` repo is available and that its runtime dependencies "
            "from `requirements.txt` are installed."
        )


def _resolve_torch_device(torch_module, requested_device: str):
    device = str(requested_device or "auto").strip().lower()
    if device in {"auto", "gpu"}:
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if device == "cuda" and not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    return torch_module.device(device)


def _load_pipspp_model(
    *,
    torch_module,
    checkpoint_path: str | Path,
    repo_root: Optional[Path],
    device,
):
    if repo_root is None:
        raise FileNotFoundError(
            "Could not resolve the PIPs++ repo. Pass repo_path pointing at the local `pipspp` clone."
        )

    checkpoint_file = _resolve_checkpoint_file(Path(checkpoint_path))
    pips_module = _import_pipspp_module(repo_root, "nets.pips2")

    cache_key = (
        str(repo_root.resolve()),
        str(checkpoint_file),
        str(device),
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    Pips = getattr(pips_module, "Pips")
    model = Pips(stride=8).to(device).eval()
    checkpoint = _torch_load_checkpoint(torch_module, checkpoint_file)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    _MODEL_CACHE[cache_key] = model
    return model


def _torch_load_checkpoint(torch_module, checkpoint_file: Path):
    try:
        return torch_module.load(checkpoint_file, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(checkpoint_file, map_location="cpu")


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


def _normalize_pipspp_resolution(target: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(target, int):
        target_h = target_w = int(target)
    else:
        target_h = int(target[0])
        target_w = int(target[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid PIPs++ inference resolution: {(target_h, target_w)}")
    if target_h % 32 != 0 or target_w % 32 != 0:
        raise ValueError(
            "PIPs++ inference resolution should be divisible by 32; "
            f"got {(target_h, target_w)}."
        )
    return (target_h, target_w)


def _resize_video_numpy(frames: np.ndarray, target: int | tuple[int, int]) -> np.ndarray:
    cv2 = importlib.import_module("cv2")
    target_h, target_w = _normalize_pipspp_resolution(target)
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


def _extract_query_seed_frames(query_points_tyx: np.ndarray, total_frames: int) -> np.ndarray:
    seed_frames_float = np.asarray(query_points_tyx[:, 0], dtype=np.float32)
    seed_frames = np.rint(seed_frames_float).astype(np.int32)
    if np.any(np.abs(seed_frames_float - seed_frames) > 1e-3):
        raise ValueError("PIPs++ query seed frames must be integer-valued.")
    if np.any(seed_frames < 0) or np.any(seed_frames >= total_frames):
        raise ValueError(
            f"PIPs++ query seed frame out of range for sequence length {total_frames}."
        )
    return seed_frames


def _build_inbounds_visibility_mask(
    tracks_xy: np.ndarray,
    frame_resolution: tuple[int, int],
) -> np.ndarray:
    height, width = frame_resolution
    visible = np.isfinite(tracks_xy[..., 0]) & np.isfinite(tracks_xy[..., 1])
    visible &= tracks_xy[..., 0] >= 0.0
    visible &= tracks_xy[..., 0] <= float(max(0, width - 1))
    visible &= tracks_xy[..., 1] >= 0.0
    visible &= tracks_xy[..., 1] <= float(max(0, height - 1))
    return visible
