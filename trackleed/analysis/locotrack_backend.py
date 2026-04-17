"""LocoTrack environment diagnostics and smoke-test helpers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Any
import importlib
import inspect
import json
import subprocess
import sys

import numpy as np


DEFAULT_LOCOTRACK_MODELS_DIR = Path("tracking_models") / "locotrack"
DEFAULT_LOCOTRACK_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "locotrack",
    Path(__file__).resolve().parents[2] / "locotrack",
)
LOCOTRACK_IMPORT_SUBDIR = "locotrack_pytorch"
OFFLINE_CHECKPOINT_PRIORITY = (
    "locotrack_small.ckpt",
    "locotrack_base.ckpt",
)
DEFAULT_LOCOTRACK_SUBPROCESS_SCRIPT = Path("scripts") / "run_locotrack_subprocess.py"
_MODEL_CACHE: dict[tuple[str, str, str, str], Any] = {}


@dataclass
class LocotrackEnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    available_checkpoints: list[Path] = field(default_factory=list)
    selected_checkpoint: Optional[Path] = None
    checkpoint_exists: bool = False
    torch_importable: bool = False
    torch_version: Optional[str] = None
    cuda_available: bool = False
    locotrack_importable: bool = False
    locotrack_module_path: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return (
            self.repo_path is not None
            and self.torch_importable
            and self.locotrack_importable
            and self.checkpoint_exists
        )


@dataclass
class LocotrackSmokeTestResult:
    tracks_shape: tuple[int, ...]
    occlusion_shape: tuple[int, ...]
    expected_dist_shape: tuple[int, ...]
    inference_resolution: tuple[int, int]
    first_track_point: list[float]
    model_size: str
    device: str


@dataclass
class LocotrackRuntimeSettings:
    repo_path: str | Path
    checkpoint_path: str | Path
    python_executable: str | Path | None = None
    backend_mode: str = "subprocess"
    model_size: str = "auto"
    device: str = "auto"
    inference_resolution: int = 256
    query_chunk_size: int = 64
    use_visibility_gating: bool = True
    visibility_threshold: float = 0.5
    use_uncertainty_gating: bool = True
    expected_dist_threshold: float = 0.5
    subprocess_timeout_sec: float = 1800.0


@dataclass
class LocotrackOfflineResult:
    tracks_xy: np.ndarray
    visible_mask: np.ndarray
    occlusion_logits: np.ndarray
    expected_dist_logits: np.ndarray
    inference_resolution: tuple[int, int]
    original_resolution: tuple[int, int]
    model_size: str


def discover_locotrack_checkpoints(models_dir: str | Path | None = None) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_LOCOTRACK_MODELS_DIR
    if not directory.exists():
        return []
    return sorted(directory.glob("*.ckpt"), key=lambda path: path.name.lower())


def select_default_locotrack_checkpoint(checkpoints: list[Path]) -> Optional[Path]:
    if not checkpoints:
        return None
    by_name = {path.name: path for path in checkpoints}
    for preferred_name in OFFLINE_CHECKPOINT_PRIORITY:
        if preferred_name in by_name:
            return by_name[preferred_name]
    return checkpoints[0]


def probe_locotrack_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> LocotrackEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_LOCOTRACK_MODELS_DIR
    directory = directory.resolve()
    checkpoints = discover_locotrack_checkpoints(directory)
    selected_checkpoint = _resolve_checkpoint_path(checkpoint_path, checkpoints, directory)
    resolved_repo_path = _resolve_locotrack_repo_path(repo_path)

    status = LocotrackEnvironmentStatus(
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
            "No LocoTrack checkpoint detected in tracking_models/locotrack. "
            "Add a .ckpt checkpoint before running LocoTrack."
        )

    if resolved_repo_path is None:
        status.errors.append(
            "Could not resolve the local LocoTrack repository. "
            "Clone `locotrack` next to the project or pass --repo-path."
        )
        status.hints.append(
            "Expected repo root containing `locotrack_pytorch/models/locotrack_model.py`."
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
            "`pip install torch torchvision torchaudio`."
        )

    if resolved_repo_path is not None:
        try:
            locotrack_module = _import_locotrack_model_module(resolved_repo_path)
            status.locotrack_importable = True
            status.locotrack_module_path = str(
                Path(inspect.getfile(locotrack_module)).resolve()
            )
        except Exception as exc:
            status.errors.append(f"LocoTrack import failed: {exc}")
            _append_locotrack_import_hints(status, exc)

    if status.locotrack_importable and status.torch_importable and not status.cuda_available:
        status.warnings.append(
            "PyTorch CUDA is not available. LocoTrack can still run on CPU, "
            "but longer LEED sequences will be slow."
        )

    return status


def format_locotrack_environment_report(status: LocotrackEnvironmentStatus) -> str:
    selected_checkpoint = (
        str(status.selected_checkpoint) if status.selected_checkpoint is not None else "None"
    )
    available = ", ".join(path.name for path in status.available_checkpoints) or "none"
    lines = [
        "LocoTrack environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Selected checkpoint: {selected_checkpoint}",
        f"- Available checkpoints: {available}",
        f"- PyTorch import: {'ok' if status.torch_importable else 'failed'}",
        f"- PyTorch version: {status.torch_version or 'unknown'}",
        f"- CUDA available: {'yes' if status.cuda_available else 'no'}",
        f"- LocoTrack import: {'ok' if status.locotrack_importable else 'failed'}",
    ]
    if status.locotrack_module_path:
        lines.append(f"- LocoTrack module: {status.locotrack_module_path}")
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


def run_locotrack_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: LocotrackRuntimeSettings,
) -> LocotrackOfflineResult:
    """
    Run offline PyTorch LocoTrack on a sequence of frames.

    Args:
        frames: [T, H, W] grayscale or [T, H, W, C] uint8/float frames.
        query_points_tyx: [Q, 3] points in [t, y, x] in the original frame grid.
        settings: LocoTrack runtime settings including repo/checkpoint paths.

    Returns:
        LocotrackOfflineResult with tracks in original image coordinates.
    """
    if frames is None or np.asarray(frames).size == 0:
        raise ValueError("No frames provided for LocoTrack inference.")
    frames_np = np.asarray(frames)
    if frames_np.ndim not in (3, 4):
        raise ValueError(f"Expected frames with ndim 3 or 4, got shape {frames_np.shape}.")

    query_np = np.asarray(query_points_tyx, dtype=np.float32)
    if query_np.ndim != 2 or query_np.shape[1] != 3:
        raise ValueError(
            "query_points_tyx must have shape [num_queries, 3] in [t, y, x] format."
        )

    backend_mode = str(settings.backend_mode or "subprocess").strip().lower()
    if backend_mode == "auto":
        backend_mode = "subprocess"

    if backend_mode == "direct":
        return _run_locotrack_direct_offline(frames_np, query_np, settings)
    if backend_mode != "subprocess":
        raise ValueError(
            f"Unsupported LocoTrack backend mode: {settings.backend_mode}. "
            "Expected `subprocess` or `direct`."
        )
    return _run_locotrack_subprocess_offline(frames_np, query_np, settings)


def _run_locotrack_subprocess_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: LocotrackRuntimeSettings,
) -> LocotrackOfflineResult:
    python_executable = _resolve_locotrack_python_executable(settings.python_executable)
    script_path = (Path(__file__).resolve().parents[2] / DEFAULT_LOCOTRACK_SUBPROCESS_SCRIPT).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"LocoTrack subprocess script not found: {script_path}")

    with TemporaryDirectory(prefix="trackleed_locotrack_") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "locotrack_input.npz"
        output_path = temp_root / "locotrack_output.npz"
        np.savez_compressed(
            input_path,
            frames=np.asarray(frames),
            query_points_tyx=np.asarray(query_points_tyx, dtype=np.float32),
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
            "--model-size",
            str(settings.model_size),
            "--device",
            str(settings.device),
            "--inference-resolution",
            str(int(settings.inference_resolution)),
            "--query-chunk-size",
            str(int(max(1, settings.query_chunk_size))),
            "--visibility-threshold",
            str(float(settings.visibility_threshold)),
            "--expected-dist-threshold",
            str(float(settings.expected_dist_threshold)),
        ]
        if settings.repo_path is not None:
            command.extend(["--repo-path", str(settings.repo_path)])
        if not bool(settings.use_visibility_gating):
            command.append("--disable-visibility-gating")
        if not bool(settings.use_uncertainty_gating):
            command.append("--disable-uncertainty-gating")

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
                "LocoTrack subprocess inference failed:\n"
                + json.dumps(payload, indent=2, ensure_ascii=False)
            )
        if not output_path.exists():
            raise FileNotFoundError(f"LocoTrack subprocess did not create output: {output_path}")

        with np.load(output_path, allow_pickle=False) as data:
            return LocotrackOfflineResult(
                tracks_xy=np.asarray(data["tracks_xy"], dtype=np.float32),
                visible_mask=np.asarray(data["visible_mask"], dtype=bool),
                occlusion_logits=np.asarray(data["occlusion_logits"], dtype=np.float32),
                expected_dist_logits=np.asarray(data["expected_dist_logits"], dtype=np.float32),
                inference_resolution=tuple(int(v) for v in data["inference_resolution"].tolist()),
                original_resolution=tuple(int(v) for v in data["original_resolution"].tolist()),
                model_size=str(data["model_size"].tolist()),
            )


def _run_locotrack_direct_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: LocotrackRuntimeSettings,
) -> LocotrackOfflineResult:
    torch = importlib.import_module("torch")
    resolved_repo_path = _resolve_locotrack_repo_path(settings.repo_path)
    if resolved_repo_path is None:
        raise FileNotFoundError(
            f"Could not resolve LocoTrack repo from: {settings.repo_path}"
        )

    device = _resolve_torch_device(torch, settings.device)
    model = _load_locotrack_model(torch, settings, resolved_repo_path, device)

    prepared_frames, original_resolution = _prepare_frames_rgb(frames)
    resized_frames = _resize_video_numpy(prepared_frames, settings.inference_resolution)
    resized_resolution = (int(resized_frames.shape[1]), int(resized_frames.shape[2]))
    scaled_query_points = _scale_query_points_tyx(
        query_points_tyx,
        source_resolution=original_resolution,
        target_resolution=resized_resolution,
    )

    frames_tensor = torch.tensor(resized_frames, dtype=torch.float32, device=device)
    frames_tensor = _preprocess_frames_tensor(frames_tensor)[None]
    query_tensor = torch.tensor(scaled_query_points, dtype=torch.float32, device=device)[
        None
    ]

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else nullcontext()
    )
    with torch.no_grad():
        with autocast_context:
            feature_grids = model.get_feature_grids(frames_tensor)
            outputs = model(
                frames_tensor,
                query_tensor,
                feature_grids=feature_grids,
                query_chunk_size=int(max(1, settings.query_chunk_size)),
            )

    tracks = outputs["tracks"][0].detach().to(dtype=torch.float32).cpu().numpy()
    occlusion = outputs["occlusion"][0].detach().to(dtype=torch.float32).cpu().numpy()
    expected_dist = outputs["expected_dist"][0].detach().to(dtype=torch.float32).cpu().numpy()

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

    return LocotrackOfflineResult(
        tracks_xy=tracks_xy,
        visible_mask=visible_mask,
        occlusion_logits=occlusion,
        expected_dist_logits=expected_dist,
        inference_resolution=resized_resolution,
        original_resolution=original_resolution,
        model_size=str(settings.model_size),
    )


def run_locotrack_smoke_test(
    *,
    repo_path: str | Path,
    checkpoint_path: str | Path,
    model_size: str = "small",
    device: str = "auto",
) -> LocotrackSmokeTestResult:
    torch = importlib.import_module("torch")
    locotrack_module = _import_locotrack_model_module(Path(repo_path))
    load_model = getattr(locotrack_module, "load_model")

    resolved_device = _resolve_torch_device(torch, device)
    model = load_model(str(checkpoint_path), model_size=model_size).to(resolved_device)
    model.eval()

    frames = torch.zeros((1, 4, 64, 64, 3), dtype=torch.float32, device=resolved_device)
    frames = frames * 2.0 - 1.0
    query_points = torch.tensor(
        [[[0.0, 32.0, 32.0]]],
        dtype=torch.float32,
        device=resolved_device,
    )

    with torch.no_grad():
        feature_grids = model.get_feature_grids(frames)
        output = model(frames, query_points, feature_grids=feature_grids)

    tracks = output["tracks"].detach().cpu().numpy()
    occlusion = output["occlusion"].detach().cpu().numpy()
    expected_dist = output["expected_dist"].detach().cpu().numpy()

    return LocotrackSmokeTestResult(
        tracks_shape=tuple(int(dim) for dim in tracks.shape),
        occlusion_shape=tuple(int(dim) for dim in occlusion.shape),
        expected_dist_shape=tuple(int(dim) for dim in expected_dist.shape),
        inference_resolution=(64, 64),
        first_track_point=[float(value) for value in tracks[0, 0, 0].tolist()],
        model_size=model_size,
        device=resolved_device,
    )


def _load_locotrack_model(
    torch_module: Any,
    settings: LocotrackRuntimeSettings,
    repo_root: Path,
    device: str,
) -> Any:
    checkpoint_path = Path(settings.checkpoint_path).resolve()
    resolved_model_size = _resolve_locotrack_model_size(settings.model_size, checkpoint_path)
    cache_key = (
        str(repo_root),
        str(checkpoint_path),
        str(resolved_model_size),
        str(device),
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"LocoTrack checkpoint not found: {checkpoint_path}")

    locotrack_module = _import_locotrack_model_module(repo_root)
    load_model = getattr(locotrack_module, "load_model")
    try:
        model = load_model(str(checkpoint_path), model_size=resolved_model_size)
    except RuntimeError as exc:
        inferred_model_size = _infer_locotrack_model_size_from_checkpoint(checkpoint_path)
        if inferred_model_size and inferred_model_size != resolved_model_size:
            try:
                model = load_model(str(checkpoint_path), model_size=inferred_model_size)
                resolved_model_size = inferred_model_size
            except RuntimeError:
                raise ValueError(
                    "Failed to load the selected LocoTrack checkpoint. "
                    f"The checkpoint appears to be '{inferred_model_size}', but the active model size was "
                    f"'{resolved_model_size}'."
                ) from exc
        else:
            raise
    model = model.to(device)
    model = model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _resolve_locotrack_model_size(model_size: str | None, checkpoint_path: Path) -> str:
    normalized = str(model_size or "auto").strip().lower()
    if normalized in {"small", "base"}:
        return normalized
    inferred = _infer_locotrack_model_size_from_checkpoint(checkpoint_path)
    return inferred or "small"


def _infer_locotrack_model_size_from_checkpoint(checkpoint_path: Path) -> Optional[str]:
    name = checkpoint_path.name.lower()
    if "base" in name:
        return "base"
    if "small" in name:
        return "small"
    return None


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoints: list[Path],
    models_dir: Path,
) -> Optional[Path]:
    if checkpoint_path is None:
        return select_default_locotrack_checkpoint(checkpoints)

    candidate = Path(checkpoint_path)
    if not candidate.is_absolute():
        candidate = (models_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _resolve_locotrack_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidate_paths: list[Path] = []
    if repo_path is not None:
        candidate_paths.append(Path(repo_path).expanduser())
    else:
        candidate_paths.extend(DEFAULT_LOCOTRACK_REPO_CANDIDATES)

    for candidate in candidate_paths:
        normalized = _normalize_locotrack_repo_candidate(candidate)
        if normalized is not None:
            return normalized
    return None


def _normalize_locotrack_repo_candidate(candidate: Path) -> Optional[Path]:
    candidate = candidate.resolve()
    repo_marker = candidate / LOCOTRACK_IMPORT_SUBDIR / "models" / "locotrack_model.py"
    if repo_marker.exists():
        return candidate

    subdir_marker = candidate / "models" / "locotrack_model.py"
    if subdir_marker.exists() and candidate.name == LOCOTRACK_IMPORT_SUBDIR:
        return candidate.parent.resolve()
    return None


def _import_locotrack_model_module(repo_root: Path):
    module_dir = (repo_root / LOCOTRACK_IMPORT_SUBDIR).resolve()
    if not module_dir.exists():
        raise FileNotFoundError(f"LocoTrack import directory not found: {module_dir}")

    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)

    existing_models = sys.modules.get("models")
    existing_file = getattr(existing_models, "__file__", "") if existing_models else ""
    if existing_file and module_dir_str not in str(existing_file):
        for module_name in list(sys.modules):
            if module_name == "models" or module_name.startswith("models."):
                del sys.modules[module_name]

    importlib.invalidate_caches()
    return importlib.import_module("models.locotrack_model")


def _resolve_torch_device(torch_module, device: str) -> str:
    requested = (device or "auto").strip().lower()
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "gpu":
        requested = "cuda"
    if requested.startswith("cuda") and not torch_module.cuda.is_available():
        return "cpu"
    return requested


def _resolve_locotrack_python_executable(
    python_executable: str | Path | None,
) -> Path:
    if python_executable is not None:
        return Path(python_executable).expanduser().resolve()
    return Path(sys.executable).resolve()


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
        raise ValueError(f"Invalid LocoTrack inference resolution: {(target_h, target_w)}")

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


def _append_locotrack_import_hints(status: LocotrackEnvironmentStatus, exc: Exception):
    message = str(exc).lower()
    if "lightning" in message:
        status.hints.append("Install Lightning in the active env, e.g. `pip install lightning==2.3.3`.")
    elif "tensorflow" in message:
        status.hints.append(
            "Install TensorFlow-related runtime dependencies from the LocoTrack PyTorch README."
        )
    elif "einops" in message:
        status.hints.append("Install `einops` in the active environment.")
    else:
        status.hints.append(
            "Check that the local LocoTrack repo is cloned and dependencies from "
            "`locotrack_pytorch/README.md` are installed."
        )
