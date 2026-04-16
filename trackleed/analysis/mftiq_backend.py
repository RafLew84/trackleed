"""MFTIQ environment diagnostics and smoke-test helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Optional
import importlib
import inspect
import json
import os
import subprocess
import sys

import numpy as np


DEFAULT_MFTIQ_MODELS_DIR = Path("tracking_models") / "mftiq"
DEFAULT_MFTIQ_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "mftiq",
    Path(__file__).resolve().parents[2] / "mftiq",
)
DEFAULT_MFTIQ_CONFIG = Path("configs") / "MFTIQ4_ROMA_200k_cfg.py"
DEFAULT_MFTIQ_SUBPROCESS_SCRIPT = Path("scripts") / "run_mftiq_subprocess.py"
DEFAULT_MFTIQ_DEMO_INPUT_CANDIDATES = (
    Path("demo_in") / "camel",
    Path("demo_in") / "ugsJtsO9w1A-00.00.24.457-00.00.29.462_HD.mp4",
)
CHECKPOINT_PRIORITY = ("UOM_bs4_200k.pth",)


@dataclass
class MftiqEnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    available_checkpoints: list[Path] = field(default_factory=list)
    selected_checkpoint: Optional[Path] = None
    checkpoint_exists: bool = False
    default_config_path: Optional[Path] = None
    config_exists: bool = False
    demo_input_path: Optional[Path] = None
    demo_input_ready: bool = False
    torch_importable: bool = False
    torch_version: Optional[str] = None
    cuda_available: bool = False
    cv2_importable: bool = False
    einops_importable: bool = False
    xformers_importable: bool = False
    xformers_version: Optional[str] = None
    triton_importable: bool = False
    spatial_correlation_sampler_importable: bool = False
    mftiq_importable: bool = False
    mftiq_module_path: Optional[str] = None
    config_loader_importable: bool = False
    point_tracking_importable: bool = False
    io_utils_importable: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return (
            self.repo_path is not None
            and self.checkpoint_exists
            and self.config_exists
            and self.torch_importable
            and self.cuda_available
            and self.cv2_importable
            and self.einops_importable
            and self.xformers_importable
            and self.spatial_correlation_sampler_importable
            and self.mftiq_importable
            and self.config_loader_importable
            and self.point_tracking_importable
            and self.io_utils_importable
        )


@dataclass
class MftiqSmokeTestResult:
    coords_shape: tuple[int, ...]
    occlusions_shape: tuple[int, ...]
    first_coord: list[float]
    first_occlusion: float
    device: str
    checkpoint_name: str
    config_name: str
    demo_input_name: str


@dataclass
class MftiqInferenceProbeResult:
    tracks_xy: np.ndarray
    occlusion_scores: np.ndarray
    sigma_scores: np.ndarray
    visible_mask: np.ndarray
    device: str
    checkpoint_name: str
    config_name: str
    demo_input_name: str
    num_frames: int
    num_queries: int
    point_query_adapter_supported: bool
    quality_estimation_supported: bool
    supports_native_mid_sequence_queries: bool
    direct_import_runtime_viable: bool
    recommended_backend_mode: str
    notes: list[str] = field(default_factory=list)


@dataclass
class MftiqRuntimeSettings:
    checkpoint_path: str | Path
    repo_path: str | Path | None = None
    python_executable: str | Path | None = None
    config_path: str | Path | None = None
    device: str = "auto"
    backend_mode: str = "subprocess"
    inference_resolution: int | tuple[int, int] | None = None
    occlusion_threshold: float = 0.5
    subprocess_timeout_sec: float = 1800.0


@dataclass
class MftiqOfflineResult:
    tracks_xy: np.ndarray
    visible_mask: np.ndarray
    occlusion_scores: np.ndarray
    sigma_scores: np.ndarray
    inference_resolution: tuple[int, int]
    original_resolution: tuple[int, int]
    checkpoint_name: str
    config_name: str
    backend_mode: str


def discover_mftiq_checkpoints(
    models_dir: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_MFTIQ_MODELS_DIR
    discovered: dict[str, Path] = {}

    for candidate in _discover_checkpoints_in_directory(directory):
        discovered[str(candidate.resolve())] = candidate.resolve()

    repo_root = _resolve_mftiq_repo_path(repo_path)
    if repo_root is not None:
        for candidate in _discover_checkpoints_in_directory(repo_root / "checkpoints"):
            discovered[str(candidate.resolve())] = candidate.resolve()

    return sorted(discovered.values(), key=lambda path: path.name.lower())


def select_default_mftiq_checkpoint(checkpoints: list[Path]) -> Optional[Path]:
    if not checkpoints:
        return None
    by_name = {path.name: path for path in checkpoints}
    for preferred_name in CHECKPOINT_PRIORITY:
        if preferred_name in by_name:
            return by_name[preferred_name]
    return checkpoints[0]


def probe_mftiq_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
    config_path: str | Path | None = None,
    demo_input_path: str | Path | None = None,
) -> MftiqEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_MFTIQ_MODELS_DIR
    directory = directory.resolve()
    resolved_repo_path = _resolve_mftiq_repo_path(repo_path)
    checkpoints = discover_mftiq_checkpoints(directory, resolved_repo_path)
    selected_checkpoint = _resolve_checkpoint_path(
        checkpoint_path,
        checkpoints,
        directory,
        resolved_repo_path,
    )
    resolved_config_path = _resolve_config_path(config_path, resolved_repo_path)
    resolved_demo_input = _resolve_demo_input_path(demo_input_path, resolved_repo_path)

    status = MftiqEnvironmentStatus(
        models_dir=directory,
        repo_path=resolved_repo_path,
        available_checkpoints=checkpoints,
        selected_checkpoint=selected_checkpoint,
        checkpoint_exists=bool(selected_checkpoint and selected_checkpoint.exists()),
        default_config_path=resolved_config_path,
        config_exists=bool(resolved_config_path and resolved_config_path.exists()),
        demo_input_path=resolved_demo_input,
        demo_input_ready=bool(resolved_demo_input and resolved_demo_input.exists()),
    )

    if not directory.exists():
        status.warnings.append(
            f"Checkpoint directory does not exist yet: {directory}. "
            "This is fine if you keep MFTIQ weights inside the local repo/checkpoints directory."
        )

    if selected_checkpoint is None:
        status.warnings.append(
            "No MFTIQ checkpoint detected in tracking_models/mftiq or in repo/checkpoints. "
            "Add `UOM_bs4_200k.pth` before running diagnostics."
        )
    elif checkpoint_path and not status.checkpoint_exists:
        status.errors.append(f"Selected checkpoint not found: {selected_checkpoint}")

    if resolved_repo_path is None:
        status.errors.append(
            "Could not resolve the local MFTIQ repo. Clone `MFTIQ` as `mftiq` next to the "
            "project or pass --repo-path."
        )
        status.hints.append(
            "Expected repo root containing `demo.py`, `configs/`, and `src/MFTIQ/__init__.py`."
        )

    if resolved_config_path is None or not status.config_exists:
        status.errors.append(
            "Could not resolve the default MFTIQ config `configs/MFTIQ4_ROMA_200k_cfg.py`."
        )

    if resolved_demo_input is None or not status.demo_input_ready:
        status.warnings.append(
            "No default demo input detected under `demo_in/`. Environment diagnostics can still "
            "pass, but the optional smoke test will need --demo-input."
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
            "Install the MFTIQ runtime torch stack, e.g. "
            "`torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118`."
        )

    try:
        importlib.import_module("cv2")
        status.cv2_importable = True
    except Exception as exc:
        status.errors.append(f"OpenCV import failed: {exc}")
        status.hints.append("Install the base MFTIQ package via `pip install .`.")

    try:
        importlib.import_module("einops")
        status.einops_importable = True
    except Exception as exc:
        status.errors.append(f"einops import failed: {exc}")
        status.hints.append("Install the base MFTIQ package via `pip install .`.")

    try:
        xformers = importlib.import_module("xformers")
        status.xformers_importable = True
        status.xformers_version = getattr(xformers, "__version__", None)
    except Exception as exc:
        status.errors.append(f"xformers import failed: {exc}")
        status.hints.append(
            "Install a wheel compatible with `torch==2.0.1`, e.g. `xformers==0.0.22`."
        )

    if status.xformers_importable:
        try:
            importlib.import_module("triton")
            status.triton_importable = True
        except Exception:
            status.warnings.append(
                "Triton is not available. MFTIQ can still run, but some xformers optimizations "
                "stay disabled on Windows."
            )

    try:
        importlib.import_module("spatial_correlation_sampler")
        status.spatial_correlation_sampler_importable = True
    except Exception as exc:
        status.errors.append(f"spatial-correlation-sampler import failed: {exc}")
        status.hints.append(
            "Install `spatial-correlation-sampler==0.4.0` in a CUDA 11.8 + MSVC session, "
            "typically with `pip install --no-build-isolation spatial-correlation-sampler==0.4.0`."
        )

    if resolved_repo_path is not None:
        try:
            mftiq_module = _import_mftiq_module(resolved_repo_path, "MFTIQ")
            status.mftiq_importable = True
            status.mftiq_module_path = str(Path(inspect.getfile(mftiq_module)).resolve())
        except Exception as exc:
            status.errors.append(f"MFTIQ package import failed: {exc}")
            _append_mftiq_import_hints(status, exc)

        try:
            _import_mftiq_module(resolved_repo_path, "MFTIQ.config")
            status.config_loader_importable = True
        except Exception as exc:
            status.errors.append(f"MFTIQ config import failed: {exc}")
            _append_mftiq_import_hints(status, exc)

        try:
            _import_mftiq_module(resolved_repo_path, "MFTIQ.point_tracking")
            status.point_tracking_importable = True
        except Exception as exc:
            status.errors.append(f"MFTIQ point-tracking import failed: {exc}")
            _append_mftiq_import_hints(status, exc)

        try:
            _import_mftiq_module(resolved_repo_path, "MFTIQ.utils.io")
            status.io_utils_importable = True
        except Exception as exc:
            status.errors.append(f"MFTIQ io-utils import failed: {exc}")
            _append_mftiq_import_hints(status, exc)

    if status.torch_importable and not status.cuda_available:
        status.warnings.append(
            "Current upstream MFTIQ assumes CUDA-backed models and tracker internals. "
            "CPU runtime is not supported by this diagnostic backend."
        )

    if (
        status.selected_checkpoint is not None
        and status.repo_path is not None
        and str(status.selected_checkpoint).startswith(str((status.repo_path / "checkpoints").resolve()))
    ):
        status.warnings.append(
            "Selected checkpoint is repo-local. This is supported, but you can also move it to "
            "`tracking_models/mftiq/` for consistency with other Trackleed backends."
        )

    return status


def format_mftiq_environment_report(status: MftiqEnvironmentStatus) -> str:
    available = ", ".join(path.name for path in status.available_checkpoints) or "none"
    lines = [
        "MFTIQ environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Selected checkpoint: {status.selected_checkpoint or 'None'}",
        f"- Available checkpoints: {available}",
        f"- Default config: {status.default_config_path or 'None'}",
        f"- Demo input: {status.demo_input_path or 'None'}",
        f"- PyTorch import: {'ok' if status.torch_importable else 'failed'}",
        f"- PyTorch version: {status.torch_version or 'unknown'}",
        f"- CUDA available: {'yes' if status.cuda_available else 'no'}",
        f"- OpenCV import: {'ok' if status.cv2_importable else 'failed'}",
        f"- einops import: {'ok' if status.einops_importable else 'failed'}",
        f"- xformers import: {'ok' if status.xformers_importable else 'failed'}",
        f"- xformers version: {status.xformers_version or 'unknown'}",
        f"- Triton import: {'ok' if status.triton_importable else 'missing'}",
        f"- spatial-correlation-sampler import: "
        f"{'ok' if status.spatial_correlation_sampler_importable else 'failed'}",
        f"- MFTIQ package import: {'ok' if status.mftiq_importable else 'failed'}",
        f"- MFTIQ config import: {'ok' if status.config_loader_importable else 'failed'}",
        f"- MFTIQ point-tracking import: {'ok' if status.point_tracking_importable else 'failed'}",
        f"- MFTIQ io-utils import: {'ok' if status.io_utils_importable else 'failed'}",
    ]
    if status.mftiq_module_path:
        lines.append(f"- MFTIQ module: {status.mftiq_module_path}")
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


def run_mftiq_smoke_test(
    *,
    checkpoint_path: str | Path,
    repo_path: str | Path | None = None,
    device: str = "auto",
    config_path: str | Path | None = None,
    demo_input_path: str | Path | None = None,
) -> MftiqSmokeTestResult:
    torch = importlib.import_module("torch")
    resolved_device = _resolve_torch_device(torch, device)
    if resolved_device.type != "cuda":
        raise RuntimeError(
            "Current upstream MFTIQ smoke test requires CUDA because the tracker and optical-flow "
            "backends create CUDA tensors internally."
        )
    if resolved_device.index is not None:
        torch.cuda.set_device(resolved_device)

    resolved_repo_path = _resolve_mftiq_repo_path(repo_path)
    if resolved_repo_path is None:
        raise FileNotFoundError(
            "Could not resolve the MFTIQ repo. Pass repo_path pointing at the local `mftiq` clone."
        )

    checkpoint_file = _resolve_existing_checkpoint_file(Path(checkpoint_path))
    resolved_config_path = _resolve_config_path(config_path, resolved_repo_path)
    if resolved_config_path is None or not resolved_config_path.exists():
        raise FileNotFoundError(
            f"MFTIQ config not found: {resolved_config_path or DEFAULT_MFTIQ_CONFIG}"
        )
    resolved_demo_input = _resolve_demo_input_path(demo_input_path, resolved_repo_path)
    if resolved_demo_input is None or not resolved_demo_input.exists():
        raise FileNotFoundError(
            f"MFTIQ smoke-test input not found: {resolved_demo_input or 'None'}"
        )

    config_module = _import_mftiq_module(resolved_repo_path, "MFTIQ.config")
    point_tracking_module = _import_mftiq_module(resolved_repo_path, "MFTIQ.point_tracking")
    io_utils_module = _import_mftiq_module(resolved_repo_path, "MFTIQ.utils.io")

    load_config = getattr(config_module, "load_config")
    convert_to_point_tracking = getattr(point_tracking_module, "convert_to_point_tracking")
    get_video_frames = getattr(io_utils_module, "get_video_frames")

    with _temporary_cwd(resolved_repo_path):
        conf = load_config(str(resolved_config_path))
        conf.uom_config.checkpoint = str(checkpoint_file)
        tracker_class = getattr(conf, "tracker_class", None)
        if tracker_class is None:
            raise RuntimeError("Resolved MFTIQ config does not define `tracker_class`.")

        tracker = tracker_class(conf)
        frame_iter = get_video_frames(resolved_demo_input)
        frame0 = _next_video_frame(frame_iter, resolved_demo_input, index=0)
        frame1 = _next_video_frame(frame_iter, resolved_demo_input, index=1)

        height, width = frame0.shape[:2]
        query = torch.tensor(
            [[float(width) * 0.5, float(height) * 0.5]],
            dtype=torch.float32,
            device=resolved_device,
        )

        with torch.inference_mode():
            tracker.init(frame0, keep_on_gpu=True)
            meta = tracker.track(frame1, keep_on_gpu=True)
            coords, occlusions = convert_to_point_tracking(meta.result, query)

    coords_np = np.asarray(coords, dtype=np.float32)
    occlusions_np = np.asarray(occlusions, dtype=np.float32)
    if coords_np.ndim != 2 or coords_np.shape[0] < 1:
        raise RuntimeError(f"Unexpected MFTIQ coords shape from smoke test: {coords_np.shape}")
    if occlusions_np.size < 1:
        raise RuntimeError(
            f"Unexpected MFTIQ occlusion shape from smoke test: {occlusions_np.shape}"
        )

    return MftiqSmokeTestResult(
        coords_shape=tuple(int(dim) for dim in coords_np.shape),
        occlusions_shape=tuple(int(dim) for dim in occlusions_np.shape),
        first_coord=[float(value) for value in coords_np[0].tolist()],
        first_occlusion=float(occlusions_np.reshape(-1)[0]),
        device=str(resolved_device),
        checkpoint_name=checkpoint_file.name,
        config_name=resolved_config_path.name,
        demo_input_name=resolved_demo_input.name,
    )


def format_mftiq_inference_probe_report(result: MftiqInferenceProbeResult) -> str:
    lines = [
        "MFTIQ inference probe",
        f"- Recommended backend mode: {result.recommended_backend_mode}",
        f"- Direct-import runtime viable: {'yes' if result.direct_import_runtime_viable else 'no'}",
        f"- Point-query adapter supported: {'yes' if result.point_query_adapter_supported else 'no'}",
        f"- Quality estimation supported: {'yes' if result.quality_estimation_supported else 'no'}",
        f"- Native mid-sequence queries: {'yes' if result.supports_native_mid_sequence_queries else 'no'}",
        f"- Device: {result.device}",
        f"- Checkpoint: {result.checkpoint_name}",
        f"- Config: {result.config_name}",
        f"- Demo input: {result.demo_input_name}",
        f"- Num frames: {result.num_frames}",
        f"- Num queries: {result.num_queries}",
        f"- Tracks shape: {tuple(int(dim) for dim in result.tracks_xy.shape)}",
        f"- Occlusion shape: {tuple(int(dim) for dim in result.occlusion_scores.shape)}",
        f"- Sigma shape: {tuple(int(dim) for dim in result.sigma_scores.shape)}",
        f"- Visible shape: {tuple(int(dim) for dim in result.visible_mask.shape)}",
        f"- First track point: {[float(value) for value in result.tracks_xy[0, 0].tolist()]}",
        f"- First occlusion: {float(result.occlusion_scores[0, 0])}",
        f"- First sigma: {float(result.sigma_scores[0, 0])}",
    ]
    if result.notes:
        lines.append("- Notes:")
        lines.extend(f"  - {item}" for item in result.notes)
    return "\n".join(lines)


def run_mftiq_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: MftiqRuntimeSettings,
) -> MftiqOfflineResult:
    """
    Run offline MFTIQ on a sequence of frames and sparse query points.

    Args:
        frames: [T, H, W] grayscale or [T, H, W, C] frames.
        query_points_tyx: [Q, 3] points in [t, y, x] in the original frame grid.
        settings: MFTIQ runtime settings.

    Returns:
        MftiqOfflineResult with tracks in original image coordinates.
    """
    frames_np = np.asarray(frames)
    if frames_np.ndim not in (3, 4):
        raise ValueError(f"Expected frames with ndim 3 or 4, got shape {frames_np.shape}.")
    if frames_np.size == 0:
        raise ValueError("No frames provided for MFTIQ inference.")

    query_np = np.asarray(query_points_tyx, dtype=np.float32)
    if query_np.ndim != 2 or query_np.shape[1] != 3:
        raise ValueError(
            "query_points_tyx must have shape [num_queries, 3] in [t, y, x] format."
        )
    if query_np.shape[0] == 0:
        raise ValueError("No query points provided for MFTIQ inference.")

    backend_mode = str(settings.backend_mode or "subprocess").strip().lower()
    if backend_mode == "auto":
        backend_mode = "subprocess"

    if backend_mode == "direct":
        return _run_mftiq_direct_offline(frames_np, query_np, settings)
    if backend_mode != "subprocess":
        raise ValueError(
            f"Unsupported MFTIQ backend mode: {settings.backend_mode}. "
            "Expected `subprocess` or `direct`."
        )
    return _run_mftiq_subprocess_offline(frames_np, query_np, settings)


def run_mftiq_inference_probe(
    *,
    checkpoint_path: str | Path,
    repo_path: str | Path | None = None,
    device: str = "auto",
    config_path: str | Path | None = None,
    demo_input_path: str | Path | None = None,
    num_frames: int = 4,
    occlusion_threshold: float = 0.5,
) -> MftiqInferenceProbeResult:
    torch = importlib.import_module("torch")
    resolved_device = _resolve_torch_device(torch, device)
    if resolved_device.type != "cuda":
        raise RuntimeError(
            "Current upstream MFTIQ inference probe requires CUDA because the tracker and "
            "optical-flow backends create CUDA tensors internally."
        )
    if resolved_device.index is not None:
        torch.cuda.set_device(resolved_device)

    if num_frames < 2:
        raise ValueError("MFTIQ inference probe requires at least two frames.")

    resolved_repo_path = _resolve_mftiq_repo_path(repo_path)
    if resolved_repo_path is None:
        raise FileNotFoundError(
            "Could not resolve the MFTIQ repo. Pass repo_path pointing at the local `mftiq` clone."
        )

    checkpoint_file = _resolve_existing_checkpoint_file(Path(checkpoint_path))
    resolved_config_path = _resolve_config_path(config_path, resolved_repo_path)
    if resolved_config_path is None or not resolved_config_path.exists():
        raise FileNotFoundError(
            f"MFTIQ config not found: {resolved_config_path or DEFAULT_MFTIQ_CONFIG}"
        )
    resolved_demo_input = _resolve_demo_input_path(demo_input_path, resolved_repo_path)
    if resolved_demo_input is None or not resolved_demo_input.exists():
        raise FileNotFoundError(
            f"MFTIQ inference-probe input not found: {resolved_demo_input or 'None'}"
        )

    config_module = _import_mftiq_module(resolved_repo_path, "MFTIQ.config")
    point_tracking_module = _import_mftiq_module(resolved_repo_path, "MFTIQ.point_tracking")
    io_utils_module = _import_mftiq_module(resolved_repo_path, "MFTIQ.utils.io")

    load_config = getattr(config_module, "load_config")
    convert_to_point_tracking = getattr(point_tracking_module, "convert_to_point_tracking")
    get_video_frames = getattr(io_utils_module, "get_video_frames")

    notes = [
        "Official MFTIQ runtime naturally tracks points queried on the init frame.",
        "Mid-sequence seed points require re-initializing the tracker on a subclip starting at the seed frame.",
        "Quality estimation is available via sampled `occlusion` and `sigma` values from FlowOUTrackingResult.",
        "Trackleed should prefer a subprocess backend to keep the heavy MFTIQ stack isolated from the main app env.",
    ]

    with _temporary_cwd(resolved_repo_path):
        conf = load_config(str(resolved_config_path))
        conf.uom_config.checkpoint = str(checkpoint_file)
        tracker_class = getattr(conf, "tracker_class", None)
        if tracker_class is None:
            raise RuntimeError("Resolved MFTIQ config does not define `tracker_class`.")

        tracker = tracker_class(conf)
        frame_iter = get_video_frames(resolved_demo_input)
        frames = [
            _next_video_frame(frame_iter, resolved_demo_input, index=index)
            for index in range(num_frames)
        ]
        frame0 = frames[0]
        queries = _build_mftiq_probe_queries(
            torch_module=torch,
            frame_shape=frame0.shape[:2],
            device=resolved_device,
        )

        results = []
        with torch.inference_mode():
            init_meta = tracker.init(frame0, keep_on_gpu=True)
            results.append(init_meta.result.clone())
            for frame in frames[1:]:
                meta = tracker.track(frame, keep_on_gpu=True)
                results.append(meta.result.clone())

        tracks_xy = np.zeros((queries.shape[0], len(results), 2), dtype=np.float32)
        occlusion_scores = np.zeros((queries.shape[0], len(results)), dtype=np.float32)
        sigma_scores = np.zeros((queries.shape[0], len(results)), dtype=np.float32)

        for frame_index, result in enumerate(results):
            coords, occlusions = convert_to_point_tracking(result, queries)
            _, _, sampled_sigmas = result.sample(queries)

            coords_np = np.asarray(coords, dtype=np.float32)
            occlusions_np = np.asarray(occlusions, dtype=np.float32).reshape(-1)
            sigmas_np = (
                sampled_sigmas.detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)
            )

            if coords_np.shape[0] != queries.shape[0]:
                raise RuntimeError(
                    "MFTIQ inference probe returned unexpected coord count: "
                    f"{coords_np.shape[0]} vs {queries.shape[0]}"
                )

            tracks_xy[:, frame_index, :] = coords_np
            occlusion_scores[:, frame_index] = occlusions_np
            sigma_scores[:, frame_index] = sigmas_np

    visible_mask = (
        np.isfinite(tracks_xy[..., 0])
        & np.isfinite(tracks_xy[..., 1])
        & np.isfinite(occlusion_scores)
        & (occlusion_scores <= float(occlusion_threshold))
    )

    return MftiqInferenceProbeResult(
        tracks_xy=tracks_xy,
        occlusion_scores=occlusion_scores,
        sigma_scores=sigma_scores,
        visible_mask=visible_mask,
        device=str(resolved_device),
        checkpoint_name=checkpoint_file.name,
        config_name=resolved_config_path.name,
        demo_input_name=resolved_demo_input.name,
        num_frames=int(len(results)),
        num_queries=int(queries.shape[0]),
        point_query_adapter_supported=True,
        quality_estimation_supported=True,
        supports_native_mid_sequence_queries=False,
        direct_import_runtime_viable=True,
        recommended_backend_mode="subprocess",
        notes=notes,
    )


def _run_mftiq_subprocess_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: MftiqRuntimeSettings,
) -> MftiqOfflineResult:
    python_executable = _resolve_mftiq_python_executable(settings.python_executable)
    script_path = (Path(__file__).resolve().parents[2] / DEFAULT_MFTIQ_SUBPROCESS_SCRIPT).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"MFTIQ subprocess script not found: {script_path}")

    inference_resolution = _normalize_optional_resolution(settings.inference_resolution)

    with TemporaryDirectory(prefix="trackleed_mftiq_") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "mftiq_input.npz"
        output_path = temp_root / "mftiq_output.npz"
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
            "--device",
            str(settings.device),
            "--occlusion-threshold",
            str(float(settings.occlusion_threshold)),
        ]
        if settings.repo_path is not None:
            command.extend(["--repo-path", str(settings.repo_path)])
        if settings.config_path is not None:
            command.extend(["--config", str(settings.config_path)])
        if inference_resolution is not None:
            command.extend(
                [
                    "--height",
                    str(int(inference_resolution[0])),
                    "--width",
                    str(int(inference_resolution[1])),
                ]
            )

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
                "MFTIQ subprocess inference failed:\n"
                + json.dumps(payload, indent=2, ensure_ascii=False)
            )

        if not output_path.exists():
            raise FileNotFoundError(f"MFTIQ subprocess did not create output: {output_path}")

        with np.load(output_path, allow_pickle=False) as data:
            return MftiqOfflineResult(
                tracks_xy=np.asarray(data["tracks_xy"], dtype=np.float32),
                visible_mask=np.asarray(data["visible_mask"], dtype=bool),
                occlusion_scores=np.asarray(data["occlusion_scores"], dtype=np.float32),
                sigma_scores=np.asarray(data["sigma_scores"], dtype=np.float32),
                inference_resolution=tuple(int(v) for v in data["inference_resolution"].tolist()),
                original_resolution=tuple(int(v) for v in data["original_resolution"].tolist()),
                checkpoint_name=str(data["checkpoint_name"].tolist()),
                config_name=str(data["config_name"].tolist()),
                backend_mode="subprocess",
            )


def _run_mftiq_direct_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: MftiqRuntimeSettings,
) -> MftiqOfflineResult:
    torch = importlib.import_module("torch")
    resolved_device = _resolve_torch_device(torch, settings.device)
    if resolved_device.type != "cuda":
        raise RuntimeError(
            "Current upstream MFTIQ runtime requires CUDA because the tracker and optical-flow "
            "backends create CUDA tensors internally."
        )
    if resolved_device.index is not None:
        torch.cuda.set_device(resolved_device)

    resolved_repo_path = _resolve_mftiq_repo_path(settings.repo_path)
    if resolved_repo_path is None:
        raise FileNotFoundError(
            "Could not resolve the MFTIQ repo. Pass repo_path pointing at the local `mftiq` clone."
        )

    checkpoint_file = _resolve_existing_checkpoint_file(Path(settings.checkpoint_path))
    resolved_config_path = _resolve_config_path(settings.config_path, resolved_repo_path)
    if resolved_config_path is None or not resolved_config_path.exists():
        raise FileNotFoundError(
            f"MFTIQ config not found: {resolved_config_path or DEFAULT_MFTIQ_CONFIG}"
        )

    prepared_frames, original_resolution = _prepare_frames_bgr_uint8(np.asarray(frames))
    inference_resolution = _normalize_optional_resolution(settings.inference_resolution)
    if inference_resolution is None:
        inference_resolution = original_resolution
    resized_frames = _resize_video_numpy(prepared_frames, inference_resolution)
    scaled_queries = _scale_query_points_tyx(
        np.asarray(query_points_tyx, dtype=np.float32),
        source_resolution=original_resolution,
        target_resolution=inference_resolution,
    )
    seed_frames = _extract_query_seed_frames(scaled_queries, total_frames=resized_frames.shape[0])

    config_module = _import_mftiq_module(resolved_repo_path, "MFTIQ.config")
    point_tracking_module = _import_mftiq_module(resolved_repo_path, "MFTIQ.point_tracking")
    load_config = getattr(config_module, "load_config")
    convert_to_point_tracking = getattr(point_tracking_module, "convert_to_point_tracking")

    num_queries = int(scaled_queries.shape[0])
    num_frames = int(resized_frames.shape[0])
    tracks_xy_resized = np.full((num_queries, num_frames, 2), np.nan, dtype=np.float32)
    occlusion_scores = np.full((num_queries, num_frames), np.inf, dtype=np.float32)
    sigma_scores = np.full((num_queries, num_frames), np.inf, dtype=np.float32)

    with _temporary_cwd(resolved_repo_path):
        conf = load_config(str(resolved_config_path))
        conf.uom_config.checkpoint = str(checkpoint_file)
        tracker_class = getattr(conf, "tracker_class", None)
        if tracker_class is None:
            raise RuntimeError("Resolved MFTIQ config does not define `tracker_class`.")
        tracker = tracker_class(conf)

        for seed_frame in np.unique(seed_frames).tolist():
            query_indices = np.where(seed_frames == seed_frame)[0]
            local_frames = resized_frames[seed_frame:]
            if local_frames.shape[0] <= 0:
                continue

            query_xy = np.stack(
                [
                    scaled_queries[query_indices, 2],
                    scaled_queries[query_indices, 1],
                ],
                axis=-1,
            ).astype(np.float32, copy=False)
            query_tensor = torch.tensor(
                query_xy,
                dtype=torch.float32,
                device=resolved_device,
            )

            with torch.inference_mode():
                init_meta = tracker.init(local_frames[0], keep_on_gpu=True)
                _record_mftiq_frame_result(
                    result=init_meta.result,
                    convert_to_point_tracking=convert_to_point_tracking,
                    queries=query_tensor,
                    tracks_xy=tracks_xy_resized,
                    occlusion_scores=occlusion_scores,
                    sigma_scores=sigma_scores,
                    query_indices=query_indices,
                    frame_index=int(seed_frame),
                )

                for local_index, frame in enumerate(local_frames[1:], start=1):
                    meta = tracker.track(frame, keep_on_gpu=True)
                    _record_mftiq_frame_result(
                        result=meta.result,
                        convert_to_point_tracking=convert_to_point_tracking,
                        queries=query_tensor,
                        tracks_xy=tracks_xy_resized,
                        occlusion_scores=occlusion_scores,
                        sigma_scores=sigma_scores,
                        query_indices=query_indices,
                        frame_index=int(seed_frame + local_index),
                    )

    tracks_xy = _scale_tracks_xy(
        tracks_xy_resized,
        source_resolution=inference_resolution,
        target_resolution=original_resolution,
    )
    visible_mask = _build_mftiq_visibility_mask(
        tracks_xy=tracks_xy,
        occlusion_scores=occlusion_scores,
        frame_resolution=original_resolution,
        occlusion_threshold=float(settings.occlusion_threshold),
    )
    backend_mode = str(settings.backend_mode or "direct").strip().lower()
    return MftiqOfflineResult(
        tracks_xy=tracks_xy,
        visible_mask=visible_mask,
        occlusion_scores=occlusion_scores,
        sigma_scores=sigma_scores,
        inference_resolution=inference_resolution,
        original_resolution=original_resolution,
        checkpoint_name=checkpoint_file.name,
        config_name=resolved_config_path.name,
        backend_mode="direct" if backend_mode == "direct" else backend_mode,
    )


def _discover_checkpoints_in_directory(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        (path.resolve() for path in directory.glob("*.pth") if path.is_file()),
        key=lambda path: path.name.lower(),
    )


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoints: list[Path],
    models_dir: Path,
    repo_path: Optional[Path],
) -> Optional[Path]:
    if checkpoint_path is None:
        return select_default_mftiq_checkpoint(checkpoints)

    candidate = Path(checkpoint_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    available_by_name = {path.name: path for path in checkpoints}
    if candidate.name in available_by_name:
        return available_by_name[candidate.name]

    candidate_paths = [(models_dir / candidate).resolve()]
    if repo_path is not None:
        candidate_paths.append((repo_path / "checkpoints" / candidate).resolve())
        candidate_paths.append((repo_path / candidate).resolve())
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    return candidate_paths[0]


def _resolve_existing_checkpoint_file(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"MFTIQ checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix.lower() != ".pth":
        raise FileNotFoundError(f"MFTIQ checkpoint is not a .pth file: {checkpoint_path}")
    return checkpoint_path


def _resolve_mftiq_python_executable(
    python_executable: str | Path | None,
) -> Path:
    if python_executable is not None:
        return Path(python_executable).expanduser().resolve()
    return Path(sys.executable).resolve()


def _resolve_mftiq_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidate_paths: list[Path] = []
    if repo_path is not None:
        candidate_paths.append(Path(repo_path).expanduser())
    else:
        candidate_paths.extend(DEFAULT_MFTIQ_REPO_CANDIDATES)

    for candidate in candidate_paths:
        normalized = _normalize_mftiq_repo_candidate(candidate)
        if normalized is not None:
            return normalized
    return None


def _normalize_mftiq_repo_candidate(candidate: Path) -> Optional[Path]:
    candidate = candidate.resolve()
    if (
        (candidate / "demo.py").exists()
        and (candidate / "configs").is_dir()
        and (candidate / "src" / "MFTIQ" / "__init__.py").exists()
    ):
        return candidate
    return None


def _resolve_config_path(
    config_path: str | Path | None,
    repo_path: Optional[Path],
) -> Optional[Path]:
    if config_path is None:
        if repo_path is None:
            return None
        return (repo_path / DEFAULT_MFTIQ_CONFIG).resolve()

    candidate = Path(config_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    if repo_path is not None:
        candidate_paths = [
            (repo_path / candidate).resolve(),
            (repo_path / "configs" / candidate).resolve(),
        ]
        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return candidate_path
        return candidate_paths[0]

    return candidate.resolve()


def _resolve_demo_input_path(
    demo_input_path: str | Path | None,
    repo_path: Optional[Path],
) -> Optional[Path]:
    if demo_input_path is not None:
        candidate = Path(demo_input_path).expanduser()
        if candidate.exists():
            return candidate.resolve()
        if repo_path is not None:
            repo_candidate = (repo_path / candidate).resolve()
            if repo_candidate.exists():
                return repo_candidate
            return repo_candidate
        return candidate.resolve()

    if repo_path is None:
        return None
    for relative_path in DEFAULT_MFTIQ_DEMO_INPUT_CANDIDATES:
        candidate = (repo_path / relative_path).resolve()
        if candidate.exists():
            return candidate
    return None


def _ensure_repo_paths_on_sys_path(repo_root: Optional[Path]) -> None:
    if repo_root is None:
        return
    repo_paths = [repo_root.resolve(), (repo_root / "src").resolve()]
    for path in repo_paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _import_mftiq_module(repo_root: Optional[Path], module_name: str) -> Any:
    _ensure_repo_paths_on_sys_path(repo_root)
    return importlib.import_module(module_name)


def _append_mftiq_import_hints(status: MftiqEnvironmentStatus, exc: Exception) -> None:
    message = str(exc).lower()
    if "no module named 'mftiq'" in message:
        status.hints.append(
            "Pass --repo-path pointing at the local `mftiq` repo root so that both repo/ and "
            "repo/src are added to sys.path."
        )
    if "no module named 'xformers'" in message:
        status.hints.append("Install `xformers==0.0.22` in the `mftiq` environment.")
    if "no module named 'spatial_correlation_sampler'" in message:
        status.hints.append(
            "Install `spatial-correlation-sampler==0.4.0` with CUDA 11.8 and MSVC loaded."
        )
    if "no module named 'pkg_resources'" in message:
        status.hints.append(
            "Use `setuptools<81` in the `mftiq` environment, e.g. `setuptools==80.10.2`."
        )
    if "no module named 'cv2'" in message:
        status.hints.append("Install the base MFTIQ dependencies via `pip install .`.")
    if "no module named 'einops'" in message:
        status.hints.append("Install the base MFTIQ dependencies via `pip install .`.")
    if not status.hints:
        status.hints.append(
            "Verify that the local `mftiq` repo is available and that its base + full runtime "
            "dependencies are installed in the active environment."
        )


def _resolve_torch_device(torch_module, requested_device: str):
    device = str(requested_device or "auto").strip().lower()
    if device in {"auto", "gpu"}:
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if device == "cuda" and not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    return torch_module.device(device)


def _next_video_frame(frame_iter, input_path: Path, *, index: int):
    try:
        return next(frame_iter)
    except StopIteration as exc:
        raise RuntimeError(
            f"MFTIQ smoke test expected at least {index + 1} frames in {input_path}."
        ) from exc


def _build_mftiq_probe_queries(torch_module, frame_shape: tuple[int, int], device) -> Any:
    height, width = frame_shape
    raw_points = np.array(
        [
            [0.50 * width, 0.50 * height],
            [0.30 * width, 0.30 * height],
            [0.70 * width, 0.30 * height],
            [0.50 * width, 0.70 * height],
        ],
        dtype=np.float32,
    )
    raw_points[:, 0] = np.clip(raw_points[:, 0], 1.0, max(1.0, float(width - 2)))
    raw_points[:, 1] = np.clip(raw_points[:, 1], 1.0, max(1.0, float(height - 2)))
    return torch_module.tensor(raw_points, dtype=torch_module.float32, device=device)


def _record_mftiq_frame_result(
    *,
    result,
    convert_to_point_tracking,
    queries,
    tracks_xy: np.ndarray,
    occlusion_scores: np.ndarray,
    sigma_scores: np.ndarray,
    query_indices: np.ndarray,
    frame_index: int,
) -> None:
    coords, occlusions = convert_to_point_tracking(result, queries)
    _, _, sampled_sigmas = result.sample(queries)

    coords_np = np.asarray(coords, dtype=np.float32)
    occlusions_np = np.asarray(occlusions, dtype=np.float32).reshape(-1)
    sigmas_np = sampled_sigmas.detach().to(dtype=queries.dtype).cpu().numpy().reshape(-1)

    tracks_xy[query_indices, frame_index, :] = coords_np
    occlusion_scores[query_indices, frame_index] = occlusions_np
    sigma_scores[query_indices, frame_index] = sigmas_np.astype(np.float32, copy=False)


def _prepare_frames_bgr_uint8(frames: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    frames_np = np.asarray(frames)
    if frames_np.ndim == 3:
        frames_np = np.stack([frames_np] * 3, axis=-1)
    elif frames_np.ndim == 4 and frames_np.shape[-1] == 1:
        frames_np = np.repeat(frames_np, 3, axis=-1)
    elif frames_np.ndim == 4 and frames_np.shape[-1] >= 3:
        frames_np = frames_np[..., :3]
    else:
        raise ValueError(f"Unsupported frame shape for MFTIQ: {frames_np.shape}")

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
    target: int | tuple[int, int] | None,
) -> Optional[tuple[int, int]]:
    if target is None:
        return None
    if isinstance(target, int):
        target_h = target_w = int(target)
    else:
        target_h = int(target[0])
        target_w = int(target[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid MFTIQ inference resolution: {(target_h, target_w)}")
    return (target_h, target_w)


def _resize_video_numpy(frames: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    cv2 = importlib.import_module("cv2")
    target_h, target_w = target
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
        raise ValueError("MFTIQ query seed frames must be integer-valued.")
    if np.any(seed_frames < 0) or np.any(seed_frames >= total_frames):
        raise ValueError(
            f"MFTIQ query seed frame out of range for sequence length {total_frames}."
        )
    return seed_frames


def _build_mftiq_visibility_mask(
    *,
    tracks_xy: np.ndarray,
    occlusion_scores: np.ndarray,
    frame_resolution: tuple[int, int],
    occlusion_threshold: float,
) -> np.ndarray:
    height, width = frame_resolution
    visible = np.isfinite(tracks_xy[..., 0]) & np.isfinite(tracks_xy[..., 1])
    visible &= np.isfinite(occlusion_scores)
    visible &= occlusion_scores <= float(occlusion_threshold)
    visible &= tracks_xy[..., 0] >= 0.0
    visible &= tracks_xy[..., 0] <= float(max(0, width - 1))
    visible &= tracks_xy[..., 1] >= 0.0
    visible &= tracks_xy[..., 1] <= float(max(0, height - 1))
    return visible


@contextmanager
def _temporary_cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
