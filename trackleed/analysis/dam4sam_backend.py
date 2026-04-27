"""DAM4SAM environment diagnostics and smoke-test helpers."""

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
from typing import Any, Iterator, Optional

import numpy as np


DEFAULT_DAM4SAM_MODELS_DIR = Path("tracking_models") / "dam4sam"
DEFAULT_DAM4SAM_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "dam4sam",
    Path(__file__).resolve().parents[3] / "DAM4SAM",
    Path(__file__).resolve().parents[2] / "dam4sam",
)
DEFAULT_DAM4SAM_TRACKER_NAME = "sam21pp-L"
DEFAULT_DAM4SAM_IMPORT_PROBE_TIMEOUT_SEC = 60.0
DEFAULT_DAM4SAM_SMOKE_TIMEOUT_SEC = 600.0
DEFAULT_DAM4SAM_SUBPROCESS_TIMEOUT_SEC = 1800.0
DEFAULT_DAM4SAM_SUBPROCESS_SCRIPT = Path("scripts") / "run_dam4sam_subprocess.py"
_IMPORT_PROBE_PREFIX = "__TRACKLEED_DAM4SAM_IMPORT_PROBE__"
_SMOKE_TEST_PREFIX = "__TRACKLEED_DAM4SAM_SMOKE_TEST__"

DAM4SAM_TRACKER_VARIANTS = {
    "sam21pp-L": ("sam2.1_hiera_large.pt", "sam21pp_hiera_l.yaml"),
    "sam21pp-B": ("sam2.1_hiera_base_plus.pt", "sam21pp_hiera_b+.yaml"),
    "sam21pp-S": ("sam2.1_hiera_small.pt", "sam21pp_hiera_s.yaml"),
    "sam21pp-T": ("sam2.1_hiera_tiny.pt", "sam21pp_hiera_t.yaml"),
    "sam2pp-L": ("sam2_hiera_large.pt", "sam2pp_hiera_l.yaml"),
    "sam2pp-B": ("sam2_hiera_base_plus.pt", "sam2pp_hiera_b+.yaml"),
    "sam2pp-S": ("sam2_hiera_small.pt", "sam2pp_hiera_s.yaml"),
    "sam2pp-T": ("sam2_hiera_tiny.pt", "sam2pp_hiera_t.yaml"),
}


@dataclass
class Dam4SamImportProbeResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any]


@dataclass
class Dam4SamEnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    python_executable: Optional[Path] = None
    tracker_name: str = DEFAULT_DAM4SAM_TRACKER_NAME
    config_name: Optional[str] = None
    config_path: Optional[Path] = None
    config_exists: bool = False
    available_checkpoints: list[Path] = field(default_factory=list)
    selected_checkpoint: Optional[Path] = None
    checkpoint_exists: bool = False
    upstream_checkpoint_path: Optional[Path] = None
    upstream_checkpoint_exists: bool = False
    import_probe_returncode: Optional[int] = None
    import_probe_stdout: str = ""
    import_probe_stderr: str = ""
    torch_importable: bool = False
    torch_version: Optional[str] = None
    torch_cuda_version: Optional[str] = None
    cuda_available: bool = False
    torchvision_importable: bool = False
    cv2_importable: bool = False
    numpy_importable: bool = False
    numpy_version: Optional[str] = None
    yaml_importable: bool = False
    hydra_importable: bool = False
    omegaconf_importable: bool = False
    pillow_importable: bool = False
    vot_importable: bool = False
    sam2_importable: bool = False
    sam2_module_path: Optional[str] = None
    build_sam_importable: bool = False
    dam4sam_tracker_importable: bool = False
    dam4sam_tracker_module_path: Optional[str] = None
    vot_wrapper_importable: bool = False
    vot_wrapper_module_path: Optional[str] = None
    cuda_extension_importable: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return (
            self.repo_path is not None
            and self.python_executable is not None
            and self.python_executable.exists()
            and self.config_exists
            and self.checkpoint_exists
            and self.torch_importable
            and self.cuda_available
            and self.torchvision_importable
            and self.cv2_importable
            and self.numpy_importable
            and self.yaml_importable
            and self.hydra_importable
            and self.omegaconf_importable
            and self.pillow_importable
            and self.vot_importable
            and self.sam2_importable
            and self.build_sam_importable
            and self.dam4sam_tracker_importable
            and self.vot_wrapper_importable
        )


@dataclass
class Dam4SamSmokeTestResult:
    command: list[str]
    tracker_name: str
    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any]

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and bool(self.payload.get("ok"))


@dataclass
class Dam4SamRuntimeSettings:
    checkpoint_path: str | Path
    repo_path: str | Path | None = None
    python_executable: str | Path | None = None
    backend_mode: str = "subprocess"
    tracker_name: str = DEFAULT_DAM4SAM_TRACKER_NAME
    device: str = "auto"
    representative_point_mode: str = "mask_centroid"
    inference_resolution: int | tuple[int, int] | None = None
    initial_bbox_padding_px: float = 12.0
    subprocess_timeout_sec: float = DEFAULT_DAM4SAM_SUBPROCESS_TIMEOUT_SEC


@dataclass
class Dam4SamOfflineResult:
    masks: np.ndarray
    tracks_xy: np.ndarray
    visible_mask: np.ndarray
    mask_areas: np.ndarray
    mask_bboxes_xyxy: np.ndarray
    mask_bboxes_xywh: np.ndarray
    mask_scores: np.ndarray
    inference_resolution: tuple[int, int]
    original_resolution: tuple[int, int]
    checkpoint_name: str
    tracker_name: str
    config_name: str
    representative_point_mode: str
    backend_mode: str


@dataclass
class Dam4SamBackendSmokeTestResult:
    checkpoint_name: str
    tracker_name: str
    backend_mode: str
    representative_point_mode: str
    masks_shape: tuple[int, int, int, int]
    tracks_shape: tuple[int, int, int]
    visible_shape: tuple[int, int]
    visible_count: int
    first_track_point_xy: tuple[float, float]
    last_visible_point_xy: tuple[float, float]
    first_mask_area: float
    last_mask_area: float
    first_bbox_xyxy: tuple[float, float, float, float]
    last_bbox_xyxy: tuple[float, float, float, float]


def discover_dam4sam_checkpoints(
    models_dir: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_DAM4SAM_MODELS_DIR
    discovered: dict[str, Path] = {}
    for candidate in _discover_checkpoints_in_directory(directory):
        discovered[str(candidate.resolve())] = candidate.resolve()

    repo_root = _resolve_dam4sam_repo_path(repo_path)
    if repo_root is not None:
        for candidate in _discover_checkpoints_in_directory(repo_root / "checkpoints"):
            discovered[str(candidate.resolve())] = candidate.resolve()

    return sorted(discovered.values(), key=lambda path: (path.name.lower(), str(path).lower()))


def select_default_dam4sam_checkpoint(
    checkpoints: list[Path],
    tracker_name: str = DEFAULT_DAM4SAM_TRACKER_NAME,
    models_dir: str | Path | None = None,
) -> Optional[Path]:
    if not checkpoints:
        return None
    expected_name = _checkpoint_name_for_tracker(tracker_name)
    if expected_name is None:
        return checkpoints[0]

    matches = [path for path in checkpoints if path.name.lower() == expected_name.lower()]
    if not matches:
        return checkpoints[0]

    preferred_dir = (
        Path(models_dir).expanduser().resolve()
        if models_dir is not None
        else DEFAULT_DAM4SAM_MODELS_DIR.resolve()
    )
    for path in matches:
        try:
            path.relative_to(preferred_dir)
            return path
        except ValueError:
            continue
    return matches[0]


def probe_dam4sam_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
    python_executable: str | Path | None = None,
    tracker_name: str = DEFAULT_DAM4SAM_TRACKER_NAME,
    import_probe_timeout_sec: float = DEFAULT_DAM4SAM_IMPORT_PROBE_TIMEOUT_SEC,
) -> Dam4SamEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_DAM4SAM_MODELS_DIR
    directory = directory.resolve()
    normalized_tracker_name = _normalize_tracker_name(tracker_name)
    resolved_repo_path = _resolve_dam4sam_repo_path(repo_path)
    resolved_python = _resolve_dam4sam_python_executable(python_executable)
    config_name = _config_name_for_tracker(normalized_tracker_name)
    config_path = (
        resolved_repo_path / "sam2" / config_name
        if resolved_repo_path is not None and config_name is not None
        else None
    )
    upstream_checkpoint_path = (
        resolved_repo_path / "checkpoints" / _checkpoint_name_for_tracker(normalized_tracker_name)
        if resolved_repo_path is not None and _checkpoint_name_for_tracker(normalized_tracker_name)
        else None
    )
    checkpoints = discover_dam4sam_checkpoints(directory, resolved_repo_path)
    selected_checkpoint = _resolve_checkpoint_path(
        checkpoint_path,
        checkpoints,
        directory,
        normalized_tracker_name,
    )

    status = Dam4SamEnvironmentStatus(
        models_dir=directory,
        repo_path=resolved_repo_path,
        python_executable=resolved_python,
        tracker_name=normalized_tracker_name,
        config_name=config_name,
        config_path=config_path,
        config_exists=bool(config_path and config_path.exists()),
        available_checkpoints=checkpoints,
        selected_checkpoint=selected_checkpoint,
        checkpoint_exists=bool(selected_checkpoint and selected_checkpoint.exists()),
        upstream_checkpoint_path=upstream_checkpoint_path,
        upstream_checkpoint_exists=bool(
            upstream_checkpoint_path and upstream_checkpoint_path.exists()
        ),
    )

    if repo_path and resolved_repo_path is None:
        status.errors.append(f"Could not resolve DAM4SAM repo from: {repo_path}")
    elif resolved_repo_path is None:
        status.errors.append(
            "Could not resolve DAM4SAM repo. Pass --repo-path pointing at the local DAM4SAM clone."
        )

    if resolved_python is None or not resolved_python.exists():
        status.errors.append(
            f"DAM4SAM Python executable not found: {resolved_python or python_executable}"
        )

    if not directory.exists():
        status.warnings.append(
            f"Checkpoint directory does not exist yet: {directory}. "
            "Place DAM4SAM/SAM2.1 checkpoints inside tracking_models/dam4sam."
        )

    if normalized_tracker_name not in DAM4SAM_TRACKER_VARIANTS:
        status.errors.append(
            f"Unsupported DAM4SAM tracker variant: {tracker_name}. "
            f"Supported variants: {', '.join(sorted(DAM4SAM_TRACKER_VARIANTS))}."
        )

    if config_path is None:
        status.errors.append("DAM4SAM config path cannot be resolved without repo path.")
    elif not config_path.exists():
        status.errors.append(f"DAM4SAM config not found: {config_path}")

    if selected_checkpoint is None:
        status.errors.append(
            "No DAM4SAM/SAM2 checkpoint detected. Expected SAM2.1 checkpoints in "
            "tracking_models/dam4sam or in repo/checkpoints."
        )
    elif checkpoint_path and not status.checkpoint_exists:
        status.errors.append(f"Selected checkpoint not found: {selected_checkpoint}")

    if resolved_python is not None and resolved_python.exists():
        probe = _run_dam4sam_import_probe(
            resolved_python,
            resolved_repo_path,
            timeout_sec=import_probe_timeout_sec,
        )
        _apply_import_probe_result(status, probe)

    if not status.cuda_available and status.torch_importable:
        status.errors.append("PyTorch CUDA is not available in the DAM4SAM environment.")

    if _numpy_major_version(status.numpy_version) is not None:
        numpy_major = _numpy_major_version(status.numpy_version)
        if numpy_major is not None and numpy_major >= 2 and str(status.torch_version or "").startswith("2.1"):
            status.errors.append(
                "NumPy 2.x is not compatible with the validated torch 2.1.0 DAM4SAM stack. "
                "Pin numpy==1.26.4."
            )

    if not status.cuda_extension_importable:
        status.warnings.append(
            "SAM2 custom CUDA extension is not importable. This is acceptable for feasibility; "
            "mask post-processing may be skipped, as in the SAM2 integration."
        )

    if normalized_tracker_name in {"sam21pp-S", "sam21pp-T", "sam2pp-S", "sam2pp-T"}:
        status.warnings.append(
            "Upstream DAM4SAM determine_tracker() may look for Small/Tiny checkpoints in the "
            "repo root instead of repo/checkpoints. Prefer sam21pp-L/sam21pp-B for initial checks "
            "unless this is patched or the files are copied to the expected location."
        )

    if not status.is_ready:
        status.hints.append(
            "Run from the dedicated `dam4sam` Conda env or pass "
            "--python-executable C:\\Users\\rlewa\\anaconda3\\envs\\dam4sam\\python.exe."
        )
        status.hints.append(
            "Validated feasibility stack: Python 3.10, torch 2.1.0+cu121, "
            "torchvision 0.16.0, numpy 1.26.4, vot-toolkit 0.7.1."
        )

    return status


def format_dam4sam_environment_report(status: Dam4SamEnvironmentStatus) -> str:
    available = _format_available_checkpoints(status.available_checkpoints)
    selected = str(status.selected_checkpoint) if status.selected_checkpoint else "None"
    lines = [
        "DAM4SAM environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Python executable: {status.python_executable or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Tracker variant: {status.tracker_name}",
        f"- Config name: {status.config_name or 'None'}",
        f"- Config path: {status.config_path or 'None'}",
        f"- Config exists: {'yes' if status.config_exists else 'no'}",
        f"- Selected checkpoint: {selected}",
        f"- Selected checkpoint exists: {'yes' if status.checkpoint_exists else 'no'}",
        f"- Upstream checkpoint: {status.upstream_checkpoint_path or 'None'}",
        f"- Upstream checkpoint exists: {'yes' if status.upstream_checkpoint_exists else 'no'}",
        f"- Available checkpoints: {available}",
        f"- Import probe returncode: {status.import_probe_returncode}",
        f"- PyTorch import: {'ok' if status.torch_importable else 'failed'}",
        f"- PyTorch version: {status.torch_version or 'unknown'}",
        f"- PyTorch CUDA version: {status.torch_cuda_version or 'unknown'}",
        f"- CUDA available: {'yes' if status.cuda_available else 'no'}",
        f"- torchvision import: {'ok' if status.torchvision_importable else 'failed'}",
        f"- OpenCV import: {'ok' if status.cv2_importable else 'failed'}",
        f"- NumPy import: {'ok' if status.numpy_importable else 'failed'}",
        f"- NumPy version: {status.numpy_version or 'unknown'}",
        f"- PyYAML import: {'ok' if status.yaml_importable else 'failed'}",
        f"- hydra import: {'ok' if status.hydra_importable else 'failed'}",
        f"- omegaconf import: {'ok' if status.omegaconf_importable else 'failed'}",
        f"- Pillow import: {'ok' if status.pillow_importable else 'failed'}",
        f"- VOT import: {'ok' if status.vot_importable else 'failed'}",
        f"- SAM2 import: {'ok' if status.sam2_importable else 'failed'}",
        f"- SAM2 build_sam import: {'ok' if status.build_sam_importable else 'failed'}",
        f"- DAM4SAM tracker import: {'ok' if status.dam4sam_tracker_importable else 'failed'}",
        f"- VOT wrapper import: {'ok' if status.vot_wrapper_importable else 'failed'}",
        f"- SAM2 CUDA extension: {'ok' if status.cuda_extension_importable else 'missing'}",
    ]
    if status.sam2_module_path:
        lines.append(f"- SAM2 module: {status.sam2_module_path}")
    if status.dam4sam_tracker_module_path:
        lines.append(f"- DAM4SAM tracker module: {status.dam4sam_tracker_module_path}")
    if status.vot_wrapper_module_path:
        lines.append(f"- VOT wrapper module: {status.vot_wrapper_module_path}")
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


def run_dam4sam_smoke_test(
    *,
    repo_path: str | Path,
    python_executable: str | Path | None = None,
    tracker_name: str = DEFAULT_DAM4SAM_TRACKER_NAME,
    timeout_sec: float = DEFAULT_DAM4SAM_SMOKE_TIMEOUT_SEC,
) -> Dam4SamSmokeTestResult:
    resolved_repo_path = Path(repo_path).expanduser().resolve()
    resolved_python = _resolve_dam4sam_python_executable(python_executable)
    if resolved_python is None:
        raise FileNotFoundError("DAM4SAM Python executable could not be resolved.")

    command = [str(resolved_python), "-c", _DAM4SAM_SMOKE_TEST_CODE]
    env = _build_windows_conda_subprocess_env(resolved_python, resolved_repo_path)
    env["TRACKLEED_DAM4SAM_TRACKER_NAME"] = _normalize_tracker_name(tracker_name)
    completed = subprocess.run(
        command,
        cwd=str(resolved_repo_path),
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_sec),
        env=env,
    )
    return Dam4SamSmokeTestResult(
        command=command,
        tracker_name=_normalize_tracker_name(tracker_name),
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        payload=_parse_prefixed_json_stdout(completed.stdout, _SMOKE_TEST_PREFIX),
    )


def format_dam4sam_smoke_test_report(result: Dam4SamSmokeTestResult) -> str:
    payload = result.payload
    lines = [
        "DAM4SAM smoke test",
        f"- tracker variant: {result.tracker_name}",
        f"- returncode: {result.returncode}",
        f"- ok: {'yes' if result.succeeded else 'no'}",
    ]
    for key in (
        "checkpoint",
        "model_cfg",
        "device",
        "frame_shape",
        "init_bbox_xywh",
        "init_mask_shape",
        "tracked_mask_shape",
        "init_mask_area",
        "tracked_mask_area",
        "tracked_mask_bbox_xyxy",
        "elapsed_sec",
    ):
        if key in payload:
            lines.append(f"- {key.replace('_', ' ')}: {payload[key]}")
    if result.returncode != 0 or not result.succeeded:
        if result.stdout.strip():
            lines.append("- stdout:")
            lines.append(result.stdout.strip())
        if result.stderr.strip():
            lines.append("- stderr:")
            lines.append(result.stderr.strip())
    return "\n".join(lines)


def run_dam4sam_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray | None,
    settings: Dam4SamRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
    query_initial_masks: np.ndarray | None = None,
) -> Dam4SamOfflineResult:
    """Run DAM4SAM on a sequence using bbox or mask prompts."""

    if frames is None or np.asarray(frames).size == 0:
        raise ValueError("No frames provided for DAM4SAM inference.")
    frames_np = np.asarray(frames)
    if frames_np.ndim not in (3, 4):
        raise ValueError(f"Expected frames with ndim 3 or 4, got shape {frames_np.shape}.")

    backend_mode = str(settings.backend_mode or "subprocess").strip().lower()
    if backend_mode == "auto":
        backend_mode = "subprocess"
    if backend_mode == "direct":
        return _run_dam4sam_direct_offline(
            frames_np,
            query_points_tyx,
            settings,
            query_boxes_xyxy=query_boxes_xyxy,
            query_initial_masks=query_initial_masks,
            backend_mode="direct",
        )
    if backend_mode != "subprocess":
        raise ValueError(
            f"Unsupported DAM4SAM backend mode: {settings.backend_mode}. "
            "Expected `subprocess` or `direct`."
        )
    return _run_dam4sam_subprocess_offline(
        frames_np,
        query_points_tyx,
        settings,
        query_boxes_xyxy=query_boxes_xyxy,
        query_initial_masks=query_initial_masks,
    )


def run_dam4sam_backend_smoke_test(
    *,
    repo_path: str | Path,
    checkpoint_path: str | Path,
    python_executable: str | Path | None = None,
    tracker_name: str = DEFAULT_DAM4SAM_TRACKER_NAME,
    device: str = "auto",
    backend_mode: str = "subprocess",
    representative_point_mode: str = "mask_centroid",
    timeout_sec: float = DEFAULT_DAM4SAM_SUBPROCESS_TIMEOUT_SEC,
) -> Dam4SamBackendSmokeTestResult:
    frames, query_points_tyx, query_boxes_xyxy = _build_dam4sam_backend_smoke_inputs()
    result = run_dam4sam_offline(
        frames,
        query_points_tyx,
        Dam4SamRuntimeSettings(
            checkpoint_path=checkpoint_path,
            repo_path=repo_path,
            python_executable=python_executable,
            backend_mode=backend_mode,
            tracker_name=tracker_name,
            device=device,
            representative_point_mode=representative_point_mode,
            initial_bbox_padding_px=0.0,
            subprocess_timeout_sec=timeout_sec,
        ),
        query_boxes_xyxy=query_boxes_xyxy,
    )

    visible_indexes = np.flatnonzero(result.visible_mask[0])
    if visible_indexes.size == 0:
        first_index = last_index = 0
        first_point = last_point = (float("nan"), float("nan"))
        first_bbox = last_bbox = (float("nan"), float("nan"), float("nan"), float("nan"))
    else:
        first_index = int(visible_indexes[0])
        last_index = int(visible_indexes[-1])
        first_point = tuple(float(v) for v in result.tracks_xy[0, first_index].tolist())
        last_point = tuple(float(v) for v in result.tracks_xy[0, last_index].tolist())
        first_bbox = tuple(float(v) for v in result.mask_bboxes_xyxy[0, first_index].tolist())
        last_bbox = tuple(float(v) for v in result.mask_bboxes_xyxy[0, last_index].tolist())

    return Dam4SamBackendSmokeTestResult(
        checkpoint_name=result.checkpoint_name,
        tracker_name=result.tracker_name,
        backend_mode=result.backend_mode,
        representative_point_mode=result.representative_point_mode,
        masks_shape=tuple(int(v) for v in result.masks.shape),
        tracks_shape=tuple(int(v) for v in result.tracks_xy.shape),
        visible_shape=tuple(int(v) for v in result.visible_mask.shape),
        visible_count=int(np.count_nonzero(result.visible_mask)),
        first_track_point_xy=first_point,
        last_visible_point_xy=last_point,
        first_mask_area=float(result.mask_areas[0, first_index]),
        last_mask_area=float(result.mask_areas[0, last_index]),
        first_bbox_xyxy=first_bbox,
        last_bbox_xyxy=last_bbox,
    )


def format_dam4sam_backend_smoke_test_report(result: Dam4SamBackendSmokeTestResult) -> str:
    return "\n".join(
        [
            "DAM4SAM backend smoke test",
            f"- checkpoint: {result.checkpoint_name}",
            f"- backend mode: {result.backend_mode}",
            f"- tracker variant: {result.tracker_name}",
            f"- representative point mode: {result.representative_point_mode}",
            f"- masks shape: {result.masks_shape}",
            f"- tracks shape: {result.tracks_shape}",
            f"- visible mask shape: {result.visible_shape}",
            f"- visible count: {result.visible_count}",
            f"- first track point: {list(result.first_track_point_xy)}",
            f"- last visible point: {list(result.last_visible_point_xy)}",
            f"- first mask area: {result.first_mask_area:g}",
            f"- last mask area: {result.last_mask_area:g}",
            f"- first bbox xyxy: {list(result.first_bbox_xyxy)}",
            f"- last bbox xyxy: {list(result.last_bbox_xyxy)}",
        ]
    )


def _run_dam4sam_direct_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray | None,
    settings: Dam4SamRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
    query_initial_masks: np.ndarray | None = None,
    backend_mode: str = "direct",
) -> Dam4SamOfflineResult:
    resolved_repo_path = _resolve_dam4sam_repo_path(settings.repo_path)
    if resolved_repo_path is None:
        raise FileNotFoundError(
            "Could not resolve the DAM4SAM repo. Pass repo_path pointing at the local clone."
        )
    checkpoint_file = _resolve_existing_dam4sam_checkpoint_file(settings.checkpoint_path)
    tracker_name = _normalize_tracker_name(settings.tracker_name)
    config_name = _config_name_for_tracker(tracker_name)
    if config_name is None:
        raise ValueError(f"Unsupported DAM4SAM tracker variant: {tracker_name}")
    representative_point_mode = _normalize_representative_point_mode(
        settings.representative_point_mode
    )

    frames_rgb, original_resolution = _prepare_frames_rgb(frames)
    inference_resolution = _normalize_optional_resolution(
        settings.inference_resolution,
        original_resolution,
    )
    resized_frames = _resize_video_numpy(frames_rgb, inference_resolution)
    frames_gray_original = _video_to_gray_float(frames_rgb)

    num_queries = _infer_num_queries(
        query_points_tyx=query_points_tyx,
        query_boxes_xyxy=query_boxes_xyxy,
        query_initial_masks=query_initial_masks,
    )
    query_points = _normalize_query_points_tyx(query_points_tyx, num_queries)
    query_boxes = _normalize_query_boxes_xyxy(query_boxes_xyxy, num_queries)
    query_masks = _normalize_query_initial_masks(
        query_initial_masks,
        num_queries=num_queries,
        frame_shape=original_resolution,
    )
    scaled_query_points = (
        _scale_query_points_tyx(
            query_points,
            source_resolution=original_resolution,
            target_resolution=inference_resolution,
        )
        if query_points is not None
        else None
    )
    scaled_query_boxes = (
        _scale_boxes_xyxy(
            query_boxes,
            source_resolution=original_resolution,
            target_resolution=inference_resolution,
        )
        if query_boxes is not None
        else None
    )
    scaled_query_masks = (
        np.stack(
            [
                _resize_mask_bool(mask, inference_resolution)
                for mask in np.asarray(query_masks, dtype=bool)
            ],
            axis=0,
        )
        if query_masks is not None
        else None
    )
    seed_frames = _extract_query_seed_frames(
        query_points,
        total_frames=int(frames_rgb.shape[0]),
        num_queries=num_queries,
    )
    seed_boxes_xywh = _build_seed_boxes_xywh(
        scaled_query_points,
        scaled_query_boxes,
        resolution=inference_resolution,
        initial_bbox_padding_px=float(settings.initial_bbox_padding_px),
        num_queries=num_queries,
    )

    num_frames = int(frames_rgb.shape[0])
    masks = np.zeros((num_queries, num_frames, original_resolution[0], original_resolution[1]), dtype=bool)
    tracks_xy = np.full((num_queries, num_frames, 2), np.nan, dtype=np.float32)
    visible_mask = np.zeros((num_queries, num_frames), dtype=bool)
    mask_areas = np.zeros((num_queries, num_frames), dtype=np.float32)
    mask_bboxes_xyxy = np.full((num_queries, num_frames, 4), np.nan, dtype=np.float32)
    mask_bboxes_xywh = np.full((num_queries, num_frames, 4), np.nan, dtype=np.float32)
    mask_scores = np.full((num_queries, num_frames), np.nan, dtype=np.float32)

    with _dam4sam_import_context(resolved_repo_path):
        torch_module = importlib.import_module("torch")
        _validate_dam4sam_device(torch_module, settings.device)
        dam4sam_module = importlib.import_module("dam4sam_tracker")
        pil_image_module = importlib.import_module("PIL.Image")

        for query_index in range(num_queries):
            tracker = _build_dam4sam_tracker(
                dam4sam_module,
                tracker_name=tracker_name,
                checkpoint_path=checkpoint_file,
                config_name=config_name,
            )
            seed_frame = int(seed_frames[query_index])
            seed_bbox_xywh = tuple(float(v) for v in seed_boxes_xywh[query_index].tolist())
            initial_mask = (
                np.asarray(scaled_query_masks[query_index], dtype=np.uint8)
                if scaled_query_masks is not None
                else None
            )
            image = pil_image_module.fromarray(resized_frames[seed_frame])
            output = tracker.initialize(image, initial_mask, bbox=seed_bbox_xywh)
            _record_dam4sam_output(
                output,
                query_index=query_index,
                frame_index=seed_frame,
                masks=masks,
                tracks_xy=tracks_xy,
                visible_mask=visible_mask,
                mask_areas=mask_areas,
                mask_bboxes_xyxy=mask_bboxes_xyxy,
                mask_bboxes_xywh=mask_bboxes_xywh,
                mask_scores=mask_scores,
                frame_gray=frames_gray_original[seed_frame],
                representative_point_mode=representative_point_mode,
                original_resolution=original_resolution,
            )

            for frame_index in range(seed_frame + 1, num_frames):
                image = pil_image_module.fromarray(resized_frames[frame_index])
                output = tracker.track(image)
                _record_dam4sam_output(
                    output,
                    query_index=query_index,
                    frame_index=frame_index,
                    masks=masks,
                    tracks_xy=tracks_xy,
                    visible_mask=visible_mask,
                    mask_areas=mask_areas,
                    mask_bboxes_xyxy=mask_bboxes_xyxy,
                    mask_bboxes_xywh=mask_bboxes_xywh,
                    mask_scores=mask_scores,
                    frame_gray=frames_gray_original[frame_index],
                    representative_point_mode=representative_point_mode,
                    original_resolution=original_resolution,
                )

    return Dam4SamOfflineResult(
        masks=masks,
        tracks_xy=tracks_xy,
        visible_mask=visible_mask,
        mask_areas=mask_areas,
        mask_bboxes_xyxy=mask_bboxes_xyxy,
        mask_bboxes_xywh=mask_bboxes_xywh,
        mask_scores=mask_scores,
        inference_resolution=tuple(int(v) for v in inference_resolution),
        original_resolution=tuple(int(v) for v in original_resolution),
        checkpoint_name=checkpoint_file.name,
        tracker_name=tracker_name,
        config_name=config_name,
        representative_point_mode=representative_point_mode,
        backend_mode=str(backend_mode),
    )


def _run_dam4sam_subprocess_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray | None,
    settings: Dam4SamRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
    query_initial_masks: np.ndarray | None = None,
) -> Dam4SamOfflineResult:
    resolved_repo_path = _resolve_dam4sam_repo_path(settings.repo_path)
    if resolved_repo_path is None:
        raise FileNotFoundError(
            "Could not resolve the DAM4SAM repo. Pass repo_path pointing at the local clone."
        )
    resolved_python = _resolve_dam4sam_python_executable(settings.python_executable)
    if resolved_python is None or not resolved_python.exists():
        raise FileNotFoundError(f"DAM4SAM Python executable not found: {resolved_python}")
    checkpoint_file = _resolve_existing_dam4sam_checkpoint_file(settings.checkpoint_path)
    representative_point_mode = _normalize_representative_point_mode(
        settings.representative_point_mode
    )
    inference_resolution = settings.inference_resolution

    script_path = Path(__file__).resolve().parents[2] / DEFAULT_DAM4SAM_SUBPROCESS_SCRIPT
    if not script_path.exists():
        raise FileNotFoundError(f"DAM4SAM subprocess script not found: {script_path}")

    with TemporaryDirectory(prefix="trackleed_dam4sam_") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "dam4sam_input.npz"
        output_path = temp_root / "dam4sam_output.npz"
        np.savez_compressed(
            input_path,
            frames=np.asarray(frames),
            has_query_points=np.asarray(query_points_tyx is not None),
            query_points_tyx=(
                np.asarray(query_points_tyx, dtype=np.float32)
                if query_points_tyx is not None
                else np.empty((0, 3), dtype=np.float32)
            ),
            has_query_boxes=np.asarray(query_boxes_xyxy is not None),
            query_boxes_xyxy=(
                np.asarray(query_boxes_xyxy, dtype=np.float32)
                if query_boxes_xyxy is not None
                else np.empty((0, 4), dtype=np.float32)
            ),
            has_query_initial_masks=np.asarray(query_initial_masks is not None),
            query_initial_masks=(
                np.asarray(query_initial_masks, dtype=bool)
                if query_initial_masks is not None
                else np.empty((0, 1, 1), dtype=bool)
            ),
        )

        command = [
            str(resolved_python),
            str(script_path),
            "--input-npz",
            str(input_path),
            "--output-npz",
            str(output_path),
            "--checkpoint",
            str(checkpoint_file),
            "--repo-path",
            str(resolved_repo_path),
            "--tracker-name",
            _normalize_tracker_name(settings.tracker_name),
            "--device",
            str(settings.device or "auto"),
            "--representative-point-mode",
            representative_point_mode,
            "--initial-bbox-padding-px",
            str(float(settings.initial_bbox_padding_px)),
            "--subprocess-timeout-sec",
            str(float(settings.subprocess_timeout_sec)),
        ]
        if inference_resolution is not None:
            height, width = _normalize_optional_resolution(inference_resolution, (1, 1))
            command.extend(["--height", str(int(height)), "--width", str(int(width))])

        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            check=False,
            timeout=float(settings.subprocess_timeout_sec),
            env=_build_windows_conda_subprocess_env(resolved_python, resolved_repo_path),
        )
        if completed.returncode != 0 or not output_path.exists():
            details = {
                "command": command,
                "returncode": int(completed.returncode),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
            raise RuntimeError(
                "DAM4SAM subprocess inference failed:\n"
                + json.dumps(details, indent=2, ensure_ascii=False)
            )

        with np.load(output_path, allow_pickle=False) as data:
            return Dam4SamOfflineResult(
                masks=np.asarray(data["masks"], dtype=bool),
                tracks_xy=np.asarray(data["tracks_xy"], dtype=np.float32),
                visible_mask=np.asarray(data["visible_mask"], dtype=bool),
                mask_areas=np.asarray(data["mask_areas"], dtype=np.float32),
                mask_bboxes_xyxy=np.asarray(data["mask_bboxes_xyxy"], dtype=np.float32),
                mask_bboxes_xywh=np.asarray(data["mask_bboxes_xywh"], dtype=np.float32),
                mask_scores=np.asarray(data["mask_scores"], dtype=np.float32),
                inference_resolution=tuple(int(v) for v in np.asarray(data["inference_resolution"]).tolist()),
                original_resolution=tuple(int(v) for v in np.asarray(data["original_resolution"]).tolist()),
                checkpoint_name=str(np.asarray(data["checkpoint_name"]).tolist()),
                tracker_name=str(np.asarray(data["tracker_name"]).tolist()),
                config_name=str(np.asarray(data["config_name"]).tolist()),
                representative_point_mode=str(np.asarray(data["representative_point_mode"]).tolist()),
                backend_mode=str(np.asarray(data["backend_mode"]).tolist()),
            )


def _resolve_dam4sam_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidates = [Path(repo_path)] if repo_path else list(DEFAULT_DAM4SAM_REPO_CANDIDATES)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "dam4sam_tracker.py").exists() and (
            resolved / "sam2" / "build_sam.py"
        ).exists():
            return resolved
    return None


def _resolve_dam4sam_python_executable(python_executable: str | Path | None) -> Optional[Path]:
    if python_executable:
        return Path(python_executable).expanduser().resolve()
    return Path(sys.executable).resolve()


def _discover_checkpoints_in_directory(directory: str | Path | None) -> list[Path]:
    if directory is None:
        return []
    root = Path(directory).expanduser()
    if not root.exists():
        return []
    checkpoints = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.lower().endswith((".pt", ".pth", ".pth.tar", ".tar")):
            checkpoints.append(path.resolve())
    return checkpoints


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoints: list[Path],
    models_dir: Path,
    tracker_name: str,
) -> Optional[Path]:
    if checkpoint_path:
        candidate = Path(checkpoint_path).expanduser()
        if not candidate.is_absolute():
            candidate = models_dir / candidate
        return candidate.resolve()
    return select_default_dam4sam_checkpoint(checkpoints, tracker_name, models_dir)


def _checkpoint_name_for_tracker(tracker_name: str) -> Optional[str]:
    variant = DAM4SAM_TRACKER_VARIANTS.get(_normalize_tracker_name(tracker_name))
    return variant[0] if variant is not None else None


def _config_name_for_tracker(tracker_name: str) -> Optional[str]:
    variant = DAM4SAM_TRACKER_VARIANTS.get(_normalize_tracker_name(tracker_name))
    return variant[1] if variant is not None else None


def _normalize_tracker_name(tracker_name: str | None) -> str:
    value = str(tracker_name or DEFAULT_DAM4SAM_TRACKER_NAME).strip()
    return value or DEFAULT_DAM4SAM_TRACKER_NAME


def _run_dam4sam_import_probe(
    python_executable: Path,
    repo_path: Optional[Path],
    *,
    timeout_sec: float,
) -> Dam4SamImportProbeResult:
    command = [str(python_executable), "-c", _DAM4SAM_IMPORT_PROBE_CODE]
    completed = subprocess.run(
        command,
        cwd=str(repo_path) if repo_path is not None else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_sec),
        env=_build_windows_conda_subprocess_env(python_executable, repo_path),
    )
    return Dam4SamImportProbeResult(
        command=command,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        payload=_parse_prefixed_json_stdout(completed.stdout, _IMPORT_PROBE_PREFIX),
    )


def _apply_import_probe_result(
    status: Dam4SamEnvironmentStatus,
    probe: Dam4SamImportProbeResult,
) -> None:
    status.import_probe_returncode = probe.returncode
    status.import_probe_stdout = probe.stdout
    status.import_probe_stderr = probe.stderr
    payload = probe.payload
    if probe.returncode != 0:
        status.errors.append(
            "DAM4SAM import probe failed with return code "
            f"{probe.returncode}: {probe.stderr.strip() or probe.stdout.strip()}"
        )
    if not payload:
        status.errors.append("DAM4SAM import probe did not return parseable diagnostics JSON.")
        return

    status.torch_importable = bool(payload.get("torch_importable"))
    status.torch_version = payload.get("torch_version")
    status.torch_cuda_version = payload.get("torch_cuda_version")
    status.cuda_available = bool(payload.get("cuda_available"))
    status.torchvision_importable = bool(payload.get("torchvision_importable"))
    status.cv2_importable = bool(payload.get("cv2_importable"))
    status.numpy_importable = bool(payload.get("numpy_importable"))
    status.numpy_version = payload.get("numpy_version")
    status.yaml_importable = bool(payload.get("yaml_importable"))
    status.hydra_importable = bool(payload.get("hydra_importable"))
    status.omegaconf_importable = bool(payload.get("omegaconf_importable"))
    status.pillow_importable = bool(payload.get("pillow_importable"))
    status.vot_importable = bool(payload.get("vot_importable"))
    status.sam2_importable = bool(payload.get("sam2_importable"))
    status.sam2_module_path = payload.get("sam2_module_path")
    status.build_sam_importable = bool(payload.get("sam2_build_sam_importable"))
    status.dam4sam_tracker_importable = bool(payload.get("dam4sam_tracker_importable"))
    status.dam4sam_tracker_module_path = payload.get("dam4sam_tracker_module_path")
    status.vot_wrapper_importable = bool(payload.get("vot_wrapper_dam4sam_importable"))
    status.vot_wrapper_module_path = payload.get("vot_wrapper_dam4sam_module_path")
    status.cuda_extension_importable = bool(payload.get("sam2_cuda_extension_importable"))

    for key, value in sorted(payload.items()):
        if key.endswith("_error") and value:
            if key == "sam2_cuda_extension_error":
                continue
            status.errors.append(f"{key[:-6]} import failed: {value}")


def _parse_prefixed_json_stdout(stdout: str, prefix: str) -> dict[str, Any]:
    for line in reversed(str(stdout or "").splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    return {}


def _format_available_checkpoints(checkpoints: list[Path]) -> str:
    if not checkpoints:
        return "none"
    by_name: dict[str, list[Path]] = {}
    for path in checkpoints:
        by_name.setdefault(path.name, []).append(path)

    parts: list[str] = []
    for name in sorted(by_name, key=str.lower):
        paths = by_name[name]
        if len(paths) == 1:
            parts.append(name)
        else:
            parts.append(f"{name} ({len(paths)} paths)")
    return ", ".join(parts)


def _build_windows_conda_subprocess_env(
    python_executable: Path,
    repo_path: Optional[Path] = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if repo_path is not None:
        env["TRACKLEED_DAM4SAM_REPO"] = str(repo_path)
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(repo_path)] + ([current_pythonpath] if current_pythonpath else [])
        )
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


def _numpy_major_version(version: str | None) -> Optional[int]:
    if not version:
        return None
    first_chunk = str(version).split(".", 1)[0]
    try:
        return int(first_chunk)
    except ValueError:
        return None


@contextmanager
def _dam4sam_import_context(repo_path: Path) -> Iterator[None]:
    repo_str = str(repo_path)
    old_path = list(sys.path)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)
    try:
        yield
    finally:
        sys.path[:] = old_path


def _resolve_existing_dam4sam_checkpoint_file(checkpoint_path: str | Path) -> Path:
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"DAM4SAM checkpoint not found: {checkpoint_file}")
    return checkpoint_file


def _validate_dam4sam_device(torch_module: Any, device: str | None) -> None:
    normalized = str(device or "auto").strip().lower()
    if normalized in {"auto", ""}:
        normalized = "cuda"
    if normalized not in {"cuda", "cuda:0"}:
        raise ValueError(
            "DAM4SAM upstream tracker currently hardcodes CUDA device 0. "
            "Use --device auto, cuda, or cuda:0."
        )
    if not bool(torch_module.cuda.is_available()):
        raise RuntimeError("DAM4SAM requires CUDA, but torch.cuda.is_available() is false.")


def _build_dam4sam_tracker(
    dam4sam_module: Any,
    *,
    tracker_name: str,
    checkpoint_path: Path,
    config_name: str,
) -> Any:
    original_determine_tracker = getattr(dam4sam_module, "determine_tracker", None)

    def determine_tracker_override(_: str) -> tuple[str, str]:
        return str(checkpoint_path), str(config_name)

    dam4sam_module.determine_tracker = determine_tracker_override
    try:
        return dam4sam_module.DAM4SAMTracker(tracker_name=tracker_name)
    finally:
        if original_determine_tracker is not None:
            dam4sam_module.determine_tracker = original_determine_tracker


def _prepare_frames_rgb(frames: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    frames_np = np.asarray(frames)
    if frames_np.ndim == 3:
        frames_np = np.stack([frames_np] * 3, axis=-1)
    elif frames_np.ndim == 4 and frames_np.shape[-1] == 1:
        frames_np = np.repeat(frames_np, 3, axis=-1)
    elif frames_np.ndim == 4 and frames_np.shape[-1] >= 3:
        frames_np = frames_np[..., :3]
    else:
        raise ValueError(f"Unsupported DAM4SAM frame shape: {frames_np.shape}")

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
        raise ValueError(f"Invalid DAM4SAM resolution: {resolution}")
    height, width = int(resolution[0]), int(resolution[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid DAM4SAM resolution: {(height, width)}")
    return (height, width)


def _resize_video_numpy(frames: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target[0]), int(target[1])
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


def _resize_mask_bool(mask: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target[0]), int(target[1])
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape == (target_h, target_w):
        return mask_bool
    pil_image_module = importlib.import_module("PIL.Image")
    resized = pil_image_module.fromarray(mask_bool.astype(np.uint8) * 255).resize(
        (target_w, target_h),
        resample=pil_image_module.Resampling.NEAREST,
    )
    return np.asarray(resized, dtype=np.uint8) > 0


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


def _infer_num_queries(
    *,
    query_points_tyx: np.ndarray | None,
    query_boxes_xyxy: np.ndarray | None,
    query_initial_masks: np.ndarray | None,
) -> int:
    candidates: list[int] = []
    if query_points_tyx is not None:
        candidates.append(int(np.asarray(query_points_tyx).shape[0]))
    if query_boxes_xyxy is not None:
        candidates.append(int(np.asarray(query_boxes_xyxy).shape[0]))
    if query_initial_masks is not None:
        candidates.append(int(np.asarray(query_initial_masks).shape[0]))
    if not candidates:
        raise ValueError("DAM4SAM requires at least one query point, bbox, or initial mask.")
    if len(set(candidates)) != 1:
        raise ValueError(f"DAM4SAM query inputs disagree on object count: {candidates}.")
    num_queries = candidates[0]
    if num_queries <= 0:
        raise ValueError("DAM4SAM requires at least one query.")
    return num_queries


def _normalize_query_points_tyx(
    query_points_tyx: np.ndarray | None,
    num_queries: int,
) -> np.ndarray | None:
    if query_points_tyx is None:
        return None
    points = np.asarray(query_points_tyx, dtype=np.float32)
    if points.shape != (num_queries, 3):
        raise ValueError(
            f"query_points_tyx must have shape {(num_queries, 3)}, got {points.shape}."
        )
    return points


def _normalize_query_boxes_xyxy(
    query_boxes_xyxy: np.ndarray | None,
    num_queries: int,
) -> np.ndarray | None:
    if query_boxes_xyxy is None:
        return None
    boxes = np.asarray(query_boxes_xyxy, dtype=np.float32)
    if boxes.shape != (num_queries, 4):
        raise ValueError(
            f"query_boxes_xyxy must have shape {(num_queries, 4)}, got {boxes.shape}."
        )
    return boxes


def _normalize_query_initial_masks(
    query_initial_masks: np.ndarray | None,
    *,
    num_queries: int,
    frame_shape: tuple[int, int],
) -> np.ndarray | None:
    if query_initial_masks is None:
        return None
    masks = np.asarray(query_initial_masks, dtype=bool)
    expected_shape = (num_queries, int(frame_shape[0]), int(frame_shape[1]))
    if masks.shape != expected_shape:
        raise ValueError(f"query_initial_masks must have shape {expected_shape}, got {masks.shape}.")
    return masks


def _extract_query_seed_frames(
    query_points_tyx: np.ndarray | None,
    *,
    total_frames: int,
    num_queries: int,
) -> np.ndarray:
    if query_points_tyx is None:
        return np.zeros((num_queries,), dtype=np.int32)
    seed_frames = np.rint(np.asarray(query_points_tyx, dtype=np.float32)[:, 0]).astype(np.int32)
    if np.any(seed_frames < 0) or np.any(seed_frames >= int(total_frames)):
        raise ValueError(
            f"DAM4SAM seed frame index out of range 0..{int(total_frames) - 1}: "
            f"{seed_frames.tolist()}"
        )
    return seed_frames


def _scale_query_points_tyx(
    query_points_tyx: np.ndarray,
    source_resolution: tuple[int, int],
    target_resolution: tuple[int, int],
) -> np.ndarray:
    src_h, src_w = source_resolution
    dst_h, dst_w = target_resolution
    scaled = np.asarray(query_points_tyx, dtype=np.float32).copy()
    scaled[:, 1] *= float(dst_h) / float(src_h)
    scaled[:, 2] *= float(dst_w) / float(src_w)
    return scaled


def _scale_boxes_xyxy(
    boxes_xyxy: np.ndarray,
    source_resolution: tuple[int, int],
    target_resolution: tuple[int, int],
) -> np.ndarray:
    src_h, src_w = source_resolution
    dst_h, dst_w = target_resolution
    scaled = np.asarray(boxes_xyxy, dtype=np.float32).copy()
    scaled[:, [0, 2]] *= float(dst_w) / float(src_w)
    scaled[:, [1, 3]] *= float(dst_h) / float(src_h)
    return scaled


def _build_seed_boxes_xywh(
    query_points_tyx: np.ndarray | None,
    query_boxes_xyxy: np.ndarray | None,
    *,
    resolution: tuple[int, int],
    initial_bbox_padding_px: float,
    num_queries: int,
) -> np.ndarray:
    height, width = int(resolution[0]), int(resolution[1])
    seed_boxes = np.zeros((num_queries, 4), dtype=np.float32)
    if query_boxes_xyxy is not None:
        boxes = np.asarray(query_boxes_xyxy, dtype=np.float32)
        x0 = np.minimum(boxes[:, 0], boxes[:, 2])
        y0 = np.minimum(boxes[:, 1], boxes[:, 3])
        x1 = np.maximum(boxes[:, 0], boxes[:, 2])
        y1 = np.maximum(boxes[:, 1], boxes[:, 3])
        seed_boxes[:, 0] = x0
        seed_boxes[:, 1] = y0
        seed_boxes[:, 2] = np.maximum(1.0, x1 - x0)
        seed_boxes[:, 3] = np.maximum(1.0, y1 - y0)
    elif query_points_tyx is not None:
        points = np.asarray(query_points_tyx, dtype=np.float32)
        half = max(1.0, float(initial_bbox_padding_px))
        seed_boxes[:, 0] = points[:, 2] - half
        seed_boxes[:, 1] = points[:, 1] - half
        seed_boxes[:, 2] = half * 2.0
        seed_boxes[:, 3] = half * 2.0
    else:
        seed_boxes[:, :] = np.asarray([0.0, 0.0, float(width), float(height)], dtype=np.float32)

    for index in range(num_queries):
        seed_boxes[index] = np.asarray(
            _clip_bbox_xywh(seed_boxes[index], image_shape=(height, width)),
            dtype=np.float32,
        )
    return seed_boxes


def _clip_bbox_xywh(
    bbox_xywh: np.ndarray,
    *,
    image_shape: tuple[int, int],
) -> tuple[float, float, float, float]:
    height, width = int(image_shape[0]), int(image_shape[1])
    x, y, w, h = [float(v) for v in np.asarray(bbox_xywh, dtype=np.float32).tolist()]
    x0 = min(max(0.0, x), max(0.0, width - 1.0))
    y0 = min(max(0.0, y), max(0.0, height - 1.0))
    x1 = min(max(x0 + 1.0, x + max(1.0, w)), float(width))
    y1 = min(max(y0 + 1.0, y + max(1.0, h)), float(height))
    return (x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0))


def _record_dam4sam_output(
    output: dict[str, Any],
    *,
    query_index: int,
    frame_index: int,
    masks: np.ndarray,
    tracks_xy: np.ndarray,
    visible_mask: np.ndarray,
    mask_areas: np.ndarray,
    mask_bboxes_xyxy: np.ndarray,
    mask_bboxes_xywh: np.ndarray,
    mask_scores: np.ndarray,
    frame_gray: np.ndarray,
    representative_point_mode: str,
    original_resolution: tuple[int, int],
) -> None:
    if "pred_mask" not in output:
        raise RuntimeError("DAM4SAM output did not contain `pred_mask`.")
    mask_bool = np.asarray(output["pred_mask"]) > 0
    mask_bool = _resize_mask_bool(mask_bool, original_resolution)
    masks[query_index, frame_index] = mask_bool
    area = float(np.count_nonzero(mask_bool))
    mask_areas[query_index, frame_index] = area
    if area <= 0:
        return

    bbox_xyxy = _mask_to_bbox_xyxy(mask_bool)
    if bbox_xyxy is None:
        return
    bbox_xywh = _bbox_xyxy_to_xywh(bbox_xyxy)
    mask_bboxes_xyxy[query_index, frame_index, :] = np.asarray(bbox_xyxy, dtype=np.float32)
    mask_bboxes_xywh[query_index, frame_index, :] = np.asarray(bbox_xywh, dtype=np.float32)

    score = output.get("score")
    if score is not None:
        try:
            mask_scores[query_index, frame_index] = float(score)
        except (TypeError, ValueError):
            pass

    point = _mask_to_representative_point(
        mask_bool,
        frame_gray=frame_gray,
        mode=representative_point_mode,
    )
    if point is None:
        return
    x, y = point
    if not np.isfinite(x) or not np.isfinite(y):
        return
    tracks_xy[query_index, frame_index, :] = np.asarray([x, y], dtype=np.float32)
    visible_mask[query_index, frame_index] = True


def _mask_to_bbox_xyxy(mask_bool: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return None
    x0 = float(np.min(xs))
    y0 = float(np.min(ys))
    x1 = float(np.max(xs) + 1)
    y1 = float(np.max(ys) + 1)
    return (x0, y0, x1, y1)


def _bbox_xyxy_to_xywh(
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox_xyxy
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


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
    raise ValueError(f"Unsupported DAM4SAM representative point mode: {mode}")


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


def _normalize_representative_point_mode(mode: str | None) -> str:
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
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported DAM4SAM representative point mode: {mode}. "
            "Expected mask_centroid, peak_in_mask, or bbox_center."
        )
    return aliases[normalized]


def _build_dam4sam_backend_smoke_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = []
    for dx in (0, 3, 6):
        frame = np.zeros((112, 144, 3), dtype=np.uint8)
        frame[..., 0] = 12
        frame[..., 1] = 14
        frame[..., 2] = 18
        yy, xx = np.ogrid[:112, :144]
        target = ((xx - (64 + dx)) ** 2) / (12.0 ** 2) + ((yy - 54) ** 2) / (12.0 ** 2) <= 1.0
        distractor = ((xx - 105) ** 2) / (11.0 ** 2) + ((yy - 54) ** 2) / (10.0 ** 2) <= 1.0
        frame[target] = np.asarray([240, 205, 80], dtype=np.uint8)
        frame[distractor] = np.asarray([80, 130, 210], dtype=np.uint8)
        frames.append(frame)
    frames_np = np.stack(frames, axis=0)
    query_points_tyx = np.asarray([[0.0, 54.0, 64.0]], dtype=np.float32)
    query_boxes_xyxy = np.asarray([[52.0, 42.0, 76.0, 66.0]], dtype=np.float32)
    return frames_np, query_points_tyx, query_boxes_xyxy



_DAM4SAM_IMPORT_PROBE_CODE = r"""
import importlib
import json
import os
import sys

PREFIX = "__TRACKLEED_DAM4SAM_IMPORT_PROBE__"
result = {"python_executable": sys.executable}
repo_path = os.environ.get("TRACKLEED_DAM4SAM_REPO")
if repo_path and repo_path not in sys.path:
    sys.path.insert(0, repo_path)


def probe(name, label):
    try:
        module = importlib.import_module(name)
        result[label + "_importable"] = True
        result[label + "_module_path"] = getattr(module, "__file__", None)
        return module
    except Exception as exc:
        result[label + "_importable"] = False
        result[label + "_error"] = type(exc).__name__ + ": " + str(exc)
        return None


torch = probe("torch", "torch")
if torch is not None:
    result["torch_version"] = getattr(torch, "__version__", None)
    result["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        result["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        result["cuda_available"] = False
        result["cuda_error"] = type(exc).__name__ + ": " + str(exc)
else:
    result["cuda_available"] = False

numpy = probe("numpy", "numpy")
if numpy is not None:
    result["numpy_version"] = getattr(numpy, "__version__", None)

probe("torchvision", "torchvision")
probe("cv2", "cv2")
probe("yaml", "yaml")
probe("hydra", "hydra")
probe("omegaconf", "omegaconf")
probe("PIL.Image", "pillow")
probe("vot.region.raster", "vot")
probe("sam2", "sam2")
probe("sam2.build_sam", "sam2_build_sam")
probe("dam4sam_tracker", "dam4sam_tracker")
probe("vot_wrapper_dam4sam", "vot_wrapper_dam4sam")
probe("sam2._C", "sam2_cuda_extension")

print(PREFIX + json.dumps(result, sort_keys=True))
"""


_DAM4SAM_SMOKE_TEST_CODE = r"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

PREFIX = "__TRACKLEED_DAM4SAM_SMOKE_TEST__"
repo_path = os.environ.get("TRACKLEED_DAM4SAM_REPO")
if repo_path and repo_path not in sys.path:
    sys.path.insert(0, repo_path)

tracker_name = os.environ.get("TRACKLEED_DAM4SAM_TRACKER_NAME") or "sam21pp-L"


def make_frame(dx):
    image = Image.new("RGB", (144, 112), (12, 14, 18))
    draw = ImageDraw.Draw(image)
    draw.ellipse((52 + dx, 42, 76 + dx, 66), fill=(240, 205, 80))
    draw.ellipse((94, 44, 116, 64), fill=(80, 130, 210))
    return image


def mask_bbox(mask):
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


start = time.perf_counter()
from dam4sam_tracker import DAM4SAMTracker
import torch

tracker = DAM4SAMTracker(tracker_name=tracker_name)
frame0 = make_frame(0)
frame1 = make_frame(3)
init_bbox = [52, 42, 24, 24]
out0 = tracker.initialize(frame0, None, bbox=init_bbox)
out1 = tracker.track(frame1)
init_mask = np.asarray(out0["pred_mask"])
tracked_mask = np.asarray(out1["pred_mask"])
payload = {
    "ok": bool(init_mask.size and tracked_mask.size and np.count_nonzero(tracked_mask) > 0),
    "tracker_name": tracker_name,
    "checkpoint": str(getattr(tracker, "checkpoint", "")),
    "model_cfg": str(getattr(tracker, "model_cfg", "")),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "frame_shape": [112, 144, 3],
    "init_bbox_xywh": init_bbox,
    "init_mask_shape": list(init_mask.shape),
    "tracked_mask_shape": list(tracked_mask.shape),
    "init_mask_area": int(np.count_nonzero(init_mask)),
    "tracked_mask_area": int(np.count_nonzero(tracked_mask)),
    "tracked_mask_bbox_xyxy": mask_bbox(tracked_mask),
    "elapsed_sec": round(time.perf_counter() - start, 3),
}
print(PREFIX + json.dumps(payload, sort_keys=True))
"""
