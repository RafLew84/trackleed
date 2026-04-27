"""SAMURAI environment diagnostics and smoke-test helpers."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
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

from trackleed.analysis.dam4sam_backend import (
    _build_seed_boxes_xywh,
    _infer_num_queries,
    _normalize_optional_resolution,
    _normalize_query_boxes_xyxy,
    _normalize_query_points_tyx,
    _prepare_frames_rgb,
    _resize_mask_bool,
    _resize_video_numpy,
    _scale_boxes_xyxy,
    _scale_query_points_tyx,
    _video_to_gray_float,
)

DEFAULT_SAMURAI_MODELS_DIR = Path("tracking_models") / "samurai"
DEFAULT_SAMURAI_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "samurai",
    Path(__file__).resolve().parents[3] / "SAMURAI",
    Path(__file__).resolve().parents[2] / "samurai",
)
DEFAULT_SAMURAI_MODEL_VARIANT = "sam2.1_hiera_base_plus"
DEFAULT_SAMURAI_IMPORT_PROBE_TIMEOUT_SEC = 60.0
DEFAULT_SAMURAI_SMOKE_TIMEOUT_SEC = 600.0
DEFAULT_SAMURAI_SUBPROCESS_TIMEOUT_SEC = 1800.0
DEFAULT_SAMURAI_SUBPROCESS_SCRIPT = Path("scripts") / "run_samurai_subprocess.py"
_IMPORT_PROBE_PREFIX = "__TRACKLEED_SAMURAI_IMPORT_PROBE__"
_SMOKE_TEST_PREFIX = "__TRACKLEED_SAMURAI_SMOKE_TEST__"

SAMURAI_MODEL_VARIANTS = {
    "sam2.1_hiera_large": (
        "sam2.1_hiera_large.pt",
        "sam2.1_hiera_l.yaml",
        "configs/samurai/sam2.1_hiera_l.yaml",
    ),
    "sam2.1_hiera_base_plus": (
        "sam2.1_hiera_base_plus.pt",
        "sam2.1_hiera_b+.yaml",
        "configs/samurai/sam2.1_hiera_b+.yaml",
    ),
    "sam2.1_hiera_small": (
        "sam2.1_hiera_small.pt",
        "sam2.1_hiera_s.yaml",
        "configs/samurai/sam2.1_hiera_s.yaml",
    ),
    "sam2.1_hiera_tiny": (
        "sam2.1_hiera_tiny.pt",
        "sam2.1_hiera_t.yaml",
        "configs/samurai/sam2.1_hiera_t.yaml",
    ),
}

_SAMURAI_VARIANT_ALIASES = {
    "large": "sam2.1_hiera_large",
    "l": "sam2.1_hiera_large",
    "sam21_large": "sam2.1_hiera_large",
    "sam21_hiera_large": "sam2.1_hiera_large",
    "base": "sam2.1_hiera_base_plus",
    "base_plus": "sam2.1_hiera_base_plus",
    "b+": "sam2.1_hiera_base_plus",
    "b_plus": "sam2.1_hiera_base_plus",
    "sam21_base_plus": "sam2.1_hiera_base_plus",
    "sam21_hiera_base_plus": "sam2.1_hiera_base_plus",
    "small": "sam2.1_hiera_small",
    "s": "sam2.1_hiera_small",
    "sam21_small": "sam2.1_hiera_small",
    "sam21_hiera_small": "sam2.1_hiera_small",
    "tiny": "sam2.1_hiera_tiny",
    "t": "sam2.1_hiera_tiny",
    "sam21_tiny": "sam2.1_hiera_tiny",
    "sam21_hiera_tiny": "sam2.1_hiera_tiny",
}


@dataclass
class SamuraiImportProbeResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any]


@dataclass
class SamuraiEnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    python_executable: Optional[Path] = None
    model_variant: str = DEFAULT_SAMURAI_MODEL_VARIANT
    config_name: Optional[str] = None
    config_identifier: Optional[str] = None
    config_path: Optional[Path] = None
    config_exists: bool = False
    demo_script_path: Optional[Path] = None
    demo_script_exists: bool = False
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
    jpeg4py_importable: bool = False
    lmdb_importable: bool = False
    loguru_importable: bool = False
    sam2_importable: bool = False
    sam2_module_path: Optional[str] = None
    build_sam_importable: bool = False
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
            and self.demo_script_exists
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
            and self.jpeg4py_importable
            and self.lmdb_importable
            and self.loguru_importable
            and self.sam2_importable
            and self.build_sam_importable
        )


@dataclass
class SamuraiSmokeTestResult:
    command: list[str]
    model_variant: str
    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any]

    @property
    def succeeded(self) -> bool:
        return (
            self.returncode == 0
            and bool(self.payload.get("ok"))
            and int(self.payload.get("propagated_frame_count", 0)) >= 2
        )


@dataclass
class SamuraiRuntimeSettings:
    checkpoint_path: str | Path
    repo_path: str | Path | None = None
    python_executable: str | Path | None = None
    backend_mode: str = "subprocess"
    model_variant: str = DEFAULT_SAMURAI_MODEL_VARIANT
    device: str = "auto"
    representative_point_mode: str = "mask_centroid"
    inference_resolution: int | tuple[int, int] | None = None
    initial_bbox_padding_px: float = 12.0
    subprocess_timeout_sec: float = DEFAULT_SAMURAI_SUBPROCESS_TIMEOUT_SEC


@dataclass
class SamuraiOfflineResult:
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
    model_variant: str
    config_name: str
    representative_point_mode: str
    backend_mode: str


@dataclass
class SamuraiBackendSmokeTestResult:
    checkpoint_name: str
    model_variant: str
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


def discover_samurai_checkpoints(
    models_dir: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_SAMURAI_MODELS_DIR
    discovered: dict[str, Path] = {}
    for candidate in _discover_checkpoints_in_directory(directory):
        discovered[str(candidate.resolve())] = candidate.resolve()

    repo_root = _resolve_samurai_repo_path(repo_path)
    if repo_root is not None:
        for candidate in _discover_checkpoints_in_directory(repo_root / "sam2" / "checkpoints"):
            discovered[str(candidate.resolve())] = candidate.resolve()

    return sorted(discovered.values(), key=lambda path: (path.name.lower(), str(path).lower()))


def select_default_samurai_checkpoint(
    checkpoints: list[Path],
    model_variant: str = DEFAULT_SAMURAI_MODEL_VARIANT,
    models_dir: str | Path | None = None,
) -> Optional[Path]:
    if not checkpoints:
        return None
    expected_name = _checkpoint_name_for_variant(model_variant)
    matches = [path for path in checkpoints if path.name.lower() == expected_name.lower()]
    if not matches:
        return checkpoints[0]

    preferred_dir = (
        Path(models_dir).expanduser().resolve()
        if models_dir is not None
        else DEFAULT_SAMURAI_MODELS_DIR.resolve()
    )
    for path in matches:
        try:
            path.relative_to(preferred_dir)
            return path
        except ValueError:
            continue
    return matches[0]


def probe_samurai_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
    python_executable: str | Path | None = None,
    model_variant: str = DEFAULT_SAMURAI_MODEL_VARIANT,
    import_probe_timeout_sec: float = DEFAULT_SAMURAI_IMPORT_PROBE_TIMEOUT_SEC,
) -> SamuraiEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_SAMURAI_MODELS_DIR
    directory = directory.resolve()
    normalized_variant = _normalize_model_variant(model_variant)
    resolved_repo_path = _resolve_samurai_repo_path(repo_path)
    resolved_python = _resolve_samurai_python_executable(python_executable)
    config_name = _config_name_for_variant(normalized_variant)
    config_identifier = _config_identifier_for_variant(normalized_variant)
    config_path = (
        resolved_repo_path / "sam2" / "sam2" / "configs" / "samurai" / config_name
        if resolved_repo_path is not None
        else None
    )
    demo_script_path = resolved_repo_path / "scripts" / "demo.py" if resolved_repo_path else None
    upstream_checkpoint_path = (
        resolved_repo_path / "sam2" / "checkpoints" / _checkpoint_name_for_variant(normalized_variant)
        if resolved_repo_path is not None
        else None
    )
    checkpoints = discover_samurai_checkpoints(directory, resolved_repo_path)
    selected_checkpoint = _resolve_checkpoint_path(
        checkpoint_path,
        checkpoints,
        directory,
        normalized_variant,
    )

    status = SamuraiEnvironmentStatus(
        models_dir=directory,
        repo_path=resolved_repo_path,
        python_executable=resolved_python,
        model_variant=normalized_variant,
        config_name=config_name,
        config_identifier=config_identifier,
        config_path=config_path,
        config_exists=bool(config_path and config_path.exists()),
        demo_script_path=demo_script_path,
        demo_script_exists=bool(demo_script_path and demo_script_path.exists()),
        available_checkpoints=checkpoints,
        selected_checkpoint=selected_checkpoint,
        checkpoint_exists=bool(selected_checkpoint and selected_checkpoint.exists()),
        upstream_checkpoint_path=upstream_checkpoint_path,
        upstream_checkpoint_exists=bool(
            upstream_checkpoint_path and upstream_checkpoint_path.exists()
        ),
    )

    if repo_path and resolved_repo_path is None:
        status.errors.append(f"Could not resolve SAMURAI repo from: {repo_path}")
    elif resolved_repo_path is None:
        status.errors.append(
            "Could not resolve SAMURAI repo. Pass --repo-path pointing at the local SAMURAI clone."
        )

    if resolved_python is None or not resolved_python.exists():
        status.errors.append(
            f"SAMURAI Python executable not found: {resolved_python or python_executable}"
        )

    if not directory.exists():
        status.warnings.append(
            f"Checkpoint directory does not exist yet: {directory}. "
            "Place SAM2.1 checkpoints inside tracking_models/samurai."
        )

    if normalized_variant not in SAMURAI_MODEL_VARIANTS:
        status.errors.append(
            f"Unsupported SAMURAI model variant: {model_variant}. "
            f"Supported variants: {', '.join(sorted(SAMURAI_MODEL_VARIANTS))}."
        )

    if config_path is None:
        status.errors.append("SAMURAI config path cannot be resolved without repo path.")
    elif not config_path.exists():
        status.errors.append(f"SAMURAI config not found: {config_path}")

    if demo_script_path is None:
        status.errors.append("SAMURAI demo script cannot be resolved without repo path.")
    elif not demo_script_path.exists():
        status.errors.append(f"SAMURAI demo script not found: {demo_script_path}")

    if selected_checkpoint is None:
        status.errors.append(
            "No SAMURAI/SAM2.1 checkpoint detected. Expected SAM2.1 checkpoints in "
            "tracking_models/samurai or in repo/sam2/checkpoints."
        )
    elif checkpoint_path and not status.checkpoint_exists:
        status.errors.append(f"Selected checkpoint not found: {selected_checkpoint}")

    if resolved_python is not None and resolved_python.exists():
        probe = _run_samurai_import_probe(
            resolved_python,
            resolved_repo_path,
            timeout_sec=import_probe_timeout_sec,
        )
        _apply_import_probe_result(status, probe)

    if not status.cuda_available and status.torch_importable:
        status.errors.append("PyTorch CUDA is not available in the SAMURAI environment.")

    numpy_major = _numpy_major_version(status.numpy_version)
    if numpy_major is not None and numpy_major >= 2:
        status.warnings.append(
            "NumPy 2.x can break older compiled packages in SAM2-style stacks. "
            "Use numpy==1.26.4 if imports or torch/numpy conversion fail."
        )

    if not status.cuda_extension_importable:
        status.warnings.append(
            "SAM2 custom CUDA extension is not importable. This is acceptable for feasibility; "
            "mask post-processing may be skipped, as in the SAM2/DAM4SAM integrations."
        )

    if not status.is_ready:
        status.hints.append(
            "Run from the dedicated `samurai` Conda env or pass "
            "--python-executable C:\\Users\\rlewa\\anaconda3\\envs\\samurai\\python.exe."
        )
        status.hints.append(
            "Validated feasibility stack: Python 3.10, torch 2.5.1+cu121, "
            "torchvision 0.20.1, numpy 1.26.4."
        )

    return status


def format_samurai_environment_report(status: SamuraiEnvironmentStatus) -> str:
    available = _format_available_checkpoints(status.available_checkpoints)
    selected = str(status.selected_checkpoint) if status.selected_checkpoint else "None"
    lines = [
        "SAMURAI environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Python executable: {status.python_executable or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Model variant: {status.model_variant}",
        f"- Config name: {status.config_name or 'None'}",
        f"- Config identifier: {status.config_identifier or 'None'}",
        f"- Config path: {status.config_path or 'None'}",
        f"- Config exists: {'yes' if status.config_exists else 'no'}",
        f"- Demo script: {status.demo_script_path or 'None'}",
        f"- Demo script exists: {'yes' if status.demo_script_exists else 'no'}",
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
        f"- jpeg4py import: {'ok' if status.jpeg4py_importable else 'failed'}",
        f"- lmdb import: {'ok' if status.lmdb_importable else 'failed'}",
        f"- loguru import: {'ok' if status.loguru_importable else 'failed'}",
        f"- SAM2 import: {'ok' if status.sam2_importable else 'failed'}",
        f"- SAM2 build_sam import: {'ok' if status.build_sam_importable else 'failed'}",
        f"- SAM2 CUDA extension: {'ok' if status.cuda_extension_importable else 'missing'}",
    ]
    if status.sam2_module_path:
        lines.append(f"- SAM2 module: {status.sam2_module_path}")
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


def run_samurai_smoke_test(
    *,
    repo_path: str | Path,
    checkpoint_path: str | Path,
    python_executable: str | Path | None = None,
    model_variant: str = DEFAULT_SAMURAI_MODEL_VARIANT,
    device: str = "auto",
    timeout_sec: float = DEFAULT_SAMURAI_SMOKE_TIMEOUT_SEC,
) -> SamuraiSmokeTestResult:
    normalized_variant = _normalize_model_variant(model_variant)
    resolved_repo_path = Path(repo_path).expanduser().resolve()
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    resolved_python = _resolve_samurai_python_executable(python_executable)
    if resolved_python is None:
        raise FileNotFoundError("SAMURAI Python executable could not be resolved.")
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"SAMURAI checkpoint not found: {resolved_checkpoint}")

    command = [
        str(resolved_python),
        "-c",
        _SAMURAI_SMOKE_TEST_CODE,
        _path_for_subprocess_argument(resolved_repo_path, resolved_python),
        _path_for_subprocess_argument(resolved_checkpoint, resolved_python),
        _config_identifier_for_variant(normalized_variant),
        str(device or "auto"),
    ]
    env = _build_samurai_subprocess_env(resolved_python, resolved_repo_path)
    completed = subprocess.run(
        command,
        cwd=str(_samurai_safe_subprocess_cwd(resolved_repo_path)),
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_sec),
        env=env,
    )
    return SamuraiSmokeTestResult(
        command=command,
        model_variant=normalized_variant,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        payload=_parse_prefixed_json_stdout(completed.stdout, _SMOKE_TEST_PREFIX),
    )


def format_samurai_smoke_test_report(result: SamuraiSmokeTestResult) -> str:
    payload = result.payload
    lines = [
        "SAMURAI smoke test",
        f"- model variant: {result.model_variant}",
        f"- returncode: {result.returncode}",
        f"- ok: {'yes' if result.succeeded else 'no'}",
    ]
    for key in (
        "checkpoint",
        "config_identifier",
        "device",
        "frame_shape",
        "init_bbox_xywh",
        "init_mask_shape",
        "init_mask_area",
        "propagated_frame_count",
        "last_mask_shape",
        "last_mask_area",
        "last_mask_bbox_xyxy",
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


def run_samurai_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray | None,
    settings: SamuraiRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
    query_boxes_xywh: np.ndarray | None = None,
) -> SamuraiOfflineResult:
    """Run SAMURAI on a sequence using bbox prompts."""

    if frames is None or np.asarray(frames).size == 0:
        raise ValueError("No frames provided for SAMURAI inference.")
    frames_np = np.asarray(frames)
    if frames_np.ndim not in (3, 4):
        raise ValueError(f"Expected frames with ndim 3 or 4, got shape {frames_np.shape}.")

    backend_mode = str(settings.backend_mode or "subprocess").strip().lower()
    if backend_mode == "auto":
        backend_mode = "subprocess"
    if backend_mode == "direct":
        return _run_samurai_direct_offline(
            frames_np,
            query_points_tyx,
            settings,
            query_boxes_xyxy=query_boxes_xyxy,
            query_boxes_xywh=query_boxes_xywh,
            backend_mode="direct",
        )
    if backend_mode != "subprocess":
        raise ValueError(
            f"Unsupported SAMURAI backend mode: {settings.backend_mode}. "
            "Expected `subprocess` or `direct`."
        )
    return _run_samurai_subprocess_offline(
        frames_np,
        query_points_tyx,
        settings,
        query_boxes_xyxy=query_boxes_xyxy,
        query_boxes_xywh=query_boxes_xywh,
    )


def run_samurai_backend_smoke_test(
    *,
    repo_path: str | Path,
    checkpoint_path: str | Path,
    python_executable: str | Path | None = None,
    model_variant: str = DEFAULT_SAMURAI_MODEL_VARIANT,
    device: str = "auto",
    backend_mode: str = "subprocess",
    representative_point_mode: str = "mask_centroid",
    timeout_sec: float = DEFAULT_SAMURAI_SUBPROCESS_TIMEOUT_SEC,
) -> SamuraiBackendSmokeTestResult:
    frames, query_points_tyx, query_boxes_xyxy = _build_samurai_backend_smoke_inputs()
    result = run_samurai_offline(
        frames,
        query_points_tyx,
        SamuraiRuntimeSettings(
            checkpoint_path=checkpoint_path,
            repo_path=repo_path,
            python_executable=python_executable,
            backend_mode=backend_mode,
            model_variant=model_variant,
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

    return SamuraiBackendSmokeTestResult(
        checkpoint_name=result.checkpoint_name,
        model_variant=result.model_variant,
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


def format_samurai_backend_smoke_test_report(result: SamuraiBackendSmokeTestResult) -> str:
    return "\n".join(
        [
            "SAMURAI backend smoke test",
            f"- checkpoint: {result.checkpoint_name}",
            f"- backend mode: {result.backend_mode}",
            f"- model variant: {result.model_variant}",
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


def _run_samurai_direct_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray | None,
    settings: SamuraiRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
    query_boxes_xywh: np.ndarray | None = None,
    backend_mode: str = "direct",
) -> SamuraiOfflineResult:
    resolved_repo_path = _resolve_samurai_repo_path(settings.repo_path)
    if resolved_repo_path is None:
        raise FileNotFoundError(
            "Could not resolve the SAMURAI repo. Pass repo_path pointing at the local clone."
        )
    checkpoint_file = _resolve_existing_samurai_checkpoint_file(settings.checkpoint_path)
    model_variant = _normalize_model_variant(settings.model_variant)
    config_identifier = _config_identifier_for_variant(model_variant)
    config_name = _config_name_for_variant(model_variant)
    representative_point_mode = _normalize_samurai_representative_point_mode(
        settings.representative_point_mode
    )

    frames_rgb, original_resolution = _prepare_frames_rgb(frames)
    inference_resolution = _normalize_optional_resolution(
        settings.inference_resolution,
        original_resolution,
    )
    resized_frames = _resize_video_numpy(frames_rgb, inference_resolution)
    frames_gray_original = _video_to_gray_float(frames_rgb)

    if query_boxes_xyxy is None and query_boxes_xywh is not None:
        query_boxes_xyxy = _boxes_xywh_to_xyxy(np.asarray(query_boxes_xywh, dtype=np.float32))

    num_queries = _infer_num_queries(
        query_points_tyx=query_points_tyx,
        query_boxes_xyxy=query_boxes_xyxy,
        query_initial_masks=None,
    )
    query_points = _normalize_query_points_tyx(query_points_tyx, num_queries)
    query_boxes = _normalize_query_boxes_xyxy(query_boxes_xyxy, num_queries)
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
    seed_frames = _extract_samurai_query_seed_frames(
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

    with _samurai_import_context(resolved_repo_path):
        torch_module = importlib.import_module("torch")
        resolved_device = _resolve_samurai_torch_device(torch_module, settings.device)
        build_sam_module = importlib.import_module("sam2.build_sam")
        pil_image_module = importlib.import_module("PIL.Image")

        with TemporaryDirectory(prefix="trackleed_samurai_frames_") as temp_dir:
            frame_dir = Path(temp_dir) / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for frame_index, frame in enumerate(resized_frames):
                pil_image_module.fromarray(frame).save(
                    frame_dir / f"{frame_index:06d}.jpg",
                    quality=95,
                )

            predictor = build_sam_module.build_sam2_video_predictor(
                config_identifier,
                str(checkpoint_file),
                device=resolved_device,
            )
            autocast_ctx = (
                torch_module.autocast("cuda", dtype=torch_module.float16)
                if str(resolved_device).startswith("cuda")
                else nullcontext()
            )
            with torch_module.inference_mode(), autocast_ctx:
                for query_index in range(num_queries):
                    state = predictor.init_state(str(frame_dir), offload_video_to_cpu=True)
                    seed_frame = int(seed_frames[query_index])
                    seed_bbox_xyxy = _bbox_xywh_to_xyxy(seed_boxes_xywh[query_index])
                    predictor.add_new_points_or_box(
                        state,
                        box=np.asarray(seed_bbox_xyxy, dtype=np.float32),
                        frame_idx=seed_frame,
                        obj_id=0,
                    )
                    for frame_index, object_ids, mask_logits in predictor.propagate_in_video(state):
                        for obj_id, mask_tensor in zip(object_ids, mask_logits):
                            if int(obj_id) != 0:
                                continue
                            _record_samurai_mask(
                                mask_tensor,
                                query_index=query_index,
                                frame_index=int(frame_index),
                                masks=masks,
                                tracks_xy=tracks_xy,
                                visible_mask=visible_mask,
                                mask_areas=mask_areas,
                                mask_bboxes_xyxy=mask_bboxes_xyxy,
                                mask_bboxes_xywh=mask_bboxes_xywh,
                                mask_scores=mask_scores,
                                frame_gray=frames_gray_original[int(frame_index)],
                                representative_point_mode=representative_point_mode,
                                original_resolution=original_resolution,
                            )
                    _release_samurai_state(torch_module, state)
            _release_samurai_predictor(torch_module, predictor)

    return SamuraiOfflineResult(
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
        model_variant=model_variant,
        config_name=config_name,
        representative_point_mode=representative_point_mode,
        backend_mode=str(backend_mode),
    )


def _run_samurai_subprocess_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray | None,
    settings: SamuraiRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
    query_boxes_xywh: np.ndarray | None = None,
) -> SamuraiOfflineResult:
    resolved_repo_path = _resolve_samurai_repo_path(settings.repo_path)
    if resolved_repo_path is None:
        raise FileNotFoundError(
            "Could not resolve the SAMURAI repo. Pass repo_path pointing at the local clone."
        )
    resolved_python = _resolve_samurai_python_executable(settings.python_executable)
    if resolved_python is None or not resolved_python.exists():
        raise FileNotFoundError(f"SAMURAI Python executable not found: {resolved_python}")
    checkpoint_file = _resolve_existing_samurai_checkpoint_file(settings.checkpoint_path)
    model_variant = _normalize_model_variant(settings.model_variant)
    representative_point_mode = _normalize_samurai_representative_point_mode(
        settings.representative_point_mode
    )

    script_path = Path(__file__).resolve().parents[2] / DEFAULT_SAMURAI_SUBPROCESS_SCRIPT
    if not script_path.exists():
        raise FileNotFoundError(f"SAMURAI subprocess script not found: {script_path}")

    with TemporaryDirectory(prefix="trackleed_samurai_") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "samurai_input.npz"
        output_path = temp_root / "samurai_output.npz"
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
            has_query_boxes_xywh=np.asarray(query_boxes_xywh is not None),
            query_boxes_xywh=(
                np.asarray(query_boxes_xywh, dtype=np.float32)
                if query_boxes_xywh is not None
                else np.empty((0, 4), dtype=np.float32)
            ),
        )

        command = [
            str(resolved_python),
            _path_for_subprocess_argument(script_path, resolved_python),
            "--input-npz",
            _path_for_subprocess_argument(input_path, resolved_python),
            "--output-npz",
            _path_for_subprocess_argument(output_path, resolved_python),
            "--checkpoint",
            _path_for_subprocess_argument(checkpoint_file, resolved_python),
            "--repo-path",
            _path_for_subprocess_argument(resolved_repo_path, resolved_python),
            "--model-variant",
            model_variant,
            "--device",
            str(settings.device or "auto"),
            "--representative-point-mode",
            representative_point_mode,
            "--initial-bbox-padding-px",
            str(float(settings.initial_bbox_padding_px)),
            "--subprocess-timeout-sec",
            str(float(settings.subprocess_timeout_sec)),
        ]
        if settings.inference_resolution is not None:
            height, width = _normalize_optional_resolution(settings.inference_resolution, (1, 1))
            command.extend(["--height", str(int(height)), "--width", str(int(width))])

        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            check=False,
            timeout=float(settings.subprocess_timeout_sec),
            env=_build_samurai_subprocess_env(resolved_python, resolved_repo_path),
        )
        if completed.returncode != 0 or not output_path.exists():
            details = {
                "command": command,
                "returncode": int(completed.returncode),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
            raise RuntimeError(
                "SAMURAI subprocess inference failed:\n"
                + json.dumps(details, indent=2, ensure_ascii=False)
            )

        with np.load(output_path, allow_pickle=False) as data:
            return SamuraiOfflineResult(
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
                model_variant=str(np.asarray(data["model_variant"]).tolist()),
                config_name=str(np.asarray(data["config_name"]).tolist()),
                representative_point_mode=str(np.asarray(data["representative_point_mode"]).tolist()),
                backend_mode=str(np.asarray(data["backend_mode"]).tolist()),
            )


def _resolve_samurai_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidates = [Path(repo_path)] if repo_path else list(DEFAULT_SAMURAI_REPO_CANDIDATES)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "scripts" / "demo.py").exists() and (
            resolved / "sam2" / "sam2" / "build_sam.py"
        ).exists():
            return resolved
    return None


def _resolve_samurai_python_executable(python_executable: str | Path | None) -> Optional[Path]:
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
    model_variant: str,
) -> Optional[Path]:
    if checkpoint_path:
        candidate = Path(checkpoint_path).expanduser()
        if not candidate.is_absolute():
            candidate = models_dir / candidate
        return candidate.resolve()
    return select_default_samurai_checkpoint(checkpoints, model_variant, models_dir)


def _checkpoint_name_for_variant(model_variant: str) -> str:
    return SAMURAI_MODEL_VARIANTS[_normalize_model_variant(model_variant)][0]


def _config_name_for_variant(model_variant: str) -> str:
    return SAMURAI_MODEL_VARIANTS[_normalize_model_variant(model_variant)][1]


def _config_identifier_for_variant(model_variant: str) -> str:
    return SAMURAI_MODEL_VARIANTS[_normalize_model_variant(model_variant)][2]


def _normalize_model_variant(model_variant: str | None) -> str:
    raw = str(model_variant or DEFAULT_SAMURAI_MODEL_VARIANT).strip()
    if not raw:
        return DEFAULT_SAMURAI_MODEL_VARIANT
    name = Path(raw).name
    for suffix in (".pth.tar", ".pth", ".pt", ".tar", ".yaml", ".yml"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    key = name.strip().lower().replace("-", "_").replace("sam2_1", "sam2.1")
    key = key.replace("hiera_b+", "hiera_base_plus").replace("b+", "base_plus")
    key = _SAMURAI_VARIANT_ALIASES.get(key, key)
    if key in SAMURAI_MODEL_VARIANTS:
        return key
    return raw


def _run_samurai_import_probe(
    python_executable: Path,
    repo_path: Optional[Path],
    *,
    timeout_sec: float,
) -> SamuraiImportProbeResult:
    command = [str(python_executable), "-c", _SAMURAI_IMPORT_PROBE_CODE]
    completed = subprocess.run(
        command,
        cwd=str(_samurai_safe_subprocess_cwd(repo_path)) if repo_path is not None else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_sec),
        env=_build_samurai_subprocess_env(python_executable, repo_path),
    )
    return SamuraiImportProbeResult(
        command=command,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        payload=_parse_prefixed_json_stdout(completed.stdout, _IMPORT_PROBE_PREFIX),
    )


def _apply_import_probe_result(
    status: SamuraiEnvironmentStatus,
    probe: SamuraiImportProbeResult,
) -> None:
    status.import_probe_returncode = probe.returncode
    status.import_probe_stdout = probe.stdout
    status.import_probe_stderr = probe.stderr
    payload = probe.payload
    if probe.returncode != 0:
        status.errors.append(
            "SAMURAI import probe failed with return code "
            f"{probe.returncode}: {probe.stderr.strip() or probe.stdout.strip()}"
        )
    if not payload:
        status.errors.append("SAMURAI import probe did not return parseable diagnostics JSON.")
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
    status.jpeg4py_importable = bool(payload.get("jpeg4py_importable"))
    status.lmdb_importable = bool(payload.get("lmdb_importable"))
    status.loguru_importable = bool(payload.get("loguru_importable"))
    status.sam2_importable = bool(payload.get("sam2_importable"))
    status.sam2_module_path = payload.get("sam2_module_path")
    status.build_sam_importable = bool(payload.get("sam2_build_sam_importable"))
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


def _build_samurai_subprocess_env(
    python_executable: Path,
    repo_path: Optional[Path] = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if repo_path is not None:
        env["TRACKLEED_SAMURAI_REPO"] = _path_for_subprocess_argument(
            repo_path,
            python_executable,
        )
        current_pythonpath = env.get("PYTHONPATH", "")
        path_separator = ";" if _is_windows_python_executable(python_executable) else os.pathsep
        pythonpath_entries = [
            _path_for_subprocess_argument(repo_path / "sam2", python_executable)
        ]
        env["PYTHONPATH"] = path_separator.join(
            pythonpath_entries + ([current_pythonpath] if current_pythonpath else [])
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


def _samurai_safe_subprocess_cwd(repo_path: Path) -> Path:
    sam2_repo = repo_path / "sam2"
    return sam2_repo if sam2_repo.exists() else repo_path


def _is_windows_python_executable(python_executable: Path) -> bool:
    return str(python_executable).lower().endswith(".exe")


def _path_for_subprocess_argument(path: str | Path, python_executable: Path) -> str:
    path_text = str(Path(path).expanduser().resolve())
    if not _is_windows_python_executable(python_executable):
        return path_text
    if not path_text.startswith("/mnt/"):
        return path_text

    parts = Path(path_text).parts
    if len(parts) < 4:
        return path_text
    drive = parts[2].upper()
    tail = "\\".join(parts[3:])
    return f"{drive}:\\{tail}"


def _numpy_major_version(version: str | None) -> Optional[int]:
    if not version:
        return None
    first_chunk = str(version).split(".", 1)[0]
    try:
        return int(first_chunk)
    except ValueError:
        return None


def _resolve_existing_samurai_checkpoint_file(checkpoint_path: str | Path) -> Path:
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"SAMURAI checkpoint not found: {checkpoint_file}")
    return checkpoint_file


@contextmanager
def _samurai_import_context(repo_path: Path) -> Iterator[None]:
    sam2_root = repo_path / "sam2"
    old_path = list(sys.path)
    old_cwd = os.getcwd()
    for candidate in (str(sam2_root), str(repo_path)):
        if candidate in sys.path:
            sys.path.remove(candidate)
        sys.path.insert(0, candidate)
    try:
        if sam2_root.exists():
            os.chdir(sam2_root)
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def _resolve_samurai_torch_device(torch_module: Any, device: str | None) -> str:
    normalized = str(device or "auto").strip().lower()
    if normalized in {"", "auto", "cuda"}:
        normalized = "cuda:0"
    if normalized.startswith("cuda") and not bool(torch_module.cuda.is_available()):
        raise RuntimeError("SAMURAI requested CUDA, but torch.cuda.is_available() is false.")
    return normalized


def _extract_samurai_query_seed_frames(
    query_points_tyx: np.ndarray | None,
    *,
    total_frames: int,
    num_queries: int,
) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("SAMURAI requires at least one frame.")
    if query_points_tyx is None:
        return np.zeros((num_queries,), dtype=np.int32)
    seed_frames = np.rint(np.asarray(query_points_tyx, dtype=np.float32)[:, 0]).astype(np.int32)
    if np.any(seed_frames < 0) or np.any(seed_frames >= int(total_frames)):
        raise ValueError(
            f"SAMURAI seed frame index out of range 0..{int(total_frames) - 1}: "
            f"{seed_frames.tolist()}"
        )
    return seed_frames


def _bbox_xywh_to_xyxy(
    bbox_xywh: np.ndarray | tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, width, height = [
        float(value) for value in np.asarray(bbox_xywh, dtype=np.float32).reshape(4)
    ]
    return (x, y, x + width, y + height)


def _boxes_xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes_xywh, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"query_boxes_xywh must have shape (N, 4), got {boxes.shape}.")
    converted = boxes.copy()
    converted[:, 2] = converted[:, 0] + np.maximum(1.0, converted[:, 2])
    converted[:, 3] = converted[:, 1] + np.maximum(1.0, converted[:, 3])
    return converted


def _record_samurai_mask(
    mask_tensor: Any,
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
    if not (0 <= frame_index < masks.shape[1]):
        return
    mask_bool = _mask_from_samurai_tensor(mask_tensor)
    mask_bool = _resize_mask_bool(mask_bool, original_resolution)
    masks[query_index, frame_index] = mask_bool
    area = float(np.count_nonzero(mask_bool))
    mask_areas[query_index, frame_index] = area
    if area <= 0.0:
        return

    bbox_xyxy = _mask_to_bbox_xyxy(mask_bool)
    if bbox_xyxy is None:
        return
    bbox_xywh = _bbox_xyxy_to_xywh(bbox_xyxy)
    mask_bboxes_xyxy[query_index, frame_index, :] = np.asarray(bbox_xyxy, dtype=np.float32)
    mask_bboxes_xywh[query_index, frame_index, :] = np.asarray(bbox_xywh, dtype=np.float32)
    mask_scores[query_index, frame_index] = 1.0

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


def _mask_from_samurai_tensor(mask_tensor: Any) -> np.ndarray:
    if hasattr(mask_tensor, "detach"):
        arr = mask_tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(mask_tensor)
    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim > 3:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SAMURAI mask tensor shape: {arr.shape}")
    return np.asarray(arr) > 0.0


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
    bbox_xywh = np.asarray([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)

    if mode == "bbox_center":
        return (float((x0 + x1) * 0.5), float((y0 + y1) * 0.5))
    if mode == "peak_in_bbox":
        peak = _peak_point_in_bbox_xywh(frame_gray, bbox_xywh)
        if peak is not None:
            return peak
    if mode == "gaussian_in_bbox":
        gaussian = _gaussian_point_in_bbox_xywh(frame_gray, bbox_xywh)
        if gaussian is not None:
            return gaussian
        peak = _peak_point_in_bbox_xywh(frame_gray, bbox_xywh)
        if peak is not None:
            return peak

    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0 or ys.size == 0:
        return None
    return (float(np.mean(xs)), float(np.mean(ys)))


def _peak_point_in_bbox_xywh(
    frame: np.ndarray,
    bbox_xywh: np.ndarray,
) -> tuple[float, float] | None:
    frame_np = np.asarray(frame)
    if frame_np.ndim not in (2, 3) or frame_np.size == 0:
        return None

    x, y, width, height = [
        float(value) for value in np.asarray(bbox_xywh, dtype=np.float32).reshape(4)
    ]
    if not np.all(np.isfinite([x, y, width, height])) or width <= 0.0 or height <= 0.0:
        return None

    frame_h, frame_w = int(frame_np.shape[0]), int(frame_np.shape[1])
    x0 = max(0, min(frame_w, int(np.floor(x))))
    y0 = max(0, min(frame_h, int(np.floor(y))))
    x1 = max(0, min(frame_w, int(np.ceil(x + width))))
    y1 = max(0, min(frame_h, int(np.ceil(y + height))))
    if x1 <= x0 or y1 <= y0:
        return None

    roi = frame_np[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    if roi.ndim == 3:
        roi = np.mean(roi.astype(np.float32, copy=False), axis=-1)
    else:
        roi = roi.astype(np.float32, copy=False)

    finite_mask = np.isfinite(roi)
    if not np.any(finite_mask):
        return None

    safe_roi = np.where(finite_mask, roi, -np.inf)
    peak_row, peak_col = np.unravel_index(int(np.argmax(safe_roi)), safe_roi.shape)
    return (float(x0 + peak_col) + 0.5, float(y0 + peak_row) + 0.5)


def _gaussian_point_in_bbox_xywh(
    frame: np.ndarray,
    bbox_xywh: np.ndarray,
) -> tuple[float, float] | None:
    try:
        from trackleed.analysis.peak_analyzer import fit_spot_to_gaussian
    except Exception:
        return None

    frame_np = np.asarray(frame)
    if frame_np.ndim not in (2, 3) or frame_np.size == 0:
        return None

    x, y, width, height = [
        float(value) for value in np.asarray(bbox_xywh, dtype=np.float32).reshape(4)
    ]
    if not np.all(np.isfinite([x, y, width, height])) or width <= 0.0 or height <= 0.0:
        return None

    frame_h, frame_w = int(frame_np.shape[0]), int(frame_np.shape[1])
    x0 = max(0, min(frame_w, int(np.floor(x))))
    y0 = max(0, min(frame_h, int(np.floor(y))))
    x1 = max(0, min(frame_w, int(np.ceil(x + width))))
    y1 = max(0, min(frame_h, int(np.ceil(y + height))))
    if x1 <= x0 or y1 <= y0:
        return None

    roi = frame_np[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    fit = fit_spot_to_gaussian(roi, center_preference_factor=0.35)
    if not fit:
        return None
    center_x = float(fit.get("center_x", np.nan))
    center_y = float(fit.get("center_y", np.nan))
    if not np.isfinite(center_x) or not np.isfinite(center_y):
        return None
    if not (0.0 <= center_x < float(max(1, x1 - x0)) and 0.0 <= center_y < float(max(1, y1 - y0))):
        return None
    return (float(x0 + center_x), float(y0 + center_y))


def _normalize_samurai_representative_point_mode(mode: str | None) -> str:
    normalized = str(mode or "mask_centroid").strip().lower().replace(" ", "_")
    aliases = {
        "centroid": "mask_centroid",
        "mask_centroid": "mask_centroid",
        "maskcenter": "mask_centroid",
        "bbox_center": "bbox_center",
        "bboxcenter": "bbox_center",
        "peak": "peak_in_bbox",
        "peak_in_bbox": "peak_in_bbox",
        "bbox_peak": "peak_in_bbox",
        "gaussian": "gaussian_in_bbox",
        "gaussian_in_bbox": "gaussian_in_bbox",
        "bbox_gaussian": "gaussian_in_bbox",
        "2d_gauss": "gaussian_in_bbox",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported SAMURAI representative point mode: {mode}. "
            "Expected mask_centroid, bbox_center, peak_in_bbox, or gaussian_in_bbox."
        )
    return aliases[normalized]


def _release_samurai_state(torch_module: Any, state: Any) -> None:
    del state
    if hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "empty_cache"):
        try:
            torch_module.cuda.empty_cache()
        except Exception:
            pass


def _release_samurai_predictor(torch_module: Any, predictor: Any) -> None:
    del predictor
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    if hasattr(torch_module, "clear_autocast_cache"):
        try:
            torch_module.clear_autocast_cache()
        except Exception:
            pass
    if hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "empty_cache"):
        try:
            torch_module.cuda.empty_cache()
        except Exception:
            pass


def _build_samurai_backend_smoke_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


_SAMURAI_IMPORT_PROBE_CODE = r"""
import importlib
import json
import os
import sys

PREFIX = "__TRACKLEED_SAMURAI_IMPORT_PROBE__"
result = {"python_executable": sys.executable}
repo_path = os.environ.get("TRACKLEED_SAMURAI_REPO")
if repo_path:
    sam2_root = os.path.join(repo_path, "sam2")
    if sam2_root and sam2_root not in sys.path:
        sys.path.insert(0, sam2_root)


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
probe("jpeg4py", "jpeg4py")
probe("lmdb", "lmdb")
probe("loguru", "loguru")
probe("sam2", "sam2")
probe("sam2.build_sam", "sam2_build_sam")
probe("sam2._C", "sam2_cuda_extension")

print(PREFIX + json.dumps(result, sort_keys=True))
"""


_SAMURAI_SMOKE_TEST_CODE = r"""
import json
import os
import sys
import tempfile
import time
from contextlib import nullcontext

import numpy as np
from PIL import Image, ImageDraw

PREFIX = "__TRACKLEED_SAMURAI_SMOKE_TEST__"
repo_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TRACKLEED_SAMURAI_REPO")
if repo_path:
    sam2_root = os.path.join(repo_path, "sam2")
    if sam2_root and sam2_root not in sys.path:
        sys.path.insert(0, sam2_root)

checkpoint = sys.argv[2] if len(sys.argv) > 2 else os.environ["TRACKLEED_SAMURAI_CHECKPOINT"]
config_identifier = sys.argv[3] if len(sys.argv) > 3 else os.environ["TRACKLEED_SAMURAI_CONFIG"]
device_request = (
    sys.argv[4] if len(sys.argv) > 4 else os.environ.get("TRACKLEED_SAMURAI_DEVICE", "auto")
).strip().lower()


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


def mask_from_tensor(mask_tensor):
    arr = mask_tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return arr > 0.0


start = time.perf_counter()
import torch
from sam2.build_sam import build_sam2_video_predictor

cuda_available = bool(torch.cuda.is_available())
if device_request in {"", "auto", "cuda", "cuda:0"}:
    if not cuda_available:
        raise RuntimeError("CUDA is not available for the SAMURAI smoke test.")
    device = "cuda:0"
elif device_request == "cpu":
    device = "cpu"
else:
    device = device_request

with tempfile.TemporaryDirectory(prefix="trackleed_samurai_smoke_") as temp_dir:
    frame_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    make_frame(0).save(os.path.join(frame_dir, "000000.jpg"), quality=95)
    make_frame(3).save(os.path.join(frame_dir, "000001.jpg"), quality=95)

    predictor = build_sam2_video_predictor(config_identifier, checkpoint, device=device)
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.float16)
        if str(device).startswith("cuda")
        else nullcontext()
    )
    frame_payloads = []
    init_bbox_xywh = [52, 42, 24, 24]
    init_bbox_xyxy = np.asarray([52, 42, 76, 66], dtype=np.float32)
    with torch.inference_mode(), autocast_ctx:
        state = predictor.init_state(frame_dir, offload_video_to_cpu=True)
        _, _object_ids, init_masks = predictor.add_new_points_or_box(
            state,
            box=init_bbox_xyxy,
            frame_idx=0,
            obj_id=0,
        )
        init_mask = mask_from_tensor(init_masks[0])
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            for obj_id, mask_tensor in zip(object_ids, masks):
                if int(obj_id) != 0:
                    continue
                mask = mask_from_tensor(mask_tensor)
                frame_payloads.append(
                    {
                        "frame_idx": int(frame_idx),
                        "mask_shape": list(mask.shape),
                        "mask_area": int(np.count_nonzero(mask)),
                        "mask_bbox_xyxy": mask_bbox(mask),
                    }
                )

payload = {
    "ok": bool(frame_payloads and frame_payloads[-1]["mask_area"] > 0),
    "checkpoint": str(checkpoint),
    "config_identifier": str(config_identifier),
    "device": device,
    "frame_shape": [112, 144, 3],
    "init_bbox_xywh": init_bbox_xywh,
    "init_mask_shape": list(init_mask.shape),
    "init_mask_area": int(np.count_nonzero(init_mask)),
    "propagated_frame_count": len(frame_payloads),
    "last_mask_shape": frame_payloads[-1]["mask_shape"] if frame_payloads else None,
    "last_mask_area": frame_payloads[-1]["mask_area"] if frame_payloads else 0,
    "last_mask_bbox_xyxy": frame_payloads[-1]["mask_bbox_xyxy"] if frame_payloads else None,
    "elapsed_sec": round(time.perf_counter() - start, 3),
}
print(PREFIX + json.dumps(payload, sort_keys=True))
"""
