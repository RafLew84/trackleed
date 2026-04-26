"""OSTrack environment diagnostics and smoke-test helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Optional

import numpy as np


DEFAULT_OSTRACK_MODELS_DIR = Path("tracking_models") / "ostrack"
DEFAULT_OSTRACK_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "ostrack",
    Path(__file__).resolve().parents[3] / "OSTrack",
    Path(__file__).resolve().parents[2] / "ostrack",
)
DEFAULT_OSTRACK_CONFIG_NAME = "vitb_256_mae_ce_32x4_ep300"
DEFAULT_OSTRACK_CHECKPOINT_NAME = "OSTrack_ep0300.pth.tar"
DEFAULT_OSTRACK_SUBPROCESS_SCRIPT = Path("scripts") / "run_ostrack_subprocess.py"
DEFAULT_OSTRACK_PROFILE_TIMEOUT_SEC = 180.0
DEFAULT_OSTRACK_PROBE_TIMEOUT_SEC = 900.0
DEFAULT_OSTRACK_SUBPROCESS_TIMEOUT_SEC = 1800.0
DEFAULT_OSTRACK_BBOX_PADDING_CANDIDATES = (0, 4, 8, 12, 16)
_IMPORT_PROBE_PREFIX = "__TRACKLEED_OSTRACK_IMPORT_PROBE__"
SUPPORTED_OSTRACK_CONFIG_NAMES = (
    DEFAULT_OSTRACK_CONFIG_NAME,
    "vitb_384_mae_ce_32x4_ep300",
    "vitb_256_mae_32x4_ep300",
    "vitb_384_mae_32x4_ep300",
    "vitb_256_mae_ce_32x4_got10k_ep100",
    "vitb_384_mae_ce_32x4_got10k_ep100",
)


@dataclass
class OstrackEnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    python_executable: Optional[Path] = None
    config_name: str = DEFAULT_OSTRACK_CONFIG_NAME
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
    yaml_importable: bool = False
    easydict_importable: bool = False
    timm_importable: bool = False
    lmdb_importable: bool = False
    jpeg4py_importable: bool = False
    ostrack_parameter_importable: bool = False
    ostrack_tracker_importable: bool = False
    ostrack_parameter_module_path: Optional[str] = None
    ostrack_tracker_module_path: Optional[str] = None
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
            and self.easydict_importable
            and self.timm_importable
            and self.lmdb_importable
            and self.jpeg4py_importable
            and self.ostrack_parameter_importable
            and self.ostrack_tracker_importable
        )


@dataclass
class OstrackImportProbeResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any]


@dataclass
class OstrackProfileSmokeTestResult:
    command: list[str]
    config_name: str
    returncode: int
    stdout: str
    stderr: str
    macs: Optional[str]
    params: Optional[str]
    latency_ms: Optional[float]
    fps: Optional[float]


@dataclass
class OstrackRuntimeSettings:
    checkpoint_path: str | Path
    repo_path: str | Path | None = None
    python_executable: str | Path | None = None
    backend_mode: str = "subprocess"
    model_variant: str = DEFAULT_OSTRACK_CONFIG_NAME
    device: str = "auto"
    representative_point_mode: str = "bbox_center"
    inference_resolution: int | tuple[int, int] | None = None
    initial_bbox_padding_px: float = 0.0
    subprocess_timeout_sec: float = DEFAULT_OSTRACK_SUBPROCESS_TIMEOUT_SEC


@dataclass
class OstrackOfflineResult:
    tracks_xy: np.ndarray
    visible_mask: np.ndarray
    tracked_bboxes_xywh: np.ndarray
    tracked_bboxes_xyxy: np.ndarray
    bbox_scores: np.ndarray
    inference_resolution: tuple[int, int]
    original_resolution: tuple[int, int]
    checkpoint_name: str
    model_variant: str
    representative_point_mode: str
    initial_bbox_padding_px: float


@dataclass
class OstrackBackendSmokeTestResult:
    checkpoint_name: str
    model_variant: str
    backend_mode: str
    representative_point_mode: str
    initial_bbox_padding_px: float
    tracks_shape: tuple[int, int, int]
    visible_shape: tuple[int, int]
    bboxes_shape: tuple[int, int, int]
    visible_count: int
    first_track_point_xy: tuple[float, float]
    last_visible_point_xy: tuple[float, float]
    first_bbox_xywh: tuple[float, float, float, float]
    last_bbox_xywh: tuple[float, float, float, float]


@dataclass
class OstrackInitialBBoxPaddingResult:
    padding_px: int
    init_bbox_xywh: tuple[int, int, int, int]
    init_bbox_iou: float
    mean_tracking_iou: float
    min_tracking_iou: float
    last_tracking_iou: float
    mean_center_error_px: float
    max_center_error_px: float
    aggregate_score: float


@dataclass
class OstrackInitialBBoxSampleResult:
    sample_name: str
    frame_shape: tuple[int, int, int]
    frame_count: int
    reference_init_bbox_xywh: tuple[int, int, int, int]
    padding_results: list[OstrackInitialBBoxPaddingResult]


@dataclass
class OstrackInitialBBoxProbeResult:
    config_name: str
    checkpoint_name: str
    samples: list[OstrackInitialBBoxSampleResult]
    candidate_paddings_px: list[int]
    recommended_padding_px: int
    aggregate_mean_scores: dict[int, float]
    notes: list[str] = field(default_factory=list)


@dataclass
class _SyntheticOstrackProbeSample:
    name: str
    frames: np.ndarray
    gt_bboxes_xywh: np.ndarray


def discover_ostrack_checkpoints(models_dir: str | Path | None = None) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_OSTRACK_MODELS_DIR
    if not directory.exists():
        return []
    checkpoints = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith((".pth", ".pt", ".tar", ".pth.tar")):
            checkpoints.append(path.resolve())
    return sorted(set(checkpoints), key=lambda item: item.name.lower())


def select_default_ostrack_checkpoint(
    checkpoints: list[Path],
    config_name: str = DEFAULT_OSTRACK_CONFIG_NAME,
) -> Optional[Path]:
    if not checkpoints:
        return None
    normalized_config_name = str(config_name or DEFAULT_OSTRACK_CONFIG_NAME).lower()
    by_name = {path.name.lower(): path for path in checkpoints}
    preferred_names = (
        f"{config_name}.pth.tar",
        f"{config_name}.pth",
        f"{config_name}.pt",
        DEFAULT_OSTRACK_CHECKPOINT_NAME,
    )
    for name in preferred_names[:-1]:
        checkpoint = by_name.get(name.lower())
        if checkpoint is not None:
            return checkpoint
    for path in checkpoints:
        if (
            path.name.lower() == DEFAULT_OSTRACK_CHECKPOINT_NAME.lower()
            and path.parent.name.lower() == normalized_config_name
        ):
            return path
    for path in checkpoints:
        normalized_parts = {part.lower() for part in path.parts}
        if (
            path.name.lower() == DEFAULT_OSTRACK_CHECKPOINT_NAME.lower()
            and normalized_config_name in normalized_parts
        ):
            return path
    for path in checkpoints:
        if normalized_config_name in path.name.lower():
            return path
    generic_checkpoint = by_name.get(DEFAULT_OSTRACK_CHECKPOINT_NAME.lower())
    if generic_checkpoint is not None:
        return generic_checkpoint
    return checkpoints[0]


def probe_ostrack_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
    python_executable: str | Path | None = None,
    config_name: str = DEFAULT_OSTRACK_CONFIG_NAME,
    import_probe_timeout_sec: float = 60.0,
) -> OstrackEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_OSTRACK_MODELS_DIR
    directory = directory.resolve()
    resolved_repo_path = _resolve_ostrack_repo_path(repo_path)
    resolved_python = _resolve_ostrack_python_executable(python_executable)
    config_path = (
        resolved_repo_path / "experiments" / "ostrack" / f"{config_name}.yaml"
        if resolved_repo_path is not None
        else None
    )
    upstream_checkpoint_path = (
        resolved_repo_path
        / "output"
        / "checkpoints"
        / "train"
        / "ostrack"
        / config_name
        / DEFAULT_OSTRACK_CHECKPOINT_NAME
        if resolved_repo_path is not None
        else None
    )
    checkpoints = discover_ostrack_checkpoints(directory)
    selected_checkpoint = _resolve_checkpoint_path(
        checkpoint_path,
        checkpoints,
        directory,
        upstream_checkpoint_path,
        config_name,
    )

    status = OstrackEnvironmentStatus(
        models_dir=directory,
        repo_path=resolved_repo_path,
        python_executable=resolved_python,
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
        status.errors.append(f"Could not resolve OSTrack repo from: {repo_path}")
    elif resolved_repo_path is None:
        status.errors.append(
            "Could not resolve OSTrack repo. Pass --repo-path pointing at the local OSTrack clone."
        )

    if resolved_python is None or not resolved_python.exists():
        status.errors.append(
            f"OSTrack Python executable not found: {resolved_python or python_executable}"
        )

    if not directory.exists():
        status.warnings.append(
            f"Checkpoint directory does not exist yet: {directory}. "
            "Place OSTrack checkpoints inside tracking_models/ostrack."
        )

    if config_path is None:
        status.errors.append("OSTrack config path cannot be resolved without repo path.")
    elif not config_path.exists():
        status.errors.append(f"OSTrack config not found: {config_path}")

    if selected_checkpoint is None:
        status.errors.append(
            "No OSTrack checkpoint detected. Expected either "
            f"{config_name}.pth.tar in tracking_models/ostrack or "
            f"{DEFAULT_OSTRACK_CHECKPOINT_NAME} in the upstream output/checkpoints path."
        )
    elif checkpoint_path and not status.checkpoint_exists:
        status.errors.append(f"Selected checkpoint not found: {selected_checkpoint}")

    if resolved_python is not None and resolved_python.exists():
        probe = _run_ostrack_import_probe(
            resolved_python,
            resolved_repo_path,
            timeout_sec=import_probe_timeout_sec,
        )
        _apply_import_probe_result(status, probe)

    if not status.cuda_available and status.torch_importable:
        status.errors.append("PyTorch CUDA is not available in the OSTrack environment.")

    if not status.is_ready:
        status.hints.append(
            "Run from the dedicated `ostrack` Conda env or pass "
            "--python-executable C:\\Users\\rlewa\\anaconda3\\envs\\ostrack\\python.exe."
        )
        status.hints.append(
            "The validated Windows stack is Python 3.8 with torch 1.13.1+cu117, "
            "torchvision 0.14.1+cu117, timm 0.5.4, python-lmdb and jpeg4py."
        )

    return status


def format_ostrack_environment_report(status: OstrackEnvironmentStatus) -> str:
    available = ", ".join(path.name for path in status.available_checkpoints) or "none"
    selected = str(status.selected_checkpoint) if status.selected_checkpoint else "None"
    lines = [
        "OSTrack environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Python executable: {status.python_executable or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Config name: {status.config_name}",
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
        f"- PyYAML import: {'ok' if status.yaml_importable else 'failed'}",
        f"- easydict import: {'ok' if status.easydict_importable else 'failed'}",
        f"- timm import: {'ok' if status.timm_importable else 'failed'}",
        f"- lmdb import: {'ok' if status.lmdb_importable else 'failed'}",
        f"- jpeg4py import: {'ok' if status.jpeg4py_importable else 'failed'}",
        f"- OSTrack parameter import: {'ok' if status.ostrack_parameter_importable else 'failed'}",
        f"- OSTrack tracker import: {'ok' if status.ostrack_tracker_importable else 'failed'}",
    ]
    if status.ostrack_parameter_module_path:
        lines.append(f"- OSTrack parameter module: {status.ostrack_parameter_module_path}")
    if status.ostrack_tracker_module_path:
        lines.append(f"- OSTrack tracker module: {status.ostrack_tracker_module_path}")
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


def run_ostrack_profile_smoke_test(
    *,
    repo_path: str | Path,
    python_executable: str | Path | None = None,
    config_name: str = DEFAULT_OSTRACK_CONFIG_NAME,
    timeout_sec: float = DEFAULT_OSTRACK_PROFILE_TIMEOUT_SEC,
) -> OstrackProfileSmokeTestResult:
    resolved_repo_path = Path(repo_path).expanduser().resolve()
    resolved_python = _resolve_ostrack_python_executable(python_executable)
    if resolved_python is None:
        raise FileNotFoundError("OSTrack Python executable could not be resolved.")
    profile_script = resolved_repo_path / "tracking" / "profile_model.py"
    command = [
        str(resolved_python),
        str(profile_script),
        "--script",
        "ostrack",
        "--config",
        config_name,
    ]
    completed = subprocess.run(
        command,
        cwd=str(resolved_repo_path),
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_sec),
        env=_build_windows_conda_subprocess_env(resolved_python, resolved_repo_path),
    )
    return OstrackProfileSmokeTestResult(
        command=command,
        config_name=config_name,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        macs=_extract_first_match(r"overall macs is\s+([^\r\n]+)", completed.stdout),
        params=_extract_first_match(r"overall params is\s+([^\r\n]+)", completed.stdout),
        latency_ms=_extract_float(
            r"The average overall latency is\s+([0-9.]+)\s*ms",
            completed.stdout,
        ),
        fps=_extract_float(r"FPS is\s+([0-9.]+)\s*fps", completed.stdout),
    )


def probe_ostrack_initial_bbox_padding(
    *,
    repo_path: str | Path,
    checkpoint_path: str | Path,
    python_executable: str | Path | None = None,
    config_name: str = DEFAULT_OSTRACK_CONFIG_NAME,
    candidate_paddings_px: list[int] | tuple[int, ...] | None = None,
    timeout_sec: float = DEFAULT_OSTRACK_PROBE_TIMEOUT_SEC,
) -> OstrackInitialBBoxProbeResult:
    """Run OSTrack on synthetic LEED-like spot sequences for bbox padding selection."""

    resolved_repo_path = Path(repo_path).expanduser().resolve()
    resolved_python = _resolve_ostrack_python_executable(python_executable)
    if resolved_python is None:
        raise FileNotFoundError("OSTrack Python executable could not be resolved.")
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"OSTrack checkpoint not found: {checkpoint_file}")

    padding_candidates = _normalize_ostrack_padding_candidates(
        candidate_paddings_px or DEFAULT_OSTRACK_BBOX_PADDING_CANDIDATES
    )
    samples = _build_ostrack_bbox_probe_samples()
    predicted_bboxes_xywh = _run_ostrack_bbox_probe_subprocess(
        repo_path=resolved_repo_path,
        python_executable=resolved_python,
        checkpoint_path=checkpoint_file,
        config_name=config_name,
        samples=samples,
        candidate_paddings_px=padding_candidates,
        timeout_sec=timeout_sec,
    )

    aggregate_scores: dict[int, list[float]] = {padding_px: [] for padding_px in padding_candidates}
    sample_results: list[OstrackInitialBBoxSampleResult] = []
    for sample_index, sample in enumerate(samples):
        padding_results: list[OstrackInitialBBoxPaddingResult] = []
        for padding_index, padding_px in enumerate(padding_candidates):
            init_bbox = _expand_bbox_xywh(
                sample.gt_bboxes_xywh[0],
                padding_px=int(padding_px),
                image_shape=sample.frames[0].shape[:2],
            )
            metrics = _evaluate_ostrack_probe_run(
                gt_bboxes_xywh=sample.gt_bboxes_xywh,
                predicted_bboxes_xywh=predicted_bboxes_xywh[sample_index, padding_index],
                padding_px=int(padding_px),
                init_bbox_xywh=init_bbox,
            )
            aggregate_scores[int(padding_px)].append(metrics.aggregate_score)
            padding_results.append(metrics)

        sample_results.append(
            OstrackInitialBBoxSampleResult(
                sample_name=sample.name,
                frame_shape=tuple(int(v) for v in sample.frames[0].shape),
                frame_count=int(sample.frames.shape[0]),
                reference_init_bbox_xywh=tuple(int(v) for v in sample.gt_bboxes_xywh[0].tolist()),
                padding_results=padding_results,
            )
        )

    aggregate_mean_scores = {
        padding_px: float(np.mean(scores)) if scores else float("-inf")
        for padding_px, scores in aggregate_scores.items()
    }
    recommended_padding_px = padding_candidates[0]
    best_score = float("-inf")
    for padding_px in padding_candidates:
        score = aggregate_mean_scores.get(int(padding_px), float("-inf"))
        if score > best_score:
            best_score = score
            recommended_padding_px = int(padding_px)

    return OstrackInitialBBoxProbeResult(
        config_name=config_name,
        checkpoint_name=checkpoint_file.name,
        samples=sample_results,
        candidate_paddings_px=[int(v) for v in padding_candidates],
        recommended_padding_px=int(recommended_padding_px),
        aggregate_mean_scores=aggregate_mean_scores,
        notes=[
            "Probe korzysta z syntetycznych sekwencji malych plam LEED z dryfem, zmiana skali, dystraktorem i niskim kontrastem.",
            "Score preferuje wysoka srednia i minimalna IoU po inicjalizacji, ale uwzglednia tez degradacje pierwszego bboxa przez padding.",
            "To jest sygnal startowy dla MVP; wynik trzeba zweryfikowac na realnych sekwencjach LEED.",
        ],
    )


def format_ostrack_initial_bbox_probe_report(result: OstrackInitialBBoxProbeResult) -> str:
    score_summary = ", ".join(
        f"{padding}px={score:.4f}"
        for padding, score in sorted(result.aggregate_mean_scores.items(), key=lambda item: item[0])
    )
    lines = [
        "OSTrack initial bbox padding probe",
        f"- Recommended padding px: {result.recommended_padding_px}",
        f"- Config: {result.config_name}",
        f"- Checkpoint: {result.checkpoint_name}",
        f"- Candidate paddings px: {', '.join(str(v) for v in result.candidate_paddings_px)}",
        f"- Aggregate mean scores: {score_summary}",
    ]
    for sample in result.samples:
        lines.append(f"- Sample: {sample.sample_name}")
        lines.append(f"  - Frame shape: {sample.frame_shape}")
        lines.append(f"  - Frame count: {sample.frame_count}")
        lines.append(f"  - Reference init bbox: {sample.reference_init_bbox_xywh}")
        for padding_result in sample.padding_results:
            lines.append(
                "  - Padding "
                f"{padding_result.padding_px}px: "
                f"init IoU={padding_result.init_bbox_iou:.4f}, "
                f"mean IoU={padding_result.mean_tracking_iou:.4f}, "
                f"min IoU={padding_result.min_tracking_iou:.4f}, "
                f"last IoU={padding_result.last_tracking_iou:.4f}, "
                f"mean center err={padding_result.mean_center_error_px:.3f}px, "
                f"max center err={padding_result.max_center_error_px:.3f}px, "
                f"aggregate={padding_result.aggregate_score:.4f}"
            )
    if result.notes:
        lines.append("- Notes:")
        lines.extend(f"  - {note}" for note in result.notes)
    return "\n".join(lines)


def run_ostrack_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: OstrackRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
) -> OstrackOfflineResult:
    """
    Run offline OSTrack on a sequence of frames and sparse query boxes.

    Args:
        frames: [T, H, W] grayscale or [T, H, W, C] frames.
        query_points_tyx: [Q, 3] points in [t, y, x] format.
        settings: OSTrack runtime settings including repo/checkpoint paths.
        query_boxes_xyxy: Optional [Q, 4] seed boxes in original coordinates.

    Returns:
        OSTrack representative points, visibility, tracked boxes and metadata.
    """
    if frames is None or np.asarray(frames).size == 0:
        raise ValueError("No frames provided for OSTrack inference.")
    frames_np = np.asarray(frames)
    if frames_np.ndim not in (3, 4):
        raise ValueError(f"Expected frames with ndim 3 or 4, got shape {frames_np.shape}.")

    query_np = np.asarray(query_points_tyx, dtype=np.float32)
    if query_np.ndim != 2 or query_np.shape[1] != 3:
        raise ValueError("query_points_tyx must have shape [num_queries, 3] in [t, y, x] format.")
    if query_np.shape[0] == 0:
        raise ValueError("No query points provided for OSTrack inference.")

    backend_mode = str(settings.backend_mode or "subprocess").strip().lower()
    if backend_mode == "auto":
        backend_mode = "subprocess"
    if backend_mode == "direct":
        return _run_ostrack_direct_offline(
            frames_np,
            query_np,
            settings,
            query_boxes_xyxy=query_boxes_xyxy,
        )
    if backend_mode != "subprocess":
        raise ValueError(
            f"Unsupported OSTrack backend mode: {settings.backend_mode}. "
            "Expected `subprocess` or `direct`."
        )
    return _run_ostrack_subprocess_offline(
        frames_np,
        query_np,
        settings,
        query_boxes_xyxy=query_boxes_xyxy,
    )


def run_ostrack_backend_smoke_test(
    *,
    repo_path: str | Path,
    checkpoint_path: str | Path,
    python_executable: str | Path | None = None,
    config_name: str = DEFAULT_OSTRACK_CONFIG_NAME,
    device: str = "auto",
    backend_mode: str = "subprocess",
    representative_point_mode: str = "bbox_center",
    initial_bbox_padding_px: float = 0.0,
    timeout_sec: float = DEFAULT_OSTRACK_SUBPROCESS_TIMEOUT_SEC,
) -> OstrackBackendSmokeTestResult:
    sample = _build_ostrack_bbox_probe_samples()[0]
    frames = sample.frames
    first_bbox = sample.gt_bboxes_xywh[0]
    query_points_tyx = np.asarray(
        [[0.0, first_bbox[1] + first_bbox[3] * 0.5, first_bbox[0] + first_bbox[2] * 0.5]],
        dtype=np.float32,
    )
    query_boxes_xyxy = np.asarray(
        [[first_bbox[0], first_bbox[1], first_bbox[0] + first_bbox[2], first_bbox[1] + first_bbox[3]]],
        dtype=np.float32,
    )
    result = run_ostrack_offline(
        frames,
        query_points_tyx,
        OstrackRuntimeSettings(
            checkpoint_path=checkpoint_path,
            repo_path=repo_path,
            python_executable=python_executable,
            backend_mode=backend_mode,
            model_variant=config_name,
            device=device,
            representative_point_mode=representative_point_mode,
            initial_bbox_padding_px=initial_bbox_padding_px,
            subprocess_timeout_sec=timeout_sec,
        ),
        query_boxes_xyxy=query_boxes_xyxy,
    )

    visible_indexes = np.flatnonzero(result.visible_mask[0])
    if visible_indexes.size == 0:
        first_point = (float("nan"), float("nan"))
        last_point = (float("nan"), float("nan"))
        first_bbox_tuple = (float("nan"), float("nan"), float("nan"), float("nan"))
        last_bbox_tuple = (float("nan"), float("nan"), float("nan"), float("nan"))
    else:
        first_index = int(visible_indexes[0])
        last_index = int(visible_indexes[-1])
        first_point = tuple(float(v) for v in result.tracks_xy[0, first_index].tolist())
        last_point = tuple(float(v) for v in result.tracks_xy[0, last_index].tolist())
        first_bbox_tuple = tuple(float(v) for v in result.tracked_bboxes_xywh[0, first_index].tolist())
        last_bbox_tuple = tuple(float(v) for v in result.tracked_bboxes_xywh[0, last_index].tolist())

    return OstrackBackendSmokeTestResult(
        checkpoint_name=result.checkpoint_name,
        model_variant=result.model_variant,
        backend_mode=str(backend_mode),
        representative_point_mode=result.representative_point_mode,
        initial_bbox_padding_px=float(result.initial_bbox_padding_px),
        tracks_shape=tuple(int(v) for v in result.tracks_xy.shape),
        visible_shape=tuple(int(v) for v in result.visible_mask.shape),
        bboxes_shape=tuple(int(v) for v in result.tracked_bboxes_xywh.shape),
        visible_count=int(np.count_nonzero(result.visible_mask)),
        first_track_point_xy=first_point,
        last_visible_point_xy=last_point,
        first_bbox_xywh=first_bbox_tuple,
        last_bbox_xywh=last_bbox_tuple,
    )


def format_ostrack_backend_smoke_test_report(result: OstrackBackendSmokeTestResult) -> str:
    return "\n".join(
        [
            "OSTrack backend smoke test",
            f"- checkpoint: {result.checkpoint_name}",
            f"- backend mode: {result.backend_mode}",
            f"- model variant: {result.model_variant}",
            f"- representative point mode: {result.representative_point_mode}",
            f"- initial bbox padding px: {result.initial_bbox_padding_px:g}",
            f"- tracks shape: {result.tracks_shape}",
            f"- visible mask shape: {result.visible_shape}",
            f"- bboxes shape: {result.bboxes_shape}",
            f"- visible count: {result.visible_count}",
            f"- first track point: {list(result.first_track_point_xy)}",
            f"- last visible point: {list(result.last_visible_point_xy)}",
            f"- first bbox xywh: {list(result.first_bbox_xywh)}",
            f"- last bbox xywh: {list(result.last_bbox_xywh)}",
        ]
    )


def _run_ostrack_direct_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: OstrackRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
) -> OstrackOfflineResult:
    resolved_repo_path = _resolve_ostrack_repo_path(settings.repo_path)
    if resolved_repo_path is None:
        raise FileNotFoundError(
            "Could not resolve the OSTrack repo. Pass repo_path pointing at the local `ostrack` clone."
        )

    checkpoint_file = _resolve_existing_ostrack_checkpoint_file(settings.checkpoint_path)
    model_variant = str(settings.model_variant or DEFAULT_OSTRACK_CONFIG_NAME)
    representative_point_mode = _normalize_ostrack_representative_point_mode(
        settings.representative_point_mode
    )

    prepared_frames, original_resolution = _prepare_frames_rgb(frames)
    inference_resolution = _resolve_ostrack_inference_resolution(
        settings.inference_resolution,
        original_resolution,
    )
    resized_frames = _resize_video_numpy(prepared_frames, inference_resolution)
    scaled_query_points = _scale_query_points_tyx(
        query_points_tyx,
        source_resolution=original_resolution,
        target_resolution=inference_resolution,
    )
    scaled_query_boxes_xyxy = (
        _scale_boxes_xyxy(
            np.asarray(query_boxes_xyxy, dtype=np.float32),
            source_resolution=original_resolution,
            target_resolution=inference_resolution,
        )
        if query_boxes_xyxy is not None
        else None
    )
    seed_frames = _extract_query_seed_frames(
        scaled_query_points,
        total_frames=int(resized_frames.shape[0]),
    )
    seed_boxes_xywh = _build_ostrack_seed_boxes_xywh(
        scaled_query_points,
        resolution=inference_resolution,
        initial_bbox_padding_px=float(settings.initial_bbox_padding_px),
        provided_boxes_xyxy=scaled_query_boxes_xyxy,
    )

    num_queries = int(scaled_query_points.shape[0])
    num_frames = int(resized_frames.shape[0])
    tracks_xy_resized = np.full((num_queries, num_frames, 2), np.nan, dtype=np.float32)
    visible_mask = np.zeros((num_queries, num_frames), dtype=bool)
    tracked_bboxes_xywh_resized = np.full((num_queries, num_frames, 4), np.nan, dtype=np.float32)
    bbox_scores = np.zeros((num_queries, num_frames), dtype=np.float32)

    with _ostrack_import_context(resolved_repo_path):
        torch_module = importlib.import_module("torch")
        resolved_device = _resolve_ostrack_torch_device(torch_module, settings.device)
        if getattr(resolved_device, "type", str(resolved_device)) != "cuda":
            raise RuntimeError(
                "OSTrack upstream tracker requires CUDA in this integration because it calls `.cuda()` internally."
            )

        params = _build_ostrack_tracker_params(
            repo_path=resolved_repo_path,
            checkpoint_path=checkpoint_file,
            config_name=model_variant,
        )
        tracker_class_module = importlib.import_module("lib.test.tracker.ostrack")
        OSTrack = getattr(tracker_class_module, "OSTrack")
        tracker = OSTrack(params, "trackleed")

        for query_index in range(num_queries):
            seed_frame = int(seed_frames[query_index])
            seed_bbox_xywh = tuple(float(v) for v in seed_boxes_xywh[query_index].tolist())

            with torch_module.no_grad():
                tracker.initialize(
                    resized_frames[seed_frame],
                    {"init_bbox": list(seed_bbox_xywh)},
                )

            _record_ostrack_frame_output(
                tracked_bboxes_xywh=tracked_bboxes_xywh_resized,
                tracks_xy=tracks_xy_resized,
                visible_mask=visible_mask,
                bbox_scores=bbox_scores,
                query_index=query_index,
                frame_index=seed_frame,
                bbox_xywh=seed_bbox_xywh,
                score=1.0,
                frame=resized_frames[seed_frame],
                frame_resolution=inference_resolution,
                representative_point_mode=representative_point_mode,
            )

            for frame_index in range(seed_frame + 1, num_frames):
                with torch_module.no_grad():
                    tracker_output = tracker.track(resized_frames[frame_index], {}) or {}
                bbox_xywh = _normalize_optional_bbox_xywh(tracker_output.get("target_bbox"))
                _record_ostrack_frame_output(
                    tracked_bboxes_xywh=tracked_bboxes_xywh_resized,
                    tracks_xy=tracks_xy_resized,
                    visible_mask=visible_mask,
                    bbox_scores=bbox_scores,
                    query_index=query_index,
                    frame_index=frame_index,
                    bbox_xywh=bbox_xywh,
                    score=1.0 if bbox_xywh is not None else 0.0,
                    frame=resized_frames[frame_index],
                    frame_resolution=inference_resolution,
                    representative_point_mode=representative_point_mode,
                )

    tracked_bboxes_xywh = _scale_boxes_xywh(
        tracked_bboxes_xywh_resized,
        source_resolution=inference_resolution,
        target_resolution=original_resolution,
    )
    tracked_bboxes_xyxy = _boxes_xywh_to_xyxy(tracked_bboxes_xywh)
    tracks_xy = _scale_tracks_xy(
        tracks_xy_resized,
        source_resolution=inference_resolution,
        target_resolution=original_resolution,
    )

    return OstrackOfflineResult(
        tracks_xy=tracks_xy,
        visible_mask=visible_mask,
        tracked_bboxes_xywh=tracked_bboxes_xywh,
        tracked_bboxes_xyxy=tracked_bboxes_xyxy,
        bbox_scores=bbox_scores,
        inference_resolution=inference_resolution,
        original_resolution=original_resolution,
        checkpoint_name=checkpoint_file.name,
        model_variant=model_variant,
        representative_point_mode=representative_point_mode,
        initial_bbox_padding_px=float(settings.initial_bbox_padding_px),
    )


def _run_ostrack_subprocess_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: OstrackRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
) -> OstrackOfflineResult:
    python_executable = _resolve_ostrack_python_executable(settings.python_executable)
    if python_executable is None:
        raise FileNotFoundError("OSTrack Python executable could not be resolved.")
    script_path = (Path(__file__).resolve().parents[2] / DEFAULT_OSTRACK_SUBPROCESS_SCRIPT).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"OSTrack subprocess script not found: {script_path}")

    with TemporaryDirectory(prefix="trackleed_ostrack_") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "ostrack_input.npz"
        output_path = temp_root / "ostrack_output.npz"
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
            "--model-variant",
            str(settings.model_variant),
            "--representative-point-mode",
            str(settings.representative_point_mode),
            "--initial-bbox-padding-px",
            str(float(settings.initial_bbox_padding_px)),
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
            command.extend(
                [
                    "--height",
                    str(int(resolution[0])),
                    "--width",
                    str(int(resolution[1])),
                ]
            )

        subprocess_timeout_sec = float(settings.subprocess_timeout_sec)
        subprocess_cwd = str(Path(__file__).resolve().parents[2])
        try:
            completed = subprocess.run(
                command,
                cwd=subprocess_cwd,
                capture_output=True,
                text=True,
                check=False,
                timeout=subprocess_timeout_sec,
                env=_build_windows_conda_subprocess_env(python_executable, settings.repo_path),
            )
        except subprocess.TimeoutExpired as error:
            payload = _build_ostrack_subprocess_failure_payload(
                command=command,
                timeout_sec=subprocess_timeout_sec,
                stdout=error.stdout,
                stderr=error.stderr,
            )
            raise RuntimeError(
                "OSTrack subprocess inference timed out:\n"
                + json.dumps(payload, indent=2, ensure_ascii=False)
            ) from error
        if completed.returncode != 0:
            payload = _build_ostrack_subprocess_failure_payload(
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
            raise RuntimeError(
                "OSTrack subprocess inference failed:\n"
                + json.dumps(payload, indent=2, ensure_ascii=False)
            )
        if not output_path.exists():
            payload = _build_ostrack_subprocess_failure_payload(
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                expected_output=str(output_path),
            )
            raise RuntimeError(
                "OSTrack subprocess did not create output:\n"
                + json.dumps(payload, indent=2, ensure_ascii=False)
            )

        with np.load(output_path, allow_pickle=False) as data:
            return OstrackOfflineResult(
                tracks_xy=np.asarray(data["tracks_xy"], dtype=np.float32),
                visible_mask=np.asarray(data["visible_mask"], dtype=bool),
                tracked_bboxes_xywh=np.asarray(data["tracked_bboxes_xywh"], dtype=np.float32),
                tracked_bboxes_xyxy=np.asarray(data["tracked_bboxes_xyxy"], dtype=np.float32),
                bbox_scores=np.asarray(data["bbox_scores"], dtype=np.float32),
                inference_resolution=tuple(int(v) for v in data["inference_resolution"].tolist()),
                original_resolution=tuple(int(v) for v in data["original_resolution"].tolist()),
                checkpoint_name=str(data["checkpoint_name"].tolist()),
                model_variant=str(data["model_variant"].tolist()),
                representative_point_mode=str(data["representative_point_mode"].tolist()),
                initial_bbox_padding_px=float(data["initial_bbox_padding_px"].tolist()),
            )


def _normalize_subprocess_output(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace").strip()
    return str(value).strip()


def _build_ostrack_subprocess_failure_payload(
    *,
    command: list[str],
    returncode: int | None = None,
    timeout_sec: float | None = None,
    stdout=None,
    stderr=None,
    expected_output: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "command": command,
        "stdout": _normalize_subprocess_output(stdout),
        "stderr": _normalize_subprocess_output(stderr),
    }
    if returncode is not None:
        payload["returncode"] = int(returncode)
    if timeout_sec is not None:
        payload["timeout_sec"] = float(timeout_sec)
    if expected_output is not None:
        payload["expected_output"] = str(expected_output)
    return payload


def _run_ostrack_bbox_probe_subprocess(
    *,
    repo_path: Path,
    python_executable: Path,
    checkpoint_path: Path,
    config_name: str,
    samples: list[_SyntheticOstrackProbeSample],
    candidate_paddings_px: list[int],
    timeout_sec: float,
) -> np.ndarray:
    with TemporaryDirectory(prefix="trackleed_ostrack_probe_") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "ostrack_probe_input.npz"
        output_path = temp_root / "ostrack_probe_output.npz"
        np.savez_compressed(
            input_path,
            frames=np.asarray([sample.frames for sample in samples], dtype=np.uint8),
            gt_bboxes_xywh=np.asarray(
                [sample.gt_bboxes_xywh for sample in samples],
                dtype=np.float32,
            ),
            paddings_px=np.asarray(candidate_paddings_px, dtype=np.int32),
        )

        env = _build_windows_conda_subprocess_env(python_executable, repo_path)
        env["TRACKLEED_OSTRACK_PROBE_INPUT"] = str(input_path)
        env["TRACKLEED_OSTRACK_PROBE_OUTPUT"] = str(output_path)
        env["TRACKLEED_OSTRACK_CHECKPOINT"] = str(checkpoint_path)
        env["TRACKLEED_OSTRACK_CONFIG"] = str(config_name)

        command = [str(python_executable), "-c", _OSTRACK_BBOX_PROBE_CODE]
        completed = subprocess.run(
            command,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
            timeout=float(timeout_sec),
            env=env,
        )
        if completed.returncode != 0:
            payload = {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
            raise RuntimeError(
                "OSTrack bbox padding probe failed:\n"
                + json.dumps(payload, indent=2, ensure_ascii=False)
            )
        if not output_path.exists():
            raise FileNotFoundError(f"OSTrack bbox probe did not create output: {output_path}")

        with np.load(output_path, allow_pickle=False) as data:
            predicted = np.asarray(data["predicted_bboxes_xywh"], dtype=np.float32)

    expected_shape = (
        len(samples),
        len(candidate_paddings_px),
        int(samples[0].frames.shape[0]),
        4,
    )
    if predicted.shape != expected_shape:
        raise RuntimeError(
            "OSTrack bbox probe returned invalid shape: "
            f"{predicted.shape}, expected {expected_shape}."
        )
    return predicted


def _resolve_ostrack_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidates = [Path(repo_path)] if repo_path else list(DEFAULT_OSTRACK_REPO_CANDIDATES)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "experiments" / "ostrack").exists() and (
            resolved / "lib" / "test" / "tracker" / "ostrack.py"
        ).exists():
            return resolved
    return None


def _resolve_ostrack_python_executable(python_executable: str | Path | None) -> Optional[Path]:
    if python_executable:
        return Path(python_executable).expanduser().resolve()
    return Path(sys.executable).resolve()


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoints: list[Path],
    models_dir: Path,
    upstream_checkpoint_path: Optional[Path],
    config_name: str,
) -> Optional[Path]:
    if checkpoint_path:
        candidate = Path(checkpoint_path).expanduser()
        if not candidate.is_absolute():
            candidate = models_dir / candidate
        return candidate.resolve()
    selected = select_default_ostrack_checkpoint(checkpoints, config_name)
    if selected is not None:
        return selected
    if upstream_checkpoint_path is not None:
        return upstream_checkpoint_path
    return None


def _run_ostrack_import_probe(
    python_executable: Path,
    repo_path: Optional[Path],
    *,
    timeout_sec: float,
) -> OstrackImportProbeResult:
    command = [str(python_executable), "-c", _OSTRACK_IMPORT_PROBE_CODE]
    completed = subprocess.run(
        command,
        cwd=str(repo_path) if repo_path is not None else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_sec),
        env=_build_windows_conda_subprocess_env(python_executable, repo_path),
    )
    payload = _parse_import_probe_stdout(completed.stdout)
    return OstrackImportProbeResult(
        command=command,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        payload=payload,
    )


def _apply_import_probe_result(
    status: OstrackEnvironmentStatus,
    probe: OstrackImportProbeResult,
) -> None:
    status.import_probe_returncode = probe.returncode
    status.import_probe_stdout = probe.stdout
    status.import_probe_stderr = probe.stderr
    payload = probe.payload
    if probe.returncode != 0:
        status.errors.append(
            "OSTrack import probe failed with return code "
            f"{probe.returncode}: {probe.stderr.strip() or probe.stdout.strip()}"
        )
    if not payload:
        status.errors.append("OSTrack import probe did not return parseable diagnostics JSON.")
        return

    status.torch_importable = bool(payload.get("torch_importable"))
    status.torch_version = payload.get("torch_version")
    status.torch_cuda_version = payload.get("torch_cuda_version")
    status.cuda_available = bool(payload.get("cuda_available"))
    status.torchvision_importable = bool(payload.get("torchvision_importable"))
    status.cv2_importable = bool(payload.get("cv2_importable"))
    status.numpy_importable = bool(payload.get("numpy_importable"))
    status.yaml_importable = bool(payload.get("yaml_importable"))
    status.easydict_importable = bool(payload.get("easydict_importable"))
    status.timm_importable = bool(payload.get("timm_importable"))
    status.lmdb_importable = bool(payload.get("lmdb_importable"))
    status.jpeg4py_importable = bool(payload.get("jpeg4py_importable"))
    status.ostrack_parameter_importable = bool(payload.get("ostrack_parameter_importable"))
    status.ostrack_tracker_importable = bool(payload.get("ostrack_tracker_importable"))
    status.ostrack_parameter_module_path = payload.get("ostrack_parameter_module_path")
    status.ostrack_tracker_module_path = payload.get("ostrack_tracker_module_path")

    for key, value in sorted(payload.items()):
        if key.endswith("_error") and value:
            status.errors.append(f"{key[:-6]} import failed: {value}")


def _parse_import_probe_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(str(stdout or "").splitlines()):
        if line.startswith(_IMPORT_PROBE_PREFIX):
            payload = line[len(_IMPORT_PROBE_PREFIX):]
            return json.loads(payload)
    return {}


def _build_windows_conda_subprocess_env(
    python_executable: Path,
    repo_path: Optional[Path] = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if repo_path is not None:
        env["TRACKLEED_OSTRACK_REPO"] = str(repo_path)
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


def _extract_first_match(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text)
    if match is None:
        return None
    return match.group(1).strip()


def _extract_float(pattern: str, text: str) -> Optional[float]:
    value = _extract_first_match(pattern, text)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _normalize_ostrack_padding_candidates(
    values: list[int] | tuple[int, ...],
) -> list[int]:
    candidates: list[int] = []
    for value in values:
        padding_px = int(value)
        if padding_px < 0:
            raise ValueError("OSTrack bbox padding candidates must be non-negative.")
        if padding_px not in candidates:
            candidates.append(padding_px)
    if not candidates:
        raise ValueError("At least one OSTrack bbox padding candidate is required.")
    return candidates


def _build_ostrack_bbox_probe_samples() -> list[_SyntheticOstrackProbeSample]:
    return [
        _synthesize_ostrack_probe_sequence(
            name="tiny_drift",
            gt_bboxes_xywh=[
                (104, 78, 12, 12),
                (105, 79, 12, 12),
                (107, 79, 12, 12),
                (108, 80, 12, 12),
            ],
            color_mode="amber",
            distractors=[
                (126, 80, 13, 13, (235, 190, 72)),
                (64, 118, 18, 8, (80, 120, 220)),
            ],
        ),
        _synthesize_ostrack_probe_sequence(
            name="small_scale_change",
            gt_bboxes_xywh=[
                (84, 64, 20, 24),
                (85, 65, 22, 25),
                (87, 66, 24, 27),
                (89, 67, 25, 28),
            ],
            color_mode="cyan",
            distractors=[
                (148, 58, 22, 24, (70, 180, 210)),
                (36, 96, 26, 10, (180, 70, 160)),
            ],
        ),
        _synthesize_ostrack_probe_sequence(
            name="close_distractor",
            gt_bboxes_xywh=[
                (112, 82, 16, 16),
                (113, 83, 16, 16),
                (115, 84, 16, 16),
                (116, 85, 16, 16),
            ],
            color_mode="amber",
            distractors=[
                (134, 84, 16, 16, (244, 205, 76)),
                (152, 108, 20, 12, (70, 150, 210)),
            ],
        ),
        _synthesize_ostrack_probe_sequence(
            name="low_contrast",
            gt_bboxes_xywh=[
                (76, 98, 18, 18),
                (77, 98, 18, 18),
                (79, 99, 18, 18),
                (80, 99, 18, 18),
            ],
            color_mode="low_contrast",
            distractors=[
                (100, 104, 18, 18, (80, 87, 96)),
                (154, 52, 28, 10, (112, 94, 130)),
            ],
        ),
        _synthesize_ostrack_probe_sequence(
            name="elongated_jitter",
            gt_bboxes_xywh=[
                (116, 88, 14, 26),
                (115, 87, 14, 26),
                (117, 88, 15, 27),
                (118, 89, 15, 27),
            ],
            color_mode="magenta",
            distractors=[
                (140, 90, 15, 26, (180, 90, 210)),
                (82, 36, 30, 12, (60, 210, 120)),
            ],
        ),
    ]


def _synthesize_ostrack_probe_sequence(
    *,
    name: str,
    gt_bboxes_xywh: list[tuple[int, int, int, int]],
    color_mode: str,
    distractors: list[tuple[int, int, int, int, tuple[int, int, int]]],
) -> _SyntheticOstrackProbeSample:
    height = 192
    width = 256
    frames: list[np.ndarray] = []

    for frame_index, bbox in enumerate(gt_bboxes_xywh):
        yy, xx = np.indices((height, width))
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = np.clip(18 + 0.06 * xx + 0.03 * yy + frame_index * 2.0, 0, 255).astype(
            np.uint8
        )
        frame[..., 1] = np.clip(20 + 0.05 * yy + 0.02 * xx, 0, 255).astype(np.uint8)
        frame[..., 2] = np.clip(14 + 0.08 * xx + 0.01 * yy + frame_index * 1.0, 0, 255).astype(
            np.uint8
        )
        stripe = ((xx // 16 + yy // 12 + frame_index) % 2) * 8
        frame[..., 1] = np.clip(
            frame[..., 1].astype(np.int16) + stripe.astype(np.int16),
            0,
            255,
        ).astype(np.uint8)

        for dx, dy, dw, dh, color in distractors:
            x0 = int(np.clip(dx, 0, width - 1))
            y0 = int(np.clip(dy, 0, height - 1))
            x1 = int(np.clip(dx + dw, x0 + 1, width))
            y1 = int(np.clip(dy + dh, y0 + 1, height))
            frame[y0:y1, x0:x1] = np.asarray(color, dtype=np.uint8)

        x, y, w, h = bbox
        frame[y : y + h, x : x + w] = _build_ostrack_target_patch(
            h,
            w,
            color_mode=color_mode,
        )
        frames.append(frame)

    return _SyntheticOstrackProbeSample(
        name=name,
        frames=np.asarray(frames, dtype=np.uint8),
        gt_bboxes_xywh=np.asarray(gt_bboxes_xywh, dtype=np.float32),
    )


def _build_ostrack_target_patch(
    height: int,
    width: int,
    *,
    color_mode: str,
) -> np.ndarray:
    patch_y = np.broadcast_to(
        np.linspace(0.0, 1.0, int(height), dtype=np.float32)[:, None],
        (int(height), int(width)),
    )
    patch_x = np.broadcast_to(
        np.linspace(0.0, 1.0, int(width), dtype=np.float32)[None, :],
        (int(height), int(width)),
    )

    if color_mode == "amber":
        patch = np.stack(
            [
                210 + 25 * patch_x,
                150 + 75 * patch_y,
                40 + 25 * (1.0 - patch_x),
            ],
            axis=-1,
        )
    elif color_mode == "cyan":
        patch = np.stack(
            [
                40 + 20 * patch_y,
                165 + 60 * patch_x,
                190 + 40 * (1.0 - patch_y),
            ],
            axis=-1,
        )
    elif color_mode == "low_contrast":
        patch = np.stack(
            [
                82 + 18 * patch_x,
                88 + 16 * patch_y,
                98 + 18 * (1.0 - patch_x),
            ],
            axis=-1,
        )
    else:
        patch = np.stack(
            [
                180 + 45 * patch_x,
                40 + 25 * patch_y,
                170 + 50 * (1.0 - patch_x),
            ],
            axis=-1,
        )

    patch = np.clip(patch, 0.0, 255.0).astype(np.uint8)
    if color_mode != "low_contrast":
        patch[
            max(1, height // 5) : max(2, height // 5 + height // 6),
            max(1, width // 5) : max(2, width - width // 4),
        ] = np.array([250, 240, 92], dtype=np.uint8)
        patch[
            max(1, height // 2) : max(2, height - height // 6),
            max(1, width // 3) : max(2, width // 3 + max(2, width // 6)),
        ] = np.array([25, 25, 25], dtype=np.uint8)
    return patch


def _expand_bbox_xywh(
    bbox_xywh: np.ndarray | tuple[float, float, float, float],
    *,
    padding_px: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    height, width = image_shape
    x0 = max(0.0, x - float(padding_px))
    y0 = max(0.0, y - float(padding_px))
    x1 = min(float(width), x + w + float(padding_px))
    y1 = min(float(height), y + h + float(padding_px))
    new_w = max(2.0, x1 - x0)
    new_h = max(2.0, y1 - y0)
    return (
        int(round(x0)),
        int(round(y0)),
        int(round(new_w)),
        int(round(new_h)),
    )


def _evaluate_ostrack_probe_run(
    *,
    gt_bboxes_xywh: np.ndarray,
    predicted_bboxes_xywh: np.ndarray,
    padding_px: int,
    init_bbox_xywh: tuple[int, int, int, int],
) -> OstrackInitialBBoxPaddingResult:
    tracking_ious = [
        _bbox_iou_xywh(predicted_bboxes_xywh[index], gt_bboxes_xywh[index])
        for index in range(1, int(len(gt_bboxes_xywh)))
    ]
    center_errors = [
        _bbox_center_error_px(predicted_bboxes_xywh[index], gt_bboxes_xywh[index])
        for index in range(1, int(len(gt_bboxes_xywh)))
    ]
    init_bbox_iou = _bbox_iou_xywh(np.asarray(init_bbox_xywh, dtype=np.float32), gt_bboxes_xywh[0])
    mean_tracking_iou = float(np.mean(tracking_ious)) if tracking_ious else init_bbox_iou
    min_tracking_iou = float(np.min(tracking_ious)) if tracking_ious else init_bbox_iou
    last_tracking_iou = float(tracking_ious[-1]) if tracking_ious else init_bbox_iou
    mean_center_error_px = float(np.mean(center_errors)) if center_errors else 0.0
    max_center_error_px = float(np.max(center_errors)) if center_errors else 0.0
    aggregate_score = (
        0.65 * mean_tracking_iou
        + 0.20 * min_tracking_iou
        + 0.15 * init_bbox_iou
    )

    return OstrackInitialBBoxPaddingResult(
        padding_px=int(padding_px),
        init_bbox_xywh=tuple(int(v) for v in init_bbox_xywh),
        init_bbox_iou=float(init_bbox_iou),
        mean_tracking_iou=float(mean_tracking_iou),
        min_tracking_iou=float(min_tracking_iou),
        last_tracking_iou=float(last_tracking_iou),
        mean_center_error_px=float(mean_center_error_px),
        max_center_error_px=float(max_center_error_px),
        aggregate_score=float(aggregate_score),
    )


def _bbox_iou_xywh(
    bbox_a_xywh: np.ndarray | tuple[float, float, float, float],
    bbox_b_xywh: np.ndarray | tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = [float(v) for v in bbox_a_xywh]
    bx, by, bw, bh = [float(v) for v in bbox_b_xywh]
    ax1 = ax + max(0.0, aw)
    ay1 = ay + max(0.0, ah)
    bx1 = bx + max(0.0, bw)
    by1 = by + max(0.0, bh)

    inter_x0 = max(ax, bx)
    inter_y0 = max(ay, by)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def _bbox_center_error_px(
    bbox_a_xywh: np.ndarray | tuple[float, float, float, float],
    bbox_b_xywh: np.ndarray | tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = [float(v) for v in bbox_a_xywh]
    bx, by, bw, bh = [float(v) for v in bbox_b_xywh]
    acx = ax + aw * 0.5
    acy = ay + ah * 0.5
    bcx = bx + bw * 0.5
    bcy = by + bh * 0.5
    return float(np.hypot(acx - bcx, acy - bcy))


def _resolve_existing_ostrack_checkpoint_file(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"OSTrack checkpoint not found: {checkpoint}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"OSTrack checkpoint is not a file: {checkpoint}")
    return checkpoint


@contextmanager
def _ostrack_import_context(repo_path: Path) -> Iterator[None]:
    repo_text = str(repo_path)
    inserted = False
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
        inserted = True
    previous_repo = os.environ.get("TRACKLEED_OSTRACK_REPO")
    os.environ["TRACKLEED_OSTRACK_REPO"] = repo_text
    try:
        yield
    finally:
        if previous_repo is None:
            os.environ.pop("TRACKLEED_OSTRACK_REPO", None)
        else:
            os.environ["TRACKLEED_OSTRACK_REPO"] = previous_repo
        if inserted:
            try:
                sys.path.remove(repo_text)
            except ValueError:
                pass


def _build_ostrack_tracker_params(
    *,
    repo_path: Path,
    checkpoint_path: Path,
    config_name: str,
):
    config_module = importlib.import_module("lib.config.ostrack.config")
    tracker_params_module = importlib.import_module("lib.test.utils")
    cfg = getattr(config_module, "cfg")
    update_config_from_file = getattr(config_module, "update_config_from_file")
    TrackerParams = getattr(tracker_params_module, "TrackerParams")

    yaml_file = repo_path / "experiments" / "ostrack" / f"{config_name}.yaml"
    if not yaml_file.exists():
        raise FileNotFoundError(f"OSTrack config not found: {yaml_file}")
    update_config_from_file(str(yaml_file))

    params = TrackerParams()
    params.cfg = cfg
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = str(checkpoint_path)
    params.debug = 0
    params.save_all_boxes = False
    return params


def _resolve_ostrack_torch_device(torch_module, requested_device: str):
    device_name = str(requested_device or "auto").strip().lower()
    if device_name in {"", "auto", "gpu"}:
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if device_name in {"cuda", "cuda:0"} and not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    return torch_module.device(device_name)


def _normalize_ostrack_representative_point_mode(mode: str) -> str:
    normalized = str(mode or "bbox_center").strip().lower()
    if normalized in {"bbox_center", "center", "bbox-centre", "bbox_centre"}:
        return "bbox_center"
    if normalized in {"peak_in_bbox", "peak", "bbox_peak"}:
        return "peak_in_bbox"
    if normalized in {"gaussian_in_bbox", "gaussian", "bbox_gaussian", "2d_gauss"}:
        return "gaussian_in_bbox"
    raise ValueError(
        "Unsupported OSTrack representative point mode: "
        f"{mode}. Expected `bbox_center`, `peak_in_bbox`, or `gaussian_in_bbox`."
    )


def _resolve_ostrack_inference_resolution(
    target: int | tuple[int, int] | None,
    original_resolution: tuple[int, int],
) -> tuple[int, int]:
    if target is None:
        return (int(original_resolution[0]), int(original_resolution[1]))
    if isinstance(target, int):
        size = int(target)
        if size <= 0:
            raise ValueError(f"Invalid OSTrack inference resolution: {target}")
        return (size, size)
    height = int(target[0])
    width = int(target[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid OSTrack inference resolution: {(height, width)}")
    return (height, width)


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


def _resize_video_numpy(
    frames: np.ndarray,
    target_resolution: tuple[int, int],
) -> np.ndarray:
    target_h, target_w = int(target_resolution[0]), int(target_resolution[1])
    if frames.shape[1] == target_h and frames.shape[2] == target_w:
        return np.asarray(frames)
    cv2 = importlib.import_module("cv2")
    resized_frames = [
        cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        for frame in np.asarray(frames)
    ]
    return np.asarray(resized_frames, dtype=np.uint8)


def _scale_query_points_tyx(
    query_points_tyx: np.ndarray,
    *,
    source_resolution: tuple[int, int],
    target_resolution: tuple[int, int],
) -> np.ndarray:
    scaled = np.array(query_points_tyx, dtype=np.float32, copy=True)
    scale_y = float(target_resolution[0]) / float(max(1, source_resolution[0]))
    scale_x = float(target_resolution[1]) / float(max(1, source_resolution[1]))
    scaled[:, 1] *= scale_y
    scaled[:, 2] *= scale_x
    return scaled


def _scale_tracks_xy(
    tracks_xy: np.ndarray,
    *,
    source_resolution: tuple[int, int],
    target_resolution: tuple[int, int],
) -> np.ndarray:
    scaled = np.array(tracks_xy, dtype=np.float32, copy=True)
    scale_x = float(target_resolution[1]) / float(max(1, source_resolution[1]))
    scale_y = float(target_resolution[0]) / float(max(1, source_resolution[0]))
    scaled[..., 0] *= scale_x
    scaled[..., 1] *= scale_y
    return scaled


def _scale_boxes_xyxy(
    boxes_xyxy: np.ndarray,
    *,
    source_resolution: tuple[int, int],
    target_resolution: tuple[int, int],
) -> np.ndarray:
    scaled = np.array(boxes_xyxy, dtype=np.float32, copy=True)
    scale_x = float(target_resolution[1]) / float(max(1, source_resolution[1]))
    scale_y = float(target_resolution[0]) / float(max(1, source_resolution[0]))
    scaled[..., 0] *= scale_x
    scaled[..., 2] *= scale_x
    scaled[..., 1] *= scale_y
    scaled[..., 3] *= scale_y
    return scaled


def _scale_boxes_xywh(
    boxes_xywh: np.ndarray,
    *,
    source_resolution: tuple[int, int],
    target_resolution: tuple[int, int],
) -> np.ndarray:
    scaled = np.array(boxes_xywh, dtype=np.float32, copy=True)
    scale_x = float(target_resolution[1]) / float(max(1, source_resolution[1]))
    scale_y = float(target_resolution[0]) / float(max(1, source_resolution[0]))
    scaled[..., 0] *= scale_x
    scaled[..., 2] *= scale_x
    scaled[..., 1] *= scale_y
    scaled[..., 3] *= scale_y
    return scaled


def _extract_query_seed_frames(
    query_points_tyx: np.ndarray,
    *,
    total_frames: int,
) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0.")
    seed_frames = np.rint(query_points_tyx[:, 0]).astype(np.int32, copy=False)
    return np.clip(seed_frames, 0, max(0, total_frames - 1))


def _build_ostrack_seed_boxes_xywh(
    query_points_tyx: np.ndarray,
    *,
    resolution: tuple[int, int],
    initial_bbox_padding_px: float,
    provided_boxes_xyxy: np.ndarray | None = None,
) -> np.ndarray:
    num_queries = int(query_points_tyx.shape[0])
    boxes_xywh = np.zeros((num_queries, 4), dtype=np.float32)
    height, width = int(resolution[0]), int(resolution[1])
    padding = max(0.0, float(initial_bbox_padding_px))

    if provided_boxes_xyxy is not None:
        boxes_np = np.asarray(provided_boxes_xyxy, dtype=np.float32)
        if boxes_np.shape != (num_queries, 4):
            raise ValueError(
                f"query_boxes_xyxy must have shape {(num_queries, 4)}, got {boxes_np.shape}."
            )
        for index in range(num_queries):
            boxes_xywh[index] = _xyxy_to_clipped_xywh(
                boxes_np[index],
                image_shape=(height, width),
                padding_px=padding,
            )
        return boxes_xywh

    half_size = max(1.0, padding)
    for index in range(num_queries):
        x = float(query_points_tyx[index, 2])
        y = float(query_points_tyx[index, 1])
        boxes_xywh[index] = _xyxy_to_clipped_xywh(
            np.asarray([x - half_size, y - half_size, x + half_size, y + half_size], dtype=np.float32),
            image_shape=(height, width),
            padding_px=0.0,
        )
    return boxes_xywh


def _xyxy_to_clipped_xywh(
    box_xyxy: np.ndarray,
    *,
    image_shape: tuple[int, int],
    padding_px: float,
) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    x_min = min(x0, x1) - float(padding_px)
    y_min = min(y0, y1) - float(padding_px)
    x_max = max(x0, x1) + float(padding_px)
    y_max = max(y0, y1) + float(padding_px)
    x_min = float(np.clip(x_min, 0.0, float(width - 1 if width > 0 else 0)))
    y_min = float(np.clip(y_min, 0.0, float(height - 1 if height > 0 else 0)))
    x_max = float(np.clip(x_max, x_min + 1.0, float(width)))
    y_max = float(np.clip(y_max, y_min + 1.0, float(height)))
    return np.asarray(
        [x_min, y_min, max(1.0, x_max - x_min), max(1.0, y_max - y_min)],
        dtype=np.float32,
    )


def _normalize_optional_bbox_xywh(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 4 or not np.all(np.isfinite(arr)):
        return None
    if float(arr[2]) <= 0.0 or float(arr[3]) <= 0.0:
        return None
    if np.all(arr < 0.0):
        return None
    return tuple(float(v) for v in arr.tolist())


def _record_ostrack_frame_output(
    *,
    tracked_bboxes_xywh: np.ndarray,
    tracks_xy: np.ndarray,
    visible_mask: np.ndarray,
    bbox_scores: np.ndarray,
    query_index: int,
    frame_index: int,
    bbox_xywh: tuple[float, float, float, float] | None,
    score: float,
    frame: np.ndarray,
    frame_resolution: tuple[int, int],
    representative_point_mode: str,
) -> None:
    bbox_scores[query_index, frame_index] = float(score) if np.isfinite(score) else 0.0
    if bbox_xywh is None:
        return

    bbox_arr = np.asarray(bbox_xywh, dtype=np.float32)
    if not _bbox_xywh_is_valid(bbox_arr, frame_resolution):
        return

    tracked_bboxes_xywh[query_index, frame_index, :] = bbox_arr
    representative_point = _representative_point_from_bbox_xywh(
        bbox_arr,
        frame=frame,
        mode=representative_point_mode,
    )
    if representative_point is None:
        return

    tracks_xy[query_index, frame_index, 0] = float(representative_point[0])
    tracks_xy[query_index, frame_index, 1] = float(representative_point[1])
    visible_mask[query_index, frame_index] = True


def _bbox_xywh_is_valid(
    bbox_xywh: np.ndarray,
    frame_resolution: tuple[int, int],
) -> bool:
    if bbox_xywh.shape != (4,) or not np.all(np.isfinite(bbox_xywh)):
        return False
    x, y, w, h = [float(v) for v in bbox_xywh]
    if w <= 0.0 or h <= 0.0:
        return False
    cx = x + w * 0.5
    cy = y + h * 0.5
    return (
        0.0 <= cx <= float(max(0, frame_resolution[1] - 1))
        and 0.0 <= cy <= float(max(0, frame_resolution[0] - 1))
    )


def _representative_point_from_bbox_xywh(
    bbox_xywh: np.ndarray,
    *,
    frame: np.ndarray,
    mode: str,
) -> tuple[float, float] | None:
    normalized_mode = _normalize_ostrack_representative_point_mode(mode)
    if normalized_mode == "gaussian_in_bbox":
        gaussian_point = _gaussian_point_in_bbox_xywh(frame, bbox_xywh)
        if gaussian_point is not None:
            return gaussian_point
    if normalized_mode == "peak_in_bbox":
        peak_point = _peak_point_in_bbox_xywh(frame, bbox_xywh)
        if peak_point is not None:
            return peak_point
    if normalized_mode == "gaussian_in_bbox":
        peak_point = _peak_point_in_bbox_xywh(frame, bbox_xywh)
        if peak_point is not None:
            return peak_point
    x, y, w, h = [float(v) for v in bbox_xywh]
    return (x + w * 0.5, y + h * 0.5)


def _peak_point_in_bbox_xywh(
    frame: np.ndarray,
    bbox_xywh: np.ndarray,
) -> tuple[float, float] | None:
    frame_np = np.asarray(frame)
    if frame_np.ndim not in (2, 3) or frame_np.size == 0:
        return None

    x, y, w, h = [float(v) for v in np.asarray(bbox_xywh, dtype=np.float32).reshape(4)]
    if not np.all(np.isfinite([x, y, w, h])) or w <= 0.0 or h <= 0.0:
        return None

    frame_h, frame_w = int(frame_np.shape[0]), int(frame_np.shape[1])
    x0 = max(0, min(frame_w, int(np.floor(x))))
    y0 = max(0, min(frame_h, int(np.floor(y))))
    x1 = max(0, min(frame_w, int(np.ceil(x + w))))
    y1 = max(0, min(frame_h, int(np.ceil(y + h))))
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

    x, y, w, h = [float(v) for v in np.asarray(bbox_xywh, dtype=np.float32).reshape(4)]
    if not np.all(np.isfinite([x, y, w, h])) or w <= 0.0 or h <= 0.0:
        return None

    frame_h, frame_w = int(frame_np.shape[0]), int(frame_np.shape[1])
    x0 = max(0, min(frame_w, int(np.floor(x))))
    y0 = max(0, min(frame_h, int(np.floor(y))))
    x1 = max(0, min(frame_w, int(np.ceil(x + w))))
    y1 = max(0, min(frame_h, int(np.ceil(y + h))))
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


def _boxes_xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    boxes_xyxy = np.array(boxes_xywh, dtype=np.float32, copy=True)
    boxes_xyxy[..., 2] = boxes_xyxy[..., 0] + boxes_xyxy[..., 2]
    boxes_xyxy[..., 3] = boxes_xyxy[..., 1] + boxes_xyxy[..., 3]
    invalid = ~np.isfinite(boxes_xyxy[..., 0]) | ~np.isfinite(boxes_xyxy[..., 1])
    boxes_xyxy[invalid] = np.nan
    return boxes_xyxy


_OSTRACK_BBOX_PROBE_CODE = r"""
import os
import sys

import numpy as np

repo_path = os.environ.get("TRACKLEED_OSTRACK_REPO")
if repo_path and repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from lib.config.ostrack.config import cfg, update_config_from_file
from lib.test.tracker.ostrack import OSTrack
from lib.test.utils import TrackerParams


def expand_bbox_xywh(bbox_xywh, padding_px, image_shape):
    x, y, w, h = [float(v) for v in bbox_xywh]
    height, width = image_shape
    x0 = max(0.0, x - float(padding_px))
    y0 = max(0.0, y - float(padding_px))
    x1 = min(float(width), x + w + float(padding_px))
    y1 = min(float(height), y + h + float(padding_px))
    return [
        int(round(x0)),
        int(round(y0)),
        int(round(max(2.0, x1 - x0))),
        int(round(max(2.0, y1 - y0))),
    ]


input_path = os.environ["TRACKLEED_OSTRACK_PROBE_INPUT"]
output_path = os.environ["TRACKLEED_OSTRACK_PROBE_OUTPUT"]
checkpoint_path = os.environ["TRACKLEED_OSTRACK_CHECKPOINT"]
config_name = os.environ["TRACKLEED_OSTRACK_CONFIG"]

with np.load(input_path, allow_pickle=False) as data:
    frames = np.asarray(data["frames"], dtype=np.uint8)
    gt_bboxes_xywh = np.asarray(data["gt_bboxes_xywh"], dtype=np.float32)
    paddings_px = np.asarray(data["paddings_px"], dtype=np.int32)

yaml_file = os.path.join(repo_path, "experiments", "ostrack", config_name + ".yaml")
update_config_from_file(yaml_file)
params = TrackerParams()
params.cfg = cfg
params.template_factor = cfg.TEST.TEMPLATE_FACTOR
params.template_size = cfg.TEST.TEMPLATE_SIZE
params.search_factor = cfg.TEST.SEARCH_FACTOR
params.search_size = cfg.TEST.SEARCH_SIZE
params.checkpoint = checkpoint_path
params.debug = 0
params.save_all_boxes = False
tracker = OSTrack(params, "trackleed")

num_samples = int(frames.shape[0])
num_paddings = int(paddings_px.shape[0])
num_frames = int(frames.shape[1])
predicted = np.full((num_samples, num_paddings, num_frames, 4), np.nan, dtype=np.float32)

for sample_index in range(num_samples):
    for padding_index, padding_px in enumerate(paddings_px.tolist()):
        init_bbox = expand_bbox_xywh(
            gt_bboxes_xywh[sample_index, 0],
            int(padding_px),
            frames[sample_index, 0].shape[:2],
        )
        tracker.initialize(frames[sample_index, 0], {"init_bbox": [float(v) for v in init_bbox]})
        predicted[sample_index, padding_index, 0] = np.asarray(init_bbox, dtype=np.float32)
        for frame_index in range(1, num_frames):
            out = tracker.track(frames[sample_index, frame_index], {}) or {}
            predicted[sample_index, padding_index, frame_index] = np.asarray(
                out.get("target_bbox"),
                dtype=np.float32,
            ).reshape(4)

np.savez_compressed(output_path, predicted_bboxes_xywh=predicted)
"""


_OSTRACK_IMPORT_PROBE_CODE = r"""
import importlib
import json
import os
import sys

PREFIX = "__TRACKLEED_OSTRACK_IMPORT_PROBE__"
result = {"python_executable": sys.executable}
repo_path = os.environ.get("TRACKLEED_OSTRACK_REPO")
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

probe("torchvision", "torchvision")
probe("cv2", "cv2")
probe("numpy", "numpy")
probe("yaml", "yaml")
probe("easydict", "easydict")
probe("timm", "timm")
probe("lmdb", "lmdb")
probe("jpeg4py", "jpeg4py")
probe("lib.test.parameter.ostrack", "ostrack_parameter")
probe("lib.test.tracker.ostrack", "ostrack_tracker")

print(PREFIX + json.dumps(result, sort_keys=True))
"""
