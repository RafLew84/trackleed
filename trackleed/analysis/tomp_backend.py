"""ToMP / TaMOs environment diagnostics and smoke-test helpers."""

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
import time

import numpy as np

from trackleed.analysis.tracking_filter_utils import presence_score_is_accepted


DEFAULT_TOMP_MODELS_DIR = Path("tracking_models") / "tomp"
DEFAULT_TAMOS_MODELS_DIR = Path("tracking_models") / "tamos"
DEFAULT_TOMP_SUBPROCESS_SCRIPT = Path("scripts") / "run_tomp_subprocess.py"
DEFAULT_PYTRACKING_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "pytracking",
    Path(__file__).resolve().parents[2] / "pytracking",
)
DEFAULT_TOMP_CHECKPOINT = "tomp50.pth.tar"
DEFAULT_TOMP101_CHECKPOINT = "tomp101.pth.tar"
DEFAULT_TAMOS_CHECKPOINT = "tamos_resnet50.pth.tar"
DEFAULT_TAMOS_SWIN_BASE_CHECKPOINT = "tamos_swin_base.pth.tar"
DEFAULT_TOMP_BBOX_PADDING_CANDIDATES = (0, 4, 8, 12, 16)
TOMP_CHECKPOINT_PRIORITY = (
    DEFAULT_TOMP_CHECKPOINT,
    DEFAULT_TOMP101_CHECKPOINT,
)
TAMOS_CHECKPOINT_PRIORITY = (
    DEFAULT_TAMOS_CHECKPOINT,
    DEFAULT_TAMOS_SWIN_BASE_CHECKPOINT,
)
PYTRACKING_IMPORT_ROOTS = ("pytracking", "ltr")


@dataclass
class TompEnvironmentStatus:
    tomp_models_dir: Path
    tamos_models_dir: Path
    repo_path: Optional[Path] = None
    repo_networks_dir: Optional[Path] = None
    precise_roi_pooling_path: Optional[Path] = None
    precise_roi_pooling_submodule_present: bool = False
    evaluation_local_path: Optional[Path] = None
    evaluation_local_exists: bool = False
    evaluation_local_syntax_ok: bool = False
    ltr_local_path: Optional[Path] = None
    ltr_local_exists: bool = False
    ltr_local_syntax_ok: bool = False
    available_tomp_checkpoints: list[Path] = field(default_factory=list)
    available_tamos_checkpoints: list[Path] = field(default_factory=list)
    selected_tomp_checkpoint: Optional[Path] = None
    selected_tamos_checkpoint: Optional[Path] = None
    tomp_checkpoint_exists: bool = False
    tamos_checkpoint_exists: bool = False
    torch_importable: bool = False
    torch_version: Optional[str] = None
    cuda_available: bool = False
    torchvision_importable: bool = False
    cv2_importable: bool = False
    pillow_importable: bool = False
    matplotlib_importable: bool = False
    pandas_importable: bool = False
    timm_importable: bool = False
    pytracking_tracker_importable: bool = False
    pytracking_tracker_module_path: Optional[str] = None
    tomp_parameter_importable: bool = False
    tomp_parameter_module_path: Optional[str] = None
    tamos_parameter_importable: bool = False
    tamos_parameter_module_path: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def core_runtime_ready(self) -> bool:
        return (
            self.repo_path is not None
            and self.torch_importable
            and self.torchvision_importable
            and self.cv2_importable
            and self.pillow_importable
            and self.matplotlib_importable
            and self.pandas_importable
            and self.timm_importable
            and self.pytracking_tracker_importable
        )

    @property
    def tomp_ready(self) -> bool:
        return (
            self.core_runtime_ready
            and self.tomp_checkpoint_exists
            and self.tomp_parameter_importable
        )

    @property
    def tamos_ready(self) -> bool:
        return (
            self.core_runtime_ready
            and self.tamos_checkpoint_exists
            and self.tamos_parameter_importable
        )

    @property
    def is_ready(self) -> bool:
        return self.tomp_ready or self.tamos_ready


@dataclass
class TompSmokeTestResult:
    variant: str
    tracker_name: str
    parameter_name: str
    checkpoint_name: str
    device: str
    frame_shape: tuple[int, int, int]
    init_bbox_xywh: tuple[int, int, int, int]
    tracked_bbox_xywh: tuple[float, float, float, float]
    object_presence_score: float
    init_time_sec: float
    track_time_sec: float


@dataclass
class TompInitialBBoxPaddingResult:
    padding_px: int
    init_bbox_xywh: tuple[int, int, int, int]
    init_bbox_iou: float
    mean_tracking_iou: float
    min_tracking_iou: float
    last_tracking_iou: float
    mean_center_error_px: float
    max_center_error_px: float
    mean_presence_score: float
    min_presence_score: float
    aggregate_score: float


@dataclass
class TompInitialBBoxVariantResult:
    variant: str
    tracker_name: str
    parameter_name: str
    checkpoint_name: str
    padding_results: list[TompInitialBBoxPaddingResult]


@dataclass
class TompInitialBBoxSampleResult:
    sample_name: str
    frame_shape: tuple[int, int, int]
    frame_count: int
    reference_init_bbox_xywh: tuple[int, int, int, int]
    variant_results: list[TompInitialBBoxVariantResult]


@dataclass
class TompInitialBBoxProbeResult:
    samples: list[TompInitialBBoxSampleResult]
    evaluated_variants: list[str]
    candidate_paddings_px: list[int]
    recommended_padding_px: int
    aggregate_mean_scores: dict[int, float]
    notes: list[str] = field(default_factory=list)


@dataclass
class _TompSequenceRunResult:
    variant: str
    tracker_name: str
    parameter_name: str
    checkpoint_name: str
    device: str
    predicted_bboxes_xywh: np.ndarray
    presence_scores: np.ndarray
    init_time_sec: float
    track_times_sec: np.ndarray


@dataclass
class TompRuntimeSettings:
    checkpoint_path: str | Path
    repo_path: str | Path | None = None
    python_executable: str | Path | None = None
    backend_mode: str = "subprocess"
    model_variant: str = "tomp50"
    device: str = "auto"
    representative_point_mode: str = "bbox_center"
    inference_resolution: int | tuple[int, int] | None = None
    initial_bbox_padding_px: float = 0.0
    use_presence_score_gating: bool = False
    presence_threshold: float = 0.0
    subprocess_timeout_sec: float = 1800.0


@dataclass
class TompOfflineResult:
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
class _SyntheticTompProbeSample:
    name: str
    frames: np.ndarray
    gt_bboxes_xywh: np.ndarray


def discover_tomp_checkpoints(
    models_dir: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_TOMP_MODELS_DIR
    return _discover_variant_checkpoints(
        models_dir=directory,
        repo_path=repo_path,
        name_priority=TOMP_CHECKPOINT_PRIORITY,
        name_keywords=("tomp",),
    )


def discover_tamos_checkpoints(
    models_dir: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_TAMOS_MODELS_DIR
    return _discover_variant_checkpoints(
        models_dir=directory,
        repo_path=repo_path,
        name_priority=TAMOS_CHECKPOINT_PRIORITY,
        name_keywords=("tamos",),
    )


def select_default_tomp_checkpoint(checkpoints: list[Path]) -> Optional[Path]:
    return _select_default_checkpoint(checkpoints, TOMP_CHECKPOINT_PRIORITY)


def select_default_tamos_checkpoint(checkpoints: list[Path]) -> Optional[Path]:
    return _select_default_checkpoint(checkpoints, TAMOS_CHECKPOINT_PRIORITY)


def expected_tomp_checkpoint_name_for_variant(variant: str) -> str:
    key = _normalize_tomp_variant_key(variant)
    if key == "tomp101":
        return DEFAULT_TOMP101_CHECKPOINT
    if key == "tamos_resnet50":
        return DEFAULT_TAMOS_CHECKPOINT
    if key == "tamos_swin_base":
        return DEFAULT_TAMOS_SWIN_BASE_CHECKPOINT
    return DEFAULT_TOMP_CHECKPOINT


def select_default_tomp_checkpoint_for_variant(
    checkpoints: list[Path],
    variant: str,
) -> Optional[Path]:
    key = _normalize_tomp_variant_key(variant)
    if key == "tomp101":
        priority = (DEFAULT_TOMP101_CHECKPOINT, DEFAULT_TOMP_CHECKPOINT)
    else:
        priority = (DEFAULT_TOMP_CHECKPOINT, DEFAULT_TOMP101_CHECKPOINT)
    return _select_default_checkpoint(checkpoints, priority)


def select_default_tamos_checkpoint_for_variant(
    checkpoints: list[Path],
    variant: str,
) -> Optional[Path]:
    key = _normalize_tomp_variant_key(variant)
    if key == "tamos_swin_base":
        priority = (DEFAULT_TAMOS_SWIN_BASE_CHECKPOINT, DEFAULT_TAMOS_CHECKPOINT)
    else:
        priority = (DEFAULT_TAMOS_CHECKPOINT, DEFAULT_TAMOS_SWIN_BASE_CHECKPOINT)
    return _select_default_checkpoint(checkpoints, priority)


def probe_tomp_environment(
    *,
    tomp_models_dir: str | Path | None = None,
    tamos_models_dir: str | Path | None = None,
    tomp_checkpoint_path: str | Path | None = None,
    tamos_checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> TompEnvironmentStatus:
    resolved_tomp_models_dir = (
        Path(tomp_models_dir) if tomp_models_dir is not None else DEFAULT_TOMP_MODELS_DIR
    ).resolve()
    resolved_tamos_models_dir = (
        Path(tamos_models_dir) if tamos_models_dir is not None else DEFAULT_TAMOS_MODELS_DIR
    ).resolve()
    resolved_repo_path = _resolve_pytracking_repo_path(repo_path)
    repo_networks_dir = _resolve_pytracking_networks_dir(resolved_repo_path)
    tomp_checkpoints = discover_tomp_checkpoints(resolved_tomp_models_dir, resolved_repo_path)
    tamos_checkpoints = discover_tamos_checkpoints(resolved_tamos_models_dir, resolved_repo_path)
    selected_tomp_checkpoint = _resolve_checkpoint_path(
        tomp_checkpoint_path,
        tomp_checkpoints,
        resolved_tomp_models_dir,
        repo_networks_dir,
        TOMP_CHECKPOINT_PRIORITY,
    )
    selected_tamos_checkpoint = _resolve_checkpoint_path(
        tamos_checkpoint_path,
        tamos_checkpoints,
        resolved_tamos_models_dir,
        repo_networks_dir,
        TAMOS_CHECKPOINT_PRIORITY,
    )

    evaluation_local_path = (
        resolved_repo_path / "pytracking" / "evaluation" / "local.py"
        if resolved_repo_path is not None
        else None
    )
    ltr_local_path = (
        resolved_repo_path / "ltr" / "admin" / "local.py"
        if resolved_repo_path is not None
        else None
    )
    precise_roi_pooling_path = (
        resolved_repo_path / "ltr" / "external" / "PreciseRoIPooling"
        if resolved_repo_path is not None
        else None
    )

    evaluation_local_exists = bool(evaluation_local_path and evaluation_local_path.exists())
    evaluation_local_syntax_ok = (
        _python_source_syntax_ok(evaluation_local_path) if evaluation_local_exists else False
    )
    ltr_local_exists = bool(ltr_local_path and ltr_local_path.exists())
    ltr_local_syntax_ok = _python_source_syntax_ok(ltr_local_path) if ltr_local_exists else False

    status = TompEnvironmentStatus(
        tomp_models_dir=resolved_tomp_models_dir,
        tamos_models_dir=resolved_tamos_models_dir,
        repo_path=resolved_repo_path,
        repo_networks_dir=repo_networks_dir,
        precise_roi_pooling_path=precise_roi_pooling_path,
        precise_roi_pooling_submodule_present=bool(
            precise_roi_pooling_path and precise_roi_pooling_path.exists()
        ),
        evaluation_local_path=evaluation_local_path,
        evaluation_local_exists=evaluation_local_exists,
        evaluation_local_syntax_ok=evaluation_local_syntax_ok,
        ltr_local_path=ltr_local_path,
        ltr_local_exists=ltr_local_exists,
        ltr_local_syntax_ok=ltr_local_syntax_ok,
        available_tomp_checkpoints=tomp_checkpoints,
        available_tamos_checkpoints=tamos_checkpoints,
        selected_tomp_checkpoint=selected_tomp_checkpoint,
        selected_tamos_checkpoint=selected_tamos_checkpoint,
        tomp_checkpoint_exists=bool(selected_tomp_checkpoint and selected_tomp_checkpoint.exists()),
        tamos_checkpoint_exists=bool(selected_tamos_checkpoint and selected_tamos_checkpoint.exists()),
    )

    if not resolved_tomp_models_dir.exists():
        status.warnings.append(f"ToMP checkpoint directory does not exist: {resolved_tomp_models_dir}")
    if not resolved_tamos_models_dir.exists():
        status.warnings.append(
            f"TaMOs checkpoint directory does not exist: {resolved_tamos_models_dir}"
        )

    if selected_tomp_checkpoint is None:
        status.warnings.append(
            "No ToMP checkpoint detected in tracking_models/tomp or in pytracking/networks."
        )
        status.hints.append(f"Add `{DEFAULT_TOMP_CHECKPOINT}` before running ToMP diagnostics.")
    elif tomp_checkpoint_path and not status.tomp_checkpoint_exists:
        status.errors.append(f"Selected ToMP checkpoint not found: {selected_tomp_checkpoint}")

    if selected_tamos_checkpoint is None:
        status.warnings.append(
            "No TaMOs checkpoint detected in tracking_models/tamos or in pytracking/networks."
        )
        status.hints.append(f"Add `{DEFAULT_TAMOS_CHECKPOINT}` before running TaMOs diagnostics.")
    elif tamos_checkpoint_path and not status.tamos_checkpoint_exists:
        status.errors.append(f"Selected TaMOs checkpoint not found: {selected_tamos_checkpoint}")

    if resolved_repo_path is None:
        status.errors.append(
            "Could not resolve the local PyTracking repo. Clone `pytracking` next to the project "
            "or pass --repo-path."
        )
        status.hints.append(
            "Expected repo root containing `pytracking/evaluation/tracker.py`, "
            "`pytracking/parameter/tomp/tomp50.py` or `tomp101.py`, and "
            "`pytracking/parameter/tamos/tamos_resnet50.py` or `tamos_swin_base.py`."
        )
    else:
        if not status.precise_roi_pooling_submodule_present:
            status.warnings.append(
                "PyTracking submodule `ltr/external/PreciseRoIPooling` is missing. "
                "ToMP/TaMOs do not rely on it directly, but the clone looks incomplete."
            )
        if not status.evaluation_local_exists:
            status.errors.append(
                f"Missing PyTracking evaluation local settings file: {evaluation_local_path}"
            )
            status.hints.append(
                "Run `python -c \"from pytracking.evaluation.environment import "
                "create_default_local_file; create_default_local_file()\"` inside the `tomp` env."
            )
        elif not status.evaluation_local_syntax_ok:
            status.errors.append(
                f"PyTracking evaluation local settings contain invalid Python syntax: {evaluation_local_path}"
            )
            status.hints.append(
                "Edit `pytracking/evaluation/local.py` and replace Windows backslashes "
                "with forward slashes, e.g. `C:/Users/...`."
            )
        if not status.ltr_local_exists:
            status.warnings.append(f"Missing LTR local settings file: {ltr_local_path}")
            status.hints.append(
                "Run `python -c \"from ltr.admin.environment import create_default_local_file; "
                "create_default_local_file()\"` inside the `tomp` env."
            )
        elif not status.ltr_local_syntax_ok:
            status.warnings.append(
                f"LTR local settings contain invalid Python syntax: {ltr_local_path}"
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
            "Install PyTorch in the dedicated `tomp` env, e.g. "
            "`conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 "
            "pytorch-cuda=11.7 -c pytorch -c nvidia`."
        )

    _probe_regular_import(
        status,
        module_name="torchvision",
        attr_name="torchvision_importable",
        error_label="TorchVision import failed",
        hint="Install TorchVision in the active environment.",
    )
    _probe_regular_import(
        status,
        module_name="cv2",
        attr_name="cv2_importable",
        error_label="OpenCV import failed",
        hint="Install `opencv-python` in the active environment.",
    )
    _probe_regular_import(
        status,
        module_name="PIL",
        attr_name="pillow_importable",
        error_label="Pillow import failed",
        hint="Install `Pillow` in the active environment.",
    )
    _probe_regular_import(
        status,
        module_name="matplotlib",
        attr_name="matplotlib_importable",
        error_label="matplotlib import failed",
        hint="Install `matplotlib` in the active environment.",
    )
    _probe_regular_import(
        status,
        module_name="pandas",
        attr_name="pandas_importable",
        error_label="pandas import failed",
        hint="Install `pandas` in the active environment.",
    )
    _probe_regular_import(
        status,
        module_name="timm",
        attr_name="timm_importable",
        error_label="timm import failed",
        hint="Install `timm==0.4.12` in the active environment.",
    )

    if resolved_repo_path is not None:
        try:
            with _pytracking_import_context(resolved_repo_path):
                tracker_module = importlib.import_module("pytracking.evaluation.tracker")
            status.pytracking_tracker_importable = True
            status.pytracking_tracker_module_path = str(
                Path(inspect.getfile(tracker_module)).resolve()
            )
        except Exception as exc:
            status.errors.append(f"PyTracking Tracker import failed: {exc}")
            _append_pytracking_import_hints(status, exc)

        try:
            with _pytracking_import_context(resolved_repo_path):
                tomp_module = importlib.import_module("pytracking.parameter.tomp.tomp50")
            status.tomp_parameter_importable = True
            status.tomp_parameter_module_path = str(Path(inspect.getfile(tomp_module)).resolve())
        except Exception as exc:
            status.errors.append(f"ToMP parameter import failed: {exc}")
            _append_pytracking_import_hints(status, exc)

        try:
            with _pytracking_import_context(resolved_repo_path):
                tamos_module = importlib.import_module("pytracking.parameter.tamos.tamos_resnet50")
            status.tamos_parameter_importable = True
            status.tamos_parameter_module_path = str(Path(inspect.getfile(tamos_module)).resolve())
        except Exception as exc:
            status.errors.append(f"TaMOs parameter import failed: {exc}")
            _append_pytracking_import_hints(status, exc)

    if status.torch_importable and not status.cuda_available:
        status.warnings.append(
            "PyTorch CUDA is not available. ToMP/TaMOs can still run on CPU, but longer "
            "LEED sequences will be slow."
        )

    return status


def format_tomp_environment_report(status: TompEnvironmentStatus) -> str:
    lines = [
        "ToMP / TaMOs environment diagnostics",
        f"- Ready: {_format_bool(status.is_ready)}",
        f"- Core runtime ready: {_format_bool(status.core_runtime_ready)}",
        f"- ToMP ready: {_format_bool(status.tomp_ready)}",
        f"- TaMOs ready: {_format_bool(status.tamos_ready)}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Repo networks dir: {status.repo_networks_dir or 'None'}",
        f"- PreciseRoIPooling submodule: {_format_bool(status.precise_roi_pooling_submodule_present)}",
        f"- PreciseRoIPooling path: {status.precise_roi_pooling_path or 'None'}",
        f"- Evaluation local.py: {status.evaluation_local_path or 'None'}",
        f"- Evaluation local exists: {_format_bool(status.evaluation_local_exists)}",
        f"- Evaluation local syntax: {_format_bool(status.evaluation_local_syntax_ok)}",
        f"- LTR local.py: {status.ltr_local_path or 'None'}",
        f"- LTR local exists: {_format_bool(status.ltr_local_exists)}",
        f"- LTR local syntax: {_format_bool(status.ltr_local_syntax_ok)}",
        f"- ToMP models dir: {status.tomp_models_dir}",
        f"- Selected ToMP checkpoint: {status.selected_tomp_checkpoint or 'None'}",
        "- Available ToMP checkpoints: "
        + (", ".join(path.name for path in status.available_tomp_checkpoints) or "none"),
        f"- TaMOs models dir: {status.tamos_models_dir}",
        f"- Selected TaMOs checkpoint: {status.selected_tamos_checkpoint or 'None'}",
        "- Available TaMOs checkpoints: "
        + (", ".join(path.name for path in status.available_tamos_checkpoints) or "none"),
        f"- PyTorch import: {_format_bool(status.torch_importable)}",
        f"- PyTorch version: {status.torch_version or 'unknown'}",
        f"- CUDA available: {_format_bool(status.cuda_available)}",
        f"- torchvision import: {_format_bool(status.torchvision_importable)}",
        f"- OpenCV import: {_format_bool(status.cv2_importable)}",
        f"- Pillow import: {_format_bool(status.pillow_importable)}",
        f"- matplotlib import: {_format_bool(status.matplotlib_importable)}",
        f"- pandas import: {_format_bool(status.pandas_importable)}",
        f"- timm import: {_format_bool(status.timm_importable)}",
        f"- PyTracking Tracker import: {_format_bool(status.pytracking_tracker_importable)}",
        f"- ToMP parameter import: {_format_bool(status.tomp_parameter_importable)}",
        f"- TaMOs parameter import: {_format_bool(status.tamos_parameter_importable)}",
    ]
    if status.pytracking_tracker_module_path:
        lines.append(f"- PyTracking Tracker module: {status.pytracking_tracker_module_path}")
    if status.tomp_parameter_module_path:
        lines.append(f"- ToMP parameter module: {status.tomp_parameter_module_path}")
    if status.tamos_parameter_module_path:
        lines.append(f"- TaMOs parameter module: {status.tamos_parameter_module_path}")
    if status.errors:
        lines.append("- Errors:")
        lines.extend(f"  - {message}" for message in status.errors)
    if status.warnings:
        lines.append("- Warnings:")
        lines.extend(f"  - {message}" for message in status.warnings)
    if status.hints:
        lines.append("- Hints:")
        lines.extend(f"  - {message}" for message in status.hints)
    return "\n".join(lines)


def run_tomp_smoke_test(
    *,
    repo_path: str | Path,
    checkpoint_path: str | Path,
    variant: str = "tomp",
    device: str = "auto",
) -> TompSmokeTestResult:
    frames, init_bbox = _build_synthetic_tracking_frames()
    run = _run_tomp_variant_on_frames(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        variant=variant,
        device=device,
        frames=frames,
        init_bbox_xywh=init_bbox,
    )
    _, _, variant_name = _normalize_tomp_variant(variant)

    return TompSmokeTestResult(
        variant=variant_name,
        tracker_name=run.tracker_name,
        parameter_name=run.parameter_name,
        checkpoint_name=run.checkpoint_name,
        device=run.device,
        frame_shape=tuple(int(v) for v in frames[0].shape),
        init_bbox_xywh=init_bbox,
        tracked_bbox_xywh=tuple(float(v) for v in run.predicted_bboxes_xywh[1].tolist()),
        object_presence_score=float(run.presence_scores[1]),
        init_time_sec=float(run.init_time_sec),
        track_time_sec=float(run.track_times_sec[0]),
    )


def probe_tomp_initial_bbox_padding(
    *,
    repo_path: str | Path,
    tomp_checkpoint_path: str | Path | None = None,
    tamos_checkpoint_path: str | Path | None = None,
    variants: list[str] | tuple[str, ...] | None = None,
    candidate_paddings_px: list[int] | tuple[int, ...] | None = None,
    device: str = "auto",
) -> TompInitialBBoxProbeResult:
    normalized_variants = _normalize_tomp_probe_variants(variants or ("tomp", "tamos"))
    padding_candidates = _normalize_tomp_padding_candidates(
        candidate_paddings_px or DEFAULT_TOMP_BBOX_PADDING_CANDIDATES
    )
    samples = _build_tomp_bbox_probe_samples()
    aggregate_scores: dict[int, list[float]] = {padding_px: [] for padding_px in padding_candidates}
    sample_results: list[TompInitialBBoxSampleResult] = []

    variant_checkpoint_names: dict[str, str] = {}
    for variant in normalized_variants:
        checkpoint_path = (
            tamos_checkpoint_path
            if str(variant).startswith("tamos")
            else tomp_checkpoint_path
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Missing checkpoint path for `{variant}` in ToMP bbox probe."
            )
        variant_checkpoint_names[variant] = _resolve_existing_checkpoint_file(
            Path(checkpoint_path)
        ).name

    for sample in samples:
        variant_results: list[TompInitialBBoxVariantResult] = []
        for variant in normalized_variants:
            variant_padding_results: list[TompInitialBBoxPaddingResult] = []
            checkpoint_path = (
                tamos_checkpoint_path
                if str(variant).startswith("tamos")
                else tomp_checkpoint_path
            )
            assert checkpoint_path is not None
            tracker_name, parameter_name, variant_display_name = _normalize_tomp_variant(variant)

            for padding_px in padding_candidates:
                init_bbox = _expand_bbox_xywh(
                    sample.gt_bboxes_xywh[0],
                    padding_px=int(padding_px),
                    image_shape=sample.frames[0].shape[:2],
                )
                run = _run_tomp_variant_on_frames(
                    repo_path=repo_path,
                    checkpoint_path=checkpoint_path,
                    variant=variant,
                    device=device,
                    frames=sample.frames,
                    init_bbox_xywh=init_bbox,
                )
                metrics = _evaluate_tomp_probe_run(
                    gt_bboxes_xywh=sample.gt_bboxes_xywh,
                    predicted_bboxes_xywh=run.predicted_bboxes_xywh,
                    presence_scores=run.presence_scores,
                    padding_px=int(padding_px),
                    init_bbox_xywh=init_bbox,
                )
                aggregate_scores[int(padding_px)].append(metrics.aggregate_score)
                variant_padding_results.append(metrics)

            variant_results.append(
                TompInitialBBoxVariantResult(
                    variant=variant_display_name,
                    tracker_name=tracker_name,
                    parameter_name=parameter_name,
                    checkpoint_name=variant_checkpoint_names[variant],
                    padding_results=variant_padding_results,
                )
            )

        sample_results.append(
            TompInitialBBoxSampleResult(
                sample_name=sample.name,
                frame_shape=tuple(int(v) for v in sample.frames[0].shape),
                frame_count=int(sample.frames.shape[0]),
                reference_init_bbox_xywh=tuple(int(v) for v in sample.gt_bboxes_xywh[0].tolist()),
                variant_results=variant_results,
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

    notes = [
        "Probe korzysta z kilku syntetycznych sekwencji z malym obiektem, lekkim ruchem i prostymi dystraktorami.",
        "Score preferuje wysoka srednia i minimalna IoU po inicjalizacji, ale uwzglednia tez jak bardzo padding degraduje pierwszy bbox.",
        "To jest tylko sygnal startowy dla MVP; wynik trzeba jeszcze zweryfikowac na realnych sekwencjach LEED.",
    ]
    return TompInitialBBoxProbeResult(
        samples=sample_results,
        evaluated_variants=[
            _normalize_tomp_variant(variant)[2] for variant in normalized_variants
        ],
        candidate_paddings_px=[int(v) for v in padding_candidates],
        recommended_padding_px=int(recommended_padding_px),
        aggregate_mean_scores=aggregate_mean_scores,
        notes=notes,
    )


def format_tomp_initial_bbox_probe_report(result: TompInitialBBoxProbeResult) -> str:
    score_summary = ", ".join(
        f"{padding}px={score:.4f}"
        for padding, score in sorted(result.aggregate_mean_scores.items(), key=lambda item: item[0])
    )
    lines = [
        "ToMP / TaMOs initial bbox padding probe",
        f"- Recommended padding px: {result.recommended_padding_px}",
        f"- Evaluated variants: {', '.join(result.evaluated_variants)}",
        f"- Candidate paddings px: {', '.join(str(v) for v in result.candidate_paddings_px)}",
        f"- Aggregate mean scores: {score_summary}",
    ]
    for sample in result.samples:
        lines.append(f"- Sample: {sample.sample_name}")
        lines.append(f"  - Frame shape: {sample.frame_shape}")
        lines.append(f"  - Frame count: {sample.frame_count}")
        lines.append(f"  - Reference init bbox: {sample.reference_init_bbox_xywh}")
        for variant_result in sample.variant_results:
            lines.append(f"  - Variant: {variant_result.variant}")
            for padding_result in variant_result.padding_results:
                lines.append(
                    "    - Padding "
                    f"{padding_result.padding_px}px: "
                    f"init IoU={padding_result.init_bbox_iou:.4f}, "
                    f"mean IoU={padding_result.mean_tracking_iou:.4f}, "
                    f"min IoU={padding_result.min_tracking_iou:.4f}, "
                    f"last IoU={padding_result.last_tracking_iou:.4f}, "
                    f"mean center err={padding_result.mean_center_error_px:.3f}px, "
                    f"max center err={padding_result.max_center_error_px:.3f}px, "
                    f"mean score={padding_result.mean_presence_score:.4f}, "
                    f"min score={padding_result.min_presence_score:.4f}, "
                    f"aggregate={padding_result.aggregate_score:.4f}"
                )
    if result.notes:
        lines.append("- Notes:")
        lines.extend(f"  - {note}" for note in result.notes)
    return "\n".join(lines)


def run_tomp_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: TompRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
) -> TompOfflineResult:
    """
    Run offline ToMP / TaMOs on a sequence of frames and sparse query points.

    Args:
        frames: [T, H, W] grayscale or [T, H, W, C] uint8/float frames.
        query_points_tyx: [Q, 3] points in [t, y, x] in the original frame grid.
        settings: ToMP/TaMOs runtime settings including repo/checkpoint paths.
        query_boxes_xyxy: Optional [Q, 4] seed boxes in original image coordinates.

    Returns:
        TompOfflineResult with representative points, visibility and tracked boxes
        in original image coordinates.
    """
    if frames is None or np.asarray(frames).size == 0:
        raise ValueError("No frames provided for ToMP / TaMOs inference.")
    frames_np = np.asarray(frames)
    if frames_np.ndim not in (3, 4):
        raise ValueError(f"Expected frames with ndim 3 or 4, got shape {frames_np.shape}.")

    query_np = np.asarray(query_points_tyx, dtype=np.float32)
    if query_np.ndim != 2 or query_np.shape[1] != 3:
        raise ValueError(
            "query_points_tyx must have shape [num_queries, 3] in [t, y, x] format."
        )
    if query_np.shape[0] == 0:
        raise ValueError("No query points provided for ToMP / TaMOs inference.")

    backend_mode = str(settings.backend_mode or "subprocess").strip().lower()
    if backend_mode == "auto":
        backend_mode = "subprocess"
    if backend_mode == "direct":
        return _run_tomp_direct_offline(
            frames_np,
            query_np,
            settings,
            query_boxes_xyxy=query_boxes_xyxy,
        )
    if backend_mode != "subprocess":
        raise ValueError(
            f"Unsupported ToMP / TaMOs backend mode: {settings.backend_mode}. "
            "Expected `subprocess` or `direct`."
        )
    return _run_tomp_subprocess_offline(
        frames_np,
        query_np,
        settings,
        query_boxes_xyxy=query_boxes_xyxy,
    )


def _run_tomp_direct_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: TompRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
) -> TompOfflineResult:
    resolved_repo_path = _resolve_pytracking_repo_path(settings.repo_path)
    if resolved_repo_path is None:
        raise FileNotFoundError(
            "Could not resolve the PyTracking repo. Pass repo_path pointing at the local `pytracking` clone."
        )

    checkpoint_file = _resolve_existing_checkpoint_file(Path(settings.checkpoint_path))
    tracker_name, parameter_name, variant_name = _normalize_tomp_variant(settings.model_variant)
    representative_point_mode = _normalize_tomp_representative_point_mode(
        settings.representative_point_mode
    )

    prepared_frames, original_resolution = _prepare_frames_rgb(frames)
    inference_resolution = _resolve_tomp_inference_resolution(
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
    seed_boxes_xywh = _build_tomp_seed_boxes_xywh(
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

    with _pytracking_import_context(resolved_repo_path):
        torch_module = importlib.import_module("torch")
        evaluation_module = importlib.import_module("pytracking.evaluation")
        Tracker = getattr(evaluation_module, "Tracker")

        tracker_wrapper = Tracker(tracker_name, parameter_name)
        params = tracker_wrapper.get_parameters()
        resolved_device = _resolve_torch_device(torch_module, settings.device)
        use_gpu = resolved_device.type == "cuda"
        params.use_gpu = use_gpu
        params.visualization = False
        params.debug = 0
        params.device = "cuda" if use_gpu else "cpu"
        if hasattr(params, "net"):
            params.net.use_gpu = use_gpu
            params.net.net_path = str(checkpoint_file)

        tracker = tracker_wrapper.create_tracker(params)
        no_grad_context = torch_module.no_grad

        for query_index in range(num_queries):
            seed_frame = int(seed_frames[query_index])
            seed_bbox_xywh = tuple(float(v) for v in seed_boxes_xywh[query_index].tolist())
            init_info = {"init_bbox": list(seed_bbox_xywh)}

            with no_grad_context():
                tracker.initialize(resized_frames[seed_frame], init_info)

            _record_tomp_frame_output(
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
                use_presence_score_gating=bool(settings.use_presence_score_gating),
                presence_threshold=float(settings.presence_threshold),
            )

            previous_output: dict[str, Any] = {}
            for frame_index in range(seed_frame + 1, num_frames):
                with no_grad_context():
                    tracker_output = tracker.track(
                        resized_frames[frame_index],
                        {"previous_output": previous_output},
                    ) or {}
                previous_output = tracker_output

                bbox_value = _extract_single_object_value(tracker_output.get("target_bbox"))
                score_value = _extract_single_object_value(
                    tracker_output.get("object_presence_score")
                )
                score = float(score_value) if score_value is not None else float("nan")
                bbox_xywh = _normalize_optional_bbox_xywh(bbox_value)

                _record_tomp_frame_output(
                    tracked_bboxes_xywh=tracked_bboxes_xywh_resized,
                    tracks_xy=tracks_xy_resized,
                    visible_mask=visible_mask,
                    bbox_scores=bbox_scores,
                    query_index=query_index,
                    frame_index=frame_index,
                    bbox_xywh=bbox_xywh,
                    score=score,
                    frame=resized_frames[frame_index],
                    frame_resolution=inference_resolution,
                    representative_point_mode=representative_point_mode,
                    use_presence_score_gating=bool(settings.use_presence_score_gating),
                    presence_threshold=float(settings.presence_threshold),
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

    return TompOfflineResult(
        tracks_xy=tracks_xy,
        visible_mask=visible_mask,
        tracked_bboxes_xywh=tracked_bboxes_xywh,
        tracked_bboxes_xyxy=tracked_bboxes_xyxy,
        bbox_scores=bbox_scores,
        inference_resolution=inference_resolution,
        original_resolution=original_resolution,
        checkpoint_name=checkpoint_file.name,
        model_variant=variant_name,
        representative_point_mode=representative_point_mode,
        initial_bbox_padding_px=float(settings.initial_bbox_padding_px),
    )


def _run_tomp_subprocess_offline(
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    settings: TompRuntimeSettings,
    *,
    query_boxes_xyxy: np.ndarray | None = None,
) -> TompOfflineResult:
    python_executable = _resolve_tomp_python_executable(settings.python_executable)
    script_path = (Path(__file__).resolve().parents[2] / DEFAULT_TOMP_SUBPROCESS_SCRIPT).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"ToMP subprocess script not found: {script_path}")

    with TemporaryDirectory(prefix="trackleed_tomp_") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "tomp_input.npz"
        output_path = temp_root / "tomp_output.npz"
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
            "--presence-threshold",
            str(float(settings.presence_threshold)),
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
        if settings.use_presence_score_gating:
            command.append("--use-presence-score-gating")

        subprocess_env = _build_windows_conda_subprocess_env(python_executable)
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            check=False,
            timeout=float(settings.subprocess_timeout_sec),
            env=subprocess_env,
        )
        if completed.returncode != 0:
            payload = {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
            raise RuntimeError(
                "ToMP / TaMOs subprocess inference failed:\n"
                + json.dumps(payload, indent=2, ensure_ascii=False)
            )
        if not output_path.exists():
            raise FileNotFoundError(f"ToMP subprocess did not create output: {output_path}")

        with np.load(output_path, allow_pickle=False) as data:
            return TompOfflineResult(
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


def _discover_variant_checkpoints(
    *,
    models_dir: Path,
    repo_path: str | Path | None,
    name_priority: tuple[str, ...],
    name_keywords: tuple[str, ...],
) -> list[Path]:
    resolved_models_dir = models_dir.resolve()
    repo_networks_dir = _resolve_pytracking_networks_dir(_resolve_pytracking_repo_path(repo_path))
    discovered: dict[str, Path] = {}

    for directory in (resolved_models_dir, repo_networks_dir):
        if directory is None or not directory.exists():
            continue
        for path in _discover_checkpoints_in_directory(directory):
            lower_name = path.name.lower()
            if lower_name not in {name.lower() for name in name_priority} and not any(
                keyword in lower_name for keyword in name_keywords
            ):
                continue
            discovered.setdefault(lower_name, path.resolve())

    return sorted(discovered.values(), key=lambda path: path.name.lower())


def _discover_checkpoints_in_directory(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    checkpoints: list[Path] = []
    for pattern in ("*.pth.tar", "*.pth", "*.pt"):
        checkpoints.extend(directory.glob(pattern))
    return sorted(set(path.resolve() for path in checkpoints), key=lambda path: path.name.lower())


def _select_default_checkpoint(
    checkpoints: list[Path],
    priority_names: tuple[str, ...],
) -> Optional[Path]:
    if not checkpoints:
        return None
    by_name = {path.name: path for path in checkpoints}
    for preferred_name in priority_names:
        if preferred_name in by_name:
            return by_name[preferred_name]
    return checkpoints[0]


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoints: list[Path],
    models_dir: Path,
    repo_networks_dir: Path | None,
    priority_names: tuple[str, ...],
) -> Optional[Path]:
    if checkpoint_path is None:
        return _select_default_checkpoint(checkpoints, priority_names)

    candidate = Path(checkpoint_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    for base_dir in (models_dir, repo_networks_dir):
        if base_dir is None:
            continue
        resolved = (base_dir / candidate).resolve()
        if resolved.exists():
            return resolved
    return (models_dir / candidate).resolve()


def _resolve_existing_checkpoint_file(path: Path) -> Path:
    checkpoint = path.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def _resolve_tomp_python_executable(
    python_executable: str | Path | None,
) -> Path:
    if python_executable is not None:
        return Path(python_executable).expanduser().resolve()
    return Path(sys.executable).resolve()


def _build_windows_conda_subprocess_env(python_executable: Path) -> dict[str, str]:
    env = os.environ.copy()
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


def _resolve_pytracking_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidate_paths: list[Path] = []
    if repo_path is not None:
        candidate_paths.append(Path(repo_path).expanduser())
    else:
        candidate_paths.extend(DEFAULT_PYTRACKING_REPO_CANDIDATES)

    for candidate in candidate_paths:
        normalized = _normalize_pytracking_repo_candidate(candidate)
        if normalized is not None:
            return normalized
    return None


def _normalize_pytracking_repo_candidate(candidate: Path) -> Optional[Path]:
    resolved = candidate.resolve()
    expected_paths = (
        resolved / "pytracking" / "evaluation" / "tracker.py",
        resolved / "pytracking" / "parameter" / "tomp" / "tomp50.py",
        resolved / "pytracking" / "parameter" / "tamos" / "tamos_resnet50.py",
        resolved / "ltr" / "models" / "tracking" / "tompnet.py",
    )
    if all(path.exists() for path in expected_paths):
        return resolved
    return None


def _resolve_pytracking_networks_dir(repo_root: Optional[Path]) -> Optional[Path]:
    if repo_root is None:
        return None
    candidate = repo_root / "pytracking" / "networks"
    return candidate.resolve()


def _python_source_syntax_ok(path: Optional[Path]) -> bool:
    if path is None or not path.exists():
        return False
    try:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
    except Exception:
        return False
    return True


def _probe_regular_import(
    status: TompEnvironmentStatus,
    *,
    module_name: str,
    attr_name: str,
    error_label: str,
    hint: str,
) -> None:
    try:
        importlib.import_module(module_name)
        setattr(status, attr_name, True)
    except Exception as exc:
        status.errors.append(f"{error_label}: {exc}")
        status.hints.append(hint)


@contextmanager
def _pytracking_import_context(repo_path: Optional[Path]) -> Iterator[None]:
    if repo_path is None:
        yield
        return

    resolved_repo_path = repo_path.resolve()
    repo_root_str = str(resolved_repo_path)
    original_sys_path = list(sys.path)
    filtered_sys_path: list[str] = []

    repo_keys = {
        _normalize_path_key(resolved_repo_path),
        _normalize_path_key(resolved_repo_path.parent),
    }
    for entry in original_sys_path:
        entry_key = _normalize_path_key(entry)
        if entry_key and entry_key in repo_keys:
            continue
        filtered_sys_path.append(entry)

    original_modules: dict[str, Any] = {}
    for module_name, module in list(sys.modules.items()):
        if module_name.split(".", 1)[0] in PYTRACKING_IMPORT_ROOTS:
            original_modules[module_name] = module

    sys.path[:] = [repo_root_str, *filtered_sys_path]
    for module_name in list(sys.modules):
        if module_name.split(".", 1)[0] in PYTRACKING_IMPORT_ROOTS:
            sys.modules.pop(module_name, None)

    try:
        yield
    finally:
        for module_name in list(sys.modules):
            if module_name.split(".", 1)[0] in PYTRACKING_IMPORT_ROOTS:
                sys.modules.pop(module_name, None)
        sys.modules.update(original_modules)
        sys.path[:] = original_sys_path


def _normalize_path_key(path_like: str | Path | None) -> str | None:
    if path_like is None:
        return None
    try:
        return str(Path(path_like).resolve()).lower()
    except Exception:
        return None


def _append_pytracking_import_hints(status: TompEnvironmentStatus, exc: Exception) -> None:
    message = str(exc)
    message_lower = message.lower()
    if "unicodeescape" in message_lower or "local.py" in message_lower:
        status.hints.append(
            "Fix `pytracking/evaluation/local.py` and use forward slashes in Windows paths, "
            "e.g. `C:/Users/...`."
        )
    if "you have not setup your local.py" in message_lower:
        status.hints.append(
            "Run `create_default_local_file()` for `pytracking.evaluation.environment` and "
            "then fix Windows paths in `local.py`."
        )
    if "no module named 'timm'" in message_lower:
        status.hints.append("Install `timm==0.4.12` in the active environment.")
    if "no module named 'cv2'" in message_lower:
        status.hints.append("Install `opencv-python` in the active environment.")
    if "no module named 'matplotlib'" in message_lower:
        status.hints.append("Install `matplotlib` in the active environment.")
    if "no module named 'pandas'" in message_lower:
        status.hints.append("Install `pandas` in the active environment.")
    if "no module named 'visdom'" in message_lower:
        status.hints.append("Install `visdom` in the active environment.")
    if "no module named 'jpeg4py'" in message_lower:
        status.hints.append("Install `jpeg4py` in the active environment.")
    if "no module named 'pytracking'" in message_lower or "no module named 'ltr'" in message_lower:
        status.hints.append("Pass --repo-path pointing at the local `pytracking` repo root.")
    if not status.hints:
        status.hints.append(
            "Verify that the local `pytracking` repo is available and that its runtime "
            "dependencies are installed in the active environment."
        )


def _format_bool(value: bool) -> str:
    return "yes" if bool(value) else "no"


def _normalize_tomp_variant(variant: str) -> tuple[str, str, str]:
    key = _normalize_tomp_variant_key(variant)
    if key == "tomp50":
        return "tomp", "tomp50", "ToMP-50"
    if key == "tomp101":
        return "tomp", "tomp101", "ToMP-101"
    if key == "tamos_resnet50":
        return "tamos", "tamos_resnet50", "TaMOs-resnet50"
    if key == "tamos_swin_base":
        return "tamos", "tamos_swin_base", "TaMOs-swin-base"
    raise ValueError(
        "Unsupported ToMP/TaMOs variant: "
        f"{variant}. Expected `tomp50`, `tomp101`, `tamos_resnet50`, or `tamos_swin_base`."
    )


def _normalize_tomp_probe_variants(
    variants: list[str] | tuple[str, ...],
) -> list[str]:
    normalized: list[str] = []
    for variant in variants:
        variant_key = _normalize_tomp_variant_key(variant)
        if variant_key not in normalized:
            normalized.append(variant_key)
    if not normalized:
        raise ValueError("No ToMP/TaMOs variants selected for the bbox probe.")
    return normalized


def _normalize_tomp_variant_key(variant: str | None) -> str:
    key = str(variant or "tomp50").strip().lower().replace("-", "_")
    if key in {"tomp", "tomp50", "tomp_50"}:
        return "tomp50"
    if key in {"tomp101", "tomp_101"}:
        return "tomp101"
    if key in {"tamos", "tamos50", "tamos_resnet50", "tamos_resnet_50"}:
        return "tamos_resnet50"
    if key in {"tamos_swin", "tamos_swin_base", "tamos_swinbase"}:
        return "tamos_swin_base"
    raise ValueError(
        "Unsupported ToMP/TaMOs variant: "
        f"{variant}. Expected `tomp50`, `tomp101`, `tamos_resnet50`, or `tamos_swin_base`."
    )


def _normalize_tomp_padding_candidates(
    paddings_px: list[int] | tuple[int, ...],
) -> list[int]:
    normalized: list[int] = []
    for value in paddings_px:
        padding = int(value)
        if padding < 0:
            raise ValueError(f"Padding candidates must be >= 0, got {padding}.")
        if padding not in normalized:
            normalized.append(padding)
    if not normalized:
        raise ValueError("No padding candidates provided for the ToMP bbox probe.")
    return normalized


def _resolve_torch_device(torch_module, requested_device: str):
    device_name = str(requested_device or "auto").strip().lower()
    if device_name in {"", "auto", "gpu"}:
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if device_name in {"cuda", "cuda:0"} and not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    return torch_module.device(device_name)


def _normalize_tomp_representative_point_mode(mode: str) -> str:
    normalized = str(mode or "bbox_center").strip().lower()
    if normalized in {"bbox_center", "center", "bbox-centre", "bbox_centre"}:
        return "bbox_center"
    if normalized in {"peak_in_bbox", "peak", "bbox_peak"}:
        return "peak_in_bbox"
    if normalized in {"gaussian_in_bbox", "gaussian", "bbox_gaussian", "2d_gauss"}:
        return "gaussian_in_bbox"
    raise ValueError(
        "Unsupported ToMP representative point mode: "
        f"{mode}. Expected `bbox_center`, `peak_in_bbox`, or `gaussian_in_bbox`."
    )


def _resolve_tomp_inference_resolution(
    target: int | tuple[int, int] | None,
    original_resolution: tuple[int, int],
) -> tuple[int, int]:
    if target is None:
        return (int(original_resolution[0]), int(original_resolution[1]))
    if isinstance(target, int):
        size = int(target)
        if size <= 0:
            raise ValueError(f"Invalid ToMP inference resolution: {target}")
        return (size, size)
    height = int(target[0])
    width = int(target[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid ToMP inference resolution: {(height, width)}")
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


def _build_tomp_seed_boxes_xywh(
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
        x0 = x - half_size
        y0 = y - half_size
        x1 = x + half_size
        y1 = y + half_size
        boxes_xywh[index] = _xyxy_to_clipped_xywh(
            np.asarray([x0, y0, x1, y1], dtype=np.float32),
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


def _record_tomp_frame_output(
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
    use_presence_score_gating: bool,
    presence_threshold: float,
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
    if not presence_score_is_accepted(
        score,
        use_presence_score_gating=use_presence_score_gating,
        presence_threshold=presence_threshold,
    ):
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
    normalized_mode = _normalize_tomp_representative_point_mode(mode)
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

    fit = fit_spot_to_gaussian(
        roi,
        center_preference_factor=0.35,
    )
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


def _run_tomp_variant_on_frames(
    *,
    repo_path: str | Path,
    checkpoint_path: str | Path,
    variant: str,
    device: str,
    frames: np.ndarray,
    init_bbox_xywh: tuple[int, int, int, int] | tuple[float, float, float, float],
) -> _TompSequenceRunResult:
    resolved_repo_path = _resolve_pytracking_repo_path(repo_path)
    if resolved_repo_path is None:
        raise FileNotFoundError(
            "Could not resolve the PyTracking repo. Pass repo_path pointing at the local `pytracking` clone."
        )

    checkpoint_file = _resolve_existing_checkpoint_file(Path(checkpoint_path))
    tracker_name, parameter_name, variant_name = _normalize_tomp_variant(variant)

    with _pytracking_import_context(resolved_repo_path):
        torch_module = importlib.import_module("torch")
        evaluation_module = importlib.import_module("pytracking.evaluation")
        Tracker = getattr(evaluation_module, "Tracker")

        tracker_wrapper = Tracker(tracker_name, parameter_name)
        params = tracker_wrapper.get_parameters()
        resolved_device = _resolve_torch_device(torch_module, device)
        use_gpu = resolved_device.type == "cuda"
        params.use_gpu = use_gpu
        params.visualization = False
        params.debug = 0
        params.device = "cuda" if use_gpu else "cpu"
        if hasattr(params, "net"):
            params.net.use_gpu = use_gpu
            params.net.net_path = str(checkpoint_file)

        tracker = tracker_wrapper.create_tracker(params)
        init_info = {"init_bbox": [float(v) for v in init_bbox_xywh]}

        init_start = time.perf_counter()
        init_output = tracker.initialize(frames[0], init_info) or {}
        init_time_sec = time.perf_counter() - init_start

        predicted_bboxes = [np.asarray(init_bbox_xywh, dtype=np.float32)]
        presence_scores = [1.0]
        track_times_sec: list[float] = []
        previous_output = init_output

        for frame in frames[1:]:
            track_start = time.perf_counter()
            track_output = tracker.track(frame, {"previous_output": previous_output}) or {}
            track_time_sec = time.perf_counter() - track_start
            previous_output = track_output

            tracked_bbox_value = _extract_single_object_value(track_output.get("target_bbox"))
            tracked_bbox = _normalize_bbox_xywh(tracked_bbox_value)
            predicted_bboxes.append(np.asarray(tracked_bbox, dtype=np.float32))

            object_presence_score_value = _extract_single_object_value(
                track_output.get("object_presence_score")
            )
            presence_scores.append(
                float(object_presence_score_value)
                if object_presence_score_value is not None
                else float("nan")
            )
            track_times_sec.append(float(track_time_sec))

    return _TompSequenceRunResult(
        variant=variant_name,
        tracker_name=tracker_name,
        parameter_name=parameter_name,
        checkpoint_name=checkpoint_file.name,
        device=str(resolved_device),
        predicted_bboxes_xywh=np.asarray(predicted_bboxes, dtype=np.float32),
        presence_scores=np.asarray(presence_scores, dtype=np.float32),
        init_time_sec=float(init_time_sec),
        track_times_sec=np.asarray(track_times_sec, dtype=np.float32),
    )


def _build_synthetic_tracking_frames() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height = 192
    width = 256
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.indices((height, width))
    frame[..., 0] = np.clip(18 + 0.10 * xx + 0.04 * yy, 0, 255).astype(np.uint8)
    frame[..., 1] = np.clip(24 + 0.06 * xx + 0.07 * yy, 0, 255).astype(np.uint8)
    frame[..., 2] = np.clip(12 + 0.04 * xx + 0.11 * yy, 0, 255).astype(np.uint8)

    x = 92
    y = 74
    w = 46
    h = 52
    patch_y = np.broadcast_to(
        np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None],
        (h, w),
    )
    patch_x = np.broadcast_to(
        np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :],
        (h, w),
    )
    patch = np.stack(
        [
            190 + 35 * patch_x,
            60 + 160 * patch_y,
            210 - 90 * patch_x,
        ],
        axis=-1,
    )
    frame[y : y + h, x : x + w] = np.clip(patch, 0, 255).astype(np.uint8)
    frame[y + 8 : y + 16, x + 8 : x + 38] = np.array([255, 240, 90], dtype=np.uint8)
    frame[y + 24 : y + 44, x + 14 : x + 20] = np.array([30, 30, 30], dtype=np.uint8)

    return np.stack([frame, frame.copy()], axis=0), (x, y, w, h)


def _build_tomp_bbox_probe_samples() -> list[_SyntheticTompProbeSample]:
    return [
        _synthesize_tomp_probe_sequence(
            name="tiny_drift",
            gt_bboxes_xywh=[
                (104, 78, 12, 12),
                (105, 79, 12, 12),
                (107, 79, 12, 12),
                (108, 80, 12, 12),
            ],
            color_mode="amber",
            distractors=[
                (126, 80, 13, 13, (245, 208, 82)),
                (64, 118, 18, 8, (80, 120, 220)),
            ],
        ),
        _synthesize_tomp_probe_sequence(
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
        _synthesize_tomp_probe_sequence(
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


def _synthesize_tomp_probe_sequence(
    *,
    name: str,
    gt_bboxes_xywh: list[tuple[int, int, int, int]],
    color_mode: str,
    distractors: list[tuple[int, int, int, int, tuple[int, int, int]]],
) -> _SyntheticTompProbeSample:
    height = 192
    width = 256
    frames: list[np.ndarray] = []

    for frame_index, bbox in enumerate(gt_bboxes_xywh):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        yy, xx = np.indices((height, width))
        frame[..., 0] = np.clip(18 + 0.06 * xx + 0.03 * yy + frame_index * 2.0, 0, 255).astype(
            np.uint8
        )
        frame[..., 1] = np.clip(20 + 0.05 * yy + 0.02 * xx, 0, 255).astype(np.uint8)
        frame[..., 2] = np.clip(14 + 0.08 * xx + 0.01 * yy + frame_index * 1.0, 0, 255).astype(
            np.uint8
        )

        stripe = ((xx // 16 + yy // 12 + frame_index) % 2) * 8
        frame[..., 1] = np.clip(frame[..., 1].astype(np.int16) + stripe.astype(np.int16), 0, 255).astype(
            np.uint8
        )

        for dx, dy, dw, dh, color in distractors:
            x0 = int(np.clip(dx, 0, width - 1))
            y0 = int(np.clip(dy, 0, height - 1))
            x1 = int(np.clip(dx + dw, x0 + 1, width))
            y1 = int(np.clip(dy + dh, y0 + 1, height))
            frame[y0:y1, x0:x1] = np.asarray(color, dtype=np.uint8)

        target_patch = _build_tomp_target_patch(bbox[3], bbox[2], color_mode=color_mode)
        x, y, w, h = bbox
        frame[y : y + h, x : x + w] = target_patch
        frames.append(frame)

    return _SyntheticTompProbeSample(
        name=name,
        frames=np.asarray(frames, dtype=np.uint8),
        gt_bboxes_xywh=np.asarray(gt_bboxes_xywh, dtype=np.float32),
    )


def _build_tomp_target_patch(
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
    else:
        patch = np.stack(
            [
                180 + 45 * patch_x,
                40 + 25 * patch_y,
                170 + 50 * (1.0 - patch_x),
            ],
            axis=-1,
        )

    patch = np.clip(patch, 0.0, 255.0)
    patch = patch.astype(np.uint8)
    patch[max(1, height // 5) : max(2, height // 5 + height // 6), max(1, width // 5) : max(2, width - width // 4)] = np.array(
        [250, 240, 92],
        dtype=np.uint8,
    )
    patch[max(1, height // 2) : max(2, height - height // 6), max(1, width // 3) : max(2, width // 3 + max(2, width // 6))] = np.array(
        [25, 25, 25],
        dtype=np.uint8,
    )
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


def _evaluate_tomp_probe_run(
    *,
    gt_bboxes_xywh: np.ndarray,
    predicted_bboxes_xywh: np.ndarray,
    presence_scores: np.ndarray,
    padding_px: int,
    init_bbox_xywh: tuple[int, int, int, int],
) -> TompInitialBBoxPaddingResult:
    tracking_ious = [
        _bbox_iou_xywh(predicted_bboxes_xywh[index], gt_bboxes_xywh[index])
        for index in range(1, int(len(gt_bboxes_xywh)))
    ]
    center_errors = [
        _bbox_center_error_px(predicted_bboxes_xywh[index], gt_bboxes_xywh[index])
        for index in range(1, int(len(gt_bboxes_xywh)))
    ]
    tracking_presence_scores = presence_scores[1:]
    init_bbox_iou = _bbox_iou_xywh(np.asarray(init_bbox_xywh, dtype=np.float32), gt_bboxes_xywh[0])
    mean_tracking_iou = float(np.mean(tracking_ious)) if tracking_ious else init_bbox_iou
    min_tracking_iou = float(np.min(tracking_ious)) if tracking_ious else init_bbox_iou
    last_tracking_iou = float(tracking_ious[-1]) if tracking_ious else init_bbox_iou
    mean_center_error_px = float(np.mean(center_errors)) if center_errors else 0.0
    max_center_error_px = float(np.max(center_errors)) if center_errors else 0.0
    mean_presence_score = (
        float(np.nanmean(tracking_presence_scores))
        if tracking_presence_scores.size > 0
        else 1.0
    )
    min_presence_score = (
        float(np.nanmin(tracking_presence_scores))
        if tracking_presence_scores.size > 0
        else 1.0
    )
    aggregate_score = (
        0.65 * mean_tracking_iou
        + 0.20 * min_tracking_iou
        + 0.15 * init_bbox_iou
    )

    return TompInitialBBoxPaddingResult(
        padding_px=int(padding_px),
        init_bbox_xywh=tuple(int(v) for v in init_bbox_xywh),
        init_bbox_iou=float(init_bbox_iou),
        mean_tracking_iou=float(mean_tracking_iou),
        min_tracking_iou=float(min_tracking_iou),
        last_tracking_iou=float(last_tracking_iou),
        mean_center_error_px=float(mean_center_error_px),
        max_center_error_px=float(max_center_error_px),
        mean_presence_score=float(mean_presence_score),
        min_presence_score=float(min_presence_score),
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


def _extract_single_object_value(value):
    if isinstance(value, dict):
        if "1" in value:
            return value["1"]
        if 1 in value:
            return value[1]
        for candidate in value.values():
            return candidate
        return None
    return value


def _normalize_bbox_xywh(value: Any) -> tuple[float, float, float, float]:
    if value is None:
        raise RuntimeError("Tracker smoke test did not return a target_bbox.")
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 4 or not np.all(np.isfinite(arr)):
        raise RuntimeError(f"Invalid tracker bbox returned by smoke test: {value!r}")
    return tuple(float(v) for v in arr.tolist())
