"""Headless catalog adapters for bounding-box seeded tracking."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from importlib import import_module
from math import isfinite
from pathlib import Path
from typing import Any

from trackleed.logic.workflows.bbox_tracking import (
    BboxBackendMode,
    BoundingBoxTrackerConfig,
    OstrackBoundingBoxConfig,
    TompBoundingBoxConfig,
    TompModelVariant,
)
from trackleed.logic.workflows.seeded_tracking import (
    RunContext,
    SeededRetrackRequest,
    SeededTrackingCapabilities,
    SeededTrackingConfig,
    SeededTrackingMethod,
    SeededTrackingRequest,
    SeedRepresentation,
    TrackingFamily,
    TrackingPreflightIssue,
    TrackingResult,
)


@dataclass(frozen=True)
class BboxTrackerDefinition:
    method_id: str
    display_name: str
    config_type: type[BoundingBoxTrackerConfig]
    default_config: BoundingBoxTrackerConfig
    backend_module: str
    probe_name: str
    source_names: tuple[str, ...]
    supports_subprocess: bool = True


_BBOX_METHODS = (
    BboxTrackerDefinition(
        "tomp_offline",
        "Offline ToMP / TaMOs",
        TompBoundingBoxConfig,
        TompBoundingBoxConfig(),
        "trackleed.analysis.tomp_backend",
        "probe_tomp_environment",
        ("tomp_offline", "tamos_offline"),
    ),
    BboxTrackerDefinition(
        "ostrack_offline",
        "Offline OSTrack",
        OstrackBoundingBoxConfig,
        OstrackBoundingBoxConfig(),
        "trackleed.analysis.ostrack_backend",
        "probe_ostrack_environment",
        ("ostrack_offline",),
    ),
)


class BboxTrackerRuntime:
    def __init__(self, definition: BboxTrackerDefinition) -> None:
        self._definition = definition

    def validate_config(
        self,
        config: SeededTrackingConfig,
    ) -> tuple[TrackingPreflightIssue, ...]:
        if not isinstance(config, self._definition.config_type):
            return (
                TrackingPreflightIssue(
                    "invalid_config",
                    f"{self._definition.display_name} requires its bounding-box configuration.",
                ),
            )
        return _config_issues(config, self._definition)

    def preflight(
        self,
        config: SeededTrackingConfig,
        context: RunContext | None = None,
    ) -> tuple[TrackingPreflightIssue, ...]:
        issues = self.validate_config(config)
        if issues:
            return issues
        assert isinstance(config, BoundingBoxTrackerConfig)
        if context is not None and context.bbox_tracker_preflight is not None:
            issues = context.bbox_tracker_preflight(
                self._definition.method_id,
                config,
            )
        else:
            issues = _probe_environment(config, self._definition)
        if context is None:
            return issues
        executor = (
            context.bbox_tracker_retrack_executor
            if context.preflight_operation == "retrack"
            else context.bbox_tracker_executor
        )
        if executor is None:
            issues += (
                TrackingPreflightIssue(
                    "executor_unavailable",
                    f"{self._definition.display_name} execution adapter is unavailable.",
                ),
            )
        return issues

    def run(
        self,
        request: SeededTrackingRequest,
        context: RunContext,
    ) -> TrackingResult:
        if context.bbox_tracker_executor is None:
            raise RuntimeError("Bounding-box execution adapter is unavailable")
        return context.bbox_tracker_executor(
            self._definition.method_id,
            request,
            context,
        )

    def retrack(
        self,
        request: SeededRetrackRequest,
        context: RunContext,
    ) -> TrackingResult:
        if context.bbox_tracker_retrack_executor is None:
            raise RuntimeError("Bounding-box re-track adapter is unavailable")
        return context.bbox_tracker_retrack_executor(
            self._definition.method_id,
            request,
            context,
        )


def bbox_tracking_definitions() -> tuple[BboxTrackerDefinition, ...]:
    return _BBOX_METHODS


def bbox_tracking_definition(method_id: str) -> BboxTrackerDefinition:
    for definition in _BBOX_METHODS:
        if definition.method_id == method_id:
            return definition
    raise KeyError(f"Unknown bounding-box tracker ID: {method_id}")


def default_bbox_tracker_configs(
    *,
    roi_size: int = 32,
) -> tuple[tuple[str, BoundingBoxTrackerConfig], ...]:
    return tuple(
        (
            definition.method_id,
            replace(definition.default_config, roi_size=roi_size),
        )
        for definition in _BBOX_METHODS
    )


def bbox_tracking_methods() -> tuple[SeededTrackingMethod, ...]:
    capabilities = SeededTrackingCapabilities(
        SeedRepresentation.BOUNDING_BOX,
        supports_background=False,
        preserves_gaps=True,
        supports_retrack=True,
    )
    return tuple(
        SeededTrackingMethod(
            definition.method_id,
            definition.display_name,
            TrackingFamily.BOUNDING_BOX,
            capabilities,
            BboxTrackerRuntime(definition),
            BboxTrackerRuntime(definition),
        )
        for definition in _BBOX_METHODS
    )


def _config_issues(
    config: BoundingBoxTrackerConfig,
    definition: BboxTrackerDefinition,
) -> tuple[TrackingPreflightIssue, ...]:
    resolution = config.inference_resolution
    resolution_valid = (
        resolution is None
        or isinstance(resolution, int)
        and resolution > 0
        or isinstance(resolution, tuple)
        and len(resolution) == 2
        and all(isinstance(value, int) and value > 0 for value in resolution)
    )
    valid = (
        config.backend_mode in {BboxBackendMode.DIRECT, BboxBackendMode.SUBPROCESS}
        and config.roi_size > 0
        and config.window_size > 0
        and 0 <= config.window_overlap < config.window_size
        and config.initial_bbox_padding_px >= 0
        and config.subprocess_timeout_sec > 0
        and 0 <= config.presence_threshold <= 1
        and config.max_step_distance_px >= 0
        and config.max_reacquisition_gap >= 0
        and config.terminate_after_invisible_frames >= 0
        and config.lateral_angle_tolerance_deg >= 0
        and config.max_radial_jump_px >= 0
        and _manual_center_valid(config.manual_center)
        and resolution_valid
        and _model_variant_valid(config)
    )
    if valid:
        return ()
    return (
        TrackingPreflightIssue(
            "invalid_config",
            f"{definition.display_name} settings are outside their valid ranges.",
        ),
    )


def _probe_environment(
    config: BoundingBoxTrackerConfig,
    definition: BboxTrackerDefinition,
) -> tuple[TrackingPreflightIssue, ...]:
    try:
        probe = getattr(import_module(definition.backend_module), definition.probe_name)
        status = probe(**_probe_kwargs(config))
    except Exception as exc:  # noqa: BLE001 - optional probes must fail closed
        return (
            TrackingPreflightIssue(
                "environment_probe_failed",
                f"{definition.display_name} environment probe failed: {exc}",
            ),
        )
    issues = list(_structural_issues(config, definition, status))
    if issues:
        return tuple(issues)
    if (
        isinstance(config, TompBoundingBoxConfig)
        and config.backend_mode is BboxBackendMode.SUBPROCESS
    ):
        return _subprocess_environment_issues(config, definition)
    if not _variant_ready(config, status):
        issues.append(
            TrackingPreflightIssue(
                "dependency_unavailable",
                f"{definition.display_name} runtime dependencies are unavailable.",
            )
        )
    return tuple(issues)


def _probe_kwargs(config: BoundingBoxTrackerConfig) -> dict[str, Any]:
    if isinstance(config, OstrackBoundingBoxConfig):
        return {
            "models_dir": config.models_dir,
            "checkpoint_path": config.checkpoint_path,
            "repo_path": config.repo_path,
            "python_executable": (
                config.python_executable
                if config.backend_mode is BboxBackendMode.SUBPROCESS
                else None
            ),
            "config_name": config.model_variant,
        }
    assert isinstance(config, TompBoundingBoxConfig)
    is_tamos = config.model_variant.value.startswith("tamos")
    return {
        "tomp_models_dir": config.tomp_models_dir,
        "tamos_models_dir": config.tamos_models_dir,
        "tomp_checkpoint_path": None if is_tamos else config.checkpoint_path,
        "tamos_checkpoint_path": config.checkpoint_path if is_tamos else None,
        "repo_path": config.repo_path,
    }


def _subprocess_environment_issues(
    config: TompBoundingBoxConfig,
    definition: BboxTrackerDefinition,
) -> tuple[TrackingPreflightIssue, ...]:
    executable = _interpreter_path(config.python_executable)
    if not executable.is_file():
        return (_issue("interpreter_unavailable", definition, "interpreter"),)
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "probe_seeded_bbox_subprocess.py"
    )
    kwargs = {
        name: None if value is None else str(value)
        for name, value in _probe_kwargs(config).items()
    }
    ready_attr = (
        "tamos_ready"
        if config.model_variant.value.startswith("tamos")
        else "tomp_ready"
    )
    try:
        completed = subprocess.run(
            (
                str(executable),
                str(script_path),
                definition.backend_module,
                definition.probe_name,
                ready_attr,
                json.dumps(kwargs),
            ),
            cwd=script_path.parents[1],
            capture_output=True,
            text=True,
            check=False,
            timeout=60.0,
        )
    except (OSError, subprocess.SubprocessError):
        return (_issue("interpreter_unavailable", definition, "interpreter"),)
    if completed.returncode == 0:
        return ()
    return (
        TrackingPreflightIssue(
            "dependency_unavailable",
            f"{definition.display_name} subprocess environment is unavailable.",
        ),
    )


def _structural_issues(
    config: BoundingBoxTrackerConfig,
    definition: BboxTrackerDefinition,
    status: Any,
) -> tuple[TrackingPreflightIssue, ...]:
    issues: list[TrackingPreflightIssue] = []
    if getattr(status, "repo_path", None) is None:
        issues.append(_issue("repository_unavailable", definition, "repository"))
    if not _checkpoint_exists(config, status):
        issues.append(_issue("checkpoint_unavailable", definition, "checkpoint"))
    if isinstance(config, OstrackBoundingBoxConfig) and not bool(
        getattr(status, "config_exists", False)
    ):
        issues.append(_issue("config_unavailable", definition, "configuration"))
    if config.backend_mode is BboxBackendMode.SUBPROCESS and not _interpreter_exists(
        config.python_executable
    ):
        issues.append(_issue("interpreter_unavailable", definition, "interpreter"))
    return tuple(issues)


def _checkpoint_exists(config: BoundingBoxTrackerConfig, status: Any) -> bool:
    if isinstance(config, OstrackBoundingBoxConfig):
        return bool(getattr(status, "checkpoint_exists", False))
    assert isinstance(config, TompBoundingBoxConfig)
    if config.model_variant.value.startswith("tamos"):
        return bool(getattr(status, "tamos_checkpoint_exists", False))
    return bool(getattr(status, "tomp_checkpoint_exists", False))


def _variant_ready(config: BoundingBoxTrackerConfig, status: Any) -> bool:
    if isinstance(config, OstrackBoundingBoxConfig):
        return bool(getattr(status, "is_ready", False))
    assert isinstance(config, TompBoundingBoxConfig)
    ready_attr = (
        "tamos_ready"
        if config.model_variant.value.startswith("tamos")
        else "tomp_ready"
    )
    return bool(getattr(status, ready_attr, False))


def _interpreter_exists(path: Path | None) -> bool:
    return _interpreter_path(path).is_file()


def _interpreter_path(path: Path | None) -> Path:
    return Path(path or sys.executable).expanduser()


def _manual_center_valid(value: tuple[float, float]) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, (int, float)) and isfinite(item) for item in value)
    )


def _model_variant_valid(config: BoundingBoxTrackerConfig) -> bool:
    if isinstance(config, TompBoundingBoxConfig):
        return isinstance(config.model_variant, TompModelVariant)
    return isinstance(config, OstrackBoundingBoxConfig) and bool(
        config.model_variant.strip()
    )


def _issue(
    code: str,
    definition: BboxTrackerDefinition,
    resource: str,
) -> TrackingPreflightIssue:
    return TrackingPreflightIssue(
        code,
        f"{definition.display_name} {resource} is unavailable.",
    )
