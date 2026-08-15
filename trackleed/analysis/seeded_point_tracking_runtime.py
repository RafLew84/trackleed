"""Headless catalog adapters for point-family seeded tracking."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from importlib import import_module
from math import isfinite
from pathlib import Path
from typing import Any

from trackleed.logic.workflows.point_tracking import (
    BackendMode,
    CotrackerPointConfig,
    LocotrackPointConfig,
    MftiqPointConfig,
    PipsppPointConfig,
    PointTrackerConfig,
    TapirPointConfig,
    TapirRuntimeMode,
    TapnextPointConfig,
    TrackonrPointConfig,
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
class PointTrackerDefinition:
    method_id: str
    display_name: str
    config_type: type[PointTrackerConfig]
    default_config: PointTrackerConfig
    backend_module: str
    probe_name: str
    requires_repo: bool = False
    requires_config: bool = False
    supports_subprocess: bool = False
    runtime_mode: TapirRuntimeMode | None = None


_POINT_METHODS = (
    PointTrackerDefinition(
        "tapir_offline",
        "Offline TAPIR",
        TapirPointConfig,
        TapirPointConfig(),
        "trackleed.analysis.tapir_backend",
        "probe_tapir_environment",
        runtime_mode=TapirRuntimeMode.OFFLINE,
    ),
    PointTrackerDefinition(
        "tapir_online",
        "BootsTAPIR Online / Causal",
        TapirPointConfig,
        TapirPointConfig(runtime_mode=TapirRuntimeMode.ONLINE),
        "trackleed.analysis.tapir_backend",
        "probe_tapir_environment",
        runtime_mode=TapirRuntimeMode.ONLINE,
    ),
    PointTrackerDefinition(
        "locotrack_offline",
        "Offline LocoTrack",
        LocotrackPointConfig,
        LocotrackPointConfig(),
        "trackleed.analysis.locotrack_backend",
        "probe_locotrack_environment",
        requires_repo=True,
        supports_subprocess=True,
    ),
    PointTrackerDefinition(
        "cotracker3_offline",
        "Offline CoTracker3",
        CotrackerPointConfig,
        CotrackerPointConfig(),
        "trackleed.analysis.cotracker_backend",
        "probe_cotracker_environment",
    ),
    PointTrackerDefinition(
        "tapnext_offline",
        "Offline TAPNext",
        TapnextPointConfig,
        TapnextPointConfig(),
        "trackleed.analysis.tapnext_backend",
        "probe_tapnext_environment",
    ),
    PointTrackerDefinition(
        "trackonr_offline",
        "Offline Track-On-R",
        TrackonrPointConfig,
        TrackonrPointConfig(),
        "trackleed.analysis.trackonr_backend",
        "probe_trackonr_environment",
        requires_repo=True,
        supports_subprocess=True,
    ),
    PointTrackerDefinition(
        "pipspp_offline",
        "Offline PIPs++",
        PipsppPointConfig,
        PipsppPointConfig(),
        "trackleed.analysis.pipspp_backend",
        "probe_pipspp_environment",
        requires_repo=True,
        supports_subprocess=True,
    ),
    PointTrackerDefinition(
        "mftiq_offline",
        "Offline MFTIQ",
        MftiqPointConfig,
        MftiqPointConfig(),
        "trackleed.analysis.mftiq_backend",
        "probe_mftiq_environment",
        requires_repo=True,
        requires_config=True,
        supports_subprocess=True,
    ),
)


class PointTrackerRuntime:
    def __init__(self, definition: PointTrackerDefinition) -> None:
        self._definition = definition

    def validate_config(
        self,
        config: SeededTrackingConfig,
    ) -> tuple[TrackingPreflightIssue, ...]:
        if not isinstance(config, self._definition.config_type):
            return (
                TrackingPreflightIssue(
                    "invalid_config",
                    f"{self._definition.display_name} requires its point configuration.",
                ),
            )
        if (
            self._definition.runtime_mode is not None
            and getattr(config, "runtime_mode", None) != self._definition.runtime_mode
        ):
            return (
                TrackingPreflightIssue(
                    "invalid_config",
                    f"{self._definition.display_name} has the wrong TAPIR runtime mode.",
                ),
            )
        config_issues = _config_issues(config, self._definition)
        if config_issues:
            return config_issues
        return ()

    def preflight(
        self,
        config: SeededTrackingConfig,
        context: RunContext | None = None,
    ) -> tuple[TrackingPreflightIssue, ...]:
        config_issues = self.validate_config(config)
        if config_issues:
            return config_issues
        assert isinstance(config, PointTrackerConfig)
        issues: list[TrackingPreflightIssue] = []
        if context is not None and context.point_tracker_preflight is not None:
            issues.extend(
                context.point_tracker_preflight(self._definition.method_id, config)
            )
        else:
            issues.extend(_probe_environment(config, self._definition))
        if context is not None:
            executor = (
                context.point_tracker_retrack_executor
                if context.preflight_operation == "retrack"
                else context.point_tracker_executor
            )
            if executor is None:
                issues.append(
                    TrackingPreflightIssue(
                        "executor_unavailable",
                        f"{self._definition.display_name} execution adapter is unavailable.",
                    )
                )
        return tuple(issues)

    def run(
        self,
        request: SeededTrackingRequest,
        context: RunContext,
    ) -> TrackingResult:
        if context.point_tracker_executor is None:
            raise RuntimeError("Point-tracker execution adapter is unavailable")
        return context.point_tracker_executor(
            self._definition.method_id,
            request,
            context,
        )

    def retrack(
        self,
        request: SeededRetrackRequest,
        context: RunContext,
    ) -> TrackingResult:
        if context.point_tracker_retrack_executor is None:
            raise RuntimeError("Point-tracker re-track adapter is unavailable")
        return context.point_tracker_retrack_executor(
            self._definition.method_id,
            request,
            context,
        )


def point_tracking_definitions() -> tuple[PointTrackerDefinition, ...]:
    return _POINT_METHODS


def point_tracking_definition(method_id: str) -> PointTrackerDefinition:
    for definition in _POINT_METHODS:
        if definition.method_id == method_id:
            return definition
    raise KeyError(f"Unknown point tracker ID: {method_id}")


def default_point_tracker_configs(
    *,
    roi_size: int = 32,
) -> tuple[tuple[str, PointTrackerConfig], ...]:
    return tuple(
        (
            definition.method_id,
            replace(definition.default_config, roi_size=roi_size),
        )
        for definition in _POINT_METHODS
    )


def point_tracking_methods() -> tuple[SeededTrackingMethod, ...]:
    capabilities = SeededTrackingCapabilities(
        SeedRepresentation.POINT,
        supports_background=False,
        preserves_gaps=True,
        supports_retrack=True,
    )
    return tuple(
        SeededTrackingMethod(
            definition.method_id,
            definition.display_name,
            TrackingFamily.POINT,
            capabilities,
            PointTrackerRuntime(definition),
            PointTrackerRuntime(definition),
        )
        for definition in _POINT_METHODS
    )


def _config_issues(
    config: PointTrackerConfig,
    definition: PointTrackerDefinition,
) -> tuple[TrackingPreflightIssue, ...]:
    valid_backend_modes = (
        {BackendMode.DIRECT, BackendMode.SUBPROCESS}
        if definition.supports_subprocess
        else {BackendMode.DIRECT}
    )
    resolution = config.inference_resolution
    resolution_valid = (
        resolution is None
        or isinstance(resolution, int)
        and resolution > 0
        or isinstance(resolution, tuple)
        and len(resolution) == 2
        and all(isinstance(value, int) and value > 0 for value in resolution)
    )
    common_values_valid = (
        config.backend_mode in valid_backend_modes
        and config.roi_size > 0
        and config.window_size > 0
        and 0 <= config.window_overlap < config.window_size
        and config.max_step_distance_px >= 0
        and config.max_reacquisition_gap >= 0
        and config.terminate_after_invisible_frames >= 0
        and config.lateral_angle_tolerance_deg >= 0
        and config.max_radial_jump_px >= 0
        and config.recenter_patch_multiplier > 0
        and 0 <= config.centroid_rel_threshold <= 1
        and config.centroid_max_shift_ratio >= 0
        and all(isfinite(value) for value in config.manual_center)
        and resolution_valid
        and _specific_config_is_valid(config)
    )
    if common_values_valid:
        return ()
    return (
        TrackingPreflightIssue(
            "invalid_config",
            f"{definition.display_name} settings are outside their valid ranges.",
        ),
    )


def _probe_environment(
    config: PointTrackerConfig,
    definition: PointTrackerDefinition,
) -> tuple[TrackingPreflightIssue, ...]:
    try:
        module = import_module(definition.backend_module)
        probe = getattr(module, definition.probe_name)
        status = probe(**_probe_kwargs(config, definition))
    except Exception as exc:  # noqa: BLE001 - optional probes must fail closed
        return (
            TrackingPreflightIssue(
                "environment_probe_failed",
                f"{definition.display_name} environment probe failed: {exc}",
            ),
        )
    issues = list(_structural_environment_issues(config, definition, status))
    if config.backend_mode is BackendMode.SUBPROCESS:
        issues.extend(_subprocess_environment_issues(config, definition))
    if issues:
        return tuple(issues)
    if config.backend_mode is BackendMode.SUBPROCESS:
        return ()
    if not bool(getattr(status, "is_ready", False)):
        return (
            TrackingPreflightIssue(
                "dependency_unavailable",
                _environment_message(definition, status),
            ),
        )
    return ()


def _probe_kwargs(
    config: PointTrackerConfig,
    definition: PointTrackerDefinition,
) -> dict[str, Path | None]:
    kwargs: dict[str, Path | None] = {
        "models_dir": config.models_dir,
        "checkpoint_path": config.checkpoint_path,
    }
    if definition.requires_repo or config.repo_path is not None:
        kwargs["repo_path"] = config.repo_path
    if definition.requires_config:
        kwargs["config_path"] = getattr(config, "config_path", None)
    return kwargs


def _specific_config_is_valid(config: PointTrackerConfig) -> bool:
    if isinstance(config, TapirPointConfig):
        return (
            config.query_chunk_size > 0
            and 0 <= config.visibility_threshold <= 1
            and config.expected_dist_threshold >= 0
        )
    if isinstance(config, LocotrackPointConfig):
        return (
            config.query_chunk_size > 0
            and config.subprocess_timeout_sec > 0
            and 0 <= config.visibility_threshold <= 1
            and config.expected_dist_threshold >= 0
        )
    if isinstance(config, CotrackerPointConfig):
        return config.window_len > 0
    if isinstance(config, TapnextPointConfig):
        return (
            0 <= config.visibility_threshold <= 1
            and 0 <= config.certainty_threshold <= 1
        )
    if isinstance(config, TrackonrPointConfig):
        return (
            config.support_grid_size > 0
            and config.subprocess_timeout_sec > 0
            and 0 <= config.visibility_threshold <= 1
        )
    if isinstance(config, PipsppPointConfig):
        return config.iters > 0 and config.subprocess_timeout_sec > 0
    if isinstance(config, MftiqPointConfig):
        return (
            0 <= config.occlusion_threshold <= 1 and config.subprocess_timeout_sec > 0
        )
    return False


def _structural_environment_issues(
    config: PointTrackerConfig,
    definition: PointTrackerDefinition,
    status: Any,
) -> tuple[TrackingPreflightIssue, ...]:
    issues: list[TrackingPreflightIssue] = []
    if not bool(getattr(status, "checkpoint_exists", False)):
        issues.append(
            TrackingPreflightIssue(
                "checkpoint_unavailable",
                f"{definition.display_name} checkpoint is unavailable.",
            )
        )
    if definition.requires_repo and getattr(status, "repo_path", None) is None:
        issues.append(
            TrackingPreflightIssue(
                "repository_unavailable",
                f"{definition.display_name} local repository is unavailable.",
            )
        )
    if definition.requires_config and not bool(getattr(status, "config_exists", False)):
        issues.append(
            TrackingPreflightIssue(
                "config_unavailable",
                f"{definition.display_name} runtime configuration is unavailable.",
            )
        )
    if definition.runtime_mode is TapirRuntimeMode.ONLINE and not bool(
        getattr(status, "selected_checkpoint_is_online", False)
    ):
        issues.append(
            TrackingPreflightIssue(
                "online_checkpoint_required",
                "BootsTAPIR Online requires a causal / online checkpoint.",
            )
        )
    return tuple(issues)


def _subprocess_environment_issues(
    config: PointTrackerConfig,
    definition: PointTrackerDefinition,
) -> tuple[TrackingPreflightIssue, ...]:
    executable = Path(config.python_executable or sys.executable).expanduser()
    if not executable.is_file():
        return (_interpreter_issue(definition),)
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / ("probe_seeded_point_subprocess.py")
    )
    kwargs = {
        name: None if value is None else str(value)
        for name, value in _probe_kwargs(config, definition).items()
    }
    try:
        completed = subprocess.run(
            (
                str(executable),
                str(script_path),
                definition.backend_module,
                definition.probe_name,
                json.dumps(kwargs),
            ),
            cwd=script_path.parents[1],
            capture_output=True,
            text=True,
            check=False,
            timeout=60.0,
        )
    except (OSError, subprocess.SubprocessError):
        return (_interpreter_issue(definition),)
    if completed.returncode == 0:
        return ()
    details = completed.stdout.strip() or completed.stderr.strip()
    return (
        TrackingPreflightIssue(
            "dependency_unavailable",
            f"{definition.display_name} subprocess dependencies are unavailable"
            f"{': ' + details if details else '.'}",
        ),
    )


def _interpreter_issue(
    definition: PointTrackerDefinition,
) -> TrackingPreflightIssue:
    return TrackingPreflightIssue(
        "interpreter_unavailable",
        f"{definition.display_name} subprocess interpreter is unavailable.",
    )


def _environment_message(definition: PointTrackerDefinition, status: Any) -> str:
    details = tuple(getattr(status, "errors", ())) + tuple(
        getattr(status, "warnings", ())
    )
    if not details:
        return f"{definition.display_name} optional dependencies are unavailable."
    return f"{definition.display_name} is unavailable: {'; '.join(details)}"
