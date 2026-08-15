"""Headless catalog adapters for mask-family seeded tracking."""

from __future__ import annotations

import sys
import subprocess
from dataclasses import dataclass, replace
from importlib import import_module
from math import isfinite
from pathlib import Path
from typing import Any

from trackleed.logic.workflows.mask_tracking import (
    CutieMaskConfig,
    Dam4SamMaskConfig,
    DevaMaskConfig,
    InitialMaskMode,
    MaskBackendMode,
    MaskCenterMode,
    MaskDeviceMode,
    MaskDirectionMode,
    MaskRepresentativePointMode,
    MaskTrackerConfig,
    Sam2PromptMode,
    Sam2MaskConfig,
    SamuraiMaskConfig,
    Xmem2MaskConfig,
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
class MaskTrackerDefinition:
    method_id: str
    display_name: str
    config_type: type[MaskTrackerConfig]
    default_config: MaskTrackerConfig
    backend_module: str
    probe_name: str
    requires_polygon_seed: bool = False
    supports_subprocess: bool = True


_MASK_METHODS = (
    MaskTrackerDefinition(
        "cutie_offline",
        "Offline Cutie",
        CutieMaskConfig,
        CutieMaskConfig(),
        "trackleed.analysis.cutie_backend",
        "probe_cutie_environment",
        requires_polygon_seed=True,
    ),
    MaskTrackerDefinition(
        "xmem2_offline",
        "Offline XMem2",
        Xmem2MaskConfig,
        Xmem2MaskConfig(),
        "trackleed.analysis.xmem2_backend",
        "probe_xmem2_environment",
        requires_polygon_seed=True,
    ),
    MaskTrackerDefinition(
        "sam2_offline",
        "Offline SAM2",
        Sam2MaskConfig,
        Sam2MaskConfig(),
        "trackleed.analysis.sam2_backend",
        "probe_sam2_environment",
    ),
    MaskTrackerDefinition(
        "dam4sam_offline",
        "Offline DAM4SAM",
        Dam4SamMaskConfig,
        Dam4SamMaskConfig(),
        "trackleed.analysis.dam4sam_backend",
        "probe_dam4sam_environment",
    ),
    MaskTrackerDefinition(
        "samurai_offline",
        "Offline SAMURAI",
        SamuraiMaskConfig,
        SamuraiMaskConfig(),
        "trackleed.analysis.samurai_backend",
        "probe_samurai_environment",
    ),
    MaskTrackerDefinition(
        "deva_offline",
        "Offline DEVA",
        DevaMaskConfig,
        DevaMaskConfig(),
        "trackleed.analysis.deva_backend",
        "probe_deva_environment",
    ),
)


class MaskTrackerRuntime:
    def __init__(self, definition: MaskTrackerDefinition) -> None:
        self._definition = definition

    def validate_config(
        self,
        config: SeededTrackingConfig,
    ) -> tuple[TrackingPreflightIssue, ...]:
        if not isinstance(config, self._definition.config_type):
            return (
                TrackingPreflightIssue(
                    "invalid_config",
                    f"{self._definition.display_name} requires its mask configuration.",
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
        assert isinstance(config, MaskTrackerConfig)
        if context is not None and context.mask_tracker_preflight is not None:
            issues = context.mask_tracker_preflight(self._definition.method_id, config)
        else:
            issues = _probe_environment(config, self._definition)
        if context is None:
            return issues
        executor = (
            context.mask_tracker_retrack_executor
            if context.preflight_operation == "retrack"
            else context.mask_tracker_executor
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
        if context.mask_tracker_executor is None:
            raise RuntimeError("Mask execution adapter is unavailable")
        return context.mask_tracker_executor(self._definition.method_id, request, context)

    def retrack(
        self,
        request: SeededRetrackRequest,
        context: RunContext,
    ) -> TrackingResult:
        if context.mask_tracker_retrack_executor is None:
            raise RuntimeError("Mask re-track adapter is unavailable")
        return context.mask_tracker_retrack_executor(
            self._definition.method_id,
            request,
            context,
        )


def mask_tracking_definitions() -> tuple[MaskTrackerDefinition, ...]:
    return _MASK_METHODS


def mask_tracking_definition(method_id: str) -> MaskTrackerDefinition:
    for definition in _MASK_METHODS:
        if definition.method_id == method_id:
            return definition
    raise KeyError(f"Unknown mask tracker ID: {method_id}")


def default_mask_tracker_configs(
    *,
    roi_size: int = 32,
) -> tuple[tuple[str, MaskTrackerConfig], ...]:
    return tuple(
        (
            definition.method_id,
            replace(definition.default_config, roi_size=roi_size),
        )
        for definition in _MASK_METHODS
    )


def mask_tracking_methods() -> tuple[SeededTrackingMethod, ...]:
    return tuple(
        SeededTrackingMethod(
            definition.method_id,
            definition.display_name,
            TrackingFamily.MASK,
            SeededTrackingCapabilities(
                SeedRepresentation.MASK,
                supports_background=False,
                preserves_gaps=True,
                supports_retrack=True,
                requires_polygon_seed=definition.requires_polygon_seed,
            ),
            MaskTrackerRuntime(definition),
            MaskTrackerRuntime(definition),
        )
        for definition in _MASK_METHODS
    )


def _config_issues(
    config: MaskTrackerConfig,
    definition: MaskTrackerDefinition,
) -> tuple[TrackingPreflightIssue, ...]:
    try:
        valid = _common_config_valid(config) and _variant_config_valid(config)
    except (TypeError, ValueError):
        valid = False
    if valid:
        return ()
    return (
        TrackingPreflightIssue(
            "invalid_config",
            f"{definition.display_name} settings are outside their valid ranges.",
        ),
    )


def _common_config_valid(config: MaskTrackerConfig) -> bool:
    return (
        isinstance(config.models_dir, Path)
        and isinstance(config.backend_mode, MaskBackendMode)
        and isinstance(config.device, MaskDeviceMode)
        and isinstance(config.representative_point_mode, MaskRepresentativePointMode)
        and config.roi_size > 0
        and config.window_size > 0
        and 0 <= config.window_overlap < config.window_size
        and config.initial_bbox_padding_px >= 0
        and config.subprocess_timeout_sec > 0
        and config.max_step_distance_px >= 0
        and config.max_reacquisition_gap >= 0
        and config.terminate_after_invisible_frames >= 0
        and config.max_mask_area_ratio_change >= 0
        and config.max_bbox_aspect_ratio_change >= 0
        and config.max_mask_components >= 0
        and config.max_bbox_center_jump_px >= 0
        and isinstance(config.center_mode, MaskCenterMode)
        and _manual_center_valid(config.manual_center)
        and isinstance(config.direction_mode, MaskDirectionMode)
        and config.lateral_angle_tolerance_deg >= 0
        and config.max_radial_jump_px >= 0
    )


def _variant_config_valid(config: MaskTrackerConfig) -> bool:
    if isinstance(config, (CutieMaskConfig, Xmem2MaskConfig)):
        return (
            _resolution_valid(config.inference_resolution)
            and isinstance(config.initial_mask_mode, InitialMaskMode)
            and 0 <= config.initial_mask_threshold_rel <= 1
            and 0 < config.initial_mask_disk_radius_ratio <= 1
            and (config.mem_every is None or config.mem_every > 0)
        )
    if isinstance(config, Sam2MaskConfig):
        return (
            _resolution_valid(config.inference_resolution)
            and isinstance(config.prompt_mode, Sam2PromptMode)
            and 0 <= config.mask_threshold <= 1
            and config.min_mask_area >= 0
            and (config.max_mask_area is None or config.max_mask_area >= config.min_mask_area)
        )
    if isinstance(config, Dam4SamMaskConfig):
        return _resolution_valid(config.inference_resolution) and bool(
            config.tracker_name.strip()
        )
    if isinstance(config, SamuraiMaskConfig):
        return _resolution_valid(config.inference_resolution) and bool(
            config.model_variant.strip()
        )
    if isinstance(config, DevaMaskConfig):
        return (
            (config.inference_size is None or config.inference_size > 0)
            and config.chunk_size > 0
            and config.initial_point_radius_px > 0
        )
    return False


def _resolution_valid(value: Any) -> bool:
    return (
        value is None
        or isinstance(value, int)
        and value > 0
        or isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, int) and item > 0 for item in value)
    )


def _manual_center_valid(value: tuple[float, float]) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, (int, float)) and isfinite(item) for item in value)
    )


def _probe_environment(
    config: MaskTrackerConfig,
    definition: MaskTrackerDefinition,
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
    if not bool(getattr(status, "is_ready", False)):
        if (
            config.backend_mode is MaskBackendMode.SUBPROCESS
            and config.python_executable is not None
        ):
            issues.extend(_subprocess_environment_issues(config, definition))
        else:
            issues.append(
                _issue("dependency_unavailable", definition, "runtime dependencies")
            )
    return tuple(issues)


def _probe_kwargs(config: MaskTrackerConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "models_dir": config.models_dir,
        "checkpoint_path": config.checkpoint_path,
        "repo_path": config.repo_path,
    }
    if isinstance(config, Sam2MaskConfig):
        kwargs["config_path"] = config.config_path
    elif isinstance(config, Dam4SamMaskConfig):
        kwargs.update(
            python_executable=_probe_interpreter(config),
            tracker_name=config.tracker_name,
        )
    elif isinstance(config, SamuraiMaskConfig):
        kwargs.update(
            python_executable=_probe_interpreter(config),
            model_variant=config.model_variant,
        )
    return kwargs


def _structural_issues(
    config: MaskTrackerConfig,
    definition: MaskTrackerDefinition,
    status: Any,
) -> tuple[TrackingPreflightIssue, ...]:
    issues: list[TrackingPreflightIssue] = []
    if getattr(status, "repo_path", None) is None:
        issues.append(_issue("repository_unavailable", definition, "repository"))
    if not bool(getattr(status, "checkpoint_exists", False)):
        issues.append(_issue("checkpoint_unavailable", definition, "checkpoint"))
    if isinstance(config, (Sam2MaskConfig, Dam4SamMaskConfig, SamuraiMaskConfig)):
        if not bool(getattr(status, "config_exists", False)):
            issues.append(_issue("config_unavailable", definition, "configuration"))
    if (
        config.backend_mode.value == "subprocess"
        and not _interpreter_path(config.python_executable).is_file()
    ):
        issues.append(_issue("interpreter_unavailable", definition, "interpreter"))
    return tuple(issues)


def _probe_interpreter(config: MaskTrackerConfig) -> Path | None:
    if config.backend_mode.value == "subprocess":
        return _interpreter_path(config.python_executable)
    return None


def _subprocess_environment_issues(
    config: MaskTrackerConfig,
    definition: MaskTrackerDefinition,
) -> tuple[TrackingPreflightIssue, ...]:
    executable = _interpreter_path(config.python_executable)
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / f"check_{definition.method_id.removesuffix('_offline')}_env.py"
    )
    if not executable.is_file() or not script_path.is_file():
        return (_issue("interpreter_unavailable", definition, "interpreter"),)
    try:
        completed = subprocess.run(
            _subprocess_probe_command(executable, script_path, config),
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
    return (_issue("dependency_unavailable", definition, "runtime dependencies"),)


def _subprocess_probe_command(
    executable: Path,
    script_path: Path,
    config: MaskTrackerConfig,
) -> tuple[str, ...]:
    command = [
        str(executable),
        str(script_path),
        "--models-dir",
        str(config.models_dir),
    ]
    for option, value in (
        ("--repo-path", config.repo_path),
        ("--checkpoint", config.checkpoint_path),
    ):
        if value is not None:
            command.extend((option, str(value)))
    if isinstance(config, Sam2MaskConfig) and config.config_path is not None:
        command.extend(("--config", str(config.config_path)))
    elif isinstance(config, Dam4SamMaskConfig):
        command.extend(("--python-executable", str(executable)))
        command.extend(("--tracker-name", config.tracker_name))
    elif isinstance(config, SamuraiMaskConfig):
        command.extend(("--python-executable", str(executable)))
        command.extend(("--model-variant", config.model_variant))
    return tuple(command)


def _interpreter_path(path: Path | None) -> Path:
    return Path(path or sys.executable).expanduser()


def _issue(
    code: str,
    definition: MaskTrackerDefinition,
    resource: str,
) -> TrackingPreflightIssue:
    return TrackingPreflightIssue(
        code,
        f"{definition.display_name} {resource} is unavailable.",
    )
