"""Qt/model compatibility boundary for mask tracking workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any

from trackleed.core.data_models import Track
from trackleed.gui.seeded_tracking_legacy_models import (
    ContextCancellationSignal,
    legacy_files,
    legacy_spot,
    legacy_track,
    legacy_tracker_gui_spec,
    neutral_result,
    plain_legacy_value,
    progress_callback,
    raise_if_cancelled,
    raise_none_result,
)
from trackleed.logic.workflows.mask_tracking import (
    Dam4SamMaskConfig,
    MaskTrackerConfig,
    SamuraiMaskConfig,
)
from trackleed.logic.workflows.seeded_tracking import (
    RunContext,
    SeededRetrackRequest,
    SeededTrackingRequest,
    TrackingResult,
)


@dataclass(frozen=True)
class MaskTrackerLegacyBinding:
    method_id: str
    tracking_runner_name: str
    retrack_runner_name: str
    review_argument_name: str
    source_names: tuple[str, ...]
    settings_type_name: str
    runtime_settings_type_name: str
    collect_params_method_name: str
    settings_attr_name: str
    panel_group_attr: str
    warning_label_attr: str
    backend_mode_widget_attr: str
    repo_path_attr: str
    python_executable_attr: str
    default_conda_env_name: str
    checkpoint_selector_attr: str
    variant_selector_attr: str | None = None


@dataclass(frozen=True)
class MaskReviewSpec:
    tracker_id: str
    has_visualizer_retrack: bool = True


_MASK_BINDINGS = (
    MaskTrackerLegacyBinding(
        "cutie_offline",
        "track_spots_cutie_offline",
        "retrack_single_spot_cutie_offline",
        "cutie_params",
        ("cutie_offline",),
        "CutieTrackingSettings",
        "CutieRuntimeSettings",
        "_collect_cutie_tracking_params",
        "cutie_settings",
        "cutie_group",
        "cutie_warning_label",
        "cutie_backend_mode_combo",
        "cutie_repo_path_edit",
        "cutie_python_executable_edit",
        "cutie",
        "cutie_checkpoint_combo",
    ),
    MaskTrackerLegacyBinding(
        "xmem2_offline",
        "track_spots_xmem2_offline",
        "retrack_single_spot_xmem2_offline",
        "xmem2_params",
        ("xmem2_offline",),
        "XMem2TrackingSettings",
        "XMem2RuntimeSettings",
        "_collect_xmem2_tracking_params",
        "xmem2_settings",
        "xmem2_group",
        "xmem2_warning_label",
        "xmem2_backend_mode_combo",
        "xmem2_repo_path_edit",
        "xmem2_python_executable_edit",
        "xmem2",
        "xmem2_checkpoint_combo",
    ),
    MaskTrackerLegacyBinding(
        "sam2_offline",
        "track_spots_sam2_offline",
        "retrack_single_spot_sam2_offline",
        "sam2_params",
        ("sam2_offline",),
        "Sam2TrackingSettings",
        "Sam2RuntimeSettings",
        "_collect_sam2_tracking_params",
        "sam2_settings",
        "sam2_group",
        "sam2_warning_label",
        "sam2_backend_mode_combo",
        "sam2_repo_path_edit",
        "sam2_python_executable_edit",
        "sam2",
        "sam2_checkpoint_combo",
    ),
    MaskTrackerLegacyBinding(
        "dam4sam_offline",
        "track_spots_dam4sam_offline",
        "retrack_single_spot_dam4sam_offline",
        "dam4sam_params",
        ("dam4sam_offline",),
        "Dam4SamTrackingSettings",
        "Dam4SamRuntimeSettings",
        "_collect_dam4sam_tracking_params",
        "dam4sam_settings",
        "dam4sam_group",
        "dam4sam_warning_label",
        "dam4sam_backend_mode_combo",
        "dam4sam_repo_path_edit",
        "dam4sam_python_executable_edit",
        "dam4sam",
        "dam4sam_checkpoint_combo",
        "dam4sam_tracker_variant_combo",
    ),
    MaskTrackerLegacyBinding(
        "samurai_offline",
        "track_spots_samurai_offline",
        "retrack_single_spot_samurai_offline",
        "samurai_params",
        ("samurai_offline",),
        "SamuraiTrackingSettings",
        "SamuraiRuntimeSettings",
        "_collect_samurai_tracking_params",
        "samurai_settings",
        "samurai_group",
        "samurai_warning_label",
        "samurai_backend_mode_combo",
        "samurai_repo_path_edit",
        "samurai_python_executable_edit",
        "samurai",
        "samurai_checkpoint_combo",
        "samurai_model_variant_combo",
    ),
    MaskTrackerLegacyBinding(
        "deva_offline",
        "track_spots_deva_offline",
        "retrack_single_spot_deva_offline",
        "deva_params",
        ("deva_offline",),
        "DevaTrackingSettings",
        "DevaRuntimeSettings",
        "_collect_deva_tracking_params",
        "deva_settings",
        "deva_group",
        "deva_warning_label",
        "deva_backend_mode_combo",
        "deva_repo_path_edit",
        "deva_python_executable_edit",
        "deva",
        "deva_checkpoint_combo",
    ),
)


def mask_tracker_legacy_bindings() -> tuple[MaskTrackerLegacyBinding, ...]:
    return _MASK_BINDINGS


def mask_tracker_legacy_binding(method_id: str) -> MaskTrackerLegacyBinding:
    normalized = str(method_id).strip().lower()
    for binding in _MASK_BINDINGS:
        if binding.method_id == normalized:
            return binding
    raise KeyError(f"Unknown mask tracker ID: {method_id}")


def mask_review_spec_for_source(source_name: str) -> MaskReviewSpec | None:
    normalized = str(source_name).strip().lower()
    for binding in _MASK_BINDINGS:
        if normalized in binding.source_names:
            return MaskReviewSpec(binding.method_id)
    return None


def mask_tracker_gui_spec(method_id: str) -> Any:
    """Expose legacy panel metadata without re-registering seeded methods."""
    from trackleed.analysis.seeded_mask_tracking_runtime import (
        mask_tracking_definition,
    )

    definition = mask_tracking_definition(method_id)
    binding = mask_tracker_legacy_binding(method_id)
    extra_fields = ("variant_selector_attr",) if binding.variant_selector_attr else ()
    return legacy_tracker_gui_spec(
        definition,
        binding,
        family="mask",
        source_names=binding.source_names,
        tracking_module_name="trackleed.analysis.tracking_mask_backends",
        extra_binding_fields=extra_fields,
    )


def legacy_mask_tracker_params(
    method_id: str,
    config: MaskTrackerConfig,
) -> dict[str, Any]:
    params = {
        key: plain_legacy_value(value) for key, value in asdict(config).items()
    }
    prefix = method_id.removesuffix("_offline")
    repo_path = params.get("repo_path")
    params[f"{prefix}_models_dir"] = params.pop("models_dir")
    params[f"{prefix}_repo_path"] = repo_path
    if isinstance(config, Dam4SamMaskConfig):
        params["dam4sam_tracker_name"] = params["tracker_name"]
    elif isinstance(config, SamuraiMaskConfig):
        params["samurai_model_variant"] = params["model_variant"]
    return params


def run_legacy_mask_tracker(
    method_id: str,
    request: SeededTrackingRequest,
    context: RunContext,
    *,
    runner: Callable[..., list[Track] | None] | None = None,
) -> TrackingResult:
    """Run an existing mask backend behind the GUI/model boundary."""
    raise_if_cancelled(context)
    if not isinstance(request.config, MaskTrackerConfig):
        raise TypeError("Mask tracking requires its typed configuration")
    files_by_id = legacy_files(request.frames, context)
    files_by_id[request.reference_frame_id].spots = [
        legacy_spot(
            seed.seed_id,
            seed.roi,
            seed.center_x,
            seed.center_y,
            seed.intensity,
            seed.source,
            seed.polygon_seed_points,
        )
        for seed in request.seeds
    ]
    frame_ids = [frame.frame_id for frame in request.frames]
    selected_runner = runner or _legacy_runner(method_id, retrack=False)
    tracks = selected_runner(
        files_by_id,
        frame_ids,
        request.reference_frame_id,
        legacy_mask_tracker_params(method_id, request.config),
        progress_callback=progress_callback(context, frame_ids[-1]),
        cancel_event=ContextCancellationSignal(context),
    )
    if tracks is None:
        raise_none_result(method_id, context)
    return neutral_result(tracks, frame_ids, family_label="Mask")


def run_legacy_mask_retracker(
    method_id: str,
    request: SeededRetrackRequest,
    context: RunContext,
    *,
    runner: Callable[..., Track | None] | None = None,
) -> TrackingResult:
    """Run an existing mask re-tracker at the GUI/model boundary."""
    raise_if_cancelled(context)
    if not isinstance(request.config, MaskTrackerConfig):
        raise TypeError("Mask re-tracking requires its typed configuration")
    files_by_id = legacy_files(request.frames, context)
    frame_ids = [frame.frame_id for frame in request.frames]
    selected_runner = runner or _legacy_runner(method_id, retrack=True)
    track = selected_runner(
        legacy_track(request.track),
        request.start_frame_id,
        files_by_id,
        frame_ids,
        legacy_mask_tracker_params(method_id, request.config),
        progress_callback=progress_callback(context, frame_ids[-1]),
        cancel_event=ContextCancellationSignal(context),
    )
    if track is None:
        raise_none_result(method_id, context)
    return neutral_result([track], frame_ids, family_label="Mask")


def _legacy_runner(method_id: str, *, retrack: bool) -> Callable[..., Any]:
    binding = mask_tracker_legacy_binding(method_id)
    module_name = (
        "trackleed.analysis.tracking_retrack"
        if retrack
        else "trackleed.analysis.tracking_mask_backends"
    )
    function_name = (
        binding.retrack_runner_name if retrack else binding.tracking_runner_name
    )
    return getattr(import_module(module_name), function_name)
