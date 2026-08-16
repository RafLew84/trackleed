"""Qt/model compatibility boundary for mask tracking workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any

from trackleed.core.data_models import SpotROI, Track
from trackleed.gui.seeded_tracking_legacy_models import (
    LegacySeededFamily,
    legacy_spot,
    plain_legacy_value,
    run_legacy_seeded,
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
    TrackingSeed,
)


@dataclass(frozen=True)
class MaskTrackerLegacyBinding:
    method_id: str
    tracking_runner_name: str
    retrack_runner_name: str
    source_names: tuple[str, ...]


@dataclass(frozen=True)
class MaskReviewSpec:
    tracker_id: str
    has_visualizer_retrack: bool = True


_MASK_BINDINGS = (
    MaskTrackerLegacyBinding(
        "cutie_offline", "track_spots_cutie_offline",
        "retrack_single_spot_cutie_offline", ("cutie_offline",),
    ),
    MaskTrackerLegacyBinding(
        "xmem2_offline", "track_spots_xmem2_offline",
        "retrack_single_spot_xmem2_offline", ("xmem2_offline",),
    ),
    MaskTrackerLegacyBinding(
        "sam2_offline", "track_spots_sam2_offline",
        "retrack_single_spot_sam2_offline", ("sam2_offline",),
    ),
    MaskTrackerLegacyBinding(
        "dam4sam_offline", "track_spots_dam4sam_offline",
        "retrack_single_spot_dam4sam_offline", ("dam4sam_offline",),
    ),
    MaskTrackerLegacyBinding(
        "samurai_offline", "track_spots_samurai_offline",
        "retrack_single_spot_samurai_offline", ("samurai_offline",),
    ),
    MaskTrackerLegacyBinding(
        "deva_offline", "track_spots_deva_offline",
        "retrack_single_spot_deva_offline", ("deva_offline",),
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
    return run_legacy_seeded(
        method_id, request, context, _MASK_FAMILY, runner=runner
    )


def run_legacy_mask_retracker(
    method_id: str,
    request: SeededRetrackRequest,
    context: RunContext,
    *,
    runner: Callable[..., Track | None] | None = None,
) -> TrackingResult:
    """Run an existing mask re-tracker at the GUI/model boundary."""
    return run_legacy_seeded(
        method_id, request, context, _MASK_FAMILY, runner=runner
    )


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


def _mask_parameters(method_id: str, config: object) -> dict[str, Any]:
    if not isinstance(config, MaskTrackerConfig):
        raise TypeError("Mask tracking requires its typed configuration")
    return legacy_mask_tracker_params(method_id, config)


def _mask_seed(seed: TrackingSeed) -> SpotROI:
    return legacy_spot(
        seed.seed_id,
        seed.roi,
        seed.center_x,
        seed.center_y,
        seed.intensity,
        seed.source,
        seed.polygon_seed_points,
    )


_MASK_FAMILY = LegacySeededFamily(
    "Mask",
    _mask_parameters,
    _mask_seed,
    _legacy_runner,
)
