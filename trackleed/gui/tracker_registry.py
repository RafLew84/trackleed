from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from trackleed.analysis.cotracker_backend import (
    format_cotracker_environment_report,
    probe_cotracker_environment,
)
from trackleed.analysis.cutie_backend import (
    format_cutie_environment_report,
    probe_cutie_environment,
)
from trackleed.analysis.dam4sam_backend import (
    format_dam4sam_environment_report,
    probe_dam4sam_environment,
)
from trackleed.analysis.deva_backend import (
    format_deva_environment_report,
    probe_deva_environment,
)
from trackleed.analysis.locotrack_backend import (
    format_locotrack_environment_report,
    probe_locotrack_environment,
)
from trackleed.analysis.mftiq_backend import (
    format_mftiq_environment_report,
    probe_mftiq_environment,
)
from trackleed.analysis.ostrack_backend import (
    format_ostrack_environment_report,
    probe_ostrack_environment,
)
from trackleed.analysis.pipspp_backend import (
    format_pipspp_environment_report,
    probe_pipspp_environment,
)
from trackleed.analysis.sam2_backend import (
    format_sam2_environment_report,
    probe_sam2_environment,
)
from trackleed.analysis.samurai_backend import (
    format_samurai_environment_report,
    probe_samurai_environment,
)
from trackleed.analysis.tapir_backend import (
    format_tapir_environment_report,
    probe_tapir_environment,
)
from trackleed.analysis.tapnext_backend import (
    format_tapnext_environment_report,
    probe_tapnext_environment,
)
from trackleed.analysis.tomp_backend import (
    format_tomp_environment_report,
    probe_tomp_environment,
)
from trackleed.analysis.trackonr_backend import (
    format_trackonr_environment_report,
    probe_trackonr_environment,
)
from trackleed.analysis.tracking_bbox_backends import (
    track_spots_ostrack_offline,
    track_spots_tomp_offline,
)
from trackleed.analysis.tracking_mask_backends import (
    track_spots_cutie_offline,
    track_spots_dam4sam_offline,
    track_spots_deva_offline,
    track_spots_sam2_offline,
    track_spots_samurai_offline,
    track_spots_xmem2_offline,
)
from trackleed.analysis.tracking_point_backends import (
    track_spots_cotracker3_offline,
    track_spots_locotrack_offline,
    track_spots_mftiq_offline,
    track_spots_pipspp_offline,
    track_spots_tapir_offline,
    track_spots_tapir_online,
    track_spots_tapnext_offline,
    track_spots_trackonr_offline,
)
from trackleed.analysis.tracking_retrack import (
    retrack_single_spot_cotracker3_offline,
    retrack_single_spot_cutie_offline,
    retrack_single_spot_dam4sam_offline,
    retrack_single_spot_deva_offline,
    retrack_single_spot_locotrack_offline,
    retrack_single_spot_mftiq_offline,
    retrack_single_spot_ostrack_offline,
    retrack_single_spot_pipspp_offline,
    retrack_single_spot_sam2_offline,
    retrack_single_spot_samurai_offline,
    retrack_single_spot_tapir_offline,
    retrack_single_spot_tapir_online,
    retrack_single_spot_tapnext_offline,
    retrack_single_spot_tomp_offline,
    retrack_single_spot_trackonr_offline,
    retrack_single_spot_xmem2_offline,
)
from trackleed.analysis.xmem2_backend import (
    format_xmem2_environment_report,
    probe_xmem2_environment,
)


TrackerFamily = Literal["point", "bbox", "mask"]


@dataclass(frozen=True)
class TrackerSpec:
    tracker_id: str
    display_name: str
    scenario_name: str
    family: TrackerFamily
    source_names: tuple[str, ...]
    settings_type_name: str
    runtime_settings_type_name: str
    probe_environment: Callable[..., Any]
    format_environment_report: Callable[[Any], str]
    collect_params_method_name: str
    run_tracking: Callable[..., Any]
    run_retrack: Callable[..., Any]
    supports_direct: bool
    supports_subprocess: bool
    has_panel: bool = True
    has_visualizer_retrack: bool = True
    settings_attr_name: str | None = None
    params_argument_name: str | None = None
    panel_group_attr: str | None = None
    warning_label_attr: str | None = None
    backend_mode_widget_attr: str | None = None
    repo_path_attr: str | None = None
    python_executable_attr: str | None = None
    default_conda_env_name: str | None = None
    checkpoint_selector_attr: str | None = None
    variant_selector_attr: str | None = None


TRACKER_SPECS: tuple[TrackerSpec, ...] = (
    TrackerSpec(
        tracker_id="tapir_offline",
        display_name="Offline TAPIR",
        scenario_name="Offline TAPIR",
        family="point",
        source_names=("tapir_offline",),
        settings_type_name="TapirTrackingSettings",
        runtime_settings_type_name="TapirRuntimeSettings",
        probe_environment=probe_tapir_environment,
        format_environment_report=format_tapir_environment_report,
        collect_params_method_name="_collect_tapir_tracking_params",
        run_tracking=track_spots_tapir_offline,
        run_retrack=retrack_single_spot_tapir_offline,
        supports_direct=True,
        supports_subprocess=False,
        settings_attr_name="tapir_settings",
        params_argument_name="tapir_params",
        panel_group_attr="tapir_group",
        warning_label_attr="tapir_warning_label",
        checkpoint_selector_attr="tapir_checkpoint_combo",
    ),
    TrackerSpec(
        tracker_id="tapir_online",
        display_name="BootsTAPIR Online / Causal",
        scenario_name="BootsTAPIR Online / Causal",
        family="point",
        source_names=("tapir_online",),
        settings_type_name="TapirTrackingSettings",
        runtime_settings_type_name="TapirOnlineRuntimeSettings",
        probe_environment=probe_tapir_environment,
        format_environment_report=format_tapir_environment_report,
        collect_params_method_name="_collect_tapir_tracking_params",
        run_tracking=track_spots_tapir_online,
        run_retrack=retrack_single_spot_tapir_online,
        supports_direct=True,
        supports_subprocess=False,
        settings_attr_name="tapir_settings",
        params_argument_name="tapir_params",
        panel_group_attr="tapir_group",
        warning_label_attr="tapir_warning_label",
        checkpoint_selector_attr="tapir_checkpoint_combo",
    ),
    TrackerSpec(
        tracker_id="locotrack_offline",
        display_name="Offline LocoTrack",
        scenario_name="Offline LocoTrack",
        family="point",
        source_names=("locotrack_offline",),
        settings_type_name="LocotrackTrackingSettings",
        runtime_settings_type_name="LocotrackRuntimeSettings",
        probe_environment=probe_locotrack_environment,
        format_environment_report=format_locotrack_environment_report,
        collect_params_method_name="_collect_locotrack_tracking_params",
        run_tracking=track_spots_locotrack_offline,
        run_retrack=retrack_single_spot_locotrack_offline,
        supports_direct=True,
        supports_subprocess=True,
        settings_attr_name="locotrack_settings",
        params_argument_name="locotrack_params",
        panel_group_attr="locotrack_group",
        warning_label_attr="locotrack_warning_label",
        backend_mode_widget_attr="locotrack_backend_mode_combo",
        repo_path_attr="locotrack_repo_path_edit",
        python_executable_attr="locotrack_python_executable_edit",
        default_conda_env_name="locotrack-pytorch",
        checkpoint_selector_attr="locotrack_checkpoint_combo",
    ),
    TrackerSpec(
        tracker_id="cotracker3_offline",
        display_name="Offline CoTracker3",
        scenario_name="Offline CoTracker3",
        family="point",
        source_names=("cotracker3_offline",),
        settings_type_name="CotrackerTrackingSettings",
        runtime_settings_type_name="CotrackerRuntimeSettings",
        probe_environment=probe_cotracker_environment,
        format_environment_report=format_cotracker_environment_report,
        collect_params_method_name="_collect_cotracker_tracking_params",
        run_tracking=track_spots_cotracker3_offline,
        run_retrack=retrack_single_spot_cotracker3_offline,
        supports_direct=True,
        supports_subprocess=False,
        settings_attr_name="cotracker_settings",
        params_argument_name="cotracker_params",
        panel_group_attr="cotracker_group",
        warning_label_attr="cotracker_warning_label",
        checkpoint_selector_attr="cotracker_checkpoint_combo",
    ),
    TrackerSpec(
        tracker_id="tapnext_offline",
        display_name="Offline TAPNext",
        scenario_name="Offline TAPNext",
        family="point",
        source_names=("tapnext_offline",),
        settings_type_name="TapnextTrackingSettings",
        runtime_settings_type_name="TapnextRuntimeSettings",
        probe_environment=probe_tapnext_environment,
        format_environment_report=format_tapnext_environment_report,
        collect_params_method_name="_collect_tapnext_tracking_params",
        run_tracking=track_spots_tapnext_offline,
        run_retrack=retrack_single_spot_tapnext_offline,
        supports_direct=True,
        supports_subprocess=False,
        settings_attr_name="tapnext_settings",
        params_argument_name="tapnext_params",
        panel_group_attr="tapnext_group",
        warning_label_attr="tapnext_warning_label",
        checkpoint_selector_attr="tapnext_checkpoint_combo",
    ),
    TrackerSpec(
        tracker_id="trackonr_offline",
        display_name="Offline Track-On-R",
        scenario_name="Offline Track-On-R",
        family="point",
        source_names=("trackonr_offline",),
        settings_type_name="TrackonrTrackingSettings",
        runtime_settings_type_name="TrackonrRuntimeSettings",
        probe_environment=probe_trackonr_environment,
        format_environment_report=format_trackonr_environment_report,
        collect_params_method_name="_collect_trackonr_tracking_params",
        run_tracking=track_spots_trackonr_offline,
        run_retrack=retrack_single_spot_trackonr_offline,
        supports_direct=True,
        supports_subprocess=True,
        settings_attr_name="trackonr_settings",
        params_argument_name="trackonr_params",
        panel_group_attr="trackonr_group",
        warning_label_attr="trackonr_warning_label",
        backend_mode_widget_attr="trackonr_backend_mode_combo",
        repo_path_attr="trackonr_repo_path_edit",
        python_executable_attr="trackonr_python_executable_edit",
        default_conda_env_name="trackonr2",
        checkpoint_selector_attr="trackonr_checkpoint_combo",
    ),
    TrackerSpec(
        tracker_id="pipspp_offline",
        display_name="Offline PIPs++",
        scenario_name="Offline PIPs++",
        family="point",
        source_names=("pipspp_offline",),
        settings_type_name="PipsppTrackingSettings",
        runtime_settings_type_name="PipsppRuntimeSettings",
        probe_environment=probe_pipspp_environment,
        format_environment_report=format_pipspp_environment_report,
        collect_params_method_name="_collect_pipspp_tracking_params",
        run_tracking=track_spots_pipspp_offline,
        run_retrack=retrack_single_spot_pipspp_offline,
        supports_direct=True,
        supports_subprocess=True,
        settings_attr_name="pipspp_settings",
        params_argument_name="pipspp_params",
        panel_group_attr="pipspp_group",
        warning_label_attr="pipspp_warning_label",
        backend_mode_widget_attr="pipspp_backend_mode_combo",
        repo_path_attr="pipspp_repo_path_edit",
        python_executable_attr="pipspp_python_executable_edit",
        default_conda_env_name="pipspp",
        checkpoint_selector_attr="pipspp_checkpoint_combo",
    ),
    TrackerSpec(
        tracker_id="mftiq_offline",
        display_name="Offline MFTIQ",
        scenario_name="Offline MFTIQ",
        family="point",
        source_names=("mftiq_offline",),
        settings_type_name="MftiqTrackingSettings",
        runtime_settings_type_name="MftiqRuntimeSettings",
        probe_environment=probe_mftiq_environment,
        format_environment_report=format_mftiq_environment_report,
        collect_params_method_name="_collect_mftiq_tracking_params",
        run_tracking=track_spots_mftiq_offline,
        run_retrack=retrack_single_spot_mftiq_offline,
        supports_direct=True,
        supports_subprocess=True,
        settings_attr_name="mftiq_settings",
        params_argument_name="mftiq_params",
        panel_group_attr="mftiq_group",
        warning_label_attr="mftiq_warning_label",
        backend_mode_widget_attr="mftiq_backend_mode_combo",
        repo_path_attr="mftiq_repo_path_edit",
        python_executable_attr="mftiq_python_executable_edit",
        default_conda_env_name="mftiq",
        checkpoint_selector_attr="mftiq_checkpoint_combo",
    ),
    TrackerSpec(
        tracker_id="tomp_offline",
        display_name="Offline ToMP / TaMOs",
        scenario_name="Offline ToMP",
        family="bbox",
        source_names=("tomp_offline", "tamos_offline"),
        settings_type_name="TompTrackingSettings",
        runtime_settings_type_name="TompRuntimeSettings",
        probe_environment=probe_tomp_environment,
        format_environment_report=format_tomp_environment_report,
        collect_params_method_name="_collect_tomp_tracking_params",
        run_tracking=track_spots_tomp_offline,
        run_retrack=retrack_single_spot_tomp_offline,
        supports_direct=True,
        supports_subprocess=True,
        settings_attr_name="tomp_settings",
        params_argument_name="tomp_params",
        panel_group_attr="tomp_group",
        warning_label_attr="tomp_warning_label",
        backend_mode_widget_attr="tomp_backend_mode_combo",
        repo_path_attr="tomp_repo_path_edit",
        python_executable_attr="tomp_python_executable_edit",
        default_conda_env_name="tomp",
        checkpoint_selector_attr="tomp_checkpoint_combo",
        variant_selector_attr="tomp_model_variant_combo",
    ),
    TrackerSpec(
        tracker_id="ostrack_offline",
        display_name="Offline OSTrack",
        scenario_name="Offline OSTrack",
        family="bbox",
        source_names=("ostrack_offline",),
        settings_type_name="OstrackTrackingSettings",
        runtime_settings_type_name="OstrackRuntimeSettings",
        probe_environment=probe_ostrack_environment,
        format_environment_report=format_ostrack_environment_report,
        collect_params_method_name="_collect_ostrack_tracking_params",
        run_tracking=track_spots_ostrack_offline,
        run_retrack=retrack_single_spot_ostrack_offline,
        supports_direct=True,
        supports_subprocess=True,
        settings_attr_name="ostrack_settings",
        params_argument_name="ostrack_params",
        panel_group_attr="ostrack_group",
        warning_label_attr="ostrack_warning_label",
        backend_mode_widget_attr="ostrack_backend_mode_combo",
        repo_path_attr="ostrack_repo_path_edit",
        python_executable_attr="ostrack_python_executable_edit",
        default_conda_env_name="ostrack",
        checkpoint_selector_attr="ostrack_checkpoint_combo",
        variant_selector_attr="ostrack_model_variant_combo",
    ),
    TrackerSpec(
        tracker_id="cutie_offline",
        display_name="Offline Cutie",
        scenario_name="Offline Cutie",
        family="mask",
        source_names=("cutie_offline",),
        settings_type_name="CutieTrackingSettings",
        runtime_settings_type_name="CutieRuntimeSettings",
        probe_environment=probe_cutie_environment,
        format_environment_report=format_cutie_environment_report,
        collect_params_method_name="_collect_cutie_tracking_params",
        run_tracking=track_spots_cutie_offline,
        run_retrack=retrack_single_spot_cutie_offline,
        supports_direct=True,
        supports_subprocess=True,
        settings_attr_name="cutie_settings",
        params_argument_name="cutie_params",
        panel_group_attr="cutie_group",
        warning_label_attr="cutie_warning_label",
        backend_mode_widget_attr="cutie_backend_mode_combo",
        repo_path_attr="cutie_repo_path_edit",
        python_executable_attr="cutie_python_executable_edit",
        default_conda_env_name="cutie",
        checkpoint_selector_attr="cutie_checkpoint_combo",
    ),
    TrackerSpec(
        tracker_id="xmem2_offline",
        display_name="Offline XMem2",
        scenario_name="Offline XMem2",
        family="mask",
        source_names=("xmem2_offline",),
        settings_type_name="XMem2TrackingSettings",
        runtime_settings_type_name="XMem2RuntimeSettings",
        probe_environment=probe_xmem2_environment,
        format_environment_report=format_xmem2_environment_report,
        collect_params_method_name="_collect_xmem2_tracking_params",
        run_tracking=track_spots_xmem2_offline,
        run_retrack=retrack_single_spot_xmem2_offline,
        supports_direct=True,
        supports_subprocess=True,
        settings_attr_name="xmem2_settings",
        params_argument_name="xmem2_params",
        panel_group_attr="xmem2_group",
        warning_label_attr="xmem2_warning_label",
        backend_mode_widget_attr="xmem2_backend_mode_combo",
        repo_path_attr="xmem2_repo_path_edit",
        python_executable_attr="xmem2_python_executable_edit",
        default_conda_env_name="xmem2",
        checkpoint_selector_attr="xmem2_checkpoint_combo",
    ),
    TrackerSpec(
        tracker_id="sam2_offline",
        display_name="Offline SAM2",
        scenario_name="Offline SAM2",
        family="mask",
        source_names=("sam2_offline",),
        settings_type_name="Sam2TrackingSettings",
        runtime_settings_type_name="Sam2RuntimeSettings",
        probe_environment=probe_sam2_environment,
        format_environment_report=format_sam2_environment_report,
        collect_params_method_name="_collect_sam2_tracking_params",
        run_tracking=track_spots_sam2_offline,
        run_retrack=retrack_single_spot_sam2_offline,
        supports_direct=True,
        supports_subprocess=True,
        settings_attr_name="sam2_settings",
        params_argument_name="sam2_params",
        panel_group_attr="sam2_group",
        warning_label_attr="sam2_warning_label",
        backend_mode_widget_attr="sam2_backend_mode_combo",
        repo_path_attr="sam2_repo_path_edit",
        python_executable_attr="sam2_python_executable_edit",
        default_conda_env_name="sam2",
        checkpoint_selector_attr="sam2_checkpoint_combo",
    ),
    TrackerSpec(
        tracker_id="dam4sam_offline",
        display_name="Offline DAM4SAM",
        scenario_name="Offline DAM4SAM",
        family="mask",
        source_names=("dam4sam_offline",),
        settings_type_name="Dam4SamTrackingSettings",
        runtime_settings_type_name="Dam4SamRuntimeSettings",
        probe_environment=probe_dam4sam_environment,
        format_environment_report=format_dam4sam_environment_report,
        collect_params_method_name="_collect_dam4sam_tracking_params",
        run_tracking=track_spots_dam4sam_offline,
        run_retrack=retrack_single_spot_dam4sam_offline,
        supports_direct=True,
        supports_subprocess=True,
        has_visualizer_retrack=True,
        settings_attr_name="dam4sam_settings",
        params_argument_name="dam4sam_params",
        panel_group_attr="dam4sam_group",
        warning_label_attr="dam4sam_warning_label",
        backend_mode_widget_attr="dam4sam_backend_mode_combo",
        repo_path_attr="dam4sam_repo_path_edit",
        python_executable_attr="dam4sam_python_executable_edit",
        default_conda_env_name="dam4sam",
        checkpoint_selector_attr="dam4sam_checkpoint_combo",
        variant_selector_attr="dam4sam_tracker_variant_combo",
    ),
    TrackerSpec(
        tracker_id="samurai_offline",
        display_name="Offline SAMURAI",
        scenario_name="Offline SAMURAI",
        family="mask",
        source_names=("samurai_offline",),
        settings_type_name="SamuraiTrackingSettings",
        runtime_settings_type_name="SamuraiRuntimeSettings",
        probe_environment=probe_samurai_environment,
        format_environment_report=format_samurai_environment_report,
        collect_params_method_name="_collect_samurai_tracking_params",
        run_tracking=track_spots_samurai_offline,
        run_retrack=retrack_single_spot_samurai_offline,
        supports_direct=True,
        supports_subprocess=True,
        has_visualizer_retrack=True,
        settings_attr_name="samurai_settings",
        params_argument_name="samurai_params",
        panel_group_attr="samurai_group",
        warning_label_attr="samurai_warning_label",
        backend_mode_widget_attr="samurai_backend_mode_combo",
        repo_path_attr="samurai_repo_path_edit",
        python_executable_attr="samurai_python_executable_edit",
        default_conda_env_name="samurai",
        checkpoint_selector_attr="samurai_checkpoint_combo",
        variant_selector_attr="samurai_model_variant_combo",
    ),
    TrackerSpec(
        tracker_id="deva_offline",
        display_name="Offline DEVA",
        scenario_name="Offline DEVA",
        family="mask",
        source_names=("deva_offline",),
        settings_type_name="DevaTrackingSettings",
        runtime_settings_type_name="DevaRuntimeSettings",
        probe_environment=probe_deva_environment,
        format_environment_report=format_deva_environment_report,
        collect_params_method_name="_collect_deva_tracking_params",
        run_tracking=track_spots_deva_offline,
        run_retrack=retrack_single_spot_deva_offline,
        supports_direct=True,
        supports_subprocess=True,
        settings_attr_name="deva_settings",
        params_argument_name="deva_params",
        panel_group_attr="deva_group",
        warning_label_attr="deva_warning_label",
        backend_mode_widget_attr="deva_backend_mode_combo",
        repo_path_attr="deva_repo_path_edit",
        python_executable_attr="deva_python_executable_edit",
        default_conda_env_name="deva",
        checkpoint_selector_attr="deva_checkpoint_combo",
    ),
)


def _build_unique_index(
    pairs: list[tuple[str, TrackerSpec]],
    *,
    index_name: str,
) -> dict[str, TrackerSpec]:
    index: dict[str, TrackerSpec] = {}
    for key, spec in pairs:
        normalized_key = str(key).strip().lower()
        if not normalized_key:
            raise ValueError(f"Empty key in tracker registry index: {index_name}")
        previous = index.get(normalized_key)
        if previous is not None and previous != spec:
            raise ValueError(
                f"Duplicate tracker registry key `{normalized_key}` in index `{index_name}`. "
                f"Conflicting specs: `{previous.tracker_id}` vs `{spec.tracker_id}`."
            )
        index[normalized_key] = spec
    return index


TRACKER_SPEC_BY_ID = _build_unique_index(
    [(spec.tracker_id, spec) for spec in TRACKER_SPECS],
    index_name="tracker_id",
)

TRACKER_SPEC_BY_SCENARIO = _build_unique_index(
    [(spec.scenario_name, spec) for spec in TRACKER_SPECS],
    index_name="scenario_name",
)

TRACKER_SPEC_BY_SOURCE = _build_unique_index(
    [(source_name, spec) for spec in TRACKER_SPECS for source_name in spec.source_names],
    index_name="source_name",
)

TRACKING_SCENARIO_NAMES = tuple(spec.scenario_name for spec in TRACKER_SPECS)


def iter_tracker_specs() -> tuple[TrackerSpec, ...]:
    return TRACKER_SPECS


def get_tracker_spec(tracker_id: str) -> TrackerSpec | None:
    return TRACKER_SPEC_BY_ID.get(str(tracker_id).strip().lower())


def get_tracker_spec_for_scenario(scenario_name: str) -> TrackerSpec | None:
    return TRACKER_SPEC_BY_SCENARIO.get(str(scenario_name).strip().lower())


def get_tracker_spec_for_source(source_name: str) -> TrackerSpec | None:
    return TRACKER_SPEC_BY_SOURCE.get(str(source_name).strip().lower())


def require_tracker_spec(tracker_id: str) -> TrackerSpec:
    spec = get_tracker_spec(tracker_id)
    if spec is None:
        raise KeyError(f"Unknown tracker id: {tracker_id}")
    return spec


def require_tracker_spec_for_scenario(scenario_name: str) -> TrackerSpec:
    spec = get_tracker_spec_for_scenario(scenario_name)
    if spec is None:
        raise KeyError(f"Unknown tracking scenario: {scenario_name}")
    return spec


def require_tracker_spec_for_source(source_name: str) -> TrackerSpec:
    spec = get_tracker_spec_for_source(source_name)
    if spec is None:
        raise KeyError(f"Unknown tracking source: {source_name}")
    return spec


__all__ = [
    "TRACKER_SPECS",
    "TRACKER_SPEC_BY_ID",
    "TRACKER_SPEC_BY_SCENARIO",
    "TRACKER_SPEC_BY_SOURCE",
    "TRACKING_SCENARIO_NAMES",
    "TrackerFamily",
    "TrackerSpec",
    "get_tracker_spec",
    "get_tracker_spec_for_scenario",
    "get_tracker_spec_for_source",
    "iter_tracker_specs",
    "require_tracker_spec",
    "require_tracker_spec_for_scenario",
    "require_tracker_spec_for_source",
]
