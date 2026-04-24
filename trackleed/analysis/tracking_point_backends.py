from __future__ import annotations

from importlib import import_module
from threading import Event
from typing import Callable, Dict, List, Optional

from trackleed.core.data_models import LeedFile, Track


def _tracking_support():
    return import_module("trackleed.analysis.tracking")


def _run_legacy_point_tracking(
    legacy_name: str,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    return getattr(_tracking_support(), legacy_name)(
        all_files_map,
        sorted_frame_ids,
        reference_frame_id,
        params,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )


def _run_legacy_point_retrack(
    legacy_name: str,
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    return getattr(_tracking_support(), legacy_name)(
        track_to_update,
        start_frame_id,
        all_files_map,
        sorted_frame_ids,
        params,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )


def track_spots_tapir_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    return _run_legacy_point_tracking(
        "_legacy_track_spots_tapir_offline",
        all_files_map,
        sorted_frame_ids,
        reference_frame_id,
        params,
        progress_callback,
        cancel_event,
    )


def track_spots_tapir_online(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    return _run_legacy_point_tracking(
        "_legacy_track_spots_tapir_online",
        all_files_map,
        sorted_frame_ids,
        reference_frame_id,
        params,
        progress_callback,
        cancel_event,
    )


def track_spots_locotrack_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    return _run_legacy_point_tracking(
        "_legacy_track_spots_locotrack_offline",
        all_files_map,
        sorted_frame_ids,
        reference_frame_id,
        params,
        progress_callback,
        cancel_event,
    )


def track_spots_cotracker3_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    return _run_legacy_point_tracking(
        "_legacy_track_spots_cotracker3_offline",
        all_files_map,
        sorted_frame_ids,
        reference_frame_id,
        params,
        progress_callback,
        cancel_event,
    )


def track_spots_tapnext_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    return _run_legacy_point_tracking(
        "_legacy_track_spots_tapnext_offline",
        all_files_map,
        sorted_frame_ids,
        reference_frame_id,
        params,
        progress_callback,
        cancel_event,
    )


def track_spots_trackonr_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    return _run_legacy_point_tracking(
        "_legacy_track_spots_trackonr_offline",
        all_files_map,
        sorted_frame_ids,
        reference_frame_id,
        params,
        progress_callback,
        cancel_event,
    )


def track_spots_pipspp_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    return _run_legacy_point_tracking(
        "_legacy_track_spots_pipspp_offline",
        all_files_map,
        sorted_frame_ids,
        reference_frame_id,
        params,
        progress_callback,
        cancel_event,
    )


def track_spots_mftiq_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    return _run_legacy_point_tracking(
        "_legacy_track_spots_mftiq_offline",
        all_files_map,
        sorted_frame_ids,
        reference_frame_id,
        params,
        progress_callback,
        cancel_event,
    )


def retrack_single_spot_tapir_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    return _run_legacy_point_retrack(
        "_legacy_retrack_single_spot_tapir_offline",
        track_to_update,
        start_frame_id,
        all_files_map,
        sorted_frame_ids,
        params,
        progress_callback,
        cancel_event,
    )


def retrack_single_spot_tapir_online(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    return _run_legacy_point_retrack(
        "_legacy_retrack_single_spot_tapir_online",
        track_to_update,
        start_frame_id,
        all_files_map,
        sorted_frame_ids,
        params,
        progress_callback,
        cancel_event,
    )


def retrack_single_spot_locotrack_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    return _run_legacy_point_retrack(
        "_legacy_retrack_single_spot_locotrack_offline",
        track_to_update,
        start_frame_id,
        all_files_map,
        sorted_frame_ids,
        params,
        progress_callback,
        cancel_event,
    )


def retrack_single_spot_cotracker3_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    return _run_legacy_point_retrack(
        "_legacy_retrack_single_spot_cotracker3_offline",
        track_to_update,
        start_frame_id,
        all_files_map,
        sorted_frame_ids,
        params,
        progress_callback,
        cancel_event,
    )


def retrack_single_spot_tapnext_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    return _run_legacy_point_retrack(
        "_legacy_retrack_single_spot_tapnext_offline",
        track_to_update,
        start_frame_id,
        all_files_map,
        sorted_frame_ids,
        params,
        progress_callback,
        cancel_event,
    )


def retrack_single_spot_trackonr_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    return _run_legacy_point_retrack(
        "_legacy_retrack_single_spot_trackonr_offline",
        track_to_update,
        start_frame_id,
        all_files_map,
        sorted_frame_ids,
        params,
        progress_callback,
        cancel_event,
    )


def retrack_single_spot_pipspp_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    return _run_legacy_point_retrack(
        "_legacy_retrack_single_spot_pipspp_offline",
        track_to_update,
        start_frame_id,
        all_files_map,
        sorted_frame_ids,
        params,
        progress_callback,
        cancel_event,
    )


def retrack_single_spot_mftiq_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    return _run_legacy_point_retrack(
        "_legacy_retrack_single_spot_mftiq_offline",
        track_to_update,
        start_frame_id,
        all_files_map,
        sorted_frame_ids,
        params,
        progress_callback,
        cancel_event,
    )
