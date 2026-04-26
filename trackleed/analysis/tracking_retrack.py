from __future__ import annotations

from importlib import import_module
from threading import Event
from typing import Dict, List, Optional, Callable

from trackleed.analysis.tracking_bbox_backends import (
    retrack_single_spot_ostrack_offline,
    retrack_single_spot_tomp_offline,
)
from trackleed.analysis.tracking_mask_backends import (
    retrack_single_spot_cutie_offline,
    retrack_single_spot_deva_offline,
    retrack_single_spot_sam2_offline,
    retrack_single_spot_xmem2_offline,
)
from trackleed.analysis.tracking_point_backends import (
    retrack_single_spot_cotracker3_offline,
    retrack_single_spot_locotrack_offline,
    retrack_single_spot_mftiq_offline,
    retrack_single_spot_pipspp_offline,
    retrack_single_spot_tapir_offline,
    retrack_single_spot_tapir_online,
    retrack_single_spot_tapnext_offline,
    retrack_single_spot_trackonr_offline,
)
from trackleed.core.data_models import LeedFile, Track


def _tracking_support():
    return import_module("trackleed.analysis.tracking")


def retrack_single_spot_optical_flow(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    return _tracking_support()._legacy_retrack_single_spot_optical_flow(
        track_to_update,
        start_frame_id,
        all_files_map,
        sorted_frame_ids,
        params,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
