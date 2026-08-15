"""Launch point-family review re-tracking through the public workflow."""

from __future__ import annotations

from threading import Event
from typing import Any

from PyQt6.QtWidgets import QMessageBox

from trackleed.analysis.seeded_point_tracking_runtime import (
    point_tracking_definition,
)
from trackleed.gui.seeded_tracking_adapter import (
    build_seeded_retrack_request,
    run_seeded_retrack_job,
)
from trackleed.io.image_reader import load_tiff
from trackleed.logic.generic_worker import GenericWorker


def launch_seeded_point_retrack(
    dialog: Any,
    method_id: str,
    _spot_id: int,
) -> None:
    """Build and launch one catalog-dispatched re-track from the review dialog."""
    definition = point_tracking_definition(method_id)
    target = dialog._get_retrack_target()
    if target is None:
        return
    track, start_frame_id = target
    config = dialog.seeded_point_configs.get(method_id)
    if config is None:
        QMessageBox.warning(
            dialog,
            "Re-tracking unavailable",
            f"No {definition.display_name} configuration is available for review.",
        )
        return
    files = tuple(dialog.file_map[frame_id] for frame_id in dialog.sorted_frame_ids)
    request = build_seeded_retrack_request(
        files=files,
        track=track,
        start_frame_id=start_frame_id,
        method_id=method_id,
        config=config,
    )
    cancel_event = Event()
    worker = GenericWorker(
        run_seeded_retrack_job,
        request,
        load_tiff,
        cancel_event,
    )
    dialog.launch_retrack_worker(
        worker,
        title=f"Re-tracking spot with {definition.display_name}...",
        cancel_event=cancel_event,
        method_label=f"{definition.display_name} re-tracking",
    )
