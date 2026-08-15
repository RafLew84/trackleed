"""Launch catalog-backed review re-tracking through the public workflow."""

from __future__ import annotations

from threading import Event
from typing import Any

from PyQt6.QtWidgets import QMessageBox

from trackleed.gui.seeded_tracking_adapter import (
    build_seeded_retrack_request,
    run_seeded_retrack_job,
)
from trackleed.gui.seeded_bbox_tracking_compatibility import (
    bbox_review_spec_for_source,
)
from trackleed.gui.seeded_mask_tracking_compatibility import (
    mask_review_spec_for_source,
)
from trackleed.gui.seeded_point_tracking_compatibility import (
    point_review_spec_for_source,
)
from trackleed.io.image_reader import load_tiff
from trackleed.logic.generic_worker import GenericWorker
from trackleed.logic.workflows.seeded_tracking import default_seeded_tracking_catalog


def review_spec_for_source(source_name: str) -> Any | None:
    """Resolve a catalog-backed review specification for a track source."""
    normalized = str(source_name).strip().lower()
    return (
        point_review_spec_for_source(normalized)
        or bbox_review_spec_for_source(normalized)
        or mask_review_spec_for_source(normalized)
    )


def launch_seeded_retrack(
    dialog: Any,
    method_id: str,
    _spot_id: int,
) -> None:
    """Build and launch one catalog-dispatched review re-track."""
    method = default_seeded_tracking_catalog().method(method_id)
    target = dialog._get_retrack_target()
    if target is None:
        return
    track, start_frame_id = target
    config = dialog.seeded_tracking_configs.get(method_id)
    if config is None:
        QMessageBox.warning(
            dialog,
            "Re-tracking unavailable",
            f"No {method.display_name} configuration is available for review.",
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
        title=f"Re-tracking spot with {method.display_name}...",
        cancel_event=cancel_event,
        method_label=f"{method.display_name} re-tracking",
    )
