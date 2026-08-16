"""Launch catalog-backed review re-tracking through the public workflow."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Protocol, cast

from PyQt6.QtWidgets import QMessageBox, QWidget

from trackleed.core.data_models import LeedFile, Track
from trackleed.gui.seeded_tracking_adapter import (
    build_seeded_retrack_request,
    run_seeded_retrack_job,
)
from trackleed.gui.seeded_bbox_tracking_compatibility import (
    BboxReviewSpec,
    bbox_review_spec_for_source,
)
from trackleed.gui.seeded_mask_tracking_compatibility import (
    MaskReviewSpec,
    mask_review_spec_for_source,
)
from trackleed.gui.seeded_point_tracking_compatibility import (
    PointReviewSpec,
    point_review_spec_for_source,
)
from trackleed.io.image_reader import load_tiff
from trackleed.logic.generic_worker import GenericWorker
from trackleed.logic.workflows.seeded_tracking import (
    SeededTrackingConfig,
    default_seeded_tracking_catalog,
)


class SeededRetrackWorkerHost(Protocol):
    def launch_retrack_worker(
        self,
        worker: GenericWorker,
        title: str,
        cancel_event: Event | None = None,
        method_label: str = "Re-tracking",
    ) -> None: ...


@dataclass(frozen=True)
class SeededRetrackLaunchRequest:
    method_id: str
    files: tuple[LeedFile, ...]
    track: Track
    start_frame_id: int
    config: SeededTrackingConfig | None


def review_spec_for_source(
    source_name: str,
) -> PointReviewSpec | BboxReviewSpec | MaskReviewSpec | None:
    """Resolve a catalog-backed review specification for a track source."""
    normalized = str(source_name).strip().lower()
    return (
        point_review_spec_for_source(normalized)
        or bbox_review_spec_for_source(normalized)
        or mask_review_spec_for_source(normalized)
    )


def launch_seeded_retrack(
    host: SeededRetrackWorkerHost,
    launch_request: SeededRetrackLaunchRequest,
) -> None:
    """Build and launch one catalog-dispatched review re-track."""
    method = default_seeded_tracking_catalog().method(launch_request.method_id)
    if launch_request.config is None:
        QMessageBox.warning(
            cast(QWidget, host),
            "Re-tracking unavailable",
            f"No {method.display_name} configuration is available for review.",
        )
        return
    request = build_seeded_retrack_request(
        files=launch_request.files,
        track=launch_request.track,
        start_frame_id=launch_request.start_frame_id,
        method_id=launch_request.method_id,
        config=launch_request.config,
    )
    cancel_event = Event()
    worker = GenericWorker(
        run_seeded_retrack_job,
        request,
        load_tiff,
        cancel_event,
    )
    host.launch_retrack_worker(
        worker,
        title=f"Re-tracking spot with {method.display_name}...",
        cancel_event=cancel_event,
        method_label=f"{method.display_name} re-tracking",
    )
