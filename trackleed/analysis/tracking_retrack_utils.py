from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from trackleed.core.data_models import SpotROI, Track


@dataclass(frozen=True)
class RetrackStartContext:
    start_spot: SpotROI
    start_frame_index: int
    subset_frame_ids: list[int]


def trim_track_after_frame(track: Track, frame_id: int) -> None:
    frames_to_remove = [fid for fid in track.spots_by_image_id if fid > frame_id]
    for fid in frames_to_remove:
        del track.spots_by_image_id[fid]


def prepare_retrack_start_context(
    track: Track,
    *,
    start_frame_id: int,
    sorted_frame_ids: list[int],
    operation_label: str,
) -> RetrackStartContext:
    if start_frame_id not in track.spots_by_image_id:
        raise ValueError(
            f"Cannot start {operation_label}. Track {track.track_id} "
            f"has no point on frame {start_frame_id}."
        )
    if start_frame_id not in sorted_frame_ids:
        raise ValueError("Start frame is missing from the provided frame list.")

    trim_track_after_frame(track, start_frame_id)

    start_frame_index = sorted_frame_ids.index(start_frame_id)
    return RetrackStartContext(
        start_spot=track.spots_by_image_id[start_frame_id],
        start_frame_index=start_frame_index,
        subset_frame_ids=sorted_frame_ids[start_frame_index:],
    )


def finish_retrack_if_too_short(
    track: Track,
    *,
    subset_frame_ids: list[int],
    progress_callback: Callable[[int], None] | None = None,
) -> Track | None:
    if len(subset_frame_ids) >= 2:
        return None
    if progress_callback:
        progress_callback(100)
    return track
