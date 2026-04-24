from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackingWindowPlan:
    index: int
    start_idx: int
    end_idx: int
    frame_ids: list[int]
    seed_frame_ids: set[int]


@dataclass(frozen=True)
class TrackingWindowContext:
    plans: list[TrackingWindowPlan]
    overlap: int
    require_reseed_overlap: bool


def compute_tracking_window_starts(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> list[int]:
    """Return monotonically increasing window starts covering the whole sequence."""
    if total_frames <= 1:
        return [0]
    if window_size <= 1 or window_size >= total_frames:
        return [0]

    overlap = min(max(0, int(window_overlap)), max(0, int(window_size) - 1))
    step = max(1, int(window_size) - overlap)
    starts = list(range(0, max(1, int(total_frames) - 1), step))
    last_start = max(0, int(total_frames) - int(window_size))
    if starts[-1] != last_start:
        starts.append(last_start)
    return sorted(set(starts))


def resolve_reseed_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    """
    Resolve overlap for backends that restart each window from an already-tracked seed.

    Such backends need at least one shared frame between neighboring windows when
    running in multi-window mode.
    """
    overlap = min(max(0, int(window_overlap)), max(0, int(window_size) - 1))
    if int(total_frames) > 1 and 1 < int(window_size) < int(total_frames):
        overlap = max(1, overlap)
    return overlap


def resolve_plain_window_overlap(
    window_size: int,
    window_overlap: int,
) -> int:
    """Resolve overlap for backends that do not require an enforced shared seed frame."""
    return min(max(0, int(window_overlap)), max(0, int(window_size) - 1))


def window_seed_frame_ids(window_frame_ids: list[int], overlap: int) -> set[int]:
    return set(window_frame_ids[: max(1, int(overlap) + 1)])


def advance_window_coverage(covered_until: int, end_idx: int) -> int:
    return max(int(covered_until), int(end_idx) - 1)


def build_tracking_window_plans(
    frame_ids: list[int],
    window_size: int,
    window_overlap: int,
    *,
    require_reseed_overlap: bool,
) -> tuple[list[TrackingWindowPlan], int]:
    total_frames = len(frame_ids)
    overlap = (
        resolve_reseed_window_overlap(total_frames, window_size, window_overlap)
        if require_reseed_overlap
        else resolve_plain_window_overlap(window_size, window_overlap)
    )
    starts = compute_tracking_window_starts(total_frames, window_size, overlap)
    plans: list[TrackingWindowPlan] = []
    for window_idx, start_idx in enumerate(starts):
        end_idx = min(total_frames, int(start_idx) + int(window_size))
        window_frame_ids = list(frame_ids[start_idx:end_idx])
        if len(window_frame_ids) < 2:
            continue
        plans.append(
            TrackingWindowPlan(
                index=int(window_idx),
                start_idx=int(start_idx),
                end_idx=int(end_idx),
                frame_ids=window_frame_ids,
                seed_frame_ids=window_seed_frame_ids(window_frame_ids, overlap),
            )
        )
    return plans, overlap


def build_reseed_tracking_window_context(
    frame_ids: list[int],
    window_size: int,
    window_overlap: int,
) -> TrackingWindowContext:
    plans, overlap = build_tracking_window_plans(
        frame_ids,
        window_size,
        window_overlap,
        require_reseed_overlap=True,
    )
    return TrackingWindowContext(
        plans=plans,
        overlap=overlap,
        require_reseed_overlap=True,
    )


def build_plain_tracking_window_context(
    frame_ids: list[int],
    window_size: int,
    window_overlap: int,
) -> TrackingWindowContext:
    plans, overlap = build_tracking_window_plans(
        frame_ids,
        window_size,
        window_overlap,
        require_reseed_overlap=False,
    )
    return TrackingWindowContext(
        plans=plans,
        overlap=overlap,
        require_reseed_overlap=False,
    )
