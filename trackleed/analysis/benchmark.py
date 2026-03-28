"""Benchmark helpers for comparing tracking backends on the same LEED sequence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from trackleed.analysis.metrics import (
    FrameAssociation,
    TrackingMetrics,
    compute_tracking_metrics,
)
from trackleed.core.data_models import SpotROI, Track


@dataclass
class TrackBenchmarkRow:
    method: str
    track_id: int
    start_frame_id: int
    end_frame_id: int
    length: int
    span_frames: int
    internal_gaps: int
    tail_missing_frames: int
    coverage_ratio: float
    mean_step_px: float
    median_step_px: float
    step_std_px: float
    smoothness_px: float
    visible_ratio: Optional[float] = None
    rejected_points: int = 0


@dataclass
class MethodBenchmarkSummary:
    method: str
    runtime_seconds: float
    track_count: int
    total_points: int
    mean_track_length: float
    median_track_length: float
    max_track_length: int
    total_internal_gaps: int
    total_tail_missing_frames: int
    overall_coverage_ratio: float
    mean_step_px: float
    median_step_px: float
    mean_smoothness_px: float
    visible_ratio: Optional[float] = None
    rejected_points: int = 0
    gt_precision: Optional[float] = None
    gt_recall: Optional[float] = None
    gt_f1: Optional[float] = None
    gt_idf1: Optional[float] = None
    gt_mota: Optional[float] = None
    gt_motp: Optional[float] = None
    gt_matches: Optional[int] = None
    gt_missed_ground_truth: Optional[int] = None
    gt_missed_predictions: Optional[int] = None
    gt_identity_switches: Optional[int] = None


def summarize_tracks(
    tracks: Iterable[Track],
    benchmark_frame_ids: List[int],
    method: str,
    runtime_seconds: float,
    ground_truth_tracks: Optional[Iterable[Track]] = None,
    gt_match_threshold_px: float = 12.0,
) -> tuple[MethodBenchmarkSummary, List[TrackBenchmarkRow]]:
    """
    Compute benchmark summary and per-track rows for one tracking method.

    When `ground_truth_tracks` is given, MOT-style metrics are also computed.
    """
    frame_index_map = {frame_id: idx for idx, frame_id in enumerate(benchmark_frame_ids)}
    track_rows = [
        summarize_single_track(track, benchmark_frame_ids, frame_index_map, method)
        for track in tracks
    ]

    total_frames = max(1, len(benchmark_frame_ids))
    total_points = int(sum(row.length for row in track_rows))
    visible_values = [row.visible_ratio for row in track_rows if row.visible_ratio is not None]

    summary = MethodBenchmarkSummary(
        method=method,
        runtime_seconds=float(runtime_seconds),
        track_count=len(track_rows),
        total_points=total_points,
        mean_track_length=_mean([row.length for row in track_rows]),
        median_track_length=_median([row.length for row in track_rows]),
        max_track_length=max((row.length for row in track_rows), default=0),
        total_internal_gaps=int(sum(row.internal_gaps for row in track_rows)),
        total_tail_missing_frames=int(sum(row.tail_missing_frames for row in track_rows)),
        overall_coverage_ratio=float(total_points / max(1, len(track_rows) * total_frames)) if track_rows else 0.0,
        mean_step_px=_mean([row.mean_step_px for row in track_rows if np.isfinite(row.mean_step_px)]),
        median_step_px=_median([row.median_step_px for row in track_rows if np.isfinite(row.median_step_px)]),
        mean_smoothness_px=_mean([row.smoothness_px for row in track_rows if np.isfinite(row.smoothness_px)]),
        visible_ratio=_mean(visible_values) if visible_values else None,
        rejected_points=int(sum(row.rejected_points for row in track_rows)),
    )

    if ground_truth_tracks is not None:
        gt_metrics = compare_tracks_to_ground_truth(
            predicted_tracks=tracks,
            ground_truth_tracks=ground_truth_tracks,
            benchmark_frame_ids=benchmark_frame_ids,
            match_threshold_px=gt_match_threshold_px,
        )
        summary.gt_precision = gt_metrics.precision
        summary.gt_recall = gt_metrics.recall
        summary.gt_f1 = gt_metrics.f1
        summary.gt_idf1 = gt_metrics.idf1
        summary.gt_mota = gt_metrics.mota
        summary.gt_motp = gt_metrics.motp
        summary.gt_matches = gt_metrics.matches
        summary.gt_missed_ground_truth = gt_metrics.missed_ground_truth
        summary.gt_missed_predictions = gt_metrics.missed_predictions
        summary.gt_identity_switches = gt_metrics.identity_switches

    return summary, track_rows


def summarize_single_track(
    track: Track,
    benchmark_frame_ids: List[int],
    frame_index_map: Dict[int, int],
    method: str,
) -> TrackBenchmarkRow:
    """Summarize one predicted track over the benchmark frame span."""
    ordered = _ordered_track_points(track, frame_index_map)
    if not ordered:
        return TrackBenchmarkRow(
            method=method,
            track_id=int(track.track_id),
            start_frame_id=-1,
            end_frame_id=-1,
            length=0,
            span_frames=len(benchmark_frame_ids),
            internal_gaps=0,
            tail_missing_frames=len(benchmark_frame_ids),
            coverage_ratio=0.0,
            mean_step_px=float("nan"),
            median_step_px=float("nan"),
            step_std_px=float("nan"),
            smoothness_px=float("nan"),
        )

    first_frame_id = ordered[0][0]
    last_frame_id = ordered[-1][0]
    first_idx = ordered[0][1]
    last_idx = ordered[-1][1]
    length = len(ordered)
    internal_gaps = max(0, (last_idx - first_idx + 1) - length)
    tail_missing = max(0, len(benchmark_frame_ids) - (last_idx + 1))
    coverage_ratio = float(length / max(1, len(benchmark_frame_ids)))

    step_per_frame, velocity_vectors = _step_vectors_per_frame(ordered)
    smoothness = _smoothness_from_velocities(velocity_vectors)
    visible_ratio, rejected_points = _visibility_and_rejections(ordered)

    return TrackBenchmarkRow(
        method=method,
        track_id=int(track.track_id),
        start_frame_id=int(first_frame_id),
        end_frame_id=int(last_frame_id),
        length=length,
        span_frames=len(benchmark_frame_ids),
        internal_gaps=internal_gaps,
        tail_missing_frames=tail_missing,
        coverage_ratio=coverage_ratio,
        mean_step_px=_mean(step_per_frame),
        median_step_px=_median(step_per_frame),
        step_std_px=_std(step_per_frame),
        smoothness_px=smoothness,
        visible_ratio=visible_ratio,
        rejected_points=rejected_points,
    )


def compare_tracks_to_ground_truth(
    predicted_tracks: Iterable[Track],
    ground_truth_tracks: Iterable[Track],
    benchmark_frame_ids: List[int],
    match_threshold_px: float = 12.0,
) -> TrackingMetrics:
    """
    Compare predicted tracks against a ground-truth track set using MOT-style metrics.
    """
    prev_matches: Dict[int, int] = {}
    frame_associations: List[FrameAssociation] = []

    for frame_id in benchmark_frame_ids:
        gt_spots = _spots_on_frame(ground_truth_tracks, frame_id)
        pred_spots = _spots_on_frame(predicted_tracks, frame_id)

        if not gt_spots and not pred_spots:
            continue

        matches: List[Tuple[int, int, float]] = []
        matched_gt = set()
        matched_pred = set()
        identity_switches = 0

        if gt_spots and pred_spots:
            cost = np.zeros((len(gt_spots), len(pred_spots)), dtype=np.float32)
            for i, (_, gt_spot) in enumerate(gt_spots):
                gt_xy = np.array([float(gt_spot.center_x), float(gt_spot.center_y)], dtype=np.float32)
                for j, (_, pred_spot) in enumerate(pred_spots):
                    pred_xy = np.array([float(pred_spot.center_x), float(pred_spot.center_y)], dtype=np.float32)
                    cost[i, j] = float(np.linalg.norm(gt_xy - pred_xy))

            gt_idx, pred_idx = linear_sum_assignment(cost)
            for gi, pj in zip(gt_idx.tolist(), pred_idx.tolist()):
                distance = float(cost[gi, pj])
                if distance > float(match_threshold_px):
                    continue
                gt_id = int(gt_spots[gi][0])
                pred_id = int(pred_spots[pj][0])
                matches.append((gt_id, pred_id, distance))
                matched_gt.add(gi)
                matched_pred.add(pj)

                prev_pred_id = prev_matches.get(gt_id)
                if prev_pred_id is not None and prev_pred_id != pred_id:
                    identity_switches += 1
                prev_matches[gt_id] = pred_id

        missed_gt = max(0, len(gt_spots) - len(matched_gt))
        missed_pred = max(0, len(pred_spots) - len(matched_pred))
        frame_associations.append(
            FrameAssociation(
                frame_id=int(frame_id),
                matches=matches,
                missed_ground_truth=missed_gt,
                missed_predictions=missed_pred,
                identity_switches=identity_switches,
            )
        )

    return compute_tracking_metrics(frame_associations)


def track_rows_to_dicts(rows: Iterable[TrackBenchmarkRow]) -> List[Dict[str, object]]:
    return [asdict(row) for row in rows]


def summary_to_dict(summary: MethodBenchmarkSummary) -> Dict[str, object]:
    return asdict(summary)


def _ordered_track_points(
    track: Track,
    frame_index_map: Dict[int, int],
) -> List[Tuple[int, int, SpotROI]]:
    ordered = []
    for frame_id, spot in track.spots_by_image_id.items():
        if frame_id not in frame_index_map:
            continue
        ordered.append((int(frame_id), int(frame_index_map[frame_id]), spot))
    ordered.sort(key=lambda item: item[1])
    return ordered


def _step_vectors_per_frame(
    ordered_points: List[Tuple[int, int, SpotROI]],
) -> tuple[List[float], List[np.ndarray]]:
    step_lengths: List[float] = []
    velocities: List[np.ndarray] = []
    for prev, curr in zip(ordered_points, ordered_points[1:]):
        prev_idx = prev[1]
        curr_idx = curr[1]
        dt = max(1, curr_idx - prev_idx)
        delta = np.array(
            [
                float(curr[2].center_x) - float(prev[2].center_x),
                float(curr[2].center_y) - float(prev[2].center_y),
            ],
            dtype=np.float32,
        )
        velocity = delta / float(dt)
        velocities.append(velocity)
        step_lengths.append(float(np.linalg.norm(velocity)))
    return step_lengths, velocities


def _smoothness_from_velocities(velocities: List[np.ndarray]) -> float:
    if len(velocities) < 2:
        return float("nan")
    accelerations = []
    for prev_v, curr_v in zip(velocities, velocities[1:]):
        accelerations.append(float(np.linalg.norm(curr_v - prev_v)))
    return _mean(accelerations)


def _visibility_and_rejections(
    ordered_points: List[Tuple[int, int, SpotROI]],
) -> tuple[Optional[float], int]:
    visible_values: List[float] = []
    rejected_points = 0
    for _, _, spot in ordered_points:
        visibility = getattr(spot, "tapir_visibility", None)
        if visibility is not None:
            visible_values.append(1.0 if bool(visibility) else 0.0)
        if bool(getattr(spot, "tapir_filter_rejected", False)):
            rejected_points += 1
    visible_ratio = _mean(visible_values) if visible_values else None
    return visible_ratio, rejected_points


def _spots_on_frame(tracks: Iterable[Track], frame_id: int) -> List[Tuple[int, SpotROI]]:
    spots = []
    for track in tracks:
        spot = track.spots_by_image_id.get(frame_id)
        if spot is not None:
            spots.append((int(track.track_id), spot))
    return spots


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.std(vals)) if vals else float("nan")
