"""
Tracking-quality metrics (IDF1, MOTA, MOTP, precision/recall) helpers.

The functions here operate on lightweight per-frame association summaries so that
existing and future trackers can log their matches without pulling in heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class FrameAssociation:
    """
    Summary of how predicted spots were associated with ground truth on a frame.

    Attributes:
        frame_id: Numerical frame identifier.
        matches: List of (ground_truth_id, prediction_id, distance) tuples for accepted matches.
        missed_ground_truth: Count of ground-truth spots without an associated prediction.
        missed_predictions: Count of predictions that were not matched to any ground truth.
        identity_switches: Number of ID switches detected on this frame (optional).
    """

    frame_id: int
    matches: List[Tuple[int, int, float]]
    missed_ground_truth: int
    missed_predictions: int
    identity_switches: int = 0


@dataclass
class TrackingMetrics:
    """Aggregated MOT-style metrics."""

    precision: float
    recall: float
    f1: float
    idf1: float
    mota: float
    motp: float
    matches: int
    missed_ground_truth: int
    missed_predictions: int
    identity_switches: int


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_precision(matches: int, false_positives: int) -> float:
    return _safe_divide(matches, matches + false_positives)


def compute_recall(matches: int, false_negatives: int) -> float:
    return _safe_divide(matches, matches + false_negatives)


def compute_f1(precision: float, recall: float) -> float:
    return (
        _safe_divide(2 * precision * recall, precision + recall)
        if precision and recall
        else 0.0
    )


def compute_idf1(idtp: int, idfp: int, idfn: int) -> float:
    return _safe_divide(2 * idtp, 2 * idtp + idfp + idfn)


def compute_mota(
    misses: int, false_positives: int, identity_switches: int, total_ground_truth: int
) -> float:
    return 1.0 - _safe_divide(misses + false_positives + identity_switches, total_ground_truth)


def compute_motp(total_distance: float, matches: int) -> float:
    return _safe_divide(total_distance, matches)


def compute_tracking_metrics(
    frame_associations: Iterable[FrameAssociation],
) -> TrackingMetrics:
    """
    Aggregate precision/recall/F1, IDF1, MOTA, MOTP over all frames.

    Args:
        frame_associations: Per-frame summaries.

    Returns:
        TrackingMetrics with aggregated statistics.
    """

    total_matches = 0
    total_distance = 0.0
    total_missed_gt = 0
    total_missed_pred = 0
    total_switches = 0

    for assoc in frame_associations:
        total_matches += len(assoc.matches)
        total_distance += sum(distance for _, _, distance in assoc.matches)
        total_missed_gt += assoc.missed_ground_truth
        total_missed_pred += assoc.missed_predictions
        total_switches += assoc.identity_switches

    total_gt = total_matches + total_missed_gt

    precision = compute_precision(total_matches, total_missed_pred)
    recall = compute_recall(total_matches, total_missed_gt)
    f1 = compute_f1(precision, recall)
    idf1 = compute_idf1(total_matches, total_missed_pred, total_missed_gt)
    mota = compute_mota(total_missed_gt, total_missed_pred, total_switches, total_gt) if total_gt else 0.0
    motp = compute_motp(total_distance, total_matches)

    return TrackingMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        idf1=idf1,
        mota=mota,
        motp=motp,
        matches=total_matches,
        missed_ground_truth=total_missed_gt,
        missed_predictions=total_missed_pred,
        identity_switches=total_switches,
    )
