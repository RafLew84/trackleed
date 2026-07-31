"""Pair and gap outcomes for reconstructed Ag(111) candidates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from trackleed.analysis.scientific_panel_common import (
    BASELINE_PACKAGE,
    TracerValidationError,
    read_json,
)
from trackleed.analysis.scientific_panel_records import PanelGapRecord, PanelPairRecord
from trackleed.analysis.scientific_tracer_common import BRANCHES
from trackleed.analysis.scientific_tracer_records import (
    CandidateRecord,
    EvaluationRecord,
    read_raw_csv,
)


def reconstruct_panel_pairs(
    candidates: Sequence[CandidateRecord],
    evaluations: Sequence[EvaluationRecord],
    workspace_root: Path,
    case_id: str,
    series_id: str,
    pair_id: str,
    method: str,
) -> list[PanelPairRecord]:
    evaluation_by_key = {row["record_key"]: row for row in evaluations}
    consistent: dict[int, dict[str, CandidateRecord]] = {}
    for candidate in candidates:
        evaluation = evaluation_by_key[candidate["record_key"]]
        if evaluation["measurement_status"] == "branch_consistent":
            consistent.setdefault(int(candidate["frame"]), {})[
                str(candidate["branch"])
            ] = candidate
    summary_path = (
        workspace_root
        / BASELINE_PACKAGE
        / "frozen_inputs"
        / "cases"
        / case_id
        / "summary.json"
    )
    case_summary = read_json(summary_path)
    seeds = {str(seed["branch"]): seed for seed in case_summary["frozen_seeds"]}
    reference_distance = math.hypot(
        float(seeds["B2"]["center_x"]) - float(seeds["B1"]["center_x"]),
        float(seeds["B2"]["center_y"]) - float(seeds["B1"]["center_y"]),
    )
    reference_midpoint = np.asarray(
        [
            (float(seeds["B1"]["center_x"]) + float(seeds["B2"]["center_x"]))
            / 2.0,
            (float(seeds["B1"]["center_y"]) + float(seeds["B2"]["center_y"]))
            / 2.0,
        ]
    )
    reference_energy = float(case_summary["reference_energy_ev"])
    rows: list[PanelPairRecord] = []
    for frame, branches in sorted(consistent.items()):
        if set(branches) != set(BRANCHES):
            continue
        first = np.asarray(
            [float(branches["B1"]["center_x"]), float(branches["B1"]["center_y"])]
        )
        second = np.asarray(
            [float(branches["B2"]["center_x"]), float(branches["B2"]["center_y"])]
        )
        energy_ev = float(branches["B1"]["energy_ev"])
        distance = float(np.linalg.norm(second - first))
        predicted = reference_distance * math.sqrt(reference_energy / energy_ev)
        midpoint = (first + second) / 2.0
        rows.append(
            PanelPairRecord(
                record_key=f"case={case_id}|method={method}|frame={frame}",
                case_id=case_id,
                series_id=series_id,
                pair_id=pair_id,
                method=method,
                frame=frame,
                energy_ev=energy_ev,
                pair_distance_px=distance,
                predicted_distance_px=predicted,
                distance_residual_px=distance - predicted,
                midpoint_x=float(midpoint[0]),
                midpoint_y=float(midpoint[1]),
                midpoint_drift_px=float(
                    np.linalg.norm(midpoint - reference_midpoint)
                ),
            )
        )
    return rows


def reconstruct_panel_gaps(
    evaluations: Sequence[EvaluationRecord],
    workspace_root: Path,
    case_id: str,
    series_id: str,
    pair_id: str,
    method: str,
    gap_specs: Sequence[Mapping[str, str]],
) -> list[PanelGapRecord]:
    case_dir = BASELINE_PACKAGE / "frozen_inputs" / "cases" / case_id
    _, baseline_rows = read_raw_csv(
        workspace_root / case_dir / "scan_measurements.csv"
    )
    status = {
        (int(row["frame"]), row["branch"]): row["status"]
        for row in baseline_rows
    }
    energy_by_frame = {
        int(row["frame"]): float(row["energy_ev"]) for row in baseline_rows
    }
    evaluation = {
        (int(row["frame"]), str(row["branch"])): row for row in evaluations
    }
    reference_frame = int(
        read_json(workspace_root / case_dir / "summary.json")["reference_frame"]
    )
    frozen_by_identity = {
        (
            row["branch"],
            row["direction"],
            float(row["start_energy_ev"]),
            float(row["end_energy_ev"]),
        ): row
        for row in gap_specs
    }
    rows: list[PanelGapRecord] = []
    for branch in BRANCHES:
        gap_frames = [
            frame
            for frame in sorted(energy_by_frame)
            if status[(frame, branch)] == "iv_visibility_gap"
        ]
        runs: list[list[int]] = []
        for frame in gap_frames:
            if not runs or frame != runs[-1][-1] + 1:
                runs.append([frame])
            else:
                runs[-1].append(frame)
        for run in runs:
            start, end = run[0], run[-1]
            if start > reference_frame:
                direction = "high"
                post_frames = range(end + 1, max(energy_by_frame) + 1)
            elif end < reference_frame:
                direction = "low"
                post_frames = range(start - 1, -1, -1)
            else:
                continue
            first_expected = None
            for frame in post_frames:
                current = status[(frame, branch)]
                if current == "visible_fit":
                    first_expected = frame
                    break
                if current in {"apparatus_absence", "beam_onset_or_termination"}:
                    break
            if first_expected is None:
                continue
            identity = (
                branch,
                direction,
                energy_by_frame[start],
                energy_by_frame[end],
            )
            frozen_gap = frozen_by_identity.get(identity)
            if frozen_gap is None:
                raise TracerValidationError(
                    f"Frozen gap identity changed: {case_id}/{identity}"
                )
            step = 1 if direction == "high" else -1
            recovery = None
            delay = 0
            frame = first_expected
            while frame in energy_by_frame and status[(frame, branch)] == "visible_fit":
                if evaluation[(frame, branch)]["measurement_status"] == "branch_consistent":
                    recovery = frame
                    break
                delay += 1
                frame += step
            rows.append(
                PanelGapRecord(
                    gap_key=str(frozen_gap["gap_key"]),
                    case_id=case_id,
                    series_id=series_id,
                    pair_id=pair_id,
                    method=method,
                    branch=branch,
                    direction=direction,
                    gap_start_energy_ev=energy_by_frame[start],
                    gap_end_energy_ev=energy_by_frame[end],
                    gap_frame_count=len(run),
                    first_expected_recovery_energy_ev=energy_by_frame[first_expected],
                    recovered=recovery is not None,
                    recovery_energy_ev=(
                        energy_by_frame[recovery] if recovery is not None else None
                    ),
                    recovery_delay_frames=(delay if recovery is not None else None),
                )
            )
    if len(rows) != len(gap_specs):
        raise TracerValidationError(f"Frozen gap set changed: {case_id}")
    return rows




__all__ = ["reconstruct_panel_gaps", "reconstruct_panel_pairs"]
