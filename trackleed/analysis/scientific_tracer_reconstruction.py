"""Scientific reconstruction and threshold-only evaluation for the tracer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import tifffile
from skimage.util import img_as_float

from trackleed.analysis.peak_analyzer import fit_spot_to_gaussian
from trackleed.analysis.scientific_tracer_evaluation import (
    evaluate_candidates,
    signed_gate_margins,
)
from trackleed.analysis.scientific_tracer_common import (
    BRANCHES,
    CASE_MEASUREMENTS_PATH,
    CASE_SUMMARY_PATH,
    MEASUREMENT_CONTRACT_PATH,
    OPPORTUNITIES_PATH,
    PROPOSAL_PATH,
    TracerValidationError,
    read_json,
    relative_file,
    sha256,
)
from trackleed.analysis.scientific_tracer_records import (
    CANDIDATE_FIELDS,
    CandidateRecord,
    EvaluationRecord,
    GapRecord,
    PairRecord,
    read_raw_csv,
)


@dataclass(frozen=True)
class TracerArtifacts:
    candidates: list[CandidateRecord]
    evaluations: list[EvaluationRecord]
    pairs: list[PairRecord]
    gaps: list[GapRecord]
    summary: dict[str, Any]


def central_contract(workspace_root: Path) -> dict[str, Any]:
    contract = read_json(workspace_root / MEASUREMENT_CONTRACT_PATH)
    central = contract.get("central_contract")
    if not isinstance(central, dict):
        raise TracerValidationError("Central measurement contract is missing")
    return central


def record_key(frame: int, branch: str) -> str:
    return (
        f"case=scan_a_pair_1|method=locotrack|frame={frame}|"
        f"branch={branch}|roi=48"
    )


def _load_opportunity_keys(workspace_root: Path) -> dict[tuple[int, str], str]:
    _, rows = read_raw_csv(workspace_root / OPPORTUNITIES_PATH)
    return {
        (int(row["frame"]), row["branch"]): row["opportunity_key"]
        for row in rows
        if row["case_id"] == "scan_a_pair_1"
    }


def _load_baseline_statuses(workspace_root: Path) -> dict[tuple[int, str], str]:
    _, rows = read_raw_csv(workspace_root / CASE_MEASUREMENTS_PATH)
    return {
        (int(row["frame"]), row["branch"]): row["status"] for row in rows
    }


def reconstruct_candidates(
    workspace_root: Path,
    tracer_dir: Path,
    contract: Mapping[str, Any],
) -> list[CandidateRecord]:
    """Fit every frozen proposal once and retain the complete candidate layer."""

    measurement_contract = central_contract(workspace_root)
    roi_side = int(measurement_contract["roi_side_px"])
    half = roi_side // 2
    center_preference = float(measurement_contract["center_preference_factor"])
    compatibility = contract["estimator_compatibility"]
    theta_bounds = tuple(
        float(value) for value in compatibility["optimizer_theta_bounds_rad"]
    )

    case_summary = read_json(workspace_root / CASE_SUMMARY_PATH)
    geometry = case_summary["geometry"]
    pattern_center = np.asarray(
        [float(geometry["center_x"]), float(geometry["center_y"])],
        dtype=float,
    )
    reference_energy = float(case_summary["reference_energy_ev"])
    seeds = {
        str(seed["branch"]): seed for seed in case_summary["frozen_seeds"]
    }
    if set(seeds) != set(BRANCHES):
        raise TracerValidationError("Frozen branch seeds changed")

    _, image_rows = read_raw_csv(tracer_dir / "frozen" / "frame_images.csv")
    if len(image_rows) != 35:
        raise TracerValidationError("Frozen frame-image map changed")
    proposal_path = workspace_root / PROPOSAL_PATH
    proposal_sha256 = sha256(proposal_path)
    with np.load(proposal_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if arrays.get("tracks_xy", np.empty(0)).shape != (2, 35, 2):
        raise TracerValidationError("Frozen LocoTrack proposal shape changed")
    if arrays.get("visible_mask", np.empty(0)).shape != (2, 35):
        raise TracerValidationError("Frozen LocoTrack visibility shape changed")

    statuses = _load_baseline_statuses(workspace_root)
    opportunities = _load_opportunity_keys(workspace_root)
    rows: list[CandidateRecord] = []
    for branch_index, branch in enumerate(BRANCHES):
        seed = seeds[branch]
        seed_xy = np.asarray(
            [float(seed["center_x"]), float(seed["center_y"])], dtype=float
        )
        for image_index, image_record in enumerate(image_rows):
            frame = int(image_record["frame"])
            energy_ev = float(image_record["energy_ev"])
            if frame != image_index:
                raise TracerValidationError("Frozen frame-image order changed")
            image_path = relative_file(
                workspace_root, image_record["image_source_path"]
            )
            if sha256(image_path) != image_record["image_sha256"]:
                raise TracerValidationError(
                    f"Raw tracer image hash mismatch: frame {frame}"
                )
            proposal = arrays["tracks_xy"][branch_index, image_index]
            proposal_available = bool(
                arrays["visible_mask"][branch_index, image_index]
            ) and bool(np.all(np.isfinite(proposal)))
            scale = math.sqrt(reference_energy / energy_ev)
            expected_xy = pattern_center + (seed_xy - pattern_center) * scale
            x0 = int(round(float(proposal[0]))) - half if proposal_available else None
            y0 = int(round(float(proposal[1]))) - half if proposal_available else None
            x1 = x0 + roi_side if x0 is not None else None
            y1 = y0 + roi_side if y0 is not None else None

            row = cast(CandidateRecord, {field: None for field in CANDIDATE_FIELDS})
            row.update(
                {
                    "record_key": record_key(frame, branch),
                    "case_id": "scan_a_pair_1",
                    "series_id": "scan_a",
                    "pair_id": "pair_1",
                    "method": "locotrack",
                    "frame": frame,
                    "energy_ev": energy_ev,
                    "branch": branch,
                    "physics_baseline_status": statuses[(frame, branch)],
                    "opportunity_key": opportunities.get((frame, branch)),
                    "image_artifact_id": image_record["image_artifact_id"],
                    "image_source_path": image_record["image_source_path"],
                    "image_sha256": image_record["image_sha256"],
                    "proposal_source_path": PROPOSAL_PATH.as_posix(),
                    "proposal_sha256": proposal_sha256,
                    "proposal_available": proposal_available,
                    "native_x": float(proposal[0]) if proposal_available else None,
                    "native_y": float(proposal[1]) if proposal_available else None,
                    "native_mask_area": float(
                        arrays["mask_areas"][branch_index, image_index]
                    ),
                    "native_mask_score": float(
                        arrays["mask_scores"][branch_index, image_index]
                    ),
                    "native_mask_component_count": int(
                        arrays["mask_component_counts"][branch_index, image_index]
                    ),
                    "predicted_x": float(expected_xy[0]),
                    "predicted_y": float(expected_xy[1]),
                    "roi_side_px": roi_side,
                    "crop_x0": x0,
                    "crop_y0": y0,
                    "crop_x1": x1,
                    "crop_y1": y1,
                    "roi_valid": False,
                    "solver_status": "not_run",
                    "finite_candidate": False,
                    "missing_state": "proposal_unavailable",
                }
            )
            if not proposal_available:
                rows.append(row)
                continue

            image = img_as_float(tifffile.imread(image_path))
            if image.ndim != 2:
                raise TracerValidationError(f"Expected grayscale TIFF: {image_path}")
            roi_valid = bool(
                x0 is not None
                and y0 is not None
                and x1 is not None
                and y1 is not None
                and x0 >= 0
                and y0 >= 0
                and x1 <= image.shape[1]
                and y1 <= image.shape[0]
            )
            row["roi_valid"] = roi_valid
            if not roi_valid:
                row["missing_state"] = "roi_outside_image"
                rows.append(row)
                continue

            fit = fit_spot_to_gaussian(
                image[y0:y1, x0:x1],
                center_preference_factor=center_preference,
                theta_bounds=theta_bounds,
            )
            if fit is None:
                row.update(
                    {"solver_status": "failed", "missing_state": "solver_failure"}
                )
                rows.append(row)
                continue

            center_x = x0 + float(fit["center_x"])
            center_y = y0 + float(fit["center_y"])
            proposal_offset = math.hypot(
                center_x - float(proposal[0]), center_y - float(proposal[1])
            )
            residual = np.asarray([center_x, center_y]) - expected_xy
            expected_radius = float(np.linalg.norm(expected_xy - pattern_center))
            radial_unit = (expected_xy - pattern_center) / expected_radius
            tangent_unit = np.asarray([-radial_unit[1], radial_unit[0]])
            branch_error = float(np.linalg.norm(residual))
            finite_values = (
                center_x,
                center_y,
                float(fit["amplitude"]),
                float(fit["r_squared"]),
                float(fit["sigma_x"]),
                float(fit["sigma_y"]),
            )
            finite_candidate = all(math.isfinite(value) for value in finite_values)
            row.update(
                {
                    "solver_status": "completed",
                    "finite_candidate": finite_candidate,
                    "missing_state": (
                        "none" if finite_candidate else "non_finite_parameters"
                    ),
                    "center_x": center_x,
                    "center_y": center_y,
                    "amplitude": float(fit["amplitude"]),
                    "r_squared": float(fit["r_squared"]),
                    "volume": float(fit["volume"]),
                    "sigma_x": float(fit["sigma_x"]),
                    "sigma_y": float(fit["sigma_y"]),
                    "theta": float(fit["theta"]),
                    "offset": float(fit["offset"]),
                    "center_x_error": float(fit["center_x_error"]),
                    "center_y_error": float(fit["center_y_error"]),
                    "amplitude_error": float(fit["amplitude_error"]),
                    "sigma_x_error": float(fit["sigma_x_error"]),
                    "sigma_y_error": float(fit["sigma_y_error"]),
                    "theta_error": float(fit["theta_error"]),
                    "offset_error": float(fit["offset_error"]),
                    "volume_error": float(fit["volume_error"]),
                    "proposal_offset_px": proposal_offset,
                    "branch_error_px": branch_error,
                    "radial_residual_px": float(np.dot(residual, radial_unit)),
                    "tangential_residual_px": float(np.dot(residual, tangent_unit)),
                }
            )
            row.update(signed_gate_margins(row, measurement_contract))
            rows.append(row)
    return rows


def reconstruct_pairs(
    candidates: Sequence[CandidateRecord],
    evaluations: Sequence[EvaluationRecord],
    workspace_root: Path,
) -> list[PairRecord]:
    evaluation_by_key = {row["record_key"]: row for row in evaluations}
    consistent: dict[int, dict[str, CandidateRecord]] = {}
    for candidate in candidates:
        evaluation = evaluation_by_key[candidate["record_key"]]
        if evaluation["measurement_status"] == "branch_consistent":
            consistent.setdefault(int(candidate["frame"]), {})[
                str(candidate["branch"])
            ] = candidate
    case_summary = read_json(workspace_root / CASE_SUMMARY_PATH)
    seeds = {
        str(seed["branch"]): seed for seed in case_summary["frozen_seeds"]
    }
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
    rows: list[PairRecord] = []
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
            PairRecord(
                tracker="locotrack",
                frame=frame,
                energy_ev=energy_ev,
                pair_distance_px=distance,
                predicted_distance_px=predicted,
                distance_residual_px=distance - predicted,
                midpoint_x=float(midpoint[0]),
                midpoint_y=float(midpoint[1]),
                midpoint_drift_px=float(np.linalg.norm(midpoint - reference_midpoint)),
            )
        )
    return rows


def reconstruct_gaps(
    evaluations: Sequence[EvaluationRecord],
    workspace_root: Path,
) -> list[GapRecord]:
    _, baseline_rows = read_raw_csv(workspace_root / CASE_MEASUREMENTS_PATH)
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
        read_json(workspace_root / CASE_SUMMARY_PATH)["reference_frame"]
    )
    rows: list[GapRecord] = []
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
            step = 1 if direction == "high" else -1
            recovery = None
            delay = 0
            frame = first_expected
            while (
                frame in energy_by_frame
                and status[(frame, branch)] == "visible_fit"
            ):
                if (
                    evaluation[(frame, branch)]["measurement_status"]
                    == "branch_consistent"
                ):
                    recovery = frame
                    break
                delay += 1
                frame += step
            rows.append(
                GapRecord(
                    tracker="locotrack",
                    branch=branch,
                    direction=direction,
                    gap_start_energy_ev=energy_by_frame[start],
                    gap_end_energy_ev=energy_by_frame[end],
                    gap_frame_count=len(run),
                    first_expected_recovery_energy_ev=energy_by_frame[
                        first_expected
                    ],
                    recovered=recovery is not None,
                    recovery_energy_ev=(
                        energy_by_frame[recovery] if recovery is not None else None
                    ),
                    recovery_delay_frames=delay if recovery is not None else None,
                )
            )
    return rows


def summarize(
    candidates: Sequence[CandidateRecord],
    evaluations: Sequence[EvaluationRecord],
    pairs: Sequence[PairRecord],
    gaps: Sequence[GapRecord],
) -> dict[str, Any]:
    by_key = {row["record_key"]: row for row in candidates}
    baseline_visible = [
        row for row in evaluations if row["physics_baseline_status"] == "visible_fit"
    ]
    consistent = [
        row
        for row in baseline_visible
        if row["measurement_status"] == "branch_consistent"
    ]
    accepted = [row for row in evaluations if bool(row["measurement_valid"])]
    inconsistent = [
        row
        for row in accepted
        if row["measurement_status"] == "geometry_inconsistent"
    ]
    residuals = [abs(float(row["distance_residual_px"])) for row in pairs]
    recovered = [row for row in gaps if bool(row["recovered"])]
    return {
        "tracker": "locotrack",
        "physics_baseline_visible_opportunities": len(baseline_visible),
        "native_visible_on_baseline_visible": sum(
            bool(by_key[row["record_key"]]["proposal_available"])
            for row in baseline_visible
        ),
        "branch_consistent_measurements": len(consistent),
        "fit_coverage_on_baseline_visible": len(consistent) / len(baseline_visible),
        "accepted_measurements_all_frames": len(accepted),
        "geometry_inconsistent_accepted_fits": len(inconsistent),
        "jointly_consistent_pair_frames": len(pairs),
        "absolute_pair_distance_residual_px": {
            "median": float(np.median(residuals)),
            "p95": float(np.percentile(residuals, 95)),
            "max": max(residuals),
        },
        "post_gap_recovery": {
            "evaluated_gap_count": len(gaps),
            "recovered_gap_count": len(recovered),
            "rate": len(recovered) / len(gaps),
            "zero_delay_count": sum(
                int(row["recovery_delay_frames"]) == 0 for row in recovered
            ),
        },
    }


def reader_facing(
    summary: Mapping[str, Any],
    rounding: Mapping[str, Any],
) -> dict[str, object]:
    visible = int(summary["physics_baseline_visible_opportunities"])
    consistent = int(summary["branch_consistent_measurements"])
    accepted = int(summary["accepted_measurements_all_frames"])
    inconsistent = int(summary["geometry_inconsistent_accepted_fits"])
    gaps = summary["post_gap_recovery"]
    rate_places = int(rounding["rates_percent"]["decimal_places"])
    residual_places = int(
        rounding["pair_distance_residual_px"]["decimal_places"]
    )
    return {
        "coverage": (
            f"{consistent}/{visible} "
            f"({100.0 * consistent / visible:.{rate_places}f}%)"
        ),
        "post_gap_recovery": (
            f"{int(gaps['recovered_gap_count'])}/{int(gaps['evaluated_gap_count'])} "
            f"({100.0 * float(gaps['rate']):.{rate_places}f}%)"
        ),
        "geometry_inconsistency": (
            f"{inconsistent}/{accepted} "
            f"({100.0 * inconsistent / accepted:.{rate_places}f}%)"
        ),
        "conditional_pair_residual_median": (
            f"{float(summary['absolute_pair_distance_residual_px']['median']):.{residual_places}f} px"
        ),
    }


def reconstruct_tracer_artifacts(
    workspace_root: Path,
    tracer_dir: Path,
    contract: Mapping[str, Any],
) -> TracerArtifacts:
    """Reconstruct every scientific artifact behind one typed interface."""

    candidates = reconstruct_candidates(workspace_root, tracer_dir, contract)
    evaluations = evaluate_candidates(candidates, central_contract(workspace_root))
    pairs = reconstruct_pairs(candidates, evaluations, workspace_root)
    gaps = reconstruct_gaps(evaluations, workspace_root)
    return TracerArtifacts(
        candidates=candidates,
        evaluations=evaluations,
        pairs=pairs,
        gaps=gaps,
        summary=summarize(candidates, evaluations, pairs, gaps),
    )


__all__ = [
    "TracerArtifacts",
    "central_contract",
    "reader_facing",
    "record_key",
    "reconstruct_tracer_artifacts",
]
