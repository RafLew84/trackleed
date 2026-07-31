"""Candidate fitting for the complete Ag(111) central panel."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import tifffile
from skimage.util import img_as_float

from trackleed.analysis.peak_analyzer import fit_spot_to_gaussian
from trackleed.analysis.scientific_panel_common import (
    BASELINE_INPUT_MANIFEST_PATH,
    BASELINE_PACKAGE,
    CASE_SPECS,
    MEASUREMENT_CONTRACT_PATH,
    METHOD_BY_ID,
    METHODS,
    OPPORTUNITIES_PATH,
    TracerValidationError,
    read_json,
    relative_file,
    sha256,
)
from trackleed.analysis.scientific_tracer_common import BRANCHES
from trackleed.analysis.scientific_tracer_evaluation import signed_gate_margins
from trackleed.analysis.scientific_tracer_records import (
    CANDIDATE_FIELDS,
    CandidateRecord,
    read_raw_csv,
)


def panel_record_key(
    case_id: str,
    method: str,
    frame: int,
    branch: str,
    roi_side: int,
) -> str:
    return (
        f"case={case_id}|method={method}|frame={frame}|branch={branch}|roi={roi_side}"
    )


def central_contract(workspace_root: Path) -> dict[str, Any]:
    contract = read_json(workspace_root / MEASUREMENT_CONTRACT_PATH)
    central = contract.get("central_contract")
    if not isinstance(central, dict):
        raise TracerValidationError("Central measurement contract is missing")
    return central


def _input_index(
    workspace_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], dict[str, Any]]]:
    manifest = read_json(workspace_root / BASELINE_INPUT_MANIFEST_PATH)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise TracerValidationError("Frozen baseline input manifest is invalid")
    images: dict[str, list[dict[str, Any]]] = {}
    proposals: dict[tuple[str, str], dict[str, Any]] = {}
    for value in artifacts:
        if not isinstance(value, dict):
            raise TracerValidationError("Frozen baseline artifact is invalid")
        role = value.get("role")
        if role == "raw_image":
            images.setdefault(str(value["series_id"]), []).append(value)
        elif role == "native_proposals":
            proposals[(str(value["case_id"]), str(value["method"]))] = value
    for records in images.values():
        records.sort(key=lambda row: str(row["source_path"]))
    return images, proposals


def _proposal_arrays(
    workspace_root: Path,
    proposal_record: Mapping[str, Any],
    method: str,
    frame_count: int,
    reference_frame: int,
) -> dict[str, np.ndarray]:
    package_path = str(proposal_record.get("package_path", ""))
    proposal_path = relative_file(
        workspace_root, (BASELINE_PACKAGE / package_path).as_posix()
    )
    proposal_codec = METHOD_BY_ID[method].proposal_codec
    if proposal_codec == "npz" and proposal_path.suffix == ".npz":
        with np.load(proposal_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    elif proposal_codec == "easyleed_csv" and proposal_path.suffix == ".csv":
        tracks = np.full((2, frame_count, 2), np.nan, dtype=np.float32)
        visible = np.zeros((2, frame_count), dtype=bool)
        areas = np.zeros((2, frame_count), dtype=np.float32)
        scores = np.full((2, frame_count), np.nan, dtype=np.float32)
        components = np.zeros((2, frame_count), dtype=np.int32)
        _, rows = read_raw_csv(proposal_path)
        branch_index = {branch: index for index, branch in enumerate(BRANCHES)}
        for row in rows:
            frame = int(row["frame"])
            if row["direction"] == "low" and frame == reference_frame:
                continue
            branch = row["branch"]
            if branch not in branch_index or not 0 <= frame < frame_count:
                raise TracerValidationError("Frozen EasyLEED proposal key changed")
            proposal = np.asarray(
                [float(row["native_x"]), float(row["native_y"])],
                dtype=np.float32,
            )
            index = branch_index[branch]
            tracks[index, frame] = proposal
            visible[index, frame] = bool(np.all(np.isfinite(proposal)))
        arrays = {
            "tracks_xy": tracks,
            "visible_mask": visible,
            "mask_areas": areas,
            "mask_scores": scores,
            "mask_component_counts": components,
        }
    else:
        raise TracerValidationError(
            f"Unsupported frozen proposal format: {proposal_path}"
        )

    shapes = {
        "tracks_xy": (2, frame_count, 2),
        "visible_mask": (2, frame_count),
        "mask_areas": (2, frame_count),
        "mask_scores": (2, frame_count),
        "mask_component_counts": (2, frame_count),
    }
    for name, shape in shapes.items():
        if arrays.get(name, np.empty(0)).shape != shape:
            raise TracerValidationError(
                f"Frozen {method} proposal shape changed: {name}"
            )
    return arrays


def reconstruct_panel_candidates(
    workspace_root: Path,
    contract: Mapping[str, Any],
    *,
    measurement_contract: Mapping[str, Any] | None = None,
) -> list[CandidateRecord]:
    """Fit every method/case/branch/image proposal for one ROI contract."""

    measurement_contract = measurement_contract or central_contract(workspace_root)
    roi_side = int(measurement_contract["roi_side_px"])
    half = roi_side // 2
    center_preference = float(measurement_contract["center_preference_factor"])
    compatibility = contract["estimator_compatibility"]
    theta_bounds = tuple(
        float(value) for value in compatibility["optimizer_theta_bounds_rad"]
    )
    images_by_series, proposal_index = _input_index(workspace_root)
    _, opportunity_rows = read_raw_csv(workspace_root / OPPORTUNITIES_PATH)
    opportunities = {
        (row["case_id"], int(row["frame"]), row["branch"]): row["opportunity_key"]
        for row in opportunity_rows
    }
    image_cache: dict[Path, np.ndarray] = {}
    rows: list[CandidateRecord] = []

    for case_spec in CASE_SPECS:
        case_id = case_spec.case_id
        series_id = case_spec.series_id
        pair_id = case_spec.pair_id
        frame_count = case_spec.frame_count
        case_dir = BASELINE_PACKAGE / "frozen_inputs" / "cases" / case_id
        case_summary = read_json(workspace_root / case_dir / "summary.json")
        _, baseline_rows = read_raw_csv(
            workspace_root / case_dir / "scan_measurements.csv"
        )
        statuses = {
            (int(row["frame"]), row["branch"]): row["status"] for row in baseline_rows
        }
        energy_by_frame = {
            int(row["frame"]): float(row["energy_ev"]) for row in baseline_rows
        }
        geometry = case_summary["geometry"]
        pattern_center = np.asarray(
            [float(geometry["center_x"]), float(geometry["center_y"])],
            dtype=float,
        )
        reference_energy = float(case_summary["reference_energy_ev"])
        reference_frame = int(case_summary["reference_frame"])
        seeds = {str(seed["branch"]): seed for seed in case_summary["frozen_seeds"]}
        if set(seeds) != set(BRANCHES):
            raise TracerValidationError(f"Frozen branch seeds changed: {case_id}")
        image_records = images_by_series.get(series_id, [])
        if len(image_records) != frame_count:
            raise TracerValidationError(f"Frozen image count changed: {case_id}")

        for method in METHODS:
            proposal_record = proposal_index.get((case_id, method))
            if proposal_record is None:
                raise TracerValidationError(
                    f"Frozen proposal is missing: {case_id}/{method}"
                )
            proposal_path_text = (
                BASELINE_PACKAGE / str(proposal_record["package_path"])
            ).as_posix()
            proposal_path = relative_file(workspace_root, proposal_path_text)
            if sha256(proposal_path) != proposal_record["source_sha256"]:
                raise TracerValidationError(
                    f"Frozen proposal hash mismatch: {case_id}/{method}"
                )
            arrays = _proposal_arrays(
                workspace_root,
                proposal_record,
                method,
                frame_count,
                reference_frame,
            )
            proposal_sha256 = sha256(proposal_path)
            for branch_index, branch in enumerate(BRANCHES):
                seed = seeds[branch]
                seed_xy = np.asarray(
                    [float(seed["center_x"]), float(seed["center_y"])],
                    dtype=float,
                )
                for frame, image_record in enumerate(image_records):
                    energy_ev = energy_by_frame[frame]
                    image_path_text = str(image_record["source_path"])
                    image_path = relative_file(workspace_root, image_path_text)
                    if image_path not in image_cache:
                        if sha256(image_path) != image_record["source_sha256"]:
                            raise TracerValidationError(
                                f"Raw panel image hash mismatch: {image_path_text}"
                            )
                        image = img_as_float(tifffile.imread(image_path))
                        if image.ndim != 2:
                            raise TracerValidationError(
                                f"Expected grayscale TIFF: {image_path}"
                            )
                        image_cache[image_path] = image
                    image = image_cache[image_path]
                    proposal = arrays["tracks_xy"][branch_index, frame]
                    proposal_available = bool(
                        arrays["visible_mask"][branch_index, frame]
                    ) and bool(np.all(np.isfinite(proposal)))
                    scale = math.sqrt(reference_energy / energy_ev)
                    expected_xy = pattern_center + (seed_xy - pattern_center) * scale
                    x0 = (
                        int(round(float(proposal[0]))) - half
                        if proposal_available
                        else None
                    )
                    y0 = (
                        int(round(float(proposal[1]))) - half
                        if proposal_available
                        else None
                    )
                    x1 = x0 + roi_side if x0 is not None else None
                    y1 = y0 + roi_side if y0 is not None else None
                    row = cast(
                        CandidateRecord,
                        {field: None for field in CANDIDATE_FIELDS},
                    )
                    row.update(
                        {
                            "record_key": panel_record_key(
                                case_id, method, frame, branch, roi_side
                            ),
                            "case_id": case_id,
                            "series_id": series_id,
                            "pair_id": pair_id,
                            "method": method,
                            "frame": frame,
                            "energy_ev": energy_ev,
                            "branch": branch,
                            "physics_baseline_status": statuses[(frame, branch)],
                            "opportunity_key": opportunities.get(
                                (case_id, frame, branch)
                            ),
                            "image_artifact_id": str(image_record["artifact_id"]),
                            "image_source_path": image_path_text,
                            "image_sha256": str(image_record["source_sha256"]),
                            "proposal_source_path": proposal_path_text,
                            "proposal_sha256": proposal_sha256,
                            "proposal_available": proposal_available,
                            "native_x": (
                                float(proposal[0]) if proposal_available else None
                            ),
                            "native_y": (
                                float(proposal[1]) if proposal_available else None
                            ),
                            "native_mask_area": float(
                                arrays["mask_areas"][branch_index, frame]
                            ),
                            "native_mask_score": float(
                                arrays["mask_scores"][branch_index, frame]
                            ),
                            "native_mask_component_count": int(
                                arrays["mask_component_counts"][branch_index, frame]
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
                            {
                                "solver_status": "failed",
                                "missing_state": "solver_failure",
                            }
                        )
                        rows.append(row)
                        continue
                    center_x = x0 + float(fit["center_x"])
                    center_y = y0 + float(fit["center_y"])
                    proposal_offset = math.hypot(
                        center_x - float(proposal[0]),
                        center_y - float(proposal[1]),
                    )
                    residual = np.asarray([center_x, center_y]) - expected_xy
                    expected_radius = float(
                        np.linalg.norm(expected_xy - pattern_center)
                    )
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
                    finite_candidate = all(
                        math.isfinite(value) for value in finite_values
                    )
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
                            "tangential_residual_px": float(
                                np.dot(residual, tangent_unit)
                            ),
                        }
                    )
                    row.update(signed_gate_margins(row, measurement_contract))
                    rows.append(row)
    return rows


__all__ = ["central_contract", "panel_record_key", "reconstruct_panel_candidates"]
