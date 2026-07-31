"""Headless reconstruction tracer for Scan A, pair 1 and LocoTrack."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version as distribution_version
import json
import math
from pathlib import Path
import platform
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import tifffile
from skimage.util import img_as_float

from trackleed.analysis.peak_analyzer import fit_spot_to_gaussian
from trackleed.analysis.scientific_baseline_validation import validate_package


SCHEMA_VERSION = 1
TRACER_ID = "scan_a_pair_1_locotrack_central"
BASELINE_PACKAGE = Path("manuscript/scientific_baseline")
PROPOSAL_PATH = Path(
    "manuscript/scientific_baseline/frozen_inputs/proposals/"
    "scan_a_pair_1/locotrack.npz"
)
CASE_SUMMARY_PATH = Path(
    "manuscript/scientific_baseline/frozen_inputs/cases/"
    "scan_a_pair_1/summary.json"
)
CASE_MEASUREMENTS_PATH = Path(
    "manuscript/scientific_baseline/frozen_inputs/cases/"
    "scan_a_pair_1/scan_measurements.csv"
)
MEASUREMENT_CONTRACT_PATH = Path(
    "manuscript/scientific_baseline/measurement_contract.json"
)
OPPORTUNITIES_PATH = Path("manuscript/scientific_baseline/opportunities.csv")
PUBLISHED_CASE_RESULTS_PATH = Path(
    "manuscript/figures/figure03_independent_scane_stress/"
    "source_data/case_results.csv"
)
BRANCHES = ("B1", "B2")
FROZEN_PACKAGE_HASHES = {
    "README.md": "4d755f9b0f842f6ae29236fa5595d7162ab0e585207b0ddf88e62122d955c895",
    "frozen/expected_central_records.csv": "e739350ebc78894de4a54fa1731494d1b141e251b81ea3710de507df81c9c1b7",
    "frozen/expected_gap_records.csv": "8f5a4497ea7f13f8c61940d5db3ac67dcf4e851c469d6dd38a40723ef722f44d",
    "frozen/expected_pair_records.csv": "3cd977f08eb2e670948e670bb0232d0a28624a1e39e9b5aca4f931ad475dc8fa",
    "frozen/expected_summary.json": "346d8f079d48a041a72897bf4d553f2e3c2f3433f5eec703832e8eca26c21ee8",
    "frozen/frame_images.csv": "7630f1ddbf60b4ebe8906aed6899c8e6fcee05c5cf415bd5a2e73c8c9f735acf",
    "frozen/tracer_contract.json": "71485886573a03ecdb455ac5c84e7e14ff04b8d8cec1d70ec5730805019a7472",
}

CANDIDATE_FIELDS = (
    "record_key",
    "case_id",
    "series_id",
    "pair_id",
    "method",
    "frame",
    "energy_ev",
    "branch",
    "physics_baseline_status",
    "opportunity_key",
    "image_artifact_id",
    "image_source_path",
    "image_sha256",
    "proposal_source_path",
    "proposal_sha256",
    "proposal_available",
    "native_x",
    "native_y",
    "native_mask_area",
    "native_mask_score",
    "native_mask_component_count",
    "predicted_x",
    "predicted_y",
    "roi_side_px",
    "crop_x0",
    "crop_y0",
    "crop_x1",
    "crop_y1",
    "roi_valid",
    "solver_status",
    "finite_candidate",
    "missing_state",
    "center_x",
    "center_y",
    "amplitude",
    "r_squared",
    "volume",
    "sigma_x",
    "sigma_y",
    "theta",
    "offset",
    "center_x_error",
    "center_y_error",
    "amplitude_error",
    "sigma_x_error",
    "sigma_y_error",
    "theta_error",
    "offset_error",
    "volume_error",
    "proposal_offset_px",
    "branch_error_px",
    "radial_residual_px",
    "tangential_residual_px",
    "proposal_to_fit_margin_px",
    "r_squared_margin",
    "amplitude_margin",
    "sigma_x_lower_margin_px",
    "sigma_x_upper_margin_px",
    "sigma_y_lower_margin_px",
    "sigma_y_upper_margin_px",
    "branch_margin_px",
)

EVALUATION_FIELDS = (
    "record_key",
    "case_id",
    "method",
    "frame",
    "energy_ev",
    "branch",
    "physics_baseline_status",
    "opportunity_key",
    "variant_id",
    "accepted_fit",
    "measurement_valid",
    "measurement_status",
    "classification",
    "first_rejection_cause",
    "full_rejection_causes",
    "legacy_fit_reason",
    "proposal_to_fit_margin_px",
    "r_squared_margin",
    "amplitude_margin",
    "sigma_x_lower_margin_px",
    "sigma_x_upper_margin_px",
    "sigma_y_lower_margin_px",
    "sigma_y_upper_margin_px",
    "branch_margin_px",
)

PAIR_FIELDS = (
    "tracker",
    "frame",
    "energy_ev",
    "pair_distance_px",
    "predicted_distance_px",
    "distance_residual_px",
    "midpoint_x",
    "midpoint_y",
    "midpoint_drift_px",
)

GAP_FIELDS = (
    "tracker",
    "branch",
    "direction",
    "gap_start_energy_ev",
    "gap_end_energy_ev",
    "gap_frame_count",
    "first_expected_recovery_energy_ev",
    "recovered",
    "recovery_energy_ev",
    "recovery_delay_frames",
)


class TracerValidationError(ValueError):
    """The reconstruction tracer differs from frozen evidence or contract."""


@dataclass(frozen=True)
class TracerReport:
    candidate_count: int
    evaluation_count: int
    branch_consistent_count: int
    evaluated_gap_count: int
    recovered_gap_count: int
    matches_frozen_central_record: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256(path: Path) -> str:
    """Hash source text independently of a checkout's newline convention."""

    canonical = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TracerValidationError(f"Cannot read tracer JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TracerValidationError(f"Expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            return fields, list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise TracerValidationError(f"Cannot read tracer CSV: {path}") from exc


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _relative_file(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise TracerValidationError(f"Tracer path must be relative: {value}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise TracerValidationError(f"Tracer path escapes workspace: {value}")
    return resolved


def _validate_package_files(tracer_dir: Path) -> dict[str, Any]:
    manifest = _read_json(tracer_dir / "package_manifest.json")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("tracer_id") != TRACER_ID
        or manifest.get("immutable") is not True
    ):
        raise TracerValidationError("Frozen tracer package identity changed")
    expected = manifest.get("files_sha256")
    if not isinstance(expected, dict):
        raise TracerValidationError("Frozen tracer manifest has no file hash map")
    if expected != FROZEN_PACKAGE_HASHES:
        raise TracerValidationError("Frozen tracer manifest contract changed")
    actual = {
        path.relative_to(tracer_dir).as_posix(): path
        for path in sorted(tracer_dir.rglob("*"))
        if path.is_file()
        and path.name != "package_manifest.json"
        and "output" not in path.relative_to(tracer_dir).parts
    }
    if set(actual) != set(expected):
        raise TracerValidationError("Frozen tracer package inventory changed")
    for relative, path in actual.items():
        if _sha256(path) != expected[relative]:
            raise TracerValidationError(f"Frozen tracer package hash mismatch: {relative}")
    contract = _read_json(tracer_dir / "frozen" / "tracer_contract.json")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("tracer_id") != TRACER_ID
        or contract.get("case_id") != "scan_a_pair_1"
        or contract.get("method") != "locotrack"
    ):
        raise TracerValidationError("Frozen tracer contract identity changed")
    schemas = contract.get("expected_output_schemas")
    if not isinstance(schemas, dict):
        raise TracerValidationError("Frozen tracer output schemas are missing")
    if schemas.get("candidate_records.csv") != list(CANDIDATE_FIELDS):
        raise TracerValidationError("Frozen candidate schema changed")
    if schemas.get("evaluation_records.csv") != list(EVALUATION_FIELDS):
        raise TracerValidationError("Frozen evaluation schema changed")
    _validate_contract_semantics(contract)
    return contract


def _validate_contract_semantics(contract: Mapping[str, Any]) -> None:
    if contract.get("expected_counts") != {
        "branch_consistent_measurements": 44,
        "candidate_records": 70,
        "evaluated_gaps": 5,
        "evaluation_records": 70,
        "geometry_inconsistent_accepted_fits": 9,
        "jointly_consistent_pair_frames": 18,
        "recovered_gaps": 5,
    }:
        raise TracerValidationError("Frozen tracer expected counts changed")
    if contract.get("comparison_tolerances") != {
        "continuous_values": {"atol": 1e-8, "rtol": 1e-9},
        "discrete_values": "exact",
        "reader_facing_values": "exact_after_declared_rounding",
    }:
        raise TracerValidationError("Frozen tracer comparison tolerances changed")
    compatibility = contract.get("estimator_compatibility")
    if not isinstance(compatibility, dict) or any(
        (
            compatibility.get("center_preference_factor") != 0.34,
            compatibility.get("image_loading")
            != "tifffile.imread followed by skimage.util.img_as_float",
            compatibility.get("mode") != "historical_publication_central_record",
            compatibility.get("optimizer_theta_bounds_rad") != [-math.pi, math.pi],
        )
    ):
        raise TracerValidationError("Frozen tracer estimator compatibility changed")
    if contract.get("reader_facing_rounding") != {
        "counts": "exact",
        "pair_distance_residual_px": {"decimal_places": 3},
        "rates_percent": {"decimal_places": 1},
    }:
        raise TracerValidationError("Frozen tracer reader-facing rounding changed")
    if contract.get("execution_contract") != {
        "device": "cpu",
        "runtime": {
            "numpy": "2.2.5",
            "python": "3.12.9",
            "scikit-image": "0.25.2",
            "scipy": "1.15.2",
            "tifffile": "2025.3.30",
        },
    }:
        raise TracerValidationError("Frozen tracer execution contract changed")
    expected_paths = {
        "case_measurements": CASE_MEASUREMENTS_PATH.as_posix(),
        "case_summary": CASE_SUMMARY_PATH.as_posix(),
        "frame_images": "frozen/frame_images.csv",
        "gaps": "manuscript/scientific_baseline/gaps.csv",
        "measurement_contract": MEASUREMENT_CONTRACT_PATH.as_posix(),
        "opportunities": OPPORTUNITIES_PATH.as_posix(),
        "proposals": PROPOSAL_PATH.as_posix(),
        "published_case_results": PUBLISHED_CASE_RESULTS_PATH.as_posix(),
    }
    inputs = contract.get("frozen_inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(expected_paths):
        raise TracerValidationError("Frozen tracer input set changed")
    for name, path in expected_paths.items():
        if not isinstance(inputs[name], dict) or inputs[name].get("path") != path:
            raise TracerValidationError(f"Frozen tracer input path changed: {name}")


def _runtime_fingerprint() -> dict[str, str]:
    return {
        "numpy": distribution_version("numpy"),
        "python": platform.python_version(),
        "scikit-image": distribution_version("scikit-image"),
        "scipy": distribution_version("scipy"),
        "tifffile": distribution_version("tifffile"),
    }


def _validate_runtime(contract: Mapping[str, Any]) -> dict[str, str]:
    runtime = _runtime_fingerprint()
    expected = contract["execution_contract"]["runtime"]
    if runtime != expected:
        raise TracerValidationError(
            f"Tracer runtime differs from frozen CPU environment: {runtime!r}"
        )
    return runtime


def _validate_input_hashes(
    workspace_root: Path,
    tracer_dir: Path,
    contract: Mapping[str, Any],
) -> None:
    inputs = contract.get("frozen_inputs")
    if not isinstance(inputs, dict):
        raise TracerValidationError("Frozen tracer inputs are missing")
    for name, record in inputs.items():
        if not isinstance(record, dict):
            raise TracerValidationError(f"Invalid frozen tracer input: {name}")
        path_text = str(record.get("path", ""))
        base = tracer_dir if path_text.startswith("frozen/") else workspace_root
        path = _relative_file(base, path_text)
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise TracerValidationError(f"Frozen tracer input hash mismatch: {name}")


def _central_contract(workspace_root: Path) -> dict[str, Any]:
    contract = _read_json(workspace_root / MEASUREMENT_CONTRACT_PATH)
    central = contract.get("central_contract")
    if not isinstance(central, dict):
        raise TracerValidationError("Central measurement contract is missing")
    return central


def _record_key(frame: int, branch: str) -> str:
    return (
        f"case=scan_a_pair_1|method=locotrack|frame={frame}|"
        f"branch={branch}|roi=48"
    )


def _load_opportunity_keys(workspace_root: Path) -> dict[tuple[int, str], str]:
    _, rows = _read_csv(workspace_root / OPPORTUNITIES_PATH)
    return {
        (int(row["frame"]), row["branch"]): row["opportunity_key"]
        for row in rows
        if row["case_id"] == "scan_a_pair_1"
    }


def _load_baseline_statuses(workspace_root: Path) -> dict[tuple[int, str], str]:
    _, rows = _read_csv(workspace_root / CASE_MEASUREMENTS_PATH)
    return {
        (int(row["frame"]), row["branch"]): row["status"] for row in rows
    }


def _candidate_rows(
    workspace_root: Path,
    tracer_dir: Path,
    contract: Mapping[str, Any],
) -> list[dict[str, object]]:
    central = _central_contract(workspace_root)
    roi_side = int(central["roi_side_px"])
    half = roi_side // 2
    width_gate = central["width_gate"]
    proposal_max = float(central["proposal_to_fit_max_px"])
    r_squared_min = float(central["r_squared_min"])
    amplitude_min = float(central["amplitude_reference_min"]) * float(
        central["amplitude_gate_multiplier"]
    )
    sigma_min = float(width_gate["sigma_min_px"])
    sigma_max = float(width_gate["sigma_max_px"])
    branch_tolerance = float(central["branch_tolerance_px"])
    center_preference = float(central["center_preference_factor"])
    compatibility = contract["estimator_compatibility"]
    theta_bounds = tuple(
        float(value) for value in compatibility["optimizer_theta_bounds_rad"]
    )

    summary = _read_json(workspace_root / CASE_SUMMARY_PATH)
    geometry = summary["geometry"]
    pattern_center = np.asarray(
        [float(geometry["center_x"]), float(geometry["center_y"])],
        dtype=float,
    )
    reference_energy = float(summary["reference_energy_ev"])
    seeds = {str(seed["branch"]): seed for seed in summary["frozen_seeds"]}
    if set(seeds) != set(BRANCHES):
        raise TracerValidationError("Frozen branch seeds changed")

    _, image_rows = _read_csv(tracer_dir / "frozen" / "frame_images.csv")
    if len(image_rows) != 35:
        raise TracerValidationError("Frozen frame-image map changed")
    proposal_path = workspace_root / PROPOSAL_PATH
    proposal_sha256 = _sha256(proposal_path)
    with np.load(proposal_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if arrays.get("tracks_xy", np.empty(0)).shape != (2, 35, 2):
        raise TracerValidationError("Frozen LocoTrack proposal shape changed")
    if arrays.get("visible_mask", np.empty(0)).shape != (2, 35):
        raise TracerValidationError("Frozen LocoTrack visibility shape changed")

    statuses = _load_baseline_statuses(workspace_root)
    opportunities = _load_opportunity_keys(workspace_root)
    rows: list[dict[str, object]] = []
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
            image_path = _relative_file(
                workspace_root, image_record["image_source_path"]
            )
            if _sha256(image_path) != image_record["image_sha256"]:
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

            row: dict[str, object] = {field: None for field in CANDIDATE_FIELDS}
            row.update(
                {
                    "record_key": _record_key(frame, branch),
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
                    "missing_state": "none" if finite_candidate else "non_finite_parameters",
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
                    "proposal_to_fit_margin_px": proposal_max - proposal_offset,
                    "r_squared_margin": float(fit["r_squared"]) - r_squared_min,
                    "amplitude_margin": float(fit["amplitude"]) - amplitude_min,
                    "sigma_x_lower_margin_px": float(fit["sigma_x"]) - sigma_min,
                    "sigma_x_upper_margin_px": sigma_max - float(fit["sigma_x"]),
                    "sigma_y_lower_margin_px": float(fit["sigma_y"]) - sigma_min,
                    "sigma_y_upper_margin_px": sigma_max - float(fit["sigma_y"]),
                    "branch_margin_px": branch_tolerance - branch_error,
                }
            )
            rows.append(row)
    return rows


def evaluate_candidates(
    candidates: Sequence[Mapping[str, object]],
    central_contract: Mapping[str, Any],
    *,
    variant_id: str = "roi_48_proposal_15",
) -> list[dict[str, object]]:
    """Classify stored candidate fits without rerunning or copying fit values."""

    proposal_max = float(central_contract["proposal_to_fit_max_px"])
    r_squared_min = float(central_contract["r_squared_min"])
    amplitude_min = float(central_contract["amplitude_reference_min"]) * float(
        central_contract["amplitude_gate_multiplier"]
    )
    sigma_min = float(central_contract["width_gate"]["sigma_min_px"])
    sigma_max = float(central_contract["width_gate"]["sigma_max_px"])
    branch_tolerance = float(central_contract["branch_tolerance_px"])
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        margins: dict[str, float | None] = {
            "proposal_to_fit_margin_px": None,
            "r_squared_margin": None,
            "amplitude_margin": None,
            "sigma_x_lower_margin_px": None,
            "sigma_x_upper_margin_px": None,
            "sigma_y_lower_margin_px": None,
            "sigma_y_upper_margin_px": None,
            "branch_margin_px": None,
        }
        if bool(candidate["finite_candidate"]):
            margins.update(
                {
                    "proposal_to_fit_margin_px": proposal_max
                    - float(candidate["proposal_offset_px"]),
                    "r_squared_margin": float(candidate["r_squared"])
                    - r_squared_min,
                    "amplitude_margin": float(candidate["amplitude"])
                    - amplitude_min,
                    "sigma_x_lower_margin_px": float(candidate["sigma_x"])
                    - sigma_min,
                    "sigma_x_upper_margin_px": sigma_max
                    - float(candidate["sigma_x"]),
                    "sigma_y_lower_margin_px": float(candidate["sigma_y"])
                    - sigma_min,
                    "sigma_y_upper_margin_px": sigma_max
                    - float(candidate["sigma_y"]),
                    "branch_margin_px": branch_tolerance
                    - float(candidate["branch_error_px"]),
                }
            )

        causes: list[str] = []
        if not bool(candidate["proposal_available"]):
            causes.append("proposal_unavailable")
        elif not bool(candidate["roi_valid"]):
            causes.append("roi_invalid")
        elif candidate["solver_status"] != "completed":
            causes.append("solver_failure")
        elif not bool(candidate["finite_candidate"]):
            causes.append("non_finite_parameters")
        else:
            if float(margins["proposal_to_fit_margin_px"]) < 0:
                causes.append("proposal_to_fit")
            if float(margins["r_squared_margin"]) < 0:
                causes.append("r_squared")
            if float(margins["amplitude_margin"]) < 0:
                causes.append("amplitude")
            if any(
                float(margins[name]) < 0
                for name in (
                    "sigma_x_lower_margin_px",
                    "sigma_x_upper_margin_px",
                    "sigma_y_lower_margin_px",
                    "sigma_y_upper_margin_px",
                )
            ):
                causes.append("width")
            if float(margins["branch_margin_px"]) < 0:
                causes.append("physics_branch_consistency")

        gaussian_causes = [
            cause for cause in causes if cause != "physics_branch_consistency"
        ]
        accepted_fit = not gaussian_causes
        if gaussian_causes:
            if gaussian_causes[0] in {
                "proposal_unavailable",
                "roi_invalid",
                "solver_failure",
                "non_finite_parameters",
            }:
                measurement_status = gaussian_causes[0]
            else:
                measurement_status = "gaussian_rejected"
        elif "physics_branch_consistency" in causes:
            measurement_status = "geometry_inconsistent"
        else:
            measurement_status = "branch_consistent"

        legacy_reasons: list[str] = []
        if bool(candidate["finite_candidate"]):
            if float(margins["r_squared_margin"]) < 0:
                legacy_reasons.append("low_r2")
            if float(margins["amplitude_margin"]) < 0:
                legacy_reasons.append("low_amplitude")
            if "width" in causes:
                legacy_reasons.append("invalid_sigma")
            if float(margins["proposal_to_fit_margin_px"]) < 0:
                legacy_reasons.append("center_outside_convergence_zone")
        legacy_fit_reason = "accepted" if not legacy_reasons else "+".join(legacy_reasons)
        row = {
            "record_key": candidate["record_key"],
            "case_id": candidate["case_id"],
            "method": candidate["method"],
            "frame": candidate["frame"],
            "energy_ev": candidate["energy_ev"],
            "branch": candidate["branch"],
            "physics_baseline_status": candidate["physics_baseline_status"],
            "opportunity_key": candidate["opportunity_key"],
            "variant_id": variant_id,
            "accepted_fit": accepted_fit,
            "measurement_valid": accepted_fit,
            "measurement_status": measurement_status,
            "classification": measurement_status,
            "first_rejection_cause": causes[0] if causes else None,
            "full_rejection_causes": "+".join(causes),
            "legacy_fit_reason": legacy_fit_reason,
            **margins,
        }
        rows.append(row)
    return rows


def _pair_rows(
    candidates: Sequence[Mapping[str, object]],
    evaluations: Sequence[Mapping[str, object]],
    workspace_root: Path,
) -> list[dict[str, object]]:
    evaluation_by_key = {row["record_key"]: row for row in evaluations}
    consistent: dict[int, dict[str, Mapping[str, object]]] = {}
    for candidate in candidates:
        evaluation = evaluation_by_key[candidate["record_key"]]
        if evaluation["measurement_status"] == "branch_consistent":
            consistent.setdefault(int(candidate["frame"]), {})[
                str(candidate["branch"])
            ] = candidate
    summary = _read_json(workspace_root / CASE_SUMMARY_PATH)
    seeds = {str(seed["branch"]): seed for seed in summary["frozen_seeds"]}
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
    reference_energy = float(summary["reference_energy_ev"])
    rows: list[dict[str, object]] = []
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
            {
                "tracker": "locotrack",
                "frame": frame,
                "energy_ev": energy_ev,
                "pair_distance_px": distance,
                "predicted_distance_px": predicted,
                "distance_residual_px": distance - predicted,
                "midpoint_x": float(midpoint[0]),
                "midpoint_y": float(midpoint[1]),
                "midpoint_drift_px": float(np.linalg.norm(midpoint - reference_midpoint)),
            }
        )
    return rows


def _gap_rows(
    evaluations: Sequence[Mapping[str, object]],
    workspace_root: Path,
) -> list[dict[str, object]]:
    _, baseline_rows = _read_csv(workspace_root / CASE_MEASUREMENTS_PATH)
    status = {
        (int(row["frame"]), row["branch"]): row["status"] for row in baseline_rows
    }
    energy_by_frame: dict[int, float] = {}
    for row in baseline_rows:
        energy_by_frame[int(row["frame"])] = float(row["energy_ev"])
    evaluation = {
        (int(row["frame"]), str(row["branch"])): row for row in evaluations
    }
    reference_frame = int(_read_json(workspace_root / CASE_SUMMARY_PATH)["reference_frame"])
    rows: list[dict[str, object]] = []
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
            while frame in energy_by_frame and status[(frame, branch)] == "visible_fit":
                if evaluation[(frame, branch)]["measurement_status"] == "branch_consistent":
                    recovery = frame
                    break
                delay += 1
                frame += step
            rows.append(
                {
                    "tracker": "locotrack",
                    "branch": branch,
                    "direction": direction,
                    "gap_start_energy_ev": energy_by_frame[start],
                    "gap_end_energy_ev": energy_by_frame[end],
                    "gap_frame_count": len(run),
                    "first_expected_recovery_energy_ev": energy_by_frame[first_expected],
                    "recovered": recovery is not None,
                    "recovery_energy_ev": energy_by_frame[recovery]
                    if recovery is not None
                    else None,
                    "recovery_delay_frames": delay if recovery is not None else None,
                }
            )
    return rows


def _summary(
    candidates: Sequence[Mapping[str, object]],
    evaluations: Sequence[Mapping[str, object]],
    pairs: Sequence[Mapping[str, object]],
    gaps: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_key = {row["record_key"]: row for row in candidates}
    baseline_visible = [
        row for row in evaluations if row["physics_baseline_status"] == "visible_fit"
    ]
    consistent = [
        row for row in baseline_visible if row["measurement_status"] == "branch_consistent"
    ]
    accepted = [row for row in evaluations if bool(row["measurement_valid"])]
    inconsistent = [
        row for row in accepted if row["measurement_status"] == "geometry_inconsistent"
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


def _reader_facing(summary: Mapping[str, Any]) -> dict[str, object]:
    visible = int(summary["physics_baseline_visible_opportunities"])
    consistent = int(summary["branch_consistent_measurements"])
    accepted = int(summary["accepted_measurements_all_frames"])
    inconsistent = int(summary["geometry_inconsistent_accepted_fits"])
    gaps = summary["post_gap_recovery"]
    return {
        "coverage": f"{consistent}/{visible} ({100.0 * consistent / visible:.1f}%)",
        "post_gap_recovery": (
            f"{int(gaps['recovered_gap_count'])}/{int(gaps['evaluated_gap_count'])} "
            f"({100.0 * float(gaps['rate']):.1f}%)"
        ),
        "geometry_inconsistency": (
            f"{inconsistent}/{accepted} ({100.0 * inconsistent / accepted:.1f}%)"
        ),
        "conditional_pair_residual_median": (
            f"{float(summary['absolute_pair_distance_residual_px']['median']):.3f} px"
        ),
    }


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise TracerValidationError(f"Expected frozen boolean, got {value!r}")


def _float_matches(actual: object, expected: str, atol: float, rtol: float) -> bool:
    if expected == "":
        return actual is None or actual == ""
    expected_value = float(expected)
    if math.isnan(expected_value):
        try:
            return math.isnan(float(actual))
        except (TypeError, ValueError):
            return False
    try:
        return math.isclose(float(actual), expected_value, abs_tol=atol, rel_tol=rtol)
    except (TypeError, ValueError):
        return False


def _validate_central_result(
    tracer_dir: Path,
    contract: Mapping[str, Any],
    candidates: Sequence[Mapping[str, object]],
    evaluations: Sequence[Mapping[str, object]],
    pairs: Sequence[Mapping[str, object]],
    gaps: Sequence[Mapping[str, object]],
    summary: Mapping[str, Any],
    workspace_root: Path,
) -> None:
    tolerance = contract["comparison_tolerances"]["continuous_values"]
    atol = float(tolerance["atol"])
    rtol = float(tolerance["rtol"])
    candidate_by_key = {row["record_key"]: row for row in candidates}
    evaluation_by_key = {row["record_key"]: row for row in evaluations}
    _, expected_rows = _read_csv(
        tracer_dir / contract["expected_files"]["central_records"]
    )
    if len(expected_rows) != len(candidates) or len(candidate_by_key) != len(candidates):
        raise TracerValidationError("Central candidate record keys changed")
    continuous = (
        "energy_ev",
        "native_x",
        "native_y",
        "native_mask_area",
        "native_mask_score",
        "predicted_x",
        "predicted_y",
        "center_x",
        "center_y",
        "proposal_offset_px",
        "r_squared",
        "amplitude",
        "volume",
        "sigma_x",
        "sigma_y",
        "branch_error_px",
        "radial_residual_px",
        "tangential_residual_px",
    )
    for expected in expected_rows:
        key = _record_key(int(expected["frame"]), expected["branch"])
        candidate = candidate_by_key.get(key)
        evaluation = evaluation_by_key.get(key)
        if candidate is None or evaluation is None:
            raise TracerValidationError(f"Central record key changed: {key}")
        discrete_pairs = {
            "method": expected["tracker"],
            "frame": int(expected["frame"]),
            "branch": expected["branch"],
            "physics_baseline_status": expected["physics_baseline_status"],
            "proposal_available": _parse_bool(expected["native_visible"]),
        }
        for field, expected_value in discrete_pairs.items():
            if candidate[field] != expected_value:
                raise TracerValidationError(
                    f"Central discrete value changed: {key}/{field}"
                )
        if bool(evaluation["accepted_fit"]) != _parse_bool(expected["fit_valid"]):
            raise TracerValidationError(f"Central fit status changed: {key}")
        if evaluation["legacy_fit_reason"] != expected["fit_reason"]:
            raise TracerValidationError(f"Central fit classification changed: {key}")
        if bool(evaluation["measurement_valid"]) != _parse_bool(
            expected["measurement_valid"]
        ):
            raise TracerValidationError(f"Central measurement status changed: {key}")
        if evaluation["measurement_status"] != expected["measurement_status"]:
            raise TracerValidationError(
                f"Central measurement classification changed: {key}"
            )
        for field in continuous:
            # The historical evaluation omitted post-fit geometry for Gaussian
            # rejections. The candidate layer intentionally retains it, so an
            # empty historical cell is not evidence that the value must be empty.
            if expected[field] == "":
                continue
            if not _float_matches(candidate.get(field), expected[field], atol, rtol):
                raise TracerValidationError(
                    f"Central continuous value changed: {key}/{field}"
                )

    _validate_expected_rows(
        tracer_dir / contract["expected_files"]["pair_records"],
        pairs,
        PAIR_FIELDS,
        atol,
        rtol,
    )
    _validate_expected_rows(
        tracer_dir / contract["expected_files"]["gap_records"],
        gaps,
        GAP_FIELDS,
        atol,
        rtol,
    )
    expected_summary = _read_json(
        tracer_dir / contract["expected_files"]["summary"]
    )
    _compare_nested(summary, expected_summary, "summary", atol, rtol)
    _validate_published_case(workspace_root, summary, atol, rtol)


def _validate_expected_rows(
    expected_path: Path,
    actual_rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
    atol: float,
    rtol: float,
) -> None:
    expected_fields, expected_rows = _read_csv(expected_path)
    if expected_fields != list(fields) or len(expected_rows) != len(actual_rows):
        raise TracerValidationError(f"Frozen expected rows changed: {expected_path.name}")
    for row_index, (expected, actual) in enumerate(zip(expected_rows, actual_rows)):
        for field in fields:
            expected_value = expected[field]
            actual_value = actual[field]
            if expected_value in {"True", "False"}:
                matches = bool(actual_value) == _parse_bool(expected_value)
            else:
                try:
                    float(expected_value)
                except ValueError:
                    matches = str(actual_value) == expected_value
                else:
                    matches = _float_matches(actual_value, expected_value, atol, rtol)
            if not matches:
                raise TracerValidationError(
                    f"Frozen {expected_path.name} changed: row {row_index}/{field}"
                )


def _compare_nested(
    actual: object,
    expected: object,
    location: str,
    atol: float,
    rtol: float,
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise TracerValidationError(f"Central {location} keys changed")
        for key, expected_value in expected.items():
            _compare_nested(actual[key], expected_value, f"{location}.{key}", atol, rtol)
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise TracerValidationError(f"Central {location} changed")
        return
    if isinstance(expected, (int, float)):
        if isinstance(expected, int):
            matches = actual == expected
        else:
            matches = math.isclose(float(actual), expected, abs_tol=atol, rel_tol=rtol)
        if not matches:
            raise TracerValidationError(f"Central {location} changed")
        return
    raise TracerValidationError(f"Unsupported frozen value at {location}")


def _validate_published_case(
    workspace_root: Path,
    summary: Mapping[str, Any],
    atol: float,
    rtol: float,
) -> None:
    _, rows = _read_csv(workspace_root / PUBLISHED_CASE_RESULTS_PATH)
    expected = next(
        row
        for row in rows
        if row["series"] == "26a21"
        and row["pair"] == "pair1"
        and row["method"] == "locotrack"
    )
    exact = {
        "visible_opportunities": "physics_baseline_visible_opportunities",
        "branch_consistent_measurements": "branch_consistent_measurements",
        "accepted_measurements_all_frames": "accepted_measurements_all_frames",
        "geometry_inconsistent_accepted_fits": "geometry_inconsistent_accepted_fits",
        "jointly_consistent_pair_frames": "jointly_consistent_pair_frames",
    }
    for source_field, summary_field in exact.items():
        if int(expected[source_field]) != int(summary[summary_field]):
            raise TracerValidationError(
                f"Published case result changed: {source_field}"
            )
    median = summary["absolute_pair_distance_residual_px"]["median"]
    if not math.isclose(
        float(median),
        float(expected["case_median_abs_distance_residual_px"]),
        abs_tol=atol,
        rel_tol=rtol,
    ):
        raise TracerValidationError("Published case residual changed")


def _output_manifest(output_dir: Path) -> dict[str, object]:
    names = (
        "candidate_records.csv",
        "evaluation_records.csv",
        "pair_records.csv",
        "gap_records.csv",
        "summary.json",
        "provenance.json",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "tracer_id": TRACER_ID,
        "files_sha256": {name: _sha256(output_dir / name) for name in names},
    }


def _candidate_from_csv(row: Mapping[str, str]) -> dict[str, object]:
    bool_fields = {
        "proposal_available",
        "roi_valid",
        "finite_candidate",
    }
    int_fields = {
        "frame",
        "native_mask_component_count",
        "roi_side_px",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
    }
    numeric_fields = set(CANDIDATE_FIELDS) - bool_fields - int_fields - {
        "record_key",
        "case_id",
        "series_id",
        "pair_id",
        "method",
        "branch",
        "physics_baseline_status",
        "opportunity_key",
        "image_artifact_id",
        "image_source_path",
        "image_sha256",
        "proposal_source_path",
        "proposal_sha256",
        "solver_status",
        "missing_state",
    }
    parsed: dict[str, object] = {}
    for field in CANDIDATE_FIELDS:
        value = row[field]
        if value == "":
            parsed[field] = None
        elif field in bool_fields:
            parsed[field] = _parse_bool(value)
        elif field in int_fields:
            parsed[field] = int(value)
        elif field in numeric_fields:
            parsed[field] = float(value)
        else:
            parsed[field] = value
    return parsed


def _evaluation_from_csv(row: Mapping[str, str]) -> dict[str, object]:
    bool_fields = {"accepted_fit", "measurement_valid"}
    numeric_fields = {
        "energy_ev",
        "proposal_to_fit_margin_px",
        "r_squared_margin",
        "amplitude_margin",
        "sigma_x_lower_margin_px",
        "sigma_x_upper_margin_px",
        "sigma_y_lower_margin_px",
        "sigma_y_upper_margin_px",
        "branch_margin_px",
    }
    parsed: dict[str, object] = {}
    for field in EVALUATION_FIELDS:
        value = row[field]
        if value == "":
            parsed[field] = None
        elif field in bool_fields:
            parsed[field] = _parse_bool(value)
        elif field == "frame":
            parsed[field] = int(value)
        elif field in numeric_fields:
            parsed[field] = float(value)
        else:
            parsed[field] = value
    return parsed


def _pair_from_csv(row: Mapping[str, str]) -> dict[str, object]:
    return {
        field: (
            row[field]
            if field == "tracker"
            else int(row[field])
            if field == "frame"
            else float(row[field])
        )
        for field in PAIR_FIELDS
    }


def _gap_from_csv(row: Mapping[str, str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for field in GAP_FIELDS:
        value = row[field]
        if value == "":
            parsed[field] = None
        elif field in {"tracker", "branch", "direction"}:
            parsed[field] = value
        elif field == "recovered":
            parsed[field] = _parse_bool(value)
        elif field in {"gap_frame_count", "recovery_delay_frames"}:
            parsed[field] = int(value)
        else:
            parsed[field] = float(value)
    return parsed


def _validate_output_manifest(output_dir: Path) -> None:
    manifest = _read_json(output_dir / "output_manifest.json")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("tracer_id") != TRACER_ID
    ):
        raise TracerValidationError("Tracer output manifest identity changed")
    expected_hashes = manifest.get("files_sha256")
    if not isinstance(expected_hashes, dict):
        raise TracerValidationError("Tracer output manifest has no file hash map")
    expected_names = {
        "candidate_records.csv",
        "evaluation_records.csv",
        "pair_records.csv",
        "gap_records.csv",
        "summary.json",
        "provenance.json",
    }
    if set(expected_hashes) != expected_names:
        raise TracerValidationError("Tracer output manifest inventory changed")
    actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_names != expected_names | {
        "output_manifest.json",
        "validation_report.json",
    }:
        raise TracerValidationError("Tracer output file inventory changed")
    for name, expected_hash in expected_hashes.items():
        if _sha256(output_dir / name) != expected_hash:
            raise TracerValidationError(f"Tracer output hash mismatch: {name}")


def validate_tracer_output(
    workspace_root: Path,
    tracer_dir: Path,
    output_dir: Path,
) -> TracerReport:
    """Validate existing tracer artifacts without rewriting any file."""

    workspace_root = workspace_root.resolve()
    tracer_dir = tracer_dir.resolve()
    output_dir = output_dir.resolve()
    validate_package(workspace_root, workspace_root / BASELINE_PACKAGE)
    contract = _validate_package_files(tracer_dir)
    runtime = _validate_runtime(contract)
    _validate_input_hashes(workspace_root, tracer_dir, contract)
    _validate_output_manifest(output_dir)

    candidate_fields, candidate_rows = _read_csv(
        output_dir / "candidate_records.csv"
    )
    evaluation_fields, evaluation_rows = _read_csv(
        output_dir / "evaluation_records.csv"
    )
    pair_fields, pair_rows = _read_csv(output_dir / "pair_records.csv")
    gap_fields, gap_rows = _read_csv(output_dir / "gap_records.csv")
    if candidate_fields != list(CANDIDATE_FIELDS):
        raise TracerValidationError("Tracer candidate output schema changed")
    if evaluation_fields != list(EVALUATION_FIELDS):
        raise TracerValidationError("Tracer evaluation output schema changed")
    if pair_fields != list(PAIR_FIELDS):
        raise TracerValidationError("Tracer pair output schema changed")
    if gap_fields != list(GAP_FIELDS):
        raise TracerValidationError("Tracer gap output schema changed")
    candidates = [_candidate_from_csv(row) for row in candidate_rows]
    evaluations = [_evaluation_from_csv(row) for row in evaluation_rows]
    pairs = [_pair_from_csv(row) for row in pair_rows]
    gaps = [_gap_from_csv(row) for row in gap_rows]
    summary_document = _read_json(output_dir / "summary.json")
    if set(summary_document) != {"central_result", "reader_facing"}:
        raise TracerValidationError("Tracer summary schema changed")
    summary = summary_document["central_result"]
    if not isinstance(summary, dict):
        raise TracerValidationError("Tracer central summary is not an object")

    _validate_central_result(
        tracer_dir,
        contract,
        candidates,
        evaluations,
        pairs,
        gaps,
        summary,
        workspace_root,
    )
    if summary_document["reader_facing"] != _reader_facing(summary):
        raise TracerValidationError("Tracer reader-facing rounding changed")
    provenance = _read_json(output_dir / "provenance.json")
    if (
        provenance.get("tracer_id") != TRACER_ID
        or provenance.get("execution_device") != "cpu"
        or provenance.get("runtime") != runtime
        or provenance.get("inputs") != contract["frozen_inputs"]
        or provenance.get("estimator_compatibility")
        != contract["estimator_compatibility"]
    ):
        raise TracerValidationError("Tracer provenance changed")
    runner_source = _relative_file(
        workspace_root, str(provenance.get("runner_source", ""))
    )
    if _source_sha256(runner_source) != provenance.get("runner_sha256"):
        raise TracerValidationError("Tracer runner source hash changed")
    validation_report = _read_json(output_dir / "validation_report.json")
    if (
        validation_report.get("tracer_id") != TRACER_ID
        or validation_report.get("fail_closed") is not True
        or validation_report.get("output_manifest_sha256")
        != _sha256(output_dir / "output_manifest.json")
    ):
        raise TracerValidationError("Tracer validation report changed")
    report = TracerReport(
        candidate_count=len(candidates),
        evaluation_count=len(evaluations),
        branch_consistent_count=int(summary["branch_consistent_measurements"]),
        evaluated_gap_count=int(summary["post_gap_recovery"]["evaluated_gap_count"]),
        recovered_gap_count=int(summary["post_gap_recovery"]["recovered_gap_count"]),
        matches_frozen_central_record=True,
    )
    report_values = asdict(report)
    if any(validation_report.get(key) != value for key, value in report_values.items()):
        raise TracerValidationError("Tracer validation report counts changed")
    return report


def run_tracer(
    workspace_root: Path,
    tracer_dir: Path,
    output_dir: Path,
) -> TracerReport:
    """Reconstruct and validate the frozen tracer without tracker inference."""

    workspace_root = workspace_root.resolve()
    tracer_dir = tracer_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace tracer output: {output_dir}")
    validate_package(workspace_root, workspace_root / BASELINE_PACKAGE)
    contract = _validate_package_files(tracer_dir)
    runtime = _validate_runtime(contract)
    _validate_input_hashes(workspace_root, tracer_dir, contract)
    candidates = _candidate_rows(workspace_root, tracer_dir, contract)
    evaluations = evaluate_candidates(candidates, _central_contract(workspace_root))
    pairs = _pair_rows(candidates, evaluations, workspace_root)
    gaps = _gap_rows(evaluations, workspace_root)
    summary = _summary(candidates, evaluations, pairs, gaps)
    _validate_central_result(
        tracer_dir,
        contract,
        candidates,
        evaluations,
        pairs,
        gaps,
        summary,
        workspace_root,
    )

    output_dir.mkdir(parents=True)
    _write_csv(output_dir / "candidate_records.csv", CANDIDATE_FIELDS, candidates)
    _write_csv(output_dir / "evaluation_records.csv", EVALUATION_FIELDS, evaluations)
    _write_csv(output_dir / "pair_records.csv", PAIR_FIELDS, pairs)
    _write_csv(output_dir / "gap_records.csv", GAP_FIELDS, gaps)
    _write_json(
        output_dir / "summary.json",
        {"central_result": summary, "reader_facing": _reader_facing(summary)},
    )
    _write_json(
        output_dir / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "tracer_id": TRACER_ID,
            "execution_device": "cpu",
            "runtime": runtime,
            "inputs": contract["frozen_inputs"],
            "estimator_compatibility": contract["estimator_compatibility"],
            "runner_source": Path(__file__).relative_to(workspace_root).as_posix(),
            "runner_sha256": _source_sha256(Path(__file__)),
        },
    )
    manifest = _output_manifest(output_dir)
    _write_json(output_dir / "output_manifest.json", manifest)
    report = TracerReport(
        candidate_count=len(candidates),
        evaluation_count=len(evaluations),
        branch_consistent_count=int(summary["branch_consistent_measurements"]),
        evaluated_gap_count=int(summary["post_gap_recovery"]["evaluated_gap_count"]),
        recovered_gap_count=int(summary["post_gap_recovery"]["recovered_gap_count"]),
        matches_frozen_central_record=True,
    )
    _write_json(
        output_dir / "validation_report.json",
        {
            **asdict(report),
            "checks": {
                "candidate_schema_and_sources": "passed",
                "continuous_value_tolerances": "passed",
                "evaluation_classifications": "passed",
                "frozen_input_hashes": "passed",
                "reader_facing_rounding": "passed",
                "record_keys_and_counts": "passed",
            },
            "fail_closed": True,
            "output_manifest_sha256": _sha256(output_dir / "output_manifest.json"),
            "schema_version": SCHEMA_VERSION,
            "tracer_id": TRACER_ID,
        },
    )
    return validate_tracer_output(workspace_root, tracer_dir, output_dir)


__all__ = [
    "CANDIDATE_FIELDS",
    "EVALUATION_FIELDS",
    "TracerReport",
    "TracerValidationError",
    "evaluate_candidates",
    "run_tracer",
    "validate_tracer_output",
]
