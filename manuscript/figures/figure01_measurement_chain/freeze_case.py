#!/usr/bin/env python3
"""Freeze and validate the representative source case for Figure 1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Sequence


FIGURE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = FIGURE_ROOT.parents[2]
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
ARTIFACT_ROOT = (
    REPOSITORY_ROOT
    / ".scratch/trackleed-physics-method-paper/artifacts/"
    "cu111-tracker-evaluation-subset"
)
RUN_ROOT = ARTIFACT_ROOT / "runs/five-methods-v1"
CPU_RUN_ROOT = ARTIFACT_ROOT / "runs/cpu-baselines-v1"

METHOD_ORDER = (
    "locotrack_base_offline",
    "ostrack_vitb_256_mae_ce",
    "dam4sam_sam21pp_l",
    "lucas_kanade_skip_anchor",
    "easyleed_kalman",
)
METHOD_METADATA = {
    "locotrack_base_offline": {
        "display_method": "LocoTrack",
        "comparison_category": "point tracking family",
        "native_representation": "point",
        "full_native_geometry_archived": True,
        "archive_status": "native point retained",
    },
    "ostrack_vitb_256_mae_ce": {
        "display_method": "OSTrack",
        "comparison_category": "bounding-box tracking family",
        "native_representation": "bounding box",
        "full_native_geometry_archived": False,
        "archive_status": "bbox reduced to center; width and height not retained",
    },
    "dam4sam_sam21pp_l": {
        "display_method": "DAM4SAM",
        "comparison_category": "mask tracking family",
        "native_representation": "mask",
        "full_native_geometry_archived": False,
        "archive_status": "mask reduced to centroid; mask pixels not retained",
    },
    "lucas_kanade_skip_anchor": {
        "display_method": "Lucas-Kanade",
        "comparison_category": "classical image-motion baseline",
        "native_representation": "point",
        "full_native_geometry_archived": True,
        "archive_status": "accepted optical-flow point retained",
    },
    "easyleed_kalman": {
        "display_method": "EasyLEED",
        "comparison_category": "LEED-domain tracking baseline",
        "native_representation": "position",
        "full_native_geometry_archived": True,
        "archive_status": "native Kalman position retained",
    },
}

SELECTION_RULE = (
    "Use the predeclared diagnostic probe whose selection_reason is "
    "'evaluation_range_end', then use the lowest canonical core spot ID. "
    "Do not inspect tracker outputs when choosing the case."
)
CORE_SPOT_IDS = (1, 3, 5, 6)
CODE_FILES = ("freeze_case.py", "test_freeze_case.py")
SOURCE_DATA_FILES = (
    "representative_initialization.json",
    "representative_method_outputs.csv",
    "representative_spot_crop.json",
    "representative_spot_crop.png",
    "source_provenance.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _csv_value(row.get(field)) for field in fields}
            )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".17g") if math.isfinite(value) else ""
    return value


def _as_int(value: object) -> int:
    return int(float(str(value)))


def _as_float(value: object) -> float:
    return float(str(value))


def _as_optional_float(value: object) -> float | None:
    text = str(value)
    return float(text) if text else None


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _portable_path(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def _resolve_native_payload(
    method: str,
    stored_path: str,
    spot_id: int,
) -> Path:
    candidate = (RUN_ROOT / stored_path).resolve()
    if candidate.is_file():
        return candidate
    if method in {"lucas_kanade_skip_anchor", "easyleed_kalman"}:
        candidate = (
            CPU_RUN_ROOT
            / "native"
            / method
            / f"{method}_spot_{spot_id:02d}.npz"
        )
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing native payload for {method}: {candidate}")
    return candidate


def _single_row(
    rows: Sequence[dict[str, str]],
    *,
    method: str,
    frame_order: int,
    spot_id: int,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["method"] == method
        and _as_int(row["frame_order"]) == frame_order
        and _as_int(row["spot_id"]) == spot_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for {method}, frame {frame_order}, spot {spot_id}"
        )
    return matches[0]


def _method_output(
    method: str,
    native: dict[str, str],
    measurement: dict[str, str],
    *,
    spot_id: int,
) -> dict[str, object]:
    if _as_int(native["frame_id"]) != _as_int(measurement["frame_id"]):
        raise ValueError(f"Frame mismatch for {method}")
    if _as_bool(native["proposal_available"]) != _as_bool(
        measurement["proposal_available"]
    ):
        raise ValueError(f"Proposal status mismatch for {method}")
    payload = _resolve_native_payload(method, native["native_payload_path"], spot_id)
    metadata = METHOD_METADATA[method]
    return {
        "method": method,
        "display_method": metadata["display_method"],
        "comparison_category": metadata["comparison_category"],
        "native_representation": metadata["native_representation"],
        "archive_status": metadata["archive_status"],
        "full_native_geometry_archived": metadata[
            "full_native_geometry_archived"
        ],
        "native_payload_path": _portable_path(payload),
        "native_payload_sha256": sha256_file(payload),
        "proposal_reduction_rule": native["proposal_reduction_rule"],
        "proposal_available": _as_bool(native["proposal_available"]),
        "proposal_x_original_px": _as_float(
            native["measurement_center_x_original_px"]
        ),
        "proposal_y_original_px": _as_float(
            native["measurement_center_y_original_px"]
        ),
        "native_visibility_signal_name": native[
            "native_visibility_signal_name"
        ],
        "native_visibility_signal": _as_optional_float(
            native["native_visibility_signal"]
        ),
        "roi_x0_original_px": _as_int(measurement["roi_x0_original_px"]),
        "roi_y0_original_px": _as_int(measurement["roi_y0_original_px"]),
        "roi_size_px": _as_int(measurement["roi_size_px"]),
        "gaussian_x_original_px": _as_float(measurement["global_center_x"]),
        "gaussian_y_original_px": _as_float(measurement["global_center_y"]),
        "proposal_offset_px": _as_float(measurement["proposal_offset_px"]),
        "fit_valid": _as_bool(measurement["fit_valid"]),
        "amplitude": _as_float(measurement["amplitude"]),
        "sigma_x_px": _as_float(measurement["sigma_x"]),
        "sigma_y_px": _as_float(measurement["sigma_y"]),
        "r_squared": _as_float(measurement["r_squared"]),
        "rejection_reason": measurement["rejection_reason"],
    }


def _load_comparison_contract(spot_id: int) -> dict[str, object]:
    artifact_contract = json.loads(
        (ARTIFACT_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    if artifact_contract.get("status") != "frozen_ready_for_runner":
        raise ValueError("Cu(111) comparison contract is not frozen")
    if tuple(artifact_contract["main_methods"]) != METHOD_ORDER:
        raise ValueError("Cu(111) comparison method order changed")
    core_spot_ids = tuple(
        _as_int(value) for value in artifact_contract["cu_scope"]["core_spot_ids"]
    )
    if core_spot_ids != CORE_SPOT_IDS or spot_id != min(core_spot_ids):
        raise ValueError("Canonical Cu(111) spot contract changed")

    seed_rows = [
        row
        for row in read_csv(ARTIFACT_ROOT / "seed_prompts.csv")
        if _as_int(row["spot_id"]) == spot_id
    ]
    if len(seed_rows) != 1:
        raise ValueError(f"Expected one seed row for spot {spot_id}")
    seed = seed_rows[0]
    initialization = artifact_contract["initialization"]
    series = artifact_contract["series"]
    bbox_x0 = _as_float(seed["bbox_x0_original_px"])
    bbox_y0 = _as_float(seed["bbox_y0_original_px"])
    bbox_x1 = _as_float(seed["bbox_x1_original_px"])
    bbox_y1 = _as_float(seed["bbox_y1_original_px"])
    bbox_width = bbox_x1 - bbox_x0
    bbox_height = bbox_y1 - bbox_y0
    expected_bbox_size = _as_float(initialization["prompt_bbox_original_px"])
    if not math.isclose(bbox_width, expected_bbox_size) or not math.isclose(
        bbox_height,
        expected_bbox_size,
    ):
        raise ValueError("Shared initialization bbox is not 48 x 48 pixels")
    if _as_int(seed["seed_frame_order"]) != _as_int(
        initialization["seed_frame_order"]
    ):
        raise ValueError("Seed-frame contract mismatch")

    return {
        "same_sequence": True,
        "same_selected_spot": True,
        "same_initialization": True,
        "series_id": series["series_id"],
        "frame_count": _as_int(series["frame_count"]),
        "sequence_order": artifact_contract["design"]["inference_sequence"],
        "methods": list(METHOD_ORDER),
        "selected_spot_id": spot_id,
        "identity_label": seed["identity_label"],
        "pair": seed["pair"],
        "seed_frame_order": _as_int(seed["seed_frame_order"]),
        "seed_frame_id": _as_int(seed["seed_frame_id"]),
        "seed_source": seed["seed_source"],
        "point_methods_receive": initialization["point_methods_receive"],
        "box_and_mask_methods_receive": initialization[
            "box_and_mask_methods_receive"
        ],
        "seed_center_x_original_px": _as_float(
            seed["center_x_original_px"]
        ),
        "seed_center_y_original_px": _as_float(
            seed["center_y_original_px"]
        ),
        "shared_bbox_x0_original_px": bbox_x0,
        "shared_bbox_y0_original_px": bbox_y0,
        "shared_bbox_x1_original_px": bbox_x1,
        "shared_bbox_y1_original_px": bbox_y1,
        "shared_bbox_width_original_px": expected_bbox_size,
        "shared_bbox_height_original_px": expected_bbox_size,
        "seed_is_ground_truth": not _as_bool(seed["not_ground_truth"]),
        "manual_correction_or_reinitialization": "forbidden",
    }


def select_representative_case() -> dict[str, object]:
    frame_rows = read_csv(ARTIFACT_ROOT / "diagnostic_probe_frames.csv")
    selected_frames = [
        row
        for row in frame_rows
        if row["selection_reason"] == "evaluation_range_end"
        and _as_bool(row["diagnostic_probe"])
    ]
    if len(selected_frames) != 1:
        raise ValueError("Expected one predeclared evaluation-range-end probe")
    selected = selected_frames[0]
    frame_order = _as_int(selected["frame_order"])
    spot_id = min(CORE_SPOT_IDS)
    comparison_contract = _load_comparison_contract(spot_id)

    native_rows = read_csv(RUN_ROOT / "native_proposals.csv")
    measurement_rows = read_csv(RUN_ROOT / "gaussian_measurements.csv")
    outputs = []
    for method in METHOD_ORDER:
        native = _single_row(
            native_rows,
            method=method,
            frame_order=frame_order,
            spot_id=spot_id,
        )
        measurement = _single_row(
            measurement_rows,
            method=method,
            frame_order=frame_order,
            spot_id=spot_id,
        )
        outputs.append(
            _method_output(method, native, measurement, spot_id=spot_id)
        )
    if not all(row["proposal_available"] for row in outputs):
        raise ValueError("Representative case has an unavailable proposal")
    if not all(row["fit_valid"] for row in outputs):
        raise ValueError("Representative case has a rejected common fit")

    raw_path = REPOSITORY_ROOT / selected["raw_relative_path"]
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing representative raw image: {raw_path}")
    if sha256_file(raw_path) != selected["raw_sha256"]:
        raise ValueError("Representative raw-image hash mismatch")

    return {
        "selection_rule": SELECTION_RULE,
        "frame": {
            "frame_order": frame_order,
            "frame_id": _as_int(selected["frame_id"]),
            "temperature_c": _as_float(selected["temperature_c"]),
            "energy_ev": _as_float(selected["energy_ev"]),
            "raw_relative_path": selected["raw_relative_path"],
            "raw_sha256": selected["raw_sha256"],
            "selection_group": selected["selection_group"],
            "selection_reason": selected["selection_reason"],
        },
        "spot_id": spot_id,
        "comparison_contract": comparison_contract,
        "method_outputs": outputs,
    }


def _max_pairwise_distance(
    rows: Sequence[dict[str, object]],
    x_field: str,
    y_field: str,
) -> float:
    points = [
        (_as_float(row[x_field]), _as_float(row[y_field])) for row in rows
    ]
    return max(
        math.hypot(first[0] - second[0], first[1] - second[1])
        for first, second in combinations(points, 2)
    )


def _write_crop(
    case: dict[str, object],
    source_data_directory: Path,
) -> dict[str, object]:
    from PIL import Image

    outputs = case["method_outputs"]
    center_x = statistics.median(
        _as_float(row["gaussian_x_original_px"]) for row in outputs
    )
    center_y = statistics.median(
        _as_float(row["gaussian_y_original_px"]) for row in outputs
    )
    crop_size = 128
    x0 = int(round(center_x)) - crop_size // 2
    y0 = int(round(center_y)) - crop_size // 2
    raw_path = REPOSITORY_ROOT / case["frame"]["raw_relative_path"]
    with Image.open(raw_path) as image:
        if image.size != (1000, 1000):
            raise ValueError(f"Unexpected representative image size: {image.size}")
        crop = image.crop((x0, y0, x0 + crop_size, y0 + crop_size))
        if crop.mode != "L":
            crop = crop.convert("L")
        crop_path = source_data_directory / "representative_spot_crop.png"
        crop.save(
            crop_path,
            format="PNG",
            optimize=False,
            compress_level=9,
        )

    metadata = {
        "raw_relative_path": case["frame"]["raw_relative_path"],
        "raw_sha256": case["frame"]["raw_sha256"],
        "crop_x0_original_px": x0,
        "crop_y0_original_px": y0,
        "crop_width_px": crop_size,
        "crop_height_px": crop_size,
        "reference_center_x_original_px": center_x,
        "reference_center_y_original_px": center_y,
        "intensity_encoding": (
            "Lossless uint8 crop from the raw TIFF; no background subtraction "
            "or contrast transformation."
        ),
    }
    write_json(
        source_data_directory / "representative_spot_crop.json",
        metadata,
    )
    return metadata


def _source_provenance(case: dict[str, object]) -> dict[str, object]:
    paths = {
        "contract_manifest": ARTIFACT_ROOT / "manifest.json",
        "diagnostic_probe_frames": ARTIFACT_ROOT / "diagnostic_probe_frames.csv",
        "seed_prompts": ARTIFACT_ROOT / "seed_prompts.csv",
        "five_method_run_manifest": RUN_ROOT / "run_manifest.json",
        "native_proposals": RUN_ROOT / "native_proposals.csv",
        "gaussian_measurements": RUN_ROOT / "gaussian_measurements.csv",
        "representative_raw_image": (
            REPOSITORY_ROOT / case["frame"]["raw_relative_path"]
        ),
    }
    for row in case["method_outputs"]:
        method = str(row["method"])
        paths[f"native_payload_{method}"] = (
            REPOSITORY_ROOT / str(row["native_payload_path"])
        )
    return {
        "schema_version": "1.0",
        "figure": "Figure 1",
        "sources": {
            name: {
                "path": _portable_path(path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }


def _source_data_hashes(figure_root: Path) -> dict[str, str]:
    source_data_directory = figure_root / "source_data"
    return {
        f"source_data/{filename}": sha256_file(
            source_data_directory / filename
        )
        for filename in SOURCE_DATA_FILES
    }


def build(output_root: Path = FIGURE_ROOT) -> None:
    case = select_representative_case()
    source_data_directory = output_root / "source_data"
    source_data_directory.mkdir(parents=True, exist_ok=True)
    write_json(
        source_data_directory / "representative_initialization.json",
        case["comparison_contract"],
    )
    write_csv(
        source_data_directory / "representative_method_outputs.csv",
        case["method_outputs"],
    )
    crop_metadata = _write_crop(case, source_data_directory)
    provenance = _source_provenance(case)
    write_json(
        source_data_directory / "source_provenance.json",
        provenance,
    )

    outputs = case["method_outputs"]
    contract = {
        "schema_version": "1.0-frozen",
        "status": "representative_case_frozen",
        "figure": "Figure 1",
        "comparison_contract": case["comparison_contract"],
        "selection": {
            "rule": SELECTION_RULE,
            "uses_tracker_outputs": False,
            "frame": case["frame"],
            "spot_id": case["spot_id"],
            "spot_rule": "lowest canonical core spot ID",
        },
        "completeness": {
            "methods": list(METHOD_ORDER),
            "native_proposals": "5/5",
            "valid_common_gaussian_fits": "5/5",
            "manual_annotations": 0,
        },
        "descriptive_geometry": {
            "maximum_pairwise_native_proposal_separation_px": (
                _max_pairwise_distance(
                    outputs,
                    "proposal_x_original_px",
                    "proposal_y_original_px",
                )
            ),
            "maximum_pairwise_gaussian_center_separation_px": (
                _max_pairwise_distance(
                    outputs,
                    "gaussian_x_original_px",
                    "gaussian_y_original_px",
                )
            ),
        },
        "crop": crop_metadata,
        "native_geometry_archive_gap": {
            "status": "explicit",
            "affected_methods": [
                "ostrack_vitb_256_mae_ce",
                "dam4sam_sam21pp_l",
            ],
            "available": (
                "Reduced bbox center for OSTrack and reduced mask centroid for "
                "DAM4SAM."
            ),
            "not_available": (
                "The tracked bbox width/height and the native mask pixels for "
                "this frame were not retained by the publication runner."
            ),
            "rendering_constraint": (
                "Do not draw an empirical OSTrack bbox or DAM4SAM mask from "
                "these archived files. Either perform a targeted native-geometry "
                "export or label those glyphs as schematic."
            ),
        },
        "intended_use": (
            "Reader-facing explanation of heterogeneous proposal "
            "representations and their reduction to one common Gaussian "
            "measurement contract."
        ),
        "claim_limit": (
            "This case is explanatory and was selected without tracker outputs; "
            "it is not a qualitative performance ranking."
        ),
    }
    write_json(output_root / "contract.json", contract)

    manifest = {
        "schema_version": "1.0",
        "status": "representative_case_frozen",
        "figure": "Figure 1",
        "code_sha256": {
            filename: sha256_file(FIGURE_ROOT / filename)
            for filename in CODE_FILES
        },
        "source_artifacts": provenance["sources"],
        "source_data_sha256": _source_data_hashes(output_root),
        "contract_sha256": sha256_file(output_root / "contract.json"),
    }
    write_json(output_root / "manifest.json", manifest)


def _validate_offline_integrity() -> tuple[dict[str, object], dict[str, object]]:
    manifest_path = FIGURE_ROOT / "manifest.json"
    contract_path = FIGURE_ROOT / "contract.json"
    if not manifest_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError("Frozen Figure 1 case is incomplete; run build")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "representative_case_frozen":
        raise ValueError("Figure 1 case manifest is not frozen")
    if contract.get("status") != "representative_case_frozen":
        raise ValueError("Figure 1 case contract is not frozen")
    expected_code = {
        filename: sha256_file(FIGURE_ROOT / filename) for filename in CODE_FILES
    }
    if manifest.get("code_sha256") != expected_code:
        raise ValueError("Figure 1 case code hash mismatch")
    if manifest.get("contract_sha256") != sha256_file(contract_path):
        raise ValueError("Figure 1 case contract hash mismatch")
    expected_source_data_paths = {
        f"source_data/{filename}" for filename in SOURCE_DATA_FILES
    }
    if set(manifest.get("source_data_sha256", {})) != expected_source_data_paths:
        raise ValueError("Figure 1 source-data file set changed")
    for relative_path, expected_hash in manifest["source_data_sha256"].items():
        path = FIGURE_ROOT / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Figure 1 source-data hash mismatch: {relative_path}")

    rows = read_csv(
        SOURCE_DATA_DIRECTORY / "representative_method_outputs.csv"
    )
    if [row["method"] for row in rows] != list(METHOD_ORDER):
        raise ValueError("Figure 1 method order changed")
    if not all(_as_bool(row["proposal_available"]) for row in rows):
        raise ValueError("Figure 1 representative proposal became unavailable")
    if not all(_as_bool(row["fit_valid"]) for row in rows):
        raise ValueError("Figure 1 representative fit became invalid")
    geometry_archived = {
        row["method"]: _as_bool(row["full_native_geometry_archived"])
        for row in rows
    }
    if geometry_archived != {
        method: bool(METHOD_METADATA[method]["full_native_geometry_archived"])
        for method in METHOD_ORDER
    }:
        raise ValueError("Figure 1 native-geometry availability changed")

    initialization = json.loads(
        (
            SOURCE_DATA_DIRECTORY / "representative_initialization.json"
        ).read_text(encoding="utf-8")
    )
    if initialization != contract.get("comparison_contract"):
        raise ValueError("Figure 1 shared-initialization contract mismatch")
    if (
        initialization.get("seed_frame_order") != 0
        or initialization.get("shared_bbox_width_original_px") != 48.0
        or initialization.get("shared_bbox_height_original_px") != 48.0
    ):
        raise ValueError("Figure 1 common initialization changed")
    return manifest, contract


def _validate_upstream_sources(manifest: dict[str, object]) -> None:
    for name, source in manifest["source_artifacts"].items():
        path = REPOSITORY_ROOT / source["path"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing upstream Figure 1 source {name}: {path}"
            )
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"Upstream Figure 1 source hash mismatch: {name}")


def _validate_deterministic_rebuild() -> None:
    relative_paths = [
        "contract.json",
        "manifest.json",
        *(f"source_data/{filename}" for filename in SOURCE_DATA_FILES),
    ]
    with tempfile.TemporaryDirectory(
        prefix="trackleed-figure01-case-"
    ) as temporary:
        rebuilt_root = Path(temporary) / "figure01_measurement_chain"
        build(rebuilt_root)
        for relative_path in relative_paths:
            committed = FIGURE_ROOT / relative_path
            rebuilt = rebuilt_root / relative_path
            if not rebuilt.is_file() or sha256_file(rebuilt) != sha256_file(
                committed
            ):
                raise ValueError(
                    f"Non-deterministic Figure 1 case rebuild: {relative_path}"
                )


def validate(*, check_upstream: bool = True) -> None:
    manifest, _ = _validate_offline_integrity()
    if check_upstream:
        _validate_upstream_sources(manifest)
        _validate_deterministic_rebuild()
        validation_scope = "upstream hashes and deterministic rebuild verified"
    else:
        validation_scope = "offline committed-file integrity only"
    print(
        "Figure 1 representative case validation passed: frame 700, spot 1, "
        "5/5 proposals and fits; OSTrack bbox and DAM4SAM mask geometry are "
        f"explicitly marked as not archived; {validation_scope}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate upstream hashes and a deterministic rebuild without "
            "rewriting committed files."
        ),
    )
    parser.add_argument(
        "--offline-integrity-only",
        action="store_true",
        help=(
            "Validate only committed code, contract, and source-data hashes "
            "when upstream artifacts are unavailable."
        ),
    )
    arguments = parser.parse_args()
    if arguments.validate_only and arguments.offline_integrity_only:
        parser.error(
            "--validate-only and --offline-integrity-only are mutually exclusive"
        )
    if arguments.offline_integrity_only:
        validate(check_upstream=False)
        return
    if arguments.validate_only:
        validate()
        return
    build()
    validate()


if __name__ == "__main__":
    main()
