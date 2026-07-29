#!/usr/bin/env python3
"""Re-export the native OSTrack bbox and DAM4SAM mask for Figure 1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any, Sequence

import numpy as np
from PIL import Image


FIGURE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = FIGURE_ROOT.parents[2]
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
ARTIFACT_ROOT = (
    REPOSITORY_ROOT
    / ".scratch/trackleed-physics-method-paper/artifacts/"
    "cu111-tracker-evaluation-subset"
)
GPU_NATIVE_ROOT = ARTIFACT_ROOT / "runs/gpu-native-v1"
UPSTREAM_RUNNER_PATH = ARTIFACT_ROOT / "run_gpu_cu.py"
OUTPUT_JSON_NAME = "representative_native_geometry.json"
MASK_PNG_NAME = "representative_dam4sam_mask_cache.png"

OSTRACK_METHOD = "ostrack_vitb_256_mae_ce"
DAM4SAM_METHOD = "dam4sam_sam21pp_l"
EXPORT_METHODS = (OSTRACK_METHOD, DAM4SAM_METHOD)
SPOT_ID = 1
SPOT_INDEX = 0
SEED_FRAME_ORDER = 0
TARGET_FRAME_ORDER = 700
FRAME_COUNT = 701
ORIGINAL_TO_CACHE = 0.256
CENTER_TOLERANCE_CACHE_PX = 5e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _float_list(values: Sequence[float] | np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def _center_error(
    actual_original_px: np.ndarray,
    expected_original_px: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(actual_original_px, dtype=np.float64)
            - np.asarray(expected_original_px, dtype=np.float64)
        )
    )


def _require_matching_center(
    actual_original_px: np.ndarray,
    expected_original_px: np.ndarray,
    *,
    geometry_name: str,
) -> float:
    error_original_px = _center_error(
        actual_original_px,
        expected_original_px,
    )
    tolerance_original_px = CENTER_TOLERANCE_CACHE_PX / ORIGINAL_TO_CACHE
    if error_original_px > tolerance_original_px:
        raise ValueError(
            f"{geometry_name} does not reproduce the archived reduced center: "
            f"error {error_original_px:.9f} original px exceeds "
            f"{tolerance_original_px:.9f}."
        )
    return error_original_px


def summarize_ostrack_bbox(
    bbox_xywh_cache_px: np.ndarray,
    *,
    expected_center_original_px: np.ndarray,
) -> dict[str, object]:
    bbox = np.asarray(bbox_xywh_cache_px, dtype=np.float64).reshape(-1)
    if (
        bbox.shape != (4,)
        or not np.isfinite(bbox).all()
        or bbox[2] <= 0.0
        or bbox[3] <= 0.0
    ):
        raise ValueError("OSTrack native bbox must be finite with positive size")
    bbox_xyxy = np.asarray(
        [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
        dtype=np.float64,
    )
    center_cache = np.asarray(
        [bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5],
        dtype=np.float64,
    )
    center_original = center_cache / ORIGINAL_TO_CACHE
    center_error = _require_matching_center(
        center_original,
        expected_center_original_px,
        geometry_name="OSTrack bbox",
    )
    return {
        "geometry_type": "bounding_box",
        "coordinate_convention": "xywh and half-open xyxy",
        "bbox_xywh_cache_px": _float_list(bbox),
        "bbox_xyxy_cache_px": _float_list(bbox_xyxy),
        "bbox_xywh_original_px": _float_list(bbox / ORIGINAL_TO_CACHE),
        "bbox_xyxy_original_px": _float_list(
            bbox_xyxy / ORIGINAL_TO_CACHE
        ),
        "reduction_rule": "x+w/2, y+h/2 from native bbox xywh",
        "reduced_center_cache_px": _float_list(center_cache),
        "reduced_center_original_px": _float_list(center_original),
        "archived_reduced_center_original_px": _float_list(
            expected_center_original_px
        ),
        "center_consistency_error_original_px": center_error,
    }


def summarize_dam4sam_mask(
    mask_cache: np.ndarray,
    *,
    expected_center_original_px: np.ndarray,
) -> dict[str, object]:
    mask = np.asarray(mask_cache, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"DAM4SAM mask must be two-dimensional: {mask.shape}")
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("DAM4SAM native mask is empty")
    bbox_xyxy = np.asarray(
        [
            float(np.min(xs)),
            float(np.min(ys)),
            float(np.max(xs) + 1),
            float(np.max(ys) + 1),
        ],
        dtype=np.float64,
    )
    bbox_xywh = np.asarray(
        [
            bbox_xyxy[0],
            bbox_xyxy[1],
            bbox_xyxy[2] - bbox_xyxy[0],
            bbox_xyxy[3] - bbox_xyxy[1],
        ],
        dtype=np.float64,
    )
    centroid_cache = np.asarray(
        [float(np.mean(xs)), float(np.mean(ys))],
        dtype=np.float64,
    )
    centroid_original = centroid_cache / ORIGINAL_TO_CACHE
    center_error = _require_matching_center(
        centroid_original,
        expected_center_original_px,
        geometry_name="DAM4SAM mask centroid",
    )
    return {
        "geometry_type": "binary_mask",
        "coordinate_convention": (
            "mask[y,x] on the 256x256 tracking cache; bbox is half-open xyxy"
        ),
        "mask_shape_cache_px": [int(mask.shape[0]), int(mask.shape[1])],
        "mask_area_cache_px": int(xs.size),
        "mask_bbox_xyxy_cache_px": _float_list(bbox_xyxy),
        "mask_bbox_xywh_cache_px": _float_list(bbox_xywh),
        "mask_bbox_xyxy_original_px": _float_list(
            bbox_xyxy / ORIGINAL_TO_CACHE
        ),
        "mask_bbox_xywh_original_px": _float_list(
            bbox_xywh / ORIGINAL_TO_CACHE
        ),
        "reduction_rule": (
            "arithmetic mean of x indices and y indices of all true mask pixels"
        ),
        "reduced_centroid_cache_px": _float_list(centroid_cache),
        "reduced_centroid_original_px": _float_list(centroid_original),
        "archived_reduced_center_original_px": _float_list(
            expected_center_original_px
        ),
        "center_consistency_error_original_px": center_error,
    }


def _load_upstream_runner() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "figure01_frozen_gpu_runner",
        UPSTREAM_RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load frozen GPU runner: {UPSTREAM_RUNNER_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _archived_payload(method: str) -> dict[str, np.ndarray]:
    path = (
        GPU_NATIVE_ROOT
        / method
        / f"{method}_spot_{SPOT_ID:02d}.npz"
    )
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _validate_replayed_trajectory(
    method: str,
    *,
    tracks_xy_cache: np.ndarray,
    proposal_available: np.ndarray,
) -> dict[str, object]:
    archived = _archived_payload(method)
    actual_tracks = np.asarray(tracks_xy_cache, dtype=np.float64)
    actual_available = np.asarray(proposal_available, dtype=bool)
    expected_tracks = np.stack(
        (
            np.asarray(archived["x_cache_px"], dtype=np.float64),
            np.asarray(archived["y_cache_px"], dtype=np.float64),
        ),
        axis=-1,
    )
    expected_available = np.asarray(
        archived["proposal_available"],
        dtype=bool,
    )
    if actual_tracks.shape != (FRAME_COUNT, 2):
        raise ValueError(
            f"{method} replay has unexpected trajectory shape {actual_tracks.shape}"
        )
    if not np.array_equal(actual_available, expected_available):
        mismatch_count = int(
            np.count_nonzero(actual_available != expected_available)
        )
        raise ValueError(
            f"{method} replay availability differs in {mismatch_count} frames"
        )
    distances = np.linalg.norm(
        actual_tracks[expected_available] - expected_tracks[expected_available],
        axis=-1,
    )
    maximum_error = float(np.max(distances)) if distances.size else 0.0
    if maximum_error > CENTER_TOLERANCE_CACHE_PX:
        raise ValueError(
            f"{method} replay trajectory differs from the archived reduction: "
            f"{maximum_error:.9f} cache px"
        )
    return {
        "archived_payload_path": (
            path := GPU_NATIVE_ROOT
            / method
            / f"{method}_spot_{SPOT_ID:02d}.npz"
        ).resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix(),
        "archived_payload_sha256": sha256_file(path),
        "availability_matches_all_frames": True,
        "reduced_center_matches_all_available_frames": True,
        "available_frame_count": int(np.count_nonzero(expected_available)),
        "maximum_reduced_center_error_cache_px": maximum_error,
        "tolerance_cache_px": CENTER_TOLERANCE_CACHE_PX,
    }


def _run_ostrack(
    runner: ModuleType,
    *,
    frames: np.ndarray,
    query_points: np.ndarray,
    query_boxes: np.ndarray,
    manifest: dict[str, Any],
) -> dict[str, object]:
    from trackleed.analysis.ostrack_backend import (
        OstrackRuntimeSettings,
        run_ostrack_offline,
    )

    method = OSTRACK_METHOD
    settings = manifest["method_configurations"][method]
    paths = runner.runtime_paths(method, manifest)
    result = run_ostrack_offline(
        frames,
        query_points,
        OstrackRuntimeSettings(
            checkpoint_path=paths["checkpoint"],
            repo_path=paths["repo"],
            python_executable=paths["python"],
            backend_mode="subprocess",
            model_variant=str(settings["model_variant"]),
            device=str(settings["device"]),
            representative_point_mode=str(
                settings["representative_point_mode"]
            ),
            inference_resolution=tuple(settings["inference_resolution"]),
            initial_bbox_padding_px=float(
                settings["initial_bbox_padding_px"]
            ),
            subprocess_timeout_sec=float(
                settings["subprocess_timeout_seconds"]
            ),
        ),
        query_boxes_xyxy=query_boxes,
    )
    available = (
        np.isfinite(result.tracked_bboxes_xywh[0]).all(axis=-1)
        & (result.tracked_bboxes_xywh[0, :, 2] > 0.0)
        & (result.tracked_bboxes_xywh[0, :, 3] > 0.0)
    )
    trajectory = _validate_replayed_trajectory(
        method,
        tracks_xy_cache=result.tracks_xy[0],
        proposal_available=available,
    )
    archived = _archived_payload(method)
    archived_center = np.asarray(
        [
            archived["x_original_px"][TARGET_FRAME_ORDER],
            archived["y_original_px"][TARGET_FRAME_ORDER],
        ],
        dtype=np.float64,
    )
    summary = summarize_ostrack_bbox(
        result.tracked_bboxes_xywh[0, TARGET_FRAME_ORDER],
        expected_center_original_px=archived_center,
    )
    summary.update(
        {
            "method": method,
            "display_method": "OSTrack",
            "native_bbox_score": float(
                result.bbox_scores[0, TARGET_FRAME_ORDER]
            ),
            "trajectory_replay": trajectory,
        }
    )
    return summary


def _run_dam4sam(
    runner: ModuleType,
    *,
    frames: np.ndarray,
    query_points: np.ndarray,
    query_boxes: np.ndarray,
    manifest: dict[str, Any],
    output_directory: Path,
) -> dict[str, object]:
    from trackleed.analysis.dam4sam_backend import (
        Dam4SamRuntimeSettings,
        run_dam4sam_offline,
    )

    method = DAM4SAM_METHOD
    settings = manifest["method_configurations"][method]
    paths = runner.runtime_paths(method, manifest)
    result = run_dam4sam_offline(
        frames,
        query_points,
        Dam4SamRuntimeSettings(
            checkpoint_path=paths["checkpoint"],
            repo_path=paths["repo"],
            python_executable=paths["python"],
            backend_mode="subprocess",
            tracker_name=str(settings["tracker_name"]),
            device=str(settings["device"]),
            representative_point_mode=str(
                settings["representative_point_mode"]
            ),
            inference_resolution=tuple(settings["inference_resolution"]),
            initial_bbox_padding_px=float(
                settings["initial_bbox_padding_px"]
            ),
            subprocess_timeout_sec=float(
                settings["subprocess_timeout_seconds"]
            ),
        ),
        query_boxes_xyxy=query_boxes,
    )
    available = (
        np.isfinite(result.tracks_xy[0]).all(axis=-1)
        & np.isfinite(result.mask_areas[0])
        & (result.mask_areas[0] > 0.0)
    )
    trajectory = _validate_replayed_trajectory(
        method,
        tracks_xy_cache=result.tracks_xy[0],
        proposal_available=available,
    )
    archived = _archived_payload(method)
    archived_center = np.asarray(
        [
            archived["x_original_px"][TARGET_FRAME_ORDER],
            archived["y_original_px"][TARGET_FRAME_ORDER],
        ],
        dtype=np.float64,
    )
    mask = np.asarray(
        result.masks[0, TARGET_FRAME_ORDER],
        dtype=bool,
    )
    summary = summarize_dam4sam_mask(
        mask,
        expected_center_original_px=archived_center,
    )
    if int(result.mask_areas[0, TARGET_FRAME_ORDER]) != int(
        summary["mask_area_cache_px"]
    ):
        raise ValueError("DAM4SAM reported mask area differs from mask pixels")
    mask_path = output_directory / MASK_PNG_NAME
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
        mask_path,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    summary.update(
        {
            "method": method,
            "display_method": "DAM4SAM",
            "mask_path": f"source_data/{MASK_PNG_NAME}",
            "mask_sha256": sha256_file(mask_path),
            "native_mask_score": (
                float(result.mask_scores[0, TARGET_FRAME_ORDER])
                if math.isfinite(
                    float(result.mask_scores[0, TARGET_FRAME_ORDER])
                )
                else None
            ),
            "trajectory_replay": trajectory,
        }
    )
    return summary


def _base_payload(manifest: dict[str, Any]) -> dict[str, object]:
    sources: dict[str, dict[str, str]] = {}
    for name, path in {
        "contract_manifest": ARTIFACT_ROOT / "manifest.json",
        "inference_frames": ARTIFACT_ROOT / "inference_frames.csv",
        "seed_prompts": ARTIFACT_ROOT / "seed_prompts.csv",
        "frozen_gpu_runner": UPSTREAM_RUNNER_PATH,
        "exporter": Path(__file__).resolve(),
    }.items():
        sources[name] = {
            "path": path.resolve()
            .relative_to(REPOSITORY_ROOT.resolve())
            .as_posix(),
            "sha256": sha256_file(path),
        }
    method_contracts = {}
    for method in EXPORT_METHODS:
        configuration = manifest["method_configurations"][method]
        resource_key = runner_resource_key(method)
        resource = manifest["method_resources"][resource_key]
        method_contracts[method] = {
            "method_configuration_sha256": manifest[
                "method_configuration_sha256"
            ][method],
            "adapter_source": configuration["adapter_source"],
            "adapter_source_sha256": configuration[
                "adapter_source_sha256"
            ],
            "checkpoint_path": resource["relative_path"],
            "checkpoint_sha256": resource["sha256"],
            "upstream_revision": configuration["upstream_revision"],
        }
    return {
        "schema_version": "1.0-frozen",
        "status": "partial",
        "figure": "Figure 1",
        "selection": {
            "series_id": manifest["series"]["series_id"],
            "frame_count": FRAME_COUNT,
            "seed_frame_order": SEED_FRAME_ORDER,
            "target_frame_order": TARGET_FRAME_ORDER,
            "target_frame_id": TARGET_FRAME_ORDER,
            "spot_id": SPOT_ID,
        },
        "replay_contract": {
            "full_chronological_sequence_replayed": True,
            "inclusive_frame_order_range": [0, TARGET_FRAME_ORDER],
            "query_scope": "selected spot 1 only",
            "query_independence": (
                "Both frozen adapters process each query independently; later "
                "queries cannot affect spot 1."
            ),
            "correction_or_reinitialization": "forbidden",
            "input_cache_shape": [256, 256],
            "coordinate_scale_original_to_cache": ORIGINAL_TO_CACHE,
        },
        "method_contracts": method_contracts,
        "sources": sources,
        "methods": {},
    }


def runner_resource_key(method: str) -> str:
    if method == OSTRACK_METHOD:
        return "ostrack"
    if method == DAM4SAM_METHOD:
        return "dam4sam"
    raise ValueError(method)


def export(methods: Sequence[str], output_directory: Path) -> None:
    runner = _load_upstream_runner()
    manifest, frame_rows, seeds = runner.load_contract(OSTRACK_METHOD)
    runner.load_contract(DAM4SAM_METHOD)
    if len(frame_rows) != FRAME_COUNT:
        raise ValueError("Targeted re-export requires all 701 frames")
    frames = runner.load_cache_frames(frame_rows)
    query_points, query_boxes = runner.build_prompts(seeds)
    query_points = query_points[SPOT_INDEX : SPOT_INDEX + 1]
    query_boxes = query_boxes[SPOT_INDEX : SPOT_INDEX + 1]
    if (
        int(query_points[0, 0]) != SEED_FRAME_ORDER
        or int(seeds[SPOT_INDEX]["spot_id"]) != SPOT_ID
    ):
        raise ValueError("Targeted re-export seed contract changed")

    output_directory.mkdir(parents=True, exist_ok=True)
    output_json_path = output_directory / OUTPUT_JSON_NAME
    payload = _base_payload(manifest)
    if output_json_path.is_file():
        existing = json.loads(output_json_path.read_text(encoding="utf-8"))
        if existing.get("schema_version") == payload["schema_version"]:
            payload["methods"] = existing.get("methods", {})

    for method in methods:
        started = perf_counter()
        if method == OSTRACK_METHOD:
            result = _run_ostrack(
                runner,
                frames=frames,
                query_points=query_points,
                query_boxes=query_boxes,
                manifest=manifest,
            )
        elif method == DAM4SAM_METHOD:
            result = _run_dam4sam(
                runner,
                frames=frames,
                query_points=query_points,
                query_boxes=query_boxes,
                manifest=manifest,
                output_directory=output_directory,
            )
        else:
            raise ValueError(f"Unsupported export method: {method}")
        payload["methods"][method] = result
        payload["status"] = (
            "native_geometry_reexported"
            if set(payload["methods"]) == set(EXPORT_METHODS)
            else "partial"
        )
        write_json(output_json_path, payload)
        print(
            f"[{method}] native geometry exported in "
            f"{perf_counter() - started:.1f} s",
            flush=True,
        )


def validate(output_directory: Path = SOURCE_DATA_DIRECTORY) -> None:
    geometry_path = output_directory / OUTPUT_JSON_NAME
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    if payload.get("status") != "native_geometry_reexported":
        raise ValueError("Figure 1 native-geometry re-export is incomplete")
    if set(payload.get("methods", {})) != set(EXPORT_METHODS):
        raise ValueError("Figure 1 native-geometry method set changed")
    for name, source in payload["sources"].items():
        source_path = REPOSITORY_ROOT / source["path"]
        if not source_path.is_file() or sha256_file(source_path) != source["sha256"]:
            raise ValueError(f"Native-geometry source hash mismatch: {name}")
    for method, contract in payload["method_contracts"].items():
        for path_key, hash_key in (
            ("adapter_source", "adapter_source_sha256"),
            ("checkpoint_path", "checkpoint_sha256"),
        ):
            source_path = REPOSITORY_ROOT / contract[path_key]
            if (
                not source_path.is_file()
                or sha256_file(source_path) != contract[hash_key]
            ):
                raise ValueError(
                    f"Native-geometry {method} {path_key} hash mismatch"
                )

    ostrack = payload["methods"][OSTRACK_METHOD]
    summarize_ostrack_bbox(
        np.asarray(ostrack["bbox_xywh_cache_px"], dtype=np.float64),
        expected_center_original_px=np.asarray(
            ostrack["archived_reduced_center_original_px"],
            dtype=np.float64,
        ),
    )
    dam4sam = payload["methods"][DAM4SAM_METHOD]
    mask_path = output_directory / Path(dam4sam["mask_path"]).name
    if not mask_path.is_file() or sha256_file(mask_path) != dam4sam["mask_sha256"]:
        raise ValueError("DAM4SAM native-mask hash mismatch")
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L")) > 0
    mask_summary = summarize_dam4sam_mask(
        mask,
        expected_center_original_px=np.asarray(
            dam4sam["archived_reduced_center_original_px"],
            dtype=np.float64,
        ),
    )
    if mask_summary["mask_area_cache_px"] != dam4sam["mask_area_cache_px"]:
        raise ValueError("DAM4SAM native-mask area changed")
    print(
        "Figure 1 native geometry validation passed: empirical OSTrack bbox "
        "and DAM4SAM mask reproduce the archived reduced trajectories.",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        action="append",
        choices=EXPORT_METHODS,
        help="Method to export; repeat to select both. Defaults to both.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SOURCE_DATA_DIRECTORY,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the committed geometry without rerunning GPU inference.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    output_directory = arguments.output_dir.expanduser().resolve()
    if arguments.validate_only:
        validate(output_directory)
        return
    export(
        tuple(arguments.method or EXPORT_METHODS),
        output_directory,
    )
    if set(arguments.method or EXPORT_METHODS) == set(EXPORT_METHODS):
        validate(output_directory)


if __name__ == "__main__":
    main()
