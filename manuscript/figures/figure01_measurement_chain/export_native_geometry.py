#!/usr/bin/env python3
"""Re-export the native OSTrack bbox and DAM4SAM mask for Figure 1."""

from __future__ import annotations

import argparse
import csv
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
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from trackleed.analysis.publication_native_geometry import (  # noqa: E402
    ORIGINAL_TO_CACHE,
    compare_reduced_trajectories,
    summarize_dam4sam_mask,
    summarize_ostrack_bbox,
)

SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
ARTIFACT_ROOT = (
    REPOSITORY_ROOT / ".scratch/trackleed-physics-method-paper/artifacts/"
    "cu111-tracker-evaluation-subset"
)
GPU_NATIVE_ROOT = ARTIFACT_ROOT / "runs/gpu-native-v1"
UPSTREAM_RUNNER_PATH = ARTIFACT_ROOT / "run_gpu_cu.py"
OUTPUT_JSON_NAME = "representative_native_geometry.json"
MASK_PNG_NAME = "representative_dam4sam_mask_cache.png"
REPLAY_CSV_NAME = "representative_native_geometry_replay.csv"

OSTRACK_METHOD = "ostrack_vitb_256_mae_ce"
DAM4SAM_METHOD = "dam4sam_sam21pp_l"
EXPORT_METHODS = (OSTRACK_METHOD, DAM4SAM_METHOD)
SPOT_ID = 1
SPOT_INDEX = 0
SEED_FRAME_ORDER = 0
TARGET_FRAME_ORDER = 700
FRAME_COUNT = 701
RESOURCE_KEYS = {
    OSTRACK_METHOD: "ostrack",
    DAM4SAM_METHOD: "dam4sam",
}


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = (
        "method",
        "frame_order",
        "frame_id",
        "proposal_available",
        "x_cache_px",
        "y_cache_px",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    path = GPU_NATIVE_ROOT / method / f"{method}_spot_{SPOT_ID:02d}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _compare_with_archived_trajectory(
    method: str,
    *,
    tracks_xy_cache: np.ndarray,
    proposal_available: np.ndarray,
) -> dict[str, object]:
    archived = _archived_payload(method)
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
    comparison = compare_reduced_trajectories(
        tracks_xy_cache,
        proposal_available,
        expected_tracks,
        expected_available,
        method=method,
    )
    comparison.update(
        {
            "archived_payload_path": (
                path := GPU_NATIVE_ROOT / method / f"{method}_spot_{SPOT_ID:02d}.npz"
            )
            .resolve()
            .relative_to(REPOSITORY_ROOT.resolve())
            .as_posix(),
            "archived_payload_sha256": sha256_file(path),
        }
    )
    return comparison


def _run_ostrack(
    runner: ModuleType,
    *,
    frames: np.ndarray,
    query_points: np.ndarray,
    query_boxes: np.ndarray,
    manifest: dict[str, Any],
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
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
            representative_point_mode=str(settings["representative_point_mode"]),
            inference_resolution=tuple(settings["inference_resolution"]),
            initial_bbox_padding_px=float(settings["initial_bbox_padding_px"]),
            subprocess_timeout_sec=float(settings["subprocess_timeout_seconds"]),
        ),
        query_boxes_xyxy=query_boxes,
    )
    normalized = runner.normalize_gpu_proposals(
        method,
        tracks_xy=result.tracks_xy,
        visibility_signal=result.bbox_scores,
        boxes_xywh=result.tracked_bboxes_xywh,
    )
    replay_tracks = np.asarray(normalized["tracks_xy"][0])
    available = np.asarray(normalized["proposal_available"][0], dtype=bool)
    trajectory = _compare_with_archived_trajectory(
        method,
        tracks_xy_cache=replay_tracks,
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
            "native_bbox_score": float(result.bbox_scores[0, TARGET_FRAME_ORDER]),
            "trajectory_replay": trajectory,
        }
    )
    return summary, replay_tracks, available


def _run_dam4sam(
    runner: ModuleType,
    *,
    frames: np.ndarray,
    query_points: np.ndarray,
    query_boxes: np.ndarray,
    manifest: dict[str, Any],
    output_directory: Path,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
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
            representative_point_mode=str(settings["representative_point_mode"]),
            inference_resolution=tuple(settings["inference_resolution"]),
            initial_bbox_padding_px=float(settings["initial_bbox_padding_px"]),
            subprocess_timeout_sec=float(settings["subprocess_timeout_seconds"]),
        ),
        query_boxes_xyxy=query_boxes,
    )
    normalized = runner.normalize_gpu_proposals(
        method,
        tracks_xy=result.tracks_xy,
        visibility_signal=result.mask_scores,
        mask_areas=result.mask_areas,
    )
    replay_tracks = np.asarray(normalized["tracks_xy"][0])
    available = np.asarray(normalized["proposal_available"][0], dtype=bool)
    trajectory = _compare_with_archived_trajectory(
        method,
        tracks_xy_cache=replay_tracks,
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
                if math.isfinite(float(result.mask_scores[0, TARGET_FRAME_ORDER]))
                else None
            ),
            "trajectory_replay": trajectory,
        }
    )
    return summary, replay_tracks, available


def _write_replay_trajectory(
    output_directory: Path,
    *,
    method: str,
    frame_rows: Sequence[dict[str, str]],
    tracks_xy_cache: np.ndarray,
    proposal_available: np.ndarray,
) -> Path:
    replay_path = output_directory / REPLAY_CSV_NAME
    retained_rows = (
        [row for row in read_csv(replay_path) if row["method"] != method]
        if replay_path.is_file()
        else []
    )
    tracks = np.asarray(tracks_xy_cache, dtype=np.float64)
    available = np.asarray(proposal_available, dtype=bool)
    if tracks.shape != (FRAME_COUNT, 2) or available.shape != (FRAME_COUNT,):
        raise ValueError(f"{method} replay arrays have unexpected shapes")
    new_rows: list[dict[str, object]] = []
    for frame_order, frame in enumerate(frame_rows):
        is_available = bool(available[frame_order])
        new_rows.append(
            {
                "method": method,
                "frame_order": frame_order,
                "frame_id": int(frame["frame_id"]),
                "proposal_available": ("true" if is_available else "false"),
                "x_cache_px": (
                    format(float(tracks[frame_order, 0]), ".17g")
                    if is_available
                    else ""
                ),
                "y_cache_px": (
                    format(float(tracks[frame_order, 1]), ".17g")
                    if is_available
                    else ""
                ),
            }
        )
    method_order = {name: index for index, name in enumerate(EXPORT_METHODS)}
    combined = retained_rows + new_rows
    combined.sort(
        key=lambda row: (
            method_order[str(row["method"])],
            int(row["frame_order"]),
        )
    )
    write_csv(replay_path, combined)
    return replay_path


def _base_payload(
    manifest: dict[str, Any],
    seeds: Sequence[dict[str, str]],
) -> dict[str, object]:
    sources: dict[str, dict[str, str]] = {}
    for name, path in {
        "contract_manifest": ARTIFACT_ROOT / "manifest.json",
        "inference_frames": ARTIFACT_ROOT / "inference_frames.csv",
        "seed_prompts": ARTIFACT_ROOT / "seed_prompts.csv",
        "frozen_gpu_runner": UPSTREAM_RUNNER_PATH,
        "exporter": Path(__file__).resolve(),
        "native_geometry_analysis": (
            REPOSITORY_ROOT / "trackleed/analysis/publication_native_geometry.py"
        ),
    }.items():
        sources[name] = {
            "path": path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix(),
            "sha256": sha256_file(path),
        }
    method_contracts = {}
    for method in EXPORT_METHODS:
        configuration = manifest["method_configurations"][method]
        resource_key = RESOURCE_KEYS[method]
        resource = manifest["method_resources"][resource_key]
        method_contracts[method] = {
            "method_configuration_sha256": manifest["method_configuration_sha256"][
                method
            ],
            "adapter_source": configuration["adapter_source"],
            "adapter_source_sha256": configuration["adapter_source_sha256"],
            "checkpoint_path": resource["relative_path"],
            "checkpoint_sha256": resource["sha256"],
            "upstream_revision": configuration["upstream_revision"],
        }
    return {
        "schema_version": "1.1-frozen",
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
            "shared_initialization": {
                "seed_center_original_px": [
                    float(seeds[SPOT_INDEX]["center_x_original_px"]),
                    float(seeds[SPOT_INDEX]["center_y_original_px"]),
                ],
                "seed_bbox_xyxy_original_px": [
                    float(seeds[SPOT_INDEX]["bbox_x0_original_px"]),
                    float(seeds[SPOT_INDEX]["bbox_y0_original_px"]),
                    float(seeds[SPOT_INDEX]["bbox_x1_original_px"]),
                    float(seeds[SPOT_INDEX]["bbox_y1_original_px"]),
                ],
                "seed_source": seeds[SPOT_INDEX]["seed_source"],
                "seed_is_ground_truth": False,
            },
        },
        "method_contracts": method_contracts,
        "sources": sources,
        "methods": {},
    }


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
    payload = _base_payload(manifest, seeds)
    if output_json_path.is_file():
        existing = json.loads(output_json_path.read_text(encoding="utf-8"))
        compatible_keys = (
            "schema_version",
            "selection",
            "replay_contract",
            "method_contracts",
            "sources",
        )
        if all(existing.get(key) == payload.get(key) for key in compatible_keys):
            payload["methods"] = existing.get("methods", {})

    method_runners = {
        OSTRACK_METHOD: _run_ostrack,
        DAM4SAM_METHOD: _run_dam4sam,
    }
    for method in methods:
        started = perf_counter()
        keyword_arguments = {
            "runner": runner,
            "frames": frames,
            "query_points": query_points,
            "query_boxes": query_boxes,
            "manifest": manifest,
        }
        if method == DAM4SAM_METHOD:
            keyword_arguments["output_directory"] = output_directory
        result, replay_tracks, replay_available = method_runners[method](
            **keyword_arguments
        )
        replay_path = _write_replay_trajectory(
            output_directory,
            method=method,
            frame_rows=frame_rows,
            tracks_xy_cache=replay_tracks,
            proposal_available=replay_available,
        )
        payload["methods"][method] = result
        replay_hash = sha256_file(replay_path)
        for method_payload in payload["methods"].values():
            method_payload["trajectory_replay"].update(
                {
                    "reexported_trajectory_path": (f"source_data/{REPLAY_CSV_NAME}"),
                    "reexported_trajectory_sha256": replay_hash,
                }
            )
        payload["status"] = (
            "native_geometry_reexported"
            if set(payload["methods"]) == set(EXPORT_METHODS)
            else "partial"
        )
        write_json(output_json_path, payload)
        print(
            f"[{method}] native geometry exported in {perf_counter() - started:.1f} s",
            flush=True,
        )


def validate(output_directory: Path = SOURCE_DATA_DIRECTORY) -> None:
    geometry_path = output_directory / OUTPUT_JSON_NAME
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    if payload.get("status") != "native_geometry_reexported":
        raise ValueError("Figure 1 native-geometry re-export is incomplete")
    if set(payload.get("methods", {})) != set(EXPORT_METHODS):
        raise ValueError("Figure 1 native-geometry method set changed")
    expected_selection = {
        "series_id": "cu111_56ev",
        "frame_count": FRAME_COUNT,
        "seed_frame_order": SEED_FRAME_ORDER,
        "target_frame_order": TARGET_FRAME_ORDER,
        "target_frame_id": TARGET_FRAME_ORDER,
        "spot_id": SPOT_ID,
    }
    if payload.get("selection") != expected_selection:
        raise ValueError("Figure 1 native-geometry selection contract changed")
    replay_contract = payload.get("replay_contract", {})
    expected_replay_fields = {
        "full_chronological_sequence_replayed": True,
        "inclusive_frame_order_range": [0, TARGET_FRAME_ORDER],
        "query_scope": "selected spot 1 only",
        "correction_or_reinitialization": "forbidden",
        "input_cache_shape": [256, 256],
        "coordinate_scale_original_to_cache": ORIGINAL_TO_CACHE,
    }
    for field, expected in expected_replay_fields.items():
        if replay_contract.get(field) != expected:
            raise ValueError(f"Figure 1 native-geometry replay field changed: {field}")
    seed_rows = [
        row
        for row in read_csv(ARTIFACT_ROOT / "seed_prompts.csv")
        if int(row["spot_id"]) == SPOT_ID
    ]
    if len(seed_rows) != 1:
        raise ValueError("Figure 1 native-geometry seed row changed")
    seed = seed_rows[0]
    expected_initialization = {
        "seed_center_original_px": [
            float(seed["center_x_original_px"]),
            float(seed["center_y_original_px"]),
        ],
        "seed_bbox_xyxy_original_px": [
            float(seed["bbox_x0_original_px"]),
            float(seed["bbox_y0_original_px"]),
            float(seed["bbox_x1_original_px"]),
            float(seed["bbox_y1_original_px"]),
        ],
        "seed_source": seed["seed_source"],
        "seed_is_ground_truth": False,
    }
    if replay_contract.get("shared_initialization") != expected_initialization:
        raise ValueError("Figure 1 native-geometry shared initialization changed")
    for name, source in payload["sources"].items():
        source_path = REPOSITORY_ROOT / source["path"]
        if not source_path.is_file() or sha256_file(source_path) != source["sha256"]:
            raise ValueError(f"Native-geometry source hash mismatch: {name}")
    for method, contract in payload["method_contracts"].items():
        adapter_path = REPOSITORY_ROOT / contract["adapter_source"]
        if (
            not adapter_path.is_file()
            or sha256_file(adapter_path) != contract["adapter_source_sha256"]
        ):
            raise ValueError(f"Native-geometry {method} adapter_source hash mismatch")
        checkpoint_path = Path(str(contract["checkpoint_path"]))
        checkpoint_sha256 = str(contract["checkpoint_sha256"])
        if (
            checkpoint_path.is_absolute()
            or ".." in checkpoint_path.parts
            or len(checkpoint_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in checkpoint_sha256
            )
        ):
            raise ValueError(f"Native-geometry {method} checkpoint declaration changed")

    replay_path = output_directory / REPLAY_CSV_NAME
    replay_hash = sha256_file(replay_path)
    replay_rows = read_csv(replay_path)
    inference_rows = read_csv(ARTIFACT_ROOT / "inference_frames.csv")
    if len(replay_rows) != FRAME_COUNT * len(EXPORT_METHODS):
        raise ValueError("Figure 1 native-geometry replay row count changed")
    for method in EXPORT_METHODS:
        method_rows = [row for row in replay_rows if row["method"] == method]
        if [int(row["frame_order"]) for row in method_rows] != list(range(FRAME_COUNT)):
            raise ValueError(f"Figure 1 {method} replay frame order changed")
        if [int(row["frame_id"]) for row in method_rows] != [
            int(row["frame_id"]) for row in inference_rows
        ]:
            raise ValueError(f"Figure 1 {method} replay frame IDs changed")
        actual_available = np.asarray(
            [row["proposal_available"] == "true" for row in method_rows],
            dtype=bool,
        )
        actual_tracks = np.asarray(
            [
                [
                    float(row["x_cache_px"]) if row["x_cache_px"] else np.nan,
                    float(row["y_cache_px"]) if row["y_cache_px"] else np.nan,
                ]
                for row in method_rows
            ],
            dtype=np.float64,
        )
        recomputed = _compare_with_archived_trajectory(
            method,
            tracks_xy_cache=actual_tracks,
            proposal_available=actual_available,
        )
        recorded = payload["methods"][method]["trajectory_replay"]
        for key, expected in recomputed.items():
            actual = recorded.get(key)
            if isinstance(expected, float):
                matches = math.isclose(
                    float(actual),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(f"Figure 1 {method} replay metric changed: {key}")
        if (
            recorded.get("reexported_trajectory_path")
            != f"source_data/{REPLAY_CSV_NAME}"
            or recorded.get("reexported_trajectory_sha256") != replay_hash
        ):
            raise ValueError(f"Figure 1 {method} replay trajectory hash changed")

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
