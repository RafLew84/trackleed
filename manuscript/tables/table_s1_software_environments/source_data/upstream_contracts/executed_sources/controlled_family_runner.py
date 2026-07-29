#!/usr/bin/env python3
"""Throwaway matched SAM2-family continuation of the ScanE tracer.

Run this script with the project's Windows ``netcat`` environment. It calls
the dedicated SAM2, SAMURAI and DAM4SAM environments through the existing
TrackLEED subprocess backends, then applies the shared Gaussian measurement
contract.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
BASELINE_OUTPUT = HERE / "output"
TRACKER_OUTPUT = BASELINE_OUTPUT / "trackers"
RAW_OUTPUT = TRACKER_OUTPUT / "raw"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))
import run as baseline  # noqa: E402

from trackleed.analysis.dam4sam_backend import (  # noqa: E402
    Dam4SamRuntimeSettings,
    run_dam4sam_offline,
)
from trackleed.analysis.sam2_backend import (  # noqa: E402
    Sam2RuntimeSettings,
    run_sam2_offline,
)
from trackleed.analysis.samurai_backend import (  # noqa: E402
    SamuraiRuntimeSettings,
    run_samurai_offline,
)


TRACKERS = ("sam2", "samurai", "dam4sam")
DIRECTIONS = ("high", "low")
BRANCHES = ("B1", "B2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the matched SAM2-family ScanE tracer."
    )
    parser.add_argument(
        "--tracker",
        choices=("both", *TRACKERS),
        default="both",
        help="Run one tracker or the complete controlled family.",
    )
    parser.add_argument(
        "--direction",
        choices=("both", *DIRECTIONS),
        default="both",
        help="Run the increasing-energy side, reversed decreasing side, or both.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore completed raw caches and rerun the selected GPU work.",
    )
    parser.add_argument(
        "--baseline-output",
        type=Path,
        default=None,
        help="Alternative physics-baseline output directory for a replicated pair.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="ScanE source directory matching the selected baseline output.",
    )
    return parser.parse_args()


def assert_runtime() -> None:
    if os.name != "nt":
        raise RuntimeError(
            "Run this prototype with the Windows netcat Python environment; "
            "the dedicated tracker environments are Windows Conda envs."
        )
    if not (BASELINE_OUTPUT / "summary.json").exists():
        raise RuntimeError("Run the physics baseline first; summary.json is missing.")


def runtime_paths() -> dict[str, Path]:
    conda_root = Path(sys.executable).resolve().parents[2]
    projects_root = REPO_ROOT.parent
    paths = {
        "sam2_python": conda_root / "envs" / "sam2" / "python.exe",
        "dam4sam_python": conda_root / "envs" / "dam4sam" / "python.exe",
        "samurai_python": conda_root / "envs" / "samurai" / "python.exe",
        "sam2_repo": projects_root / "sam2",
        "dam4sam_repo": projects_root / "dam4sam",
        "samurai_repo": projects_root / "samurai",
        "sam2_checkpoint": (
            REPO_ROOT
            / "tracking_models"
            / "sam2"
            / "sam2.1_hiera_base_plus.pt"
        ),
        "dam4sam_checkpoint": (
            REPO_ROOT
            / "tracking_models"
            / "dam4sam"
            / "sam2.1_hiera_base_plus.pt"
        ),
        "samurai_checkpoint": (
            REPO_ROOT
            / "tracking_models"
            / "samurai"
            / "sam2.1_hiera_base_plus.pt"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing tracker runtime paths:\n" + "\n".join(missing))
    return paths


def load_baseline_contract() -> tuple[
    list[baseline.Frame],
    list[dict[str, float]],
    int,
    list[dict[str, str]],
]:
    summary = json.loads(
        (BASELINE_OUTPUT / "summary.json").read_text(encoding="utf-8")
    )
    baseline.REFERENCE_ENERGY_EV = float(summary["reference_energy_ev"])
    seeds = [
        {
            "branch": str(seed["branch"]),
            "center_x": float(seed["center_x"]),
            "center_y": float(seed["center_y"]),
        }
        for seed in summary["frozen_seeds"]
    ]
    frames = baseline.load_frames()
    reference_index = min(
        range(len(frames)),
        key=lambda index: abs(
            frames[index].energy_ev - float(summary["reference_energy_ev"])
        ),
    )
    with (BASELINE_OUTPUT / "scan_measurements.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        baseline_rows = list(csv.DictReader(handle))
    return frames, seeds, reference_index, baseline_rows


def load_frame_stack(frames: list[baseline.Frame]) -> np.ndarray:
    arrays = [np.asarray(tifffile.imread(frame.path)) for frame in frames]
    stack = np.stack(arrays, axis=0)
    if stack.ndim != 3:
        raise ValueError(f"Expected [T,H,W] grayscale stack, got {stack.shape}")
    return stack


def sequence_indices(
    frame_count: int, reference_index: int, direction: str
) -> list[int]:
    if direction == "high":
        return list(range(reference_index, frame_count))
    if direction == "low":
        return list(range(reference_index, -1, -1))
    raise ValueError(direction)


def seed_box(seed: dict[str, float]) -> np.ndarray:
    half = baseline.ROI_SIZE / 2.0
    return np.asarray(
        [
            seed["center_x"] - half,
            seed["center_y"] - half,
            seed["center_x"] + half,
            seed["center_y"] + half,
        ],
        dtype=np.float32,
    )


def raw_cache_path(tracker: str, direction: str, branch: str) -> Path:
    return RAW_OUTPUT / f"{tracker}_{direction}_{branch}.npz"


def compact_result_payload(result: Any, tracker: str) -> dict[str, np.ndarray]:
    tracks = np.asarray(result.tracks_xy, dtype=np.float32)
    visible = np.asarray(result.visible_mask, dtype=bool)
    if tracks.shape[0] != 1 or visible.shape[0] != 1:
        raise ValueError(
            f"Expected one-object result, got tracks={tracks.shape}, visible={visible.shape}"
        )
    component_counts = getattr(result, "mask_component_counts", None)
    if component_counts is None:
        component_counts = np.zeros_like(visible, dtype=np.int32)
    return {
        "tracks_xy": tracks[0],
        "visible_mask": visible[0],
        "mask_areas": np.asarray(result.mask_areas, dtype=np.float32)[0],
        "mask_bboxes_xyxy": np.asarray(
            result.mask_bboxes_xyxy, dtype=np.float32
        )[0],
        "mask_scores": np.asarray(result.mask_scores, dtype=np.float32)[0],
        "mask_component_counts": np.asarray(component_counts, dtype=np.int32)[0],
        "inference_resolution": np.asarray(
            result.inference_resolution, dtype=np.int32
        ),
        "original_resolution": np.asarray(
            result.original_resolution, dtype=np.int32
        ),
        "checkpoint_name": np.asarray(result.checkpoint_name),
        "config_name": np.asarray(result.config_name),
        "representative_point_mode": np.asarray(
            result.representative_point_mode
        ),
        "tracker": np.asarray(tracker),
    }


def run_one(
    *,
    tracker: str,
    direction: str,
    branch: str,
    seed: dict[str, float],
    frames: np.ndarray,
    paths: dict[str, Path],
    force: bool,
) -> Path:
    cache = raw_cache_path(tracker, direction, branch)
    if cache.exists() and not force:
        print(f"CACHE {tracker} {direction} {branch}: {cache}", flush=True)
        return cache

    query = np.asarray(
        [[0.0, seed["center_y"], seed["center_x"]]], dtype=np.float32
    )
    box = seed_box(seed)[None, :]
    print(
        f"RUN {tracker} {direction} {branch}: frames={frames.shape[0]}, "
        f"seed=({seed['center_x']:.2f},{seed['center_y']:.2f})",
        flush=True,
    )

    if tracker == "sam2":
        result = run_sam2_offline(
            frames,
            query,
            Sam2RuntimeSettings(
                checkpoint_path=paths["sam2_checkpoint"],
                repo_path=paths["sam2_repo"],
                python_executable=paths["sam2_python"],
                backend_mode="subprocess",
                device="cuda",
                prompt_mode="bbox",
                representative_point_mode="mask_centroid",
                inference_resolution=None,
                initial_bbox_padding_px=0.0,
                mask_threshold=0.5,
                min_mask_area=4.0,
                max_mask_area=5000.0,
                use_vos_optimized=False,
                apply_postprocessing=True,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False,
                subprocess_timeout_sec=1800.0,
            ),
            query_boxes_xyxy=box,
        )
    elif tracker == "dam4sam":
        result = run_dam4sam_offline(
            frames,
            query,
            Dam4SamRuntimeSettings(
                checkpoint_path=paths["dam4sam_checkpoint"],
                repo_path=paths["dam4sam_repo"],
                python_executable=paths["dam4sam_python"],
                backend_mode="subprocess",
                tracker_name="sam21pp-B",
                device="cuda",
                representative_point_mode="mask_centroid",
                inference_resolution=None,
                initial_bbox_padding_px=0.0,
                subprocess_timeout_sec=1800.0,
            ),
            query_boxes_xyxy=box,
        )
    elif tracker == "samurai":
        result = run_samurai_offline(
            frames,
            query,
            SamuraiRuntimeSettings(
                checkpoint_path=paths["samurai_checkpoint"],
                repo_path=paths["samurai_repo"],
                python_executable=paths["samurai_python"],
                backend_mode="subprocess",
                model_variant="sam2.1_hiera_base_plus",
                device="cuda",
                representative_point_mode="mask_centroid",
                inference_resolution=None,
                initial_bbox_padding_px=0.0,
                subprocess_timeout_sec=1800.0,
            ),
            query_boxes_xyxy=box,
        )
    else:
        raise ValueError(tracker)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **compact_result_payload(result, tracker))
    print(f"SAVED {tracker} {direction} {branch}: {cache}", flush=True)
    del result
    gc.collect()
    return cache


def selected_values(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    return allowed if value == "both" else (value,)


def load_raw(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def merge_tracker_outputs(
    tracker: str,
    frames: list[baseline.Frame],
    reference_index: int,
) -> dict[str, np.ndarray]:
    frame_count = len(frames)
    tracks_xy = np.full((2, frame_count, 2), np.nan, dtype=np.float32)
    visible_mask = np.zeros((2, frame_count), dtype=bool)
    mask_areas = np.zeros((2, frame_count), dtype=np.float32)
    mask_bboxes = np.full((2, frame_count, 4), np.nan, dtype=np.float32)
    mask_scores = np.full((2, frame_count), np.nan, dtype=np.float32)
    component_counts = np.zeros((2, frame_count), dtype=np.int32)

    for branch_index, branch in enumerate(BRANCHES):
        for direction in DIRECTIONS:
            cache = raw_cache_path(tracker, direction, branch)
            if not cache.exists():
                raise FileNotFoundError(
                    f"Missing {cache}; both directions are required for analysis."
                )
            raw = load_raw(cache)
            indices = sequence_indices(frame_count, reference_index, direction)
            if len(indices) != int(raw["tracks_xy"].shape[0]):
                raise ValueError(
                    f"Cache length mismatch for {cache}: "
                    f"{raw['tracks_xy'].shape[0]} != {len(indices)}"
                )
            for local_index, global_index in enumerate(indices):
                # The high-side seed output wins the duplicate reference frame.
                if direction == "low" and global_index == reference_index:
                    continue
                tracks_xy[branch_index, global_index] = raw["tracks_xy"][local_index]
                visible_mask[branch_index, global_index] = bool(
                    raw["visible_mask"][local_index]
                )
                mask_areas[branch_index, global_index] = float(
                    raw["mask_areas"][local_index]
                )
                mask_bboxes[branch_index, global_index] = raw[
                    "mask_bboxes_xyxy"
                ][local_index]
                mask_scores[branch_index, global_index] = float(
                    raw["mask_scores"][local_index]
                )
                component_counts[branch_index, global_index] = int(
                    raw["mask_component_counts"][local_index]
                )
    return {
        "tracks_xy": tracks_xy,
        "visible_mask": visible_mask,
        "mask_areas": mask_areas,
        "mask_bboxes_xyxy": mask_bboxes,
        "mask_scores": mask_scores,
        "mask_component_counts": component_counts,
    }


def baseline_status_lookup(
    baseline_rows: list[dict[str, str]],
) -> dict[tuple[int, str], str]:
    return {
        (int(row["frame"]), str(row["branch"])): str(row["status"])
        for row in baseline_rows
    }


def measure_tracker(
    tracker: str,
    merged: dict[str, np.ndarray],
    frame_stack: np.ndarray,
    frames: list[baseline.Frame],
    seeds: list[dict[str, float]],
    geometry: baseline.Geometry,
    baseline_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    reference_status = baseline_status_lookup(baseline_rows)
    rows: list[dict[str, object]] = []
    center = np.array([geometry.center_x, geometry.center_y], dtype=float)
    for branch_index, seed in enumerate(seeds):
        branch = str(seed["branch"])
        for frame_index, frame in enumerate(frames):
            proposal = merged["tracks_xy"][branch_index, frame_index]
            native_visible = bool(merged["visible_mask"][branch_index, frame_index])
            predicted_x, predicted_y = baseline.predicted_position(
                seed, geometry, frame.energy_ev
            )
            row: dict[str, object] = {
                "tracker": tracker,
                "frame": frame.frame,
                "energy_ev": frame.energy_ev,
                "branch": branch,
                "physics_baseline_status": reference_status[(frame.frame, branch)],
                "native_visible": native_visible,
                "native_x": float(proposal[0]) if np.isfinite(proposal[0]) else None,
                "native_y": float(proposal[1]) if np.isfinite(proposal[1]) else None,
                "native_mask_area": float(
                    merged["mask_areas"][branch_index, frame_index]
                ),
                "native_mask_score": float(
                    merged["mask_scores"][branch_index, frame_index]
                ),
                "native_mask_component_count": int(
                    merged["mask_component_counts"][branch_index, frame_index]
                ),
                "predicted_x": predicted_x,
                "predicted_y": predicted_y,
            }
            if not native_visible or not np.all(np.isfinite(proposal)):
                row.update(
                    {
                        "measurement_valid": False,
                        "measurement_status": "tracker_not_visible",
                    }
                )
                rows.append(row)
                continue

            image = baseline.img_as_float(frame_stack[frame_index])
            result = baseline.measure(image, float(proposal[0]), float(proposal[1]))
            row.update(result)
            if not result.get("fit_valid"):
                row.update(
                    {
                        "measurement_valid": False,
                        "measurement_status": "gaussian_rejected",
                    }
                )
                rows.append(row)
                continue

            measured = np.array(
                [float(result["center_x"]), float(result["center_y"])], dtype=float
            )
            expected = np.array([predicted_x, predicted_y], dtype=float)
            residual = measured - expected
            radial_unit = (expected - center) / np.linalg.norm(expected - center)
            tangent_unit = np.asarray([-radial_unit[1], radial_unit[0]])
            branch_error = float(np.linalg.norm(residual))
            row.update(
                {
                    "measurement_valid": True,
                    "measurement_status": (
                        "branch_consistent"
                        if branch_error <= baseline.BRANCH_TOLERANCE_PX
                        else "geometry_inconsistent"
                    ),
                    "branch_error_px": branch_error,
                    "radial_residual_px": float(np.dot(residual, radial_unit)),
                    "tangential_residual_px": float(np.dot(residual, tangent_unit)),
                }
            )
            rows.append(row)
    return rows


def build_pair_rows(
    tracker_rows: list[dict[str, object]],
    seeds: list[dict[str, float]],
) -> list[dict[str, object]]:
    by_frame: dict[int, dict[str, dict[str, object]]] = {}
    for row in tracker_rows:
        if row.get("measurement_status") != "branch_consistent":
            continue
        by_frame.setdefault(int(row["frame"]), {})[str(row["branch"])] = row

    seed_1, seed_2 = seeds
    reference_distance = math.hypot(
        seed_2["center_x"] - seed_1["center_x"],
        seed_2["center_y"] - seed_1["center_y"],
    )
    reference_midpoint = np.asarray(
        [
            (seed_1["center_x"] + seed_2["center_x"]) / 2.0,
            (seed_1["center_y"] + seed_2["center_y"]) / 2.0,
        ]
    )
    pair_rows: list[dict[str, object]] = []
    for frame, branches in sorted(by_frame.items()):
        if set(branches) != set(BRANCHES):
            continue
        first = np.asarray(
            [
                float(branches["B1"]["center_x"]),
                float(branches["B1"]["center_y"]),
            ]
        )
        second = np.asarray(
            [
                float(branches["B2"]["center_x"]),
                float(branches["B2"]["center_y"]),
            ]
        )
        energy_ev = float(branches["B1"]["energy_ev"])
        distance = float(np.linalg.norm(second - first))
        predicted = reference_distance * math.sqrt(
            baseline.REFERENCE_ENERGY_EV / energy_ev
        )
        midpoint = (first + second) / 2.0
        pair_rows.append(
            {
                "tracker": branches["B1"]["tracker"],
                "frame": frame,
                "energy_ev": energy_ev,
                "pair_distance_px": distance,
                "predicted_distance_px": predicted,
                "distance_residual_px": distance - predicted,
                "midpoint_x": float(midpoint[0]),
                "midpoint_y": float(midpoint[1]),
                "midpoint_drift_px": float(
                    np.linalg.norm(midpoint - reference_midpoint)
                ),
            }
        )
    return pair_rows


def contiguous_gap_runs(
    frames: list[baseline.Frame],
    baseline_rows: list[dict[str, str]],
    branch: str,
) -> list[tuple[int, int]]:
    lookup = baseline_status_lookup(baseline_rows)
    indices = [
        index
        for index, frame in enumerate(frames)
        if lookup[(frame.frame, branch)] == "iv_visibility_gap"
    ]
    runs: list[tuple[int, int]] = []
    for index in indices:
        if not runs or index != runs[-1][1] + 1:
            runs.append((index, index))
        else:
            runs[-1] = (runs[-1][0], index)
    return runs


def evaluate_gap_recovery(
    tracker_rows: list[dict[str, object]],
    frames: list[baseline.Frame],
    baseline_rows: list[dict[str, str]],
    reference_index: int,
) -> list[dict[str, object]]:
    row_lookup = {
        (int(row["frame"]), str(row["branch"])): row for row in tracker_rows
    }
    baseline_lookup = baseline_status_lookup(baseline_rows)
    results: list[dict[str, object]] = []
    for branch in BRANCHES:
        for start, end in contiguous_gap_runs(frames, baseline_rows, branch):
            if start > reference_index:
                direction = "high"
                post_indices = range(end + 1, len(frames))
            elif end < reference_index:
                direction = "low"
                post_indices = range(start - 1, -1, -1)
            else:
                continue

            first_expected_index = None
            for index in post_indices:
                status = baseline_lookup[(frames[index].frame, branch)]
                if status == "visible_fit":
                    first_expected_index = index
                    break
                if status in {"apparatus_absence", "beam_onset_or_termination"}:
                    break
            if first_expected_index is None:
                continue

            recovery_index = None
            step = 1 if direction == "high" else -1
            index = first_expected_index
            delay = 0
            while 0 <= index < len(frames):
                if baseline_lookup[(frames[index].frame, branch)] != "visible_fit":
                    break
                tracker_row = row_lookup[(frames[index].frame, branch)]
                if tracker_row.get("measurement_status") == "branch_consistent":
                    recovery_index = index
                    break
                delay += 1
                index += step

            results.append(
                {
                    "tracker": tracker_rows[0]["tracker"],
                    "branch": branch,
                    "direction": direction,
                    "gap_start_energy_ev": frames[start].energy_ev,
                    "gap_end_energy_ev": frames[end].energy_ev,
                    "gap_frame_count": end - start + 1,
                    "first_expected_recovery_energy_ev": frames[
                        first_expected_index
                    ].energy_ev,
                    "recovered": recovery_index is not None,
                    "recovery_energy_ev": (
                        frames[recovery_index].energy_ev
                        if recovery_index is not None
                        else None
                    ),
                    "recovery_delay_frames": delay if recovery_index is not None else None,
                }
            )
    return results


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tracker_summary(
    tracker: str,
    rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    gap_rows: list[dict[str, object]],
) -> dict[str, object]:
    baseline_visible = [
        row for row in rows if row["physics_baseline_status"] == "visible_fit"
    ]
    consistent = [
        row
        for row in baseline_visible
        if row.get("measurement_status") == "branch_consistent"
    ]
    accepted_all = [row for row in rows if row.get("measurement_valid")]
    inconsistent = [
        row for row in accepted_all if row["measurement_status"] == "geometry_inconsistent"
    ]
    pair_residuals = [
        abs(float(row["distance_residual_px"])) for row in pair_rows
    ]
    recovered = [row for row in gap_rows if row["recovered"]]
    return {
        "tracker": tracker,
        "physics_baseline_visible_opportunities": len(baseline_visible),
        "native_visible_on_baseline_visible": sum(
            bool(row["native_visible"]) for row in baseline_visible
        ),
        "branch_consistent_measurements": len(consistent),
        "fit_coverage_on_baseline_visible": (
            len(consistent) / len(baseline_visible) if baseline_visible else None
        ),
        "accepted_measurements_all_frames": len(accepted_all),
        "geometry_inconsistent_accepted_fits": len(inconsistent),
        "jointly_consistent_pair_frames": len(pair_rows),
        "absolute_pair_distance_residual_px": {
            "median": (
                float(np.median(pair_residuals)) if pair_residuals else None
            ),
            "p95": (
                float(np.percentile(pair_residuals, 95))
                if pair_residuals
                else None
            ),
            "max": max(pair_residuals) if pair_residuals else None,
        },
        "post_gap_recovery": {
            "evaluated_gap_count": len(gap_rows),
            "recovered_gap_count": len(recovered),
            "rate": len(recovered) / len(gap_rows) if gap_rows else None,
            "zero_delay_count": sum(
                int(row["recovery_delay_frames"]) == 0 for row in recovered
            ),
        },
    }


def plot_comparison() -> None:
    summaries = sorted(TRACKER_OUTPUT.glob("*_pair_measurements.csv"))
    if not summaries:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    baseline_pair_path = BASELINE_OUTPUT / "pair_measurements.csv"
    if baseline_pair_path.exists():
        with baseline_pair_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        axes[0].plot(
            [float(row["energy_ev"]) for row in rows],
            [float(row["distance_residual_px"]) for row in rows],
            marker=".",
            color="black",
            alpha=0.55,
            label="physics baseline",
        )
        axes[1].plot(
            [float(row["energy_ev"]) for row in rows],
            [float(row["midpoint_drift_px"]) for row in rows],
            marker=".",
            color="black",
            alpha=0.55,
            label="physics baseline",
        )
    for path in summaries:
        tracker = path.name.split("_", 1)[0]
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        energies = [float(row["energy_ev"]) for row in rows]
        residuals = [float(row["distance_residual_px"]) for row in rows]
        axes[0].plot(energies, residuals, marker=".", label=tracker)
        axes[1].plot(
            energies,
            [float(row["midpoint_drift_px"]) for row in rows],
            marker=".",
            label=tracker,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_xlabel("Energy [eV]")
    axes[0].set_ylabel("Pair-distance residual [px]")
    axes[0].set_title("Common Gaussian measurement")
    axes[1].set_xlabel("Energy [eV]")
    axes[1].set_ylabel("Pair midpoint drift [px]")
    axes[1].set_title("Common-mode diagnostic")
    tracker_summaries: dict[str, dict[str, object]] = {}
    for tracker in TRACKERS:
        path = TRACKER_OUTPUT / f"{tracker}_summary.json"
        if path.exists():
            tracker_summaries[tracker] = json.loads(path.read_text(encoding="utf-8"))
    names = list(tracker_summaries)
    x = np.arange(len(names), dtype=float)
    axes[2].bar(
        x - 0.18,
        [
            100.0 * float(tracker_summaries[name]["fit_coverage_on_baseline_visible"])
            for name in names
        ],
        width=0.36,
        label="fit coverage",
    )
    axes[2].bar(
        x + 0.18,
        [
            100.0 * float(tracker_summaries[name]["post_gap_recovery"]["rate"])
            for name in names
        ],
        width=0.36,
        label="gap recovery",
    )
    axes[2].set_xticks(x, labels=names)
    axes[2].set_ylim(0.0, 105.0)
    axes[2].set_ylabel("Rate [%]")
    axes[2].set_title("Measurement-chain outcomes")
    axes[2].legend(fontsize=8)
    for axis in axes[:2]:
        axis.legend()
    fig.tight_layout()
    fig.savefig(TRACKER_OUTPUT / "tracker_pair_comparison.png", dpi=180)
    plt.close(fig)


def plot_status_timeline() -> None:
    status_colors = {
        "branch_consistent": "#1b9e77",
        "tracker_not_visible": "#666666",
        "gaussian_rejected": "#d95f02",
        "geometry_inconsistent": "#e7298a",
    }
    fig, ax = plt.subplots(figsize=(12.0, 4.2))
    labels: list[str] = []
    row_number = 0
    for tracker in TRACKERS:
        path = TRACKER_OUTPUT / f"{tracker}_measurements.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for branch in BRANCHES:
            labels.append(f"{tracker} {branch}")
            branch_rows = [row for row in rows if str(row["branch"]) == branch]
            for row in branch_rows:
                status = str(row["measurement_status"])
                ax.scatter(
                    float(row["energy_ev"]),
                    row_number,
                    color=status_colors[status],
                    s=22,
                )
            row_number += 1
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Energy [eV]")
    ax.set_title("Native proposal → common Gaussian measurement status")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color,
            label=status.replace("_", " "),
        )
        for status, color in status_colors.items()
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(TRACKER_OUTPUT / "tracker_status_timeline.png", dpi=180)
    plt.close(fig)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_comparison_summary(paths: dict[str, Path]) -> None:
    summary_paths = {
        tracker: TRACKER_OUTPUT / f"{tracker}_summary.json"
        for tracker in TRACKERS
    }
    if not all(path.exists() for path in summary_paths.values()):
        return
    method_results = {
        tracker: json.loads(path.read_text(encoding="utf-8"))
        for tracker, path in summary_paths.items()
    }
    sam2_hash = sha256_file(paths["sam2_checkpoint"])
    samurai_hash = sha256_file(paths["samurai_checkpoint"])
    dam4sam_hash = sha256_file(paths["dam4sam_checkpoint"])
    sam2 = method_results["sam2"]
    samurai = method_results["samurai"]
    dam4sam = method_results["dam4sam"]

    def contrast(first: dict[str, object], second: dict[str, object]) -> dict[str, float | int]:
        return {
            "fit_coverage_percentage_points": 100.0
            * (
                float(first["fit_coverage_on_baseline_visible"])
                - float(second["fit_coverage_on_baseline_visible"])
            ),
            "post_gap_recovery_percentage_points": 100.0
            * (
                float(first["post_gap_recovery"]["rate"])
                - float(second["post_gap_recovery"]["rate"])
            ),
            "jointly_consistent_pair_frames": int(
                first["jointly_consistent_pair_frames"]
            )
            - int(second["jointly_consistent_pair_frames"]),
            "geometry_inconsistent_accepted_fits_all_frames": int(
                first["geometry_inconsistent_accepted_fits"]
            )
            - int(second["geometry_inconsistent_accepted_fits"]),
        }

    comparison = {
        "scope": (
            "Single automatically frozen diametric pair in one ScanE series; "
            "annotation-free matched-control prototype, not absolute accuracy."
        ),
        "frozen_contract": {
            "reference_energy_ev": baseline.REFERENCE_ENERGY_EV,
            "prompt": "single-object bbox 48x48 px",
            "representative_point": "mask_centroid",
            "measurement_roi_px": baseline.ROI_SIZE,
            "common_estimator": "TrackLEED 2D elliptical Gaussian fit",
            "sam2_variant": "sam2.1_hiera_base_plus",
            "samurai_variant": "sam2.1_hiera_base_plus with motion-aware memory",
            "dam4sam_variant": "sam21pp-B",
            "sam2_checkpoint_sha256": sam2_hash,
            "samurai_checkpoint_sha256": samurai_hash,
            "dam4sam_checkpoint_sha256": dam4sam_hash,
            "identical_checkpoint_weights": (
                sam2_hash == samurai_hash == dam4sam_hash
            ),
            "directions": [
                "187.9 to 220.2 eV",
                "187.9 to 45.0 eV with reversed frame order",
            ],
        },
        "methods": method_results,
        "paired_contrasts": {
            "samurai_minus_sam2": contrast(samurai, sam2),
            "dam4sam_minus_sam2": contrast(dam4sam, sam2),
            "dam4sam_minus_samurai": contrast(dam4sam, samurai),
        },
        "prototype_readout": (
            "This matched SAM2-family control separates vanilla propagation, "
            "motion-aware memory, and distractor-aware memory under identical "
            "weights and measurement. The conclusion remains provisional "
            "because this is one pair in one series."
        ),
    }
    (TRACKER_OUTPUT / "comparison_summary.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def analyze_tracker(
    tracker: str,
    frame_stack: np.ndarray,
    frames: list[baseline.Frame],
    seeds: list[dict[str, float]],
    reference_index: int,
    baseline_rows: list[dict[str, str]],
) -> dict[str, object]:
    summary = json.loads(
        (BASELINE_OUTPUT / "summary.json").read_text(encoding="utf-8")
    )
    geometry = baseline.Geometry(
        center_x=float(summary["geometry"]["center_x"]),
        center_y=float(summary["geometry"]["center_y"]),
        aperture_radius=float(summary["geometry"]["aperture_radius_px"]),
        screen_radius=float(summary["geometry"]["screen_radius_px"]),
    )
    merged = merge_tracker_outputs(tracker, frames, reference_index)
    rows = measure_tracker(
        tracker,
        merged,
        frame_stack,
        frames,
        seeds,
        geometry,
        baseline_rows,
    )
    pair_rows = build_pair_rows(rows, seeds)
    gap_rows = evaluate_gap_recovery(
        rows, frames, baseline_rows, reference_index
    )
    result = tracker_summary(tracker, rows, pair_rows, gap_rows)

    write_csv(TRACKER_OUTPUT / f"{tracker}_measurements.csv", rows)
    write_csv(TRACKER_OUTPUT / f"{tracker}_pair_measurements.csv", pair_rows)
    write_csv(TRACKER_OUTPUT / f"{tracker}_gap_recovery.csv", gap_rows)
    (TRACKER_OUTPUT / f"{tracker}_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(TRACKER_OUTPUT / f"{tracker}_merged_native.npz", **merged)
    return result


def main() -> int:
    global BASELINE_OUTPUT, TRACKER_OUTPUT, RAW_OUTPUT
    args = parse_args()
    if args.baseline_output is not None:
        BASELINE_OUTPUT = args.baseline_output.expanduser().resolve()
        TRACKER_OUTPUT = BASELINE_OUTPUT / "trackers"
        RAW_OUTPUT = TRACKER_OUTPUT / "raw"
    if args.data_dir is not None:
        baseline.DATA_DIR = args.data_dir.expanduser().resolve()
    assert_runtime()
    paths = runtime_paths()
    frames, seeds, reference_index, baseline_rows = load_baseline_contract()
    frame_stack = load_frame_stack(frames)
    trackers = selected_values(args.tracker, TRACKERS)
    directions = selected_values(args.direction, DIRECTIONS)

    for tracker in trackers:
        for direction in directions:
            indices = sequence_indices(len(frames), reference_index, direction)
            sequence = frame_stack[indices]
            for seed in seeds:
                run_one(
                    tracker=tracker,
                    direction=direction,
                    branch=str(seed["branch"]),
                    seed=seed,
                    frames=sequence,
                    paths=paths,
                    force=bool(args.force),
                )

    TRACKER_OUTPUT.mkdir(parents=True, exist_ok=True)
    completed: dict[str, object] = {}
    for tracker in trackers:
        if all(
            raw_cache_path(tracker, direction, branch).exists()
            for direction in DIRECTIONS
            for branch in BRANCHES
        ):
            completed[tracker] = analyze_tracker(
                tracker,
                frame_stack,
                frames,
                seeds,
                reference_index,
                baseline_rows,
            )
            print(json.dumps(completed[tracker], indent=2), flush=True)

    plot_comparison()
    plot_status_timeline()
    write_comparison_summary(paths)
    print(f"Results: {TRACKER_OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
