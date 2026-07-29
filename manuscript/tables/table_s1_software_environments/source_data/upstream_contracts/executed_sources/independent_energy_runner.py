#!/usr/bin/env python3
"""PROTOTYPE: compare point, bbox, mask and LK proposals on frozen ScanE pairs.

This runner answers one publication-design question. It is not production
benchmark infrastructure and deliberately reuses the frozen measurement
contract from ``run_trackers.py``.
"""

from __future__ import annotations

import argparse
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


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
BASELINE_OUTPUT = HERE / "output"
OUTPUT_DIR = BASELINE_OUTPUT / "representatives"
RAW_DIR = OUTPUT_DIR / "raw"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))
import run as baseline  # noqa: E402
import run_trackers as common  # noqa: E402

from trackleed.analysis.locotrack_backend import (  # noqa: E402
    LocotrackRuntimeSettings,
    run_locotrack_offline,
)
from trackleed.analysis.ostrack_backend import (  # noqa: E402
    OstrackRuntimeSettings,
    run_ostrack_offline,
)
from trackleed.analysis.tracking import (  # noqa: E402
    _load_and_prep_image,
    optical_flow_with_skip_anchor,
)


METHODS = ("locotrack", "ostrack", "lk")
DISPLAY_METHODS = ("locotrack", "ostrack", "dam4sam", "lk")
DIRECTIONS = common.DIRECTIONS
BRANCHES = common.BRANCHES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the throwaway LocoTrack–OSTrack–LK ScanE representative-family "
            "prototype."
        )
    )
    parser.add_argument(
        "--method",
        choices=("all", *METHODS),
        default="all",
        help="Run one new representative or all three.",
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
        help="Ignore completed raw caches and rerun selected inference.",
    )
    parser.add_argument(
        "--baseline-output",
        type=Path,
        default=None,
        help="Alternative frozen physics-baseline output, for example pair2.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="ScanE source directory matching the selected baseline output.",
    )
    return parser.parse_args()


def configure_output(baseline_output: Path | None) -> None:
    global BASELINE_OUTPUT, OUTPUT_DIR, RAW_DIR
    if baseline_output is not None:
        BASELINE_OUTPUT = baseline_output.expanduser().resolve()
    OUTPUT_DIR = BASELINE_OUTPUT / "representatives"
    RAW_DIR = OUTPUT_DIR / "raw"

    # Reuse the already validated common measurement and summary functions.
    common.BASELINE_OUTPUT = BASELINE_OUTPUT
    common.TRACKER_OUTPUT = OUTPUT_DIR
    common.RAW_OUTPUT = RAW_DIR


def runtime_paths() -> dict[str, Path]:
    conda_root = Path(sys.executable).resolve().parents[2]
    projects_root = REPO_ROOT.parent
    paths = {
        "locotrack_python": (
            conda_root / "envs" / "locotrack-pytorch" / "python.exe"
        ),
        "locotrack_repo": projects_root / "locotrack",
        "locotrack_checkpoint": (
            REPO_ROOT / "tracking_models" / "locotrack" / "locotrack_base.ckpt"
        ),
        "ostrack_python": conda_root / "envs" / "ostrack" / "python.exe",
        "ostrack_repo": projects_root / "ostrack",
        "ostrack_checkpoint": (
            REPO_ROOT
            / "tracking_models"
            / "ostrack"
            / "vitb_256_mae_ce_32x4_ep300"
            / "OSTrack_ep0300.pth.tar"
        ),
        "dam4sam_checkpoint": (
            REPO_ROOT
            / "tracking_models"
            / "dam4sam"
            / "sam2.1_hiera_base_plus.pt"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing representative-family runtime paths:\n" + "\n".join(missing)
        )
    return paths


def assert_runtime() -> None:
    if os.name != "nt":
        raise RuntimeError(
            "Run this prototype with the Windows netcat Python environment; "
            "the dedicated tracker environments are Windows Conda envs."
        )
    if not (BASELINE_OUTPUT / "summary.json").exists():
        raise RuntimeError("Frozen physics-baseline summary.json is missing.")
    if not (BASELINE_OUTPUT / "trackers" / "dam4sam_summary.json").exists():
        raise RuntimeError(
            "Run the controlled SAM2-family prototype first; the frozen "
            "DAM4SAM result is required as the mask representative."
        )


def raw_path(method: str, direction: str, branch: str) -> Path:
    return RAW_DIR / f"{method}_{direction}_{branch}.npz"


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def standard_payload(
    *,
    method: str,
    tracks_xy: np.ndarray,
    visible_mask: np.ndarray,
    proposal_areas: np.ndarray | None,
    proposal_boxes_xyxy: np.ndarray | None,
    proposal_scores: np.ndarray | None,
    inference_resolution: tuple[int, int],
    original_resolution: tuple[int, int],
    checkpoint_name: str,
    config_name: str,
    representative_point_mode: str,
    extra: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    tracks = np.asarray(tracks_xy, dtype=np.float32)
    visible = np.asarray(visible_mask, dtype=bool)
    frame_count = int(tracks.shape[0])
    payload = {
        "tracks_xy": tracks,
        "visible_mask": visible,
        # These names match the common prototype adapter. For point and LK
        # proposals, area and bbox remain undefined rather than synthesized.
        "mask_areas": (
            np.asarray(proposal_areas, dtype=np.float32)
            if proposal_areas is not None
            else np.full(frame_count, np.nan, dtype=np.float32)
        ),
        "mask_bboxes_xyxy": (
            np.asarray(proposal_boxes_xyxy, dtype=np.float32)
            if proposal_boxes_xyxy is not None
            else np.full((frame_count, 4), np.nan, dtype=np.float32)
        ),
        "mask_scores": (
            np.asarray(proposal_scores, dtype=np.float32)
            if proposal_scores is not None
            else np.full(frame_count, np.nan, dtype=np.float32)
        ),
        "mask_component_counts": np.zeros(frame_count, dtype=np.int32),
        "inference_resolution": np.asarray(
            inference_resolution, dtype=np.int32
        ),
        "original_resolution": np.asarray(original_resolution, dtype=np.int32),
        "checkpoint_name": np.asarray(checkpoint_name),
        "config_name": np.asarray(config_name),
        "representative_point_mode": np.asarray(representative_point_mode),
        "tracker": np.asarray(method),
    }
    if extra:
        payload.update(extra)
    return payload


def run_one(
    *,
    method: str,
    direction: str,
    branch: str,
    seed: dict[str, float],
    frames: np.ndarray,
    frame_paths: list[Path],
    paths: dict[str, Path],
    force: bool,
) -> Path:
    cache = raw_path(method, direction, branch)
    if cache.exists() and not force:
        print(f"CACHE {method} {direction} {branch}: {cache}", flush=True)
        return cache

    query = np.asarray(
        [[0.0, seed["center_y"], seed["center_x"]]], dtype=np.float32
    )
    box = common.seed_box(seed)[None, :]
    print(
        f"RUN {method} {direction} {branch}: frames={frames.shape[0]}, "
        f"seed=({seed['center_x']:.2f},{seed['center_y']:.2f})",
        flush=True,
    )

    if method == "locotrack":
        result = run_locotrack_offline(
            frames,
            query,
            LocotrackRuntimeSettings(
                repo_path=paths["locotrack_repo"],
                checkpoint_path=paths["locotrack_checkpoint"],
                python_executable=paths["locotrack_python"],
                backend_mode="subprocess",
                model_size="base",
                device="cuda",
                inference_resolution=256,
                query_chunk_size=1,
                # Preserve raw logits and apply the official combined rule below.
                use_visibility_gating=False,
                use_uncertainty_gating=False,
                subprocess_timeout_sec=1800.0,
            ),
        )
        occlusion = np.asarray(result.occlusion_logits[0], dtype=np.float32)
        expected_dist = np.asarray(
            result.expected_dist_logits[0], dtype=np.float32
        )
        visible_probability = (1.0 - sigmoid(occlusion)) * (
            1.0 - sigmoid(expected_dist)
        )
        payload = standard_payload(
            method=method,
            tracks_xy=result.tracks_xy[0],
            visible_mask=visible_probability > 0.5,
            proposal_areas=None,
            proposal_boxes_xyxy=None,
            proposal_scores=visible_probability,
            inference_resolution=result.inference_resolution,
            original_resolution=result.original_resolution,
            checkpoint_name=paths["locotrack_checkpoint"].name,
            config_name="locotrack_base_offline",
            representative_point_mode="native_point",
            extra={
                "occlusion_logits": occlusion,
                "expected_dist_logits": expected_dist,
            },
        )
    elif method == "ostrack":
        result = run_ostrack_offline(
            frames,
            query,
            OstrackRuntimeSettings(
                checkpoint_path=paths["ostrack_checkpoint"],
                repo_path=paths["ostrack_repo"],
                python_executable=paths["ostrack_python"],
                backend_mode="subprocess",
                model_variant="vitb_256_mae_ce_32x4_ep300",
                device="cuda",
                representative_point_mode="bbox_center",
                inference_resolution=(256, 256),
                initial_bbox_padding_px=0.0,
                subprocess_timeout_sec=1800.0,
            ),
            query_boxes_xyxy=box,
        )
        boxes = np.asarray(result.tracked_bboxes_xyxy[0], dtype=np.float32)
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
            0.0, boxes[:, 3] - boxes[:, 1]
        )
        payload = standard_payload(
            method=method,
            tracks_xy=result.tracks_xy[0],
            visible_mask=result.visible_mask[0],
            proposal_areas=areas,
            proposal_boxes_xyxy=boxes,
            proposal_scores=result.bbox_scores[0],
            inference_resolution=result.inference_resolution,
            original_resolution=result.original_resolution,
            checkpoint_name=result.checkpoint_name,
            config_name=result.model_variant,
            representative_point_mode=result.representative_point_mode,
        )
    elif method == "lk":
        prepared = [_load_and_prep_image(path) for path in frame_paths]
        if any(frame is None for frame in prepared):
            raise RuntimeError("LK preprocessing failed for at least one frame.")
        positions = optical_flow_with_skip_anchor(
            frames=prepared,
            points0=[(float(seed["center_x"]), float(seed["center_y"]))],
            lk_params={
                "winSize": (41, 41),
                "maxLevel": 4,
                "criteria": (
                    3,  # cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
                    50,
                    0.01,
                ),
                "minEigThreshold": 1e-5,
            },
            err_threshold=30.0,
            fb_threshold=8.0,
            use_forward_backward=True,
            max_skip_frames_per_point=0,
            oob_behavior="skip",
            use_ncc_check=False,
            use_centroid_refinement=False,
            roi_size=baseline.ROI_SIZE,
        )
        if positions is None:
            raise RuntimeError("LK prototype was cancelled unexpectedly.")
        tracks = np.full((len(prepared), 2), np.nan, dtype=np.float32)
        visible = np.zeros(len(prepared), dtype=bool)
        for frame_index, x, y in positions[0]:
            tracks[int(frame_index)] = (float(x), float(y))
            visible[int(frame_index)] = True
        resolution = (int(frames.shape[1]), int(frames.shape[2]))
        payload = standard_payload(
            method=method,
            tracks_xy=tracks,
            visible_mask=visible,
            proposal_areas=None,
            proposal_boxes_xyxy=None,
            proposal_scores=visible.astype(np.float32),
            inference_resolution=resolution,
            original_resolution=resolution,
            checkpoint_name="none",
            config_name="pyramidal_lk_skip_anchor",
            representative_point_mode="native_point",
        )
    else:
        raise ValueError(method)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    print(f"SAVED {method} {direction} {branch}: {cache}", flush=True)
    if "result" in locals():
        del result
    gc.collect()
    return cache


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_comparison(paths: dict[str, Path]) -> None:
    new_paths = {
        method: OUTPUT_DIR / f"{method}_summary.json" for method in METHODS
    }
    if not all(path.exists() for path in new_paths.values()):
        return
    dam_path = BASELINE_OUTPUT / "trackers" / "dam4sam_summary.json"
    methods = {
        "locotrack": load_summary(new_paths["locotrack"]),
        "ostrack": load_summary(new_paths["ostrack"]),
        "dam4sam": load_summary(dam_path),
        "lk": load_summary(new_paths["lk"]),
    }
    controlled_summary_path = (
        BASELINE_OUTPUT / "trackers" / "comparison_summary.json"
    )
    dam4sam_hash = (
        load_summary(controlled_summary_path)["frozen_contract"][
            "dam4sam_checkpoint_sha256"
        ]
        if controlled_summary_path.exists()
        else sha256_file(paths["dam4sam_checkpoint"])
    )
    comparison = {
        "scope": (
            "Two automatically frozen spots forming one ScanE pair; "
            "annotation-free representative-family prototype, not absolute accuracy."
        ),
        "frozen_contract": {
            "reference_energy_ev": baseline.REFERENCE_ENERGY_EV,
            "common_measurement_roi_px": baseline.ROI_SIZE,
            "common_estimator": "TrackLEED 2D elliptical Gaussian fit",
            "directions": [
                "187.9 to 220.2 eV",
                "187.9 to 45.0 eV with reversed frame order",
            ],
            "representatives": {
                "point": "LocoTrack Base offline, official combined visibility",
                "bbox": "OSTrack ViT-B/256 MAE-CE, bbox center",
                "mask": "DAM4SAM sam21pp-B from controlled SAM2-family run",
                "optical_flow": "pyramidal LK skip-anchor",
            },
            "prompt_by_family": {
                "locotrack": "Gaussian-refined seed point",
                "ostrack": "48x48 px bbox around the same seed",
                "dam4sam": "48x48 px bbox around the same seed",
                "lk": "Gaussian-refined seed point",
            },
            "checkpoint_sha256": {
                "locotrack": sha256_file(paths["locotrack_checkpoint"]),
                "ostrack": sha256_file(paths["ostrack_checkpoint"]),
                "dam4sam": dam4sam_hash,
            },
        },
        "methods": methods,
        "prototype_readout": (
            "The representative-family panel compares proposal robustness after "
            "family-native point or bbox initialization and before one common "
            "Gaussian measurement. Coverage, gap recovery and physically "
            "inconsistent accepted fits must be interpreted jointly."
        ),
    }
    (OUTPUT_DIR / "comparison_summary.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def plot_comparison() -> None:
    summary_path = OUTPUT_DIR / "comparison_summary.json"
    if not summary_path.exists():
        return
    comparison = load_summary(summary_path)
    methods = comparison["methods"]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.3))

    for method in DISPLAY_METHODS:
        pair_path = (
            BASELINE_OUTPUT / "trackers" / "dam4sam_pair_measurements.csv"
            if method == "dam4sam"
            else OUTPUT_DIR / f"{method}_pair_measurements.csv"
        )
        if not pair_path.exists():
            continue
        import csv

        with pair_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        axes[0].plot(
            [float(row["energy_ev"]) for row in rows],
            [float(row["distance_residual_px"]) for row in rows],
            marker=".",
            label=method,
        )

    x = np.arange(len(DISPLAY_METHODS), dtype=float)
    coverage = [
        100.0 * float(methods[name]["fit_coverage_on_baseline_visible"])
        for name in DISPLAY_METHODS
    ]
    recovery = [
        100.0 * float(methods[name]["post_gap_recovery"]["rate"])
        for name in DISPLAY_METHODS
    ]
    inconsistent = [
        int(methods[name]["geometry_inconsistent_accepted_fits"])
        for name in DISPLAY_METHODS
    ]
    axes[1].bar(x - 0.18, coverage, width=0.36, label="fit coverage")
    axes[1].bar(x + 0.18, recovery, width=0.36, label="gap recovery")
    axes[1].set_xticks(x, DISPLAY_METHODS, rotation=15)
    axes[1].set_ylim(0.0, 105.0)
    axes[1].set_ylabel("Rate [%]")
    axes[1].legend(fontsize=8)
    axes[2].bar(x, inconsistent)
    axes[2].set_xticks(x, DISPLAY_METHODS, rotation=15)
    axes[2].set_ylabel("Accepted fits [count]")

    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_xlabel("Energy [eV]")
    axes[0].set_ylabel("Pair-distance residual [px]")
    axes[0].set_title("Common Gaussian measurement")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Visible-regime outcomes")
    axes[2].set_title("Geometry-inconsistent fits, all regimes")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "representative_family_comparison.png", dpi=180)
    plt.close(fig)


def write_cross_pair_artifacts() -> None:
    primary_path = (
        HERE / "output" / "representatives" / "comparison_summary.json"
    )
    replicate_path = (
        HERE
        / "output"
        / "pair2"
        / "representatives"
        / "comparison_summary.json"
    )
    if not primary_path.exists() or not replicate_path.exists():
        return

    cases = {
        "pair1": load_summary(primary_path),
        "pair2": load_summary(replicate_path),
    }
    aggregates: dict[str, dict[str, Any]] = {}
    for method in DISPLAY_METHODS:
        first = cases["pair1"]["methods"][method]
        second = cases["pair2"]["methods"][method]
        visible_total = sum(
            int(case["physics_baseline_visible_opportunities"])
            for case in (first, second)
        )
        consistent_total = sum(
            int(case["branch_consistent_measurements"])
            for case in (first, second)
        )
        gap_total = sum(
            int(case["post_gap_recovery"]["evaluated_gap_count"])
            for case in (first, second)
        )
        recovered_total = sum(
            int(case["post_gap_recovery"]["recovered_gap_count"])
            for case in (first, second)
        )
        aggregates[method] = {
            "branch_consistent_measurements": consistent_total,
            "physics_baseline_visible_opportunities": visible_total,
            "fit_coverage": consistent_total / visible_total,
            "post_gap_recovered": recovered_total,
            "post_gap_evaluated": gap_total,
            "post_gap_recovery_rate": recovered_total / gap_total,
            "geometry_inconsistent_accepted_fits": sum(
                int(case["geometry_inconsistent_accepted_fits"])
                for case in (first, second)
            ),
            "pair_residual_p95_px_by_pair": [
                float(first["absolute_pair_distance_residual_px"]["p95"]),
                float(second["absolute_pair_distance_residual_px"]["p95"]),
            ],
        }

    output_root = HERE / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    cross_summary = {
        "scope": (
            "Cross-pair view of the two disjoint frozen ScanE pairs. "
            "The aggregate is descriptive and is not an overall score."
        ),
        "cases": {
            name: case["methods"] for name, case in cases.items()
        },
        "descriptive_aggregates": aggregates,
    }
    (output_root / "representative_cross_pair_summary.json").write_text(
        json.dumps(cross_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    x = np.arange(len(DISPLAY_METHODS), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    for case_index, (case_name, case) in enumerate(cases.items()):
        offset = (-0.5 + case_index) * width
        values = [case["methods"][name] for name in DISPLAY_METHODS]
        axes[0, 0].bar(
            x + offset,
            [
                100.0 * float(value["fit_coverage_on_baseline_visible"])
                for value in values
            ],
            width=width,
            label=case_name,
        )
        axes[0, 1].bar(
            x + offset,
            [
                100.0 * float(value["post_gap_recovery"]["rate"])
                for value in values
            ],
            width=width,
            label=case_name,
        )
        axes[1, 0].bar(
            x + offset,
            [
                int(value["geometry_inconsistent_accepted_fits"])
                for value in values
            ],
            width=width,
            label=case_name,
        )
        axes[1, 1].bar(
            x + offset,
            [
                float(value["absolute_pair_distance_residual_px"]["p95"])
                for value in values
            ],
            width=width,
            label=case_name,
        )

    titles = (
        "Fit coverage on measurable frames",
        "Post-gap recovery",
        "Geometry-inconsistent accepted fits",
        "Conditional pair-residual P95",
    )
    ylabels = ("Rate [%]", "Rate [%]", "Count", "Pixels")
    for axis, title, ylabel in zip(axes.flat, titles, ylabels):
        axis.set_xticks(x, DISPLAY_METHODS, rotation=15)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
    axes[0, 0].set_ylim(0.0, 105.0)
    axes[0, 1].set_ylim(0.0, 105.0)
    fig.suptitle(
        "Two disjoint ScanE pairs — robustness outcomes, not absolute accuracy",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(
        output_root / "representative_cross_pair_comparison.png", dpi=180
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    configure_output(args.baseline_output)
    if args.data_dir is not None:
        baseline.DATA_DIR = args.data_dir.expanduser().resolve()
    assert_runtime()
    paths = runtime_paths()
    frames, seeds, reference_index, baseline_rows = common.load_baseline_contract()
    frame_stack = common.load_frame_stack(frames)
    methods = METHODS if args.method == "all" else (args.method,)
    directions = common.selected_values(args.direction, DIRECTIONS)

    for method in methods:
        for direction in directions:
            indices = common.sequence_indices(
                len(frames), reference_index, direction
            )
            sequence = frame_stack[indices]
            sequence_paths = [frames[index].path for index in indices]
            for seed in seeds:
                run_one(
                    method=method,
                    direction=direction,
                    branch=str(seed["branch"]),
                    seed=seed,
                    frames=sequence,
                    frame_paths=sequence_paths,
                    paths=paths,
                    force=bool(args.force),
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for method in methods:
        if all(
            raw_path(method, direction, branch).exists()
            for direction in DIRECTIONS
            for branch in BRANCHES
        ):
            result = common.analyze_tracker(
                method,
                frame_stack,
                frames,
                seeds,
                reference_index,
                baseline_rows,
            )
            print(json.dumps(result, indent=2), flush=True)

    write_comparison(paths)
    plot_comparison()
    # Cross-pair figures belong to the original two-pair tracer study. A run
    # rooted elsewhere must not silently rewrite those original artifacts.
    try:
        BASELINE_OUTPUT.relative_to(HERE / "output")
    except ValueError:
        pass
    else:
        write_cross_pair_artifacts()
    print(f"Results: {OUTPUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
