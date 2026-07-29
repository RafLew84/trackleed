#!/usr/bin/env python3
"""Apply the shared Gaussian and ScanE metric contract to EasyLEED proposals."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ARTIFACT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ARTIFACT_ROOT.parents[3]
TRACER_DIR = (
    REPO_ROOT
    / ".scratch"
    / "trackleed-physics-method-paper"
    / "prototypes"
    / "scane_tracer"
)
sys.path.insert(0, str(TRACER_DIR))
sys.path.insert(0, str(REPO_ROOT))
import run as baseline  # noqa: E402
import run_trackers as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze native EasyLEED proposals with the common contract."
    )
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_output = args.baseline_output.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = baseline_output / "easyleed"
    native_path = output_dir / "easyleed_native.csv"
    native_summary_path = output_dir / "native_summary.json"
    if not native_path.exists() or not native_summary_path.exists():
        raise FileNotFoundError("Run run_easyleed.py in the EasyLEED env first.")

    baseline.DATA_DIR = data_dir
    common.BASELINE_OUTPUT = baseline_output
    frames, seeds, reference_index, baseline_rows = (
        common.load_baseline_contract()
    )
    frame_stack = common.load_frame_stack(frames)
    baseline_summary = json.loads(
        (baseline_output / "summary.json").read_text(encoding="utf-8")
    )
    geometry_data = baseline_summary["geometry"]
    geometry = baseline.Geometry(
        center_x=float(geometry_data["center_x"]),
        center_y=float(geometry_data["center_y"]),
        aperture_radius=float(geometry_data["aperture_radius_px"]),
        screen_radius=float(geometry_data["screen_radius_px"]),
    )

    with native_path.open(newline="", encoding="utf-8") as handle:
        native_rows = list(csv.DictReader(handle))
    frame_index = {frame.frame: index for index, frame in enumerate(frames)}
    branch_index = {
        str(seed["branch"]): index for index, seed in enumerate(seeds)
    }
    tracks_xy = np.full((2, len(frames), 2), np.nan, dtype=np.float32)
    visible_mask = np.zeros((2, len(frames)), dtype=bool)
    mask_areas = np.zeros((2, len(frames)), dtype=np.float32)
    mask_bboxes = np.full((2, len(frames), 4), np.nan, dtype=np.float32)
    mask_scores = np.full((2, len(frames)), np.nan, dtype=np.float32)
    component_counts = np.zeros((2, len(frames)), dtype=np.int32)
    reference_frame = frames[reference_index].frame

    for row in native_rows:
        frame = int(row["frame"])
        branch = str(row["branch"])
        if row["direction"] == "low" and frame == reference_frame:
            continue
        b_index = branch_index[branch]
        f_index = frame_index[frame]
        proposal = np.asarray(
            [float(row["native_x"]), float(row["native_y"])],
            dtype=np.float32,
        )
        tracks_xy[b_index, f_index] = proposal
        visible_mask[b_index, f_index] = bool(np.all(np.isfinite(proposal)))

    merged = {
        "tracks_xy": tracks_xy,
        "visible_mask": visible_mask,
        "mask_areas": mask_areas,
        "mask_bboxes_xyxy": mask_bboxes,
        "mask_scores": mask_scores,
        "mask_component_counts": component_counts,
    }
    rows = common.measure_tracker(
        "easyleed",
        merged,
        frame_stack,
        frames,
        seeds,
        geometry,
        baseline_rows,
    )
    pair_rows = common.build_pair_rows(rows, seeds)
    gap_rows = common.evaluate_gap_recovery(
        rows,
        frames,
        baseline_rows,
        reference_index,
    )
    common.write_csv(output_dir / "easyleed_measurements.csv", rows)
    common.write_csv(
        output_dir / "easyleed_pair_measurements.csv", pair_rows
    )
    common.write_csv(output_dir / "easyleed_gap_recovery.csv", gap_rows)

    summary = common.tracker_summary(
        "easyleed", rows, pair_rows, gap_rows
    )
    summary["native_contract"] = json.loads(
        native_summary_path.read_text(encoding="utf-8")
    )
    summary["merged_proposal_count"] = int(visible_mask.sum())
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
