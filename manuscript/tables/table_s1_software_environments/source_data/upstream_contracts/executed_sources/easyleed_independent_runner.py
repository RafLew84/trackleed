#!/usr/bin/env python3
"""Run native EasyLEED proposals on one frozen independent ScanE pair."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import easyleed
import numpy as np
from easyleed import config
from easyleed.base import Tracker
from easyleed.io import AllImageLoader, PILImageLoader


ENERGY_RE = re.compile(r"_E(\d{4})_")
FRAME_RE = re.compile(r"_(\d{4})_T")
TRACKING_RADIUS_PX = 12.0
DIRECTIONS = ("high", "low")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EasyLEED on a frozen two-branch ScanE case."
    )
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    return parser.parse_args()


def load_records(data_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in data_dir.glob("*.tiff"):
        energy_match = ENERGY_RE.search(path.name)
        frame_match = FRAME_RE.search(path.name)
        if energy_match is None or frame_match is None:
            continue
        records.append(
            {
                "frame": int(frame_match.group(1)),
                "energy_ev": int(energy_match.group(1)) / 10.0,
                "path": path.resolve(),
            }
        )
    if not records:
        raise RuntimeError(f"No ScanE TIFF files found in {data_dir}")
    return sorted(records, key=lambda row: int(row["frame"]))


def sequence_indices(
    frame_count: int, reference_index: int, direction: str
) -> list[int]:
    if direction == "high":
        return list(range(reference_index, frame_count))
    if direction == "low":
        return list(range(reference_index, -1, -1))
    raise ValueError(direction)


def physics_prediction(
    seed_xy: tuple[float, float],
    center_xy: tuple[float, float],
    reference_energy_ev: float,
    energy_ev: float,
) -> tuple[float, float]:
    scale = math.sqrt(reference_energy_ev / energy_ev)
    return (
        center_xy[0] + (seed_xy[0] - center_xy[0]) * scale,
        center_xy[1] + (seed_xy[1] - center_xy[1]) * scale,
    )


def main() -> None:
    args = parse_args()
    baseline_output = args.baseline_output.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = baseline_output / "easyleed"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = json.loads(
        (baseline_output / "summary.json").read_text(encoding="utf-8")
    )
    records = load_records(data_dir)
    if len(records) != int(baseline_summary["frame_count"]):
        raise RuntimeError(
            f"Frame count mismatch: {len(records)} != "
            f"{baseline_summary['frame_count']}"
        )

    reference_frame = int(baseline_summary["reference_frame"])
    reference_energy_ev = float(baseline_summary["reference_energy_ev"])
    reference_index = next(
        index
        for index, record in enumerate(records)
        if int(record["frame"]) == reference_frame
    )
    geometry = baseline_summary["geometry"]
    center_xy = (
        float(geometry["center_x"]),
        float(geometry["center_y"]),
    )
    seeds = {
        str(seed["branch"]): (
            float(seed["center_x"]),
            float(seed["center_y"]),
        )
        for seed in baseline_summary["frozen_seeds"]
    }

    # Record both failure modes of EasyLEED's standard energy-keyed loader.
    # The original acquisition names end in C0000, which the default regex
    # interprets as energy zero. Parseable proxy names expose the second issue:
    # equal terminal energies still collapse to one dictionary entry.
    original_loader = AllImageLoader(
        [str(record["path"]) for record in records],
        config.IO_energyRegex,
    )
    proxy_names = [
        f"frame_{int(record['frame']):04d}_{float(record['energy_ev']):.1f}.tiff"
        for record in records
    ]
    proxy_loader = AllImageLoader(proxy_names, config.IO_energyRegex)

    # Preserve frame identity while retaining EasyLEED's native image reader,
    # Tracker, Gaussian observation and Kalman dynamics.
    images = [
        PILImageLoader.get_image(str(record["path"])) for record in records
    ]
    rows: list[dict[str, object]] = []
    for direction in DIRECTIONS:
        indices = sequence_indices(len(records), reference_index, direction)
        for branch, seed_xy in seeds.items():
            tracker = Tracker(
                seed_xy[0],
                seed_xy[1],
                TRACKING_RADIUS_PX,
                reference_energy_ev,
                center_xy[0],
                center_xy[1],
                input_precision=config.Tracking_inputPrecision,
                window_scaling=config.Tracking_windowScalingOn,
            )
            failure: str | None = None
            for sequence_index, record_index in enumerate(indices):
                record = records[record_index]
                energy_ev = float(record["energy_ev"])
                if failure is None:
                    try:
                        x, y, intensity, _, radius = tracker.feed_image(
                            (images[record_index], energy_ev)
                        )
                        native_status = "proposal"
                        native_error = ""
                    except Exception as exc:  # native failure is an outcome
                        failure = f"{type(exc).__name__}: {exc}"
                        x = y = intensity = radius = float("nan")
                        native_status = "tracker_exception"
                        native_error = failure
                else:
                    x = y = intensity = radius = float("nan")
                    native_status = "unavailable_after_tracker_exception"
                    native_error = failure
                expected_x, expected_y = physics_prediction(
                    seed_xy,
                    center_xy,
                    reference_energy_ev,
                    energy_ev,
                )
                rows.append(
                    {
                        "direction": direction,
                        "sequence_index": sequence_index,
                        "frame": int(record["frame"]),
                        "energy_ev": energy_ev,
                        "branch": branch,
                        "native_x": float(x),
                        "native_y": float(y),
                        "native_intensity": float(intensity),
                        "native_radius": float(radius),
                        "native_status": native_status,
                        "native_error": native_error,
                        "physics_x": expected_x,
                        "physics_y": expected_y,
                        "physics_branch_error_px": (
                            math.hypot(
                                float(x) - expected_x,
                                float(y) - expected_y,
                            )
                            if np.isfinite(x) and np.isfinite(y)
                            else float("nan")
                        ),
                    }
                )

    csv_path = output_dir / "easyleed_native.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    duplicate_groups: dict[float, list[int]] = {}
    for record in records:
        duplicate_groups.setdefault(float(record["energy_ev"]), []).append(
            int(record["frame"])
        )
    duplicate_groups = {
        energy: frames
        for energy, frames in duplicate_groups.items()
        if len(frames) > 1
    }
    summary = {
        "status": "completed",
        "scope": (
            "Native EasyLEED proposals for one frozen independent ScanE pair; "
            "final acceptance uses the shared TrackLEED Gaussian contract."
        ),
        "easyleed_version": easyleed.__version__,
        "numpy_version": np.__version__,
        "frame_count": len(records),
        "unique_energy_count": len(
            {float(record["energy_ev"]) for record in records}
        ),
        "reference_frame": reference_frame,
        "reference_energy_ev": reference_energy_ev,
        "branch_count": len(seeds),
        "direction_count": len(DIRECTIONS),
        "native_row_count": len(rows),
        "native_tracker_exceptions": sum(
            row["native_status"] == "tracker_exception" for row in rows
        ),
        "unavailable_rows_after_exception": sum(
            row["native_status"] == "unavailable_after_tracker_exception"
            for row in rows
        ),
        "duplicate_energy_contract": {
            "groups": [
                {"energy_ev": energy, "frames": frames}
                for energy, frames in sorted(duplicate_groups.items())
            ],
            "standard_loader_input_frames": len(records),
            "original_filename_retained_energies": len(
                original_loader.energies
            ),
            "parseable_proxy_retained_unique_energies": len(
                proxy_loader.energies
            ),
            "adapter_retained_frames": len(records),
            "adapter": (
                "Preserve manifest order and frame_id; read every image with "
                "EasyLEED PILImageLoader; feed a fresh native Tracker in each "
                "direction, including both equal-energy terminal frames."
            ),
        },
        "settings": {
            "tracking_radius_px": TRACKING_RADIUS_PX,
            "input_precision_px": config.Tracking_inputPrecision,
            "window_scaling": config.Tracking_windowScalingOn,
            "gaussian_r_squared_min": config.Tracking_minRsq,
            "validation_gamma": config.Tracking_gamma,
        },
        "outputs": [csv_path.name],
    }
    (output_dir / "native_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
