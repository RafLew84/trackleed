#!/usr/bin/env python3
"""Run EasyLEED on the frozen 15-frame, two-pair ScanE segment."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import easyleed
import numpy as np
from easyleed import config
from easyleed.base import Tracker
from easyleed.io import AllImageLoader, PILImageLoader


ARTIFACT_DIR = Path(__file__).resolve().parent
FRAME_DIR = ARTIFACT_DIR / "frames"
MANIFEST_PATH = ARTIFACT_DIR / "input_manifest.csv"
OUTPUT_DIR = ARTIFACT_DIR / "easyleed"

REFERENCE_ENERGY_EV = 187.9
APPARATUS_CENTER = (491.00030779190064, 472.01122086402023)
TRACKING_RADIUS_PX = 12.0
SEEDS = {
    "P1_B1": (272.7697717354655, 515.74845510848),
    "P1_B2": (714.4012068520402, 464.38229982316034),
    "P2_B1": (578.7745601084399, 278.99980724180637),
    "P2_B2": (403.01171432256814, 693.4746658283458),
}


def read_manifest() -> list[dict[str, object]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for row in source_rows:
        rows.append(
            {
                "segment_index": int(row["segment_index"]),
                "frame": int(row["source_frame"]),
                "energy_ev": float(row["energy_ev"]),
                "path": ARTIFACT_DIR / row["analysis_file"],
            }
        )
    if [row["frame"] for row in rows] != list(range(54, 69)):
        raise RuntimeError("Manifest does not contain frames 54-68 in order")
    if [row["energy_ev"] for row in rows][-2:] != [220.2, 220.2]:
        raise RuntimeError("Manifest lost the duplicate terminal energy")
    return rows


def physics_prediction(
    seed_xy: tuple[float, float], energy_ev: float
) -> tuple[float, float]:
    center_x, center_y = APPARATUS_CENTER
    scale = math.sqrt(REFERENCE_ENERGY_EV / energy_ev)
    return (
        center_x + (seed_xy[0] - center_x) * scale,
        center_y + (seed_xy[1] - center_y) * scale,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = read_manifest()
    frame_paths = [Path(record["path"]) for record in records]

    # This deliberately exercises and records EasyLEED's standard contract.
    # ImageLoader stores path by energy, so the two 220.2 eV frames collapse.
    standard_loader = AllImageLoader(
        [str(path) for path in frame_paths],
        config.IO_energyRegex,
    )
    standard_loader_unique_count = len(standard_loader.energies)
    if standard_loader_unique_count != 14:
        raise RuntimeError(
            "Expected EasyLEED's energy-keyed loader to retain 14 unique "
            f"energies, got {standard_loader_unique_count}"
        )

    # Preserve frame identity while keeping EasyLEED's native image reader,
    # Tracker, real energy values and zero-dt terminal Kalman update.
    images = [
        PILImageLoader.get_image(str(record["path"])) for record in records
    ]
    trackers = {
        branch: Tracker(
            seed_x,
            seed_y,
            TRACKING_RADIUS_PX,
            REFERENCE_ENERGY_EV,
            APPARATUS_CENTER[0],
            APPARATUS_CENTER[1],
            input_precision=config.Tracking_inputPrecision,
            window_scaling=config.Tracking_windowScalingOn,
        )
        for branch, (seed_x, seed_y) in SEEDS.items()
    }

    rows: list[dict[str, object]] = []
    for record, image in zip(records, images, strict=True):
        energy_ev = float(record["energy_ev"])
        for branch, tracker in trackers.items():
            x, y, intensity, _, radius = tracker.feed_image((image, energy_ev))
            expected_x, expected_y = physics_prediction(
                SEEDS[branch], energy_ev
            )
            rows.append(
                {
                    "segment_index": int(record["segment_index"]),
                    "frame": int(record["frame"]),
                    "energy_ev": energy_ev,
                    "branch": branch,
                    "native_x": float(x),
                    "native_y": float(y),
                    "native_intensity": float(intensity),
                    "native_radius": float(radius),
                    "physics_x": expected_x,
                    "physics_y": expected_y,
                    "physics_branch_error_px": math.hypot(
                        float(x) - expected_x,
                        float(y) - expected_y,
                    ),
                }
            )

    csv_path = OUTPUT_DIR / "easyleed_native.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    branch_errors = {
        branch: [
            float(row["physics_branch_error_px"])
            for row in rows
            if row["branch"] == branch
        ]
        for branch in SEEDS
    }
    summary = {
        "status": "completed",
        "scope": (
            "Native EasyLEED proposals on the frozen 15-frame, two-pair "
            "high-energy contract; not an accuracy ranking."
        ),
        "easyleed_version": easyleed.__version__,
        "numpy_version": np.__version__,
        "frame_count": len(records),
        "unique_energy_count": len(
            {float(record["energy_ev"]) for record in records}
        ),
        "energy_range_ev": [
            float(records[0]["energy_ev"]),
            float(records[-1]["energy_ev"]),
        ],
        "reference_energy_ev": REFERENCE_ENERGY_EV,
        "branch_count": len(SEEDS),
        "row_count": len(rows),
        "duplicate_energy_contract": {
            "energy_ev": 220.2,
            "frames": [67, 68],
            "standard_loader_input_frames": len(records),
            "standard_loader_retained_unique_energies": (
                standard_loader_unique_count
            ),
            "adapter": (
                "Preserve manifest order and frame_id; read each image with "
                "EasyLEED PILImageLoader; feed native Tracker twice at "
                "220.2 eV, with dt=0 for the second frame."
            ),
        },
        "settings": {
            "tracking_radius_px": TRACKING_RADIUS_PX,
            "input_precision_px": config.Tracking_inputPrecision,
            "window_scaling": config.Tracking_windowScalingOn,
            "gaussian_r_squared_min": config.Tracking_minRsq,
            "validation_gamma": config.Tracking_gamma,
        },
        "physics_branch_error_px": {
            branch: {
                "median": float(np.median(values)),
                "maximum": float(np.max(values)),
            }
            for branch, values in branch_errors.items()
        },
        "outputs": [csv_path.name],
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
