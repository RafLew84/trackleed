#!/usr/bin/env python3
"""Freeze source-linked geometry-inconsistent examples for Figure S5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path

from PIL import Image


FIGURE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = FIGURE_ROOT.parents[2]
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
UPSTREAM_ROOT = (
    REPOSITORY_ROOT / ".scratch/trackleed-physics-method-paper/artifacts/"
    "independent-scane-validation"
)
AGGREGATE_PATH = UPSTREAM_ROOT / "aggregate_method_results.csv"
BRANCH_GATE_PX = 18.0
METHOD_ORDER = ("locotrack", "ostrack", "dam4sam", "lk", "easyleed")
DISPLAY_NAMES = {
    "locotrack": "LocoTrack",
    "ostrack": "OSTrack",
    "dam4sam": "DAM4SAM",
    "lk": "Lucas-Kanade",
    "easyleed": "EasyLEED",
}
CASES = (
    ("scan_a_pair_1", "26a21", "pair1", "Scan A", "pair 1", "data/26a21_ScanE"),
    ("scan_a_pair_2", "26a21", "pair2", "Scan A", "pair 2", "data/26a21_ScanE"),
    ("scan_b_pair_1", "26a21b", "pair1", "Scan B", "pair 1", "data/26a21b_ScanE"),
    ("scan_b_pair_2", "26a21b", "pair2", "Scan B", "pair 2", "data/26a21b_ScanE"),
)
SELECTION_RULE = (
    "For each method in the frozen display order, scan frozen cases in "
    "Scan A pair 1, Scan A pair 2, Scan B pair 1, Scan B pair 2 order and "
    "select the first frame/branch row that is an accepted Gaussian fit "
    "outside the 18 px branch-consistency gate."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _measurement_path(series_id: str, pair_id: str, method: str) -> Path:
    relative = (
        f"trackers/{method}_measurements.csv"
        if method == "dam4sam"
        else f"easyleed/{method}_measurements.csv"
        if method == "easyleed"
        else f"representatives/{method}_measurements.csv"
    )
    return UPSTREAM_ROOT / series_id / pair_id / relative


def _find_image(data_path: str, frame: int) -> Path:
    matches = sorted((REPOSITORY_ROOT / data_path).glob(f"*_{frame:04d}_*.tiff"))
    if len(matches) != 1:
        raise ValueError(f"Expected one source image for {data_path} frame {frame}")
    return matches[0]


def _select_examples() -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        selected: dict[str, object] | None = None
        for case_id, series_id, pair_id, series, pair, data_path in CASES:
            measurement_path = _measurement_path(series_id, pair_id, method)
            candidates = sorted(
                read_csv(measurement_path),
                key=lambda row: (int(row["frame"]), row["branch"]),
            )
            row = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate["fit_valid"] == "True"
                    and float(candidate["branch_error_px"]) > BRANCH_GATE_PX
                ),
                None,
            )
            if row is None:
                continue
            frame = int(row["frame"])
            image_path = _find_image(data_path, frame)
            coordinates = [
                float(row[field])
                for field in (
                    "predicted_x",
                    "predicted_y",
                    "native_x",
                    "native_y",
                    "center_x",
                    "center_y",
                )
            ]
            x_values = coordinates[0::2]
            y_values = coordinates[1::2]
            with Image.open(image_path) as image:
                width, height = image.size
            margin = 36
            crop_x0 = max(0, math.floor(min(x_values) - margin))
            crop_y0 = max(0, math.floor(min(y_values) - margin))
            crop_x1 = min(width, math.ceil(max(x_values) + margin))
            crop_y1 = min(height, math.ceil(max(y_values) + margin))
            selected = {
                "method": method,
                "display_method": DISPLAY_NAMES[method],
                "case_id": case_id,
                "series": series,
                "pair": pair,
                "frame": frame,
                "energy_ev": row["energy_ev"],
                "branch": row["branch"],
                "physics_baseline_status": row["physics_baseline_status"],
                "fit_valid": "true",
                "fit_reason": row["fit_reason"],
                "native_x": row["native_x"],
                "native_y": row["native_y"],
                "predicted_x": row["predicted_x"],
                "predicted_y": row["predicted_y"],
                "center_x": row["center_x"],
                "center_y": row["center_y"],
                "branch_error_px": row["branch_error_px"],
                "r_squared": row["r_squared"],
                "amplitude": row["amplitude"],
                "sigma_x": row["sigma_x"],
                "sigma_y": row["sigma_y"],
                "crop_x0": crop_x0,
                "crop_y0": crop_y0,
                "crop_x1": crop_x1,
                "crop_y1": crop_y1,
                "measurement_source_path": measurement_path.relative_to(
                    REPOSITORY_ROOT
                ).as_posix(),
                "measurement_source_sha256": sha256_file(measurement_path),
                "image_source_path": image_path.relative_to(REPOSITORY_ROOT).as_posix(),
                "image_source_sha256": sha256_file(image_path),
                "crop_path": f"crops/{method}.png",
            }
            break
        if selected is None:
            raise ValueError(f"No geometry-inconsistent example for {method}")
        examples.append(selected)
    return examples


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _selection_contract() -> dict[str, object]:
    aggregate = {row["method"]: row for row in read_csv(AGGREGATE_PATH)}
    return {
        "schema_version": "1.0",
        "branch_consistency_gate_px": BRANCH_GATE_PX,
        "selection_rule": SELECTION_RULE,
        "selection_uses_error_magnitude": False,
        "examples_are_ground_truth_identity_switches": False,
        "supports_tracker_ranking": False,
        "geometry_inconsistent_count_by_method": {
            method: int(aggregate[method]["geometry_inconsistent_accepted_fits"])
            for method in METHOD_ORDER
        },
        "accepted_fit_denominator_by_method": {
            method: int(aggregate[method]["accepted_measurements_all_frames"])
            for method in METHOD_ORDER
        },
    }


def build(output_root: Path = FIGURE_ROOT) -> None:
    output = output_root / "source_data"
    crops = output / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    examples = _select_examples()
    for row in examples:
        image_path = REPOSITORY_ROOT / str(row["image_source_path"])
        crop_path = output / str(row["crop_path"])
        with Image.open(image_path) as image:
            crop = image.crop(
                (
                    int(row["crop_x0"]),
                    int(row["crop_y0"]),
                    int(row["crop_x1"]),
                    int(row["crop_y1"]),
                )
            )
            crop.save(crop_path, format="PNG", compress_level=9, optimize=False)
        row["crop_sha256"] = sha256_file(crop_path)
    _write_csv(output / "example_records.csv", examples)
    _write_json(output / "selection_contract.json", _selection_contract())
    provenance_sources = {
        "aggregate": {
            "path": AGGREGATE_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_file(AGGREGATE_PATH),
        }
    }
    for row in examples:
        provenance_sources[f"{row['method']}:measurement"] = {
            "path": row["measurement_source_path"],
            "sha256": row["measurement_source_sha256"],
        }
        provenance_sources[f"{row['method']}:image"] = {
            "path": row["image_source_path"],
            "sha256": row["image_source_sha256"],
        }
    _write_json(
        output / "source_provenance.json",
        {"figure": "Figure S5", "schema_version": "1.0", "sources": provenance_sources},
    )


def validate_source_package(
    source_directory: Path = SOURCE_DATA_DIRECTORY,
    *,
    check_upstream: bool = True,
) -> None:
    rows = read_csv(source_directory / "example_records.csv")
    if [row["method"] for row in rows] != list(METHOD_ORDER):
        raise ValueError("Figure S5 method order changed")
    if any(
        row["fit_valid"] != "true" or float(row["branch_error_px"]) <= BRANCH_GATE_PX
        for row in rows
    ):
        raise ValueError("Figure S5 contains a non-qualifying example")
    selection = json.loads(
        (source_directory / "selection_contract.json").read_text(encoding="utf-8")
    )
    if (
        selection["selection_uses_error_magnitude"]
        or selection["supports_tracker_ranking"]
    ):
        raise ValueError("Figure S5 selection became rank-oriented")
    if selection["geometry_inconsistent_count_by_method"] != {
        "locotrack": 36,
        "ostrack": 34,
        "dam4sam": 36,
        "lk": 14,
        "easyleed": 4,
    }:
        raise ValueError("Figure S5 geometry-inconsistent counts changed")
    for row in rows:
        crop = source_directory / row["crop_path"]
        if not crop.is_file() or sha256_file(crop) != row["crop_sha256"]:
            raise ValueError(f"Figure S5 crop hash mismatch: {row['method']}")
    provenance = json.loads(
        (source_directory / "source_provenance.json").read_text(encoding="utf-8")
    )
    if check_upstream:
        for name, source in provenance["sources"].items():
            path = (REPOSITORY_ROOT / source["path"]).resolve()
            if REPOSITORY_ROOT.resolve() not in path.parents:
                raise ValueError(f"Figure S5 provenance escapes repository: {name}")
            if not path.is_file() or sha256_file(path) != source["sha256"]:
                raise ValueError(f"Figure S5 upstream hash mismatch: {name}")


def validate() -> None:
    validate_source_package()
    with tempfile.TemporaryDirectory(prefix="trackleed-figure-s5-source-") as tmp:
        rebuilt = Path(tmp) / "figure_s5_geometry_inconsistent_continuation"
        build(rebuilt)
        committed_files = sorted(
            path for path in SOURCE_DATA_DIRECTORY.rglob("*") if path.is_file()
        )
        for committed in committed_files:
            relative = committed.relative_to(SOURCE_DATA_DIRECTORY)
            candidate = rebuilt / "source_data" / relative
            if not candidate.is_file() or sha256_file(committed) != sha256_file(
                candidate
            ):
                raise ValueError(f"Non-deterministic Figure S5 source: {relative}")
    print("Figure S5 source validation passed: five deterministic examples.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.validate_only:
        build()
    validate()


if __name__ == "__main__":
    main()
