#!/usr/bin/env python3
"""Analyze tracked LEED ROIs without launching the Qt user interface.

The command reads a TrackLEED ``.lts.json`` tracking session, performs the
same rotated 2D Gaussian fit and per-frame circle-center calculation used by
the desktop analysis workflow, and writes a new, non-destructive result
directory.  An optional legacy per-spot CSV can be supplied for automatic
track matching and numerical comparison.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from skimage.color import rgb2gray
from skimage.util import img_as_float


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.peak_analyzer import (
    fit_circle_center,
    fit_spot_to_gaussian,
)
from trackleed.analysis.result_parameters import (
    GAUSSIAN_DERIVED_EXPORT_FIELDS,
)
from trackleed.io.fwhm_series_exporter import export_fwhm_series
from trackleed.io.fwhm_series_plots import (
    PALETTE,
    PlotSeries,
    _render_line_plot,
)
from trackleed.io.image_reader import load_tiff


MIN_R_SQUARED = 0.85
MIN_AMPLITUDE = 0.05
PREFERRED_RESULT_FIELDS = (
    "track_id",
    "frame_id",
    "temperature",
    "energy",
    "coverage",
    "global_center_x",
    "center_x_error",
    "global_center_y",
    "center_y_error",
    "radius",
    "radius_error",
    "volume",
    "volume_error",
    "amplitude",
    "amplitude_error",
    "sigma_x",
    "sigma_x_error",
    "sigma_y",
    "sigma_y_error",
    "theta",
    "theta_error",
    "offset",
    "offset_error",
    "fwhm_x",
    "fwhm_x_error",
    "fwhm_y",
    "fwhm_y_error",
    *GAUSSIAN_DERIVED_EXPORT_FIELDS,
    "theta_major",
    "theta_major_error",
    "asymmetry",
    "r_squared",
    "chi_squared_reduced",
)
COMPARISON_METRICS = (
    "center_distance",
    "radius_delta",
    "fwhm_effective_delta",
    "fwhm_minor_delta",
    "fwhm_major_delta",
    "r_squared_delta",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gaussian/circle analysis for a tracked .lts.json session and "
            "write a non-destructive complete export."
        )
    )
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        help="Optional earlier full_analysis_per_spot.csv to compare.",
    )
    return parser.parse_args()


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = _finite(value)
        return "" if number is None else format(number, ".10g")
    return str(value)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_image_path(session_path: Path, file_payload: Mapping) -> Path:
    local_candidate = session_path.parent / str(file_payload["filename"])
    if local_candidate.is_file():
        return local_candidate
    stored_candidate = Path(str(file_payload.get("filepath", "")))
    if stored_candidate.is_file():
        return stored_candidate
    raise FileNotFoundError(
        f"Cannot resolve image for frame {file_payload.get('id')}: "
        f"{file_payload.get('filename')}"
    )


def _fit_is_valid(fit: Mapping | None, roi_shape: tuple[int, ...]) -> bool:
    if not fit or len(roi_shape) < 2:
        return False
    height, width = roi_shape[:2]
    return (
        float(fit.get("r_squared", 0.0)) >= MIN_R_SQUARED
        and float(fit.get("amplitude", 0.0)) >= MIN_AMPLITUDE
        and -2.0 < float(fit.get("center_x", 0.0)) < width + 2.0
        and -2.0 < float(fit.get("center_y", 0.0)) < height + 2.0
        and 0.5 < float(fit.get("sigma_x", 0.0)) < width
        and 0.5 < float(fit.get("sigma_y", 0.0)) < height
    )


def _gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim >= 3:
        if image.shape[-1] >= 3:
            image = rgb2gray(image[..., :3])
        else:
            image = image[..., 0]
    return img_as_float(image)


def _apply_circle_radius(
    fit: dict[str, object],
    center: Mapping[str, object],
) -> None:
    dx = float(fit["global_center_x"]) - float(center["center_x"])
    dy = float(fit["global_center_y"]) - float(center["center_y"])
    radius = math.hypot(dx, dy)
    fit["radius"] = radius

    variance_x = (
        float(fit.get("center_x_error", 0.0)) ** 2
        + float(center.get("error_x", 0.0)) ** 2
    )
    variance_y = (
        float(fit.get("center_y_error", 0.0)) ** 2
        + float(center.get("error_y", 0.0)) ** 2
    )
    fit["radius_error"] = (
        math.sqrt(dx * dx * variance_x + dy * dy * variance_y) / radius
        if radius > 1e-6
        else 0.0
    )


def _fit_session(
    payload: dict,
    session_path: Path,
) -> tuple[
    dict[int, dict[int, dict[str, object] | None]],
    dict[int, dict[str, object]],
    dict[int, dict[str, float]],
    dict[str, int],
]:
    files_by_id = {
        int(file_payload["id"]): file_payload
        for file_payload in payload.get("data_files", [])
    }
    tracks = payload.get("tracks", [])
    results: dict[int, dict[int, dict[str, object] | None]] = {
        int(track["track_id"]): {} for track in tracks
    }
    centers: dict[int, dict[str, object]] = {}
    averages: dict[int, dict[str, float]] = {}
    counts = {
        "frames": len(files_by_id),
        "tracks": len(tracks),
        "expected_track_frames": len(files_by_id) * len(tracks),
        "tracked_rois": 0,
        "missing_tracked_rois": 0,
        "valid_fits": 0,
        "invalid_fits": 0,
    }

    for index, frame_id in enumerate(sorted(files_by_id), start=1):
        file_payload = files_by_id[frame_id]
        image = load_tiff(_resolve_image_path(session_path, file_payload))
        if image is None or image.size == 0:
            raise RuntimeError(f"Failed to load frame {frame_id}")
        frame_positions: list[dict[str, float]] = []

        for track in tracks:
            track_id = int(track["track_id"])
            spot = track.get("spots_by_image_id", {}).get(str(frame_id))
            if not isinstance(spot, dict):
                continue
            counts["tracked_rois"] += 1
            roi = spot["roi"]
            x = int(roi["x"])
            y = int(roi["y"])
            width = int(roi["w"])
            height = int(roi["h"])
            roi_image = image[y : y + height, x : x + width].copy()
            fit = fit_spot_to_gaussian(_gray_float(roi_image))
            if not _fit_is_valid(fit, roi_image.shape):
                results[track_id][frame_id] = None
                counts["invalid_fits"] += 1
                continue

            fit = dict(fit)
            fit["global_center_x"] = x + float(fit["center_x"])
            fit["global_center_y"] = y + float(fit["center_y"])
            results[track_id][frame_id] = fit
            counts["valid_fits"] += 1
            frame_positions.append(
                {
                    "center_x": float(fit["global_center_x"]),
                    "center_y": float(fit["global_center_y"]),
                }
            )

        center = fit_circle_center(frame_positions)
        if center is not None:
            centers[frame_id] = dict(center)
            radii = []
            for track_results in results.values():
                fit = track_results.get(frame_id)
                if fit:
                    _apply_circle_radius(fit, center)
                    radii.append(float(fit["radius"]))
            if len(radii) > 1:
                averages[frame_id] = {
                    "avg_radius": float(np.mean(radii)),
                    "avg_radius_error": float(
                        np.std(radii) / math.sqrt(len(radii))
                    ),
                    "valid_spot_count": len(radii),
                }

        if index == 1 or index % 10 == 0 or index == len(files_by_id):
            print(
                f"[headless-analysis] frames {index}/{len(files_by_id)} | "
                f"valid fits {counts['valid_fits']} | "
                f"invalid fits {counts['invalid_fits']}",
                flush=True,
            )
    counts["missing_tracked_rois"] = (
        counts["expected_track_frames"] - counts["tracked_rois"]
    )
    return results, centers, averages, counts


def _result_rows(
    payload: Mapping,
    results: Mapping[int, Mapping[int, Mapping | None]],
) -> list[dict[str, object]]:
    files_by_id = {
        int(file_payload["id"]): file_payload
        for file_payload in payload.get("data_files", [])
    }
    rows = []
    for track_id in sorted(results):
        for frame_id, fit in sorted(results[track_id].items()):
            if not fit:
                continue
            file_payload = files_by_id[frame_id]
            row = {
                "track_id": track_id,
                "frame_id": frame_id,
                "temperature": file_payload.get("temperature"),
                "energy": file_payload.get("energy"),
                "coverage": file_payload.get("coverage"),
            }
            row.update(
                {
                    key: value
                    for key, value in fit.items()
                    if key != "covariance_matrix"
                }
            )
            rows.append(row)
    return rows


def _write_csv(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    names = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in names})


def _write_numerical_exports(
    staging: Path,
    payload: Mapping,
    results: Mapping[int, Mapping[int, Mapping | None]],
    centers: Mapping[int, Mapping],
    averages: Mapping[int, Mapping],
) -> list[dict[str, object]]:
    files_by_id = {
        int(file_payload["id"]): file_payload
        for file_payload in payload.get("data_files", [])
    }
    rows = _result_rows(payload, results)
    available = {key for row in rows for key in row}
    fields = [field for field in PREFERRED_RESULT_FIELDS if field in available]
    extras = sorted(available.difference(fields))
    _write_csv(staging / "full_analysis_per_spot.csv", fields + extras, rows)

    center_rows = [
        {
            "frame_id": frame_id,
            "temperature": files_by_id[frame_id].get("temperature"),
            "energy": files_by_id[frame_id].get("energy"),
            "coverage": files_by_id[frame_id].get("coverage"),
            **center,
        }
        for frame_id, center in sorted(centers.items())
    ]
    center_fields = (
        "frame_id",
        "temperature",
        "energy",
        "coverage",
        "center_x",
        "center_y",
        "error_x",
        "error_y",
        "circle_radius",
    )
    _write_csv(staging / "pattern_center_data.csv", center_fields, center_rows)

    average_rows = [
        {
            "frame_id": frame_id,
            "temperature": files_by_id[frame_id].get("temperature"),
            "energy": files_by_id[frame_id].get("energy"),
            "coverage": files_by_id[frame_id].get("coverage"),
            **average,
        }
        for frame_id, average in sorted(averages.items())
    ]
    average_fields = (
        "frame_id",
        "temperature",
        "energy",
        "coverage",
        "avg_radius",
        "avg_radius_error",
        "valid_spot_count",
    )
    _write_csv(
        staging / "average_radius_per_frame.csv",
        average_fields,
        average_rows,
    )

    legend_rows = []
    for track in payload.get("tracks", []):
        spots = track.get("spots_by_image_id", {})
        if not spots:
            continue
        start_frame = min(map(int, spots))
        roi = spots[str(start_frame)]["roi"]
        legend_rows.append(
            {
                "track_id": track["track_id"],
                "start_frame_id": start_frame,
                "initial_x_pos": int(roi["x"]) + int(roi["w"]) / 2.0,
                "initial_y_pos": int(roi["y"]) + int(roi["h"]) / 2.0,
            }
        )
    _write_csv(
        staging / "track_legend.csv",
        (
            "track_id",
            "start_frame_id",
            "initial_x_pos",
            "initial_y_pos",
        ),
        legend_rows,
    )
    return rows


def _save_analyzed_session(
    staging: Path,
    payload: dict,
    results: Mapping[int, Mapping[int, Mapping | None]],
    centers: Mapping[int, Mapping],
    averages: Mapping[int, Mapping],
) -> None:
    for track in payload.get("tracks", []):
        track_id = int(track["track_id"])
        track["analysis_results"] = {
            str(frame_id): _json_safe(
                {
                    key: value
                    for key, value in (fit or {}).items()
                    if key != "covariance_matrix"
                }
            )
            if fit
            else None
            for frame_id, fit in sorted(results[track_id].items())
        }
    for file_payload in payload.get("data_files", []):
        frame_id = int(file_payload["id"])
        file_payload["pattern_center"] = _json_safe(centers.get(frame_id))
        file_payload["post_analysis_results"] = _json_safe(
            averages.get(frame_id, {})
        )
    (staging / "analyzed_session.lts.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )


def _image_to_rgb(image: np.ndarray) -> Image.Image:
    data = np.asarray(image)
    if data.ndim >= 3:
        if data.shape[-1] >= 3:
            data = rgb2gray(data[..., :3])
        else:
            data = data[..., 0]
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        normalized = np.zeros(data.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [1.0, 99.5])
        if high <= low:
            high = low + 1.0
        normalized = np.clip((data - low) / (high - low), 0.0, 1.0)
        normalized = np.asarray(255.0 * normalized, dtype=np.uint8)
    return Image.fromarray(normalized).convert("RGB")


def _write_track_visualizations(
    staging: Path,
    session_path: Path,
    payload: Mapping,
    results: Mapping[int, Mapping[int, Mapping | None]],
) -> None:
    output_dir = staging / "track_visualizations"
    output_dir.mkdir()
    files = sorted(payload.get("data_files", []), key=lambda item: int(item["id"]))
    tracks = payload.get("tracks", [])
    font = ImageFont.load_default()
    for index, file_payload in enumerate(files):
        if index % 5 != 0:
            continue
        frame_id = int(file_payload["id"])
        image = _image_to_rgb(
            load_tiff(_resolve_image_path(session_path, file_payload))
        )
        draw = ImageDraw.Draw(image)
        for track_index, track in enumerate(tracks):
            track_id = int(track["track_id"])
            spot = track.get("spots_by_image_id", {}).get(str(frame_id))
            if not spot:
                continue
            color = PALETTE[track_index % len(PALETTE)]
            roi = spot["roi"]
            x, y = int(roi["x"]), int(roi["y"])
            width, height = int(roi["w"]), int(roi["h"])
            draw.rectangle(
                (x, y, x + width, y + height),
                outline=color,
                width=2,
            )
            fit = results.get(track_id, {}).get(frame_id)
            if fit:
                cx = float(fit["global_center_x"])
                cy = float(fit["global_center_y"])
                draw.line((cx - 4, cy, cx + 4, cy), fill=color, width=2)
                draw.line((cx, cy - 4, cx, cy + 4), fill=color, width=2)
            draw.text((x + 2, y + 2), str(track_id), fill=color, font=font)
        image.save(output_dir / f"frame_{frame_id:04d}.png")


def _read_result_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _derived_width(row: Mapping[str, object], kind: str) -> float | None:
    direct = _finite(row.get(f"fwhm_{kind}"))
    if direct is not None:
        return direct
    x = _finite(row.get("fwhm_x"))
    y = _finite(row.get("fwhm_y"))
    if x is None or y is None or x <= 0.0 or y <= 0.0:
        return None
    if kind == "effective":
        return math.sqrt(x * y)
    if kind == "minor":
        return min(x, y)
    if kind == "major":
        return max(x, y)
    raise ValueError(kind)


def _rows_by_track_frame(
    rows: Iterable[Mapping[str, object]],
) -> dict[int, dict[int, Mapping[str, object]]]:
    grouped: dict[int, dict[int, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        track_id = int(float(str(row["track_id"])))
        frame_id = int(float(str(row["frame_id"])))
        grouped[track_id][frame_id] = row
    return dict(grouped)


def _track_mapping(
    current: Mapping[int, Mapping[int, Mapping[str, object]]],
    reference: Mapping[int, Mapping[int, Mapping[str, object]]],
) -> list[dict[str, object]]:
    current_ids = sorted(current)
    reference_ids = sorted(reference)
    costs = np.full((len(current_ids), len(reference_ids)), 1e9)
    common_counts = np.zeros_like(costs, dtype=int)
    for row_index, current_id in enumerate(current_ids):
        for column_index, reference_id in enumerate(reference_ids):
            common = sorted(
                set(current[current_id]).intersection(reference[reference_id])
            )
            distances = []
            for frame_id in common:
                current_row = current[current_id][frame_id]
                reference_row = reference[reference_id][frame_id]
                cx = _finite(current_row.get("global_center_x"))
                cy = _finite(current_row.get("global_center_y"))
                rx = _finite(reference_row.get("global_center_x"))
                ry = _finite(reference_row.get("global_center_y"))
                if None not in (cx, cy, rx, ry):
                    distances.append(math.hypot(cx - rx, cy - ry))
            if distances:
                costs[row_index, column_index] = float(np.median(distances))
                common_counts[row_index, column_index] = len(distances)

    row_indices, column_indices = linear_sum_assignment(costs)
    return [
        {
            "current_track_id": current_ids[row_index],
            "reference_track_id": reference_ids[column_index],
            "median_center_distance_px": costs[row_index, column_index],
            "common_frame_count": common_counts[row_index, column_index],
        }
        for row_index, column_index in zip(row_indices, column_indices)
    ]


def _comparison_rows(
    current: Mapping[int, Mapping[int, Mapping[str, object]]],
    reference: Mapping[int, Mapping[int, Mapping[str, object]]],
    mapping: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    comparisons = []
    for mapped in mapping:
        current_id = int(mapped["current_track_id"])
        reference_id = int(mapped["reference_track_id"])
        common = sorted(set(current[current_id]).intersection(reference[reference_id]))
        for frame_id in common:
            current_row = current[current_id][frame_id]
            reference_row = reference[reference_id][frame_id]
            cx = float(current_row["global_center_x"])
            cy = float(current_row["global_center_y"])
            rx = float(reference_row["global_center_x"])
            ry = float(reference_row["global_center_y"])
            row: dict[str, object] = {
                "current_track_id": current_id,
                "reference_track_id": reference_id,
                "frame_id": frame_id,
                "center_dx": cx - rx,
                "center_dy": cy - ry,
                "center_distance": math.hypot(cx - rx, cy - ry),
            }
            for metric in ("radius", "r_squared"):
                current_value = _finite(current_row.get(metric))
                reference_value = _finite(reference_row.get(metric))
                row[f"{metric}_current"] = current_value
                row[f"{metric}_reference"] = reference_value
                row[f"{metric}_delta"] = (
                    current_value - reference_value
                    if current_value is not None and reference_value is not None
                    else None
                )
            for kind in ("effective", "minor", "major"):
                current_value = _derived_width(current_row, kind)
                reference_value = _derived_width(reference_row, kind)
                row[f"fwhm_{kind}_current"] = current_value
                row[f"fwhm_{kind}_reference"] = reference_value
                row[f"fwhm_{kind}_delta"] = (
                    current_value - reference_value
                    if current_value is not None and reference_value is not None
                    else None
                )
            comparisons.append(row)
    return comparisons


def _metric_summary(
    rows: Iterable[Mapping[str, object]],
    metric: str,
) -> dict[str, object]:
    values = [
        value
        for row in rows
        if (value := _finite(row.get(metric))) is not None
    ]
    if not values:
        return {
            "metric": metric,
            "count": 0,
            "mean": None,
            "median": None,
            "mean_absolute": None,
            "rmse": None,
            "p95_absolute": None,
            "maximum_absolute": None,
        }
    array = np.asarray(values, dtype=float)
    absolute = np.abs(array)
    return {
        "metric": metric,
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "mean_absolute": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(array**2))),
        "p95_absolute": float(np.percentile(absolute, 95.0)),
        "maximum_absolute": float(np.max(absolute)),
    }


def _write_comparison_plots(
    comparison_dir: Path,
    rows: list[Mapping[str, object]],
) -> None:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["current_track_id"])].append(row)
    plot_specs = (
        (
            "center_distance",
            "Center difference: current vs previous",
            "Center distance (px)",
        ),
        (
            "radius_delta",
            "Radius difference: current - previous",
            "Radius delta (px)",
        ),
        (
            "fwhm_effective_delta",
            "Effective FWHM difference: current - previous",
            "Effective FWHM delta (px)",
        ),
    )
    for metric, title, y_label in plot_specs:
        series = []
        for index, track_id in enumerate(sorted(grouped)):
            track_rows = sorted(grouped[track_id], key=lambda row: int(row["frame_id"]))
            valid_rows = [
                row for row in track_rows if _finite(row.get(metric)) is not None
            ]
            series.append(
                PlotSeries(
                    label=f"Track {track_id}",
                    color=PALETTE[index % len(PALETTE)],
                    line_width=2,
                    x_values=[float(row["frame_id"]) for row in valid_rows],
                    y_values=[float(row[metric]) for row in valid_rows],
                )
            )
        _render_line_plot(
            comparison_dir / f"{metric}_by_frame.png",
            title=title,
            y_label=y_label,
            reference_y=0.0 if metric.endswith("_delta") else None,
            series=series,
        )


def _frozen_radius_comparison(
    current: Mapping[int, Mapping[int, Mapping[str, object]]],
    reference: Mapping[int, Mapping[int, Mapping[str, object]]],
    mapping: list[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    maximum_common_count = max(
        int(row["common_frame_count"]) for row in mapping
    )
    complete_pairs = [
        (
            int(row["current_track_id"]),
            int(row["reference_track_id"]),
        )
        for row in mapping
        if int(row["common_frame_count"]) == maximum_common_count
    ]
    common_frames: set[int] | None = None
    for current_id, reference_id in complete_pairs:
        pair_frames = {
            frame_id
            for frame_id in set(current[current_id]).intersection(
                reference[reference_id]
            )
            if _finite(current[current_id][frame_id].get("radius")) is not None
            and _finite(reference[reference_id][frame_id].get("radius"))
            is not None
        }
        common_frames = (
            pair_frames
            if common_frames is None
            else common_frames.intersection(pair_frames)
        )
    frame_ids = sorted(common_frames or [])
    if not complete_pairs or not frame_ids:
        return [], {
            "complete_pairs": complete_pairs,
            "frame_count": 0,
        }

    radius_rows = []
    for frame_id in frame_ids:
        current_radii = [
            float(current[current_id][frame_id]["radius"])
            for current_id, _reference_id in complete_pairs
        ]
        reference_radii = [
            float(reference[reference_id][frame_id]["radius"])
            for _current_id, reference_id in complete_pairs
        ]
        radius_rows.append(
            {
                "frame_id": frame_id,
                "current_average_radius": float(np.mean(current_radii)),
                "reference_average_radius": float(np.mean(reference_radii)),
            }
        )

    current_baseline = float(radius_rows[0]["current_average_radius"])
    reference_baseline = float(radius_rows[0]["reference_average_radius"])
    for row in radius_rows:
        current_relative = (
            float(row["current_average_radius"]) / current_baseline - 1.0
        )
        reference_relative = (
            float(row["reference_average_radius"]) / reference_baseline - 1.0
        )
        row["current_relative_change_percent"] = 100.0 * current_relative
        row["reference_relative_change_percent"] = 100.0 * reference_relative
        row["relative_change_delta_percentage_points"] = 100.0 * (
            current_relative - reference_relative
        )

    current_values = np.asarray(
        [
            float(row["current_relative_change_percent"])
            for row in radius_rows
        ]
    )
    reference_values = np.asarray(
        [
            float(row["reference_relative_change_percent"])
            for row in radius_rows
        ]
    )
    deltas = current_values - reference_values
    correlation = float(np.corrcoef(current_values, reference_values)[0, 1])
    summary = {
        "complete_pairs": [
            {
                "current_track_id": current_id,
                "reference_track_id": reference_id,
            }
            for current_id, reference_id in complete_pairs
        ],
        "frame_count": len(radius_rows),
        "correlation": correlation,
        "mean_absolute_delta_percentage_points": float(
            np.mean(np.abs(deltas))
        ),
        "rmse_delta_percentage_points": float(
            np.sqrt(np.mean(deltas**2))
        ),
        "final_current_relative_change_percent": float(current_values[-1]),
        "final_reference_relative_change_percent": float(reference_values[-1]),
    }
    return radius_rows, summary


def _write_frozen_radius_comparison(
    comparison_dir: Path,
    rows: list[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    _write_csv(
        comparison_dir / "frozen_subset_average_radius_per_frame.csv",
        (
            "frame_id",
            "current_average_radius",
            "reference_average_radius",
            "current_relative_change_percent",
            "reference_relative_change_percent",
            "relative_change_delta_percentage_points",
        ),
        rows,
    )
    (comparison_dir / "frozen_subset_radius_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2),
        encoding="utf-8",
    )
    if not rows:
        return
    _render_line_plot(
        comparison_dir / "frozen_subset_relative_radius_comparison.png",
        title="Relative average-radius change: current vs previous",
        y_label="Relative radius change (%)",
        reference_y=0.0,
        series=[
            PlotSeries(
                label="Current DAM4SAM",
                color=PALETTE[0],
                line_width=3,
                x_values=[float(row["frame_id"]) for row in rows],
                y_values=[
                    float(row["current_relative_change_percent"])
                    for row in rows
                ],
            ),
            PlotSeries(
                label="Previous analysis",
                color=PALETTE[1],
                line_width=2,
                x_values=[float(row["frame_id"]) for row in rows],
                y_values=[
                    float(row["reference_relative_change_percent"])
                    for row in rows
                ],
            ),
        ],
    )


def _write_comparison(
    staging: Path,
    rows: list[dict[str, object]],
    reference_csv: Path,
    counts: Mapping[str, int],
) -> dict[str, object]:
    comparison_dir = staging / "comparison_with_previous"
    comparison_dir.mkdir()
    current_grouped = _rows_by_track_frame(rows)
    reference_grouped = _rows_by_track_frame(_read_result_csv(reference_csv))
    mapping = _track_mapping(current_grouped, reference_grouped)
    comparisons = _comparison_rows(current_grouped, reference_grouped, mapping)
    frozen_radius_rows, frozen_radius_summary = _frozen_radius_comparison(
        current_grouped,
        reference_grouped,
        mapping,
    )
    summaries = [
        _metric_summary(comparisons, metric) for metric in COMPARISON_METRICS
    ]

    _write_csv(
        comparison_dir / "track_mapping.csv",
        (
            "current_track_id",
            "reference_track_id",
            "median_center_distance_px",
            "common_frame_count",
        ),
        mapping,
    )
    comparison_fields = (
        "current_track_id",
        "reference_track_id",
        "frame_id",
        "center_dx",
        "center_dy",
        "center_distance",
        "radius_current",
        "radius_reference",
        "radius_delta",
        "fwhm_effective_current",
        "fwhm_effective_reference",
        "fwhm_effective_delta",
        "fwhm_minor_current",
        "fwhm_minor_reference",
        "fwhm_minor_delta",
        "fwhm_major_current",
        "fwhm_major_reference",
        "fwhm_major_delta",
        "r_squared_current",
        "r_squared_reference",
        "r_squared_delta",
    )
    _write_csv(
        comparison_dir / "comparison_per_spot_frame.csv",
        comparison_fields,
        comparisons,
    )
    summary_fields = (
        "metric",
        "count",
        "mean",
        "median",
        "mean_absolute",
        "rmse",
        "p95_absolute",
        "maximum_absolute",
    )
    _write_csv(
        comparison_dir / "comparison_summary.csv",
        summary_fields,
        summaries,
    )
    _write_comparison_plots(comparison_dir, comparisons)
    _write_frozen_radius_comparison(
        comparison_dir,
        frozen_radius_rows,
        frozen_radius_summary,
    )

    summary_by_metric = {str(row["metric"]): row for row in summaries}
    center = summary_by_metric["center_distance"]
    radius = summary_by_metric["radius_delta"]
    fwhm = summary_by_metric["fwhm_effective_delta"]
    typical_rows = [
        row for row in comparisons if float(row["center_distance"]) <= 1.0
    ]
    typical_center = _metric_summary(typical_rows, "center_distance")
    typical_radius = _metric_summary(typical_rows, "radius_delta")
    typical_fwhm = _metric_summary(typical_rows, "fwhm_effective_delta")
    fwhm_relative = [
        100.0
        * float(row["fwhm_effective_delta"])
        / float(row["fwhm_effective_reference"])
        for row in typical_rows
        if _finite(row.get("fwhm_effective_reference"))
        and float(row["fwhm_effective_reference"]) != 0.0
    ]
    outlier_rows = [
        row for row in comparisons if float(row["center_distance"]) > 1.0
    ]
    outliers_by_track: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in outlier_rows:
        outliers_by_track[
            (
                int(row["current_track_id"]),
                int(row["reference_track_id"]),
            )
        ].append(int(row["frame_id"]))
    outlier_descriptions = [
        (
            f"current track {current_id} / previous track {reference_id}: "
            + ", ".join(map(str, sorted(frame_ids)))
        )
        for (current_id, reference_id), frame_ids in sorted(
            outliers_by_track.items()
        )
    ]
    lines = [
        "# Comparison with previous TrackLEED analysis",
        "",
        "## Completeness",
        "",
        (
            f"- Frames: {counts['frames']}; tracks: {counts['tracks']}; "
            f"expected track/frame combinations: "
            f"{counts['expected_track_frames']}."
        ),
        (
            f"- Saved tracking ROIs: {counts['tracked_rois']}; missing ROIs: "
            f"{counts['missing_tracked_rois']}."
        ),
        (
            f"- Accepted Gaussian fits: {counts['valid_fits']}; rejected fits: "
            f"{counts['invalid_fits']}."
        ),
        "",
        "## Inputs and matching",
        "",
        f"- Reference CSV: `{reference_csv}`",
        f"- Automatically matched tracks: {len(mapping)}",
        f"- Compared spot/frame pairs: {len(comparisons)}",
        "- Matching criterion: minimum median distance between Gaussian-refined centers over common frames.",
        "",
        "## Main numerical differences",
        "",
        (
            f"- Center distance: median {float(center['median']):.4f} px, "
            f"mean {float(center['mean']):.4f} px, "
            f"95th percentile {float(center['p95_absolute']):.4f} px."
        ),
        (
            f"- Radius: mean absolute difference "
            f"{float(radius['mean_absolute']):.4f} px, "
            f"bias {float(radius['mean']):+.4f} px."
        ),
        (
            f"- Effective FWHM: mean absolute difference "
            f"{float(fwhm['mean_absolute']):.4f} px, "
            f"bias {float(fwhm['mean']):+.4f} px."
        ),
        "",
        "## Typical agreement after isolating center outliers",
        "",
        (
            f"- Rows with center distance <= 1 px: {len(typical_rows)} of "
            f"{len(comparisons)}."
        ),
        (
            f"- Center distance: median "
            f"{float(typical_center['median']):.4f} px, "
            f"95th percentile "
            f"{float(typical_center['p95_absolute']):.4f} px."
        ),
        (
            f"- Radius: mean absolute difference "
            f"{float(typical_radius['mean_absolute']):.4f} px, "
            f"bias {float(typical_radius['mean']):+.4f} px."
        ),
        (
            f"- Effective FWHM: median difference "
            f"{float(typical_fwhm['median']):+.4f} px, "
            f"mean absolute difference "
            f"{float(typical_fwhm['mean_absolute']):.4f} px, "
            f"median relative difference "
            f"{float(np.median(fwhm_relative)):+.2f}%."
        ),
        "",
        "## Primary radius endpoint on a frozen complete-track subset",
        "",
        (
            "- Matched track pairs: "
            + ", ".join(
                (
                    f"{pair['current_track_id']}"
                    f"/{pair['reference_track_id']}"
                )
                for pair in frozen_radius_summary["complete_pairs"]
            )
            + " (current/previous)."
        ),
        (
            f"- Compared frames: {frozen_radius_summary['frame_count']}."
        ),
        (
            f"- Correlation of relative average-radius curves: "
            f"{float(frozen_radius_summary['correlation']):.6f}."
        ),
        (
            f"- Mean absolute curve difference: "
            f"{float(frozen_radius_summary['mean_absolute_delta_percentage_points']):.4f} "
            "percentage points."
        ),
        (
            f"- Final relative radius change: current "
            f"{float(frozen_radius_summary['final_current_relative_change_percent']):+.3f}% "
            "vs previous "
            f"{float(frozen_radius_summary['final_reference_relative_change_percent']):+.3f}%."
        ),
        "",
        "## Localized discrepancies",
        "",
        (
            f"- Center-distance outliers above 1 px: {len(outlier_rows)}."
        ),
        *[f"- {description}" for description in outlier_descriptions],
        "",
        "## Interpretation",
        "",
        (
            "Near-zero center differences indicate that both analyses identify "
            "the same physical diffraction spots. FWHM can differ despite "
            "matching centers because Gaussian background and width estimates "
            "depend on the ROI/bounding-box extent; the current DAM4SAM session "
            "uses its saved 32 x 32 px boxes."
        ),
        (
            "The localized late-series discrepancy should be reviewed as a "
            "tracking/visibility case rather than averaged into the otherwise "
            "well-matched population."
        ),
        "",
    ]
    (comparison_dir / "comparison_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return {
        "reference_csv": str(reference_csv.resolve()),
        "reference_sha256": _sha256(reference_csv),
        "matched_tracks": len(mapping),
        "compared_rows": len(comparisons),
        "mapping": _json_safe(mapping),
        "summary": _json_safe(summaries),
        "frozen_complete_track_radius_comparison": _json_safe(
            frozen_radius_summary
        ),
        "typical_center_distance_at_most_1px": {
            "row_count": len(typical_rows),
            "center_summary": _json_safe(typical_center),
            "radius_summary": _json_safe(typical_radius),
            "fwhm_effective_summary": _json_safe(typical_fwhm),
            "fwhm_effective_relative_median_percent": float(
                np.median(fwhm_relative)
            ),
        },
        "center_outliers_above_1px": {
            "row_count": len(outlier_rows),
            "by_track": outlier_descriptions,
        },
    }


def _write_manifest(
    staging: Path,
    *,
    session_path: Path,
    reference_csv: Path | None,
    counts: Mapping[str, int],
    comparison: Mapping[str, object] | None,
) -> None:
    files = sorted(
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    manifest = {
        "version": 1,
        "source_session": str(session_path.resolve()),
        "source_session_sha256": _sha256(session_path),
        "reference_csv": str(reference_csv.resolve()) if reference_csv else None,
        "settings": {
            "gaussian_model": "rotated_2d",
            "center_mode": "per_frame_circle_fit",
            "background_subtraction": False,
            "minimum_r_squared": MIN_R_SQUARED,
            "minimum_amplitude": MIN_AMPLITUDE,
        },
        "counts": dict(counts),
        "comparison": comparison,
        "files": files,
    }
    (staging / "analysis_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    session_path = args.session.resolve()
    output_dir = args.output_dir.resolve()
    reference_csv = (
        args.reference_csv.resolve() if args.reference_csv is not None else None
    )
    if not session_path.is_file():
        raise FileNotFoundError(session_path)
    if reference_csv is not None and not reference_csv.is_file():
        raise FileNotFoundError(reference_csv)
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists and will not be overwritten: {output_dir}"
        )

    payload = json.loads(session_path.read_text(encoding="utf-8"))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        results, centers, averages, counts = _fit_session(
            payload,
            session_path,
        )
        rows = _write_numerical_exports(
            staging,
            payload,
            results,
            centers,
            averages,
        )
        _save_analyzed_session(staging, payload, results, centers, averages)
        _write_track_visualizations(staging, session_path, payload, results)
        export_fwhm_series(
            staging / "full_analysis_per_spot.csv",
            staging / "fwhm",
        )
        comparison = (
            _write_comparison(staging, rows, reference_csv, counts)
            if reference_csv is not None
            else None
        )
        _write_manifest(
            staging,
            session_path=session_path,
            reference_csv=reference_csv,
            counts=counts,
            comparison=comparison,
        )
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Complete analysis written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
