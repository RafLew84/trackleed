#!/usr/bin/env python3
"""Throwaway ScanE tracer for the annotation-free publication map.

This is intentionally a single-file batch prototype. It validates the
measurement/metric contract before the controlled tracker family is run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import affine_transform, gaussian_filter, shift
from skimage.feature import peak_local_max
from skimage.morphology import disk, white_tophat
from skimage.util import img_as_float


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data" / "26a24_ScanE"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# The prototype intentionally reuses the application's common estimator.
sys.path.insert(0, str(REPO_ROOT))
from trackleed.analysis.peak_analyzer import fit_spot_to_gaussian  # noqa: E402


ROI_SIZE = 48
R2_MIN = 0.85
AMPLITUDE_MIN = 0.012
SIGMA_MIN = 0.6
SIGMA_MAX = 13.0
CENTER_TOLERANCE_PX = 15.0
BRANCH_TOLERANCE_PX = 18.0
REFERENCE_ENERGY_TARGET_EV = 187.9
REFERENCE_ENERGY_EV = REFERENCE_ENERGY_TARGET_EV

ENERGY_RE = re.compile(r"_E(\d{4})_")
FRAME_RE = re.compile(r"_(\d{4})_T")


@dataclass(frozen=True)
class Frame:
    frame: int
    energy_ev: float
    path: Path


@dataclass(frozen=True)
class Geometry:
    center_x: float
    center_y: float
    aperture_radius: float
    screen_radius: float


def load_frames() -> list[Frame]:
    frames: list[Frame] = []
    for path in DATA_DIR.glob("*.tiff"):
        energy_match = ENERGY_RE.search(path.name)
        frame_match = FRAME_RE.search(path.name)
        if energy_match is None or frame_match is None:
            continue
        frames.append(
            Frame(
                frame=int(frame_match.group(1)),
                energy_ev=int(energy_match.group(1)) / 10.0,
                path=path,
            )
        )
    if not frames:
        raise RuntimeError(f"No TIFF frames found in {DATA_DIR}")
    return sorted(frames, key=lambda item: item.energy_ev)


def read_float(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image, got {image.shape}: {path}")
    return img_as_float(image)


def fit_circle(points_xy: np.ndarray) -> tuple[float, float, float]:
    """Algebraic circle fit, sufficient for frozen apparatus geometry."""
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    design = np.column_stack((x, y, np.ones_like(x)))
    rhs = -(x * x + y * y)
    a, b, c = np.linalg.lstsq(design, rhs, rcond=None)[0]
    center_x = -a / 2.0
    center_y = -b / 2.0
    radius_sq = center_x * center_x + center_y * center_y - c
    return float(center_x), float(center_y), float(math.sqrt(max(radius_sq, 0.0)))


def radial_transition_points(
    image: np.ndarray,
    origin_xy: tuple[float, float],
    radius_range: range,
    *,
    transition: str,
    threshold: float,
) -> np.ndarray:
    """Sample stable dark/bright boundaries along many radial rays."""
    origin_x, origin_y = origin_xy
    points: list[tuple[float, float]] = []
    radii = np.asarray(list(radius_range), dtype=float)
    for angle in np.linspace(0.0, 2.0 * np.pi, 1080, endpoint=False):
        xs = np.rint(origin_x + radii * np.cos(angle)).astype(int)
        ys = np.rint(origin_y + radii * np.sin(angle)).astype(int)
        inside = (
            (xs >= 0)
            & (xs < image.shape[1])
            & (ys >= 0)
            & (ys < image.shape[0])
        )
        xs = xs[inside]
        ys = ys[inside]
        rs = radii[inside]
        if len(rs) < 5:
            continue
        bright = image[ys, xs] > threshold
        if transition == "dark_to_bright":
            stable = np.convolve(bright.astype(int), np.ones(4, dtype=int), mode="same")
            candidates = np.flatnonzero(stable >= 4)
        elif transition == "bright_to_dark":
            stable = np.convolve((~bright).astype(int), np.ones(4, dtype=int), mode="same")
            candidates = np.flatnonzero(stable >= 4)
        else:
            raise ValueError(transition)
        if candidates.size:
            index = int(candidates[0])
            points.append((float(xs[index]), float(ys[index])))
    return np.asarray(points, dtype=float)


def robust_circle(points_xy: np.ndarray) -> tuple[float, float, float]:
    points = points_xy.copy()
    for _ in range(5):
        center_x, center_y, radius = fit_circle(points)
        residuals = np.abs(
            np.hypot(points[:, 0] - center_x, points[:, 1] - center_y) - radius
        )
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        limit = median + max(2.5 * 1.4826 * mad, 1.5)
        kept = points[residuals <= limit]
        if len(kept) == len(points) or len(kept) < 20:
            break
        points = kept
    return fit_circle(points)


def estimate_geometry(reference: np.ndarray) -> Geometry:
    nominal = ((reference.shape[1] - 1) / 2.0, (reference.shape[0] - 1) / 2.0)
    aperture_points = radial_transition_points(
        reference,
        nominal,
        range(45, 190),
        transition="dark_to_bright",
        threshold=0.025,
    )
    if len(aperture_points) < 40:
        raise RuntimeError("Could not estimate the central aperture.")
    center_x, center_y, aperture_radius = robust_circle(aperture_points)

    screen_points = radial_transition_points(
        reference,
        (center_x, center_y),
        range(330, 550),
        transition="bright_to_dark",
        threshold=0.018,
    )
    if len(screen_points) < 40:
        raise RuntimeError("Could not estimate the active screen.")
    screen_x, screen_y, screen_radius = robust_circle(screen_points)

    # The aperture is mechanically tied to the diffraction center and is the
    # cleaner center estimate. The outer-screen fit is used only as a sanity
    # check because labels touch the rim.
    center_delta = math.hypot(screen_x - center_x, screen_y - center_y)
    if center_delta > 35.0:
        raise RuntimeError(
            f"Inconsistent apparatus centers: aperture/screen differ by {center_delta:.1f}px"
        )
    return Geometry(center_x, center_y, aperture_radius, screen_radius)


def apparatus_mask(reference: np.ndarray, geometry: Geometry) -> np.ndarray:
    yy, xx = np.indices(reference.shape)
    radius = np.hypot(xx - geometry.center_x, yy - geometry.center_y)
    annulus = (radius >= geometry.aperture_radius + ROI_SIZE / 2) & (
        radius <= geometry.screen_radius - ROI_SIZE / 2
    )
    # Heavy smoothing keeps the fixed apparatus silhouette while suppressing
    # individual diffraction spots.
    screen = gaussian_filter(reference, sigma=18.0) > 0.018
    return annulus & screen


def detect_reference_spots(
    reference: np.ndarray, geometry: Geometry, available: np.ndarray
) -> list[dict[str, float]]:
    residual = reference - gaussian_filter(reference, sigma=20.0)
    enhanced = white_tophat(residual, footprint=disk(8))
    coordinates = peak_local_max(
        enhanced,
        min_distance=18,
        threshold_rel=0.09,
        exclude_border=True,
    )
    candidates: list[dict[str, float]] = []
    for y, x in coordinates:
        if not available[y, x]:
            continue
        radius = math.hypot(x - geometry.center_x, y - geometry.center_y)
        if reference[y, x] < 0.075:
            continue
        candidates.append(
            {
                "x": float(x),
                "y": float(y),
                "radius": radius,
                "intensity": float(enhanced[y, x]),
            }
        )
    return sorted(candidates, key=lambda item: item["intensity"], reverse=True)


def select_diametric_pair(
    candidates: list[dict[str, float]],
    geometry: Geometry,
    minimum_energy_ev: float,
    pair_rank: int = 1,
) -> tuple[dict[str, float], dict[str, float]]:
    if len(candidates) < 2:
        raise RuntimeError("Too few automatically detected spots.")
    max_intensity = max(item["intensity"] for item in candidates)
    scored: list[tuple[float, dict[str, float], dict[str, float]]] = []
    center = np.array([geometry.center_x, geometry.center_y], dtype=float)
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            v1 = np.array([first["x"], first["y"]]) - center
            v2 = np.array([second["x"], second["y"]]) - center
            r1 = float(np.linalg.norm(v1))
            r2 = float(np.linalg.norm(v2))
            cosine = float(np.dot(v1, v2) / (r1 * r2))
            mean_radius = (r1 + r2) / 2.0
            radius_mismatch = abs(r1 - r2) / mean_radius
            midpoint_error = float(
                np.linalg.norm((np.array([first["x"], first["y"]])
                                + np.array([second["x"], second["y"]])) / 2.0 - center)
                / mean_radius
            )
            if cosine > -0.92 or radius_mismatch > 0.18:
                continue
            # A tracer pair on the rim is nearly absent over the low-energy
            # half of this scan. Retain only pairs whose physics-scaled radius
            # reaches the low-energy screen with at most a small edge margin.
            low_energy_radius = mean_radius * math.sqrt(
                REFERENCE_ENERGY_EV / minimum_energy_ev
            )
            if low_energy_radius > geometry.screen_radius + ROI_SIZE / 3:
                continue
            intensity_reward = (
                first["intensity"] + second["intensity"]
            ) / (2.0 * max_intensity)
            coverage_reward = 1.0 - min(
                1.0, low_energy_radius / geometry.screen_radius
            )
            score = (
                (cosine + 1.0)
                + radius_mismatch
                + midpoint_error
                - 0.35 * intensity_reward
                - 0.20 * coverage_reward
            )
            scored.append((score, first, second))
    if not scored:
        raise RuntimeError("No approximately diametric pair found.")
    # Greedily freeze independent pairs so pair_rank=2 cannot reuse either
    # spot from the primary pair. This keeps the replication independent
    # without manual spot selection.
    independent_pairs: list[
        tuple[float, dict[str, float], dict[str, float]]
    ] = []
    used_candidate_ids: set[int] = set()
    for item in sorted(scored, key=lambda value: value[0]):
        _, first, second = item
        if id(first) in used_candidate_ids or id(second) in used_candidate_ids:
            continue
        independent_pairs.append(item)
        used_candidate_ids.update((id(first), id(second)))
    if pair_rank < 1 or pair_rank > len(independent_pairs):
        raise RuntimeError(
            f"Requested independent pair rank {pair_rank}, "
            f"but only {len(independent_pairs)} pairs are available."
        )
    _, first, second = independent_pairs[pair_rank - 1]
    return first, second


def extract_roi(
    image: np.ndarray, predicted_x: float, predicted_y: float
) -> tuple[np.ndarray | None, int, int]:
    half = ROI_SIZE // 2
    x0 = int(round(predicted_x)) - half
    y0 = int(round(predicted_y)) - half
    x1 = x0 + ROI_SIZE
    y1 = y0 + ROI_SIZE
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        return None, x0, y0
    return image[y0:y1, x0:x1], x0, y0


def measure(
    image: np.ndarray, predicted_x: float, predicted_y: float
) -> dict[str, float | str | bool]:
    roi, x0, y0 = extract_roi(image, predicted_x, predicted_y)
    if roi is None:
        return {"fit_valid": False, "fit_reason": "roi_outside_image"}
    fit = fit_spot_to_gaussian(roi, center_preference_factor=0.34)
    if fit is None:
        return {"fit_valid": False, "fit_reason": "fit_failed"}

    center_x = x0 + float(fit["center_x"])
    center_y = y0 + float(fit["center_y"])
    proposal_offset = math.hypot(center_x - predicted_x, center_y - predicted_y)
    reasons: list[str] = []
    if float(fit["r_squared"]) < R2_MIN:
        reasons.append("low_r2")
    if float(fit["amplitude"]) < AMPLITUDE_MIN:
        reasons.append("low_amplitude")
    if not (
        SIGMA_MIN <= float(fit["sigma_x"]) <= SIGMA_MAX
        and SIGMA_MIN <= float(fit["sigma_y"]) <= SIGMA_MAX
    ):
        reasons.append("invalid_sigma")
    if proposal_offset > CENTER_TOLERANCE_PX:
        reasons.append("center_outside_convergence_zone")

    return {
        "fit_valid": not reasons,
        "fit_reason": "accepted" if not reasons else "+".join(reasons),
        "center_x": center_x,
        "center_y": center_y,
        "proposal_offset_px": proposal_offset,
        "r_squared": float(fit["r_squared"]),
        "amplitude": float(fit["amplitude"]),
        "volume": float(fit["volume"]),
        "sigma_x": float(fit["sigma_x"]),
        "sigma_y": float(fit["sigma_y"]),
    }


def point_available(mask: np.ndarray, x: float, y: float) -> bool:
    xi = int(round(x))
    yi = int(round(y))
    half_window = 5
    if (
        xi - half_window < 0
        or yi - half_window < 0
        or xi + half_window >= mask.shape[1]
        or yi + half_window >= mask.shape[0]
    ):
        return False
    patch = mask[
        yi - half_window : yi + half_window + 1,
        xi - half_window : xi + half_window + 1,
    ]
    return float(np.mean(patch)) >= 0.75


def gaussian_refine_seeds(
    reference: np.ndarray,
    selected: Iterable[dict[str, float]],
) -> list[dict[str, float]]:
    seeds: list[dict[str, float]] = []
    for branch_index, detection in enumerate(selected, start=1):
        result = measure(reference, detection["x"], detection["y"])
        if not result.get("fit_valid"):
            raise RuntimeError(
                f"Reference fit failed for branch {branch_index}: {result.get('fit_reason')}"
            )
        seeds.append(
            {
                "branch": f"B{branch_index}",
                "detection_x": detection["x"],
                "detection_y": detection["y"],
                "center_x": float(result["center_x"]),
                "center_y": float(result["center_y"]),
                "reference_r2": float(result["r_squared"]),
                "reference_amplitude": float(result["amplitude"]),
            }
        )
    return seeds


def predicted_position(
    seed: dict[str, float], geometry: Geometry, energy_ev: float
) -> tuple[float, float]:
    scale = math.sqrt(REFERENCE_ENERGY_EV / energy_ev)
    return (
        geometry.center_x + (seed["center_x"] - geometry.center_x) * scale,
        geometry.center_y + (seed["center_y"] - geometry.center_y) * scale,
    )


def classify_missing_runs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Classify invalid in-apparatus runs after the full branch is visible."""
    gap_summaries: list[dict[str, object]] = []
    by_branch: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_branch.setdefault(str(row["branch"]), []).append(row)

    for branch, branch_rows in by_branch.items():
        branch_rows.sort(key=lambda row: float(row["energy_ev"]))
        visible_indices = [
            index
            for index, row in enumerate(branch_rows)
            if row["status"] == "visible_fit"
        ]
        first_visible = min(visible_indices) if visible_indices else None
        last_visible = max(visible_indices) if visible_indices else None

        index = 0
        while index < len(branch_rows):
            row = branch_rows[index]
            if row["status"] != "unmeasurable_in_apparatus":
                index += 1
                continue
            start = index
            while (
                index + 1 < len(branch_rows)
                and branch_rows[index + 1]["status"] == "unmeasurable_in_apparatus"
            ):
                index += 1
            end = index
            bracketed = (
                first_visible is not None
                and last_visible is not None
                and start > first_visible
                and end < last_visible
                and branch_rows[start - 1]["status"] == "visible_fit"
                and branch_rows[end + 1]["status"] == "visible_fit"
            )
            classification = "iv_visibility_gap" if bracketed else "beam_onset_or_termination"
            for gap_row in branch_rows[start : end + 1]:
                gap_row["status"] = classification
            gap_summaries.append(
                {
                    "branch": branch,
                    "classification": classification,
                    "start_energy_ev": branch_rows[start]["energy_ev"],
                    "end_energy_ev": branch_rows[end]["energy_ev"],
                    "frame_count": end - start + 1,
                    "recovered_without_reinitialization": bool(bracketed),
                    "recovery_energy_ev": (
                        branch_rows[end + 1]["energy_ev"] if bracketed else None
                    ),
                }
            )
            index += 1
    return gap_summaries


def run_scan(
    frames: list[Frame],
    seeds: list[dict[str, float]],
    geometry: Geometry,
    available: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    center = np.array([geometry.center_x, geometry.center_y], dtype=float)
    for frame in frames:
        image = read_float(frame.path)
        for seed in seeds:
            predicted_x, predicted_y = predicted_position(seed, geometry, frame.energy_ev)
            row: dict[str, object] = {
                "frame": frame.frame,
                "energy_ev": frame.energy_ev,
                "branch": seed["branch"],
                "predicted_x": predicted_x,
                "predicted_y": predicted_y,
            }
            if not point_available(available, predicted_x, predicted_y):
                row.update(
                    {
                        "status": "apparatus_absence",
                        "fit_valid": False,
                        "fit_reason": "outside_active_annulus_or_in_apparatus_shadow",
                    }
                )
                rows.append(row)
                continue

            result = measure(image, predicted_x, predicted_y)
            row.update(result)
            if not result.get("fit_valid"):
                row["status"] = "unmeasurable_in_apparatus"
                rows.append(row)
                continue

            measured = np.array(
                [float(result["center_x"]), float(result["center_y"])], dtype=float
            )
            predicted = np.array([predicted_x, predicted_y], dtype=float)
            radial_unit = (predicted - center) / np.linalg.norm(predicted - center)
            tangent_unit = np.array([-radial_unit[1], radial_unit[0]])
            residual = measured - predicted
            radial_residual = float(np.dot(residual, radial_unit))
            tangential_residual = float(np.dot(residual, tangent_unit))
            branch_error = float(np.linalg.norm(residual))
            row.update(
                {
                    "radial_residual_px": radial_residual,
                    "tangential_residual_px": tangential_residual,
                    "branch_error_px": branch_error,
                    "measured_radius_px": float(np.linalg.norm(measured - center)),
                    "status": (
                        "visible_fit"
                        if branch_error <= BRANCH_TOLERANCE_PX
                        else "geometry_inconsistent_fit"
                    ),
                }
            )
            rows.append(row)
    gaps = classify_missing_runs(rows)
    return rows, gaps


def build_pair_diagnostics(
    scan_rows: list[dict[str, object]],
    seeds: list[dict[str, float]],
) -> list[dict[str, object]]:
    """Compute translation-invariant pair geometry on jointly visible frames."""
    by_frame: dict[tuple[int, float], dict[str, dict[str, object]]] = {}
    for row in scan_rows:
        if row["status"] != "visible_fit":
            continue
        key = (int(row["frame"]), float(row["energy_ev"]))
        by_frame.setdefault(key, {})[str(row["branch"])] = row

    seed_by_branch = {str(seed["branch"]): seed for seed in seeds}
    seed_1 = seed_by_branch["B1"]
    seed_2 = seed_by_branch["B2"]
    reference_distance = math.hypot(
        seed_2["center_x"] - seed_1["center_x"],
        seed_2["center_y"] - seed_1["center_y"],
    )
    reference_midpoint = np.array(
        [
            (seed_1["center_x"] + seed_2["center_x"]) / 2.0,
            (seed_1["center_y"] + seed_2["center_y"]) / 2.0,
        ],
        dtype=float,
    )

    pair_rows: list[dict[str, object]] = []
    for (frame, energy_ev), branches in sorted(
        by_frame.items(), key=lambda item: item[0][1]
    ):
        if "B1" not in branches or "B2" not in branches:
            continue
        first = np.array(
            [float(branches["B1"]["center_x"]), float(branches["B1"]["center_y"])],
            dtype=float,
        )
        second = np.array(
            [float(branches["B2"]["center_x"]), float(branches["B2"]["center_y"])],
            dtype=float,
        )
        midpoint = (first + second) / 2.0
        distance = float(np.linalg.norm(second - first))
        predicted_distance = reference_distance * math.sqrt(
            REFERENCE_ENERGY_EV / energy_ev
        )
        midpoint_delta = midpoint - reference_midpoint
        pair_rows.append(
            {
                "frame": frame,
                "energy_ev": energy_ev,
                "pair_distance_px": distance,
                "predicted_distance_px": predicted_distance,
                "distance_residual_px": distance - predicted_distance,
                "midpoint_x": float(midpoint[0]),
                "midpoint_y": float(midpoint[1]),
                "midpoint_dx_px": float(midpoint_delta[0]),
                "midpoint_dy_px": float(midpoint_delta[1]),
                "midpoint_drift_px": float(np.linalg.norm(midpoint_delta)),
            }
        )
    return pair_rows


def radial_scale_image(
    image: np.ndarray, geometry: Geometry, scale_factor: float
) -> np.ndarray:
    # scipy affine_transform consumes coordinates in (y, x) order and maps
    # output coordinates back into the input.
    center_yx = np.array([geometry.center_y, geometry.center_x], dtype=float)
    inverse = np.eye(2) / scale_factor
    offset = center_yx - inverse @ center_yx
    return affine_transform(
        image,
        inverse,
        offset=offset,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def run_transform_checks(
    reference: np.ndarray,
    seeds: list[dict[str, float]],
    geometry: Geometry,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    translations = [(-14.0, -9.0), (-8.0, 11.0), (13.0, -6.0), (10.0, 12.0)]
    for dx, dy in translations:
        transformed = shift(
            reference,
            shift=(dy, dx),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        for seed in seeds:
            expected_x = seed["center_x"] + dx
            expected_y = seed["center_y"] + dy
            result = measure(transformed, expected_x, expected_y)
            error = (
                math.hypot(
                    float(result["center_x"]) - expected_x,
                    float(result["center_y"]) - expected_y,
                )
                if result.get("fit_valid")
                else None
            )
            rows.append(
                {
                    "transform": "translation",
                    "parameter": f"dx={dx:g},dy={dy:g}",
                    "branch": seed["branch"],
                    "expected_x": expected_x,
                    "expected_y": expected_y,
                    "equivariance_error_px": error,
                    **result,
                }
            )

    for scale_factor in [0.94, 0.97, 1.03, 1.06]:
        transformed = radial_scale_image(reference, geometry, scale_factor)
        for seed in seeds:
            expected_x = geometry.center_x + (
                seed["center_x"] - geometry.center_x
            ) * scale_factor
            expected_y = geometry.center_y + (
                seed["center_y"] - geometry.center_y
            ) * scale_factor
            result = measure(transformed, expected_x, expected_y)
            error = (
                math.hypot(
                    float(result["center_x"]) - expected_x,
                    float(result["center_y"]) - expected_y,
                )
                if result.get("fit_valid")
                else None
            )
            rows.append(
                {
                    "transform": "radial_scale",
                    "parameter": f"scale={scale_factor:.2f}",
                    "branch": seed["branch"],
                    "expected_x": expected_x,
                    "expected_y": expected_y,
                    "equivariance_error_px": error,
                    **result,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_reference(
    reference: np.ndarray,
    geometry: Geometry,
    candidates: list[dict[str, float]],
    seeds: list[dict[str, float]],
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    ax.imshow(reference, cmap="gray", vmin=0.0, vmax=np.percentile(reference, 99.8))
    for index, candidate in enumerate(candidates, start=1):
        ax.scatter(candidate["x"], candidate["y"], s=22, facecolors="none", edgecolors="cyan")
        ax.text(candidate["x"] + 7, candidate["y"] - 5, str(index), color="cyan", fontsize=7)
    for seed in seeds:
        ax.scatter(
            seed["center_x"],
            seed["center_y"],
            s=120,
            facecolors="none",
            edgecolors="yellow",
            linewidths=2.0,
        )
        ax.text(
            seed["center_x"] + 10,
            seed["center_y"] + 14,
            seed["branch"],
            color="yellow",
            fontsize=10,
            weight="bold",
        )
    for radius, color in [
        (geometry.aperture_radius, "orange"),
        (geometry.screen_radius, "lime"),
    ]:
        ax.add_patch(
            plt.Circle(
                (geometry.center_x, geometry.center_y),
                radius,
                fill=False,
                color=color,
                linewidth=1.0,
            )
        )
    ax.scatter(geometry.center_x, geometry.center_y, marker="+", s=120, color="red")
    ax.set_title(
        "Automatic detections and frozen diametric pair at "
        f"{REFERENCE_ENERGY_EV:.1f} eV"
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "reference_selection.png", dpi=180)
    plt.close(fig)


STATUS_COLORS = {
    "visible_fit": "#1b9e77",
    "iv_visibility_gap": "#d95f02",
    "beam_onset_or_termination": "#7570b3",
    "apparatus_absence": "#666666",
    "geometry_inconsistent_fit": "#e7298a",
}


def plot_scan(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True)
    for branch in sorted({str(row["branch"]) for row in rows}):
        branch_rows = sorted(
            [row for row in rows if row["branch"] == branch],
            key=lambda row: float(row["energy_ev"]),
        )
        visible = [row for row in branch_rows if row["status"] == "visible_fit"]
        energies = [float(row["energy_ev"]) for row in visible]
        axes[0, 0].plot(
            energies,
            [float(row["measured_radius_px"]) for row in visible],
            marker=".",
            label=branch,
        )
        axes[0, 1].plot(
            energies,
            [float(row["radial_residual_px"]) for row in visible],
            marker=".",
            label=branch,
        )
        axes[0, 1].plot(
            energies,
            [float(row["tangential_residual_px"]) for row in visible],
            marker="x",
            linestyle="--",
            alpha=0.75,
        )
        axes[1, 0].plot(
            energies,
            [float(row["volume"]) for row in visible],
            marker=".",
            label=branch,
        )
        branch_y = 1 if branch == "B1" else 2
        for row in branch_rows:
            axes[1, 1].scatter(
                float(row["energy_ev"]),
                branch_y,
                color=STATUS_COLORS.get(str(row["status"]), "black"),
                s=18,
            )

    axes[0, 0].set_ylabel("Gaussian-refined radius [px]")
    axes[0, 0].set_title("Measured radial branch")
    axes[0, 0].legend()
    axes[0, 1].axhline(0.0, color="black", linewidth=0.7)
    axes[0, 1].set_ylabel("Residual vs 1/sqrt(E) [px]")
    axes[0, 1].set_title("Radial (dot) / tangential (cross)")
    axes[1, 0].set_ylabel("Gaussian volume [a.u.]")
    axes[1, 0].set_xlabel("Energy [eV]")
    axes[1, 0].set_title("Physical I(V) observable")
    axes[1, 1].set_yticks([1, 2], labels=["B1", "B2"])
    axes[1, 1].set_xlabel("Energy [eV]")
    axes[1, 1].set_title("Visibility / absence classification")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color,
            label=status.replace("_", " "),
        )
        for status, color in STATUS_COLORS.items()
    ]
    axes[1, 1].legend(handles=handles, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scan_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_transforms(rows: list[dict[str, object]]) -> None:
    accepted = [
        row for row in rows if row.get("equivariance_error_px") is not None
    ]
    labels = [
        f"{row['transform']}\n{row['parameter']}\n{row['branch']}" for row in accepted
    ]
    errors = [float(row["equivariance_error_px"]) for row in accepted]
    colors = [
        "#4c78a8" if row["transform"] == "translation" else "#f58518"
        for row in accepted
    ]
    fig, ax = plt.subplots(figsize=(12.0, 4.5))
    ax.bar(range(len(errors)), errors, color=colors)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Equivariance error [px]")
    ax.set_title("Fixed-ROI Gaussian measurement under exact transforms")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "transform_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_pair_diagnostics(rows: list[dict[str, object]]) -> None:
    energies = [float(row["energy_ev"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    axes[0].plot(
        energies,
        [float(row["pair_distance_px"]) for row in rows],
        marker=".",
        label="Gaussian pair distance",
    )
    axes[0].plot(
        energies,
        [float(row["predicted_distance_px"]) for row in rows],
        linestyle="--",
        label=r"$D_{ref}\sqrt{E_{ref}/E}$",
    )
    axes[0].set_ylabel("Pair distance D [px]")
    axes[0].set_xlabel("Energy [eV]")
    axes[0].set_title("Translation-invariant radial contraction")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        energies,
        [float(row["distance_residual_px"]) for row in rows],
        marker=".",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_ylabel("D residual [px]")
    axes[1].set_xlabel("Energy [eV]")
    axes[1].set_title("Pairwise energy-law residual")

    axes[2].plot(
        energies,
        [float(row["midpoint_dx_px"]) for row in rows],
        marker=".",
        label="dx",
    )
    axes[2].plot(
        energies,
        [float(row["midpoint_dy_px"]) for row in rows],
        marker="x",
        linestyle="--",
        label="dy",
    )
    axes[2].axhline(0.0, color="black", linewidth=0.7)
    axes[2].set_ylabel("Pair midpoint shift [px]")
    axes[2].set_xlabel("Energy [eV]")
    axes[2].set_title("Common-mode apparatus/projection drift")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pair_diagnostics.png", dpi=180)
    plt.close(fig)


def summarize(
    frames: list[Frame],
    reference_frame: Frame,
    geometry: Geometry,
    candidates: list[dict[str, float]],
    seeds: list[dict[str, float]],
    scan_rows: list[dict[str, object]],
    gaps: list[dict[str, object]],
    transform_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    pair_rank: int,
) -> dict[str, object]:
    transform_errors = [
        float(row["equivariance_error_px"])
        for row in transform_rows
        if row.get("equivariance_error_px") is not None
    ]
    status_counts: dict[str, int] = {}
    for row in scan_rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    visible_errors = [
        float(row["branch_error_px"])
        for row in scan_rows
        if row["status"] == "visible_fit"
    ]
    pair_residuals = [
        abs(float(row["distance_residual_px"])) for row in pair_rows
    ]
    midpoint_drifts = [
        float(row["midpoint_drift_px"]) for row in pair_rows
    ]
    return {
        "prototype_scope": (
            "Physics-scaling proposal plus common Gaussian measurement; "
            "not a tracker comparison."
        ),
        "dataset": str(DATA_DIR.relative_to(REPO_ROOT)),
        "frame_count": len(frames),
        "energy_range_ev": [frames[0].energy_ev, frames[-1].energy_ev],
        "reference_frame": reference_frame.frame,
        "reference_energy_ev": reference_frame.energy_ev,
        "geometry": {
            "center_x": geometry.center_x,
            "center_y": geometry.center_y,
            "aperture_radius_px": geometry.aperture_radius,
            "screen_radius_px": geometry.screen_radius,
        },
        "automatic_candidate_count": len(candidates),
        "frozen_independent_pair_rank": pair_rank,
        "frozen_seeds": seeds,
        "thresholds": {
            "roi_size_px": ROI_SIZE,
            "r_squared_min": R2_MIN,
            "amplitude_min": AMPLITUDE_MIN,
            "center_tolerance_px": CENTER_TOLERANCE_PX,
            "branch_tolerance_px": BRANCH_TOLERANCE_PX,
        },
        "synthetic_transform_contract": {
            "attempt_count": len(transform_rows),
            "accepted_count": len(transform_errors),
            "median_equivariance_error_px": (
                float(np.median(transform_errors)) if transform_errors else None
            ),
            "max_equivariance_error_px": (
                max(transform_errors) if transform_errors else None
            ),
        },
        "scan_status_counts": status_counts,
        "visible_branch_error_px": {
            "median": float(np.median(visible_errors)) if visible_errors else None,
            "p95": float(np.percentile(visible_errors, 95)) if visible_errors else None,
        },
        "pairwise_scan_geometry": {
            "jointly_visible_frame_count": len(pair_rows),
            "absolute_distance_residual_px": {
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
            "midpoint_drift_px": {
                "median": (
                    float(np.median(midpoint_drifts)) if midpoint_drifts else None
                ),
                "p95": (
                    float(np.percentile(midpoint_drifts, 95))
                    if midpoint_drifts
                    else None
                ),
                "max": max(midpoint_drifts) if midpoint_drifts else None,
            },
        },
        "visibility_gaps": gaps,
        "decision_checks": {
            "known_transforms_recovered": (
                bool(transform_errors)
                and len(transform_errors) == len(transform_rows)
                and max(transform_errors) < 1.5
            ),
            "coherent_pairwise_energy_law": (
                len(pair_residuals) >= 20
                and float(np.percentile(pair_residuals, 95)) < 8.0
            ),
            "natural_iv_gap_available": any(
                gap["classification"] == "iv_visibility_gap" for gap in gaps
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the annotation-free ScanE physics baseline."
    )
    parser.add_argument(
        "--pair-rank",
        type=int,
        default=1,
        help="One-based rank among greedily selected independent diametric pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory; defaults to the prototype output directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional ScanE directory; defaults to data/26a24_ScanE.",
    )
    return parser.parse_args()


def main() -> None:
    global DATA_DIR, OUTPUT_DIR, REFERENCE_ENERGY_EV
    args = parse_args()
    if args.data_dir is not None:
        DATA_DIR = args.data_dir.expanduser().resolve()
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir.expanduser().resolve()
    frames = load_frames()
    reference_frame = min(
        frames,
        key=lambda item: abs(
            item.energy_ev - REFERENCE_ENERGY_TARGET_EV
        ),
    )
    REFERENCE_ENERGY_EV = reference_frame.energy_ev
    reference = read_float(reference_frame.path)
    geometry = estimate_geometry(reference)
    available = apparatus_mask(reference, geometry)
    candidates = detect_reference_spots(reference, geometry, available)
    selected_pair = select_diametric_pair(
        candidates,
        geometry,
        minimum_energy_ev=frames[0].energy_ev,
        pair_rank=int(args.pair_rank),
    )
    seeds = gaussian_refine_seeds(reference, selected_pair)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    scan_rows, gaps = run_scan(frames, seeds, geometry, available)
    pair_rows = build_pair_diagnostics(scan_rows, seeds)
    transform_rows = run_transform_checks(reference, seeds, geometry)
    summary = summarize(
        frames,
        reference_frame,
        geometry,
        candidates,
        seeds,
        scan_rows,
        gaps,
        transform_rows,
        pair_rows,
        int(args.pair_rank),
    )

    write_csv(OUTPUT_DIR / "scan_measurements.csv", scan_rows)
    write_csv(OUTPUT_DIR / "pair_measurements.csv", pair_rows)
    write_csv(OUTPUT_DIR / "transform_measurements.csv", transform_rows)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_reference(reference, geometry, candidates, seeds)
    plot_scan(scan_rows)
    plot_pair_diagnostics(pair_rows)
    plot_transforms(transform_rows)

    print(json.dumps(summary["decision_checks"], indent=2))
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
