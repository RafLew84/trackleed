#!/usr/bin/env python3
"""Run frozen Cu(111) trackers through one common Gaussian measurement chain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw
from skimage.util import img_as_float


ARTIFACT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ARTIFACT_ROOT.parents[3]
GAUSSIAN_FITTER_SOURCE = (
    REPO_ROOT / "trackleed" / "analysis" / "peak_analyzer.py"
)
sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.peak_analyzer import fit_spot_to_gaussian  # noqa: E402
METHODS = (
    "locotrack_base_offline",
    "ostrack_vitb_256_mae_ce",
    "dam4sam_sam21pp_l",
    "lucas_kanade_skip_anchor",
    "easyleed_kalman",
)
CPU_METHODS = (
    "lucas_kanade_skip_anchor",
    "easyleed_kalman",
)
GPU_METHODS = (
    "locotrack_base_offline",
    "ostrack_vitb_256_mae_ce",
    "dam4sam_sam21pp_l",
)
SPOTS = (1, 3, 5, 6)
PAIRS = ((1, 6), (3, 5))
FRAME_COUNT = 701
ROI_SIZE = 48
ROI_HALF = ROI_SIZE // 2
MINIMUM_REFERENCE_COUNT = 8
OUTPUT_FILES = {
    "run_audit_rows": "run_audit.csv",
    "native_proposal_rows": "native_proposals.csv",
    "gaussian_measurement_rows": "gaussian_measurements.csv",
    "pair_endpoint_rows": "pair_endpoints.csv",
}
FIT_RESULT_FIELDS = (
    "global_center_x",
    "global_center_y",
    "center_x_error",
    "center_y_error",
    "amplitude",
    "sigma_x",
    "sigma_y",
    "r_squared",
    "proposal_offset_px",
)


def summarize_distribution(
    values_by_frame: list[tuple[int, float]],
) -> dict[str, object]:
    if not values_by_frame:
        return {
            "count": 0,
            "median": None,
            "p95_linear": None,
            "maximum": None,
            "maximum_frame_order": None,
        }
    values = np.asarray(
        [value for _, value in values_by_frame], dtype=np.float64
    )
    maximum_index = int(np.argmax(values))
    return {
        "count": len(values_by_frame),
        "median": float(np.median(values)),
        "p95_linear": float(np.percentile(values, 95)),
        "maximum": float(values[maximum_index]),
        "maximum_frame_order": int(values_by_frame[maximum_index][0]),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def csv_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite values must be represented as blanks")
        return format(value, ".17g")
    return value


def write_csv(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: csv_value(row.get(field, "")) for field in fieldnames}
            )


def measure_proposal(
    raw_float_image: np.ndarray,
    proposal_x: float,
    proposal_y: float,
) -> dict[str, object]:
    """Apply the frozen raw-image crop, Gaussian fit and acceptance gates."""

    if raw_float_image.ndim != 2:
        raise ValueError("The common fit requires a two-dimensional image")
    image_height, image_width = raw_float_image.shape
    x0 = int(round(float(proposal_x))) - ROI_HALF
    y0 = int(round(float(proposal_y))) - ROI_HALF
    result: dict[str, object] = {
        "proposal_available": True,
        "fit_valid": False,
        "roi_x0_original_px": x0,
        "roi_y0_original_px": y0,
        "roi_size_px": ROI_SIZE,
        **{field: "" for field in FIT_RESULT_FIELDS},
        "rejection_reason": "",
    }
    if (
        x0 < 0
        or y0 < 0
        or x0 + ROI_SIZE > image_width
        or y0 + ROI_SIZE > image_height
    ):
        result["rejection_reason"] = "roi_outside_image"
        return result

    roi = raw_float_image[y0 : y0 + ROI_SIZE, x0 : x0 + ROI_SIZE]
    fit = fit_spot_to_gaussian(roi, center_preference_factor=None)
    if fit is None:
        result["rejection_reason"] = "fit_failed"
        return result

    diagnostics = {
        "global_center_x": x0 + float(fit["center_x"]),
        "global_center_y": y0 + float(fit["center_y"]),
        "center_x_error": float(fit["center_x_error"]),
        "center_y_error": float(fit["center_y_error"]),
        "amplitude": float(fit["amplitude"]),
        "sigma_x": float(fit["sigma_x"]),
        "sigma_y": float(fit["sigma_y"]),
        "r_squared": float(fit["r_squared"]),
    }
    diagnostics["proposal_offset_px"] = math.hypot(
        diagnostics["global_center_x"] - float(proposal_x),
        diagnostics["global_center_y"] - float(proposal_y),
    )
    if (
        not all(math.isfinite(value) for value in diagnostics.values())
        or diagnostics["center_x_error"] < 0.0
        or diagnostics["center_y_error"] < 0.0
    ):
        result["rejection_reason"] = "fit_failed"
        return result

    result.update(diagnostics)
    reasons: list[str] = []
    local_x = float(fit["center_x"])
    local_y = float(fit["center_y"])
    if diagnostics["r_squared"] < 0.85:
        reasons.append("low_r_squared")
    if diagnostics["amplitude"] < 0.05:
        reasons.append("low_amplitude")
    if not (-2.0 < local_x < ROI_SIZE + 2.0):
        reasons.append("center_x_outside")
    if not (-2.0 < local_y < ROI_SIZE + 2.0):
        reasons.append("center_y_outside")
    if not (0.5 < diagnostics["sigma_x"] < ROI_SIZE):
        reasons.append("sigma_x_outside")
    if not (0.5 < diagnostics["sigma_y"] < ROI_SIZE):
        reasons.append("sigma_y_outside")
    result["fit_valid"] = not reasons
    result["rejection_reason"] = "accepted" if not reasons else "+".join(reasons)
    return result


def unavailable_measurement() -> dict[str, object]:
    return {
        "proposal_available": False,
        "fit_valid": False,
        "roi_x0_original_px": "",
        "roi_y0_original_px": "",
        "roi_size_px": "",
        **{field: "" for field in FIT_RESULT_FIELDS},
        "rejection_reason": "proposal_unavailable",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected frozen Cu(111) methods and apply the same raw-image "
            "Gaussian measurement and pairwise delta R/R."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(CPU_METHODS),
    )
    parser.add_argument(
        "--easyleed-native-dir",
        type=Path,
        help="Full output directory created by run_easyleed_cu.py.",
    )
    parser.add_argument(
        "--gpu-native-root",
        type=Path,
        help=(
            "Directory containing one subdirectory per GPU method, each "
            "created by run_gpu_cu.py."
        ),
    )
    return parser.parse_args()


def load_contract() -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    manifest = json.loads(
        (ARTIFACT_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (ARTIFACT_ROOT / "output_contract.json").read_text(encoding="utf-8")
    )
    frames = read_csv(ARTIFACT_ROOT / "inference_frames.csv")
    seeds = read_csv(ARTIFACT_ROOT / "seed_prompts.csv")
    if manifest["status"] != "frozen_ready_for_runner":
        raise RuntimeError("Contract manifest is not frozen for execution")
    if len(frames) != FRAME_COUNT:
        raise RuntimeError(f"Expected 701 frames, found {len(frames)}")
    if [int(row["frame_order"]) for row in frames] != list(range(FRAME_COUNT)):
        raise RuntimeError("Inference inventory is not the exact range 0..700")
    if tuple(int(row["spot_id"]) for row in seeds) != SPOTS:
        raise RuntimeError("Seed inventory must contain spots 1, 3, 5 and 6")
    for filename in (
        "inference_frames.csv",
        "seed_prompts.csv",
        "output_contract.json",
        "run_gpu_cu.py",
        "run_headless.py",
    ):
        if sha256_file(ARTIFACT_ROOT / filename) != manifest["files"][filename]:
            raise RuntimeError(f"Frozen contract hash mismatch: {filename}")
    measurement = manifest["measurement"]
    if REPO_ROOT / measurement["fit_source"] != GAUSSIAN_FITTER_SOURCE:
        raise RuntimeError("Unexpected common Gaussian fitter source")
    if (
        sha256_file(GAUSSIAN_FITTER_SOURCE)
        != measurement["fit_source_sha256"]
    ):
        raise RuntimeError("Frozen common Gaussian fitter hash mismatch")
    return manifest, contract, frames, seeds


def canonical_selected_methods(requested: Iterable[str]) -> tuple[str, ...]:
    requested_set = set(requested)
    if not requested_set:
        raise ValueError("At least one method is required")
    unknown = requested_set.difference(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    return tuple(method for method in METHODS if method in requested_set)


def relative_to_output(path: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), output_dir.resolve())).as_posix()


def save_native_payload(
    path: Path,
    frame_orders: np.ndarray,
    frame_ids: np.ndarray,
    x_cache_px: np.ndarray,
    y_cache_px: np.ndarray,
    x_original_px: np.ndarray,
    y_original_px: np.ndarray,
    proposal_available: np.ndarray,
    runtime_seconds: np.ndarray,
    native_status: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_order=frame_orders.astype(np.int32),
        frame_id=frame_ids.astype(np.int32),
        x_cache_px=x_cache_px.astype(np.float64),
        y_cache_px=y_cache_px.astype(np.float64),
        x_original_px=x_original_px.astype(np.float64),
        y_original_px=y_original_px.astype(np.float64),
        proposal_available=proposal_available.astype(np.bool_),
        runtime_seconds=runtime_seconds.astype(np.float64),
        native_status=native_status.astype(np.str_),
    )


def run_lucas_kanade(
    output_dir: Path,
    manifest: dict[str, Any],
    frames: list[dict[str, str]],
    seeds: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]]]:
    import cv2
    from trackleed.analysis.tracking import optical_flow_with_skip_anchor

    method = "lucas_kanade_skip_anchor"
    settings = manifest["method_configurations"][method]
    cache_frames: list[np.ndarray] = []
    for row in frames:
        path = REPO_ROOT / row["cache_relative_path"]
        if sha256_file(path) != row["cache_sha256"]:
            raise RuntimeError(
                f"LK cache hash mismatch at frame {row['frame_order']}"
            )
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8)
        if array.shape != (256, 256):
            raise RuntimeError(
                f"Unexpected LK cache shape {array.shape} at frame "
                f"{row['frame_order']}"
            )
        cache_frames.append(array)
    points = [
        (float(seed["center_x_cache_px"]), float(seed["center_y_cache_px"]))
        for seed in seeds
    ]
    criteria = settings["termination"]
    started = time.perf_counter()
    tracks = optical_flow_with_skip_anchor(
        frames=cache_frames,
        points0=points,
        lk_params={
            "winSize": tuple(settings["win_size"]),
            "maxLevel": int(settings["max_level"]),
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(criteria["max_iterations"]),
                float(criteria["epsilon"]),
            ),
            "minEigThreshold": float(settings["min_eigen_threshold"]),
        },
        err_threshold=float(settings["error_threshold"]),
        fb_threshold=float(settings["forward_backward_threshold"]),
        use_forward_backward=bool(settings["forward_backward_check"]),
        max_skip_frames_per_point=int(settings["max_skip_frames"]),
        oob_behavior=str(settings["out_of_bounds_behavior"]),
        use_ncc_check=bool(settings["ncc_check"]),
        use_centroid_refinement=bool(settings["centroid_refinement"]),
        roi_size=ROI_SIZE,
    )
    elapsed = time.perf_counter() - started
    if tracks is None or len(tracks) != len(SPOTS):
        raise RuntimeError("Lucas-Kanade did not return four tracks")

    scale = float(
        manifest["input_transform"]["coordinate_scale_original_to_cache"]
    )
    amortized_runtime = elapsed / (FRAME_COUNT * len(SPOTS))
    native_rows: list[dict[str, object]] = []
    payload_info: dict[int, dict[str, object]] = {}
    frame_orders = np.arange(FRAME_COUNT, dtype=np.int32)
    frame_ids = np.asarray(
        [int(row["frame_id"]) for row in frames], dtype=np.int32
    )
    for spot_id, positions in zip(SPOTS, tracks):
        x_cache = np.full(FRAME_COUNT, np.nan, dtype=np.float64)
        y_cache = np.full(FRAME_COUNT, np.nan, dtype=np.float64)
        available = np.zeros(FRAME_COUNT, dtype=np.bool_)
        for frame_order, x_value, y_value in positions:
            x_cache[int(frame_order)] = float(x_value)
            y_cache[int(frame_order)] = float(y_value)
            available[int(frame_order)] = True
        x_original = x_cache / scale
        y_original = y_cache / scale
        runtimes = np.full(FRAME_COUNT, amortized_runtime, dtype=np.float64)
        statuses = np.where(available, "proposal", "lk_gate_rejected")
        payload_path = (
            output_dir / "native" / method / f"{method}_spot_{spot_id:02d}.npz"
        )
        save_native_payload(
            payload_path,
            frame_orders,
            frame_ids,
            x_cache,
            y_cache,
            x_original,
            y_original,
            available,
            runtimes,
            statuses,
        )
        payload_info[spot_id] = {
            "path": payload_path,
            "sha256": sha256_file(payload_path),
            "proposal_count": int(available.sum()),
            "elapsed_seconds": elapsed,
            "terminal_error": None,
        }
        relative_payload = relative_to_output(payload_path, output_dir)
        for frame_order, frame_id in zip(frame_orders, frame_ids):
            index = int(frame_order)
            is_available = bool(available[index])
            native_rows.append(
                {
                    "method": method,
                    "frame_order": index,
                    "frame_id": int(frame_id),
                    "spot_id": spot_id,
                    "proposal_available": is_available,
                    "native_visibility_signal_name": "lk_gate_passed",
                    "native_visibility_signal": 1.0 if is_available else 0.0,
                    "native_payload_path": relative_payload,
                    "proposal_reduction_rule": settings[
                        "proposal_reduction_rule"
                    ],
                    "measurement_center_x_original_px": (
                        float(x_original[index]) if is_available else ""
                    ),
                    "measurement_center_y_original_px": (
                        float(y_original[index]) if is_available else ""
                    ),
                    "runtime_seconds": float(runtimes[index]),
                }
            )
    return native_rows, payload_info


def load_easyleed(
    output_dir: Path,
    native_dir: Path,
    manifest: dict[str, Any],
    frames: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]]]:
    method = "easyleed_kalman"
    settings = manifest["method_configurations"][method]
    summary_path = native_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["publication_run"] is not True:
        raise RuntimeError("EasyLEED input is a smoke run, not a full run")
    if summary["status"] not in {
        "completed",
        "completed_with_tracker_failures",
    }:
        raise RuntimeError(f"Unexpected EasyLEED status: {summary['status']}")
    if summary["frame_count"] != FRAME_COUNT:
        raise RuntimeError("EasyLEED native output does not contain 701 frames")
    if summary["method_configuration_sha256"] != manifest[
        "method_configuration_sha256"
    ][method]:
        raise RuntimeError("EasyLEED method-configuration hash mismatch")
    if summary["adapter_source_sha256"] != settings["adapter_source_sha256"]:
        raise RuntimeError("EasyLEED adapter-source hash mismatch")
    if summary.get("easyleed_version") != settings["version"]:
        raise RuntimeError("EasyLEED version mismatch")
    if summary.get("numpy_version") != settings["numpy_version"]:
        raise RuntimeError("EasyLEED NumPy version mismatch")
    if summary["corrections_or_reinitializations"] != 0:
        raise RuntimeError("EasyLEED output reports a forbidden reset")

    payload_by_spot = {
        int(item["spot_id"]): item for item in summary["payloads"]
    }
    if tuple(sorted(payload_by_spot)) != SPOTS:
        raise RuntimeError("EasyLEED output must contain the four frozen spots")
    native_rows: list[dict[str, object]] = []
    payload_info: dict[int, dict[str, object]] = {}
    expected_orders = np.arange(FRAME_COUNT, dtype=np.int32)
    expected_ids = np.asarray(
        [int(row["frame_id"]) for row in frames], dtype=np.int32
    )
    for spot_id in SPOTS:
        item = payload_by_spot[spot_id]
        payload_path = native_dir / item["path"]
        if sha256_file(payload_path) != item["sha256"]:
            raise RuntimeError(f"EasyLEED payload hash mismatch for spot {spot_id}")
        with np.load(payload_path, allow_pickle=False) as payload:
            frame_orders = payload["frame_order"]
            frame_ids = payload["frame_id"]
            available = payload["proposal_available"].astype(bool)
            x_original = payload["x_original_px"].astype(np.float64)
            y_original = payload["y_original_px"].astype(np.float64)
            runtimes = payload["runtime_seconds"].astype(np.float64)
        if not np.array_equal(frame_orders, expected_orders):
            raise RuntimeError(f"EasyLEED frame order mismatch for spot {spot_id}")
        if not np.array_equal(frame_ids, expected_ids):
            raise RuntimeError(f"EasyLEED frame ids mismatch for spot {spot_id}")
        if any(
            len(values) != FRAME_COUNT
            for values in (available, x_original, y_original, runtimes)
        ):
            raise RuntimeError(f"Truncated EasyLEED payload for spot {spot_id}")
        if not np.isfinite(runtimes).all() or np.any(runtimes < 0.0):
            raise RuntimeError(f"Invalid EasyLEED runtimes for spot {spot_id}")
        finite_positions = np.isfinite(x_original) & np.isfinite(y_original)
        if not np.array_equal(available, finite_positions):
            raise RuntimeError(
                f"EasyLEED availability does not match positions for spot {spot_id}"
            )
        payload_info[spot_id] = {
            "path": payload_path,
            "sha256": item["sha256"],
            "proposal_count": int(available.sum()),
            "elapsed_seconds": float(summary["elapsed_seconds"]),
            "terminal_error": item["terminal_error"],
        }
        relative_payload = relative_to_output(payload_path, output_dir)
        for index in range(FRAME_COUNT):
            is_available = bool(available[index])
            native_rows.append(
                {
                    "method": method,
                    "frame_order": index,
                    "frame_id": int(frame_ids[index]),
                    "spot_id": spot_id,
                    "proposal_available": is_available,
                    "native_visibility_signal_name": "finite_kalman_position",
                    "native_visibility_signal": 1.0 if is_available else 0.0,
                    "native_payload_path": relative_payload,
                    "proposal_reduction_rule": settings[
                        "proposal_reduction_rule"
                    ],
                    "measurement_center_x_original_px": (
                        float(x_original[index]) if is_available else ""
                    ),
                    "measurement_center_y_original_px": (
                        float(y_original[index]) if is_available else ""
                    ),
                    "runtime_seconds": float(runtimes[index]),
                }
            )
    return native_rows, payload_info


def load_gpu_native(
    output_dir: Path,
    native_dir: Path,
    method: str,
    manifest: dict[str, Any],
    frames: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]]]:
    if method not in GPU_METHODS:
        raise ValueError(f"Not a frozen GPU method: {method}")
    settings = manifest["method_configurations"][method]
    summary_path = native_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("publication_run") is not True:
        raise RuntimeError(f"{method} input is not a full publication run")
    if summary.get("status") != "completed":
        raise RuntimeError(
            f"Unexpected {method} native status: {summary.get('status')}"
        )
    if summary.get("method") != method:
        raise RuntimeError(
            f"GPU native method mismatch: {summary.get('method')} != {method}"
        )
    if summary.get("frame_count") != FRAME_COUNT:
        raise RuntimeError(f"{method} native output does not contain 701 frames")
    if summary.get("method_configuration_sha256") != manifest[
        "method_configuration_sha256"
    ][method]:
        raise RuntimeError(f"{method} method-configuration hash mismatch")
    if summary.get("adapter_source_sha256") != settings[
        "adapter_source_sha256"
    ]:
        raise RuntimeError(f"{method} adapter-source hash mismatch")
    if summary.get("producer_source_sha256") != manifest["files"][
        "run_gpu_cu.py"
    ]:
        raise RuntimeError(f"{method} GPU producer-source hash mismatch")
    if summary.get("corrections_or_reinitializations") != 0:
        raise RuntimeError(f"{method} output reports a forbidden reset")
    if summary.get("proposal_reduction_rule") != settings[
        "proposal_reduction_rule"
    ]:
        raise RuntimeError(f"{method} proposal-reduction rule mismatch")

    payload_by_spot = {
        int(item["spot_id"]): item for item in summary["payloads"]
    }
    if tuple(sorted(payload_by_spot)) != SPOTS:
        raise RuntimeError(f"{method} output must contain four frozen spots")
    native_rows: list[dict[str, object]] = []
    payload_info: dict[int, dict[str, object]] = {}
    expected_orders = np.arange(FRAME_COUNT, dtype=np.int32)
    expected_ids = np.asarray(
        [int(row["frame_id"]) for row in frames],
        dtype=np.int32,
    )
    signal_name = str(summary["native_visibility_signal_name"])
    for spot_id in SPOTS:
        item = payload_by_spot[spot_id]
        payload_path = native_dir / item["path"]
        if sha256_file(payload_path) != item["sha256"]:
            raise RuntimeError(
                f"{method} payload hash mismatch for spot {spot_id}"
            )
        with np.load(payload_path, allow_pickle=False) as payload:
            frame_orders = payload["frame_order"]
            frame_ids = payload["frame_id"]
            available = payload["proposal_available"].astype(bool)
            x_original = payload["x_original_px"].astype(np.float64)
            y_original = payload["y_original_px"].astype(np.float64)
            runtimes = payload["runtime_seconds"].astype(np.float64)
            visibility = payload["native_visibility_signal"].astype(
                np.float64
            )
        if not np.array_equal(frame_orders, expected_orders):
            raise RuntimeError(
                f"{method} frame order mismatch for spot {spot_id}"
            )
        if not np.array_equal(frame_ids, expected_ids):
            raise RuntimeError(f"{method} frame ids mismatch for spot {spot_id}")
        if any(
            len(values) != FRAME_COUNT
            for values in (
                available,
                x_original,
                y_original,
                runtimes,
                visibility,
            )
        ):
            raise RuntimeError(f"Truncated {method} payload for spot {spot_id}")
        if not np.isfinite(runtimes).all() or np.any(runtimes < 0.0):
            raise RuntimeError(f"Invalid {method} runtimes for spot {spot_id}")
        finite_positions = np.isfinite(x_original) & np.isfinite(y_original)
        if not np.array_equal(available, finite_positions):
            raise RuntimeError(
                f"{method} availability does not match stored positions "
                f"for spot {spot_id}"
            )
        if int(available.sum()) != int(item["proposal_count"]):
            raise RuntimeError(
                f"{method} proposal count mismatch for spot {spot_id}"
            )
        payload_info[spot_id] = {
            "path": payload_path,
            "sha256": item["sha256"],
            "proposal_count": int(available.sum()),
            "elapsed_seconds": float(summary["elapsed_seconds"]),
            "terminal_error": item.get("terminal_error"),
        }
        relative_payload = relative_to_output(payload_path, output_dir)
        for index in range(FRAME_COUNT):
            is_available = bool(available[index])
            signal = float(visibility[index])
            native_rows.append(
                {
                    "method": method,
                    "frame_order": index,
                    "frame_id": int(frame_ids[index]),
                    "spot_id": spot_id,
                    "proposal_available": is_available,
                    "native_visibility_signal_name": signal_name,
                    "native_visibility_signal": (
                        signal if math.isfinite(signal) else ""
                    ),
                    "native_payload_path": relative_payload,
                    "proposal_reduction_rule": settings[
                        "proposal_reduction_rule"
                    ],
                    "measurement_center_x_original_px": (
                        float(x_original[index]) if is_available else ""
                    ),
                    "measurement_center_y_original_px": (
                        float(y_original[index]) if is_available else ""
                    ),
                    "runtime_seconds": float(runtimes[index]),
                }
            )
    return native_rows, payload_info


def build_audit_rows(
    output_dir: Path,
    methods: tuple[str, ...],
    manifest: dict[str, Any],
    payloads: dict[tuple[str, int], dict[str, object]],
) -> list[dict[str, object]]:
    runner_path = Path(__file__).resolve()
    runner_relative = relative_to_output(runner_path, output_dir)
    runner_hash = sha256_file(runner_path)
    rows: list[dict[str, object]] = []
    for method in methods:
        configuration = manifest["method_configurations"][method]
        for spot_id in SPOTS:
            payload = payloads[(method, spot_id)]
            terminal_error = payload.get("terminal_error")
            rows.append(
                {
                    "method": method,
                    "spot_id": spot_id,
                    "terminal_status": (
                        "completed_with_native_failure"
                        if terminal_error
                        else "completed"
                    ),
                    "processed_frame_count": FRAME_COUNT,
                    "first_frame_order": 0,
                    "last_frame_order": FRAME_COUNT - 1,
                    "reset_count": 0,
                    "seed_prompts_sha256": manifest["files"][
                        "seed_prompts.csv"
                    ],
                    "input_inventory_sha256": manifest["files"][
                        "inference_frames.csv"
                    ],
                    "method_configuration_sha256": manifest[
                        "method_configuration_sha256"
                    ][method],
                    "adapter_source_sha256": configuration[
                        "adapter_source_sha256"
                    ],
                    "runner_source_path": runner_relative,
                    "runner_source_sha256": runner_hash,
                    "native_output_sha256": payload["sha256"],
                    "error_message": terminal_error or "",
                }
            )
    return rows


def build_measurement_rows(
    methods: tuple[str, ...],
    frames: list[dict[str, str]],
    native_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    native_by_key = {
        (
            str(row["method"]),
            int(row["frame_order"]),
            int(row["spot_id"]),
        ): row
        for row in native_rows
    }
    rows: list[dict[str, object]] = []
    for frame_index, frame in enumerate(frames):
        raw_path = REPO_ROOT / frame["raw_relative_path"]
        if sha256_file(raw_path) != frame["raw_sha256"]:
            raise RuntimeError(f"Raw TIFF hash mismatch at frame {frame_index}")
        with Image.open(raw_path) as image:
            raw = np.asarray(image)
        if raw.shape != (1000, 1000) or raw.dtype != np.uint8:
            raise RuntimeError(
                f"Expected 1000x1000 uint8 TIFF at frame {frame_index}, "
                f"found {raw.shape} {raw.dtype}"
            )
        raw_float = img_as_float(raw)
        for method in methods:
            for spot_id in SPOTS:
                native = native_by_key[(method, frame_index, spot_id)]
                if not bool(native["proposal_available"]):
                    measurement = unavailable_measurement()
                else:
                    measurement = measure_proposal(
                        raw_float,
                        float(native["measurement_center_x_original_px"]),
                        float(native["measurement_center_y_original_px"]),
                    )
                rows.append(
                    {
                        "method": method,
                        "frame_order": frame_index,
                        "frame_id": int(frame["frame_id"]),
                        "spot_id": spot_id,
                        **measurement,
                    }
                )
        if (
            frame_index == 0
            or (frame_index + 1) % 25 == 0
            or frame_index + 1 == len(frames)
        ):
            valid_count = sum(bool(row["fit_valid"]) for row in rows)
            print(
                f"[common-fit] frames {frame_index + 1}/{len(frames)} | "
                f"valid fits {valid_count}/{len(rows)}",
                flush=True,
            )
    return rows


def build_pair_rows(
    methods: tuple[str, ...],
    measurement_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_key = {
        (
            str(row["method"]),
            int(row["frame_order"]),
            int(row["spot_id"]),
        ): row
        for row in measurement_rows
    }
    pair_rows: list[dict[str, object]] = []
    for method in methods:
        for first_spot, second_spot in PAIRS:
            pair_name = f"{first_spot}-{second_spot}"
            reference_distances: list[float] = []
            for frame_order in range(10):
                first = by_key[(method, frame_order, first_spot)]
                second = by_key[(method, frame_order, second_spot)]
                if not (bool(first["fit_valid"]) and bool(second["fit_valid"])):
                    continue
                reference_distances.append(
                    math.hypot(
                        float(first["global_center_x"])
                        - float(second["global_center_x"]),
                        float(first["global_center_y"])
                        - float(second["global_center_y"]),
                    )
                )
            reference_count = len(reference_distances)
            reference = (
                statistics.median(reference_distances)
                if reference_count >= MINIMUM_REFERENCE_COUNT
                else None
            )
            for frame_order in range(FRAME_COUNT):
                first = by_key[(method, frame_order, first_spot)]
                second = by_key[(method, frame_order, second_spot)]
                pair_valid = (
                    reference is not None
                    and bool(first["fit_valid"])
                    and bool(second["fit_valid"])
                )
                if pair_valid:
                    distance = math.hypot(
                        float(first["global_center_x"])
                        - float(second["global_center_x"]),
                        float(first["global_center_y"])
                        - float(second["global_center_y"]),
                    )
                    delta = distance / float(reference) - 1.0
                    numeric: dict[str, object] = {
                        "distance_px": distance,
                        "reference_distance_px": float(reference),
                        "delta_r_over_r": delta,
                        "delta_r_over_r_percent": 100.0 * delta,
                    }
                    reason = "accepted"
                else:
                    numeric = {
                        "distance_px": "",
                        "reference_distance_px": "",
                        "delta_r_over_r": "",
                        "delta_r_over_r_percent": "",
                    }
                    reason = (
                        "insufficient_reference_coverage"
                        if reference is None
                        else "missing_pair_measurement"
                    )
                pair_rows.append(
                    {
                        "method": method,
                        "frame_order": frame_order,
                        "frame_id": frame_order,
                        "pair": pair_name,
                        "pair_valid": pair_valid,
                        **numeric,
                        "reference_valid_frame_count": reference_count,
                        "rejection_reason": reason,
                    }
                )
    return pair_rows


def render_diagnostic_overlays(
    output_dir: Path,
    methods: tuple[str, ...],
    frames: list[dict[str, str]],
    native_rows: list[dict[str, object]],
    measurement_rows: list[dict[str, object]],
) -> dict[str, object]:
    colors = {
        "locotrack_base_offline": (255, 170, 0),
        "ostrack_vitb_256_mae_ce": (40, 210, 40),
        "dam4sam_sam21pp_l": (255, 60, 60),
        "lucas_kanade_skip_anchor": (0, 220, 255),
        "easyleed_kalman": (255, 0, 255),
    }
    short_names = {
        "locotrack_base_offline": "Loco",
        "ostrack_vitb_256_mae_ce": "OST",
        "dam4sam_sam21pp_l": "DAM",
        "lucas_kanade_skip_anchor": "LK",
        "easyleed_kalman": "Easy",
    }
    native_by_key = {
        (
            str(row["method"]),
            int(row["frame_order"]),
            int(row["spot_id"]),
        ): row
        for row in native_rows
    }
    measurement_by_key = {
        (
            str(row["method"]),
            int(row["frame_order"]),
            int(row["spot_id"]),
        ): row
        for row in measurement_rows
    }
    probe_rows = [
        row for row in frames if row["diagnostic_probe"] == "true"
    ]
    if len(probe_rows) != 24:
        raise RuntimeError(f"Expected 24 diagnostic probes, found {len(probe_rows)}")
    overlay_dir = output_dir / "diagnostic_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    file_hashes: dict[str, str] = {}
    for frame in probe_rows:
        frame_order = int(frame["frame_order"])
        raw_path = REPO_ROOT / frame["raw_relative_path"]
        with Image.open(raw_path) as source:
            canvas = source.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        legend_x = 12
        for method_index, method in enumerate(methods):
            color = colors[method]
            legend_y = 12 + method_index * 20
            draw.rectangle(
                (legend_x, legend_y, legend_x + 12, legend_y + 12),
                fill=color,
            )
            draw.text(
                (legend_x + 18, legend_y),
                f"{short_names[method]}: circle=proposal, cross=Gaussian",
                fill=color,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
            for spot_id in SPOTS:
                native = native_by_key[(method, frame_order, spot_id)]
                measurement = measurement_by_key[
                    (method, frame_order, spot_id)
                ]
                if bool(native["proposal_available"]):
                    proposal_x = float(
                        native["measurement_center_x_original_px"]
                    )
                    proposal_y = float(
                        native["measurement_center_y_original_px"]
                    )
                    radius = 10 + method_index * 2
                    draw.ellipse(
                        (
                            proposal_x - radius,
                            proposal_y - radius,
                            proposal_x + radius,
                            proposal_y + radius,
                        ),
                        outline=color,
                        width=2,
                    )
                    draw.text(
                        (
                            proposal_x + radius + 2,
                            proposal_y
                            - radius
                            - 16
                            + method_index * 28,
                        ),
                        f"{short_names[method]}:{spot_id}",
                        fill=color,
                        stroke_width=2,
                        stroke_fill=(0, 0, 0),
                    )
                if bool(measurement["fit_valid"]):
                    center_x = float(measurement["global_center_x"])
                    center_y = float(measurement["global_center_y"])
                    draw.line(
                        (center_x - 6, center_y, center_x + 6, center_y),
                        fill=color,
                        width=2,
                    )
                    draw.line(
                        (center_x, center_y - 6, center_x, center_y + 6),
                        fill=color,
                        width=2,
                    )
                    if bool(native["proposal_available"]):
                        draw.line(
                            (
                                proposal_x,
                                proposal_y,
                                center_x,
                                center_y,
                            ),
                            fill=color,
                            width=1,
                        )
        filename = f"frame_{frame_order:04d}.png"
        output_path = overlay_dir / filename
        canvas.save(output_path, format="PNG", optimize=True)
        file_hash = sha256_file(output_path)
        file_hashes[filename] = file_hash
        manifest_rows.append(
            {
                "frame_order": frame_order,
                "frame_id": int(frame["frame_id"]),
                "temperature_c": float(frame["temperature_c"]),
                "filename": filename,
                "sha256": file_hash,
                "selected_methods": ",".join(methods),
                "spot_ids": ",".join(str(spot) for spot in SPOTS),
                "manual_annotations": 0,
            }
        )
    overlay_manifest_path = overlay_dir / "manifest.csv"
    write_csv(
        overlay_manifest_path,
        (
            "frame_order",
            "frame_id",
            "temperature_c",
            "filename",
            "sha256",
            "selected_methods",
            "spot_ids",
            "manual_annotations",
        ),
        manifest_rows,
    )
    return {
        "count": len(manifest_rows),
        "directory": relative_to_output(overlay_dir, output_dir),
        "manifest": relative_to_output(overlay_manifest_path, output_dir),
        "manifest_sha256": sha256_file(overlay_manifest_path),
        "files_sha256": file_hashes,
    }


def build_summary(
    methods: tuple[str, ...],
    native_rows: list[dict[str, object]],
    measurement_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    payloads: dict[tuple[str, int], dict[str, object]],
) -> dict[str, object]:
    summary_methods: dict[str, object] = {}
    for method in methods:
        method_native = [row for row in native_rows if row["method"] == method]
        method_measurements = [
            row for row in measurement_rows if row["method"] == method
        ]
        method_pairs = [row for row in pair_rows if row["method"] == method]
        pair_summary: dict[str, object] = {}
        for pair_name in ("1-6", "3-5"):
            selected = [row for row in method_pairs if row["pair"] == pair_name]
            valid = [row for row in selected if bool(row["pair_valid"])]
            plateau = [
                float(row["delta_r_over_r_percent"])
                for row in valid
                if 372 <= int(row["frame_order"]) <= 376
            ]
            minimum = (
                min(
                    valid,
                    key=lambda row: float(row["delta_r_over_r_percent"]),
                )
                if valid
                else None
            )
            pair_summary[pair_name] = {
                "valid_count": len(valid),
                "reference_valid_frame_count": int(
                    selected[0]["reference_valid_frame_count"]
                ),
                "reference_distance_px": (
                    float(valid[0]["reference_distance_px"])
                    if valid
                    else None
                ),
                "plateau_372_376_median_delta_r_over_r_percent": (
                    float(statistics.median(plateau)) if plateau else None
                ),
                "minimum_frame_order": (
                    int(minimum["frame_order"]) if minimum else None
                ),
                "minimum_delta_r_over_r_percent": (
                    float(minimum["delta_r_over_r_percent"])
                    if minimum
                    else None
                ),
                "final_delta_r_over_r_percent": (
                    float(valid[-1]["delta_r_over_r_percent"])
                    if valid and int(valid[-1]["frame_order"]) == 700
                    else None
                ),
            }
        summary_methods[method] = {
            "native_proposal_count": sum(
                bool(row["proposal_available"]) for row in method_native
            ),
            "valid_gaussian_fit_count": sum(
                bool(row["fit_valid"]) for row in method_measurements
            ),
            "pair_endpoints": pair_summary,
            "native_elapsed_seconds": max(
                float(payloads[(method, spot)]["elapsed_seconds"])
                for spot in SPOTS
            ),
            "terminal_native_failures": {
                str(spot): payloads[(method, spot)]["terminal_error"]
                for spot in SPOTS
                if payloads[(method, spot)].get("terminal_error")
            },
        }

    native_by_key = {
        (
            str(row["method"]),
            int(row["frame_order"]),
            int(row["spot_id"]),
        ): row
        for row in native_rows
    }
    measurement_by_key = {
        (
            str(row["method"]),
            int(row["frame_order"]),
            int(row["spot_id"]),
        ): row
        for row in measurement_rows
    }
    pair_by_key = {
        (str(row["method"]), int(row["frame_order"]), str(row["pair"])): row
        for row in pair_rows
    }
    comparisons: dict[str, object] = {}
    for first_method, second_method in combinations(methods, 2):
        comparison: dict[str, object] = {
            "native_proposal_distance_original_px": {},
            "gaussian_center_distance_original_px": {},
            "absolute_endpoint_difference_percentage_points": {},
        }
        for spot_id in SPOTS:
            native_distances: list[tuple[int, float]] = []
            gaussian_distances: list[tuple[int, float]] = []
            for frame_order in range(FRAME_COUNT):
                first_native = native_by_key[
                    (first_method, frame_order, spot_id)
                ]
                second_native = native_by_key[
                    (second_method, frame_order, spot_id)
                ]
                if bool(first_native["proposal_available"]) and bool(
                    second_native["proposal_available"]
                ):
                    native_distances.append(
                        (
                            frame_order,
                            math.hypot(
                                float(
                                    first_native[
                                        "measurement_center_x_original_px"
                                    ]
                                )
                                - float(
                                    second_native[
                                        "measurement_center_x_original_px"
                                    ]
                                ),
                                float(
                                    first_native[
                                        "measurement_center_y_original_px"
                                    ]
                                )
                                - float(
                                    second_native[
                                        "measurement_center_y_original_px"
                                    ]
                                ),
                            ),
                        )
                    )
                first_fit = measurement_by_key[
                    (first_method, frame_order, spot_id)
                ]
                second_fit = measurement_by_key[
                    (second_method, frame_order, spot_id)
                ]
                if bool(first_fit["fit_valid"]) and bool(
                    second_fit["fit_valid"]
                ):
                    gaussian_distances.append(
                        (
                            frame_order,
                            math.hypot(
                                float(first_fit["global_center_x"])
                                - float(second_fit["global_center_x"]),
                                float(first_fit["global_center_y"])
                                - float(second_fit["global_center_y"]),
                            ),
                        )
                    )
            comparison["native_proposal_distance_original_px"][
                str(spot_id)
            ] = summarize_distribution(native_distances)
            comparison["gaussian_center_distance_original_px"][
                str(spot_id)
            ] = summarize_distribution(gaussian_distances)
        for pair_name in ("1-6", "3-5"):
            differences: list[tuple[int, float]] = []
            for frame_order in range(FRAME_COUNT):
                first_pair = pair_by_key[
                    (first_method, frame_order, pair_name)
                ]
                second_pair = pair_by_key[
                    (second_method, frame_order, pair_name)
                ]
                if bool(first_pair["pair_valid"]) and bool(
                    second_pair["pair_valid"]
                ):
                    differences.append(
                        (
                            frame_order,
                            abs(
                                float(first_pair["delta_r_over_r_percent"])
                                - float(
                                    second_pair[
                                        "delta_r_over_r_percent"
                                    ]
                                )
                            ),
                        )
                    )
            comparison[
                "absolute_endpoint_difference_percentage_points"
            ][pair_name] = summarize_distribution(differences)
        comparisons[f"{first_method}__vs__{second_method}"] = comparison
    return {
        "status": (
            "completed"
            if tuple(methods) == METHODS
            else "completed_partial"
        ),
        "selected_methods": list(methods),
        "frame_count": FRAME_COUNT,
        "spot_ids": list(SPOTS),
        "pairs": [f"{first}-{second}" for first, second in PAIRS],
        "manual_annotations": 0,
        "methods": summary_methods,
        "cross_method_comparisons": comparisons,
    }


def render_results_readme(summary: dict[str, object]) -> str:
    def number(value: object, digits: int = 6) -> str:
        return "NA" if value is None else f"{float(value):.{digits}f}"

    method_rows: list[str] = []
    endpoint_rows: list[str] = []
    methods = summary["methods"]
    for method in summary["selected_methods"]:
        result = methods[method]
        method_rows.append(
            f"| `{method}` | {result['native_proposal_count']}/2804 | "
            f"{result['valid_gaussian_fit_count']}/2804 | "
            f"{result['native_elapsed_seconds']:.3f} |"
        )
        for pair_name, pair in result["pair_endpoints"].items():
            endpoint_rows.append(
                f"| `{method}` | `{pair_name}` | {pair['valid_count']}/701 | "
                f"{number(pair['plateau_372_376_median_delta_r_over_r_percent'])} | "
                f"{pair['minimum_frame_order'] if pair['minimum_frame_order'] is not None else 'NA'} "
                f"({number(pair['minimum_delta_r_over_r_percent'])}) | "
                f"{number(pair['final_delta_r_over_r_percent'])} |"
            )
    comparison_rows: list[str] = []
    for name, comparison in summary["cross_method_comparisons"].items():
        for spot_id, native in comparison[
            "native_proposal_distance_original_px"
        ].items():
            gaussian = comparison["gaussian_center_distance_original_px"][
                spot_id
            ]
            comparison_rows.append(
                f"| `{name}` | spot {spot_id} | "
                f"{number(native['median'])} | "
                f"{number(native['maximum'])} | "
                f"{number(gaussian['median'])} | "
                f"{number(gaussian['maximum'])} |"
            )
    complete = set(summary["selected_methods"]) == set(METHODS)
    scope_text = (
        "This is the completed five-method run of the frozen 701-frame "
        "contract."
        if complete
        else "This is a completed partial-method run of the frozen "
        "701-frame contract."
    )
    remaining_text = (
        ""
        if complete
        else " Only the methods listed below are present in this run."
    )
    return f"""# Cu(111) tracker-evaluation run

{scope_text}{remaining_text} No manual correction, reinitialization or
annotation was used.

## Coverage

| Method | Native proposals | Valid common fits | Native wall time (s) |
|---|---:|---:|---:|
{chr(10).join(method_rows)}

## Pairwise endpoint

Values are percentage `delta R/R = delta D/D`. The plateau is the median over
frames 372--376.

| Method | Pair | Valid | Plateau (%) | Minimum: frame (%) | Final (%) |
|---|---|---:|---:|---:|---:|
{chr(10).join(endpoint_rows)}

## Proposal versus measurement

Distances are in original-image pixels. This comparison shows whether native
proposal differences survive the common Gaussian measurement.

| Comparison | Spot | Native median | Native maximum | Gaussian median | Gaussian maximum |
|---|---|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

The complete machine-readable statistics, including 95th percentiles and
endpoint differences, are in `summary.json`. These results assess the frozen
measurement chain; they are not an absolute tracking-accuracy ranking.
"""


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    methods = canonical_selected_methods(args.methods)
    if "easyleed_kalman" in methods and args.easyleed_native_dir is None:
        raise SystemExit("--easyleed-native-dir is required for EasyLEED")
    if set(methods).intersection(GPU_METHODS) and args.gpu_native_root is None:
        raise SystemExit(
            "--gpu-native-root is required when a GPU method is selected"
        )

    manifest, contract, frames, seeds = load_contract()
    for method in methods:
        configuration = manifest["method_configurations"][method]
        adapter_path = REPO_ROOT / configuration["adapter_source"]
        if sha256_file(adapter_path) != configuration["adapter_source_sha256"]:
            raise RuntimeError(f"Frozen adapter-source hash mismatch: {method}")
    output_dir.mkdir(parents=True, exist_ok=True)
    native_rows: list[dict[str, object]] = []
    payloads: dict[tuple[str, int], dict[str, object]] = {}
    if "lucas_kanade_skip_anchor" in methods:
        rows, info = run_lucas_kanade(output_dir, manifest, frames, seeds)
        native_rows.extend(rows)
        payloads.update(
            {
                ("lucas_kanade_skip_anchor", spot_id): value
                for spot_id, value in info.items()
            }
        )
    if "easyleed_kalman" in methods:
        rows, info = load_easyleed(
            output_dir,
            args.easyleed_native_dir.expanduser().resolve(),
            manifest,
            frames,
        )
        native_rows.extend(rows)
        payloads.update(
            {
                ("easyleed_kalman", spot_id): value
                for spot_id, value in info.items()
            }
        )
    if set(methods).intersection(GPU_METHODS):
        gpu_native_root = args.gpu_native_root.expanduser().resolve()
        for method in GPU_METHODS:
            if method not in methods:
                continue
            rows, info = load_gpu_native(
                output_dir,
                gpu_native_root / method,
                method,
                manifest,
                frames,
            )
            native_rows.extend(rows)
            payloads.update(
                {
                    (method, spot_id): value
                    for spot_id, value in info.items()
                }
            )
    method_order = {method: index for index, method in enumerate(METHODS)}
    native_rows.sort(
        key=lambda row: (
            method_order[str(row["method"])],
            int(row["frame_order"]),
            int(row["spot_id"]),
        )
    )

    audit_rows = build_audit_rows(output_dir, methods, manifest, payloads)
    measurement_rows = build_measurement_rows(methods, frames, native_rows)
    pair_rows = build_pair_rows(methods, measurement_rows)
    overlay_summary = render_diagnostic_overlays(
        output_dir,
        methods,
        frames,
        native_rows,
        measurement_rows,
    )
    table_rows = {
        "run_audit_rows": audit_rows,
        "native_proposal_rows": native_rows,
        "gaussian_measurement_rows": measurement_rows,
        "pair_endpoint_rows": pair_rows,
    }
    file_hashes: dict[str, str] = {}
    for section, filename in OUTPUT_FILES.items():
        path = output_dir / filename
        write_csv(
            path,
            contract[section]["required_columns"],
            table_rows[section],
        )
        file_hashes[filename] = sha256_file(path)

    summary = build_summary(
        methods,
        native_rows,
        measurement_rows,
        pair_rows,
        payloads,
    )
    summary["diagnostic_overlays"] = overlay_summary
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    results_readme_path = output_dir / "README.md"
    results_readme_path.write_text(
        render_results_readme(summary),
        encoding="utf-8",
    )
    run_manifest = {
        "schema_version": "1.0",
        "status": (
            "completed"
            if tuple(methods) == METHODS
            else "completed_partial"
        ),
        "selected_methods": list(methods),
        "contract_manifest_sha256": sha256_file(
            ARTIFACT_ROOT / "manifest.json"
        ),
        "seed_prompts_sha256": manifest["files"]["seed_prompts.csv"],
        "input_inventory_sha256": manifest["files"]["inference_frames.csv"],
        "method_configuration_sha256": manifest[
            "method_configuration_sha256"
        ],
        "output_files_sha256": file_hashes,
        "summary_sha256": sha256_file(summary_path),
        "results_readme_sha256": sha256_file(results_readme_path),
        "diagnostic_overlays": overlay_summary,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
