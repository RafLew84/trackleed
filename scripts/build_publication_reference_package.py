"""Build the frozen-input draft for the TrackLEED publication pilot.

This script deliberately creates only pre-inference artifacts. Existing session
coordinates are frozen as common initialization proposals, never as ground truth.
The pilot evaluates the Gaussian-refined measurement chain without new manual
annotations or claims of absolute tracking accuracy.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".scratch"
    / "trackleed-physics-method-paper"
    / "artifacts"
    / "benchmark-reference-v1"
)
FRAME_PATTERN = re.compile(
    r"_(?P<frame_id>\d+)_T(?P<temperature>\d+)_E(?P<energy>\d+)"
    r"_C(?P<coverage>\d+)\.(?:tif|tiff)$",
    re.IGNORECASE,
)
HASH_CHUNK_BYTES = 8 * 1024 * 1024
CACHE_SIZE = 256
ORIGINAL_SIZE = 1000
EXPECTED_TOTAL_TIFF_BYTES = 2_426_606_500
METADATA_ANCHORS_PER_SERIES = 12
TEMPORAL_QUANTILES_PER_SERIES = 12
EVALUATION_FRAMES_PER_SERIES = (
    METADATA_ANCHORS_PER_SERIES + TEMPORAL_QUANTILES_PER_SERIES
)


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    data_dir: str
    proposal_session: str
    expected_frames: int
    expected_spots: int
    physical_role: str


@dataclass(frozen=True)
class CheckpointSpec:
    method: str
    family: str
    variant: str
    relative_path: str
    official_url: str


SERIES = (
    SeriesSpec(
        series_id="cu111_56ev",
        data_dir="data/25w28_ThExten_56eV_UD",
        proposal_session="data/25w28_ThExten_56eV_UD/s2.lts.json",
        expected_frames=701,
        expected_spots=6,
        physical_role="primary_physics_case",
    ),
    SeriesSpec(
        series_id="cu3au_au04_122ev",
        data_dir="data/26b32_ThExt_600CUD_122eV_Au04",
        proposal_session=(
            "data/26b32_ThExt_600CUD_122eV_Au04/"
            "26b32_ThExt_600CUD_122eV_Au04_4spots.lts.json"
        ),
        expected_frames=1198,
        expected_spots=4,
        physical_role="long_tracking_stress_case",
    ),
    SeriesSpec(
        series_id="cu3au_au10_set1_122ev",
        data_dir="data/26c22_ThExt_122eV_500CUD_Au10",
        proposal_session=(
            "data/26c22_ThExt_122eV_500CUD_Au10/"
            "26c22_ThExt_122eV_500CUD_Au10_set1.lts.json"
        ),
        expected_frames=527,
        expected_spots=4,
        physical_role="independent_tracking_replication",
    ),
)

CHECKPOINTS = (
    CheckpointSpec(
        method="locotrack",
        family="point",
        variant="base_offline",
        relative_path="tracking_models/locotrack/locotrack_base.ckpt",
        official_url="https://github.com/cvlab-kaist/locotrack",
    ),
    CheckpointSpec(
        method="ostrack",
        family="bounding_box",
        variant="vitb_256_mae_ce_32x4_ep300",
        relative_path=(
            "tracking_models/ostrack/vitb_256_mae_ce_32x4_ep300/"
            "OSTrack_ep0300.pth.tar"
        ),
        official_url="https://github.com/botaoye/OSTrack",
    ),
    CheckpointSpec(
        method="dam4sam",
        family="mask",
        variant="sam21pp-L",
        relative_path="tracking_models/dam4sam/sam2.1_hiera_large.pt",
        official_url="https://github.com/jovanavidenovic/DAM4SAM",
    ),
    CheckpointSpec(
        method="sam2",
        family="controlled_mask_comparator",
        variant="sam2.1_hiera_large",
        relative_path="tracking_models/sam2/sam2.1_hiera_large.pt",
        official_url="https://github.com/facebookresearch/sam2",
    ),
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or validate the pre-inference publication reference package."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Artifact directory. Default: the publication wayfinder artifact directory.",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Build manifests without creating 256x256 PNGs.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Recreate cache PNGs even when they already exist.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing generated package without modifying it.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_repo(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def load_session(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("data_files"), list):
        raise ValueError(f"Unsupported session structure: {path}")
    return payload


def discover_frames(spec: SeriesSpec) -> list[dict]:
    data_dir = REPO_ROOT / spec.data_dir
    session = load_session(REPO_ROOT / spec.proposal_session)
    session_frames = {
        int(item["id"]): item
        for item in session["data_files"]
        if isinstance(item, dict) and item.get("id") is not None
    }
    frames: list[dict] = []
    for path in data_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        match = FRAME_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"Cannot parse frame metadata: {path}")
        frame_id = int(match.group("frame_id"))
        session_frame = session_frames.get(frame_id)
        if session_frame is None:
            raise ValueError(f"Frame {frame_id} is absent from proposal session: {path}")
        frames.append(
            {
                "series_id": spec.series_id,
                "frame_id": frame_id,
                "filename": path.name,
                "raw_path": relative_to_repo(path),
                "raw_abspath": path,
                "temperature_c": float(match.group("temperature")),
                "energy_ev": float(match.group("energy")) / 10.0,
                "coverage_nominal": float(match.group("coverage")),
                "proposal_session_frame": session_frame,
            }
        )
    frames.sort(key=lambda item: (int(item["frame_id"]), str(item["filename"])))
    if len(frames) != spec.expected_frames:
        raise ValueError(
            f"{spec.series_id}: expected {spec.expected_frames} TIFFs, found {len(frames)}"
        )
    frame_ids = [int(item["frame_id"]) for item in frames]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError(f"{spec.series_id}: duplicate frame IDs")
    for order, frame in enumerate(frames):
        frame["frame_order"] = order
    return frames


def nearest_available_index(
    frames: Sequence[Mapping[str, object]],
    candidate_indices: Iterable[int],
    *,
    target_temperature: float | None,
    target_order: float,
    used: set[int],
    minimum_index: int = 1,
    maximum_index: int | None = None,
) -> int:
    upper = len(frames) - 1 if maximum_index is None else maximum_index
    candidates = [
        index
        for index in candidate_indices
        if minimum_index <= index <= upper and index not in used
    ]
    if not candidates:
        candidates = [
            index
            for index in range(minimum_index, upper + 1)
            if index not in used
        ]
    if not candidates:
        raise ValueError("No unused evaluation frame remains")

    def key(index: int) -> tuple[float, float, int]:
        temperature_distance = (
            0.0
            if target_temperature is None
            else abs(float(frames[index]["temperature_c"]) - target_temperature)
        )
        return (temperature_distance, abs(index - target_order), index)

    return min(candidates, key=key)


def select_evaluation_frames(frames: Sequence[Mapping[str, object]]) -> list[dict]:
    """Select 12 metadata anchors and 12 temporal quantiles without the seed."""

    if len(frames) < EVALUATION_FRAMES_PER_SERIES + 1:
        raise ValueError(
            "Too few frames for the frozen resource-constrained evaluation set"
        )
    used: set[int] = set()
    selected: list[dict] = []
    temperatures = [float(frame["temperature_c"]) for frame in frames]
    peak_temperature = max(temperatures[1:])
    peak_indices = [
        index
        for index, temperature in enumerate(temperatures)
        if index > 0 and temperature == peak_temperature
    ]
    first_peak = peak_indices[0]
    last_peak = peak_indices[-1]
    start_index = 1
    end_index = len(frames) - 1
    start_temperature = temperatures[start_index]
    end_temperature = temperatures[end_index]

    def add(
        label: str,
        candidates: Iterable[int],
        target_temperature: float | None,
        target_order: float,
        minimum_index: int = 1,
        maximum_index: int | None = None,
    ) -> None:
        index = nearest_available_index(
            frames,
            candidates,
            target_temperature=target_temperature,
            target_order=target_order,
            used=used,
            minimum_index=minimum_index,
            maximum_index=maximum_index,
        )
        used.add(index)
        selected.append(
            {
                "frame_order": index,
                "selection_group": "physical_metadata_anchor",
                "selection_reason": label,
            }
        )

    add("evaluation_range_start_after_seed", [start_index], start_temperature, start_index)
    for fraction in (0.2, 0.4, 0.6, 0.8):
        target_temperature = start_temperature + fraction * (
            peak_temperature - start_temperature
        )
        target_order = start_index + fraction * (first_peak - start_index)
        add(
            f"heating_temperature_fraction_{int(fraction * 100):02d}",
            range(start_index, first_peak + 1),
            target_temperature,
            target_order,
            minimum_index=start_index,
            maximum_index=first_peak,
        )
    add(
        "temperature_peak_entry",
        range(max(start_index, first_peak - 3), min(end_index, first_peak + 3) + 1),
        peak_temperature,
        first_peak,
    )
    add(
        "temperature_peak_exit",
        range(last_peak, min(end_index, last_peak + 3) + 1),
        peak_temperature,
        last_peak,
        minimum_index=last_peak,
    )
    for fraction in (0.8, 0.6, 0.4, 0.2):
        target_temperature = end_temperature + fraction * (
            peak_temperature - end_temperature
        )
        target_order = last_peak + (1.0 - fraction) * (end_index - last_peak)
        add(
            f"cooling_temperature_fraction_{int(fraction * 100):02d}",
            range(last_peak, end_index + 1),
            target_temperature,
            target_order,
            minimum_index=last_peak,
            maximum_index=end_index,
        )
    add("evaluation_range_end", [end_index], end_temperature, end_index)

    if len(selected) != 12:
        raise AssertionError(f"Expected 12 metadata anchors, got {len(selected)}")

    quantile_denominator = TEMPORAL_QUANTILES_PER_SERIES + 1
    for quantile_index in range(1, quantile_denominator):
        target_order = quantile_index * end_index / quantile_denominator
        rounded = int(round(target_order))
        index = nearest_available_index(
            frames,
            range(max(1, rounded - 60), min(end_index, rounded + 60) + 1),
            target_temperature=None,
            target_order=target_order,
            used=used,
        )
        used.add(index)
        selected.append(
            {
                "frame_order": index,
                "selection_group": "temporal_quantile",
                "selection_reason": (
                    f"time_quantile_{quantile_index:02d}_of_"
                    f"{quantile_denominator}"
                ),
            }
        )

    if (
        len(selected) != EVALUATION_FRAMES_PER_SERIES
        or len(used) != EVALUATION_FRAMES_PER_SERIES
        or 0 in used
    ):
        raise AssertionError(
            "Evaluation selection is not the expected set of unique non-seed frames"
        )
    selected.sort(key=lambda item: int(item["frame_order"]))
    return selected


def seed_spot_proposals(spec: SeriesSpec, frames: Sequence[Mapping[str, object]]) -> list[dict]:
    seed = frames[0]
    spots = seed["proposal_session_frame"].get("spots")
    if not isinstance(spots, list) or len(spots) != spec.expected_spots:
        raise ValueError(
            f"{spec.series_id}: expected {spec.expected_spots} seed spots, "
            f"found {0 if not isinstance(spots, list) else len(spots)}"
        )
    proposals: list[dict] = []
    for spot in sorted(spots, key=lambda item: int(item["id"])):
        roi = spot.get("roi") or {}
        proposals.append(
            {
                "series_id": spec.series_id,
                "frame_order": 0,
                "frame_id": int(seed["frame_id"]),
                "filename": str(seed["filename"]),
                "spot_id": int(spot["id"]),
                "identity_label": f"{spec.series_id}:spot_{int(spot['id']):02d}",
                "proposal_center_x_px": float(spot["center_x"]),
                "proposal_center_y_px": float(spot["center_y"]),
                "proposal_bbox_x_px": int(roi["x"]),
                "proposal_bbox_y_px": int(roi["y"]),
                "proposal_bbox_w_px": int(roi["w"]),
                "proposal_bbox_h_px": int(roi["h"]),
                "proposal_source": str(spot.get("source") or "unknown"),
                "proposal_session": spec.proposal_session,
                "not_ground_truth": "true",
            }
        )
    return proposals


def make_cache(
    frame: Mapping[str, object],
    output: Path,
    *,
    rebuild: bool,
) -> tuple[str, str]:
    cache_path = (
        output
        / "cache256"
        / str(frame["series_id"])
        / f"{int(frame['frame_order']):04d}_{int(frame['frame_id']):04d}.png"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild or not cache_path.exists():
        image = tifffile.imread(Path(frame["raw_abspath"]))
        if image.shape != (ORIGINAL_SIZE, ORIGINAL_SIZE):
            raise ValueError(
                f"Unexpected TIFF shape {image.shape}: {frame['raw_abspath']}"
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 TIFF, got {image.dtype}: {frame['raw_abspath']}"
            )
        resized = Image.fromarray(image, mode="L").resize(
            (CACHE_SIZE, CACHE_SIZE),
            resample=Image.Resampling.BOX,
        )
        resized.save(cache_path, format="PNG", compress_level=9, optimize=False)
    with Image.open(cache_path) as cached:
        if cached.mode != "L" or cached.size != (CACHE_SIZE, CACHE_SIZE):
            raise ValueError(f"Invalid cache image: {cache_path}")
    return cache_path.relative_to(output).as_posix(), sha256_file(cache_path)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_rows() -> list[dict]:
    rows: list[dict] = []
    for spec in CHECKPOINTS:
        path = REPO_ROOT / spec.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        rows.append(
            {
                "method": spec.method,
                "family": spec.family,
                "variant": spec.variant,
                "relative_path": spec.relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "official_url": spec.official_url,
                "redistribution_approved": "false",
            }
        )
    return rows


def build(output: Path, *, skip_cache: bool, rebuild_cache: bool) -> None:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_frame_rows: list[dict] = []
    all_seed_proposals: list[dict] = []
    manifest_series: list[dict] = []
    raw_total_bytes = 0

    for spec in SERIES:
        frames = discover_frames(spec)
        selection = select_evaluation_frames(frames)
        selection_by_order = {
            int(item["frame_order"]): item for item in selection
        }
        proposals = seed_spot_proposals(spec, frames)
        all_seed_proposals.extend(proposals)
        series_raw_bytes = 0
        for frame in frames:
            raw_path = Path(frame["raw_abspath"])
            raw_size = raw_path.stat().st_size
            raw_hash = sha256_file(raw_path)
            series_raw_bytes += raw_size
            raw_total_bytes += raw_size
            cache_path = ""
            cache_hash = ""
            if not skip_cache:
                cache_path, cache_hash = make_cache(
                    frame,
                    output,
                    rebuild=rebuild_cache,
                )
            selection_info = selection_by_order.get(int(frame["frame_order"]), {})
            all_frame_rows.append(
                {
                    "series_id": spec.series_id,
                    "frame_order": int(frame["frame_order"]),
                    "frame_id": int(frame["frame_id"]),
                    "filename": str(frame["filename"]),
                    "raw_relative_path": str(frame["raw_path"]),
                    "raw_size_bytes": raw_size,
                    "raw_sha256": raw_hash,
                    "temperature_c": float(frame["temperature_c"]),
                    "energy_ev": float(frame["energy_ev"]),
                    "coverage_nominal": float(frame["coverage_nominal"]),
                    "cache_relative_path": cache_path,
                    "cache_sha256": cache_hash,
                    "evaluation_selected": (
                        "true" if int(frame["frame_order"]) in selection_by_order else "false"
                    ),
                    "selection_group": selection_info.get("selection_group", ""),
                    "selection_reason": selection_info.get("selection_reason", ""),
                }
            )
        manifest_series.append(
            {
                "series_id": spec.series_id,
                "raw_data_directory": spec.data_dir,
                "proposal_session": spec.proposal_session,
                "proposal_is_ground_truth": False,
                "frame_count": len(frames),
                "raw_size_bytes": series_raw_bytes,
                "spot_count": spec.expected_spots,
                "seed_frame_order": 0,
                "seed_frame_id": int(frames[0]["frame_id"]),
                "seed_filename": str(frames[0]["filename"]),
                "evaluation_frame_count": len(selection),
                "evaluation_frame_orders": [
                    int(item["frame_order"]) for item in selection
                ],
                "evaluation_frame_ids": [
                    int(frames[int(item["frame_order"])]["frame_id"])
                    for item in selection
                ],
                "physical_role": spec.physical_role,
            }
        )

    if raw_total_bytes != EXPECTED_TOTAL_TIFF_BYTES:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_TIFF_BYTES} raw bytes, got {raw_total_bytes}"
        )

    manifest_dir = output / "manifests"
    annotation_dir = output / "annotations"
    frame_fields = (
        "series_id",
        "frame_order",
        "frame_id",
        "filename",
        "raw_relative_path",
        "raw_size_bytes",
        "raw_sha256",
        "temperature_c",
        "energy_ev",
        "coverage_nominal",
        "cache_relative_path",
        "cache_sha256",
        "evaluation_selected",
        "selection_group",
        "selection_reason",
    )
    write_csv(manifest_dir / "frames.csv", frame_fields, all_frame_rows)

    checkpoint_data = checkpoint_rows()
    checkpoint_fields = tuple(checkpoint_data[0].keys())
    write_csv(manifest_dir / "checkpoints.csv", checkpoint_fields, checkpoint_data)

    seed_proposal_fields = tuple(all_seed_proposals[0].keys())
    write_csv(
        annotation_dir / "seed_proposals_not_ground_truth.csv",
        seed_proposal_fields,
        all_seed_proposals,
    )

    manifest = {
        "schema_version": "1.1-draft",
        "package_status": "ready_for_ground_truth_free_pilot",
        "frozen": False,
        "created_date": "2026-07-27",
        "repository_revision": "f75001c71a68148dbbcf5f52bd944f0aa33802a4",
        "publication_protocol": (
            ".scratch/trackleed-physics-method-paper/research/"
            "02-protokol-uczciwego-porownania-trackerow.md"
        ),
        "paths_are_relative_to": "repository_root_or_artifact_root_as_documented",
        "raw_input": {
            "format": "TIFF",
            "dtype": "uint8",
            "height": ORIGINAL_SIZE,
            "width": ORIGINAL_SIZE,
            "background_subtraction": False,
            "local_contrast_enhancement": False,
            "method_specific_denoising": False,
            "frame_count": len(all_frame_rows),
            "size_bytes": raw_total_bytes,
            "inventory": "manifests/frames.csv",
        },
        "common_cache": {
            "present": not skip_cache,
            "format": "PNG",
            "mode": "L",
            "height": CACHE_SIZE,
            "width": CACHE_SIZE,
            "resampling": "Pillow Image.Resampling.BOX",
            "pillow_version": Image.__version__,
            "coordinate_scale_x": CACHE_SIZE / ORIGINAL_SIZE,
            "coordinate_scale_y": CACHE_SIZE / ORIGINAL_SIZE,
            "rgb_adapter": "repeat the grayscale plane into three identical channels",
            "inventory": "manifests/frames.csv",
        },
        "automated_probe_selection": {
            "seed_excluded": True,
            "frames_per_series": EVALUATION_FRAMES_PER_SERIES,
            "temporal_quantiles_per_series": TEMPORAL_QUANTILES_PER_SERIES,
            "metadata_anchors_per_series": METADATA_ANCHORS_PER_SERIES,
            "metadata_anchor_rule": (
                "first non-seed frame; 20/40/60/80 percent temperature levels "
                "on heating; peak entry and exit; 80/60/40/20 percent "
                "temperature levels on cooling; final frame"
            ),
            "selection_uses_tracker_outputs": False,
        },
        "manual_reference": {
            "new_annotation_required": False,
            "absolute_tracking_accuracy_claimed": False,
            "tap_vid_metrics_enabled": False,
            "seed_proposals": (
                "annotations/seed_proposals_not_ground_truth.csv"
            ),
            "seed_proposal_role": (
                "frozen common initialization input, not ground truth"
            ),
        },
        "cu111_physics_roles": {
            "complete_core_spot_ids": [1, 3, 5, 6],
            "diametric_core_pairs": [[1, 6], [3, 5]],
            "missingness_sensitivity_pair": [2, 4],
            "common_gaussian_roi_size_px": 32,
            "pairwise_radius_rule": (
                "half the Euclidean distance between Gaussian centers in each "
                "predefined diametric pair; relative delta R/R equals delta D/D"
            ),
            "reference_baseline_rule": (
                "for each method, median valid Gaussian-refined pair distance "
                "over the first 10 Cu(111) frames"
            ),
            "pattern_center_required": False,
            "dense_method_curves_are_ground_truth": False,
            "dense_method_curve_validation": (
                "known image transformations, prompt perturbations, "
                "forward-backward consistency, fit validity, and LEED pattern "
                "geometry; do not claim absolute framewise accuracy"
            ),
        },
        "trackers": {
            "main": [
                "LocoTrack Base offline",
                "OSTrack ViT-B/256 MAE-CE",
                "DAM4SAM sam21pp-L",
                "Lucas-Kanade skip-anchor",
            ],
            "controlled_pair": ["DAM4SAM sam21pp-L", "SAM2.1 Hiera Large"],
            "checkpoint_inventory": "manifests/checkpoints.csv",
            "checkpoint_redistribution_approved": False,
        },
        "series": manifest_series,
    }
    manifest_path = manifest_dir / "benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    validate(output, require_cache=not skip_cache)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate(output: Path, *, require_cache: bool | None = None) -> None:
    manifest_path = output / "manifests" / "benchmark_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    frames = read_csv(output / "manifests" / "frames.csv")
    if len(frames) != 2426:
        raise ValueError(f"Expected 2426 frame rows, found {len(frames)}")
    raw_bytes = sum(int(row["raw_size_bytes"]) for row in frames)
    if raw_bytes != EXPECTED_TOTAL_TIFF_BYTES:
        raise ValueError(f"Unexpected raw byte total: {raw_bytes}")
    for row in frames:
        raw_path = REPO_ROOT / row["raw_relative_path"]
        if not raw_path.is_file() or sha256_file(raw_path) != row["raw_sha256"]:
            raise ValueError(f"Raw input validation failed: {raw_path}")
    cache_expected = (
        bool(manifest["common_cache"]["present"])
        if require_cache is None
        else require_cache
    )
    if cache_expected:
        for row in frames:
            cache_path = output / row["cache_relative_path"]
            if not cache_path.is_file() or sha256_file(cache_path) != row["cache_sha256"]:
                raise ValueError(f"Cache validation failed: {cache_path}")

    selected_by_series: dict[str, list[dict]] = {}
    for row in frames:
        if row["evaluation_selected"] == "true":
            selected_by_series.setdefault(row["series_id"], []).append(row)
    for series_id, selected in selected_by_series.items():
        if len(selected) != EVALUATION_FRAMES_PER_SERIES:
            raise ValueError(
                f"{series_id}: expected "
                f"{EVALUATION_FRAMES_PER_SERIES} automated probe frames"
            )
        if any(int(row["frame_order"]) == 0 for row in selected):
            raise ValueError(f"{series_id}: seed frame leaked into evaluation")
        groups = [row["selection_group"] for row in selected]
        if groups.count("temporal_quantile") != TEMPORAL_QUANTILES_PER_SERIES:
            raise ValueError(f"{series_id}: temporal quantile count is wrong")
        if (
            groups.count("physical_metadata_anchor")
            != METADATA_ANCHORS_PER_SERIES
        ):
            raise ValueError(f"{series_id}: metadata anchor count is wrong")

    seed_path = output / "annotations" / "seed_proposals_not_ground_truth.csv"
    if len(read_csv(seed_path)) != 14:
        raise ValueError("Common initialization proposal inventory must have 14 rows")
    if len(read_csv(output / "manifests" / "checkpoints.csv")) != 4:
        raise ValueError("Checkpoint inventory does not have four rows")
    if manifest.get("frozen") is not False:
        raise ValueError("Draft manifest must not claim to be frozen")
    if manifest.get("manual_reference", {}).get("new_annotation_required") is not False:
        raise ValueError("Ground-truth-free pilot must not require new annotation")


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if args.validate_only:
        validate(output)
        print(f"Validated draft reference package: {output}")
        return 0
    build(
        output,
        skip_cache=bool(args.skip_cache),
        rebuild_cache=bool(args.rebuild_cache),
    )
    print(f"Built draft reference package: {output}")
    print("Status: ready for the ground-truth-free measurement-chain pilot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
