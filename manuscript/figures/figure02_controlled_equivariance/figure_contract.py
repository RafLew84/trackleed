"""Frozen data contract and source-data preparation for Figure 2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence


FIGURE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = FIGURE_ROOT.parents[2]
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
OUTPUT_DIRECTORY = FIGURE_ROOT / "output"
FIGURE_BASENAME = "figure02_controlled_equivariance"

CONTROLLED_TRANSLATION_ROOT = (
    REPOSITORY_ROOT
    / ".scratch/trackleed-physics-method-paper/artifacts/"
    "controlled-translation-chain"
)
CONTROLLED_TRANSLATION_CONTRACT = CONTROLLED_TRANSLATION_ROOT / "contract.json"
CONTROLLED_TRANSLATION_SUMMARY = CONTROLLED_TRANSLATION_ROOT / "method_summary.csv"

METHOD_ORDER = (
    "locotrack_base_offline",
    "ostrack_vitb_256_mae_ce",
    "dam4sam_sam21pp_l",
    "lucas_kanade_skip_anchor",
    "easyleed_kalman",
)
ORACLE_METHOD = "oracle_roi_control"
DISPLAY_NAMES = {
    "locotrack_base_offline": "LocoTrack",
    "ostrack_vitb_256_mae_ce": "OSTrack",
    "dam4sam_sam21pp_l": "DAM4SAM",
    "lucas_kanade_skip_anchor": "Lucas-Kanade",
    "easyleed_kalman": "EasyLEED",
    ORACLE_METHOD: "Oracle ROI",
}
METHOD_COLORS = {
    "locotrack_base_offline": "#0072B2",
    "ostrack_vitb_256_mae_ce": "#E69F00",
    "dam4sam_sam21pp_l": "#009E73",
    "lucas_kanade_skip_anchor": "#CC79A7",
    "easyleed_kalman": "#D55E00",
}
PAIR_COLOR = "#0072B2"
NEUTRAL_COLOR = "#4D4D4D"
FONT_SIZES_PT = {
    "base": 7.2,
    "axis_title": 8.0,
    "axis_label": 7.2,
    "tick_label": 6.6,
    "legend": 6.4,
    "panel_label": 9.0,
    "value_annotation": 6.3,
    "minor_annotation": 6.0,
}

SOURCE_FILES = {
    "controlled_translation_contract": CONTROLLED_TRANSLATION_CONTRACT,
    "controlled_translation_method_summary": CONTROLLED_TRANSLATION_SUMMARY,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".17g")
    return value


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: csv_value(row.get(field)) for field in fieldnames}
            )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def as_int(value: object) -> int:
    return int(float(str(value)))


def as_float(value: object) -> float:
    return float(str(value))


def as_optional_float(value: object) -> float | None:
    text = str(value)
    return float(text) if text else None


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def portable_path(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def load_translation_path(contract_path: Path) -> list[dict[str, object]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_ready_for_runner":
        raise ValueError("Controlled-translation contract is not frozen")
    translations = contract.get("translations_original_px")
    if not isinstance(translations, list) or len(translations) != 6:
        raise ValueError("Expected the frozen six-step translation path")
    points: list[dict[str, object]] = []
    for step, translation in enumerate(translations):
        points.append(
            {
                "step": step,
                "dx_px": as_float(translation["dx"]),
                "dy_px": as_float(translation["dy"]),
                "is_closed_cycle_step": step == len(translations) - 1,
            }
        )
    if (points[0]["dx_px"], points[0]["dy_px"]) != (
        points[-1]["dx_px"],
        points[-1]["dy_px"],
    ):
        raise ValueError("Translation path is not closed")
    return points


def _parse_method_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "method": row["method"],
        "display_method": DISPLAY_NAMES[row["method"]],
        "role": row["role"],
        "evaluation_spot_opportunities": as_int(
            row["evaluation_spot_opportunities"]
        ),
        "valid_gaussian_measurements": as_int(
            row["valid_gaussian_measurements"]
        ),
        "measurement_coverage": as_float(row["measurement_coverage"]),
        "native_error_median_px": as_optional_float(
            row["native_error_median_px"]
        ),
        "measurement_error_median_px": as_float(
            row["measurement_error_median_px"]
        ),
        "measurement_error_p95_px": as_float(
            row["measurement_error_p95_px"]
        ),
        "pair_opportunities": as_int(row["pair_opportunities"]),
        "valid_pair_measurements": as_int(row["valid_pair_measurements"]),
        "abs_delta_d_over_d_max_pp": as_float(
            row["abs_delta_d_over_d_max_pp"]
        ),
    }


def load_all_method_metrics(path: Path) -> list[dict[str, object]]:
    rows = [_parse_method_row(row) for row in read_csv(path)]
    by_method = {str(row["method"]): row for row in rows}
    expected = (*METHOD_ORDER, ORACLE_METHOD)
    if set(by_method) != set(expected):
        raise ValueError("Method summary does not match the frozen method set")
    return [by_method[method] for method in expected]


def load_method_metrics(path: Path) -> list[dict[str, object]]:
    return [
        row
        for row in load_all_method_metrics(path)
        if row["method"] in METHOD_ORDER
    ]

def _prepare_representative_crop(contract: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from PIL import Image

    first_trial = contract["trials"][0]
    raw_path = REPOSITORY_ROOT / first_trial["raw_relative_path"]
    if sha256_file(raw_path) != first_trial["raw_sha256"]:
        raise ValueError("Representative raw-image hash mismatch")
    seeds = {
        as_int(seed["spot_id"]): seed for seed in first_trial["seeds"]
    }
    seed = seeds[1]
    center_x = as_float(seed["center_x_original_px"])
    center_y = as_float(seed["center_y_original_px"])
    half_size = 64
    center_x_integer = int(round(center_x))
    center_y_integer = int(round(center_y))
    x0 = center_x_integer - half_size
    y0 = center_y_integer - half_size
    x1 = center_x_integer + half_size
    y1 = center_y_integer + half_size

    with Image.open(raw_path) as image:
        array = np.asarray(image)
    crop = array[y0:y1, x0:x1].astype(np.float64)
    if crop.shape != (2 * half_size, 2 * half_size):
        raise ValueError("Representative crop extends beyond the raw image")
    lower = float(np.percentile(crop, 1.0))
    upper = float(np.percentile(crop, 99.8))
    if upper <= lower:
        raise ValueError("Representative crop has no display range")
    display = np.clip((crop - lower) / (upper - lower), 0.0, 1.0)
    display_u8 = np.round(display * 255.0).astype(np.uint8)
    crop_path = SOURCE_DATA_DIRECTORY / "representative_cu_crop.png"
    Image.fromarray(display_u8, mode="L").save(
        crop_path,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return {
        "raw_relative_path": first_trial["raw_relative_path"],
        "raw_sha256": first_trial["raw_sha256"],
        "temperature_c": as_float(first_trial["temperature_c"]),
        "energy_ev": as_float(first_trial["energy_ev"]),
        "spot_id": 1,
        "crop_x0_original_px": x0,
        "crop_y0_original_px": y0,
        "crop_width_px": 2 * half_size,
        "crop_height_px": 2 * half_size,
        "seed_x_crop_px": center_x - x0,
        "seed_y_crop_px": center_y - y0,
        "roi_size_original_px": as_int(contract["roi_size_original_px"]),
        "display_percentile_low": 1.0,
        "display_percentile_high": 99.8,
        "display_intensity_low": lower,
        "display_intensity_high": upper,
    }


def refresh_source_data() -> None:
    missing = [path for path in SOURCE_FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen source artifact: {missing[0]}")
    SOURCE_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    contract = json.loads(
        CONTROLLED_TRANSLATION_CONTRACT.read_text(encoding="utf-8")
    )
    translation_rows = load_translation_path(CONTROLLED_TRANSLATION_CONTRACT)
    method_rows = load_all_method_metrics(CONTROLLED_TRANSLATION_SUMMARY)
    crop_metadata = _prepare_representative_crop(contract)

    write_csv(SOURCE_DATA_DIRECTORY / "translation_path.csv", translation_rows)
    write_csv(SOURCE_DATA_DIRECTORY / "method_metrics.csv", method_rows)
    write_json(SOURCE_DATA_DIRECTORY / "representative_cu_crop.json", crop_metadata)

    provenance_sources = {
        name: {
            "path": portable_path(path),
            "sha256": sha256_file(path),
        }
        for name, path in SOURCE_FILES.items()
    }
    raw_path = REPOSITORY_ROOT / crop_metadata["raw_relative_path"]
    provenance_sources["representative_cu_raw_image"] = {
        "path": portable_path(raw_path),
        "sha256": sha256_file(raw_path),
    }
    write_json(
        SOURCE_DATA_DIRECTORY / "source_provenance.json",
        {
            "schema_version": "1.0",
            "figure": "Figure 2",
            "sources": provenance_sources,
        },
    )


def load_translation_source_data() -> list[dict[str, object]]:
    return [
        {
            "step": as_int(row["step"]),
            "dx_px": as_float(row["dx_px"]),
            "dy_px": as_float(row["dy_px"]),
            "is_closed_cycle_step": as_bool(row["is_closed_cycle_step"]),
        }
        for row in read_csv(SOURCE_DATA_DIRECTORY / "translation_path.csv")
    ]
