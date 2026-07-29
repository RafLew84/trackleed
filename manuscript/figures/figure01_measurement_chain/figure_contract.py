"""Frozen data contract and shared constants for publication Figure 1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


FIGURE_ROOT = Path(__file__).resolve().parent
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
OUTPUT_DIRECTORY = FIGURE_ROOT / "output"
FIGURE_BASENAME = "figure01_measurement_chain"
SOURCE_CASE_MANIFEST = FIGURE_ROOT / "manifest.json"
RENDER_MANIFEST = FIGURE_ROOT / "render_manifest.json"

ROI_SIZE_PX = 48
CACHE_SIZE_PX = 256
ORIGINAL_IMAGE_SIZE_PX = 1000
FIGURE_SIZE_INCHES = (7.2, 6.8)
PNG_DPI = 400
FONT_FAMILY = "DejaVu Sans"


@dataclass(frozen=True)
class MethodSpec:
    method: str
    display_name: str
    representation_label: str
    color: str


METHOD_SPECS = (
    MethodSpec("locotrack_base_offline", "LocoTrack", "point", "#0072B2"),
    MethodSpec(
        "ostrack_vitb_256_mae_ce",
        "OSTrack",
        "bounding box",
        "#E69F00",
    ),
    MethodSpec(
        "dam4sam_sam21pp_l",
        "DAM4SAM",
        "binary mask",
        "#009E73",
    ),
    MethodSpec(
        "lucas_kanade_skip_anchor",
        "Lucas-Kanade",
        "optical-flow point",
        "#CC79A7",
    ),
    MethodSpec(
        "easyleed_kalman",
        "EasyLEED",
        "Kalman position",
        "#D55E00",
    ),
)
METHOD_ORDER = tuple(spec.method for spec in METHOD_SPECS)
DISPLAY_NAMES = {spec.method: spec.display_name for spec in METHOD_SPECS}
REPRESENTATION_LABELS = {
    spec.method: spec.representation_label for spec in METHOD_SPECS
}
METHOD_COLORS = {spec.method: spec.color for spec in METHOD_SPECS}
NEUTRAL_COLOR = "#4D4D4D"
LIGHT_NEUTRAL = "#D9D9D9"
MEASUREMENT_COLOR = "#56B4E9"
EVIDENCE_ROLES = (
    {
        "regime": "Cu(111) thermal sequence",
        "role": "Positive control",
        "question": "Do methods converge on the same physical endpoint?",
        "symbol": "cu",
    },
    {
        "regime": "Known translations",
        "role": "Equivariance",
        "question": "Does the complete chain recover a prescribed shift?",
        "symbol": "translation",
    },
    {
        "regime": "Ag(111) energy scans",
        "role": "Motion + I(V) gaps",
        "question": "Is spot identity recovered through radial motion and gaps?",
        "symbol": "energy",
    },
    {
        "regime": "SAM2 family",
        "role": "Memory mechanism",
        "question": "How does memory policy alter recovery?",
        "symbol": "memory",
    },
)
FONT_SIZES_PT = {
    "base": 7.2,
    "axis_title": 8.0,
    "axis_label": 7.2,
    "tick_label": 6.5,
    "legend": 6.3,
    "panel_label": 9.0,
    "method": 7.1,
    "representation": 6.2,
    "stage": 6.8,
    "minor_annotation": 5.9,
    "formula": 8.4,
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


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_optional_float(value: object) -> float | None:
    text = str(value).strip()
    return float(text) if text else None


def load_contract() -> dict[str, object]:
    contract = json.loads((FIGURE_ROOT / "contract.json").read_text(encoding="utf-8"))
    if contract.get("status") != "representative_case_frozen":
        raise ValueError("Figure 1 representative case is not frozen")
    if contract.get("claim_limit") != (
        "This case is explanatory and was selected without tracker outputs; "
        "it is not a qualitative performance ranking."
    ):
        raise ValueError("Figure 1 claim limit changed")
    return contract


def load_crop_metadata() -> dict[str, object]:
    metadata = json.loads(
        (SOURCE_DATA_DIRECTORY / "representative_spot_crop.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        int(metadata["crop_width_px"]),
        int(metadata["crop_height_px"]),
    ) != (128, 128):
        raise ValueError("Representative crop dimensions changed")
    return metadata


def load_method_outputs() -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    for row in read_csv(
        SOURCE_DATA_DIRECTORY / "representative_method_outputs.csv"
    ):
        parsed.append(
            {
                **row,
                "full_native_geometry_archived": _as_bool(
                    row["full_native_geometry_archived"]
                ),
                "proposal_available": _as_bool(row["proposal_available"]),
                "proposal_x_original_px": float(row["proposal_x_original_px"]),
                "proposal_y_original_px": float(row["proposal_y_original_px"]),
                "native_visibility_signal": _as_optional_float(
                    row["native_visibility_signal"]
                ),
                "roi_x0_original_px": int(float(row["roi_x0_original_px"])),
                "roi_y0_original_px": int(float(row["roi_y0_original_px"])),
                "roi_size_px": int(float(row["roi_size_px"])),
                "gaussian_x_original_px": float(row["gaussian_x_original_px"]),
                "gaussian_y_original_px": float(row["gaussian_y_original_px"]),
                "proposal_offset_px": float(row["proposal_offset_px"]),
                "fit_valid": _as_bool(row["fit_valid"]),
                "amplitude": float(row["amplitude"]),
                "sigma_x_px": float(row["sigma_x_px"]),
                "sigma_y_px": float(row["sigma_y_px"]),
                "r_squared": float(row["r_squared"]),
            }
        )
    by_method = {str(row["method"]): row for row in parsed}
    if set(by_method) != set(METHOD_ORDER):
        raise ValueError("Figure 1 method set differs from the frozen contract")
    rows = [by_method[method] for method in METHOD_ORDER]
    if any(row["display_method"] != DISPLAY_NAMES[row["method"]] for row in rows):
        raise ValueError("Figure 1 method display names changed")
    return rows


def load_native_geometry() -> dict[str, object]:
    geometry = json.loads(
        (
            SOURCE_DATA_DIRECTORY / "representative_native_geometry.json"
        ).read_text(encoding="utf-8")
    )
    if geometry.get("status") != "native_geometry_reexported":
        raise ValueError("Empirical native geometry is not frozen")
    if set(geometry.get("methods", {})) != {
        "ostrack_vitb_256_mae_ce",
        "dam4sam_sam21pp_l",
    }:
        raise ValueError("Empirical native geometry method set changed")
    return geometry


def load_dam4sam_mask() -> object:
    import numpy as np
    from PIL import Image

    path = SOURCE_DATA_DIRECTORY / "representative_dam4sam_mask_cache.png"
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 0
    geometry = load_native_geometry()
    expected = geometry["methods"]["dam4sam_sam21pp_l"]
    if list(mask.shape) != expected["mask_shape_cache_px"]:
        raise ValueError("DAM4SAM native mask shape changed")
    if int(mask.sum()) != int(expected["mask_area_cache_px"]):
        raise ValueError("DAM4SAM native mask area changed")
    if sha256_file(path) != expected["mask_sha256"]:
        raise ValueError("DAM4SAM native mask hash changed")
    return mask


def proposal_in_crop(
    row: dict[str, object],
    crop: dict[str, object],
) -> tuple[float, float]:
    return (
        float(row["proposal_x_original_px"])
        - float(crop["crop_x0_original_px"]),
        float(row["proposal_y_original_px"])
        - float(crop["crop_y0_original_px"]),
    )


def gaussian_in_crop(
    row: dict[str, object],
    crop: dict[str, object],
) -> tuple[float, float]:
    return (
        float(row["gaussian_x_original_px"])
        - float(crop["crop_x0_original_px"]),
        float(row["gaussian_y_original_px"])
        - float(crop["crop_y0_original_px"]),
    )


def ostrack_bbox_in_crop(
    geometry: dict[str, object],
    crop: dict[str, object],
) -> tuple[float, float, float, float]:
    bbox = geometry["methods"]["ostrack_vitb_256_mae_ce"][
        "bbox_xywh_original_px"
    ]
    return (
        float(bbox[0]) - float(crop["crop_x0_original_px"]),
        float(bbox[1]) - float(crop["crop_y0_original_px"]),
        float(bbox[2]),
        float(bbox[3]),
    )


def mask_extent_in_crop(
    crop: dict[str, object],
) -> tuple[float, float, float, float]:
    """Return native-mask image extent in representative-crop coordinates."""

    scale = ORIGINAL_IMAGE_SIZE_PX / CACHE_SIZE_PX
    x0 = -float(crop["crop_x0_original_px"])
    y0 = -float(crop["crop_y0_original_px"])
    return (x0, x0 + CACHE_SIZE_PX * scale, y0 + CACHE_SIZE_PX * scale, y0)


def maximum_pairwise_separation(
    rows: Sequence[dict[str, object]],
    *,
    x_field: str,
    y_field: str,
) -> float:
    maximum = 0.0
    for first_index, first in enumerate(rows):
        for second in rows[first_index + 1 :]:
            distance = math.hypot(
                float(first[x_field]) - float(second[x_field]),
                float(first[y_field]) - float(second[y_field]),
            )
            maximum = max(maximum, distance)
    return maximum


def validate_render_contract() -> None:
    contract = load_contract()
    rows = load_method_outputs()
    geometry = load_native_geometry()
    crop = load_crop_metadata()
    load_dam4sam_mask()

    if any(
        row["roi_size_px"] != ROI_SIZE_PX
        or row["roi_x0_original_px"]
        != round(row["proposal_x_original_px"]) - ROI_SIZE_PX // 2
        or row["roi_y0_original_px"]
        != round(row["proposal_y_original_px"]) - ROI_SIZE_PX // 2
        for row in rows
    ):
        raise ValueError("Common 48 x 48 ROI construction changed")
    if any(
        not row["proposal_available"]
        or not row["fit_valid"]
        or not row["full_native_geometry_archived"]
        for row in rows
    ):
        raise ValueError("Representative case is incomplete")

    for fields, contract_field in (
        (
            ("proposal_x_original_px", "proposal_y_original_px"),
            "maximum_pairwise_native_proposal_separation_px",
        ),
        (
            ("gaussian_x_original_px", "gaussian_y_original_px"),
            "maximum_pairwise_gaussian_center_separation_px",
        ),
    ):
        actual = maximum_pairwise_separation(
            rows,
            x_field=fields[0],
            y_field=fields[1],
        )
        expected = float(contract["descriptive_geometry"][contract_field])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Frozen descriptive geometry changed: {contract_field}")

    ostrow = next(
        row for row in rows if row["method"] == "ostrack_vitb_256_mae_ce"
    )
    ostcenter = geometry["methods"]["ostrack_vitb_256_mae_ce"][
        "reduced_center_original_px"
    ]
    if not math.isclose(
        ostrow["proposal_x_original_px"],
        float(ostcenter[0]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("OSTrack bbox center differs from the frozen proposal")
    if tuple(int(crop[field]) for field in ("crop_width_px", "crop_height_px")) != (
        128,
        128,
    ):
        raise ValueError("Representative crop geometry changed")
