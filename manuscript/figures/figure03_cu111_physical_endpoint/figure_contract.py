"""Frozen source-data contract for publication Figure 3."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


FIGURE_ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = FIGURE_ROOT / "contract.json"
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
OUTPUT_DIRECTORY = FIGURE_ROOT / "output"
FIGURE_BASENAME = "figure03_cu111_physical_endpoint"
MANIFEST_PATH = FIGURE_ROOT / "manifest.json"
FIGURE_SIZE_INCHES = (7.2, 8.0)
PNG_DPI = 400
FONT_FAMILY = "DejaVu Sans"
ANCHOR_ANNOTATION = "2804/2804 measurements per method; 701/701 endpoints per pair"


@dataclass(frozen=True)
class MethodSpec:
    method: str
    display_name: str
    color: str
    marker: str
    linestyle: object


@dataclass(frozen=True)
class PrimaryRecords:
    """Validated records defining the Cu(111) positive-control display."""

    run: dict[str, object]
    endpoints: list[dict[str, object]]
    summaries: list[dict[str, object]]


METHOD_SPECS = (
    MethodSpec("locotrack_base_offline", "LocoTrack", "#0072B2", "o", "solid"),
    MethodSpec("ostrack_vitb_256_mae_ce", "OSTrack", "#E69F00", "s", (0, (5, 2))),
    MethodSpec("dam4sam_sam21pp_l", "DAM4SAM", "#009E73", "^", (0, (2, 1.5))),
    MethodSpec(
        "lucas_kanade_skip_anchor",
        "Lucas-Kanade",
        "#CC79A7",
        "D",
        (0, (7, 2, 1.5, 2)),
    ),
    MethodSpec("easyleed_kalman", "EasyLEED", "#D55E00", "P", (0, (1, 1.5))),
)
METHOD_ORDER = (*(spec.method for spec in METHOD_SPECS),)
DISPLAY_NAMES = {spec.method: spec.display_name for spec in METHOD_SPECS}
METHOD_COLORS = {spec.method: spec.color for spec in METHOD_SPECS}
METHOD_MARKERS = {spec.method: spec.marker for spec in METHOD_SPECS}
METHOD_LINESTYLES = {spec.method: spec.linestyle for spec in METHOD_SPECS}
PAIR_ORDER = ("1-6", "3-5")
PANEL_ORDER = (
    "thermal_cycle_landmarks",
    "pair_1_6_endpoint",
    "pair_3_5_endpoint",
    "between_method_endpoint_spread",
)
NEUTRAL_COLOR = "#4D4D4D"
PAIR_COLORS = {"1-6": "#4C78A8", "3-5": "#E45756"}
FONT_SIZES_PT = {
    "base": 7.2,
    "axis_title": 8.1,
    "axis_label": 7.2,
    "tick_label": 6.4,
    "legend": 6.0,
    "panel_label": 9.2,
    "value_annotation": 6.0,
    "minor_annotation": 5.7,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_contract() -> dict[str, object]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("status") != "source_contract_frozen":
        raise ValueError("Figure 3 source contract is not frozen")
    if tuple(contract["primary_scope"]["methods"]) != METHOD_ORDER:
        raise ValueError("Figure 3 primary method order changed")
    if tuple(contract["primary_scope"]["pairs"]) != PAIR_ORDER:
        raise ValueError("Figure 3 primary pair order changed")
    if tuple(contract["panel_order"]) != PANEL_ORDER:
        raise ValueError("Figure 3 panel order changed")
    if contract["run_contract_source"] != "source_data/five_method_run_contract.json":
        raise ValueError("Figure 3 completed-run source changed")
    return contract


def load_temperature_profile() -> list[dict[str, object]]:
    return [
        {
            **row,
            "frame_order": int(row["frame_order"]),
            "frame_id": int(row["frame_id"]),
            "temperature_c": float(row["temperature_c"]),
            "energy_ev": float(row["energy_ev"]),
        }
        for row in read_csv(SOURCE_DATA_DIRECTORY / "temperature_profile.csv")
    ]


def load_cycle_landmarks() -> list[dict[str, object]]:
    return [
        {
            **row,
            "frame_start": int(row["frame_start"]),
            "frame_end": int(row["frame_end"]),
            "temperature_min_c": float(row["temperature_min_c"]),
            "temperature_max_c": float(row["temperature_max_c"]),
            "outcome_selected": as_bool(row["outcome_selected"]),
        }
        for row in read_csv(SOURCE_DATA_DIRECTORY / "cycle_landmarks.csv")
    ]


def load_pair_endpoint_timeseries() -> list[dict[str, object]]:
    return [
        {
            **row,
            "frame_order": int(row["frame_order"]),
            "frame_id": int(row["frame_id"]),
            "pair_valid": as_bool(row["pair_valid"]),
            "distance_px": float(row["distance_px"]),
            "reference_distance_px": float(row["reference_distance_px"]),
            "reference_valid_frame_count": int(row["reference_valid_frame_count"]),
            "delta_r_over_r": float(row["delta_r_over_r"]),
            "delta_r_over_r_percent": float(row["delta_r_over_r_percent"]),
        }
        for row in read_csv(SOURCE_DATA_DIRECTORY / "pair_endpoint_timeseries.csv")
    ]


def load_endpoint_summaries() -> list[dict[str, object]]:
    float_fields = (
        "reference_distance_px",
        "plateau_median_percent",
        "minimum_percent",
        "final_percent",
    )
    return [
        {
            **row,
            "valid_frame_count": int(row["valid_frame_count"]),
            "reference_valid_frame_count": int(row["reference_valid_frame_count"]),
            "minimum_frame_order": int(row["minimum_frame_order"]),
            **{field: float(row[field]) for field in float_fields},
        }
        for row in read_csv(SOURCE_DATA_DIRECTORY / "endpoint_summaries.csv")
    ]


def load_between_method_spread() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(SOURCE_DATA_DIRECTORY / "between_method_spread.csv"):
        rows.append(
            {
                **row,
                "frame_order": (
                    int(row["frame_order"]) if row["frame_order"] else None
                ),
                "minimum_value_percent": float(row["minimum_value_percent"]),
                "maximum_value_percent": float(row["maximum_value_percent"]),
                "spread_percentage_points": float(row["spread_percentage_points"]),
                "display_in_panel_d": as_bool(row["display_in_panel_d"]),
            }
        )
    return rows


def load_five_method_run_contract() -> dict[str, object]:
    return json.loads(
        (SOURCE_DATA_DIRECTORY / "five_method_run_contract.json").read_text(
            encoding="utf-8"
        )
    )


def load_primary_records() -> PrimaryRecords:
    """Validate and load the records at the public artifact seam."""

    validate_source_contract()
    return PrimaryRecords(
        run=load_five_method_run_contract(),
        endpoints=load_pair_endpoint_timeseries(),
        summaries=load_endpoint_summaries(),
    )


def _validate_primary_timeseries(
    source_directory: Path, contract: dict[str, object]
) -> None:
    rows = read_csv(source_directory / "pair_endpoint_timeseries.csv")
    expected = {
        (method, pair, str(frame))
        for method in METHOD_ORDER
        for pair in PAIR_ORDER
        for frame in range(701)
    }
    identities = {(row["method"], row["pair"], row["frame_order"]) for row in rows}
    if identities != expected or len(rows) != len(expected):
        raise ValueError("Figure 3 primary endpoint identities changed")
    series_label = contract["primary_scope"]["scientific_series_label"]
    if {row["scientific_series_label"] for row in rows} != {series_label}:
        raise ValueError("Figure 3 reader-facing series label changed")
    if {row["evidence_role"] for row in rows} != {"five-method positive control"}:
        raise ValueError("Figure 3 primary evidence role changed")
    if any(not as_bool(row["pair_valid"]) for row in rows):
        raise ValueError("Figure 3 primary endpoint coverage changed")
    if {int(row["reference_valid_frame_count"]) for row in rows} != {10}:
        raise ValueError("Figure 3 reference denominator changed")


def _validate_five_method_run_contract(
    source_directory: Path, contract: dict[str, object]
) -> None:
    run = json.loads(
        (source_directory / "five_method_run_contract.json").read_text(encoding="utf-8")
    )
    scope = contract["primary_scope"]
    if (
        run["status"] != "completed"
        or int(run["frame_count"]) != int(scope["frame_count"])
        or tuple(run["methods"]) != METHOD_ORDER
        or run["spot_ids"] != scope["spots"]
        or run["pairs"] != scope["pairs"]
        or int(run["seed_frame_order"]) != 0
        or not run["single_forward_pass"]
        or not run["correction_and_reinitialization_forbidden"]
        or int(run["manual_annotations"]) != 0
        or not run["manifest_hash_chain_valid"]
    ):
        raise ValueError("Figure 3 completed-run contract changed")
    expected_proposals = int(scope["native_proposals_per_method"])
    expected_fits = int(scope["valid_gaussian_fits_per_method"])
    if run["native_proposals_per_method"] != {
        method: expected_proposals for method in METHOD_ORDER
    }:
        raise ValueError("Figure 3 proposal denominator changed")
    if run["valid_gaussian_fits_per_method"] != {
        method: expected_fits for method in METHOD_ORDER
    }:
        raise ValueError("Figure 3 Gaussian-fit denominator changed")


def _validate_cycle_sources(
    source_directory: Path, contract: dict[str, object]
) -> None:
    profile = read_csv(source_directory / "temperature_profile.csv")
    if [int(row["frame_order"]) for row in profile] != list(range(701)):
        raise ValueError("Figure 3 temperature profile changed")
    if {float(row["energy_ev"]) for row in profile} != {
        float(contract["primary_scope"]["energy_ev"])
    }:
        raise ValueError("Figure 3 fixed-energy scope changed")

    landmarks = read_csv(source_directory / "cycle_landmarks.csv")
    expected_points = contract["summary_points"]
    if [row["landmark"] for row in landmarks] != [
        point["name"] for point in expected_points
    ]:
        raise ValueError("Figure 3 landmark order changed")
    for row, point in zip(landmarks, expected_points):
        if (
            row["reader_label"] != point["reader_label"]
            or int(row["frame_start"]) != int(point["frame_start"])
            or int(row["frame_end"]) != int(point["frame_end"])
            or as_bool(row["outcome_selected"]) != bool(point["outcome_selected"])
        ):
            raise ValueError(f"Figure 3 landmark changed: {point['name']}")
        reader_label = row["reader_label"].lower()
        if "frame" in reader_label or "25w28" in reader_label:
            raise ValueError("Figure 3 reader label leaks archive metadata")


def _validate_endpoint_summaries(source_directory: Path) -> None:
    rows = read_csv(source_directory / "endpoint_summaries.csv")
    expected = {(method, pair) for method in METHOD_ORDER for pair in PAIR_ORDER}
    if {(row["method"], row["pair"]) for row in rows} != expected:
        raise ValueError("Figure 3 endpoint summary identities changed")
    if any(int(row["valid_frame_count"]) != 701 for row in rows):
        raise ValueError("Figure 3 endpoint summary coverage changed")
    if {int(row["minimum_frame_order"]) for row in rows} != {378}:
        raise ValueError("Figure 3 common post-plateau minimum changed")

    spread = read_csv(source_directory / "between_method_spread.csv")
    points = (
        "plateau",
        "post_plateau_minimum",
        "final",
        "entire_sequence_maximum",
    )
    if {(row["pair"], row["summary_point"]) for row in spread} != {
        (pair, point) for pair in PAIR_ORDER for point in points
    }:
        raise ValueError("Figure 3 spread-summary identities changed")
    if any(
        as_bool(row["display_in_panel_d"])
        != (row["summary_point"] != "entire_sequence_maximum")
        for row in spread
    ):
        raise ValueError("Figure 3 panel-d selection changed")
    sequence_maxima = {
        row["pair"]: float(row["spread_percentage_points"])
        for row in spread
        if row["summary_point"] == "entire_sequence_maximum"
    }
    expected_maxima = {
        "1-6": 0.0017096162773233559,
        "3-5": 0.002236605543060044,
    }
    if any(
        not math.isclose(sequence_maxima[pair], expected, abs_tol=1e-15)
        for pair, expected in expected_maxima.items()
    ):
        raise ValueError("Figure 3 complete-sequence spread changed")


def _validate_provenance(source_directory: Path) -> None:
    provenance = json.loads(
        (source_directory / "source_provenance.json").read_text(encoding="utf-8")
    )
    roles = {
        name: source["evidence_role"] for name, source in provenance["sources"].items()
    }
    if roles != {
        "frozen_subset_manifest": "frozen five-method experimental design",
        "completed_run_manifest": "completed-run hash chain",
        "completed_run_summary": "proposal and common-fit denominators",
        "primary_pair_endpoints": "five-method positive control",
        "temperature_profile": ("thermal-cycle coordinates for the five-method run"),
    }:
        raise ValueError("Figure 3 provenance roles changed")
    if any(len(source["sha256"]) != 64 for source in provenance["sources"].values()):
        raise ValueError("Figure 3 provenance hash is invalid")


def validate_source_contract(
    source_directory: Path = SOURCE_DATA_DIRECTORY,
) -> None:
    contract = load_contract()
    _validate_five_method_run_contract(source_directory, contract)
    _validate_primary_timeseries(source_directory, contract)
    _validate_cycle_sources(source_directory, contract)
    _validate_endpoint_summaries(source_directory)
    _validate_provenance(source_directory)
