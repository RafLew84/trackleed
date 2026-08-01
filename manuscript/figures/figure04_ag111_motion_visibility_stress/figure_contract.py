"""Frozen data and rendering contract for publication Figure 4."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


FIGURE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = FIGURE_ROOT.parents[2]
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
OUTPUT_DIRECTORY = FIGURE_ROOT / "output"
FIGURE_BASENAME = "figure04_ag111_motion_visibility_stress"
CONTRACT_PATH = FIGURE_ROOT / "contract.json"
MANIFEST_PATH = FIGURE_ROOT / "manifest.json"

FIGURE_SIZE_INCHES = (7.2, 7.4)
PNG_DPI = 400
FONT_FAMILY = "DejaVu Sans"


@dataclass(frozen=True)
class MethodSpec:
    method: str
    display_name: str
    color: str
    marker: str


METHOD_SPECS = (
    MethodSpec("locotrack", "LocoTrack", "#0072B2", "o"),
    MethodSpec("ostrack", "OSTrack", "#E69F00", "s"),
    MethodSpec("dam4sam", "DAM4SAM", "#009E73", "^"),
    MethodSpec("lk", "Lucas-Kanade", "#CC79A7", "D"),
    MethodSpec("easyleed", "EasyLEED", "#D55E00", "P"),
)
METHOD_ORDER = tuple(spec.method for spec in METHOD_SPECS)
DISPLAY_NAMES = {spec.method: spec.display_name for spec in METHOD_SPECS}
METHOD_COLORS = {spec.method: spec.color for spec in METHOD_SPECS}
METHOD_MARKERS = {spec.method: spec.marker for spec in METHOD_SPECS}
CASE_COUNT_FIELDS = (
    "visible_opportunities",
    "branch_consistent_measurements",
    "evaluated_gaps",
    "recovered_gaps",
    "accepted_measurements_all_frames",
    "geometry_inconsistent_accepted_fits",
    "jointly_consistent_pair_frames",
)
PANEL_ORDER = (
    "representative_trajectory",
    "measurement_coverage",
    "gap_recovery",
    "geometry_inconsistency",
    "conditional_pair_residual",
)
REGIME_COLORS = {
    "visible_fit": "#E8F3E8",
    "iv_visibility_gap": "#F7C6C7",
    "beam_onset_or_termination": "#E6E6E6",
    "apparatus_absence": "#BDBDBD",
}
NEUTRAL_COLOR = "#4D4D4D"
EXPECTED_COLOR = "#333333"
ADVERSE_COLOR = "#A23B3B"
FONT_SIZES_PT = {
    "base": 7.2,
    "axis_title": 8.0,
    "axis_label": 7.2,
    "tick_label": 6.5,
    "legend": 6.1,
    "panel_label": 9.0,
    "value_annotation": 6.2,
    "minor_annotation": 5.8,
    "card_header": 7.0,
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


def as_int(value: object) -> int:
    return int(float(str(value)))


def as_float(value: object) -> float:
    return float(str(value))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_contract() -> dict[str, object]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("status") != "source_contract_frozen":
        raise ValueError("Figure 4 source contract is not frozen")
    if tuple(contract.get("panel_order", ())) != PANEL_ORDER:
        raise ValueError("Figure 4 panel order changed")
    if tuple(contract["main_comparison"]["methods"]) != METHOD_ORDER:
        raise ValueError("Figure 4 method order changed")
    return contract


def parse_aggregate_row(row: dict[str, str]) -> dict[str, object]:
    float_fields = (
        "weighted_fit_coverage",
        "weighted_gap_recovery_rate",
        "geometry_inconsistent_fraction",
        "median_abs_distance_residual_px",
        "p95_abs_distance_residual_px",
        "maximum_abs_distance_residual_px",
    )
    return {
        **row,
        "display_method": DISPLAY_NAMES[row["method"]],
        **{field: as_int(row[field]) for field in CASE_COUNT_FIELDS},
        **{field: as_float(row[field]) for field in float_fields},
    }


def parse_case_row(row: dict[str, str]) -> dict[str, object]:
    float_fields = (
        "fit_coverage",
        "gap_recovery_rate",
        "geometry_inconsistent_fraction",
        "case_median_abs_distance_residual_px",
    )
    return {
        **row,
        "display_method": DISPLAY_NAMES[row["method"]],
        **{field: as_int(row[field]) for field in CASE_COUNT_FIELDS},
        **{field: as_float(row[field]) for field in float_fields},
    }


def load_aggregate_source_data() -> list[dict[str, object]]:
    rows = [
        parse_aggregate_row(row)
        for row in read_csv(SOURCE_DATA_DIRECTORY / "aggregate_method_results.csv")
    ]
    by_method = {str(row["method"]): row for row in rows}
    if set(by_method) != set(METHOD_ORDER):
        raise ValueError("Figure 4 aggregate method set changed")
    return [by_method[method] for method in METHOD_ORDER]


def load_case_source_data() -> list[dict[str, object]]:
    return [
        parse_case_row(row)
        for row in read_csv(SOURCE_DATA_DIRECTORY / "case_results.csv")
    ]


def load_representative_pair_source_data() -> list[dict[str, object]]:
    rows = []
    for row in read_csv(SOURCE_DATA_DIRECTORY / "representative_pair_measurements.csv"):
        rows.append(
            {
                **row,
                "frame": as_int(row["frame"]),
                "energy_ev": as_float(row["energy_ev"]),
                "pair_distance_px": as_float(row["pair_distance_px"]),
                "predicted_distance_px": as_float(row["predicted_distance_px"]),
                "distance_residual_px": as_float(row["distance_residual_px"]),
            }
        )
    if {str(row["method"]) for row in rows} != set(METHOD_ORDER):
        raise ValueError("Representative trajectory method set changed")
    return rows


def load_visibility_source_data() -> list[dict[str, object]]:
    rows = []
    for row in read_csv(
        SOURCE_DATA_DIRECTORY / "representative_visibility_regimes.csv"
    ):
        rows.append(
            {
                **row,
                "frame": as_int(row["frame"]),
                "energy_ev": as_float(row["energy_ev"]),
                "fit_valid": as_bool(row["fit_valid"]),
                "predicted_x": as_float(row["predicted_x"]),
                "predicted_y": as_float(row["predicted_y"]),
            }
        )
    return rows


def load_representative_case_source_data() -> dict[str, object]:
    return json.loads(
        (SOURCE_DATA_DIRECTORY / "representative_case.json").read_text(encoding="utf-8")
    )


def validate_source_contract() -> None:
    contract = load_contract()
    rows = load_aggregate_source_data()
    if any(row["visible_opportunities"] != 221 for row in rows):
        raise ValueError("Figure 4 visible-opportunity denominator changed")
    if any(row["evaluated_gaps"] != 24 for row in rows):
        raise ValueError("Figure 4 gap denominator changed")
    if [row["branch_consistent_measurements"] for row in rows] != [
        217,
        216,
        216,
        188,
        214,
    ]:
        raise ValueError("Figure 4 coverage numerators changed")
    if [row["recovered_gaps"] for row in rows] != [23, 21, 23, 17, 22]:
        raise ValueError("Figure 4 recovery numerators changed")
    cases = load_case_source_data()
    for aggregate_row in rows:
        method_rows = [row for row in cases if row["method"] == aggregate_row["method"]]
        if len(method_rows) != 4:
            raise ValueError("Figure 4 frozen-case count changed")
        for field in CASE_COUNT_FIELDS:
            if sum(row[field] for row in method_rows) != aggregate_row[field]:
                raise ValueError(f"Figure 4 case-to-aggregate mismatch: {field}")

    representative = load_representative_case_source_data()
    frozen = contract["representative_case"]
    for field in (
        "archive_series_id",
        "archive_pair_id",
        "scientific_series_label",
        "scientific_pair_label",
        "selection_rule",
    ):
        if representative.get(field) != frozen[field]:
            raise ValueError(f"Representative-case contract changed: {field}")
    if representative.get("iv_gap_event_count") != 7:
        raise ValueError("Representative natural-gap count changed")

    visibility = load_visibility_source_data()
    if len(visibility) != 70:
        raise ValueError("Representative visibility cardinality changed")
    trajectories = load_representative_pair_source_data()
    expected_counts = {
        "locotrack": 19,
        "ostrack": 20,
        "dam4sam": 20,
        "lk": 14,
        "easyleed": 18,
    }
    actual_counts = {
        method: sum(row["method"] == method for row in trajectories)
        for method in METHOD_ORDER
    }
    if actual_counts != expected_counts:
        raise ValueError("Representative trajectory cardinality changed")

    for row in rows:
        expected_fraction = (
            row["geometry_inconsistent_accepted_fits"]
            / row["accepted_measurements_all_frames"]
        )
        if not math.isclose(
            row["geometry_inconsistent_fraction"],
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Geometry-inconsistency fraction changed")
