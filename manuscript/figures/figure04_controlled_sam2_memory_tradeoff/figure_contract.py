"""Frozen source-data contract for publication Figure 4."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import tifffile


FIGURE_ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = FIGURE_ROOT / "contract.json"
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
OUTPUT_DIRECTORY = FIGURE_ROOT / "output"
FIGURE_BASENAME = "figure04_controlled_sam2_memory_tradeoff"
MANIFEST_PATH = FIGURE_ROOT / "manifest.json"

FIGURE_SIZE_INCHES = (7.2, 9.0)
PNG_DPI = 400
FONT_FAMILY = "DejaVu Sans"


@dataclass(frozen=True)
class MethodSpec:
    method: str
    display_name: str
    color: str
    marker: str


METHOD_SPECS = (
    MethodSpec("sam2", "SAM2", "#0072B2", "o"),
    MethodSpec("samurai", "SAMURAI", "#D55E00", "s"),
    MethodSpec("dam4sam", "DAM4SAM", "#009E73", "^"),
)
METHOD_ORDER = tuple(spec.method for spec in METHOD_SPECS)
DISPLAY_NAMES = {spec.method: spec.display_name for spec in METHOD_SPECS}
METHOD_COLORS = {spec.method: spec.color for spec in METHOD_SPECS}
METHOD_MARKERS = {spec.method: spec.marker for spec in METHOD_SPECS}
PANEL_ORDER = (
    "representative_visibility_gap",
    "coverage_by_pair",
    "gap_recovery_by_pair",
    "failure_cost_and_conditional_residual_by_pair",
    "mechanistic_tradeoff_summary",
)
NEUTRAL_COLOR = "#4D4D4D"
ADVERSE_COLOR = "#A23B3B"
ACCEPTED_COLOR = "#F0E442"
FONT_SIZES_PT = {
    "base": 7.1,
    "axis_title": 8.0,
    "axis_label": 7.1,
    "tick_label": 6.3,
    "legend": 6.0,
    "panel_label": 9.2,
    "value_annotation": 5.9,
    "minor_annotation": 5.6,
    "card_header": 7.0,
    "card_value": 6.2,
}
COUNT_FIELDS = (
    "visible_opportunities",
    "branch_consistent_measurements",
    "evaluated_gaps",
    "recovered_gaps",
    "accepted_measurements_all_frames",
    "geometry_inconsistent_accepted_fits",
    "jointly_consistent_pair_frames",
)
EXPECTED_CROPS = (
    "b1_169p2ev.tiff",
    "b1_166p5ev.tiff",
    "b1_163p9ev.tiff",
    "b1_160p9ev.tiff",
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


def load_contract() -> dict[str, object]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("status") != "source_contract_frozen":
        raise ValueError("Figure 4 source contract is not frozen")
    if tuple(contract["controlled_family"]["methods"]) != METHOD_ORDER:
        raise ValueError("Figure 4 controlled method order changed")
    if tuple(contract["panel_order"]) != PANEL_ORDER:
        raise ValueError("Figure 4 panel order changed")
    return contract


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_int(value: object) -> int:
    return int(float(str(value)))


def as_float(value: object) -> float:
    return float(str(value))


def optional_float(value: object) -> float | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return float(text)


def load_aggregate_source_data() -> list[dict[str, object]]:
    integer_fields = COUNT_FIELDS
    float_fields = (
        "fit_coverage",
        "gap_recovery_rate",
        "geometry_inconsistent_fraction",
        "pair1_conditional_median_abs_distance_residual_px",
        "pair2_conditional_median_abs_distance_residual_px",
    )
    rows = [
        {
            **row,
            "identical_checkpoint_weights": as_bool(
                row["identical_checkpoint_weights"]
            ),
            **{field: as_int(row[field]) for field in integer_fields},
            **{field: as_float(row[field]) for field in float_fields},
        }
        for row in read_csv(SOURCE_DATA_DIRECTORY / "family_aggregate.csv")
    ]
    by_method = {str(row["method"]): row for row in rows}
    if set(by_method) != set(METHOD_ORDER):
        raise ValueError("Figure 4 aggregate method set changed")
    return [by_method[method] for method in METHOD_ORDER]


def load_pair_source_data() -> list[dict[str, object]]:
    integer_fields = (
        *COUNT_FIELDS,
        "zero_delay_recoveries",
    )
    float_fields = (
        "fit_coverage",
        "gap_recovery_rate",
        "geometry_inconsistent_fraction",
        "conditional_median_abs_distance_residual_px",
        "conditional_p95_abs_distance_residual_px",
    )
    rows = [
        {
            **row,
            **{field: as_int(row[field]) for field in integer_fields},
            **{field: as_float(row[field]) for field in float_fields},
        }
        for row in read_csv(SOURCE_DATA_DIRECTORY / "family_by_pair.csv")
    ]
    expected = {
        (pair, method)
        for pair in ("pair 1", "pair 2")
        for method in METHOD_ORDER
    }
    if {(row["pair"], row["method"]) for row in rows} != expected:
        raise ValueError("Figure 4 pair-by-method identities changed")
    by_identity = {
        (str(row["pair"]), str(row["method"])): row for row in rows
    }
    return [
        by_identity[(pair, method)]
        for pair in ("pair 1", "pair 2")
        for method in METHOD_ORDER
    ]


def load_representative_gap_source_data() -> list[dict[str, object]]:
    boolean_fields = ("native_visible", "fit_valid", "measurement_valid")
    optional_fields = (
        "native_x",
        "native_y",
        "gaussian_center_x",
        "gaussian_center_y",
        "branch_error_px",
    )
    rows = [
        {
            **row,
            "energy_ev": as_float(row["energy_ev"]),
            **{field: as_bool(row[field]) for field in boolean_fields},
            **{field: optional_float(row[field]) for field in optional_fields},
        }
        for row in read_csv(
            SOURCE_DATA_DIRECTORY / "representative_gap_measurements.csv"
        )
    ]
    if len(rows) != 12:
        raise ValueError("Figure 4 representative-gap cardinality changed")
    return rows


def load_representative_gap_metadata() -> dict[str, object]:
    return json.loads(
        (SOURCE_DATA_DIRECTORY / "representative_gap.json").read_text(
            encoding="utf-8"
        )
    )


def load_representative_crops() -> dict[float, object]:
    metadata = load_representative_gap_metadata()
    crops = {
        float(energy): tifffile.imread(
            SOURCE_DATA_DIRECTORY / str(relative_path)
        )
        for energy, relative_path in metadata["crop_files"].items()
    }
    expected_energies = {
        float(energy) for energy in metadata["display_energies_ev"]
    }
    if set(crops) != expected_energies:
        raise ValueError("Figure 4 representative crop energies changed")
    return crops


def shared_crop_intensity_limits() -> tuple[float, float]:
    crops = load_representative_crops()
    lower = min(float(crop.min()) for crop in crops.values())
    upper = max(float(crop.max()) for crop in crops.values())
    if not lower < upper:
        raise ValueError("Figure 4 crop intensity range collapsed")
    return lower, upper


def validate_source_contract(
    source_directory: Path = SOURCE_DATA_DIRECTORY,
) -> None:
    contract = load_contract()
    aggregate = read_csv(source_directory / "family_aggregate.csv")
    if [row["method"] for row in aggregate] != list(METHOD_ORDER):
        raise ValueError("Figure 4 aggregate method order changed")
    controlled = contract["controlled_family"]
    if any(
        int(row["visible_opportunities"])
        != controlled["visible_opportunities_per_method"]
        for row in aggregate
    ):
        raise ValueError("Figure 4 visible-opportunity denominator changed")
    if any(
        int(row["evaluated_gaps"]) != controlled["evaluated_gaps_per_method"]
        for row in aggregate
    ):
        raise ValueError("Figure 4 natural-gap denominator changed")
    if {
        row["checkpoint_sha256"] for row in aggregate
    } != {controlled["checkpoint_sha256"]}:
        raise ValueError("Figure 4 checkpoint identity changed")
    if any(row["identical_checkpoint_weights"] != "true" for row in aggregate):
        raise ValueError("Figure 4 weights are no longer a matched control")

    by_pair = read_csv(source_directory / "family_by_pair.csv")
    if len(by_pair) != 6:
        raise ValueError("Figure 4 pair-by-method cardinality changed")
    required_panel_metrics = {
        metric
        for metrics in contract["panel_metrics"].values()
        for metric in metrics
    }
    if not required_panel_metrics.issubset(by_pair[0]):
        missing = sorted(required_panel_metrics - set(by_pair[0]))
        raise ValueError(f"Figure 4 panel metrics missing: {missing}")
    for aggregate_row in aggregate:
        method_rows = [
            row for row in by_pair if row["method"] == aggregate_row["method"]
        ]
        if len(method_rows) != 2:
            raise ValueError("Figure 4 pair count changed")
        for field in COUNT_FIELDS:
            if sum(int(row[field]) for row in method_rows) != int(
                aggregate_row[field]
            ):
                raise ValueError(
                    f"Figure 4 pair-to-aggregate mismatch: {field}"
                )
    if any(
        int(row["zero_delay_recoveries"]) > int(row["recovered_gaps"])
        for row in by_pair
    ):
        raise ValueError("Figure 4 zero-delay recovery counts are invalid")

    pair_definition = contract["pair_definition"]
    pair_seeds = read_csv(source_directory / "pair_seeds.csv")
    if {
        (row["pair"], row["branch"]) for row in pair_seeds
    } != {
        ("pair 1", "B1"),
        ("pair 1", "B2"),
        ("pair 2", "B1"),
        ("pair 2", "B2"),
    }:
        raise ValueError("Figure 4 frozen pair identities changed")
    if any(
        float(row["reference_energy_ev"])
        != float(pair_definition["reference_energy_ev"])
        for row in pair_seeds
    ):
        raise ValueError("Figure 4 pair reference energy changed")
    pair_1 = [row for row in pair_seeds if row["pair"] == "pair 1"]
    pair_2 = [row for row in pair_seeds if row["pair"] == "pair 2"]
    minimum_separation = float(
        pair_definition["minimum_cross_pair_seed_separation_px"]
    )
    if any(
        math.hypot(
            float(first["center_x"]) - float(second["center_x"]),
            float(first["center_y"]) - float(second["center_y"]),
        )
        < minimum_separation
        for first in pair_1
        for second in pair_2
    ):
        raise ValueError("Figure 4 frozen pairs are not spatially disjoint")

    representative = json.loads(
        (source_directory / "representative_gap.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_gap = contract["representative_gap"]
    for field in (
        "scientific_series_label",
        "scientific_pair_label",
        "branch",
        "direction",
        "display_energies_ev",
        "display_normalization",
        "selection_uses_method_outcomes",
    ):
        if representative[field] != frozen_gap[field]:
            raise ValueError(f"Figure 4 representative gap changed: {field}")

    gap_rows = read_csv(
        source_directory / "representative_gap_measurements.csv"
    )
    if len(gap_rows) != 12:
        raise ValueError("Figure 4 representative-gap cardinality changed")
    if {row["method"] for row in gap_rows} != set(METHOD_ORDER):
        raise ValueError("Figure 4 representative-gap method set changed")
    if {
        float(row["energy_ev"]) for row in gap_rows
    } != set(frozen_gap["display_energies_ev"]):
        raise ValueError("Figure 4 representative-gap energies changed")

    for filename in EXPECTED_CROPS:
        crop = tifffile.imread(source_directory / "crops" / filename)
        if crop.shape != (128, 128) or crop.dtype.name != "uint8":
            raise ValueError(f"Figure 4 crop contract changed: {filename}")
