"""Frozen source-data contract for publication Figure 4."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import tifffile


FIGURE_ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = FIGURE_ROOT / "contract.json"
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
METHOD_ORDER = ("sam2", "samurai", "dam4sam")
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_contract() -> dict[str, object]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("status") != "source_contract_frozen":
        raise ValueError("Figure 4 source contract is not frozen")
    if tuple(contract["controlled_family"]["methods"]) != METHOD_ORDER:
        raise ValueError("Figure 4 controlled method order changed")
    return contract


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
