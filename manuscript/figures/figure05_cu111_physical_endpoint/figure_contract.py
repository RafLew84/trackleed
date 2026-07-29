"""Frozen source-data contract for publication Figure 5."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


FIGURE_ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = FIGURE_ROOT / "contract.json"
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
METHOD_ORDER = (
    "locotrack_base_offline",
    "ostrack_vitb_256_mae_ce",
    "dam4sam_sam21pp_l",
    "lucas_kanade_skip_anchor",
    "easyleed_kalman",
)
PAIR_ORDER = ("1-6", "3-5")
PANEL_ORDER = (
    "thermal_cycle_landmarks",
    "pair_1_6_endpoint",
    "pair_3_5_endpoint",
    "between_method_endpoint_spread",
    "separate_pair_2_4_fit_missingness",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_contract() -> dict[str, object]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("status") != "source_contract_frozen":
        raise ValueError("Figure 5 source contract is not frozen")
    if tuple(contract["primary_scope"]["methods"]) != METHOD_ORDER:
        raise ValueError("Figure 5 primary method order changed")
    if tuple(contract["primary_scope"]["pairs"]) != PAIR_ORDER:
        raise ValueError("Figure 5 primary pair order changed")
    if tuple(contract["panel_order"]) != PANEL_ORDER:
        raise ValueError("Figure 5 panel order changed")
    if (
        contract["run_contract_source"]
        != "source_data/five_method_run_contract.json"
    ):
        raise ValueError("Figure 5 completed-run source changed")
    return contract


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
    identities = {
        (row["method"], row["pair"], row["frame_order"]) for row in rows
    }
    if identities != expected or len(rows) != len(expected):
        raise ValueError("Figure 5 primary endpoint identities changed")
    series_label = contract["primary_scope"]["scientific_series_label"]
    if {row["scientific_series_label"] for row in rows} != {series_label}:
        raise ValueError("Figure 5 reader-facing series label changed")
    if {row["evidence_role"] for row in rows} != {
        "five-method positive control"
    }:
        raise ValueError("Figure 5 primary evidence role changed")
    if any(not as_bool(row["pair_valid"]) for row in rows):
        raise ValueError("Figure 5 primary endpoint coverage changed")
    if {int(row["reference_valid_frame_count"]) for row in rows} != {10}:
        raise ValueError("Figure 5 reference denominator changed")


def _validate_five_method_run_contract(
    source_directory: Path, contract: dict[str, object]
) -> None:
    run = json.loads(
        (source_directory / "five_method_run_contract.json").read_text(
            encoding="utf-8"
        )
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
        raise ValueError("Figure 5 completed-run contract changed")
    expected_proposals = int(scope["native_proposals_per_method"])
    expected_fits = int(scope["valid_gaussian_fits_per_method"])
    if run["native_proposals_per_method"] != {
        method: expected_proposals for method in METHOD_ORDER
    }:
        raise ValueError("Figure 5 proposal denominator changed")
    if run["valid_gaussian_fits_per_method"] != {
        method: expected_fits for method in METHOD_ORDER
    }:
        raise ValueError("Figure 5 Gaussian-fit denominator changed")


def _validate_cycle_sources(
    source_directory: Path, contract: dict[str, object]
) -> None:
    profile = read_csv(source_directory / "temperature_profile.csv")
    if [int(row["frame_order"]) for row in profile] != list(range(701)):
        raise ValueError("Figure 5 temperature profile changed")
    if {float(row["energy_ev"]) for row in profile} != {
        float(contract["primary_scope"]["energy_ev"])
    }:
        raise ValueError("Figure 5 fixed-energy scope changed")

    landmarks = read_csv(source_directory / "cycle_landmarks.csv")
    expected_points = contract["summary_points"]
    if [row["landmark"] for row in landmarks] != [
        point["name"] for point in expected_points
    ]:
        raise ValueError("Figure 5 landmark order changed")
    for row, point in zip(landmarks, expected_points):
        if (
            row["reader_label"] != point["reader_label"]
            or int(row["frame_start"]) != int(point["frame_start"])
            or int(row["frame_end"]) != int(point["frame_end"])
            or as_bool(row["outcome_selected"])
            != bool(point["outcome_selected"])
        ):
            raise ValueError(f"Figure 5 landmark changed: {point['name']}")
        reader_label = row["reader_label"].lower()
        if "frame" in reader_label or "25w28" in reader_label:
            raise ValueError("Figure 5 reader label leaks archive metadata")


def _validate_endpoint_summaries(source_directory: Path) -> None:
    rows = read_csv(source_directory / "endpoint_summaries.csv")
    expected = {(method, pair) for method in METHOD_ORDER for pair in PAIR_ORDER}
    if {(row["method"], row["pair"]) for row in rows} != expected:
        raise ValueError("Figure 5 endpoint summary identities changed")
    if any(int(row["valid_frame_count"]) != 701 for row in rows):
        raise ValueError("Figure 5 endpoint summary coverage changed")
    if {int(row["minimum_frame_order"]) for row in rows} != {378}:
        raise ValueError("Figure 5 common post-plateau minimum changed")

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
        raise ValueError("Figure 5 spread-summary identities changed")
    if any(
        as_bool(row["display_in_panel_d"])
        != (row["summary_point"] != "entire_sequence_maximum")
        for row in spread
    ):
        raise ValueError("Figure 5 panel-d selection changed")
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
        raise ValueError("Figure 5 complete-sequence spread changed")


def _validate_separate_diagnostic(
    source_directory: Path, contract: dict[str, object]
) -> None:
    rows = read_csv(
        source_directory / "pair_2_4_missingness_diagnostic.csv"
    )
    diagnostic = contract["separate_diagnostic"]
    if [int(row["frame_id"]) for row in rows] != list(range(356, 382)):
        raise ValueError("Figure 5 pair 2-4 diagnostic context changed")
    if {row["pair"] for row in rows} != {diagnostic["pair"]}:
        raise ValueError("Figure 5 diagnostic pair changed")
    if {row["tracking_source"] for row in rows} != {
        diagnostic["tracking_source"]
    }:
        raise ValueError("Figure 5 diagnostic tracking source changed")
    if any(as_bool(row["inside_five_method_run"]) for row in rows):
        raise ValueError("Figure 5 diagnostic was mixed into five-method run")
    rejected = [row for row in rows if not as_bool(row["refit_accepted"])]
    if len(rejected) != int(diagnostic["rejected_fit_count"]):
        raise ValueError("Figure 5 diagnostic rejection count changed")
    if any(
        not as_bool(row["tracking_roi_present"])
        or not as_bool(row["r_squared_gate"])
        or as_bool(row["amplitude_gate"])
        or not as_bool(row["center_gate"])
        or not as_bool(row["sigma_gate"])
        or row["rejection_reason"]
        != "amplitude_below_or_outside_threshold"
        for row in rejected
    ):
        raise ValueError("Figure 5 diagnostic is no longer amplitude-only")


def _validate_provenance(source_directory: Path) -> None:
    provenance = json.loads(
        (source_directory / "source_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    roles = {
        name: source["evidence_role"]
        for name, source in provenance["sources"].items()
    }
    if roles != {
        "frozen_subset_manifest": "frozen five-method experimental design",
        "completed_run_manifest": "completed-run hash chain",
        "completed_run_summary": "proposal and common-fit denominators",
        "primary_pair_endpoints": "five-method positive control",
        "temperature_profile": (
            "thermal-cycle coordinates for the five-method run"
        ),
        "pair_2_4_fit_diagnostics": (
            "separate saved-YOLO missingness diagnostic"
        ),
    }:
        raise ValueError("Figure 5 provenance roles changed")
    if any(
        len(source["sha256"]) != 64
        for source in provenance["sources"].values()
    ):
        raise ValueError("Figure 5 provenance hash is invalid")


def validate_source_contract(
    source_directory: Path = SOURCE_DATA_DIRECTORY,
) -> None:
    contract = load_contract()
    _validate_five_method_run_contract(source_directory, contract)
    _validate_primary_timeseries(source_directory, contract)
    _validate_cycle_sources(source_directory, contract)
    _validate_endpoint_summaries(source_directory)
    _validate_separate_diagnostic(source_directory, contract)
    _validate_provenance(source_directory)
