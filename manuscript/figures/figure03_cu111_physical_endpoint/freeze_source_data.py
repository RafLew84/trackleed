#!/usr/bin/env python3
"""Freeze and validate the self-contained source package for Figure 3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from statistics import median
from typing import Sequence

from figure_contract import as_bool, read_csv


FIGURE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = FIGURE_ROOT.parents[2]
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
FIVE_METHOD_ROOT = (
    REPOSITORY_ROOT / ".scratch/trackleed-physics-method-paper/artifacts/"
    "cu111-tracker-evaluation-subset/runs/five-methods-v1"
)
PAIR_ENDPOINTS = FIVE_METHOD_ROOT / "pair_endpoints.csv"
INFERENCE_FRAMES = FIVE_METHOD_ROOT.parents[1] / "inference_frames.csv"
SUBSET_MANIFEST = FIVE_METHOD_ROOT.parents[1] / "manifest.json"
RUN_MANIFEST = FIVE_METHOD_ROOT / "run_manifest.json"
RUN_SUMMARY = FIVE_METHOD_ROOT / "summary.json"
METHOD_ORDER = (
    "locotrack_base_offline",
    "ostrack_vitb_256_mae_ce",
    "dam4sam_sam21pp_l",
    "lucas_kanade_skip_anchor",
    "easyleed_kalman",
)
PAIR_ORDER = ("1-6", "3-5")
DISPLAY_NAMES = {
    "locotrack_base_offline": "LocoTrack",
    "ostrack_vitb_256_mae_ce": "OSTrack",
    "dam4sam_sam21pp_l": "DAM4SAM",
    "lucas_kanade_skip_anchor": "Lucas-Kanade",
    "easyleed_kalman": "EasyLEED",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".17g")
    if value is None:
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_five_method_run_contract() -> dict[str, object]:
    """Load the frozen run contract and verify its internal hash chain."""

    subset = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    run = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    summary = json.loads(RUN_SUMMARY.read_text(encoding="utf-8"))
    methods = list(METHOD_ORDER)
    if (
        subset["main_methods"] != methods
        or run["selected_methods"] != methods
        or summary["selected_methods"] != methods
    ):
        raise ValueError("Figure 3 five-method run identity changed")
    method_results = summary["methods"]
    hash_chain_valid = (
        sha256_file(SUBSET_MANIFEST) == run["contract_manifest_sha256"]
        and sha256_file(RUN_SUMMARY) == run["summary_sha256"]
        and sha256_file(PAIR_ENDPOINTS)
        == run["output_files_sha256"]["pair_endpoints.csv"]
        and sha256_file(INFERENCE_FRAMES) == run["input_inventory_sha256"]
    )
    return {
        "status": run["status"],
        "frame_count": int(summary["frame_count"]),
        "methods": methods,
        "spot_ids": [int(spot) for spot in summary["spot_ids"]],
        "pairs": list(summary["pairs"]),
        "seed_frame_order": int(subset["initialization"]["seed_frame_order"]),
        "single_forward_pass": (
            "one forward pass from frame 0" in subset["design"]["inference_sequence"]
        ),
        "correction_and_reinitialization_forbidden": (
            "no frame skipping, correction or reinitialization"
            in subset["design"]["inference_sequence"]
            and all(
                configuration["correction_or_reinitialization"] == "forbidden"
                for configuration in subset["method_configurations"].values()
            )
        ),
        "manual_annotations": int(summary["manual_annotations"]),
        "native_proposals_per_method": {
            method: int(method_results[method]["native_proposal_count"])
            for method in methods
        },
        "valid_gaussian_fits_per_method": {
            method: int(method_results[method]["valid_gaussian_fit_count"])
            for method in methods
        },
        "manifest_hash_chain_valid": hash_chain_valid,
    }


def load_pair_endpoint_timeseries(
    path: Path = PAIR_ENDPOINTS,
) -> list[dict[str, object]]:
    """Load the primary endpoint without admitting the saved-YOLO diagnostic."""

    parsed = [
        {
            **row,
            "scientific_series_label": "Cu(111) thermal cycle",
            "evidence_role": "five-method positive control",
            "display_method": DISPLAY_NAMES[row["method"]],
            "frame_order": int(row["frame_order"]),
            "frame_id": int(row["frame_id"]),
            "pair_valid": as_bool(row["pair_valid"]),
            "distance_px": float(row["distance_px"]),
            "reference_distance_px": float(row["reference_distance_px"]),
            "reference_valid_frame_count": int(row["reference_valid_frame_count"]),
            "delta_r_over_r": float(row["delta_r_over_r"]),
            "delta_r_over_r_percent": float(row["delta_r_over_r_percent"]),
        }
        for row in read_csv(path)
    ]
    by_identity = {
        (row["method"], row["pair"], row["frame_order"]): row for row in parsed
    }
    expected = {
        (method, pair, frame)
        for method in METHOD_ORDER
        for pair in PAIR_ORDER
        for frame in range(701)
    }
    if set(by_identity) != expected:
        raise ValueError("Figure 3 primary endpoint identity set changed")
    return [
        by_identity[(method, pair, frame)]
        for method in METHOD_ORDER
        for pair in PAIR_ORDER
        for frame in range(701)
    ]


def load_temperature_profile(
    path: Path = INFERENCE_FRAMES,
) -> list[dict[str, object]]:
    """Load only scientific coordinates needed to render the thermal cycle."""

    rows = [
        {
            "frame_order": int(row["frame_order"]),
            "frame_id": int(row["frame_id"]),
            "temperature_c": float(row["temperature_c"]),
            "energy_ev": float(row["energy_ev"]),
        }
        for row in read_csv(path)
    ]
    if [row["frame_order"] for row in rows] != list(range(701)):
        raise ValueError("Figure 3 temperature-profile identity set changed")
    return rows


def build_endpoint_summaries(
    timeseries: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize the predefined plateau, post-plateau minimum, and final."""

    summaries: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        for pair in PAIR_ORDER:
            rows = [
                row
                for row in timeseries
                if row["method"] == method and row["pair"] == pair
            ]
            valid = [row for row in rows if row["pair_valid"]]
            plateau = [
                float(row["delta_r_over_r_percent"])
                for row in valid
                if 372 <= int(row["frame_order"]) <= 376
            ]
            post_plateau = [row for row in valid if int(row["frame_order"]) >= 377]
            minimum = min(
                post_plateau,
                key=lambda row: float(row["delta_r_over_r_percent"]),
            )
            final = next(row for row in valid if int(row["frame_order"]) == 700)
            summaries.append(
                {
                    "method": method,
                    "display_method": DISPLAY_NAMES[method],
                    "pair": pair,
                    "valid_frame_count": len(valid),
                    "reference_valid_frame_count": int(
                        rows[0]["reference_valid_frame_count"]
                    ),
                    "reference_distance_px": float(rows[0]["reference_distance_px"]),
                    "plateau_median_percent": median(plateau),
                    "minimum_frame_order": int(minimum["frame_order"]),
                    "minimum_percent": float(minimum["delta_r_over_r_percent"]),
                    "final_percent": float(final["delta_r_over_r_percent"]),
                }
            )
    return summaries


def build_cycle_landmarks(
    temperature_profile: list[dict[str, object]],
    endpoint_summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Expose display labels and retain frame identities only as provenance."""

    temperature = {
        int(row["frame_order"]): float(row["temperature_c"])
        for row in temperature_profile
    }
    minimum_frames = {int(row["minimum_frame_order"]) for row in endpoint_summaries}
    if minimum_frames != {378}:
        raise ValueError("Figure 3 methods no longer share the endpoint minimum")

    specifications = (
        (
            "reference",
            "Reference (10-16 °C)",
            0,
            9,
            "median valid pair distance; at least 8 of 10 required",
            False,
        ),
        (
            "plateau",
            "800 °C plateau",
            372,
            376,
            "predefined five-image median",
            False,
        ),
        (
            "post_plateau_minimum",
            "Post-plateau minimum (797 °C)",
            378,
            378,
            "minimum after the 800 °C plateau; common to all methods and pairs",
            True,
        ),
        (
            "final",
            "Final cooled state (92 °C)",
            700,
            700,
            "predefined final image",
            False,
        ),
    )
    return [
        {
            "landmark": name,
            "reader_label": label,
            "frame_start": start,
            "frame_end": end,
            "temperature_min_c": min(
                temperature[frame] for frame in range(start, end + 1)
            ),
            "temperature_max_c": max(
                temperature[frame] for frame in range(start, end + 1)
            ),
            "selection_basis": basis,
            "outcome_selected": outcome_selected,
        }
        for name, label, start, end, basis, outcome_selected in specifications
    ]


def _build_spread_row(
    *,
    pair: str,
    summary_point: str,
    reader_label: str,
    values: list[tuple[str, float]],
    frame_order: int | None,
    display_in_panel_d: bool,
) -> dict[str, object]:
    minimum_method, minimum_value = min(values, key=lambda item: item[1])
    maximum_method, maximum_value = max(values, key=lambda item: item[1])
    return {
        "pair": pair,
        "summary_point": summary_point,
        "reader_label": reader_label,
        "frame_order": frame_order,
        "minimum_method": minimum_method,
        "minimum_value_percent": minimum_value,
        "maximum_method": maximum_method,
        "maximum_value_percent": maximum_value,
        "spread_percentage_points": maximum_value - minimum_value,
        "display_in_panel_d": display_in_panel_d,
    }


def build_between_method_spread(
    timeseries: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Compute method range at reader-facing landmarks and over all images."""

    summaries = build_endpoint_summaries(timeseries)
    output: list[dict[str, object]] = []
    for pair in PAIR_ORDER:
        pair_summaries = [row for row in summaries if row["pair"] == pair]
        point_specs = (
            (
                "plateau",
                "800 °C plateau",
                "plateau_median_percent",
                None,
                True,
            ),
            (
                "post_plateau_minimum",
                "Post-plateau minimum (797 °C)",
                "minimum_percent",
                378,
                True,
            ),
            (
                "final",
                "Final cooled state (92 °C)",
                "final_percent",
                700,
                True,
            ),
        )
        for name, label, value_field, frame_order, displayed in point_specs:
            values = [
                (str(row["method"]), float(row[value_field])) for row in pair_summaries
            ]
            output.append(
                _build_spread_row(
                    pair=pair,
                    summary_point=name,
                    reader_label=label,
                    values=values,
                    frame_order=frame_order,
                    display_in_panel_d=displayed,
                )
            )

        by_frame: dict[int, list[tuple[str, float]]] = {}
        for row in timeseries:
            if row["pair"] != pair:
                continue
            frame = int(row["frame_order"])
            by_frame.setdefault(frame, []).append(
                (str(row["method"]), float(row["delta_r_over_r_percent"]))
            )
        frame_order, values = max(
            by_frame.items(),
            key=lambda item: max(value for _, value in item[1])
            - min(value for _, value in item[1]),
        )
        output.append(
            _build_spread_row(
                pair=pair,
                summary_point="entire_sequence_maximum",
                reader_label="Maximum over complete sequence",
                values=values,
                frame_order=frame_order,
                display_in_panel_d=False,
            )
        )
    return output


def _portable_path(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def _source_provenance() -> dict[str, object]:
    sources = {
        "frozen_subset_manifest": (
            SUBSET_MANIFEST,
            "frozen five-method experimental design",
        ),
        "completed_run_manifest": (
            RUN_MANIFEST,
            "completed-run hash chain",
        ),
        "completed_run_summary": (
            RUN_SUMMARY,
            "proposal and common-fit denominators",
        ),
        "primary_pair_endpoints": (
            PAIR_ENDPOINTS,
            "five-method positive control",
        ),
        "temperature_profile": (
            INFERENCE_FRAMES,
            "thermal-cycle coordinates for the five-method run",
        ),
    }
    return {
        "schema_version": "1.0",
        "figure": "Figure 3",
        "sources": {
            name: {
                "path": _portable_path(path),
                "sha256": sha256_file(path),
                "evidence_role": evidence_role,
            }
            for name, (path, evidence_role) in sources.items()
        },
    }


def build(output_root: Path = FIGURE_ROOT) -> None:
    source = output_root / "source_data"
    timeseries = load_pair_endpoint_timeseries()
    summaries = build_endpoint_summaries(timeseries)
    profile = load_temperature_profile()
    _write_json(
        source / "five_method_run_contract.json",
        load_five_method_run_contract(),
    )
    _write_csv(source / "temperature_profile.csv", profile)
    _write_csv(
        source / "cycle_landmarks.csv",
        build_cycle_landmarks(profile, summaries),
    )
    _write_csv(source / "pair_endpoint_timeseries.csv", timeseries)
    _write_csv(source / "endpoint_summaries.csv", summaries)
    _write_csv(
        source / "between_method_spread.csv",
        build_between_method_spread(timeseries),
    )
    _write_json(source / "source_provenance.json", _source_provenance())


def validate(*, check_upstream: bool = True) -> None:
    from figure_contract import validate_source_contract

    validate_source_contract(SOURCE_DATA_DIRECTORY)
    provenance = json.loads(
        (SOURCE_DATA_DIRECTORY / "source_provenance.json").read_text(encoding="utf-8")
    )
    if check_upstream:
        for name, source in provenance["sources"].items():
            path = REPOSITORY_ROOT / source["path"]
            if not path.is_file() or sha256_file(path) != source["sha256"]:
                raise ValueError(f"Figure 3 upstream hash mismatch: {name}")

        with tempfile.TemporaryDirectory(
            prefix="trackleed-figure03-source-"
        ) as temporary:
            rebuilt_root = Path(temporary)
            build(rebuilt_root)
            committed_files = sorted(
                path.relative_to(FIGURE_ROOT)
                for path in SOURCE_DATA_DIRECTORY.rglob("*")
                if path.is_file()
            )
            rebuilt_files = sorted(
                path.relative_to(rebuilt_root)
                for path in (rebuilt_root / "source_data").rglob("*")
                if path.is_file()
            )
            if committed_files != rebuilt_files:
                raise ValueError("Figure 3 source-data file set changed")
            for relative_path in committed_files:
                if sha256_file(FIGURE_ROOT / relative_path) != sha256_file(
                    rebuilt_root / relative_path
                ):
                    raise ValueError(
                        f"Non-deterministic Figure 3 source: {relative_path}"
                    )
    print(
        "Figure 3 source validation passed: one 701-image Cu(111) thermal "
        "cycle, five methods, and two primary pairs."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate committed source data without rewriting it.",
    )
    parser.add_argument(
        "--offline-integrity-only",
        action="store_true",
        help="Skip upstream hashes and deterministic source rebuild.",
    )
    arguments = parser.parse_args()
    if arguments.validate_only:
        validate(check_upstream=not arguments.offline_integrity_only)
        return
    build()
    validate()


if __name__ == "__main__":
    main()
