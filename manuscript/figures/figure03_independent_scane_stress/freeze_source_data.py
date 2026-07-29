#!/usr/bin/env python3
"""Freeze and validate the self-contained source package for Figure 3."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Sequence

from figure_contract import (
    CONTRACT_PATH,
    DISPLAY_NAMES,
    FIGURE_ROOT,
    METHOD_ORDER,
    REPOSITORY_ROOT,
    SOURCE_DATA_DIRECTORY,
    as_bool,
    as_float,
    as_int,
    parse_aggregate_row,
    parse_case_row,
    read_csv,
    sha256_file,
    validate_source_contract,
)


INDEPENDENT_ROOT = (
    REPOSITORY_ROOT
    / ".scratch/trackleed-physics-method-paper/artifacts/"
    "independent-scane-validation"
)
INDEPENDENT_AGGREGATE = INDEPENDENT_ROOT / "aggregate_method_results.csv"
INDEPENDENT_CASE_RESULTS = INDEPENDENT_ROOT / "case_results.csv"
INDEPENDENT_SUMMARY = INDEPENDENT_ROOT / "summary.json"
EXTERNAL_ROOT = (
    REPOSITORY_ROOT
    / ".scratch/trackleed-physics-method-paper/artifacts/"
    "domain-baseline-high-energy"
)
EXTERNAL_SUMMARY = EXTERNAL_ROOT / "summary.json"
EXTERNAL_GAP_RECOVERY = EXTERNAL_ROOT / "gap_recovery.csv"
VIPERLEED_PATTERN_X = (
    EXTERNAL_ROOT / "viperleed/viperleed_high_energy_x.csv"
)

SOURCE_DATA_FILES = (
    "aggregate_method_results.csv",
    "case_results.csv",
    "representative_pair_measurements.csv",
    "representative_visibility_regimes.csv",
    "representative_case.json",
    "external_reference_summary.json",
    "source_provenance.json",
)
SELECTION_RULE = (
    "Order frozen cases by scientific series and pair identity; select the "
    "first case that passed the predeclared coherent-pairwise-energy-law "
    "check and contains at least one natural I(V) visibility gap. Do not "
    "inspect method outcomes."
)


def load_aggregate_results(
    path: Path = INDEPENDENT_AGGREGATE,
) -> list[dict[str, object]]:
    rows = [parse_aggregate_row(row) for row in read_csv(path)]
    by_method = {str(row["method"]): row for row in rows}
    if set(by_method) != set(METHOD_ORDER):
        raise ValueError("Independent ScanE aggregate method set changed")
    return [by_method[method] for method in METHOD_ORDER]


def load_case_results(
    path: Path = INDEPENDENT_CASE_RESULTS,
) -> list[dict[str, object]]:
    return [parse_case_row(row) for row in read_csv(path)]


def select_representative_case(
    summary: dict[str, object],
) -> dict[str, object]:
    series_order = {"26a21": (0, "Scan A"), "26a21b": (1, "Scan B")}
    cases = sorted(
        summary["baseline_cases"],
        key=lambda case: (
            series_order[str(case["series"])][0],
            int(str(case["pair"]).replace("pair", "")),
        ),
    )
    for case in cases:
        checks = case["decision_checks"]
        if (
            checks["coherent_pairwise_energy_law"]
            and checks["natural_iv_gap_available"]
        ):
            iv_gaps = [
                gap
                for gap in case["visibility_gaps"]
                if gap["classification"] == "iv_visibility_gap"
            ]
            return {
                "archive_series_id": case["series"],
                "archive_pair_id": case["pair"],
                "scientific_series_label": series_order[case["series"]][1],
                "scientific_pair_label": (
                    f"pair {int(str(case['pair']).replace('pair', ''))}"
                ),
                "selection_rule": SELECTION_RULE,
                "selection_uses_method_outcomes": False,
                "frame_count": as_int(case["frame_count"]),
                "energy_range_ev": [
                    as_float(value) for value in case["energy_range_ev"]
                ],
                "reference_energy_ev": as_float(case["reference_energy_ev"]),
                "reference_frame": as_int(case["reference_frame"]),
                "decision_checks": checks,
                "iv_gap_event_count": len(iv_gaps),
                "visibility_gaps": case["visibility_gaps"],
            }
    raise ValueError("No representative case satisfies the frozen physics rule")


def _representative_pair_paths(
    selected: dict[str, object],
) -> dict[str, Path]:
    root = (
        INDEPENDENT_ROOT
        / str(selected["archive_series_id"])
        / str(selected["archive_pair_id"])
    )
    return {
        "locotrack": root
        / "representatives/locotrack_pair_measurements.csv",
        "ostrack": root / "representatives/ostrack_pair_measurements.csv",
        "dam4sam": root / "trackers/dam4sam_pair_measurements.csv",
        "lk": root / "representatives/lk_pair_measurements.csv",
        "easyleed": root / "easyleed/easyleed_pair_measurements.csv",
    }


def build_representative_pair_rows(
    selected: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, path in _representative_pair_paths(selected).items():
        for source in read_csv(path):
            rows.append(
                {
                    "scientific_series_label": selected[
                        "scientific_series_label"
                    ],
                    "scientific_pair_label": selected["scientific_pair_label"],
                    "method": method,
                    "display_method": DISPLAY_NAMES[method],
                    "frame": as_int(source["frame"]),
                    "energy_ev": as_float(source["energy_ev"]),
                    "pair_distance_px": as_float(source["pair_distance_px"]),
                    "predicted_distance_px": as_float(
                        source["predicted_distance_px"]
                    ),
                    "distance_residual_px": as_float(
                        source["distance_residual_px"]
                    ),
                }
            )
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    return sorted(rows, key=lambda row: (order[row["method"]], row["frame"]))


def build_visibility_rows(
    selected: dict[str, object],
) -> list[dict[str, object]]:
    path = (
        INDEPENDENT_ROOT
        / str(selected["archive_series_id"])
        / str(selected["archive_pair_id"])
        / "scan_measurements.csv"
    )
    return [
        {
            "scientific_series_label": selected["scientific_series_label"],
            "scientific_pair_label": selected["scientific_pair_label"],
            "frame": as_int(row["frame"]),
            "energy_ev": as_float(row["energy_ev"]),
            "branch": row["branch"],
            "status": row["status"],
            "fit_valid": as_bool(row["fit_valid"]),
            "predicted_x": as_float(row["predicted_x"]),
            "predicted_y": as_float(row["predicted_y"]),
        }
        for row in read_csv(path)
    ]


def _viperleed_beam_count() -> int:
    with VIPERLEED_PATTERN_X.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    return len(header) - 1


def build_external_reference_summary() -> dict[str, object]:
    summary = json.loads(EXTERNAL_SUMMARY.read_text(encoding="utf-8"))
    methods = summary["methods"]
    fit_counts = {
        as_int(methods[name]["valid_common_fits"]) for name in methods
    }
    proposal_counts = {
        as_int(methods[name]["proposal_rows"]) for name in methods
    }
    gap_checks = {
        (
            as_int(methods[name]["branch_gap_checks_passed"]),
            as_int(methods[name]["branch_gap_checks_total"]),
        )
        for name in methods
    }
    if len(fit_counts) != 1 or len(proposal_counts) != 1 or len(gap_checks) != 1:
        raise ValueError("External methods no longer share the frozen denominator")
    gap_passed, gap_total = next(iter(gap_checks))
    per_method = {
        name: {
            "valid_common_fits": as_int(payload["valid_common_fits"]),
            "branch_opportunities": as_int(payload["proposal_rows"]),
            "pair_1_valid_frames": as_int(
                payload["per_pair"]["P1"]["valid_pair_frames"]
            ),
            "pair_2_valid_frames": as_int(
                payload["per_pair"]["P2"]["valid_pair_frames"]
            ),
            "pair_1_median_abs_residual_px": as_float(
                payload["per_pair"]["P1"]["pair_abs_residual_px"]["median"]
            ),
            "pair_2_median_abs_residual_px": as_float(
                payload["per_pair"]["P2"]["pair_abs_residual_px"]["median"]
            ),
        }
        for name, payload in methods.items()
    }
    return {
        "scientific_label": "Restricted high-energy external reference",
        "methods": ["EasyLEED", "ViPErLEED"],
        "frame_count": as_int(summary["frame_count"]),
        "energy_range_ev": [
            as_float(value) for value in summary["energy_range_ev"]
        ],
        "branch_opportunities_per_method": next(iter(proposal_counts)),
        "valid_common_fits_per_method": next(iter(fit_counts)),
        "branch_gap_checks": f"{gap_passed}/{gap_total}",
        "per_method": per_method,
        "initialization_contracts": {
            "EasyLEED": "one Kalman tracker per frozen spot",
            "ViPErLEED": (
                f"registered {_viperleed_beam_count()}-beam "
                "whole-pattern trajectory"
            ),
        },
        "cross_method_contract_agreement_px": summary[
            "cross_method_contract_agreement_px"
        ],
        "part_of_main_five_method_matrix": False,
        "claim_limit": summary["scope"],
    }


def _csv_value(value: object) -> object:
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
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _csv_value(row.get(field, "")) for field in fields}
            )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _portable_path(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def _source_hashes(selected: dict[str, object]) -> dict[str, object]:
    visibility_path = (
        INDEPENDENT_ROOT
        / str(selected["archive_series_id"])
        / str(selected["archive_pair_id"])
        / "scan_measurements.csv"
    )
    sources = {
        "independent_aggregate": INDEPENDENT_AGGREGATE,
        "independent_case_results": INDEPENDENT_CASE_RESULTS,
        "independent_summary": INDEPENDENT_SUMMARY,
        "representative_visibility_regimes": visibility_path,
        "external_summary": EXTERNAL_SUMMARY,
        "external_gap_recovery": EXTERNAL_GAP_RECOVERY,
        "viperleed_registered_pattern_x": VIPERLEED_PATTERN_X,
        **{
            f"representative_pair_{method}": path
            for method, path in _representative_pair_paths(selected).items()
        },
    }
    return {
        name: {"path": _portable_path(path), "sha256": sha256_file(path)}
        for name, path in sources.items()
    }


def build(output_root: Path = FIGURE_ROOT) -> None:
    summary = json.loads(INDEPENDENT_SUMMARY.read_text(encoding="utf-8"))
    selected = select_representative_case(summary)
    output_source = output_root / "source_data"
    output_source.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_source / "aggregate_method_results.csv",
        load_aggregate_results(),
    )
    write_csv(
        output_source / "case_results.csv",
        load_case_results(),
    )
    write_csv(
        output_source / "representative_pair_measurements.csv",
        build_representative_pair_rows(selected),
    )
    write_csv(
        output_source / "representative_visibility_regimes.csv",
        build_visibility_rows(selected),
    )
    write_json(output_source / "representative_case.json", selected)
    write_json(
        output_source / "external_reference_summary.json",
        build_external_reference_summary(),
    )
    write_json(
        output_source / "source_provenance.json",
        {
            "schema_version": "1.0",
            "figure": "Figure 3",
            "sources": _source_hashes(selected),
        },
    )


def validate(*, check_upstream: bool = True) -> None:
    validate_source_contract()
    provenance = json.loads(
        (SOURCE_DATA_DIRECTORY / "source_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    if check_upstream:
        for name, source in provenance["sources"].items():
            path = REPOSITORY_ROOT / source["path"]
            if not path.is_file() or sha256_file(path) != source["sha256"]:
                raise ValueError(f"Figure 3 upstream hash mismatch: {name}")

        relative_paths = [f"source_data/{name}" for name in SOURCE_DATA_FILES]
        with tempfile.TemporaryDirectory(
            prefix="trackleed-figure03-source-"
        ) as temporary:
            rebuilt_root = Path(temporary) / "figure03_independent_scane_stress"
            build(rebuilt_root)
            for relative_path in relative_paths:
                committed = FIGURE_ROOT / relative_path
                rebuilt = rebuilt_root / relative_path
                if sha256_file(committed) != sha256_file(rebuilt):
                    raise ValueError(
                        f"Non-deterministic Figure 3 source: {relative_path}"
                    )
    print(
        "Figure 3 source validation passed: 2 scans, 4 cases, 5 methods, "
        "221 visible opportunities and 24 natural gaps per method; external "
        "reference kept separate at 60 opportunities per method."
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
