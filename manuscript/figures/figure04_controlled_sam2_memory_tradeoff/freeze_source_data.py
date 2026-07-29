#!/usr/bin/env python3
"""Freeze and validate the self-contained source package for Figure 4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Sequence


FIGURE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = FIGURE_ROOT.parents[2]
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
EVIDENCE_ROOT = (
    REPOSITORY_ROOT
    / ".scratch/trackleed-physics-method-paper/artifacts/"
    "cross-regime-evidence-matrix"
)
FAMILY_AGGREGATE = EVIDENCE_ROOT / "sam2_family_tradeoff.csv"
PROTOTYPE_ROOT = (
    REPOSITORY_ROOT
    / ".scratch/trackleed-physics-method-paper/prototypes/scane_tracer/output"
)
PAIR_SUMMARIES = (
    ("pair 1", PROTOTYPE_ROOT / "trackers/comparison_summary.json"),
    ("pair 2", PROTOTYPE_ROOT / "pair2/trackers/comparison_summary.json"),
)
METHOD_ORDER = ("sam2", "samurai", "dam4sam")
PAIR1_TRACKER_ROOT = PROTOTYPE_ROOT / "trackers"
PAIR1_MEASUREMENTS = {
    method: PAIR1_TRACKER_ROOT / f"{method}_measurements.csv"
    for method in METHOD_ORDER
}
DISPLAY_NAMES = {
    "sam2": "SAM2",
    "samurai": "SAMURAI",
    "dam4sam": "DAM4SAM",
}
MEMORY_STRATEGIES = {
    "sam2": "vanilla SAM2 propagation",
    "samurai": "motion-aware memory",
    "dam4sam": "distractor-aware memory",
}
REPRESENTATIVE_GAP_ENERGIES = (169.2, 166.5, 163.9, 160.9)
REPRESENTATIVE_GAP_ROLES = {
    169.2: "pre-gap visible",
    166.5: "I(V) gap",
    163.9: "I(V) gap",
    160.9: "first expected recovery",
}
REPRESENTATIVE_GAP_SELECTION_RULE = (
    "Starting at the common 187.9 eV seed and following the decreasing-energy "
    "branch, select the first natural I(V) gap for which the three controlled "
    "methods have different recovery outcomes. Use this outcome-selected case "
    "only as an explanatory mechanism panel; quantitative claims use both "
    "frozen pairs."
)
DATA_ROOT = REPOSITORY_ROOT / "data/26a24_ScanE"
SCAN_FRAMES = {
    169.2: DATA_ROOT / "Ag111-3G_0047_T0000_E1692_C0000.tiff",
    166.5: DATA_ROOT / "Ag111-3G_0046_T0000_E1665_C0000.tiff",
    163.9: DATA_ROOT / "Ag111-3G_0045_T0000_E1639_C0000.tiff",
    160.9: DATA_ROOT / "Ag111-3G_0044_T0000_E1609_C0000.tiff",
}
CROP_BOX_XYXY = (200, 456, 328, 584)
CROP_FILENAMES = {
    169.2: "b1_169p2ev.tiff",
    166.5: "b1_166p5ev.tiff",
    163.9: "b1_163p9ev.tiff",
    160.9: "b1_160p9ev.tiff",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def optional_float(value: object) -> float | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return float(text)


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
                {field: _csv_value(row.get(field)) for field in fields}
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


def load_family_aggregate(
    path: Path = FAMILY_AGGREGATE,
) -> list[dict[str, object]]:
    integer_fields = (
        "visible_opportunities",
        "branch_consistent_measurements",
        "evaluated_gaps",
        "recovered_gaps",
        "accepted_measurements_all_frames",
        "geometry_inconsistent_accepted_fits",
        "jointly_consistent_pair_frames",
    )
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
            "identical_checkpoint_weights": (
                row["identical_checkpoint_weights"].strip().lower() == "true"
            ),
            **{field: int(row[field]) for field in integer_fields},
            **{field: float(row[field]) for field in float_fields},
        }
        for row in read_csv(path)
    ]
    by_method = {str(row["method"]): row for row in rows}
    if set(by_method) != set(METHOD_ORDER):
        raise ValueError("Figure 4 SAM2-family method set changed")
    return [by_method[method] for method in METHOD_ORDER]


def load_pair_results() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair, path in PAIR_SUMMARIES:
        summary = json.loads(path.read_text(encoding="utf-8"))
        for method in METHOD_ORDER:
            result = summary["methods"][method]
            accepted = int(result["accepted_measurements_all_frames"])
            inconsistent = int(result["geometry_inconsistent_accepted_fits"])
            gap = result["post_gap_recovery"]
            residual = result["absolute_pair_distance_residual_px"]
            rows.append(
                {
                    "pair": pair,
                    "method": method,
                    "display_method": DISPLAY_NAMES[method],
                    "memory_strategy": MEMORY_STRATEGIES[method],
                    "visible_opportunities": int(
                        result["physics_baseline_visible_opportunities"]
                    ),
                    "branch_consistent_measurements": int(
                        result["branch_consistent_measurements"]
                    ),
                    "fit_coverage": float(
                        result["fit_coverage_on_baseline_visible"]
                    ),
                    "evaluated_gaps": int(gap["evaluated_gap_count"]),
                    "recovered_gaps": int(gap["recovered_gap_count"]),
                    "zero_delay_recoveries": int(gap["zero_delay_count"]),
                    "gap_recovery_rate": float(gap["rate"]),
                    "accepted_measurements_all_frames": accepted,
                    "geometry_inconsistent_accepted_fits": inconsistent,
                    "geometry_inconsistent_fraction": inconsistent / accepted,
                    "jointly_consistent_pair_frames": int(
                        result["jointly_consistent_pair_frames"]
                    ),
                    "conditional_median_abs_distance_residual_px": float(
                        residual["median"]
                    ),
                    "conditional_p95_abs_distance_residual_px": float(
                        residual["p95"]
                    ),
                }
            )
    return rows


def select_representative_gap() -> dict[str, object]:
    return {
        "archive_series_id": "26a24",
        "scientific_series_label": "Controlled scan",
        "scientific_pair_label": "pair 1",
        "branch": "B1",
        "direction": "decreasing energy",
        "reference_energy_ev": 187.9,
        "gap_energies_ev": [166.5, 163.9],
        "first_expected_recovery_energy_ev": 160.9,
        "display_energies_ev": list(REPRESENTATIVE_GAP_ENERGIES),
        "selection_rule": REPRESENTATIVE_GAP_SELECTION_RULE,
        "selection_uses_method_outcomes": True,
        "role": "explanatory mechanism panel, not independent quantitative evidence",
        "display_normalization": (
            "one shared linear intensity scale across all four raw crops"
        ),
    }


def build_representative_gap_rows(
    selected: dict[str, object],
) -> list[dict[str, object]]:
    target_energies = set(selected["display_energies_ev"])
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        selected_rows = [
            row
            for row in read_csv(PAIR1_MEASUREMENTS[method])
            if row["branch"] == selected["branch"]
            and float(row["energy_ev"]) in target_energies
        ]
        by_energy = {float(row["energy_ev"]): row for row in selected_rows}
        if set(by_energy) != target_energies:
            raise ValueError(
                f"Figure 4 representative-gap frames changed for {method}"
            )
        for energy in REPRESENTATIVE_GAP_ENERGIES:
            source = by_energy[energy]
            rows.append(
                {
                    "scientific_series_label": selected[
                        "scientific_series_label"
                    ],
                    "scientific_pair_label": selected["scientific_pair_label"],
                    "branch": selected["branch"],
                    "direction": selected["direction"],
                    "sequence_role": REPRESENTATIVE_GAP_ROLES[energy],
                    "energy_ev": energy,
                    "method": method,
                    "display_method": DISPLAY_NAMES[method],
                    "memory_strategy": MEMORY_STRATEGIES[method],
                    "physics_baseline_status": source[
                        "physics_baseline_status"
                    ],
                    "native_visible": as_bool(source["native_visible"]),
                    "native_x": optional_float(source["native_x"]),
                    "native_y": optional_float(source["native_y"]),
                    "fit_valid": as_bool(source["fit_valid"]),
                    "fit_reason": source["fit_reason"],
                    "gaussian_center_x": optional_float(source["center_x"]),
                    "gaussian_center_y": optional_float(source["center_y"]),
                    "measurement_valid": as_bool(source["measurement_valid"]),
                    "measurement_status": source["measurement_status"],
                    "branch_error_px": optional_float(source["branch_error_px"]),
                }
            )
    return rows


def _source_provenance() -> dict[str, object]:
    sources = {
        "family_aggregate": FAMILY_AGGREGATE,
        **{
            f"{pair.replace(' ', '_')}_summary": path
            for pair, path in PAIR_SUMMARIES
        },
        **{
            f"pair_1_{method}_measurements": path
            for method, path in PAIR1_MEASUREMENTS.items()
        },
        **{
            f"representative_frame_{str(energy).replace('.', 'p')}_ev": path
            for energy, path in SCAN_FRAMES.items()
        },
    }
    return {
        "schema_version": "1.0",
        "figure": "Figure 4",
        "sources": {
            name: {
                "path": _portable_path(path),
                "sha256": sha256_file(path),
            }
            for name, path in sources.items()
        },
    }


def build(output_root: Path = FIGURE_ROOT) -> None:
    import tifffile

    source = output_root / "source_data"
    crops = source / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    write_csv(source / "family_aggregate.csv", load_family_aggregate())
    write_csv(source / "family_by_pair.csv", load_pair_results())

    selected = {
        **select_representative_gap(),
        "crop_box_xyxy": list(CROP_BOX_XYXY),
        "crop_files": {
            str(energy): f"crops/{CROP_FILENAMES[energy]}"
            for energy in REPRESENTATIVE_GAP_ENERGIES
        },
    }
    write_csv(
        source / "representative_gap_measurements.csv",
        build_representative_gap_rows(selected),
    )
    write_json(source / "representative_gap.json", selected)

    x0, y0, x1, y1 = CROP_BOX_XYXY
    for energy, path in SCAN_FRAMES.items():
        frame = tifffile.imread(path)
        crop = frame[y0:y1, x0:x1]
        tifffile.imwrite(
            crops / CROP_FILENAMES[energy],
            crop,
            photometric="minisblack",
            metadata=None,
        )
    write_json(source / "source_provenance.json", _source_provenance())


def validate(*, check_upstream: bool = True) -> None:
    from figure_contract import validate_source_contract

    validate_source_contract(SOURCE_DATA_DIRECTORY)
    provenance = json.loads(
        (SOURCE_DATA_DIRECTORY / "source_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    if check_upstream:
        for name, source in provenance["sources"].items():
            path = REPOSITORY_ROOT / source["path"]
            if not path.is_file() or sha256_file(path) != source["sha256"]:
                raise ValueError(f"Figure 4 upstream hash mismatch: {name}")

        with tempfile.TemporaryDirectory(
            prefix="trackleed-figure04-source-"
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
                raise ValueError("Figure 4 source-data file set changed")
            for relative_path in committed_files:
                if sha256_file(FIGURE_ROOT / relative_path) != sha256_file(
                    rebuilt_root / relative_path
                ):
                    raise ValueError(
                        f"Non-deterministic Figure 4 source: {relative_path}"
                    )
    print(
        "Figure 4 source validation passed: one controlled energy scan, "
        "two disjoint pairs, three SAM2-family memory strategies, 182 visible "
        "opportunities and 13 natural gaps per method."
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
