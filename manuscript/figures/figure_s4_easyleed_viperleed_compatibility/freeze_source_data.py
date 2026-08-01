#!/usr/bin/env python3
"""Freeze the separate EasyLEED-ViPErLEED source contract for Figure S4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


FIGURE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = FIGURE_ROOT.parents[2]
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
UPSTREAM_ROOT = (
    REPOSITORY_ROOT / ".scratch/trackleed-physics-method-paper/artifacts/"
    "domain-baseline-high-energy"
)
UPSTREAM_FILES = {
    "common_measurements.csv": UPSTREAM_ROOT / "common_measurements.csv",
    "common_pair_measurements.csv": UPSTREAM_ROOT / "common_pair_measurements.csv",
    "gap_recovery.csv": UPSTREAM_ROOT / "gap_recovery.csv",
    "input_manifest.csv": UPSTREAM_ROOT / "input_manifest.csv",
}
SOURCE_FILES = tuple(UPSTREAM_FILES) + (
    "compatibility_contract.json",
    "source_provenance.json",
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


def _compatibility_contract() -> dict[str, object]:
    summary = json.loads((UPSTREAM_ROOT / "summary.json").read_text(encoding="utf-8"))
    methods = summary["methods"]
    pair_denominators = {
        pair: int(methods["EasyLEED"]["per_pair"][pair]["valid_pair_frames"])
        for pair in ("P1", "P2")
    }
    if any(
        int(methods["ViPErLEED"]["per_pair"][pair]["valid_pair_frames"])
        != pair_denominators[pair]
        for pair in pair_denominators
    ):
        raise ValueError("External-reference pair denominators diverged")
    return {
        "scientific_label": "Restricted high-energy external reference",
        "methods": ["EasyLEED", "ViPErLEED"],
        "frame_count": int(summary["frame_count"]),
        "energy_range_ev": summary["energy_range_ev"],
        "branch_opportunities_per_method": 60,
        "valid_common_fits_per_method": 38,
        "conditional_pair_frames_per_method": pair_denominators,
        "branch_gap_checks_per_method": "4/4",
        "initialization_contracts": {
            "EasyLEED": "one Kalman tracker per frozen spot",
            "ViPErLEED": "registered 18-beam whole-pattern trajectory",
        },
        "per_method": {
            method: {
                "per_branch_valid_common_fits": {
                    branch: int(payload["valid_common_fits"])
                    for branch, payload in methods[method]["per_branch"].items()
                },
                "pair_median_abs_residual_px": {
                    pair: payload["pair_abs_residual_px"]["median"]
                    for pair, payload in methods[method]["per_pair"].items()
                },
            }
            for method in ("EasyLEED", "ViPErLEED")
        },
        "cross_method_contract_agreement_px": summary[
            "cross_method_contract_agreement_px"
        ],
        "part_of_main_five_method_matrix": False,
        "claims_position_ground_truth": False,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build(output_root: Path = FIGURE_ROOT) -> None:
    output = output_root / "source_data"
    output.mkdir(parents=True, exist_ok=True)
    for name, source in UPSTREAM_FILES.items():
        shutil.copyfile(source, output / name)
    _write_json(output / "compatibility_contract.json", _compatibility_contract())
    sources = {
        name: {
            "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in {
            **UPSTREAM_FILES,
            "summary.json": UPSTREAM_ROOT / "summary.json",
        }.items()
    }
    _write_json(
        output / "source_provenance.json",
        {"figure": "Figure S4", "schema_version": "1.0", "sources": sources},
    )


def validate_source_package(
    source_directory: Path = SOURCE_DATA_DIRECTORY,
    *,
    check_upstream: bool = True,
) -> None:
    measurements = read_csv(source_directory / "common_measurements.csv")
    pairs = read_csv(source_directory / "common_pair_measurements.csv")
    gaps = read_csv(source_directory / "gap_recovery.csv")
    frames = read_csv(source_directory / "input_manifest.csv")
    contract = json.loads(
        (source_directory / "compatibility_contract.json").read_text(encoding="utf-8")
    )
    if len(measurements) != 120 or len(pairs) != 60 or len(gaps) != 8:
        raise ValueError("Figure S4 record cardinality changed")
    if len(frames) != 15 or len({row["source_frame"] for row in frames}) != 15:
        raise ValueError("Figure S4 frame identity changed")
    for method in ("EasyLEED", "ViPErLEED"):
        method_rows = [row for row in measurements if row["method"] == method]
        if len(method_rows) != 60:
            raise ValueError(f"Figure S4 opportunity denominator changed: {method}")
        if sum(row["fit_valid"] == "True" for row in method_rows) != 38:
            raise ValueError(f"Figure S4 valid-fit denominator changed: {method}")
        method_gaps = [row for row in gaps if row["method"] == method]
        if len(method_gaps) != 4 or not all(
            row["recovery_fit_valid"] == "True" for row in method_gaps
        ):
            raise ValueError(f"Figure S4 gap recovery contract changed: {method}")
    if contract["conditional_pair_frames_per_method"] != {"P1": 5, "P2": 3}:
        raise ValueError("Figure S4 conditional pair denominator changed")
    if contract["part_of_main_five_method_matrix"]:
        raise ValueError("Figure S4 external reference entered main matrix")

    provenance = json.loads(
        (source_directory / "source_provenance.json").read_text(encoding="utf-8")
    )
    if check_upstream:
        for name, source in provenance["sources"].items():
            path = (REPOSITORY_ROOT / source["path"]).resolve()
            if REPOSITORY_ROOT.resolve() not in path.parents:
                raise ValueError(f"Figure S4 provenance escapes repository: {name}")
            if not path.is_file() or sha256_file(path) != source["sha256"]:
                raise ValueError(f"Figure S4 upstream hash mismatch: {name}")


def validate() -> None:
    validate_source_package()
    with tempfile.TemporaryDirectory(prefix="trackleed-figure-s4-source-") as tmp:
        rebuilt = Path(tmp) / "figure_s4_easyleed_viperleed_compatibility"
        build(rebuilt)
        for name in SOURCE_FILES:
            if sha256_file(SOURCE_DATA_DIRECTORY / name) != sha256_file(
                rebuilt / "source_data" / name
            ):
                raise ValueError(f"Non-deterministic Figure S4 source: {name}")
    print("Figure S4 source validation passed: 60 opportunities per method.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.validate_only:
        build()
    validate()


if __name__ == "__main__":
    main()
