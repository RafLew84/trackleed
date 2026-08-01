"""Reader-facing evidence derived from frozen controlled translations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
from typing import Any, TypedDict


METHOD_ORDER = (
    "locotrack_base_offline",
    "ostrack_vitb_256_mae_ce",
    "dam4sam_sam21pp_l",
    "lucas_kanade_skip_anchor",
    "easyleed_kalman",
    "oracle_roi_control",
)
TRIAL_ORDER = (0, 372, 378, 700)
TRIAL_LABELS = {
    0: "initial low-temperature state",
    372: "800 deg C plateau onset",
    378: "post-plateau contraction minimum",
    700: "final cooled state",
}
SPOT_OPPORTUNITIES_PER_TRIAL = 20
PAIR_OPPORTUNITIES_PER_TRIAL = 10
SOURCE_FILE_MAP = {
    "contract.json": "contract.json",
    "equivariance_results.csv": "equivariance_results.csv",
    "pair_invariance.csv": "pair_invariance.csv",
    "method_summary.csv": "method_summary.csv",
    "manifest.json": "upstream_manifest.json",
}


class TrialOutcome(TypedDict):
    source_frame_order: int
    trial: str
    temperature_c: int
    method: str
    display_method: str
    role: str
    available_proposal_count: int
    spot_opportunity_n: int
    proposal_coverage: str
    accepted_measurement_count: int
    measurement_coverage: str
    conditional_measurement_error_n: int
    measurement_error_median_px: float
    measurement_error_p95_px: float
    measurement_error_max_px: float
    valid_pair_count: int
    pair_opportunity_n: int
    pair_coverage: str
    conditional_pair_invariance_n: int
    maximum_abs_delta_d_over_d_pp: float


class ControlledTranslationSourceReport(TypedDict):
    spot_record_count: int
    pair_record_count: int
    trial_count: int
    method_count: int


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read controlled-translation JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Controlled-translation JSON is not an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def linear_percentile(values: list[float], percentile: float) -> float:
    """Return the linearly interpolated percentile of nonempty values."""

    if not values:
        raise ValueError("A conditional percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def _fraction(numerator: int, denominator: int) -> str:
    if denominator <= 0 or not 0 <= numerator <= denominator:
        raise ValueError("Coverage requires 0 <= numerator <= denominator")
    return f"{numerator}/{denominator}"


def _validate_source_matrix(
    spot_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
) -> None:
    if len(spot_rows) != 576 or len(pair_rows) != 288:
        raise ValueError("Controlled-translation source cardinality changed")
    if {row["method"] for row in spot_rows} != set(METHOD_ORDER):
        raise ValueError("Controlled-translation method inventory changed")
    if {
        int(row["source_frame_order"]) for row in spot_rows
    } != set(TRIAL_ORDER):
        raise ValueError("Controlled-translation trial inventory changed")


def _validate_upstream_output(origin_directory: Path) -> None:
    manifest = read_json(origin_directory / "manifest.json")
    output_hashes = manifest.get("output_files_sha256")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("status") != "completed"
        or not isinstance(output_hashes, dict)
    ):
        raise ValueError("Controlled-translation upstream manifest changed")
    for name in (
        "equivariance_results.csv",
        "pair_invariance.csv",
        "method_summary.csv",
    ):
        path = origin_directory / name
        if not path.is_file() or output_hashes.get(name) != sha256_file(path):
            raise ValueError(f"Controlled-translation upstream hash changed: {name}")
    contract_path = origin_directory / "contract.json"
    if manifest.get("contract_sha256") != sha256_file(contract_path):
        raise ValueError("Controlled-translation upstream contract changed")
    if read_json(contract_path).get("status") != "frozen_ready_for_runner":
        raise ValueError("Controlled-translation contract is not frozen")


def freeze_controlled_translation_source_data(
    origin_directory: Path,
    destination_directory: Path,
    *,
    workspace_root: Path,
) -> None:
    """Freeze the authenticated controlled-translation tables exactly once."""

    origin_directory = origin_directory.resolve()
    destination_directory = destination_directory.resolve()
    workspace_root = workspace_root.resolve()
    _validate_upstream_output(origin_directory)
    destination_directory.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict[str, str]] = {}
    origin_root = origin_directory.relative_to(workspace_root).as_posix()
    for origin_name, frozen_name in SOURCE_FILE_MAP.items():
        origin_path = origin_directory / origin_name
        frozen_path = destination_directory / frozen_name
        shutil.copyfile(origin_path, frozen_path)
        digest = sha256_file(origin_path)
        sources[frozen_name] = {
            "frozen_sha256": sha256_file(frozen_path),
            "origin_path": f"{origin_root}/{origin_name}",
            "origin_sha256": digest,
        }
    (destination_directory / "source_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "origin_required_at_build": False,
                "sources": sources,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def validate_controlled_translation_source_data(
    source_directory: Path,
    *,
    origin_directory: Path | None = None,
) -> ControlledTranslationSourceReport:
    """Validate frozen source hashes and optionally their authenticated origin."""

    source_directory = source_directory.resolve()
    provenance = read_json(source_directory / "source_provenance.json")
    sources = provenance.get("sources")
    expected_names = set(SOURCE_FILE_MAP.values())
    actual_names = {
        path.name
        for path in source_directory.iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    if (
        provenance.get("schema_version") != 1
        or provenance.get("origin_required_at_build") is not False
        or not isinstance(sources, dict)
        or set(sources) != expected_names
        or actual_names != expected_names | {"source_provenance.json"}
    ):
        raise ValueError("Controlled-translation source provenance changed")
    for frozen_name, record in sources.items():
        if (
            not isinstance(record, dict)
            or set(record) != {"frozen_sha256", "origin_path", "origin_sha256"}
        ):
            raise ValueError(
                f"Controlled-translation source record changed: {frozen_name}"
            )
        frozen_hash = sha256_file(source_directory / frozen_name)
        if record["frozen_sha256"] != frozen_hash:
            raise ValueError(
                f"Frozen controlled-translation source changed: {frozen_name}"
            )
        for field in ("frozen_sha256", "origin_sha256"):
            digest = str(record[field])
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    f"Malformed controlled-translation hash: {frozen_name}"
                )

    upstream_manifest = read_json(source_directory / "upstream_manifest.json")
    upstream_hashes = upstream_manifest.get("output_files_sha256")
    if not isinstance(upstream_hashes, dict):
        raise ValueError("Frozen controlled-translation manifest changed")
    for name in (
        "equivariance_results.csv",
        "pair_invariance.csv",
        "method_summary.csv",
    ):
        if upstream_hashes.get(name) != sha256_file(source_directory / name):
            raise ValueError(
                f"Controlled-translation source is not manifest-linked: {name}"
            )
    if upstream_manifest.get("contract_sha256") != sha256_file(
        source_directory / "contract.json"
    ):
        raise ValueError("Frozen controlled-translation contract is not linked")

    spot_rows = read_csv(source_directory / "equivariance_results.csv")
    pair_rows = read_csv(source_directory / "pair_invariance.csv")
    _validate_source_matrix(spot_rows, pair_rows)
    if origin_directory is not None:
        origin_directory = origin_directory.resolve()
        _validate_upstream_output(origin_directory)
        for origin_name, frozen_name in SOURCE_FILE_MAP.items():
            origin_hash = sha256_file(origin_directory / origin_name)
            record = sources[frozen_name]
            if (
                record["origin_sha256"] != origin_hash
                or record["frozen_sha256"] != origin_hash
            ):
                raise ValueError(
                    f"Controlled-translation source diverges from origin: "
                    f"{origin_name}"
                )
    return ControlledTranslationSourceReport(
        spot_record_count=len(spot_rows),
        pair_record_count=len(pair_rows),
        trial_count=len(TRIAL_ORDER),
        method_count=len(METHOD_ORDER),
    )


def build_trial_outcomes(source_directory: Path) -> list[TrialOutcome]:
    """Build one full-precision outcome row per source trial and method."""

    source_directory = source_directory.resolve()
    validate_controlled_translation_source_data(source_directory)
    spot_rows = read_csv(source_directory / "equivariance_results.csv")
    pair_rows = read_csv(source_directory / "pair_invariance.csv")
    _validate_source_matrix(spot_rows, pair_rows)
    display_names = {row["method"]: row["display_method"] for row in spot_rows}
    roles = {row["method"]: row["role"] for row in spot_rows}
    outcomes: list[TrialOutcome] = []
    for trial in TRIAL_ORDER:
        for method in METHOD_ORDER:
            trial_spots = [
                row
                for row in spot_rows
                if row["method"] == method
                and int(row["source_frame_order"]) == trial
                and as_bool(row["is_evaluation_step"])
            ]
            trial_pairs = [
                row
                for row in pair_rows
                if row["method"] == method
                and int(row["source_frame_order"]) == trial
                and as_bool(row["is_evaluation_step"])
            ]
            if (
                len(trial_spots) != SPOT_OPPORTUNITIES_PER_TRIAL
                or len(trial_pairs) != PAIR_OPPORTUNITIES_PER_TRIAL
            ):
                raise ValueError(
                    f"Controlled-translation opportunity set changed: {trial}/{method}"
                )
            measurement_errors = [
                float(row["measurement_displacement_error_px"])
                for row in trial_spots
                if as_bool(row["fit_valid"])
            ]
            pair_changes = [
                abs(float(row["delta_d_over_d_percent"]))
                for row in trial_pairs
                if as_bool(row["pair_valid"])
            ]
            if not measurement_errors or not pair_changes:
                raise ValueError(
                    f"Controlled-translation conditional outcome is empty: "
                    f"{trial}/{method}"
                )
            proposal_count = sum(
                as_bool(row["proposal_available"]) for row in trial_spots
            )
            measurement_count = len(measurement_errors)
            pair_count = len(pair_changes)
            outcomes.append(
                TrialOutcome(
                    source_frame_order=trial,
                    trial=TRIAL_LABELS[trial],
                    temperature_c=int(float(trial_spots[0]["temperature_c"])),
                    method=method,
                    display_method=display_names[method],
                    role=roles[method],
                    available_proposal_count=proposal_count,
                    spot_opportunity_n=SPOT_OPPORTUNITIES_PER_TRIAL,
                    proposal_coverage=_fraction(
                        proposal_count,
                        SPOT_OPPORTUNITIES_PER_TRIAL,
                    ),
                    accepted_measurement_count=measurement_count,
                    measurement_coverage=_fraction(
                        measurement_count,
                        SPOT_OPPORTUNITIES_PER_TRIAL,
                    ),
                    conditional_measurement_error_n=measurement_count,
                    measurement_error_median_px=statistics.median(
                        measurement_errors
                    ),
                    measurement_error_p95_px=linear_percentile(
                        measurement_errors,
                        95.0,
                    ),
                    measurement_error_max_px=max(measurement_errors),
                    valid_pair_count=pair_count,
                    pair_opportunity_n=PAIR_OPPORTUNITIES_PER_TRIAL,
                    pair_coverage=_fraction(
                        pair_count,
                        PAIR_OPPORTUNITIES_PER_TRIAL,
                    ),
                    conditional_pair_invariance_n=pair_count,
                    maximum_abs_delta_d_over_d_pp=max(pair_changes),
                )
            )
    if len(outcomes) != len(TRIAL_ORDER) * len(METHOD_ORDER):
        raise ValueError("Controlled-translation trial matrix changed")
    return outcomes


__all__ = [
    "METHOD_ORDER",
    "PAIR_OPPORTUNITIES_PER_TRIAL",
    "SPOT_OPPORTUNITIES_PER_TRIAL",
    "TRIAL_LABELS",
    "TRIAL_ORDER",
    "TrialOutcome",
    "as_bool",
    "build_trial_outcomes",
    "freeze_controlled_translation_source_data",
    "linear_percentile",
    "read_csv",
    "sha256_file",
    "validate_controlled_translation_source_data",
]
