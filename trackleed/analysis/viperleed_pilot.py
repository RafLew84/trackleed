"""Build and validate the frozen native ViPErLEED pilot contract."""

from __future__ import annotations

import csv
import hashlib
from io import StringIO
import json
import math
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Mapping, cast


DEFAULT_PACKAGE_PATH = Path("manuscript/cross_contract/viperleed_pilot")
EXPECTED_CONTRACT_ID = "native-viperleed-pilot-reference"
EXPECTED_STATUS = "frozen-native-viperleed-pilot-contract"
EXPECTED_CONTRACT_SHA256 = (
    "5afa547db8eca73a3e2a6479ac4138b90e15c357322fe1b805120fdbacbc1618"
)
EXPECTED_BRANCH_MAPPING = {
    "P1_B1": (-1, 0, "(-1|0) [0]"),
    "P1_B2": (1, 0, "(1|0) [0]"),
    "P2_B1": (1, -1, "(1|-1) [0]"),
    "P2_B2": (-1, 1, "(-1|1) [0]"),
}
DERIVED_PATHS = (
    "pilot_records.csv",
    "pilot_summary.json",
    "manifest.json",
    "validation_report.json",
)
NATIVE_TABLE_PATHS = {
    "x": "frozen/native/x.csv",
    "y": "frozen/native/y.csv",
    "significance": "frozen/native/significance.csv",
    "dx_raw": "frozen/native/dx_raw.csv",
    "dy_raw": "frozen/native/dy_raw.csv",
    "dx_smooth": "frozen/native/dx_smooth.csv",
    "dy_smooth": "frozen/native/dy_smooth.csv",
    "raw_intensity": "frozen/native/raw_intensity.csv",
    "background": "frozen/native/background.csv",
    "background_sigma": "frozen/native/background_sigma.csv",
    "radial_size": "frozen/native/radial_size.csv",
    "tangential_size": "frozen/native/tangential_size.csv",
    "iv_curves": "frozen/native/iv_curves.csv",
}
PILOT_RECORD_FIELDS = (
    "segment_index",
    "frame_id",
    "energy_ev",
    "branch_id",
    "beam_h",
    "beam_k",
    "beam_column",
    "significance",
    "dx_raw",
    "dy_raw",
    "dx_smooth",
    "dy_smooth",
    "final_x",
    "final_y",
    "model_x",
    "model_y",
    "raw_x",
    "raw_y",
    "viper_native_direct_detectability",
    "gaussian_contract_measurable",
    "gaussian_rejection_reason",
    "gaussian_x",
    "gaussian_y",
    "cross_contract_dx",
    "cross_contract_dy",
    "cross_contract_distance",
    "radial_component",
    "tangential_component",
    "high_signal_calibration_subset",
)


class ViperleedPilotValidationError(ValueError):
    """Raised when the native ViPErLEED pilot package fails closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ViperleedPilotValidationError(
            f"Cannot read {label}: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise ViperleedPilotValidationError(f"{label} must be a JSON object")
    return value


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ViperleedPilotValidationError(f"Cannot read {label}: {path}") from exc


def _read_pattern_csv(path: Path) -> list[dict[str, str]]:
    """Read a ViPErLEED pattern file while retaining only its CSV body."""

    try:
        body = "".join(
            line
            for line in path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
            if not line.lstrip().startswith("#")
        )
        return list(csv.DictReader(StringIO(body)))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ViperleedPilotValidationError(
            f"Cannot read spot pattern: {path}"
        ) from exc


def _optional_float(value: object, label: str) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        number = float(str(value))
    except ValueError as exc:
        raise ViperleedPilotValidationError(f"Invalid numeric value for {label}") from exc
    return number if math.isfinite(number) else None


def _required_float(value: object, label: str) -> float:
    number = _optional_float(value, label)
    if number is None:
        raise ViperleedPilotValidationError(f"Missing finite value for {label}")
    return number


def _required_int(value: object, label: str) -> int:
    number = _required_float(value, label)
    integer = int(number)
    if float(integer) != number:
        raise ViperleedPilotValidationError(f"Non-integral value for {label}")
    return integer


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    output = StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=PILOT_RECORD_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _format_value(row[field]) for field in PILOT_RECORD_FIELDS})
    return output.getvalue().encode("utf-8")


def _within(root: Path, value: object, label: str, *, require_file: bool) -> Path:
    if not isinstance(value, str) or not value:
        raise ViperleedPilotValidationError(f"{label} path changed")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ViperleedPilotValidationError(f"{label} escaped workspace: {value}")
    if require_file and not path.is_file():
        raise ViperleedPilotValidationError(f"{label} is unavailable: {value}")
    return path


def _contract_path(package_root: Path) -> Path:
    path = package_root / "contract.json"
    if not path.is_file():
        raise ViperleedPilotValidationError(
            f"ViPErLEED pilot contract is unavailable: {package_root}"
        )
    return path


def _validate_contract(
    workspace_root: Path,
    package_root: Path,
) -> dict[str, Any]:
    contract_path = _contract_path(package_root)
    if _sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise ViperleedPilotValidationError("ViPErLEED pilot contract hash changed")
    contract = _load_json(contract_path, "ViPErLEED pilot contract")
    if (
        contract.get("schema_version") != "1.0"
        or contract.get("contract_id") != EXPECTED_CONTRACT_ID
        or contract.get("status") != EXPECTED_STATUS
    ):
        raise ViperleedPilotValidationError("ViPErLEED pilot contract identity changed")

    mapping = contract.get("beam_mapping")
    if not isinstance(mapping, dict) or set(mapping) != set(EXPECTED_BRANCH_MAPPING):
        raise ViperleedPilotValidationError("Frozen branch mapping changed")
    for branch_id, expected in EXPECTED_BRANCH_MAPPING.items():
        record = mapping[branch_id]
        actual = (
            record.get("h") if isinstance(record, dict) else None,
            record.get("k") if isinstance(record, dict) else None,
            record.get("beam_column") if isinstance(record, dict) else None,
        )
        if actual != expected:
            raise ViperleedPilotValidationError(
                f"Frozen branch mapping changed: {branch_id}"
            )

    direct = contract.get("viper_native_direct_detectability")
    tracking = contract.get("tracking_options")
    calibration = contract.get("calibration")
    evidence_role = contract.get("evidence_role")
    if (
        not isinstance(direct, dict)
        or not isinstance(tracking, dict)
        or not isinstance(calibration, dict)
        or not isinstance(evidence_role, dict)
    ):
        raise ViperleedPilotValidationError("Pilot contract section changed")
    if (
        direct.get("significance_threshold") != 2.5
        or direct.get("final_coordinates_alone_are_sufficient") is not False
        or direct.get("smoothed_or_extrapolated_coordinates_alone_are_sufficient")
        is not False
        or tracking.get("noise_rejection_significance") != 2.5
        or tracking.get("position_smoothing_range_ev") != 30.0
        or tracking.get("search_beam_unseen_for_ev") != 10.0
        or tracking.get("minimum_energy_range_per_beam_ev") != 0.0
    ):
        raise ViperleedPilotValidationError("Native observation policy changed")
    if (
        calibration.get("cross_contract_tolerance_px") != 1.0
        or calibration.get("expected_record_count") != 22
        or calibration.get("safety_factor") != 2.0
        or calibration.get("round_up_increment_px") != 1.0
        or calibration.get("eligibility", {}).get("minimum_native_significance")
        != 20.0
    ):
        raise ViperleedPilotValidationError("Pilot tolerance calibration changed")
    if evidence_role != {
        "may_be_reused_as_independent_confirmation": False,
        "principal_scan_outcomes_inspected_for_calibration": False,
        "role": "calibration-only",
    }:
        raise ViperleedPilotValidationError("Pilot evidence role changed")

    scientific = contract.get("scientific_contract")
    if not isinstance(scientific, dict):
        raise ViperleedPilotValidationError("Scientific contract link changed")
    scientific_path = _within(
        workspace_root,
        scientific.get("path"),
        "scientific contract",
        require_file=True,
    )
    if _sha256(scientific_path) != scientific.get("sha256"):
        raise ViperleedPilotValidationError("Scientific contract hash drift")

    inputs = contract.get("frozen_inputs")
    if not isinstance(inputs, list) or len(inputs) != 22:
        raise ViperleedPilotValidationError("Frozen pilot input inventory changed")
    frozen_paths = [
        item.get("frozen_path") if isinstance(item, dict) else None for item in inputs
    ]
    source_hashes = [
        item.get("sha256") if isinstance(item, dict) else None for item in inputs
    ]
    if (
        any(not isinstance(path, str) for path in frozen_paths)
        or len(set(frozen_paths)) != len(frozen_paths)
        or not set(NATIVE_TABLE_PATHS.values()).issubset(set(frozen_paths))
        or any(
            not isinstance(digest, str) or len(digest) != 64
            for digest in source_hashes
        )
    ):
        raise ViperleedPilotValidationError("Frozen pilot input inventory changed")
    return contract


def _validate_executed_configuration(
    contract: Mapping[str, Any],
    package_root: Path,
) -> None:
    macro = (package_root / "frozen/run_viperleed.ijm").read_text(encoding="utf-8")
    log = (package_root / "frozen/viperleed_log.txt").read_text(encoding="utf-8")
    checkpoint = (package_root / "frozen/macro_checkpoint.txt").read_text(
        encoding="utf-8"
    )
    required_macro_fragments = (
        'Ext.LEEDsetValue("minRange_eV", 0);',
        'Ext.LEEDsetValue("neighborBackground", 0);',
        'Ext.LEEDsetValue("minSignificanceSelecting", 4);',
        "Ext.LEEDselectSliceForE(187.9);",
        "makeOval(21.000308, 2.011221, 940, 940);",
        "makeOval(366.000308, 347.011221, 250, 250);",
    )
    if any(fragment not in macro for fragment in required_macro_fragments):
        raise ViperleedPilotValidationError("Executed ViPErLEED macro changed")
    if 'Ext.LEEDsetInput("dark"' in macro or 'Ext.LEEDsetInput("flat"' in macro:
        raise ViperleedPilotValidationError("Pilot correction inputs changed")
    serialized = str(contract["serialized_tracker_parameters"])
    required_log_fragments = (
        "Integration Radius: 8.33 -> 8.28 circ.",
        "Indices defined at 187.9 eV",
        "cubic, ?rms=3.2pxl, 18 spots",
        f"leed.tracker_n={serialized}",
    )
    if any(fragment not in log for fragment in required_log_fragments):
        raise ViperleedPilotValidationError("Executed ViPErLEED parameters changed")
    if "macro completed" not in checkpoint:
        raise ViperleedPilotValidationError("ViPErLEED pilot run is incomplete")


def _native_tables(
    contract: Mapping[str, Any],
    package_root: Path,
    energies: list[float],
) -> dict[str, list[dict[str, str]]]:
    tables = {
        name: _read_csv(package_root / relative_path, f"native {name} table")
        for name, relative_path in NATIVE_TABLE_PATHS.items()
    }
    columns = {
        record["beam_column"]
        for record in contract["beam_mapping"].values()
        if isinstance(record, dict)
    }
    expected_headers: tuple[str, ...] | None = None
    for name, rows in tables.items():
        if len(rows) != len(energies):
            raise ViperleedPilotValidationError(
                f"Native {name} row count changed: {len(rows)}"
            )
        headers = tuple(rows[0]) if rows else ()
        if expected_headers is None:
            expected_headers = headers
        elif headers != expected_headers:
            raise ViperleedPilotValidationError("Native output schemas diverged")
        if "energy" not in headers or not columns.issubset(headers):
            raise ViperleedPilotValidationError(
                f"Native {name} beam columns changed"
            )
        table_energies = [
            _required_float(row.get("energy"), f"{name} energy") for row in rows
        ]
        if any(abs(actual - expected) > 1e-6 for actual, expected in zip(table_energies, energies)):
            raise ViperleedPilotValidationError(
                f"Native {name} energy sequence changed"
            )
    return tables


def _build_records_and_summary(
    contract: Mapping[str, Any],
    package_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest_rows = _read_csv(
        package_root / "frozen/input_manifest.csv",
        "pilot input manifest",
    )
    if len(manifest_rows) != 15:
        raise ViperleedPilotValidationError("Pilot frame inventory changed")
    segments = [_required_int(row.get("segment_index"), "segment_index") for row in manifest_rows]
    frames = [_required_int(row.get("source_frame"), "source_frame") for row in manifest_rows]
    energies = [_required_float(row.get("energy_ev"), "energy_ev") for row in manifest_rows]
    if segments != list(range(15)) or frames != list(range(54, 69)):
        raise ViperleedPilotValidationError("Pilot frame identity changed")
    if energies[-2:] != [220.2, 220.2]:
        raise ViperleedPilotValidationError("Duplicate terminal energy was not retained")

    pattern_rows = _read_pattern_csv(package_root / "frozen/pattern.csv")
    if len(pattern_rows) != contract["initialization"]["pattern_beam_count"]:
        raise ViperleedPilotValidationError("Pattern beam count changed")
    pattern_indices = {
        (
            _required_int(row.get("h"), "pattern h"),
            _required_int(row.get("k"), "pattern k"),
        )
        for row in pattern_rows
    }
    if any((h, k) not in pattern_indices for h, k, _ in EXPECTED_BRANCH_MAPPING.values()):
        raise ViperleedPilotValidationError("Mapped beam is absent from the pattern")

    common_rows = [
        row
        for row in _read_csv(
            package_root / "frozen/common_measurements.csv",
            "common measurement table",
        )
        if row.get("method") == "ViPErLEED"
    ]
    if len(common_rows) != contract["pilot_scope"]["expected_branch_frame_records"]:
        raise ViperleedPilotValidationError("Pilot branch-frame denominator changed")
    common_by_key: dict[tuple[int, str], dict[str, str]] = {}
    for row in common_rows:
        key = (
            _required_int(row.get("segment_index"), "common segment_index"),
            str(row.get("branch")),
        )
        if key in common_by_key or key[1] not in EXPECTED_BRANCH_MAPPING:
            raise ViperleedPilotValidationError("Common measurement key changed")
        common_by_key[key] = row
    expected_keys = {
        (segment, branch)
        for segment in range(15)
        for branch in EXPECTED_BRANCH_MAPPING
    }
    if set(common_by_key) != expected_keys:
        raise ViperleedPilotValidationError("Pilot opportunity set changed")

    tables = _native_tables(contract, package_root, energies)
    threshold = float(
        contract["viper_native_direct_detectability"]["significance_threshold"]
    )
    high_signal = float(
        contract["calibration"]["eligibility"]["minimum_native_significance"]
    )
    origin_x, origin_y = (
        float(value)
        for value in contract["cross_contract_displacement"]["radial_origin_xy_px"]
    )
    records: list[dict[str, object]] = []
    for segment in range(15):
        for branch_id in EXPECTED_BRANCH_MAPPING:
            common = common_by_key[(segment, branch_id)]
            mapping = contract["beam_mapping"][branch_id]
            column = str(mapping["beam_column"])
            values = {
                name: _optional_float(rows[segment].get(column), f"{name}:{segment}:{branch_id}")
                for name, rows in tables.items()
            }
            significance = values["significance"]
            dx_raw = values["dx_raw"]
            dy_raw = values["dy_raw"]
            raw_deviations_finite = dx_raw is not None and dy_raw is not None
            direct = (
                significance is not None
                and significance >= threshold
                and raw_deviations_finite
            )
            final_x = values["x"]
            final_y = values["y"]
            dx_smooth = values["dx_smooth"]
            dy_smooth = values["dy_smooth"]
            model_x: float | None
            model_y: float | None
            raw_x: float | None
            raw_y: float | None
            if (
                final_x is not None
                and final_y is not None
                and dx_smooth is not None
                and dy_smooth is not None
                and dx_raw is not None
                and dy_raw is not None
            ):
                model_x = final_x - dx_smooth
                model_y = final_y - dy_smooth
                raw_x = model_x + dx_raw
                raw_y = model_y + dy_raw
            else:
                model_x = model_y = raw_x = raw_y = None
            if direct and raw_x is None:
                raise ViperleedPilotValidationError(
                    f"Direct native record lacks reconstructable position: {segment}/{branch_id}"
                )

            fit_value = common.get("fit_valid")
            if fit_value not in {"True", "False"}:
                raise ViperleedPilotValidationError("Gaussian fit validity changed")
            gaussian_valid = fit_value == "True"
            gaussian_x = (
                _required_float(common.get("center_x"), "Gaussian x")
                if gaussian_valid
                else None
            )
            gaussian_y = (
                _required_float(common.get("center_y"), "Gaussian y")
                if gaussian_valid
                else None
            )

            delta_x = delta_y = distance = radial = tangential = None
            if direct and gaussian_valid:
                assert raw_x is not None and raw_y is not None
                assert model_x is not None and model_y is not None
                assert gaussian_x is not None and gaussian_y is not None
                delta_x = raw_x - gaussian_x
                delta_y = raw_y - gaussian_y
                distance = math.hypot(delta_x, delta_y)
                radius = math.hypot(model_x - origin_x, model_y - origin_y)
                if radius <= 0:
                    raise ViperleedPilotValidationError("Radial axis is undefined")
                unit_x = (model_x - origin_x) / radius
                unit_y = (model_y - origin_y) / radius
                radial = delta_x * unit_x + delta_y * unit_y
                # Image coordinates are y-down; invert y when projecting onto
                # the counter-clockwise tangent of the physical y-up plane.
                tangential = delta_x * unit_y - delta_y * unit_x
            calibration_eligible = bool(
                direct
                and gaussian_valid
                and significance is not None
                and significance >= high_signal
            )
            records.append(
                {
                    "segment_index": segment,
                    "frame_id": frames[segment],
                    "energy_ev": energies[segment],
                    "branch_id": branch_id,
                    "beam_h": mapping["h"],
                    "beam_k": mapping["k"],
                    "beam_column": column,
                    "significance": significance,
                    "dx_raw": dx_raw,
                    "dy_raw": dy_raw,
                    "dx_smooth": values["dx_smooth"],
                    "dy_smooth": values["dy_smooth"],
                    "final_x": values["x"],
                    "final_y": values["y"],
                    "model_x": model_x,
                    "model_y": model_y,
                    "raw_x": raw_x,
                    "raw_y": raw_y,
                    "viper_native_direct_detectability": direct,
                    "gaussian_contract_measurable": gaussian_valid,
                    "gaussian_rejection_reason": (
                        "" if gaussian_valid else str(common.get("fit_reason", ""))
                    ),
                    "gaussian_x": gaussian_x,
                    "gaussian_y": gaussian_y,
                    "cross_contract_dx": delta_x,
                    "cross_contract_dy": delta_y,
                    "cross_contract_distance": distance,
                    "radial_component": radial,
                    "tangential_component": tangential,
                    "high_signal_calibration_subset": calibration_eligible,
                }
            )

    rejected = [record for record in records if not record["gaussian_contract_measurable"]]
    rejected_above = [
        record
        for record in rejected
        if record["significance"] is not None
        and float(cast(float, record["significance"])) > threshold
    ]
    rejected_above_finite = [
        record
        for record in rejected_above
        if record["dx_raw"] is not None and record["dy_raw"] is not None
    ]
    calibration_records = [
        record for record in records if record["high_signal_calibration_subset"]
    ]
    calibration_distances = [
        float(cast(float, record["cross_contract_distance"]))
        for record in calibration_records
        if record["cross_contract_distance"] is not None
    ]
    if not calibration_distances:
        raise ViperleedPilotValidationError("High-signal calibration set is empty")
    maximum = max(calibration_distances)
    increment = float(contract["calibration"]["round_up_increment_px"])
    safety_factor = float(contract["calibration"]["safety_factor"])
    tolerance = math.ceil(safety_factor * maximum / increment) * increment

    summary: dict[str, object] = {
        "branch_mapping": {
            branch: [values[0], values[1]]
            for branch, values in EXPECTED_BRANCH_MAPPING.items()
        },
        "calibration": {
            "cross_contract_tolerance_px": tolerance,
            "high_signal_maximum_distance_px": maximum,
            "high_signal_minimum_significance": high_signal,
            "record_count": len(calibration_records),
            "round_up_increment_px": increment,
            "safety_factor": safety_factor,
        },
        "cross_contract_displacement_record_count": sum(
            1 for record in records if record["cross_contract_distance"] is not None
        ),
        "viper_native_direct_detectability": {
            "count": sum(
                1
                for record in records
                if record["viper_native_direct_detectability"] is True
            ),
            "definition": "significance >= 2.5 and finite dx_raw and dy_raw",
            "threshold": threshold,
        },
        "gaussian_rejections": {
            "above_native_threshold": len(rejected_above),
            "above_threshold_with_finite_raw": len(rejected_above_finite),
            "count": len(rejected),
        },
        "pilot_evidence_role": contract["evidence_role"]["role"],
        "pilot_record_count": len(records),
        "schema_version": "1.0",
        "status": "frozen-native-viperleed-pilot-summary",
    }
    return records, summary


def _build_manifest(
    package_root: Path,
    contract: Mapping[str, Any],
) -> dict[str, object]:
    file_roles = {
        ".gitattributes": "checkout byte-preservation rules",
        "contract.json": "normative pilot contract",
    }
    for item in contract["frozen_inputs"]:
        file_roles[str(item["frozen_path"])] = str(item["role"])
    file_roles.update(
        {
            "pilot_records.csv": "normalized native four-branch pilot records",
            "pilot_summary.json": "derived pilot calibration summary",
        }
    )
    files = []
    for relative_path, role in sorted(file_roles.items()):
        path = package_root / relative_path
        if not path.is_file():
            raise ViperleedPilotValidationError(
                f"Manifest input is unavailable: {relative_path}"
            )
        files.append(
            {
                "path": relative_path,
                "role": role,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    image_rows = _read_csv(
        package_root / "frozen/input_manifest.csv",
        "source-image inventory",
    )
    return {
        "file_count": len(files),
        "files": files,
        "package_id": EXPECTED_CONTRACT_ID,
        "schema_version": "1.0",
        "source_image_inventory": {
            "frame_count": len(image_rows),
            "identity": "source frame plus energy; duplicate energies remain distinct",
            "manifest_path": "frozen/input_manifest.csv",
        },
        "status": "frozen-viperleed-pilot-manifest",
    }


def _computed_report(
    package_root: Path,
    contract: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, object]:
    manifest_path = package_root / "manifest.json"
    return {
        "branch_mapping": summary["branch_mapping"],
        "contract_sha256": _sha256(package_root / "contract.json"),
        "cross_contract_displacement_record_count": summary[
            "cross_contract_displacement_record_count"
        ],
        "cross_contract_tolerance_px": summary["calibration"][
            "cross_contract_tolerance_px"
        ],
        "gaussian_rejection_count": summary["gaussian_rejections"]["count"],
        "high_signal_calibration_count": summary["calibration"]["record_count"],
        "manifest_sha256": _sha256(manifest_path),
        "viper_native_direct_detectability_count": summary[
            "viper_native_direct_detectability"
        ]["count"],
        "pilot_evidence_role": contract["evidence_role"]["role"],
        "pilot_record_count": summary["pilot_record_count"],
        "rejections_above_native_threshold": summary["gaussian_rejections"][
            "above_native_threshold"
        ],
        "rejections_above_threshold_with_finite_raw": summary[
            "gaussian_rejections"
        ]["above_threshold_with_finite_raw"],
        "schema_version": "1.0",
        "status": "validated-viperleed-pilot-contract",
    }


def _validate_package_files(
    workspace_root: Path,
    package_root: Path,
    *,
    require_report: bool,
) -> dict[str, object]:
    contract = _validate_contract(workspace_root, package_root)
    manifest_path = package_root / "manifest.json"
    if not manifest_path.is_file():
        raise ViperleedPilotValidationError("ViPErLEED pilot manifest is unavailable")
    manifest = _load_json(manifest_path, "ViPErLEED pilot manifest")
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("package_id") != EXPECTED_CONTRACT_ID
        or manifest.get("status") != "frozen-viperleed-pilot-manifest"
        or not isinstance(files, list)
        or manifest.get("file_count") != len(files)
    ):
        raise ViperleedPilotValidationError("ViPErLEED pilot manifest changed")
    contract_input_hashes = {
        str(item["frozen_path"]): str(item["sha256"])
        for item in contract["frozen_inputs"]
    }
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ViperleedPilotValidationError("Pilot manifest record changed")
        relative_path = item.get("path")
        path = _within(
            package_root,
            relative_path,
            "manifested pilot file",
            require_file=True,
        )
        assert isinstance(relative_path, str)
        if relative_path in seen:
            raise ViperleedPilotValidationError("Duplicate pilot manifest path")
        seen.add(relative_path)
        actual_hash = _sha256(path)
        contract_hash = contract_input_hashes.get(relative_path)
        if contract_hash is not None and (
            actual_hash != contract_hash or item.get("sha256") != contract_hash
        ):
            raise ViperleedPilotValidationError(
                f"Frozen pilot contract-bound evidence hash drift: {relative_path}"
            )
        if actual_hash != item.get("sha256") or path.stat().st_size != item.get(
            "size_bytes"
        ):
            raise ViperleedPilotValidationError(
                f"Frozen pilot evidence hash drift: {relative_path}"
            )
    expected_paths = {
        ".gitattributes",
        "contract.json",
        "pilot_records.csv",
        "pilot_summary.json",
    }
    expected_paths.update(str(item["frozen_path"]) for item in contract["frozen_inputs"])
    if seen != expected_paths:
        raise ViperleedPilotValidationError("Pilot manifest coverage changed")

    _validate_executed_configuration(contract, package_root)
    records, summary = _build_records_and_summary(contract, package_root)
    if (package_root / "pilot_records.csv").read_bytes() != _csv_bytes(records):
        raise ViperleedPilotValidationError(
            "Pilot records no longer reproduce from frozen native inputs"
        )
    if (package_root / "pilot_summary.json").read_bytes() != _json_bytes(summary):
        raise ViperleedPilotValidationError(
            "Pilot summary no longer reproduces from frozen native inputs"
        )

    calibration = cast(dict[str, Any], summary["calibration"])
    if (
        calibration["record_count"] != contract["calibration"]["expected_record_count"]
        or abs(
            calibration["high_signal_maximum_distance_px"]
            - contract["calibration"]["observed_max_distance_px"]
        )
        > 1e-12
        or calibration["cross_contract_tolerance_px"]
        != contract["calibration"]["cross_contract_tolerance_px"]
    ):
        raise ViperleedPilotValidationError("Frozen tolerance does not reproduce")
    if cast(dict[str, Any], summary["gaussian_rejections"]) != {
        "above_native_threshold": 21,
        "above_threshold_with_finite_raw": 21,
        "count": 22,
    }:
        raise ViperleedPilotValidationError("Pilot 21-of-22 finding changed")
    if (
        cast(dict[str, Any], summary["viper_native_direct_detectability"])["count"]
        != 59
    ):
        raise ViperleedPilotValidationError(
            "ViPEr-native direct-detectability count changed"
        )

    report = _computed_report(package_root, contract, summary)
    report_path = package_root / "validation_report.json"
    if require_report:
        if not report_path.is_file():
            raise ViperleedPilotValidationError(
                "ViPErLEED pilot validation result is unavailable"
            )
        if report_path.read_bytes() != _json_bytes(report):
            raise ViperleedPilotValidationError(
                "Committed ViPErLEED pilot validation result changed"
            )
    return report


def freeze_viperleed_pilot(
    workspace_root: Path,
    package_root: Path | None = None,
) -> dict[str, object]:
    """Freeze source copies and derived pilot artifacts exactly once."""

    workspace_root = workspace_root.resolve()
    package_root = (
        package_root.resolve()
        if package_root is not None
        else workspace_root / DEFAULT_PACKAGE_PATH
    )
    contract = _validate_contract(workspace_root, package_root)
    if (package_root / "frozen").exists() or any(
        (package_root / relative_path).exists() for relative_path in DERIVED_PATHS
    ):
        raise FileExistsError(
            f"Refusing to replace frozen ViPErLEED pilot package: {package_root}"
        )

    package_root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="viperleed-pilot-",
        dir=package_root.parent,
    ) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        shutil.copyfile(package_root / "contract.json", stage / "contract.json")
        shutil.copyfile(package_root / ".gitattributes", stage / ".gitattributes")
        readme_path = package_root / "README.md"
        if readme_path.is_file():
            shutil.copyfile(readme_path, stage / "README.md")
        for item in contract["frozen_inputs"]:
            origin = _within(
                workspace_root,
                item.get("origin_path"),
                "pilot origin",
                require_file=True,
            )
            if _sha256(origin) != item.get("sha256"):
                raise ViperleedPilotValidationError(
                    f"Pilot origin hash drift: {item.get('origin_path')}"
                )
            destination = _within(
                stage,
                item.get("frozen_path"),
                "frozen pilot destination",
                require_file=False,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, destination)

        records, summary = _build_records_and_summary(contract, stage)
        (stage / "pilot_records.csv").write_bytes(_csv_bytes(records))
        (stage / "pilot_summary.json").write_bytes(_json_bytes(summary))
        manifest = _build_manifest(stage, contract)
        (stage / "manifest.json").write_bytes(_json_bytes(manifest))
        report = _validate_package_files(
            workspace_root,
            stage,
            require_report=False,
        )
        (stage / "validation_report.json").write_bytes(_json_bytes(report))
        _validate_package_files(workspace_root, stage, require_report=True)

        shutil.copytree(stage / "frozen", package_root / "frozen")
        for relative_path in DERIVED_PATHS:
            os.replace(stage / relative_path, package_root / relative_path)
    return validate_viperleed_pilot(workspace_root, package_root)


def validate_viperleed_pilot(
    workspace_root: Path,
    package_root: Path | None = None,
) -> dict[str, object]:
    """Validate the frozen pilot package without mutating it."""

    workspace_root = workspace_root.resolve()
    package_root = (
        package_root.resolve()
        if package_root is not None
        else workspace_root / DEFAULT_PACKAGE_PATH
    )
    _contract_path(package_root)
    return _validate_package_files(
        workspace_root,
        package_root,
        require_report=True,
    )


__all__ = [
    "ViperleedPilotValidationError",
    "freeze_viperleed_pilot",
    "validate_viperleed_pilot",
]
