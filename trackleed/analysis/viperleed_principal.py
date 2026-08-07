"""Freeze and validate native ViPErLEED results for the principal scans."""

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

from trackleed.analysis.viperleed_pilot import validate_viperleed_pilot


DEFAULT_PACKAGE_PATH = Path("manuscript/cross_contract/viperleed_principal")
EXECUTION_OUTPUT_PATH = Path(
    ".scratch/trackleed-cross-contract-manuscript/principal-viperleed-execution"
)
EXPECTED_CONTRACT_SHA256 = (
    "200715029bd4bcfb944fff59b632372e497c6b220dc2039e9a2aab242837c4a0"
)
EXPECTED_RUN_AUTHENTICATION_SHA256 = (
    "14db2e014f046ef7f45bfa7ec92714ddf7c818209cd72f1fc03488bab33b5e05"
)
EXPECTED_MANIFEST_SHA256 = (
    "df97fa62bfbd02ee51d0364e469d31a339aaf5fcf2b1c1099a33433cba75dced"
)
PREREGISTRATION_COMMITS = (
    "9997c720cae6c33bb19f5c93cae182124a6ab4fa",
    "c5c5cab7536940249e15ca9442c35a00f1578b46",
)
SCAN_IDS = ("scan_a", "scan_b")
BRANCH_MAPPING = {
    "P1_B1": (-1, 0, "(-1|0) [0]"),
    "P1_B2": (1, 0, "(1|0) [0]"),
    "P2_B1": (1, -1, "(1|-1) [0]"),
    "P2_B2": (-1, 1, "(-1|1) [0]"),
}
TABLE_SUFFIXES = {
    "final_x": "x",
    "final_y": "y",
    "significance": "signif",
    "dx_raw": "dx_raw",
    "dy_raw": "dy_raw",
    "dx_smooth": "dx_smooth",
    "dy_smooth": "dy_smooth",
    "raw_intensity": "rawInt",
    "background": "backgr",
    "background_sigma": "bsigma",
    "radial_size": "rSize",
    "tangential_size": "tSize",
    "iv_curve": "IVcurves",
}
DERIVED_PATHS = (
    "run_authentication.json",
    "four_branch_records.csv",
    "run_summary.json",
    "manifest.json",
    "validation_report.json",
)
RECORD_FIELDS = (
    "scan_id",
    "segment_index",
    "source_frame",
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
    "raw_intensity",
    "background",
    "background_sigma",
    "radial_size",
    "tangential_size",
    "iv_curve",
    "raw_deviations_finite",
    "smoothed_deviations_finite",
    "final_position_available",
    "viper_native_direct_detectability",
    "position_semantics",
)


class ViperleedPrincipalValidationError(ValueError):
    """Raised when the principal native ViPErLEED package fails closed."""


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
        raise ViperleedPrincipalValidationError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ViperleedPrincipalValidationError(f"{label} must be a JSON object")
    return value


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ViperleedPrincipalValidationError(f"Cannot read {label}: {path}") from exc


def _optional_float(value: object, label: str) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        number = float(str(value))
    except ValueError as exc:
        raise ViperleedPrincipalValidationError(
            f"Invalid numeric value for {label}"
        ) from exc
    return number if math.isfinite(number) else None


def _required_float(value: object, label: str) -> float:
    number = _optional_float(value, label)
    if number is None:
        raise ViperleedPrincipalValidationError(f"Missing finite value for {label}")
    return number


def _required_int(value: object, label: str) -> int:
    number = _required_float(value, label)
    integer = int(number)
    if float(integer) != number:
        raise ViperleedPrincipalValidationError(f"Non-integral value for {label}")
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
        fieldnames=RECORD_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _format_value(row[field]) for field in RECORD_FIELDS})
    return output.getvalue().encode("utf-8")


def _within(root: Path, value: object, label: str, *, require_file: bool) -> Path:
    if not isinstance(value, str) or not value:
        raise ViperleedPrincipalValidationError(f"{label} path changed")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ViperleedPrincipalValidationError(f"{label} escaped root: {value}")
    if require_file and not path.is_file():
        raise ViperleedPrincipalValidationError(f"{label} is unavailable: {value}")
    return path


def _contract_path(package_root: Path) -> Path:
    path = package_root / "contract.json"
    if not path.is_file():
        raise ViperleedPrincipalValidationError(
            f"principal ViPErLEED contract is unavailable: {package_root}"
        )
    return path


def _bound_workspace_file(
    workspace_root: Path,
    record: object,
    label: str,
) -> Path:
    if not isinstance(record, dict):
        raise ViperleedPrincipalValidationError(f"{label} binding changed")
    path = _within(workspace_root, record.get("path"), label, require_file=True)
    if _sha256(path) != record.get("sha256"):
        raise ViperleedPrincipalValidationError(f"{label} hash drift")
    return path


def _validate_contract(
    workspace_root: Path,
    package_root: Path,
) -> dict[str, Any]:
    path = _contract_path(package_root)
    if _sha256(path) != EXPECTED_CONTRACT_SHA256:
        raise ViperleedPrincipalValidationError(
            "principal ViPErLEED contract hash changed"
        )
    contract = _load_json(path, "principal ViPErLEED contract")
    if (
        contract.get("schema_version") != "1.0"
        or contract.get("contract_id") != "native-viperleed-principal-scans"
        or contract.get("status")
        != "predeclared-principal-viperleed-inference-contract"
    ):
        raise ViperleedPrincipalValidationError(
            "principal ViPErLEED contract identity changed"
        )
    if set(cast(dict[str, Any], contract.get("scans", {}))) != set(SCAN_IDS):
        raise ViperleedPrincipalValidationError("principal scan set changed")
    if contract.get("branch_mapping") != {
        branch: [mapping[0], mapping[1]] for branch, mapping in BRANCH_MAPPING.items()
    }:
        raise ViperleedPrincipalValidationError("principal branch mapping changed")
    validate_viperleed_pilot(workspace_root)
    _bound_workspace_file(
        workspace_root, contract.get("pilot_contract"), "pilot contract"
    )
    _bound_workspace_file(
        workspace_root,
        contract.get("scientific_contract"),
        "scientific contract",
    )
    predecessor = contract.get("predecessor_scientific_core")
    if not isinstance(predecessor, dict):
        raise ViperleedPrincipalValidationError(
            "predecessor scientific-core binding changed"
        )
    current_predecessor_path = _within(
        workspace_root,
        predecessor.get("fingerprint_path"),
        "predecessor scientific-core fingerprint",
        require_file=True,
    )
    expected_predecessor_sha256 = predecessor.get("fingerprint_sha256")
    if (
        not isinstance(expected_predecessor_sha256, str)
        or len(expected_predecessor_sha256) != 64
        or predecessor.get("required_transition")
        != "controlled scientific-core rebaseline after cross-contract analysis "
        "and claim-boundary approval"
    ):
        raise ViperleedPrincipalValidationError(
            "predecessor scientific-core binding changed"
        )
    history_root = current_predecessor_path.parent / "history" / "fingerprints"
    predecessor_candidates = [current_predecessor_path]
    if history_root.is_dir():
        predecessor_candidates.extend(sorted(history_root.glob("*.json")))
    matching_predecessors = [
        path
        for path in predecessor_candidates
        if path.is_file() and _sha256(path) == expected_predecessor_sha256
    ]
    if len(matching_predecessors) != 1:
        raise ViperleedPrincipalValidationError(
            "predecessor scientific-core binding changed"
        )
    configuration = contract.get("configuration")
    if not isinstance(configuration, dict):
        raise ViperleedPrincipalValidationError("principal configuration changed")
    _bound_workspace_file(workspace_root, configuration.get("pattern"), "spot pattern")
    _bound_workspace_file(
        workspace_root,
        configuration.get("screenfit"),
        "transferred screenfit",
    )
    screenfit = cast(dict[str, Any], configuration["screenfit"])
    _bound_workspace_file(
        workspace_root,
        screenfit.get("source_log"),
        "pilot screenfit source log",
    )
    for scan_id in SCAN_IDS:
        scan = cast(dict[str, Any], contract["scans"][scan_id])
        _bound_workspace_file(
            workspace_root,
            scan.get("input_manifest"),
            f"{scan_id} input manifest",
        )
        _bound_workspace_file(
            workspace_root,
            scan.get("macro"),
            f"{scan_id} execution macro",
        )
    return contract


def _input_rows(
    workspace_root: Path,
    contract: Mapping[str, Any],
) -> dict[str, list[dict[str, str]]]:
    inventories: dict[str, list[dict[str, str]]] = {}
    seen_observations: set[tuple[str, int, float]] = set()
    for scan_id in SCAN_IDS:
        scan = cast(dict[str, Any], contract["scans"][scan_id])
        manifest_record = cast(dict[str, Any], scan["input_manifest"])
        manifest_path = _within(
            workspace_root,
            manifest_record["path"],
            f"{scan_id} input manifest",
            require_file=True,
        )
        rows = _read_csv(manifest_path, f"{scan_id} input manifest")
        frame_count = int(scan["frame_count"])
        if len(rows) != frame_count:
            raise ViperleedPrincipalValidationError(
                f"{scan_id} source-frame count changed"
            )
        energies: list[float] = []
        source_paths: set[str] = set()
        dataset = (workspace_root / str(scan["dataset"])).resolve()
        for segment, row in enumerate(rows):
            frame = _required_int(row.get("source_frame"), "source_frame")
            energy = _required_float(row.get("energy_ev"), "energy_ev")
            if (
                row.get("scan_id") != scan_id
                or _required_int(row.get("segment_index"), "segment_index") != segment
                or frame != segment
            ):
                raise ViperleedPrincipalValidationError(
                    f"{scan_id} source-frame identity changed"
                )
            key = (scan_id, frame, energy)
            if key in seen_observations:
                raise ViperleedPrincipalValidationError(
                    "duplicate principal observation key"
                )
            seen_observations.add(key)
            source_path = row.get("source_path")
            source = _within(
                workspace_root,
                source_path,
                f"{scan_id} source image",
                require_file=True,
            )
            if not source.is_relative_to(dataset):
                raise ViperleedPrincipalValidationError(
                    f"{scan_id} source image escaped its dataset"
                )
            if (
                not isinstance(source_path, str)
                or source_path in source_paths
                or _sha256(source) != row.get("sha256")
                or source.stat().st_size
                != _required_int(row.get("size_bytes"), "size_bytes")
            ):
                raise ViperleedPrincipalValidationError(
                    f"{scan_id} source image inventory drift"
                )
            source_paths.add(source_path)
            energies.append(energy)
        expected_duplicate = [float(value) for value in scan["terminal_duplicate"]]
        if energies[-2:] != expected_duplicate:
            raise ViperleedPrincipalValidationError(
                f"{scan_id} terminal duplicate energy changed"
            )
        inventories[scan_id] = rows
    return inventories


def _native_filename(scan_id: str, suffix: str) -> str:
    return f"viperleed_{scan_id}_{suffix}.csv"


def _native_file_names(scan_id: str) -> tuple[str, ...]:
    prefix = f"viperleed_{scan_id}"
    return (
        *(_native_filename(scan_id, suffix) for suffix in TABLE_SUFFIXES.values()),
        f"{prefix}_log.txt",
        f"{prefix}_spotIndicesImage.tif.zip",
        "macro_checkpoint.txt",
    )


def _expected_beam_headers(workspace_root: Path) -> tuple[str, ...]:
    pilot_x = (
        workspace_root / "manuscript/cross_contract/viperleed_pilot/frozen/native/x.csv"
    )
    rows = _read_csv(pilot_x, "pilot native x table")
    if not rows:
        raise ViperleedPrincipalValidationError("pilot native beam schema is empty")
    headers = tuple(rows[0])
    if len(headers) != 19 or headers[0] != "energy":
        raise ViperleedPrincipalValidationError("pilot native beam schema changed")
    return headers


def _native_tables(
    workspace_root: Path,
    contract: Mapping[str, Any],
    package_root: Path,
    input_rows: Mapping[str, list[dict[str, str]]],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    expected_headers = _expected_beam_headers(workspace_root)
    result: dict[str, dict[str, list[dict[str, str]]]] = {}
    for scan_id in SCAN_IDS:
        scan = cast(dict[str, Any], contract["scans"][scan_id])
        expected_energies = [
            _required_float(row.get("energy_ev"), "input energy")
            for row in input_rows[scan_id]
        ]
        tables: dict[str, list[dict[str, str]]] = {}
        for field, suffix in TABLE_SUFFIXES.items():
            path = package_root / "frozen" / scan_id / _native_filename(scan_id, suffix)
            rows = _read_csv(path, f"{scan_id} native {field}")
            if len(rows) != int(scan["frame_count"]):
                raise ViperleedPrincipalValidationError(
                    f"{scan_id} native {field} row count changed"
                )
            headers = tuple(rows[0]) if rows else ()
            if headers != expected_headers:
                raise ViperleedPrincipalValidationError(
                    f"{scan_id} native {field} columns changed"
                )
            energies = [
                _required_float(row.get("energy"), f"{scan_id} native energy")
                for row in rows
            ]
            if any(
                abs(actual - expected) > 1e-6
                for actual, expected in zip(energies, expected_energies)
            ):
                raise ViperleedPrincipalValidationError(
                    f"{scan_id} native {field} energy sequence changed"
                )
            tables[field] = rows
        result[scan_id] = tables
    return result


def _validate_executed_configuration(
    contract: Mapping[str, Any],
    package_root: Path,
) -> dict[str, dict[str, str]]:
    serialized = str(
        contract["configuration"]["serialized_tracker_parameters_expected"]
    )
    outcomes: dict[str, dict[str, str]] = {}
    for scan_id in SCAN_IDS:
        prefix = f"viperleed_{scan_id}"
        root = package_root / "frozen" / scan_id
        checkpoint = (root / "macro_checkpoint.txt").read_text(encoding="utf-8")
        log = (root / f"{prefix}_log.txt").read_text(encoding="utf-8")
        required_checkpoint = (
            "configuration predeclared; no manual outcome annotation",
            "indices set once; no per-frame correction permitted",
            "native whole-pattern data saved before proposal reduction",
            "macro completed",
        )
        if any(fragment not in checkpoint for fragment in required_checkpoint):
            raise ViperleedPrincipalValidationError(
                f"{scan_id} execution checkpoint changed"
            )
        required_log = (
            "Indices defined at 188.5 eV",
            "18 beams; highest raw intensity=",
            f"leed.tracker_n={serialized}",
            "no I0",
        )
        if any(fragment not in log for fragment in required_log):
            raise ViperleedPrincipalValidationError(
                f"{scan_id} executed ViPErLEED configuration changed"
            )
        integration = next(
            (
                line
                for line in log.splitlines()
                if line.startswith("Integration Radius:")
            ),
            "",
        )
        screenfit = next(
            (line for line in log.splitlines() if line.startswith("leed.screenfit=")),
            "",
        )
        indices = next(
            (line for line in log.splitlines() if "spots" in line and "rms=" in line),
            "",
        )
        if not integration or not screenfit or not indices:
            raise ViperleedPrincipalValidationError(
                f"{scan_id} initialization outcome is incomplete"
            )
        outcomes[scan_id] = {
            "index_fit": indices,
            "integration_radius": integration.removeprefix("Integration Radius: "),
            "screenfit": screenfit.removeprefix("leed.screenfit="),
        }
    return outcomes


def _build_records_and_summary(
    workspace_root: Path,
    contract: Mapping[str, Any],
    package_root: Path,
    input_rows: Mapping[str, list[dict[str, str]]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    tables_by_scan = _native_tables(
        workspace_root,
        contract,
        package_root,
        input_rows,
    )
    threshold = float(contract["configuration"]["noise_rejection_significance"])
    records: list[dict[str, object]] = []
    scan_summaries: dict[str, dict[str, object]] = {}
    whole_pattern_beam_frame_count = 0
    for scan_id in SCAN_IDS:
        tables = tables_by_scan[scan_id]
        inputs = input_rows[scan_id]
        direct_count = 0
        final_count = 0
        for segment, input_row in enumerate(inputs):
            frame = _required_int(input_row.get("source_frame"), "source_frame")
            energy = _required_float(input_row.get("energy_ev"), "energy_ev")
            for branch_id, (beam_h, beam_k, beam_column) in BRANCH_MAPPING.items():
                values = {
                    field: _optional_float(
                        rows[segment].get(beam_column),
                        f"{scan_id}:{segment}:{branch_id}:{field}",
                    )
                    for field, rows in tables.items()
                }
                dx_raw = values["dx_raw"]
                dy_raw = values["dy_raw"]
                dx_smooth = values["dx_smooth"]
                dy_smooth = values["dy_smooth"]
                final_x = values["final_x"]
                final_y = values["final_y"]
                raw_pair = dx_raw is not None and dy_raw is not None
                smooth_pair = dx_smooth is not None and dy_smooth is not None
                final_pair = final_x is not None and final_y is not None
                if (dx_raw is None) != (dy_raw is None):
                    raise ViperleedPrincipalValidationError(
                        f"{scan_id} raw deviation pair is incomplete"
                    )
                if (dx_smooth is None) != (dy_smooth is None):
                    raise ViperleedPrincipalValidationError(
                        f"{scan_id} smoothed deviation pair is incomplete"
                    )
                if (final_x is None) != (final_y is None):
                    raise ViperleedPrincipalValidationError(
                        f"{scan_id} final position pair is incomplete"
                    )
                significance = values["significance"]
                direct = bool(
                    significance is not None and significance >= threshold and raw_pair
                )
                model_x = model_y = raw_x = raw_y = None
                if final_pair and smooth_pair:
                    assert final_x is not None and final_y is not None
                    assert dx_smooth is not None and dy_smooth is not None
                    model_x = final_x - dx_smooth
                    model_y = final_y - dy_smooth
                    if raw_pair:
                        assert dx_raw is not None and dy_raw is not None
                        raw_x = model_x + dx_raw
                        raw_y = model_y + dy_raw
                if direct and (raw_x is None or raw_y is None):
                    raise ViperleedPrincipalValidationError(
                        f"{scan_id} direct record lacks a raw native position"
                    )
                semantics = (
                    "direct"
                    if direct
                    else "inferred-or-subthreshold"
                    if final_pair
                    else "missing"
                )
                direct_count += int(direct)
                final_count += int(final_pair)
                records.append(
                    {
                        "scan_id": scan_id,
                        "segment_index": segment,
                        "source_frame": frame,
                        "energy_ev": energy,
                        "branch_id": branch_id,
                        "beam_h": beam_h,
                        "beam_k": beam_k,
                        "beam_column": beam_column,
                        **values,
                        "model_x": model_x,
                        "model_y": model_y,
                        "raw_x": raw_x,
                        "raw_y": raw_y,
                        "raw_deviations_finite": raw_pair,
                        "smoothed_deviations_finite": smooth_pair,
                        "final_position_available": final_pair,
                        "viper_native_direct_detectability": direct,
                        "position_semantics": semantics,
                    }
                )
        frame_count = len(inputs)
        opportunity_count = frame_count * len(BRANCH_MAPPING)
        scan = cast(dict[str, Any], contract["scans"][scan_id])
        scan_summaries[scan_id] = {
            "direct_detectability_count": direct_count,
            "final_position_count": final_count,
            "frame_count": frame_count,
            "opportunity_count": opportunity_count,
            "retained_non_direct_count": opportunity_count - direct_count,
            "terminal_duplicate_energy_ev": float(scan["terminal_duplicate"][-1]),
        }
        whole_pattern_beam_frame_count += frame_count * 18
    direct_total = sum(
        int(record["viper_native_direct_detectability"] is True) for record in records
    )
    final_total = sum(
        int(record["final_position_available"] is True) for record in records
    )
    summary: dict[str, object] = {
        "four_branch_opportunity_count": len(records),
        "frame_count": sum(len(rows) for rows in input_rows.values()),
        "inference_scope": "new inference on existing acquisitions",
        "native_table_count_per_scan": len(TABLE_SUFFIXES),
        "scan_count": len(SCAN_IDS),
        "scans": scan_summaries,
        "scientific_core_rebaseline": "required",
        "status": "frozen-principal-viperleed-summary",
        "viper_native_direct_detectability_count": direct_total,
        "retained_non_direct_count": len(records) - direct_total,
        "final_position_count": final_total,
        "whole_pattern_beam_frame_count": whole_pattern_beam_frame_count,
    }
    return records, summary


def _build_run_authentication(
    contract: Mapping[str, Any],
    package_root: Path,
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for scan_id in SCAN_IDS:
        origin_root = EXECUTION_OUTPUT_PATH / scan_id / "native"
        for name in _native_file_names(scan_id):
            frozen_path = Path("frozen") / scan_id / name
            path = package_root / frozen_path
            files.append(
                {
                    "origin_path": (origin_root / name).as_posix(),
                    "path": frozen_path.as_posix(),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "executed_initialization": _validate_executed_configuration(
            contract,
            package_root,
        ),
        "files": files,
        "manual_intervention": False,
        "native_whole_pattern_preserved_before_proposal_reduction": True,
        "preregistration_commits": list(PREREGISTRATION_COMMITS),
        "schema_version": "1.0",
        "status": "authenticated-principal-viperleed-runs",
    }


def _manifest_roles(
    contract: Mapping[str, Any],
    authentication: Mapping[str, Any],
) -> dict[str, str]:
    roles = {
        ".gitattributes": "native byte-preservation rules",
        "contract.json": "predeclared principal inference contract",
        "execution/frozen_pilot_screenfit.csv": "transferred final pilot screenfit",
        "run_authentication.json": "immutable native-run authentication",
        "four_branch_records.csv": "lossless four-branch native records",
        "run_summary.json": "native denominator and finite-semantics summary",
    }
    for scan_id in SCAN_IDS:
        scan = cast(dict[str, Any], contract["scans"][scan_id])
        for field in ("input_manifest", "macro"):
            path = Path(str(scan[field]["path"]))
            relative = path.relative_to(DEFAULT_PACKAGE_PATH).as_posix()
            roles[relative] = f"predeclared {scan_id} {field.replace('_', ' ')}"
    for item in authentication["files"]:
        roles[str(item["path"])] = "native whole-pattern output"
    return roles


def _build_manifest(
    contract: Mapping[str, Any],
    authentication: Mapping[str, Any],
    package_root: Path,
) -> dict[str, object]:
    files = []
    for relative_path, role in sorted(
        _manifest_roles(contract, authentication).items()
    ):
        path = package_root / relative_path
        if not path.is_file():
            raise ViperleedPrincipalValidationError(
                f"principal manifest input is unavailable: {relative_path}"
            )
        files.append(
            {
                "path": relative_path,
                "role": role,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "file_count": len(files),
        "files": files,
        "package_id": "native-viperleed-principal-scans",
        "schema_version": "1.0",
        "source_image_inventory": {
            "distinct_observation_count": 81,
            "identity": "scan plus source frame plus energy; duplicate energies remain distinct",
        },
        "status": "frozen-principal-viperleed-manifest",
    }


def _computed_report(
    package_root: Path,
    summary: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "final_position_count": summary["final_position_count"],
        "four_branch_opportunity_count": summary["four_branch_opportunity_count"],
        "frame_count": summary["frame_count"],
        "inference_scope": summary["inference_scope"],
        "manifest_sha256": _sha256(package_root / "manifest.json"),
        "native_table_count_per_scan": summary["native_table_count_per_scan"],
        "retained_non_direct_count": summary["retained_non_direct_count"],
        "run_authentication_sha256": _sha256(package_root / "run_authentication.json"),
        "scan_count": summary["scan_count"],
        "scans": summary["scans"],
        "scientific_core_rebaseline": summary["scientific_core_rebaseline"],
        "schema_version": "1.0",
        "status": "validated-principal-viperleed-scans",
        "viper_native_direct_detectability_count": summary[
            "viper_native_direct_detectability_count"
        ],
        "whole_pattern_beam_frame_count": summary["whole_pattern_beam_frame_count"],
    }


def _validate_frozen_package(
    workspace_root: Path,
    package_root: Path,
    *,
    enforce_frozen_hashes: bool,
) -> dict[str, object]:
    contract = _validate_contract(workspace_root, package_root)
    inputs = _input_rows(workspace_root, contract)
    authentication_path = package_root / "run_authentication.json"
    manifest_path = package_root / "manifest.json"
    if not authentication_path.is_file() or not manifest_path.is_file():
        raise ViperleedPrincipalValidationError(
            f"principal ViPErLEED manifest is unavailable: {package_root}"
        )
    if enforce_frozen_hashes:
        if _sha256(authentication_path) != EXPECTED_RUN_AUTHENTICATION_SHA256:
            raise ViperleedPrincipalValidationError(
                "principal run authentication hash changed"
            )
        if _sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
            raise ViperleedPrincipalValidationError(
                "principal package manifest hash changed"
            )
    authentication = _load_json(
        authentication_path,
        "principal run authentication",
    )
    if (
        authentication.get("schema_version") != "1.0"
        or authentication.get("status") != "authenticated-principal-viperleed-runs"
        or authentication.get("contract_sha256") != EXPECTED_CONTRACT_SHA256
        or authentication.get("preregistration_commits")
        != list(PREREGISTRATION_COMMITS)
        or authentication.get("manual_intervention") is not False
        or authentication.get(
            "native_whole_pattern_preserved_before_proposal_reduction"
        )
        is not True
    ):
        raise ViperleedPrincipalValidationError("principal run authentication changed")
    authenticated_files = authentication.get("files")
    if not isinstance(authenticated_files, list) or len(authenticated_files) != 32:
        raise ViperleedPrincipalValidationError(
            "principal native output inventory changed"
        )
    authenticated_hashes: dict[str, str] = {}
    for item in authenticated_files:
        if not isinstance(item, dict):
            raise ViperleedPrincipalValidationError(
                "principal native authentication record changed"
            )
        path = _within(
            package_root,
            item.get("path"),
            "authenticated native output",
            require_file=True,
        )
        relative = str(item["path"])
        if relative in authenticated_hashes:
            raise ViperleedPrincipalValidationError(
                "duplicate authenticated native output"
            )
        digest = _sha256(path)
        if digest != item.get("sha256") or path.stat().st_size != item.get(
            "size_bytes"
        ):
            raise ViperleedPrincipalValidationError(
                f"authenticated principal native output drift: {relative}"
            )
        authenticated_hashes[relative] = digest
    expected_native_paths = {
        (Path("frozen") / scan_id / name).as_posix()
        for scan_id in SCAN_IDS
        for name in _native_file_names(scan_id)
    }
    if set(authenticated_hashes) != expected_native_paths:
        raise ViperleedPrincipalValidationError(
            "principal authenticated native output coverage changed"
        )
    executed = _validate_executed_configuration(contract, package_root)
    if authentication.get("executed_initialization") != executed:
        raise ViperleedPrincipalValidationError(
            "principal initialization authentication changed"
        )

    manifest = _load_json(manifest_path, "principal package manifest")
    manifest_files = manifest.get("files")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("status") != "frozen-principal-viperleed-manifest"
        or manifest.get("package_id") != "native-viperleed-principal-scans"
        or not isinstance(manifest_files, list)
        or manifest.get("file_count") != len(manifest_files)
    ):
        raise ViperleedPrincipalValidationError("principal package manifest changed")
    manifested: set[str] = set()
    for item in manifest_files:
        if not isinstance(item, dict):
            raise ViperleedPrincipalValidationError("principal manifest record changed")
        path = _within(
            package_root,
            item.get("path"),
            "manifested principal file",
            require_file=True,
        )
        relative = str(item["path"])
        if relative in manifested:
            raise ViperleedPrincipalValidationError("duplicate principal manifest path")
        if _sha256(path) != item.get("sha256") or path.stat().st_size != item.get(
            "size_bytes"
        ):
            raise ViperleedPrincipalValidationError(
                f"principal manifested file drift: {relative}"
            )
        if (
            relative in authenticated_hashes
            and item.get("sha256") != authenticated_hashes[relative]
        ):
            raise ViperleedPrincipalValidationError(
                f"principal authentication/manifest mismatch: {relative}"
            )
        manifested.add(relative)
    expected_manifested = set(_manifest_roles(contract, authentication))
    if manifested != expected_manifested:
        raise ViperleedPrincipalValidationError("principal manifest coverage changed")

    records, summary = _build_records_and_summary(
        workspace_root,
        contract,
        package_root,
        inputs,
    )
    if (package_root / "four_branch_records.csv").read_bytes() != _csv_bytes(records):
        raise ViperleedPrincipalValidationError(
            "principal four-branch records no longer reproduce"
        )
    if (package_root / "run_summary.json").read_bytes() != _json_bytes(summary):
        raise ViperleedPrincipalValidationError(
            "principal run summary no longer reproduces"
        )
    expected_summary = {
        "frame_count": 81,
        "four_branch_opportunity_count": 324,
        "viper_native_direct_detectability_count": 305,
        "retained_non_direct_count": 19,
        "final_position_count": 324,
        "whole_pattern_beam_frame_count": 1458,
    }
    if any(summary[key] != value for key, value in expected_summary.items()):
        raise ViperleedPrincipalValidationError(
            "principal native denominator or finite semantics changed"
        )
    report = _computed_report(package_root, summary)
    report_path = package_root / "validation_report.json"
    if not report_path.is_file() or report_path.read_bytes() != _json_bytes(report):
        raise ViperleedPrincipalValidationError(
            "committed principal validation report changed"
        )
    return report


def freeze_principal_viperleed_scans(
    workspace_root: Path,
    package_root: Path | None = None,
) -> dict[str, object]:
    """Freeze completed preregistered native runs exactly once."""

    workspace_root = workspace_root.resolve()
    package_root = (
        package_root.resolve()
        if package_root is not None
        else workspace_root / DEFAULT_PACKAGE_PATH
    )
    contract = _validate_contract(workspace_root, package_root)
    inputs = _input_rows(workspace_root, contract)
    if (package_root / "frozen").exists() or any(
        (package_root / path).exists() for path in DERIVED_PATHS
    ):
        raise FileExistsError(
            f"Refusing to replace frozen principal ViPErLEED package: {package_root}"
        )
    execution_root = workspace_root / EXECUTION_OUTPUT_PATH
    for scan_id in SCAN_IDS:
        origin = execution_root / scan_id / "native"
        actual = {path.name for path in origin.iterdir() if path.is_file()}
        expected = set(_native_file_names(scan_id))
        if actual != expected:
            raise ViperleedPrincipalValidationError(
                f"{scan_id} native execution output set changed"
            )

    with TemporaryDirectory(
        prefix="viperleed-principal-",
        dir=package_root.parent,
    ) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        shutil.copyfile(package_root / ".gitattributes", stage / ".gitattributes")
        shutil.copyfile(package_root / "contract.json", stage / "contract.json")
        shutil.copytree(package_root / "execution", stage / "execution")
        for scan_id in SCAN_IDS:
            source = execution_root / scan_id / "native"
            destination = stage / "frozen" / scan_id
            shutil.copytree(source, destination)
        records, summary = _build_records_and_summary(
            workspace_root,
            contract,
            stage,
            inputs,
        )
        (stage / "four_branch_records.csv").write_bytes(_csv_bytes(records))
        (stage / "run_summary.json").write_bytes(_json_bytes(summary))
        authentication = _build_run_authentication(contract, stage)
        (stage / "run_authentication.json").write_bytes(_json_bytes(authentication))
        manifest = _build_manifest(contract, authentication, stage)
        (stage / "manifest.json").write_bytes(_json_bytes(manifest))
        report = _computed_report(stage, summary)
        (stage / "validation_report.json").write_bytes(_json_bytes(report))
        validated_report = _validate_frozen_package(
            workspace_root,
            stage,
            enforce_frozen_hashes=True,
        )

        shutil.copytree(stage / "frozen", package_root / "frozen")
        for path in DERIVED_PATHS:
            os.replace(stage / path, package_root / path)
    return validated_report


def validate_principal_viperleed_scans(
    workspace_root: Path,
    package_root: Path | None = None,
) -> dict[str, object]:
    """Validate the frozen principal-scan package without mutating it."""

    workspace_root = workspace_root.resolve()
    package_root = (
        package_root.resolve()
        if package_root is not None
        else workspace_root / DEFAULT_PACKAGE_PATH
    )
    _contract_path(package_root)
    return _validate_frozen_package(
        workspace_root,
        package_root,
        enforce_frozen_hashes=True,
    )


__all__ = [
    "ViperleedPrincipalValidationError",
    "freeze_principal_viperleed_scans",
    "validate_principal_viperleed_scans",
]
