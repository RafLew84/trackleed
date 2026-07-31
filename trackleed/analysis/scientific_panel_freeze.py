"""One-time construction of the authenticated Ag(111) panel package."""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from trackleed.analysis.scientific_panel_common import (
    BASELINE_MANIFEST_PATH,
    CASE_SPECS,
    MEASUREMENT_CONTRACT_PATH,
    METHODS,
    PANEL_AUTHENTICATION_PATH,
    PANEL_ID,
    PUBLISHED_AGGREGATE_RESULTS_PATH,
    PUBLISHED_CASE_RESULTS_PATH,
    SCHEMA_VERSION,
    TracerValidationError,
    read_json,
    sha256,
    write_json,
)
from trackleed.analysis.scientific_panel_contract import (
    PANEL_SCHEMAS,
    panel_package_manifest,
    validate_panel_package,
)
from trackleed.analysis.scientific_panel_reconstruction import (
    PanelArtifacts,
    reconstruct_panel_artifacts,
)
from trackleed.analysis.scientific_panel_records import (
    HISTORICAL_CENTRAL_FIELDS,
    AggregateReportRecord,
    CaseReportRecord,
    PanelGapRecord,
    PanelPairRecord,
)
from trackleed.analysis.scientific_panel_validation import validate_panel_result
from trackleed.analysis.scientific_tracer_contract import runtime_fingerprint
from trackleed.analysis.scientific_tracer_records import (
    CandidateRecord,
    EvaluationRecord,
    read_raw_csv,
    write_csv,
)


README = """# Complete Ag(111) central-panel reconstruction

This immutable package freezes the historical record-by-record result for the
five-method, four-case Ag(111) panel and the expected outputs reconstructed
from the separately frozen scientific-baseline inputs.

The runtime path is headless and CPU-only. It does not run a tracker, load a
checkpoint, open the GUI, select a configuration or update the scientific
baseline. Candidate and evaluation records retain proposal, ROI, solver and
gate outcomes separately. Coverage and recovery use the fixed 221/24 key sets;
geometry inconsistency and pair residuals report their conditional per-method
denominators.

Run:

```text
python -m scripts.run_publication_panel --clean --validate
```

Validation authenticates this package, the scientific-baseline package and
the existing publication source-data anchors before comparing every discrete
and continuous record under the predeclared tolerances.
"""


def _historical_path(
    historical_root: Path,
    series: str,
    pair: str,
    method: str,
) -> Path:
    case_dir = historical_root / series / pair
    if method == "easyleed":
        return case_dir / "easyleed" / "easyleed_measurements.csv"
    if method == "dam4sam":
        return case_dir / "trackers" / "dam4sam_measurements.csv"
    return case_dir / "representatives" / f"{method}_measurements.csv"


def _freeze_historical_records(
    workspace_root: Path,
    historical_root: Path,
    frozen_dir: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, object]] = []
    inventory: list[dict[str, str]] = []
    for case_id, series_id, pair_id, series, pair, _frame_count in CASE_SPECS:
        for method in METHODS:
            path = _historical_path(historical_root, series, pair, method)
            fields, source_rows = read_raw_csv(path)
            required = set(HISTORICAL_CENTRAL_FIELDS) - {
                "case_id",
                "series_id",
                "pair_id",
            }
            if set(fields) != required:
                raise TracerValidationError(
                    f"Historical central schema changed: {path}"
                )
            for source_row in source_rows:
                rows.append(
                    {
                        "case_id": case_id,
                        "series_id": series_id,
                        "pair_id": pair_id,
                        **source_row,
                    }
                )
            try:
                source_path = path.resolve().relative_to(
                    workspace_root.resolve()
                ).as_posix()
            except ValueError:
                source_path = path.resolve().as_posix()
            inventory.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "source_path": source_path,
                    "source_sha256": sha256(path),
                }
            )
    if len(rows) != 1620:
        raise TracerValidationError(
            f"Expected 1620 historical central records, found {len(rows)}"
        )
    from csv import DictWriter

    historical_path = frozen_dir / "expected_historical_central_records.csv"
    with historical_path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(
            handle,
            fieldnames=list(HISTORICAL_CENTRAL_FIELDS),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return inventory


def _base_contract(workspace_root: Path) -> dict[str, Any]:
    measurement = read_json(workspace_root / MEASUREMENT_CONTRACT_PATH)
    return {
        "schema_version": SCHEMA_VERSION,
        "tracer_id": PANEL_ID,
        "scope": {
            "cases": [case[0] for case in CASE_SPECS],
            "methods": list(METHODS),
        },
        "comparison_tolerances": measurement["comparison_tolerances"],
        "reader_facing_rounding": measurement["reader_facing_rounding"],
        "denominators": measurement["denominators"],
        "estimator_compatibility": {
            "center_preference_factor": 0.34,
            "image_loading": (
                "tifffile.imread followed by skimage.util.img_as_float"
            ),
            "mode": "historical_publication_central_record",
            "optimizer_theta_bounds_rad": [-math.pi, math.pi],
            "reason": (
                "The frozen central record predates the unbounded-periodic-angle "
                "optimizer correction; the application default remains unbounded."
            ),
        },
        "execution_contract": {
            "device": "cpu",
            "runtime": runtime_fingerprint(),
        },
        "frozen_inputs": {
            "scientific_baseline_manifest": {
                "path": BASELINE_MANIFEST_PATH.as_posix(),
                "sha256": sha256(workspace_root / BASELINE_MANIFEST_PATH),
            },
            "published_case_results": {
                "path": PUBLISHED_CASE_RESULTS_PATH.as_posix(),
                "sha256": sha256(workspace_root / PUBLISHED_CASE_RESULTS_PATH),
            },
            "published_aggregate_results": {
                "path": PUBLISHED_AGGREGATE_RESULTS_PATH.as_posix(),
                "sha256": sha256(
                    workspace_root / PUBLISHED_AGGREGATE_RESULTS_PATH
                ),
            },
        },
        "expected_files": {
            "candidate_records": "frozen/expected_candidate_records.csv",
            "evaluation_records": "frozen/expected_evaluation_records.csv",
            "pair_records": "frozen/expected_pair_records.csv",
            "gap_records": "frozen/expected_gap_records.csv",
            "case_results": "frozen/expected_case_results.csv",
            "aggregate_method_results": (
                "frozen/expected_aggregate_method_results.csv"
            ),
            "historical_central_records": (
                "frozen/expected_historical_central_records.csv"
            ),
            "summary": "frozen/expected_summary.json",
        },
        "expected_output_schemas": {
            name: list(fields) for name, fields in PANEL_SCHEMAS.items()
        },
    }


def _write_expected_records(
    panel_dir: Path,
    contract: dict[str, Any],
    artifacts: PanelArtifacts,
) -> None:
    expected = contract["expected_files"]
    write_csv(
        panel_dir / expected["candidate_records"],
        CandidateRecord,
        artifacts.candidates,
    )
    write_csv(
        panel_dir / expected["evaluation_records"],
        EvaluationRecord,
        artifacts.evaluations,
    )
    write_csv(
        panel_dir / expected["pair_records"],
        PanelPairRecord,
        artifacts.pairs,
    )
    write_csv(
        panel_dir / expected["gap_records"],
        PanelGapRecord,
        artifacts.gaps,
    )
    write_csv(
        panel_dir / expected["case_results"],
        CaseReportRecord,
        artifacts.case_reports,
    )
    write_csv(
        panel_dir / expected["aggregate_method_results"],
        AggregateReportRecord,
        artifacts.aggregate_reports,
    )
    write_json(panel_dir / expected["summary"], artifacts.summary)
    contract["expected_counts"] = {
        "candidate_records": len(artifacts.candidates),
        "evaluation_records": len(artifacts.evaluations),
        "pair_records": len(artifacts.pairs),
        "gap_records": len(artifacts.gaps),
        "case_results": len(artifacts.case_reports),
        "aggregate_method_results": len(artifacts.aggregate_reports),
        "historical_central_records": 1620,
    }


def build_panel_package(
    workspace_root: Path,
    panel_dir: Path,
    historical_root: Path,
) -> dict[str, object]:
    """Build a new package; refuse to replace any existing panel package."""

    workspace_root = workspace_root.resolve()
    panel_dir = panel_dir.resolve()
    historical_root = historical_root.resolve()
    if panel_dir.exists():
        raise FileExistsError(f"Refusing to replace frozen panel: {panel_dir}")
    if not historical_root.is_dir():
        raise FileNotFoundError(historical_root)
    authentication_path = workspace_root / PANEL_AUTHENTICATION_PATH
    if authentication_path.exists():
        raise FileExistsError(
            f"Refusing to replace panel authentication: {authentication_path}"
        )
    panel_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=panel_dir.parent) as temporary:
        staged = Path(temporary) / panel_dir.name
        frozen_dir = staged / "frozen"
        frozen_dir.mkdir(parents=True)
        (staged / "README.md").write_text(README, encoding="utf-8", newline="\n")
        inventory = _freeze_historical_records(
            workspace_root, historical_root, frozen_dir
        )
        write_json(
            frozen_dir / "historical_source_inventory.json",
            {"schema_version": SCHEMA_VERSION, "sources": inventory},
        )
        contract = _base_contract(workspace_root)
        artifacts = reconstruct_panel_artifacts(workspace_root, contract)
        _write_expected_records(staged, contract, artifacts)
        write_json(frozen_dir / "panel_contract.json", contract)
        write_json(staged / "package_manifest.json", panel_package_manifest(staged))
        staged_authentication = Path(temporary) / "authentication.json"
        authentication = {
            "package_manifest_sha256": sha256(staged / "package_manifest.json"),
            "schema_version": SCHEMA_VERSION,
            "tracer_id": PANEL_ID,
        }
        write_json(staged_authentication, authentication)
        validate_panel_package(
            workspace_root,
            staged,
            authentication_path=staged_authentication,
        )
        validate_panel_result(workspace_root, staged, contract, artifacts)
        staged.rename(panel_dir)
    write_json(authentication_path, authentication)
    return authentication


__all__ = ["build_panel_package"]
