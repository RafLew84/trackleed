"""Reader-facing evidence for measurement-window and capture-contract sensitivity."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
from typing import TypedDict, get_type_hints

from trackleed.analysis.scientific_panel_common import METHODS
from trackleed.analysis.scientific_sensitivity_records import SensitivityResultRecord
from trackleed.analysis.scientific_tracer_records import read_csv, write_csv
from trackleed.publication.scientific_sensitivity_sources import (
    sha256_file,
    validate_sensitivity_source_data,
)


VARIANT_COUNT = 15
CASE_RESULT_COUNT = 300
METHOD_RESULT_COUNT = 75
RESULT_COUNT = CASE_RESULT_COUNT + METHOD_RESULT_COUNT
METHOD_DISPLAY_NAMES = {
    "easyleed": "EasyLEED",
    "locotrack": "LocoTrack",
    "ostrack": "OSTrack",
    "dam4sam": "DAM4SAM",
    "lk": "Lucas–Kanade",
}
FULL_PRECISION_CSV = "table_s6_method_aggregate_full_precision.csv"
CANONICAL_CSV = "table_s6_measurement_sensitivity.csv"
CODEBOOK_CSV = "table_s6_variant_codebook.csv"
WORKBOOK_XLSX = "table_s6_measurement_sensitivity.xlsx"


class VariantCodebookRecord(TypedDict):
    variant_id: str
    reader_code: str
    meaning: str
    contract_change: str


VARIANT_CODEBOOK: tuple[VariantCodebookRecord, ...] = (
    {
        "variant_id": "roi_32_proposal_12",
        "reader_code": "R32-P12",
        "meaning": "ROI 32 px; proposal-to-fit 12 px",
        "contract_change": "R32, P12; B18, R2-85, A1.00",
    },
    {
        "variant_id": "roi_32_proposal_15",
        "reader_code": "R32-P15",
        "meaning": "ROI 32 px; proposal-to-fit 15 px",
        "contract_change": "R32, P15; B18, R2-85, A1.00",
    },
    {
        "variant_id": "roi_32_proposal_18",
        "reader_code": "R32-P18",
        "meaning": "ROI 32 px; proposal-to-fit 18 px",
        "contract_change": "R32, P18; B18, R2-85, A1.00",
    },
    {
        "variant_id": "roi_48_proposal_12",
        "reader_code": "R48-P12",
        "meaning": "ROI 48 px; proposal-to-fit 12 px",
        "contract_change": "R48, P12; B18, R2-85, A1.00",
    },
    {
        "variant_id": "roi_48_proposal_15",
        "reader_code": "R48-P15*",
        "meaning": "frozen central contract",
        "contract_change": "central: R48, P15, B18, R2-85, A1.00",
    },
    {
        "variant_id": "roi_48_proposal_18",
        "reader_code": "R48-P18",
        "meaning": "ROI 48 px; proposal-to-fit 18 px",
        "contract_change": "R48, P18; B18, R2-85, A1.00",
    },
    {
        "variant_id": "roi_64_proposal_12",
        "reader_code": "R64-P12",
        "meaning": "ROI 64 px; proposal-to-fit 12 px",
        "contract_change": "R64, P12; B18, R2-85, A1.00",
    },
    {
        "variant_id": "roi_64_proposal_15",
        "reader_code": "R64-P15",
        "meaning": "ROI 64 px; proposal-to-fit 15 px",
        "contract_change": "R64, P15; B18, R2-85, A1.00",
    },
    {
        "variant_id": "roi_64_proposal_18",
        "reader_code": "R64-P18",
        "meaning": "ROI 64 px; proposal-to-fit 18 px",
        "contract_change": "R64, P18; B18, R2-85, A1.00",
    },
    {
        "variant_id": "branch_tolerance_12",
        "reader_code": "B12",
        "meaning": "branch tolerance 12 px",
        "contract_change": "R48, P15; B12, R2-85, A1.00",
    },
    {
        "variant_id": "branch_tolerance_24",
        "reader_code": "B24",
        "meaning": "branch tolerance 24 px",
        "contract_change": "R48, P15; B24, R2-85, A1.00",
    },
    {
        "variant_id": "r_squared_0p80",
        "reader_code": "R2-80",
        "meaning": "R² minimum 0.80",
        "contract_change": "R48, P15; B18, R2-80, A1.00",
    },
    {
        "variant_id": "r_squared_0p90",
        "reader_code": "R2-90",
        "meaning": "R² minimum 0.90",
        "contract_change": "R48, P15; B18, R2-90, A1.00",
    },
    {
        "variant_id": "amplitude_multiplier_0p75",
        "reader_code": "A0.75",
        "meaning": "amplitude gate multiplier 0.75",
        "contract_change": (
            "R48, P15; B18, R2-85, A0.75 (threshold 0.009)"
        ),
    },
    {
        "variant_id": "amplitude_multiplier_1p25",
        "reader_code": "A1.25",
        "meaning": "amplitude gate multiplier 1.25",
        "contract_change": (
            "R48, P15; B18, R2-85, A1.25 (threshold 0.015)"
        ),
    },
)
VARIANT_DISPLAY = {row["variant_id"]: row for row in VARIANT_CODEBOOK}
IMPLEMENTATION_PATHS = (
    Path("trackleed/publication/scientific_sensitivity_evidence.py"),
    Path("trackleed/publication/scientific_sensitivity_sources.py"),
)
TABLE_S6_BEGIN = "<!-- BEGIN TABLE S6 -->"
TABLE_S6_END = "<!-- END TABLE S6 -->"


@dataclass(frozen=True)
class TableS6Report:
    result_count: int
    method_result_count: int
    case_result_count: int
    variant_count: int


def _load_sensitivity_results(
    source_directory: Path,
) -> list[SensitivityResultRecord]:
    validate_sensitivity_source_data(source_directory)
    _, rows = read_csv(
        source_directory / "sensitivity_results.csv",
        SensitivityResultRecord,
    )
    scopes = Counter(row["scope"] for row in rows)
    variants = {row["variant_id"] for row in rows}
    method_keys = {
        (row["variant_id"], row["method"])
        for row in rows
        if row["scope"] == "method"
    }
    if (
        len(rows) != RESULT_COUNT
        or scopes != {"case": CASE_RESULT_COUNT, "method": METHOD_RESULT_COUNT}
        or len(variants) != VARIANT_COUNT
        or variants != set(VARIANT_DISPLAY)
        or method_keys != {(variant, method) for variant in variants for method in METHODS}
    ):
        raise ValueError("Table S6 sensitivity result matrix changed")
    for row in rows:
        if (
            (
                row["scope"] == "method"
                and (
                    row["physics_baseline_visible_opportunities"] != 221
                    or row["evaluated_gap_count"] != 24
                )
            )
            or row["geometry_inconsistency_denominator"]
            != row["accepted_measurements_all_frames"]
            or row["pair_residual_denominator"]
            != row["jointly_consistent_pair_frames"]
        ):
            raise ValueError("Table S6 denominator contract changed")
    return rows


def _render_table_s6(rows: list[SensitivityResultRecord]) -> str:
    method_rows = [row for row in rows if row["scope"] == "method"]
    lines = [
        (
            "**Table S6. Measurement-window and capture-contract "
            "sensitivity.** For the independent Ag(111) stress test, coverage "
            "uses the fixed "
            "denominator 221 and post-gap branch-consistent recovery uses the "
            "fixed denominator 24. Geometry inconsistency is conditional on all "
            "accepted fits, while the pair-distance residual retains its method- "
            "and variant-specific conditional n. Reader-facing residuals are "
            "rounded to three decimals; full-precision values remain in the "
            "accompanying CSV and XLSX exports. No comparative rank or automatic "
            "configuration selection is implied. The asterisk marks only the "
            "frozen central contract."
        ),
        "",
        "**Variant codebook.**",
        "",
        *[
            f"- `{record['reader_code']}`: {record['meaning']}."
            for record in VARIANT_CODEBOOK
        ],
        "",
        (
            '<!-- table-s6-row-style reader_code="R48-P15*" '
            'fill="#EAF2F8" non_color_marker="*" -->'
        ),
        "",
        (
            "| Variant | Method | Contract change | Coverage | Recovery | "
            "Geometry inconsistency | Conditional median / P95 [px] (`n`) |"
        ),
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        display = VARIANT_DISPLAY[row["variant_id"]]
        residual = (
            "— (`n=0`)"
            if row["pair_residual_denominator"] == 0
            else (
                f"{row['median_absolute_pair_distance_residual_px']:.3f} / "
                f"{row['p95_absolute_pair_distance_residual_px']:.3f} "
                f"(`n={row['pair_residual_denominator']}`)"
            )
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{display['reader_code']}`",
                    METHOD_DISPLAY_NAMES[row["method"]],
                    display["contract_change"],
                    (
                        f"{row['branch_consistent_measurements']}/"
                        f"{row['physics_baseline_visible_opportunities']}"
                    ),
                    (
                        f"{row['branch_consistent_recovered_gap_count']}/"
                        f"{row['evaluated_gap_count']}"
                    ),
                    (
                        f"{row['geometry_inconsistent_accepted_fits']}/"
                        f"{row['geometry_inconsistency_denominator']}"
                    ),
                    residual,
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _write_table_s6_workbook(
    path: Path,
    method_rows: list[SensitivityResultRecord],
    case_rows: list[SensitivityResultRecord],
) -> None:
    """Write a deterministic, typed convenience workbook from canonical rows."""

    try:
        import xlsxwriter
    except ImportError as exc:  # pragma: no cover - exercised by runtime setup
        raise RuntimeError(
            "Table S6 XLSX export requires XlsxWriter"
        ) from exc

    fields = list(get_type_hints(SensitivityResultRecord))
    field_types = get_type_hints(SensitivityResultRecord)
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True})
    workbook.set_properties(
        {
            "title": "Table S6 full-precision sensitivity records",
            "subject": "Measurement-window and capture-contract sensitivity",
            "author": "TrackLEED",
            "company": "TrackLEED",
            "comments": "CSV is the canonical source of truth.",
            "created": datetime(1980, 1, 1),
        }
    )
    header = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#17365D",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
        }
    )
    base_properties = {"border": 1, "border_color": "#D9E2F3"}
    central_properties = {**base_properties, "bg_color": "#EAF2F8"}
    formats: dict[tuple[str, bool], object] = {}
    number_formats = {
        "proposal_to_fit_max_px": "0",
        "branch_tolerance_px": "0",
        "r_squared_min": "0.00",
        "amplitude_threshold_min": "0.000",
        "sigma_min_px": "0.0",
        "sigma_max_px": "0.0",
        "branch_consistent_coverage": "0.00%",
        "branch_consistent_recovery_rate": "0.00%",
        "geometry_inconsistent_fraction": "0.00%",
        "median_absolute_pair_distance_residual_px": "0.000",
        "p95_absolute_pair_distance_residual_px": "0.000",
        "maximum_absolute_pair_distance_residual_px": "0.000",
    }

    def cell_format(field: str, central: bool) -> object:
        annotation = field_types[field]
        category = "text"
        if annotation is int:
            category = "integer"
        elif annotation is float:
            category = number_formats.get(field, "0.000")
        key = (category, central)
        if key not in formats:
            properties = dict(central_properties if central else base_properties)
            if category == "integer":
                properties["num_format"] = "0"
            elif category != "text":
                properties["num_format"] = category
            formats[key] = workbook.add_format(properties)
        return formats[key]

    for sheet_name, rows in (
        ("method_aggregate", method_rows),
        ("case_records", case_rows),
    ):
        worksheet = workbook.add_worksheet(sheet_name)
        worksheet.hide_gridlines(2)
        worksheet.freeze_panes(1, 7)
        worksheet.set_row(0, 42)
        worksheet.write_row(0, 0, fields, header)
        for row_index, row in enumerate(rows, start=1):
            central = row["variant_id"] == "roi_48_proposal_15"
            for column_index, field in enumerate(fields):
                value = row[field]  # type: ignore[literal-required]
                fmt = cell_format(field, central)
                if value is None:
                    worksheet.write_blank(row_index, column_index, None, fmt)
                elif isinstance(value, bool):
                    worksheet.write_boolean(row_index, column_index, value, fmt)
                elif isinstance(value, (int, float)):
                    worksheet.write_number(row_index, column_index, value, fmt)
                else:
                    worksheet.write_string(row_index, column_index, value, fmt)
        worksheet.autofilter(0, 0, len(rows), len(fields) - 1)
        for column_index, field in enumerate(fields):
            width = min(max(len(field) + 2, 12), 30)
            if field in {"variant_id", "family", "case_id", "series_id", "pair_id"}:
                width = 24
            worksheet.set_column(column_index, column_index, width)
    workbook.close()


def build_table_s6(
    source_directory: Path,
    output_directory: Path,
) -> TableS6Report:
    """Build the full-precision source table and reader-facing Markdown."""

    rows = _load_sensitivity_results(source_directory.resolve())
    method_rows = [row for row in rows if row["scope"] == "method"]
    case_rows = [row for row in rows if row["scope"] == "case"]
    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_directory / CANONICAL_CSV,
        SensitivityResultRecord,
        rows,
    )
    write_csv(
        output_directory / FULL_PRECISION_CSV,
        SensitivityResultRecord,
        method_rows,
    )
    write_csv(
        output_directory / CODEBOOK_CSV,
        VariantCodebookRecord,
        VARIANT_CODEBOOK,
    )
    _write_table_s6_workbook(
        output_directory / WORKBOOK_XLSX,
        method_rows,
        case_rows,
    )
    (output_directory / "table_s6_measurement_sensitivity.md").write_text(
        _render_table_s6(rows),
        encoding="utf-8",
        newline="\n",
    )
    return TableS6Report(
        result_count=len(rows),
        method_result_count=len(method_rows),
        case_result_count=len(case_rows),
        variant_count=len({row["variant_id"] for row in rows}),
    )


def build_table_s6_artifact(
    workspace_root: Path,
    source_directory: Path,
    output_directory: Path,
    manifest_path: Path,
) -> TableS6Report:
    """Build Table S6 and bind sources, implementation, and outputs by hash."""

    workspace_root = workspace_root.resolve()
    source_directory = source_directory.resolve()
    output_directory = output_directory.resolve()
    report = build_table_s6(source_directory, output_directory)
    manifest = {
        "schema_version": 2,
        "table": "Table S6",
        "title": "Measurement-window and capture-contract sensitivity",
        "status": "completed",
        "result_count": report.result_count,
        "method_result_count": report.method_result_count,
        "case_result_count": report.case_result_count,
        "variant_count": report.variant_count,
        "source_data_sha256": {
            path.name: sha256_file(path)
            for path in sorted(source_directory.iterdir())
            if path.is_file()
        },
        "implementation_sha256": {
            path.as_posix(): sha256_file(workspace_root / path)
            for path in IMPLEMENTATION_PATHS
        },
        "outputs_sha256": {
            path.name: sha256_file(path)
            for path in sorted(output_directory.iterdir())
            if path.is_file()
        },
        "denominator_policy": (
            "fixed reference opportunity sets with explicit variant-specific "
            "denominators for conditional outcomes"
        ),
        "automatic_configuration_selection": False,
        "xlsx_generator": {
            "package": "XlsxWriter",
            "version": version("XlsxWriter"),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def synchronize_table_s6(
    table_markdown_path: Path,
    supplementary_material_path: Path,
) -> None:
    """Insert or replace the generated Table S6 block in the supplement."""

    table_markdown = table_markdown_path.read_text(encoding="utf-8").rstrip()
    supplement = supplementary_material_path.read_text(encoding="utf-8")
    block = (
        f"{TABLE_S6_BEGIN}\n"
        f"{table_markdown}\n"
        f"{TABLE_S6_END}"
    )
    begin_count = supplement.count(TABLE_S6_BEGIN)
    end_count = supplement.count(TABLE_S6_END)
    if (begin_count, end_count) == (0, 0):
        supplement = (
            supplement.rstrip()
            + "\n\n## Table S6 — Measurement-window and capture-contract sensitivity\n\n"
            + block
            + "\n"
        )
    elif (begin_count, end_count) == (1, 1):
        before, remainder = supplement.split(TABLE_S6_BEGIN, 1)
        _, after = remainder.split(TABLE_S6_END, 1)
        supplement = before + block + after
    else:
        raise ValueError("Table S6 supplement markers changed")
    supplementary_material_path.write_text(
        supplement,
        encoding="utf-8",
        newline="\n",
    )


__all__ = [
    "TableS6Report",
    "build_table_s6",
    "build_table_s6_artifact",
    "synchronize_table_s6",
]
