#!/usr/bin/env python3
"""Build supplementary Table S4 from the frozen Figure 3 endpoint contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path


TABLE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = TABLE_ROOT.parents[2]
SOURCE_DATA_DIRECTORY = TABLE_ROOT / "source_data"
PROVENANCE_PATH = SOURCE_DATA_DIRECTORY / "source_provenance.json"
ENDPOINTS_PATH = SOURCE_DATA_DIRECTORY / "endpoint_summaries.csv"
LANDMARKS_PATH = SOURCE_DATA_DIRECTORY / "cycle_landmarks.csv"
SPREAD_PATH = SOURCE_DATA_DIRECTORY / "between_method_spread.csv"
OUTPUT_DIRECTORY = TABLE_ROOT / "output"
MANIFEST_PATH = TABLE_ROOT / "manifest.json"
SUPPLEMENT_PATH = TABLE_ROOT.parents[1] / "supplementary-material.md"

SUPPLEMENT_BEGIN_MARKER = "<!-- BEGIN TABLE S4 -->"
SUPPLEMENT_END_MARKER = "<!-- END TABLE S4 -->"
SUPPLEMENT_HEADING = "## Table S4 — Cu(111) pairwise physical endpoints"

ENDPOINT_COLUMNS = (
    "method",
    "display_method",
    "pair",
    "valid_frame_count",
    "reference_valid_frame_count",
    "reference_distance_px",
    "plateau_median_percent",
    "minimum_frame_order",
    "minimum_percent",
    "final_percent",
)
LANDMARK_COLUMNS = (
    "landmark",
    "reader_label",
    "frame_start",
    "frame_end",
    "temperature_min_c",
    "temperature_max_c",
    "selection_basis",
    "outcome_selected",
)
SPREAD_COLUMNS = (
    "pair",
    "summary_point",
    "reader_label",
    "frame_order",
    "minimum_method",
    "minimum_value_percent",
    "maximum_method",
    "maximum_value_percent",
    "spread_percentage_points",
    "display_in_panel_d",
)
OUTPUT_COLUMNS = (
    "pair",
    "method",
    "valid_endpoints",
    "reference_valid_images",
    "reference_distance_px",
    "plateau_median_percent",
    "post_plateau_minimum_percent",
    "final_cooled_state_percent",
)

METHODS = (
    (
        "locotrack_base_offline",
        "LocoTrack",
        "LocoTrack",
    ),
    (
        "ostrack_vitb_256_mae_ce",
        "OSTrack",
        "OSTrack",
    ),
    (
        "dam4sam_sam21pp_l",
        "DAM4SAM",
        "DAM4SAM",
    ),
    (
        "lucas_kanade_skip_anchor",
        "Lucas-Kanade",
        "Lucas–Kanade",
    ),
    (
        "easyleed_kalman",
        "EasyLEED",
        "EasyLEED",
    ),
)
METHOD_ORDER = tuple(method[2] for method in METHODS)
METHOD_BY_SOURCE = {method[0]: method for method in METHODS}
PAIR_LABELS = {"1-6": "1–6", "3-5": "3–5"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: object, *, field: str) -> None:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"Malformed SHA-256 in {field}: {digest}")


def _read_contract_csv(
    path: Path,
    expected_columns: tuple[str, ...],
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError(f"Unexpected columns in {path.name}.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No records in {path.name}.")
    if any(value is None for row in rows for value in row.values()):
        raise ValueError(f"Malformed record in {path.name}.")
    return rows


def validate_provenance() -> dict[str, object]:
    loaded = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    if (
        not isinstance(loaded, dict)
        or set(loaded) != {"freeze_policy", "schema_version", "sources"}
        or loaded["schema_version"] != "1.0"
        or not isinstance(loaded["freeze_policy"], str)
        or not loaded["freeze_policy"].strip()
    ):
        raise ValueError("Malformed source provenance.")
    sources = loaded["sources"]
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("Source provenance must list three frozen sources.")
    required_keys = {
        "frozen_path",
        "frozen_sha256",
        "origin_path",
        "origin_sha256",
        "role",
    }
    for source in sources:
        if not isinstance(source, dict) or set(source) != required_keys:
            raise ValueError("Malformed source-provenance entry.")
        for text_key in ("frozen_path", "origin_path", "role"):
            if (
                not isinstance(source[text_key], str)
                or not source[text_key].strip()
            ):
                raise ValueError(f"Malformed provenance field: {text_key}")
        _validate_sha256(source["frozen_sha256"], field="frozen_sha256")
        _validate_sha256(source["origin_sha256"], field="origin_sha256")

        frozen_path = (TABLE_ROOT / str(source["frozen_path"])).resolve()
        if TABLE_ROOT not in frozen_path.parents:
            raise ValueError("Frozen provenance path escapes the artifact.")
        if not frozen_path.is_file():
            raise ValueError(f"Missing frozen source: {frozen_path.name}")
        if sha256_file(frozen_path) != source["frozen_sha256"]:
            raise ValueError(f"Frozen source hash changed: {frozen_path.name}")

        origin_path = (
            REPOSITORY_ROOT / str(source["origin_path"])
        ).resolve()
        if (
            origin_path != REPOSITORY_ROOT
            and REPOSITORY_ROOT not in origin_path.parents
        ):
            raise ValueError("Origin provenance path escapes the repository.")
        if not origin_path.is_file():
            raise ValueError(f"Missing origin source: {origin_path.name}")
        if sha256_file(origin_path) != source["origin_sha256"]:
            raise ValueError(f"Origin source hash changed: {origin_path.name}")
    return loaded


def _finite_decimal(value: str, *, field: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except Exception as error:
        raise ValueError(f"Invalid numeric value in {field}: {value}") from error
    if not parsed.is_finite():
        raise ValueError(f"Nonfinite numeric value in {field}: {value}")
    return parsed


def load_and_validate_endpoints() -> list[dict[str, str]]:
    source_rows = _read_contract_csv(ENDPOINTS_PATH, ENDPOINT_COLUMNS)
    expected_keys = {
        (source_method, source_pair)
        for source_method in METHOD_BY_SOURCE
        for source_pair in PAIR_LABELS
    }
    actual_keys = {(row["method"], row["pair"]) for row in source_rows}
    if len(source_rows) != 10 or actual_keys != expected_keys:
        raise ValueError("Expected five methods for each of two primary pairs.")

    output: list[dict[str, str]] = []
    for row in source_rows:
        source_method = row["method"]
        expected_display = METHOD_BY_SOURCE[source_method][1]
        reader_method = METHOD_BY_SOURCE[source_method][2]
        if row["display_method"] != expected_display:
            raise ValueError(f"Unexpected display method for {source_method}.")
        if row["valid_frame_count"] != "701":
            raise ValueError("Every primary pair must retain 701 endpoints.")
        if row["reference_valid_frame_count"] != "10":
            raise ValueError("Every reference must retain all first ten images.")
        if row["minimum_frame_order"] != "378":
            raise ValueError("The common post-plateau minimum must be retained.")

        reference_distance = _finite_decimal(
            row["reference_distance_px"],
            field="reference_distance_px",
        )
        plateau = _finite_decimal(
            row["plateau_median_percent"],
            field="plateau_median_percent",
        )
        minimum = _finite_decimal(
            row["minimum_percent"],
            field="minimum_percent",
        )
        final = _finite_decimal(
            row["final_percent"],
            field="final_percent",
        )
        if reference_distance <= 0:
            raise ValueError("Reference pair distance must be positive.")
        if not minimum < plateau < final:
            raise ValueError("Cu(111) endpoint landmark ordering changed.")

        output.append(
            {
                "source_method": source_method,
                "source_pair": row["pair"],
                "pair": PAIR_LABELS[row["pair"]],
                "method": reader_method,
                "valid_endpoints": "701/701",
                "reference_valid_images": "10/10",
                "reference_distance_px": row["reference_distance_px"],
                "plateau_median_percent": row["plateau_median_percent"],
                "minimum_source_frame": row["minimum_frame_order"],
                "post_plateau_minimum_percent": row["minimum_percent"],
                "final_cooled_state_percent": row["final_percent"],
            }
        )
    return sorted(
        output,
        key=lambda row: (
            tuple(PAIR_LABELS.values()).index(row["pair"]),
            METHOD_ORDER.index(row["method"]),
        ),
    )


def load_and_validate_landmarks() -> list[dict[str, str]]:
    rows = _read_contract_csv(LANDMARKS_PATH, LANDMARK_COLUMNS)
    expected = {
        "reference": {
            "reader_label": "Reference (10-16 °C)",
            "frame_start": "0",
            "frame_end": "9",
            "temperature_min_c": "10",
            "temperature_max_c": "16",
            "outcome_selected": "false",
        },
        "plateau": {
            "reader_label": "800 °C plateau",
            "frame_start": "372",
            "frame_end": "376",
            "temperature_min_c": "800",
            "temperature_max_c": "800",
            "outcome_selected": "false",
        },
        "post_plateau_minimum": {
            "reader_label": "Post-plateau minimum (797 °C)",
            "frame_start": "378",
            "frame_end": "378",
            "temperature_min_c": "797",
            "temperature_max_c": "797",
            "outcome_selected": "true",
        },
        "final": {
            "reader_label": "Final cooled state (92 °C)",
            "frame_start": "700",
            "frame_end": "700",
            "temperature_min_c": "92",
            "temperature_max_c": "92",
            "outcome_selected": "false",
        },
    }
    if len(rows) != 4 or {row["landmark"] for row in rows} != set(expected):
        raise ValueError("Unexpected Cu(111) cycle-landmark contract.")
    output: list[dict[str, str]] = []
    for row in rows:
        expected_fields = expected[row["landmark"]]
        if any(row[key] != value for key, value in expected_fields.items()):
            raise ValueError(f"Landmark contract changed: {row['landmark']}")
        if row["landmark"] == "plateau" and (
            row["selection_basis"] != "predefined five-image median"
        ):
            raise ValueError("Plateau aggregation contract changed.")
        if row["landmark"] == "post_plateau_minimum" and (
            row["selection_basis"]
            != "minimum after the 800 °C plateau; common to all methods and pairs"
        ):
            raise ValueError("Post-plateau selection contract changed.")
        temperature_min = row["temperature_min_c"]
        temperature_max = row["temperature_max_c"]
        temperature = (
            f"{temperature_min} °C"
            if temperature_min == temperature_max
            else f"{temperature_min}–{temperature_max} °C"
        )
        output.append(
            {
                **row,
                "temperature": temperature,
                "selection": (
                    "outcome-selected"
                    if row["outcome_selected"] == "true"
                    else "predefined"
                ),
            }
        )
    return output


def validate_between_method_spread(
    endpoint_rows: list[dict[str, str]],
) -> None:
    spread_rows = _read_contract_csv(SPREAD_PATH, SPREAD_COLUMNS)
    if len(spread_rows) != 8:
        raise ValueError("Expected six displayed and two audit-only spread rows.")
    displayed = [
        row for row in spread_rows if row["display_in_panel_d"] == "true"
    ]
    audit_only = [
        row for row in spread_rows if row["display_in_panel_d"] == "false"
    ]
    if (
        len(displayed) != 6
        or {row["summary_point"] for row in audit_only}
        != {"entire_sequence_maximum"}
    ):
        raise ValueError("Unexpected between-method spread roles.")

    endpoint_fields = {
        "plateau": "plateau_median_percent",
        "post_plateau_minimum": "post_plateau_minimum_percent",
        "final": "final_cooled_state_percent",
    }
    expected_frames = {
        "plateau": "",
        "post_plateau_minimum": "378",
        "final": "700",
    }
    displayed_keys = {
        (row["pair"], row["summary_point"]) for row in displayed
    }
    expected_keys = {
        (pair, point)
        for pair in PAIR_LABELS
        for point in endpoint_fields
    }
    if displayed_keys != expected_keys:
        raise ValueError("Displayed spread rows do not cover all endpoints.")

    for spread in displayed:
        pair_rows = [
            row
            for row in endpoint_rows
            if row["source_pair"] == spread["pair"]
        ]
        field = endpoint_fields[spread["summary_point"]]
        ordered = sorted(
            pair_rows,
            key=lambda row: Decimal(row[field]),
        )
        minimum, maximum = ordered[0], ordered[-1]
        if spread["frame_order"] != expected_frames[spread["summary_point"]]:
            raise ValueError("Spread landmark frame contract changed.")
        if (
            spread["minimum_method"] != minimum["source_method"]
            or spread["maximum_method"] != maximum["source_method"]
            or Decimal(spread["minimum_value_percent"])
            != Decimal(minimum[field])
            or Decimal(spread["maximum_value_percent"])
            != Decimal(maximum[field])
        ):
            raise ValueError("Spread extrema do not match endpoint summaries.")
        observed_spread = float(spread["spread_percentage_points"])
        expected_spread = float(
            Decimal(maximum[field]) - Decimal(minimum[field])
        )
        if not math.isclose(
            observed_spread,
            expected_spread,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("Between-method spread value changed.")


def _display_percent(value: str) -> str:
    return f"{float(value):.6f}".replace("-", "−")


def render_markdown(rows: list[dict[str, str]]) -> str:
    caption = (
        "**Table S4. Cu(111) pairwise relative spot-radius endpoints by "
        "tracking method.** Values are ΔR/R = ΔD/D in percent after the "
        "native proposal from each method entered the common Gaussian "
        "measurement. Every method produced 701/701 valid endpoints for each "
        "primary pair. The 800 °C plateau is the median of a predefined "
        "five-image interval. The post-plateau minimum was selected as the "
        "lowest value after that interval; it occurred in the common image "
        "acquired at 797 °C and is therefore outcome-selected. The final "
        "cooled state is the predefined final image acquired at 92 °C. D_ref "
        "is the median valid pair "
        "distance in the first ten images, with at least eight required. "
        "Cross-method concordance after common refinement is not external "
        "absolute accuracy or an overall tracker ranking. The two pairs come "
        "from one series, are not independent replicates, and do not establish "
        "crystallographic anisotropy."
    )
    lines = [
        caption,
        "",
        (
            "| Pair | Method | Valid endpoints | 800 °C plateau median "
            "ΔR/R [%] | Post-plateau minimum (797 °C) ΔR/R [%] | "
            "Final cooled state (92 °C) ΔR/R [%] |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {pair} | {method} | {valid} | {plateau} | {minimum} | "
            "{final} |".format(
                pair=row["pair"],
                method=row["method"],
                valid=row["valid_endpoints"],
                plateau=_display_percent(row["plateau_median_percent"]),
                minimum=_display_percent(
                    row["post_plateau_minimum_percent"]
                ),
                final=_display_percent(row["final_cooled_state_percent"]),
            )
        )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {column: row[column] for column in OUTPUT_COLUMNS}
            for row in rows
        )


def _source_hashes(
    provenance: dict[str, object],
) -> dict[str, str]:
    paths = {
        PROVENANCE_PATH,
        Path(__file__).resolve(),
    }
    for source in provenance["sources"]:
        paths.add(TABLE_ROOT / str(source["frozen_path"]))
    return {
        path.relative_to(TABLE_ROOT).as_posix(): sha256_file(path)
        for path in sorted(paths)
    }


def build_artifact(
    *,
    output_directory: Path = OUTPUT_DIRECTORY,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, str]:
    provenance = validate_provenance()
    rows = load_and_validate_endpoints()
    landmarks = load_and_validate_landmarks()
    validate_between_method_spread(rows)

    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "table_s4_cu111_endpoints.csv": (
            output_directory / "table_s4_cu111_endpoints.csv"
        ),
        "table_s4_cu111_endpoints.md": (
            output_directory / "table_s4_cu111_endpoints.md"
        ),
    }
    _write_csv(paths["table_s4_cu111_endpoints.csv"], rows)
    paths["table_s4_cu111_endpoints.md"].write_text(
        render_markdown(rows),
        encoding="utf-8",
        newline="\n",
    )

    output_hashes = {
        name: sha256_file(path) for name, path in sorted(paths.items())
    }
    landmark_manifest = {
        row["landmark"]: {
            "reader_label": row["reader_label"],
            "selection": row["selection"],
            "temperature": row["temperature"],
        }
        for row in landmarks
    }
    manifest = {
        "artifact": "Supplementary Table S4",
        "landmarks": landmark_manifest,
        "outputs": output_hashes,
        "reporting_unit": (
            "method–pair within one Cu(111) thermal-cycle series"
        ),
        "row_count": len(rows),
        "schema_version": "1.0",
        "sources": _source_hashes(provenance),
        "status": "completed",
        "title": "Cu(111) pairwise physical endpoints",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_hashes


def _supplement_marker_bounds(
    supplement: str,
    *,
    allow_absent: bool,
) -> tuple[int, int] | None:
    begin_count = supplement.count(SUPPLEMENT_BEGIN_MARKER)
    end_count = supplement.count(SUPPLEMENT_END_MARKER)
    if begin_count == 0 and end_count == 0:
        if allow_absent:
            return None
        raise ValueError("Table S4 markers are missing from the supplement.")
    if begin_count != 1 or end_count != 1:
        raise ValueError(
            "Supplement must contain exactly one ordered Table S4 marker pair."
        )
    begin = supplement.index(SUPPLEMENT_BEGIN_MARKER)
    end = supplement.index(SUPPLEMENT_END_MARKER)
    if end < begin:
        raise ValueError("Table S4 end marker precedes its begin marker.")
    return begin, end


def extract_supplement_block(supplement: str) -> str:
    bounds = _supplement_marker_bounds(supplement, allow_absent=False)
    assert bounds is not None
    begin, end = bounds
    content_start = begin + len(SUPPLEMENT_BEGIN_MARKER)
    return supplement[content_start:end].strip() + "\n"


def sync_supplement(markdown: str) -> None:
    supplement = SUPPLEMENT_PATH.read_text(encoding="utf-8")
    bounds = _supplement_marker_bounds(supplement, allow_absent=True)
    block = (
        f"{SUPPLEMENT_BEGIN_MARKER}\n"
        f"{markdown.rstrip()}\n"
        f"{SUPPLEMENT_END_MARKER}"
    )
    if bounds is not None:
        begin, end = bounds
        before = supplement[:begin].rstrip()
        after = supplement[
            end + len(SUPPLEMENT_END_MARKER):
        ].lstrip()
        updated = f"{before}\n\n{block}\n"
        if after:
            updated += f"\n{after}"
    else:
        if SUPPLEMENT_HEADING in supplement:
            raise ValueError("Table S4 heading exists without generated markers.")
        updated = (
            supplement.rstrip()
            + f"\n\n{SUPPLEMENT_HEADING}\n\n"
            + block
            + "\n"
        )
    SUPPLEMENT_PATH.write_text(updated, encoding="utf-8", newline="\n")


def main() -> None:
    build_artifact()
    markdown_path = OUTPUT_DIRECTORY / "table_s4_cu111_endpoints.md"
    sync_supplement(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
