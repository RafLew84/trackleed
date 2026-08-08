"""Freeze and validate the constrained cross-contract supplement."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pypdf import PdfReader

DEFAULT_PACKAGE_PATH = Path("manuscript/cross_contract/supplement")
EXPECTED_CONTRACT_SHA256 = (
    "3e9f3f2d11761de216a64322d5756123650496cdd6eeada81ce8bb15da94bf70"
)
EXPECTED_MANIFEST_SHA256 = (
    "39a33d38b0978ff7051e7596625f4ad11a709e01b72a59f2aaa3a1b0a3024c34"
)
STATIC_PATHS = ("README.md", "contract.json", "supplement.md")
DERIVED_ROLES = {
    "output/figure_s1_pilot_calibration.png": "single-column pilot calibration figure",
    "output/figure_s2_sensitivity.png": "single-column sensitivity figure",
    "output/pdf/figure_s1_pilot_calibration.pdf": "vector pilot calibration figure",
    "output/pdf/figure_s2_sensitivity.pdf": "vector sensitivity figure",
    "source_data/figure_s1_pilot_calibration.csv": "compact pilot calibration figure data",
    "source_data/figure_s2_sensitivity.csv": "compact sensitivity figure data",
    "source_data/table_s1_measurement_contracts.csv": "measurement-contract table data",
    "source_data/table_s2_pilot_validation.csv": "restricted-pilot table data",
}
STATIC_ROLES = {
    "README.md": "supplement scope, workflow, and repository boundary",
    "contract.json": "frozen supplement content, display, and source contract",
    "supplement.md": "reader-facing constrained Supplementary Information",
}
OUTPUT_ROLES = {**STATIC_ROLES, **DERIVED_ROLES}
GENERATED_PATHS = tuple(DERIVED_ROLES) + ("manifest.json", "validation_report.json")
METHOD_ORDER = ("easyleed", "locotrack", "ostrack", "dam4sam", "lk")
METHOD_LABELS = {
    "easyleed": "EasyLEED",
    "locotrack": "LocoTrack",
    "ostrack": "OSTrack",
    "dam4sam": "DAM4SAM",
    "lk": "Lucas-Kanade",
}
METHOD_COLORS = {
    "easyleed": "#4477AA",
    "locotrack": "#EE6677",
    "ostrack": "#228833",
    "dam4sam": "#CCBB44",
    "lk": "#AA3377",
}
METHOD_MARKERS = {
    "easyleed": "o",
    "locotrack": "o",
    "ostrack": "s",
    "dam4sam": "x",
    "lk": "D",
}
METHOD_LINESTYLES = {
    "easyleed": "-",
    "locotrack": "-",
    "ostrack": "-.",
    "dam4sam": "--",
    "lk": ":",
}
BRANCH_COLORS = {
    "P1_B1": "#4477AA",
    "P1_B2": "#EE6677",
    "P2_B1": "#228833",
    "P2_B2": "#AA3377",
}
WORD_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", flags=re.UNICODE)
PDF_TIMESTAMP = datetime(2026, 8, 8, tzinfo=timezone.utc)


class CrossContractSupplementError(RuntimeError):
    """Raised when the cross-contract supplement package fails validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CrossContractSupplementError(
            f"cross-contract supplement source is unavailable: {path}"
        ) from exc
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CrossContractSupplementError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CrossContractSupplementError(f"invalid {label}: {path}")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise CrossContractSupplementError(f"invalid {label}: {path}") from exc


def _write_csv(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_contract(workspace_root: Path, package: Path) -> dict[str, Any]:
    contract_path = package / "contract.json"
    if not contract_path.is_file():
        raise CrossContractSupplementError(
            "cross-contract supplement contract is unavailable"
        )
    if EXPECTED_CONTRACT_SHA256 and _sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise CrossContractSupplementError("cross-contract supplement contract changed")
    contract = _load_json(contract_path, "cross-contract supplement contract")
    if (
        contract.get("schema_version") != "1.0"
        or contract.get("supplement_id") != "constrained-cross-contract-supplement"
        or contract.get("status") != "frozen-before-supplement-build"
        or contract.get("required_sections")
        != [
            "Scope and evidence boundary",
            "Frozen measurement contracts",
            "Restricted pilot calibration evidence",
            "Limited predeclared sensitivity",
            "Repository evidence boundary",
        ]
    ):
        raise CrossContractSupplementError(
            "cross-contract supplement contract identity changed"
        )
    displays = contract.get("display_inventory")
    if (
        not isinstance(displays, dict)
        or len(displays.get("figures", [])) != 2
        or len(displays.get("tables", [])) != 2
        or displays.get("single_column_width_mm") != 88.0
        or displays.get("minimum_font_size_pt") != 7.0
    ):
        raise CrossContractSupplementError(
            "cross-contract supplement display contract changed"
        )
    bindings = contract.get("source_bindings")
    if not isinstance(bindings, dict) or len(bindings) != 15:
        raise CrossContractSupplementError(
            "cross-contract supplement source bindings changed"
        )
    for binding_id, binding in bindings.items():
        if not isinstance(binding, dict):
            raise CrossContractSupplementError(
                f"invalid cross-contract supplement source binding: {binding_id}"
            )
        relative = binding.get("path")
        expected_sha256 = binding.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_sha256, str):
            raise CrossContractSupplementError(
                f"invalid cross-contract supplement source binding: {binding_id}"
            )
        if _sha256(workspace_root / relative) != expected_sha256:
            raise CrossContractSupplementError(
                f"cross-contract supplement source drift: {binding_id}"
            )
    return contract


def _pilot_source_rows(workspace_root: Path) -> list[dict[str, str]]:
    records = _read_csv(
        workspace_root / "manuscript/cross_contract/viperleed_pilot/pilot_records.csv",
        "restricted pilot records",
    )
    if len(records) != 60:
        raise CrossContractSupplementError(
            "restricted pilot branch-frame inventory changed"
        )
    direct_count = sum(
        row["viper_native_direct_detectability"].casefold() == "true" for row in records
    )
    displacement_count = sum(bool(row["cross_contract_distance"]) for row in records)
    calibration = [
        row
        for row in records
        if row["high_signal_calibration_subset"].casefold() == "true"
    ]
    distances = [float(row["cross_contract_distance"]) for row in calibration]
    if (
        direct_count != 59
        or displacement_count != 38
        or len(calibration) != 22
        or abs(max(distances) - 0.4548687654337075) > 1e-12
    ):
        raise CrossContractSupplementError(
            "restricted pilot supporting evidence changed"
        )
    return [
        {
            "energy_ev": f"{float(row['energy_ev']):.1f}",
            "branch_id": row["branch_id"],
            "native_significance": f"{float(row['significance']):.6f}",
            "cross_contract_distance_px": f"{float(row['cross_contract_distance']):.6f}",
        }
        for row in calibration
    ]


def _sensitivity_source_rows(workspace_root: Path) -> list[dict[str, str]]:
    rows = _read_csv(
        workspace_root / "manuscript/cross_contract/outcomes/sensitivity_results.csv",
        "cross-contract sensitivity results",
    )
    thresholds = {float(row["native_significance_threshold"]) for row in rows}
    tolerances = {float(row["cross_contract_positional_tolerance_px"]) for row in rows}
    methods = {row["method"] for row in rows}
    if (
        len(rows) != 45
        or thresholds != {2.0, 2.5, 3.0}
        or tolerances != {0.5, 1.0, 2.0}
        or methods != set(METHOD_ORDER)
    ):
        raise CrossContractSupplementError(
            "predeclared cross-contract sensitivity inventory changed"
        )
    invariant_fields = (
        "opportunity_count",
        "both_contracts_observe",
        "gaussian_only",
        "viperleed_only",
        "neither_contract_observes",
        "gaussian_gap_event_count",
        "viperleed_gap_event_count",
        "reference_visible_opportunities",
        "branch_consistent_on_reference_visible",
        "accepted_gaussian_all_frames",
        "out_of_branch_all_frames",
        "out_of_branch_unresolved",
    )
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        for field in invariant_fields:
            if len({row[field] for row in method_rows}) != 1:
                raise CrossContractSupplementError(
                    f"sensitivity denominator changed across variants: {method}/{field}"
                )
    nominal_threshold_rows = [
        row for row in rows if float(row["native_significance_threshold"]) == 2.5
    ]
    nominal_threshold_rows.sort(
        key=lambda row: (
            METHOD_ORDER.index(row["method"]),
            float(row["cross_contract_positional_tolerance_px"]),
        )
    )
    return [
        {
            "method": row["method"],
            "positional_tolerance_px": row["cross_contract_positional_tolerance_px"],
            "accepted_out_of_branch_events": row["out_of_branch_all_frames"],
            "branch_model_disagreement": row["out_of_branch_branch_model_disagreement"],
            "cross_contract_target_inconsistency": row[
                "out_of_branch_cross_contract_target_inconsistency"
            ],
            "unresolved_cross_contract_outcome": row["out_of_branch_unresolved"],
        }
        for row in nominal_threshold_rows
    ]


def _table_s1_rows() -> list[dict[str, str]]:
    return [
        {
            "measurement_path": "Local proposal-to-Gaussian",
            "qualifying_endpoint": "finite accepted Gaussian-refined position",
            "denominator_and_retention": "324 per method; rejection retained as G=0",
        },
        {
            "measurement_path": "Native indexed whole-pattern",
            "qualifying_endpoint": "significance >=2.5 and finite raw dx/dy",
            "denominator_and_retention": "324 mapped; non-direct retained as V=0",
        },
    ]


def _table_s2_rows() -> list[dict[str, str]]:
    return [
        {
            "pilot_diagnostic": "Pilot opportunity inventory",
            "frozen_value": "15 frames x 4 branches = 60",
            "evidence_role": "calibration-only; not an independent replicate",
        },
        {
            "pilot_diagnostic": "Native direct endpoint",
            "frozen_value": "59/60",
            "evidence_role": "contract check; not physical visibility",
        },
        {
            "pilot_diagnostic": "Cross-contract displacement",
            "frozen_value": "38/60",
            "evidence_role": "paired endpoints; not absolute accuracy",
        },
        {
            "pilot_diagnostic": "High-signal calibration subset",
            "frozen_value": "22 at significance >=20",
            "evidence_role": "predeclared tolerance input",
        },
        {
            "pilot_diagnostic": "Maximum calibration distance",
            "frozen_value": "0.4549 px",
            "evidence_role": "operational disagreement; not true error",
        },
        {
            "pilot_diagnostic": "Frozen positional tolerance",
            "frozen_value": "1.0 px",
            "evidence_role": "rounded safety rule; no optimization",
        },
    ]


def _figure_context() -> dict[str, object]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        "savefig.facecolor": "white",
    }


def _save_figure(figure: plt.Figure, png_path: Path, pdf_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        png_path,
        dpi=300,
        metadata={"Software": "TrackLEED"},
    )
    figure.savefig(
        pdf_path,
        metadata={
            "Creator": "TrackLEED",
            "Producer": "TrackLEED",
            "CreationDate": PDF_TIMESTAMP,
            "ModDate": PDF_TIMESTAMP,
        },
    )
    plt.close(figure)


def _render_pilot_figure(package: Path, rows: list[dict[str, str]]) -> None:
    with plt.rc_context(_figure_context()):
        figure, axis = plt.subplots(
            figsize=(88.0 / 25.4, 75.0 / 25.4),
            layout="constrained",
        )
        for branch, color in BRANCH_COLORS.items():
            branch_rows = [row for row in rows if row["branch_id"] == branch]
            axis.scatter(
                [float(row["energy_ev"]) for row in branch_rows],
                [float(row["cross_contract_distance_px"]) for row in branch_rows],
                color=color,
                edgecolor="white",
                linewidth=0.35,
                s=23,
                label=branch.replace("_", " "),
                zorder=3,
            )
        axis.axhline(1.0, color="#BB4444", linestyle="--", label="frozen 1.0 px")
        axis.axhline(
            0.4548687654337075,
            color="#666666",
            linestyle=":",
            label="observed max 0.455 px",
        )
        axis.set_title("Restricted pilot calibration subset (n=22)")
        axis.set_xlabel("Energy (eV)")
        axis.set_ylabel("Raw-native to Gaussian distance (px)")
        axis.set_ylim(0.0, 1.08)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.legend(loc="upper left", ncols=2, frameon=False, handletextpad=0.4)
        _save_figure(
            figure,
            package / "output/figure_s1_pilot_calibration.png",
            package / "output/pdf/figure_s1_pilot_calibration.pdf",
        )


def _render_sensitivity_figure(package: Path, rows: list[dict[str, str]]) -> None:
    with plt.rc_context(_figure_context()):
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(88.0 / 25.4, 100.0 / 25.4),
            sharex=True,
            layout="constrained",
        )
        for method in METHOD_ORDER:
            method_rows = [row for row in rows if row["method"] == method]
            tolerances = [float(row["positional_tolerance_px"]) for row in method_rows]
            axes[0].plot(
                tolerances,
                [
                    int(row["cross_contract_target_inconsistency"])
                    for row in method_rows
                ],
                marker=METHOD_MARKERS[method],
                markersize=3.3,
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                markerfacecolor=(
                    "white" if method == "locotrack" else METHOD_COLORS[method]
                ),
                markeredgewidth=0.8,
                label=METHOD_LABELS[method],
            )
            axes[1].plot(
                tolerances,
                [int(row["branch_model_disagreement"]) for row in method_rows],
                marker=METHOD_MARKERS[method],
                markersize=3.3,
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                markerfacecolor=(
                    "white" if method == "locotrack" else METHOD_COLORS[method]
                ),
                markeredgewidth=0.8,
            )
        axes[0].set_title("Direct-event interpretation across frozen tolerances")
        axes[0].set_ylabel("Cross-contract\ntarget inconsistency")
        axes[1].set_ylabel("Branch-model\ndisagreement")
        axes[1].set_xlabel("Positional tolerance (px)")
        axes[1].set_xticks([0.5, 1.0, 2.0])
        for axis in axes:
            axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axes[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncols=3,
            frameon=False,
            columnspacing=0.8,
            handletextpad=0.35,
        )
        _save_figure(
            figure,
            package / "output/figure_s2_sensitivity.png",
            package / "output/pdf/figure_s2_sensitivity.pdf",
        )


def _generate_outputs(workspace_root: Path, package: Path) -> None:
    pilot_rows = _pilot_source_rows(workspace_root)
    sensitivity_rows = _sensitivity_source_rows(workspace_root)
    _write_csv(
        package / "source_data/figure_s1_pilot_calibration.csv",
        (
            "energy_ev",
            "branch_id",
            "native_significance",
            "cross_contract_distance_px",
        ),
        pilot_rows,
    )
    _write_csv(
        package / "source_data/figure_s2_sensitivity.csv",
        (
            "method",
            "positional_tolerance_px",
            "accepted_out_of_branch_events",
            "branch_model_disagreement",
            "cross_contract_target_inconsistency",
            "unresolved_cross_contract_outcome",
        ),
        sensitivity_rows,
    )
    _write_csv(
        package / "source_data/table_s1_measurement_contracts.csv",
        (
            "measurement_path",
            "qualifying_endpoint",
            "denominator_and_retention",
        ),
        _table_s1_rows(),
    )
    _write_csv(
        package / "source_data/table_s2_pilot_validation.csv",
        ("pilot_diagnostic", "frozen_value", "evidence_role"),
        _table_s2_rows(),
    )
    _render_pilot_figure(package, pilot_rows)
    _render_sensitivity_figure(package, sensitivity_rows)


def _word_count(text: str) -> int:
    without_figures = re.sub(
        r"^!\[[^\]]*\]\([^\r\n]+\)\s*$", "", text, flags=re.MULTILINE
    )
    without_captions = re.sub(
        r"^\*\*(?:Figure|Table) S[1-2]\.[^\r\n]*$",
        "",
        without_figures,
        flags=re.MULTILINE,
    )
    without_tables = re.sub(r"^\s*\|.*\|\s*$", "", without_captions, flags=re.MULTILINE)
    without_headings = re.sub(
        r"^#{1,6}\s+[^\r\n]+$", "", without_tables, flags=re.MULTILINE
    )
    without_paths = re.sub(r"`[^`]+`", "", without_headings)
    return len(WORD_PATTERN.findall(without_paths))


def _analyze_supplement(package: Path, contract: dict[str, Any]) -> dict[str, object]:
    try:
        supplement = (package / "supplement.md").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CrossContractSupplementError(
            "cross-contract supplement source is unavailable"
        ) from exc
    if not supplement.startswith(f"# {contract['title']}\n"):
        raise CrossContractSupplementError("cross-contract supplement title changed")
    sections = [
        match.group(2)
        for match in re.finditer(
            r"^## S([1-5])\. ([^\r\n]+)$", supplement, flags=re.MULTILINE
        )
    ]
    if sections != contract["required_sections"]:
        raise CrossContractSupplementError(
            "cross-contract supplement section structure changed"
        )
    prose_word_count = _word_count(supplement)
    if not 1800 <= prose_word_count <= 2200:
        raise CrossContractSupplementError(
            "cross-contract supplement prose target changed: "
            f"{prose_word_count} words, expected 1800-2200"
        )
    displays = contract["display_inventory"]
    figure_directives = re.findall(
        r"^!\[(Figure S[1-2])\]\(([^\r\n]+)\)$",
        supplement,
        flags=re.MULTILINE,
    )
    expected_figures = [
        (str(item["id"]), str(item["path"])) for item in displays["figures"]
    ]
    if figure_directives != expected_figures:
        raise CrossContractSupplementError(
            "cross-contract supplement figure inventory changed"
        )
    table_separators = re.findall(
        r"^\|(?:\s*---\s*\|){3}$", supplement, flags=re.MULTILINE
    )
    if len(table_separators) != 2:
        raise CrossContractSupplementError(
            "cross-contract supplement table inventory changed"
        )
    supporting_items: list[str] = []
    for item in [*displays["figures"], *displays["tables"]]:
        item_id = str(item["id"])
        if not item.get("supporting_validation"):
            raise CrossContractSupplementError(
                f"supplement display is not supporting validation: {item_id}"
            )
        if supplement.count(f"**{item_id}. Supporting validation.**") != 1:
            raise CrossContractSupplementError(
                f"supplement display label changed: {item_id}"
            )
        supporting_items.append(item_id)
    supporting_items.sort(key=lambda value: (value.split()[0], value))
    expected_items = ["Figure S1", "Figure S2", "Table S1", "Table S2"]
    if supporting_items != expected_items:
        raise CrossContractSupplementError(
            "cross-contract supporting-validation inventory changed"
        )
    semantic = re.sub(r"\s+", " ", re.sub(r"[*_`]", "", supplement))
    required_terms = (
        "Gaussian-contract measurability",
        "ViPEr-native direct detectability",
        "Gaussian-contract gap",
        "ViPEr-native visibility gap",
        "I(V) visibility gap",
        "Cross-contract target inconsistency",
        "Unresolved cross-contract outcome",
        "calibration-only",
        "not an independent experimental replicate",
        "not an absolute accuracy reference",
        "conditional surviving set",
        "Authorized deferred publication inputs",
        "Page-level render acceptance",
        "submission-ready",
    )
    for term in required_terms:
        if term not in semantic:
            raise CrossContractSupplementError(
                f"required cross-contract supplement term is missing: {term}"
            )
    noncanonical_terms = (
        r"\btarget-inconsistenc(?:y|ies)\b",
        r"(?<!Cross-contract )\btarget inconsistenc(?:y|ies)\b",
        r"\bunresolved (?:event|events|class|category)\b",
    )
    for pattern in noncanonical_terms:
        if re.search(pattern, semantic, flags=re.IGNORECASE):
            raise CrossContractSupplementError(
                "cross-contract supplement uses noncanonical outcome terminology"
            )
    required_anchors = (
        "324 fixed",
        "305 direct and 19 non-direct",
        "221 frozen",
        "206-272 accepted",
        "124 accepted",
        "45 method-variant rows",
        "60 branch-frame records",
        "59 ViPEr-native directly detectable",
        "38 records",
        "22 predeclared high-signal records",
        "0.4549 px",
        "1.0 px",
        "2.0-3.0 range",
    )
    for anchor in required_anchors:
        if anchor not in semantic:
            raise CrossContractSupplementError(
                f"required cross-contract supplement anchor is missing: {anchor}"
            )
    prohibited_claims = (
        "the pilot independently confirms",
        "the pilot proves",
        "ViPErLEED is ground truth",
        "absolute tracking accuracy was established",
        "the best tracker",
    )
    normalized = semantic.casefold()
    for claim in prohibited_claims:
        if claim.casefold() in normalized:
            raise CrossContractSupplementError(
                f"prohibited cross-contract supplement claim remains: {claim}"
            )
    if re.search(r"\b[0-9a-f]{64}\b", supplement, flags=re.IGNORECASE):
        raise CrossContractSupplementError(
            "reader-facing supplement contains a raw hash inventory"
        )
    if "TODO[" in supplement:
        raise CrossContractSupplementError(
            "cross-contract supplement contains an unauthorized placeholder"
        )
    fixed_denominators = contract.get("fixed_denominators")
    if fixed_denominators != {
        "accepted_out_of_branch_events": 124,
        "opportunities_per_method": 324,
        "pilot_branch_frame_records": 60,
        "reference_visible_per_method": 221,
        "sensitivity_method_variant_rows": 45,
    }:
        raise CrossContractSupplementError(
            "cross-contract supplement fixed denominators changed"
        )
    return {
        "figure_count": len(figure_directives),
        "fixed_denominators": fixed_denominators,
        "prose_word_count": prose_word_count,
        "repository_only_full_records": True,
        "sections": sections,
        "supporting_validation_items": expected_items,
        "table_count": len(table_separators),
    }


def _validate_displays(package: Path, contract: dict[str, Any]) -> dict[str, object]:
    reports: dict[str, object] = {}
    for item in contract["display_inventory"]["figures"]:
        item_id = str(item["id"])
        size_mm = [float(value) for value in item["size_mm"]]
        png_path = package / str(item["path"])
        pdf_path = package / str(item["pdf_path"])
        expected_pixels = tuple(int(value / 25.4 * 300) for value in size_mm)
        try:
            with Image.open(png_path) as image:
                actual_pixels = image.size
        except (OSError, ValueError) as exc:
            raise CrossContractSupplementError(
                f"invalid cross-contract supplement figure: {item_id}"
            ) from exc
        if actual_pixels != expected_pixels:
            raise CrossContractSupplementError(
                "cross-contract supplement figure dimensions changed: "
                f"{item_id} actual={actual_pixels} expected={expected_pixels}"
            )
        try:
            page = PdfReader(pdf_path).pages[0]
            pdf_size = [float(page.mediabox.width), float(page.mediabox.height)]
        except (OSError, ValueError, IndexError) as exc:
            raise CrossContractSupplementError(
                f"invalid cross-contract supplement PDF: {item_id}"
            ) from exc
        expected_pdf = [value / 25.4 * 72.0 for value in size_mm]
        if any(
            abs(actual - expected) > 0.15
            for actual, expected in zip(pdf_size, expected_pdf)
        ):
            raise CrossContractSupplementError(
                f"cross-contract supplement PDF dimensions changed: {item_id}"
            )
        reports[item_id] = {
            "minimum_font_size_pt": contract["display_inventory"][
                "minimum_font_size_pt"
            ],
            "pixel_size_300_dpi": list(actual_pixels),
            "size_mm": size_mm,
        }
    return reports


def _build_manifest(package: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for relative, role in sorted(OUTPUT_ROLES.items()):
        path = package / relative
        if not path.is_file():
            raise CrossContractSupplementError(
                f"cross-contract supplement manifest input is unavailable: {relative}"
            )
        files.append(
            {
                "path": relative,
                "role": role,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "file_count": len(files),
        "files": files,
        "schema_version": "1.0",
        "status": "frozen-cross-contract-supplement-manifest",
        "supplement_id": "constrained-cross-contract-supplement",
    }


def _validation_report(
    package: Path,
    contract: dict[str, Any],
) -> dict[str, object]:
    return {
        **_analyze_supplement(package, contract),
        "display_validation": _validate_displays(package, contract),
        "fail_closed": True,
        "manifest_sha256": _sha256(package / "manifest.json"),
        "schema_version": "1.0",
        "status": "validated-cross-contract-supplement",
    }


def _package_file(package: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise CrossContractSupplementError("manifested supplement path changed")
    path = (package / value).resolve()
    if not path.is_relative_to(package) or not path.is_file():
        raise CrossContractSupplementError(
            f"manifested cross-contract supplement file is unavailable: {value}"
        )
    return path


def _validate_frozen_package(
    workspace_root: Path,
    package: Path,
) -> dict[str, object]:
    contract = _load_contract(workspace_root, package)
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        raise CrossContractSupplementError(
            f"cross-contract supplement manifest is unavailable: {package}"
        )
    if EXPECTED_MANIFEST_SHA256 and _sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise CrossContractSupplementError("cross-contract supplement manifest changed")
    manifest = _load_json(manifest_path, "cross-contract supplement manifest")
    manifested_files = manifest.get("files")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("supplement_id") != "constrained-cross-contract-supplement"
        or manifest.get("status") != "frozen-cross-contract-supplement-manifest"
        or not isinstance(manifested_files, list)
        or manifest.get("file_count") != len(manifested_files)
    ):
        raise CrossContractSupplementError(
            "cross-contract supplement manifest identity changed"
        )
    manifested: set[str] = set()
    for item in manifested_files:
        if not isinstance(item, dict):
            raise CrossContractSupplementError(
                "cross-contract supplement manifest record changed"
            )
        relative = item.get("path")
        path = _package_file(package, relative)
        if not isinstance(relative, str) or relative in manifested:
            raise CrossContractSupplementError(
                "duplicate cross-contract supplement manifest path"
            )
        if (
            item.get("role") != OUTPUT_ROLES.get(relative)
            or item.get("sha256") != _sha256(path)
            or item.get("size_bytes") != path.stat().st_size
        ):
            raise CrossContractSupplementError(
                f"manifested cross-contract supplement drift: {relative}"
            )
        manifested.add(relative)
    if manifested != set(OUTPUT_ROLES):
        raise CrossContractSupplementError(
            "cross-contract supplement manifest coverage changed"
        )
    actual = {path.relative_to(package).as_posix() for path in package.rglob("*")}
    expected_files = set(OUTPUT_ROLES) | {"manifest.json", "validation_report.json"}
    expected_entries = set(expected_files)
    for relative in expected_files:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_entries.add(parent.as_posix())
            parent = parent.parent
    if actual != expected_entries:
        raise CrossContractSupplementError(
            "cross-contract supplement package file inventory changed"
        )
    if manifest_path.read_bytes() != _json_bytes(_build_manifest(package)):
        raise CrossContractSupplementError(
            "cross-contract supplement manifest no longer reproduces"
        )
    report = _validation_report(package, contract)
    report_path = package / "validation_report.json"
    if not report_path.is_file() or report_path.read_bytes() != _json_bytes(report):
        raise CrossContractSupplementError(
            "cross-contract supplement validation report changed"
        )
    with TemporaryDirectory(
        prefix="cross-contract-supplement-validation-"
    ) as temporary:
        reproduction = Path(temporary) / "supplement"
        reproduction.mkdir()
        for relative in STATIC_PATHS:
            shutil.copyfile(package / relative, reproduction / relative)
        _generate_outputs(workspace_root, reproduction)
        for relative in DERIVED_ROLES:
            if (reproduction / relative).read_bytes() != (
                package / relative
            ).read_bytes():
                raise CrossContractSupplementError(
                    f"cross-contract supplement output no longer reproduces: {relative}"
                )
    return report


def freeze_cross_contract_supplement(
    workspace_root: Path,
    package_path: Path | None = None,
) -> dict[str, object]:
    """Build, manifest, validate, and publish the supplement package once."""

    workspace_root = workspace_root.resolve()
    package = (
        package_path.resolve()
        if package_path is not None
        else workspace_root / DEFAULT_PACKAGE_PATH
    )
    _load_contract(workspace_root, package)
    if any((package / relative).exists() for relative in GENERATED_PATHS):
        raise FileExistsError(
            f"Refusing to replace frozen cross-contract supplement: {package}"
        )
    for relative in STATIC_PATHS:
        if not (package / relative).is_file():
            raise CrossContractSupplementError(
                f"cross-contract supplement source is unavailable: {relative}"
            )
    with TemporaryDirectory(
        prefix="cross-contract-supplement-", dir=package.parent
    ) as temporary:
        stage = Path(temporary) / "supplement"
        stage.mkdir()
        for relative in STATIC_PATHS:
            shutil.copyfile(package / relative, stage / relative)
        contract = _load_contract(workspace_root, stage)
        _generate_outputs(workspace_root, stage)
        (stage / "manifest.json").write_bytes(_json_bytes(_build_manifest(stage)))
        (stage / "validation_report.json").write_bytes(
            _json_bytes(_validation_report(stage, contract))
        )
        validated = _validate_frozen_package(workspace_root, stage)
        if any(
            _sha256(package / relative) != _sha256(stage / relative)
            for relative in STATIC_PATHS
        ):
            raise CrossContractSupplementError(
                "cross-contract supplement sources changed during staged validation"
            )
        for relative in GENERATED_PATHS:
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / relative, destination)
    return validated


def validate_cross_contract_supplement(
    workspace_root: Path,
    package_path: Path | None = None,
) -> dict[str, object]:
    """Validate the frozen constrained Supplementary Information package."""

    workspace_root = workspace_root.resolve()
    package = (
        package_path.resolve()
        if package_path is not None
        else workspace_root / DEFAULT_PACKAGE_PATH
    )
    if not (package / "contract.json").is_file():
        raise CrossContractSupplementError(
            "cross-contract supplement contract is unavailable"
        )
    return _validate_frozen_package(workspace_root, package)


__all__ = [
    "CrossContractSupplementError",
    "freeze_cross_contract_supplement",
    "validate_cross_contract_supplement",
]
