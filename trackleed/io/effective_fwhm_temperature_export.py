"""Publication-ready scalar FWHM exports split by thermal branch."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from trackleed.analysis.fwhm_series import numeric_sort_key
from trackleed.analysis.thermal_branches import (
    classify_thermal_branches,
    normalize_thermal_branch,
)


DATA_FILENAME = "fwhm_vs_temperature_data.csv"
MANIFEST_FILENAME = "fwhm_vs_temperature_manifest.json"
OUTPUT_FIELDS = (
    "track_id",
    "frame_id",
    "frame_order",
    "temperature_C",
    "energy_eV",
    "coverage",
    "thermal_branch",
    "plotted",
    "quality_note",
    "missing_reason",
    "fwhm_minor_px",
    "fwhm_minor_error_px",
    "fwhm_major_px",
    "fwhm_major_error_px",
    "fwhm_px",
    "fwhm_error_px",
    "r_squared",
)
PLOT_FORMATS = ("png", "pdf", "svg")
PLOT_BRANCHES = ("heating", "cooling")
GEOMETRIC_RTOL = 1e-6


@dataclass(frozen=True)
class EffectiveFwhmTemperatureExportResult:
    """Files produced from one stored fit series."""

    output_dir: Path
    data_path: Path
    plot_paths: tuple[Path, ...]
    manifest_path: Path


def export_effective_fwhm_temperature(
    *,
    fit_csv: Path | str,
    series_csv: Path | str,
    output_dir: Path | str,
) -> EffectiveFwhmTemperatureExportResult:
    """Join canonical fit widths to plot metadata and export scalar FWHM.

    ``fit_csv`` is the canonical ``full_analysis_per_spot.csv``. ``series_csv``
    supplies the already curated frame order, branch and fit-quality status.
    The output deliberately exposes no legacy or detector-axis X/Y widths.
    """

    fit_path = Path(fit_csv)
    series_path = Path(series_csv)
    destination = Path(output_dir)
    fit_rows, fit_fields = _read_csv(fit_path)
    series_rows, series_fields = _read_csv(series_path)
    _require_fields(
        fit_path,
        fit_fields,
        {
            "track_id",
            "frame_id",
            "fwhm_minor",
            "fwhm_major",
            "fwhm_effective",
        },
    )
    _require_fields(
        series_path,
        series_fields,
        {"track_id", "frame_id", "temperature_C", "plotted"},
    )

    fits_by_key = _index_fit_rows(fit_rows)
    frame_orders = _frame_orders(series_rows)
    frame_branches = _frame_branches(series_rows, frame_orders)
    output_rows = [
        _build_output_row(
            series_row,
            fits_by_key.get(_row_key(series_row)),
            frame_order=frame_orders[str(series_row["frame_id"])],
            branch=frame_branches[str(series_row["frame_id"])],
        )
        for series_row in series_rows
    ]

    present_branches = tuple(
        branch
        for branch in PLOT_BRANCHES
        if any(row["thermal_branch"] == branch for row in output_rows)
    )
    expected_paths = [destination / DATA_FILENAME]
    for branch in present_branches:
        expected_paths.extend(
            destination / f"fwhm_vs_temperature_{branch}.{extension}"
            for extension in PLOT_FORMATS
        )
    expected_paths.append(destination / MANIFEST_FILENAME)
    existing_paths = [path for path in expected_paths if path.exists()]
    if existing_paths:
        raise FileExistsError(
            "Refusing to overwrite existing export files: "
            + ", ".join(str(path) for path in existing_paths)
        )

    destination.mkdir(parents=True, exist_ok=True)
    data_path = destination / DATA_FILENAME
    _write_csv(data_path, output_rows)

    plot_paths: list[Path] = []
    for branch in present_branches:
        plot_paths.extend(_write_branch_plot(destination, output_rows, branch))

    manifest_path = destination / MANIFEST_FILENAME
    manifest = {
        "version": 1,
        "fit_source_csv": str(fit_path.resolve()),
        "fit_source_sha256": _sha256(fit_path),
        "curation_source_role": (
            "stored frame order, thermal branch, and fit-quality status"
        ),
        "curation_source_sha256": _sha256(series_path),
        "fwhm_definition": "sqrt(fwhm_minor * fwhm_major)",
        "plot_y_axis": "FWHM (px)",
        "output_columns": list(OUTPUT_FIELDS),
        "row_count": len(output_rows),
        "track_count": len({row["track_id"] for row in output_rows}),
        "branches": dict(
            sorted(Counter(row["thermal_branch"] for row in output_rows).items())
        ),
        "plotted_rows": sum(_is_true(row["plotted"]) for row in output_rows),
        "plots": [path.name for path in plot_paths],
        "data_csv": data_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return EffectiveFwhmTemperatureExportResult(
        output_dir=destination,
        data_path=data_path,
        plot_paths=tuple(plot_paths),
        manifest_path=manifest_path,
    )


def _read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows, fields


def _require_fields(path: Path, fields: Iterable[str], required: set[str]) -> None:
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _index_fit_rows(
    rows: Iterable[Mapping[str, str]],
) -> dict[tuple[str, str], Mapping[str, str]]:
    indexed: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        key = _row_key(row)
        if key in indexed:
            raise ValueError(
                f"Duplicate canonical fit for track {key[0]}, frame {key[1]}"
            )
        if _validated_widths(row, key) is None:
            continue
        indexed[key] = row
    return indexed


def _validated_widths(
    row: Mapping[str, str],
    key: tuple[str, str],
) -> tuple[float, float, float] | None:
    minor = _finite_float(row.get("fwhm_minor"))
    major = _finite_float(row.get("fwhm_major"))
    effective = _finite_float(row.get("fwhm_effective"))
    if minor is None and major is None and effective is None:
        return None
    if minor is None or major is None or effective is None:
        raise ValueError(
            f"Incomplete canonical FWHM for track {key[0]}, frame {key[1]}"
        )
    if minor <= 0 or major <= 0 or effective <= 0:
        raise ValueError(
            f"Non-positive canonical FWHM for track {key[0]}, frame {key[1]}"
        )
    geometric = math.sqrt(minor * major)
    if not math.isclose(effective, geometric, rel_tol=GEOMETRIC_RTOL):
        raise ValueError(
            "Stored effective FWHM is inconsistent with the geometric mean "
            f"for track {key[0]}, frame {key[1]}: "
            f"stored={effective}, geometric={geometric}"
        )
    return minor, major, effective


def _frame_orders(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    orders: dict[str, int] = {}
    for row in rows:
        frame_id = str(row["frame_id"])
        stored_order = _finite_float(row.get("frame_order"))
        if stored_order is None and frame_id in orders:
            continue
        order = int(stored_order) if stored_order is not None else len(orders)
        previous = orders.setdefault(frame_id, order)
        if previous != order:
            raise ValueError(f"Inconsistent frame order for frame {frame_id}")
    return orders


def _frame_branches(
    rows: Iterable[Mapping[str, str]],
    frame_orders: Mapping[str, int],
) -> dict[str, str]:
    frame_temperatures: dict[str, float] = {}
    stored_branches: dict[str, str] = {}
    for row in rows:
        frame_id = str(row["frame_id"])
        temperature = _finite_float(row.get("temperature_C"))
        if temperature is None:
            raise ValueError(f"Invalid temperature for frame {frame_id}")
        previous_temperature = frame_temperatures.setdefault(frame_id, temperature)
        if not math.isclose(previous_temperature, temperature):
            raise ValueError(f"Inconsistent temperature for frame {frame_id}")
        stored_branch = str(row.get("thermal_branch", "")).strip()
        branch = normalize_thermal_branch(stored_branch) if stored_branch else ""
        if branch:
            previous_branch = stored_branches.setdefault(frame_id, branch)
            if previous_branch != branch:
                raise ValueError(f"Inconsistent thermal branch for frame {frame_id}")

    return classify_thermal_branches(
        frame_temperatures,
        frame_orders,
        stored_branches,
    )


def _build_output_row(
    series_row: Mapping[str, str],
    fit_row: Mapping[str, str] | None,
    *,
    frame_order: int,
    branch: str,
) -> dict[str, str]:
    widths = (None, None, None)
    if fit_row is not None:
        validated_widths = _validated_widths(fit_row, _row_key(series_row))
        if validated_widths is not None:
            widths = validated_widths
    minor, major, effective = widths
    plotted = _is_true(series_row.get("plotted")) and fit_row is not None
    return {
        "track_id": str(series_row.get("track_id", "")),
        "frame_id": str(series_row.get("frame_id", "")),
        "frame_order": str(frame_order),
        "temperature_C": str(series_row.get("temperature_C", "")),
        "energy_eV": str(series_row.get("energy_eV", "")),
        "coverage": str(series_row.get("coverage", "")),
        "thermal_branch": branch,
        "plotted": "True" if plotted else "False",
        "quality_note": str(series_row.get("quality_note", "")),
        "missing_reason": str(series_row.get("missing_reason", "")),
        "fwhm_minor_px": _format_number(minor),
        "fwhm_minor_error_px": _fit_value(fit_row, "fwhm_minor_error"),
        "fwhm_major_px": _format_number(major),
        "fwhm_major_error_px": _fit_value(fit_row, "fwhm_major_error"),
        "fwhm_px": _format_number(effective),
        "fwhm_error_px": _fit_value(fit_row, "fwhm_effective_error"),
        "r_squared": _fit_value(fit_row, "r_squared")
        or str(series_row.get("r_squared", "")),
    }


def _fit_value(row: Mapping[str, str] | None, column: str) -> str:
    if row is None:
        return ""
    return _format_number(_finite_float(row.get(column)))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_branch_plot(
    destination: Path,
    rows: list[dict[str, str]],
    branch: str,
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    branch_rows = [row for row in rows if row["thermal_branch"] == branch]
    rows_by_track: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in branch_rows:
        rows_by_track[row["track_id"]].append(row)
    track_ids = sorted(rows_by_track, key=numeric_sort_key)
    column_count = min(3, len(track_ids))
    row_count = math.ceil(len(track_ids) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.6 * column_count, 4.3 * row_count),
        squeeze=False,
    )
    color = "#0072B2" if branch == "heating" else "#D55E00"
    for axis, track_id in zip(axes.flat, track_ids):
        track_rows = sorted(
            rows_by_track[track_id],
            key=lambda row: int(row["frame_order"]),
        )
        temperatures = [float(row["temperature_C"]) for row in track_rows]
        widths = [
            float(row["fwhm_px"])
            if _is_true(row["plotted"]) and row["fwhm_px"]
            else math.nan
            for row in track_rows
        ]
        axis.plot(temperatures, widths, color=color, linewidth=1.6)
        axis.set_title(f"Track {track_id}")
        axis.grid(True, alpha=0.25)
    for axis in list(axes.flat)[len(track_ids) :]:
        axis.set_visible(False)
    figure.supxlabel("Temperature (\N{DEGREE SIGN}C)")
    figure.supylabel("FWHM (px)")
    figure.tight_layout(rect=(0.04, 0.04, 1.0, 1.0))

    paths = tuple(
        destination / f"fwhm_vs_temperature_{branch}.{extension}"
        for extension in PLOT_FORMATS
    )
    for path in paths:
        save_kwargs = {"bbox_inches": "tight"}
        if path.suffix == ".png":
            save_kwargs["dpi"] = 300
        figure.savefig(path, **save_kwargs)
    plt.close(figure)
    return paths


def _row_key(row: Mapping[str, str]) -> tuple[str, str]:
    return str(row.get("track_id", "")), str(row.get("frame_id", ""))


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_number(value: float | None) -> str:
    return "" if value is None else format(value, ".10g")


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
