"""Thermal-cycle branch classification from stored frame metadata."""

from __future__ import annotations

import math
from typing import Mapping


HEATING = "heating"
COOLING = "cooling"


def classify_thermal_branches(
    frame_temperatures: Mapping[str, float],
    frame_orders: Mapping[str, int],
    stored_branches: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return heating/cooling assignments, treating the peak as heating."""

    if not frame_temperatures:
        raise ValueError("Cannot classify an empty thermal series")
    missing_orders = sorted(set(frame_temperatures).difference(frame_orders))
    if missing_orders:
        raise ValueError(
            "Missing frame order for thermal frames: " + ", ".join(missing_orders)
        )

    normalized_stored = {
        frame_id: normalize_thermal_branch(branch)
        for frame_id, branch in (stored_branches or {}).items()
    }
    unknown_frames = sorted(
        set(normalized_stored).difference(frame_temperatures)
    )
    if unknown_frames:
        raise ValueError(
            "Stored thermal branch refers to unknown frames: "
            + ", ".join(unknown_frames)
        )

    max_temperature = max(frame_temperatures.values())
    last_peak_order = max(
        frame_orders[frame_id]
        for frame_id, temperature in frame_temperatures.items()
        if math.isclose(temperature, max_temperature)
    )
    return {
        frame_id: normalized_stored.get(
            frame_id,
            HEATING if frame_orders[frame_id] <= last_peak_order else COOLING,
        )
        for frame_id in frame_temperatures
    }


def normalize_thermal_branch(value: str) -> str:
    """Normalize stored peak/hold labels to the two publication branches."""

    normalized = value.strip().lower().replace("-", "_").replace("/", "_")
    if normalized in {HEATING, "peak_hold", "hold", "heating_hold"}:
        return HEATING
    if normalized == COOLING:
        return COOLING
    raise ValueError(f"Unsupported thermal branch: {value}")
