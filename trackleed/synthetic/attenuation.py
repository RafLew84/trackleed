"""Declared per-reflection attenuation, independent of observation schedules."""
from dataclasses import dataclass
import math
import re
from typing import Any


@dataclass(frozen=True)
class Attenuation:
    id: str
    reflection_id: str
    axis: str
    knots: tuple[int | float, ...]
    minimum_fraction: float

    def coordinate(self, time_ns: int | None, energy_ev: float | None) -> int | float:
        value = time_ns if self.axis == "time_ns" else energy_ev
        if value is None:
            raise ValueError(f"attenuation {self.id} requires {self.axis}")
        return value

    def factor(self, coordinate: int | float) -> float:
        a, b, c, d = self.knots
        phase = self.phase(coordinate)
        if phase in ("before", "after"):
            return 1.0
        if phase == "minimum":
            return self.minimum_fraction
        fraction = ((b - coordinate) / (b - a) if phase == "fade"
                    else (coordinate - c) / (d - c))
        factor = self.minimum_fraction + (1 - self.minimum_fraction) * fraction
        if factor == 0:
            raise ValueError("positive attenuation ramp cannot be represented")
        return factor

    def phase(self, coordinate: int | float) -> str:
        a, b, c, d = self.knots
        if coordinate < a or (coordinate == a and a < b):
            return "before"
        if coordinate >= d:
            return "after"
        if coordinate < b:
            return "fade"
        return "minimum" if coordinate <= c else "return"


def parse_attenuations(raw: Any) -> tuple[Attenuation, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError("scene.attenuations must be an array")
    events = []
    names = {"id", "reflection_id", "axis", "knots", "minimum_fraction"}
    for value in raw:
        if not isinstance(value, dict) or set(value) != names:
            raise ValueError(f"attenuation requires exactly {sorted(names)}")
        if not isinstance(value["id"], str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value["id"]) is None:
            raise ValueError("attenuation.id must be a portable identifier")
        if not isinstance(value["reflection_id"], str):
            raise ValueError("attenuation.reflection_id must be a string")
        if value["axis"] not in ("time_ns", "energy_ev"):
            raise ValueError("attenuation.axis must be time_ns or energy_ev")
        knots = value["knots"]
        if not isinstance(knots, (list, tuple)) or len(knots) != 4:
            raise ValueError("attenuation.knots requires four coordinates")
        for point in knots:
            if value["axis"] == "time_ns":
                if type(point) is not int or not 0 <= point <= 2**63 - 1:
                    raise ValueError("attenuation time_ns knots require nonnegative int64")
            elif not _finite(point) or point <= 0:
                raise ValueError("attenuation energy_ev knots must be positive finite numbers")
        a, b, c, d = knots
        if not a <= b < c <= d:
            raise ValueError("attenuation knots require fade_start <= minimum_start < minimum_end <= return_end")
        minimum = value["minimum_fraction"]
        if not _finite(minimum) or not 0 <= minimum < 1:
            raise ValueError("attenuation.minimum_fraction must be in [0,1)")
        events.append(Attenuation(value["id"], value["reflection_id"], value["axis"], tuple(knots), minimum))
    if len({e.id for e in events}) != len(events):
        raise ValueError("attenuation.id must be unique")
    if len({e.reflection_id for e in events}) != len(events):
        raise ValueError("only one attenuation per reflection_id is supported")
    return tuple(events)


def _finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate_attenuations(events: tuple[Attenuation, ...], amplitudes: dict[str, float],
                          *, has_time: bool, has_energy: bool) -> None:
    for event in events:
        if event.reflection_id not in amplitudes:
            raise ValueError(f"Unknown attenuation reflection_id: {event.reflection_id}")
        base = amplitudes[event.reflection_id]
        if base <= 0:
            raise ValueError("attenuation requires a positive base amplitude")
        if event.minimum_fraction > 0 and base * event.minimum_fraction == 0:
            raise ValueError("attenuation positive minimum cannot be represented")
        if (event.axis == "time_ns" and not has_time) or (event.axis == "energy_ev" and not has_energy):
            raise ValueError(f"attenuation requires known {event.axis} observations")


def amplitude_at(events: tuple[Attenuation, ...], reflection_id: str, base: float,
                 time_ns: int | None, energy_ev: float | None) -> float:
    for event in events:
        if event.reflection_id == reflection_id:
            factor = event.factor(event.coordinate(time_ns, energy_ev))
            value = base * factor
            if factor > 0 and value == 0:
                raise ValueError("positive attenuated amplitude cannot be represented")
            return value
    return base
