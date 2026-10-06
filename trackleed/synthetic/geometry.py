"""Ideal reciprocal-lattice sites and their persistent identities."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LatticeSite:
    reflection_id: str
    h: int
    k: int
    x: float
    y: float
    parent_id: str | None = None
    role: str = "primary"


def lattice_sites(pattern: str, radius: float, satellite_offset: float) -> tuple[LatticeSite, ...]:
    """Return a deterministic selected lattice fragment, excluding the central beam."""
    order = 1 if pattern == "H6" else 2
    primary = tuple(
        LatticeSite(f"h{h:+d}_k{k:+d}", h, k,
                    radius * (h + k / 2), radius * math.sqrt(3) * k / 2)
        for h in range(-order, order + 1) for k in range(-order, order + 1)
        if 1 <= max(abs(h), abs(k), abs(h + k)) <= order
    )
    if pattern != "H54":
        return primary
    satellites = tuple(
        LatticeSite(f"{parent.reflection_id}_s{j}", parent.h, parent.k,
                    parent.x + satellite_offset * math.cos(j * math.pi / 3),
                    parent.y + satellite_offset * math.sin(j * math.pi / 3),
                    parent.reflection_id, "satellite")
        for parent in primary if max(abs(parent.h), abs(parent.k), abs(parent.h + parent.k)) == 1
        for j in range(6)
    )
    return primary + satellites
