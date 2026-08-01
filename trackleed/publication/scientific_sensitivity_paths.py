"""Canonical filesystem layout for publication sensitivity evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SENSITIVITY_SOURCE_RELATIVE_PATH = Path(
    "manuscript/source_data/measurement_sensitivity"
)


def sensitivity_source_root(workspace_root: Path) -> Path:
    """Return the canonical frozen sensitivity source-package directory."""

    return workspace_root.resolve() / SENSITIVITY_SOURCE_RELATIVE_PATH


@dataclass(frozen=True)
class SensitivityPublicationPaths:
    """Named locations consumed by the sensitivity publication bundle."""

    source_root: Path
    table_root: Path
    figure_root: Path
    adjudication_root: Path
    supplementary_material: Path
    adjudication_contract: Path

    @classmethod
    def for_workspace(cls, workspace_root: Path) -> SensitivityPublicationPaths:
        workspace_root = workspace_root.resolve()
        manuscript_root = workspace_root / "manuscript"
        adjudication_root = manuscript_root / "claim_adjudication"
        return cls(
            source_root=sensitivity_source_root(workspace_root),
            table_root=(
                manuscript_root / "tables" / "table_s6_measurement_sensitivity"
            ),
            figure_root=(
                manuscript_root
                / "figures"
                / "figure_s2_measurement_sensitivity"
            ),
            adjudication_root=adjudication_root,
            supplementary_material=manuscript_root / "supplementary-material.md",
            adjudication_contract=(
                adjudication_root / "adjudication_contract.json"
            ),
        )

    def resolved(self) -> SensitivityPublicationPaths:
        """Return an equivalent layout with every path made absolute."""

        return SensitivityPublicationPaths(
            source_root=self.source_root.resolve(),
            table_root=self.table_root.resolve(),
            figure_root=self.figure_root.resolve(),
            adjudication_root=self.adjudication_root.resolve(),
            supplementary_material=self.supplementary_material.resolve(),
            adjudication_contract=self.adjudication_contract.resolve(),
        )


__all__ = [
    "SENSITIVITY_SOURCE_RELATIVE_PATH",
    "SensitivityPublicationPaths",
    "sensitivity_source_root",
]
