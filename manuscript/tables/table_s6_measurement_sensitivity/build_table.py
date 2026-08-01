"""Build the frozen publication artifact for Table S6."""

from pathlib import Path

from trackleed.publication.scientific_sensitivity_bundle import (
    SENSITIVITY_SOURCE_RELATIVE_PATH,
)
from trackleed.publication.scientific_sensitivity_evidence import (
    build_table_s6_artifact,
    synchronize_table_s6,
)


ARTIFACT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ARTIFACT_ROOT.parents[2]
SOURCE_DATA_ROOT = REPO_ROOT / SENSITIVITY_SOURCE_RELATIVE_PATH


if __name__ == "__main__":
    build_table_s6_artifact(
        REPO_ROOT,
        SOURCE_DATA_ROOT,
        ARTIFACT_ROOT / "output",
        ARTIFACT_ROOT / "manifest.json",
    )
    synchronize_table_s6(
        ARTIFACT_ROOT / "output" / "table_s6_measurement_sensitivity.md",
        REPO_ROOT / "manuscript" / "supplementary-material.md",
    )
