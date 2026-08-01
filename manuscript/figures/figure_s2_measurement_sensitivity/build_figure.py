"""Build the frozen publication artifact for Figure S2."""

from pathlib import Path

from trackleed.publication.scientific_sensitivity_bundle import (
    SENSITIVITY_SOURCE_RELATIVE_PATH,
)
from trackleed.publication.scientific_sensitivity_figure import (
    build_figure_s2_artifact,
)


ARTIFACT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ARTIFACT_ROOT.parents[2]
SOURCE_DATA_ROOT = REPO_ROOT / SENSITIVITY_SOURCE_RELATIVE_PATH


if __name__ == "__main__":
    build_figure_s2_artifact(
        REPO_ROOT,
        SOURCE_DATA_ROOT,
        ARTIFACT_ROOT / "output",
        ARTIFACT_ROOT / "manifest.json",
    )
