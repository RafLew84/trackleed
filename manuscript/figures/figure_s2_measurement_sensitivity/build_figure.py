"""Build the frozen publication artifact for Figure S2."""

from pathlib import Path

from trackleed.publication.scientific_sensitivity_figure import (
    build_figure_s2_artifact,
)
from trackleed.publication.scientific_sensitivity_paths import (
    sensitivity_source_root,
)


ARTIFACT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ARTIFACT_ROOT.parents[2]
SOURCE_DATA_ROOT = sensitivity_source_root(REPO_ROOT)


if __name__ == "__main__":
    build_figure_s2_artifact(
        REPO_ROOT,
        SOURCE_DATA_ROOT,
        ARTIFACT_ROOT / "output",
        ARTIFACT_ROOT / "manifest.json",
    )
