"""Build the frozen publication artifact for Figure S2."""

from pathlib import Path

from trackleed.publication.scientific_sensitivity_figure import (
    build_figure_s2_artifact,
)


ARTIFACT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ARTIFACT_ROOT.parents[2]


if __name__ == "__main__":
    build_figure_s2_artifact(
        REPO_ROOT,
        ARTIFACT_ROOT / "source_data",
        ARTIFACT_ROOT / "output",
        ARTIFACT_ROOT / "manifest.json",
    )
