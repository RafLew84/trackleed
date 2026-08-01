"""Build the sensitivity- and metadata-constrained claim boundary."""

import argparse
from pathlib import Path

from trackleed.publication.scientific_claim_adjudication import (
    build_claim_adjudication,
    confirm_claim_adjudication,
)
from trackleed.publication.scientific_sensitivity_paths import (
    sensitivity_source_root,
)


ARTIFACT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ARTIFACT_ROOT.parents[1]
SOURCE_DATA_ROOT = sensitivity_source_root(REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--confirmed-by")
    parser.add_argument("--confirmed-date")
    arguments = parser.parse_args()
    build_claim_adjudication(
        workspace_root=REPO_ROOT,
        contract_path=ARTIFACT_ROOT / "adjudication_contract.json",
        sensitivity_source_directory=SOURCE_DATA_ROOT,
        metadata_directory=REPO_ROOT / "manuscript" / "experimental_metadata",
        claim_map_path=REPO_ROOT / "manuscript" / "claim-evidence-map.md",
        output_directory=ARTIFACT_ROOT / "output",
    )
    if arguments.confirm:
        if not arguments.confirmed_by or not arguments.confirmed_date:
            parser.error("--confirm requires --confirmed-by and --confirmed-date")
        confirm_claim_adjudication(
            ARTIFACT_ROOT / "output",
            confirmed_by=arguments.confirmed_by,
            confirmed_date=arguments.confirmed_date,
        )


if __name__ == "__main__":
    main()
