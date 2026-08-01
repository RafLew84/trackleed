"""Publication bundle for sensitivity and scientific claim-boundary adjudication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from trackleed.publication.scientific_claim_adjudication import (
    ClaimAdjudicationReport,
    build_claim_adjudication,
    validate_author_checkpoint,
)
from trackleed.publication.scientific_sensitivity_evidence import (
    TABLE_S6_BEGIN,
    TABLE_S6_END,
    TableS6Report,
    build_table_s6_artifact,
    synchronize_table_s6,
)
from trackleed.publication.scientific_sensitivity_figure import (
    FigureS2Report,
    build_figure_s2_artifact,
)
from trackleed.publication.scientific_sensitivity_paths import (
    SensitivityPublicationPaths,
)
from trackleed.publication.scientific_sensitivity_sources import (
    freeze_sensitivity_source_data,
    read_json,
    sha256_file,
    validate_sensitivity_source_data,
)


@dataclass(frozen=True)
class SensitivityPublicationBundleReport:
    table_s6: TableS6Report
    figure_s2: FigureS2Report
    claim_adjudication: ClaimAdjudicationReport


def _require_matching_file(expected: Path, actual: Path) -> None:
    if not actual.is_file() or expected.read_bytes() != actual.read_bytes():
        raise ValueError(f"Publication sensitivity artifact changed: {actual}")


def _validate_claim_manifest(
    workspace_root: Path,
    adjudication_root: Path,
    checkpoint_status: str,
) -> None:
    output_directory = adjudication_root / "output"
    manifest = read_json(output_directory / "manifest.json")
    output_hashes = manifest.get("outputs_sha256")
    implementation_hashes = manifest.get("implementation_sha256")
    expected_status = (
        "completed" if checkpoint_status == "confirmed" else "pending_author_checkpoint"
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != expected_status
        or manifest.get("author_checkpoint_status") != checkpoint_status
        or manifest.get("contract_sha256")
        != sha256_file(adjudication_root / "adjudication_contract.json")
        or not isinstance(output_hashes, dict)
        or set(output_hashes)
        != {
            "author_checkpoint.json",
            "final_claim_boundary.json",
            "sensitivity_claim_evidence.json",
        }
        or not isinstance(implementation_hashes, dict)
    ):
        raise ValueError("Claim-adjudication manifest changed")
    for name, digest in output_hashes.items():
        if sha256_file(output_directory / name) != digest:
            raise ValueError(f"Claim-adjudication output changed: {name}")
    for relative_path, digest in implementation_hashes.items():
        if sha256_file(workspace_root / relative_path) != digest:
            raise ValueError(
                f"Claim-adjudication implementation changed: {relative_path}"
            )


def build_sensitivity_publication_bundle(
    *,
    workspace_root: Path,
    sensitivity_output_directory: Path,
    paths: SensitivityPublicationPaths | None = None,
) -> SensitivityPublicationBundleReport:
    """Freeze validated results and build their only publication displays."""

    workspace_root = workspace_root.resolve()
    manuscript_root = workspace_root / "manuscript"
    paths = (
        paths or SensitivityPublicationPaths.for_workspace(workspace_root)
    ).resolved()

    freeze_sensitivity_source_data(
        sensitivity_output_directory,
        paths.source_root,
        workspace_root=workspace_root,
    )

    table_report = build_table_s6_artifact(
        workspace_root,
        paths.source_root,
        paths.table_root / "output",
        paths.table_root / "manifest.json",
    )
    figure_report = build_figure_s2_artifact(
        workspace_root,
        paths.source_root,
        paths.figure_root / "output",
        paths.figure_root / "manifest.json",
    )
    claim_report = build_claim_adjudication(
        workspace_root=workspace_root,
        contract_path=paths.adjudication_contract,
        sensitivity_source_directory=paths.source_root,
        metadata_directory=manuscript_root / "experimental_metadata",
        claim_map_path=manuscript_root / "claim-evidence-map.md",
        output_directory=paths.adjudication_root / "output",
    )
    synchronize_table_s6(
        paths.table_root / "output" / "table_s6_measurement_sensitivity.md",
        paths.supplementary_material,
    )
    return SensitivityPublicationBundleReport(
        table_s6=table_report,
        figure_s2=figure_report,
        claim_adjudication=claim_report,
    )


def validate_sensitivity_publication_bundle(
    *,
    workspace_root: Path,
    sensitivity_output_directory: Path | None = None,
) -> SensitivityPublicationBundleReport:
    """Validate committed reader-facing artifacts without rewriting them."""

    workspace_root = workspace_root.resolve()
    paths = SensitivityPublicationPaths.for_workspace(workspace_root)
    origin_directory = (
        sensitivity_output_directory
        or workspace_root
        / "manuscript"
        / "scientific_panel"
        / "output"
        / "sensitivity"
    )
    validate_sensitivity_source_data(
        paths.source_root,
        origin_directory=origin_directory,
    )
    with TemporaryDirectory() as temporary:
        generated_root = Path(temporary)
        table_report = build_table_s6_artifact(
            workspace_root,
            paths.source_root,
            generated_root / "table-output",
            generated_root / "table-manifest.json",
        )
        figure_report = build_figure_s2_artifact(
            workspace_root,
            paths.source_root,
            generated_root / "figure-output",
            generated_root / "figure-manifest.json",
        )
        claim_report = build_claim_adjudication(
            workspace_root=workspace_root,
            contract_path=paths.adjudication_contract,
            sensitivity_source_directory=paths.source_root,
            metadata_directory=workspace_root / "manuscript" / "experimental_metadata",
            claim_map_path=workspace_root / "manuscript" / "claim-evidence-map.md",
            output_directory=generated_root / "claim-output",
        )
        _require_matching_file(
            generated_root / "table-manifest.json",
            paths.table_root / "manifest.json",
        )
        _require_matching_file(
            generated_root / "figure-manifest.json",
            paths.figure_root / "manifest.json",
        )
        for generated, committed in (
            (generated_root / "table-output", paths.table_root / "output"),
            (generated_root / "figure-output", paths.figure_root / "output"),
        ):
            generated_files = {path.name: path for path in generated.iterdir()}
            committed_files = {path.name: path for path in committed.iterdir()}
            if set(generated_files) != set(committed_files):
                raise ValueError(f"Publication artifact inventory changed: {committed}")
            for name, expected in generated_files.items():
                _require_matching_file(expected, committed_files[name])
        for name in (
            "sensitivity_claim_evidence.json",
            "final_claim_boundary.json",
        ):
            _require_matching_file(
                generated_root / "claim-output" / name,
                paths.adjudication_root / "output" / name,
            )

    boundary_path = paths.adjudication_root / "output" / "final_claim_boundary.json"
    checkpoint = read_json(
        paths.adjudication_root / "output" / "author_checkpoint.json"
    )
    validate_author_checkpoint(checkpoint, sha256_file(boundary_path))
    checkpoint_status = str(checkpoint["status"])
    _validate_claim_manifest(
        workspace_root,
        paths.adjudication_root,
        checkpoint_status,
    )
    table_markdown = (
        paths.table_root / "output" / "table_s6_measurement_sensitivity.md"
    ).read_text(encoding="utf-8").rstrip()
    supplement = paths.supplementary_material.read_text(encoding="utf-8")
    expected_block = (
        f"{TABLE_S6_BEGIN}\n{table_markdown}\n{TABLE_S6_END}"
    )
    if (
        supplement.count(TABLE_S6_BEGIN) != 1
        or supplement.count(TABLE_S6_END) != 1
        or expected_block not in supplement
    ):
        raise ValueError("Supplementary Table S6 changed")
    return SensitivityPublicationBundleReport(
        table_s6=table_report,
        figure_s2=figure_report,
        claim_adjudication=ClaimAdjudicationReport(
            claim_count=claim_report.claim_count,
            preserved_count=claim_report.preserved_count,
            narrowed_count=claim_report.narrowed_count,
            removed_count=claim_report.removed_count,
            author_checkpoint_status=checkpoint_status,
        ),
    )


__all__ = [
    "SensitivityPublicationBundleReport",
    "build_sensitivity_publication_bundle",
    "validate_sensitivity_publication_bundle",
]
