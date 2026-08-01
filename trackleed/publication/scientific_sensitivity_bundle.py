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
from trackleed.publication.scientific_sensitivity_sources import (
    freeze_sensitivity_source_data,
    read_json,
    sha256_file,
    validate_sensitivity_source_data,
)


SENSITIVITY_SOURCE_RELATIVE_PATH = Path(
    "manuscript/source_data/measurement_sensitivity"
)


@dataclass(frozen=True)
class SensitivityPublicationBundleReport:
    table_s6: TableS6Report
    figure_s2: FigureS2Report
    claim_adjudication: ClaimAdjudicationReport


def _resolved_bundle_paths(
    workspace_root: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    manuscript_root = workspace_root / "manuscript"
    return (
        workspace_root / SENSITIVITY_SOURCE_RELATIVE_PATH,
        manuscript_root / "tables" / "table_s6_measurement_sensitivity",
        manuscript_root / "figures" / "figure_s2_measurement_sensitivity",
        manuscript_root / "claim_adjudication",
        manuscript_root / "supplementary-material.md",
    )


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
    sensitivity_source_root: Path | None = None,
    table_root: Path | None = None,
    figure_root: Path | None = None,
    adjudication_root: Path | None = None,
    supplementary_material_path: Path | None = None,
    adjudication_contract_path: Path | None = None,
) -> SensitivityPublicationBundleReport:
    """Freeze validated results and build their only publication displays."""

    workspace_root = workspace_root.resolve()
    manuscript_root = workspace_root / "manuscript"
    sensitivity_source_root = (
        sensitivity_source_root
        or workspace_root / SENSITIVITY_SOURCE_RELATIVE_PATH
    ).resolve()
    table_root = (
        table_root
        or manuscript_root / "tables" / "table_s6_measurement_sensitivity"
    ).resolve()
    figure_root = (
        figure_root
        or manuscript_root / "figures" / "figure_s2_measurement_sensitivity"
    ).resolve()
    adjudication_root = (
        adjudication_root or manuscript_root / "claim_adjudication"
    ).resolve()
    supplementary_material_path = (
        supplementary_material_path or manuscript_root / "supplementary-material.md"
    ).resolve()
    adjudication_contract_path = (
        adjudication_contract_path
        or adjudication_root / "adjudication_contract.json"
    ).resolve()

    freeze_sensitivity_source_data(
        sensitivity_output_directory,
        sensitivity_source_root,
        workspace_root=workspace_root,
    )

    table_report = build_table_s6_artifact(
        workspace_root,
        sensitivity_source_root,
        table_root / "output",
        table_root / "manifest.json",
    )
    figure_report = build_figure_s2_artifact(
        workspace_root,
        sensitivity_source_root,
        figure_root / "output",
        figure_root / "manifest.json",
    )
    claim_report = build_claim_adjudication(
        workspace_root=workspace_root,
        contract_path=adjudication_contract_path,
        sensitivity_source_directory=sensitivity_source_root,
        metadata_directory=manuscript_root / "experimental_metadata",
        claim_map_path=manuscript_root / "claim-evidence-map.md",
        output_directory=adjudication_root / "output",
    )
    synchronize_table_s6(
        table_root / "output" / "table_s6_measurement_sensitivity.md",
        supplementary_material_path,
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
    source_root, table_root, figure_root, adjudication_root, supplement_path = (
        _resolved_bundle_paths(workspace_root)
    )
    origin_directory = (
        sensitivity_output_directory
        or workspace_root
        / "manuscript"
        / "scientific_panel"
        / "output"
        / "sensitivity"
    )
    validate_sensitivity_source_data(
        source_root,
        origin_directory=origin_directory,
    )
    with TemporaryDirectory() as temporary:
        generated_root = Path(temporary)
        table_report = build_table_s6_artifact(
            workspace_root,
            source_root,
            generated_root / "table-output",
            generated_root / "table-manifest.json",
        )
        figure_report = build_figure_s2_artifact(
            workspace_root,
            source_root,
            generated_root / "figure-output",
            generated_root / "figure-manifest.json",
        )
        claim_report = build_claim_adjudication(
            workspace_root=workspace_root,
            contract_path=adjudication_root / "adjudication_contract.json",
            sensitivity_source_directory=source_root,
            metadata_directory=workspace_root / "manuscript" / "experimental_metadata",
            claim_map_path=workspace_root / "manuscript" / "claim-evidence-map.md",
            output_directory=generated_root / "claim-output",
        )
        _require_matching_file(
            generated_root / "table-manifest.json",
            table_root / "manifest.json",
        )
        _require_matching_file(
            generated_root / "figure-manifest.json",
            figure_root / "manifest.json",
        )
        for generated, committed in (
            (generated_root / "table-output", table_root / "output"),
            (generated_root / "figure-output", figure_root / "output"),
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
                adjudication_root / "output" / name,
            )

    boundary_path = adjudication_root / "output" / "final_claim_boundary.json"
    checkpoint = read_json(adjudication_root / "output" / "author_checkpoint.json")
    validate_author_checkpoint(checkpoint, sha256_file(boundary_path))
    checkpoint_status = str(checkpoint["status"])
    _validate_claim_manifest(workspace_root, adjudication_root, checkpoint_status)
    table_markdown = (
        table_root / "output" / "table_s6_measurement_sensitivity.md"
    ).read_text(encoding="utf-8").rstrip()
    supplement = supplement_path.read_text(encoding="utf-8")
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
    "SENSITIVITY_SOURCE_RELATIVE_PATH",
    "SensitivityPublicationBundleReport",
    "build_sensitivity_publication_bundle",
    "validate_sensitivity_publication_bundle",
]
