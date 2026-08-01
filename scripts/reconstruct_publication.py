"""Single acceptance seam for frozen publication reconstruction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
from typing import Sequence

from manuscript.manuscript_contract import validate_manuscript_package
from trackleed.publication.scientific_panel import (
    TracerValidationError,
    run_central_panel,
    validate_central_panel_output,
)
from trackleed.publication.scientific_sensitivity import (
    run_sensitivity,
    validate_sensitivity_output,
)
from trackleed.publication.scientific_sensitivity_bundle import (
    validate_sensitivity_publication_bundle,
)
from trackleed.publication.scientific_core import (
    ScientificCoreValidationError,
    validate_fingerprint,
)
from trackleed.publication.scientific_reconstruction import (
    ScientificReconstructionError,
    rebuild_reader_facing_artifacts,
    validate_reader_facing_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the frozen Ag(111) central panel and its predeclared "
            "measurement-window and capture-contract sensitivity."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--panel-package", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser


def _paths(arguments: argparse.Namespace) -> tuple[Path, Path, Path]:
    workspace = arguments.workspace_root.resolve()
    package = (
        arguments.panel_package.resolve()
        if arguments.panel_package is not None
        else workspace / "manuscript" / "scientific_panel"
    )
    output = (
        arguments.output.resolve()
        if arguments.output is not None
        else package / "output"
    )
    return workspace, package, output


def _remove_output(output: Path, workspace: Path, package: Path) -> None:
    if output in {
        Path(output.anchor).resolve(),
        Path.home().resolve(),
        workspace,
        package,
        package.parent,
    }:
        raise ValueError(f"Refusing unsafe clean target: {output}")
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Panel output is not a directory: {output}")
        shutil.rmtree(output)


def _write_scientific_core_report(
    output: Path,
    results: dict[str, object],
) -> None:
    report_directory = output / "reconstruction_report"
    report = {
        "schema_version": 1,
        "status": "scientific-baseline-complete",
        "command": "python -m scripts.reconstruct_publication --clean --validate",
        "execution_contract": {
            "device": "cpu",
            "headless": True,
            "gui": False,
            "gpu": False,
            "model_checkpoints": False,
            "tracker_inference": False,
            "editorial_rendering": False,
        },
        "results": results,
    }
    report_directory.mkdir(parents=True, exist_ok=True)
    (report_directory / "scientific_core_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = "\n".join(
        (
            "# Scientific-core validation report",
            "",
            "Status: `scientific-baseline-complete`.",
            "",
            "Command: `python -m scripts.reconstruct_publication --clean --validate`.",
            "",
            "The validated path is headless and CPU-only. It does not use the "
            "GUI, GPU, model checkpoints, tracker inference, or editorial "
            "document rendering.",
            "",
            "The machine-readable details are recorded in "
            "`scientific_core_validation_report.json`.",
            "",
        )
    )
    (report_directory / "scientific_core_validation_report.md").write_text(
        markdown,
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace, package, output = _paths(arguments)
    sensitivity_output = output / "sensitivity"
    scientific_core_root = workspace / "manuscript" / "scientific_core"
    try:
        if arguments.clean:
            _remove_output(output, workspace, package)
        if output.exists() and arguments.validate and not arguments.clean:
            central_report = validate_central_panel_output(workspace, package, output)
            sensitivity_report = validate_sensitivity_output(
                workspace,
                package,
                sensitivity_output,
            )
            publication_bundle_report = validate_sensitivity_publication_bundle(
                workspace_root=workspace,
                sensitivity_output_directory=sensitivity_output,
            )
            reader_facing_report = validate_reader_facing_artifacts(workspace)
        else:
            # Both runners validate their complete output before returning.
            central_report = run_central_panel(workspace, package, output)
            sensitivity_report = run_sensitivity(
                workspace,
                package,
                sensitivity_output,
            )
            publication_bundle_report = validate_sensitivity_publication_bundle(
                workspace_root=workspace,
                sensitivity_output_directory=sensitivity_output,
            )
            reader_facing_report = rebuild_reader_facing_artifacts(workspace)
        manuscript_report = validate_manuscript_package(workspace / "manuscript")
        scientific_core_report = validate_fingerprint(
            workspace,
            scientific_core_root / "fingerprint_contract.json",
            scientific_core_root / "fingerprint.json",
        )
        results = {
            "central_panel": asdict(central_report),
            "measurement_window_and_capture_contract_sensitivity": asdict(
                sensitivity_report
            ),
            "reader_facing_sensitivity_bundle": asdict(publication_bundle_report),
            "reader_facing_artifacts": asdict(reader_facing_report),
            "manuscript_contract": manuscript_report,
            "scientific_core": asdict(scientific_core_report),
        }
        _write_scientific_core_report(output, results)
    except (
        FileExistsError,
        FileNotFoundError,
        ScientificCoreValidationError,
        ScientificReconstructionError,
        TracerValidationError,
        ValueError,
    ) as exc:
        print(f"publication reconstruction failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
