"""Single acceptance seam for frozen publication reconstruction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
from typing import Sequence

from trackleed.publication.scientific_panel import (
    TracerValidationError,
    run_central_panel,
    validate_central_panel_output,
)
from trackleed.publication.scientific_sensitivity import (
    run_sensitivity,
    validate_sensitivity_output,
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


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace, package, output = _paths(arguments)
    sensitivity_output = output / "sensitivity"
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
        else:
            central_report = run_central_panel(workspace, package, output)
            sensitivity_report = run_sensitivity(
                workspace,
                package,
                sensitivity_output,
            )
            if arguments.validate:
                central_report = validate_central_panel_output(
                    workspace, package, output
                )
                sensitivity_report = validate_sensitivity_output(
                    workspace,
                    package,
                    sensitivity_output,
                )
    except (
        FileExistsError,
        FileNotFoundError,
        TracerValidationError,
        ValueError,
    ) as exc:
        print(f"publication reconstruction failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "central_panel": asdict(central_report),
                "measurement_window_and_capture_contract_sensitivity": asdict(
                    sensitivity_report
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
