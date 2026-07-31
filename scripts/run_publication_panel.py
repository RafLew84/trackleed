"""Run or validate the frozen complete Ag(111) central panel."""

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the frozen five-method, four-case Ag(111) central panel."
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
    try:
        if arguments.clean:
            _remove_output(output, workspace, package)
        if output.exists() and arguments.validate and not arguments.clean:
            report = validate_central_panel_output(workspace, package, output)
        else:
            report = run_central_panel(workspace, package, output)
            if arguments.validate:
                report = validate_central_panel_output(
                    workspace, package, output
                )
    except (
        FileExistsError,
        FileNotFoundError,
        TracerValidationError,
        ValueError,
    ) as exc:
        print(f"publication panel failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
