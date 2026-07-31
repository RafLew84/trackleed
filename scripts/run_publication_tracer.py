"""Run the frozen Scan A / pair 1 / LocoTrack publication tracer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
from typing import Sequence

from trackleed.publication.scientific_tracer import (
    TracerValidationError,
    run_tracer,
    validate_tracer_output,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the frozen Scan A / pair 1 / LocoTrack result from "
            "stored proposals and raw images."
        )
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--tracer-package",
        type=Path,
        help="Frozen tracer package (default: manuscript/scientific_tracer).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: <tracer-package>/output).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove only the resolved output directory before reconstruction.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate a retained output, or validate again after reconstruction.",
    )
    return parser


def _resolve_paths(arguments: argparse.Namespace) -> tuple[Path, Path, Path]:
    workspace_root = arguments.workspace_root.resolve()
    tracer_package = (
        arguments.tracer_package.resolve()
        if arguments.tracer_package is not None
        else workspace_root / "manuscript" / "scientific_tracer"
    )
    output = (
        arguments.output.resolve()
        if arguments.output is not None
        else tracer_package / "output"
    )
    return workspace_root, tracer_package, output


def _remove_output(output: Path, workspace_root: Path, tracer_package: Path) -> None:
    filesystem_root = Path(output.anchor).resolve()
    if output in {
        filesystem_root,
        Path.home().resolve(),
        workspace_root,
        tracer_package,
        tracer_package.parent,
    }:
        raise ValueError(f"Refusing unsafe clean target: {output}")
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Tracer output is not a directory: {output}")
        shutil.rmtree(output)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace_root, tracer_package, output = _resolve_paths(arguments)
    try:
        if arguments.clean:
            _remove_output(output, workspace_root, tracer_package)

        if output.exists() and arguments.validate and not arguments.clean:
            report = validate_tracer_output(workspace_root, tracer_package, output)
        else:
            report = run_tracer(workspace_root, tracer_package, output)
            if arguments.validate:
                report = validate_tracer_output(
                    workspace_root,
                    tracer_package,
                    output,
                )
    except (
        FileExistsError,
        FileNotFoundError,
        TracerValidationError,
        ValueError,
    ) as error:
        print(f"publication tracer failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
