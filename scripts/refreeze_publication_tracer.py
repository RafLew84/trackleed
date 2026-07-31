"""Explicitly refreeze an already validated scientific tracer reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Sequence

from trackleed.analysis.scientific_tracer_common import (
    TRACER_AUTHENTICATION_PATH,
    TracerValidationError,
)
from trackleed.analysis.scientific_tracer_freeze import refreeze_tracer_package
from trackleed.publication.scientific_tracer import (
    stage_tracer_refreeze,
    validate_tracer_output,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh frozen tracer records, schemas, manifests and authentication "
            "after an intentional implementation-only change."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--tracer-package", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--approve-refreeze",
        action="store_true",
        help="Required acknowledgement that the immutable package will be updated.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.approve_refreeze:
        print("refreeze refused: --approve-refreeze is required", file=sys.stderr)
        return 2
    workspace_root = arguments.workspace_root.resolve()
    tracer_dir = (
        arguments.tracer_package.resolve()
        if arguments.tracer_package is not None
        else workspace_root / "manuscript" / "scientific_tracer"
    )
    try:
        authentication_path = workspace_root / TRACER_AUTHENTICATION_PATH
        if arguments.output is not None:
            output_dir = arguments.output.resolve()
            authentication = refreeze_tracer_package(
                workspace_root,
                tracer_dir,
                output_dir,
                authentication_path,
            )
            validate_tracer_output(workspace_root, tracer_dir, output_dir)
        else:
            with TemporaryDirectory() as temporary:
                staged_output = Path(temporary) / "tracer-output"
                stage_tracer_refreeze(
                    workspace_root,
                    tracer_dir,
                    staged_output,
                    authentication_path,
                )
                authentication = refreeze_tracer_package(
                    workspace_root,
                    tracer_dir,
                    staged_output,
                    authentication_path,
                )
                output_dir = tracer_dir / "output"
                if output_dir.exists():
                    if not output_dir.is_dir():
                        raise ValueError(
                            f"Tracer output is not a directory: {output_dir}"
                        )
                    shutil.rmtree(output_dir)
                shutil.copytree(staged_output, output_dir)
                validate_tracer_output(workspace_root, tracer_dir, output_dir)
    except (OSError, TracerValidationError, ValueError) as error:
        print(f"refreeze failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(authentication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
