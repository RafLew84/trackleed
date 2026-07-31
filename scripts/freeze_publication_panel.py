"""Create the immutable expected-record package for the full Ag(111) panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from trackleed.analysis.scientific_panel_common import TracerValidationError
from trackleed.analysis.scientific_panel_freeze import build_panel_package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the pre-existing five-method Ag(111) central record."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--panel-package", type=Path)
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument(
        "--build",
        action="store_true",
        help="Required acknowledgement that a new immutable package is created.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.build:
        print("panel freeze refused: --build is required", file=sys.stderr)
        return 2
    workspace_root = arguments.workspace_root.resolve()
    panel_dir = (
        arguments.panel_package.resolve()
        if arguments.panel_package is not None
        else workspace_root / "manuscript" / "scientific_panel"
    )
    historical_root = (
        arguments.historical_root.resolve()
        if arguments.historical_root is not None
        else workspace_root
        / ".scratch"
        / "trackleed-physics-method-paper"
        / "artifacts"
        / "independent-scane-validation"
    )
    try:
        authentication = build_panel_package(
            workspace_root, panel_dir, historical_root
        )
    except (FileExistsError, FileNotFoundError, OSError, TracerValidationError) as exc:
        print(f"panel freeze failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(authentication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
