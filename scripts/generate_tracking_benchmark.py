#!/usr/bin/env python3
"""Generate or verify a complete static or energy-scanned H6 benchmark without GUI or model imports."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trackleed.synthetic.config import load_config  # noqa: E402
from trackleed.synthetic.package import (  # noqa: E402
    check_output, generate_benchmark, verify_package,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path, help="Versioned benchmark JSON")
    mode.add_argument("--verify", type=Path, help="Read and verify a published package")
    parser.add_argument("--output", type=Path, help="New output directory; parent must exist")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify and (args.output or args.validate_only):
        parser.error("--verify cannot be combined with --output or --validate-only")
    if args.config and not args.validate_only and args.output is None:
        parser.error("--output is required for generation")
    try:
        if args.verify:
            result = {"status": "verified", **asdict(verify_package(args.verify))}
        else:
            config = load_config(args.config)
            if args.output:
                check_output(args.output)
            if args.validate_only:
                result = {"status": "valid", "plan": config.plan(), "config": config.expanded()}
            else:
                print(json.dumps({"plan": config.plan()}), file=sys.stderr)
                result = {"status": "complete", **asdict(generate_benchmark(args.config, args.output))}
        print(json.dumps(result, default=str, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Benchmark interrupted; no incomplete package is published.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
