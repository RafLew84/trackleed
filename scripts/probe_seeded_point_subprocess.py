#!/usr/bin/env python3
"""Probe a seeded point backend in its configured execution interpreter."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 3:
        print(json.dumps({"errors": ["Expected module, probe, and kwargs."]}))
        return 2
    module_name, probe_name, kwargs_json = arguments
    try:
        kwargs: dict[str, Any] = json.loads(kwargs_json)
        module = importlib.import_module(module_name)
        status = getattr(module, probe_name)(**kwargs)
        errors = [str(item) for item in getattr(status, "errors", ())]
        ready = bool(getattr(status, "is_ready", False))
    except Exception as exc:  # noqa: BLE001 - probe reports optional failures
        print(json.dumps({"errors": [str(exc)]}, ensure_ascii=False))
        return 2
    print(json.dumps({"ready": ready, "errors": errors}, ensure_ascii=False))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
