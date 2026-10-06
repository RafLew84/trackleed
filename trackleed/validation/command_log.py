"""Persist validation diagnostics while keeping the command output short."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import tempfile
from typing import IO
import traceback


def run_logged_validation(
    repository_root: Path, action: Callable[[], tuple[int, str]],
) -> int:
    directory = repository_root / ".scratch" / "validation"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now(timezone.utc).strftime("workflow-%Y%m%dT%H%M%SZ-")
        log = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=prefix,
            suffix=".log", dir=directory, delete=False,
        )
    except OSError as cause:
        print(f"Cannot create full validation log: {cause}", file=sys.stderr)
        return 1
    log_path = Path(log.name).resolve()
    with log, _capture_output(log.file):
        try:
            status, summary = action()
        except Exception as cause:
            traceback.print_exc()
            status, summary = 1, f"Current workflow validation failed: {cause}"
        print(summary, flush=True)
    console = sys.stdout if status == 0 else sys.stderr
    compact = " ".join(summary.splitlines()[:2])
    print(compact if len(compact) <= 600 else compact[:597] + "...", file=console)
    print(f"Full log: {log_path}", file=console)
    return status


@contextmanager
def _capture_output(log: IO[str]) -> Iterator[None]:
    # File descriptors also capture native libraries, existing log handlers and
    # child processes that do not write through Python's current sys.stderr.
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    try:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        with redirect_stdout(log), redirect_stderr(log):
            yield
    finally:
        log.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
