"""Small, atomic status snapshots for an existing validation run."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
import io
import json
import os
from pathlib import Path
import tempfile
import time
import sys
from typing import TextIO
import unittest


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_summary(summary: str) -> str:
    compact = " ".join(summary.splitlines()[:2])
    return compact if len(compact) <= 600 else compact[:597] + "..."


@dataclass
class _RunState:
    schema_version: int
    pid: int
    status: str
    started_at: str
    updated_at: str
    stage: str
    stage_started_at: str
    log_path: str | None
    last_completed_step: str | None = None
    finished_at: str | None = None
    tests_completed: int = 0
    tests_total: int | None = None
    last_completed_test: str | None = None
    summary: str | None = None


class ValidationProgress:
    def __init__(self, path: Path | None = None, log_path: Path | None = None):
        now = _now()
        self.path = path
        self._last_test_snapshot = 0.0
        self._state = _RunState(
            1, os.getpid(), "running", now, now, "startup", now,
            str(log_path) if log_path is not None else None,
        )
        self._write()

    def start_step(self, name: str) -> None:
        self._state.stage = name
        self._state.stage_started_at = _now()
        self._write()

    def complete_step(self) -> None:
        self._state.last_completed_step = self._state.stage
        self._write()

    def tests_completed(self, completed: int, total: int, last: str | None = None) -> None:
        self._state.tests_completed = completed
        self._state.tests_total = total
        if last is not None:
            self._state.last_completed_test = last
        now = time.monotonic()
        if completed <= 1 or now - self._last_test_snapshot >= 1:
            self._write()
            self._last_test_snapshot = now

    def finish(self, code: int, summary: str) -> None:
        self._state.status = "interrupted" if code == 130 else ("passed" if code == 0 else "failed")
        self._state.finished_at = _now()
        self._state.summary = compact_summary(summary)
        self._write()

    def _write(self) -> None:
        if self.path is None:
            return
        self._state.updated_at = _now()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=self.path.parent,
            prefix=self.path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(asdict(self._state), stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            except BaseException:
                stream.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            for attempt in range(3):
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError:
                    if os.name != "nt" or attempt == 2:
                        raise
                    # A Windows reader may briefly hold the previous snapshot.
                    time.sleep(.02)
        finally:
            temporary.unlink(missing_ok=True)


def read_validation_status(root: Path, selected: str = "latest") -> dict:
    if selected == "latest":
        candidates = sorted((root / ".scratch/validation").glob("workflow-*.status.json"))
        if not candidates:
            raise ValueError("No validation status exists; start validation first.")
        path = candidates[-1]
    else:
        path = Path(selected)
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError(f"Unsupported validation status: {path}")
    end = datetime.fromisoformat(state["finished_at"]) if state.get("finished_at") else datetime.now(timezone.utc)
    state["elapsed_seconds"] = round((end - datetime.fromisoformat(state["started_at"])).total_seconds(), 3)
    state["status_path"] = str(path.resolve())
    return state


class _LiveTestOutput:
    def __init__(self, buffer: io.StringIO, destination: TextIO):
        self.buffer = buffer
        self.destination = destination

    def write(self, text: str) -> int:
        self.destination.write(text)
        self.destination.flush()
        return self.buffer.write(text)

    def flush(self) -> None:
        self.destination.flush()


class _ProgressResult(unittest.TextTestResult):
    def __init__(self, *args, progress: ValidationProgress, total: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress = progress
        self.total = total

    def stopTest(self, test) -> None:
        super().stopTest(test)
        self.progress.tests_completed(self.testsRun, self.total, test.id())


def run_test_suite(suite: unittest.TestSuite, stream: io.StringIO,
                   progress: ValidationProgress) -> unittest.TestResult:
    total = suite.countTestCases()
    progress.tests_completed(0, total)
    output = _LiveTestOutput(stream, sys.stdout) if progress.path is not None else stream
    return unittest.TextTestRunner(
        stream=output, verbosity=2,
        resultclass=partial(_ProgressResult, progress=progress, total=total),
    ).run(suite)
