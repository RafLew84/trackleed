from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import cast, TypedDict

from trackleed.validation import ValidationReport


SCHEMA_VERSION = 1
PRODUCTION_LIMITS = (1000, 50)
TEST_LIMITS = (1500, 75)
FUNCTION_LINE_LIMIT = 80
EXCLUDED_COMPONENTS = frozenset({"generated", "vendored", "frozen", "third_party"})


@dataclass(frozen=True)
class CodeSizeReport(ValidationReport):
    checked_files: int


@dataclass(frozen=True)
class FileMetrics:
    lines: int
    definitions: int
    oversized_definitions: dict[str, int]


class ManifestEntry(TypedDict):
    lines: int
    definitions: int
    oversized_definitions: dict[str, int]


@dataclass(frozen=True)
class BaselineManifest:
    fixed_point: str
    files: dict[str, ManifestEntry]


class _DefinitionVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.line_counts: dict[str, int] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        base_name = ".".join([*self.scope, node.name])
        qualified_name = base_name
        duplicate = 2
        while qualified_name in self.line_counts:
            qualified_name = f"{base_name}#{duplicate}"
            duplicate += 1
        start_line = min(
            [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
        )
        self.line_counts[qualified_name] = (node.end_lineno or node.lineno) - start_line + 1
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()


def validate_code_size(repo_root: Path, manifest_path: Path) -> CodeSizeReport:
    root = repo_root.resolve()
    manifest, manifest_errors = _load_manifest(manifest_path)
    if manifest is None:
        return CodeSizeReport(checked_files=0, errors=manifest_errors)

    fixed_metrics, fixed_errors = _read_fixed_point_metrics(root, manifest.fixed_point)
    current_metrics, current_errors = _read_current_metrics(root)
    history_errors = _validate_manifest_history(root, manifest_path, manifest)
    errors = [*history_errors, *fixed_errors, *current_errors]
    if not fixed_errors:
        errors.extend(_validate_manifest(manifest, fixed_metrics, current_metrics))
    errors.extend(_validate_current_files(manifest, current_metrics))
    return CodeSizeReport(checked_files=len(current_metrics), errors=tuple(errors))


def _load_manifest(path: Path) -> tuple[BaselineManifest | None, tuple[str, ...]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return None, (f"Cannot read code-size manifest: {exc}",)
    return _parse_manifest_text(text, "code-size manifest")


def _parse_manifest_text(
    text: str,
    source: str,
) -> tuple[BaselineManifest | None, tuple[str, ...]]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, (f"Cannot parse {source}: {exc}",)
    if not isinstance(raw, dict):
        return None, (f"{source.capitalize()} root must be an object.",)

    errors: list[str] = []
    if raw.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"Code-size manifest must use schema_version {SCHEMA_VERSION}.")
    fixed_point = raw.get("fixed_point")
    if not isinstance(fixed_point, str) or not fixed_point:
        errors.append("Code-size manifest is missing fixed_point.")
    files = raw.get("files")
    if not isinstance(files, dict):
        errors.append("Code-size manifest is missing files.")
    elif not errors:
        errors.extend(_validate_manifest_entries(files))
    if errors:
        return None, tuple(errors)
    typed_files = cast(dict[str, ManifestEntry], files)
    return BaselineManifest(fixed_point=fixed_point, files=typed_files), ()


def _validate_manifest_history(
    root: Path,
    manifest_path: Path,
    current: BaselineManifest,
) -> list[str]:
    try:
        relative_path = manifest_path.resolve().relative_to(root).as_posix()
    except ValueError:
        return ["Code-size manifest must be located inside the repository."]

    shallow = _run_git(root, "rev-parse", "--is-shallow-repository")
    if shallow.returncode != 0:
        return ["Cannot determine whether code-size manifest history is complete."]
    if shallow.stdout.strip() == "true":
        return ["Cannot validate code-size manifest history in a shallow repository."]

    history = _run_git(root, "rev-list", "HEAD", "--", relative_path)
    if history.returncode != 0:
        return ["Cannot read code-size manifest commit history."]
    history_errors: list[str] = []
    for revision in history.stdout.splitlines():
        previous, errors = _read_manifest_at_revision(root, revision, relative_path)
        history_errors.extend(errors)
        if previous is not None and previous != current:
            history_errors.extend(_validate_manifest_reduction(previous, current))
    return history_errors


def _read_manifest_at_revision(
    root: Path,
    revision: str,
    relative_path: str,
) -> tuple[BaselineManifest | None, list[str]]:
    listing = _run_git(root, "ls-tree", "--name-only", revision, "--", relative_path)
    if listing.returncode != 0:
        return None, [f"Cannot inspect code-size manifest at {revision}."]
    if relative_path not in listing.stdout.splitlines():
        return None, []
    shown = _run_git(root, "show", f"{revision}:{relative_path}")
    if shown.returncode != 0:
        return None, [f"Cannot read code-size manifest at {revision}."]
    previous, errors = _parse_manifest_text(
        shown.stdout,
        f"code-size manifest at {revision}",
    )
    return previous, list(errors)


def _validate_manifest_reduction(
    previous: BaselineManifest,
    current: BaselineManifest,
) -> list[str]:
    errors: list[str] = []
    if current.fixed_point != previous.fixed_point:
        errors.append(
            "Code-size manifest fixed point changed from "
            f"{previous.fixed_point} to {current.fixed_point}."
        )
    for relative_path, entry in current.files.items():
        prior_entry = previous.files.get(relative_path)
        if prior_entry is None:
            errors.append(f"Code-size manifest re-added or introduced {relative_path}.")
            continue
        for metric_name in ("lines", "definitions"):
            current_value = entry[metric_name]
            previous_value = prior_entry[metric_name]
            if current_value > previous_value:
                errors.append(
                    f"{relative_path} manifest {metric_name} ratchet was raised "
                    f"from {previous_value} to {current_value}."
                )
        for qualified_name, line_count in entry["oversized_definitions"].items():
            previous_count = prior_entry["oversized_definitions"].get(qualified_name)
            if previous_count is None:
                errors.append(
                    f"{relative_path}:{qualified_name} manifest ratchet was re-added or introduced."
                )
            elif line_count > previous_count:
                errors.append(
                    f"{relative_path}:{qualified_name} manifest ratchet was raised "
                    f"from {previous_count} to {line_count}."
                )
    return errors


def _validate_manifest_entries(files: dict[object, object]) -> list[str]:
    errors: list[str] = []
    for relative_path, entry in files.items():
        if not isinstance(relative_path, str) or not _is_repository_python_path(relative_path):
            errors.append(f"Invalid code-size manifest path: {relative_path!r}.")
            continue
        if not isinstance(entry, dict):
            errors.append(f"Manifest entry for {relative_path} must be an object.")
            continue
        for metric_name in ("lines", "definitions"):
            value = entry.get(metric_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"Manifest {relative_path}.{metric_name} must be a non-negative integer.")
        oversized = entry.get("oversized_definitions")
        if not isinstance(oversized, dict):
            errors.append(f"Manifest {relative_path}.oversized_definitions must be an object.")
            continue
        for qualified_name, line_count in oversized.items():
            if not isinstance(qualified_name, str) or not qualified_name:
                errors.append(f"Manifest {relative_path} has an invalid definition name.")
            if not isinstance(line_count, int) or isinstance(line_count, bool) or line_count <= 80:
                errors.append(
                    f"Manifest {relative_path}:{qualified_name} must exceed 80 lines."
                )
    return errors


def _read_fixed_point_metrics(
    root: Path,
    fixed_point: str,
) -> tuple[dict[str, FileMetrics], list[str]]:
    listing = _run_git(root, "ls-tree", "-r", "--name-only", fixed_point)
    if listing.returncode != 0:
        message = listing.stderr.strip() or "unknown Git error"
        return {}, [f"Cannot read fixed point {fixed_point}: {message}"]

    metrics: dict[str, FileMetrics] = {}
    errors: list[str] = []
    for relative_path in listing.stdout.splitlines():
        if not _is_repository_python_path(relative_path):
            continue
        shown = _run_git(root, "show", f"{fixed_point}:{relative_path}")
        if shown.returncode != 0:
            errors.append(f"Cannot read {relative_path} at fixed point.")
            continue
        measured, error = _measure_source(shown.stdout, relative_path)
        if error:
            errors.append(f"{relative_path} cannot be parsed at fixed point: {error}")
        elif measured:
            metrics[relative_path] = measured
    return metrics, errors


def _read_current_metrics(root: Path) -> tuple[dict[str, FileMetrics], list[str]]:
    listing = _run_git(
        root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "*.py",
    )
    if listing.returncode != 0:
        message = listing.stderr.strip() or "unknown Git error"
        return {}, [f"Cannot enumerate repository Python files: {message}"]

    metrics: dict[str, FileMetrics] = {}
    errors: list[str] = []
    for relative_path in listing.stdout.splitlines():
        path = root / relative_path
        if not path.is_file() or not _is_repository_python_path(relative_path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"{relative_path} cannot be read: {exc}")
            continue
        measured, error = _measure_source(source, relative_path)
        if error:
            errors.append(f"{relative_path} cannot be parsed: {error}")
        elif measured:
            metrics[relative_path] = measured
    return metrics, errors


def _measure_source(source: str, relative_path: str) -> tuple[FileMetrics | None, str | None]:
    try:
        module = ast.parse(source, filename=relative_path)
    except SyntaxError as exc:
        return None, str(exc)
    visitor = _DefinitionVisitor()
    visitor.visit(module)
    oversized = {
        name: line_count
        for name, line_count in visitor.line_counts.items()
        if line_count > FUNCTION_LINE_LIMIT
    }
    return FileMetrics(
        lines=len(source.splitlines()),
        definitions=len(visitor.line_counts),
        oversized_definitions=oversized,
    ), None


def _validate_manifest(
    manifest: BaselineManifest,
    fixed_metrics: dict[str, FileMetrics],
    current_metrics: dict[str, FileMetrics],
) -> list[str]:
    errors: list[str] = []
    for relative_path, entry in manifest.files.items():
        fixed = fixed_metrics.get(relative_path)
        if fixed is None or not _is_oversized(relative_path, fixed):
            errors.append(f"{relative_path} has a manifest entry but was not legacy-oversized.")
            continue
        if entry["lines"] > fixed.lines:
            errors.append(
                f"{relative_path} manifest line ratchet {entry['lines']} exceeds "
                f"fixed-point value {fixed.lines}."
            )
        if entry["definitions"] > fixed.definitions:
            errors.append(
                f"{relative_path} manifest definition ratchet {entry['definitions']} "
                f"exceeds fixed-point value {fixed.definitions}."
            )
        errors.extend(_validate_definition_ratchets(relative_path, entry, fixed))

    for relative_path, fixed in fixed_metrics.items():
        current = current_metrics.get(relative_path)
        if current is None or not _is_oversized(relative_path, current):
            continue
        if _is_oversized(relative_path, fixed) and relative_path not in manifest.files:
            errors.append(
                f"{relative_path} is oversized at the fixed point but has no manifest entry."
            )
    return errors


def _validate_definition_ratchets(
    relative_path: str,
    entry: ManifestEntry,
    fixed: FileMetrics,
) -> list[str]:
    errors: list[str] = []
    ratchets = entry["oversized_definitions"]
    for qualified_name, line_count in ratchets.items():
        fixed_line_count = fixed.oversized_definitions.get(qualified_name)
        if fixed_line_count is None:
            errors.append(
                f"{relative_path}:{qualified_name} has a manifest ratchet but was not oversized."
            )
        elif line_count > fixed_line_count:
            errors.append(
                f"{relative_path}:{qualified_name} manifest ratchet {line_count} exceeds "
                f"fixed-point value {fixed_line_count}."
            )
    return errors


def _validate_current_files(
    manifest: BaselineManifest,
    current_metrics: dict[str, FileMetrics],
) -> list[str]:
    errors: list[str] = []
    for relative_path, metrics in current_metrics.items():
        entry = manifest.files.get(relative_path)
        default_lines, default_definitions = _limits_for(relative_path)
        line_ceiling = entry["lines"] if entry else default_lines
        definition_ceiling = entry["definitions"] if entry else default_definitions
        if metrics.lines > line_ceiling:
            if entry:
                errors.append(
                    f"{relative_path} grew from {line_ceiling} to {metrics.lines} lines."
                )
            else:
                errors.append(
                    f"{relative_path} has {metrics.lines} lines; limit is {line_ceiling}."
                )
        if metrics.definitions > definition_ceiling:
            if entry:
                errors.append(
                    f"{relative_path} grew from {definition_ceiling} to "
                    f"{metrics.definitions} definitions."
                )
            else:
                errors.append(
                    f"{relative_path} has {metrics.definitions} definitions; "
                    f"limit is {definition_ceiling}."
                )
        errors.extend(_validate_current_definitions(relative_path, metrics, entry))
    return errors


def _validate_current_definitions(
    relative_path: str,
    metrics: FileMetrics,
    entry: ManifestEntry | None,
) -> list[str]:
    errors: list[str] = []
    ratchets = entry["oversized_definitions"] if entry else {}
    for qualified_name, line_count in metrics.oversized_definitions.items():
        ceiling = ratchets.get(qualified_name, FUNCTION_LINE_LIMIT)
        if line_count <= ceiling:
            continue
        if qualified_name in ratchets:
            errors.append(
                f"{relative_path}:{qualified_name} grew from {ceiling} to {line_count} lines."
            )
        else:
            errors.append(
                f"{relative_path}:{qualified_name} has {line_count} lines; limit is 80."
            )
    return errors


def _limits_for(relative_path: str) -> tuple[int, int]:
    path = PurePosixPath(relative_path)
    components = {component.casefold() for component in path.parts}
    is_test = "tests" in components or path.stem.casefold().startswith("test_")
    return TEST_LIMITS if is_test else PRODUCTION_LIMITS


def _is_oversized(relative_path: str, metrics: FileMetrics) -> bool:
    line_limit, definition_limit = _limits_for(relative_path)
    return (
        metrics.lines > line_limit
        or metrics.definitions > definition_limit
        or bool(metrics.oversized_definitions)
    )


def _is_repository_python_path(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or path.suffix.casefold() != ".py":
        return False
    components = {component.casefold() for component in path.parts}
    return components.isdisjoint(EXCLUDED_COMPONENTS)


def _run_git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            check=False,
        )
    except (OSError, UnicodeError) as exc:
        return subprocess.CompletedProcess(
            args=["git", *arguments],
            returncode=1,
            stdout="",
            stderr=str(exc),
        )
