"""Validate changed repository text without normalizing or rewriting any bytes."""
from __future__ import annotations

from pathlib import Path
import subprocess

from trackleed.validation import ValidationReport


_TEXT_SUFFIXES = frozenset({
    ".py", ".pyi", ".md", ".rst", ".txt", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".csv", ".tsv", ".ps1", ".sh", ".bat", ".cmd",
    ".html", ".css", ".js", ".ts", ".tsx", ".jsx", ".xml", ".svg", ".tex",
})
_BINARY_SUFFIXES = frozenset({
    ".zip", ".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".pdf", ".docx",
    ".xlsx", ".pptx", ".npy", ".npz", ".pt", ".pth", ".ico", ".woff", ".woff2",
})


def validate_changed_text(root: Path, *, base: str | None = None) -> ValidationReport:
    """Check current changed files and independently check staged versions.

    With no explicit base, include the latest commit (first parent) plus local
    changes. An initial commit is compared with the empty tree. A missing base
    or missing history is an error, never an empty successful check.
    """
    root = root.resolve()
    try:
        staged = _names(root, "diff", "--cached", "--name-only", "-z",
                        "--diff-filter=ACMRT", "--no-renames", "--")
        names = _committed_paths(root, base) | staged
        names |= _names(root, "diff", "--name-only", "-z", "--diff-filter=ACMRT",
                        "--no-renames", "--")
        names |= _names(root, "ls-files", "--others", "--exclude-standard", "-z")
        attributes = _text_attributes(root, names)
        errors = []
        for name in sorted(names):
            if attributes.get(name) == "unset":
                continue  # Explicit binary / frozen upstream bytes.
            path = root / name
            if path.is_file() and not path.is_symlink():
                errors.extend(_check_bytes(name, "working tree", path.read_bytes(),
                                           attributes.get(name)))
            if name in staged:
                data = _git(root, "show", f":{name}")
                errors.extend(_check_bytes(name, "staged", data, attributes.get(name)))
        return ValidationReport(errors=tuple(errors))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return ValidationReport(errors=(f"Changed-text validation unavailable: {exc}",))


def _git(root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "--no-optional-locks", *arguments], cwd=root, input=input_bytes, capture_output=True,
        check=False, timeout=300,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout


def _names(root: Path, *arguments: str) -> set[str]:
    return {name.decode("utf-8") for name in _git(root, *arguments).split(b"\0") if name}


def _committed_paths(root: Path, base: str | None) -> set[str]:
    if base == "EMPTY":
        return _names(root, "ls-tree", "-r", "--name-only", "-z", "HEAD")
    if base is None:
        header = _git(root, "cat-file", "-p", "HEAD").split(b"\n\n", 1)[0]
        parents = [line[7:].decode("ascii") for line in header.splitlines()
                   if line.startswith(b"parent ")]
        if not parents:
            return _names(root, "diff-tree", "--root", "--no-commit-id", "-r",
                          "--name-only", "-z", "--diff-filter=ACMRT", "HEAD", "--")
        base = parents[0]
    resolved = _git(root, "rev-parse", "--verify", "--end-of-options",
                    f"{base}^{{commit}}").decode("ascii").strip()
    return _names(root, "diff", "--name-only", "-z", "--diff-filter=ACMRT",
                  "--no-renames", resolved, "HEAD", "--")


def _text_attributes(root: Path, names: set[str]) -> dict[str, str]:
    if not names:
        return {}
    output = _git(root, "check-attr", "-z", "--stdin", "text",
                  input_bytes=b"".join(name.encode("utf-8") + b"\0" for name in sorted(names)))
    fields = output.split(b"\0")
    return {fields[i].decode("utf-8"): fields[i + 2].decode("ascii")
            for i in range(0, len(fields) - 1, 3)}


def _check_bytes(name: str, snapshot: str, data: bytes, attribute: str | None) -> list[str]:
    suffix = Path(name).suffix.lower()
    known_text = suffix in _TEXT_SUFFIXES or attribute == "set"
    if not known_text and (suffix in _BINARY_SUFFIXES or b"\0" in data[:8000]):
        return []
    prefix = f"{name} ({snapshot})"
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        line = data[:exc.start].count(b"\n") + 1
        return [f"{prefix}:{line}: invalid UTF-8 at byte {exc.start}; save as UTF-8."]
    if b"\0" in data:
        return [f"{prefix}: NUL byte in a text file; save as UTF-8 text."]
    if b"\r\r\n" in data:
        return [f"{prefix}: damaged CRCRLF line endings; use consistent LF or CRLF."]
    without_crlf = data.replace(b"\r\n", b"")
    if b"\r" in without_crlf:
        return [f"{prefix}: bare CR line ending; use consistent LF or CRLF."]
    if b"\r\n" in data and b"\n" in without_crlf:
        return [f"{prefix}: mixed LF/CRLF line endings; use one consistent style."]
    return []
