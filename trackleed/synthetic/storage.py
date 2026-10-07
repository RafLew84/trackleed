"""Portable package encoding, fingerprints and generator provenance."""
from __future__ import annotations

from hashlib import sha256
import importlib.metadata
import ctypes
import errno
import ntpath
import shutil
import os
import sys
import json
from pathlib import Path
import platform
import re
import subprocess
from typing import Any

import numpy as np
from numpy.typing import NDArray

PACKAGE_TYPE = "trackleed.synthetic-benchmark"
MARKER = ".trackleed-synthetic-benchmark"
STAGING_PREFIX = ".trackleed-incomplete-"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, allow_nan=False,
                               indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_descriptor(array: NDArray) -> dict[str, Any]:
    dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array, dtype=dtype)
    header = {"dtype": dtype.str, "shape": list(array.shape), "order": "C"}
    digest = sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return {**header, "pixel_sha256": digest.hexdigest()}


def generator_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "trackleed" / "synthetic").glob("*.py"))
    paths += [root / "scripts" / "generate_tracking_benchmark.py"]
    sources = {p.relative_to(root).as_posix(): file_sha256(p) for p in paths if p.is_file()}
    digest = sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    revision = _git(root, "rev-parse", "HEAD")
    dirty = _working_tree_dirty(root, set(sources))
    return {
        "name": PACKAGE_TYPE, "version": "1.4-stage5", "git_revision": revision,
        "git_dirty": dirty, "git_dirty_scope": "tracked_files_and_generator_sources",
        "source_sha256": digest, "source_files": sources,
        "python": platform.python_version(), "platform": platform.platform(),
        "libraries": {name: importlib.metadata.version(name)
                      for name in ("numpy", "tifffile", "Pillow")},
    }


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True,
                                text=True, timeout=15, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def publish_directory(stage: Path, target: Path) -> None:
    """Atomically publish without replacing even an empty competing directory."""
    if os.name == "nt":
        os.rename(stage, target)  # Windows rename never replaces an existing directory.
        return
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        try:
            rename = library.renameat2
        except AttributeError as exc:
            raise OSError("Atomic no-replace publication is unavailable") from exc
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(stage), -100, os.fsencode(target), 1) != 0:
            error = ctypes.get_errno()
            if error in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS) and "microsoft" in platform.release().lower():
                _publish_windows_mount(stage, target)
                return
            raise OSError(error, os.strerror(error), str(target))
        return
    raise OSError("Atomic no-replace publication supports Windows and Linux")


def refuse_benchmark_source(source: Path) -> None:
    """Protect benchmark roots and descendants from legacy PNG16 conversion."""
    resolved = source.resolve()
    for ancestor in (resolved, *resolved.parents):
        marker = ancestor / MARKER
        manifest = ancestor / "manifest.json"
        recognized = os.path.lexists(marker)
        if manifest.is_file():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
                recognized |= isinstance(payload, dict) and payload.get("package_type") == PACKAGE_TYPE
            except (OSError, UnicodeError, ValueError):
                # The permanent marker identifies corrupt new packages too.
                if not recognized:
                    try:
                        recognized = PACKAGE_TYPE in manifest.read_text(encoding="utf-8-sig")
                    except (OSError, UnicodeError):
                        pass
        if recognized:
            candidates = [ancestor / "images", *sorted((ancestor / "sequences").glob("*/images"))]
            folders = ", ".join(str(path) for path in candidates if path.is_dir())
            location = folders or f"the images folders of {ancestor} (package is incomplete)"
            raise ValueError(
                f"Synthetic benchmark cannot use the legacy converter; "
                f"prepared PNG uint8 images are in {location}"
            )


def validate_provenance(value: Any) -> None:
    """Validate recorded provenance without claiming it is the current runtime."""
    required = {"name", "version", "git_revision", "git_dirty", "source_sha256",
                "source_files", "python", "platform", "libraries", "git_dirty_scope"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Invalid generator provenance fields")
    if value["git_dirty_scope"] != "tracked_files_and_generator_sources":
        raise ValueError("Invalid generator git_dirty_scope")
    if value["name"] != PACKAGE_TYPE:
        raise ValueError("Invalid generator name")
    for key in ("version", "python", "platform"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"Invalid generator {key}")
    if value["git_dirty"] is not None and type(value["git_dirty"]) is not bool:
        raise ValueError("Invalid generator git_dirty")
    revision = value["git_revision"]
    if revision is not None and (not isinstance(revision, str)
                                 or re.fullmatch(r"[0-9a-f]{40,64}", revision) is None):
        raise ValueError("Invalid generator git_revision")
    sources = value["source_files"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Missing generator source fingerprints")
    for path, digest in sources.items():
        if (not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts
                or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError("Invalid generator source fingerprint")
    actual = sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if value["source_sha256"] != actual:
        raise ValueError("Generator source checksum mismatch")
    libraries = value["libraries"]
    if not isinstance(libraries, dict) or set(libraries) != {"numpy", "tifffile", "Pillow"}:
        raise ValueError("Invalid generator library versions")
    if any(not isinstance(v, str) or not v for v in libraries.values()):
        raise ValueError("Invalid generator library version")


def _working_tree_dirty(root: Path, source_paths: set[str]) -> bool | None:
    tracked = _git(root, "ls-files", "--", "trackleed/synthetic",
                   "scripts/generate_tracking_benchmark.py")
    if tracked is None:
        return None
    if source_paths - set(tracked.splitlines()):
        return True
    try:
        result = subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"],
                                cwd=root, capture_output=True, timeout=15, check=False)
        return bool(result.returncode) if result.returncode in (0, 1) else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _publish_windows_mount(stage: Path, target: Path) -> None:
    # DrvFS lacks renameat2 flags. Execute a native no-replace Windows directory move.
    if stage.resolve().parent != target.parent.resolve() or not stage.name.startswith(STAGING_PREFIX):
        raise ValueError("Publication requires a sibling staging directory")
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        raise OSError("Windows-mounted publication requires PowerShell interoperability")
    source_win, target_win = _windows_path(stage), _windows_path(target)
    if ntpath.dirname(source_win).casefold() != ntpath.dirname(target_win).casefold():
        raise ValueError("Publication paths must share one Windows directory")
    command = """
$ErrorActionPreference = 'Stop'
$paths = [Console]::In.ReadToEnd() | ConvertFrom-Json
try {
    [System.IO.Directory]::Move($paths.source, $paths.target)
} catch {
    if (Test-Path -LiteralPath $paths.target) { exit 17 }
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
"""
    # Paths are JSON data on stdin, never interpolated into shell code.
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
        input=json.dumps({"source": source_win, "target": target_win}),
        text=True, encoding="utf-8", errors="replace", capture_output=True, check=False,
    )
    if result.returncode == 17:
        raise FileExistsError(errno.EEXIST, "Output already exists", str(target))
    if result.returncode != 0:
        raise OSError(f"Native Windows publication failed: {result.stderr.strip()}")


def _windows_path(path: Path) -> str:
    result = subprocess.run(["wslpath", "-w", str(path.absolute())],
                            capture_output=True, text=True, check=False)
    value = result.stdout.strip()
    if result.returncode or re.match(r"^[A-Za-z]:\\", value) is None:
        raise OSError("Publication fallback requires a Windows drive path")
    if any(part.endswith((" ", ".")) for part in Path(path).parts):
        raise ValueError("Windows publication paths cannot end in spaces or dots")
    return value
