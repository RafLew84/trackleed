#!/usr/bin/env python3
"""Capture path-free package records from retained Windows Conda environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from importlib.metadata import distributions
from pathlib import Path
from urllib.parse import urlparse


ENVIRONMENT_NAMES = (
    "netcat",
    "locotrack-pytorch",
    "ostrack",
    "dam4sam",
    "sam2",
    "samurai",
    "easyleed",
)
ALLOWED_CHANNEL_HOSTS = {
    "conda.anaconda.org",
    "repo.anaconda.com",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _safe_channel(raw_channel: object) -> str:
    """Retain reviewed public channel URLs and reject local/private paths."""

    channel = str(raw_channel or "not recorded")
    if channel == "not recorded":
        return channel
    parsed = urlparse(channel)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_CHANNEL_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Conda channel is not an approved public, credential-free URL: "
            f"{channel!r}"
        )
    return channel


def _python_version(environment: Path) -> str:
    records = []
    for record_path in (environment / "conda-meta").glob("python-*.json"):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("name") == "python":
            records.append(str(record["version"]))
    if len(records) != 1:
        raise ValueError(
            f"Expected one Python package record for {environment.name}."
        )
    return records[0]


def _installed_distributions(environment: Path) -> list[dict[str, str]]:
    site_packages = environment / "Lib" / "site-packages"
    packages: dict[str, str] = {}
    for distribution in distributions(path=[str(site_packages)]):
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = _canonical_name(raw_name)
        version = str(distribution.version)
        previous = packages.get(name)
        if previous is not None and previous != version:
            raise ValueError(
                f"Conflicting versions for {name} in {environment.name}."
            )
        packages[name] = version
    return [
        {"name": name, "version": version}
        for name, version in sorted(packages.items())
    ]


def _conda_packages(environment: Path) -> list[dict[str, object]]:
    packages = []
    for record_path in sorted((environment / "conda-meta").glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not {"name", "version", "build"}.issubset(record):
            continue
        packages.append(
            {
                "name": str(record["name"]),
                "version": str(record["version"]),
                "build": str(record["build"]),
                "build_number": int(record.get("build_number", 0)),
                "channel": _safe_channel(record.get("channel")),
                "subdir": str(record.get("subdir", "not recorded")),
            }
        )
    return sorted(
        packages,
        key=lambda package: (
            str(package["name"]),
            str(package["version"]),
            str(package["build"]),
        ),
    )


def _torch_cuda_build(environment: Path) -> str:
    version_path = (
        environment / "Lib" / "site-packages" / "torch" / "version.py"
    )
    if not version_path.is_file():
        return "not applicable"
    text = version_path.read_text(encoding="utf-8")
    match = re.search(
        r"^cuda(?:\s*:\s*[^=\n]+)?\s*=\s*['\"]([^'\"]+)['\"]",
        text,
        flags=re.MULTILINE,
    )
    return match.group(1) if match else "not recorded"


def capture_environment(
    environment: Path,
    capture_date: str,
) -> dict[str, object]:
    """Return a normalized, path-free snapshot of one retained environment."""

    history_path = environment / "conda-meta" / "history"
    if not history_path.is_file():
        raise FileNotFoundError(history_path)
    return {
        "schema_version": "1.0",
        "status": "retained_environment_snapshot",
        "capture_date": capture_date,
        "environment_label": environment.name,
        "runtime_kind": "python-conda",
        "platform_family": "Windows 64-bit",
        "python_version": _python_version(environment),
        "torch_cuda_build": _torch_cuda_build(environment),
        "conda_history_sha256": sha256_file(history_path),
        "installed_distributions": _installed_distributions(environment),
        "conda_packages": _conda_packages(environment),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-date", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ENVIRONMENT_NAMES:
        record = capture_environment(
            args.conda_root / "envs" / name,
            args.capture_date,
        )
        output_path = args.output_dir / f"{name}.json"
        output_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
