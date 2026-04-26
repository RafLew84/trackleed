"""OSTrack environment diagnostics and smoke-test helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Optional


DEFAULT_OSTRACK_MODELS_DIR = Path("tracking_models") / "ostrack"
DEFAULT_OSTRACK_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "ostrack",
    Path(__file__).resolve().parents[3] / "OSTrack",
    Path(__file__).resolve().parents[2] / "ostrack",
)
DEFAULT_OSTRACK_CONFIG_NAME = "vitb_256_mae_ce_32x4_ep300"
DEFAULT_OSTRACK_CHECKPOINT_NAME = "OSTrack_ep0300.pth.tar"
DEFAULT_OSTRACK_PROFILE_TIMEOUT_SEC = 180.0
_IMPORT_PROBE_PREFIX = "__TRACKLEED_OSTRACK_IMPORT_PROBE__"


@dataclass
class OstrackEnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    python_executable: Optional[Path] = None
    config_name: str = DEFAULT_OSTRACK_CONFIG_NAME
    config_path: Optional[Path] = None
    config_exists: bool = False
    available_checkpoints: list[Path] = field(default_factory=list)
    selected_checkpoint: Optional[Path] = None
    checkpoint_exists: bool = False
    upstream_checkpoint_path: Optional[Path] = None
    upstream_checkpoint_exists: bool = False
    import_probe_returncode: Optional[int] = None
    import_probe_stdout: str = ""
    import_probe_stderr: str = ""
    torch_importable: bool = False
    torch_version: Optional[str] = None
    torch_cuda_version: Optional[str] = None
    cuda_available: bool = False
    torchvision_importable: bool = False
    cv2_importable: bool = False
    numpy_importable: bool = False
    yaml_importable: bool = False
    easydict_importable: bool = False
    timm_importable: bool = False
    lmdb_importable: bool = False
    jpeg4py_importable: bool = False
    ostrack_parameter_importable: bool = False
    ostrack_tracker_importable: bool = False
    ostrack_parameter_module_path: Optional[str] = None
    ostrack_tracker_module_path: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return (
            self.repo_path is not None
            and self.python_executable is not None
            and self.python_executable.exists()
            and self.config_exists
            and self.checkpoint_exists
            and self.torch_importable
            and self.cuda_available
            and self.torchvision_importable
            and self.cv2_importable
            and self.numpy_importable
            and self.yaml_importable
            and self.easydict_importable
            and self.timm_importable
            and self.lmdb_importable
            and self.jpeg4py_importable
            and self.ostrack_parameter_importable
            and self.ostrack_tracker_importable
        )


@dataclass
class OstrackImportProbeResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any]


@dataclass
class OstrackProfileSmokeTestResult:
    command: list[str]
    config_name: str
    returncode: int
    stdout: str
    stderr: str
    macs: Optional[str]
    params: Optional[str]
    latency_ms: Optional[float]
    fps: Optional[float]


def discover_ostrack_checkpoints(models_dir: str | Path | None = None) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_OSTRACK_MODELS_DIR
    if not directory.exists():
        return []
    checkpoints = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith((".pth", ".pt", ".tar", ".pth.tar")):
            checkpoints.append(path.resolve())
    return sorted(set(checkpoints), key=lambda item: item.name.lower())


def select_default_ostrack_checkpoint(
    checkpoints: list[Path],
    config_name: str = DEFAULT_OSTRACK_CONFIG_NAME,
) -> Optional[Path]:
    if not checkpoints:
        return None
    by_name = {path.name.lower(): path for path in checkpoints}
    preferred_names = (
        f"{config_name}.pth.tar",
        f"{config_name}.pth",
        DEFAULT_OSTRACK_CHECKPOINT_NAME,
    )
    for name in preferred_names:
        checkpoint = by_name.get(name.lower())
        if checkpoint is not None:
            return checkpoint
    return checkpoints[0]


def probe_ostrack_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
    python_executable: str | Path | None = None,
    config_name: str = DEFAULT_OSTRACK_CONFIG_NAME,
    import_probe_timeout_sec: float = 60.0,
) -> OstrackEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_OSTRACK_MODELS_DIR
    directory = directory.resolve()
    resolved_repo_path = _resolve_ostrack_repo_path(repo_path)
    resolved_python = _resolve_ostrack_python_executable(python_executable)
    config_path = (
        resolved_repo_path / "experiments" / "ostrack" / f"{config_name}.yaml"
        if resolved_repo_path is not None
        else None
    )
    upstream_checkpoint_path = (
        resolved_repo_path
        / "output"
        / "checkpoints"
        / "train"
        / "ostrack"
        / config_name
        / DEFAULT_OSTRACK_CHECKPOINT_NAME
        if resolved_repo_path is not None
        else None
    )
    checkpoints = discover_ostrack_checkpoints(directory)
    selected_checkpoint = _resolve_checkpoint_path(
        checkpoint_path,
        checkpoints,
        directory,
        upstream_checkpoint_path,
        config_name,
    )

    status = OstrackEnvironmentStatus(
        models_dir=directory,
        repo_path=resolved_repo_path,
        python_executable=resolved_python,
        config_name=config_name,
        config_path=config_path,
        config_exists=bool(config_path and config_path.exists()),
        available_checkpoints=checkpoints,
        selected_checkpoint=selected_checkpoint,
        checkpoint_exists=bool(selected_checkpoint and selected_checkpoint.exists()),
        upstream_checkpoint_path=upstream_checkpoint_path,
        upstream_checkpoint_exists=bool(
            upstream_checkpoint_path and upstream_checkpoint_path.exists()
        ),
    )

    if repo_path and resolved_repo_path is None:
        status.errors.append(f"Could not resolve OSTrack repo from: {repo_path}")
    elif resolved_repo_path is None:
        status.errors.append(
            "Could not resolve OSTrack repo. Pass --repo-path pointing at the local OSTrack clone."
        )

    if resolved_python is None or not resolved_python.exists():
        status.errors.append(
            f"OSTrack Python executable not found: {resolved_python or python_executable}"
        )

    if not directory.exists():
        status.warnings.append(
            f"Checkpoint directory does not exist yet: {directory}. "
            "Place OSTrack checkpoints inside tracking_models/ostrack."
        )

    if config_path is None:
        status.errors.append("OSTrack config path cannot be resolved without repo path.")
    elif not config_path.exists():
        status.errors.append(f"OSTrack config not found: {config_path}")

    if selected_checkpoint is None:
        status.errors.append(
            "No OSTrack checkpoint detected. Expected either "
            f"{config_name}.pth.tar in tracking_models/ostrack or "
            f"{DEFAULT_OSTRACK_CHECKPOINT_NAME} in the upstream output/checkpoints path."
        )
    elif checkpoint_path and not status.checkpoint_exists:
        status.errors.append(f"Selected checkpoint not found: {selected_checkpoint}")

    if resolved_python is not None and resolved_python.exists():
        probe = _run_ostrack_import_probe(
            resolved_python,
            resolved_repo_path,
            timeout_sec=import_probe_timeout_sec,
        )
        _apply_import_probe_result(status, probe)

    if not status.cuda_available and status.torch_importable:
        status.errors.append("PyTorch CUDA is not available in the OSTrack environment.")

    if not status.is_ready:
        status.hints.append(
            "Run from the dedicated `ostrack` Conda env or pass "
            "--python-executable C:\\Users\\rlewa\\anaconda3\\envs\\ostrack\\python.exe."
        )
        status.hints.append(
            "The validated Windows stack is Python 3.8 with torch 1.13.1+cu117, "
            "torchvision 0.14.1+cu117, timm 0.5.4, python-lmdb and jpeg4py."
        )

    return status


def format_ostrack_environment_report(status: OstrackEnvironmentStatus) -> str:
    available = ", ".join(path.name for path in status.available_checkpoints) or "none"
    selected = str(status.selected_checkpoint) if status.selected_checkpoint else "None"
    lines = [
        "OSTrack environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Python executable: {status.python_executable or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Config name: {status.config_name}",
        f"- Config path: {status.config_path or 'None'}",
        f"- Config exists: {'yes' if status.config_exists else 'no'}",
        f"- Selected checkpoint: {selected}",
        f"- Selected checkpoint exists: {'yes' if status.checkpoint_exists else 'no'}",
        f"- Upstream checkpoint: {status.upstream_checkpoint_path or 'None'}",
        f"- Upstream checkpoint exists: {'yes' if status.upstream_checkpoint_exists else 'no'}",
        f"- Available checkpoints: {available}",
        f"- Import probe returncode: {status.import_probe_returncode}",
        f"- PyTorch import: {'ok' if status.torch_importable else 'failed'}",
        f"- PyTorch version: {status.torch_version or 'unknown'}",
        f"- PyTorch CUDA version: {status.torch_cuda_version or 'unknown'}",
        f"- CUDA available: {'yes' if status.cuda_available else 'no'}",
        f"- torchvision import: {'ok' if status.torchvision_importable else 'failed'}",
        f"- OpenCV import: {'ok' if status.cv2_importable else 'failed'}",
        f"- NumPy import: {'ok' if status.numpy_importable else 'failed'}",
        f"- PyYAML import: {'ok' if status.yaml_importable else 'failed'}",
        f"- easydict import: {'ok' if status.easydict_importable else 'failed'}",
        f"- timm import: {'ok' if status.timm_importable else 'failed'}",
        f"- lmdb import: {'ok' if status.lmdb_importable else 'failed'}",
        f"- jpeg4py import: {'ok' if status.jpeg4py_importable else 'failed'}",
        f"- OSTrack parameter import: {'ok' if status.ostrack_parameter_importable else 'failed'}",
        f"- OSTrack tracker import: {'ok' if status.ostrack_tracker_importable else 'failed'}",
    ]
    if status.ostrack_parameter_module_path:
        lines.append(f"- OSTrack parameter module: {status.ostrack_parameter_module_path}")
    if status.ostrack_tracker_module_path:
        lines.append(f"- OSTrack tracker module: {status.ostrack_tracker_module_path}")
    if status.errors:
        lines.append("- Errors:")
        lines.extend(f"  - {item}" for item in status.errors)
    if status.warnings:
        lines.append("- Warnings:")
        lines.extend(f"  - {item}" for item in status.warnings)
    if status.hints:
        lines.append("- Hints:")
        lines.extend(f"  - {item}" for item in status.hints)
    return "\n".join(lines)


def run_ostrack_profile_smoke_test(
    *,
    repo_path: str | Path,
    python_executable: str | Path | None = None,
    config_name: str = DEFAULT_OSTRACK_CONFIG_NAME,
    timeout_sec: float = DEFAULT_OSTRACK_PROFILE_TIMEOUT_SEC,
) -> OstrackProfileSmokeTestResult:
    resolved_repo_path = Path(repo_path).expanduser().resolve()
    resolved_python = _resolve_ostrack_python_executable(python_executable)
    if resolved_python is None:
        raise FileNotFoundError("OSTrack Python executable could not be resolved.")
    profile_script = resolved_repo_path / "tracking" / "profile_model.py"
    command = [
        str(resolved_python),
        str(profile_script),
        "--script",
        "ostrack",
        "--config",
        config_name,
    ]
    completed = subprocess.run(
        command,
        cwd=str(resolved_repo_path),
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_sec),
        env=_build_windows_conda_subprocess_env(resolved_python, resolved_repo_path),
    )
    return OstrackProfileSmokeTestResult(
        command=command,
        config_name=config_name,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        macs=_extract_first_match(r"overall macs is\s+([^\r\n]+)", completed.stdout),
        params=_extract_first_match(r"overall params is\s+([^\r\n]+)", completed.stdout),
        latency_ms=_extract_float(
            r"The average overall latency is\s+([0-9.]+)\s*ms",
            completed.stdout,
        ),
        fps=_extract_float(r"FPS is\s+([0-9.]+)\s*fps", completed.stdout),
    )


def _resolve_ostrack_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidates = [Path(repo_path)] if repo_path else list(DEFAULT_OSTRACK_REPO_CANDIDATES)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "experiments" / "ostrack").exists() and (
            resolved / "lib" / "test" / "tracker" / "ostrack.py"
        ).exists():
            return resolved
    return None


def _resolve_ostrack_python_executable(python_executable: str | Path | None) -> Optional[Path]:
    if python_executable:
        return Path(python_executable).expanduser().resolve()
    return Path(sys.executable).resolve()


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoints: list[Path],
    models_dir: Path,
    upstream_checkpoint_path: Optional[Path],
    config_name: str,
) -> Optional[Path]:
    if checkpoint_path:
        candidate = Path(checkpoint_path).expanduser()
        if not candidate.is_absolute():
            candidate = models_dir / candidate
        return candidate.resolve()
    selected = select_default_ostrack_checkpoint(checkpoints, config_name)
    if selected is not None:
        return selected
    if upstream_checkpoint_path is not None:
        return upstream_checkpoint_path
    return None


def _run_ostrack_import_probe(
    python_executable: Path,
    repo_path: Optional[Path],
    *,
    timeout_sec: float,
) -> OstrackImportProbeResult:
    command = [str(python_executable), "-c", _OSTRACK_IMPORT_PROBE_CODE]
    completed = subprocess.run(
        command,
        cwd=str(repo_path) if repo_path is not None else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_sec),
        env=_build_windows_conda_subprocess_env(python_executable, repo_path),
    )
    payload = _parse_import_probe_stdout(completed.stdout)
    return OstrackImportProbeResult(
        command=command,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        payload=payload,
    )


def _apply_import_probe_result(
    status: OstrackEnvironmentStatus,
    probe: OstrackImportProbeResult,
) -> None:
    status.import_probe_returncode = probe.returncode
    status.import_probe_stdout = probe.stdout
    status.import_probe_stderr = probe.stderr
    payload = probe.payload
    if probe.returncode != 0:
        status.errors.append(
            "OSTrack import probe failed with return code "
            f"{probe.returncode}: {probe.stderr.strip() or probe.stdout.strip()}"
        )
    if not payload:
        status.errors.append("OSTrack import probe did not return parseable diagnostics JSON.")
        return

    status.torch_importable = bool(payload.get("torch_importable"))
    status.torch_version = payload.get("torch_version")
    status.torch_cuda_version = payload.get("torch_cuda_version")
    status.cuda_available = bool(payload.get("cuda_available"))
    status.torchvision_importable = bool(payload.get("torchvision_importable"))
    status.cv2_importable = bool(payload.get("cv2_importable"))
    status.numpy_importable = bool(payload.get("numpy_importable"))
    status.yaml_importable = bool(payload.get("yaml_importable"))
    status.easydict_importable = bool(payload.get("easydict_importable"))
    status.timm_importable = bool(payload.get("timm_importable"))
    status.lmdb_importable = bool(payload.get("lmdb_importable"))
    status.jpeg4py_importable = bool(payload.get("jpeg4py_importable"))
    status.ostrack_parameter_importable = bool(payload.get("ostrack_parameter_importable"))
    status.ostrack_tracker_importable = bool(payload.get("ostrack_tracker_importable"))
    status.ostrack_parameter_module_path = payload.get("ostrack_parameter_module_path")
    status.ostrack_tracker_module_path = payload.get("ostrack_tracker_module_path")

    for key, value in sorted(payload.items()):
        if key.endswith("_error") and value:
            status.errors.append(f"{key[:-6]} import failed: {value}")


def _parse_import_probe_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(str(stdout or "").splitlines()):
        if line.startswith(_IMPORT_PROBE_PREFIX):
            payload = line[len(_IMPORT_PROBE_PREFIX):]
            return json.loads(payload)
    return {}


def _build_windows_conda_subprocess_env(
    python_executable: Path,
    repo_path: Optional[Path] = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if repo_path is not None:
        env["TRACKLEED_OSTRACK_REPO"] = str(repo_path)
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(repo_path)] + ([current_pythonpath] if current_pythonpath else [])
        )
    if os.name != "nt":
        return env

    exe_path = Path(python_executable).expanduser().resolve()
    env_root = exe_path.parent
    dll_dirs = [
        env_root,
        env_root / "Library" / "bin",
        env_root / "DLLs",
        env_root / "Scripts",
        env_root / "Lib" / "site-packages" / "torch" / "lib",
    ]
    existing = [str(path) for path in dll_dirs if path.exists()]
    current_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(existing + ([current_path] if current_path else []))
    env.setdefault("CONDA_PREFIX", str(env_root))
    env.setdefault("CONDA_DEFAULT_ENV", env_root.name)
    return env


def _extract_first_match(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text)
    if match is None:
        return None
    return match.group(1).strip()


def _extract_float(pattern: str, text: str) -> Optional[float]:
    value = _extract_first_match(pattern, text)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


_OSTRACK_IMPORT_PROBE_CODE = r"""
import importlib
import json
import os
import sys

PREFIX = "__TRACKLEED_OSTRACK_IMPORT_PROBE__"
result = {"python_executable": sys.executable}
repo_path = os.environ.get("TRACKLEED_OSTRACK_REPO")
if repo_path and repo_path not in sys.path:
    sys.path.insert(0, repo_path)


def probe(name, label):
    try:
        module = importlib.import_module(name)
        result[label + "_importable"] = True
        result[label + "_module_path"] = getattr(module, "__file__", None)
        return module
    except Exception as exc:
        result[label + "_importable"] = False
        result[label + "_error"] = type(exc).__name__ + ": " + str(exc)
        return None


torch = probe("torch", "torch")
if torch is not None:
    result["torch_version"] = getattr(torch, "__version__", None)
    result["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        result["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        result["cuda_available"] = False
        result["cuda_error"] = type(exc).__name__ + ": " + str(exc)
else:
    result["cuda_available"] = False

probe("torchvision", "torchvision")
probe("cv2", "cv2")
probe("numpy", "numpy")
probe("yaml", "yaml")
probe("easydict", "easydict")
probe("timm", "timm")
probe("lmdb", "lmdb")
probe("jpeg4py", "jpeg4py")
probe("lib.test.parameter.ostrack", "ostrack_parameter")
probe("lib.test.tracker.ostrack", "ostrack_tracker")

print(PREFIX + json.dumps(result, sort_keys=True))
"""
