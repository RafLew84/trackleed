"""DAM4SAM environment diagnostics and smoke-test helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Optional


DEFAULT_DAM4SAM_MODELS_DIR = Path("tracking_models") / "dam4sam"
DEFAULT_DAM4SAM_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "dam4sam",
    Path(__file__).resolve().parents[3] / "DAM4SAM",
    Path(__file__).resolve().parents[2] / "dam4sam",
)
DEFAULT_DAM4SAM_TRACKER_NAME = "sam21pp-L"
DEFAULT_DAM4SAM_IMPORT_PROBE_TIMEOUT_SEC = 60.0
DEFAULT_DAM4SAM_SMOKE_TIMEOUT_SEC = 600.0
_IMPORT_PROBE_PREFIX = "__TRACKLEED_DAM4SAM_IMPORT_PROBE__"
_SMOKE_TEST_PREFIX = "__TRACKLEED_DAM4SAM_SMOKE_TEST__"

DAM4SAM_TRACKER_VARIANTS = {
    "sam21pp-L": ("sam2.1_hiera_large.pt", "sam21pp_hiera_l.yaml"),
    "sam21pp-B": ("sam2.1_hiera_base_plus.pt", "sam21pp_hiera_b+.yaml"),
    "sam21pp-S": ("sam2.1_hiera_small.pt", "sam21pp_hiera_s.yaml"),
    "sam21pp-T": ("sam2.1_hiera_tiny.pt", "sam21pp_hiera_t.yaml"),
    "sam2pp-L": ("sam2_hiera_large.pt", "sam2pp_hiera_l.yaml"),
    "sam2pp-B": ("sam2_hiera_base_plus.pt", "sam2pp_hiera_b+.yaml"),
    "sam2pp-S": ("sam2_hiera_small.pt", "sam2pp_hiera_s.yaml"),
    "sam2pp-T": ("sam2_hiera_tiny.pt", "sam2pp_hiera_t.yaml"),
}


@dataclass
class Dam4SamImportProbeResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any]


@dataclass
class Dam4SamEnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    python_executable: Optional[Path] = None
    tracker_name: str = DEFAULT_DAM4SAM_TRACKER_NAME
    config_name: Optional[str] = None
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
    numpy_version: Optional[str] = None
    yaml_importable: bool = False
    hydra_importable: bool = False
    omegaconf_importable: bool = False
    pillow_importable: bool = False
    vot_importable: bool = False
    sam2_importable: bool = False
    sam2_module_path: Optional[str] = None
    build_sam_importable: bool = False
    dam4sam_tracker_importable: bool = False
    dam4sam_tracker_module_path: Optional[str] = None
    vot_wrapper_importable: bool = False
    vot_wrapper_module_path: Optional[str] = None
    cuda_extension_importable: bool = False
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
            and self.hydra_importable
            and self.omegaconf_importable
            and self.pillow_importable
            and self.vot_importable
            and self.sam2_importable
            and self.build_sam_importable
            and self.dam4sam_tracker_importable
            and self.vot_wrapper_importable
        )


@dataclass
class Dam4SamSmokeTestResult:
    command: list[str]
    tracker_name: str
    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any]

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and bool(self.payload.get("ok"))


def discover_dam4sam_checkpoints(
    models_dir: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_DAM4SAM_MODELS_DIR
    discovered: dict[str, Path] = {}
    for candidate in _discover_checkpoints_in_directory(directory):
        discovered[str(candidate.resolve())] = candidate.resolve()

    repo_root = _resolve_dam4sam_repo_path(repo_path)
    if repo_root is not None:
        for candidate in _discover_checkpoints_in_directory(repo_root / "checkpoints"):
            discovered[str(candidate.resolve())] = candidate.resolve()

    return sorted(discovered.values(), key=lambda path: (path.name.lower(), str(path).lower()))


def select_default_dam4sam_checkpoint(
    checkpoints: list[Path],
    tracker_name: str = DEFAULT_DAM4SAM_TRACKER_NAME,
    models_dir: str | Path | None = None,
) -> Optional[Path]:
    if not checkpoints:
        return None
    expected_name = _checkpoint_name_for_tracker(tracker_name)
    if expected_name is None:
        return checkpoints[0]

    matches = [path for path in checkpoints if path.name.lower() == expected_name.lower()]
    if not matches:
        return checkpoints[0]

    preferred_dir = (
        Path(models_dir).expanduser().resolve()
        if models_dir is not None
        else DEFAULT_DAM4SAM_MODELS_DIR.resolve()
    )
    for path in matches:
        try:
            path.relative_to(preferred_dir)
            return path
        except ValueError:
            continue
    return matches[0]


def probe_dam4sam_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
    python_executable: str | Path | None = None,
    tracker_name: str = DEFAULT_DAM4SAM_TRACKER_NAME,
    import_probe_timeout_sec: float = DEFAULT_DAM4SAM_IMPORT_PROBE_TIMEOUT_SEC,
) -> Dam4SamEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_DAM4SAM_MODELS_DIR
    directory = directory.resolve()
    normalized_tracker_name = _normalize_tracker_name(tracker_name)
    resolved_repo_path = _resolve_dam4sam_repo_path(repo_path)
    resolved_python = _resolve_dam4sam_python_executable(python_executable)
    config_name = _config_name_for_tracker(normalized_tracker_name)
    config_path = (
        resolved_repo_path / "sam2" / config_name
        if resolved_repo_path is not None and config_name is not None
        else None
    )
    upstream_checkpoint_path = (
        resolved_repo_path / "checkpoints" / _checkpoint_name_for_tracker(normalized_tracker_name)
        if resolved_repo_path is not None and _checkpoint_name_for_tracker(normalized_tracker_name)
        else None
    )
    checkpoints = discover_dam4sam_checkpoints(directory, resolved_repo_path)
    selected_checkpoint = _resolve_checkpoint_path(
        checkpoint_path,
        checkpoints,
        directory,
        normalized_tracker_name,
    )

    status = Dam4SamEnvironmentStatus(
        models_dir=directory,
        repo_path=resolved_repo_path,
        python_executable=resolved_python,
        tracker_name=normalized_tracker_name,
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
        status.errors.append(f"Could not resolve DAM4SAM repo from: {repo_path}")
    elif resolved_repo_path is None:
        status.errors.append(
            "Could not resolve DAM4SAM repo. Pass --repo-path pointing at the local DAM4SAM clone."
        )

    if resolved_python is None or not resolved_python.exists():
        status.errors.append(
            f"DAM4SAM Python executable not found: {resolved_python or python_executable}"
        )

    if not directory.exists():
        status.warnings.append(
            f"Checkpoint directory does not exist yet: {directory}. "
            "Place DAM4SAM/SAM2.1 checkpoints inside tracking_models/dam4sam."
        )

    if normalized_tracker_name not in DAM4SAM_TRACKER_VARIANTS:
        status.errors.append(
            f"Unsupported DAM4SAM tracker variant: {tracker_name}. "
            f"Supported variants: {', '.join(sorted(DAM4SAM_TRACKER_VARIANTS))}."
        )

    if config_path is None:
        status.errors.append("DAM4SAM config path cannot be resolved without repo path.")
    elif not config_path.exists():
        status.errors.append(f"DAM4SAM config not found: {config_path}")

    if selected_checkpoint is None:
        status.errors.append(
            "No DAM4SAM/SAM2 checkpoint detected. Expected SAM2.1 checkpoints in "
            "tracking_models/dam4sam or in repo/checkpoints."
        )
    elif checkpoint_path and not status.checkpoint_exists:
        status.errors.append(f"Selected checkpoint not found: {selected_checkpoint}")

    if resolved_python is not None and resolved_python.exists():
        probe = _run_dam4sam_import_probe(
            resolved_python,
            resolved_repo_path,
            timeout_sec=import_probe_timeout_sec,
        )
        _apply_import_probe_result(status, probe)

    if not status.cuda_available and status.torch_importable:
        status.errors.append("PyTorch CUDA is not available in the DAM4SAM environment.")

    if _numpy_major_version(status.numpy_version) is not None:
        numpy_major = _numpy_major_version(status.numpy_version)
        if numpy_major is not None and numpy_major >= 2 and str(status.torch_version or "").startswith("2.1"):
            status.errors.append(
                "NumPy 2.x is not compatible with the validated torch 2.1.0 DAM4SAM stack. "
                "Pin numpy==1.26.4."
            )

    if not status.cuda_extension_importable:
        status.warnings.append(
            "SAM2 custom CUDA extension is not importable. This is acceptable for feasibility; "
            "mask post-processing may be skipped, as in the SAM2 integration."
        )

    if normalized_tracker_name in {"sam21pp-S", "sam21pp-T", "sam2pp-S", "sam2pp-T"}:
        status.warnings.append(
            "Upstream DAM4SAM determine_tracker() may look for Small/Tiny checkpoints in the "
            "repo root instead of repo/checkpoints. Prefer sam21pp-L/sam21pp-B for initial checks "
            "unless this is patched or the files are copied to the expected location."
        )

    if not status.is_ready:
        status.hints.append(
            "Run from the dedicated `dam4sam` Conda env or pass "
            "--python-executable C:\\Users\\rlewa\\anaconda3\\envs\\dam4sam\\python.exe."
        )
        status.hints.append(
            "Validated feasibility stack: Python 3.10, torch 2.1.0+cu121, "
            "torchvision 0.16.0, numpy 1.26.4, vot-toolkit 0.7.1."
        )

    return status


def format_dam4sam_environment_report(status: Dam4SamEnvironmentStatus) -> str:
    available = ", ".join(path.name for path in status.available_checkpoints) or "none"
    selected = str(status.selected_checkpoint) if status.selected_checkpoint else "None"
    lines = [
        "DAM4SAM environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Python executable: {status.python_executable or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Tracker variant: {status.tracker_name}",
        f"- Config name: {status.config_name or 'None'}",
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
        f"- NumPy version: {status.numpy_version or 'unknown'}",
        f"- PyYAML import: {'ok' if status.yaml_importable else 'failed'}",
        f"- hydra import: {'ok' if status.hydra_importable else 'failed'}",
        f"- omegaconf import: {'ok' if status.omegaconf_importable else 'failed'}",
        f"- Pillow import: {'ok' if status.pillow_importable else 'failed'}",
        f"- VOT import: {'ok' if status.vot_importable else 'failed'}",
        f"- SAM2 import: {'ok' if status.sam2_importable else 'failed'}",
        f"- SAM2 build_sam import: {'ok' if status.build_sam_importable else 'failed'}",
        f"- DAM4SAM tracker import: {'ok' if status.dam4sam_tracker_importable else 'failed'}",
        f"- VOT wrapper import: {'ok' if status.vot_wrapper_importable else 'failed'}",
        f"- SAM2 CUDA extension: {'ok' if status.cuda_extension_importable else 'missing'}",
    ]
    if status.sam2_module_path:
        lines.append(f"- SAM2 module: {status.sam2_module_path}")
    if status.dam4sam_tracker_module_path:
        lines.append(f"- DAM4SAM tracker module: {status.dam4sam_tracker_module_path}")
    if status.vot_wrapper_module_path:
        lines.append(f"- VOT wrapper module: {status.vot_wrapper_module_path}")
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


def run_dam4sam_smoke_test(
    *,
    repo_path: str | Path,
    python_executable: str | Path | None = None,
    tracker_name: str = DEFAULT_DAM4SAM_TRACKER_NAME,
    timeout_sec: float = DEFAULT_DAM4SAM_SMOKE_TIMEOUT_SEC,
) -> Dam4SamSmokeTestResult:
    resolved_repo_path = Path(repo_path).expanduser().resolve()
    resolved_python = _resolve_dam4sam_python_executable(python_executable)
    if resolved_python is None:
        raise FileNotFoundError("DAM4SAM Python executable could not be resolved.")

    command = [str(resolved_python), "-c", _DAM4SAM_SMOKE_TEST_CODE]
    env = _build_windows_conda_subprocess_env(resolved_python, resolved_repo_path)
    env["TRACKLEED_DAM4SAM_TRACKER_NAME"] = _normalize_tracker_name(tracker_name)
    completed = subprocess.run(
        command,
        cwd=str(resolved_repo_path),
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_sec),
        env=env,
    )
    return Dam4SamSmokeTestResult(
        command=command,
        tracker_name=_normalize_tracker_name(tracker_name),
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        payload=_parse_prefixed_json_stdout(completed.stdout, _SMOKE_TEST_PREFIX),
    )


def format_dam4sam_smoke_test_report(result: Dam4SamSmokeTestResult) -> str:
    payload = result.payload
    lines = [
        "DAM4SAM smoke test",
        f"- tracker variant: {result.tracker_name}",
        f"- returncode: {result.returncode}",
        f"- ok: {'yes' if result.succeeded else 'no'}",
    ]
    for key in (
        "checkpoint",
        "model_cfg",
        "device",
        "frame_shape",
        "init_bbox_xywh",
        "init_mask_shape",
        "tracked_mask_shape",
        "init_mask_area",
        "tracked_mask_area",
        "tracked_mask_bbox_xyxy",
        "elapsed_sec",
    ):
        if key in payload:
            lines.append(f"- {key.replace('_', ' ')}: {payload[key]}")
    if result.returncode != 0 or not result.succeeded:
        if result.stdout.strip():
            lines.append("- stdout:")
            lines.append(result.stdout.strip())
        if result.stderr.strip():
            lines.append("- stderr:")
            lines.append(result.stderr.strip())
    return "\n".join(lines)


def _resolve_dam4sam_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidates = [Path(repo_path)] if repo_path else list(DEFAULT_DAM4SAM_REPO_CANDIDATES)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "dam4sam_tracker.py").exists() and (
            resolved / "sam2" / "build_sam.py"
        ).exists():
            return resolved
    return None


def _resolve_dam4sam_python_executable(python_executable: str | Path | None) -> Optional[Path]:
    if python_executable:
        return Path(python_executable).expanduser().resolve()
    return Path(sys.executable).resolve()


def _discover_checkpoints_in_directory(directory: str | Path | None) -> list[Path]:
    if directory is None:
        return []
    root = Path(directory).expanduser()
    if not root.exists():
        return []
    checkpoints = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.lower().endswith((".pt", ".pth", ".pth.tar", ".tar")):
            checkpoints.append(path.resolve())
    return checkpoints


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoints: list[Path],
    models_dir: Path,
    tracker_name: str,
) -> Optional[Path]:
    if checkpoint_path:
        candidate = Path(checkpoint_path).expanduser()
        if not candidate.is_absolute():
            candidate = models_dir / candidate
        return candidate.resolve()
    return select_default_dam4sam_checkpoint(checkpoints, tracker_name, models_dir)


def _checkpoint_name_for_tracker(tracker_name: str) -> Optional[str]:
    variant = DAM4SAM_TRACKER_VARIANTS.get(_normalize_tracker_name(tracker_name))
    return variant[0] if variant is not None else None


def _config_name_for_tracker(tracker_name: str) -> Optional[str]:
    variant = DAM4SAM_TRACKER_VARIANTS.get(_normalize_tracker_name(tracker_name))
    return variant[1] if variant is not None else None


def _normalize_tracker_name(tracker_name: str | None) -> str:
    value = str(tracker_name or DEFAULT_DAM4SAM_TRACKER_NAME).strip()
    return value or DEFAULT_DAM4SAM_TRACKER_NAME


def _run_dam4sam_import_probe(
    python_executable: Path,
    repo_path: Optional[Path],
    *,
    timeout_sec: float,
) -> Dam4SamImportProbeResult:
    command = [str(python_executable), "-c", _DAM4SAM_IMPORT_PROBE_CODE]
    completed = subprocess.run(
        command,
        cwd=str(repo_path) if repo_path is not None else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_sec),
        env=_build_windows_conda_subprocess_env(python_executable, repo_path),
    )
    return Dam4SamImportProbeResult(
        command=command,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        payload=_parse_prefixed_json_stdout(completed.stdout, _IMPORT_PROBE_PREFIX),
    )


def _apply_import_probe_result(
    status: Dam4SamEnvironmentStatus,
    probe: Dam4SamImportProbeResult,
) -> None:
    status.import_probe_returncode = probe.returncode
    status.import_probe_stdout = probe.stdout
    status.import_probe_stderr = probe.stderr
    payload = probe.payload
    if probe.returncode != 0:
        status.errors.append(
            "DAM4SAM import probe failed with return code "
            f"{probe.returncode}: {probe.stderr.strip() or probe.stdout.strip()}"
        )
    if not payload:
        status.errors.append("DAM4SAM import probe did not return parseable diagnostics JSON.")
        return

    status.torch_importable = bool(payload.get("torch_importable"))
    status.torch_version = payload.get("torch_version")
    status.torch_cuda_version = payload.get("torch_cuda_version")
    status.cuda_available = bool(payload.get("cuda_available"))
    status.torchvision_importable = bool(payload.get("torchvision_importable"))
    status.cv2_importable = bool(payload.get("cv2_importable"))
    status.numpy_importable = bool(payload.get("numpy_importable"))
    status.numpy_version = payload.get("numpy_version")
    status.yaml_importable = bool(payload.get("yaml_importable"))
    status.hydra_importable = bool(payload.get("hydra_importable"))
    status.omegaconf_importable = bool(payload.get("omegaconf_importable"))
    status.pillow_importable = bool(payload.get("pillow_importable"))
    status.vot_importable = bool(payload.get("vot_importable"))
    status.sam2_importable = bool(payload.get("sam2_importable"))
    status.sam2_module_path = payload.get("sam2_module_path")
    status.build_sam_importable = bool(payload.get("sam2_build_sam_importable"))
    status.dam4sam_tracker_importable = bool(payload.get("dam4sam_tracker_importable"))
    status.dam4sam_tracker_module_path = payload.get("dam4sam_tracker_module_path")
    status.vot_wrapper_importable = bool(payload.get("vot_wrapper_dam4sam_importable"))
    status.vot_wrapper_module_path = payload.get("vot_wrapper_dam4sam_module_path")
    status.cuda_extension_importable = bool(payload.get("sam2_cuda_extension_importable"))

    for key, value in sorted(payload.items()):
        if key.endswith("_error") and value:
            status.errors.append(f"{key[:-6]} import failed: {value}")


def _parse_prefixed_json_stdout(stdout: str, prefix: str) -> dict[str, Any]:
    for line in reversed(str(stdout or "").splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    return {}


def _build_windows_conda_subprocess_env(
    python_executable: Path,
    repo_path: Optional[Path] = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if repo_path is not None:
        env["TRACKLEED_DAM4SAM_REPO"] = str(repo_path)
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


def _numpy_major_version(version: str | None) -> Optional[int]:
    if not version:
        return None
    first_chunk = str(version).split(".", 1)[0]
    try:
        return int(first_chunk)
    except ValueError:
        return None


_DAM4SAM_IMPORT_PROBE_CODE = r"""
import importlib
import json
import os
import sys

PREFIX = "__TRACKLEED_DAM4SAM_IMPORT_PROBE__"
result = {"python_executable": sys.executable}
repo_path = os.environ.get("TRACKLEED_DAM4SAM_REPO")
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

numpy = probe("numpy", "numpy")
if numpy is not None:
    result["numpy_version"] = getattr(numpy, "__version__", None)

probe("torchvision", "torchvision")
probe("cv2", "cv2")
probe("yaml", "yaml")
probe("hydra", "hydra")
probe("omegaconf", "omegaconf")
probe("PIL.Image", "pillow")
probe("vot.region.raster", "vot")
probe("sam2", "sam2")
probe("sam2.build_sam", "sam2_build_sam")
probe("dam4sam_tracker", "dam4sam_tracker")
probe("vot_wrapper_dam4sam", "vot_wrapper_dam4sam")
probe("sam2._C", "sam2_cuda_extension")

print(PREFIX + json.dumps(result, sort_keys=True))
"""


_DAM4SAM_SMOKE_TEST_CODE = r"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

PREFIX = "__TRACKLEED_DAM4SAM_SMOKE_TEST__"
repo_path = os.environ.get("TRACKLEED_DAM4SAM_REPO")
if repo_path and repo_path not in sys.path:
    sys.path.insert(0, repo_path)

tracker_name = os.environ.get("TRACKLEED_DAM4SAM_TRACKER_NAME") or "sam21pp-L"


def make_frame(dx):
    image = Image.new("RGB", (144, 112), (12, 14, 18))
    draw = ImageDraw.Draw(image)
    draw.ellipse((52 + dx, 42, 76 + dx, 66), fill=(240, 205, 80))
    draw.ellipse((94, 44, 116, 64), fill=(80, 130, 210))
    return image


def mask_bbox(mask):
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


start = time.perf_counter()
from dam4sam_tracker import DAM4SAMTracker
import torch

tracker = DAM4SAMTracker(tracker_name=tracker_name)
frame0 = make_frame(0)
frame1 = make_frame(3)
init_bbox = [52, 42, 24, 24]
out0 = tracker.initialize(frame0, None, bbox=init_bbox)
out1 = tracker.track(frame1)
init_mask = np.asarray(out0["pred_mask"])
tracked_mask = np.asarray(out1["pred_mask"])
payload = {
    "ok": bool(init_mask.size and tracked_mask.size and np.count_nonzero(tracked_mask) > 0),
    "tracker_name": tracker_name,
    "checkpoint": str(getattr(tracker, "checkpoint", "")),
    "model_cfg": str(getattr(tracker, "model_cfg", "")),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "frame_shape": [112, 144, 3],
    "init_bbox_xywh": init_bbox,
    "init_mask_shape": list(init_mask.shape),
    "tracked_mask_shape": list(tracked_mask.shape),
    "init_mask_area": int(np.count_nonzero(init_mask)),
    "tracked_mask_area": int(np.count_nonzero(tracked_mask)),
    "tracked_mask_bbox_xyxy": mask_bbox(tracked_mask),
    "elapsed_sec": round(time.perf_counter() - start, 3),
}
print(PREFIX + json.dumps(payload, sort_keys=True))
"""
