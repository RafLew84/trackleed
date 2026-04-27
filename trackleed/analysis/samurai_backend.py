"""SAMURAI environment diagnostics and smoke-test helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Optional


DEFAULT_SAMURAI_MODELS_DIR = Path("tracking_models") / "samurai"
DEFAULT_SAMURAI_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "samurai",
    Path(__file__).resolve().parents[3] / "SAMURAI",
    Path(__file__).resolve().parents[2] / "samurai",
)
DEFAULT_SAMURAI_MODEL_VARIANT = "sam2.1_hiera_base_plus"
DEFAULT_SAMURAI_IMPORT_PROBE_TIMEOUT_SEC = 60.0
DEFAULT_SAMURAI_SMOKE_TIMEOUT_SEC = 600.0
_IMPORT_PROBE_PREFIX = "__TRACKLEED_SAMURAI_IMPORT_PROBE__"
_SMOKE_TEST_PREFIX = "__TRACKLEED_SAMURAI_SMOKE_TEST__"

SAMURAI_MODEL_VARIANTS = {
    "sam2.1_hiera_large": (
        "sam2.1_hiera_large.pt",
        "sam2.1_hiera_l.yaml",
        "configs/samurai/sam2.1_hiera_l.yaml",
    ),
    "sam2.1_hiera_base_plus": (
        "sam2.1_hiera_base_plus.pt",
        "sam2.1_hiera_b+.yaml",
        "configs/samurai/sam2.1_hiera_b+.yaml",
    ),
    "sam2.1_hiera_small": (
        "sam2.1_hiera_small.pt",
        "sam2.1_hiera_s.yaml",
        "configs/samurai/sam2.1_hiera_s.yaml",
    ),
    "sam2.1_hiera_tiny": (
        "sam2.1_hiera_tiny.pt",
        "sam2.1_hiera_t.yaml",
        "configs/samurai/sam2.1_hiera_t.yaml",
    ),
}

_SAMURAI_VARIANT_ALIASES = {
    "large": "sam2.1_hiera_large",
    "l": "sam2.1_hiera_large",
    "sam21_large": "sam2.1_hiera_large",
    "sam21_hiera_large": "sam2.1_hiera_large",
    "base": "sam2.1_hiera_base_plus",
    "base_plus": "sam2.1_hiera_base_plus",
    "b+": "sam2.1_hiera_base_plus",
    "b_plus": "sam2.1_hiera_base_plus",
    "sam21_base_plus": "sam2.1_hiera_base_plus",
    "sam21_hiera_base_plus": "sam2.1_hiera_base_plus",
    "small": "sam2.1_hiera_small",
    "s": "sam2.1_hiera_small",
    "sam21_small": "sam2.1_hiera_small",
    "sam21_hiera_small": "sam2.1_hiera_small",
    "tiny": "sam2.1_hiera_tiny",
    "t": "sam2.1_hiera_tiny",
    "sam21_tiny": "sam2.1_hiera_tiny",
    "sam21_hiera_tiny": "sam2.1_hiera_tiny",
}


@dataclass
class SamuraiImportProbeResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any]


@dataclass
class SamuraiEnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    python_executable: Optional[Path] = None
    model_variant: str = DEFAULT_SAMURAI_MODEL_VARIANT
    config_name: Optional[str] = None
    config_identifier: Optional[str] = None
    config_path: Optional[Path] = None
    config_exists: bool = False
    demo_script_path: Optional[Path] = None
    demo_script_exists: bool = False
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
    jpeg4py_importable: bool = False
    lmdb_importable: bool = False
    loguru_importable: bool = False
    sam2_importable: bool = False
    sam2_module_path: Optional[str] = None
    build_sam_importable: bool = False
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
            and self.demo_script_exists
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
            and self.jpeg4py_importable
            and self.lmdb_importable
            and self.loguru_importable
            and self.sam2_importable
            and self.build_sam_importable
        )


@dataclass
class SamuraiSmokeTestResult:
    command: list[str]
    model_variant: str
    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any]

    @property
    def succeeded(self) -> bool:
        return (
            self.returncode == 0
            and bool(self.payload.get("ok"))
            and int(self.payload.get("propagated_frame_count", 0)) >= 2
        )


def discover_samurai_checkpoints(
    models_dir: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_SAMURAI_MODELS_DIR
    discovered: dict[str, Path] = {}
    for candidate in _discover_checkpoints_in_directory(directory):
        discovered[str(candidate.resolve())] = candidate.resolve()

    repo_root = _resolve_samurai_repo_path(repo_path)
    if repo_root is not None:
        for candidate in _discover_checkpoints_in_directory(repo_root / "sam2" / "checkpoints"):
            discovered[str(candidate.resolve())] = candidate.resolve()

    return sorted(discovered.values(), key=lambda path: (path.name.lower(), str(path).lower()))


def select_default_samurai_checkpoint(
    checkpoints: list[Path],
    model_variant: str = DEFAULT_SAMURAI_MODEL_VARIANT,
    models_dir: str | Path | None = None,
) -> Optional[Path]:
    if not checkpoints:
        return None
    expected_name = _checkpoint_name_for_variant(model_variant)
    matches = [path for path in checkpoints if path.name.lower() == expected_name.lower()]
    if not matches:
        return checkpoints[0]

    preferred_dir = (
        Path(models_dir).expanduser().resolve()
        if models_dir is not None
        else DEFAULT_SAMURAI_MODELS_DIR.resolve()
    )
    for path in matches:
        try:
            path.relative_to(preferred_dir)
            return path
        except ValueError:
            continue
    return matches[0]


def probe_samurai_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
    python_executable: str | Path | None = None,
    model_variant: str = DEFAULT_SAMURAI_MODEL_VARIANT,
    import_probe_timeout_sec: float = DEFAULT_SAMURAI_IMPORT_PROBE_TIMEOUT_SEC,
) -> SamuraiEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_SAMURAI_MODELS_DIR
    directory = directory.resolve()
    normalized_variant = _normalize_model_variant(model_variant)
    resolved_repo_path = _resolve_samurai_repo_path(repo_path)
    resolved_python = _resolve_samurai_python_executable(python_executable)
    config_name = _config_name_for_variant(normalized_variant)
    config_identifier = _config_identifier_for_variant(normalized_variant)
    config_path = (
        resolved_repo_path / "sam2" / "sam2" / "configs" / "samurai" / config_name
        if resolved_repo_path is not None
        else None
    )
    demo_script_path = resolved_repo_path / "scripts" / "demo.py" if resolved_repo_path else None
    upstream_checkpoint_path = (
        resolved_repo_path / "sam2" / "checkpoints" / _checkpoint_name_for_variant(normalized_variant)
        if resolved_repo_path is not None
        else None
    )
    checkpoints = discover_samurai_checkpoints(directory, resolved_repo_path)
    selected_checkpoint = _resolve_checkpoint_path(
        checkpoint_path,
        checkpoints,
        directory,
        normalized_variant,
    )

    status = SamuraiEnvironmentStatus(
        models_dir=directory,
        repo_path=resolved_repo_path,
        python_executable=resolved_python,
        model_variant=normalized_variant,
        config_name=config_name,
        config_identifier=config_identifier,
        config_path=config_path,
        config_exists=bool(config_path and config_path.exists()),
        demo_script_path=demo_script_path,
        demo_script_exists=bool(demo_script_path and demo_script_path.exists()),
        available_checkpoints=checkpoints,
        selected_checkpoint=selected_checkpoint,
        checkpoint_exists=bool(selected_checkpoint and selected_checkpoint.exists()),
        upstream_checkpoint_path=upstream_checkpoint_path,
        upstream_checkpoint_exists=bool(
            upstream_checkpoint_path and upstream_checkpoint_path.exists()
        ),
    )

    if repo_path and resolved_repo_path is None:
        status.errors.append(f"Could not resolve SAMURAI repo from: {repo_path}")
    elif resolved_repo_path is None:
        status.errors.append(
            "Could not resolve SAMURAI repo. Pass --repo-path pointing at the local SAMURAI clone."
        )

    if resolved_python is None or not resolved_python.exists():
        status.errors.append(
            f"SAMURAI Python executable not found: {resolved_python or python_executable}"
        )

    if not directory.exists():
        status.warnings.append(
            f"Checkpoint directory does not exist yet: {directory}. "
            "Place SAM2.1 checkpoints inside tracking_models/samurai."
        )

    if normalized_variant not in SAMURAI_MODEL_VARIANTS:
        status.errors.append(
            f"Unsupported SAMURAI model variant: {model_variant}. "
            f"Supported variants: {', '.join(sorted(SAMURAI_MODEL_VARIANTS))}."
        )

    if config_path is None:
        status.errors.append("SAMURAI config path cannot be resolved without repo path.")
    elif not config_path.exists():
        status.errors.append(f"SAMURAI config not found: {config_path}")

    if demo_script_path is None:
        status.errors.append("SAMURAI demo script cannot be resolved without repo path.")
    elif not demo_script_path.exists():
        status.errors.append(f"SAMURAI demo script not found: {demo_script_path}")

    if selected_checkpoint is None:
        status.errors.append(
            "No SAMURAI/SAM2.1 checkpoint detected. Expected SAM2.1 checkpoints in "
            "tracking_models/samurai or in repo/sam2/checkpoints."
        )
    elif checkpoint_path and not status.checkpoint_exists:
        status.errors.append(f"Selected checkpoint not found: {selected_checkpoint}")

    if resolved_python is not None and resolved_python.exists():
        probe = _run_samurai_import_probe(
            resolved_python,
            resolved_repo_path,
            timeout_sec=import_probe_timeout_sec,
        )
        _apply_import_probe_result(status, probe)

    if not status.cuda_available and status.torch_importable:
        status.errors.append("PyTorch CUDA is not available in the SAMURAI environment.")

    numpy_major = _numpy_major_version(status.numpy_version)
    if numpy_major is not None and numpy_major >= 2:
        status.warnings.append(
            "NumPy 2.x can break older compiled packages in SAM2-style stacks. "
            "Use numpy==1.26.4 if imports or torch/numpy conversion fail."
        )

    if not status.cuda_extension_importable:
        status.warnings.append(
            "SAM2 custom CUDA extension is not importable. This is acceptable for feasibility; "
            "mask post-processing may be skipped, as in the SAM2/DAM4SAM integrations."
        )

    if not status.is_ready:
        status.hints.append(
            "Run from the dedicated `samurai` Conda env or pass "
            "--python-executable C:\\Users\\rlewa\\anaconda3\\envs\\samurai\\python.exe."
        )
        status.hints.append(
            "Validated feasibility stack: Python 3.10, torch 2.5.1+cu121, "
            "torchvision 0.20.1, numpy 1.26.4."
        )

    return status


def format_samurai_environment_report(status: SamuraiEnvironmentStatus) -> str:
    available = _format_available_checkpoints(status.available_checkpoints)
    selected = str(status.selected_checkpoint) if status.selected_checkpoint else "None"
    lines = [
        "SAMURAI environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Python executable: {status.python_executable or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Model variant: {status.model_variant}",
        f"- Config name: {status.config_name or 'None'}",
        f"- Config identifier: {status.config_identifier or 'None'}",
        f"- Config path: {status.config_path or 'None'}",
        f"- Config exists: {'yes' if status.config_exists else 'no'}",
        f"- Demo script: {status.demo_script_path or 'None'}",
        f"- Demo script exists: {'yes' if status.demo_script_exists else 'no'}",
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
        f"- jpeg4py import: {'ok' if status.jpeg4py_importable else 'failed'}",
        f"- lmdb import: {'ok' if status.lmdb_importable else 'failed'}",
        f"- loguru import: {'ok' if status.loguru_importable else 'failed'}",
        f"- SAM2 import: {'ok' if status.sam2_importable else 'failed'}",
        f"- SAM2 build_sam import: {'ok' if status.build_sam_importable else 'failed'}",
        f"- SAM2 CUDA extension: {'ok' if status.cuda_extension_importable else 'missing'}",
    ]
    if status.sam2_module_path:
        lines.append(f"- SAM2 module: {status.sam2_module_path}")
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


def run_samurai_smoke_test(
    *,
    repo_path: str | Path,
    checkpoint_path: str | Path,
    python_executable: str | Path | None = None,
    model_variant: str = DEFAULT_SAMURAI_MODEL_VARIANT,
    device: str = "auto",
    timeout_sec: float = DEFAULT_SAMURAI_SMOKE_TIMEOUT_SEC,
) -> SamuraiSmokeTestResult:
    normalized_variant = _normalize_model_variant(model_variant)
    resolved_repo_path = Path(repo_path).expanduser().resolve()
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    resolved_python = _resolve_samurai_python_executable(python_executable)
    if resolved_python is None:
        raise FileNotFoundError("SAMURAI Python executable could not be resolved.")
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"SAMURAI checkpoint not found: {resolved_checkpoint}")

    command = [
        str(resolved_python),
        "-c",
        _SAMURAI_SMOKE_TEST_CODE,
        _path_for_subprocess_argument(resolved_repo_path, resolved_python),
        _path_for_subprocess_argument(resolved_checkpoint, resolved_python),
        _config_identifier_for_variant(normalized_variant),
        str(device or "auto"),
    ]
    env = _build_samurai_subprocess_env(resolved_python, resolved_repo_path)
    completed = subprocess.run(
        command,
        cwd=str(_samurai_safe_subprocess_cwd(resolved_repo_path)),
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_sec),
        env=env,
    )
    return SamuraiSmokeTestResult(
        command=command,
        model_variant=normalized_variant,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        payload=_parse_prefixed_json_stdout(completed.stdout, _SMOKE_TEST_PREFIX),
    )


def format_samurai_smoke_test_report(result: SamuraiSmokeTestResult) -> str:
    payload = result.payload
    lines = [
        "SAMURAI smoke test",
        f"- model variant: {result.model_variant}",
        f"- returncode: {result.returncode}",
        f"- ok: {'yes' if result.succeeded else 'no'}",
    ]
    for key in (
        "checkpoint",
        "config_identifier",
        "device",
        "frame_shape",
        "init_bbox_xywh",
        "init_mask_shape",
        "init_mask_area",
        "propagated_frame_count",
        "last_mask_shape",
        "last_mask_area",
        "last_mask_bbox_xyxy",
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


def _resolve_samurai_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidates = [Path(repo_path)] if repo_path else list(DEFAULT_SAMURAI_REPO_CANDIDATES)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "scripts" / "demo.py").exists() and (
            resolved / "sam2" / "sam2" / "build_sam.py"
        ).exists():
            return resolved
    return None


def _resolve_samurai_python_executable(python_executable: str | Path | None) -> Optional[Path]:
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
    model_variant: str,
) -> Optional[Path]:
    if checkpoint_path:
        candidate = Path(checkpoint_path).expanduser()
        if not candidate.is_absolute():
            candidate = models_dir / candidate
        return candidate.resolve()
    return select_default_samurai_checkpoint(checkpoints, model_variant, models_dir)


def _checkpoint_name_for_variant(model_variant: str) -> str:
    return SAMURAI_MODEL_VARIANTS[_normalize_model_variant(model_variant)][0]


def _config_name_for_variant(model_variant: str) -> str:
    return SAMURAI_MODEL_VARIANTS[_normalize_model_variant(model_variant)][1]


def _config_identifier_for_variant(model_variant: str) -> str:
    return SAMURAI_MODEL_VARIANTS[_normalize_model_variant(model_variant)][2]


def _normalize_model_variant(model_variant: str | None) -> str:
    raw = str(model_variant or DEFAULT_SAMURAI_MODEL_VARIANT).strip()
    if not raw:
        return DEFAULT_SAMURAI_MODEL_VARIANT
    name = Path(raw).name
    for suffix in (".pth.tar", ".pth", ".pt", ".tar", ".yaml", ".yml"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    key = name.strip().lower().replace("-", "_").replace("sam2_1", "sam2.1")
    key = key.replace("hiera_b+", "hiera_base_plus").replace("b+", "base_plus")
    key = _SAMURAI_VARIANT_ALIASES.get(key, key)
    if key in SAMURAI_MODEL_VARIANTS:
        return key
    return raw


def _run_samurai_import_probe(
    python_executable: Path,
    repo_path: Optional[Path],
    *,
    timeout_sec: float,
) -> SamuraiImportProbeResult:
    command = [str(python_executable), "-c", _SAMURAI_IMPORT_PROBE_CODE]
    completed = subprocess.run(
        command,
        cwd=str(_samurai_safe_subprocess_cwd(repo_path)) if repo_path is not None else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_sec),
        env=_build_samurai_subprocess_env(python_executable, repo_path),
    )
    return SamuraiImportProbeResult(
        command=command,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        payload=_parse_prefixed_json_stdout(completed.stdout, _IMPORT_PROBE_PREFIX),
    )


def _apply_import_probe_result(
    status: SamuraiEnvironmentStatus,
    probe: SamuraiImportProbeResult,
) -> None:
    status.import_probe_returncode = probe.returncode
    status.import_probe_stdout = probe.stdout
    status.import_probe_stderr = probe.stderr
    payload = probe.payload
    if probe.returncode != 0:
        status.errors.append(
            "SAMURAI import probe failed with return code "
            f"{probe.returncode}: {probe.stderr.strip() or probe.stdout.strip()}"
        )
    if not payload:
        status.errors.append("SAMURAI import probe did not return parseable diagnostics JSON.")
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
    status.jpeg4py_importable = bool(payload.get("jpeg4py_importable"))
    status.lmdb_importable = bool(payload.get("lmdb_importable"))
    status.loguru_importable = bool(payload.get("loguru_importable"))
    status.sam2_importable = bool(payload.get("sam2_importable"))
    status.sam2_module_path = payload.get("sam2_module_path")
    status.build_sam_importable = bool(payload.get("sam2_build_sam_importable"))
    status.cuda_extension_importable = bool(payload.get("sam2_cuda_extension_importable"))

    for key, value in sorted(payload.items()):
        if key.endswith("_error") and value:
            if key == "sam2_cuda_extension_error":
                continue
            status.errors.append(f"{key[:-6]} import failed: {value}")


def _parse_prefixed_json_stdout(stdout: str, prefix: str) -> dict[str, Any]:
    for line in reversed(str(stdout or "").splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    return {}


def _format_available_checkpoints(checkpoints: list[Path]) -> str:
    if not checkpoints:
        return "none"
    by_name: dict[str, list[Path]] = {}
    for path in checkpoints:
        by_name.setdefault(path.name, []).append(path)

    parts: list[str] = []
    for name in sorted(by_name, key=str.lower):
        paths = by_name[name]
        if len(paths) == 1:
            parts.append(name)
        else:
            parts.append(f"{name} ({len(paths)} paths)")
    return ", ".join(parts)


def _build_samurai_subprocess_env(
    python_executable: Path,
    repo_path: Optional[Path] = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if repo_path is not None:
        env["TRACKLEED_SAMURAI_REPO"] = _path_for_subprocess_argument(
            repo_path,
            python_executable,
        )
        current_pythonpath = env.get("PYTHONPATH", "")
        path_separator = ";" if _is_windows_python_executable(python_executable) else os.pathsep
        pythonpath_entries = [
            _path_for_subprocess_argument(repo_path / "sam2", python_executable)
        ]
        env["PYTHONPATH"] = path_separator.join(
            pythonpath_entries + ([current_pythonpath] if current_pythonpath else [])
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


def _samurai_safe_subprocess_cwd(repo_path: Path) -> Path:
    sam2_repo = repo_path / "sam2"
    return sam2_repo if sam2_repo.exists() else repo_path


def _is_windows_python_executable(python_executable: Path) -> bool:
    return str(python_executable).lower().endswith(".exe")


def _path_for_subprocess_argument(path: str | Path, python_executable: Path) -> str:
    path_text = str(Path(path).expanduser().resolve())
    if not _is_windows_python_executable(python_executable):
        return path_text
    if not path_text.startswith("/mnt/"):
        return path_text

    parts = Path(path_text).parts
    if len(parts) < 4:
        return path_text
    drive = parts[2].upper()
    tail = "\\".join(parts[3:])
    return f"{drive}:\\{tail}"


def _numpy_major_version(version: str | None) -> Optional[int]:
    if not version:
        return None
    first_chunk = str(version).split(".", 1)[0]
    try:
        return int(first_chunk)
    except ValueError:
        return None


_SAMURAI_IMPORT_PROBE_CODE = r"""
import importlib
import json
import os
import sys

PREFIX = "__TRACKLEED_SAMURAI_IMPORT_PROBE__"
result = {"python_executable": sys.executable}
repo_path = os.environ.get("TRACKLEED_SAMURAI_REPO")
if repo_path:
    sam2_root = os.path.join(repo_path, "sam2")
    if sam2_root and sam2_root not in sys.path:
        sys.path.insert(0, sam2_root)


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
probe("jpeg4py", "jpeg4py")
probe("lmdb", "lmdb")
probe("loguru", "loguru")
probe("sam2", "sam2")
probe("sam2.build_sam", "sam2_build_sam")
probe("sam2._C", "sam2_cuda_extension")

print(PREFIX + json.dumps(result, sort_keys=True))
"""


_SAMURAI_SMOKE_TEST_CODE = r"""
import json
import os
import sys
import tempfile
import time
from contextlib import nullcontext

import numpy as np
from PIL import Image, ImageDraw

PREFIX = "__TRACKLEED_SAMURAI_SMOKE_TEST__"
repo_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TRACKLEED_SAMURAI_REPO")
if repo_path:
    sam2_root = os.path.join(repo_path, "sam2")
    if sam2_root and sam2_root not in sys.path:
        sys.path.insert(0, sam2_root)

checkpoint = sys.argv[2] if len(sys.argv) > 2 else os.environ["TRACKLEED_SAMURAI_CHECKPOINT"]
config_identifier = sys.argv[3] if len(sys.argv) > 3 else os.environ["TRACKLEED_SAMURAI_CONFIG"]
device_request = (
    sys.argv[4] if len(sys.argv) > 4 else os.environ.get("TRACKLEED_SAMURAI_DEVICE", "auto")
).strip().lower()


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


def mask_from_tensor(mask_tensor):
    arr = mask_tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return arr > 0.0


start = time.perf_counter()
import torch
from sam2.build_sam import build_sam2_video_predictor

cuda_available = bool(torch.cuda.is_available())
if device_request in {"", "auto", "cuda", "cuda:0"}:
    if not cuda_available:
        raise RuntimeError("CUDA is not available for the SAMURAI smoke test.")
    device = "cuda:0"
elif device_request == "cpu":
    device = "cpu"
else:
    device = device_request

with tempfile.TemporaryDirectory(prefix="trackleed_samurai_smoke_") as temp_dir:
    frame_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    make_frame(0).save(os.path.join(frame_dir, "000000.jpg"), quality=95)
    make_frame(3).save(os.path.join(frame_dir, "000001.jpg"), quality=95)

    predictor = build_sam2_video_predictor(config_identifier, checkpoint, device=device)
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.float16)
        if str(device).startswith("cuda")
        else nullcontext()
    )
    frame_payloads = []
    init_bbox_xywh = [52, 42, 24, 24]
    init_bbox_xyxy = np.asarray([52, 42, 76, 66], dtype=np.float32)
    with torch.inference_mode(), autocast_ctx:
        state = predictor.init_state(frame_dir, offload_video_to_cpu=True)
        _, _object_ids, init_masks = predictor.add_new_points_or_box(
            state,
            box=init_bbox_xyxy,
            frame_idx=0,
            obj_id=0,
        )
        init_mask = mask_from_tensor(init_masks[0])
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            for obj_id, mask_tensor in zip(object_ids, masks):
                if int(obj_id) != 0:
                    continue
                mask = mask_from_tensor(mask_tensor)
                frame_payloads.append(
                    {
                        "frame_idx": int(frame_idx),
                        "mask_shape": list(mask.shape),
                        "mask_area": int(np.count_nonzero(mask)),
                        "mask_bbox_xyxy": mask_bbox(mask),
                    }
                )

payload = {
    "ok": bool(frame_payloads and frame_payloads[-1]["mask_area"] > 0),
    "checkpoint": str(checkpoint),
    "config_identifier": str(config_identifier),
    "device": device,
    "frame_shape": [112, 144, 3],
    "init_bbox_xywh": init_bbox_xywh,
    "init_mask_shape": list(init_mask.shape),
    "init_mask_area": int(np.count_nonzero(init_mask)),
    "propagated_frame_count": len(frame_payloads),
    "last_mask_shape": frame_payloads[-1]["mask_shape"] if frame_payloads else None,
    "last_mask_area": frame_payloads[-1]["mask_area"] if frame_payloads else 0,
    "last_mask_bbox_xyxy": frame_payloads[-1]["mask_bbox_xyxy"] if frame_payloads else None,
    "elapsed_sec": round(time.perf_counter() - start, 3),
}
print(PREFIX + json.dumps(payload, sort_keys=True))
"""
