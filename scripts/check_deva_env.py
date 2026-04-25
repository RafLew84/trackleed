#!/usr/bin/env python3
"""Print DEVA runtime diagnostics for the current Trackleed environment."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Iterator, Optional
import warnings


TRACKLEED_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKLEED_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKLEED_ROOT))

DEFAULT_DEVA_MODELS_DIR = TRACKLEED_ROOT / "tracking_models" / "deva"
DEFAULT_DEVA_REPO_CANDIDATES = (
    TRACKLEED_ROOT.parent / "deva",
    TRACKLEED_ROOT.parent / "Tracking-Anything-with-DEVA",
    TRACKLEED_ROOT / "deva",
)
DEFAULT_DEVA_CHECKPOINT_NAME = "DEVA-propagation.pth"
DEFAULT_DEVA_SMOKE_SIZE = 480
DEFAULT_DEVA_SMOKE_CHUNK_SIZE = 4


@dataclass
class DevaEnvironmentStatus:
    models_dir: Path
    repo_path: Optional[Path] = None
    available_checkpoints: list[Path] = field(default_factory=list)
    selected_checkpoint: Optional[Path] = None
    checkpoint_exists: bool = False
    checkpoint_size_bytes: Optional[int] = None
    example_vos_path: Optional[Path] = None
    example_vos_ready: bool = False
    torch_importable: bool = False
    torch_version: Optional[str] = None
    torch_cuda_version: Optional[str] = None
    cuda_available: bool = False
    torchvision_importable: bool = False
    pillow_importable: bool = False
    numpy_importable: bool = False
    tqdm_importable: bool = False
    deva_importable: bool = False
    deva_module_path: Optional[str] = None
    deva_network_importable: bool = False
    deva_inference_core_importable: bool = False
    deva_general_vos_dataset_importable: bool = False
    groundingdino_importable: bool = False
    segment_anything_importable: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return (
            self.repo_path is not None
            and self.checkpoint_exists
            and self.example_vos_ready
            and self.torch_importable
            and self.cuda_available
            and self.torchvision_importable
            and self.pillow_importable
            and self.numpy_importable
            and self.tqdm_importable
            and self.deva_importable
            and self.deva_network_importable
            and self.deva_inference_core_importable
            and self.deva_general_vos_dataset_importable
        )


@dataclass
class DevaSmokeTestResult:
    command: list[str]
    checkpoint_name: str
    example_dataset_path: Path
    output_dir: Path
    returncode: int
    elapsed_sec: float
    processed_frame_count: Optional[int]
    total_process_time_sec: Optional[float]
    fps: Optional[float]
    max_allocated_memory_mb: Optional[float]
    output_mask_count: int


def discover_deva_checkpoints(
    models_dir: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> list[Path]:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_DEVA_MODELS_DIR
    discovered: dict[str, Path] = {}

    for candidate in _discover_checkpoints_in_directory(directory):
        discovered[str(candidate.resolve())] = candidate.resolve()

    return sorted(discovered.values(), key=lambda path: path.name.lower())


def select_default_deva_checkpoint(checkpoints: list[Path]) -> Optional[Path]:
    if not checkpoints:
        return None
    by_name = {path.name: path for path in checkpoints}
    if DEFAULT_DEVA_CHECKPOINT_NAME in by_name:
        return by_name[DEFAULT_DEVA_CHECKPOINT_NAME]
    return checkpoints[0]


def probe_deva_environment(
    *,
    models_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> DevaEnvironmentStatus:
    directory = Path(models_dir) if models_dir is not None else DEFAULT_DEVA_MODELS_DIR
    directory = directory.resolve()
    resolved_repo_path = _resolve_deva_repo_path(repo_path)
    checkpoints = discover_deva_checkpoints(directory, resolved_repo_path)
    selected_checkpoint = _resolve_checkpoint_path(
        checkpoint_path,
        checkpoints,
        directory,
        resolved_repo_path,
    )
    example_vos_path = _resolve_example_vos_path(resolved_repo_path)

    status = DevaEnvironmentStatus(
        models_dir=directory,
        repo_path=resolved_repo_path,
        available_checkpoints=checkpoints,
        selected_checkpoint=selected_checkpoint,
        checkpoint_exists=bool(selected_checkpoint and selected_checkpoint.exists()),
        checkpoint_size_bytes=(
            selected_checkpoint.stat().st_size
            if selected_checkpoint is not None and selected_checkpoint.exists()
            else None
        ),
        example_vos_path=example_vos_path,
        example_vos_ready=bool(example_vos_path and _is_generic_vos_dataset(example_vos_path)),
    )

    if repo_path and resolved_repo_path is None:
        status.errors.append(f"Could not resolve DEVA repo from: {repo_path}")
    elif resolved_repo_path is None:
        status.errors.append(
            "Could not resolve DEVA repo. Pass --repo-path pointing at the local DEVA clone."
        )

    if not directory.exists():
        status.warnings.append(
            f"Checkpoint directory does not exist yet: {directory}. "
            "Place DEVA checkpoints inside tracking_models/deva."
        )

    if selected_checkpoint is None:
        status.errors.append(
            "No DEVA checkpoint detected in tracking_models/deva."
        )
    elif checkpoint_path and not status.checkpoint_exists:
        status.errors.append(f"Selected checkpoint not found: {selected_checkpoint}")

    if example_vos_path is None:
        status.warnings.append("DEVA example/vos dataset was not found.")
    elif not status.example_vos_ready:
        status.warnings.append(
            f"DEVA example/vos dataset is incomplete: {example_vos_path}"
        )

    try:
        torch = importlib.import_module("torch")
        status.torch_importable = True
        status.torch_version = getattr(torch, "__version__", None)
        status.torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        status.cuda_available = bool(torch.cuda.is_available())
        if not status.cuda_available:
            status.errors.append(
                "CUDA is not available. The official DEVA VOS evaluation path calls CUDA directly."
            )
    except Exception as error:
        status.errors.append(f"torch import failed: {error}")

    status.torchvision_importable = _try_import("torchvision", status.errors, "torchvision")
    status.pillow_importable = _try_import("PIL.Image", status.errors, "Pillow")
    status.numpy_importable = _try_import("numpy", status.errors, "NumPy")
    status.tqdm_importable = _try_import("tqdm", status.errors, "tqdm")

    with _deva_import_context(resolved_repo_path):
        deva_module = _try_import_module("deva")
        status.deva_importable = deva_module is not None
        if deva_module is not None:
            status.deva_module_path = getattr(deva_module, "__file__", None)
        else:
            status.errors.append("DEVA import failed: No module named 'deva'")

        status.deva_network_importable = _try_import(
            "deva.model.network",
            status.errors,
            "DEVA model network",
        )
        status.deva_inference_core_importable = _try_import(
            "deva.inference.inference_core",
            status.errors,
            "DEVA inference core",
        )
        status.deva_general_vos_dataset_importable = _try_import(
            "deva.inference.data.vos_test_datasets",
            status.errors,
            "DEVA VOS dataset loader",
        )
        status.groundingdino_importable = _try_import(
            "groundingdino.util.inference",
            [],
            "GroundingDINO",
        )
        status.segment_anything_importable = _try_import(
            "segment_anything",
            [],
            "Segment Anything",
        )

    if not status.groundingdino_importable:
        status.warnings.append(
            "GroundingDINO is not importable. This is OK for seeded VOS smoke tests, "
            "but text-prompted/automatic DEVA demos will need it."
        )
    if not status.segment_anything_importable:
        status.warnings.append(
            "segment_anything is not importable. This is OK for seeded VOS smoke tests, "
            "but automatic/text-prompted DEVA demos will need it."
        )

    if not status.is_ready:
        status.hints.append(
            "Run this checker from the dedicated `deva` Conda env and pass "
            "--repo-path C:\\Users\\rlewa\\Documents\\PROJEKTY\\deva if auto-detection fails."
        )
        status.hints.append(
            "For the first Trackleed integration path, only DEVA core and "
            "DEVA-propagation.pth are required; GroundingDINO/SAM can stay optional."
        )

    return status


def format_deva_environment_report(status: DevaEnvironmentStatus) -> str:
    lines = [
        "DEVA environment diagnostics",
        f"- Ready: {'yes' if status.is_ready else 'no'}",
        f"- Repo path: {status.repo_path or 'None'}",
        f"- Models dir: {status.models_dir}",
        f"- Selected checkpoint: {status.selected_checkpoint or 'None'}",
        f"- Selected checkpoint exists: {'yes' if status.checkpoint_exists else 'no'}",
        f"- Selected checkpoint size MB: {_format_size_mb(status.checkpoint_size_bytes)}",
        f"- Available checkpoints: {_format_checkpoint_paths(status.available_checkpoints)}",
        f"- Example VOS path: {status.example_vos_path or 'None'}",
        f"- Example VOS ready: {'yes' if status.example_vos_ready else 'no'}",
        f"- PyTorch import: {'ok' if status.torch_importable else 'failed'}",
        f"- PyTorch version: {status.torch_version or 'unknown'}",
        f"- PyTorch CUDA version: {status.torch_cuda_version or 'unknown'}",
        f"- CUDA available: {'yes' if status.cuda_available else 'no'}",
        f"- torchvision import: {'ok' if status.torchvision_importable else 'failed'}",
        f"- Pillow import: {'ok' if status.pillow_importable else 'failed'}",
        f"- NumPy import: {'ok' if status.numpy_importable else 'failed'}",
        f"- tqdm import: {'ok' if status.tqdm_importable else 'failed'}",
        f"- DEVA import: {'ok' if status.deva_importable else 'failed'}",
        f"- DEVA module: {status.deva_module_path or 'unknown'}",
        f"- DEVA network import: {'ok' if status.deva_network_importable else 'failed'}",
        f"- DEVA inference core import: {'ok' if status.deva_inference_core_importable else 'failed'}",
        "- DEVA generic VOS dataset import: "
        + ("ok" if status.deva_general_vos_dataset_importable else "failed"),
        f"- GroundingDINO import: {'ok' if status.groundingdino_importable else 'missing'}",
        f"- segment_anything import: {'ok' if status.segment_anything_importable else 'missing'}",
    ]

    if status.errors:
        lines.append("- Errors:")
        lines.extend(f"  - {error}" for error in status.errors)
    if status.warnings:
        lines.append("- Warnings:")
        lines.extend(f"  - {warning}" for warning in status.warnings)
    if status.hints:
        lines.append("- Hints:")
        lines.extend(f"  - {hint}" for hint in status.hints)
    return "\n".join(lines)


def run_deva_smoke_test(
    *,
    checkpoint_path: str | Path,
    repo_path: str | Path,
    size: int = DEFAULT_DEVA_SMOKE_SIZE,
    chunk_size: int = DEFAULT_DEVA_SMOKE_CHUNK_SIZE,
    amp: bool = True,
    output_dir: str | Path | None = None,
    timeout_sec: float = 600.0,
) -> DevaSmokeTestResult:
    repo_root = Path(repo_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    example_vos_path = repo_root / "example" / "vos"
    if not _is_generic_vos_dataset(example_vos_path):
        raise RuntimeError(f"DEVA example/vos dataset is missing or incomplete: {example_vos_path}")

    if output_dir is None:
        with TemporaryDirectory(prefix="trackleed_deva_smoke_") as temp_dir:
            return _run_deva_smoke_test_command(
                checkpoint=checkpoint,
                repo_root=repo_root,
                example_vos_path=example_vos_path,
                output_dir=Path(temp_dir),
                size=size,
                chunk_size=chunk_size,
                amp=amp,
                timeout_sec=timeout_sec,
            )

    return _run_deva_smoke_test_command(
        checkpoint=checkpoint,
        repo_root=repo_root,
        example_vos_path=example_vos_path,
        output_dir=Path(output_dir).resolve(),
        size=size,
        chunk_size=chunk_size,
        amp=amp,
        timeout_sec=timeout_sec,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check DEVA runtime availability for Trackleed.")
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path to the local DEVA repository root.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_DEVA_MODELS_DIR),
        help="Directory containing DEVA checkpoints for Trackleed.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit DEVA checkpoint path or name.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny generic VOS smoke test on DEVA's bundled example/vos sequence.",
    )
    parser.add_argument(
        "--backend-smoke-test",
        action="store_true",
        help="Run Trackleed's DEVA backend contract on a tiny synthetic seeded sequence.",
    )
    parser.add_argument(
        "--smoke-size",
        type=int,
        default=DEFAULT_DEVA_SMOKE_SIZE,
        help="Shortest side used by the DEVA smoke test.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_DEVA_SMOKE_CHUNK_SIZE,
        help="DEVA chunk_size passed to the smoke test.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision in the smoke test.",
    )
    parser.add_argument(
        "--smoke-output-dir",
        default=None,
        help="Optional output directory for the smoke-test masks. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--subprocess-timeout-sec",
        type=float,
        default=600.0,
        help="Maximum time allowed for the smoke-test subprocess.",
    )
    args = parser.parse_args()

    status = probe_deva_environment(
        models_dir=args.models_dir,
        checkpoint_path=args.checkpoint,
        repo_path=args.repo_path,
    )
    print(format_deva_environment_report(status))
    if not status.is_ready:
        return 1

    if args.smoke_test:
        result = run_deva_smoke_test(
            checkpoint_path=status.selected_checkpoint,
            repo_path=status.repo_path,
            size=args.smoke_size,
            chunk_size=args.chunk_size,
            amp=not args.no_amp,
            output_dir=args.smoke_output_dir,
            timeout_sec=args.subprocess_timeout_sec,
        )
        print("Smoke test:")
        print(f"- checkpoint: {result.checkpoint_name}")
        print(f"- example dataset: {result.example_dataset_path}")
        print(f"- output dir: {result.output_dir}")
        print(f"- returncode: {result.returncode}")
        print(f"- elapsed sec: {result.elapsed_sec:.3f}")
        print(f"- processed frame count: {result.processed_frame_count}")
        print(f"- total process time sec: {result.total_process_time_sec}")
        print(f"- FPS: {result.fps}")
        print(f"- max allocated memory MB: {result.max_allocated_memory_mb}")
        print(f"- output mask count: {result.output_mask_count}")

    if args.backend_smoke_test:
        result = _run_trackleed_deva_backend_smoke_test(
            checkpoint_path=status.selected_checkpoint,
            repo_path=status.repo_path,
            size=args.smoke_size,
            chunk_size=args.chunk_size,
            amp=not args.no_amp,
            timeout_sec=args.subprocess_timeout_sec,
        )
        print("Trackleed backend smoke test:")
        print(f"- checkpoint: {result.checkpoint_name}")
        print(f"- backend mode: {result.backend_mode}")
        print(f"- representative point mode: {result.representative_point_mode}")
        print(f"- initial mask mode: {result.initial_mask_mode}")
        print(f"- masks shape: {result.masks.shape}")
        print(f"- tracks shape: {result.tracks_xy.shape}")
        print(f"- visible count: {int(result.visible_mask.sum())}")
        print(f"- first track point: {result.tracks_xy[0, 0].tolist()}")
        print(f"- last visible point: {_last_visible_point_text(result.tracks_xy, result.visible_mask)}")

    return 0


def _run_trackleed_deva_backend_smoke_test(
    *,
    checkpoint_path: Path,
    repo_path: Path,
    size: int,
    chunk_size: int,
    amp: bool,
    timeout_sec: float,
):
    import numpy as np

    from trackleed.analysis.deva_backend import DevaRuntimeSettings, run_deva_offline

    frames = _build_synthetic_deva_backend_frames()
    query_points_tyx = np.asarray([[0.0, 56.0, 58.0]], dtype=np.float32)
    query_boxes_xyxy = np.asarray([[42.0, 44.0, 74.0, 68.0]], dtype=np.float32)
    return run_deva_offline(
        frames,
        query_points_tyx,
        DevaRuntimeSettings(
            checkpoint_path=checkpoint_path,
            repo_path=repo_path,
            python_executable=sys.executable,
            backend_mode="subprocess",
            representative_point_mode="mask_centroid",
            inference_size=int(size),
            initial_bbox_padding_px=0.0,
            amp=bool(amp),
            chunk_size=int(chunk_size),
            subprocess_timeout_sec=float(timeout_sec),
        ),
        query_boxes_xyxy=query_boxes_xyxy,
    )


def _build_synthetic_deva_backend_frames():
    import numpy as np

    height, width = 112, 144
    yy, xx = np.mgrid[:height, :width]
    frames = []
    for frame_idx in range(4):
        center_x = 58 + (frame_idx * 4)
        center_y = 56 + (frame_idx * 2)
        blob = np.exp(-(((xx - center_x) ** 2) / 150.0 + ((yy - center_y) ** 2) / 90.0))
        image = np.clip(blob * 255.0, 0.0, 255.0).astype(np.uint8)
        rgb = np.stack([image, image, image], axis=-1)
        frames.append(rgb)
    return np.stack(frames, axis=0)


def _last_visible_point_text(tracks_xy, visible_mask) -> str:
    import numpy as np

    visible_indices = np.argwhere(np.asarray(visible_mask, dtype=bool))
    if visible_indices.size == 0:
        return "None"
    object_index, frame_index = visible_indices[-1].tolist()
    return str(np.asarray(tracks_xy[object_index, frame_index], dtype=float).tolist())


def _run_deva_smoke_test_command(
    *,
    checkpoint: Path,
    repo_root: Path,
    example_vos_path: Path,
    output_dir: Path,
    size: int,
    chunk_size: int,
    amp: bool,
    timeout_sec: float,
) -> DevaSmokeTestResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(repo_root / "evaluation" / "eval_vos.py"),
        "--dataset",
        "G",
        "--generic_path",
        str(example_vos_path),
        "--model",
        str(checkpoint),
        "--output",
        str(output_dir),
        "--size",
        str(size),
        "--chunk_size",
        str(chunk_size),
    ]
    if amp:
        command.append("--amp")

    started_at = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    elapsed_sec = time.perf_counter() - started_at
    combined_output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        raise RuntimeError(
            "DEVA smoke test failed:\n"
            + json.dumps(
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout_tail": _tail_text(completed.stdout),
                    "stderr_tail": _tail_text(completed.stderr),
                },
                indent=2,
            )
        )

    return DevaSmokeTestResult(
        command=command,
        checkpoint_name=checkpoint.name,
        example_dataset_path=example_vos_path,
        output_dir=output_dir,
        returncode=completed.returncode,
        elapsed_sec=elapsed_sec,
        processed_frame_count=_parse_int_metric(combined_output, r"Total processed frames:\s*(\d+)"),
        total_process_time_sec=_parse_float_metric(
            combined_output,
            r"Total processing time:\s*([0-9.eE+-]+)",
        ),
        fps=_parse_float_metric(combined_output, r"FPS:\s*([0-9.eE+-]+)"),
        max_allocated_memory_mb=_parse_float_metric(
            combined_output,
            r"Max allocated memory \(MB\):\s*([0-9.eE+-]+)",
        ),
        output_mask_count=sum(1 for _ in output_dir.rglob("*.png")),
    )


def _discover_checkpoints_in_directory(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".pth", ".pt"}
    ]


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoints: list[Path],
    models_dir: Path,
    repo_path: Optional[Path],
) -> Optional[Path]:
    if checkpoint_path is None:
        return select_default_deva_checkpoint(checkpoints)

    candidate = Path(checkpoint_path).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(models_dir / candidate)

    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0].resolve()


def _resolve_deva_repo_path(repo_path: str | Path | None) -> Optional[Path]:
    candidates: list[Path] = []
    if repo_path is not None:
        candidates.append(Path(repo_path).expanduser())
    candidates.extend(DEFAULT_DEVA_REPO_CANDIDATES)
    candidates.append(Path.cwd())

    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "deva").is_dir() and (resolved / "evaluation" / "eval_vos.py").is_file():
            return resolved
    return None


def _resolve_example_vos_path(repo_path: Optional[Path]) -> Optional[Path]:
    if repo_path is None:
        return None
    candidate = repo_path / "example" / "vos"
    return candidate if candidate.exists() else None


def _is_generic_vos_dataset(path: Path) -> bool:
    jpeg_root = path / "JPEGImages"
    annotation_root = path / "Annotations"
    if not jpeg_root.is_dir() or not annotation_root.is_dir():
        return False
    return any(jpeg_root.glob("*/*.jpg")) and any(annotation_root.glob("*/*.png"))


@contextmanager
def _deva_import_context(repo_path: Optional[Path]) -> Iterator[None]:
    if repo_path is None:
        yield
        return

    original_sys_path = list(sys.path)
    repo_path_str = str(repo_path)
    if repo_path_str not in sys.path:
        sys.path.insert(0, repo_path_str)
    try:
        yield
    finally:
        sys.path[:] = original_sys_path


def _try_import(module_name: str, errors: list[str], label: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception as error:
        errors.append(f"{label} import failed: {error}")
        return False


def _try_import_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _format_size_mb(size_bytes: Optional[int]) -> str:
    if size_bytes is None:
        return "unknown"
    return f"{size_bytes / (1024 * 1024):.1f}"


def _format_checkpoint_paths(paths: list[Path]) -> str:
    if not paths:
        return "None"
    names = [path.name for path in paths]
    if len(set(names)) == len(names):
        return ", ".join(names)
    return ", ".join(str(path) for path in paths)


def _parse_int_metric(text: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, text)
    if not match:
        return None
    return int(match.group(1))


def _parse_float_metric(text: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))


def _tail_text(text: str, max_lines: int = 80) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        raise SystemExit(main())
