from __future__ import annotations

import ast
import unittest
from pathlib import Path

TRACKLEED_ROOT = Path(__file__).resolve().parents[1]
TRACKING_BBOX_BACKENDS_PATH = TRACKLEED_ROOT / "analysis" / "tracking_bbox_backends.py"
TOMP_BACKEND_PATH = TRACKLEED_ROOT / "analysis" / "tomp_backend.py"


def _load_module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _extract_function_source(path: Path, function_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            end_lineno = getattr(node, "end_lineno", None)
            if end_lineno is None:
                raise ValueError(f"Python AST did not provide end_lineno for `{function_name}`.")
            return "\n".join(lines[node.lineno - 1 : end_lineno])
    raise KeyError(f"Could not find function `{function_name}` in {path}.")


def _extract_imported_names(path: Path) -> set[str]:
    module = _load_module_ast(path)
    imported_names: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
    return imported_names


class TrackerRefactorReadinessTests(unittest.TestCase):
    def test_tapir_legacy_runtime_does_not_require_polygon_seed(self) -> None:
        tapir_source = _extract_function_source(
            TRACKLEED_ROOT / "analysis" / "tracking.py",
            "_legacy_track_spots_tapir_offline",
        )
        self.assertNotIn("_collect_missing_polygon_seed_spot_ids", tapir_source)
        self.assertNotIn("Offline Cutie requires a polygon seed", tapir_source)

    def test_tomp_bbox_backend_imports_quality_filter_reader(self) -> None:
        imported_names = _extract_imported_names(TRACKING_BBOX_BACKENDS_PATH)
        self.assertIn("read_bbox_quality_filter_params", imported_names)

    def test_tomp_subprocess_backend_does_not_import_pyqt_bound_tracking_helpers(self) -> None:
        tomp_backend_source = TOMP_BACKEND_PATH.read_text(encoding="utf-8")
        self.assertNotIn("tracking_filter_utils", tomp_backend_source)


if __name__ == "__main__":
    unittest.main()
