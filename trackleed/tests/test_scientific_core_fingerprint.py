from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from trackleed.publication.scientific_core import (
    ScientificCoreMismatchError,
    ScientificCoreValidationError,
    freeze_fingerprint,
    validate_fingerprint,
)
from trackleed.publication.scientific_reconstruction import (
    rebuild_reader_facing_artifacts,
    validate_reader_facing_artifacts,
)


REQUIRED_ROLES = (
    "builders",
    "central_outcomes",
    "claim_boundaries",
    "claim_evidence_map",
    "denominators",
    "displays",
    "final_prose",
    "limitations",
    "provenance",
    "sensitivity",
    "si_inventory",
    "source_data",
    "validators",
)
REPO_ROOT = Path(__file__).resolve().parents[2]


class ScientificCoreFingerprintTests(unittest.TestCase):
    def test_fingerprint_covers_renderer_independent_display_semantics(self) -> None:
        package = REPO_ROOT / "manuscript" / "scientific_core"
        contract = json.loads(
            (package / "fingerprint_contract.json").read_text(encoding="utf-8")
        )
        component = next(
            item
            for item in contract["components"]
            if item["id"] == "display-semantic-inventory"
        )
        self.assertEqual(
            set(component["roles"]),
            {
                "builders",
                "denominators",
                "displays",
                "si_inventory",
                "source_data",
                "validators",
            },
        )
        semantic = json.loads(
            (REPO_ROOT / component["path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(len(semantic["bundles"]), 20)
        self.assertEqual(semantic["status"], "scientific-baseline-complete")
        for bundle in semantic["bundles"].values():
            self.assertTrue(bundle["source_data_sha256"])
            self.assertTrue(bundle["contracts_sha256"]["builder"])
            self.assertTrue(bundle["contracts_sha256"]["caption"])
            self.assertTrue(bundle["contracts_sha256"]["validators"])
            self.assertTrue(bundle["contracts_sha256"]["manifests"])
            self.assertFalse(
                any(
                    Path(path).suffix.lower() in {".pdf", ".png"}
                    for path in bundle["text_outputs_sha256"]
                )
            )

    def test_reconstruction_runs_every_display_builder_headless(self) -> None:
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "trackleed.publication.scientific_reconstruction.subprocess.run",
            return_value=completed,
        ) as run:
            report = rebuild_reader_facing_artifacts(REPO_ROOT)

        self.assertEqual(report.bundle_count, 20)
        self.assertEqual(run.call_count, 21)
        for call in run.call_args_list:
            environment = call.kwargs["env"]
            self.assertEqual(environment["MPLBACKEND"], "Agg")
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(environment["QT_QPA_PLATFORM"], "offscreen")
        invoked_scripts = [Path(call.args[0][1]) for call in run.call_args_list]
        self.assertTrue(all(path.is_relative_to(REPO_ROOT) for path in invoked_scripts))
        self.assertEqual(
            invoked_scripts[-1],
            REPO_ROOT / "manuscript" / "publication_inventory" / "build_inventory.py",
        )

    def test_reader_facing_inventory_validates_without_rewriting(self) -> None:
        index_path = REPO_ROOT / "manuscript" / "figures-and-tables.md"
        manifest_path = (
            REPO_ROOT / "manuscript" / "publication_inventory" / "manifest.json"
        )
        before = (index_path.read_bytes(), manifest_path.read_bytes())

        report = validate_reader_facing_artifacts(REPO_ROOT)

        self.assertEqual(report.bundle_count, 20)
        self.assertEqual(
            (index_path.read_bytes(), manifest_path.read_bytes()),
            before,
        )

    def test_failed_reconstruction_restores_the_committed_artifacts(self) -> None:
        protected_paths = (
            REPO_ROOT / "manuscript" / "publication_inventory" / "manifest.json",
            REPO_ROOT / "manuscript" / "manuscript.md",
            REPO_ROOT / "manuscript" / "supplementary-material.md",
        )
        baseline = {path: path.read_bytes() for path in protected_paths}

        def fail_after_mutation(*_args: object, **_kwargs: object) -> None:
            for path in protected_paths:
                path.write_text("mutated during rebuild\n", encoding="utf-8")
            raise ScientificCoreValidationError("simulated builder failure")

        try:
            with mock.patch(
                "trackleed.publication.scientific_reconstruction._run_builder",
                side_effect=fail_after_mutation,
            ):
                with self.assertRaises(ScientificCoreValidationError):
                    rebuild_reader_facing_artifacts(REPO_ROOT)

            self.assertEqual(
                {path: path.read_bytes() for path in protected_paths},
                baseline,
            )
        finally:
            for path, content in baseline.items():
                path.write_bytes(content)

    def test_fingerprint_fails_closed_without_rewriting_the_baseline(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            component = workspace / "core.txt"
            component.write_text("approved scientific core\n", encoding="utf-8")
            contract = workspace / "fingerprint_contract.json"
            contract.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "frozen",
                        "required_roles": list(REQUIRED_ROLES),
                        "components": [
                            {
                                "id": "complete-fixture",
                                "path": "core.txt",
                                "roles": list(REQUIRED_ROLES),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            fingerprint = workspace / "fingerprint.json"

            frozen = freeze_fingerprint(workspace, contract, fingerprint)
            self.assertEqual(frozen.status, "scientific-baseline-complete")
            self.assertEqual(
                validate_fingerprint(workspace, contract, fingerprint), frozen
            )
            baseline_bytes = fingerprint.read_bytes()

            component.write_text("silently changed core\n", encoding="utf-8")

            with self.assertRaises(ScientificCoreMismatchError):
                validate_fingerprint(workspace, contract, fingerprint)
            self.assertEqual(fingerprint.read_bytes(), baseline_bytes)


if __name__ == "__main__":
    unittest.main()
