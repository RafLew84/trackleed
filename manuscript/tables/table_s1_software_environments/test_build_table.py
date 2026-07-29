#!/usr/bin/env python3
"""Contract tests for supplementary Table S1."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import build_table
import capture_environments


TABLE_ROOT = Path(__file__).resolve().parent
SUPPLEMENT_PATH = TABLE_ROOT.parents[1] / "supplementary-material.md"
SHARED_SAM2_CHECKPOINT = (
    "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
)
FAMILY_CONTRACT_SHA256 = (
    "ee579ab76263a3ee20c227f79f83bc1c1d6f26793718209a0b517f5e5131a60c"
)


class SoftwareEnvironmentsTableTest(unittest.TestCase):
    def test_components_preserve_versions_variants_and_roles(self) -> None:
        rows = build_table.load_and_validate_components()

        self.assertEqual(len(rows), 10)
        self.assertEqual(
            [row["component"] for row in rows],
            [
                "TrackLEED common measurement",
                "LocoTrack",
                "OSTrack",
                "DAM4SAM Large",
                "Lucas–Kanade",
                "EasyLEED",
                "SAM2.1",
                "SAMURAI",
                "DAM4SAM Base+",
                "ViPErLEED",
            ],
        )
        self.assertEqual(rows[5]["software_version"], "2.5.2")
        self.assertEqual(rows[9]["software_version"], "1.03")
        self.assertEqual(
            rows[1]["upstream_revision"],
            "3385a233a270160b253a69050c6a3934527f7775",
        )

    def test_controlled_family_shares_weights_and_labels_normalized_records(
        self,
    ) -> None:
        rows = build_table.load_and_validate_components()
        family = rows[6:9]

        self.assertEqual(
            {row["artifact_sha256"] for row in family},
            {SHARED_SAM2_CHECKPOINT},
        )
        self.assertEqual(
            {
                row["configuration_record_status"] for row in family
            },
            {"post-run normalized from executed source"},
        )
        for row in family:
            path = (
                build_table.CONFIGURATION_RECORDS_DIRECTORY
                / build_table.CONTROLLED_FAMILY_CONFIGURATION_FILES[
                    row["component"]
                ]
            )
            self.assertEqual(
                row["configuration_record_sha256"],
                build_table.sha256_file(path),
            )
            self.assertIn(
                build_table.CONTROLLED_FAMILY_RUNNER_SHA256,
                row["implementation_record_sha256"],
            )
        for row in rows:
            self.assertNotIn(
                FAMILY_CONTRACT_SHA256,
                row["configuration_record_sha256"],
            )

    def test_independent_energy_variants_have_separate_config_records(
        self,
    ) -> None:
        rows = {
            row["component"]: row
            for row in build_table.load_and_validate_components()
        }

        for component, filename in (
            build_table.INDEPENDENT_ENERGY_CONFIGURATION_FILES.items()
        ):
            row = rows[component]
            normalized_path = (
                build_table.CONFIGURATION_RECORDS_DIRECTORY / filename
            )
            self.assertEqual(
                row["configuration_record_status"],
                build_table.MIXED_MAIN_CONFIGURATION_STATUS,
            )
            self.assertIn(
                build_table.sha256_file(normalized_path),
                build_table._split_hashes(
                    row["configuration_record_sha256"]
                ),
            )
            self.assertIn(
                build_table.INDEPENDENT_ENERGY_RUNNER_SHA256,
                build_table._split_hashes(
                    row["implementation_record_sha256"]
                ),
            )

    def test_easyleed_archives_each_adapter_and_configuration_regime(
        self,
    ) -> None:
        row = build_table.load_and_validate_components()[5]

        self.assertEqual(
            set(
                build_table._split_hashes(
                    row["implementation_record_sha256"]
                )
            ),
            build_table.EASYLEED_IMPLEMENTATION_HASHES,
        )
        self.assertEqual(
            set(
                build_table._split_hashes(
                    row["configuration_record_sha256"]
                )
            ),
            {
                build_table.EASYLEED_RUN_FROZEN_CONFIGURATION_SHA256,
                *(
                    build_table.sha256_file(
                        build_table.CONFIGURATION_RECORDS_DIRECTORY / filename
                    )
                    for filename in build_table.EASYLEED_CONFIGURATION_FILES
                ),
            },
        )

    def test_environment_snapshots_are_complete_and_path_free(self) -> None:
        records = build_table.load_and_validate_environments()

        self.assertEqual(len(records), 8)
        by_label = {record["environment_label"]: record for record in records}
        self.assertEqual(by_label["netcat"]["python_version"], "3.12.9")
        self.assertEqual(
            build_table.package_version(by_label["netcat"], "numpy"),
            "2.2.5",
        )
        self.assertEqual(
            build_table.package_version(by_label["ostrack"], "torch"),
            "1.13.1+cu117",
        )
        self.assertEqual(
            build_table.package_version(by_label["dam4sam"], "torch"),
            "2.1.0+cu121",
        )
        self.assertEqual(by_label["sam2"]["torch_cuda_build"], "12.1")
        self.assertEqual(
            by_label["viperleed-imagej"]["runtime_components"][0]["version"],
            "1.54u",
        )
        components = build_table.load_and_validate_components()
        self.assertEqual(
            components[-1]["implementation_record_sha256"],
            "not applicable",
        )
        self.assertEqual(
            by_label["viperleed-imagej"]["runtime_components"][0][
                "artifact_sha256"
            ],
            "6acc016e4630d5c5ecd470b205f8d1abc08ce57568c6ab617a308d2451d9c90d",
        )

        serialized = json.dumps(records)
        self.assertNotIn("C:\\\\Users", serialized)
        self.assertNotIn("/mnt/c/Users", serialized)
        for label in build_table.PYTHON_ENVIRONMENT_LABELS:
            record = by_label[label]
            self.assertGreater(len(record["installed_distributions"]), 20)
            self.assertGreater(len(record["conda_packages"]), 10)
            self.assertEqual(len(record["conda_history_sha256"]), 64)

    def test_capture_rejects_local_or_credentialed_conda_channels(self) -> None:
        for unsafe_channel in (
            "file:///D:/Users/Alice/private",
            "C:/Users/Alice/channel",
            "C:\\Users\\Alice\\channel",
            "\\\\server\\share\\channel",
            "https://alice:secret@conda.anaconda.org/private/win-64",
            "https://private.example.org/channel/win-64",
        ):
            with self.subTest(channel=unsafe_channel):
                with self.assertRaises(ValueError):
                    capture_environments._safe_channel(unsafe_channel)

        self.assertEqual(
            capture_environments._safe_channel(
                "https://conda.anaconda.org/conda-forge/win-64"
            ),
            "https://conda.anaconda.org/conda-forge/win-64",
        )
        self.assertEqual(
            capture_environments._safe_channel(
                "https://repo.anaconda.com/pkgs/main/win-64"
            ),
            "https://repo.anaconda.com/pkgs/main/win-64",
        )

    def test_hash_addressed_text_records_use_stable_lf_endings(
        self,
    ) -> None:
        attributes = (TABLE_ROOT / ".gitattributes").read_text(
            encoding="utf-8"
        )

        for extension in ("csv", "json", "md", "py"):
            self.assertIn(
                f"*.{extension} text eol=lf",
                attributes,
            )

    def test_rendered_table_has_three_blocks_and_capture_caveat(self) -> None:
        components = build_table.load_and_validate_components()
        environments = build_table.build_environment_rows(
            build_table.load_and_validate_environments()
        )
        rendered = build_table.render_markdown(components, environments)

        self.assertIn("**A. Software and model identity**", rendered)
        self.assertIn(
            "**B. Configuration and implementation hashes**",
            rendered,
        )
        self.assertIn("**C. Retained execution environments**", rendered)
        self.assertIn(
            "captured on 2026-07-29 after the completed runs",
            rendered,
        )
        self.assertIn(
            "does not establish bitwise reproducibility of GPU inference",
            rendered,
        )
        self.assertIn(
            "post-run normalized record derived from the archived executed "
            "source",
            rendered,
        )
        self.assertIn("NVIDIA GeForce RTX 4070 Laptop GPU", rendered)
        self.assertIn("driver 610.62", rendered)
        self.assertNotIn("ScanE", rendered)
        self.assertNotIn("26a21", rendered)
        self.assertNotIn("C:\\Users", rendered)

    def test_provenance_uses_hash_checked_committed_snapshots(self) -> None:
        provenance = build_table.validate_provenance()

        self.assertEqual(len(provenance["sources"]), 9)
        for source in provenance["sources"]:
            frozen_path = build_table.TABLE_ROOT / source["frozen_path"]
            self.assertTrue(frozen_path.is_file())
            self.assertNotIn(".scratch", source["frozen_path"])
            self.assertEqual(
                build_table.sha256_file(frozen_path),
                source["frozen_sha256"],
            )
            origin_path = (
                build_table.REPOSITORY_ROOT / source["origin_path"]
            )
            self.assertTrue(origin_path.is_file())
            self.assertEqual(
                build_table.sha256_file(origin_path),
                source["origin_sha256"],
            )

    def test_artifact_is_deterministic_and_records_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "output"
            manifest_path = root / "manifest.json"

            first = build_table.build_artifact(
                output_directory=output_directory,
                manifest_path=manifest_path,
            )
            first_bytes = {
                path.name: path.read_bytes()
                for path in output_directory.iterdir()
            }
            second = build_table.build_artifact(
                output_directory=output_directory,
                manifest_path=manifest_path,
            )
            second_bytes = {
                path.name: path.read_bytes()
                for path in output_directory.iterdir()
            }

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(
                set(first_bytes),
                {
                    "table_s1_components.csv",
                    "table_s1_environments.csv",
                    "table_s1_versions_hashes_environments.md",
                },
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            contract = manifest["table_contract"]
            self.assertEqual(manifest["schema_version"], "1.1")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(contract["component_count"], 10)
            self.assertEqual(contract["environment_count"], 8)
            self.assertEqual(contract["python_environment_count"], 7)
            self.assertEqual(contract["imagej_environment_count"], 1)
            self.assertTrue(contract["controlled_family_weights_identical"])
            self.assertTrue(
                contract["configuration_record_provenance_explicit"]
            )
            self.assertEqual(
                contract["post_run_normalized_configuration_record_count"],
                8,
            )
            self.assertEqual(
                contract["environment_record_timing"],
                "retained environments captured after completed runs",
            )
            self.assertEqual(len(manifest["outputs"]), 3)
            self.assertEqual(len(manifest["environment_snapshots"]), 8)

            with (
                output_directory / "table_s1_components.csv"
            ).open(newline="", encoding="utf-8") as handle:
                component_rows = list(csv.DictReader(handle))
            with (
                output_directory / "table_s1_environments.csv"
            ).open(newline="", encoding="utf-8") as handle:
                environment_rows = list(csv.DictReader(handle))
            self.assertEqual(len(component_rows), 10)
            self.assertEqual(len(environment_rows), 8)

    def test_supplement_copy_matches_generated_markdown(self) -> None:
        supplement = SUPPLEMENT_PATH.read_text(encoding="utf-8")
        table_block = build_table.extract_supplement_table(supplement)
        rendered = build_table.render_markdown(
            build_table.load_and_validate_components(),
            build_table.build_environment_rows(
                build_table.load_and_validate_environments()
            ),
        )

        self.assertEqual(table_block, rendered)
        self.assertGreater(
            supplement.index(build_table.SUPPLEMENT_BEGIN_MARKER),
            supplement.index("## Table S1"),
        )


if __name__ == "__main__":
    unittest.main()
