from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from trackleed.publication.viperleed_pilot import (
    ViperleedPilotValidationError,
    freeze_viperleed_pilot,
    validate_viperleed_pilot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "manuscript/cross_contract/viperleed_pilot"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


class ViperleedPilotContractTests(unittest.TestCase):
    def test_coordinated_diagnostic_evidence_drift_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "viperleed-pilot"
            shutil.copytree(PACKAGE_ROOT, package)
            relative_path = "frozen/native/raw_intensity.csv"
            diagnostic = package / relative_path
            lines = diagnostic.read_text(encoding="utf-8").splitlines()
            first_record = lines[1].split(",")
            first_record[1] = format(float(first_record[1]) + 1.0, ".8g")
            lines[1] = ",".join(first_record)
            diagnostic.write_text("\n".join(lines) + "\n", encoding="utf-8")

            manifest_path = package / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            diagnostic_entry = next(
                item
                for item in manifest["files"]
                if item["path"] == relative_path
            )
            diagnostic_entry["sha256"] = _sha256(diagnostic)
            diagnostic_entry["size_bytes"] = diagnostic.stat().st_size
            _write_json(manifest_path, manifest)

            report_path = package / "validation_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["manifest_sha256"] = _sha256(manifest_path)
            _write_json(report_path, report)

            with self.assertRaisesRegex(
                ViperleedPilotValidationError,
                "contract-bound evidence hash drift",
            ):
                validate_viperleed_pilot(REPO_ROOT, package)

    def test_frozen_evidence_byte_drift_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "viperleed-pilot"
            shutil.copytree(PACKAGE_ROOT, package)
            significance = package / "frozen/native/significance.csv"
            significance.write_bytes(significance.read_bytes() + b"\n")

            with self.assertRaisesRegex(
                ViperleedPilotValidationError,
                "Frozen pilot contract-bound evidence hash drift",
            ):
                validate_viperleed_pilot(REPO_ROOT, package)

    def test_freeze_rejects_source_drift_before_copying_evidence(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            package = workspace / "manuscript/cross_contract/viperleed_pilot"
            package.mkdir(parents=True)
            shutil.copyfile(PACKAGE_ROOT / "contract.json", package / "contract.json")
            shutil.copyfile(
                PACKAGE_ROOT / ".gitattributes",
                package / ".gitattributes",
            )

            scientific_contract = (
                workspace / "manuscript/cross_contract/scientific_contract.json"
            )
            scientific_contract.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                REPO_ROOT / "manuscript/cross_contract/scientific_contract.json",
                scientific_contract,
            )
            source = (
                workspace
                / ".scratch/trackleed-physics-method-paper/artifacts/"
                "domain-baseline-high-energy/input_manifest.csv"
            )
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                REPO_ROOT
                / ".scratch/trackleed-physics-method-paper/artifacts/"
                "domain-baseline-high-energy/input_manifest.csv",
                source,
            )
            source.write_bytes(source.read_bytes() + b"\n")

            with self.assertRaisesRegex(
                ViperleedPilotValidationError,
                "Pilot origin hash drift",
            ):
                freeze_viperleed_pilot(workspace, package)

    def test_coordinated_contract_metadata_drift_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "viperleed-pilot"
            shutil.copytree(PACKAGE_ROOT, package)

            contract_path = package / "contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["runtime"]["imagej_version"] = "changed-after-freeze"
            _write_json(contract_path, contract)

            manifest_path = package / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            contract_entry = next(
                item for item in manifest["files"] if item["path"] == "contract.json"
            )
            contract_entry["sha256"] = _sha256(contract_path)
            contract_entry["size_bytes"] = contract_path.stat().st_size
            _write_json(manifest_path, manifest)

            report_path = package / "validation_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["contract_sha256"] = _sha256(contract_path)
            report["manifest_sha256"] = _sha256(manifest_path)
            _write_json(report_path, report)

            with self.assertRaisesRegex(
                ViperleedPilotValidationError,
                "contract hash changed",
            ):
                validate_viperleed_pilot(REPO_ROOT, package)

    def test_validation_fails_closed_when_contract_package_is_absent(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)

            with self.assertRaisesRegex(
                ViperleedPilotValidationError,
                "pilot contract is unavailable",
            ):
                validate_viperleed_pilot(workspace)

    def test_frozen_pilot_reproduces_direct_detectability_and_tolerance_contract(
        self,
    ) -> None:
        report = validate_viperleed_pilot(REPO_ROOT)

        self.assertEqual(report["status"], "validated-viperleed-pilot-contract")
        self.assertEqual(report["pilot_record_count"], 60)
        self.assertEqual(report["viper_native_direct_detectability_count"], 59)
        self.assertEqual(report["gaussian_rejection_count"], 22)
        self.assertEqual(report["rejections_above_native_threshold"], 21)
        self.assertEqual(report["rejections_above_threshold_with_finite_raw"], 21)
        self.assertEqual(report["high_signal_calibration_count"], 22)
        self.assertEqual(report["cross_contract_tolerance_px"], 1.0)
        self.assertEqual(
            report["branch_mapping"],
            {
                "P1_B1": [-1, 0],
                "P1_B2": [1, 0],
                "P2_B1": [1, -1],
                "P2_B2": [-1, 1],
            },
        )
        self.assertEqual(report["pilot_evidence_role"], "calibration-only")


if __name__ == "__main__":
    unittest.main()
