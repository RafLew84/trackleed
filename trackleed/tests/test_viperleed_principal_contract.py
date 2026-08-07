from __future__ import annotations

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from trackleed.publication.viperleed_principal import (
    ViperleedPrincipalValidationError,
    freeze_principal_viperleed_scans,
    validate_principal_viperleed_scans,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PACKAGE = REPO_ROOT / "manuscript/cross_contract/viperleed_principal"


def copy_unfrozen_package(destination: Path) -> None:
    destination.mkdir()
    shutil.copyfile(
        SOURCE_PACKAGE / ".gitattributes",
        destination / ".gitattributes",
    )
    shutil.copyfile(
        SOURCE_PACKAGE / "contract.json",
        destination / "contract.json",
    )
    shutil.copytree(
        SOURCE_PACKAGE / "execution",
        destination / "execution",
    )


class ViperleedPrincipalContractTests(unittest.TestCase):
    def test_principal_results_preserve_full_denominators_and_native_semantics(
        self,
    ) -> None:
        report = validate_principal_viperleed_scans(REPO_ROOT)

        self.assertEqual(report["status"], "validated-principal-viperleed-scans")
        self.assertEqual(report["scan_count"], 2)
        self.assertEqual(report["frame_count"], 81)
        self.assertEqual(report["whole_pattern_beam_frame_count"], 1458)
        self.assertEqual(report["four_branch_opportunity_count"], 324)
        self.assertEqual(report["viper_native_direct_detectability_count"], 305)
        self.assertEqual(report["retained_non_direct_count"], 19)
        self.assertEqual(report["final_position_count"], 324)
        self.assertEqual(report["native_table_count_per_scan"], 13)
        self.assertEqual(
            report["scans"],
            {
                "scan_a": {
                    "direct_detectability_count": 129,
                    "final_position_count": 140,
                    "frame_count": 35,
                    "opportunity_count": 140,
                    "retained_non_direct_count": 11,
                    "terminal_duplicate_energy_ev": 220.5,
                },
                "scan_b": {
                    "direct_detectability_count": 176,
                    "final_position_count": 184,
                    "frame_count": 46,
                    "opportunity_count": 184,
                    "retained_non_direct_count": 8,
                    "terminal_duplicate_energy_ev": 220.4,
                },
            },
        )
        self.assertEqual(
            report["inference_scope"],
            "new inference on existing acquisitions",
        )
        self.assertEqual(report["scientific_core_rebaseline"], "required")

    def test_validation_fails_closed_when_principal_package_is_absent(self) -> None:
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                ViperleedPrincipalValidationError,
                "principal ViPErLEED contract is unavailable",
            ):
                validate_principal_viperleed_scans(Path(temporary))

    def test_validation_fails_closed_when_native_output_drifts(self) -> None:
        with TemporaryDirectory() as temporary:
            copied_package = Path(temporary) / "viperleed_principal"
            shutil.copytree(SOURCE_PACKAGE, copied_package)
            native_table = copied_package / "frozen/scan_a/viperleed_scan_a_signif.csv"
            with native_table.open("ab") as handle:
                handle.write(b"tampered")

            with self.assertRaisesRegex(
                ViperleedPrincipalValidationError,
                "authenticated principal native output drift",
            ):
                validate_principal_viperleed_scans(
                    REPO_ROOT,
                    copied_package,
                )

    def test_freeze_validates_staging_before_publishing(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "viperleed_principal"
            copy_unfrozen_package(package)
            with (package / ".gitattributes").open("ab") as handle:
                handle.write(b"# manifest drift\n")

            with self.assertRaisesRegex(
                ViperleedPrincipalValidationError,
                "principal package manifest hash changed",
            ):
                freeze_principal_viperleed_scans(REPO_ROOT, package)

            self.assertFalse((package / "frozen").exists())
            self.assertFalse((package / "manifest.json").exists())

    def test_freeze_is_build_once_and_reproduces_the_validated_package(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "viperleed_principal"
            copy_unfrozen_package(package)

            frozen = freeze_principal_viperleed_scans(REPO_ROOT, package)

            self.assertEqual(
                frozen,
                validate_principal_viperleed_scans(REPO_ROOT, package),
            )
            with self.assertRaisesRegex(
                FileExistsError,
                "Refusing to replace frozen principal ViPErLEED package",
            ):
                freeze_principal_viperleed_scans(REPO_ROOT, package)


if __name__ == "__main__":
    unittest.main()
