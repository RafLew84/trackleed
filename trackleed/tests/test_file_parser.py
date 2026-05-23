import tempfile
import unittest
from pathlib import Path

from trackleed.core.data_models import LeedFile
from trackleed.io.file_parser import parse_directory, repair_duplicate_frame_ids


class FileParserTests(unittest.TestCase):
    def test_parses_ag111_domain_suffix_frame_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "Ag111-3G_0000_T0014_E1100_C0000.tiff").touch()
            (directory / "Ag111-3G_0001_T0015_E1100_C0000.tiff").touch()

            files = parse_directory(directory)

        self.assertEqual([file.id for file in files], [0, 1])
        self.assertEqual(files[0].element, "Ag")
        self.assertEqual(files[0].surface, 111)
        self.assertEqual(files[0].mode, "G")
        self.assertEqual(files[0].temperature, 14.0)
        self.assertEqual(files[0].energy, 110.0)
        self.assertEqual(files[0].coverage, 0.0)

    def test_fallback_prefers_frame_token_before_trailing_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "sample_frame_0007_T0014_E1100_C0000.png").touch()

            files = parse_directory(directory)

        self.assertEqual([file.id for file in files], [7])

    def test_fallback_prefers_temperature_token_when_no_explicit_frame_token(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "sample_T0014_E1100_C0000.png").touch()

            files = parse_directory(directory)

        self.assertEqual([file.id for file in files], [14])

    def test_duplicate_frame_ids_fall_back_to_temperature_token(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "Ag111-3G_0000_T0014_E1100_C0000.tiff").touch()
            (directory / "Ag111-3G_0000_T0015_E1100_C0000.tiff").touch()

            files = parse_directory(directory)

        self.assertEqual([file.id for file in files], [14, 15])

    def test_duplicate_frame_and_temperature_ids_fall_back_to_coverage_token(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "Ag111-3G_0000_T0500_E1100_C0150.tiff").touch()
            (directory / "Ag111-3G_0000_T0500_E1100_C0160.tiff").touch()

            files = parse_directory(directory)

        self.assertEqual([file.id for file in files], [150, 160])

    def test_duplicate_frame_temperature_and_coverage_ids_fall_back_to_energy_token(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "Ag111-3G_0000_T0500_E1100_C0150.tiff").touch()
            (directory / "Ag111-3G_0000_T0500_E1110_C0150.tiff").touch()

            files = parse_directory(directory)

        self.assertEqual([file.id for file in files], [1100, 1110])

    def test_png_fallback_uses_temperature_when_trailing_coverage_ids_are_duplicate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "sample_T0014_E1100_C0000.png").touch()
            (directory / "sample_T0015_E1100_C0000.png").touch()

            files = parse_directory(directory)

        self.assertEqual([file.id for file in files], [14, 15])

    def test_repair_duplicate_frame_ids_updates_loaded_files_in_memory(self):
        files = [
            LeedFile(
                filepath=Path("Ag111-3G_0000_T0014_E1100_C0000.tiff"),
                filename="Ag111-3G_0000_T0014_E1100_C0000.tiff",
                is_background=False,
                element="Ag",
                surface=111,
                mode="G",
                temperature=14.0,
                energy=110.0,
                id=0,
                coverage=0.0,
            ),
            LeedFile(
                filepath=Path("Ag111-3G_0000_T0015_E1100_C0000.tiff"),
                filename="Ag111-3G_0000_T0015_E1100_C0000.tiff",
                is_background=False,
                element="Ag",
                surface=111,
                mode="G",
                temperature=15.0,
                energy=110.0,
                id=0,
                coverage=0.0,
            ),
        ]

        repaired_source = repair_duplicate_frame_ids(files)

        self.assertEqual(repaired_source, "temperature")
        self.assertEqual([file.id for file in files], [14, 15])

    def test_repair_duplicate_frame_ids_prefers_explicit_frame_token(self):
        files = [
            LeedFile(
                filepath=Path("Ag111-3G_0000_T0014_E1100_C0000.tiff"),
                filename="Ag111-3G_0000_T0014_E1100_C0000.tiff",
                is_background=False,
                element="Ag",
                surface=111,
                mode="G",
                temperature=14.0,
                energy=110.0,
                id=0,
                coverage=0.0,
            ),
            LeedFile(
                filepath=Path("Ag111-3G_0001_T0014_E1100_C0000.tiff"),
                filename="Ag111-3G_0001_T0014_E1100_C0000.tiff",
                is_background=False,
                element="Ag",
                surface=111,
                mode="G",
                temperature=14.0,
                energy=110.0,
                id=0,
                coverage=0.0,
            ),
        ]

        repaired_source = repair_duplicate_frame_ids(files)

        self.assertEqual(repaired_source, "frame")
        self.assertEqual([file.id for file in files], [0, 1])

    def test_legacy_file_pattern_still_parses_frame_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "RuAu1G_0093_T0450_E0077_C0151_R-33.tiff").touch()

            files = parse_directory(directory)

        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].element, "RuAu")
        self.assertEqual(files[0].surface, 1)
        self.assertEqual(files[0].mode, "G")
        self.assertEqual(files[0].id, 93)
        self.assertEqual(files[0].energy, 7.7)
        self.assertEqual(files[0].coverage, 0.151)


if __name__ == "__main__":
    unittest.main()
