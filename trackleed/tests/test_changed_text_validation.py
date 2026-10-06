"""Changed-text checks through real Git repositories and the public validator."""
from pathlib import Path
import tempfile
import unittest

from trackleed.tests.validation_test_support import initialize_repository, run_git
from trackleed.validation.text_files import validate_changed_text


class ChangedTextValidationTests(unittest.TestCase):
    def test_invalid_staged_bytes_cannot_be_hidden_by_a_clean_working_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root, {"notes.md": "original\n"})
            path = root / "notes.md"
            path.write_bytes(bytes([255]))
            run_git(root, "add", "notes.md")
            path.write_text("fixed in working tree\n", encoding="utf-8")
            report = validate_changed_text(root)
            self.assertFalse(report.ok)
            self.assertIn("notes.md (staged)", report.format_errors())
            run_git(root, "add", "notes.md")
            self.assertTrue(validate_changed_text(root).ok)

    def test_utf8_lf_crlf_bom_and_extensionless_text_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root, {"notes.md": "original\n"})
            samples = {
                "lf.md": "za\u017c\u00f3\u0142\u0107\n".encode("utf-8"),
                "crlf.txt": b"first\r\nsecond\r\n",
                "bom.md": b"\xef\xbb\xbfUTF-8\n",
                "README": b"plain text\n",
            }
            for name, content in samples.items():
                (root / name).write_bytes(content)
            report = validate_changed_text(root)
            self.assertTrue(report.ok, report.format_errors())
            self.assertEqual({name: (root / name).read_bytes() for name in samples}, samples)

    def test_binary_and_explicit_frozen_bytes_are_exempt_but_text_nul_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root, {
                ".gitattributes": "frozen.dat -text\nforced.data text\n",
                ".gitignore": "ignored.md\n",
            })
            for name in ("binary.png", "frozen.dat", "ignored.md"):
                (root / name).write_bytes(b"\0\xff\xfe")
            self.assertTrue(validate_changed_text(root).ok)
            (root / "forced.data").write_bytes(b"text\0bytes")
            report = validate_changed_text(root)
            self.assertFalse(report.ok)
            self.assertIn("forced.data", report.format_errors())
            self.assertIn("NUL", report.format_errors())

    def test_renamed_unicode_paths_are_checked_and_deleted_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root, {"notes.md": "valid\n"})
            name = "za\u017c\u00f3\u0142\u0107 nowy.md"
            run_git(root, "mv", "notes.md", name)
            (root / name).write_bytes(b"broken\r\r\n")
            report = validate_changed_text(root)
            self.assertFalse(report.ok)
            self.assertIn(name, report.format_errors())
            (root / name).unlink()
            run_git(root, "add", "-u")
            self.assertTrue(validate_changed_text(root).ok)

    def test_missing_comparison_history_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root, {"notes.md": "valid\n"})
            report = validate_changed_text(root, base="missing-commit")
            self.assertFalse(report.ok)
            self.assertIn("missing-commit", report.format_errors())
            self.assertIn("unavailable", report.format_errors())

    def test_first_push_checks_all_files_and_a_working_fix_can_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root, {"early.md": "valid\n"})
            (root / "early.md").write_bytes(bytes([255]))
            run_git(root, "add", "early.md")
            run_git(root, "commit", "-m", "bad historical text")
            (root / "late.md").write_text("valid\n", encoding="utf-8")
            run_git(root, "add", "late.md")
            run_git(root, "commit", "-m", "later change")
            report = validate_changed_text(root, base="EMPTY")
            self.assertFalse(report.ok)
            self.assertIn("early.md", report.format_errors())
            (root / "early.md").write_text("fixed\n", encoding="utf-8")
            self.assertTrue(validate_changed_text(root, base="EMPTY").ok)

    def test_file_type_changes_are_checked_without_requiring_os_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root, {"notes.md": "valid\n"})
            blob = run_git(root, "rev-parse", "HEAD:notes.md").stdout.strip()
            run_git(root, "update-index", "--add", "--cacheinfo",
                    f"120000,{blob},changed.md")
            run_git(root, "commit", "-m", "historical link")
            (root / "later.md").write_text("valid\n", encoding="utf-8")
            run_git(root, "add", "later.md")
            run_git(root, "commit", "-m", "later unrelated change")
            (root / "changed.md").write_bytes(bytes([255]))
            blob = run_git(root, "hash-object", "-w", "changed.md").stdout.strip()
            run_git(root, "update-index", "--cacheinfo", f"100644,{blob},changed.md")
            report = validate_changed_text(root)
            self.assertFalse(report.ok)
            self.assertIn("changed.md", report.format_errors())
            run_git(root, "commit", "-m", "replace link with text")
            self.assertFalse(validate_changed_text(root).ok)
