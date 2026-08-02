"""Contract tests for the canonical manuscript/SI document builder."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from manuscript.publication_package import build_publication_package as builder


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]


class CanonicalPublicationPackageContractTest(unittest.TestCase):
    def test_contract_uses_required_reader_and_empty_fallback_manifest(self) -> None:
        contract = json.loads((PACKAGE_DIR / "document_contract.json").read_text(encoding="utf-8"))
        fallbacks = json.loads((PACKAGE_DIR / "math_fallbacks.json").read_text(encoding="utf-8"))

        self.assertEqual(contract["pandoc_reader"], "markdown+tex_math_single_backslash")
        self.assertEqual(
            contract["authorized_placeholders"],
            [
                "TODO[AUTHOR]",
                "TODO[EXPERIMENT]",
                "TODO[RELEASE-AUTHORIZATION]",
                "TODO[SUBMISSION]",
            ],
        )
        self.assertEqual(fallbacks["fallbacks"], [])
        self.assertEqual(contract["canonical_artifacts"]["main"]["docx"], "output/trackleed_manuscript.docx")
        self.assertEqual(contract["canonical_artifacts"]["main"]["pdf"], "output/trackleed_manuscript.pdf")
        self.assertEqual(
            contract["canonical_artifacts"]["supplementary_information"]["docx"],
            "output/trackleed_supplementary_information.docx",
        )
        self.assertEqual(
            contract["canonical_artifacts"]["supplementary_information"]["pdf"],
            "output/trackleed_supplementary_information.pdf",
        )
        self.assertTrue(contract["excluded_review_artifacts"])

    def test_math_audit_fails_closed_on_count_mismatch(self) -> None:
        with self.assertRaisesRegex(builder.BuildError, "Math transport mismatch"):
            builder.assert_math_transport(
                document_name="main",
                ast_math=["x", "y"],
                roundtrip_math=["x"],
                native_omath=["x"],
                fallback_count=0,
            )

    def test_math_audit_fails_closed_on_semantic_mismatch_or_empty_omath(self) -> None:
        with self.assertRaisesRegex(builder.BuildError, "expression mismatch"):
            builder.assert_math_transport(
                document_name="main",
                ast_math=[r"R^2"],
                roundtrip_math=[r"R^{-2}"],
                native_omath=["R-2"],
                fallback_count=0,
            )
        with self.assertRaisesRegex(builder.BuildError, "empty native OMath"):
            builder.assert_math_transport(
                document_name="main",
                ast_math=["x"],
                roundtrip_math=["x"],
                native_omath=[""],
                fallback_count=0,
            )

        builder.assert_math_transport(
            document_name="main",
            ast_math=[
                r"D(E)\propto1/\sqrt{E}",
                r"R^2",
                r"P_{\mathrm{vis}}=[1-\operatorname{sigmoid}(o)]",
            ],
            roundtrip_math=[
                r"D(E) \propto 1/\sqrt{E}",
                r"R^{2}",
                r"P_{vis}=\lbrack 1-sigmoid(o)\rbrack",
            ],
            native_omath=["D(E)∝1/E", "R2", "Pvis=1-sigmoid(o)"],
            fallback_count=0,
        )

    def test_conversion_warnings_are_fatal(self) -> None:
        with self.assertRaisesRegex(builder.BuildError, "conversion warning"):
            builder.assert_no_warnings("[WARNING] Could not fetch resource")

    def test_damaged_math_signatures_are_fatal_unless_allowlisted(self) -> None:
        with self.assertRaisesRegex(builder.BuildError, "damaged-math signature"):
            builder.assert_no_damaged_math("The result is () after conversion.", allowlist=[])
        builder.assert_no_damaged_math("The result is () after conversion.", allowlist=["()"])

    def test_ooxml_audit_counts_native_math_and_structures(self) -> None:
        document_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
          xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
          <w:body><m:oMath><m:f/><m:rad/><m:sSub/><m:sSup/><m:r><m:t>Δ≤σ</m:t></m:r></m:oMath></w:body>
        </w:document>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.docx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)
            audit = builder.inspect_docx_math(path)

        self.assertEqual(audit["native_omath_count"], 1)
        self.assertEqual(audit["structures"]["fractions"], 1)
        self.assertEqual(audit["structures"]["radicals"], 1)
        self.assertEqual(audit["structures"]["subscripts"], 1)
        self.assertEqual(audit["structures"]["superscripts"], 1)
        self.assertTrue(audit["symbols"]["Δ"])
        self.assertTrue(audit["symbols"]["≤"])
        self.assertTrue(audit["symbols"]["σ"])

    def test_docx_normalization_is_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.docx"
            second = Path(directory) / "second.docx"
            for path, timestamp in ((first, (2026, 8, 2, 12, 0, 0)), (second, (2026, 8, 2, 13, 0, 0))):
                with zipfile.ZipFile(path, "w") as archive:
                    info = zipfile.ZipInfo("word/document.xml", timestamp)
                    archive.writestr(info, "<document/>")
                    info = zipfile.ZipInfo("docProps/core.xml", timestamp)
                    archive.writestr(
                        info,
                        "<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
                        "xmlns:dcterms=\"http://purl.org/dc/terms/\"><dcterms:created>different</dcterms:created></cp:coreProperties>",
                    )
                    info = zipfile.ZipInfo("docProps/custom.xml", timestamp)
                    archive.writestr(
                        info,
                        "<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/custom-properties\" "
                        "xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\">"
                        "<property name=\"bibliography\"><vt:lpwstr>"
                        f"{path.parent / path.stem / 'manuscript' / 'references.bib'}"
                        "</vt:lpwstr></property></Properties>",
                    )
                builder.normalize_docx(path, "2026-08-02")

            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_tables_are_wrapped_in_landscape_sections(self) -> None:
        document_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body><w:p/><w:tbl><w:tr><w:tc><w:p/></w:tc></w:tr></w:tbl><w:p/><w:sectPr/></w:body>
        </w:document>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.docx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)
            builder.wrap_tables_in_landscape_sections(path)
            with zipfile.ZipFile(path) as archive:
                transformed = archive.read("word/document.xml").decode("utf-8")

        self.assertEqual(transformed.count('w:orient="landscape"'), 1)
        self.assertGreaterEqual(transformed.count("w:sectPr"), 5)
        self.assertEqual(transformed.count("w:cantSplit"), 1)
        self.assertEqual(transformed.count("w:tblHeader"), 1)

    def test_assemblies_include_all_numbered_display_items(self) -> None:
        main = builder.assemble_main(REPOSITORY_ROOT)
        si = builder.assemble_supplementary_information(REPOSITORY_ROOT)

        for number in range(1, 6):
            self.assertIn(f"**Figure {number}.", main)
        for number in range(1, 7):
            self.assertIn(f"Figure S{number}", si)
        self.assertIn("Figure S2 (continued)", si)
        self.assertEqual(main.count("{width=6.25in}"), 5)
        self.assertEqual(si.count("{width=6.25in}"), 6)
        self.assertEqual(main.count(builder.PAGE_BREAK), 8)
        self.assertEqual(si.count(builder.PAGE_BREAK), 5)

    def test_reproducibility_check_compares_every_canonical_output(self) -> None:
        first = {
            "artifacts": {
                "main": {"docx": {"sha256": "a"}, "pdf": {"sha256": "b"}},
                "supplementary_information": {
                    "docx": {"sha256": "c"},
                    "pdf": {"sha256": "d"},
                },
            }
        }
        changed = json.loads(json.dumps(first))
        changed["artifacts"]["supplementary_information"]["pdf"]["sha256"] = "changed"
        with mock.patch.object(builder, "build", side_effect=(first, changed)):
            with self.assertRaisesRegex(builder.BuildError, "not reproducible"):
                builder.build_reproducibly(pandoc=Path("pandoc"))


if __name__ == "__main__":
    unittest.main()
