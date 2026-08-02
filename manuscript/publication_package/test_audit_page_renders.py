"""Contract tests for canonical page-level render acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from manuscript.publication_package import audit_page_renders as audit


PAGE_CHECKS = (
    "mathematics",
    "tables",
    "figures",
    "captions",
    "numbering",
    "citations_and_cross_references",
    "clipping",
    "page_breaks",
    "effective_fonts",
    "image_quality",
    "missing_glyphs",
    "authorized_placeholders",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PageRenderAuditContractTest(unittest.TestCase):
    def test_builds_an_accepted_manifest_for_all_four_canonical_renditions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            package = workspace / "manuscript/publication_package"
            output = package / "output"
            render_root = package / "page_audit/rendered"
            output.mkdir(parents=True)
            render_root.mkdir(parents=True)

            artifacts: dict[str, dict[str, dict[str, object]]] = {}
            provenance: dict[str, object] = {
                "schema_version": "1.0",
                "renditions": {},
            }
            decisions: dict[str, object] = {
                "schema_version": "1.0",
                "reviewer_identity": "Test reviewer",
                "review_date": "2026-08-02",
                "final_decision": "accepted",
                "renditions": {},
            }
            for document_name in ("main", "supplementary_information"):
                artifacts[document_name] = {}
                for source_format in ("docx", "pdf"):
                    document = output / f"{document_name}.{source_format}"
                    document.write_bytes(f"{document_name}/{source_format}".encode())
                    relative_document = document.relative_to(workspace).as_posix()
                    artifacts[document_name][source_format] = {
                        "path": relative_document,
                        "sha256": sha256(document),
                        "size_bytes": document.stat().st_size,
                    }
                    rendition_id = f"{document_name}_{source_format}"
                    page_directory = render_root / rendition_id
                    page_directory.mkdir()
                    page = page_directory / "page-001.png"
                    page.write_bytes(f"page/{rendition_id}".encode())
                    render_record = {
                        "document_path": relative_document,
                        "document_sha256": sha256(document),
                        "origin_format": source_format,
                        "page_directory": page_directory.relative_to(workspace).as_posix(),
                        "renderer": {
                            "name": "Test renderer",
                            "version": "1.2.3",
                        },
                    }
                    if source_format == "docx":
                        rendered_pdf = page_directory / f"{document_name}-docx-origin.pdf"
                        rendered_pdf.write_bytes(f"pdf/{rendition_id}".encode())
                        render_record["rendered_pdf"] = {
                            "path": rendered_pdf.relative_to(workspace).as_posix(),
                            "sha256": sha256(rendered_pdf),
                        }
                    provenance["renditions"][rendition_id] = render_record
                    decisions["renditions"][rendition_id] = {
                        "pages": [
                            {
                                "page_number": 1,
                                "page_image_sha256": sha256(page),
                                "checks": {name: "pass" for name in PAGE_CHECKS},
                                "status": "pass",
                            }
                        ]
                    }

            publication_manifest = package / "manifest.json"
            publication_manifest.write_text(
                json.dumps({"schema_version": "1.0", "artifacts": artifacts}),
                encoding="utf-8",
                newline="\n",
            )
            provenance_path = render_root / "render_provenance.json"
            provenance_path.write_text(
                json.dumps(provenance), encoding="utf-8", newline="\n"
            )
            decision_path = package / "page_audit/reviewer_decision.json"
            decision_path.write_text(
                json.dumps(decisions), encoding="utf-8", newline="\n"
            )
            audit_manifest = package / "page_audit/manifest.json"

            result = audit.build_audit_manifest(
                workspace,
                render_provenance_path=provenance_path,
                reviewer_decision_path=decision_path,
                output_path=audit_manifest,
            )

            self.assertEqual(result["status"], "page-level-render-accepted")
            self.assertEqual(result["rendition_count"], 4)
            self.assertEqual(result["reviewed_page_count"], 4)
            self.assertEqual(
                result["renditions"]["main_docx"]["rendered_pdf"]["sha256"],
                provenance["renditions"]["main_docx"]["rendered_pdf"]["sha256"],
            )
            self.assertEqual(set(result["renditions"]), {
                "main_docx",
                "main_pdf",
                "supplementary_information_docx",
                "supplementary_information_pdf",
            })
            self.assertEqual(audit.verify_audit_manifest(workspace, audit_manifest), result)

            tampered_manifest = json.loads(audit_manifest.read_text(encoding="utf-8"))
            tampered_manifest["reviewer_identity"] = ""
            audit_manifest.write_text(
                json.dumps(tampered_manifest), encoding="utf-8", newline="\n"
            )
            with self.assertRaisesRegex(audit.AuditError, "Reviewer identity"):
                audit.verify_audit_manifest(workspace, audit_manifest)
            audit_manifest.write_text(
                json.dumps(result), encoding="utf-8", newline="\n"
            )

            audited_page = render_root / "main_pdf/page-001.png"
            original_page = audited_page.read_bytes()
            audited_page.write_bytes(b"changed page")
            with self.assertRaisesRegex(audit.AuditError, "Audited page drift"):
                audit.verify_audit_manifest(workspace, audit_manifest)
            audited_page.write_bytes(original_page)

            audited_page.unlink()
            with self.assertRaisesRegex(audit.AuditError, "not contiguous"):
                audit.build_audit_manifest(
                    workspace,
                    render_provenance_path=provenance_path,
                    reviewer_decision_path=decision_path,
                    output_path=audit_manifest,
                )
            audited_page.write_bytes(original_page)

            decisions["final_decision"] = "pending-hitl"
            decision_path.write_text(
                json.dumps(decisions), encoding="utf-8", newline="\n"
            )
            with self.assertRaisesRegex(audit.AuditError, "incomplete or not accepted"):
                audit.build_audit_manifest(
                    workspace,
                    render_provenance_path=provenance_path,
                    reviewer_decision_path=decision_path,
                    output_path=audit_manifest,
                )
            decisions["final_decision"] = "accepted"

            provenance["renditions"]["main_pdf"]["page_directory"] = (
                provenance["renditions"]["main_docx"]["page_directory"]
            )
            decisions["renditions"]["main_pdf"] = decisions["renditions"]["main_docx"]
            provenance_path.write_text(
                json.dumps(provenance), encoding="utf-8", newline="\n"
            )
            decision_path.write_text(
                json.dumps(decisions), encoding="utf-8", newline="\n"
            )
            with self.assertRaisesRegex(audit.AuditError, "Shared page-image set"):
                audit.build_audit_manifest(
                    workspace,
                    render_provenance_path=provenance_path,
                    reviewer_decision_path=decision_path,
                    output_path=audit_manifest,
                )


if __name__ == "__main__":
    unittest.main()
