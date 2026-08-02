#!/usr/bin/env python3
"""Build or verify fail-closed page-level render acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


REQUIRED_RENDITIONS = (
    "main_docx",
    "main_pdf",
    "supplementary_information_docx",
    "supplementary_information_pdf",
)
REQUIRED_PAGE_CHECKS = (
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
PAGE_NAME = re.compile(r"page-(\d{3})\.png")


class AuditError(ValueError):
    """The canonical page-level render acceptance contract is incomplete."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(workspace_root: Path, relative_path: str, label: str) -> Path:
    path = (workspace_root / relative_path).resolve()
    if not path.is_relative_to(workspace_root) or not path.is_file():
        raise AuditError(f"{label} is unavailable: {relative_path}")
    return path


def _relative(workspace_root: Path, path: Path) -> str:
    path = path.resolve()
    if not path.is_relative_to(workspace_root):
        raise AuditError(f"Audit input is outside the workspace: {path}")
    return path.relative_to(workspace_root).as_posix()


def _canonical_artifacts(publication_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = publication_manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "main",
        "supplementary_information",
    }:
        raise AuditError("Publication manifest must identify main and SI")
    result: dict[str, dict[str, Any]] = {}
    for document_name in ("main", "supplementary_information"):
        document = artifacts.get(document_name)
        if not isinstance(document, dict):
            raise AuditError(f"Publication artifact changed: {document_name}")
        for origin_format in ("docx", "pdf"):
            record = document.get(origin_format)
            if not isinstance(record, dict):
                raise AuditError(
                    f"Publication artifact changed: {document_name}/{origin_format}"
                )
            result[f"{document_name}_{origin_format}"] = record
    return result


def _page_paths(page_directory: Path) -> list[Path]:
    if not page_directory.is_dir():
        raise AuditError(f"Page-image directory is unavailable: {page_directory}")
    pages = sorted(
        path
        for path in page_directory.iterdir()
        if path.is_file() and PAGE_NAME.fullmatch(path.name)
    )
    expected = [f"page-{number:03d}.png" for number in range(1, len(pages) + 1)]
    if not pages or [path.name for path in pages] != expected:
        raise AuditError(f"Page-image set is not contiguous: {page_directory}")
    return pages


def _is_fully_accepted_page(
    value: Any,
    *,
    page_number: int,
    page_image_sha256: str,
) -> bool:
    if not isinstance(value, dict):
        return False
    checks = value.get("checks")
    return (
        value.get("page_number") == page_number
        and value.get("page_image_sha256") == page_image_sha256
        and value.get("status") == "pass"
        and isinstance(checks, dict)
        and set(checks) == set(REQUIRED_PAGE_CHECKS)
        and all(result == "pass" for result in checks.values())
    )


def build_audit_manifest(
    workspace_root: Path,
    *,
    render_provenance_path: Path,
    reviewer_decision_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Bind four canonical renditions to complete accepted per-page decisions."""

    workspace_root = workspace_root.resolve()
    package = workspace_root / "manuscript" / "publication_package"
    publication_manifest_path = package / "manifest.json"
    publication_manifest = _load_json(
        publication_manifest_path, "canonical publication manifest"
    )
    provenance = _load_json(render_provenance_path, "render provenance")
    decision = _load_json(reviewer_decision_path, "reviewer decision")
    canonical = _canonical_artifacts(publication_manifest)

    provenance_renditions = provenance.get("renditions")
    decision_renditions = decision.get("renditions")
    if (
        provenance.get("schema_version") != "1.0"
        or not isinstance(provenance_renditions, dict)
        or set(provenance_renditions) != set(REQUIRED_RENDITIONS)
    ):
        raise AuditError("Render provenance does not cover four canonical renditions")
    if (
        decision.get("schema_version") != "1.0"
        or decision.get("final_decision") != "accepted"
        or not isinstance(decision.get("reviewer_identity"), str)
        or not decision["reviewer_identity"].strip()
        or not isinstance(decision.get("review_date"), str)
        or not decision["review_date"].strip()
        or not isinstance(decision_renditions, dict)
        or set(decision_renditions) != set(REQUIRED_RENDITIONS)
    ):
        raise AuditError("Reviewer decision is incomplete or not accepted")

    renditions: dict[str, Any] = {}
    page_directories: dict[str, Path] = {}
    reviewed_page_count = 0
    for rendition_id in REQUIRED_RENDITIONS:
        source = canonical[rendition_id]
        source_path = _resolve(
            workspace_root, str(source.get("path", "")), "canonical document"
        )
        source_hash = _sha256(source_path)
        if source_hash != source.get("sha256"):
            raise AuditError(f"Canonical document hash drift: {rendition_id}")

        render = provenance_renditions[rendition_id]
        renderer = render.get("renderer") if isinstance(render, dict) else None
        expected_format = rendition_id.rsplit("_", 1)[1]
        if (
            not isinstance(render, dict)
            or render.get("document_path") != source.get("path")
            or render.get("document_sha256") != source_hash
            or render.get("origin_format") != expected_format
            or not isinstance(renderer, dict)
            or not isinstance(renderer.get("name"), str)
            or not renderer["name"].strip()
            or not isinstance(renderer.get("version"), str)
            or not renderer["version"].strip()
        ):
            raise AuditError(f"Render provenance changed: {rendition_id}")
        page_directory_value = render.get("page_directory")
        if not isinstance(page_directory_value, str):
            raise AuditError(f"Page-image directory changed: {rendition_id}")
        page_directory = (workspace_root / page_directory_value).resolve()
        if not page_directory.is_relative_to(workspace_root):
            raise AuditError(f"Page-image directory escaped workspace: {rendition_id}")
        pages = _page_paths(page_directory)
        page_directories[rendition_id] = page_directory

        rendered_pdf_record: dict[str, Any] | None = None
        if expected_format == "docx":
            rendered_pdf = render.get("rendered_pdf")
            if not isinstance(rendered_pdf, dict):
                raise AuditError(f"DOCX-origin PDF provenance changed: {rendition_id}")
            rendered_pdf_path = _resolve(
                workspace_root,
                str(rendered_pdf.get("path", "")),
                "DOCX-origin PDF",
            )
            rendered_pdf_hash = _sha256(rendered_pdf_path)
            if rendered_pdf_hash != rendered_pdf.get("sha256"):
                raise AuditError(f"DOCX-origin PDF hash drift: {rendition_id}")
            rendered_pdf_record = {
                "path": _relative(workspace_root, rendered_pdf_path),
                "sha256": rendered_pdf_hash,
                "size_bytes": rendered_pdf_path.stat().st_size,
            }

        rendition_decision = decision_renditions[rendition_id]
        decisions = (
            rendition_decision.get("pages")
            if isinstance(rendition_decision, dict)
            else None
        )
        if not isinstance(decisions, list) or len(decisions) != len(pages):
            raise AuditError(f"Per-page decision count changed: {rendition_id}")

        page_records: list[dict[str, Any]] = []
        for number, (page, page_decision) in enumerate(
            zip(pages, decisions, strict=True), start=1
        ):
            page_hash = _sha256(page)
            if not _is_fully_accepted_page(
                page_decision,
                page_number=number,
                page_image_sha256=page_hash,
            ):
                raise AuditError(
                    f"Page {number} is not fully accepted: {rendition_id}"
                )
            page_records.append(
                {
                    "checks": page_decision["checks"],
                    "page_image_path": _relative(workspace_root, page),
                    "page_image_sha256": page_hash,
                    "page_number": number,
                    "status": "pass",
                }
            )

        reviewed_page_count += len(page_records)
        rendition_record = {
            "document_path": source["path"],
            "document_sha256": source_hash,
            "document_size_bytes": source_path.stat().st_size,
            "origin_format": expected_format,
            "page_count": len(page_records),
            "pages": page_records,
            "renderer": renderer,
        }
        if rendered_pdf_record is not None:
            rendition_record["rendered_pdf"] = rendered_pdf_record
        renditions[rendition_id] = rendition_record

    shared_page_sets: list[dict[str, Any]] = []
    for document_name in ("main", "supplementary_information"):
        docx_id = f"{document_name}_docx"
        pdf_id = f"{document_name}_pdf"
        if page_directories[docx_id] != page_directories[pdf_id]:
            continue
        rendered_pdf = renditions[docx_id]["rendered_pdf"]
        canonical_pdf_hash = renditions[pdf_id]["document_sha256"]
        if rendered_pdf["sha256"] != canonical_pdf_hash:
            raise AuditError(
                f"Shared page-image set requires byte-identical PDFs: {document_name}"
            )
        shared_page_sets.append(
            {
                "basis": "byte-identical-pdf",
                "pdf_sha256": canonical_pdf_hash,
                "renditions": [docx_id, pdf_id],
            }
        )

    manifest = {
        "final_decision": "accepted",
        "rendition_count": len(renditions),
        "renditions": renditions,
        "review_date": decision["review_date"],
        "reviewed_page_count": reviewed_page_count,
        "reviewer_identity": decision["reviewer_identity"],
        "schema_version": "1.0",
        "shared_page_sets": shared_page_sets,
        "sources": {
            "canonical_publication_manifest": {
                "path": _relative(workspace_root, publication_manifest_path),
                "sha256": _sha256(publication_manifest_path),
            },
            "render_provenance": {
                "path": _relative(workspace_root, render_provenance_path),
                "sha256": _sha256(render_provenance_path),
            },
            "reviewer_decision": {
                "path": _relative(workspace_root, reviewer_decision_path),
                "sha256": _sha256(reviewer_decision_path),
            },
        },
        "status": "page-level-render-accepted",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def verify_audit_manifest(
    workspace_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Verify the accepted manifest and every bound document and page image."""

    workspace_root = workspace_root.resolve()
    manifest = _load_json(manifest_path, "page-level audit manifest")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("status") != "page-level-render-accepted"
        or manifest.get("final_decision") != "accepted"
        or manifest.get("rendition_count") != 4
    ):
        raise AuditError("Page-level audit manifest is not accepted")
    if (
        not isinstance(manifest.get("reviewer_identity"), str)
        or not manifest["reviewer_identity"].strip()
        or not isinstance(manifest.get("review_date"), str)
        or not manifest["review_date"].strip()
    ):
        raise AuditError("Reviewer identity or review date changed")
    sources = manifest.get("sources")
    renditions = manifest.get("renditions")
    if not isinstance(sources, dict) or not isinstance(renditions, dict):
        raise AuditError("Page-level audit manifest structure changed")
    for source in sources.values():
        if not isinstance(source, dict):
            raise AuditError("Page-level audit source changed")
        path = _resolve(workspace_root, str(source.get("path", "")), "audit source")
        if _sha256(path) != source.get("sha256"):
            raise AuditError(f"Audit source hash drift: {path}")
    if set(renditions) != set(REQUIRED_RENDITIONS):
        raise AuditError("Page-level audit rendition set changed")
    reviewed_page_count = 0
    for rendition_id, rendition in renditions.items():
        if not isinstance(rendition, dict):
            raise AuditError(f"Page-level audit rendition changed: {rendition_id}")
        document = _resolve(
            workspace_root,
            str(rendition.get("document_path", "")),
            "audited document",
        )
        if _sha256(document) != rendition.get("document_sha256"):
            raise AuditError(f"Audited document hash drift: {rendition_id}")
        if rendition_id.endswith("_docx"):
            rendered_pdf = rendition.get("rendered_pdf")
            if not isinstance(rendered_pdf, dict):
                raise AuditError(f"DOCX-origin PDF record changed: {rendition_id}")
            rendered_pdf_path = _resolve(
                workspace_root,
                str(rendered_pdf.get("path", "")),
                "audited DOCX-origin PDF",
            )
            if _sha256(rendered_pdf_path) != rendered_pdf.get("sha256"):
                raise AuditError(f"Audited DOCX-origin PDF drift: {rendition_id}")
        pages = rendition.get("pages")
        if not isinstance(pages, list) or rendition.get("page_count") != len(pages):
            raise AuditError(f"Audited page count changed: {rendition_id}")
        for number, page in enumerate(pages, start=1):
            if not isinstance(page, dict) or page.get("page_number") != number:
                raise AuditError(f"Audited page sequence changed: {rendition_id}")
            image = _resolve(
                workspace_root,
                str(page.get("page_image_path", "")),
                "audited page image",
            )
            if not _is_fully_accepted_page(
                page,
                page_number=number,
                page_image_sha256=_sha256(image),
            ):
                raise AuditError(f"Audited page drift: {rendition_id}/{number}")
        reviewed_page_count += len(pages)
    if manifest.get("reviewed_page_count") != reviewed_page_count:
        raise AuditError("Reviewed page total changed")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--render-provenance", type=Path)
    parser.add_argument("--reviewer-decision", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace_root.resolve()
    package = workspace / "manuscript" / "publication_package" / "page_audit"
    output = args.output or package / "manifest.json"
    try:
        if args.verify:
            result = verify_audit_manifest(workspace, output)
        else:
            result = build_audit_manifest(
                workspace,
                render_provenance_path=(
                    args.render_provenance or package / "render_provenance.json"
                ),
                reviewer_decision_path=(
                    args.reviewer_decision or package / "reviewer_decision.json"
                ),
                output_path=output,
            )
    except (AuditError, OSError, json.JSONDecodeError) as exc:
        print(f"Page-level render audit failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "rendition_count": result["rendition_count"],
                "reviewed_page_count": result["reviewed_page_count"],
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
