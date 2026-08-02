#!/usr/bin/env python3
"""Build and audit the canonical TrackLEED manuscript and SI DOCX/PDF pair."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

from lxml import etree
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, ByteStringObject


PACKAGE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIRECTORY.parents[1]
CONTRACT_PATH = PACKAGE_DIRECTORY / "document_contract.json"
FALLBACK_PATH = PACKAGE_DIRECTORY / "math_fallbacks.json"
MANIFEST_PATH = PACKAGE_DIRECTORY / "manifest.json"
ASSEMBLED_DIRECTORY = PACKAGE_DIRECTORY / "assembled"
OUTPUT_DIRECTORY = PACKAGE_DIRECTORY / "output"

PANDOC_READER = "markdown+tex_math_single_backslash"
PAGE_BREAK = """```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```"""
MATH_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MATH_TAGS = {
    "fractions": "f",
    "radicals": "rad",
    "subscripts": "sSub",
    "superscripts": "sSup",
    "subscript_superscripts": "sSubSup",
}
CRITICAL_SYMBOLS = ("Δ", "≤", "≥", "σ", "μ", "θ", "√", "±")
DAMAGED_MATH_PATTERNS = (
    ("empty-parentheses", re.compile(r"(?<!\w)\(\)(?!\w)")),
    ("orphan-one-slash", re.compile(r"(?<!\w)1/(?!\w|\d)")),
    ("parenthesis-underscore", re.compile(r"\(_")),
    ("empty-subscript", re.compile(r"_\{\}\s")),
)



@dataclass(frozen=True)
class DisplaySpec:
    label: str
    directory_name: str
    image_names: tuple[str, ...]
    continued_label: str | None = None


MAIN_FIGURES = (
    DisplaySpec("Figure 1", "figure01_measurement_chain", ("figure01_measurement_chain.png",)),
    DisplaySpec("Figure 2", "figure02_controlled_equivariance", ("figure02_controlled_equivariance.png",)),
    DisplaySpec("Figure 3", "figure03_independent_scane_stress", ("figure03_independent_scane_stress.png",)),
    DisplaySpec("Figure 4", "figure04_controlled_sam2_memory_tradeoff", ("figure04_controlled_sam2_memory_tradeoff.png",)),
    DisplaySpec("Figure 5", "figure05_cu111_physical_endpoint", ("figure05_cu111_physical_endpoint.png",)),
)
SUPPLEMENTARY_FIGURES = (
    DisplaySpec(
        "Figure S2",
        "figure_s2_measurement_sensitivity",
        (
            "figure_s2_measurement_sensitivity_a-d.png",
            "figure_s2_measurement_sensitivity_e-f_continued.png",
        ),
        continued_label="Figure S2 (continued)",
    ),
    DisplaySpec("Figure S3", "figure_s3_ag111_midpoint_drift", ("figure_s3_ag111_midpoint_drift.png",)),
    DisplaySpec("Figure S4", "figure_s4_easyleed_viperleed_compatibility", ("figure_s4_easyleed_viperleed_compatibility.png",)),
    DisplaySpec("Figure S5", "figure_s5_geometry_inconsistent_continuation", ("figure_s5_geometry_inconsistent_continuation.png",)),
    DisplaySpec("Figure S6", "figure_s6_cu111_pair_2_4_missingness", ("figure_s6_cu111_pair_2_4_missingness.png",)),
)


class BuildError(RuntimeError):
    """Raised when a fail-closed build or audit condition is not satisfied."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path, root: Path = REPOSITORY_ROOT) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def shift_heading_hierarchy(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise BuildError("Canonical Markdown must start with one level-one title.")
    title = lines[0][2:].strip()
    body: list[str] = []
    for line in lines[1:]:
        match = re.match(r"^(#{2,6})(\s+.*)$", line)
        if match:
            line = match.group(1)[1:] + match.group(2)
        body.append(line)
    return title, "\n".join(body).strip() + "\n"


def _section(readme: str, heading_pattern: str) -> str:
    match = re.search(
        rf"^{heading_pattern}\s*\n(?P<body>.*?)(?=^##\s|\Z)",
        readme,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise BuildError(f"Caption section matching {heading_pattern!r} is missing.")
    return match.group("body").strip()


def extract_main_caption(readme_path: Path, label: str) -> str:
    caption = _section(readme_path.read_text(encoding="utf-8"), r"## Caption(?: draft)?")
    if not caption.startswith(f"**{label}."):
        raise BuildError(f"{readme_path} does not start its caption with {label}.")
    return caption


def extract_supplementary_caption(readme_path: Path, label: str) -> str:
    text = readme_path.read_text(encoding="utf-8")
    if re.search(r"^## Caption\s*$", text, flags=re.MULTILINE):
        caption = _section(text, r"## Caption")
        if caption.startswith(f"**{label}."):
            return caption
    inline = re.search(
        r"^\*\*Caption\.\*\*\s*(?P<body>.*?)(?=^`|^##\s|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if inline is None:
        raise BuildError(f"Caption is missing from {readme_path}.")
    return f"**{label}.** {inline.group('body').strip()}"


def _metadata(title: str, subtitle: str, source_date: str) -> str:
    return "\n".join(
        (
            "---",
            f'title: "{title}"',
            f'subtitle: "{subtitle}"',
            f'date: "{source_date}"',
            "lang: en-US",
            "link-citations: true",
            "---",
        )
    )


def _image_block(
    root: Path,
    image: Path,
    caption: str,
    *,
    continuation_label: str | None = None,
    width_inches: float = 6.25,
    caption_page_break: bool = False,
) -> str:
    if not image.is_file():
        raise BuildError(f"Required display item is missing: {image}")
    label = f"\n\n**{continuation_label}.**" if continuation_label else ""
    body = "" if continuation_label else caption
    caption_break = f"\n\n{PAGE_BREAK}" if caption_page_break else ""
    return (
        f"![]({relative(image, root)}){{width={width_inches:.2f}in}}"
        f"{label}{caption_break}\n\n{body}"
    ).rstrip()


def assemble_main(root: Path = REPOSITORY_ROOT) -> str:
    contract = load_json(root / "manuscript" / "publication_package" / "document_contract.json")
    manuscript_path = root / "manuscript" / "manuscript.md"
    bibliography_path = root / "manuscript" / "references.bib"
    title, body = shift_heading_hierarchy(manuscript_path.read_text(encoding="utf-8"))
    if not bibliography_path.is_file():
        raise BuildError(f"Bibliography is missing: {bibliography_path}")

    figure_blocks: list[str] = ["# Main figures"]
    for index, display in enumerate(MAIN_FIGURES):
        directory = root / "manuscript" / "figures" / display.directory_name
        if index and index != 1:
            figure_blocks.append(PAGE_BREAK)
        figure_blocks.append(
            _image_block(
                root,
                directory / "output" / display.image_names[0],
                extract_main_caption(directory / "README.md", display.label),
                caption_page_break=index != 1,
            )
        )

    return "\n\n".join(
        (
            _metadata(title, "Main manuscript", contract["source_date"]),
            body.rstrip(),
            "# References\n\n::: {#refs}\n:::",
            PAGE_BREAK,
            "\n\n".join(figure_blocks),
        )
    ) + "\n"


def assemble_supplementary_information(root: Path = REPOSITORY_ROOT) -> str:
    contract = load_json(root / "manuscript" / "publication_package" / "document_contract.json")
    source_path = root / "manuscript" / "supplementary-material.md"
    title, body = shift_heading_hierarchy(source_path.read_text(encoding="utf-8"))
    blocks: list[str] = [
        _metadata(title, "Supplementary information", contract["source_date"]),
        body.rstrip(),
        "# Additional supplementary figures",
    ]
    for figure_index, display in enumerate(SUPPLEMENTARY_FIGURES):
        directory = root / "manuscript" / "figures" / display.directory_name
        caption = extract_supplementary_caption(directory / "README.md", display.label)
        if figure_index:
            blocks.append(PAGE_BREAK)
        blocks.append(_image_block(root, directory / "output" / display.image_names[0], caption))
        for image_name in display.image_names[1:]:
            if display.continued_label is None:
                raise BuildError(f"{display.label} has multiple image parts without a continuation label.")
            blocks.extend(
                (
                    PAGE_BREAK,
                    _image_block(
                        root,
                        directory / "output" / image_name,
                        caption,
                        continuation_label=display.continued_label,
                    ),
                )
            )
    return "\n\n".join(blocks) + "\n"


def run_command(command: list[str], *, cwd: Path = REPOSITORY_ROOT) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, encoding="utf-8", errors="replace")
    combined = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    if result.returncode:
        raise BuildError(f"Command failed ({result.returncode}): {' '.join(command)}\n{combined}")
    assert_no_warnings(combined)
    return result


def assert_no_warnings(output: str) -> None:
    if re.search(r"(?im)^.*\bwarning(?:s)?\b.*$", output):
        raise BuildError(f"A conversion warning is fatal:\n{output}")


def assert_no_damaged_math(text: str, allowlist: Iterable[str]) -> None:
    allowed = set(allowlist)
    findings: list[str] = []
    for name, pattern in DAMAGED_MATH_PATTERNS:
        for match in pattern.finditer(text):
            token = match.group(0)
            if token not in allowed:
                findings.append(f"{name}: {token!r}")
    if findings:
        raise BuildError("Unallowlisted damaged-math signature(s): " + ", ".join(findings))


def math_nodes(ast: Any) -> list[str]:
    found: list[str] = []
    if isinstance(ast, dict):
        if ast.get("t") == "Math":
            content = ast.get("c", [])
            if isinstance(content, list) and len(content) == 2 and isinstance(content[1], str):
                found.append(content[1])
        for value in ast.values():
            found.extend(math_nodes(value))
    elif isinstance(ast, list):
        for value in ast:
            found.extend(math_nodes(value))
    return found


def inspect_docx_math(path: Path) -> dict[str, Any]:
    roots: list[ET.Element] = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.startswith("word/") and name.endswith(".xml"):
                roots.append(ET.fromstring(archive.read(name)))
    if not roots:
        raise BuildError(f"No WordprocessingML was found in {path}.")

    def q(local: str) -> str:
        return f"{{{MATH_NAMESPACE}}}{local}"

    native_omath = [
        "".join(text.text or "" for text in node.iter(q("t")))
        for root in roots
        for node in root.iter(q("oMath"))
    ]
    all_math_text = "".join(native_omath)
    structures = {
        key: sum(1 for root in roots for _ in root.iter(q(tag)))
        for key, tag in MATH_TAGS.items()
    }
    return {
        "native_omath": native_omath,
        "native_omath_count": len(native_omath),
        "structures": structures,
        "symbols": {symbol: symbol in all_math_text for symbol in CRITICAL_SYMBOLS},
    }


def _replace_zip_entry(path: Path, entry_name: str, replacement: bytes) -> None:
    with zipfile.ZipFile(path) as archive:
        entries = [(copy.copy(info), archive.read(info.filename)) for info in archive.infolist()]
    temporary = path.with_suffix(path.suffix + ".rewritten")
    with zipfile.ZipFile(temporary, "w") as archive:
        for info, data in entries:
            archive.writestr(info, replacement if info.filename == entry_name else data)
    temporary.replace(path)


def wrap_tables_in_landscape_sections(path: Path) -> None:
    """Give each top-level table a full landscape page without rasterizing it."""

    namespaces = {"w": WORD_NAMESPACE}
    def qn(local: str) -> str:
        return f"{{{WORD_NAMESPACE}}}{local}"

    def break_paragraph(*, landscape: bool) -> etree._Element:
        paragraph = etree.Element(qn("p"))
        properties = etree.SubElement(paragraph, qn("pPr"))
        section = etree.SubElement(properties, qn("sectPr"))
        section_type = etree.SubElement(section, qn("type"))
        section_type.set(qn("val"), "nextPage")
        page_size = etree.SubElement(section, qn("pgSz"))
        if landscape:
            page_size.set(qn("w"), "15840")
            page_size.set(qn("h"), "12240")
            page_size.set(qn("orient"), "landscape")
        else:
            page_size.set(qn("w"), "12240")
            page_size.set(qn("h"), "15840")
        margins = etree.SubElement(section, qn("pgMar"))
        for side in ("top", "right", "bottom", "left"):
            margins.set(qn(side), "1440")
        margins.set(qn("header"), "708")
        margins.set(qn("footer"), "708")
        margins.set(qn("gutter"), "0")
        return paragraph

    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    document = etree.fromstring(xml)
    body = document.find("w:body", namespaces)
    if body is None:
        raise BuildError(f"DOCX body is missing from {path}.")
    tables = list(body.findall("w:tbl", namespaces))
    for table in tables:
        rows = table.findall("w:tr", namespaces)
        for row_index, row in enumerate(rows):
            row_properties = row.find("w:trPr", namespaces)
            if row_properties is None:
                row_properties = etree.Element(qn("trPr"))
                row.insert(0, row_properties)
            if row_properties.find("w:cantSplit", namespaces) is None:
                etree.SubElement(row_properties, qn("cantSplit"))
            if row_index == 0 and row_properties.find("w:tblHeader", namespaces) is None:
                etree.SubElement(row_properties, qn("tblHeader"))
        index = body.index(table)
        body.insert(index, break_paragraph(landscape=False))
        body.insert(body.index(table) + 1, break_paragraph(landscape=True))
    transformed = etree.tostring(document, encoding="UTF-8", xml_declaration=True, standalone=True)
    _replace_zip_entry(path, "word/document.xml", transformed)


def normalize_docx(path: Path, source_date: str) -> None:
    """Remove build-clock variability from the OPC package and core properties."""

    fixed_zip_time = (1980, 1, 1, 0, 0, 0)
    fixed_w3cdtf = f"{source_date}T00:00:00Z"
    custom_properties = "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
    custom_value_types = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"
    ET.register_namespace("cp", "http://schemas.openxmlformats.org/package/2006/metadata/core-properties")
    ET.register_namespace("dc", "http://purl.org/dc/elements/1.1/")
    dcterms = "http://purl.org/dc/terms/"
    ET.register_namespace("dcterms", dcterms)
    ET.register_namespace("dcmitype", "http://purl.org/dc/dcmitype/")
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    ET.register_namespace("", custom_properties)
    ET.register_namespace("vt", custom_value_types)
    with zipfile.ZipFile(path) as archive:
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]

    normalized: list[tuple[zipfile.ZipInfo, bytes]] = []
    for original, data in entries:
        if original.filename == "docProps/core.xml":
            root = ET.fromstring(data)
            for local in ("created", "modified"):
                node = root.find(f"{{{dcterms}}}{local}")
                if node is not None:
                    node.text = fixed_w3cdtf
            data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        elif original.filename == "docProps/custom.xml":
            root = ET.fromstring(data)
            bibliography = root.find(f"{{{custom_properties}}}property[@name='bibliography']")
            if bibliography is not None:
                value = bibliography.find(f"{{{custom_value_types}}}lpwstr")
                if value is not None:
                    value.text = "manuscript/references.bib"
            data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        info = copy.copy(original)
        info.date_time = fixed_zip_time
        normalized.append((info, data))

    temporary = path.with_suffix(path.suffix + ".normalized")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for info, data in normalized:
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    temporary.replace(path)


def normalize_pdf(path: Path, *, document_sha256: str, source_date: str) -> None:
    """Rewrite a Word PDF with fixed metadata and a content-derived trailer ID."""

    reader = PdfReader(path)
    writer = PdfWriter()
    # Clone only the page tree and its referenced resources. Full-document
    # cloning retains an unreferenced Word XMP object with a clock time/UUID.
    writer.append_pages_from_reader(reader)
    fixed_pdf_date = source_date.replace("-", "")
    writer.add_metadata(
        {
            "/CreationDate": f"D:{fixed_pdf_date}000000Z",
            "/Creator": "TrackLEED canonical publication builder",
            "/ModDate": f"D:{fixed_pdf_date}000000Z",
            "/Producer": "Microsoft Word ExportAsFixedFormat; normalized by pypdf",
        }
    )
    identifier = bytes.fromhex(document_sha256)
    writer._ID = ArrayObject((ByteStringObject(identifier), ByteStringObject(identifier)))
    temporary = path.with_suffix(path.suffix + ".normalized")
    with temporary.open("wb") as handle:
        writer.write(handle)
    temporary.replace(path)


def normalize_math_expression(expression: str) -> str:
    """Normalize only notation changes introduced by Pandoc's DOCX round trip."""

    normalized = unicodedata.normalize("NFKC", expression)
    normalized = re.sub(r"\s+", "", normalized)
    for token in (r"\left.", r"\right.", r"\left", r"\right"):
        normalized = normalized.replace(token, "")
    replacements = {
        r"\lbrack": "[",
        r"\rbrack": "]",
        r"\parallel": r"\|",
        r"\qquad": "",
        r"\quad": "",
        r"\,": "",
        r"\;": "",
        r"\!": "",
    }
    for source, replacement in replacements.items():
        normalized = normalized.replace(source, replacement)
    normalized = re.sub(
        r"\\(?:mathrm|operatorname)\{([^{}]+)\}", r"\1", normalized
    )
    normalized = re.sub(r"([_^])\{([A-Za-z0-9+\-]+)\}", r"\1\2", normalized)
    return normalized


def math_expression_digest(expressions: list[str]) -> str:
    normalized = [normalize_math_expression(expression) for expression in expressions]
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_math_transport(
    *,
    document_name: str,
    ast_math: list[str],
    roundtrip_math: list[str],
    native_omath: list[str],
    fallback_count: int,
) -> None:
    native_omath_count = len(native_omath)
    if len(ast_math) != native_omath_count + fallback_count:
        raise BuildError(
            f"Math transport mismatch for {document_name}: AST Math={len(ast_math)}, "
            f"native OMath={native_omath_count}, fallbacks={fallback_count}."
        )
    if len(roundtrip_math) != native_omath_count:
        raise BuildError(
            f"Math round-trip mismatch for {document_name}: DOCX Math={len(roundtrip_math)}, "
            f"native OMath={native_omath_count}."
        )
    empty = [index for index, expression in enumerate(native_omath) if not expression.strip()]
    if empty:
        raise BuildError(f"Math transport contains empty native OMath in {document_name}: {empty}.")
    for index, (source, roundtrip) in enumerate(zip(ast_math, roundtrip_math)):
        if normalize_math_expression(source) != normalize_math_expression(roundtrip):
            raise BuildError(
                f"Math expression mismatch for {document_name} at index {index}: "
                f"source={source!r}, DOCX={roundtrip!r}."
            )


def _pandoc_source_arguments(source: Path, bibliography: Path) -> list[str]:
    resource_path = os.pathsep.join((str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "manuscript")))
    return [
        str(source),
        f"--from={PANDOC_READER}",
        "--citeproc",
        f"--bibliography={bibliography}",
        f"--resource-path={resource_path}",
        "--fail-if-warnings",
    ]


def _pandoc_ast(pandoc: Path, source: Path, bibliography: Path) -> tuple[dict[str, Any], str]:
    result = run_command(
        [str(pandoc), *_pandoc_source_arguments(source, bibliography), "--to=json"]
    )
    return json.loads(result.stdout), result.stderr


def _pandoc_docx(pandoc: Path, source: Path, bibliography: Path, output: Path) -> None:
    run_command(
        [
            str(pandoc),
            *_pandoc_source_arguments(source, bibliography),
            "--to=docx",
            "--toc",
            "--toc-depth=2",
            f"--output={output}",
        ]
    )


def _pandoc_docx_ast(pandoc: Path, docx: Path) -> dict[str, Any]:
    result = run_command(
        [str(pandoc), str(docx), "--from=docx", "--to=json", "--fail-if-warnings"]
    )
    return json.loads(result.stdout)


def docx_reader_facing_text(docx: Path) -> str:
    """Extract visible Word text while excluding native equation internals."""

    word_text = f"{{{WORD_NAMESPACE}}}t"
    word_paragraph = f"{{{WORD_NAMESPACE}}}p"
    math_prefix = f"{{{MATH_NAMESPACE}}}"

    def visit(node: ET.Element, *, inside_math: bool = False) -> list[str]:
        if node.tag.startswith(math_prefix) and not inside_math:
            return ["[MATH]"]
        inside_math = inside_math or node.tag.startswith(math_prefix)
        if node.tag == word_text and not inside_math:
            return [node.text or ""]
        parts: list[str] = []
        for child in node:
            parts.extend(visit(child, inside_math=inside_math))
        if node.tag == word_paragraph:
            parts.append("\n")
        return parts

    with zipfile.ZipFile(docx) as archive:
        names = [
            name
            for name in archive.namelist()
            if name == "word/document.xml"
            or re.fullmatch(r"word/(?:header|footer)\d+\.xml", name)
        ]
        return "".join(
            part
            for name in names
            for part in visit(ET.fromstring(archive.read(name)))
        )


def _pandoc_version(pandoc: Path) -> str:
    output = run_command([str(pandoc), "--version"]).stdout.splitlines()
    if not output:
        raise BuildError("Pandoc did not report its version.")
    return output[0]


def export_pdf(docx: Path, pdf: Path) -> None:
    script = PACKAGE_DIRECTORY / "export_word_pdf.ps1"
    run_command(
        [
            "powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", str(script), "-DocxPath", str(docx.resolve()), "-PdfPath", str(pdf.resolve()),
        ]
    )
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise BuildError(f"Word did not create a non-empty PDF: {pdf}")


def _document_audit(pandoc: Path, source: Path, docx: Path, document_name: str, fallback_count: int, allowlist: list[str]) -> dict[str, Any]:
    ast, _ = _pandoc_ast(pandoc, source, REPOSITORY_ROOT / "manuscript" / "references.bib")
    ast_math = math_nodes(ast)
    roundtrip_math = math_nodes(_pandoc_docx_ast(pandoc, docx))
    ooxml = inspect_docx_math(docx)
    assert_math_transport(
        document_name=document_name,
        ast_math=ast_math,
        roundtrip_math=roundtrip_math,
        native_omath=ooxml["native_omath"],
        fallback_count=fallback_count,
    )
    assert_no_damaged_math(docx_reader_facing_text(docx), allowlist)
    return {
        "ast_math_count": len(ast_math),
        "ast_math_sha256": math_expression_digest(ast_math),
        "fallback_count": fallback_count,
        "native_omath_count": ooxml["native_omath_count"],
        "roundtrip_math_count": len(roundtrip_math),
        "roundtrip_math_sha256": math_expression_digest(roundtrip_math),
        "structures": ooxml["structures"],
        "symbols": ooxml["symbols"],
    }


def _source_paths(root: Path) -> list[Path]:
    paths = [
        root / "manuscript" / "manuscript.md",
        root / "manuscript" / "supplementary-material.md",
        root / "manuscript" / "references.bib",
        CONTRACT_PATH,
        FALLBACK_PATH,
        Path(__file__).resolve(),
        PACKAGE_DIRECTORY / "export_word_pdf.ps1",
    ]
    for display in MAIN_FIGURES:
        directory = root / "manuscript" / "figures" / display.directory_name
        paths.extend((directory / "README.md", directory / "output" / display.image_names[0]))
    for display in SUPPLEMENTARY_FIGURES:
        directory = root / "manuscript" / "figures" / display.directory_name
        paths.append(directory / "README.md")
        paths.extend(directory / "output" / image_name for image_name in display.image_names)
    return paths


def build(*, pandoc: Path, export_pdfs: bool = True) -> dict[str, Any]:
    contract = load_json(CONTRACT_PATH)
    if contract["pandoc_reader"] != PANDOC_READER:
        raise BuildError("document_contract.json changed the required Pandoc reader.")
    fallback_contract = load_json(FALLBACK_PATH)
    fallbacks = fallback_contract["fallbacks"]
    if fallbacks:
        raise BuildError("The initial production fallback manifest must remain empty.")

    ASSEMBLED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    main_source = ASSEMBLED_DIRECTORY / "trackleed_manuscript.md"
    si_source = ASSEMBLED_DIRECTORY / "trackleed_supplementary_information.md"
    main_source.write_text(assemble_main(), encoding="utf-8", newline="\n")
    si_source.write_text(assemble_supplementary_information(), encoding="utf-8", newline="\n")

    documents = {
        "main": (main_source, OUTPUT_DIRECTORY / "trackleed_manuscript.docx", OUTPUT_DIRECTORY / "trackleed_manuscript.pdf"),
        "supplementary_information": (si_source, OUTPUT_DIRECTORY / "trackleed_supplementary_information.docx", OUTPUT_DIRECTORY / "trackleed_supplementary_information.pdf"),
    }
    audits: dict[str, Any] = {}
    artifact_manifest: dict[str, Any] = {}
    bibliography = REPOSITORY_ROOT / "manuscript" / "references.bib"
    for name, (source, docx, pdf) in documents.items():
        _pandoc_docx(pandoc, source, bibliography, docx)
        wrap_tables_in_landscape_sections(docx)
        normalize_docx(docx, contract["source_date"])
        audits[name] = _document_audit(pandoc, source, docx, name, 0, contract["damaged_math_allowlist"])
        if export_pdfs:
            export_pdf(docx, pdf)
            normalize_pdf(pdf, document_sha256=sha256_file(docx), source_date=contract["source_date"])
        if not pdf.is_file():
            raise BuildError(f"Canonical PDF is missing: {pdf}")
        artifact_manifest[name] = {
            "docx": {"path": relative(docx), "sha256": sha256_file(docx), "size_bytes": docx.stat().st_size},
            "pdf": {"path": relative(pdf), "sha256": sha256_file(pdf), "size_bytes": pdf.stat().st_size},
            "assembled_markdown": {"path": relative(source), "sha256": sha256_file(source)},
        }

    source_paths = _source_paths(REPOSITORY_ROOT)
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise BuildError("Manifest source is missing: " + ", ".join(str(path) for path in missing))
    manifest: dict[str, Any] = {
        "artifacts": artifact_manifest,
        "audit": audits,
        "builder_version": contract["builder_version"],
        "canonical_contract": relative(CONTRACT_PATH),
        "excluded_review_artifacts": contract["excluded_review_artifacts"],
        "math_fallbacks": {"count": 0, "path": relative(FALLBACK_PATH), "sha256": sha256_file(FALLBACK_PATH)},
        "pandoc": {"reader": PANDOC_READER, "version": _pandoc_version(pandoc)},
        "pdf_generator": "Microsoft Word ExportAsFixedFormat",
        "schema_version": "1.0",
        "source_date": contract["source_date"],
        "sources": {relative(path): sha256_file(path) for path in sorted(source_paths)},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def canonical_artifact_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        f"{document_name}/{artifact_type}": artifacts[artifact_type]["sha256"]
        for document_name, artifacts in manifest["artifacts"].items()
        for artifact_type in ("docx", "pdf")
    }


def build_reproducibly(*, pandoc: Path) -> dict[str, Any]:
    """Run two real exports and fail unless all four outputs are byte-identical."""

    first = build(pandoc=pandoc)
    first_hashes = canonical_artifact_hashes(first)
    second = build(pandoc=pandoc)
    second_hashes = canonical_artifact_hashes(second)
    if first_hashes != second_hashes:
        changed = sorted(
            name for name in first_hashes if first_hashes[name] != second_hashes[name]
        )
        raise BuildError(
            "Canonical publication outputs are not reproducible: " + ", ".join(changed)
        )
    return second


def verify_manifest() -> dict[str, Any]:
    manifest = load_json(MANIFEST_PATH)
    paths: dict[str, str] = {}
    for document_name, artifacts in manifest["artifacts"].items():
        for artifact_type in ("docx", "pdf"):
            record = artifacts[artifact_type]
            path = REPOSITORY_ROOT / record["path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise BuildError(f"Manifest hash mismatch: {document_name}/{artifact_type}")
            paths[f"{document_name}/{artifact_type}"] = record["path"]
    if len(set(paths.values())) != 4:
        raise BuildError("The canonical manifest does not identify four unique main/SI DOCX/PDF artifacts.")
    for source, expected in manifest["sources"].items():
        path = REPOSITORY_ROOT / source
        if not path.is_file() or sha256_file(path) != expected:
            raise BuildError(f"Manifest source hash mismatch: {source}")
    return manifest


def discover_pandoc(explicit: str | None) -> Path:
    candidates = [
        explicit,
        os.environ.get("PANDOC"),
        shutil.which("pandoc"),
        str(Path.home() / "AppData" / "Local" / "Pandoc" / "pandoc.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise BuildError("Pandoc was not found; pass --pandoc or set PANDOC.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pandoc", help="Path to the Pandoc executable.")
    parser.add_argument("--verify", action="store_true", help="Verify the committed manifest without rebuilding.")
    parser.add_argument(
        "--check-reproducible",
        action="store_true",
        help="Build twice with real Pandoc/Word exports and compare all four canonical hashes.",
    )
    parser.add_argument("--skip-pdf", action="store_true", help="Reuse existing PDFs (targeted development only).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify:
            manifest = verify_manifest()
        elif args.check_reproducible:
            if args.skip_pdf:
                raise BuildError("--check-reproducible cannot be combined with --skip-pdf.")
            manifest = build_reproducibly(pandoc=discover_pandoc(args.pandoc))
        else:
            manifest = build(
                pandoc=discover_pandoc(args.pandoc), export_pdfs=not args.skip_pdf
            )
    except (BuildError, OSError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"artifacts": manifest["artifacts"], "status": "ok"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
