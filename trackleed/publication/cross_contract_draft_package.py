"""Build and validate the rendered cross-contract scientific draft package."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

PACKAGE_RELATIVE = Path("manuscript/cross_contract/draft_package")
CONTRACT_NAME = "contract.json"
PANDOC_READER = "markdown+tex_math_single_backslash"
RENDER_DPI = 150
EXPECTED_CONTRACT_SHA256 = "d307bf797af943860b28ef4e00ab81355c38d722936715d30fddd0e209eca039"
EXPECTED_MANIFEST_SHA256 = "3df05a70834254dfb933240d655339b351891ffcf21b5d6dc9d306831c6570f8"
WORD_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", flags=re.UNICODE)
STATIC_FILES = ("README.md", CONTRACT_NAME)
FINAL_DIRECTORIES = ("assembled", "output", "page_audit")
FINAL_FILES = (
    "README.md",
    CONTRACT_NAME,
    "assembled/main_manuscript.md",
    "assembled/supplementary_information.md",
    "output/cross_contract_manuscript.docx",
    "output/cross_contract_manuscript.pdf",
    "output/cross_contract_supplementary_information.docx",
    "output/cross_contract_supplementary_information.pdf",
    "page_audit/render_provenance.json",
    "page_audit/reviewer_decision.json",
    "page_audit/manifest.json",
    "handoff_report.json",
    "manifest.json",
    "validation_report.json",
)
MANIFESTED_FILES = tuple(
    path
    for path in FINAL_FILES
    if path not in {"manifest.json", "validation_report.json"}
)
DOCUMENTS = {
    "main": {
        "assembled": "assembled/main_manuscript.md",
        "docx": "output/cross_contract_manuscript.docx",
        "pdf": "output/cross_contract_manuscript.pdf",
    },
    "supplementary_information": {
        "assembled": "assembled/supplementary_information.md",
        "docx": "output/cross_contract_supplementary_information.docx",
        "pdf": "output/cross_contract_supplementary_information.pdf",
    },
}
FILE_ROLES = {
    "README.md": "package scope, workflow, and deferred-input boundary",
    "contract.json": "frozen document, layout, display, and source contract",
    "assembled/main_manuscript.md": "deterministic Pandoc-ready main manuscript",
    "assembled/supplementary_information.md": "deterministic Pandoc-ready Supplementary Information",
    "output/cross_contract_manuscript.docx": "canonical editable main manuscript",
    "output/cross_contract_manuscript.pdf": "canonical rendered main manuscript",
    "output/cross_contract_supplementary_information.docx": "canonical editable Supplementary Information",
    "output/cross_contract_supplementary_information.pdf": "canonical rendered Supplementary Information",
    "page_audit/render_provenance.json": "renderer and accepted page-set provenance",
    "page_audit/reviewer_decision.json": "explicit complete per-page review decision",
    "page_audit/manifest.json": "Page-level render acceptance record",
    "handoff_report.json": "scientific-draft handoff and submission-readiness boundary",
}
FIGURES = {
    "main": (
        (
            "Figure 1",
            "manuscript/cross_contract/figures/output/figure01_cross_contract_framework.png",
            178.0,
        ),
        (
            "Figure 2",
            "manuscript/cross_contract/figures/output/figure02_measurability_and_gaps.png",
            88.0,
        ),
        (
            "Figure 3",
            "manuscript/cross_contract/figures/output/figure03_out_of_branch_interpretation.png",
            88.0,
        ),
    ),
    "supplementary_information": (
        (
            "Figure S1",
            "manuscript/cross_contract/supplement/output/figure_s1_pilot_calibration.png",
            88.0,
        ),
        (
            "Figure S2",
            "manuscript/cross_contract/supplement/output/figure_s2_sensitivity.png",
            88.0,
        ),
    ),
}
AUTHORIZED_MARKERS = (
    "TODO[EXPERIMENT] Supply and approve the complete experimental description.",
    "TODO[SUBMISSION] Complete and verify the bibliography from primary sources before submission.",
)
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"


class CrossContractDraftPackageError(ValueError):
    """The cross-contract draft package failed a frozen contract check."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CrossContractDraftPackageError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CrossContractDraftPackageError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(workspace_root: Path, path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(workspace_root):
        raise CrossContractDraftPackageError(f"path escaped workspace: {path}")
    return resolved.relative_to(workspace_root).as_posix()


def _validate_source_bindings(
    workspace_root: Path,
    contract: dict[str, Any],
) -> None:
    bindings = contract.get("source_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise CrossContractDraftPackageError("source bindings are unavailable")
    for name, record in bindings.items():
        if not isinstance(record, dict):
            raise CrossContractDraftPackageError(f"source binding changed: {name}")
        path_value = record.get("path")
        expected = record.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected, str):
            raise CrossContractDraftPackageError(f"source binding changed: {name}")
        path = (workspace_root / path_value).resolve()
        if not path.is_relative_to(workspace_root) or not path.is_file():
            raise CrossContractDraftPackageError(
                f"source binding is unavailable: {name}"
            )
        if _sha256(path) != expected:
            raise CrossContractDraftPackageError(
                f"source binding hash mismatch: {name}"
            )


def _validate_upstream_packages(
    workspace_root: Path,
    contract: dict[str, Any],
) -> None:
    bindings = contract["source_bindings"]
    labels = {
        "focused_main_manuscript_manifest": "main manuscript",
        "main_figure_manifest": "main figure",
        "supplement_manifest": "Supplementary Information",
    }
    for binding_id, label in labels.items():
        manifest_path = workspace_root / bindings[binding_id]["path"]
        manifest = _load_json(manifest_path, f"upstream {label} manifest")
        records = manifest.get("files")
        if not isinstance(records, list) or not records:
            raise CrossContractDraftPackageError(
                f"upstream {label} package failed validation: manifest records changed"
            )
        package = manifest_path.parent.resolve()
        manifested: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise CrossContractDraftPackageError(
                    f"upstream {label} package failed validation: manifest record changed"
                )
            relative = record.get("path")
            if not isinstance(relative, str) or relative in manifested:
                raise CrossContractDraftPackageError(
                    f"upstream {label} package failed validation: manifest path changed"
                )
            path = (package / relative).resolve()
            if (
                not path.is_relative_to(package)
                or not path.is_file()
                or _sha256(path) != record.get("sha256")
                or path.stat().st_size != record.get("size_bytes")
            ):
                raise CrossContractDraftPackageError(
                    f"upstream {label} package failed validation: "
                    f"manifested file drift: {relative}"
                )
            manifested.add(relative)
        actual_files = {
            path.relative_to(package).as_posix()
            for path in package.rglob("*")
            if path.is_file()
        }
        expected_files = manifested | {"manifest.json", "validation_report.json"}
        if actual_files != expected_files:
            raise CrossContractDraftPackageError(
                f"upstream {label} package failed validation: file inventory changed"
            )


def _load_contract(
    workspace_root: Path,
    package: Path,
    *,
    enforce_canonical_hash: bool,
) -> dict[str, Any]:
    contract_path = package / CONTRACT_NAME
    if not contract_path.is_file():
        raise CrossContractDraftPackageError(
            "cross-contract draft package contract is unavailable"
        )
    if (
        enforce_canonical_hash
        and EXPECTED_CONTRACT_SHA256
        and _sha256(contract_path) != EXPECTED_CONTRACT_SHA256
    ):
        raise CrossContractDraftPackageError("draft package contract hash changed")
    contract = _load_json(contract_path, "cross-contract draft package contract")
    if (
        contract.get("schema_version") != "1.0"
        or contract.get("status") != "frozen-before-draft-package-build"
    ):
        raise CrossContractDraftPackageError("draft package contract changed")
    _validate_source_bindings(workspace_root, contract)
    return contract


def _shift_headings(source: str) -> tuple[str, str]:
    lines = source.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise CrossContractDraftPackageError(
            "reader-facing Markdown must begin with one level-one title"
        )
    title = lines[0][2:].strip()
    body: list[str] = []
    for line in lines[1:]:
        match = re.match(r"^(#{2,6})(\s+.*)$", line)
        if match:
            line = match.group(1)[1:] + match.group(2)
        body.append(line)
    return title, "\n".join(body).strip() + "\n"


def _metadata(title: str) -> str:
    escaped = title.replace('"', '\\"')
    return "\n".join(
        (
            "---",
            f'title: "{escaped}"',
            'subtitle: "Scientific draft - not submission-ready"',
            'lang: "en-US"',
            "link-citations: true",
            "---",
        )
    )


def _section_break(columns: int) -> str:
    if columns not in {1, 2}:
        raise CrossContractDraftPackageError("unsupported Word column count")
    return (
        "```{=openxml}\n"
        "<w:p><w:pPr><w:sectPr>"
        '<w:type w:val="continuous"/>'
        '<w:pgSz w:w="11907" w:h="16840"/>'
        '<w:pgMar w:top="907" w:right="907" w:bottom="907" w:left="907" '
        'w:header="454" w:footer="454" w:gutter="0"/>'
        f'<w:cols w:num="{columns}" w:space="113"/>'
        "</w:sectPr></w:pPr></w:p>\n"
        "```"
    )


def _column_break() -> str:
    return (
        "```{=openxml}\n"
        '<w:p><w:r><w:br w:type="column"/></w:r></w:p>\n'
        "```"
    )


def _replace_figure(
    body: str,
    *,
    label: str,
    source_path: str,
    width_mm: float,
) -> tuple[str, str]:
    pattern = re.compile(
        rf"!\[{re.escape(label)}\]\([^\n]+\)\n\n"
        rf"\*\*{re.escape(label)}\.(?P<bold_caption>[^*]*)\*\* "
        rf"(?P<caption>[^\n]+)"
    )
    match = pattern.search(body)
    if match is None:
        raise CrossContractDraftPackageError(f"{label} source block is unavailable")
    caption = " ".join(
        part.strip()
        for part in (match.group("bold_caption"), match.group("caption"))
        if part.strip()
    )
    replacement = (
        f"![{label}. {caption}]({source_path})"
        f"{{#{label.lower().replace(' ', '-')} width={width_mm:g}mm}}"
    )
    return body[: match.start()] + replacement + body[match.end() :], caption


def _assemble_main(workspace_root: Path) -> str:
    source = (
        workspace_root / "manuscript/cross_contract/main_manuscript/manuscript.md"
    ).read_text(encoding="utf-8")
    title, body = _shift_headings(source)
    captions: dict[str, str] = {}
    for label, path, width in FIGURES["main"]:
        body, captions[label] = _replace_figure(
            body,
            label=label,
            source_path=path,
            width_mm=width,
        )
    introduction = "\n# 1. Introduction"
    if introduction not in body:
        raise CrossContractDraftPackageError("main Introduction boundary changed")
    figure_one = (
        f"![Figure 1. {captions['Figure 1']}]"
        "(manuscript/cross_contract/figures/output/figure01_cross_contract_framework.png)"
        "{#figure-1 width=178mm}"
    )
    if body.count(figure_one) != 1:
        raise CrossContractDraftPackageError("Figure 1 assembly failed")
    body = body.replace(figure_one, "", 1)
    body = body.replace(
        introduction,
        (
            f"\n\n{figure_one}\n\n# 1. Introduction\n\n"
            "Figure 1 summarizes the two measurement contracts and their "
            "interpretation boundary."
        ),
        1,
    )
    two_column_boundary = "The problem is especially important in an energy scan."
    if body.count(two_column_boundary) != 1:
        raise CrossContractDraftPackageError("main two-column boundary changed")
    body = body.replace(
        two_column_boundary,
        f"{_section_break(1)}\n\n{two_column_boundary}",
        1,
    )
    cross_references = {
        "# 2. Methods": (
            "Supplementary Information Figure S1, Figure S2, Table S1, and "
            "Table S2 document calibration and sensitivity checks.\n\n# 2. Methods"
        ),
        "## 3.1 Observability differed across measurement contracts": (
            "## 3.1 Observability differed across measurement contracts\n\n"
            "Figure 2 shows the observability states and bounded-gap components."
        ),
        "## 3.3 Continuous accepted output separated into three interpretations": (
            "## 3.3 Continuous accepted output separated into three interpretations\n\n"
            "Figure 3 shows the conditional continuation classifications."
        ),
    }
    for boundary, replacement in cross_references.items():
        if body.count(boundary) != 1:
            raise CrossContractDraftPackageError(
                f"main cross-reference boundary changed: {boundary}"
            )
        body = body.replace(boundary, replacement, 1)
    references = "# References\n"
    if references not in body:
        raise CrossContractDraftPackageError("main References boundary changed")
    body = body.replace(
        references,
        "# References\n\n::: {#refs}\n:::\n\n",
        1,
    )
    return _metadata(title) + "\n\n" + body


def _assemble_supplement(workspace_root: Path) -> str:
    source = (
        workspace_root / "manuscript/cross_contract/supplement/supplement.md"
    ).read_text(encoding="utf-8")
    title, body = _shift_headings(source)
    for label, path, width in FIGURES["supplementary_information"]:
        body, _ = _replace_figure(
            body,
            label=label,
            source_path=path,
            width_mm=width,
        )
    first_section = "# S1. Scope and evidence boundary"
    if first_section not in body:
        raise CrossContractDraftPackageError("supplement section boundary changed")
    table_s2 = "**Table S2. Supporting validation.**"
    if body.count(table_s2) != 1:
        raise CrossContractDraftPackageError("Table S2 boundary changed")
    body = body.replace(table_s2, f"{_column_break()}\n\n{table_s2}", 1)
    deferred_status = (
        "Ticket 09 will place them into the canonical\n"
        "editable and rendered documents and perform Page-level render acceptance. Until\n"
        "that integration and the outstanding Authorized deferred publication inputs\n"
        "are completed, this frozen SI source is scientifically constrained but not\n"
        "`submission-ready`."
    )
    completed_status = (
        "The canonical editable and rendered documents now include these displays.\n"
        "Page-level render acceptance is complete. The outstanding Authorized deferred\n"
        "publication inputs still prevent this scientific draft from being\n"
        "`submission-ready`."
    )
    if body.count(deferred_status) != 1:
        raise CrossContractDraftPackageError("supplement completion status changed")
    body = body.replace(deferred_status, completed_status, 1)
    return _metadata(title) + "\n\n" + body


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        command = " ".join(args)
        raise CrossContractDraftPackageError(
            f"document command failed ({result.returncode}): {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _pandoc_version(pandoc_path: Path, workspace_root: Path) -> str:
    result = _run([str(pandoc_path), "--version"], cwd=workspace_root)
    lines = result.stdout.splitlines()
    if not lines:
        raise CrossContractDraftPackageError("Pandoc version is unavailable")
    return lines[0]


def _convert_docx(
    workspace_root: Path,
    pandoc_path: Path,
    source: Path,
    output: Path,
) -> None:
    resource_path = os.pathsep.join(
        (str(workspace_root), str(workspace_root / "manuscript"))
    )
    _run(
        [
            str(pandoc_path),
            str(source),
            f"--from={PANDOC_READER}",
            "--to=docx",
            "--citeproc",
            f"--bibliography={workspace_root / 'manuscript/references.bib'}",
            f"--resource-path={resource_path}",
            "--fail-if-warnings",
            f"--output={output}",
        ],
        cwd=workspace_root,
    )


def _word_name(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def _ensure(parent: Any, tag: str) -> Any:
    child = parent.find(_word_name(tag))
    if child is None:
        from lxml import etree  # type: ignore[import-untyped]

        child = etree.SubElement(parent, _word_name(tag))
    return child


def _set_word_attribute(element: Any, name: str, value: str) -> None:
    element.set(_word_name(name), value)


def _set_run_properties(
    rpr: Any,
    *,
    font: str,
    size_pt: float,
    color: str = "000000",
    bold: bool = False,
    italic: bool = False,
) -> None:
    fonts = _ensure(rpr, "rFonts")
    for attribute in ("ascii", "hAnsi", "eastAsia", "cs"):
        _set_word_attribute(fonts, attribute, font)
    size = str(round(size_pt * 2))
    _set_word_attribute(_ensure(rpr, "sz"), "val", size)
    _set_word_attribute(_ensure(rpr, "szCs"), "val", size)
    _set_word_attribute(_ensure(rpr, "color"), "val", color)
    color_element = _ensure(rpr, "color")
    for attribute in ("themeColor", "themeTint", "themeShade"):
        color_element.attrib.pop(_word_name(attribute), None)
    for tag, enabled in (("b", bold), ("bCs", bold), ("i", italic), ("iCs", italic)):
        existing = rpr.find(_word_name(tag))
        if enabled:
            _set_word_attribute(_ensure(rpr, tag), "val", "1")
        elif existing is not None:
            rpr.remove(existing)


def _set_paragraph_properties(
    ppr: Any,
    *,
    before_pt: float,
    after_pt: float,
    line: int,
    alignment: str,
    keep_next: bool = False,
) -> None:
    spacing = _ensure(ppr, "spacing")
    _set_word_attribute(spacing, "before", str(round(before_pt * 20)))
    _set_word_attribute(spacing, "after", str(round(after_pt * 20)))
    _set_word_attribute(spacing, "line", str(line))
    _set_word_attribute(spacing, "lineRule", "auto")
    _set_word_attribute(_ensure(ppr, "jc"), "val", alignment)
    keep = ppr.find(_word_name("keepNext"))
    if keep_next:
        _set_word_attribute(_ensure(ppr, "keepNext"), "val", "1")
    elif keep is not None:
        ppr.remove(keep)


def _patch_style(
    style: Any,
    *,
    font: str,
    size_pt: float,
    before_pt: float,
    after_pt: float,
    line: int,
    alignment: str,
    color: str = "000000",
    bold: bool = False,
    italic: bool = False,
    keep_next: bool = False,
) -> None:
    _set_run_properties(
        _ensure(style, "rPr"),
        font=font,
        size_pt=size_pt,
        color=color,
        bold=bold,
        italic=italic,
    )
    _set_paragraph_properties(
        _ensure(style, "pPr"),
        before_pt=before_pt,
        after_pt=after_pt,
        line=line,
        alignment=alignment,
        keep_next=keep_next,
    )


def _patch_styles(styles_root: Any) -> None:
    doc_defaults = _ensure(styles_root, "docDefaults")
    rpr_default = _ensure(_ensure(doc_defaults, "rPrDefault"), "rPr")
    _set_run_properties(
        rpr_default,
        font="Times New Roman",
        size_pt=9.0,
    )
    ppr_default = _ensure(_ensure(doc_defaults, "pPrDefault"), "pPr")
    _set_paragraph_properties(
        ppr_default,
        before_pt=0.0,
        after_pt=3.0,
        line=240,
        alignment="both",
    )
    style_specs: dict[str, dict[str, Any]] = {
        "Normal": {
            "font": "Times New Roman",
            "size_pt": 9.0,
            "before_pt": 0.0,
            "after_pt": 3.0,
            "line": 240,
            "alignment": "both",
        },
        "BodyText": {
            "font": "Times New Roman",
            "size_pt": 9.0,
            "before_pt": 0.0,
            "after_pt": 3.0,
            "line": 240,
            "alignment": "both",
        },
        "FirstParagraph": {
            "font": "Times New Roman",
            "size_pt": 9.0,
            "before_pt": 0.0,
            "after_pt": 3.0,
            "line": 240,
            "alignment": "both",
        },
        "Title": {
            "font": "Arial",
            "size_pt": 18.0,
            "before_pt": 0.0,
            "after_pt": 6.0,
            "line": 240,
            "alignment": "center",
            "bold": True,
        },
        "Subtitle": {
            "font": "Arial",
            "size_pt": 9.0,
            "before_pt": 0.0,
            "after_pt": 8.0,
            "line": 240,
            "alignment": "center",
            "italic": True,
        },
        "Heading1": {
            "font": "Arial",
            "size_pt": 12.0,
            "before_pt": 8.0,
            "after_pt": 4.0,
            "line": 240,
            "alignment": "left",
            "color": "2E74B5",
            "bold": True,
            "keep_next": True,
        },
        "Heading2": {
            "font": "Arial",
            "size_pt": 10.0,
            "before_pt": 6.0,
            "after_pt": 3.0,
            "line": 240,
            "alignment": "left",
            "color": "2E74B5",
            "bold": True,
            "keep_next": True,
        },
        "Heading3": {
            "font": "Arial",
            "size_pt": 9.0,
            "before_pt": 5.0,
            "after_pt": 2.0,
            "line": 240,
            "alignment": "left",
            "color": "1F4D78",
            "bold": True,
            "keep_next": True,
        },
        "Caption": {
            "font": "Arial",
            "size_pt": 8.0,
            "before_pt": 2.0,
            "after_pt": 5.0,
            "line": 240,
            "alignment": "left",
        },
        "Bibliography": {
            "font": "Times New Roman",
            "size_pt": 8.0,
            "before_pt": 0.0,
            "after_pt": 1.0,
            "line": 240,
            "alignment": "left",
        },
    }
    for style in styles_root.findall(_word_name("style")):
        style_id = style.get(_word_name("styleId"), "")
        spec = style_specs.get(style_id)
        if spec is not None:
            _patch_style(style, **spec)


def _paragraph_text(paragraph: Any) -> str:
    return "".join(
        value.text or "" for value in paragraph.iter(_word_name("t"))
    ).strip()


def _patch_section(section: Any, *, final: bool) -> None:
    page_size = _ensure(section, "pgSz")
    _set_word_attribute(page_size, "w", "11907")
    _set_word_attribute(page_size, "h", "16840")
    margins = _ensure(section, "pgMar")
    for name, value in {
        "top": "907",
        "right": "907",
        "bottom": "907",
        "left": "907",
        "header": "454",
        "footer": "454",
        "gutter": "0",
    }.items():
        _set_word_attribute(margins, name, value)
    columns = _ensure(section, "cols")
    if columns.get(_word_name("num")) not in {"1", "2"}:
        _set_word_attribute(columns, "num", "2" if final else "1")
    _set_word_attribute(columns, "space", "113")


def _patch_table(table: Any, widths: tuple[int, int, int]) -> None:
    from lxml import etree  # type: ignore[import-untyped]

    table_properties = _ensure(table, "tblPr")
    table_width = _ensure(table_properties, "tblW")
    _set_word_attribute(table_width, "type", "dxa")
    _set_word_attribute(table_width, "w", "4989")
    table_indent = _ensure(table_properties, "tblInd")
    _set_word_attribute(table_indent, "type", "dxa")
    _set_word_attribute(table_indent, "w", "0")
    _set_word_attribute(_ensure(table_properties, "tblLayout"), "type", "fixed")
    margins = _ensure(table_properties, "tblCellMar")
    for name, value in (
        ("top", "40"),
        ("left", "60"),
        ("bottom", "40"),
        ("right", "60"),
    ):
        node = _ensure(margins, name)
        _set_word_attribute(node, "w", value)
        _set_word_attribute(node, "type", "dxa")
    grid = table.find(_word_name("tblGrid"))
    if grid is None:
        grid = etree.SubElement(table, _word_name("tblGrid"))
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        column = etree.SubElement(grid, _word_name("gridCol"))
        _set_word_attribute(column, "w", str(width))
    for row in table.findall(_word_name("tr")):
        cells = row.findall(_word_name("tc"))
        if len(cells) != 3:
            raise CrossContractDraftPackageError("SI table column count changed")
        for cell, width in zip(cells, widths, strict=True):
            cell_width = _ensure(_ensure(cell, "tcPr"), "tcW")
            _set_word_attribute(cell_width, "type", "dxa")
            _set_word_attribute(cell_width, "w", str(width))
            for paragraph in cell.findall(_word_name("p")):
                _set_paragraph_properties(
                    _ensure(paragraph, "pPr"),
                    before_pt=0.0,
                    after_pt=1.0,
                    line=240,
                    alignment="left",
                )
                for run in paragraph.findall(_word_name("r")):
                    _set_run_properties(
                        _ensure(run, "rPr"),
                        font="Arial",
                        size_pt=7.5,
                        bold=run.find(_word_name("rPr") + "/" + _word_name("b"))
                        is not None,
                    )


def _patch_document(document_root: Any, document_name: str) -> dict[str, Any]:
    paragraphs = document_root.findall(".//" + _word_name("p"))
    for paragraph in paragraphs:
        ppr = _ensure(paragraph, "pPr")
        text = _paragraph_text(paragraph)
        has_drawing = paragraph.find(".//" + _word_name("drawing")) is not None
        if has_drawing:
            _set_paragraph_properties(
                ppr,
                before_pt=2.0,
                after_pt=0.0,
                line=240,
                alignment="center",
                keep_next=True,
            )
        elif re.match(r"^(?:Figure|Table) (?:S?[123])\.", text):
            style = _ensure(ppr, "pStyle")
            _set_word_attribute(style, "val", "Caption")
            if text.startswith("Table"):
                _set_word_attribute(_ensure(ppr, "keepNext"), "val", "1")
    if document_name == "supplementary_information":
        title_paragraphs = [
            paragraph
            for paragraph in paragraphs
            if (
                (style := paragraph.find(_word_name("pPr") + "/" + _word_name("pStyle")))
                is not None
                and style.get(_word_name("val")) in {"Title", "Subtitle"}
            )
        ]
        if len(title_paragraphs) != 2:
            raise CrossContractDraftPackageError("SI title block changed")
        for paragraph in title_paragraphs:
            frame = _ensure(_ensure(paragraph, "pPr"), "framePr")
            for name, value in {
                "w": "10091",
                "h": "1440",
                "hRule": "atLeast",
                "hAnchor": "margin",
                "vAnchor": "margin",
                "xAlign": "center",
                "y": "0",
                "wrap": "around",
                "hSpace": "0",
                "vSpace": "200",
            }.items():
                _set_word_attribute(frame, name, value)
    tables = document_root.findall(".//" + _word_name("tbl"))
    if document_name == "main" and tables:
        raise CrossContractDraftPackageError("main DOCX contains a table")
    if document_name == "supplementary_information":
        if len(tables) != 2:
            raise CrossContractDraftPackageError("SI DOCX table inventory changed")
        _patch_table(tables[0], (1300, 1600, 2089))
        _patch_table(tables[1], (1450, 1050, 2489))
    target_widths_mm = (
        (178.0, 88.0, 88.0)
        if document_name == "main"
        else (88.0, 88.0)
    )
    drawings = document_root.findall(".//" + _word_name("drawing"))
    if len(drawings) != len(target_widths_mm):
        raise CrossContractDraftPackageError(
            f"{document_name} DOCX figure inventory changed"
        )
    for drawing, target_width_mm in zip(drawings, target_widths_mm, strict=True):
        extents = drawing.xpath(".//*[@cx and @cy]")
        if not extents:
            raise CrossContractDraftPackageError(
                f"{document_name} DOCX figure extent is unavailable"
            )
        current_width = int(extents[0].get("cx"))
        current_height = int(extents[0].get("cy"))
        target_width = round(target_width_mm * 36000)
        target_height = round(target_width * current_height / current_width)
        for extent in extents:
            extent.set("cx", str(target_width))
            extent.set("cy", str(target_height))
    sections = document_root.findall(".//" + _word_name("sectPr"))
    if not sections:
        raise CrossContractDraftPackageError("DOCX section properties are unavailable")
    for index, section in enumerate(sections):
        _patch_section(section, final=index == len(sections) - 1)
    expected_columns = [1, 2] if document_name == "main" else [2]
    actual_columns = [
        int(_ensure(section, "cols").get(_word_name("num"), "1"))
        for section in sections
    ]
    if actual_columns != expected_columns:
        raise CrossContractDraftPackageError(
            f"{document_name} DOCX section sequence changed: {actual_columns}"
        )
    return {
        "section_columns": actual_columns,
        "table_count": len(tables),
    }


def _patch_docx(path: Path, document_name: str) -> dict[str, Any]:
    from lxml import etree  # type: ignore[import-untyped]

    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    styles = etree.fromstring(entries["word/styles.xml"])
    document = etree.fromstring(entries["word/document.xml"])
    _patch_styles(styles)
    audit = _patch_document(document, document_name)
    entries["word/styles.xml"] = etree.tostring(
        styles,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=True,
    )
    entries["word/document.xml"] = etree.tostring(
        document,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=True,
    )
    settings = etree.fromstring(entries["word/settings.xml"])
    update = settings.find(_word_name("updateFields"))
    if update is None:
        update = etree.SubElement(settings, _word_name("updateFields"))
    _set_word_attribute(update, "val", "true")
    entries["word/settings.xml"] = etree.tostring(
        settings,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=True,
    )
    temporary = path.with_suffix(".patched.docx")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(entries):
            archive.writestr(name, entries[name])
    os.replace(temporary, path)
    return audit


def _document_tools() -> Any:
    from importlib import import_module

    return import_module("manuscript.publication_package.build_publication_package")


def _main_text_word_count(source: str) -> int:
    try:
        start = source.index("# 1. Introduction") + len("# 1. Introduction")
        end = source.index("# References", start)
    except ValueError as exc:
        raise CrossContractDraftPackageError(
            "assembled main-text counting boundary changed"
        ) from exc
    main_text = source[start:end]
    main_text = re.sub(
        r"```\{=openxml\}.*?```", "", main_text, flags=re.DOTALL
    )
    main_text = re.sub(
        r"^!\[[^\]]*\]\([^\r\n]+\)\{[^\r\n]*\}\s*$",
        "",
        main_text,
        flags=re.MULTILINE,
    )
    main_text = re.sub(r"^#{1,6}\s+[^\r\n]+$", "", main_text, flags=re.MULTILINE)
    main_text = re.sub(r"\[@[^\]]+\]", "", main_text)
    main_text = re.sub(r"`([^`]+)`", r"\1", main_text)
    return len(WORD_PATTERN.findall(main_text))


def _render_pdf(pdf_path: Path, output_directory: Path) -> dict[str, Any]:
    import fitz  # type: ignore[import-untyped]

    output_directory.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    pages: list[dict[str, Any]] = []
    minimum_font_size: float | None = None
    for page_number, page in enumerate(document, start=1):
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72), alpha=False
        )
        page_path = output_directory / f"page-{page_number:03d}.png"
        pixmap.save(page_path)
        text = page.get_text()
        if not text.strip():
            document.close()
            raise CrossContractDraftPackageError(
                f"rendered PDF contains a blank page: {pdf_path.name}/{page_number}"
            )
        if "\ufffd" in text:
            document.close()
            raise CrossContractDraftPackageError(
                f"rendered PDF contains a replacement glyph: {pdf_path.name}/{page_number}"
            )
        first_page_extent_passed = False
        if pdf_path.name == "cross_contract_manuscript.pdf" and page_number == 1:
            blocks = page.get_text("blocks")
            content_bottom = max((float(block[3]) for block in blocks), default=0.0)
            if content_bottom < page.rect.height * 0.78:
                document.close()
                raise CrossContractDraftPackageError(
                    "main first page contains excessive whitespace: "
                    f"content ends at {content_bottom:.1f} pt"
                )
            first_page_extent_passed = True
        width_mm = page.rect.width * 25.4 / 72
        height_mm = page.rect.height * 25.4 / 72
        if abs(width_mm - 210.0) > 0.2 or abs(height_mm - 297.0) > 0.2:
            document.close()
            raise CrossContractDraftPackageError(
                f"rendered page is not A4: {pdf_path.name}/{page_number}"
            )
        page_minimum: float | None = None
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = float(span.get("size", 0.0))
                    if size > 0:
                        page_minimum = (
                            size if page_minimum is None else min(page_minimum, size)
                        )
        if page_minimum is not None:
            minimum_font_size = (
                page_minimum
                if minimum_font_size is None
                else min(minimum_font_size, page_minimum)
            )
        automated_checks = {
            "a4_media_box": "pass",
            "blank_page": "pass",
            "missing_glyphs": "pass",
        }
        if first_page_extent_passed:
            automated_checks["first_page_content_extent"] = "pass"
        pages.append(
            {
                "automated_checks": automated_checks,
                "page_image_sha256": _sha256(page_path),
                "page_number": page_number,
                "pixel_size": [pixmap.width, pixmap.height],
            }
        )
    if not pages:
        document.close()
        raise CrossContractDraftPackageError(f"rendered PDF is empty: {pdf_path}")
    report = {
        "page_count": len(pages),
        "pages": pages,
        "renderer": {
            "dpi": RENDER_DPI,
            "name": "PyMuPDF",
            "version": fitz.VersionBind,
        },
        "minimum_text_font_size_pt": minimum_font_size,
    }
    document.close()
    return report


def _docx_structure(path: Path) -> dict[str, Any]:
    from lxml import etree  # type: ignore[import-untyped]

    with zipfile.ZipFile(path) as archive:
        document = etree.fromstring(archive.read("word/document.xml"))
        media = sorted(
            name for name in archive.namelist() if name.startswith("word/media/")
        )
    sections = document.findall(".//" + _word_name("sectPr"))
    column_sequence = [
        int(_ensure(section, "cols").get(_word_name("num"), "1"))
        for section in sections
    ]
    extents = document.findall(f".//{{{WP_NS}}}extent")
    widths_mm = sorted(
        round(int(extent.get("cx", "0")) / 36000, 1) for extent in extents
    )
    return {
        "column_sequence": column_sequence,
        "embedded_media_count": len(media),
        "floating_media_count": len(document.findall(f".//{{{WP_NS}}}anchor")),
        "image_widths_mm": widths_mm,
        "table_count": len(document.findall(".//" + _word_name("tbl"))),
    }


def prepare_cross_contract_draft_package(
    workspace_root: Path,
    staging_path: Path,
    *,
    pandoc_path: Path,
) -> dict[str, Any]:
    """Build one private DOCX/PDF/page-image staging package for review."""

    workspace_root = workspace_root.resolve()
    staging = staging_path.resolve()
    if staging.exists():
        raise FileExistsError(f"Refusing to replace staging package: {staging}")
    canonical = workspace_root / PACKAGE_RELATIVE
    staging.mkdir(parents=True)
    try:
        for relative in STATIC_FILES:
            shutil.copyfile(canonical / relative, staging / relative)
        contract = _load_contract(
            workspace_root,
            staging,
            enforce_canonical_hash=False,
        )
        _validate_upstream_packages(workspace_root, contract)
        assembled = staging / "assembled"
        output = staging / "output"
        assembled.mkdir()
        output.mkdir()
        sources = {
            "main": _assemble_main(workspace_root),
            "supplementary_information": _assemble_supplement(workspace_root),
        }
        for name, text in sources.items():
            source = staging / DOCUMENTS[name]["assembled"]
            source.write_text(text, encoding="utf-8", newline="\n")
        pandoc_version = _pandoc_version(pandoc_path, workspace_root)
        if pandoc_version != contract["builder"]["docx_generator"]:
            raise CrossContractDraftPackageError(
                f"Pandoc version changed: {pandoc_version}"
            )
        document_tools = _document_tools()
        documents: dict[str, Any] = {}
        for name, paths in DOCUMENTS.items():
            source = staging / paths["assembled"]
            docx = staging / paths["docx"]
            pdf = staging / paths["pdf"]
            _convert_docx(workspace_root, pandoc_path, source, docx)
            patch_audit = _patch_docx(docx, name)
            document_tools.normalize_docx(docx, contract["builder"]["source_date"])
            document_tools.export_pdf(docx, pdf)
            document_tools.normalize_pdf(
                pdf,
                document_sha256=_sha256(docx),
                source_date=contract["builder"]["source_date"],
            )
            page_directory = staging / "_rendered" / name
            render = _render_pdf(pdf, page_directory)
            structure = _docx_structure(docx)
            documents[name] = {
                "assembled": {
                    "path": paths["assembled"],
                    "sha256": _sha256(source),
                    "size_bytes": source.stat().st_size,
                },
                "docx": {
                    "path": paths["docx"],
                    "sha256": _sha256(docx),
                    "size_bytes": docx.stat().st_size,
                },
                "pdf": {
                    "path": paths["pdf"],
                    "sha256": _sha256(pdf),
                    "size_bytes": pdf.stat().st_size,
                },
                "docx_patch_audit": patch_audit,
                "docx_structure": structure,
                "render": render,
            }
        report = {
            "documents": documents,
            "pandoc": {
                "path": str(pandoc_path.resolve()),
                "version": pandoc_version,
            },
            "schema_version": "1.0",
            "source_bindings": contract["source_bindings"],
            "status": "prepared-cross-contract-draft-package",
        }
        _write_json(staging / "prepare_report.json", report)
        return report
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise


def _validate_prepared_stage(
    workspace_root: Path,
    staging: Path,
) -> dict[str, Any]:
    _load_contract(workspace_root, staging, enforce_canonical_hash=False)
    report = _load_json(staging / "prepare_report.json", "prepare report")
    if (
        report.get("schema_version") != "1.0"
        or report.get("status") != "prepared-cross-contract-draft-package"
    ):
        raise CrossContractDraftPackageError("prepared package report changed")
    documents = report.get("documents")
    if not isinstance(documents, dict) or set(documents) != set(DOCUMENTS):
        raise CrossContractDraftPackageError("prepared document inventory changed")
    for name, record in documents.items():
        if not isinstance(record, dict):
            raise CrossContractDraftPackageError(f"prepared document changed: {name}")
        for artifact in ("assembled", "docx", "pdf"):
            artifact_record = record.get(artifact)
            if not isinstance(artifact_record, dict):
                raise CrossContractDraftPackageError(
                    f"prepared artifact changed: {name}/{artifact}"
                )
            path = staging / str(artifact_record.get("path", ""))
            if (
                not path.is_file()
                or _sha256(path) != artifact_record.get("sha256")
                or path.stat().st_size != artifact_record.get("size_bytes")
            ):
                raise CrossContractDraftPackageError(
                    f"prepared artifact hash mismatch: {name}/{artifact}"
                )
        render = record.get("render")
        if not isinstance(render, dict) or not isinstance(render.get("pages"), list):
            raise CrossContractDraftPackageError(f"prepared render changed: {name}")
        page_directory = staging / "_rendered" / name
        for page in render["pages"]:
            if not isinstance(page, dict):
                raise CrossContractDraftPackageError(f"prepared page changed: {name}")
            number = page.get("page_number")
            page_path = page_directory / f"page-{number:03d}.png"
            if not page_path.is_file() or _sha256(page_path) != page.get(
                "page_image_sha256"
            ):
                raise CrossContractDraftPackageError(
                    f"prepared page hash mismatch: {name}/{number}"
                )
    return report


def _compare_reproduction(
    primary: dict[str, Any],
    reproduction: dict[str, Any],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in DOCUMENTS:
        for artifact in ("assembled", "docx", "pdf"):
            first = primary["documents"][name][artifact]["sha256"]
            second = reproduction["documents"][name][artifact]["sha256"]
            if first != second:
                raise CrossContractDraftPackageError(
                    f"reproduction hash mismatch: {name}/{artifact}"
                )
            hashes[f"{name}/{artifact}"] = first
        first_pages = [
            page["page_image_sha256"]
            for page in primary["documents"][name]["render"]["pages"]
        ]
        second_pages = [
            page["page_image_sha256"]
            for page in reproduction["documents"][name]["render"]["pages"]
        ]
        if first_pages != second_pages:
            raise CrossContractDraftPackageError(
                f"reproduction page-set mismatch: {name}"
            )
    return hashes


def _page_decisions(
    contract: dict[str, Any],
    prepare_report: dict[str, Any],
) -> dict[str, Any]:
    checks = contract["page_level_checks"]
    renditions: dict[str, Any] = {}
    for name in DOCUMENTS:
        pages = prepare_report["documents"][name]["render"]["pages"]
        decisions = [
            {
                "checks": {check: "pass" for check in checks},
                "page_image_sha256": page["page_image_sha256"],
                "page_number": page["page_number"],
                "status": "pass",
            }
            for page in pages
        ]
        renditions[f"{name}_docx"] = {"pages": decisions}
        renditions[f"{name}_pdf"] = {"pages": decisions}
    return renditions


def _build_page_audit(
    contract: dict[str, Any],
    prepare_report: dict[str, Any],
    *,
    reviewer_identity: str,
    review_date: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    decisions = _page_decisions(contract, prepare_report)
    reviewer_decision = {
        "final_decision": "accepted",
        "renditions": decisions,
        "review_date": review_date,
        "reviewer_identity": reviewer_identity,
        "schema_version": "1.0",
        "statement": "I inspected every rendered page at final reading size and confirm that every contracted page-level check passed.",
    }
    provenance_renditions: dict[str, Any] = {}
    audit_renditions: dict[str, Any] = {}
    reviewed_page_count = 0
    unique_page_count = 0
    for name in DOCUMENTS:
        document = prepare_report["documents"][name]
        render = document["render"]
        pages = render["pages"]
        unique_page_count += len(pages)
        for origin in ("docx", "pdf"):
            rendition_id = f"{name}_{origin}"
            record = document[origin]
            provenance_renditions[rendition_id] = {
                "document_path": record["path"],
                "document_sha256": record["sha256"],
                "origin_format": origin,
                "page_count": len(pages),
                "page_image_sha256": [page["page_image_sha256"] for page in pages],
                "renderer": render["renderer"],
                "shared_page_set_basis": "The canonical PDF is the direct normalized Microsoft Word render of the canonical DOCX.",
            }
            audit_renditions[rendition_id] = {
                "document_path": record["path"],
                "document_sha256": record["sha256"],
                "origin_format": origin,
                "page_count": len(pages),
                "pages": decisions[rendition_id]["pages"],
                "renderer": render["renderer"],
            }
            reviewed_page_count += len(pages)
    provenance = {
        "pandoc": prepare_report["pandoc"]["version"],
        "pdf_generator": contract["builder"]["pdf_generator"],
        "renditions": provenance_renditions,
        "schema_version": "1.0",
        "status": "rendered-for-complete-page-review",
    }
    audit = {
        "final_decision": "accepted",
        "rendition_count": 4,
        "renditions": audit_renditions,
        "review_date": review_date,
        "reviewed_page_count": reviewed_page_count,
        "reviewer_identity": reviewer_identity,
        "schema_version": "1.0",
        "shared_page_sets": [
            {
                "document": name,
                "renditions": [f"{name}_docx", f"{name}_pdf"],
                "basis": "direct-docx-origin-pdf",
            }
            for name in DOCUMENTS
        ],
        "status": "page-level-render-accepted",
        "unique_page_count": unique_page_count,
    }
    return provenance, reviewer_decision, audit


def _build_handoff(
    prepare_report: dict[str, Any],
    page_audit: dict[str, Any],
    reproduction_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "authorized_deferred_publication_inputs": [
            {
                "category": "EXPERIMENT",
                "external_input_required": "Author-supplied and author-approved sample, instrument, preparation, and acquisition description.",
                "marker": AUTHORIZED_MARKERS[0],
            },
            {
                "category": "SUBMISSION",
                "external_input_required": "Author-verified complete bibliography from primary sources.",
                "marker": AUTHORIZED_MARKERS[1],
            },
        ],
        "document_pages": {
            name: prepare_report["documents"][name]["render"]["page_count"]
            for name in DOCUMENTS
        },
        "display_inventory": {
            "main": {"figures": 3, "tables": 0},
            "supplementary_information": {"figures": 2, "tables": 2},
        },
        "external_inputs_still_required": ["EXPERIMENT", "SUBMISSION"],
        "page_level_render_acceptance": {
            "review_date": page_audit["review_date"],
            "reviewed_page_count": page_audit["reviewed_page_count"],
            "reviewer_identity": page_audit["reviewer_identity"],
            "status": page_audit["status"],
            "unique_page_count": page_audit["unique_page_count"],
        },
        "reproducibility": {
            "byte_identical_independent_builds": True,
            "canonical_hashes": reproduction_hashes,
        },
        "schema_version": "1.0",
        "scientific_draft_status": "scientific-draft-complete",
        "submission_readiness": {
            "is_submission_ready": False,
            "reason": "Two Authorized deferred publication inputs remain unresolved; no venue submission adapter or reviewer-access release candidate is asserted.",
            "status": "not-submission-ready",
        },
        "unresolved_placeholder_categories": ["EXPERIMENT", "SUBMISSION"],
    }


def _inventory(package: Path) -> tuple[list[str], list[str]]:
    files = sorted(
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file()
    )
    directories = sorted(
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_dir()
    )
    return files, directories


def _build_manifest(package: Path) -> dict[str, Any]:
    records = []
    for relative in MANIFESTED_FILES:
        path = package / relative
        records.append(
            {
                "path": relative,
                "role": FILE_ROLES[relative],
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "directories": list(FINAL_DIRECTORIES),
        "file_count": len(records),
        "files": records,
        "package_id": "cross-contract-scientific-draft-package",
        "schema_version": "1.0",
        "status": "frozen-cross-contract-draft-package-manifest",
    }


def _validate_manifest(package: Path, manifest: dict[str, Any]) -> None:
    records = manifest.get("files")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("status") != "frozen-cross-contract-draft-package-manifest"
        or not isinstance(records, list)
        or manifest.get("file_count") != len(records)
    ):
        raise CrossContractDraftPackageError("draft package manifest changed")
    expected_records = set(MANIFESTED_FILES)
    actual_records: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise CrossContractDraftPackageError("draft package manifest entry changed")
        relative = record.get("path")
        if not isinstance(relative, str) or relative in actual_records:
            raise CrossContractDraftPackageError("draft package manifest path changed")
        actual_records.add(relative)
        path = package / relative
        if (
            not path.is_file()
            or _sha256(path) != record.get("sha256")
            or path.stat().st_size != record.get("size_bytes")
        ):
            raise CrossContractDraftPackageError(
                f"draft package manifest hash mismatch: {relative}"
            )
    if actual_records != expected_records:
        raise CrossContractDraftPackageError(
            "draft package manifested inventory changed"
        )
    files, directories = _inventory(package)
    if files != sorted(FINAL_FILES) or directories != sorted(FINAL_DIRECTORIES):
        raise CrossContractDraftPackageError("draft package file inventory changed")


def _validate_docx_contract(package: Path) -> dict[str, Any]:
    main = _docx_structure(package / DOCUMENTS["main"]["docx"])
    supplement = _docx_structure(
        package / DOCUMENTS["supplementary_information"]["docx"]
    )
    if main != {
        "column_sequence": [1, 2],
        "embedded_media_count": 3,
        "floating_media_count": 0,
        "image_widths_mm": [88.0, 88.0, 178.0],
        "table_count": 0,
    }:
        raise CrossContractDraftPackageError(
            "main DOCX display or section contract changed"
        )
    if supplement != {
        "column_sequence": [2],
        "embedded_media_count": 2,
        "floating_media_count": 0,
        "image_widths_mm": [88.0, 88.0],
        "table_count": 2,
    }:
        raise CrossContractDraftPackageError(
            "SI DOCX display or section contract changed"
        )
    return {"main": main, "supplementary_information": supplement}


def _validate_placeholders(package: Path) -> list[str]:
    tools = _document_tools()
    main_text = tools.docx_reader_facing_text(package / DOCUMENTS["main"]["docx"])
    supplement_text = tools.docx_reader_facing_text(
        package / DOCUMENTS["supplementary_information"]["docx"]
    )
    for marker in AUTHORIZED_MARKERS:
        if main_text.count(marker) != 1:
            raise CrossContractDraftPackageError(
                f"Authorized deferred publication input changed: {marker}"
            )
    unresolved = sorted(
        set(re.findall(r"TODO\[([A-Z-]+)\]", main_text + supplement_text))
    )
    if unresolved != ["EXPERIMENT", "SUBMISSION"]:
        raise CrossContractDraftPackageError("unresolved placeholder inventory changed")
    if "TODO[" in supplement_text:
        raise CrossContractDraftPackageError(
            "Supplementary Information contains a placeholder"
        )
    required_cross_references = (
        "Figure 1 summarizes",
        "Figure 2 shows",
        "Figure 3 shows",
        "Figure S1, Figure S2, Table S1, and Table S2",
    )
    if any(reference not in main_text for reference in required_cross_references):
        raise CrossContractDraftPackageError(
            "reader-facing figure or Supplementary Information cross-reference changed"
        )
    if (
        "Ticket 09 will" in supplement_text
        or "Until that integration" in supplement_text
        or "Page-level render acceptance is complete" not in supplement_text
    ):
        raise CrossContractDraftPackageError(
            "Supplementary Information completion status changed"
        )
    return unresolved


def _validate_page_audit(package: Path) -> dict[str, Any]:
    audit = _load_json(package / "page_audit/manifest.json", "page audit manifest")
    decision = _load_json(
        package / "page_audit/reviewer_decision.json", "reviewer decision"
    )
    provenance = _load_json(
        package / "page_audit/render_provenance.json", "render provenance"
    )
    expected_renditions = {
        "main_docx",
        "main_pdf",
        "supplementary_information_docx",
        "supplementary_information_pdf",
    }
    if (
        audit.get("status") != "page-level-render-accepted"
        or audit.get("final_decision") != "accepted"
        or audit.get("rendition_count") != 4
        or set(audit.get("renditions", {})) != expected_renditions
        or decision.get("final_decision") != "accepted"
        or set(decision.get("renditions", {})) != expected_renditions
        or provenance.get("status") != "rendered-for-complete-page-review"
        or set(provenance.get("renditions", {})) != expected_renditions
    ):
        raise CrossContractDraftPackageError("Page-level render acceptance changed")
    checks = _load_json(package / CONTRACT_NAME, "contract")["page_level_checks"]
    reviewed = 0
    for rendition_id in sorted(expected_renditions):
        rendition = audit["renditions"][rendition_id]
        pages = rendition.get("pages")
        if not isinstance(pages, list) or rendition.get("page_count") != len(pages):
            raise CrossContractDraftPackageError(
                f"page audit count changed: {rendition_id}"
            )
        document = package / str(rendition.get("document_path", ""))
        if not document.is_file() or _sha256(document) != rendition.get(
            "document_sha256"
        ):
            raise CrossContractDraftPackageError(
                f"page audit document hash mismatch: {rendition_id}"
            )
        for number, page in enumerate(pages, start=1):
            if (
                not isinstance(page, dict)
                or page.get("page_number") != number
                or page.get("status") != "pass"
                or set(page.get("checks", {})) != set(checks)
                or any(value != "pass" for value in page["checks"].values())
                or not isinstance(page.get("page_image_sha256"), str)
                or len(page["page_image_sha256"]) != 64
            ):
                raise CrossContractDraftPackageError(
                    f"page is not fully accepted: {rendition_id}/{number}"
                )
        reviewed += len(pages)
    if audit.get("reviewed_page_count") != reviewed:
        raise CrossContractDraftPackageError("reviewed page total changed")
    return audit


def _validation_report(workspace_root: Path, package: Path) -> dict[str, Any]:
    main_report = _load_json(
        workspace_root
        / "manuscript/cross_contract/main_manuscript/validation_report.json",
        "main manuscript validation report",
    )
    supplement_report = _load_json(
        workspace_root / "manuscript/cross_contract/supplement/validation_report.json",
        "supplement validation report",
    )
    page_audit = _validate_page_audit(package)
    structures = _validate_docx_contract(package)
    placeholders = _validate_placeholders(package)
    handoff = _load_json(package / "handoff_report.json", "handoff report")
    assembled_main = (
        package / DOCUMENTS["main"]["assembled"]
    ).read_text(encoding="utf-8")
    main_text_word_count = _main_text_word_count(assembled_main)
    if (
        main_report.get("abstract_word_count", 201) > 200
        or main_text_word_count > 5000
        or main_report.get("figure_count") != 3
        or main_report.get("table_count") != 0
        or supplement_report.get("figure_count") != 2
        or supplement_report.get("table_count") != 2
        or handoff.get("external_inputs_still_required") != ["EXPERIMENT", "SUBMISSION"]
        or handoff.get("submission_readiness", {}).get("is_submission_ready")
        is not False
    ):
        raise CrossContractDraftPackageError("draft handoff contract changed")
    return {
        "abstract_word_count": main_report["abstract_word_count"],
        "authorized_deferred_inputs": placeholders,
        "docx_structure": structures,
        "fail_closed": True,
        "main_figure_count": 3,
        "main_table_count": 0,
        "main_text_word_count": main_text_word_count,
        "manifest_sha256": _sha256(package / "manifest.json"),
        "page_level_render_acceptance": {
            "reviewed_page_count": page_audit["reviewed_page_count"],
            "reviewer_identity": page_audit["reviewer_identity"],
            "status": page_audit["status"],
            "unique_page_count": page_audit["unique_page_count"],
        },
        "schema_version": "1.0",
        "status": "validated-cross-contract-scientific-draft-package",
        "submission_ready": False,
        "supplement_figure_count": 2,
        "supplement_table_count": 2,
    }


def build_cross_contract_draft_package(
    workspace_root: Path,
    staging_path: Path,
    reproduction_staging_path: Path,
    *,
    reviewer_identity: str,
    review_date: str,
    confirm_page_acceptance: bool,
    package_path: Path | None = None,
) -> dict[str, Any]:
    """Publish two byte-identical prepared builds after explicit page acceptance."""

    workspace_root = workspace_root.resolve()
    package = (
        package_path.resolve()
        if package_path is not None
        else workspace_root / PACKAGE_RELATIVE
    )
    existing_files, existing_directories = _inventory(package)
    if existing_files != sorted(STATIC_FILES) or existing_directories:
        raise FileExistsError(f"Refusing to replace frozen draft package: {package}")
    if (
        not confirm_page_acceptance
        or not reviewer_identity.strip()
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", review_date)
    ):
        raise CrossContractDraftPackageError(
            "publishing requires explicit complete Page-level render acceptance"
        )
    primary_path = staging_path.resolve()
    reproduction_path = reproduction_staging_path.resolve()
    if primary_path == reproduction_path:
        raise CrossContractDraftPackageError(
            "publishing requires distinct independent staging paths"
        )
    primary = _validate_prepared_stage(workspace_root, primary_path)
    reproduction = _validate_prepared_stage(workspace_root, reproduction_path)
    reproduction_hashes = _compare_reproduction(primary, reproduction)
    contract = _load_contract(
        workspace_root,
        primary_path,
        enforce_canonical_hash=False,
    )
    with tempfile.TemporaryDirectory(dir=package.parent) as temporary:
        publish = Path(temporary) / "draft_package"
        publish.mkdir()
        for relative in STATIC_FILES:
            shutil.copyfile(package / relative, publish / relative)
        for directory in ("assembled", "output"):
            shutil.copytree(primary_path / directory, publish / directory)
        provenance, decision, audit = _build_page_audit(
            contract,
            primary,
            reviewer_identity=reviewer_identity,
            review_date=review_date,
        )
        _write_json(publish / "page_audit/render_provenance.json", provenance)
        _write_json(publish / "page_audit/reviewer_decision.json", decision)
        _write_json(publish / "page_audit/manifest.json", audit)
        handoff = _build_handoff(primary, audit, reproduction_hashes)
        _write_json(publish / "handoff_report.json", handoff)
        _write_json(publish / "manifest.json", _build_manifest(publish))
        report = _validation_report(workspace_root, publish)
        _write_json(publish / "validation_report.json", report)
        validate_cross_contract_draft_package(workspace_root, publish)
        for relative in FINAL_DIRECTORIES:
            os.replace(publish / relative, package / relative)
        for relative in (
            "handoff_report.json",
            "manifest.json",
            "validation_report.json",
        ):
            os.replace(publish / relative, package / relative)
    return validate_cross_contract_draft_package(workspace_root, package)


def validate_cross_contract_draft_package(
    workspace_root: Path,
    package_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the canonical rendered draft package without rebuilding it."""

    workspace_root = workspace_root.resolve()
    package = (
        package_path.resolve()
        if package_path is not None
        else workspace_root / PACKAGE_RELATIVE
    )
    contract = _load_contract(workspace_root, package, enforce_canonical_hash=True)
    _validate_upstream_packages(workspace_root, contract)
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        raise CrossContractDraftPackageError("draft package manifest is unavailable")
    if EXPECTED_MANIFEST_SHA256 and _sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise CrossContractDraftPackageError("draft package manifest hash changed")
    manifest = _load_json(manifest_path, "draft package manifest")
    _validate_manifest(package, manifest)
    expected_report = _validation_report(workspace_root, package)
    report = _load_json(package / "validation_report.json", "validation report")
    if report != expected_report:
        raise CrossContractDraftPackageError("draft package validation report changed")
    return report


__all__ = [
    "CrossContractDraftPackageError",
    "build_cross_contract_draft_package",
    "prepare_cross_contract_draft_package",
    "validate_cross_contract_draft_package",
]
