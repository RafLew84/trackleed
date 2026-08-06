#!/usr/bin/env python3
"""Assemble the review manuscript with main figures and bibliography target."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


REVIEW_DIRECTORY = Path(__file__).resolve().parent
MANUSCRIPT_DIRECTORY = REVIEW_DIRECTORY.parent
REPOSITORY_ROOT = MANUSCRIPT_DIRECTORY.parent
MANUSCRIPT_PATH = MANUSCRIPT_DIRECTORY / "manuscript.md"
BIBLIOGRAPHY_PATH = MANUSCRIPT_DIRECTORY / "references.bib"
OUTPUT_PATH = REVIEW_DIRECTORY / "article_review.md"
SOURCE_MANIFEST_PATH = REVIEW_DIRECTORY / "source_manifest.json"

BUILD_DATE = "2026-07-30"
FIGURES = (
    (
        "Figure 1",
        MANUSCRIPT_DIRECTORY
        / "figures"
        / "figure01_measurement_chain"
        / "output"
        / "figure01_measurement_chain.png",
        MANUSCRIPT_DIRECTORY / "figures" / "figure01_measurement_chain" / "README.md",
    ),
    (
        "Figure 2",
        MANUSCRIPT_DIRECTORY
        / "figures"
        / "figure02_controlled_equivariance"
        / "output"
        / "figure02_controlled_equivariance.png",
        MANUSCRIPT_DIRECTORY
        / "figures"
        / "figure02_controlled_equivariance"
        / "README.md",
    ),
    (
        "Figure 3",
        MANUSCRIPT_DIRECTORY
        / "figures"
        / "figure03_independent_scane_stress"
        / "output"
        / "figure03_independent_scane_stress.png",
        MANUSCRIPT_DIRECTORY
        / "figures"
        / "figure03_independent_scane_stress"
        / "README.md",
    ),
    (
        "Figure 4",
        MANUSCRIPT_DIRECTORY
        / "figures"
        / "figure04_controlled_sam2_memory_tradeoff"
        / "output"
        / "figure04_controlled_sam2_memory_tradeoff.png",
        MANUSCRIPT_DIRECTORY
        / "figures"
        / "figure04_controlled_sam2_memory_tradeoff"
        / "README.md",
    ),
    (
        "Figure 5",
        MANUSCRIPT_DIRECTORY
        / "figures"
        / "figure05_cu111_physical_endpoint"
        / "output"
        / "figure05_cu111_physical_endpoint.png",
        MANUSCRIPT_DIRECTORY
        / "figures"
        / "figure05_cu111_physical_endpoint"
        / "README.md",
    ),
)

PAGE_BREAK = """```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_caption(readme_path: Path, figure_label: str) -> str:
    text = readme_path.read_text(encoding="utf-8")
    match = re.search(
        r"^## Caption(?: draft)?\s*\n(?P<caption>.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Caption section missing from {readme_path}.")
    caption = match.group("caption").strip()
    if not caption.startswith(f"**{figure_label}."):
        raise ValueError(f"Caption label changed in {readme_path}.")
    return caption


def bibliography_keys(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"^@\w+\{([^,]+),", text, flags=re.MULTILINE))


def cited_keys(text: str) -> set[str]:
    return set(re.findall(r"(?<![\w])@([A-Za-z0-9_:-]+)", text))


def shift_heading_hierarchy(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("The manuscript must start with one level-one title.")
    title = lines[0][2:].strip()
    body_lines: list[str] = []
    for line in lines[1:]:
        match = re.match(r"^(#{2,6})(\s+.*)$", line)
        if match:
            line = match.group(1)[1:] + match.group(2)
        body_lines.append(line)
    return title, "\n".join(body_lines).strip() + "\n"


def figure_section() -> str:
    blocks = ["# Main figures"]
    for index, (label, image_path, readme_path) in enumerate(FIGURES):
        if index:
            blocks.append(PAGE_BREAK)
        relative_image = image_path.relative_to(REPOSITORY_ROOT).as_posix()
        caption = extract_caption(readme_path, label)
        blocks.extend(
            (
                f"![]({relative_image}){{width=6.25in}}",
                caption,
            )
        )
    return "\n\n".join(blocks) + "\n"


def build() -> dict[str, object]:
    manuscript = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    title, body = shift_heading_hierarchy(manuscript)
    missing_citations = cited_keys(manuscript) - bibliography_keys(
        BIBLIOGRAPHY_PATH
    )
    if missing_citations:
        raise ValueError(
            "Missing bibliography records: "
            + ", ".join(sorted(missing_citations))
        )
    for _, image_path, readme_path in FIGURES:
        if not image_path.is_file() or not readme_path.is_file():
            raise ValueError(f"Missing figure source: {image_path}")

    metadata = "\n".join(
        (
            "---",
            f'title: "{title}"',
            'subtitle: "Review manuscript"',
            f'date: "{BUILD_DATE}"',
            "lang: en-US",
            "link-citations: true",
            "---",
        )
    )
    assembled = "\n\n".join(
        (
            metadata,
            body.rstrip(),
            "# References\n\n::: {#refs}\n:::",
            PAGE_BREAK,
            figure_section().rstrip(),
        )
    ) + "\n"
    OUTPUT_PATH.write_text(assembled, encoding="utf-8", newline="\n")

    source_paths = [
        MANUSCRIPT_PATH,
        BIBLIOGRAPHY_PATH,
        Path(__file__).resolve(),
    ]
    for _, image_path, readme_path in FIGURES:
        source_paths.extend((image_path, readme_path))
    manifest: dict[str, object] = {
        "build_date": BUILD_DATE,
        "citation_count": len(cited_keys(manuscript)),
        "figure_count": len(FIGURES),
        "output_markdown": {
            OUTPUT_PATH.relative_to(REPOSITORY_ROOT).as_posix(): (
                sha256_file(OUTPUT_PATH)
            )
        },
        "schema_version": "1.0",
        "sources": {
            path.relative_to(REPOSITORY_ROOT).as_posix(): sha256_file(path)
            for path in source_paths
        },
        "todo_marker_count": assembled.count("TODO["),
    }
    SOURCE_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


if __name__ == "__main__":
    build()
