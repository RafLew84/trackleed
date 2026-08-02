#!/usr/bin/env python3
"""Render every PDF page and create labelled contact sheets for visual QA."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz
from PIL import Image, ImageDraw


def render(pdf_path: Path, output_directory: Path, *, dpi: int = 110) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    page_paths: list[Path] = []
    for index, page in enumerate(document, start=1):
        output_path = output_directory / f"page-{index:03d}.png"
        page.get_pixmap(matrix=matrix, alpha=False).save(output_path)
        page_paths.append(output_path)
    return page_paths


def contact_sheets(page_paths: list[Path], output_directory: Path, *, columns: int = 2, rows: int = 3) -> list[Path]:
    if not page_paths:
        return []
    sizes: list[tuple[int, int]] = []
    for page_path in page_paths:
        with Image.open(page_path) as page:
            sizes.append(page.size)
    page_width = max(width for width, _ in sizes)
    page_height = max(height for _, height in sizes)
    label_height = 28
    sheet_paths: list[Path] = []
    per_sheet = columns * rows
    for sheet_number, offset in enumerate(range(0, len(page_paths), per_sheet), start=1):
        selected = page_paths[offset : offset + per_sheet]
        sheet = Image.new("RGB", (columns * page_width, rows * (page_height + label_height)), "#D9D9D9")
        draw = ImageDraw.Draw(sheet)
        for position, page_path in enumerate(selected):
            page = Image.open(page_path).convert("RGB")
            cell_x = (position % columns) * page_width
            y = (position // columns) * (page_height + label_height)
            x = cell_x + (page_width - page.width) // 2
            page_y = y + label_height + (page_height - page.height) // 2
            sheet.paste(page, (x, page_y))
            draw.text((cell_x + 8, y + 6), f"Page {offset + position + 1}", fill="black")
            page.close()
        output_path = output_directory / f"contact-{sheet_number:02d}.png"
        sheet.save(output_path, optimize=True)
        sheet_paths.append(output_path)
    return sheet_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()
    pages = render(args.pdf, args.output_directory, dpi=args.dpi)
    sheets = contact_sheets(pages, args.output_directory)
    print(f"Rendered {len(pages)} pages into {len(sheets)} contact sheets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
