"""
PDF Ingestion Module

Extracts page-anchored text and table chunks from PDF documents
matching the Fact Knowledge Layer schema:
    {
        "doc_id": str,
        "page_number": int,    # 1-indexed
        "text": str,
        "chunk_type": "text" | "table"
    }
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import pymupdf

# Suppress PyMuPDF optional layout analyzer recommendation message
os.environ["PYMUPDF_SUGGEST_LAYOUT_ANALYZER"] = "0"
if hasattr(pymupdf, "no_recommend_layout"):
    pymupdf.no_recommend_layout()


class Chunk(TypedDict):
    doc_id: str
    page_number: int
    text: str
    chunk_type: str


def _format_table_raw(table_obj: Any, page: pymupdf.Page) -> Tuple[str, List[pymupdf.Rect]]:
    """
    Extracts raw text from a detected table, robustly preserving row labels on borderless tables.
    Returns:
        (formatted_text, absorbed_rectangles)
    """
    t_bbox = getattr(table_obj, "bbox", None)
    if not t_bbox:
        return "", []

    t_rect = pymupdf.Rect(t_bbox)
    extracted = table_obj.extract() or []

    # Detect if table has missing first-column cells (e.g. borderless row labels)
    has_empty_col0 = any(
        r and len(r) > 0 and (r[0] is None or str(r[0]).strip() == "")
        for r in extracted
    )

    # Detect if to_markdown() produced duplicate repeated header labels down col 0
    md_text = ""
    try:
        md_text = table_obj.to_markdown() or ""
    except Exception:
        pass

    repeated_header = False
    if md_text:
        lines = [l for l in md_text.splitlines() if l.startswith("|")]
        if len(lines) >= 3:
            first_cells = [l.split("|")[1].strip() for l in lines[2:]]
            if len(first_cells) >= 3 and len(set(first_cells)) == 1:
                repeated_header = True

    absorbed_rects = [t_rect]

    # If borderless or duplicated, merge adjacent and overlapping text blocks
    if has_empty_col0 or repeated_header:
        expanded_rect = pymupdf.Rect(
            max(0, t_rect.x0 - 120),
            t_rect.y0 - 5,
            t_rect.x1 + 10,
            t_rect.y1 + 5,
        )
        blocks = []
        for b in page.get_text("blocks"):
            b_rect = pymupdf.Rect(b[:4])
            if not (b_rect & expanded_rect).is_empty and b[6] == 0:
                blocks.append((b_rect, b[4].strip()))
                absorbed_rects.append(b_rect)

        blocks.sort(key=lambda item: item[0].y0)

        merged_rows = []
        for _, text in blocks:
            raw_lines = [l.strip() for l in text.splitlines() if l.strip()]
            if not raw_lines:
                continue
            merged_rows.append(" | ".join(raw_lines))

        if merged_rows:
            return "\n".join(merged_rows), absorbed_rects

    # Standard table fallback
    if md_text and md_text.strip():
        return md_text.strip(), absorbed_rects
    elif extracted:
        rows = [
            " | ".join(str(c).strip() if c is not None else "" for c in r)
            for r in extracted
        ]
        return "\n".join(rows), absorbed_rects

    return page.get_text("text", clip=t_rect).strip(), absorbed_rects


def extract_chunks(
    pdf_path: str | Path,
    doc_id: Optional[str] = None,
) -> List[Chunk]:
    """
    Given a PDF file path, extracts page-anchored chunks from each page.

    Tables are detected and tagged as `chunk_type: "table"`.
    Prose text blocks outside table boundaries are tagged as `chunk_type: "text"`.

    Args:
        pdf_path: Path to the PDF file.
        doc_id: Unique document identifier. Defaults to the filename stem.

    Returns:
        A list of Chunk dictionaries with keys:
        {doc_id, page_number, text, chunk_type}
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Provided path is not a file: {path}")

    effective_doc_id = doc_id if doc_id else path.stem
    chunks: List[Chunk] = []

    doc = pymupdf.open(str(path))
    try:
        for page_idx in range(len(doc)):
            page_number = page_idx + 1
            page = doc[page_idx]

            # 1. Detect tables on the page
            table_rects: List[pymupdf.Rect] = []
            try:
                tabs = page.find_tables()
                tables = tabs.tables if tabs else []
            except Exception:
                tables = []

            for tab in tables:
                raw_table_text, absorbed = _format_table_raw(tab, page)
                table_rects.extend(absorbed)

                if raw_table_text:
                    chunks.append(
                        {
                            "doc_id": effective_doc_id,
                            "page_number": page_number,
                            "text": raw_table_text,
                            "chunk_type": "table",
                        }
                    )

            # 2. Extract text blocks outside table bounding boxes
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
            page_has_text_chunks = False

            for b in blocks:
                block_type = b[6] if len(b) > 6 else 0
                if block_type != 0:
                    # Skip non-text blocks (e.g., images)
                    continue

                block_rect = pymupdf.Rect(b[:4])
                block_text = b[4].strip()
                if not block_text:
                    continue

                # Check if this text block significantly overlaps with any detected table
                is_inside_table = False
                block_area = block_rect.get_area()
                if block_area > 0 and table_rects:
                    for t_rect in table_rects:
                        overlap = block_rect & t_rect
                        if not overlap.is_empty and (overlap.get_area() / block_area) > 0.4:
                            is_inside_table = True
                            break

                if not is_inside_table:
                    page_has_text_chunks = True
                    chunks.append(
                        {
                            "doc_id": effective_doc_id,
                            "page_number": page_number,
                            "text": block_text,
                            "chunk_type": "text",
                        }
                    )

            # If no blocks were extracted but plain page text exists (fallback for edge cases)
            if not page_has_text_chunks and not tables:
                full_text = page.get_text().strip()
                if full_text:
                    chunks.append(
                        {
                            "doc_id": effective_doc_id,
                            "page_number": page_number,
                            "text": full_text,
                            "chunk_type": "text",
                        }
                    )
    finally:
        doc.close()

    return chunks


def main() -> None:
    # Ensure UTF-8 output encoding on Windows consoles
    if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Extract page-anchored text and table chunks from a PDF document."
    )
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to ingest.")
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Optional document ID (defaults to filename stem).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (default: 2).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional file path to write JSON output to.",
    )

    args = parser.parse_args()

    try:
        chunks = extract_chunks(args.pdf_path, doc_id=args.doc_id)
        json_output = json.dumps(chunks, ensure_ascii=False, indent=args.indent)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_output, encoding="utf-8")

        print(json_output)
    except Exception as exc:
        print(f"Error ingesting PDF: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
