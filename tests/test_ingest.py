"""
Unit tests for the PDF Ingestion Module.
Uses sample PDFs placed in data/samples/.
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest

from backend.ingest import extract_chunks

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def get_sample_pdf() -> Path:
    """Finds an available sample PDF for testing."""
    sample_candidates = [
        SAMPLES_DIR / "delhivery" / "03-delhivery-q4-fy24-earnings-presentation.pdf",
        SAMPLES_DIR / "delhivery" / "01-delhivery-prospectus-2022-excerpt.pdf",
        SAMPLES_DIR / "india-macroeconomy" / "02-rbi-annual-report-2024-25-excerpt.pdf",
    ]
    for candidate in sample_candidates:
        if candidate.exists():
            return candidate

    # Fallback to any PDF in data/samples
    found = list(SAMPLES_DIR.glob("**/*.pdf"))
    if found:
        return found[0]

    pytest.skip("No sample PDF found in data/samples/ for ingestion testing.")


def test_extract_chunks_schema_and_types():
    """Verify extracted chunks strictly adhere to {doc_id, page_number, text, chunk_type}."""
    pdf_path = get_sample_pdf()
    chunks = extract_chunks(pdf_path)

    assert len(chunks) > 0, "Ingestion should produce at least one chunk"

    required_keys = {"doc_id", "page_number", "text", "chunk_type"}
    for idx, chunk in enumerate(chunks[:50]):  # check representative sample
        assert isinstance(chunk, dict), f"Chunk {idx} is not a dict"
        assert set(chunk.keys()) == required_keys, f"Chunk {idx} keys mismatch: {chunk.keys()}"
        assert chunk["doc_id"] == pdf_path.stem, f"doc_id should default to filename stem"
        assert isinstance(chunk["page_number"], int), f"page_number must be int"
        assert chunk["page_number"] >= 1, f"page_number must be 1-indexed"
        assert chunk["chunk_type"] in {"text", "table"}, f"Unexpected chunk_type: {chunk['chunk_type']}"
        assert isinstance(chunk["text"], str), f"text must be string"
        assert len(chunk["text"].strip()) > 0, f"text must not be empty or whitespace only"


def test_extract_chunks_table_detection():
    """Verify that tables are detected and flagged as chunk_type: 'table' with raw text preserved."""
    pdf_path = SAMPLES_DIR / "delhivery" / "03-delhivery-q4-fy24-earnings-presentation.pdf"
    if not pdf_path.exists():
        pdf_path = get_sample_pdf()

    chunks = extract_chunks(pdf_path)

    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]

    assert len(text_chunks) > 0, "Expected prose text chunks to be extracted"
    assert len(table_chunks) > 0, "Expected table chunks to be extracted from financial presentation"

    # Verify table chunks preserve row structure
    sample_table = table_chunks[0]
    assert "|" in sample_table["text"] or "\n" in sample_table["text"], (
        "Table chunk should retain row or markdown separator structure"
    )


def test_custom_doc_id_and_page_anchoring():
    """Verify custom doc_id parameter overrides default and page numbers are properly anchored."""
    pdf_path = get_sample_pdf()
    custom_id = "custom_doc_identifier"
    chunks = extract_chunks(pdf_path, doc_id=custom_id)

    assert len(chunks) > 0
    assert all(c["doc_id"] == custom_id for c in chunks), "All chunks must reflect custom doc_id"

    # Verify page numbers are monotonic / valid range
    page_numbers = [c["page_number"] for c in chunks]
    assert min(page_numbers) == 1
    assert max(page_numbers) >= 1


def test_cli_ingest_output():
    """Verify running python backend/ingest.py <pdf_path> outputs valid JSON matching chunk schema."""
    pdf_path = get_sample_pdf()

    res = subprocess.run(
        [sys.executable, "backend/ingest.py", str(pdf_path), "--indent", "0"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert res.returncode == 0, f"CLI exited with code {res.returncode}. Stderr: {res.stderr}"
    data = json.loads(res.stdout)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "doc_id" in data[0]
    assert "page_number" in data[0]
    assert "text" in data[0]
    assert "chunk_type" in data[0]


def test_nonexistent_pdf_raises_error():
    """Verify proper error handling when given an invalid PDF path."""
    with pytest.raises(FileNotFoundError):
        extract_chunks("data/samples/nonexistent_file.pdf")
