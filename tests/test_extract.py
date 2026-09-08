"""
Unit tests for Fact Extraction (Phase 2).

Tests:
1. Fact schema validation, field defaults, and flexible extra dict.
2. Heuristic filtering logic (skipping pure headers/dates while retaining tables and prose).
3. Evidence grounding verification (detecting verbatim substrings vs fabricated evidence).
4. End-to-end extraction pipeline with mocked LLM structured output.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError

from backend.extract import (
    extract_document_facts,
    extract_facts_from_chunk,
    get_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    should_process_chunk,
    validate_evidence_grounding,
    validate_temporal_grounding,
)
from backend.ingest import Chunk
from backend.schema import ChunkFactsExtraction, ExtractedFact, Fact


def test_fact_schema_and_extra_flexibility():
    """Verify Fact and ExtractedFact schemas, validation rules, and open extra field."""
    # 1. Valid ExtractedFact
    ef = ExtractedFact(
        entity="Reserve Bank of India",
        attribute="policy repo rate",
        value="6.50",
        unit="%",
        normalized_value=6.50,
        normalized_unit="%",
        as_of="2024-25",
        scope="national",
        evidence_text="The policy repo rate was kept unchanged at 6.50 per cent.",
        confidence=0.98,
        extra={"monetary_policy_stance": "withdrawal of accommodation", "vote_split": "4-2"},
    )
    assert ef.entity == "Reserve Bank of India"
    assert ef.unit == "%"
    assert ef.normalized_value == 6.50
    assert ef.normalized_unit == "%"
    assert ef.extra["vote_split"] == "4-2"

    # 2. Convert to stored Fact
    fact = Fact.from_extracted(ef, source_doc_id="rbi-ar-24", page_number=5)
    assert fact.normalized_value == 6.50
    assert fact.normalized_unit == "%"
    assert fact.source_doc_id == "rbi-ar-24"
    assert fact.page_number == 5
    assert fact.id is not None and len(fact.id) > 0

    # 3. Flexible extra accepts arbitrary nested structures
    fact.extra["nested_data"] = {"revisions": [{"date": "Oct 2024", "val": 6.5}]}
    dumped = json.loads(fact.model_dump_json())
    assert dumped["extra"]["nested_data"]["revisions"][0]["val"] == 6.5

    # 4. Confidence must be between 0.0 and 1.0
    with pytest.raises(ValidationError):
        ExtractedFact(
            entity="Test",
            attribute="TestAttr",
            value="10",
            evidence_text="Test",
            confidence=1.5,  # invalid: > 1.0
        )


def test_heuristic_filter_retains_and_skips():
    """Verify heuristic filter skips low-information chunks and retains tables and prose."""
    # Retain all tables
    table_chunk: Chunk = {
        "doc_id": "test",
        "page_number": 1,
        "text": "| Metric | Q4 FY24 |\n|---|---|\n| Revenue | 2,000 |",
        "chunk_type": "table",
    }
    assert should_process_chunk(table_chunk) is True

    # Retain informative prose
    prose_chunk: Chunk = {
        "doc_id": "test",
        "page_number": 2,
        "text": "Delhivery recorded revenue from contracts with customers of INR 8,141.74 Cr in FY24.",
        "chunk_type": "text",
    }
    assert should_process_chunk(prose_chunk) is True

    # Skip standalone date
    date_chunk: Chunk = {
        "doc_id": "test",
        "page_number": 1,
        "text": "Date: May 17, 2024",
        "chunk_type": "text",
    }
    assert should_process_chunk(date_chunk) is False

    # Skip page numbers
    page_chunk: Chunk = {
        "doc_id": "test",
        "page_number": 3,
        "text": "Page 12 of 100",
        "chunk_type": "text",
    }
    assert should_process_chunk(page_chunk) is False

    # Skip short boilerplate headers
    header_chunk: Chunk = {
        "doc_id": "test",
        "page_number": 1,
        "text": "Disclaimer",
        "chunk_type": "text",
    }
    assert should_process_chunk(header_chunk) is False


def test_evidence_grounding_validation():
    """Verify evidence grounding checks exact substrings and catches invented quotes."""
    source_chunk_text = (
        "During FY24, the Indian economy expanded at a robust rate of 8.2 per cent, "
        "as per the provisional estimates released by the National Statistical Office (NSO)."
    )

    # Verbatim grounded quote
    quote_grounded = "Indian economy expanded at a robust rate of 8.2 per cent"
    is_g1, score1 = validate_evidence_grounding(quote_grounded, source_chunk_text)
    assert is_g1 is True
    assert score1 == 1.0

    # Slight punctuation / whitespace variation
    quote_multiline = "expanded at a robust rate\nof 8.2 per cent"
    is_g2, score2 = validate_evidence_grounding(quote_multiline, source_chunk_text)
    assert is_g2 is True

    # Hallucinated / invented evidence
    quote_invented = "India's inflation rate dropped to 2.1 per cent in the fourth quarter"
    is_g3, score3 = validate_evidence_grounding(quote_invented, source_chunk_text)
    assert is_g3 is False
    assert score3 < 0.5


def test_extraction_pipeline_with_mock_llm(tmp_path):
    """Verify end-to-end extraction pipeline with mocked LLM response and grounding tagging."""
    mock_facts = [
        ExtractedFact(
            entity="Indian Economy",
            attribute="real GDP growth rate",
            value="8.2",
            unit="%",
            as_of="FY24",
            scope="provisional",
            evidence_text="expanded at a robust rate of 8.2 per cent",
            confidence=0.95,
            extra={},
        ),
        ExtractedFact(
            entity="Indian Economy",
            attribute="hallucinated metric",
            value="999",
            unit="%",
            as_of="FY24",
            scope=None,
            evidence_text="this exact text does not exist anywhere in the chunk",
            confidence=0.9,
            extra={},
        ),
    ]

    sample_pdf = Path("data/samples/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf")
    if not sample_pdf.exists():
        pytest.skip("Sample PDF not found for pipeline test")

    with patch("backend.extract.extract_facts_from_chunk", return_value=mock_facts):
        facts = extract_document_facts(
            pdf_path=sample_pdf,
            doc_id="test-doc",
            max_chunks=2,
        )

        assert len(facts) >= 2
        fact1, fact2 = facts[0], facts[1]

        # First fact is properly grounded
        assert fact1.entity == "Indian Economy"
        assert fact1.attribute == "real GDP growth rate"
        assert fact1.source_doc_id == "test-doc"

        # Second fact is ungrounded -> grounding warning and reduced confidence
        assert fact2.attribute == "hallucinated metric"
        assert "grounding_warning" in fact2.extra
        assert fact2.confidence <= 0.5


def test_temporal_grounding_validation():
    """
    CRITICAL (Case 4): Verifies that validate_temporal_grounding eliminates
    hallucinated quarters/years when chunks lack column headers or period labels.
    """
    # Chunk 45: Headerless card box of sort centers
    chunk_headerless = "Automated sort centers\n21\n24\n30\n29"

    # LLM hallucinated Q1 FY24 -> MUST be caught and stripped to None
    is_g1, as_of1 = validate_temporal_grounding("Q1 FY24", chunk_headerless)
    assert is_g1 is False
    assert as_of1 is None

    # LLM hallucinated Q4 FY24 -> MUST be caught and stripped to None
    is_g2, as_of2 = validate_temporal_grounding("Q4 FY24", chunk_headerless)
    assert is_g2 is False
    assert as_of2 is None

    # Chunk 40: Legitimate table with explicit period header
    chunk_table = "As of end of / for the period Q4 FY22 | Q4 FY23 | Q3 FY24 | Q4 FY24 Pin-code reach | 18,793"

    # Legitimate Q4 FY24 -> grounded and preserved
    is_g3, as_of3 = validate_temporal_grounding("Q4 FY24", chunk_table)
    assert is_g3 is True
    assert as_of3 == "Q4 FY24"

    # Legitimate Q4 FY22 -> grounded and preserved
    is_g4, as_of4 = validate_temporal_grounding("Q4 FY22", chunk_table)
    assert is_g4 is True
    assert as_of4 == "Q4 FY22"

    # Prose with explicit filing date
    chunk_prospectus = "As of June 30, 2021, we covered 17,488 pin codes across India."
    is_g5, as_of5 = validate_temporal_grounding("June 30, 2021", chunk_prospectus)
    assert is_g5 is True
    assert as_of5 == "June 30, 2021"

    # None as_of is always valid (no claim made)
    is_g6, as_of6 = validate_temporal_grounding(None, chunk_headerless)
    assert is_g6 is True
    assert as_of6 is None


def test_checkpointing_and_resumption(tmp_path):
    """
    Verifies that batch extraction saves checkpoints atomically,
    and resumes without re-processing completed chunks.
    """
    sample_pdf = Path("data/samples/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf")
    if not sample_pdf.exists():
        pytest.skip("Sample PDF not found for checkpointing test")

    mock_fact = ExtractedFact(
        entity="Delhivery",
        attribute="Test Metric",
        value="100",
        unit="count",
        normalized_value=100.0,
        normalized_unit="count",
        as_of=None,
        scope=None,
        evidence_text="Pin-code reach",
        confidence=0.99,
        extra={},
    )

    cp_dir = tmp_path / "checkpoints"
    call_counts = {"calls": 0}

    def mock_extract(chunk, **kwargs):
        call_counts["calls"] += 1
        return [mock_fact]

    # Pass 1: Run with max_chunks=3 (processes 3 chunks, saves checkpoint)
    with patch("backend.extract.extract_facts_from_chunk", side_effect=mock_extract):
        facts_pass1 = extract_document_facts(
            pdf_path=sample_pdf,
            doc_id="test-resumable-doc",
            max_chunks=3,
            checkpoint_dir=cp_dir,
            resume=True,
            rpm=0,
            verbose=False,
        )

    assert len(facts_pass1) == 3
    assert call_counts["calls"] == 3

    # Confirm checkpoint file exists on disk
    cp_file = cp_dir / "test-resumable-doc.checkpoint.json"
    assert cp_file.exists()
    cp_data = load_checkpoint(cp_file)
    assert cp_data["doc_id"] == "test-resumable-doc"
    assert len(cp_data["completed_indices"]) == 3
    assert len(cp_data["facts"]) == 3

    # Pass 2: Resume with max_chunks=3 (all 3 chunks already completed in checkpoint!)
    call_counts["calls"] = 0
    with patch("backend.extract.extract_facts_from_chunk", side_effect=mock_extract):
        facts_pass2 = extract_document_facts(
            pdf_path=sample_pdf,
            doc_id="test-resumable-doc",
            max_chunks=3,
            checkpoint_dir=cp_dir,
            resume=True,
            rpm=0,
            verbose=False,
        )

    # All 3 chunks should be skipped (0 new LLM calls), facts restored from checkpoint!
    assert call_counts["calls"] == 0
    assert len(facts_pass2) == 3
    assert facts_pass2[0].attribute == "Test Metric"


def test_rate_limit_backoff_retry():
    """
    Verifies that extract_facts_from_chunk intercepts 429 rate limit exceptions,
    sleeps with exponential backoff, and succeeds when the API recovers.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "facts": [
            {
                "entity": "Delhivery",
                "attribute": "Active customers",
                "value": "30000",
                "unit": "count",
                "normalized_value": 30000.0,
                "normalized_unit": "count",
                "as_of": None,
                "scope": None,
                "evidence_text": "Delhivery active customers 30000",
                "confidence": 0.95,
                "extra": [],
            }
        ]
    })

    attempts = {"count": 0}

    def flaky_generate(**kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: Rate limit exceeded")
        return mock_response

    mock_client.models.generate_content.side_effect = flaky_generate

    chunk: Chunk = {
        "doc_id": "test",
        "page_number": 1,
        "text": "Delhivery active customers 30000",
        "chunk_type": "text",
    }

    with patch("time.sleep", return_value=None) as mock_sleep:
        facts = extract_facts_from_chunk(
            chunk=chunk,
            client=mock_client,
            provider="gemini",
            max_retries=5,
            initial_backoff=1.0,
        )

        assert attempts["count"] == 3
        assert mock_sleep.call_count == 2
        assert len(facts) == 1
        assert facts[0].entity == "Delhivery"
        assert facts[0].normalized_value == 30000.0

