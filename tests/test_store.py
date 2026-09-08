"""
Unit tests for Fact Storage and Embeddings (Phase 3).

Tests:
1. Schema creation and foreign key constraints on SQLite facts and relationships tables.
2. Upsert idempotency under re-extraction (generating fresh random UUIDs for the same facts).
3. Document-level as_of period inference and selective backfilling.
4. Vector embedding generation and nearest-neighbor cosine similarity search.
"""

import os
import sqlite3
import uuid
from pathlib import Path
from typing import List

import pytest

# Ensure offline mock embeddings are used during test suite
os.environ["TEST_MOCK_EMBEDDINGS"] = "1"

from backend.as_of import backfill_facts_as_of, extract_period_from_text, infer_document_as_of
from backend.db import get_connection, init_db
from backend.embeddings import (
    bytes_to_embedding,
    compute_cosine_similarity,
    embedding_to_bytes,
    fact_to_embedding_text,
    get_embedding,
    get_embeddings_batch,
)
from backend.schema import Fact, Relationship
from backend.store import (
    find_nearest_facts,
    get_stored_fact_count,
    load_facts_from_json,
    upsert_facts,
)


@pytest.fixture
def memory_db():
    """Provides a fresh in-memory SQLite connection for testing."""
    conn = init_db(db_path=":memory:")
    yield conn
    conn.close()


def test_db_schema_initialization(memory_db):
    """Verifies that facts and relationships tables and indexes are created properly."""
    cursor = memory_db.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    assert "facts" in tables
    assert "relationships" in tables

    # Check facts table column definitions
    cursor = memory_db.execute("PRAGMA table_info(facts);")
    columns = {row["name"]: row["type"] for row in cursor.fetchall()}
    assert "id" in columns
    assert "entity" in columns
    assert "attribute" in columns
    assert "normalized_value" in columns
    assert "normalized_unit" in columns
    assert "embedding" in columns
    assert "evidence_text" in columns
    assert "extra" in columns

    # Check foreign key enforcement on relationships table
    rel = Relationship(
        fact_a_id="non-existent-1",
        fact_b_id="non-existent-2",
        type="corroborates",
        reasoning="Test relationship",
    )
    with pytest.raises(sqlite3.IntegrityError):
        with memory_db:
            memory_db.execute(
                "INSERT INTO relationships (id, fact_a_id, fact_b_id, type, reasoning) VALUES (?, ?, ?, ?, ?)",
                (rel.id, rel.fact_a_id, rel.fact_b_id, rel.type, rel.reasoning),
            )


def test_upsert_idempotency_with_new_uuids(memory_db):
    """
    CRITICAL: Confirms that re-extracting the same document content (which generates
    brand-new random UUIDs for identical facts) updates records without increasing row count.
    """
    initial_facts = [
        Fact(
            id=str(uuid.uuid4()),
            entity="Delhivery",
            attribute="Pin-code reach",
            value="18,793",
            unit="count",
            normalized_value=18793.0,
            normalized_unit="count",
            as_of="FY24",
            scope="national",
            source_doc_id="delhivery-q4-deck",
            page_number=8,
            evidence_text="Pin-code reach(1) | 18,074 | 18,540 | 18,675 | 18,793",
            confidence=0.95,
            extra={"footnote": "1"},
        ),
        Fact(
            id=str(uuid.uuid4()),
            entity="Delhivery",
            attribute="Active customers",
            value="33,278",
            unit="count",
            normalized_value=33278.0,
            normalized_unit="count",
            as_of="FY24",
            scope=None,
            source_doc_id="delhivery-q4-deck",
            page_number=8,
            evidence_text="No. of Active Customers(3) | 23,613 | 27,253 | 30,598 | 33,278",
            confidence=0.95,
            extra={},
        ),
    ]

    # Pass 1: Initial load
    count1 = upsert_facts(initial_facts, conn=memory_db)
    assert count1 == 2
    assert get_stored_fact_count(conn=memory_db) == 2

    # Pass 2: Re-extraction produces identical facts with BRAND NEW random UUIDs
    reextracted_facts = [
        Fact(
            id=str(uuid.uuid4()),  # Fresh UUID!
            entity="Delhivery",
            attribute="Pin-code reach",
            value="18,793",
            unit="count",
            normalized_value=18793.0,
            normalized_unit="count",
            as_of="FY24",
            scope="national",
            source_doc_id="delhivery-q4-deck",
            page_number=8,
            evidence_text="Pin-code reach(1) | 18,074 | 18,540 | 18,675 | 18,793",
            confidence=0.98,  # Slightly updated confidence
            extra={"footnote": "1", "re_extracted": True},
        ),
        Fact(
            id=str(uuid.uuid4()),  # Fresh UUID!
            entity="Delhivery",
            attribute="Active customers",
            value="33,278",
            unit="count",
            normalized_value=33278.0,
            normalized_unit="count",
            as_of="FY24",
            scope=None,
            source_doc_id="delhivery-q4-deck",
            page_number=8,
            evidence_text="No. of Active Customers(3) | 23,613 | 27,253 | 30,598 | 33,278",
            confidence=0.98,
            extra={"re_extracted": True},
        ),
    ]

    count2 = upsert_facts(reextracted_facts, conn=memory_db)
    assert count2 == 2

    # Crucial assertion: Total row count in storage DOES NOT GROW!
    total_in_storage = get_stored_fact_count(conn=memory_db)
    assert total_in_storage == 2, f"Expected 2 facts in DB, found {total_in_storage} (duplicate insertion occurred!)"


def test_as_of_period_inference_and_backfill():
    """Verifies that clear document titles backfill missing as_of, while ambiguous text leaves null."""
    # 1. Unambiguous period extraction
    assert extract_period_from_text("Delhivery Limited Annual Report 2023-24") == "2023-24"
    assert extract_period_from_text("Q4 FY24 Earnings Presentation Delhivery") == "Q4 FY24"
    assert extract_period_from_text("Economic Survey 2024-25 Government of India") == "2024-25"
    assert extract_period_from_text("Prospectus Dated May 14, 2022") == "May 14, 2022"

    # 2. Ambiguous period extraction -> None (no guessing)
    assert extract_period_from_text("General Introduction to Indian Logistics Market") is None

    # 3. Selective backfilling
    facts = [
        Fact(
            entity="Delhivery",
            attribute="Registered Office",
            value="New Delhi",
            as_of=None,  # missing -> should backfill
            source_doc_id="delhivery-ar-fy24",
            page_number=1,
            evidence_text="Registered office in New Delhi",
        ),
        Fact(
            entity="Delhivery",
            attribute="Historical Growth",
            value="25%",
            as_of="FY22",  # specific -> must NOT overwrite
            source_doc_id="delhivery-ar-fy24",
            page_number=5,
            evidence_text="Growth of 25% in FY22",
        ),
    ]

    backfilled = backfill_facts_as_of(facts, doc_as_of="2023-24")
    assert backfilled[0].as_of == "2023-24"
    assert backfilled[0].extra.get("as_of_inferred") is True
    assert backfilled[1].as_of == "FY22"  # preserved

    # 4. Ambiguous multi-value series on same page (e.g. Sort centers: 21, 24, 30, 29 without headers)
    ambiguous_series = [
        Fact(
            entity="Delhivery",
            attribute="count of automated sort centers",
            value="21",
            as_of=None,
            source_doc_id="q4-deck",
            page_number=8,
            evidence_text="21",
        ),
        Fact(
            entity="Delhivery",
            attribute="count of automated sort centers",
            value="24",
            as_of=None,
            source_doc_id="q4-deck",
            page_number=8,
            evidence_text="24",
        ),
    ]
    backfilled_ambiguous = backfill_facts_as_of(ambiguous_series, doc_as_of="Q4 FY24")
    # Must NOT fabricate Q4 FY24 for both conflicting values
    assert backfilled_ambiguous[0].as_of is None
    assert backfilled_ambiguous[1].as_of is None
    assert "as_of_inference_skipped" in backfilled_ambiguous[0].extra



def test_nearest_neighbor_retrieval(memory_db):
    """
    Verifies that vector nearest-neighbor retrieval returns semantically relevant
    facts based on entity + attribute matching.
    """
    test_facts = [
        Fact(
            entity="Delhivery",
            attribute="Pin-code reach",
            value="18,793",
            source_doc_id="doc1",
            page_number=8,
            evidence_text="Pin-code reach 18,793",
        ),
        Fact(
            entity="Delhivery",
            attribute="Active customers",
            value="33,200",
            source_doc_id="doc1",
            page_number=8,
            evidence_text="Active customers 33,200",
        ),
        Fact(
            entity="India",
            attribute="retail inflation rate",
            value="5.4%",
            source_doc_id="doc2",
            page_number=76,
            evidence_text="retail inflation was 5.4%",
        ),
        Fact(
            entity="India",
            attribute="real GDP growth rate",
            value="8.2%",
            source_doc_id="doc2",
            page_number=1,
            evidence_text="GDP grew by 8.2%",
        ),
    ]

    upsert_facts(test_facts, conn=memory_db)

    # 1. Query for Delhivery pin codes
    results_pincode = find_nearest_facts(
        query="Delhivery: Pin codes covered",
        top_k=2,
        conn=memory_db,
    )
    assert len(results_pincode) == 2
    # Top result must be the Pin-code reach fact
    assert results_pincode[0]["attribute"] == "Pin-code reach"
    assert results_pincode[0]["similarity"] > results_pincode[1]["similarity"]

    # 2. Query for India inflation
    results_inflation = find_nearest_facts(
        query="India: CPI inflation",
        top_k=2,
        conn=memory_db,
    )
    assert results_inflation[0]["attribute"] == "retail inflation rate"
    assert results_inflation[0]["entity"] == "India"
