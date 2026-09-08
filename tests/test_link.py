"""
Unit tests for Cross-Document Fact Linking (Phase 4).

Tests:
1. Schema validity for Relationship and FactRelationshipJudgement.
2. Canonical pair key calculation and order-independent deduplication.
3. Unit mismatch multiplier heuristic detection (10x, 1,000x).
4. Cheap domain pre-filtering (skipping corporate vs macro cross-talk).
5. Mock-LLM classification confirming the reasoning field is populated with citations.
"""

import json
import sqlite3
import uuid
from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError

from backend.db import init_db
from backend.link import (
    classify_relationship,
    detect_unit_mismatch_multiplier,
    get_all_stored_relationships,
    get_existing_relationship_pairs,
    is_viable_candidate_pair,
    make_canonical_pair_key,
    store_relationship,
)
from backend.schema import Fact, FactRelationshipJudgement, Relationship


@pytest.fixture
def memory_db():
    """Provides a fresh in-memory SQLite connection with initialized schema."""
    conn = init_db(db_path=":memory:")
    yield conn
    conn.close()


def test_relationship_schema_validity():
    """Verifies that Relationship model enforces valid types and default values."""
    rel = Relationship(
        fact_a_id="fact-1",
        fact_b_id="fact-2",
        type="corroborates",
        reasoning="Both documents state Delhivery covered 18,793 pin codes as of FY24.",
        confidence=0.98,
    )
    assert rel.type == "corroborates"
    assert rel.confidence == 0.98
    assert "18,793" in rel.reasoning

    # Invalid type
    with pytest.raises(ValidationError):
        FactRelationshipJudgement(
            type="invalid_category",  # must be corroborates, contradicts, reconciled_by_context, or unrelated
            reasoning="test",
            confidence=1.0,
        )


def test_canonical_pair_key_and_order_independence():
    """Confirms that (A, B) and (B, A) resolve to the exact same canonical pair key."""
    id_1 = "aaaa-1111"
    id_2 = "bbbb-2222"

    key_forward = make_canonical_pair_key(id_1, id_2)
    key_backward = make_canonical_pair_key(id_2, id_1)

    assert key_forward == (id_1, id_2)
    assert key_backward == (id_1, id_2)
    assert key_forward == key_backward


def test_store_relationship_dedup_and_unrelated_filtering(memory_db):
    """
    CRITICAL: Confirms order-independent deduplication in SQLite.
    Storing (A, B) then (B, A) must update the row rather than creating a duplicate.
    Also confirms 'unrelated' judgements are discarded from storage.
    """
    # Insert prerequisite facts into facts table
    id_a = "11111111-1111-1111-1111-111111111111"
    id_b = "22222222-2222-2222-2222-222222222222"

    with memory_db:
        memory_db.execute(
            "INSERT INTO facts (id, entity, attribute, value, source_doc_id, page_number, evidence_text) "
            "VALUES (?, 'Delhivery', 'AttrA', '100', 'doc-1', 1, 'quote 1'), "
            "(?, 'Delhivery', 'AttrB', '100', 'doc-2', 2, 'quote 2');",
            (id_a, id_b),
        )

    # Pass 1: Insert forward pair (A, B)
    rel1 = Relationship(
        fact_a_id=id_a,
        fact_b_id=id_b,
        type="corroborates",
        reasoning="Initial corroboration statement.",
        confidence=0.90,
    )
    assert store_relationship(rel1, conn=memory_db) is True

    existing = get_existing_relationship_pairs(conn=memory_db)
    assert len(existing) == 1
    assert (id_a, id_b) in existing

    # Pass 2: Insert reversed pair (B, A) with updated reasoning
    rel2 = Relationship(
        fact_a_id=id_b,
        fact_b_id=id_a,
        type="corroborates",
        reasoning="Updated corroboration reasoning.",
        confidence=0.99,
    )
    assert store_relationship(rel2, conn=memory_db) is True

    # Row count must NOT increase
    cursor = memory_db.execute("SELECT count(*) FROM relationships;")
    assert cursor.fetchone()[0] == 1

    # Stored reasoning should be updated
    stored = get_all_stored_relationships(conn=memory_db)
    assert len(stored) == 1
    assert stored[0]["reasoning"] == "Updated corroboration reasoning."
    assert stored[0]["confidence"] == 0.99

    # Pass 3: Unrelated pairs must be discarded from storage
    rel_unrelated = Relationship(
        fact_a_id=id_a,
        fact_b_id=id_b,
        type="unrelated",
        reasoning="Different concepts entirely.",
    )
    assert store_relationship(rel_unrelated, conn=memory_db) is False

    # Row count remains 1
    cursor = memory_db.execute("SELECT count(*) FROM relationships;")
    assert cursor.fetchone()[0] == 1


def test_unit_mismatch_multiplier_detection():
    """Confirms detect_unit_mismatch_multiplier catches 10x and 1,000x unit differences."""
    fact_crore = Fact(
        entity="Delhivery",
        attribute="Revenue",
        value="8,141.74 Cr",
        normalized_value=8141.74,
        source_doc_id="doc1",
        page_number=1,
        evidence_text="8,141.74 Cr",
    )
    fact_million = Fact(
        entity="Delhivery",
        attribute="Revenue",
        value="814.17 Mn",
        normalized_value=814.174,
        source_doc_id="doc2",
        page_number=1,
        evidence_text="814.17 Mn",
    )

    # 10x crore vs million ratio
    hint_10x = detect_unit_mismatch_multiplier(fact_crore, fact_million)
    assert hint_10x is not None
    assert "10x" in hint_10x
    assert "Crore vs. Million" in hint_10x

    # 1,000x thousand vs million ratio
    fact_k = Fact(
        entity="Delhivery",
        attribute="Fleet",
        value="15.065",
        normalized_value=15.065,
        source_doc_id="doc1",
        page_number=1,
        evidence_text="15.065",
    )
    fact_units = Fact(
        entity="Delhivery",
        attribute="Fleet",
        value="15,065",
        normalized_value=15065.0,
        source_doc_id="doc2",
        page_number=1,
        evidence_text="15,065",
    )
    hint_1000x = detect_unit_mismatch_multiplier(fact_k, fact_units)
    assert hint_1000x is not None
    assert "1,000x" in hint_1000x

    # Non-round ratio (e.g. 17,488 vs 18,793 = 1.07x) -> None
    fact_pincode_1 = Fact(
        entity="Delhivery",
        attribute="Pins",
        value="17,488",
        normalized_value=17488.0,
        source_doc_id="doc1",
        page_number=1,
        evidence_text="17,488",
    )
    fact_pincode_2 = Fact(
        entity="Delhivery",
        attribute="Pins",
        value="18,793",
        normalized_value=18793.0,
        source_doc_id="doc2",
        page_number=1,
        evidence_text="18,793",
    )
    assert detect_unit_mismatch_multiplier(fact_pincode_1, fact_pincode_2) is None


def test_domain_pre_filtering():
    """Verifies that is_viable_candidate_pair eliminates cross-domain noise."""
    f_delhivery = Fact(
        entity="Delhivery",
        attribute="Revenue",
        value="8000",
        source_doc_id="doc-delhivery-1",
        page_number=1,
        evidence_text="Rev 8000",
    )
    f_delhivery_2 = Fact(
        entity="Delhivery Limited",
        attribute="Revenue",
        value="8000",
        source_doc_id="doc-delhivery-2",
        page_number=1,
        evidence_text="Rev 8000",
    )
    f_india = Fact(
        entity="India",
        attribute="Retail Inflation",
        value="5.4%",
        source_doc_id="doc-macro-1",
        page_number=1,
        evidence_text="Inf 5.4%",
    )
    f_rbi = Fact(
        entity="Reserve Bank of India",
        attribute="Repo Rate",
        value="6.5%",
        source_doc_id="doc-macro-2",
        page_number=1,
        evidence_text="Repo 6.5%",
    )

    # 1. Corporate vs Corporate across docs -> Viable
    assert is_viable_candidate_pair(f_delhivery, f_delhivery_2) is True

    # 2. Macro vs Macro across docs -> Viable
    assert is_viable_candidate_pair(f_india, f_rbi) is True

    # 3. Corporate vs Macro -> DISCARD (noise pair)
    assert is_viable_candidate_pair(f_delhivery, f_india) is False
    assert is_viable_candidate_pair(f_delhivery, f_rbi) is False

    # 4. Same document -> DISCARD
    f_same_doc = Fact(
        entity="Delhivery",
        attribute="Pins",
        value="18000",
        source_doc_id="doc-delhivery-1",
        page_number=2,
        evidence_text="Pins 18000",
    )
    assert is_viable_candidate_pair(f_delhivery, f_same_doc) is False


def test_mock_llm_relationship_reasoning_validation():
    """
    Confirms that classify_relationship parses structured LLM output and validates
    that the reasoning field contains specific metric and temporal citations.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "type": "reconciled_by_context",
        "reasoning": (
            "Fact A reports 17,488 PIN codes serviced as of December 31, 2021 in the prospectus, "
            "whereas Fact B reports 18,793 pin codes covered as of FY24 in the earnings presentation. "
            "The 1,305 pin code discrepancy is reconciled by temporal expansion between 2021 and 2024."
        ),
        "confidence": 0.98,
    })
    mock_client.models.generate_content.return_value = mock_response

    f1 = Fact(
        entity="Delhivery",
        attribute="PIN codes serviced",
        value="17,488",
        as_of="December 31, 2021",
        source_doc_id="prospectus",
        page_number=50,
        evidence_text="covered 17,488 pin codes",
    )
    f2 = Fact(
        entity="Delhivery",
        attribute="pin codes covered",
        value="18,793",
        as_of="FY24",
        source_doc_id="q4-deck",
        page_number=8,
        evidence_text="Pin-code reach 18,793",
    )

    judgement = classify_relationship(
        fact_a=f1,
        fact_b=f2,
        client=mock_client,
        provider="gemini",
    )

    assert judgement.type == "reconciled_by_context"
    assert judgement.confidence == 0.98
    # Assert reasoning is specific and grounded
    assert "17,488" in judgement.reasoning
    assert "18,793" in judgement.reasoning
    assert "2021" in judgement.reasoning
    assert "2024" in judgement.reasoning
    assert len(judgement.reasoning) > 50
