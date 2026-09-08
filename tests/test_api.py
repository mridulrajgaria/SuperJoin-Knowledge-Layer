"""
Unit tests for FastAPI Pipeline API (Phase 5).

Tests:
1. GET /stats response structure, document count, and relationship breakdown.
2. GET /facts with ?source_doc_id and ?entity query filtering.
3. GET /facts/{fact_id} 200 on success and 404 with clean detail on missing ID.
4. GET /relationships with embedded fact_a and fact_b records, and ?type filter.
5. POST /upload success with mocked pipeline execution.
6. POST /upload per-chunk failure resilience (partial facts returned with errors array).
7. POST /upload validation failure on non-PDF upload (400 Bad Request).
"""

import io
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.db import init_db
from backend.schema import ExtractedFact, Fact, Relationship


@pytest.fixture
def client_with_test_db(tmp_path, monkeypatch):
    """Sets up an isolated SQLite database populated with deterministic test data."""
    test_db_path = tmp_path / "test_api_isolated.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{test_db_path}")

    conn = init_db(db_path=test_db_path)

    # Insert test facts
    fact_1_id = "11111111-1111-1111-1111-111111111111"
    fact_2_id = "22222222-2222-2222-2222-222222222222"
    fact_3_id = "33333333-3333-3333-3333-333333333333"

    with conn:
        conn.execute(
            """
            INSERT INTO facts (
                id, entity, attribute, value, unit, normalized_value, normalized_unit,
                as_of, scope, source_doc_id, page_number, evidence_text, confidence, extra
            ) VALUES 
            (?, 'Delhivery', 'Pin codes covered', '18,793', 'count', 18793.0, 'count', 'FY24', NULL, 'doc-ar-fy24', 2, '18,793 Pin codes covered', 0.95, '{"source": "ar"}'),
            (?, 'Delhivery', 'Pin codes covered', '17,488', 'count', 17488.0, 'count', 'Dec 2021', NULL, 'doc-prospectus-2022', 1, '17,488 Pin codes covered', 0.95, '{"source": "prospectus"}'),
            (?, 'India', 'GDP growth rate', '9.7', '%', 9.7, '%', '2021-22', NULL, 'doc-rbi-2024', 24, 'GDP growth rate 9.7%', 1.0, '{}');
            """,
            (fact_1_id, fact_2_id, fact_3_id),
        )

        # Insert test relationships
        conn.execute(
            """
            INSERT INTO relationships (id, fact_a_id, fact_b_id, type, reasoning, confidence)
            VALUES 
            ('rel-1', ?, ?, 'reconciled_by_context', 'Pin codes expanded from 17,488 in 2021 to 18,793 in FY24.', 0.98);
            """,
            (fact_2_id, fact_1_id),
        )

    conn.close()
    return TestClient(app)


def test_stats_endpoint(client_with_test_db):
    """Verifies /stats aggregate counts and dashboard breakdown."""
    response = client_with_test_db.get("/stats")
    assert response.status_code == 200
    data = response.json()

    assert data["total_facts"] == 3
    assert data["total_relationships"] == 1
    assert data["total_documents"] == 3
    assert data["relationships_by_type"]["reconciled_by_context"] == 1
    assert data["relationships_by_type"]["corroborates"] == 0
    assert data["relationships_by_type"]["contradicts"] == 0
    assert len(data["documents"]) == 3


def test_get_facts_and_filtering(client_with_test_db):
    """Verifies /facts listing and query parameter filtering."""
    # 1. Fetch all facts
    r_all = client_with_test_db.get("/facts")
    assert r_all.status_code == 200
    assert len(r_all.json()) == 3

    # 2. Filter by source_doc_id
    r_doc = client_with_test_db.get("/facts", params={"source_doc_id": "doc-rbi-2024"})
    assert r_doc.status_code == 200
    assert len(r_doc.json()) == 1
    assert r_doc.json()[0]["entity"] == "India"

    # 3. Filter by entity (case-insensitive)
    r_entity = client_with_test_db.get("/facts", params={"entity": "delhivery"})
    assert r_entity.status_code == 200
    assert len(r_entity.json()) == 2
    assert all(f["entity"] == "Delhivery" for f in r_entity.json())


def test_get_single_fact(client_with_test_db):
    """Verifies /facts/{fact_id} retrieves details or returns 404."""
    fact_id = "11111111-1111-1111-1111-111111111111"
    response = client_with_test_db.get(f"/facts/{fact_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == fact_id
    assert data["attribute"] == "Pin codes covered"
    assert data["extra"] == {"source": "ar"}

    # Non-existent ID -> 404
    r_missing = client_with_test_db.get("/facts/missing-uuid-99999")
    assert r_missing.status_code == 404
    assert "not found" in r_missing.json()["detail"]


def test_get_relationships_embedded_and_type_filter(client_with_test_db):
    """Verifies /relationships embeds full fact_a and fact_b and filters by type."""
    response = client_with_test_db.get("/relationships")
    assert response.status_code == 200
    rels = response.json()
    assert len(rels) == 1

    first = rels[0]
    assert first["id"] == "rel-1"
    assert first["type"] == "reconciled_by_context"
    assert "fact_a" in first and "fact_b" in first
    assert first["fact_a"]["attribute"] == "Pin codes covered"
    assert first["fact_b"]["attribute"] == "Pin codes covered"

    # Filter by type matching
    r_match = client_with_test_db.get("/relationships", params={"type": "reconciled_by_context"})
    assert r_match.status_code == 200
    assert len(r_match.json()) == 1

    # Filter by type non-matching
    r_nomatch = client_with_test_db.get("/relationships", params={"type": "contradicts"})
    assert r_nomatch.status_code == 200
    assert len(r_nomatch.json()) == 0


def test_upload_endpoint_success(client_with_test_db):
    """Verifies POST /upload orchestrates pipeline and returns expected schema."""
    dummy_chunks = [
        {"chunk_id": "c1", "page_number": 1, "chunk_type": "text", "text": "Delhivery active customers >33,200", "table_raw": None},
    ]
    extracted_ef = [
        ExtractedFact(
            entity="Delhivery",
            attribute="active customers",
            value=">33,200",
            evidence_text="active customers >33,200",
            confidence=0.95,
        )
    ]
    mock_relationship = Relationship(
        fact_a_id="f1",
        fact_b_id="f2",
        type="corroborates",
        reasoning="Both documents agree.",
        confidence=0.95,
    )

    with (
        patch("backend.api.extract_chunks", return_value=dummy_chunks),
        patch("backend.api.should_process_chunk", return_value=True),
        patch("backend.api.extract_facts_from_chunk", return_value=extracted_ef),
        patch("backend.api.upsert_facts", return_value=1),
        patch("backend.api.link_facts", return_value=[mock_relationship]),
    ):
        file_content = b"%PDF-1.4 dummy pdf binary content"
        response = client_with_test_db.post(
            "/upload",
            files={"file": ("test_doc.pdf", io.BytesIO(file_content), "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "test_doc"
        assert data["facts_extracted"] == 1
        assert data["relationships_found"] == 1
        assert len(data["facts"]) == 1
        assert data["facts"][0]["attribute"] == "active customers"
        assert len(data["relationships"]) == 1
        assert data["errors"] == []


def test_upload_endpoint_partial_error(client_with_test_db):
    """Verifies POST /upload does not crash when some chunks fail extraction."""
    dummy_chunks = [
        {"chunk_id": "c1", "page_number": 1, "chunk_type": "text", "text": "Valid chunk", "table_raw": None},
        {"chunk_id": "c2", "page_number": 2, "chunk_type": "text", "text": "Failing chunk", "table_raw": None},
    ]
    extracted_ef = [
        ExtractedFact(
            entity="Delhivery",
            attribute="fleets",
            value="15,000",
            evidence_text="fleets 15,000",
            confidence=0.90,
        )
    ]

    def mock_extract(chunk):
        if chunk["chunk_id"] == "c2":
            raise RuntimeError("API Rate limit or timeout on chunk c2")
        return extracted_ef

    with (
        patch("backend.api.extract_chunks", return_value=dummy_chunks),
        patch("backend.api.should_process_chunk", return_value=True),
        patch("backend.api.extract_facts_from_chunk", side_effect=mock_extract),
        patch("backend.api.upsert_facts", return_value=1),
        patch("backend.api.link_facts", return_value=[]),
    ):
        file_content = b"%PDF-1.4 dummy content"
        response = client_with_test_db.post(
            "/upload",
            files={"file": ("partial_fail.pdf", io.BytesIO(file_content), "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "partial_fail"
        assert data["facts_extracted"] == 1
        assert len(data["errors"]) == 1
        assert data["errors"][0]["chunk_id"] == "c2"
        assert "API Rate limit" in data["errors"][0]["error"]


def test_upload_endpoint_invalid_file(client_with_test_db):
    """Verifies POST /upload returns 400 Bad Request for non-PDF uploads."""
    response = client_with_test_db.post(
        "/upload",
        files={"file": ("sample.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert response.status_code == 400
    assert "must be a PDF" in response.json()["detail"]
