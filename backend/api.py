"""
FastAPI Application Module (Phase 5)

Wraps the ingest, extract, store, and link pipeline into a unified REST API.
Endpoints:
- POST /upload: Synchronously runs ingest -> extract -> store -> link for uploaded PDF.
- GET /facts: Filterable list of facts (?source_doc_id, ?entity).
- GET /facts/{fact_id}: Single fact details.
- GET /relationships: Filterable relationships with embedded fact_a and fact_b (?type).
- GET /stats: Dashboard aggregate statistics.
"""

import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.as_of import backfill_facts_as_of, infer_document_as_of
from backend.db import get_connection, init_db
from backend.extract import extract_facts_from_chunk, should_process_chunk
from backend.ingest import extract_chunks
from backend.link import link_facts
from backend.schema import Fact, Relationship
from backend.store import upsert_facts

load_dotenv()

UPLOAD_DIR = Path("data/samples/uploaded")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="Fact Knowledge Layer API",
    description="REST API wrapping ingest, fact extraction, persistent storage, and cross-document fact linking.",
    version="1.0.0",
)

# CORS configuration: wildcard origin without allow_credentials (per spec requirement)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Standardized HTTP error response with clear messaging."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catches unhandled server exceptions and returns clean 500 JSON without stack traces."""
    logger.error(f"Internal server error handling {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"An unexpected server error occurred: {str(exc)}"},
    )


def get_db(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """Dependency / helper for acquiring initialized database connection."""
    conn = get_connection(db_path)
    init_db(conn=conn)
    return conn


@app.get("/stats")
def get_stats() -> Dict[str, Any]:
    """
    Returns dashboard statistics:
    - total_facts: Total count of grounded facts in storage
    - total_relationships: Total count of corroborated, contradicted, or reconciled relationships
    - relationships_by_type: Breakdown of relationships grouped by type
    - total_documents: Count of distinct source documents represented
    - documents: Per-document fact count breakdown
    """
    conn = get_db()
    cursor = conn.cursor()

    # Fact totals
    cursor.execute("SELECT COUNT(*) FROM facts;")
    total_facts = cursor.fetchone()[0]

    # Relationship totals & breakdown
    cursor.execute("SELECT COUNT(*) FROM relationships;")
    total_relationships = cursor.fetchone()[0]

    cursor.execute("SELECT type, COUNT(*) FROM relationships GROUP BY type;")
    relationships_by_type = {row["type"]: row[1] for row in cursor.fetchall()}
    # Ensure all standard types are present in dict even if count is 0
    for r_type in ["corroborates", "contradicts", "reconciled_by_context"]:
        relationships_by_type.setdefault(r_type, 0)

    # Document statistics
    cursor.execute("SELECT COUNT(DISTINCT source_doc_id) FROM facts;")
    total_documents = cursor.fetchone()[0]

    cursor.execute(
        "SELECT source_doc_id, COUNT(*) as fact_count FROM facts GROUP BY source_doc_id ORDER BY fact_count DESC;"
    )
    documents = [
        {"source_doc_id": row["source_doc_id"], "fact_count": row["fact_count"]}
        for row in cursor.fetchall()
    ]

    return {
        "total_facts": total_facts,
        "total_relationships": total_relationships,
        "relationships_by_type": relationships_by_type,
        "total_documents": total_documents,
        "documents": documents,
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Accepts a PDF upload, persists to data/samples/uploaded/<filename>, and synchronously executes:
    1. Ingestion: extract text and table chunks.
    2. Fact Extraction: extracts structured facts with per-chunk fault tolerance.
    3. Period Inference: backfills document-level as_of periods if missing.
    4. Storage: upserts facts and computes vector embeddings.
    5. Linking: links newly extracted facts against existing knowledge base.

    Returns:
    {
        "doc_id": "...",
        "facts_extracted": ...,
        "relationships_found": ...,
        "facts": [...],
        "relationships": [...],
        "errors": [...]
    }
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a PDF.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_pdf_path = UPLOAD_DIR / file.filename
    content = await file.read()
    with open(saved_pdf_path, "wb") as f:
        f.write(content)

    doc_id = Path(file.filename).stem

    # 1. Ingestion
    try:
        chunks = extract_chunks(saved_pdf_path, doc_id=doc_id)
    except Exception as e:
        logger.error(f"Failed to ingest PDF {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse and extract chunks from PDF: {str(e)}",
        )

    # 2. Fact Extraction (per-chunk error resilience)
    facts: List[Fact] = []
    errors: List[Dict[str, Any]] = []

    for chunk in chunks:
        if not should_process_chunk(chunk):
            continue
        try:
            extracted_facts = extract_facts_from_chunk(chunk)
            for ef in extracted_facts:
                facts.append(
                    Fact.from_extracted(
                        extracted=ef,
                        source_doc_id=doc_id,
                        page_number=chunk["page_number"],
                    )
                )
        except Exception as err:
            logger.warning(f"Error extracting facts from chunk {chunk.get('chunk_id')}: {err}")
            errors.append({
                "chunk_id": chunk.get("chunk_id"),
                "page_number": chunk.get("page_number"),
                "error": str(err),
            })

    # 3. Document-level as_of backfill
    doc_as_of = infer_document_as_of(doc_id=doc_id, chunks=chunks, pdf_path=saved_pdf_path)
    facts = backfill_facts_as_of(facts, doc_as_of=doc_as_of)

    # 4. Storage (upsert)
    conn = get_db()
    if facts:
        upsert_facts(facts, conn=conn)

    # 5. Cross-document Fact Linking against all stored documents
    try:
        new_relationships = link_facts(doc_id=doc_id, conn=conn, verbose=False)
    except Exception as err:
        logger.warning(f"Cross-document linking encountered an error for doc {doc_id}: {err}")
        errors.append({
            "stage": "linking",
            "error": str(err),
        })
        new_relationships = []

    return {
        "doc_id": doc_id,
        "facts_extracted": len(facts),
        "relationships_found": len(new_relationships),
        "facts": [f.model_dump() for f in facts],
        "relationships": [r.model_dump() for r in new_relationships],
        "errors": errors,
    }


def format_fact_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Serializes a facts table SQLite row into a clean dictionary."""
    extra_data = {}
    if row["extra"]:
        try:
            extra_data = json.loads(row["extra"])
        except Exception:
            extra_data = {}

    return {
        "id": row["id"],
        "entity": row["entity"],
        "attribute": row["attribute"],
        "value": row["value"],
        "unit": row["unit"],
        "normalized_value": row["normalized_value"],
        "normalized_unit": row["normalized_unit"],
        "as_of": row["as_of"],
        "scope": row["scope"],
        "source_doc_id": row["source_doc_id"],
        "page_number": row["page_number"],
        "evidence_text": row["evidence_text"],
        "confidence": row["confidence"],
        "extra": extra_data,
        "created_at": row["created_at"] if "created_at" in row.keys() else None,
    }


@app.get("/facts")
def get_facts(
    source_doc_id: Optional[str] = None,
    entity: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves facts from storage with optional filtering by source_doc_id and entity.
    """
    conn = get_db()
    query = """
    SELECT id, entity, attribute, value, unit, normalized_value, normalized_unit,
           as_of, scope, source_doc_id, page_number, evidence_text, confidence, extra, created_at
    FROM facts
    WHERE 1=1
    """
    params: List[Any] = []

    if source_doc_id:
        query += " AND source_doc_id = ?"
        params.append(source_doc_id.strip())

    if entity:
        query += " AND LOWER(entity) LIKE ?"
        params.append(f"%{entity.strip().lower()}%")

    query += " ORDER BY source_doc_id, page_number, entity, attribute;"

    cursor = conn.cursor()
    cursor.execute(query, params)
    return [format_fact_row(row) for row in cursor.fetchall()]


@app.get("/facts/{fact_id}")
def get_fact_by_id(fact_id: str) -> Dict[str, Any]:
    """
    Retrieves full details for a single fact by its ID. Returns 404 if not found.
    """
    conn = get_db()
    query = """
    SELECT id, entity, attribute, value, unit, normalized_value, normalized_unit,
           as_of, scope, source_doc_id, page_number, evidence_text, confidence, extra, created_at
    FROM facts
    WHERE id = ?;
    """
    cursor = conn.cursor()
    cursor.execute(query, (fact_id.strip(),))
    row = cursor.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fact with id '{fact_id}' not found.",
        )

    return format_fact_row(row)


@app.get("/relationships")
def get_relationships(
    type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves relationships between facts, embedding full fact_a and fact_b records
    directly in the response so callers do not need secondary lookups.
    Supports optional ?type= filter ('corroborates', 'contradicts', 'reconciled_by_context').
    """
    conn = get_db()
    query = """
    SELECT 
        r.id AS rel_id,
        r.type,
        r.reasoning,
        r.confidence,
        r.created_at,
        fa.id AS fa_id,
        fa.entity AS fa_entity,
        fa.attribute AS fa_attribute,
        fa.value AS fa_value,
        fa.unit AS fa_unit,
        fa.normalized_value AS fa_normalized_value,
        fa.normalized_unit AS fa_normalized_unit,
        fa.as_of AS fa_as_of,
        fa.scope AS fa_scope,
        fa.source_doc_id AS fa_source_doc_id,
        fa.page_number AS fa_page_number,
        fa.evidence_text AS fa_evidence_text,
        fa.confidence AS fa_confidence,
        fa.extra AS fa_extra,
        fb.id AS fb_id,
        fb.entity AS fb_entity,
        fb.attribute AS fb_attribute,
        fb.value AS fb_value,
        fb.unit AS fb_unit,
        fb.normalized_value AS fb_normalized_value,
        fb.normalized_unit AS fb_normalized_unit,
        fb.as_of AS fb_as_of,
        fb.scope AS fb_scope,
        fb.source_doc_id AS fb_source_doc_id,
        fb.page_number AS fb_page_number,
        fb.evidence_text AS fb_evidence_text,
        fb.confidence AS fb_confidence,
        fb.extra AS fb_extra
    FROM relationships r
    JOIN facts fa ON r.fact_a_id = fa.id
    JOIN facts fb ON r.fact_b_id = fb.id
    WHERE 1=1
    """
    params: List[Any] = []

    if type:
        query += " AND r.type = ?"
        params.append(type.strip().lower())

    query += " ORDER BY r.created_at DESC;"

    cursor = conn.cursor()
    cursor.execute(query, params)

    relationships = []
    for row in cursor.fetchall():
        fa_extra = {}
        if row["fa_extra"]:
            try:
                fa_extra = json.loads(row["fa_extra"])
            except Exception:
                fa_extra = {}

        fb_extra = {}
        if row["fb_extra"]:
            try:
                fb_extra = json.loads(row["fb_extra"])
            except Exception:
                fb_extra = {}

        relationships.append({
            "id": row["rel_id"],
            "type": row["type"],
            "reasoning": row["reasoning"],
            "confidence": row["confidence"],
            "created_at": row["created_at"],
            "fact_a": {
                "id": row["fa_id"],
                "entity": row["fa_entity"],
                "attribute": row["fa_attribute"],
                "value": row["fa_value"],
                "unit": row["fa_unit"],
                "normalized_value": row["fa_normalized_value"],
                "normalized_unit": row["fa_normalized_unit"],
                "as_of": row["fa_as_of"],
                "scope": row["fa_scope"],
                "source_doc_id": row["fa_source_doc_id"],
                "page_number": row["fa_page_number"],
                "evidence_text": row["fa_evidence_text"],
                "confidence": row["fa_confidence"],
                "extra": fa_extra,
            },
            "fact_b": {
                "id": row["fb_id"],
                "entity": row["fb_entity"],
                "attribute": row["fb_attribute"],
                "value": row["fb_value"],
                "unit": row["fb_unit"],
                "normalized_value": row["fb_normalized_value"],
                "normalized_unit": row["fb_normalized_unit"],
                "as_of": row["fb_as_of"],
                "scope": row["fb_scope"],
                "source_doc_id": row["fb_source_doc_id"],
                "page_number": row["fb_page_number"],
                "evidence_text": row["fb_evidence_text"],
                "confidence": row["fb_confidence"],
                "extra": fb_extra,
            },
        })

    return relationships


# Mount static frontend for web workspace
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")



