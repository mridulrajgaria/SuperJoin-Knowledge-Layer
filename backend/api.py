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
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.db import get_connection, init_db

load_dotenv()

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
