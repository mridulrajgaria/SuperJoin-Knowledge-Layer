"""
Fact Storage and Retrieval Module (Phase 3)

Provides persistent SQLite storage, embedding indexing, idempotent upserts,
document-level as_of backfill, and vector nearest-neighbor search.

CLI:
    python backend/store.py <extracted_json_path>
    python backend/store.py --search "Delhivery: Pin codes covered"
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from backend.as_of import backfill_facts_as_of, infer_document_as_of
from backend.db import get_connection, init_db
from backend.embeddings import (
    bytes_to_embedding,
    compute_cosine_similarity,
    embedding_to_bytes,
    fact_to_embedding_text,
    get_embedding,
    get_embeddings_batch,
)
from backend.schema import Fact


def upsert_facts(
    facts: List[Fact],
    conn: Optional[sqlite3.Connection] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """
    Idempotently upserts a list of Fact models into SQLite.
    Computes vector embeddings for each fact from `entity + attribute`.
    Handles conflict on composite unique key:
    (source_doc_id, page_number, entity, attribute, evidence_text).
    """
    if not facts:
        return 0

    if conn is None:
        conn = init_db(db_path=db_path)

    # 1. Compute embeddings in batch for entity + attribute
    texts = [fact_to_embedding_text(f) for f in facts]
    embeddings = get_embeddings_batch(texts)

    upsert_sql = """
    INSERT INTO facts (
        id, entity, attribute, value, unit, normalized_value, normalized_unit,
        as_of, scope, source_doc_id, page_number, evidence_text, confidence, extra, embedding
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(source_doc_id, page_number, entity, attribute, evidence_text)
    DO UPDATE SET
        value = excluded.value,
        unit = excluded.unit,
        normalized_value = excluded.normalized_value,
        normalized_unit = excluded.normalized_unit,
        as_of = COALESCE(excluded.as_of, facts.as_of),
        scope = excluded.scope,
        confidence = excluded.confidence,
        extra = excluded.extra,
        embedding = excluded.embedding;
    """

    records = []
    for fact, emb in zip(facts, embeddings):
        emb_bytes = embedding_to_bytes(emb)
        extra_json = json.dumps(fact.extra, ensure_ascii=False)
        records.append(
            (
                fact.id,
                fact.entity,
                fact.attribute,
                str(fact.value),
                fact.unit,
                fact.normalized_value,
                fact.normalized_unit,
                fact.as_of,
                fact.scope,
                fact.source_doc_id,
                fact.page_number,
                fact.evidence_text,
                fact.confidence,
                extra_json,
                emb_bytes,
            )
        )

    with conn:
        conn.executemany(upsert_sql, records)

    return len(records)


def load_facts_from_json(
    json_path: Union[str, Path],
    conn: Optional[sqlite3.Connection] = None,
    db_path: Optional[Union[str, Path]] = None,
    backfill_as_of: bool = True,
) -> int:
    """
    Loads facts from an extracted JSON file, performs document-level as_of
    backfill if enabled, computes embeddings, and idempotently upserts to storage.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Extracted JSON file not found: {path}")

    content = path.read_text(encoding="utf-8")
    raw_data = json.loads(content)

    facts: List[Fact] = [Fact.model_validate(item) for item in raw_data]
    if not facts:
        return 0

    doc_id = facts[0].source_doc_id if facts else path.stem

    # Attempt document-level as_of inference & backfill
    if backfill_as_of:
        doc_as_of = infer_document_as_of(doc_id=doc_id)
        facts = backfill_facts_as_of(facts, doc_as_of)

    return upsert_facts(facts, conn=conn, db_path=db_path)


def find_nearest_facts(
    query: Union[str, Fact, Dict[str, Any]],
    top_k: int = 5,
    exclude_doc_id: Optional[str] = None,
    exclude_fact_id: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Performs nearest-neighbor search across stored facts based on cosine similarity.
    Query is embedded via `entity + attribute` (or raw query string).
    """
    if conn is None:
        conn = get_connection(db_path=db_path)

    # 1. Determine query vector
    if isinstance(query, str):
        query_text = query
    else:
        query_text = fact_to_embedding_text(query)

    query_vec = np.asarray(get_embedding(query_text), dtype=np.float32)

    # 2. Fetch stored facts with embeddings
    sql = "SELECT * FROM facts WHERE embedding IS NOT NULL"
    params: List[Any] = []

    if exclude_doc_id:
        sql += " AND source_doc_id != ?"
        params.append(exclude_doc_id)

    if exclude_fact_id:
        sql += " AND id != ?"
        params.append(exclude_fact_id)

    cursor = conn.execute(sql, params)
    rows = cursor.fetchall()
    if not rows:
        return []

    # 3. Compute cosine similarity
    scored: List[Tuple[float, sqlite3.Row]] = []
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0.0:
        return []

    for row in rows:
        emb_blob = row["embedding"]
        target_vec = bytes_to_embedding(emb_blob)
        t_norm = np.linalg.norm(target_vec)
        if t_norm > 0.0:
            sim = float(np.dot(query_vec, target_vec) / (q_norm * t_norm))
            scored.append((sim, row))

    # Sort descending by similarity
    scored.sort(key=lambda x: x[0], reverse=True)

    results: List[Dict[str, Any]] = []
    for sim, r in scored[:top_k]:
        extra_data = {}
        try:
            extra_data = json.loads(r["extra"])
        except Exception:
            pass

        results.append(
            {
                "id": r["id"],
                "similarity": round(sim, 4),
                "entity": r["entity"],
                "attribute": r["attribute"],
                "value": r["value"],
                "unit": r["unit"],
                "normalized_value": r["normalized_value"],
                "normalized_unit": r["normalized_unit"],
                "as_of": r["as_of"],
                "scope": r["scope"],
                "source_doc_id": r["source_doc_id"],
                "page_number": r["page_number"],
                "evidence_text": r["evidence_text"],
                "confidence": r["confidence"],
                "extra": extra_data,
            }
        )

    return results


def get_stored_fact_count(conn: Optional[sqlite3.Connection] = None, db_path: Optional[Union[str, Path]] = None) -> int:
    """Returns the total number of facts stored in the database."""
    if conn is None:
        conn = get_connection(db_path=db_path)
    cursor = conn.execute("SELECT COUNT(*) FROM facts;")
    return cursor.fetchone()[0]


def main() -> None:
    if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Fact Knowledge Layer Storage CLI (Phase 3). Load extracted facts and search nearest neighbors."
    )
    parser.add_argument(
        "json_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to an extracted facts JSON file in data/extracted/ to upsert into storage.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Optional custom SQLite database path (defaults to facts.db or DATABASE_URL).",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search query text (e.g., 'Delhivery: Pin codes covered') to retrieve nearest neighbors.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest neighbors to return (default: 5).",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Display total count of facts in storage.",
    )
    parser.add_argument(
        "--no-backfill",
        action="store_true",
        help="Disable document-level as_of backfilling.",
    )

    args = parser.parse_args()

    # Ensure DB schema is initialized
    init_db(db_path=args.db)

    if args.count:
        conn = get_connection(db_path=args.db)
        cur = conn.cursor()
        cur.execute(
            "SELECT source_doc_id, page_number, count(*) FROM facts "
            "GROUP BY source_doc_id, page_number ORDER BY source_doc_id, page_number"
        )
        rows = cur.fetchall()
        total = sum(r[2] for r in rows)
        print(f"Total stored facts: {total}\n")
        print("Grouped by source_doc_id, page_number:")
        for doc_id, page_num, count in rows:
            print(f"  {doc_id} | page {page_num}: {count} facts")
        return

    if args.search:
        results = find_nearest_facts(query=args.search, top_k=args.top_k, db_path=args.db)
        print(f"\n--- Nearest Neighbors for query: '{args.search}' (top {args.top_k}) ---")
        if not results:
            print("No matching facts found.")
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if args.json_path:
        count = load_facts_from_json(
            json_path=args.json_path,
            db_path=args.db,
            backfill_as_of=not args.no_backfill,
        )
        total = get_stored_fact_count(db_path=args.db)
        print(f"Successfully upserted {count} facts from {args.json_path}.")
        print(f"Total facts now in database: {total}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
