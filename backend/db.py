"""
Database Module for Fact Knowledge Layer (Phase 3)

Manages SQLite storage for facts and relationships:
- facts table: stores grounded facts, metadata, extra JSON, and vector embedding BLOBs.
  Enforces composite UNIQUE(source_doc_id, page_number, entity, attribute, evidence_text)
  to ensure re-running extraction generates updates, not duplicates.
- relationships table: schema-ready for Phase 4 cross-document corroboration/contradiction.
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional, Union

# Default SQLite database path
DEFAULT_DB_PATH = Path("facts.db")


def get_db_path(override_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolves database file path from argument, DATABASE_URL env, or default."""
    if override_path is not None:
        return Path(override_path)
    env_url = os.getenv("DATABASE_URL")
    if env_url and env_url.startswith("sqlite:///"):
        return Path(env_url.replace("sqlite:///", ""))
    return DEFAULT_DB_PATH


def get_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """
    Creates and configures an SQLite database connection with row factory
    and enabled foreign keys.
    """
    path = get_db_path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(
    conn: Optional[sqlite3.Connection] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> sqlite3.Connection:
    """
    Initializes database schema creating facts and relationships tables if they do not exist.
    """
    if conn is None:
        conn = get_connection(db_path)

    with conn:
        # Facts Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                entity TEXT NOT NULL,
                attribute TEXT NOT NULL,
                value TEXT NOT NULL,
                unit TEXT,
                normalized_value REAL,
                normalized_unit TEXT,
                as_of TEXT,
                scope TEXT,
                source_doc_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                evidence_text TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                extra TEXT NOT NULL DEFAULT '{}',
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_doc_id, page_number, entity, attribute, evidence_text)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_entity_attr ON facts(entity, attribute);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_source_doc ON facts(source_doc_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_as_of ON facts(as_of);")

        # Relationships Table (Phase 4 schema placeholder)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                fact_a_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
                fact_b_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
                type TEXT NOT NULL CHECK(type IN ('corroborates', 'contradicts', 'reconciled_by_context')),
                reasoning TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_fact_a ON relationships(fact_a_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_fact_b ON relationships(fact_b_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(type);")

    return conn
