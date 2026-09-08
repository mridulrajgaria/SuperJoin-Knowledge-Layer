"""
Cross-Document Fact Linking Module (Phase 4)

Retrieves candidate fact pairs across documents using vector similarity,
applies domain and similarity pre-filters, and prepares candidates for LLM classification.
"""

import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from dotenv import load_dotenv

from backend.db import get_connection, init_db
from backend.embeddings import bytes_to_embedding, compute_cosine_similarity
from backend.schema import Fact

load_dotenv()

# Defined entity domain boundaries for cheap pre-filtering
CORPORATE_ENTITIES: Set[str] = {
    "delhivery",
    "delhivery limited",
    "ca swift investments",
    "fedex",
    "aramex",
    "spoton",
    "india post",
}

MACRO_ENTITIES: Set[str] = {
    "india",
    "reserve bank of india",
    "rbi",
    "central government of india",
    "central government",
    "state governments",
    "imf",
    "international monetary fund",
    "global economy",
    "global inflation",
    "food items",
}


def is_viable_candidate_pair(fact_a: Fact, fact_b: Fact) -> bool:
    """
    Cheap rule-based pre-filter to eliminate noise pairs before LLM classification:
    - Rejects pairs from the same source document.
    - Rejects pairs across mutually exclusive domains (e.g. corporate logistics vs sovereign macroeconomy).
    """
    if fact_a.source_doc_id == fact_b.source_doc_id:
        return False

    if fact_a.id == fact_b.id:
        return False

    entity_a = fact_a.entity.strip().lower()
    entity_b = fact_b.entity.strip().lower()

    # If one entity is corporate and the other is macroeconomic, skip immediately
    a_is_corp = any(c in entity_a for c in CORPORATE_ENTITIES)
    b_is_corp = any(c in entity_b for c in CORPORATE_ENTITIES)
    a_is_macro = any(m in entity_a for m in MACRO_ENTITIES)
    b_is_macro = any(m in entity_b for m in MACRO_ENTITIES)

    if (a_is_corp and b_is_macro) or (a_is_macro and b_is_corp):
        return False

    return True


def row_to_fact(row: sqlite3.Row) -> Fact:
    """Converts an SQLite row from the facts table into a Fact model."""
    import json

    extra_data = {}
    if row["extra"]:
        try:
            extra_data = json.loads(row["extra"])
        except Exception:
            extra_data = {}

    return Fact(
        id=row["id"],
        entity=row["entity"],
        attribute=row["attribute"],
        value=row["value"],
        unit=row["unit"],
        normalized_value=row["normalized_value"],
        normalized_unit=row["normalized_unit"],
        as_of=row["as_of"],
        scope=row["scope"],
        source_doc_id=row["source_doc_id"],
        page_number=row["page_number"],
        evidence_text=row["evidence_text"],
        confidence=row["confidence"],
        extra=extra_data,
    )


def get_candidate_pairs_for_fact(
    target_fact: Fact,
    target_embedding: np.ndarray,
    all_facts_with_embeddings: List[Tuple[Fact, np.ndarray]],
    top_k: int = 5,
    min_sim: float = 0.70,
) -> List[Tuple[float, Fact]]:
    """
    Retrieves top-k cross-document candidate facts for a target fact based on cosine similarity.
    Applies the domain pre-filter and similarity floor.

    Returns:
        List of (similarity_score, candidate_fact) sorted descending by similarity.
    """
    candidates = []

    for candidate_fact, candidate_emb in all_facts_with_embeddings:
        # Must be from a different document
        if candidate_fact.source_doc_id == target_fact.source_doc_id:
            continue

        # Check cheap domain pre-filter
        if not is_viable_candidate_pair(target_fact, candidate_fact):
            continue

        sim = compute_cosine_similarity(target_embedding, candidate_emb)
        if sim >= min_sim:
            candidates.append((sim, candidate_fact))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:top_k]
