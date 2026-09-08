"""
Cross-Document Fact Linking Module (Phase 4)

Retrieves candidate fact pairs across documents using vector similarity,
applies domain and similarity pre-filters, and prepares candidates for LLM classification.
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from dotenv import load_dotenv

from backend.db import get_connection, init_db
from backend.embeddings import bytes_to_embedding, compute_cosine_similarity
from backend.extract import get_llm_client
from backend.schema import Fact, FactRelationshipJudgement, Relationship

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


def detect_unit_mismatch_multiplier(fact_a: Fact, fact_b: Fact) -> Optional[str]:
    """
    Detects if two numeric facts differ by a round power-of-10 multiplier
    consistent with unit scale mismatches (e.g. crore vs million, thousand vs million).

    Returns:
        A formatted hint string for the LLM classifier if a multiplier is detected, or None.
    """
    v_a = fact_a.normalized_value
    v_b = fact_b.normalized_value

    if v_a is None or v_b is None:
        return None

    try:
        val_a = float(v_a)
        val_b = float(v_b)
    except (ValueError, TypeError):
        return None

    if val_a <= 0 or val_b <= 0:
        return None

    ratio = max(val_a, val_b) / min(val_a, val_b)

    # Multipliers with ±2% tolerance
    MULTIPLIERS = [
        (10.0, "10x", "Crore vs. Million unit scale (1 crore = 10 million)"),
        (100.0, "100x", "Percentage vs. Basis Points or Lakh vs. Thousand scale"),
        (1000.0, "1,000x", "Thousands vs. Millions or Units vs. Thousands scale"),
        (10000.0, "10,000x", "Crore vs. Thousand scale"),
        (100000.0, "100,000x", "Lakh vs. Single units scale"),
        (10000000.0, "10,000,000x (10^7)", "Crore vs. Single units scale"),
    ]

    for target_mult, label, explanation in MULTIPLIERS:
        if abs(ratio - target_mult) / target_mult <= 0.025:
            return (
                f"UNIT SCALE MISMATCH DETECTED: The normalized values ({val_a} vs {val_b}) differ by a ~{label} "
                f"multiplier (ratio {ratio:.2f}), which is consistent with a {explanation}. "
                f"Consider whether this discrepancy is 'reconciled_by_context' via unit scale differences."
            )

    return None


RELATIONSHIP_SYSTEM_PROMPT = """You are an expert factual reasoning and knowledge graph system.
Your task is to analyze two facts extracted from different source documents and classify their semantic relationship.

Relationship Categories:
1. 'corroborates': Both facts state the same or consistent claim/metric about the entity for the same time period and scope (e.g. both confirm 18,793 pin codes covered in FY24, or both report 6.5% repo rate in 2024-25). Minor variations in phrasing or precision are allowed.
2. 'contradicts': Both facts report genuinely conflicting figures, statuses, or assertions for the SAME entity, attribute, and time period/scope that CANNOT be reconciled by time, scope, or measurement units.
3. 'reconciled_by_context': The facts report different numbers or assertions, but the difference is logically explained by contextual divergence:
   - Temporal difference: Different 'as_of' periods or dates (e.g., 17,488 pin codes in Dec 2021 vs 18,793 in March 2024; network expanded over time).
   - Scope difference: Consolidated vs standalone, urban vs rural, provisional vs revised.
   - Unit/Scale difference: Values reported in different denominations (crore vs million, thousands vs units).
4. 'unrelated': The two facts are superficially similar in vocabulary or domain, but describe DIFFERENT attributes, metrics, or events (e.g. gateway count vs sort center count, or revenue vs inflation). Do NOT force unrelated facts into corroborates/contradicts/reconciled!

CRITICAL REASONING RULE:
The 'reasoning' field MUST explicitly cite the specific values, effective periods ('as_of'), scopes, and units of both facts. Avoid generic, hand-waving statements like "these facts are different" or "these facts agree". State the exact numbers and explain why they corroborate, contradict, or reconcile.
"""


def classify_relationship(
    fact_a: Fact,
    fact_b: Fact,
    client: Optional[Any] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 5,
    initial_backoff: float = 4.0,
) -> FactRelationshipJudgement:
    """
    Calls LLM with structured output to judge the relationship between two cross-document facts.
    Includes unit-mismatch hints and rate-limit backoff retry.
    """
    if client is None or provider is None:
        provider, client = get_llm_client()

    if provider == "none" or client is None:
        raise RuntimeError(
            "No LLM API key configured. Please set GEMINI_API_KEY or OPENAI_API_KEY in your environment or .env file."
        )

    # Pre-check for unit scale mismatches
    unit_hint = detect_unit_mismatch_multiplier(fact_a, fact_b)
    hint_section = f"\nPre-computed Analysis:\n{unit_hint}\n" if unit_hint else ""

    pair_prompt = (
        f"FACT A:\n"
        f"- Entity: {fact_a.entity}\n"
        f"- Attribute: {fact_a.attribute}\n"
        f"- Value: {fact_a.value}\n"
        f"- Unit: {fact_a.unit or 'none'}\n"
        f"- Normalized Value: {fact_a.normalized_value}\n"
        f"- Normalized Unit: {fact_a.normalized_unit}\n"
        f"- As Of: {fact_a.as_of or 'unspecified'}\n"
        f"- Scope: {fact_a.scope or 'none'}\n"
        f"- Source Document: {fact_a.source_doc_id} (page {fact_a.page_number})\n"
        f"- Evidence Quote: \"{fact_a.evidence_text}\"\n\n"
        f"FACT B:\n"
        f"- Entity: {fact_b.entity}\n"
        f"- Attribute: {fact_b.attribute}\n"
        f"- Value: {fact_b.value}\n"
        f"- Unit: {fact_b.unit or 'none'}\n"
        f"- Normalized Value: {fact_b.normalized_value}\n"
        f"- Normalized Unit: {fact_b.normalized_unit}\n"
        f"- As Of: {fact_b.as_of or 'unspecified'}\n"
        f"- Scope: {fact_b.scope or 'none'}\n"
        f"- Source Document: {fact_b.source_doc_id} (page {fact_b.page_number})\n"
        f"- Evidence Quote: \"{fact_b.evidence_text}\"\n"
        f"{hint_section}\n"
        "Classify the relationship between FACT A and FACT B. Cite exact values, periods, and units in your reasoning."
    )

    for attempt in range(max_retries):
        try:
            if provider == "gemini":
                from google.genai import types

                model_name = model or os.getenv("GEMINI_MODEL") or "gemini-3.6-flash"
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        {"role": "user", "parts": [{"text": RELATIONSHIP_SYSTEM_PROMPT + "\n\n" + pair_prompt}]}
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=FactRelationshipJudgement,
                        temperature=0.0,
                    ),
                )
                return FactRelationshipJudgement.model_validate_json(response.text)

            elif provider == "openai":
                model_name = model or "gpt-4o-mini"
                completion = client.beta.chat.completions.parse(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": RELATIONSHIP_SYSTEM_PROMPT},
                        {"role": "user", "content": pair_prompt},
                    ],
                    response_format=FactRelationshipJudgement,
                    temperature=0.0,
                )
                parsed = completion.choices[0].message.parsed
                if parsed is None:
                    raise RuntimeError("Failed to parse OpenAI relationship response")
                return parsed

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as err:
            err_lower = str(err).lower()
            is_transient = any(
                k in err_lower
                for k in [
                    "429",
                    "resource_exhausted",
                    "quota",
                    "too many requests",
                    "503",
                    "unavailable",
                    "high demand",
                    "overloaded",
                    "servererror",
                ]
            )
            if is_transient and attempt < max_retries - 1:
                sleep_secs = min(60.0, initial_backoff * (2 ** attempt))
                print(
                    f"Warning: Transient API condition ({type(err).__name__}) encountered during relationship classification. "
                    f"Backing off for {sleep_secs:.1f}s (retry {attempt + 1}/{max_retries})...",
                    file=sys.stderr,
                )
                time.sleep(sleep_secs)
                continue
            raise err


def make_canonical_pair_key(id_a: str, id_b: str) -> Tuple[str, str]:
    """Ensures order-independent canonical representation of a fact pair (min_id, max_id)."""
    return (min(id_a, id_b), max(id_a, id_b))


def store_relationship(
    rel: Relationship,
    conn: Optional[sqlite3.Connection] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Persists a Relationship into SQLite in canonical order (fact_a_id < fact_b_id).
    Enforces order-independent deduplication via ON CONFLICT.
    Discards 'unrelated' pairs from database insertion to keep graph clean.

    Returns:
        True if inserted/updated, False if skipped (e.g. unrelated).
    """
    if rel.type == "unrelated":
        return False

    can_a, can_b = make_canonical_pair_key(rel.fact_a_id, rel.fact_b_id)

    if conn is None:
        conn = get_connection(db_path)

    upsert_sql = """
    INSERT INTO relationships (id, fact_a_id, fact_b_id, type, reasoning, confidence)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(fact_a_id, fact_b_id) DO UPDATE SET
        type=excluded.type,
        reasoning=excluded.reasoning,
        confidence=excluded.confidence;
    """

    with conn:
        conn.execute(
            upsert_sql,
            (rel.id, can_a, can_b, rel.type, rel.reasoning, rel.confidence),
        )

    return True


def get_existing_relationship_pairs(
    conn: Optional[sqlite3.Connection] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> Set[Tuple[str, str]]:
    """Retrieves all canonical (fact_a_id, fact_b_id) pairs already evaluated or stored."""
    if conn is None:
        conn = get_connection(db_path)

    cursor = conn.cursor()
    cursor.execute("SELECT fact_a_id, fact_b_id FROM relationships;")
    pairs = {make_canonical_pair_key(r[0], r[1]) for r in cursor.fetchall()}

    # Also load pairs tracked in evaluated_pairs (which includes 'unrelated' decisions)
    try:
        cursor.execute("SELECT fact_a_id, fact_b_id FROM evaluated_pairs;")
        pairs.update({make_canonical_pair_key(r[0], r[1]) for r in cursor.fetchall()})
    except Exception:
        pass

    return pairs


def get_all_stored_relationships(
    conn: Optional[sqlite3.Connection] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves all stored relationships with joined fact details for human inspection and reporting.
    """
    if conn is None:
        conn = get_connection(db_path)

    query = """
    SELECT 
        r.id AS rel_id,
        r.type,
        r.reasoning,
        r.confidence,
        fa.id AS fact_a_id,
        fa.entity AS fact_a_entity,
        fa.attribute AS fact_a_attribute,
        fa.value AS fact_a_value,
        fa.unit AS fact_a_unit,
        fa.as_of AS fact_a_as_of,
        fa.source_doc_id AS fact_a_doc,
        fa.page_number AS fact_a_page,
        fa.evidence_text AS fact_a_evidence,
        fb.id AS fact_b_id,
        fb.entity AS fact_b_entity,
        fb.attribute AS fact_b_attribute,
        fb.value AS fact_b_value,
        fb.unit AS fact_b_unit,
        fb.as_of AS fact_b_as_of,
        fb.source_doc_id AS fact_b_doc,
        fb.page_number AS fact_b_page,
        fb.evidence_text AS fact_b_evidence
    FROM relationships r
    JOIN facts fa ON r.fact_a_id = fa.id
    JOIN facts fb ON r.fact_b_id = fb.id
    ORDER BY r.created_at DESC;
    """
    cursor = conn.cursor()
    cursor.execute(query)
    results = []
    for row in cursor.fetchall():
        results.append(dict(row))
    return results


def link_facts(
    doc_id: Optional[str] = None,
    top_k: int = 5,
    min_sim: float = 0.70,
    max_pairs: Optional[int] = None,
    model: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
    db_path: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> List[Relationship]:
    """
    Executes cross-document fact linking for a specific document or all documents in storage.

    Workflow:
    1. Loads all facts with embeddings from SQLite.
    2. Identifies target facts (filtered by doc_id if provided).
    3. Finds candidate pairs across distinct documents satisfying similarity floor and domain pre-filters.
    4. Deduplicates pairs canonically so (A, B) and (B, A) are evaluated once.
    5. Calls LLM with structured reasoning to classify relationships.
    6. Persists non-unrelated relationships into SQLite relationships table.
    """
    if conn is None:
        conn = get_connection(db_path)

    init_db(conn=conn)

    # 1. Fetch all facts with embeddings
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, entity, attribute, value, unit, normalized_value, normalized_unit, "
        "as_of, scope, source_doc_id, page_number, evidence_text, confidence, extra, embedding "
        "FROM facts WHERE embedding IS NOT NULL;"
    )
    all_rows = cursor.fetchall()

    if not all_rows:
        if verbose:
            print("No facts with embeddings found in database.", file=sys.stderr)
        return []

    all_facts_with_embeddings: List[Tuple[Fact, np.ndarray]] = []
    for r in all_rows:
        fact = row_to_fact(r)
        emb = bytes_to_embedding(r["embedding"])
        if emb is not None:
            all_facts_with_embeddings.append((fact, emb))

    # 2. Identify target facts
    if doc_id:
        target_pool = [(f, e) for f, e in all_facts_with_embeddings if f.source_doc_id == doc_id]
        if not target_pool and verbose:
            print(f"No facts found for doc-id '{doc_id}' in storage.", file=sys.stderr)
            return []
    else:
        target_pool = all_facts_with_embeddings

    if verbose:
        print(
            f"Cross-Document Linking: {len(target_pool)} target facts, "
            f"{len(all_facts_with_embeddings)} total facts in pool (top_k={top_k}, min_sim={min_sim:.2f})...",
            file=sys.stderr,
        )

    # 3. Retrieve and queue candidate pairs
    existing_pairs = get_existing_relationship_pairs(conn=conn)
    queued_pairs: List[Tuple[Tuple[str, str], Fact, Fact, float]] = []
    seen_in_batch: Set[Tuple[str, str]] = set()

    for target_fact, target_emb in target_pool:
        candidates = get_candidate_pairs_for_fact(
            target_fact=target_fact,
            target_embedding=target_emb,
            all_facts_with_embeddings=all_facts_with_embeddings,
            top_k=top_k,
            min_sim=min_sim,
        )

        for sim, cand_fact in candidates:
            pair_key = make_canonical_pair_key(target_fact.id, cand_fact.id)
            if pair_key in existing_pairs or pair_key in seen_in_batch:
                continue
            seen_in_batch.add(pair_key)
            queued_pairs.append((pair_key, target_fact, cand_fact, sim))

    # Sort queued pairs descending by similarity so strongest semantic connections evaluate first
    queued_pairs.sort(key=lambda x: x[3], reverse=True)

    if max_pairs is not None and max_pairs > 0:
        queued_pairs = queued_pairs[:max_pairs]

    if verbose:
        print(
            f"Found {len(queued_pairs)} unique candidate cross-document pairs to evaluate with LLM.",
            file=sys.stderr,
        )

    relationships_created: List[Relationship] = []

    # 4. Classify each candidate pair
    for idx, (pair_key, fact_a, fact_b, sim) in enumerate(queued_pairs):
        if verbose:
            print(
                f"[{idx + 1}/{len(queued_pairs)}] Evaluating: "
                f"[{fact_a.source_doc_id}] {fact_a.entity}: {fact_a.attribute} "
                f"vs [{fact_b.source_doc_id}] {fact_b.entity}: {fact_b.attribute} (sim={sim:.3f})...",
                file=sys.stderr,
            )

        judgement = classify_relationship(
            fact_a=fact_a,
            fact_b=fact_b,
            model=model,
        )

        if verbose:
            print(f"  -> Result: {judgement.type.upper()} (conf: {judgement.confidence:.2f})", file=sys.stderr)
            print(f"     Reasoning: {judgement.reasoning}\n", file=sys.stderr)

        if not dry_run:
            can_a, can_b = pair_key
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO evaluated_pairs (fact_a_id, fact_b_id) VALUES (?, ?);",
                    (can_a, can_b),
                )

        if judgement.type == "unrelated":
            time.sleep(1.0)
            continue

        rel = Relationship(
            fact_a_id=pair_key[0],
            fact_b_id=pair_key[1],
            type=judgement.type,
            reasoning=judgement.reasoning,
            confidence=judgement.confidence,
        )

        if not dry_run:
            store_relationship(rel, conn=conn)

        relationships_created.append(rel)

        # Pacing pause to remain well within API rate and concurrency quotas
        time.sleep(2.0)

    if verbose:
        print(
            f"\nLinking complete: {len(relationships_created)} valid relationships discovered and recorded.",
            file=sys.stderr,
        )

    return relationships_created


def main() -> None:
    if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Cross-Document Fact Linking CLI (Phase 4)"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--doc-id",
        type=str,
        help="Link facts from this specific document against all other documents in storage.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run cross-document linking across all documents currently in storage.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum nearest neighbors to retrieve per fact (default: 5).",
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.70,
        help="Minimum cosine similarity threshold floor (default: 0.70).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum candidate pairs to evaluate (evaluated in order of descending cosine similarity).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model override for relationship classification (default: gemini-3.6-flash).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Custom SQLite database path (defaults to facts.db).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate relationships without storing them to SQLite.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output created relationships as JSON to stdout.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all relationships currently stored in the database.",
    )

    args = parser.parse_args()

    if args.list:
        rels = get_all_stored_relationships(db_path=args.db)
        print(f"Total stored relationships: {len(rels)}\n")
        print(json.dumps(rels, indent=2, ensure_ascii=False))
        return

    if not args.doc_id and not args.all:
        parser.print_help()
        sys.exit(0)

    results = link_facts(
        doc_id=args.doc_id,
        top_k=args.top_k,
        min_sim=args.min_sim,
        max_pairs=args.max_pairs,
        model=args.model,
        db_path=args.db,
        dry_run=args.dry_run,
        verbose=True,
    )

    if args.json:
        rels_dicts = [r.model_dump() for r in results]
        print(json.dumps(rels_dicts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()




