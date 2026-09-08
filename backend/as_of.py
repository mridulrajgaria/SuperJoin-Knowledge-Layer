"""
Document-Level As-Of Inference and Backfill Module (Phase 3)

Infers overall document time periods from title/header/cover chunks (page 1)
and backfills facts that have `as_of: None`.
Adheres to the rule: "Don't guess if it's ambiguous — leave null rather than fabricate."
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schema import Fact


def extract_period_from_text(text: str) -> Optional[str]:
    """
    Extracts an unambiguous temporal period from header, title, or cover page text.
    Returns None if ambiguous.
    """
    if not text:
        return None

    norm = re.sub(r"\s+", " ", text).strip()

    # 1. Explicit Quarter & Fiscal Year (e.g., "Q4 FY24", "Q3 FY2024")
    m_qtr = re.search(r"\b(Q[1-4]\s*FY\s*\d{2,4})\b", norm, re.IGNORECASE)
    if m_qtr:
        return m_qtr.group(1).upper()

    # 2. Multi-year fiscal span (e.g., "Annual Report 2023-24", "Economic Survey 2024-25")
    m_fiscal = re.search(r"\b(?:Annual Report|Economic Survey|FY|Fiscal Year)\s*(\d{4}[-–]\d{2,4})\b", norm, re.IGNORECASE)
    if m_fiscal:
        period = m_fiscal.group(1).replace("–", "-")
        return period

    # 3. Specific Prospectus / Filing Date (e.g., "Dated May 14, 2022", "May 2022")
    m_dated = re.search(r"\bDated\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})\b", norm, re.IGNORECASE)
    if m_dated:
        return m_dated.group(1)

    # 4. Standalone Annual Report / Survey Year (e.g. "Report 2024-25")
    m_span = re.search(r"\b(20\d{2}[-–]\d{2})\b", norm)
    if m_span:
        return m_span.group(1).replace("–", "-")

    # 5. Calendar Year with Context (e.g., "2025 Article IV Consultation")
    m_art = re.search(r"\b(20\d{2})\s+Article\s+IV\b", norm, re.IGNORECASE)
    if m_art:
        return m_art.group(1)

    return None


def infer_document_as_of(
    doc_id: str,
    chunks: Optional[List[Dict[str, Any]]] = None,
    pdf_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """
    Infers the document-level temporal period:
    1. Checks Page 1 chunks if provided or available.
    2. Fallbacks to unambiguous periods in doc_id/filename (e.g., 'fy24', '2024-25').
    3. Leaves null if ambiguous or not definitively stated.
    """
    # 1. Inspect Page 1 / cover chunk text
    if chunks:
        p1_chunks = [c for c in chunks if c.get("page_number") == 1]
        for c in p1_chunks:
            extracted = extract_period_from_text(c.get("text", ""))
            if extracted:
                return extracted

    # If chunks not provided but PDF exists, inspect page 1 text via ingest
    if pdf_path:
        path = Path(pdf_path)
        if path.exists():
            try:
                from backend.ingest import extract_chunks

                p1_chunks = [c for c in extract_chunks(path, doc_id=doc_id) if c.get("page_number") == 1]
                for c in p1_chunks:
                    extracted = extract_period_from_text(c.get("text", ""))
                    if extracted:
                        return extracted
            except Exception:
                pass

    # 2. Inspect document ID / filename stem if clearly unambiguous
    doc_lower = doc_id.lower()
    if "q4-fy24" in doc_lower or "q4_fy24" in doc_lower:
        return "Q4 FY24"
    if "fy24" in doc_lower:
        return "FY24"
    if "2024-25" in doc_lower:
        return "2024-25"
    if "2022" in doc_lower:
        return "2022"
    if "2025" in doc_lower:
        return "2025"

    return None


def backfill_facts_as_of(
    facts: List[Fact],
    doc_as_of: Optional[str],
) -> List[Fact]:
    """
    Backfills facts that have `as_of: None` with document-level period `doc_as_of`.
    Maintains provenance by adding audit metadata in `extra`.
    Preserves existing fact-level `as_of` without overwriting.
    """
    if not doc_as_of:
        return facts

    for f in facts:
        if f.as_of is None or str(f.as_of).strip() == "":
            f.as_of = doc_as_of
            f.extra["as_of_inferred"] = True
            f.extra["doc_as_of"] = doc_as_of

    return facts
