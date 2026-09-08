"""
Fact Extraction Module

Extracts grounded facts from document chunks using structured LLM output.
Adheres to the Fact schema defined in PROJECT.md:
- Uses cheap heuristic filtering to skip non-factual chunks.
- Employs a generic, domain-agnostic extraction prompt.
- Validates evidence grounding (verifies evidence_text against source chunk).
- Supports Gemini (google-genai) and OpenAI (gpt-4o-mini) with structured outputs.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path when invoked directly as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from backend.ingest import Chunk, extract_chunks
from backend.schema import ChunkFactsExtraction, ExtractedFact, Fact

# Load local .env if present
load_dotenv()

GENERIC_EXTRACTION_SYSTEM_PROMPT = """You are an expert factual extraction system.
Your task is to extract atomic, verifiable facts from the provided text or table chunk.

Rules:
1. Grounding & Evidence:
   - Every fact must be directly stated in the input text.
   - The 'evidence_text' field MUST be an exact substring quoted directly from the input chunk that contains the fact. Do NOT invent, extrapolate, or paraphrase evidence.
2. Entity vs Attribute Rules (CRITICAL):
   - 'entity': MUST ALWAYS BE the real-world named subject — an actual company, organization, country, state, government agency, person, or specific named institution/index.
     * In corporate filings, earnings decks, or annual reports, operational metrics, facility counts, equipment, and financial results belong to the reporting enterprise (e.g. 'Delhivery', or a named subsidiary/shareholder like 'CA Swift Investments').
     * In macroeconomic documents, the entity is the country, region, central bank, or agency (e.g. 'India', 'Reserve Bank of India', 'IMF', 'Global Economy').
     * NEVER use metric types, equipment names, asset classes, or operational items as the entity! For example, '46-ft tractors', 'Fleet', 'Active customers', 'Pin codes', 'Staff', 'Revenue', 'Gateways' are NEVER entities — they are metrics or categories that belong in the 'attribute' field under the entity (e.g. entity: 'Delhivery', attribute: 'count of 46-ft tractors').
   - 'attribute': The specific property, metric, equipment count, operational figure, event, or relation being reported about that entity (e.g., 'count of 46-ft tractors', 'daily average fleet size', 'pin codes covered', 'active customer count', 'consolidated revenue', 'retail inflation rate').
   - 'value': The value or statement reported. Keep the original formatting and numeric precision as stated.
   - 'unit': The unit of measurement if applicable (e.g., '%', 'INR crore', 'USD', 'million', or null if non-numeric).
   - 'normalized_value': Standardized numeric representation of the value (as a float, e.g., 8141.74 for '8,141.74 Cr', 6.5 for '6.50%', 0.082 for '8.2%'), or null if qualitative/non-numeric.
   - 'normalized_unit': Canonical/standardized unit symbol or denomination (e.g., 'INR', 'USD', '%', 'count', 'ratio', 'sq_ft'), or null if not applicable.
   - 'as_of': The temporal period or effective date the fact refers to (e.g., 'FY24', 'Q4 FY24', 'March 31, 2024', or null if not time-bound).
     * STRICT TEMPORAL GROUNDING: 'as_of' MUST be explicitly stated within the chunk text itself (or verified directly in the table header).
     * If a chunk contains a sequence of numbers/metrics without period labels or column headers, DO NOT guess or extrapolate quarters or fiscal years (e.g. NEVER invent Q1/Q2/Q3/Q4 when headers are missing). Set 'as_of' to null!
   - 'scope': The qualifying scope or segment if specified (e.g., 'consolidated', 'standalone', 'urban', 'rural', or null).
   - 'confidence': Your confidence score between 0.0 and 1.0 that the fact is accurately stated and extracted.
   - 'extra': A flexible key-value list for domain-specific qualifiers, footnotes, or accounting notes that do not fit into the standard fields.
3. No Hallucinations:
   - If the chunk contains no verifiable facts (e.g., greetings, boilerplate legal disclaimers, pure navigation headers), return an empty list: {"facts": []}.
   - Do not perform calculations or conversions unless explicitly stated in the chunk.
"""


def should_process_chunk(chunk: Chunk) -> bool:
    """
    Cheap rule-based heuristic to filter out non-factual chunks without LLM calls.

    - Always retains table chunks (chunk_type == 'table').
    - Discards short strings (<25 chars) without numbers.
    - Discards standalone dates, page numbers, and boilerplate header lines.
    """
    if chunk.get("chunk_type") == "table":
        return True

    text = chunk.get("text", "").strip()
    if not text:
        return False

    # Skip very short fragments with no numeric information
    if len(text) < 25 and not any(ch.isdigit() for ch in text):
        return False

    # Standalone date stamp (e.g., "Date: May 17, 2024")
    if re.match(r"^(Date:?\s*)?[A-Za-z]+\s+\d{1,2},\s*\d{4}$", text, re.IGNORECASE):
        return False

    # Standalone page numbering (e.g., "Page 12", "12", "Page 2 of 50")
    if re.match(r"^(Page\s+\d+(\s+of\s+\d+)?|\d+)$", text, re.IGNORECASE):
        return False

    # Pure navigation or document structural markers
    if re.match(
        r"^(Table of Contents|Contents|Disclaimer|Index|Dear Sir/?\s*Madam,?|Thank you\.?|Encl:.*)$",
        text,
        re.IGNORECASE,
    ):
        return False

    # Skip single/double word non-numeric labels
    words = text.split()
    if len(words) <= 2 and not any(ch.isdigit() for ch in text):
        return False

    return True


def normalize_whitespace(s: str) -> str:
    """Normalizes whitespace and linebreaks for robust comparison."""
    return re.sub(r"\s+", " ", s).strip().lower()


def validate_evidence_grounding(evidence_text: str, source_text: str) -> Tuple[bool, float]:
    """
    Verifies that evidence_text is actually grounded in the source chunk.

    Returns:
        (is_grounded, grounding_score)
        - is_grounded: True if evidence_text is a normalized substring or has >= 85% token overlap.
        - grounding_score: Overlap ratio (0.0 to 1.0).
    """
    if not evidence_text or not source_text:
        return False, 0.0

    norm_evidence = normalize_whitespace(evidence_text)
    norm_source = normalize_whitespace(source_text)

    # 1. Exact normalized substring match
    if norm_evidence in norm_source:
        return True, 1.0

    # 2. Token overlap ratio for multiline or slightly formatted excerpts
    evidence_tokens = set(re.findall(r"\w+", norm_evidence))
    source_tokens = set(re.findall(r"\w+", norm_source))

    if not evidence_tokens:
        return False, 0.0

    overlap = evidence_tokens.intersection(source_tokens)
    ratio = len(overlap) / len(evidence_tokens)

    is_grounded = ratio >= 0.85
    return is_grounded, ratio


def validate_temporal_grounding(as_of: Optional[str], source_text: str) -> Tuple[bool, Optional[str]]:
    """
    Verifies that the extracted `as_of` temporal period is genuinely grounded in the source chunk,
    preventing LLM hallucination of quarters (e.g. fabricating Q1-Q4 FY24 on headerless tables).

    Returns:
        (is_grounded, cleaned_as_of)
        - is_grounded: True if temporal markers are present in source chunk or as_of is None.
        - cleaned_as_of: The original as_of if grounded, or None if ungrounded.
    """
    if not as_of or not as_of.strip():
        return True, None

    as_of_clean = as_of.strip()
    norm_source = normalize_whitespace(source_text)

    # 1. Direct normalized match
    if normalize_whitespace(as_of_clean) in norm_source:
        return True, as_of_clean

    # 2. Extract temporal tokens
    quarter_match = re.search(r"\b(Q[1-4])\b", as_of_clean, re.IGNORECASE)
    fy_match = re.search(r"\b(FY\s*\d{2,4}|\d{4}[-–]\d{2,4})\b", as_of_clean, re.IGNORECASE)
    year_match = re.search(r"\b(20\d{2})\b", as_of_clean)
    month_match = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
        as_of_clean,
        re.IGNORECASE,
    )

    temporal_tokens_checked = []

    # If a quarter was specified (e.g. Q1, Q2, Q3, Q4), it MUST be in the source chunk
    if quarter_match:
        q_token = quarter_match.group(1).lower()
        temporal_tokens_checked.append(q_token)
        if not re.search(rf"\b{re.escape(q_token)}\b", norm_source, re.IGNORECASE):
            return False, None

    # If a fiscal year or multi-year span was specified, check presence
    if fy_match:
        fy_raw = fy_match.group(1)
        fy_token = re.sub(r"\s+", "", fy_raw).lower()
        temporal_tokens_checked.append(fy_token)
        source_condensed = norm_source.replace(" ", "")

        matched_fy = False
        if fy_token in source_condensed or re.search(rf"\b{re.escape(fy_token)}\b", norm_source, re.IGNORECASE):
            matched_fy = True
        else:
            # Check if 2-digit FY matches e.g. FY24 matches 2023-24, 2024, or fy 24
            m_yr2 = re.search(r"FY(\d{2})", fy_token, re.IGNORECASE)
            if m_yr2:
                yr2 = m_yr2.group(1)
                if re.search(rf"(\bfy\s*{yr2}\b|20\d\d[-–]{yr2}|20{yr2})", norm_source, re.IGNORECASE):
                    matched_fy = True
        if not matched_fy:
            return False, None

    # If a 4-digit calendar year was specified without FY
    if year_match and not fy_match:
        yr = year_match.group(1)
        temporal_tokens_checked.append(yr)
        if yr not in norm_source:
            return False, None

    # If a month was specified
    if month_match:
        mo = month_match.group(1).lower()
        temporal_tokens_checked.append(mo)
        if not re.search(rf"\b{re.escape(mo)}\b", norm_source, re.IGNORECASE):
            return False, None

    # If no specific patterns matched but as_of is non-empty, check word overlap
    if not temporal_tokens_checked:
        tokens = [w for w in re.findall(r"\w+", as_of_clean.lower()) if len(w) > 1]
        if tokens and not any(t in norm_source for t in tokens):
            return False, None

    return True, as_of_clean


def get_checkpoint_path(doc_id: str, checkpoint_dir: Optional[Path] = None) -> Path:
    """Returns the checkpoint file path for a document."""
    cp_dir = checkpoint_dir or (PROJECT_ROOT / "data" / "extracted" / ".checkpoints")
    cp_dir.mkdir(parents=True, exist_ok=True)
    return cp_dir / f"{doc_id}.checkpoint.json"


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Loads existing extraction checkpoint if valid, otherwise returns an empty dict."""
    if checkpoint_path.exists():
        try:
            return json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}", file=sys.stderr)
            return {}
    return {}


def save_checkpoint(
    checkpoint_path: Path,
    doc_id: str,
    pdf_path: str,
    total_chunks: int,
    completed_indices: List[int],
    facts: List[Fact],
    is_completed: bool = False,
) -> None:
    """Atomically saves checkpoint to disk using a temporary file swap."""
    data = {
        "doc_id": doc_id,
        "pdf_path": str(pdf_path),
        "total_chunks": total_chunks,
        "completed_count": len(completed_indices),
        "completed_indices": sorted(list(set(completed_indices))),
        "facts": [f.model_dump() for f in facts],
        "is_completed": is_completed,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp_path = checkpoint_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(checkpoint_path)


def get_llm_client() -> Tuple[str, Any]:
    """
    Detects and initializes an LLM client based on available environment variables.
    Prefers Gemini (google-genai) with fallback to OpenAI.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        from google import genai

        client = genai.Client(api_key=gemini_key)
        return "gemini", client

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        import openai

        client = openai.OpenAI(api_key=openai_key)
        return "openai", client

    return "none", None


class GeminiExtraField(BaseModel):
    key: str = Field(description="Property or qualifier name")
    value: str = Field(description="Property value")


class GeminiExtractedFact(BaseModel):
    entity: str = Field(description="The primary subject or entity the fact is about.")
    attribute: str = Field(description="The specific property, metric, or statement being made.")
    value: str = Field(description="The extracted value as reported.")
    unit: Optional[str] = Field(default=None, description="Unit of measurement if applicable.")
    normalized_value: Optional[float] = Field(default=None, description="Standardized numeric value.")
    normalized_unit: Optional[str] = Field(default=None, description="Canonical/standardized unit.")
    as_of: Optional[str] = Field(default=None, description="Date or time period the fact refers to.")
    scope: Optional[str] = Field(default=None, description="Scope or qualifying dimension.")
    evidence_text: str = Field(description="Exact substring or quote from source chunk.")
    confidence: float = Field(default=1.0, description="Model confidence score between 0.0 and 1.0.")
    extra: List[GeminiExtraField] = Field(default_factory=list, description="List of key-value pairs for unanticipated qualifiers.")


class GeminiChunkFactsExtraction(BaseModel):
    facts: List[GeminiExtractedFact] = Field(default_factory=list)


def extract_facts_from_chunk(
    chunk: Chunk,
    client: Optional[Any] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 5,
    initial_backoff: float = 4.0,
) -> List[ExtractedFact]:
    """
    Calls LLM with structured output to extract facts from a single chunk.
    Includes rate-limit awareness with exponential backoff on 429/RESOURCE_EXHAUSTED errors.
    """
    if client is None or provider is None:
        provider, client = get_llm_client()

    if provider == "none" or client is None:
        raise RuntimeError(
            "No LLM API key configured. Please set GEMINI_API_KEY or OPENAI_API_KEY in your environment or .env file."
        )

    chunk_prompt = (
        f"Document ID: {chunk['doc_id']}\n"
        f"Page: {chunk['page_number']}\n"
        f"Chunk Type: {chunk['chunk_type']}\n\n"
        f"Content:\n{chunk['text']}\n\n"
        "Remember: 'entity' MUST ALWAYS be the real-world named subject (e.g. company, country, organization) and NEVER a metric type, equipment, or category (e.g., '46-ft tractors', 'Fleet', 'Active customers', 'Pin codes' belong in 'attribute' under entity 'Delhivery')."
    )

    for attempt in range(max_retries):
        try:
            if provider == "gemini":
                from google.genai import types

                model_name = model or os.getenv("GEMINI_MODEL") or "gemini-3.6-flash"
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        {"role": "user", "parts": [{"text": GENERIC_EXTRACTION_SYSTEM_PROMPT + "\n\n" + chunk_prompt}]}
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GeminiChunkFactsExtraction,
                        temperature=0.0,
                    ),
                )
                gemini_parsed = GeminiChunkFactsExtraction.model_validate_json(response.text)
                return [
                    ExtractedFact(
                        entity=f.entity,
                        attribute=f.attribute,
                        value=f.value,
                        unit=f.unit,
                        normalized_value=f.normalized_value,
                        normalized_unit=f.normalized_unit,
                        as_of=f.as_of,
                        scope=f.scope,
                        evidence_text=f.evidence_text,
                        confidence=f.confidence,
                        extra={item.key: item.value for item in f.extra},
                    )
                    for f in gemini_parsed.facts
                ]

            elif provider == "openai":
                model_name = model or "gpt-4o-mini"
                completion = client.beta.chat.completions.parse(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": GENERIC_EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": chunk_prompt},
                    ],
                    response_format=ChunkFactsExtraction,
                    temperature=0.0,
                )
                parsed = completion.choices[0].message.parsed
                return parsed.facts if parsed else []

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as err:
            err_lower = str(err).lower()
            is_rate_limit = any(k in err_lower for k in ["429", "resource_exhausted", "quota", "too many requests"])
            if is_rate_limit and attempt < max_retries - 1:
                sleep_secs = min(60.0, initial_backoff * (2 ** attempt))
                print(
                    f"Warning: Rate limit hit (429) on chunk (p{chunk.get('page_number')}). "
                    f"Backing off for {sleep_secs:.1f}s (retry {attempt + 1}/{max_retries})...",
                    file=sys.stderr,
                )
                time.sleep(sleep_secs)
                continue
            raise err


def extract_document_facts(
    pdf_path: str | Path,
    doc_id: Optional[str] = None,
    client: Optional[Any] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_chunks: Optional[int] = None,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    resume: bool = True,
    rpm: float = 15.0,
    checkpoint_dir: Optional[Path] = None,
    verbose: bool = True,
) -> List[Fact]:
    """
    Runs rate-limit-aware, resumable batch fact extraction on a PDF document.

    Features:
    1. Ingests PDF and applies cheap heuristic filtering.
    2. Checkpoints progress after every chunk to enable seamless resume.
    3. Handles 429 rate limit errors with exponential backoff.
    4. Enforces temporal grounding to eliminate hallucinated quarters/years.
    5. Paces API requests to respect provider RPM limits.
    """
    pdf_file = Path(pdf_path)
    effective_doc_id = doc_id if doc_id else pdf_file.stem

    raw_chunks = extract_chunks(pdf_file, doc_id=effective_doc_id)
    if start_page is not None:
        raw_chunks = [c for c in raw_chunks if c["page_number"] >= start_page]
    if end_page is not None:
        raw_chunks = [c for c in raw_chunks if c["page_number"] <= end_page]

    filtered_chunks = [c for c in raw_chunks if should_process_chunk(c)]

    if max_chunks is not None and max_chunks > 0:
        filtered_chunks = filtered_chunks[:max_chunks]

    total_chunks = len(filtered_chunks)
    cp_path = get_checkpoint_path(effective_doc_id, checkpoint_dir)
    completed_indices: set = set()
    facts: List[Fact] = []

    # Resume from checkpoint if available
    if resume and cp_path.exists():
        cp_data = load_checkpoint(cp_path)
        if cp_data.get("doc_id") == effective_doc_id:
            completed_indices = set(cp_data.get("completed_indices", []))
            raw_facts = cp_data.get("facts", [])
            facts = [Fact.model_validate(f) for f in raw_facts]
            if verbose:
                print(
                    f"Resuming '{effective_doc_id}' from checkpoint: "
                    f"{len(completed_indices)}/{total_chunks} chunks already processed "
                    f"({len(facts)} facts recovered).",
                    file=sys.stderr,
                )

    try:
        for idx, chunk in enumerate(filtered_chunks):
            if idx in completed_indices:
                continue

            t_start = time.time()
            if verbose:
                print(
                    f"[{effective_doc_id}] Processing chunk {idx + 1}/{total_chunks} "
                    f"(p{chunk['page_number']}, {chunk['chunk_type']})...",
                    file=sys.stderr,
                    end="\r",
                )

            try:
                extracted_facts = extract_facts_from_chunk(
                    chunk=chunk,
                    client=client,
                    provider=provider,
                    model=model,
                )
            except Exception as err:
                err_lower = str(err).lower()
                is_rate_limit = any(k in err_lower for k in ["429", "resource_exhausted", "quota"])
                if is_rate_limit:
                    # Save checkpoint before raising on exhausted daily quota
                    save_checkpoint(
                        checkpoint_path=cp_path,
                        doc_id=effective_doc_id,
                        pdf_path=str(pdf_file),
                        total_chunks=total_chunks,
                        completed_indices=list(completed_indices),
                        facts=facts,
                        is_completed=False,
                    )
                    print(
                        f"\n[RATE LIMIT / QUOTA EXHAUSTED] Paused at chunk {idx + 1}/{total_chunks}. "
                        f"Progress safely checkpointed to {cp_path}.\n"
                        f"Run again with --resume to continue.",
                        file=sys.stderr,
                    )
                    raise
                else:
                    print(
                        f"\nWarning: Chunk {idx + 1} (page {chunk['page_number']}) extraction failed: {err}",
                        file=sys.stderr,
                    )
                    completed_indices.add(idx)
                    save_checkpoint(
                        checkpoint_path=cp_path,
                        doc_id=effective_doc_id,
                        pdf_path=str(pdf_file),
                        total_chunks=total_chunks,
                        completed_indices=list(completed_indices),
                        facts=facts,
                        is_completed=False,
                    )
                    continue

            for ef in extracted_facts:
                # 1. Validate evidence grounding
                is_grounded, grounding_score = validate_evidence_grounding(
                    evidence_text=ef.evidence_text,
                    source_text=chunk["text"],
                )

                # 2. Validate temporal grounding (Option b: prevent fabricated quarters/dates)
                temporal_grounded, cleaned_as_of = validate_temporal_grounding(
                    as_of=ef.as_of,
                    source_text=chunk["text"],
                )

                fact = Fact.from_extracted(
                    extracted=ef,
                    source_doc_id=chunk["doc_id"],
                    page_number=chunk["page_number"],
                )

                # Record grounding metadata
                if not is_grounded:
                    fact.extra["grounding_warning"] = "Evidence text not strictly grounded in source chunk"
                    fact.confidence = round(min(fact.confidence, 0.5), 2)
                else:
                    fact.extra["grounding_score"] = round(grounding_score, 2)

                if not temporal_grounded and ef.as_of:
                    fact.extra["as_of_hallucination_detected"] = f"Removed ungrounded as_of '{ef.as_of}'"
                    fact.as_of = None
                else:
                    fact.as_of = cleaned_as_of

                facts.append(fact)

            completed_indices.add(idx)
            save_checkpoint(
                checkpoint_path=cp_path,
                doc_id=effective_doc_id,
                pdf_path=str(pdf_file),
                total_chunks=total_chunks,
                completed_indices=list(completed_indices),
                facts=facts,
                is_completed=(len(completed_indices) == total_chunks),
            )

            # Pacing delay to adhere strictly to rate limits (e.g. 15 RPM = 4.0s per request)
            if rpm > 0 and idx < total_chunks - 1:
                elapsed = time.time() - t_start
                delay = (60.0 / rpm) - elapsed
                if delay > 0:
                    time.sleep(delay)

    except KeyboardInterrupt:
        save_checkpoint(
            checkpoint_path=cp_path,
            doc_id=effective_doc_id,
            pdf_path=str(pdf_file),
            total_chunks=total_chunks,
            completed_indices=list(completed_indices),
            facts=facts,
            is_completed=False,
        )
        print(
            f"\n[INTERRUPTED] Progress checkpointed: {len(completed_indices)}/{total_chunks} chunks completed. "
            f"Resume anytime with --resume.",
            file=sys.stderr,
        )
        raise

    if verbose:
        print(
            f"\nExtraction complete for '{effective_doc_id}': "
            f"{total_chunks} chunks processed, {len(facts)} facts extracted.",
            file=sys.stderr,
        )

    return facts


def main() -> None:
    # Ensure UTF-8 output encoding on Windows consoles
    if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Extract structured facts from a PDF document with rate-limit handling and checkpointing."
    )
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to extract facts from.")
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Optional document ID (defaults to filename stem).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional LLM model override (e.g., 'gemini-2.5-flash' or 'gpt-4o-mini').",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional limit on number of chunks to process (useful for testing).",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=None,
        help="Optional 1-indexed starting page number to extract from.",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Optional 1-indexed ending page number to extract from.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing checkpoint if available (default: True).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not resume; start extraction from the beginning.",
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=15.0,
        help="Pacing rate limit in requests per minute (default: 15.0 for free tier). Set 0 to disable artificial sleep.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional custom output path (defaults to data/extracted/<doc_id>.json).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for stdout (default: 2).",
    )

    args = parser.parse_args()

    pdf_file = Path(args.pdf_path)
    if not pdf_file.exists():
        print(f"Error: File not found: {pdf_file}", file=sys.stderr)
        sys.exit(1)

    effective_doc_id = args.doc_id if args.doc_id else pdf_file.stem

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("data") / "extracted" / f"{effective_doc_id}.json"

    try:
        facts = extract_document_facts(
            pdf_path=pdf_file,
            doc_id=effective_doc_id,
            model=args.model,
            max_chunks=args.max_chunks,
            start_page=args.start_page,
            end_page=args.end_page,
            resume=args.resume,
            rpm=args.rpm,
        )

        facts_dicts = [f.model_dump() for f in facts]
        json_output = json.dumps(facts_dicts, ensure_ascii=False, indent=args.indent)

        # Save to data/extracted/<doc_id>.json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_output, encoding="utf-8")

        print(json_output)
    except Exception as exc:
        print(f"Error during extraction: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

