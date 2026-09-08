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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path when invoked directly as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

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
2. Fact Representation:
   - 'entity': The primary subject or entity the fact relates to (e.g., an organization, product, country, person, index, or concept).
   - 'attribute': The property, metric, event, or relation being asserted (e.g., 'consolidated revenue', 'headcount', 'growth rate', 'inflation rate', 'registered office').
   - 'value': The value or statement reported. Keep the original formatting and numeric precision as stated.
   - 'unit': The unit of measurement if applicable (e.g., '%', 'INR crore', 'USD', 'million', or null if non-numeric).
   - 'normalized_value': Standardized numeric representation of the value (as a float, e.g., 8141.74 for '8,141.74 Cr', 6.5 for '6.50%', 0.082 for '8.2%'), or null if qualitative/non-numeric.
   - 'normalized_unit': Canonical/standardized unit symbol or denomination (e.g., 'INR', 'USD', '%', 'count', 'ratio', 'sq_ft'), or null if not applicable.
   - 'as_of': The temporal period or effective date the fact refers to (e.g., 'FY24', 'Q4 FY24', 'March 31, 2024', or null if not time-bound).
   - 'scope': The qualifying scope or segment if specified (e.g., 'consolidated', 'standalone', 'urban', 'rural', or null).
   - 'confidence': Your confidence score between 0.0 and 1.0 that the fact is accurately stated and extracted.
   - 'extra': A flexible key-value dictionary for any domain-specific qualifiers, accounting notes, or footnotes that do not fit into the standard fields.
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


def extract_facts_from_chunk(
    chunk: Chunk,
    client: Optional[Any] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[ExtractedFact]:
    """
    Calls LLM with structured output to extract facts from a single chunk.
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
        f"Content:\n{chunk['text']}"
    )

    if provider == "gemini":
        from google.genai import types

        model_name = model or "gemini-2.5-flash"
        response = client.models.generate_content(
            model=model_name,
            contents=[
                {"role": "user", "parts": [{"text": GENERIC_EXTRACTION_SYSTEM_PROMPT + "\n\n" + chunk_prompt}]}
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ChunkFactsExtraction,
                temperature=0.0,
            ),
        )
        parsed = ChunkFactsExtraction.model_validate_json(response.text)
        return parsed.facts

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


def extract_document_facts(
    pdf_path: str | Path,
    doc_id: Optional[str] = None,
    client: Optional[Any] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_chunks: Optional[int] = None,
) -> List[Fact]:
    """
    Runs end-to-end ingestion and structured fact extraction on a PDF.

    1. Ingests PDF into chunks.
    2. Applies cheap heuristic filtering.
    3. Calls LLM with structured output.
    4. Validates evidence grounding against source chunks.
    5. Returns grounded Fact models.
    """
    raw_chunks = extract_chunks(pdf_path, doc_id=doc_id)
    filtered_chunks = [c for c in raw_chunks if should_process_chunk(c)]

    if max_chunks is not None and max_chunks > 0:
        filtered_chunks = filtered_chunks[:max_chunks]

    facts: List[Fact] = []

    for chunk in filtered_chunks:
        try:
            extracted_facts = extract_facts_from_chunk(
                chunk=chunk,
                client=client,
                provider=provider,
                model=model,
            )
        except Exception as err:
            # Continue extracting from remaining chunks on isolated failure
            print(f"Warning: Extraction error on page {chunk['page_number']}: {err}", file=sys.stderr)
            continue

        for ef in extracted_facts:
            # Validate evidence grounding
            is_grounded, grounding_score = validate_evidence_grounding(
                evidence_text=ef.evidence_text,
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

            facts.append(fact)

    return facts


def main() -> None:
    # Ensure UTF-8 output encoding on Windows consoles
    if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Extract structured facts from a PDF document using LLM structured outputs."
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
