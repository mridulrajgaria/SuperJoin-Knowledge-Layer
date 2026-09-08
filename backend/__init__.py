"""Fact Knowledge Layer - Backend Module"""

from backend.ingest import extract_chunks, Chunk
from backend.schema import Fact, ExtractedFact, ChunkFactsExtraction
from backend.extract import extract_document_facts, extract_facts_from_chunk, should_process_chunk, validate_evidence_grounding

__all__ = [
    "extract_chunks",
    "Chunk",
    "Fact",
    "ExtractedFact",
    "ChunkFactsExtraction",
    "extract_document_facts",
    "extract_facts_from_chunk",
    "should_process_chunk",
    "validate_evidence_grounding",
]
