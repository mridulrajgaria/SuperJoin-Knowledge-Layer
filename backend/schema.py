"""
Fact Schema Module

Defines the Pydantic data models for Fact extraction as specified in PROJECT.md:
- Fact: Complete stored fact model anchored to source_doc_id and page_number.
- ExtractedFact: Fact model extracted from LLM chunk output.
- ChunkFactsExtraction: Container for structured LLM responses.
"""

import uuid
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ExtractedFact(BaseModel):
    """
    Structured fact representation extracted by an LLM from a source chunk.
    """
    entity: str = Field(
        ...,
        description="The primary subject or entity the fact is about (e.g., 'Delhivery', 'Consumer Price Index').",
    )
    attribute: str = Field(
        ...,
        description="The specific property, metric, or statement being made (e.g., 'consolidated revenue', 'repo rate').",
    )
    value: Any = Field(
        ...,
        description="The extracted value as reported (numeric or qualitative string).",
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement if applicable (e.g., 'INR crore', '%', 'million', or null if non-numeric).",
    )
    normalized_value: Optional[float] = Field(
        default=None,
        description="Standardized numeric value (e.g., 8141.74, 6.5) for mathematical/cross-document comparison, or null if qualitative.",
    )
    normalized_unit: Optional[str] = Field(
        default=None,
        description="Canonical/standardized unit (e.g., 'INR', 'USD', '%', 'ratio', 'count', 'sq_ft'), or null if not applicable.",
    )
    as_of: Optional[str] = Field(
        default=None,
        description="The date or time period the fact refers to (e.g., 'FY24', 'Q4 FY24', 'March 31, 2024', or null).",
    )
    scope: Optional[str] = Field(
        default=None,
        description="The scope or qualifying dimension (e.g., 'consolidated', 'standalone', 'urban', or null).",
    )
    evidence_text: str = Field(
        ...,
        description="Exact substring or quote from the source chunk supporting this fact.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Model confidence score between 0.0 and 1.0.",
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Open bag for unanticipated attributes or context (e.g., footnotes, accounting standards).",
    )


class ChunkFactsExtraction(BaseModel):
    """
    Container schema for structured LLM output from a single chunk.
    """
    facts: List[ExtractedFact] = Field(
        default_factory=list,
        description="List of discrete facts extracted from the chunk.",
    )


class Fact(BaseModel):
    """
    Full Fact model matching PROJECT.md, anchored to document and page.
    """
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the fact.",
    )
    entity: str = Field(
        ...,
        description="The primary entity (e.g., 'Delhivery', 'India CPI inflation').",
    )
    attribute: str = Field(
        ...,
        description="The metric or property (e.g., 'consolidated revenue', 'GDP growth rate').",
    )
    value: Any = Field(
        ...,
        description="The value of the attribute.",
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement (e.g., 'INR crore', '%', or null).",
    )
    normalized_value: Optional[float] = Field(
        default=None,
        description="Standardized numeric value (e.g., 8141.74, 6.5) for mathematical/cross-document comparison, or null if qualitative.",
    )
    normalized_unit: Optional[str] = Field(
        default=None,
        description="Canonical/standardized unit (e.g., 'INR', 'USD', '%', 'ratio', 'count', 'sq_ft'), or null if not applicable.",
    )
    as_of: Optional[str] = Field(
        default=None,
        description="Date or period the fact refers to (e.g., 'FY24', 'Q4 FY24').",
    )
    scope: Optional[str] = Field(
        default=None,
        description="Scope or qualification (e.g., 'consolidated', 'standalone', 'India').",
    )
    source_doc_id: str = Field(
        ...,
        description="Identifier of the source document.",
    )
    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed page number where the fact appears.",
    )
    evidence_text: str = Field(
        ...,
        description="Exact quote or snippet supporting the fact.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence between 0.0 and 1.0.",
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Open bag for attributes that do not fit standard fields.",
    )

    @classmethod
    def from_extracted(
        cls,
        extracted: ExtractedFact,
        source_doc_id: str,
        page_number: int,
        fact_id: Optional[str] = None,
    ) -> "Fact":
        """Builds a grounded Fact instance from an ExtractedFact and chunk metadata."""
        return cls(
            id=fact_id or str(uuid.uuid4()),
            entity=extracted.entity,
            attribute=extracted.attribute,
            value=extracted.value,
            unit=extracted.unit,
            normalized_value=extracted.normalized_value,
            normalized_unit=extracted.normalized_unit,
            as_of=extracted.as_of,
            scope=extracted.scope,
            source_doc_id=source_doc_id,
            page_number=page_number,
            evidence_text=extracted.evidence_text,
            confidence=extracted.confidence,
            extra=extracted.extra,
        )


class Relationship(BaseModel):
    """
    Represents a semantic relationship between two stored facts (PROJECT.md Phase 4).
    """
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the relationship.",
    )
    fact_a_id: str = Field(
        ...,
        description="Identifier of the primary fact.",
    )
    fact_b_id: str = Field(
        ...,
        description="Identifier of the comparison fact.",
    )
    type: str = Field(
        ...,
        description="Type of relationship: 'corroborates', 'contradicts', or 'reconciled_by_context'.",
    )
    reasoning: str = Field(
        ...,
        description="LLM explanation detailing the corroboration, contradiction, or contextual difference.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this relationship judgement.",
    )


class FactRelationshipJudgement(BaseModel):
    """
    Structured response schema for LLM relationship classification.
    """
    type: Literal["corroborates", "contradicts", "reconciled_by_context", "unrelated"] = Field(
        ...,
        description="Type of relationship: 'corroborates', 'contradicts', 'reconciled_by_context', or 'unrelated'.",
    )
    reasoning: str = Field(
        ...,
        description="Detailed explanation detailing why the facts corroborate, contradict, or reconcile, citing specific values, periods, scopes, or units.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this relationship judgement.",
    )

