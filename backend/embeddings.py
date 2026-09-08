"""
Embeddings Module for Fact Knowledge Layer (Phase 3)

Generates vector embeddings for stored facts:
- Input text: `entity + attribute` (focuses on subject and metric, ignoring incidental wording).
- Provider: Google Gemini `gemini-embedding-001` (3072 dims) or OpenAI `text-embedding-3-small`.
- Deterministic offline fallback for hermetic unit testing (high semantic overlap via n-gram hashing).
"""

import hashlib
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Canonical embedding dimension for Gemini embedding-001
EMBEDDING_DIM = 3072


def fact_to_embedding_text(fact_or_entity: Union[str, Any], attribute: Optional[str] = None) -> str:
    """
    Constructs the canonical text used for fact embedding generation.
    Matches strictly on what the fact is about (entity + attribute),
    per PROJECT.md and the Phase 3 brief.
    """
    if attribute is not None:
        entity_str = str(fact_or_entity).strip()
        attr_str = str(attribute).strip()
        return f"{entity_str}: {attr_str}"

    if hasattr(fact_or_entity, "entity") and hasattr(fact_or_entity, "attribute"):
        return f"{fact_or_entity.entity.strip()}: {fact_or_entity.attribute.strip()}"

    if isinstance(fact_or_entity, dict):
        entity = fact_or_entity.get("entity", "").strip()
        attr = fact_or_entity.get("attribute", "").strip()
        return f"{entity}: {attr}"

    return str(fact_or_entity).strip()


def embedding_to_bytes(vector: Sequence[float]) -> bytes:
    """Serializes a float vector to a binary float32 BLOB for SQLite storage."""
    arr = np.asarray(vector, dtype=np.float32)
    return arr.tobytes()


def bytes_to_embedding(blob: bytes) -> np.ndarray:
    """Deserializes a binary float32 BLOB from SQLite into a NumPy float32 vector."""
    if not blob:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32)


def compute_cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Computes cosine similarity between two vectors."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _generate_mock_embedding(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """
    Generates a deterministic pseudo-semantic embedding vector using character
    and token n-gram feature hashing. Enables reproducible offline testing without
    exhausting API quotas while maintaining semantic locality.
    """
    vec = np.zeros(dim, dtype=np.float32)
    normalized = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    words = normalized.split()

    for word in words:
        # Unigram hash
        idx = int(hashlib.md5(f"w_{word}".encode()).hexdigest(), 16) % dim
        vec[idx] += 3.0

        # Character tri-grams
        if len(word) >= 3:
            for i in range(len(word) - 2):
                tri = word[i : i + 3]
                t_idx = int(hashlib.md5(f"t_{tri}".encode()).hexdigest(), 16) % dim
                vec[t_idx] += 1.0

    # Bigrams
    for i in range(len(words) - 1):
        bi = f"{words[i]}_{words[i+1]}"
        b_idx = int(hashlib.md5(f"bi_{bi}".encode()).hexdigest(), 16) % dim
        vec[b_idx] += 2.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def get_embeddings_batch(
    texts: List[str],
    model: Optional[str] = None,
    batch_size: int = 50,
) -> List[List[float]]:
    """
    Computes embedding vectors for a list of texts using Gemini (or mock fallback).
    """
    if not texts:
        return []

    # Force mock embeddings during unit tests or when explicitly set
    if os.getenv("TEST_MOCK_EMBEDDINGS") == "1":
        return [_generate_mock_embedding(t) for t in texts]

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        try:
            from google import genai

            client = genai.Client(api_key=gemini_key)
            model_name = model or os.getenv("EMBEDDING_MODEL") or "gemini-embedding-001"

            all_embeddings: List[List[float]] = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                response = client.models.embed_content(
                    model=model_name,
                    contents=chunk if len(chunk) > 1 else chunk[0],
                )
                if hasattr(response, "embeddings") and response.embeddings:
                    all_embeddings.extend([e.values for e in response.embeddings])
                elif hasattr(response, "embedding") and response.embedding:
                    all_embeddings.append(response.embedding.values)
                else:
                    raise RuntimeError(f"Unexpected response format from embed_content: {response}")

            return all_embeddings
        except Exception as err:
            # If rate-limited or unexpected API failure, fallback gracefully
            print(f"Warning: Gemini embedding failed ({err}), falling back to deterministic embedding.", file=sys.stderr)
            return [_generate_mock_embedding(t) for t in texts]

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai

            client = openai.OpenAI(api_key=openai_key)
            model_name = model or "text-embedding-3-small"
            resp = client.embeddings.create(input=texts, model=model_name)
            return [data.embedding for data in resp.data]
        except Exception as err:
            print(f"Warning: OpenAI embedding failed ({err}), falling back to deterministic embedding.", file=sys.stderr)
            return [_generate_mock_embedding(t) for t in texts]

    # Default fallback when no API key is available
    return [_generate_mock_embedding(t) for t in texts]


def get_embedding(text: str, model: Optional[str] = None) -> List[float]:
    """Computes a single embedding vector for a given text."""
    results = get_embeddings_batch([text], model=model)
    return results[0]
