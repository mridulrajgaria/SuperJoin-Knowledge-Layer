# Fact Knowledge Layer

A document intelligence and fact-verification system that ingests PDF documents, extracts structured facts grounded in source evidence, and discovers cross-document relationships (corroboration, contradiction, contextual reconciliation) using embedding retrieval and LLM-based reasoning.

## Setup and Run Instructions

### Prerequisites

- Python 3.12+
- A Google Gemini API key (used for extraction, embeddings, and relationship reasoning)

### Installation

```bash
git clone https://github.com/mridulrajgaria/SuperJoin-Knowledge-Layer.git
cd SuperJoin-Knowledge-Layer
pip install -r requirements.txt --break-system-packages
```

### Configuration

Copy the environment template and add your API key:

```bash
cp .env.example .env
```

Edit `.env` and set:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

The `.env` file is gitignored and was never committed to version control.

### Running

```bash
python -m uvicorn backend.api:app --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in a browser. The application serves both the REST API and the frontend UI from the same server.

### Running Tests

```bash
python -m pytest tests/ -q
```

31 tests pass, covering ingestion, extraction, storage, linking, and API endpoints.

---

## Video Demo

<!-- TODO: Insert video link here -->
*Video link to be added — demonstrates a PDF being processed end-to-end, a corroborated fact, a genuine contradiction, a reconciliation by context, and a documented extraction failure.*

---

## Approach

### Architecture

The system is built as a five-phase pipeline, each phase independently testable and committed separately:

```
PDF Upload → Ingestion → LLM Extraction → Storage + Embeddings → Cross-Document Linking → API/UI
```

**1. Ingestion** (`backend/ingest.py`): Parses PDFs into page-anchored chunks using PyMuPDF. Detects tables via `page.find_tables()`, extracts them separately from prose text blocks, and merges adjacent text blocks that were split across columns into coherent paragraphs. Each chunk carries `{doc_id, page_number, text, chunk_type}`.

**2. Extraction** (`backend/extract.py`): Sends each chunk to Gemini with a structured output schema (Pydantic `ExtractedFact` model) to extract atomic facts. A heuristic pre-filter skips chunks unlikely to contain factual content (boilerplate, disclaimers, ToC entries). Every extracted fact is validated with evidence grounding — the `evidence_text` field must be an exact substring of the source chunk, or the fact is rejected. Includes rate-limit backoff with retry and batch checkpointing.

**3. Storage** (`backend/store.py`, `backend/db.py`, `backend/embeddings.py`): Facts are persisted in SQLite with vector embeddings (Gemini `gemini-embedding-001`, 3072 dimensions) stored as BLOB columns. Embeddings are computed from `entity + attribute` text, focusing retrieval on what the fact is about rather than incidental wording. Upserts use a composite unique key `(source_doc_id, page_number, entity, attribute, evidence_text)` instead of random UUIDs, making re-extraction idempotent — running the pipeline twice on the same document updates existing rows rather than creating duplicates. A document-level `as_of` inference module (`backend/as_of.py`) backfills temporal periods for facts extracted without explicit time markers, using regex-based period detection from cover pages and title text.

**4. Cross-Document Linking** (`backend/link.py`): For each fact, retrieves nearest-neighbor candidates from other documents using cosine similarity on the stored embeddings. Candidate pairs pass through a domain pre-filter (corporate entities aren't compared against macroeconomic entities) and a unit-mismatch multiplier heuristic. Surviving pairs are sent to Gemini for LLM-judge classification with structured reasoning output — the model must produce one of `corroborates`, `contradicts`, or `reconciled_by_context` with a natural-language explanation referencing specific differences in period, scope, or units. Relationships are stored with canonical order-independent deduplication (fact pairs are sorted by ID before insertion) and an evaluated-pairs cache to avoid re-judging known pairs.

**5. API and UI** (`backend/api.py`, `frontend/`): A FastAPI application serves five endpoints (`GET /stats`, `GET /facts`, `GET /facts/{id}`, `GET /relationships`, `POST /upload`) and mounts a static frontend. The frontend is a single-page application (vanilla JS, no framework) with an editorial/analyst design aesthetic — light background, white card surfaces, Inter typography, restrained color coding (green for corroborated, amber for reconciled, red for contradicts). The upload endpoint runs the full pipeline synchronously (ingest → extract → store → link) and returns results including fact count, relationship count, and any chunk-level errors.

### Key Engineering Decisions

- **Composite upsert key over UUID**: Facts are keyed on `(source_doc_id, page_number, entity, attribute, evidence_text)`. This makes the entire pipeline re-runnable without duplicates, and means two independent extraction runs on the same PDF converge to the same database state.

- **Embedding-then-LLM-judge for relationships**: Rather than using pure embedding similarity thresholds (which conflate semantic closeness with factual agreement) or sending all pairs to the LLM (which is expensive), the system uses embeddings as a cheap recall stage and the LLM as a precision stage. The LLM must produce explicit reasoning referencing the specific difference (period, scope, unit) that justifies the classification.

- **SQLite with in-process vector search**: Chosen for zero-dependency deployment. Vector search is brute-force cosine similarity over in-memory numpy arrays loaded from BLOB columns — adequate for the scale of this prototype (hundreds of facts, not millions). No external vector database required.

- **Synchronous upload processing**: The `POST /upload` endpoint blocks until the full pipeline completes (typically 1–2 minutes depending on document size and API latency). This simplifies the architecture — no job queue, no polling, no WebSocket progress updates. The tradeoff is a long HTTP request, which the frontend handles with a processing spinner and status messaging.

- **Evidence grounding validation**: Every extracted fact's `evidence_text` is checked as an exact substring of the source chunk. Facts that fail this check are silently dropped. This catches LLM hallucinations where the model invents plausible-sounding evidence that doesn't actually appear in the document.

### AI Tools Used

- **Antigravity (Google)**: Used for the entire build — architecture planning, implementation, debugging, and verification. Every implementation plan was reviewed before execution and every phase completion verified against real output (live API calls, database queries, Playwright DOM inspection) before proceeding to the next phase.
- **Gemini**: Used at runtime for three distinct purposes: (1) structured fact extraction from document chunks via `gemini-2.5-flash`, (2) vector embedding generation via `gemini-embedding-001` for cross-document retrieval, and (3) LLM-judge relationship classification with structured reasoning output.

---

## Limitations and Next Steps

### Current Limitations

- **No OCR support for scanned PDFs**: The ingestion module relies on PyMuPDF's digital text extraction. Scanned or image-only PDFs with no embedded text layer will produce zero chunks and zero facts. OCR integration (e.g., Tesseract) is not implemented.

- **Extraction is page-specific, not full-document**: The starter documents are multi-page PDFs but the assignment provided page-specific excerpts. Facts are extracted only from the specified excerpt pages — for example, the Delhivery prospectus extracts from pages 1 and 50, not all 200+ pages. The database currently contains 106 facts across the 6 starter documents and 32 cross-document relationships (5 corroborates, 23 reconciled by context, 4 contradicts). Coverage within each document is partial by design for prototype scope.

- **No PDF viewer or evidence highlighting**: The UI shows evidence text as quoted strings and cites the source document and page number, but does not render the original PDF page or highlight the evidence span within it.

- **No review/approval workflow**: Extracted facts go directly into the database. There is no human-in-the-loop stage where an analyst can accept, reject, or modify facts before they enter the knowledge base.

- **Entity naming inconsistency**: The LLM extraction sometimes produces variant entity names for the same real-world entity (e.g., "Delhivery" vs "Delhivery Limited" from live upload testing). There is no entity resolution or canonicalization layer — each variant is stored as a distinct entity. A normalization pass (fuzzy matching or alias table) would improve cross-document recall.

- **Brute-force vector search**: Embedding similarity is computed via full numpy scan over all stored vectors. This is adequate at ~100 facts but would not scale to production volumes without an approximate nearest-neighbor index (FAISS, pgvector, etc.).

### Next Steps

- Entity canonicalization layer (alias table or fuzzy merge)
- PDF.js viewer with evidence highlighting
- Fact review/approval workflow with accept/reject/edit states
- Async upload processing with progress WebSocket
- Incremental relationship updates (re-link only new facts, not full re-scan)
- OCR integration for scanned documents

---

## Additional Notes

### Documented Extraction and Reasoning Failures

Two real failures were found during development and documented as part of the four required cases:

**1. `as_of` temporal hallucination**: The LLM extraction model invented fiscal quarter labels (Q1 FY24, Q2 FY24, etc.) for a series of numeric values in the Delhivery Q4 FY24 earnings presentation page 8, where the source content was a simple grid of numbers with no column headers or period labels. The model fabricated temporal markers that seemed plausible given the document context but were not grounded in the actual text.

*Fix applied*: Three-layer defense — (a) prompt hardening with explicit instructions to set `as_of` to null when period labels are absent, (b) programmatic evidence-grounding validation that checks whether the claimed `as_of` value appears in the source chunk text, and (c) a document-level `as_of` inference module that backfills null periods only from unambiguous cover-page/title patterns (regex-based, not LLM-based), preserving null when ambiguous rather than guessing.

**2. Infographic column-merge bug (2.8Bn workforce)**: The Delhivery annual report excerpt page 2 uses a visual infographic layout with large display numbers. The ingestion layer's column-detection heuristic merged the number "2.8Bn" (which referred to shipments handled) with the adjacent label "Workforce strength", producing the chunk text "2.8Bn Workforce strength". The LLM faithfully extracted this as `entity: Delhivery, attribute: workforce strength, value: 2.8Bn` — a correct extraction of incorrect input. This was then flagged as a contradiction against the Q4 earnings deck's team-size figures (63,713 / 63,144 / 60,373 / 57,307), which is the system correctly identifying that 2.8 billion employees is impossible for a logistics company.

*Status*: Documented as a genuine ingestion layout bug — the column-midpoint merge heuristic fails on infographic layouts where adjacent display numbers and labels occupy different visual columns but should be read independently. The contradiction is surfaced in the UI with a "Flagged — requires verification" badge.

### Build Process

The system was built incrementally across 39 commits over a single session, following the phased architecture in `PROJECT.md`. Every phase was planned, reviewed, and verified against real output before moving to the next. The upload pipeline was also tested against unrelated third-party documents (not part of the starter dataset) to verify generalization — the system processes arbitrary PDFs without document-specific rules or hardcoded entity lists.
