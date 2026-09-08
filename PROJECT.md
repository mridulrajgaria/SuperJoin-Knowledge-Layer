# Fact Knowledge Layer — Project Doc

## Goal

Build a system that, given any PDF (not just the starter set), extracts
meaningful facts, grounds every fact in a specific page/quote in its source
document, and identifies when facts across documents corroborate, contradict,
or are reconciled by context (time, scope, units). Provide an API or UI to
upload PDFs and inspect results.

Must generalize: no hardcoded filenames, facts, or document-specific rules.

## Fact Schema (dynamic — do not hardcode fields per document type)

```
Fact {
  id
  entity          # e.g. "Delhivery", "India CPI inflation"
  attribute       # e.g. "consolidated revenue", "GDP growth rate"
  value
  unit            # e.g. "INR crore", "%", null if not numeric
  as_of           # date or period the fact refers to, e.g. "FY24", "Q4 FY24"
  scope           # e.g. "consolidated", "standalone", "India", "urban"
  source_doc_id
  page_number
  evidence_text   # exact quote/snippet supporting the fact
  confidence      # extraction confidence, 0-1
  extra           # open bag for attributes we didn't anticipate — schema
                   # should be able to grow without a migration
}

Relationship {
  id
  fact_a_id
  fact_b_id
  type            # corroborates | contradicts | reconciled_by_context
  reasoning       # LLM-generated explanation, referencing the specific
                   # difference (period/scope/unit) when relevant
  confidence
}
```

## Architecture (phased — build and verify in this order)

1. **Ingestion**: PDF -> page-anchored chunks `{doc_id, page_number, text,
   chunk_type}`. No LLM calls yet.
2. **Extraction**: chunks -> facts via LLM structured output against the
   schema above. Store as JSON first, inspect manually before moving on.
3. **Storage**: persist facts (SQLite/Postgres) + embeddings (entity +
   attribute + value) for retrieval.
4. **Cross-document linking**: for each new fact, retrieve nearest-neighbor
   facts from other documents, then LLM-judge the relationship with explicit
   reasoning. This is the core of the assignment — don't shortcut it into a
   pure similarity threshold.
5. **API/UI**: upload endpoint, fact browser, relationship view showing both
   facts side by side with their evidence and the reasoning.

## Datasets in `data/samples/`

- `delhivery/` — 2022 IPO prospectus, FY24 annual report, Q4 FY24 earnings
  deck. Same company, different formats/dates — good source for corroborated
  facts and things that changed over time.
- `india-macroeconomy/` — Economic Survey 2024-25, RBI Annual Report 2024-25,
  IMF Article IV 2025. Same economy, different institutions/vintages — good
  source for contradictions and context-reconciled facts (fiscal vs calendar
  year, provisional vs revised data).

## Four Required Cases (target, not guaranteed until extraction is run)

1. Corroborated fact — same fact worded differently across two documents.
2. Genuine contradiction — diverging figures for the same entity/attribute
   that aren't explained by period/scope/unit.
3. Apparent contradiction reconciled by context — differs due to time, scope,
   or units.
4. A real extraction/reasoning failure, documented honestly, with what was
   done or would be done about it.

## Git Policy

- Commit after each logical unit of work, not at the end of a session.
- Conventional commit style: feat / fix / refactor / test / docs / chore.
- One coherent change per commit — don't bundle ingestion + extraction +
  storage into a single commit.
- Never commit secrets or `.env` files.
- Bug fixes found while testing on real PDFs get their own commit
  (`fix: ...`), not silently folded into the feature commit.

## Non-goals for this prototype

- No production polish, auth, or scaling work unless time permits as a
  brownie-point extension.
- A graph visualization alone is not the deliverable — the extraction and
  reasoning behind each relationship must be visible and explained.
