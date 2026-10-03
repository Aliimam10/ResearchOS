# ResearchOS

ResearchOS is a compact, local research assistant for analysing a collection of
papers and reports. Its purpose is to demonstrate an evidence-first RAG system
that is small enough to explain in an interview: it preserves citation metadata,
combines keyword and semantic retrieval, routes questions through a bounded
LangGraph workflow, uses deterministic tools for arithmetic, and measures both
retrieval and answer groundedness.

It is not a claim of enterprise search, factual-certainty guarantees, or
autonomous web browsing. In particular, the topic-discovery path uses the
[OpenAlex Works API](https://docs.openalex.org/api-entities/works) and downloads
only explicit open-access PDF links returned by that source. It never attempts
to bypass a publisher paywall.

## Current milestone: corpus ingestion

The first milestone implements the shared corpus foundation.

```text
PDF upload or OpenAlex topic discovery
                ↓
document record + source metadata
                ↓
page/section-aware citable chunks
                ↓
local JSON corpus (inspectable and portable)
```

Each chunk carries its document title, source identifier/URL, authors, year,
page number, section (when detectable), DOI, and content type. Papers without
full text still contribute an explicitly labelled `Abstract` passage; metadata
alone is retained but is never misrepresented as document evidence.

## Local setup

```bash
cd /Users/alismac/Ali/UCL/Projects/ResearchOS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Add your own PDFs
python -m researchos upload ~/Downloads/paper.pdf

# Retrieve up to 50 OpenAlex candidates, and OA PDFs where offered
python -m researchos discover "Machine learning approaches to gravitational-wave detection" --year-from 2020

python -m researchos status
pytest
```

Generated corpus JSON and downloaded PDFs stay under `data/` and are ignored by
Git. This makes it safe to experiment without committing research material.

## Planned milestones

1. **Scaffold, PDF ingestion, scholarly discovery** — this commit.
2. **Hybrid retrieval** — embeddings, BM25, metadata filters, RRF, and a small
   reranker.
3. **Research workflow** — LangGraph state orchestration plus bounded retriever,
   calculator, and paper-search tools.
4. **Groundedness and evaluation** — claim/evidence verification and a manually
   checked benchmark comparing semantic, BM25, and hybrid retrieval.
5. **Local product surface** — FastAPI, a focused evidence UI, API tests, and
   project documentation.

## Design decisions to defend

- The same `DocumentRecord` and `Chunk` schema serves uploaded PDFs and papers
  discovered through OpenAlex, avoiding two inconsistent retrieval paths.
- Chunks never cross pages, so a page citation is never fabricated. Section
  labels are best-effort because PDF structure is inconsistent.
- JSON storage is intentional for a small portfolio corpus: it is transparent,
  versionable in schema, and has no infrastructure dependency. The retrieval
  index will be rebuilt locally from it.
- Full text is optional and clearly marked. Abstract-only evidence is useful for
  discovery questions but should yield lower confidence in later verification.
# ResearchOS
