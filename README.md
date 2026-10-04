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
                ↓
semantic vectors + BM25 → Reciprocal Rank Fusion → lightweight reranking
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

# Hybrid search preserves document/page/section citation metadata
python -m researchos search "Which methods improve detection accuracy?" --year-from 2020

# Ask a corpus question; the trace contains high-level tool actions only
python -m researchos ask "What is the average improvement reported?"

# Write a reproducible retrieval and groundedness report from the fixed corpus
python -m researchos evaluate

python -m researchos status
pytest
```

## Local interface

The interface is a plain HTML/CSS/JavaScript page served by FastAPI, so there is
no Node build step or hidden frontend state. Start it with:

```bash
uvicorn app:server --reload --port 7860
```

Open `http://127.0.0.1:7860`. The **Corpus** view uploads PDFs or creates a
topic corpus from OpenAlex; **Research** sends a scoped question and optional
year/document filters to the workflow; **Evaluation** runs the isolated fixed
benchmark and renders the stored report. The browser communicates only with the
following local endpoints: `GET /api/corpus`, `POST /api/upload`,
`POST /api/discover`, `POST /api/ask`, `GET /api/evaluation`, and
`POST /api/evaluation/run`.

Generated corpus JSON and downloaded PDFs stay under `data/` and are ignored by
Git. This makes it safe to experiment without committing research material.

## Planned milestones

1. **Scaffold, PDF ingestion, scholarly discovery** — this commit.
2. **Hybrid retrieval** — embeddings, BM25, metadata filters, RRF, and a small
   reranker.
3. **Research workflow** — LangGraph state orchestration plus bounded retriever,
   calculator, and paper-search tools. It makes at most two retrieval attempts,
   uses deterministic arithmetic only on retrieved values, and returns an
   explicit insufficient-evidence response where appropriate. Its trace is an
   operational summary, not hidden reasoning.
4. **Groundedness and evaluation** — a claim verifier checks that each displayed
   claim has a matching retrieved chunk and citation. Unsupported claims are
   removed. `evaluation/` contains a fixed 12-passage corpus and 50 manually
   checked question-to-chunk labels; `researchos evaluate` reports Recall@5,
   MRR@5, nDCG@5, citation correctness, claim groundedness, unsupported-claim
   rate, and latency for semantic, BM25, and hybrid retrieval.
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
  versionable in schema, and has no infrastructure dependency. The in-process
  NumPy/BM25 index is rebuilt when the local corpus changes. That is a sound
  trade-off for hundreds or low thousands of chunks; FAISS is a natural swap for
  a genuinely large local corpus.
- Hybrid retrieval uses cosine similarity from `sentence-transformers` when a
  model is available and transparently falls back to a labelled hashing-vector
  encoder when offline. It combines semantic and BM25 ranks with Reciprocal Rank
  Fusion, then applies a small deterministic term-coverage reranker. Scores are
  retrieval signals, not evidence-quality claims.
- Verification is deliberately narrow: it checks whether ResearchOS has linked
  displayed text to the cited retrieved passage. It does not guarantee that the
  source itself is correct, representative, or applicable beyond its context.
- Full text is optional and clearly marked. Abstract-only evidence is useful for
  discovery questions but should yield lower confidence in later verification.
# ResearchOS
