"""Hybrid retrieval: semantic vectors + BM25 + metadata filters + RRF + reranking."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from .embeddings import TextEncoder, default_encoder
from .models import Chunk
from .storage import CorpusStore

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on",
    "or", "that", "the", "to", "was", "were", "what", "which", "with",
}


def tokenize(text: str) -> list[str]:
    """A small shared tokeniser is adequate for transparent, local BM25."""
    import re

    return [word.lower() for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", text) if word.lower() not in _STOP_WORDS]


@dataclass(slots=True)
class RetrievalFilters:
    year_from: int | None = None
    year_to: int | None = None
    document_ids: set[str] | None = None
    source_types: set[Literal["full_text", "abstract"]] | None = None


@dataclass(slots=True)
class RetrievalHit:
    """A ranked passage plus its retrieval provenance and source citation data."""

    chunk: Chunk
    score: float
    semantic_score: float | None = None
    bm25_score: float | None = None
    semantic_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self.chunk)
        payload.update(
            {
                "score": round(self.score, 5),
                "semantic_score": round(self.semantic_score, 5) if self.semantic_score is not None else None,
                "bm25_score": round(self.bm25_score, 5) if self.bm25_score is not None else None,
                "semantic_rank": self.semantic_rank,
                "bm25_rank": self.bm25_rank,
                "rrf_score": round(self.rrf_score, 5) if self.rrf_score is not None else None,
                "rerank_score": round(self.rerank_score, 5) if self.rerank_score is not None else None,
                "matched_terms": self.matched_terms,
            }
        )
        return payload


class BM25Index:
    """A compact BM25 implementation; no opaque search server is needed locally."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.documents = [tokenize(text) for text in texts]
        self.lengths = [len(document) for document in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.term_frequencies = [Counter(document) for document in self.documents]
        document_frequency: dict[str, int] = defaultdict(int)
        for document in self.documents:
            for term in set(document):
                document_frequency[term] += 1
        total = len(self.documents)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score(self, query: str) -> np.ndarray:
        terms = tokenize(query)
        scores = np.zeros(len(self.documents), dtype=np.float32)
        if not terms or not self.documents:
            return scores
        for row, frequencies in enumerate(self.term_frequencies):
            denominator_normaliser = self.k1 * (1 - self.b + self.b * self.lengths[row] / max(self.average_length, 1))
            for term in terms:
                frequency = frequencies.get(term, 0)
                if frequency:
                    scores[row] += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / (
                        frequency + denominator_normaliser
                    )
        return scores


class HybridRetriever:
    """Read the small local corpus and perform transparent multi-signal ranking.

    The index is deliberately in-process: rebuilding a few hundred paper chunks
    is fast, keeps the project infrastructure-free, and makes scoring/evaluation
    reproducible. A larger corpus could swap the semantic matrix for FAISS.
    """

    RRF_K = 60

    def __init__(self, store: CorpusStore, encoder: TextEncoder | None = None) -> None:
        self.store = store
        self.encoder = encoder or default_encoder()
        self._fingerprint: tuple[str, ...] = ()
        self._chunks: list[Chunk] = []
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._bm25 = BM25Index([])

    @property
    def embedding_model(self) -> str:
        return self.encoder.name

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        filters: RetrievalFilters | None = None,
        mode: Literal["semantic", "bm25", "hybrid"] = "hybrid",
    ) -> list[RetrievalHit]:
        """Return citable chunks ranked by one retrieval mode or fused hybrid search."""
        if not query.strip() or limit < 1:
            return []
        self._ensure_index()
        eligible = self._eligible_indices(filters or RetrievalFilters())
        if not eligible:
            return []
        query_vector = self.encoder.encode([query])
        semantic_scores = self._vectors @ query_vector[0] if self._vectors.size else np.zeros(len(self._chunks))
        bm25_scores = self._bm25.score(query)
        return self._rank(query, eligible, semantic_scores, bm25_scores, limit, mode)

    def _ensure_index(self) -> None:
        chunks = self.store.chunks()
        fingerprint = tuple(chunk.id for chunk in chunks)
        if fingerprint == self._fingerprint:
            return
        self._chunks = chunks
        self._fingerprint = fingerprint
        texts = [f"{chunk.title}\n{chunk.section or ''}\n{chunk.text}" for chunk in chunks]
        self._vectors = self.encoder.encode(texts) if texts else np.empty((0, 0), dtype=np.float32)
        self._bm25 = BM25Index(texts)

    def _eligible_indices(self, filters: RetrievalFilters) -> list[int]:
        indices: list[int] = []
        for index, chunk in enumerate(self._chunks):
            if filters.year_from is not None and (chunk.year is None or chunk.year < filters.year_from):
                continue
            if filters.year_to is not None and (chunk.year is None or chunk.year > filters.year_to):
                continue
            if filters.document_ids is not None and chunk.document_id not in filters.document_ids:
                continue
            if filters.source_types is not None and chunk.content_kind not in filters.source_types:
                continue
            indices.append(index)
        return indices

    def _rank(
        self,
        query: str,
        eligible: list[int],
        semantic: np.ndarray,
        bm25: np.ndarray,
        limit: int,
        mode: Literal["semantic", "bm25", "hybrid"],
    ) -> list[RetrievalHit]:
        semantic_order = sorted(eligible, key=lambda index: float(semantic[index]), reverse=True)
        bm25_order = sorted(eligible, key=lambda index: float(bm25[index]), reverse=True)
        semantic_ranks = {index: rank for rank, index in enumerate(semantic_order, start=1)}
        bm25_ranks = {index: rank for rank, index in enumerate(bm25_order, start=1)}
        query_terms = set(tokenize(query))

        if mode == "semantic":
            candidates = semantic_order[: max(limit, 20)]
            score_lookup = {index: float(semantic[index]) for index in candidates}
        elif mode == "bm25":
            candidates = [index for index in bm25_order if bm25[index] > 0][: max(limit, 20)]
            score_lookup = {index: float(bm25[index]) for index in candidates}
        else:
            # RRF balances scales that are otherwise incomparable (cosine vs BM25).
            candidate_set = set(semantic_order[:40]) | set(bm25_order[:40])
            rrf = {
                index: 1 / (self.RRF_K + semantic_ranks[index]) + 1 / (self.RRF_K + bm25_ranks[index])
                for index in candidate_set
            }
            candidates = sorted(candidate_set, key=lambda index: rrf[index], reverse=True)[: max(limit * 4, 20)]
            score_lookup = rrf

        hits: list[RetrievalHit] = []
        maximum_rrf = max(score_lookup.values(), default=1.0)
        for index in candidates:
            chunk = self._chunks[index]
            chunk_terms = set(tokenize(f"{chunk.title} {chunk.section or ''} {chunk.text}"))
            matched = sorted(query_terms & chunk_terms)
            coverage = len(matched) / max(1, len(query_terms))
            phrase_bonus = 0.1 if query.lower() in chunk.text.lower() else 0.0
            title_terms = set(tokenize(f"{chunk.title} {chunk.section or ''}"))
            title_bonus = 0.12 * len(query_terms & title_terms) / max(1, len(query_terms))
            # This small deterministic reranker improves exact question-term
            # coverage without hiding the underlying semantic/BM25 scores.
            rerank = 0.68 * (score_lookup[index] / maximum_rrf) + 0.2 * coverage + title_bonus + phrase_bonus
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=rerank,
                    semantic_score=float(semantic[index]),
                    bm25_score=float(bm25[index]),
                    semantic_rank=semantic_ranks[index],
                    bm25_rank=bm25_ranks[index],
                    rrf_score=score_lookup[index] if mode == "hybrid" else None,
                    rerank_score=rerank,
                    matched_terms=matched,
                )
            )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]
