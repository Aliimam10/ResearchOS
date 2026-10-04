"""Reproducible retrieval and grounded-answer evaluation for ResearchOS."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from .config import EVALUATION_DIR, REPORT_DIR
from .embeddings import HashingEncoder, TextEncoder
from .models import Chunk, DocumentRecord
from .retrieval import HybridRetriever
from .storage import CorpusStore
from .tools import ResearchRetriever
from .workflow import ResearchWorkflow


@dataclass(slots=True)
class BenchmarkQuestion:
    id: str
    question: str
    relevant_chunk_ids: set[str]


def load_benchmark(path: Path = EVALUATION_DIR / "benchmark.json") -> list[BenchmarkQuestion]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        BenchmarkQuestion(id=item["id"], question=item["question"], relevant_chunk_ids=set(item["relevant_chunk_ids"]))
        for item in payload["questions"]
    ]


def load_fixed_corpus(store: CorpusStore, path: Path = EVALUATION_DIR / "fixed_corpus.json") -> None:
    """Load the small hand-authored corpus into an isolated evaluation store."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    store.clear()
    chunks_by_document: dict[str, list[Chunk]] = {}
    for raw_chunk in payload["chunks"]:
        chunk = Chunk.from_dict(raw_chunk)
        chunks_by_document.setdefault(chunk.document_id, []).append(chunk)
    for raw_document in payload["documents"]:
        document = DocumentRecord.from_dict(raw_document)
        store.save_document(document, chunks_by_document.get(document.id, []))


def _metric_row(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> tuple[float, float, float]:
    top = retrieved_ids[:k]
    recall = float(bool(set(top) & relevant_ids))
    first_rank = next((rank for rank, chunk_id in enumerate(retrieved_ids[:k], start=1) if chunk_id in relevant_ids), None)
    reciprocal_rank = 1 / first_rank if first_rank else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank, chunk_id in enumerate(top, start=1) if chunk_id in relevant_ids)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant_ids)) + 1))
    return recall, reciprocal_rank, (dcg / ideal if ideal else 0.0)


def evaluate_retrieval(
    retriever: HybridRetriever, questions: Iterable[BenchmarkQuestion], *, k: int = 5
) -> dict[str, dict[str, float]]:
    """Compare semantic-only, BM25-only, and hybrid RRF retrieval over one corpus."""
    question_list = list(questions)
    report: dict[str, dict[str, float]] = {}
    for mode in ("semantic", "bm25", "hybrid"):
        scores: list[tuple[float, float, float]] = []
        latencies: list[float] = []
        for question in question_list:
            started = time.perf_counter()
            hits = retriever.search(question.question, limit=k, mode=mode)
            latencies.append((time.perf_counter() - started) * 1_000)
            scores.append(_metric_row([hit.chunk.id for hit in hits], question.relevant_chunk_ids, k))
        report[mode] = {
            f"recall_at_{k}": round(mean(row[0] for row in scores), 4),
            f"mrr_at_{k}": round(mean(row[1] for row in scores), 4),
            f"ndcg_at_{k}": round(mean(row[2] for row in scores), 4),
            "mean_latency_ms": round(mean(latencies), 3),
            "p50_latency_ms": round(median(latencies), 3),
            "questions": len(question_list),
        }
    return report


def evaluate_answers(retriever: HybridRetriever, questions: Iterable[BenchmarkQuestion]) -> dict[str, float | int]:
    """Measure citations and verifier outputs against manual relevance labels."""
    question_list = list(questions)
    workflow = ResearchWorkflow(ResearchRetriever(retriever))
    citation_total = citation_correct = claims_total = claims_supported = 0
    latencies: list[float] = []
    for question in question_list:
        started = time.perf_counter()
        result = workflow.ask(question.question)
        latencies.append((time.perf_counter() - started) * 1_000)
        citation_total += len(result.citations)
        citation_correct += sum(citation["chunk_id"] in question.relevant_chunk_ids for citation in result.citations)
        claims_total += len(result.verified_claims)
        claims_supported += sum(claim["supported"] for claim in result.verified_claims)
    return {
        "questions": len(question_list),
        "citation_correctness": round(citation_correct / citation_total, 4) if citation_total else 0.0,
        "claim_groundedness": round(claims_supported / claims_total, 4) if claims_total else 0.0,
        "unsupported_claim_rate": round(1 - claims_supported / claims_total, 4) if claims_total else 0.0,
        "mean_latency_ms": round(mean(latencies), 3),
        "p50_latency_ms": round(median(latencies), 3),
    }


def run_fixed_benchmark(
    *,
    working_dir: Path = REPORT_DIR / "evaluation_corpus",
    report_path: Path = REPORT_DIR / "evaluation_report.json",
    encoder: TextEncoder | None = None,
) -> dict[str, Any]:
    """Run the checked-in 50-question benchmark without touching a user's corpus."""
    store = CorpusStore(working_dir)
    load_fixed_corpus(store)
    questions = load_benchmark()
    selected_encoder = encoder or HashingEncoder()
    retriever = HybridRetriever(store, encoder=selected_encoder)
    report = {
        "benchmark": "ResearchOS fixed evaluation corpus",
        "question_count": len(questions),
        "embedding_backend": selected_encoder.name,
        "notes": [
            "Relevance labels are manually checked chunk identifiers in evaluation/benchmark.json.",
            "Groundedness checks whether displayed claims link to retrieved evidence; it does not prove real-world correctness.",
        ],
        "retrieval": evaluate_retrieval(retriever, questions),
        "answer_generation": evaluate_answers(retriever, questions),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
