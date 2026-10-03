"""The single entry point shared by future API, CLI, and evaluation code."""
from __future__ import annotations

from pathlib import Path

from .config import CORPUS_DIR, DOWNLOAD_DIR
from .discovery import OpenAlexClient, TopicDiscovery
from .ingestion import PDFIngestor
from .models import DiscoveryResult, DocumentRecord, IngestResult
from .retrieval import HybridRetriever, RetrievalFilters, RetrievalHit
from .storage import CorpusStore


class ResearchService:
    """Own the local corpus and route both ingestion modes through it.

    This deliberately keeps upload and discovery from becoming parallel data
    paths: both produce the same DocumentRecord + citable Chunk records.
    """

    def __init__(self, corpus_dir: Path = CORPUS_DIR, downloads_dir: Path = DOWNLOAD_DIR) -> None:
        self.store = CorpusStore(corpus_dir)
        self.ingestor = PDFIngestor(self.store)
        self.downloads_dir = downloads_dir

    def ingest_upload(self, filename: str, data: bytes) -> IngestResult:
        if not filename.lower().endswith(".pdf"):
            raise ValueError("ResearchOS currently accepts PDF uploads only.")
        return self.ingestor.ingest_bytes(data, filename)

    def discover_topic(
        self,
        topic: str,
        *,
        limit: int = 50,
        year_from: int | None = None,
        download_full_text: bool = True,
        client: OpenAlexClient | None = None,
    ) -> DiscoveryResult:
        discovery = TopicDiscovery(self.store, self.ingestor, self.downloads_dir, client=client)
        return discovery.discover(
            topic, limit=limit, year_from=year_from, download_full_text=download_full_text
        )

    def documents(self) -> list[DocumentRecord]:
        return sorted(self.store.documents(), key=lambda doc: (doc.year or 0, doc.title), reverse=True)

    def corpus_summary(self) -> dict[str, int]:
        return self.store.counts()

    def search(
        self, question: str, *, filters: RetrievalFilters | None = None, limit: int = 8
    ) -> list[RetrievalHit]:
        """Use hybrid retrieval over the current corpus with all citation metadata intact."""
        return HybridRetriever(self.store).search(question, filters=filters, limit=limit)
