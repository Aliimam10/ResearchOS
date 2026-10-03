"""OpenAlex-backed scholarly discovery with an open-access-only full-text path."""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from .config import (
    DEFAULT_DISCOVERY_LIMIT,
    MAX_DISCOVERY_LIMIT,
    MAX_DOWNLOADS_PER_RUN,
    OPENALEX_WORKS_URL,
    REQUEST_TIMEOUT_SECONDS,
)
from .ingestion import PDFIngestor, abstract_chunk
from .models import DiscoveredPaper, DiscoveryResult, DocumentRecord
from .storage import CorpusStore


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _unpack_inverted_index(index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex represents abstracts as a word-to-position inverted index."""
    if not index:
        return None
    positions = [(position, word) for word, indexes in index.items() for position in indexes]
    return " ".join(word for _, word in sorted(positions)) or None


def _authors(record: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for authorship in record.get("authorships") or []:
        author = authorship.get("author") or {}
        if name := author.get("display_name"):
            names.append(str(name))
    return names


def _oa_pdf_url(record: dict[str, Any]) -> str | None:
    """Select only explicit provider-hosted OA PDF links; never attempt a paywall."""
    oa = record.get("open_access") or {}
    locations = [record.get("best_oa_location"), record.get("primary_location")]
    locations.extend(record.get("locations") or [])
    for location in locations:
        if not location:
            continue
        url = location.get("pdf_url")
        if url and (oa.get("is_oa") or location.get("is_oa")):
            return str(url)
    return None


class OpenAlexClient:
    """Small, testable client for the OpenAlex Works endpoint."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "ResearchOS/0.1 (local portfolio project)")

    def search(self, topic: str, limit: int = DEFAULT_DISCOVERY_LIMIT, year_from: int | None = None) -> list[DiscoveredPaper]:
        if not topic.strip():
            raise ValueError("A research topic is required.")
        limit = max(1, min(int(limit), MAX_DISCOVERY_LIMIT))
        params: dict[str, Any] = {"search": topic.strip(), "per-page": limit}
        if year_from:
            params["filter"] = f"from_publication_date:{year_from}-01-01"
        try:
            response = self.session.get(OPENALEX_WORKS_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError("OpenAlex discovery failed. Check your connection and try again.") from exc

        payload = response.json()
        papers: list[DiscoveredPaper] = []
        seen: set[str] = set()
        for row in payload.get("results", []):
            title = str(row.get("display_name") or "Untitled work").strip()
            doi = row.get("doi")
            key = (str(doi).lower() if doi else title.lower())
            if not title or key in seen:
                continue
            seen.add(key)
            primary_location = row.get("primary_location") or {}
            papers.append(
                DiscoveredPaper(
                    openalex_id=str(row.get("id") or f"openalex:{len(papers)}"),
                    title=title,
                    authors=_authors(row),
                    year=row.get("publication_year"),
                    abstract=_unpack_inverted_index(row.get("abstract_inverted_index")),
                    doi=str(doi) if doi else None,
                    source_url=(primary_location.get("landing_page_url") or row.get("doi") or row.get("id")),
                    open_access_url=_oa_pdf_url(row),
                    cited_by_count=int(row.get("cited_by_count") or 0),
                    relevance_hint=float(row.get("relevance_score") or 0.0),
                )
            )
        return papers

    def download_open_access_pdf(self, url: str) -> bytes:
        """Fetch an OpenAlex-supplied OA PDF and reject anything that is not a PDF.

        This method deliberately does not follow an article landing page, guess
        publisher URLs, log in, or otherwise try to obtain unavailable text.
        """
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Open-access link does not use HTTP(S).")
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS, allow_redirects=True)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError("Open-access PDF could not be downloaded.") from exc
        data = response.content
        content_type = response.headers.get("Content-Type", "").lower()
        if len(data) > 50_000_000:
            raise ValueError("Open-access PDF is larger than the 50 MB local ingestion limit.")
        if not data.startswith(b"%PDF") and "pdf" not in content_type:
            raise ValueError("Open-access link did not return a PDF.")
        return data


def paper_to_document(paper: DiscoveredPaper) -> DocumentRecord:
    digest = hashlib.sha256(paper.openalex_id.encode("utf-8")).hexdigest()[:16]
    return DocumentRecord(
        id=f"openalex_{digest}",
        title=paper.title,
        authors=paper.authors,
        year=paper.year,
        abstract=paper.abstract,
        doi=paper.doi,
        source_url=paper.source_url,
        open_access_url=paper.open_access_url,
        source_type="openalex",
        content_status="metadata_only",
        created_at=_now(),
    )


class TopicDiscovery:
    """Search papers, deduplicate them, and build local evidence when available."""

    def __init__(
        self, store: CorpusStore, ingestor: PDFIngestor, downloads_dir: Path, client: OpenAlexClient | None = None
    ) -> None:
        self.store = store
        self.ingestor = ingestor
        self.downloads_dir = downloads_dir
        self.client = client or OpenAlexClient()

    def discover(
        self,
        topic: str,
        *,
        limit: int = DEFAULT_DISCOVERY_LIMIT,
        year_from: int | None = None,
        download_full_text: bool = True,
    ) -> DiscoveryResult:
        papers = self.client.search(topic, limit=limit, year_from=year_from)
        added = full_text = abstract_only = duplicates = 0
        warnings: list[str] = []

        for paper in papers:
            document = paper_to_document(paper)
            if self.store.find_duplicate(document):
                duplicates += 1
                continue
            added += 1
            downloaded = False
            if download_full_text and paper.open_access_url and full_text < MAX_DOWNLOADS_PER_RUN:
                try:
                    data = self.client.download_open_access_pdf(paper.open_access_url)
                    self._cache_download(document.id, data)
                    result = self.ingestor.ingest_bytes(data, f"{document.id}.pdf", document=document)
                    downloaded = result.document.content_status == "full_text"
                    full_text += int(downloaded)
                    warnings.extend(result.warnings)
                except (RuntimeError, ValueError) as exc:
                    warnings.append(f"{document.title[:60]}: {exc}")
            if not downloaded:
                chunk = abstract_chunk(document)
                if chunk:
                    document.content_status = "abstract_only"
                    abstract_only += 1
                    self.store.save_document(document, [chunk])
                else:
                    document.content_status = "metadata_only"
                    self.store.save_document(document, [])

        return DiscoveryResult(
            topic=topic,
            candidates_found=len(papers),
            documents_added=added,
            full_text_downloads=full_text,
            abstract_only=abstract_only,
            skipped_duplicates=duplicates,
            warnings=warnings[:10],
        )

    def _cache_download(self, document_id: str, data: bytes) -> None:
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        (self.downloads_dir / f"{document_id}.pdf").write_bytes(data)
