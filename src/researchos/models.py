"""Small serialisable records passed between ResearchOS pipeline stages."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ContentStatus = Literal["full_text", "abstract_only", "metadata_only", "failed"]


@dataclass(slots=True)
class DocumentRecord:
    """One source document, whether uploaded or found through discovery."""

    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    doi: str | None = None
    source_url: str | None = None
    open_access_url: str | None = None
    source_type: Literal["upload", "openalex"] = "upload"
    content_status: ContentStatus = "metadata_only"
    page_count: int = 0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentRecord":
        return cls(**payload)


@dataclass(slots=True)
class Chunk:
    """A citable passage. Metadata is copied here so retrieval is traceable."""

    id: str
    document_id: str
    text: str
    title: str
    page: int | None = None
    section: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    source_url: str | None = None
    content_kind: Literal["full_text", "abstract"] = "full_text"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Chunk":
        return cls(**payload)


@dataclass(slots=True)
class DiscoveredPaper:
    """The normalized subset of OpenAlex metadata ResearchOS needs."""

    openalex_id: str
    title: str
    authors: list[str]
    year: int | None
    abstract: str | None
    doi: str | None
    source_url: str | None
    open_access_url: str | None
    cited_by_count: int = 0
    relevance_hint: float = 0.0


@dataclass(slots=True)
class IngestResult:
    document: DocumentRecord
    chunks_created: int
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DiscoveryResult:
    topic: str
    candidates_found: int
    documents_added: int
    full_text_downloads: int
    abstract_only: int
    skipped_duplicates: int
    warnings: list[str] = field(default_factory=list)
