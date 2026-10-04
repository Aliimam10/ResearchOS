"""Structured citations derived only from retrieved passages."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Citation:
    """A human-readable citation plus the exact chunk that supports it."""

    id: int
    chunk_id: str
    document_id: str
    title: str
    location: str
    source_url: str | None
    doi: str | None
    content_kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def inline(self) -> str:
        return f"[{self.id}]"

    def reference(self) -> str:
        detail = f" — {self.location}"
        if self.doi:
            detail += f" · DOI: {self.doi}"
        elif self.source_url:
            detail += f" · {self.source_url}"
        return f"[{self.id}] {self.title}{detail}"


def citation_from_evidence(item: dict[str, Any], number: int) -> Citation:
    """Create a citation without losing the retrieved chunk identity."""
    location = f"p. {item['page']}" if item.get("page") else "Abstract"
    if item.get("section") and item["section"] != "Abstract":
        location = f"{location}, {item['section']}"
    return Citation(
        id=number,
        chunk_id=item["id"],
        document_id=item["document_id"],
        title=item["title"],
        location=location,
        source_url=item.get("source_url"),
        doi=item.get("doi"),
        content_kind=item["content_kind"],
    )
