"""PDF parsing and section-aware chunk creation for uploaded/open-access papers."""
from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from .config import CHUNK_OVERLAP_CHARS, CHUNK_TARGET_CHARS
from .models import Chunk, DocumentRecord, IngestResult
from .storage import CorpusStore

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_NUMBERED_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+){0,3}\.?\s+)([A-Z].{2,120})$")
_ALL_CAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z \-–:]{3,100}$")
_KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "methods",
    "methodology",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "references",
    "appendix",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _stable_id(prefix: str, value: bytes | str) -> str:
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:16]}"


def _parse_authors(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in re.split(r"[;,]", raw) if part.strip()]


def _find_year(*values: str | None) -> int | None:
    for value in values:
        if value:
            match = _YEAR_RE.search(value)
            if match:
                return int(match.group(0))
    return None


def _is_heading(line: str) -> str | None:
    """Return a plausible section title without overclassifying body sentences."""
    normalized = _clean_text(line)
    if not normalized or len(normalized) > 120 or normalized.endswith((".", ";", ",")):
        return None
    lowered = normalized.rstrip(":").lower()
    if lowered in _KNOWN_HEADINGS:
        return normalized.rstrip(":").title()
    numbered = _NUMBERED_HEADING_RE.match(normalized)
    if numbered:
        return numbered.group(1).rstrip(":")
    if _ALL_CAPS_HEADING_RE.match(normalized) and len(normalized.split()) <= 12:
        return normalized.title()
    return None


def _split_with_overlap(text: str, target: int, overlap: int) -> list[str]:
    """Prefer sentence/whitespace boundaries while maintaining a small overlap."""
    text = _clean_text(text)
    if len(text) <= target:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + target)
        if end < len(text):
            boundaries = [text.rfind(marker, start + target // 2, end) for marker in (". ", "? ", "! ", " ")]
            boundary = max(boundaries)
            if boundary > start:
                end = boundary + 1
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


class PDFIngestor:
    """Extract a PDF locally. The only required external parser is PyMuPDF."""

    def __init__(self, store: CorpusStore) -> None:
        self.store = store

    def ingest_path(self, path: Path) -> IngestResult:
        return self.ingest_bytes(path.read_bytes(), filename=path.name)

    def ingest_bytes(
        self, data: bytes, filename: str, document: DocumentRecord | None = None
    ) -> IngestResult:
        """Parse local bytes, preserving a discovery record when one is supplied."""
        try:
            import fitz  # PyMuPDF; loaded lazily so metadata-only discovery still works.
        except ImportError as exc:  # pragma: no cover - dependency behaviour
            raise RuntimeError("PDF upload needs PyMuPDF. Install requirements.txt first.") from exc

        try:
            pdf = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"Could not parse {filename} as a PDF") from exc

        metadata = pdf.metadata or {}
        pages = [page.get_text("text") for page in pdf]
        generated = document or DocumentRecord(
            id=_stable_id("doc", data),
            title=_clean_text(metadata.get("title") or "") or Path(filename).stem,
            authors=_parse_authors(metadata.get("author")),
            year=_find_year(metadata.get("creationDate"), filename),
            source_type="upload",
            created_at=_now(),
        )
        if not generated.created_at:
            generated.created_at = _now()
        generated.page_count = len(pages)
        return self._save_parsed_pages(generated, pages)

    def ingest_text_pages(
        self, title: str, pages: Iterable[str], *, authors: list[str] | None = None, year: int | None = None
    ) -> IngestResult:
        """Ingest pre-extracted pages; useful for tests and non-PDF import adapters."""
        page_list = list(pages)
        document = DocumentRecord(
            id=_stable_id("doc", title + "\n" + "\n".join(page_list)),
            title=title,
            authors=authors or [],
            year=year,
            source_type="upload",
            created_at=_now(),
            page_count=len(page_list),
        )
        return self._save_parsed_pages(document, page_list)

    def _save_parsed_pages(self, document: DocumentRecord, pages: list[str]) -> IngestResult:
        chunks = section_aware_chunks(document, pages)
        warnings: list[str] = []
        if chunks:
            document.content_status = "full_text"
        else:
            document.content_status = "failed"
            warnings.append("No extractable text was found in this PDF; OCR is out of scope for this local build.")
        self.store.save_document(document, chunks)
        return IngestResult(document=document, chunks_created=len(chunks), warnings=warnings)


def section_aware_chunks(document: DocumentRecord, pages: Iterable[str]) -> list[Chunk]:
    """Chunk each page independently so citations never pretend a passage spans pages.

    A lightweight heading detector records likely sections; PDF structure is too
    variable to promise semantic section extraction, so unknown sections stay
    explicitly ``None`` rather than being invented.
    """
    chunks: list[Chunk] = []
    current_section: str | None = None
    for page_number, raw_page in enumerate(pages, start=1):
        lines = [line.strip() for line in raw_page.splitlines() if _clean_text(line)]
        body_parts: list[str] = []
        for line in lines:
            heading = _is_heading(line)
            if heading:
                current_section = heading
                continue
            body_parts.append(line)
        body = "\n".join(body_parts)
        for piece_index, piece in enumerate(_split_with_overlap(body, CHUNK_TARGET_CHARS, CHUNK_OVERLAP_CHARS)):
            chunks.append(
                Chunk(
                    id=_stable_id("chunk", f"{document.id}:{page_number}:{piece_index}:{piece}"),
                    document_id=document.id,
                    text=piece,
                    title=document.title,
                    page=page_number,
                    section=current_section,
                    authors=document.authors,
                    year=document.year,
                    doi=document.doi,
                    source_url=document.source_url,
                    content_kind="full_text",
                )
            )
    return chunks


def abstract_chunk(document: DocumentRecord) -> Chunk | None:
    """Turn metadata abstracts into explicitly labelled, citable fallback evidence."""
    if not document.abstract or not _clean_text(document.abstract):
        return None
    text = _clean_text(document.abstract)
    return Chunk(
        id=_stable_id("chunk", f"{document.id}:abstract:{text}"),
        document_id=document.id,
        text=text,
        title=document.title,
        section="Abstract",
        authors=document.authors,
        year=document.year,
        doi=document.doi,
        source_url=document.source_url,
        content_kind="abstract",
    )
