"""A deliberately small JSON-backed corpus store.

The project does not need a database: human-readable JSON makes document and
chunk provenance easy to inspect during an interview, and the corpus is small.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Iterable

from .models import Chunk, DocumentRecord


def _normalise_title(title: str) -> str:
    return re.sub(r"\W+", "", title.lower())


class CorpusStore:
    """Persist document records and their citable chunks with atomic writes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.documents_path = root / "documents.json"
        self.chunks_path = root / "chunks.json"
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read(path: Path) -> list[dict]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Corpus file is not valid JSON: {path}") from exc
        if not isinstance(data, list):
            raise RuntimeError(f"Corpus file must contain a JSON list: {path}")
        return data

    @staticmethod
    def _write(path: Path, value: list[dict]) -> None:
        """Replace a corpus file atomically, avoiding half-written indexes."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            temporary_path = Path(handle.name)
        temporary_path.replace(path)

    def documents(self) -> list[DocumentRecord]:
        return [DocumentRecord.from_dict(item) for item in self._read(self.documents_path)]

    def chunks(self) -> list[Chunk]:
        return [Chunk.from_dict(item) for item in self._read(self.chunks_path)]

    def document_by_id(self, document_id: str) -> DocumentRecord | None:
        return next((doc for doc in self.documents() if doc.id == document_id), None)

    def find_duplicate(self, document: DocumentRecord) -> DocumentRecord | None:
        """Deduplicate discovery results by DOI, canonical URL, then title/year."""
        target_title = _normalise_title(document.title)
        for existing in self.documents():
            if document.doi and existing.doi and document.doi.lower() == existing.doi.lower():
                return existing
            if document.source_url and document.source_url == existing.source_url:
                return existing
            if (
                target_title
                and target_title == _normalise_title(existing.title)
                and document.year
                and document.year == existing.year
            ):
                return existing
        return None

    def save_document(self, document: DocumentRecord, chunks: Iterable[Chunk]) -> None:
        """Upsert a document and replace only that document's passages."""
        documents = self.documents()
        old_index = next((index for index, item in enumerate(documents) if item.id == document.id), None)
        if old_index is None:
            documents.append(document)
        else:
            documents[old_index] = document

        replacement = list(chunks)
        all_chunks = [chunk for chunk in self.chunks() if chunk.document_id != document.id]
        all_chunks.extend(replacement)
        self._write(self.documents_path, [item.to_dict() for item in documents])
        self._write(self.chunks_path, [item.to_dict() for item in all_chunks])

    def counts(self) -> dict[str, int]:
        documents = self.documents()
        chunks = self.chunks()
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "full_text_documents": sum(doc.content_status == "full_text" for doc in documents),
            "abstract_only_documents": sum(doc.content_status == "abstract_only" for doc in documents),
        }

    def clear(self) -> None:
        """Clear the local corpus; intentionally exposed only through explicit API/UI actions later."""
        self._write(self.documents_path, [])
        self._write(self.chunks_path, [])
