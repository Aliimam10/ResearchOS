"""Local semantic encoders with a dependable no-download fallback.

ResearchOS prefers sentence-transformers when installed, but a hashing encoder
keeps the portfolio project runnable offline and makes tests deterministic. Both
encoders expose the same normalised-vector interface to the retrieval layer.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,}")


class TextEncoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class HashingEncoder:
    """A deterministic vector baseline, not a substitute for learned embeddings.

    It uses signed token and token-bigram hashing. Its role is to make the
    project usable without a model download; the UI/API exposes its name so it
    cannot be mistaken for a sentence-transformer model.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.name = f"hashing-{dimensions}d-fallback"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            terms = [token.lower() for token in _TOKEN_RE.findall(text)]
            features = terms + [f"{left}::{right}" for left, right in zip(terms, terms[1:])]
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                index = value % self.dimensions
                vectors[row, index] += 1.0 if (value >> 63) else -1.0
        return _normalise(vectors)


class SentenceTransformerEncoder:
    """Adapter around sentence-transformers, loaded lazily to avoid import cost."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional installation
            raise RuntimeError("sentence-transformers is not installed") from exc
        self.model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers/{model_name}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def default_encoder() -> TextEncoder:
    """Prefer an actual embedding model while failing closed to a labelled fallback.

    Set ``RESEARCHOS_EMBEDDING_BACKEND=hashing`` for an entirely offline run, or
    ``RESEARCHOS_EMBEDDING_MODEL`` to select another sentence-transformer model.
    """
    backend = os.getenv("RESEARCHOS_EMBEDDING_BACKEND", "sentence-transformers").lower()
    if backend == "hashing":
        return HashingEncoder()
    try:
        return SentenceTransformerEncoder(os.getenv("RESEARCHOS_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    except Exception:
        # The project must still open when an offline user has not downloaded a
        # model. The response metadata tells them that this fallback was used.
        return HashingEncoder()
