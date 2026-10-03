"""Application paths and small, explicit configuration values."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("RESEARCHOS_DATA_DIR", ROOT / "data"))
CORPUS_DIR = DATA_DIR / "corpus"
DOWNLOAD_DIR = DATA_DIR / "downloads"
REPORT_DIR = DATA_DIR / "reports"

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_DISCOVERY_LIMIT = 50
MAX_DISCOVERY_LIMIT = 100
MAX_DOWNLOADS_PER_RUN = 20
REQUEST_TIMEOUT_SECONDS = 25

# Keeping chunks below a page-length preserves readable citations and makes
# the later retrieval/evaluation stages fast enough for a local portfolio app.
CHUNK_TARGET_CHARS = 1_200
CHUNK_OVERLAP_CHARS = 180
