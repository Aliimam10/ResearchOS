"""FastAPI adapter for ResearchOS's local evidence-first research pipeline."""
from __future__ import annotations

import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from researchos.config import REPORT_DIR  # noqa: E402
from researchos.evaluation import run_fixed_benchmark  # noqa: E402
from researchos.service import ResearchService  # noqa: E402


class DiscoveryRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=300)
    limit: int = Field(default=50, ge=1, le=100)
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    download_full_text: bool = True


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2_000)
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)
    document_ids: list[str] | None = None


@lru_cache
def get_service() -> ResearchService:
    return ResearchService()


server = FastAPI(title="ResearchOS", version="0.1.0", description="Local evidence-backed research assistant")


@server.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "researchos_ui.html").read_text(encoding="utf-8")


@server.get("/api/corpus")
def corpus(service: ResearchService = Depends(get_service)) -> dict:
    return {"summary": service.corpus_summary(), "documents": [document.to_dict() for document in service.documents()]}


@server.post("/api/upload")
async def upload(
    files: list[UploadFile] = File(...), service: ResearchService = Depends(get_service)
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one PDF.")
    results = []
    for file in files:
        filename = file.filename or "upload.pdf"
        try:
            result = service.ingest_upload(filename, await file.read())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        results.append(
            {"document": result.document.to_dict(), "chunks_created": result.chunks_created, "warnings": result.warnings}
        )
    return {"uploaded": results, "summary": service.corpus_summary()}


@server.post("/api/discover")
def discover(request: DiscoveryRequest, service: ResearchService = Depends(get_service)) -> dict:
    try:
        result = service.discover_topic(
            request.topic,
            limit=request.limit,
            year_from=request.year_from,
            download_full_text=request.download_full_text,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"result": asdict(result), "summary": service.corpus_summary()}


@server.post("/api/ask")
def ask(request: AskRequest, service: ResearchService = Depends(get_service)) -> dict:
    if request.year_from and request.year_to and request.year_from > request.year_to:
        raise HTTPException(status_code=400, detail="The start year must not be after the end year.")
    try:
        result = service.ask(
            request.question,
            filters={
                "year_from": request.year_from,
                "year_to": request.year_to,
                "document_ids": request.document_ids,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return asdict(result)


@server.get("/api/evaluation")
def read_evaluation() -> dict:
    report_path = REPORT_DIR / "evaluation_report.json"
    if not report_path.exists():
        return {"available": False, "message": "Run the fixed benchmark to create an evaluation report."}
    import json

    return {"available": True, "report": json.loads(report_path.read_text(encoding="utf-8"))}


@server.post("/api/evaluation/run")
def run_evaluation() -> dict:
    return {"available": True, "report": run_fixed_benchmark()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(server, host="127.0.0.1", port=7860)
