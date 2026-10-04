import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1]))

from app import get_service, server
from researchos.models import DiscoveryResult, DocumentRecord, IngestResult
from researchos.workflow import WorkflowResult


class FakeService:
    def corpus_summary(self):
        return {"documents": 1, "chunks": 2, "full_text_documents": 1, "abstract_only_documents": 0}

    def documents(self):
        return [DocumentRecord(id="doc-1", title="Example paper", authors=["Ada"], year=2024, content_status="full_text")]

    def ingest_upload(self, filename, data):
        if not filename.endswith(".pdf"):
            raise ValueError("ResearchOS currently accepts PDF uploads only.")
        return IngestResult(document=self.documents()[0], chunks_created=2)

    def discover_topic(self, topic, **kwargs):
        return DiscoveryResult(topic=topic, candidates_found=3, documents_added=2, full_text_downloads=1, abstract_only=1, skipped_duplicates=0)

    def ask(self, question, *, filters):
        assert filters["year_from"] == 2020
        return WorkflowResult(
            answer="This is a source-bound summary.\n\n- The paper reports a result. [1]",
            evidence_quality="Moderate",
            citations=[{"id": 1, "chunk_id": "chunk-1", "title": "Example paper", "location": "p. 2", "content_kind": "full_text", "source_url": None, "doi": None, "document_id": "doc-1"}],
            evidence=[{"id": "chunk-1", "title": "Example paper", "page": 2, "text": "The paper reports a result.", "content_kind": "full_text"}],
            tool_results=[],
            candidate_claims=[{"text": "The paper reports a result.", "citation_id": 1, "evidence_id": "chunk-1"}],
            verified_claims=[{"text": "The paper reports a result.", "supported": True, "evidence_label": "Strong"}],
            trace=["Retrieved 1 citable passage", "Verified 1/1 candidate claim"],
        )


@pytest.fixture()
def client():
    server.dependency_overrides[get_service] = lambda: FakeService()
    with TestClient(server) as test_client:
        yield test_client
    server.dependency_overrides.clear()


def test_index_serves_the_api_connected_html_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert 'id="discoverForm"' in response.text
    assert "'/api/ask'" in response.text
    assert "Run fixed benchmark" in response.text


def test_corpus_and_ask_endpoints_return_evidence_ready_data(client):
    corpus = client.get("/api/corpus")
    answer = client.post("/api/ask", json={"question": "What result was reported?", "year_from": 2020})

    assert corpus.status_code == 200
    assert corpus.json()["documents"][0]["title"] == "Example paper"
    assert answer.status_code == 200
    assert answer.json()["citations"][0]["chunk_id"] == "chunk-1"
    assert answer.json()["verified_claims"][0]["supported"] is True


def test_upload_and_discovery_routes_delegate_to_the_shared_service(client):
    upload = client.post("/api/upload", files=[("files", ("paper.pdf", b"%PDF fake", "application/pdf"))])
    discovery = client.post("/api/discover", json={"topic": "gravitational-wave methods", "limit": 25, "download_full_text": False})

    assert upload.status_code == 200
    assert upload.json()["uploaded"][0]["chunks_created"] == 2
    assert discovery.status_code == 200
    assert discovery.json()["result"]["documents_added"] == 2


def test_api_rejects_invalid_question_filter_and_non_pdf_upload(client):
    invalid_years = client.post("/api/ask", json={"question": "Compare methods", "year_from": 2025, "year_to": 2020})
    invalid_upload = client.post("/api/upload", files=[("files", ("notes.txt", b"notes", "text/plain"))])

    assert invalid_years.status_code == 400
    assert invalid_upload.status_code == 400
