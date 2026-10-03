from researchos.embeddings import HashingEncoder
from researchos.ingestion import PDFIngestor
from researchos.retrieval import HybridRetriever
from researchos.storage import CorpusStore
from researchos.tools import ResearchRetriever
from researchos.workflow import ResearchWorkflow


def _workflow(tmp_path) -> ResearchWorkflow:
    store = CorpusStore(tmp_path / "corpus")
    PDFIngestor(store).ingest_text_pages(
        "Detector model comparison",
        [
            "RESULTS\nMethod A reported a 12% improvement in detection accuracy. "
            "Method B reported a 20% improvement in detection accuracy."
        ],
        authors=["A. Author"],
        year=2024,
    )
    return ResearchWorkflow(ResearchRetriever(HybridRetriever(store, encoder=HashingEncoder())))


def test_workflow_calls_calculator_only_on_retrieved_percentages(tmp_path):
    result = _workflow(tmp_path).ask("What is the average improvement in detection accuracy?")

    assert result.evidence_quality == "Moderate"
    assert result.tool_results[0]["tool"] == "calculator"
    assert result.tool_results[0]["values"] == [12.0, 20.0]
    assert result.tool_results[0]["mean"] == 16
    assert "16%" in result.answer
    assert result.citations[0]["location"] == "p. 1, Results"
    assert any("Calculator used" in action for action in result.trace)


def test_workflow_returns_insufficient_evidence_after_one_retry(tmp_path):
    result = _workflow(tmp_path).ask("What changed in dragon-scale materials?")

    assert result.evidence_quality == "Insufficient"
    assert result.citations == []
    assert result.answer.startswith("Insufficient evidence")
    assert sum("Retrieved" in action for action in result.trace) == 2
    assert any("Reformulated" in action for action in result.trace)
    assert not any("chain" in action.lower() for action in result.trace)
