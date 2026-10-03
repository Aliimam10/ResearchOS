from researchos.embeddings import HashingEncoder
from researchos.models import Chunk, DocumentRecord
from researchos.retrieval import HybridRetriever, RetrievalFilters
from researchos.storage import CorpusStore


def _retriever(tmp_path) -> HybridRetriever:
    store = CorpusStore(tmp_path / "corpus")
    store.save_document(
        DocumentRecord(id="cnn", title="Convolutional detection", year=2021),
        [
            Chunk(
                id="cnn-results",
                document_id="cnn",
                title="Convolutional detection",
                page=4,
                section="Results",
                year=2021,
                text="A convolutional neural network improved gravitational-wave signal detection accuracy.",
            )
        ],
    )
    store.save_document(
        DocumentRecord(id="bayes", title="Bayesian baseline", year=2018),
        [
            Chunk(
                id="bayes-methods",
                document_id="bayes",
                title="Bayesian baseline",
                page=2,
                section="Methods",
                year=2018,
                text="Bayesian inference estimates detector noise and astrophysical parameters.",
            )
        ],
    )
    store.save_document(
        DocumentRecord(id="vision", title="Image classification", year=2023),
        [
            Chunk(
                id="vision-other",
                document_id="vision",
                title="Image classification",
                page=1,
                year=2023,
                text="A convolutional network classifies satellite images at high accuracy.",
            )
        ],
    )
    return HybridRetriever(store, encoder=HashingEncoder(dimensions=128))


def test_bm25_retrieval_prioritises_exact_question_terms(tmp_path):
    hits = _retriever(tmp_path).search("gravitational-wave signal detection", mode="bm25")

    assert hits[0].chunk.id == "cnn-results"
    assert hits[0].bm25_score is not None and hits[0].bm25_score > 0
    assert hits[0].semantic_score is not None
    assert hits[0].chunk.page == 4
    assert hits[0].chunk.section == "Results"


def test_hybrid_retrieval_fuses_ranks_and_returns_rerank_provenance(tmp_path):
    hits = _retriever(tmp_path).search("convolutional detection accuracy", mode="hybrid", limit=2)

    assert hits[0].chunk.id == "cnn-results"
    assert hits[0].semantic_rank is not None
    assert hits[0].bm25_rank is not None
    assert hits[0].rrf_score is not None
    assert hits[0].rerank_score == hits[0].score
    assert "convolutional" in hits[0].matched_terms


def test_metadata_filters_exclude_old_and_non_selected_documents(tmp_path):
    retriever = _retriever(tmp_path)

    recent = retriever.search("convolutional accuracy", filters=RetrievalFilters(year_from=2020))
    one_document = retriever.search(
        "convolutional accuracy", filters=RetrievalFilters(document_ids={"cnn"})
    )

    assert {hit.chunk.document_id for hit in recent} <= {"cnn", "vision"}
    assert [hit.chunk.document_id for hit in one_document] == ["cnn"]


def test_index_refreshes_when_a_new_document_is_added(tmp_path):
    retriever = _retriever(tmp_path)
    retriever.search("bayesian detector")  # Build the initial index.
    store = retriever.store
    store.save_document(
        DocumentRecord(id="transformer", title="Transformer search", year=2025),
        [
            Chunk(
                id="transformer-results",
                document_id="transformer",
                title="Transformer search",
                page=7,
                year=2025,
                text="Transformer models improve gravitational-wave detection sensitivity.",
            )
        ],
    )

    hits = retriever.search("transformer sensitivity", mode="hybrid")

    assert hits[0].chunk.id == "transformer-results"
