from researchos.ingestion import PDFIngestor, abstract_chunk, section_aware_chunks
from researchos.models import DocumentRecord
from researchos.storage import CorpusStore


def test_section_aware_chunks_keep_page_and_document_metadata(tmp_path):
    document = DocumentRecord(
        id="doc_test",
        title="Detector Methods",
        authors=["A. Researcher"],
        year=2024,
        doi="10.999/example",
        source_url="https://example.test/paper",
    )
    chunks = section_aware_chunks(
        document,
        [
            "INTRODUCTION\nThe detector receives a signal and filters background noise.",
            "2 Methods\nWe compare a convolutional classifier with a baseline model.",
        ],
    )

    assert len(chunks) == 2
    assert chunks[0].page == 1
    assert chunks[0].section == "Introduction"
    assert chunks[1].page == 2
    assert chunks[1].section == "Methods"
    assert chunks[1].title == "Detector Methods"
    assert chunks[1].doi == "10.999/example"
    assert chunks[1].content_kind == "full_text"


def test_ingestor_persists_text_pages_as_citable_full_text(tmp_path):
    store = CorpusStore(tmp_path / "corpus")
    ingestor = PDFIngestor(store)

    result = ingestor.ingest_text_pages(
        "Local report", ["Abstract\nA compact study with an important finding."], authors=["Sam"], year=2022
    )

    assert result.chunks_created == 1
    assert result.document.content_status == "full_text"
    assert store.counts() == {
        "documents": 1,
        "chunks": 1,
        "full_text_documents": 1,
        "abstract_only_documents": 0,
    }
    chunk = store.chunks()[0]
    assert chunk.document_id == result.document.id
    assert chunk.page == 1
    assert chunk.authors == ["Sam"]


def test_abstract_fallback_is_explicitly_labelled():
    document = DocumentRecord(id="doc_abs", title="Metadata-only paper", abstract="A useful abstract.")

    chunk = abstract_chunk(document)

    assert chunk is not None
    assert chunk.section == "Abstract"
    assert chunk.content_kind == "abstract"
    assert chunk.page is None
