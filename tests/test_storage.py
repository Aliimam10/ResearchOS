from researchos.models import Chunk, DocumentRecord
from researchos.storage import CorpusStore


def test_store_detects_duplicate_by_doi_and_replaces_a_documents_chunks(tmp_path):
    store = CorpusStore(tmp_path / "corpus")
    original = DocumentRecord(id="one", title="An important paper", year=2024, doi="10.1/abc")
    store.save_document(original, [Chunk(id="first", document_id="one", text="Old", title=original.title)])

    duplicate = DocumentRecord(id="two", title="Different title", year=2020, doi="10.1/abc")
    assert store.find_duplicate(duplicate) == original

    replacement = Chunk(id="second", document_id="one", text="New", title=original.title)
    store.save_document(original, [replacement])
    assert [item.id for item in store.chunks()] == ["second"]
