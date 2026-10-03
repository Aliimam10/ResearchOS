from researchos.discovery import TopicDiscovery
from researchos.ingestion import PDFIngestor
from researchos.models import DiscoveredPaper
from researchos.storage import CorpusStore


class FakeOpenAlex:
    def __init__(self):
        self.search_calls = []

    def search(self, topic, limit, year_from=None):
        self.search_calls.append((topic, limit, year_from))
        return [
            DiscoveredPaper(
                openalex_id="https://openalex.org/W123",
                title="Learning gravitational-wave signals",
                authors=["Ada Lovelace"],
                year=2023,
                abstract="A neural model detects signals in noisy interferometer data.",
                doi="https://doi.org/10.1000/example",
                source_url="https://example.test/landing",
                open_access_url=None,
                cited_by_count=7,
            ),
            DiscoveredPaper(
                openalex_id="https://openalex.org/W124",
                title="A record without an abstract",
                authors=[],
                year=2022,
                abstract=None,
                doi=None,
                source_url="https://example.test/no-abstract",
                open_access_url=None,
            ),
        ]


def test_discovery_keeps_metadata_and_abstract_when_full_text_is_unavailable(tmp_path):
    store = CorpusStore(tmp_path / "corpus")
    client = FakeOpenAlex()
    discovery = TopicDiscovery(store, PDFIngestor(store), tmp_path / "downloads", client=client)

    result = discovery.discover("gravitational waves", limit=50, year_from=2020, download_full_text=False)

    assert client.search_calls == [("gravitational waves", 50, 2020)]
    assert result.candidates_found == 2
    assert result.documents_added == 2
    assert result.abstract_only == 1
    assert store.counts()["documents"] == 2
    assert len(store.chunks()) == 1
    assert store.chunks()[0].section == "Abstract"


def test_discovery_deduplicates_a_second_topic_run(tmp_path):
    store = CorpusStore(tmp_path / "corpus")
    discovery = TopicDiscovery(store, PDFIngestor(store), tmp_path / "downloads", client=FakeOpenAlex())

    discovery.discover("waves", download_full_text=False)
    second = discovery.discover("waves", download_full_text=False)

    assert second.documents_added == 0
    assert second.skipped_duplicates == 2
    assert store.counts()["documents"] == 2
