import pytest

from researchos.models import DiscoveredPaper
from researchos.tools import CalculatorTool, PaperSearchTool


def test_calculator_is_deterministic_and_rejects_python_execution():
    calculator = CalculatorTool()

    assert calculator.evaluate("(12 + 20) / 2") == 16
    assert calculator.statistics([12, 20], label="improvements")["median"] == 16

    with pytest.raises(ValueError):
        calculator.evaluate("__import__('os').system('echo unsafe')")


def test_paper_search_uses_the_bounded_scholarly_client():
    class FakeOpenAlex:
        def search(self, topic, limit, year_from=None):
            assert (topic, limit, year_from) == ("gravitational waves", 3, 2020)
            return [
                DiscoveredPaper(
                    openalex_id="W1",
                    title="Wave paper",
                    authors=["Ada"],
                    year=2021,
                    abstract="Abstract",
                    doi=None,
                    source_url="https://example.test",
                    open_access_url=None,
                )
            ]

    papers = PaperSearchTool(client=FakeOpenAlex()).search("gravitational waves", limit=3, year_from=2020)

    assert papers[0]["title"] == "Wave paper"
