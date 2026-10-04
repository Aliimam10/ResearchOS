from researchos.embeddings import HashingEncoder
from researchos.evaluation import load_benchmark, run_fixed_benchmark


def test_fixed_benchmark_has_the_declared_manually_checked_question_count():
    questions = load_benchmark()

    assert len(questions) == 50
    assert all(question.relevant_chunk_ids for question in questions)


def test_evaluation_writes_retrieval_and_groundedness_report(tmp_path):
    report_path = tmp_path / "report.json"
    report = run_fixed_benchmark(
        working_dir=tmp_path / "corpus", report_path=report_path, encoder=HashingEncoder(dimensions=128)
    )

    assert report_path.exists()
    assert set(report["retrieval"]) == {"semantic", "bm25", "hybrid"}
    assert 0 <= report["retrieval"]["hybrid"]["recall_at_5"] <= 1
    assert 0 <= report["answer_generation"]["citation_correctness"] <= 1
    assert 0 <= report["answer_generation"]["unsupported_claim_rate"] <= 1
