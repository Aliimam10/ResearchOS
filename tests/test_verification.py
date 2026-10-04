from researchos.citations import citation_from_evidence
from researchos.verification import ClaimEvidenceVerifier


def _evidence(content_kind="full_text"):
    return {
        "id": "chunk-1",
        "document_id": "doc-1",
        "title": "Example study",
        "page": 3,
        "section": "Results",
        "text": "The experiment reported a 12% improvement in recall.",
        "content_kind": content_kind,
        "source_url": "https://example.test",
        "doi": None,
    }


def test_verifier_marks_a_direct_full_text_claim_strong_and_citable():
    evidence = _evidence()
    citation = citation_from_evidence(evidence, 1)
    outcome = ClaimEvidenceVerifier().verify(
        [{"text": "The experiment reported a 12% improvement in recall.", "evidence_id": "chunk-1", "citation_id": 1}],
        [evidence],
        [citation.to_dict()],
    )[0]

    assert outcome.supported is True
    assert outcome.evidence_label == "Strong"
    assert citation.reference() == "[1] Example study — p. 3, Results · https://example.test"


def test_verifier_rejects_a_claim_with_a_mismatched_citation():
    evidence = _evidence()
    outcome = ClaimEvidenceVerifier().verify(
        [{"text": "The experiment reported a 12% improvement in recall.", "evidence_id": "chunk-1", "citation_id": 1}],
        [evidence],
        [{"id": 1, "chunk_id": "wrong-chunk"}],
    )[0]

    assert outcome.supported is False
    assert outcome.evidence_label == "Insufficient"


def test_verifier_downgrades_abstract_only_support():
    evidence = _evidence(content_kind="abstract")
    citation = citation_from_evidence(evidence, 1)
    outcome = ClaimEvidenceVerifier().verify(
        [{"text": "The experiment reported a 12% improvement in recall.", "evidence_id": "chunk-1", "citation_id": 1}],
        [evidence],
        [citation.to_dict()],
    )[0]

    assert outcome.supported is True
    assert outcome.evidence_label == "Moderate"
