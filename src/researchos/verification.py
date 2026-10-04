"""Conservative claim/evidence checks; useful groundedness signals, not truth guarantees."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,}")
_STOP_WORDS = {"a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "with"}


def _terms(text: str) -> set[str]:
    return {word.lower() for word in _TOKEN_RE.findall(text) if word.lower() not in _STOP_WORDS}


@dataclass(slots=True)
class VerifiedClaim:
    text: str
    evidence_id: str | None
    citation_id: int | None
    support_score: float
    evidence_label: str
    supported: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClaimEvidenceVerifier:
    """Verify claim-to-passage links with transparent lexical containment.

    This checker does not establish the real-world truth of a passage. It checks
    the narrower and essential RAG property that a displayed claim is actually
    supported by the retrieved passage cited beside it.
    """

    def verify(
        self,
        claims: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        citations: list[dict[str, Any]],
    ) -> list[VerifiedClaim]:
        by_id = {item["id"]: item for item in evidence}
        citation_by_id = {int(item["id"]): item for item in citations}
        outcomes: list[VerifiedClaim] = []
        for claim in claims:
            evidence_id = claim.get("evidence_id")
            citation_id = claim.get("citation_id")
            passage = by_id.get(evidence_id)
            citation = citation_by_id.get(citation_id) if citation_id is not None else None
            if not passage or not citation or citation.get("chunk_id") != evidence_id:
                outcomes.append(
                    VerifiedClaim(
                        text=claim["text"],
                        evidence_id=evidence_id,
                        citation_id=citation_id,
                        support_score=0.0,
                        evidence_label="Insufficient",
                        supported=False,
                        reason="The claim has no matching retrieved passage and citation.",
                    )
                )
                continue

            claim_terms = _terms(claim["text"])
            passage_terms = _terms(passage["text"])
            overlap = len(claim_terms & passage_terms) / max(1, len(claim_terms))
            exact = " ".join(claim["text"].split()).lower() in " ".join(passage["text"].split()).lower()
            # Abstracts can provide discovery-level support, but should not get
            # the strongest label even if the sentence was copied verbatim.
            if exact and passage["content_kind"] == "full_text":
                label, supported, reason = "Strong", True, "Claim is directly present in the cited full-text passage."
            elif overlap >= 0.75:
                label, supported, reason = "Moderate", True, "Most claim terms occur in the cited passage."
            elif overlap >= 0.45:
                label, supported, reason = "Limited", True, "Only partial lexical support was found in the cited passage."
            else:
                label, supported, reason = "Insufficient", False, "The cited passage does not sufficiently support the claim wording."
            if passage["content_kind"] == "abstract" and label == "Strong":
                label = "Moderate"
                reason = "Claim is present in an abstract; full-text support was not available."
            outcomes.append(
                VerifiedClaim(
                    text=claim["text"],
                    evidence_id=evidence_id,
                    citation_id=citation_id,
                    support_score=round(overlap, 4),
                    evidence_label=label,
                    supported=supported,
                    reason=reason,
                )
            )
        return outcomes
