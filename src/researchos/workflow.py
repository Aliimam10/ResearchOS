"""Bounded LangGraph orchestration for evidence-first local research."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypedDict

from .tools import CalculatorTool, ResearchRetriever

try:  # Keep local tests usable before optional LangGraph has been installed.
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent fallback
    END = "__end__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False


class ResearchState(TypedDict, total=False):
    """Explicit operational state, never a model's hidden reasoning trace."""

    question: str
    filters: dict[str, Any]
    analysis: dict[str, Any]
    plan: list[str]
    retrieval_query: str
    retry_count: int
    evidence: list[dict[str, Any]]
    evidence_sufficient: bool
    evidence_quality: str
    tool_results: list[dict[str, Any]]
    candidate_claims: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    answer: str
    trace: list[str]


@dataclass(slots=True)
class WorkflowResult:
    answer: str
    evidence_quality: str
    citations: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    candidate_claims: list[dict[str, Any]]
    trace: list[str]


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,}")
_QUESTION_STOP_WORDS = {"a", "an", "and", "are", "between", "by", "do", "for", "from", "how", "in", "is", "of", "on", "the", "to", "what", "which", "with"}
_NUMERIC_REQUEST_RE = re.compile(
    r"\b(average|mean|median|calculate|percentage|percent|difference|increase|improvement|compare)\b", re.I
)


def _keywords(text: str) -> list[str]:
    return [word.lower() for word in _TOKEN_RE.findall(text) if word.lower() not in _QUESTION_STOP_WORDS]


def _traced(state: ResearchState, action: str) -> list[str]:
    return [*state.get("trace", []), action]


class ResearchWorkflow:
    """A single, understandable graph—not a complex role-playing multi-agent system.

    It retries corpus retrieval at most once. The resulting trace reports only
    user-safe operational events (e.g. passages retrieved or calculator used),
    never a private chain of thought. Claim verification is intentionally a
    separate later milestone rather than a claim that this stage guarantees truth.
    """

    def __init__(self, retriever: ResearchRetriever, calculator: CalculatorTool | None = None) -> None:
        self.retriever = retriever
        self.calculator = calculator or CalculatorTool()
        self.graph = self._build_graph() if LANGGRAPH_AVAILABLE else None

    def ask(self, question: str, *, filters: dict[str, Any] | None = None) -> WorkflowResult:
        if not question.strip():
            raise ValueError("Ask a research question before running the workflow.")
        initial: ResearchState = {
            "question": question.strip(),
            "filters": filters or {},
            "retry_count": 0,
            "evidence": [],
            "tool_results": [],
            "candidate_claims": [],
            "citations": [],
            "trace": [],
        }
        state = self.graph.invoke(initial) if self.graph else self._run_fallback(initial)
        return WorkflowResult(
            answer=state["answer"],
            evidence_quality=state["evidence_quality"],
            citations=state.get("citations", []),
            evidence=state.get("evidence", []),
            tool_results=state.get("tool_results", []),
            candidate_claims=state.get("candidate_claims", []),
            trace=state.get("trace", []),
        )

    def _build_graph(self):
        assert StateGraph is not None
        graph = StateGraph(ResearchState)
        graph.add_node("analyse", self.analyse)
        graph.add_node("plan", self.plan)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("assess", self.assess)
        graph.add_node("reformulate", self.reformulate)
        graph.add_node("calculate", self.calculate)
        graph.add_node("draft", self.draft)
        graph.add_node("insufficient", self.insufficient)
        graph.set_entry_point("analyse")
        graph.add_edge("analyse", "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "assess")
        graph.add_conditional_edges(
            "assess",
            self.next_step,
            {
                "reformulate": "reformulate",
                "calculate": "calculate",
                "draft": "draft",
                "insufficient": "insufficient",
            },
        )
        graph.add_edge("reformulate", "retrieve")
        graph.add_edge("calculate", "draft")
        graph.add_edge("draft", END)
        graph.add_edge("insufficient", END)
        return graph.compile()

    def analyse(self, state: ResearchState) -> dict[str, Any]:
        needs_calculation = bool(_NUMERIC_REQUEST_RE.search(state["question"]))
        return {
            "analysis": {
                "keywords": _keywords(state["question"]),
                "needs_calculation": needs_calculation,
                "comparison": bool(re.search(r"\b(compare|versus|vs\.?|difference|changed)\b", state["question"], re.I)),
            },
            "trace": _traced(
                state, "Analysed question" + ("; numerical tool may be required" if needs_calculation else "")
            ),
        }

    def plan(self, state: ResearchState) -> dict[str, Any]:
        actions = ["Search the local corpus using hybrid retrieval."]
        if state["analysis"]["keywords"]:
            actions.append("Use a focused keyword query if the first search is insufficient.")
        if state["analysis"]["needs_calculation"]:
            actions.append("Calculate only from explicit values in retrieved passages.")
        return {
            "plan": actions,
            "retrieval_query": state["question"],
            "trace": _traced(state, "Planned bounded hybrid retrieval"),
        }

    def retrieve(self, state: ResearchState) -> dict[str, Any]:
        evidence = self.retriever.search(
            state["retrieval_query"], limit=int(state["filters"].get("limit", 8)), filters=state["filters"]
        )
        return {
            "evidence": evidence,
            "trace": _traced(state, f"Retrieved {len(evidence)} citable passage{'s' if len(evidence) != 1 else ''}"),
        }

    def assess(self, state: ResearchState) -> dict[str, Any]:
        evidence = state.get("evidence", [])
        matched = sum(bool(item.get("matched_terms")) for item in evidence)
        best_semantic = max((float(item.get("semantic_score") or 0) for item in evidence), default=0.0)
        distinct_documents = len({item["document_id"] for item in evidence})
        # RRF ranks every candidate, even a weak one. Gate on an explicit query
        # term or a meaningful semantic score to avoid answering from mere rank.
        has_relevance_signal = bool(matched) or best_semantic >= 0.25
        if not evidence or not has_relevance_signal:
            sufficient, quality = False, "Insufficient"
        elif all(item["content_kind"] == "abstract" for item in evidence):
            sufficient, quality = True, "Limited"
        elif distinct_documents >= 2 and matched >= 2:
            sufficient, quality = True, "Strong"
        else:
            sufficient, quality = True, "Moderate"
        return {
            "evidence_sufficient": sufficient,
            "evidence_quality": quality,
            "trace": _traced(
                state,
                f"Evidence check: {quality.lower()} ({distinct_documents} source document{'s' if distinct_documents != 1 else ''})",
            ),
        }

    def next_step(self, state: ResearchState) -> str:
        if not state["evidence_sufficient"]:
            return "reformulate" if state["retry_count"] < 1 else "insufficient"
        return "calculate" if state["analysis"]["needs_calculation"] else "draft"

    def reformulate(self, state: ResearchState) -> dict[str, Any]:
        query = " ".join(state["analysis"]["keywords"][:12]) or state["question"]
        return {
            "retrieval_query": query,
            "retry_count": state["retry_count"] + 1,
            "trace": _traced(state, "Reformulated a focused keyword query and retried retrieval"),
        }

    def calculate(self, state: ResearchState) -> dict[str, Any]:
        values = self.calculator.percentages_from_evidence(state["evidence"])
        if not values:
            outcome = {
                "tool": self.calculator.name,
                "status": "not_run",
                "reason": "No explicit percentage values were found in the retrieved evidence.",
            }
            action = "Calculator not run: no explicit percentage values in retrieved evidence"
        else:
            outcome = {
                "tool": self.calculator.name,
                "status": "ok",
                **self.calculator.statistics(values, label="percentages extracted from retrieved evidence"),
            }
            action = f"Calculator used on {len(values)} retrieved percentage value{'s' if len(values) != 1 else ''}"
        return {"tool_results": [*state["tool_results"], outcome], "trace": _traced(state, action)}

    def draft(self, state: ResearchState) -> dict[str, Any]:
        evidence = state["evidence"][:4]
        claims: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        for number, item in enumerate(evidence, start=1):
            claims.append(
                {
                    "text": self._best_source_sentence(item["text"], state["analysis"]["keywords"]),
                    "evidence_id": item["id"],
                    "citation_id": number,
                }
            )
            citations.append(self._citation(item, number))
        lines = [f"- {claim['text']} [{claim['citation_id']}]" for claim in claims]
        for result in state["tool_results"]:
            if result["status"] == "ok":
                inputs = ", ".join(f"{value:g}%" for value in result["values"])
                lines.append(
                    f"- Deterministic calculation: mean of {inputs} = **{result['mean']:g}%** "
                    f"(median {result['median']:g}%; n={result['count']})."
                )
            else:
                lines.append(f"- Calculation note: {result['reason']}")
        opening = {
            "Strong": "The corpus contains multiple directly relevant full-text passages.",
            "Moderate": "This is a source-bound summary of the most relevant local passages.",
            "Limited": "This answer relies on abstract-level evidence and should be treated as preliminary.",
        }[state["evidence_quality"]]
        return {
            "candidate_claims": claims,
            "citations": citations,
            "answer": f"{opening}\n\n" + "\n".join(lines),
            "trace": _traced(state, f"Drafted {len(claims)} source-bound claim{'s' if len(claims) != 1 else ''}"),
        }

    def insufficient(self, state: ResearchState) -> dict[str, Any]:
        attempts = state["retry_count"] + 1
        return {
            "evidence_quality": "Insufficient",
            "candidate_claims": [],
            "citations": [],
            "answer": (
                "Insufficient evidence in the current corpus to support an answer. "
                f"ResearchOS made {attempts} bounded retrieval attempt{'s' if attempts != 1 else ''} and found no sufficiently relevant citable passage. "
                "Try uploading targeted documents, narrowing the question, or using Topic Discovery Mode."
            ),
            "trace": _traced(state, "Returned an explicit insufficient-evidence response"),
        }

    @staticmethod
    def _best_source_sentence(text: str, keywords: list[str]) -> str:
        candidates = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        if not candidates:
            return text.strip()
        keyword_set = set(keywords)
        return max(candidates, key=lambda sentence: len(keyword_set & set(_keywords(sentence))))

    @staticmethod
    def _citation(item: dict[str, Any], number: int) -> dict[str, Any]:
        location = f"p. {item['page']}" if item.get("page") else "Abstract"
        if item.get("section") and item["section"] != "Abstract":
            location = f"{location}, {item['section']}"
        return {
            "id": number,
            "document_id": item["document_id"],
            "title": item["title"],
            "location": location,
            "source_url": item.get("source_url"),
            "doi": item.get("doi"),
            "content_kind": item["content_kind"],
        }

    def _run_fallback(self, state: ResearchState) -> ResearchState:
        """Mirror the same bounded graph for minimal/offline test environments."""
        state = {**state, **self.analyse(state)}
        state = {**state, **self.plan(state)}
        while True:
            state = {**state, **self.retrieve(state)}
            state = {**state, **self.assess(state)}
            destination = self.next_step(state)
            if destination == "reformulate":
                state = {**state, **self.reformulate(state)}
                continue
            if destination == "calculate":
                state = {**state, **self.calculate(state)}
                state = {**state, **self.draft(state)}
            elif destination == "draft":
                state = {**state, **self.draft(state)}
            else:
                state = {**state, **self.insufficient(state)}
            return state
