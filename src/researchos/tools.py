"""The deliberately small, inspectable tool set used by ResearchOS workflows."""
from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import asdict
from statistics import mean, median
from typing import Any, Callable, Iterable

from .discovery import OpenAlexClient
from .retrieval import HybridRetriever, RetrievalFilters


class ResearchRetriever:
    """Workflow tool wrapper around hybrid search, preserving retrieval signals."""

    name = "research_retriever"

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def search(self, query: str, *, limit: int = 8, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = filters or {}
        source_types = filters.get("source_types")
        retrieval_filters = RetrievalFilters(
            year_from=filters.get("year_from"),
            year_to=filters.get("year_to"),
            document_ids=set(filters["document_ids"]) if filters.get("document_ids") else None,
            source_types=set(source_types) if source_types else None,
        )
        hits = self.retriever.search(query, limit=limit, filters=retrieval_filters, mode="hybrid")
        return [hit.to_dict() for hit in hits]


class CalculatorTool:
    """Restricted arithmetic and descriptive statistics—never arbitrary Python."""

    name = "calculator"
    _BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    _UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def evaluate(self, expression: str) -> float:
        try:
            root = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError("Calculator expression is not valid arithmetic.") from exc
        value = self._evaluate_node(root.body)
        if not math.isfinite(value):
            raise ValueError("Calculator result is not finite.")
        return value

    def _evaluate_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._UNARY:
            return self._UNARY[type(node.op)](self._evaluate_node(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in self._BINARY:
            left = self._evaluate_node(node.left)
            right = self._evaluate_node(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent is outside the calculator safety limit.")
            return self._BINARY[type(node.op)](left, right)
        raise ValueError("Calculator accepts only numbers and +, -, *, /, **.")

    def statistics(self, values: Iterable[float], *, label: str) -> dict[str, Any]:
        numbers = [float(value) for value in values]
        if not numbers:
            raise ValueError("No values supplied for deterministic calculation.")
        return {
            "operation": "descriptive_statistics",
            "label": label,
            "count": len(numbers),
            "values": numbers,
            "mean": round(mean(numbers), 6),
            "median": round(median(numbers), 6),
            "minimum": min(numbers),
            "maximum": max(numbers),
        }

    def percentages_from_evidence(self, evidence: Iterable[dict[str, Any]]) -> list[float]:
        """Extract only percentages in performance-related sentences.

        The workflow puts the exact values in its public output. This avoids
        concealing a judgment that values from unlike experiments are comparable.
        """
        values: list[float] = []
        for item in evidence:
            sentences = re.split(r"(?<=[.!?])\s+", item["text"])
            for sentence in sentences:
                if re.search(r"improv|increase|reduc|accuracy|performance", sentence, re.I):
                    values.extend(float(value) for value in re.findall(r"(-?\d+(?:\.\d+)?)\s*%", sentence))
        return values


class PaperSearchTool:
    """A bounded scholarly API tool; it does not scrape arbitrary web pages."""

    name = "paper_search"

    def __init__(self, client: OpenAlexClient | None = None) -> None:
        self.client = client or OpenAlexClient()

    def search(self, topic: str, *, limit: int = 10, year_from: int | None = None) -> list[dict[str, Any]]:
        return [asdict(paper) for paper in self.client.search(topic, limit=limit, year_from=year_from)]
