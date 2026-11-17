from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark row: a question and its ground truth, given either
    as Cypher whose result is the ground truth (the book's ch08 movies
    benchmark) or as a written ``answer`` (for questions over text, which
    no query can answer). ``category`` groups questions for reporting,
    e.g. "specific" vs "broad".
    """

    question: str
    cypher: str | None = None
    answer: str | None = None
    category: str = ""

    def __post_init__(self) -> None:
        if (self.cypher is None) == (self.answer is None):
            raise ValueError(f"Give exactly one of cypher or answer: {self.question!r}")


@dataclass(frozen=True)
class EvaluationSample:
    """A benchmark question as the system under test handled it.

    ``ground_truth`` is the written answer, or the ground-truth Cypher's
    rows as JSON. ``error`` is set (and ``answer``/``contexts`` empty) if
    answering raised; the book records ``None`` there and later scores it
    as "I don't know".
    """

    question: str
    ground_truth: str
    answer: str
    contexts: list[str]
    latency: float
    error: str | None = None
