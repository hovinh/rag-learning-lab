from __future__ import annotations

from dataclasses import dataclass, field

from graphrag.ingestion.models import Chunk
from graphrag.pipelines.models import RagAnswer
from graphrag.retrieval.retriever import ScoredMatch


@dataclass
class ChunkIndexContext:
    documents: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)


@dataclass
class AnswerContext:
    question: str
    matches: list[ScoredMatch] = field(default_factory=list)
    answer: str = ""

    def to_answer(self) -> RagAnswer:
        return RagAnswer(self.question, self.answer, [match.text for match in self.matches])
