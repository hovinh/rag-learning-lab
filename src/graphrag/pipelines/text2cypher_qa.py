from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from graphrag.graph.cypher_store import CypherGraphStore
from graphrag.graph.models import GraphSchema
from graphrag.graph.text2cypher import CypherExample, Text2Cypher
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.pipeline import ExecuteCypherStep, GenerateCypherStep, Pipeline
from graphrag.pipelines.models import CypherAnswer


@dataclass
class _CypherContext:
    question: str
    cypher: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)


class Text2CypherQA:
    """ch04 as a reusable pipeline: the LLM writes Cypher for a question
    against ``store``'s schema, which then runs on the graph.

    The schema is read when this is built, so ``store`` must already hold
    the graph; ``terminology`` and ``examples`` are the domain's prompt
    sections (e.g. ``MOVIES_TERMINOLOGY``, ``MOVIES_CYPHER_EXAMPLES``).
    """

    def __init__(
        self,
        store: CypherGraphStore,
        llm: LLMClient,
        *,
        terminology: str | None = None,
        examples: Sequence[CypherExample] = (),
        model: str = OpenAILLMClient.DEFAULT_MODEL,
    ) -> None:
        self.schema: GraphSchema = store.get_schema()
        text2cypher = Text2Cypher(
            llm, self.schema, terminology=terminology, examples=examples, model=model
        )
        self._pipeline: Pipeline[_CypherContext] = Pipeline(
            [GenerateCypherStep(text2cypher), ExecuteCypherStep(store)]
        )

    def query(self, question: str) -> CypherAnswer:
        context = self._pipeline.run(_CypherContext(question))
        return CypherAnswer(question, context.cypher, context.rows)
