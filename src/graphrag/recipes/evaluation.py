from __future__ import annotations

from collections.abc import Sequence

from graphrag.evaluation.evaluator import EvaluationReport, Evaluator
from graphrag.evaluation.metrics import AnswerCorrectness, ContextRecall, Faithfulness, Metric
from graphrag.evaluation.models import BenchmarkCase, EvaluationSample
from graphrag.evaluation.runner import BenchmarkRunner, Responder
from graphrag.graph.cypher_store import CypherGraphStore
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.retrieval.embedder import Embedder


class RagEvaluation:
    """ch08 as a recipe: ``answer`` runs benchmark questions
    through any responder (``AgenticRag.query``, ``VectorRag.query``, ...),
    computing each ground truth with the case's Cypher on ``store``; then
    ``score`` judges the answers with LLM metrics -- by default the book's
    three, answer correctness, context recall and faithfulness.

    The two are separate so saved answers can be re-scored without paying
    for answering again.
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder,
        *,
        metrics: Sequence[Metric] | None = None,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
    ) -> None:
        self._evaluator = Evaluator(
            metrics
            if metrics is not None
            else [
                AnswerCorrectness(llm, embedder, model=model),
                ContextRecall(llm, model=model),
                Faithfulness(llm, model=model),
            ]
        )

    def answer(
        self, store: CypherGraphStore, cases: Sequence[BenchmarkCase], respond: Responder
    ) -> list[EvaluationSample]:
        return BenchmarkRunner(store, respond).run(cases)

    def score(self, samples: Sequence[EvaluationSample]) -> EvaluationReport:
        return self._evaluator.evaluate(samples)
