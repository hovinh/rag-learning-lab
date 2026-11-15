from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from graphrag.evaluation.models import EvaluationSample
from graphrag.evaluation.statements import StatementExtractor
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.llm.errors import LLMResponseError
from graphrag.llm.json_response import load_json_field, load_json_object
from graphrag.prompts import PromptRepository
from graphrag.retrieval.embedder import Embedder

_NO_CONTEXT = "(no context was retrieved)"


class Metric(ABC):
    """Scores one answered benchmark sample in [0, 1] (ch08). The three
    subclasses re-implement the RAGAS metrics the book uses, as LLM-judge
    calls through ``LLMClient`` instead of the ``ragas`` package.
    """

    name: ClassVar[str]

    @abstractmethod
    def score(self, sample: EvaluationSample) -> float | None:
        """``None`` if the metric doesn't apply, e.g. faithfulness of an
        answer that makes no statements.

        Raises ``LLMResponseError`` if a judge's response can't be used.
        """


class Faithfulness(Metric):
    """Share of the answer's statements the retrieved contexts support:
    does the answer stick to what retrieval found, or hallucinate?
    """

    name = "faithfulness"

    def __init__(
        self,
        llm: LLMClient,
        *,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
        prompts: PromptRepository | None = None,
    ) -> None:
        self._llm = llm
        self._model = model
        prompts = prompts or PromptRepository()
        self._statements = StatementExtractor(llm, model=model, prompts=prompts)
        self._system_message = prompts.load("eval_faithfulness_system.txt")
        self._user_template = prompts.load("eval_faithfulness_user.txt")

    def score(self, sample: EvaluationSample) -> float | None:
        statements = self._statements.extract(sample.question, sample.answer)
        if not statements:
            return None
        response = _complete_json(
            self._llm,
            self._model,
            self._system_message,
            self._user_template.format(
                context=_format_contexts(sample.contexts), statements=json.dumps(statements)
            ),
        )
        return _mean_flag(response, "verdicts", "verdict")


class ContextRecall(Metric):
    """Share of the ground truth's facts the retrieved contexts contain:
    did retrieval find everything the answer needs?
    """

    name = "context_recall"

    def __init__(
        self,
        llm: LLMClient,
        *,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
        prompts: PromptRepository | None = None,
    ) -> None:
        self._llm = llm
        self._model = model
        prompts = prompts or PromptRepository()
        self._system_message = prompts.load("eval_context_recall_system.txt")
        self._user_template = prompts.load("eval_context_recall_user.txt")

    def score(self, sample: EvaluationSample) -> float | None:
        response = _complete_json(
            self._llm,
            self._model,
            self._system_message,
            self._user_template.format(
                question=sample.question,
                context=_format_contexts(sample.contexts),
                ground_truth=sample.ground_truth,
            ),
        )
        return _mean_flag(response, "classifications", "attributed")


class AnswerCorrectness(Metric):
    """How well the answer matches the ground truth: a weighted sum of the
    F1 score over their statements (an LLM sorts them into true/false
    positives and false negatives) and the cosine similarity of their
    embeddings. RAGAS's default weights are 0.75 and 0.25.
    """

    name = "answer_correctness"

    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder,
        *,
        factuality_weight: float = 0.75,
        similarity_weight: float = 0.25,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
        prompts: PromptRepository | None = None,
    ) -> None:
        if not math.isclose(factuality_weight + similarity_weight, 1.0):
            raise ValueError("factuality_weight and similarity_weight must sum to 1")
        self._llm = llm
        self._embedder = embedder
        self._factuality_weight = factuality_weight
        self._similarity_weight = similarity_weight
        self._model = model
        prompts = prompts or PromptRepository()
        self._statements = StatementExtractor(llm, model=model, prompts=prompts)
        self._system_message = prompts.load("eval_answer_correctness_system.txt")
        self._user_template = prompts.load("eval_answer_correctness_user.txt")

    def score(self, sample: EvaluationSample) -> float:
        f1 = self._f1(sample)
        similarity = self._similarity(sample)
        return self._factuality_weight * f1 + self._similarity_weight * similarity

    def _f1(self, sample: EvaluationSample) -> float:
        answer_statements = self._statements.extract(sample.question, sample.answer)
        ground_truth_statements = self._statements.extract(sample.question, sample.ground_truth)
        response = _complete_json(
            self._llm,
            self._model,
            self._system_message,
            self._user_template.format(
                question=sample.question,
                answer_statements=json.dumps(answer_statements),
                ground_truth_statements=json.dumps(ground_truth_statements),
            ),
        )
        data = load_json_object(response)
        counts = {}
        for key in ("TP", "FP", "FN"):
            items = data.get(key)
            if not isinstance(items, list):
                raise LLMResponseError(f"{key!r} isn't a list: {response}")
            counts[key] = len(items)
        denominator = counts["TP"] + 0.5 * (counts["FP"] + counts["FN"])
        return counts["TP"] / denominator if denominator else 0.0

    def _similarity(self, sample: EvaluationSample) -> float:
        answer, ground_truth = self._embedder.embed([sample.answer, sample.ground_truth])
        norms = math.hypot(*answer) * math.hypot(*ground_truth)
        if not norms:
            return 0.0
        return sum(a * b for a, b in zip(answer, ground_truth, strict=True)) / norms


def _complete_json(llm: LLMClient, model: str, system_message: str, user_message: str) -> str:
    return llm.complete(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        model=model,
        json_mode=True,
    )


def _format_contexts(contexts: list[str]) -> str:
    return "\n\n".join(contexts) or _NO_CONTEXT


def _mean_flag(response: str, field: str, flag: str) -> float | None:
    """Mean of the 0/1 ``flag`` across the judgements listed in ``field``;
    ``None`` if there are none.
    """
    judgements = load_json_field(response, field)
    if not isinstance(judgements, list):
        raise LLMResponseError(f"{field!r} isn't a list: {response}")
    flags: list[Any] = [j.get(flag) if isinstance(j, dict) else None for j in judgements]
    if not all(f in (0, 1) for f in flags):
        raise LLMResponseError(f"Every {field!r} item needs a 0/1 {flag!r}: {response}")
    return sum(flags) / len(flags) if flags else None
