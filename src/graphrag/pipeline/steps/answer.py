from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.observability import Trace, preview
from graphrag.prompts import PromptRepository

_ANSWER_SYSTEM_PROMPT = "answer_system.txt"
_ANSWER_USER_PROMPT = "answer_user.txt"

_trace = Trace(__name__)


class ScoredMatch(Protocol):
    """A retrieval match -- satisfied structurally by both ``ChunkMatch``
    and ``ParentMatch`` without either depending on this module. Declared
    as read-only properties (rather than plain attributes) so mypy checks
    them covariantly -- both dataclasses are frozen, so their fields are
    read-only in practice too.
    """

    @property
    def text(self) -> str: ...

    @property
    def score(self) -> float: ...


class AnswerContext(Protocol):
    question: str
    answer: str

    @property
    def matches(self) -> Sequence[ScoredMatch]: ...


class AnswerStep:
    """Logs the retrieved matches, builds the doc-stuffing prompt from
    ``answer_system.txt``/``answer_user.txt``, and streams the LLM's answer
    into ``context.answer`` -- replaces each example's own duplicated
    ``log_matches``/``build_prompt``/``ask`` functions.

    ``label`` names the retrieval the matches came from, in the step's
    pipeline section title (``Answer (vector search)``).
    """

    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptRepository | None = None,
        *,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
        label: str | None = None,
    ) -> None:
        self._llm = llm
        self._model = model
        self.title = f"Answer ({label.lower()})" if label else "Answer"
        prompts = prompts or PromptRepository()
        self._system_message = prompts.load(_ANSWER_SYSTEM_PROMPT)
        self._user_template = prompts.load(_ANSWER_USER_PROMPT)

    def __call__(self, context: AnswerContext) -> None:
        _trace.table(
            "matches",
            [
                {"#": rank, "score": match.score, "text": preview(match.text, 70)}
                for rank, match in enumerate(context.matches, start=1)
            ],
            max_width=70,
        )
        for rank, match in enumerate(context.matches, start=1):
            _trace.block(f"match {rank}", match.text, level=logging.DEBUG)

        context_text = "\n".join(match.text for match in context.matches)
        user_message = self._user_template.format(context=context_text, question=context.question)

        _trace.field("question", context.question)
        # Streamed as the book does, but logged once complete: printing
        # deltas would break up the trace on the console.
        context.answer = "".join(
            self._llm.stream(
                [
                    {"role": "system", "content": self._system_message},
                    {"role": "user", "content": user_message},
                ],
                model=self._model,
            )
        )
        _trace.field("answer", context.answer)
