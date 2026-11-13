from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from logzero import logger

from graphrag.ingestion.pdf_loader import PdfLoader
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.prompts import PromptRepository

_ANSWER_SYSTEM_PROMPT = "answer_system.txt"
_ANSWER_USER_PROMPT = "answer_user.txt"


class HasText(Protocol):
    """A context exposing a mutable ``text`` attribute, e.g. one written to
    by ``LoadPdfTextStep``.
    """

    text: str


class LoadPdfTextStep:
    """Downloads the PDF (if not already on disk) and extracts its text
    into ``context.text``. Identical across chapters -- ch02 and ch03 both
    start from the same "download a PDF, read its text" step.
    """

    def __init__(self, pdf_loader: PdfLoader, url: str, path: Path) -> None:
        self._pdf_loader = pdf_loader
        self._url = url
        self._path = path

    def __call__(self, context: HasText) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._pdf_loader.download(self._url, self._path)
        context.text = self._pdf_loader.extract_text(self._path)


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
        self._label = label
        prompts = prompts or PromptRepository()
        self._system_message = prompts.load(_ANSWER_SYSTEM_PROMPT)
        self._user_template = prompts.load(_ANSWER_USER_PROMPT)

    def __call__(self, context: AnswerContext) -> None:
        if self._label:
            logger.info(f"--- {self._label} ---")
        for match in context.matches:
            logger.info(f"{match.score} {match.text}")

        context_text = "\n".join(match.text for match in context.matches)
        user_message = self._user_template.format(context=context_text, question=context.question)

        logger.info(f"Question: {context.question}")
        deltas = []
        for delta in self._llm.stream(
            [
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": user_message},
            ],
            model=self._model,
        ):
            print(delta, end="", flush=True)
            deltas.append(delta)
        print()
        context.answer = "".join(deltas)
        logger.info(f"Answer: {context.answer}")
