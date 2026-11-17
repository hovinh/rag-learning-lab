from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from graphrag.evaluation.models import BenchmarkCase, EvaluationSample
from graphrag.graph.cypher_store import CypherGraphStore
from graphrag.observability import Trace


class AnsweredQuestion(Protocol):
    """What the system under test returns: its answer, and the retrieved
    contexts it was written from -- e.g. ``AgentResponse`` or any
    ``graphrag.recipes`` ``RagAnswer``.
    """

    @property
    def answer(self) -> str: ...

    @property
    def contexts(self) -> list[str]: ...


Responder = Callable[[str], AnsweredQuestion]

_trace = Trace(__name__)


class BenchmarkRunner:
    """Runs each benchmark question through the system under test (ch08):
    computes the ground truth by running the case's Cypher against
    ``store``, then times ``respond`` (e.g. ``Agent.respond`` or a
    recipe's ``query``) answering
    the question.
    """

    def __init__(self, store: CypherGraphStore, respond: Responder) -> None:
        self._store = store
        self._respond = respond

    def run(self, cases: Sequence[BenchmarkCase]) -> list[EvaluationSample]:
        """A ground-truth query the database rejects raises
        ``GraphQueryError``: that's a broken benchmark, not a wrong answer.
        """
        samples = []
        for index, case in enumerate(cases, start=1):
            with _trace.section(f"Answer [{index}/{len(cases)}] {case.question}"):
                samples.append(self._run_case(case))
        return samples

    def _run_case(self, case: BenchmarkCase) -> EvaluationSample:
        # default=str: Kuzu can return values JSON has no type for (e.g.
        # dates); ensure_ascii=False keeps names like "Miloš Forman" readable
        # to the judge LLM.
        rows = self._store.query(case.cypher)
        _trace.summary("truth", rows)
        ground_truth = json.dumps(rows, default=str, ensure_ascii=False)
        start = time.perf_counter()
        try:
            response = self._respond(case.question)
        # One failing question shouldn't abort the whole benchmark, so any
        # error is recorded on the sample (the book catches everything too,
        # but drops the error and scores the row as "I don't know").
        except Exception as exc:  # noqa: BLE001
            _trace.exception("Answering failed")
            return EvaluationSample(
                case.question, ground_truth, "", [], time.perf_counter() - start, repr(exc)
            )
        return EvaluationSample(
            case.question,
            ground_truth,
            response.answer,
            response.contexts,
            time.perf_counter() - start,
        )
