from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from graphrag.evaluation.metrics import Metric
from graphrag.evaluation.models import EvaluationSample
from graphrag.llm.errors import LLMResponseError
from graphrag.observability import Trace

_trace = Trace(__name__)


@dataclass(frozen=True)
class ScoredSample:
    """``scores`` maps each metric name to its score, or to ``None`` if the
    metric didn't apply, its judge's response was unusable, or the sample
    failed to answer.
    """

    sample: EvaluationSample
    scores: dict[str, float | None]


@dataclass(frozen=True)
class EvaluationReport:
    metric_names: list[str]
    rows: list[ScoredSample]

    def means(self) -> dict[str, float | None]:
        """Each metric's mean over the rows it scored (``None`` if it
        scored none). Failed samples are left out, so read this next to
        ``failed()``.
        """
        means: dict[str, float | None] = {}
        for name in self.metric_names:
            scores = [s for row in self.rows if (s := row.scores[name]) is not None]
            means[name] = sum(scores) / len(scores) if scores else None
        return means

    def mean_latency(self) -> float | None:
        latencies = [row.sample.latency for row in self.rows]
        return sum(latencies) / len(latencies) if latencies else None

    def failed(self) -> list[EvaluationSample]:
        return [row.sample for row in self.rows if row.sample.error is not None]


class Evaluator:
    """Scores benchmark samples with each metric (ch08's ``ragas.evaluate``
    step). Samples that failed to answer get no scores rather than being
    judged as an empty answer.
    """

    def __init__(self, metrics: Sequence[Metric]) -> None:
        names = [metric.name for metric in metrics]
        if len(set(names)) != len(names):
            raise ValueError(f"Metric names must be unique: {names}")
        self._metrics = list(metrics)

    def evaluate(self, samples: Sequence[EvaluationSample]) -> EvaluationReport:
        rows = []
        for index, sample in enumerate(samples, start=1):
            with _trace.section(f"Score [{index}/{len(samples)}] {sample.question}"):
                if sample.error is not None:
                    _trace.field("skipped", "no answer to score", level=logging.WARNING)
                    scores: dict[str, float | None] = {m.name: None for m in self._metrics}
                else:
                    scores = {m.name: self._score(m, sample) for m in self._metrics}
                    _trace.fields({name: format_score(score) for name, score in scores.items()})
            rows.append(ScoredSample(sample, scores))
        return EvaluationReport([m.name for m in self._metrics], rows)

    @staticmethod
    def _score(metric: Metric, sample: EvaluationSample) -> float | None:
        # By now every question has been answered (and paid for), so one
        # unusable judge response costs only that score, not the report.
        try:
            return metric.score(sample)
        except LLMResponseError:
            _trace.exception(f"{metric.name} couldn't score this answer")
            return None


def format_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.3f}"
