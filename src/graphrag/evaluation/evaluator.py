from __future__ import annotations

import csv
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from graphrag.evaluation.metrics import Metric
from graphrag.evaluation.models import EvaluationSample
from graphrag.llm.errors import LLMResponseError
from graphrag.observability import Trace, format_duration, preview

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

    def write_csv(self, path: Path) -> None:
        """One row per question, like the book's final ``test_data`` table."""
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["question", "ground_truth", "answer", "error", "latency", *self.metric_names]
            )
            for row in self.rows:
                sample = row.sample
                writer.writerow(
                    [
                        sample.question,
                        sample.ground_truth,
                        sample.answer,
                        sample.error or "",
                        f"{sample.latency:.3f}",
                        *(format_score(row.scores[name]) for name in self.metric_names),
                    ]
                )


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


def log_report(report: EvaluationReport) -> None:
    """Traces the per-question scores as a table, then the means and the
    questions that failed to answer.
    """
    _trace.table(
        "scores",
        [
            {
                "#": index,
                "question": preview(row.sample.question, 45),
                "latency": format_duration(row.sample.latency),
                **{name: format_score(score) for name, score in row.scores.items()},
            }
            for index, row in enumerate(report.rows, start=1)
        ],
        max_width=45,
    )
    mean_latency = report.mean_latency()
    _trace.fields(
        {
            **{f"mean {name}": format_score(mean) for name, mean in report.means().items()},
            "mean latency": "n/a" if mean_latency is None else format_duration(mean_latency),
            "failed": f"{len(report.failed())}/{len(report.rows)} questions",
        }
    )
    for sample in report.failed():
        _trace.field("failed", f"{sample.question}: {sample.error}", level=logging.WARNING)
