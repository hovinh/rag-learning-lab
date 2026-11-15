import pytest

from graphrag.evaluation.evaluator import Evaluator
from graphrag.evaluation.metrics import Metric
from graphrag.evaluation.models import EvaluationSample
from graphrag.llm.errors import LLMResponseError


class FixedMetric(Metric):
    """Scores each sample by looking its question up in ``scores``; an
    exception there is raised instead.
    """

    def __init__(self, name, scores):
        self.name = name
        self._scores = scores
        self.scored = []

    def score(self, sample):
        self.scored.append(sample.question)
        result = self._scores[sample.question]
        if isinstance(result, Exception):
            raise result
        return result


def make_sample(question, latency=1.0, error=None):
    return EvaluationSample(question, "[]", "" if error else "answer", [], latency, error)


def test_evaluate_scores_every_sample_with_every_metric():
    correctness = FixedMetric("correctness", {"a": 1.0, "b": 0.5})
    recall = FixedMetric("recall", {"a": 0.0, "b": None})
    samples = [make_sample("a", latency=1.0), make_sample("b", latency=3.0)]

    report = Evaluator([correctness, recall]).evaluate(samples)

    assert report.metric_names == ["correctness", "recall"]
    assert [row.scores for row in report.rows] == [
        {"correctness": 1.0, "recall": 0.0},
        {"correctness": 0.5, "recall": None},
    ]
    # None (not applicable) is left out of the mean, not counted as 0.
    assert report.means() == {"correctness": 0.75, "recall": 0.0}
    assert report.mean_latency() == 2.0
    assert report.failed() == []


def test_evaluate_doesnt_score_failed_samples():
    metric = FixedMetric("correctness", {"a": 1.0})
    failed = make_sample("b", error="RuntimeError('boom')")

    report = Evaluator([metric]).evaluate([make_sample("a"), failed])

    assert metric.scored == ["a"]
    assert report.rows[1].scores == {"correctness": None}
    assert report.means() == {"correctness": 1.0}
    assert report.failed() == [failed]


def test_evaluate_keeps_going_when_a_judge_response_is_unusable():
    metric = FixedMetric("correctness", {"a": LLMResponseError("bad"), "b": 0.5})

    report = Evaluator([metric]).evaluate([make_sample("a"), make_sample("b")])

    assert [row.scores for row in report.rows] == [{"correctness": None}, {"correctness": 0.5}]


def test_means_of_an_unscored_metric_is_none():
    report = Evaluator([FixedMetric("correctness", {"a": None})]).evaluate([make_sample("a")])

    assert report.means() == {"correctness": None}


def test_evaluator_rejects_duplicate_metric_names():
    with pytest.raises(ValueError):
        Evaluator([FixedMetric("x", {}), FixedMetric("x", {})])
