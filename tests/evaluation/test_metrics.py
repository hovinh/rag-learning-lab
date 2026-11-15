import json

import pytest

from graphrag.evaluation.metrics import AnswerCorrectness, ContextRecall, Faithfulness
from graphrag.evaluation.models import EvaluationSample
from graphrag.llm.errors import LLMResponseError


def make_sample(answer="Tom Cruise and Val Kilmer.", contexts=("Top Gun cast: Tom Cruise",)):
    return EvaluationSample(
        question="Who acted in Top Gun?",
        ground_truth='[{"p.name": "Tom Cruise"}, {"p.name": "Val Kilmer"}]',
        answer=answer,
        contexts=list(contexts),
        latency=1.0,
    )


def judgements(field, flag, *values):
    return json.dumps({field: [{"statement": "s", "reason": "r", flag: v} for v in values]})


def test_faithfulness_is_share_of_supported_statements(llm):
    llm.completions = [
        '{"statements": ["Tom Cruise acted in Top Gun.", "Val Kilmer acted in Top Gun."]}',
        judgements("verdicts", "verdict", 1, 0),
    ]

    assert Faithfulness(llm, model="test-model").score(make_sample()) == 0.5
    judge = llm.complete_calls[1]
    assert judge["model"] == "test-model"
    assert judge["json_mode"] is True
    assert "Top Gun cast: Tom Cruise" in judge["messages"][1]["content"]
    assert '"Val Kilmer acted in Top Gun."' in judge["messages"][1]["content"]


def test_faithfulness_doesnt_apply_to_an_answer_without_statements(llm):
    llm.completions = ['{"statements": []}']

    assert Faithfulness(llm).score(make_sample(answer="I don't know.")) is None
    assert len(llm.complete_calls) == 1


def test_context_recall_is_share_of_attributed_ground_truth_facts(llm):
    llm.completions = [judgements("classifications", "attributed", 1, 0, 0, 1)]

    assert ContextRecall(llm).score(make_sample()) == 0.5
    [call] = llm.complete_calls
    assert '"p.name": "Val Kilmer"' in call["messages"][1]["content"]


def test_context_recall_marks_missing_context(llm):
    llm.completions = [judgements("classifications", "attributed", 0)]

    assert ContextRecall(llm).score(make_sample(contexts=())) == 0.0
    assert "(no context was retrieved)" in llm.complete_calls[0]["messages"][1]["content"]


@pytest.mark.parametrize(
    "response",
    [
        "{}",
        '{"classifications": "x"}',
        '{"classifications": ["x"]}',
        '{"classifications": [{"attributed": 2}]}',
        '{"classifications": [{"attributed": "1"}]}',
    ],
)
def test_context_recall_rejects_unusable_responses(llm, response):
    llm.completions = [response]

    with pytest.raises(LLMResponseError):
        ContextRecall(llm).score(make_sample())


def test_answer_correctness_weights_f1_and_similarity(llm, embedder):
    sample = make_sample()
    llm.completions = [
        '{"statements": ["Tom Cruise acted in Top Gun.", "Val Kilmer acted in Top Gun."]}',
        '{"statements": ["Tom Cruise acted in Top Gun.", "Kelly McGillis acted in Top Gun."]}',
        json.dumps(
            {
                "TP": [{"statement": "Tom Cruise", "reason": "r"}],
                "FP": [{"statement": "Val Kilmer", "reason": "r"}],
                "FN": [{"statement": "Kelly McGillis", "reason": "r"}],
            }
        ),
    ]
    embedder.vectors = {sample.answer: [1.0, 0.0, 0.0], sample.ground_truth: [0.0, 1.0, 0.0]}

    score = AnswerCorrectness(llm, embedder).score(sample)

    # F1 = 1 / (1 + 0.5 * (1 + 1)) = 0.5; the embeddings are orthogonal.
    assert score == pytest.approx(0.75 * 0.5 + 0.25 * 0.0)
    judge = llm.complete_calls[2]["messages"][1]["content"]
    assert '"Val Kilmer acted in Top Gun."' in judge
    assert '"Kelly McGillis acted in Top Gun."' in judge


def test_answer_correctness_of_an_exact_match_is_one(llm, embedder):
    sample = make_sample()
    llm.completions = [
        '{"statements": ["a"]}',
        '{"statements": ["a"]}',
        '{"TP": [{"statement": "a", "reason": "r"}], "FP": [], "FN": []}',
    ]
    embedder.vectors = {sample.answer: [1.0, 2.0, 0.0], sample.ground_truth: [2.0, 4.0, 0.0]}

    assert AnswerCorrectness(llm, embedder).score(sample) == pytest.approx(1.0)


def test_answer_correctness_rejects_a_ground_truth_without_statements(llm, embedder):
    llm.completions = ['{"statements": ["Tom Hanks appeared in 8 movies."]}', '{"statements": []}']

    with pytest.raises(LLMResponseError, match="ground truth gave no statements"):
        AnswerCorrectness(llm, embedder).score(make_sample())
    # Nothing is classified or embedded once the ground truth is unusable.
    assert len(llm.complete_calls) == 2
    assert embedder.calls == []


def test_answer_correctness_rejects_missing_categories(llm, embedder):
    llm.completions = ['{"statements": []}', '{"statements": ["a"]}', '{"TP": []}']

    with pytest.raises(LLMResponseError):
        AnswerCorrectness(llm, embedder).score(make_sample())


def test_answer_correctness_weights_must_sum_to_one(llm, embedder):
    with pytest.raises(ValueError):
        AnswerCorrectness(llm, embedder, factuality_weight=0.5, similarity_weight=0.25)


def test_judges_read_results_as_answers_to_their_questions(llm):
    # Without this, "born in 1965" wasn't credited to a name returned for
    # "Who was born in 1965 and has directed a movie?".
    llm.completions = [
        '{"statements": ["s"]}',
        judgements("verdicts", "verdict", 1),
        judgements("classifications", "attributed", 1),
    ]

    Faithfulness(llm).score(make_sample())
    ContextRecall(llm).score(make_sample())

    rule = "A result returned for a question establishes what that question asks"
    for call in llm.complete_calls[1:]:
        assert rule in call["messages"][0]["content"]


def test_answer_correctness_judges_by_meaning(llm, embedder):
    sample = make_sample()
    llm.completions = [
        '{"statements": ["a"]}',
        '{"statements": ["a"]}',
        '{"TP": [{"statement": "a", "reason": "r"}], "FP": [], "FN": []}',
    ]
    embedder.vectors = {sample.answer: [1.0, 0.0, 0.0], sample.ground_truth: [1.0, 0.0, 0.0]}

    AnswerCorrectness(llm, embedder).score(sample)

    system = llm.complete_calls[2]["messages"][0]["content"]
    assert "Judge by meaning, not wording" in system
    assert "A different fact, name or number is never supported." in system
