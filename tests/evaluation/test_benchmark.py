import pytest

from graphrag.evaluation.benchmark import load_benchmark
from graphrag.evaluation.models import BenchmarkCase


def test_load_benchmark_reads_semicolon_rows_with_multiline_cypher(tmp_path):
    path = tmp_path / "benchmark.csv"
    path.write_text(
        "question;cypher\n"
        'Hello;"RETURN ""greeting"""\n'
        'Who acted in Top Gun?;"MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: ""Top Gun""})\n'
        "RETURN p.name\n"
        '"\n',
        encoding="utf-8",
        newline="\n",
    )

    assert load_benchmark(path) == [
        BenchmarkCase("Hello", 'RETURN "greeting"'),
        BenchmarkCase(
            "Who acted in Top Gun?",
            'MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: "Top Gun"})\nRETURN p.name',
        ),
    ]


def test_load_benchmark_rejects_missing_columns(tmp_path):
    path = tmp_path / "benchmark.csv"
    path.write_text("question\nHello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cypher"):
        load_benchmark(path)


def test_load_benchmark_reads_written_answers_with_categories(tmp_path):
    path = tmp_path / "benchmark.csv"
    path.write_text(
        "question;answer;category\n"
        'Who is Telemachus?;"Ulysses\' son;\nhe lives in Ithaca.";specific\n'
        "What is it about?;A homecoming.;broad\n",
        encoding="utf-8",
        newline="\n",
    )

    assert load_benchmark(path) == [
        BenchmarkCase(
            "Who is Telemachus?", answer="Ulysses' son;\nhe lives in Ithaca.", category="specific"
        ),
        BenchmarkCase("What is it about?", answer="A homecoming.", category="broad"),
    ]


def test_load_benchmark_rejects_both_ground_truth_columns(tmp_path):
    path = tmp_path / "benchmark.csv"
    path.write_text("question;cypher;answer\nHi;RETURN 1;one\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        load_benchmark(path)


@pytest.mark.parametrize("truth", [{}, {"cypher": "RETURN 1", "answer": "1"}])
def test_benchmark_case_needs_exactly_one_ground_truth(truth):
    with pytest.raises(ValueError):
        BenchmarkCase("?", **truth)
