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
