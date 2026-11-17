from __future__ import annotations

import csv
from pathlib import Path

from graphrag.evaluation.models import BenchmarkCase


def load_benchmark(path: Path) -> list[BenchmarkCase]:
    """Read a semicolon-separated benchmark CSV with a ``question`` column
    and a ground-truth column: ``cypher`` (the book's
    ``benchmark_data.csv`` format) or a written ``answer``. An optional
    ``category`` column groups the questions. Fields may span lines
    inside quotes.
    """
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        columns = set(reader.fieldnames or [])
        if "question" not in columns:
            raise ValueError(f"{path} has no 'question' column")
        truth_columns = columns & {"cypher", "answer"}
        if len(truth_columns) != 1:
            raise ValueError(f"{path} needs exactly one of a 'cypher' or 'answer' column")
        [truth] = truth_columns
        return [
            BenchmarkCase(
                row["question"],
                **{truth: row[truth].strip()},
                category=(row.get("category") or "").strip(),
            )
            for row in reader
        ]
