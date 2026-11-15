from __future__ import annotations

import csv
from pathlib import Path

from graphrag.evaluation.models import BenchmarkCase


def load_benchmark(path: Path) -> list[BenchmarkCase]:
    """Read a ``question;cypher`` CSV, the book's ``benchmark_data.csv``
    format. Cypher may span lines inside a quoted field.
    """
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        missing = {"question", "cypher"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} has no {sorted(missing)} column(s)")
        return [BenchmarkCase(row["question"], row["cypher"].strip()) for row in reader]
