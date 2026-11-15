from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from graphrag.evaluation.models import EvaluationSample


def save_samples(samples: Sequence[EvaluationSample], path: Path) -> None:
    """Write answered samples as JSON, so they can be re-scored later
    without asking the system under test again (e.g. after changing a
    judge prompt).
    """
    data = [dataclasses.asdict(sample) for sample in samples]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_samples(path: Path) -> list[EvaluationSample]:
    """Read samples written by ``save_samples``.

    Raises ``ValueError`` if the file isn't a list of samples.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        try:
            return [EvaluationSample(**item) for item in data]
        except TypeError as exc:
            raise ValueError(f"{path} has a malformed sample: {exc}") from exc
    raise ValueError(f"{path} doesn't hold a list of samples")
