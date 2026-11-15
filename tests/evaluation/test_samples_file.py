import pytest

from graphrag.evaluation.models import EvaluationSample
from graphrag.evaluation.samples_file import load_samples, save_samples


def test_samples_round_trip(tmp_path):
    samples = [
        EvaluationSample(
            "Who directed it?", '[{"p.name": "Miloš Forman"}]', "Miloš Forman.", ["c"], 1.5
        ),
        EvaluationSample("Hello", "[]", "", [], 0.2, "RuntimeError('boom')"),
    ]
    path = tmp_path / "samples.json"

    save_samples(samples, path)

    assert load_samples(path) == samples
    assert "Miloš Forman" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("content", ['{"question": "x"}', '[{"question": "x"}]'])
def test_load_samples_rejects_malformed_files(tmp_path, content):
    path = tmp_path / "samples.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        load_samples(path)
