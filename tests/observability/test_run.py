import logging

import pytest

from graphrag.observability import RunLogging, Trace, Usage, UsageMeter


def test_run_writes_header_trace_and_footer_to_its_log_file(tmp_path):
    with RunLogging("demo", tmp_path, title="Demo run", details={"model": "m"}) as run:
        run.usage.record_completion(1000, 50)
        run.usage.record_embedding(20)
        Trace("graphrag.tests.run").field("answer", "42")
        Trace("graphrag.tests.run").debug("full payload")

    text = (tmp_path / "demo.log").read_text(encoding="utf-8")
    assert " Demo run" in text
    assert "model     m" in text
    assert "answer    42" in text
    # The file always gets DEBUG lines, whatever the console shows.
    assert "full payload" in text
    assert "✓ finished in" in text
    assert "1 call · 1.1k tokens (1.0k in, 50 out)" in text
    assert "embeddings  1 call · 20 tokens" in text


def test_run_restores_logging_state(tmp_path):
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level

    with RunLogging("demo", tmp_path):
        pass

    assert root.handlers == handlers
    assert root.level == level


def test_run_logs_an_escaping_exception_and_exits_with_status_1(tmp_path):
    with pytest.raises(SystemExit) as exit_info, RunLogging("demo", tmp_path):
        raise ValueError("bad input")

    assert exit_info.value.code == 1
    text = (tmp_path / "demo.log").read_text(encoding="utf-8")
    assert "ValueError: bad input" in text
    assert "✗ failed in" in text


def test_run_overwrites_the_previous_log(tmp_path):
    with RunLogging("demo", tmp_path):
        Trace("graphrag.tests.run").info("first run")
    with RunLogging("demo", tmp_path):
        pass

    assert "first run" not in (tmp_path / "demo.log").read_text(encoding="utf-8")


def test_usage_meter_snapshots_subtract():
    meter = UsageMeter()
    meter.record_completion(10, 5)
    before = meter.snapshot()
    meter.record_completion(100, 20)
    meter.record_embedding(7)

    used = meter.snapshot() - before

    assert used == Usage(1, 100, 20, 1, 7)
    assert used.describe() == "1 LLM call · 120 tokens · 1 embedding call"
    assert Usage().describe() == ""


def test_add_arguments_adds_process_name_and_verbose():
    import argparse

    parser = argparse.ArgumentParser()
    RunLogging.add_arguments(parser, default_process_name="ch99")

    assert vars(parser.parse_args([])) == {"process_name": "ch99", "verbose": False}
    assert vars(parser.parse_args(["--process-name", "x", "--verbose"])) == {
        "process_name": "x",
        "verbose": True,
    }
