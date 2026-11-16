import io
import logging
import re

import pytest

from graphrag.observability import Trace, TraceFormatter, UsageMeter
from graphrag.observability._context import ACTIVE_METER

TIME = re.compile(r"^\d\d:\d\d:\d\d ", re.MULTILINE)


@pytest.fixture
def capture():
    """A ``Trace`` whose output is rendered by ``TraceFormatter`` into a
    buffer; ``read()`` returns it with timestamps blanked out.
    """
    logger = logging.getLogger("graphrag.tests.trace")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(TraceFormatter())
    logger.addHandler(handler)

    def read() -> str:
        return TIME.sub("HH:MM:SS ", buffer.getvalue())

    yield Trace(logger), read
    logger.removeHandler(handler)


def test_field_aligns_multiline_value_under_itself(capture):
    trace, read = capture

    trace.field("answer", "line one\nline two")

    assert read() == "HH:MM:SS answer    line one\n         " + " " * 10 + "line two\n"


def test_section_indents_its_lines_and_closes_with_duration(capture):
    trace, read = capture

    with trace.section("Question 1/2"):
        trace.field("plan", "a question")

    lines = read().splitlines()
    assert lines[0] == "HH:MM:SS ┌ Question 1/2"
    assert lines[1] == "HH:MM:SS │ plan      a question"
    assert re.fullmatch(r"HH:MM:SS └ \d+\.\ds", lines[2])


def test_section_with_nothing_logged_collapses_to_one_line(capture):
    trace, read = capture

    with trace.section("Seed"):
        pass

    assert re.fullmatch(r"HH:MM:SS • Seed · \d+\.\ds\n", read())


def test_nested_section_opens_its_parents_first(capture):
    trace, read = capture

    with trace.section("outer"), trace.section("inner"):
        trace.info("hello")

    lines = read().splitlines()
    assert lines[:3] == [
        "HH:MM:SS ┌ outer",
        "HH:MM:SS │ ┌ inner",
        "HH:MM:SS │ │ hello",
    ]
    assert lines[3].startswith("HH:MM:SS │ └ ")
    assert lines[4].startswith("HH:MM:SS └ ")


def test_section_reports_usage_made_inside_it(capture):
    trace, read = capture
    meter = UsageMeter()
    token = ACTIVE_METER.set(meter)
    try:
        with trace.section("Gather"):
            meter.record_completion(3000, 200)
            meter.record_completion(100, 10)
    finally:
        ACTIVE_METER.reset(token)

    assert read().rstrip().endswith("· 2 LLM calls · 3.3k tokens")


def test_failed_section_says_so_and_reraises(capture):
    trace, read = capture

    with pytest.raises(RuntimeError), trace.section("boom"):
        trace.info("starting")
        raise RuntimeError("no")

    assert "└ failed after" in read()


def test_summary_previews_rows_and_adds_full_rows_at_debug(capture):
    trace, read = capture
    rows = [{"title": "The Matrix", "cast": ["Keanu"]}, {"title": "Top Gun", "cast": ["Tom"]}]

    trace.summary("result", rows)

    lines = read().splitlines()
    assert lines[0] == "HH:MM:SS result    2 rows: The Matrix, [1 item] | Top Gun, [1 item]"
    assert lines[1].endswith('{"title": "The Matrix", "cast": ["Keanu"]}')
    assert lines[2].endswith('{"title": "Top Gun", "cast": ["Tom"]}')


def test_summary_that_says_everything_is_not_repeated(capture):
    trace, read = capture

    trace.summary("result", [{"movieCount": 14}])

    assert read() == "HH:MM:SS result    movieCount=14\n"


def test_summary_of_short_text_logs_it_once(capture):
    trace, read = capture

    trace.summary("result", "Hello!")

    assert read() == "HH:MM:SS result    Hello!\n"


def test_fields_share_one_value_column(capture):
    trace, read = capture

    trace.fields({"answer_correctness": "0.921", "faithfulness": "1.000"})

    assert read().splitlines() == [
        "HH:MM:SS answer_correctness  0.921",
        "HH:MM:SS faithfulness        1.000",
    ]


def test_bullets_and_their_empty_text(capture):
    trace, read = capture

    trace.bullets("critique", ["first?", "second?"])
    trace.bullets("critique", [], empty="nothing missing")

    assert read().splitlines() == [
        "HH:MM:SS critique  • first?",
        "                   • second?",
        "HH:MM:SS critique  nothing missing",
    ]


def test_warning_gets_a_marker(capture):
    trace, read = capture

    trace.warning("careful")

    assert read() == "HH:MM:SS ⚠ careful\n"


def test_progress_reports_quarters_at_info_and_every_item_at_debug(capture):
    trace, read = capture
    trace._logger.setLevel(logging.INFO)

    progress = trace.progress("chunk", 8)
    for _ in range(8):
        progress.advance()

    assert [line.split()[2] for line in read().splitlines()] == ["2/8", "4/8", "6/8", "8/8"]


def test_trace_stays_silent_without_handlers_configured():
    logger = logging.getLogger("graphrag.tests.silent")
    assert not logger.isEnabledFor(logging.INFO)

    Trace(logger).info("nobody sees this")
