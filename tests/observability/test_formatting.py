from graphrag.observability import format_count, format_duration, preview, render_table
from graphrag.observability.formatting import to_text, wrap


def test_preview_collapses_whitespace_and_truncates():
    assert preview("a  b\n c", 100) == "a b c"
    assert preview("x" * 20, 10) == "x" * 9 + "…"


def test_preview_keeps_keys_of_a_single_row():
    assert (
        preview([{"directorCount": 10, "producerCount": 4}]) == "directorCount=10, producerCount=4"
    )


def test_preview_counts_rows_and_list_values():
    rows = [{"title": "A", "cast": ["x", "y"]}, {"title": "B", "cast": ["z"]}]

    assert preview(rows) == "2 rows: A, [2 items] | B, [1 item]"


def test_preview_of_no_rows():
    assert preview([]) == "0 rows"


def test_to_text_keeps_non_ascii_characters():
    assert to_text([{"name": "Miloš Forman"}]) == '{"name": "Miloš Forman"}'
    assert to_text({"name": "Miloš"}) == '{\n  "name": "Miloš"\n}'


def test_render_table_aligns_columns_and_right_aligns_numbers():
    table = render_table([{"name": "Tom Hanks", "movies": 8}, {"name": "Ed", "movies": 12}])

    assert table.splitlines() == [
        "name       movies",
        "─────────  ──────",
        "Tom Hanks       8",
        "Ed             12",
    ]


def test_to_text_keeps_a_list_of_scalars_on_one_line():
    assert to_text([153, 256, 4156]) == "[153, 256, 4156]"


def test_wrap_breaks_only_long_lines_at_words():
    text = "short line\n" + "word " * 30

    wrapped = wrap(text, 40).splitlines()

    assert wrapped[0] == "short line"
    assert all(len(line) <= 40 for line in wrapped)
    assert " ".join(wrapped[1:]) == ("word " * 30).strip()


def test_render_table_shows_whole_floats_without_decimals():
    assert render_table([{"score": 95.0}, {"score": 0.5}]).splitlines()[2:] == ["   95", "0.500"]


def test_render_table_without_rows():
    assert render_table([]) == "(no rows)"


def test_format_duration_and_count():
    assert format_duration(4.24) == "4.2s"
    assert format_duration(125) == "2m05s"
    assert format_count(950) == "950"
    assert format_count(3210) == "3.2k"
    assert format_count(1_250_000) == "1.2M"
