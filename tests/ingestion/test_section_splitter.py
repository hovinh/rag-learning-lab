from graphrag.ingestion.section_splitter import SectionSplitter

# Body lines are deliberately long (>60 chars): the title regex only looks
# up to 60 chars past the title text for the next newline, so a short body
# line would get swallowed into the title match itself (see ch03.ipynb's
# ``split_text_by_titles`` -- this is a faithful port of that quirk, not a
# design choice made here).
TEXT = (
    "Leading text before any numbered title.\n"
    "1. Introduction\n"
    "This is a long first line of the introduction section that runs well past sixty characters.\n"
    "A second line of the introduction section.\n"
    "2A. Background\n"
    "This is a long first line of the background section that also runs well past sixty characters.\n"
    "A second line of the background section."
)


def test_split_separates_leading_text_from_first_titled_section():
    sections = SectionSplitter().split(TEXT)

    assert sections[0] == "Leading text before any numbered title."


def test_split_keeps_title_with_its_section_body():
    sections = SectionSplitter().split(TEXT)

    assert len(sections) == 3
    assert sections[1] == (
        "1. Introduction\n"
        "This is a long first line of the introduction section that runs well past sixty characters.\n"
        "A second line of the introduction section."
    )
    assert sections[2] == (
        "2A. Background\n"
        "This is a long first line of the background section that also runs well past sixty characters.\n"
        "A second line of the background section."
    )


def test_split_with_no_titles_returns_whole_text_as_one_section():
    text = "Just a plain paragraph with no numbered titles at all."

    sections = SectionSplitter().split(text)

    assert sections == [text]
