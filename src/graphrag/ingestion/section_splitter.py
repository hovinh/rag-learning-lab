from __future__ import annotations

import re

_TITLE_PATTERN = re.compile(r"(\n\d+[A-Z]?\. {1,3}.{0,60}\n)", re.DOTALL)


class SectionSplitter:
    """Splits text into sections at numbered-title boundaries (ch03),
    ported from the book's free-standing ``split_text_by_titles`` helper.

    Titles look like "1. Introduction" or "113B. Sound reproduction
    system..." on their own line; everything up to the next title (or EOF)
    becomes that title's section. Text before the first title is kept as
    its own leading section.
    """

    def split(self, text: str) -> list[str]:
        titles = _TITLE_PATTERN.findall(text)
        parts = _TITLE_PATTERN.split(text)

        sections = [parts[0]]
        for i in range(1, len(titles) + 1):
            sections.append(parts[i * 2 - 1].strip() + "\n" + parts[i * 2].strip())
        return sections
