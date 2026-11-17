from __future__ import annotations

from pathlib import Path

import requests

from graphrag.ingestion.sources import DocumentSource

# Samuel Butler's prose translation of Homer's Odyssey on Project Gutenberg,
# the corpus the book's ch07 builds its knowledge graph from.
ODYSSEY_URL = "https://www.gutenberg.org/cache/epub/1727/pg1727.txt"


class OdysseyCorpus(DocumentSource):
    """Downloads the Odyssey (cached at ``path``) and splits it into its 24
    books (ch07's ``chunk_into_books``), one document each. ``books``
    keeps only the first ones, so recipes without a ``limit`` of their own
    index the same slice.
    """

    def __init__(self, path: Path, url: str = ODYSSEY_URL, *, books: int | None = None) -> None:
        self._path = path
        self._url = url
        self._books = books

    def load(self) -> list[str]:
        return self.load_books()[: self._books]

    def load_books(self, *, timeout: float = 30.0) -> list[str]:
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(self._url, timeout=timeout)
            response.raise_for_status()
            # Cached as raw bytes: writing ``response.text`` would decode it
            # as ISO-8859-1 (requests' fallback for this file), and text-mode
            # writes on Windows turn its \r\n line endings into \r\r\n.
            self._path.write_bytes(response.content)
        return self.split_into_books(self._path.read_text(encoding="utf-8-sig"))

    @staticmethod
    def split_into_books(text: str) -> list[str]:
        """Drops the Gutenberg header, the translator's prefaces and the
        footnotes, then splits on each ``BOOK <numeral>`` heading.
        """
        # "PREFACE TO FIRST EDITION" appears twice: in the table of contents,
        # then as the preface's own heading. The text starts after the second.
        text = text.replace("\r\n", "\n")
        body = text.split("PREFACE TO FIRST EDITION")[2].split("FOOTNOTES")[0].strip()
        return body.split("\nBOOK")[1:]
