from graphrag.domains.odyssey.corpus import OdysseyCorpus

GUTENBERG_TEXT = (
    "The Project Gutenberg eBook of The Odyssey\r\n"
    "Contents\r\nPREFACE TO FIRST EDITION\r\nBOOK I\r\nBOOK II\r\n"
    "PREFACE TO FIRST EDITION\r\nThe translator's preface.\r\n"
    "BOOK I\r\nTHE GODS IN COUNCIL\r\nTell me, O Muse.\r\n"
    "BOOK II\r\nASSEMBLY OF THE PEOPLE OF ITHACA\r\nNow when the child of morning.\r\n"
    "FOOTNOTES\r\n[1] A footnote.\r\nEnd of the Project Gutenberg eBook\r\n"
)


def test_split_into_books_keeps_only_the_books():
    books = OdysseyCorpus.split_into_books(GUTENBERG_TEXT)

    assert books == [
        " I\nTHE GODS IN COUNCIL\nTell me, O Muse.",
        " II\nASSEMBLY OF THE PEOPLE OF ITHACA\nNow when the child of morning.",
    ]


class FakeResponse:
    content = ("﻿" + GUTENBERG_TEXT.replace("Muse.", "Muse—")).encode("utf-8")

    def raise_for_status(self):
        pass


def test_load_books_downloads_and_caches_the_text_unchanged(tmp_path, monkeypatch):
    requested = []
    monkeypatch.setattr(
        "graphrag.domains.odyssey.corpus.requests.get",
        lambda url, timeout: requested.append(url) or FakeResponse(),
    )
    corpus = OdysseyCorpus(tmp_path / "data" / "odyssey.txt", url="http://example.test/odyssey")

    first = corpus.load_books()
    second = corpus.load_books()

    assert requested == ["http://example.test/odyssey"]
    assert first == second
    assert first[0] == " I\nTHE GODS IN COUNCIL\nTell me, O Muse—"


def test_load_books_reads_a_cached_copy(tmp_path):
    path = tmp_path / "odyssey.txt"
    path.write_text(GUTENBERG_TEXT, encoding="utf-8")

    books = OdysseyCorpus(path, url="http://unused.invalid").load_books()

    assert len(books) == 2


def test_load_returns_the_books_as_documents(tmp_path):
    path = tmp_path / "odyssey.txt"
    path.write_text(GUTENBERG_TEXT, encoding="utf-8")
    corpus = OdysseyCorpus(path, url="http://unused.invalid")

    assert corpus.load() == corpus.load_books()
