from graphrag.ingestion.sources import PdfSource, TextFileSource


class FakePdfLoader:
    def __init__(self, text: str = "extracted text") -> None:
        self.text = text
        self.downloaded: tuple[str, object] | None = None

    def download(self, url, dest):
        self.downloaded = (url, dest)
        dest.write_text("pdf bytes", encoding="utf-8")
        return dest

    def extract_text(self, path):
        return self.text


def test_pdf_source_downloads_when_file_missing(tmp_path):
    path = tmp_path / "nested" / "doc.pdf"
    loader = FakePdfLoader()

    documents = PdfSource("https://example.com/doc.pdf", path, loader).load()

    assert loader.downloaded == ("https://example.com/doc.pdf", path)
    assert documents == ["extracted text"]


def test_pdf_source_skips_download_when_file_exists(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_text("already here", encoding="utf-8")
    loader = FakePdfLoader()

    documents = PdfSource("https://example.com/doc.pdf", path, loader).load()

    assert loader.downloaded is None
    assert documents == ["extracted text"]


def test_text_file_source_loads_one_document_per_file(tmp_path):
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("Miloš", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    assert TextFileSource(first, second).load() == ["Miloš", "second"]
