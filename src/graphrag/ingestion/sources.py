from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from graphrag.ingestion.pdf_loader import PdfLoader


class DocumentSource(ABC):
    """Where a pipeline's documents come from: each document is the text
    of one unit to chunk or extract from (a PDF, a file, a book). Pipelines
    depend on this, so the same ingestion runs over any corpus.
    """

    @abstractmethod
    def load(self) -> list[str]: ...


class PdfSource(DocumentSource):
    """One PDF, downloaded from ``url`` to ``path`` unless it's already
    there (ch02/ch03).
    """

    def __init__(self, url: str, path: Path, loader: PdfLoader | None = None) -> None:
        self._url = url
        self._path = path
        self._loader = loader or PdfLoader()

    def load(self) -> list[str]:
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._loader.download(self._url, self._path)
        return [self._loader.extract_text(self._path)]


class TextFileSource(DocumentSource):
    """UTF-8 text files, one document each (ch06's contract)."""

    def __init__(self, *paths: Path) -> None:
        self._paths = paths

    def load(self) -> list[str]:
        return [path.read_text(encoding="utf-8") for path in self._paths]
