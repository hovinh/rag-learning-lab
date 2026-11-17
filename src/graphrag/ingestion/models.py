from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A chunk of one document: ``document_index`` is the document's
    position in what the ``DocumentSource`` loaded, ``index`` the chunk's
    position within that document.
    """

    document_index: int
    index: int
    text: str
