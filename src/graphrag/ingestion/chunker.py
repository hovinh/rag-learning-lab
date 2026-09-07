from __future__ import annotations


class TextChunker:
    """Splits text into overlapping chunks (ch02), ported from the book's
    free-standing ``chunk_text`` helper in ``utils.py``.
    """

    def __init__(
        self,
        chunk_size: int,
        overlap: int,
        split_on_whitespace_only: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.split_on_whitespace_only = split_on_whitespace_only

    def split(self, text: str) -> list[str]:
        chunks = []
        index = 0

        while index < len(text):
            if self.split_on_whitespace_only:
                prev_whitespace = 0
                left_index = index - self.overlap
                while left_index >= 0:
                    if text[left_index] == " ":
                        prev_whitespace = left_index
                        break
                    left_index -= 1
                next_whitespace = text.find(" ", index + self.chunk_size)
                if next_whitespace == -1:
                    next_whitespace = len(text)
                chunk = text[prev_whitespace:next_whitespace].strip()
                chunks.append(chunk)
                index = next_whitespace + 1
            else:
                start = max(0, index - self.overlap + 1)
                end = min(index + self.chunk_size + self.overlap, len(text))
                chunk = text[start:end].strip()
                chunks.append(chunk)
                index += self.chunk_size

        return chunks
