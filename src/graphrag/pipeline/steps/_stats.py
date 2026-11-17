from __future__ import annotations

from collections.abc import Sequence

from graphrag.llm.token_counter import TokenCounter


def token_stats(token_counter: TokenCounter, texts: Sequence[str]) -> str:
    """``min 12, max 3400, mean 870`` -- how big texts are, as the book
    prints before chunking them.
    """
    tokens = [token_counter.count(text) for text in texts]
    if not tokens:
        return "no texts"
    return f"min {min(tokens)}, max {max(tokens)}, mean {sum(tokens) / len(tokens):.0f}"
