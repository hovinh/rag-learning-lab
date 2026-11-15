from __future__ import annotations

import tiktoken

from graphrag.llm.client import OpenAILLMClient


class TokenCounter:
    """Counts the tokens ``model``'s tokenizer splits a text into (the
    book's ``num_tokens_from_string``), e.g. to check that sections or
    chunks fit a context window.

    The book counts with gpt-4's tokenizer (``cl100k_base``). This defaults
    to the model the rest of the code calls, gpt-4o (``o200k_base``), which
    gives slightly fewer tokens for English text (about 0.6% fewer on the
    Odyssey).
    """

    def __init__(self, model: str = OpenAILLMClient.DEFAULT_MODEL) -> None:
        self._encoding = tiktoken.encoding_for_model(model)

    def count(self, text: str) -> int:
        # disallowed_special=(): count "<|endoftext|>" in a document as
        # plain text instead of raising, which tiktoken does by default.
        return len(self._encoding.encode(text, disallowed_special=()))
