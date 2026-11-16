from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Usage:
    """API usage counted by a ``UsageMeter``. Subtracting two snapshots
    gives the usage between them -- what a trace section reports.
    """

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_calls: int = 0
    embedding_tokens: int = 0

    @property
    def llm_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __sub__(self, other: Usage) -> Usage:
        return Usage(
            self.llm_calls - other.llm_calls,
            self.prompt_tokens - other.prompt_tokens,
            self.completion_tokens - other.completion_tokens,
            self.embedding_calls - other.embedding_calls,
            self.embedding_tokens - other.embedding_tokens,
        )

    def describe(self) -> str:
        """One line, e.g. ``7 LLM calls · 3.2k tokens · 1 embedding call``;
        empty if nothing was called.
        """
        parts = []
        if self.llm_calls:
            parts.append(
                f"{plural(self.llm_calls, 'LLM call')} · {format_count(self.llm_tokens)} tokens"
            )
        if self.embedding_calls:
            parts.append(plural(self.embedding_calls, "embedding call"))
        return " · ".join(parts)


class UsageMeter:
    """Counts LLM and embedding calls and their tokens. Pass one to
    ``OpenAILLMClient``/``OpenAIEmbedder`` (``RunLogging.usage`` is the
    run's meter) and every call they make is recorded.
    """

    def __init__(self) -> None:
        self._usage = Usage()

    def record_completion(self, prompt_tokens: int, completion_tokens: int) -> None:
        u = self._usage
        self._usage = Usage(
            u.llm_calls + 1,
            u.prompt_tokens + prompt_tokens,
            u.completion_tokens + completion_tokens,
            u.embedding_calls,
            u.embedding_tokens,
        )

    def record_embedding(self, tokens: int) -> None:
        u = self._usage
        self._usage = Usage(
            u.llm_calls,
            u.prompt_tokens,
            u.completion_tokens,
            u.embedding_calls + 1,
            u.embedding_tokens + tokens,
        )

    def snapshot(self) -> Usage:
        return self._usage


def format_count(count: int) -> str:
    """``950`` -> ``950``, ``3210`` -> ``3.2k``, ``1250000`` -> ``1.2M``."""
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1000:.1f}k"
    return f"{count / 1_000_000:.1f}M"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"
