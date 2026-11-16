from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from openai import OpenAI

from graphrag.observability.usage import UsageMeter


class Embedder(ABC):
    """Text-to-vector interface. ``dimensions`` lets a GraphStore size its
    vector column without hardcoding a model's embedding width.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...


class OpenAIEmbedder(Embedder):
    _MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(
        self,
        client: OpenAI,
        model: str = "text-embedding-3-small",
        *,
        usage: UsageMeter | None = None,
    ) -> None:
        if model not in self._MODEL_DIMENSIONS:
            raise ValueError(f"Unknown embedding model: {model}")
        self._client = client
        self._model = model
        self._usage = usage

    @property
    def dimensions(self) -> int:
        return self._MODEL_DIMENSIONS[self._model]

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        if self._usage is not None:
            self._usage.record_embedding(response.usage.total_tokens)
        return [item.embedding for item in response.data]
