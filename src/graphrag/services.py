from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from graphrag.llm.client import OpenAILLMClient
from graphrag.observability import UsageMeter
from graphrag.retrieval.embedder import OpenAIEmbedder
from graphrag.settings import Settings


@dataclass(frozen=True)
class OpenAIServices:
    """The LLM client and embedder a pipeline needs, sharing one OpenAI
    client and recording onto one ``UsageMeter`` -- the wiring every
    composition root otherwise repeats.
    """

    llm: OpenAILLMClient
    embedder: OpenAIEmbedder

    @classmethod
    def create(cls, settings: Settings, *, usage: UsageMeter | None = None) -> OpenAIServices:
        client = OpenAI(api_key=settings.openai_api_key)
        return cls(
            llm=OpenAILLMClient(client, usage=usage),
            embedder=OpenAIEmbedder(client, usage=usage),
        )
