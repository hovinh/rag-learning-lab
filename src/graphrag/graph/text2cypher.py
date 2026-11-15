from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from graphrag.graph.models import GraphSchema
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.prompts import PromptRepository

_INSTRUCTIONS_PROMPT = "text2cypher_instructions.txt"
_SCHEMA_PROMPT = "text2cypher_schema.txt"
_TERMINOLOGY_PROMPT = "text2cypher_terminology.txt"
_EXAMPLES_PROMPT = "text2cypher_examples.txt"
_QUESTION_PROMPT = "text2cypher_question.txt"
_CORRECTION_PROMPT = "text2cypher_correction.txt"


@dataclass(frozen=True)
class CypherExample:
    """A few-shot question -> Cypher pair shown to the LLM."""

    question: str
    cypher: str


class Text2Cypher:
    """Generates a Cypher query that answers a question (ch04). The prompt
    is assembled from the book's sections -- instructions, schema,
    terminology mapping, few-shot examples, question -- leaving out
    terminology/examples when none are given.

    Unlike the book's class, every section except the question is fixed at
    construction instead of set through ``set_prompt_section``, so one
    instance can serve many questions (e.g. as a ch05 agent tool).
    """

    def __init__(
        self,
        llm: LLMClient,
        schema: GraphSchema,
        *,
        terminology: str | None = None,
        examples: Sequence[CypherExample] = (),
        model: str = OpenAILLMClient.DEFAULT_MODEL,
        prompts: PromptRepository | None = None,
    ) -> None:
        self._llm = llm
        self._model = model
        prompts = prompts or PromptRepository()

        sections = [
            prompts.load(_INSTRUCTIONS_PROMPT),
            prompts.load(_SCHEMA_PROMPT).format(schema=schema.to_prompt()),
        ]
        if terminology:
            sections.append(
                prompts.load(_TERMINOLOGY_PROMPT).format(terminology=terminology.strip())
            )
        if examples:
            formatted = "\n".join(f"Question: {e.question}\nCypher: {e.cypher}" for e in examples)
            sections.append(prompts.load(_EXAMPLES_PROMPT).format(examples=formatted))
        self._prompt_prefix = "\n\n".join(sections)
        self._question_template = prompts.load(_QUESTION_PROMPT)
        self._correction_template = prompts.load(_CORRECTION_PROMPT)

    def build_prompt(self, question: str) -> str:
        question_section = self._question_template.format(question=question)
        return f"{self._prompt_prefix}\n\n{question_section}"

    def generate(self, question: str) -> str:
        return self._complete(self.build_prompt(question))

    def correct(self, question: str, cypher: str, error: str) -> str:
        """Regenerate ``cypher`` for ``question`` after the database
        rejected it with ``error`` -- the book has no such step.
        """
        correction = self._correction_template.format(cypher=cypher, error=error)
        return self._complete(f"{self.build_prompt(question)}\n\n{correction}")

    def _complete(self, prompt: str) -> str:
        return self._llm.complete([{"role": "user", "content": prompt}], model=self._model).strip()
