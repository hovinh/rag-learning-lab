from __future__ import annotations

from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.llm.errors import LLMResponseError
from graphrag.llm.json_response import load_json_field
from graphrag.prompts import PromptRepository

_SYSTEM_PROMPT = "hypothetical_questions_system.txt"


class HypotheticalQuestionGenerator:
    """Writes the questions a chunk answers, for hypothetical question
    embedding: the questions are embedded instead of the chunk, so a
    user's question is matched question-to-question rather than
    question-to-passage.

    ``purpose`` says what the index is for, e.g. who asks and about what,
    and steers which facts the questions pick out of the chunk: the
    embeddings only capture what the questions ask about.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        purpose: str,
        questions_per_chunk: int = 5,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
        prompts: PromptRepository | None = None,
    ) -> None:
        if questions_per_chunk < 1:
            raise ValueError("questions_per_chunk must be at least 1")
        self._llm = llm
        self._model = model
        self._count = questions_per_chunk
        template = (prompts or PromptRepository()).load(_SYSTEM_PROMPT)
        self._system_message = template.format(purpose=purpose, count=questions_per_chunk)

    def generate(self, chunk: str) -> list[str]:
        """At most ``questions_per_chunk`` questions.

        Raises ``LLMResponseError`` if the response holds no questions.
        """
        response = self._llm.complete(
            [
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": chunk},
            ],
            model=self._model,
            json_mode=True,
        )
        questions = load_json_field(response, "questions")
        if not isinstance(questions, list):
            raise LLMResponseError(f"'questions' isn't a list: {response}")
        kept = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
        if not kept:
            raise LLMResponseError(f"Response has no questions: {response}")
        return kept[: self._count]
