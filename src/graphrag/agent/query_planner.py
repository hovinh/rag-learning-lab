from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from graphrag.llm.client import Message


class QueryPlanner(ABC):
    """Turns a question into the questions the agent routes to its tools,
    one tool round each: ch05 rewrites it into one (``QueryUpdater``), ch08
    splits it into several (``QueryDecomposer``).
    """

    @abstractmethod
    def plan(self, question: str, history: Sequence[Message] = ()) -> list[str]:
        """Raises ``LLMResponseError`` if the response has no usable questions."""
