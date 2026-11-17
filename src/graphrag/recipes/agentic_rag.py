from __future__ import annotations

from collections.abc import Sequence

from graphrag.agent import (
    Agent,
    AgentScope,
    AnswerCritic,
    QueryDecomposer,
    QueryPlanner,
    QueryUpdater,
    Tool,
    ToolRouter,
)
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.recipes.models import RagAnswer


class AgenticRag:
    """ch05 as a recipe: an ``Agent`` wired from any tools.
    Each question is rewritten (or, with ``decompose``, split as in ch08),
    routed to a tool, critiqued for gaps with one follow-up round, then
    answered from what the tools returned.

    ``scope`` words the prompts for the domain, and ``data_description``
    (e.g. ``GraphSchema.to_prompt()``) tells the critic what data exists.
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: Sequence[Tool],
        *,
        scope: AgentScope | None = None,
        data_description: str | None = None,
        decompose: bool = False,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
    ) -> None:
        planner: QueryPlanner = (
            QueryDecomposer(llm, scope=scope, model=model)
            if decompose
            else QueryUpdater(llm, scope=scope, model=model)
        )
        self.agent = Agent(
            llm,
            planner,
            ToolRouter(llm, tools, model=model),
            AnswerCritic(llm, data_description=data_description, scope=scope, model=model),
            model=model,
        )

    def query(self, question: str) -> RagAnswer:
        """The agent traces its own steps, ending with the answer."""
        response = self.agent.respond(question)
        return RagAnswer(question, response.answer, response.contexts)
