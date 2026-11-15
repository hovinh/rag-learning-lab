from __future__ import annotations

import json
from dataclasses import dataclass

from logzero import logger

from graphrag.agent.critic import AnswerCritic
from graphrag.agent.query_updater import QueryUpdater
from graphrag.agent.router import ToolRouter
from graphrag.llm.client import LLMClient, Message, OpenAILLMClient
from graphrag.prompts import PromptRepository

# The final-answer system prompt quotes agent_tool_answer.txt's wording to
# point at the gathered answers, so change the two together.
_SYSTEM_PROMPT = "agent_main_system.txt"
_USER_PROMPT = "agent_main_user.txt"
_TOOL_ANSWER_PROMPT = "agent_tool_answer.txt"


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    contexts: list[str]


class Agent:
    """Agentic RAG (ch05): update the question, route it to a tool, have a
    critic check whether the answers so far are complete -- if not, run
    its follow-up questions through the same update/route step once --
    then answer from the gathered answers only.

    Every ``run`` starts from an empty history. The book's
    ``handle_user_input(input, answers=[])`` has a mutable default, so its
    second question silently sees the first one's answers.
    """

    def __init__(
        self,
        llm: LLMClient,
        query_updater: QueryUpdater,
        router: ToolRouter,
        critic: AnswerCritic,
        *,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
        prompts: PromptRepository | None = None,
    ) -> None:
        self._llm = llm
        self._query_updater = query_updater
        self._router = router
        self._critic = critic
        self._model = model
        prompts = prompts or PromptRepository()
        self._system_message = prompts.load(_SYSTEM_PROMPT)
        self._user_template = prompts.load(_USER_PROMPT)
        self._tool_answer_template = prompts.load(_TOOL_ANSWER_PROMPT)

    def run(self, question: str) -> str:
        return self.respond(question).answer

    def respond(self, question: str) -> AgentResponse:
        """Like ``run``, but also returns the gathered tool answers the
        final answer was written from -- what ch08 evaluates as the
        retrieved context.
        """
        history: list[Message] = []
        self._gather(question, history)

        follow_ups = self._critic.critique(question, history)
        logger.info(f"Critique: {follow_ups}")
        if follow_ups:
            self._gather(" ".join(follow_ups), history)

        answer = self._llm.complete(
            [
                {"role": "system", "content": self._system_message},
                *history,
                {"role": "user", "content": self._user_template.format(question=question)},
            ],
            model=self._model,
        )
        return AgentResponse(answer, [message["content"] for message in history])

    def _gather(self, question: str, history: list[Message]) -> None:
        updated = self._query_updater.update(question, history)
        logger.info(f"Updated question: {updated}")
        results = self._router.route(updated, history)
        # default=str: text2cypher can return values JSON has no type for
        # (e.g. dates).
        answer = json.dumps(results, default=str)
        logger.info(f"Tool results: {answer}")
        history.append(
            {
                "role": "assistant",
                "content": self._tool_answer_template.format(question=updated, answer=answer),
            }
        )
