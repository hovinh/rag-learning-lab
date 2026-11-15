from __future__ import annotations

from dataclasses import dataclass

from graphrag.graph.cypher_store import CypherGraphStore
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.llm.errors import LLMResponseError
from graphrag.llm.json_response import load_json_field
from graphrag.prompts import PromptRepository

_MAP_PROMPT = "global_map_system.txt"
_REDUCE_PROMPT = "global_reduce_system.txt"

_COMMUNITY_SUMMARIES = """
MATCH (c:Community)
WHERE c.rating >= $rating
RETURN c.summary AS summary
ORDER BY c.id
"""


@dataclass(frozen=True)
class KeyPoint:
    """One point a map call found in a community report, with the LLM's
    0-100 rating of how much it helps answer the question.
    """

    description: str
    score: float


class GlobalSearch:
    """Answers broad questions about the whole corpus ("What is this story
    about?") by map-reduce over the community reports (ch07's
    ``global_retriever``). ``map`` asks the question of each report rated at
    least ``rating_threshold``; ``reduce`` merges the points into an answer.
    """

    def __init__(
        self,
        store: CypherGraphStore,
        llm: LLMClient,
        *,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
        prompts: PromptRepository | None = None,
        rating_threshold: float = 5.0,
        response_type: str = "multiple paragraphs",
    ) -> None:
        self._store = store
        self._llm = llm
        self._model = model
        prompts = prompts or PromptRepository()
        self._map_template = prompts.load(_MAP_PROMPT)
        self._reduce_template = prompts.load(_REDUCE_PROMPT)
        self._rating_threshold = rating_threshold
        self._response_type = response_type

    def search(self, question: str) -> str:
        return self.reduce(question, self.map(question))

    def map(self, question: str) -> list[KeyPoint]:
        """Returns the points from every report, most important first.
        Points scored 0 ("I don't know") are dropped: the reduce prompt says
        its reports are ranked by importance, which the book never does.

        Raises ``LLMResponseError`` if a map response isn't the JSON the
        prompt asks for.
        """
        points: list[KeyPoint] = []
        for report in self._store.query(_COMMUNITY_SUMMARIES, {"rating": self._rating_threshold}):
            response = self._llm.complete(
                [
                    {
                        "role": "system",
                        "content": self._map_template.format(context_data=report["summary"]),
                    },
                    {"role": "user", "content": question},
                ],
                model=self._model,
                json_mode=True,
            )
            points.extend(_parse_points(response))
        relevant = [point for point in points if point.score > 0]
        return sorted(relevant, key=lambda point: point.score, reverse=True)

    def reduce(self, question: str, points: list[KeyPoint]) -> str:
        report_data = "\n\n".join(
            f"----Analyst {i}----\nImportance Score: {point.score:g}\n{point.description}"
            for i, point in enumerate(points, start=1)
        )
        system_message = self._reduce_template.format(
            report_data=report_data, response_type=self._response_type
        )
        return self._llm.complete(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
            ],
            model=self._model,
        )


def _parse_points(response: str) -> list[KeyPoint]:
    raw_points = load_json_field(response, "points")
    if not isinstance(raw_points, list):
        raise LLMResponseError(f"'points' isn't a list: {response}")
    try:
        return [KeyPoint(str(point["description"]), float(point["score"])) for point in raw_points]
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMResponseError(f"Unusable key point in: {response}") from exc
