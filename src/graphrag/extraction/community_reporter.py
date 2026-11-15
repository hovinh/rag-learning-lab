from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graphrag.graph.cypher_store import CypherGraphStore
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.llm.errors import LLMResponseError
from graphrag.llm.json_response import load_json_object
from graphrag.prompts import PromptRepository

_REPORT_PROMPT = "community_report.txt"

_COMMUNITIES = """
MATCH (e:Entity)
WHERE e.community_id IS NOT NULL
WITH e.community_id AS id, count(*) AS size
WHERE size >= $min_size
RETURN id
ORDER BY id
"""

_COMMUNITY_ENTITIES = """
MATCH (e:Entity)
WHERE e.community_id = $community_id
RETURN e.name AS id, e.summary AS description, e.type AS type
ORDER BY id
"""

# The book collects these with ``apoc.path.subgraphAll`` restricted to the
# community's nodes; Kuzu has no APOC, and the edges with both ends in the
# community are the same set.
_COMMUNITY_RELATIONSHIPS = """
MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
WHERE s.community_id = $community_id AND t.community_id = $community_id
RETURN s.name AS source, t.name AS target, r.description AS description
ORDER BY source, target
"""

_MERGE_COMMUNITY = """
MERGE (c:Community {id: $community_id})
SET c.title = $title,
    c.summary = $summary,
    c.rating = $rating,
    c.rating_explanation = $rating_explanation
WITH c
MATCH (e:Entity)
WHERE e.community_id = $community_id
MERGE (e)-[:IN_COMMUNITY]->(c)
"""


@dataclass(frozen=True)
class CommunityReport:
    community_id: int
    title: str
    summary: str
    rating: float
    rating_explanation: str


class CommunityReporter:
    """Has the LLM write a report (title, summary, importance rating) for
    each community from its entities and relationships, and stores it as a
    ``(:Community)`` node that every member links to with ``IN_COMMUNITY``
    (ch07's ``get_summarize_community_prompt`` / ``import_community_query``).
    These reports are what global search reads.
    """

    def __init__(
        self,
        store: CypherGraphStore,
        llm: LLMClient,
        *,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
        prompts: PromptRepository | None = None,
    ) -> None:
        self._store = store
        self._llm = llm
        self._model = model
        self._template = (prompts or PromptRepository()).load(_REPORT_PROMPT)

    def report_communities(self, *, min_size: int = 2) -> list[CommunityReport]:
        """Communities smaller than ``min_size`` are skipped; the book skips
        single-entity ones.

        Raises ``LLMResponseError`` if a report isn't the JSON the prompt
        asks for.
        """
        reports = []
        for row in self._store.query(_COMMUNITIES, {"min_size": min_size}):
            report = self._report(row["id"])
            self._store.query(
                _MERGE_COMMUNITY,
                {
                    "community_id": report.community_id,
                    "title": report.title,
                    "summary": report.summary,
                    "rating": report.rating,
                    "rating_explanation": report.rating_explanation,
                },
            )
            reports.append(report)
        return reports

    def _report(self, community_id: int) -> CommunityReport:
        params = {"community_id": community_id}
        entities = self._store.query(_COMMUNITY_ENTITIES, params)
        relationships = self._store.query(_COMMUNITY_RELATIONSHIPS, params)
        input_text = f"Entities\n\n{entities}\n\nRelationships\n\n{relationships}"
        response = self._llm.complete(
            [{"role": "user", "content": self._template.format(input_text=input_text)}],
            model=self._model,
            json_mode=True,
        )
        return _parse_report(community_id, response)


def _parse_report(community_id: int, response: str) -> CommunityReport:
    data = load_json_object(response)
    try:
        return CommunityReport(
            community_id=community_id,
            title=_require_str(data, "title"),
            summary=_require_str(data, "summary"),
            rating=float(data["rating"]),
            rating_explanation=_require_str(data, "rating_explanation"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMResponseError(f"Unusable community report: {response}") from exc


def _require_str(data: dict[str, Any], field: str) -> str:
    value = data[field]
    if not isinstance(value, str):
        raise TypeError(f"{field!r} isn't a string")
    return value
