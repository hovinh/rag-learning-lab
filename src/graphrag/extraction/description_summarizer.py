from __future__ import annotations

from typing import Any

from graphrag.graph.cypher_store import CypherGraphStore
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.prompts import PromptRepository

_SUMMARY_PROMPT = "description_summary.txt"

_ENTITY_DESCRIPTIONS = """
MATCH (e:Entity)
WHERE e.description IS NOT NULL
RETURN e.name AS name, e.description AS descriptions
ORDER BY name
"""

_SET_ENTITY_SUMMARIES = """
UNWIND $rows AS row
MATCH (e:Entity {name: row.name})
SET e.summary = row.summary
"""

# Each unordered pair is read once: the undirected match sees every edge
# from both ends, and ``s.name < t.name`` keeps one of them (the book uses
# ``id(s) < id(t)``).
_RELATIONSHIP_DESCRIPTIONS = """
MATCH (s:Entity)-[r:RELATIONSHIP]-(t:Entity)
WHERE s.name < t.name
WITH s.name AS source, t.name AS target, collect(r.description) AS descriptions
RETURN source, target, descriptions
ORDER BY source, target
"""

# Kuzu's MERGE needs a direction, so the summarized edge always points from
# the alphabetically smaller name; callers match it undirected.
_SET_RELATIONSHIP_SUMMARIES = """
UNWIND $rows AS row
MATCH (s:Entity {name: row.source}), (t:Entity {name: row.target})
MERGE (s)-[r:SUMMARIZED_RELATIONSHIP]->(t)
SET r.summary = row.summary, r.weight = row.weight
"""


class DescriptionSummarizer:
    """Merges the per-mention descriptions the extraction collected into one
    summary per entity and per related pair of entities (ch07's
    ``get_summarize_prompt`` with ``import_entity_summary`` /
    ``import_rels_summary``). Anything with a single description keeps it
    as-is, with no LLM call.
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
        self._template = (prompts or PromptRepository()).load(_SUMMARY_PROMPT)

    def summarize_entities(self) -> int:
        """Sets ``summary`` on every entity that has a description; returns
        how many needed the LLM.
        """
        rows: list[dict[str, Any]] = []
        summarized = 0
        for entity in self._store.query(_ENTITY_DESCRIPTIONS):
            descriptions = entity["descriptions"]
            if len(descriptions) > 1:
                summarized += 1
            rows.append(
                {"name": entity["name"], "summary": self._summarize(entity["name"], descriptions)}
            )
        if rows:
            self._store.query(_SET_ENTITY_SUMMARIES, {"rows": rows})
        return summarized

    def summarize_relationships(self) -> int:
        """Collapses every pair's ``RELATIONSHIP`` edges into one
        ``SUMMARIZED_RELATIONSHIP``, whose ``weight`` is how many edges it
        replaces; returns how many pairs needed the LLM.
        """
        rows: list[dict[str, Any]] = []
        summarized = 0
        for pair in self._store.query(_RELATIONSHIP_DESCRIPTIONS):
            descriptions = pair["descriptions"]
            if len(descriptions) > 1:
                summarized += 1
            label = f"{pair['source']} relationship to {pair['target']}"
            rows.append(
                {
                    "source": pair["source"],
                    "target": pair["target"],
                    "summary": self._summarize(label, descriptions),
                    "weight": len(descriptions),
                }
            )
        if rows:
            self._store.query(_SET_RELATIONSHIP_SUMMARIES, {"rows": rows})
        return summarized

    def _summarize(self, entity_name: str, descriptions: list[str]) -> str:
        if len(descriptions) == 1:
            return descriptions[0]
        prompt = self._template.format(entity_name=entity_name, description_list=descriptions)
        return self._llm.complete([{"role": "user", "content": prompt}], model=self._model)
