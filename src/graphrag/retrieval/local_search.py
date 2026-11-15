from __future__ import annotations

from dataclasses import dataclass, field

from graphrag.graph.cypher_store import CypherGraphStore
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.prompts import PromptRepository
from graphrag.retrieval.embedder import Embedder
from graphrag.retrieval.entity_index import ENTITY_VECTOR_INDEX

_LOCAL_SEARCH_PROMPT = "local_search_system.txt"

# The book's local_search_query gathers all four parts in one query with
# COLLECT subqueries, which Kuzu doesn't support, so each is its own query
# over the entities found by vector search.
_ENTITIES = f"""
CALL QUERY_VECTOR_INDEX('Entity', '{ENTITY_VECTOR_INDEX}', $embedding, $k)
RETURN node.name AS name, node.summary AS summary
ORDER BY distance
"""

# The book matches ``HAS_ENTITY`` here, but its import creates ``MENTIONS``
# edges, so its local search never finds any chunks.
_CHUNKS = """
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE e.name IN $names
WITH c, count(DISTINCT e) AS freq
RETURN c.text AS text
ORDER BY freq DESC, c.id
LIMIT $limit
"""

# The book orders by ``rank`` and ``weight``, which nothing ever sets
# (Neo4j ignores unknown properties; Kuzu rejects them). Reports are ordered
# by their importance rating instead, and relationships by how many
# mentions they summarize.
_REPORTS = """
MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community)
WHERE e.name IN $names
WITH DISTINCT c
RETURN c.summary AS summary
ORDER BY c.rating DESC, c.id
LIMIT $limit
"""

_RELATIONSHIPS = """
MATCH (s:Entity)-[r:SUMMARIZED_RELATIONSHIP]->(t:Entity)
WHERE s.name IN $names AND t.name IN $names
RETURN r.summary AS summary
ORDER BY r.weight DESC, s.name, t.name
LIMIT $limit
"""


@dataclass
class LocalContext:
    """What local search gathers around the entities closest to the
    question; ``to_prompt`` renders it in the shape the book passes to the
    LLM.
    """

    entities: list[str] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        return str(
            {
                "Chunks": self.chunks,
                "Reports": self.reports,
                "Relationships": self.relationships,
                "Entities": self.entities,
            }
        )


class LocalSearch:
    """Answers questions about specific entities ("Who is Ulysses?") from
    the graph around them (ch07's ``local_search``): vector-search the
    ``k_entities`` entities closest to the question, then add the chunks
    that mention the most of them, their communities' reports, and the
    summarized relationships among them.
    """

    def __init__(
        self,
        store: CypherGraphStore,
        llm: LLMClient,
        embedder: Embedder,
        *,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
        prompts: PromptRepository | None = None,
        k_entities: int = 5,
        top_chunks: int = 3,
        top_communities: int = 3,
        top_inside_rels: int = 3,
        response_type: str = "multiple paragraphs",
    ) -> None:
        self._store = store
        self._llm = llm
        self._embedder = embedder
        self._model = model
        self._template = (prompts or PromptRepository()).load(_LOCAL_SEARCH_PROMPT)
        self._k_entities = k_entities
        self._top_chunks = top_chunks
        self._top_communities = top_communities
        self._top_inside_rels = top_inside_rels
        self._response_type = response_type

    def search(self, question: str) -> str:
        return self.answer(question, self.build_context(question))

    def build_context(self, question: str) -> LocalContext:
        [embedding] = self._embedder.embed([question])
        entities = self._store.query(_ENTITIES, {"embedding": embedding, "k": self._k_entities})
        names = [entity["name"] for entity in entities]
        return LocalContext(
            entities=[entity["summary"] for entity in entities],
            chunks=self._column(_CHUNKS, names, self._top_chunks, "text"),
            reports=self._column(_REPORTS, names, self._top_communities, "summary"),
            relationships=self._column(_RELATIONSHIPS, names, self._top_inside_rels, "summary"),
        )

    def answer(self, question: str, context: LocalContext) -> str:
        system_message = self._template.format(
            context_data=context.to_prompt(), response_type=self._response_type
        )
        return self._llm.complete(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
            ],
            model=self._model,
        )

    def _column(self, cypher: str, names: list[str], limit: int, column: str) -> list[str]:
        rows = self._store.query(cypher, {"names": names, "limit": limit})
        return [row[column] for row in rows]
