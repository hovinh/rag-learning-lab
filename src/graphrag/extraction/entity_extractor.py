from __future__ import annotations

from collections.abc import Sequence

from graphrag.extraction.entity_models import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelationship,
)
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.prompts import PromptRepository

_EXTRACTION_PROMPT = "graph_extraction.txt"


class EntityRelationshipExtractor:
    """Has the LLM pick out entities of the given types and the
    relationships between them from a chunk of text, with no predefined
    schema beyond the type names (ch07's ``extract_entities``). The prompt
    asks for delimited tuple records, which ``parse`` turns into typed
    results.
    """

    TUPLE_DELIMITER = ";"
    RECORD_DELIMITER = "|"
    COMPLETION_DELIMITER = "\n\n"

    def __init__(
        self,
        llm: LLMClient,
        entity_types: Sequence[str],
        *,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
        prompts: PromptRepository | None = None,
    ) -> None:
        self._llm = llm
        self._entity_types = ",".join(entity_types)
        self._model = model
        self._template = (prompts or PromptRepository()).load(_EXTRACTION_PROMPT)

    def extract(self, text: str) -> ChunkExtraction:
        prompt = self._template.format(
            entity_types=self._entity_types,
            input_text=text,
            tuple_delimiter=self.TUPLE_DELIMITER,
            record_delimiter=self.RECORD_DELIMITER,
            completion_delimiter=self.COMPLETION_DELIMITER,
        )
        output = self._llm.complete([{"role": "user", "content": prompt}], model=self._model)
        return self.parse(output)

    @classmethod
    def parse(cls, output: str) -> ChunkExtraction:
        """Malformed and unknown records are skipped, as in the book: one bad
        record shouldn't cost the rest of the chunk.

        Names are upper-cased so the same entity written with different
        casing in different chunks still merges into one node.
        """
        extraction = ChunkExtraction()
        for raw in output.split(cls.RECORD_DELIMITER):
            record = raw.strip()
            if record.startswith("(") and record.endswith(")"):
                record = record[1:-1].strip()
            kind, _, rest = record.partition(cls.TUPLE_DELIMITER)
            kind = kind.strip(" \"'").lower()
            # The description is split off last, and the strength from the
            # right, so a description containing the delimiter survives
            # intact; the book's plain split() drops such records.
            fields = [field.strip() for field in rest.split(cls.TUPLE_DELIMITER, 2)]
            if len(fields) != 3:
                continue
            if kind == "entity":
                name, entity_type, description = fields
                extraction.entities.append(
                    ExtractedEntity(_normalize(name), _normalize(entity_type), description)
                )
            elif kind == "relationship":
                source, target, tail = fields
                description, sep, strength = tail.rpartition(cls.TUPLE_DELIMITER)
                if not sep:
                    continue
                extraction.relationships.append(
                    ExtractedRelationship(
                        _normalize(source),
                        _normalize(target),
                        description.strip(),
                        _to_float(strength),
                    )
                )
        return extraction


def _normalize(name: str) -> str:
    return name.strip(" \"'").upper()


def _to_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None
