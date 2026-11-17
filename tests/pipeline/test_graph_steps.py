from dataclasses import dataclass, field

import pytest

from graphrag.extraction.entity_models import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelationship,
)
from graphrag.ingestion.models import Chunk
from graphrag.pipeline.steps.graph_index import DetectCommunitiesStep, ExtractEntityGraphStep
from graphrag.pipeline.steps.graph_search import GlobalSearchStep, LocalSearchStep
from graphrag.retrieval.global_search import KeyPoint
from graphrag.retrieval.local_search import LocalContext


@dataclass
class ChunksContext:
    chunks: list[Chunk] = field(default_factory=list)


class FakeEntityExtractor:
    def extract(self, text):
        return ChunkExtraction(
            [ExtractedEntity(name, "PERSON", f"{name} is in: {text}") for name in text.split()],
            [ExtractedRelationship(*text.split(), "they meet", 1.0)],
        )


def test_extract_entity_graph_step_imports_each_chunk_under_its_document(
    entity_store, entity_importer
):
    context = ChunksContext(chunks=[Chunk(0, 0, "ULYSSES ATHENA"), Chunk(1, 0, "ULYSSES ZEUS")])

    ExtractEntityGraphStep(FakeEntityExtractor(), entity_importer)(context)

    chunks = entity_store.query("MATCH (c:Chunk) RETURN c.id AS id ORDER BY id")
    assert [row["id"] for row in chunks] == ["0-0", "1-0"]
    assert entity_importer.count_entities() == 3
    assert entity_importer.count_relationships() == 2


class FakeDetector:
    def detect(self):
        return {}


def test_detect_communities_step_fails_when_there_are_none():
    with pytest.raises(ValueError, match="no communities"):
        DetectCommunitiesStep(FakeDetector())(object())


@dataclass
class GlobalContext:
    question: str
    key_points: list[KeyPoint] = field(default_factory=list)
    answer: str = ""


class FakeGlobalSearch:
    def map(self, question):
        return [KeyPoint("A voyage home", 90)]

    def reduce(self, question, points):
        return f"{question} -> {points[0].description}"


def test_global_search_step_maps_then_reduces():
    context = GlobalContext("What is this about?")

    GlobalSearchStep(FakeGlobalSearch())(context)

    assert context.key_points == [KeyPoint("A voyage home", 90)]
    assert context.answer == "What is this about? -> A voyage home"


@dataclass
class LocalTestContext:
    question: str
    local_context: LocalContext = field(default_factory=LocalContext)
    answer: str = ""


class FakeLocalSearch:
    def build_context(self, question):
        return LocalContext(entities=["ULYSSES: a hero"])

    def answer(self, question, local_context):
        return f"{question} -> {local_context.entities[0]}"


def test_local_search_step_builds_context_then_answers():
    context = LocalTestContext("Who is Ulysses?")

    LocalSearchStep(FakeLocalSearch())(context)

    assert context.local_context == LocalContext(entities=["ULYSSES: a hero"])
    assert context.answer == "Who is Ulysses? -> ULYSSES: a hero"
