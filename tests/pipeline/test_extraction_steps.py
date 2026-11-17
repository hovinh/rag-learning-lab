from dataclasses import dataclass, field

from pydantic import BaseModel

from graphrag.pipeline.steps.extraction import ExtractStructuredStep, ImportExtractedStep


class Person(BaseModel):
    name: str


@dataclass
class ExtractionTestContext:
    documents: list[str] = field(default_factory=list)
    extracted: list[Person] = field(default_factory=list)
    imported_ids: list[str] = field(default_factory=list)


class FakeExtractor:
    def extract(self, document):
        return Person(name=document.upper())


def test_extract_one_model_per_document_then_import_each():
    context = ExtractionTestContext(documents=["ada", "alan"])
    imported = []

    def import_one(person):
        imported.append(person)
        return f"id-{person.name}"

    ExtractStructuredStep(FakeExtractor())(context)
    ImportExtractedStep(import_one)(context)

    assert context.extracted == [Person(name="ADA"), Person(name="ALAN")]
    assert imported == context.extracted
    assert context.imported_ids == ["id-ADA", "id-ALAN"]
