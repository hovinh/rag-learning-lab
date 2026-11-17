from dataclasses import dataclass, field

from graphrag.graph.models import ChunkMatch
from graphrag.pipeline.steps.retrieval import RetrieveStep
from graphrag.retrieval.retriever import Retriever, ScoredMatch


class FixedRetriever(Retriever):
    def __init__(self, matches):
        self.matches = matches
        self.questions = []

    def retrieve(self, question):
        self.questions.append(question)
        return tuple(self.matches)


@dataclass
class RetrieveContext:
    question: str
    matches: list[ScoredMatch] = field(default_factory=list)


def test_retrieve_step_fills_matches_as_a_list():
    retriever = FixedRetriever([ChunkMatch(0, "a", 1.0)])
    context = RetrieveContext("Why?")

    RetrieveStep(retriever)(context)

    assert retriever.questions == ["Why?"]
    assert context.matches == [ChunkMatch(0, "a", 1.0)]


def test_retrieve_step_title_names_the_retrieval():
    assert RetrieveStep(FixedRetriever([]), label="hybrid search").title == (
        "Retrieve (hybrid search)"
    )
