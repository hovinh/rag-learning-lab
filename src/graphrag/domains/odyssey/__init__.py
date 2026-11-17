"""The Odyssey domain (ch07): Homer's Odyssey as a corpus of 24 books, the
entity types its schema-free knowledge graph is extracted with, and a
benchmark over Book I for comparing the text recipes.
"""

from pathlib import Path

from graphrag.domains.odyssey.corpus import ODYSSEY_URL, OdysseyCorpus

ODYSSEY_ENTITY_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "GOD",
    "EVENT",
    "CREATURE",
    "WEAPON_OR_TOOL",
]

# ``question;answer;category`` rows over Book I, with hand-written answers
# checked against the text. "specific" questions are answered by one
# passage; "broad" ones need the whole book.
ODYSSEY_BENCHMARK_PATH = Path(__file__).parent / "benchmark.csv"

# What hypothetical question embedding is for over this corpus: it steers
# which questions each chunk is indexed under.
ODYSSEY_QUESTION_PURPOSE = (
    "Readers of Homer's Odyssey asking about its characters, gods and places: who they are, "
    "what they do and why, how they relate to each other, and what happens in the story."
)

__all__ = [
    "ODYSSEY_BENCHMARK_PATH",
    "ODYSSEY_ENTITY_TYPES",
    "ODYSSEY_QUESTION_PURPOSE",
    "ODYSSEY_URL",
    "OdysseyCorpus",
]
