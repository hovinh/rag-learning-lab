"""The Odyssey domain (ch07): Homer's Odyssey as a corpus of 24 books, and
the entity types its schema-free knowledge graph is extracted with.
"""

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

__all__ = ["ODYSSEY_ENTITY_TYPES", "ODYSSEY_URL", "OdysseyCorpus"]
