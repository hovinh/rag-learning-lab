from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentScope:
    """What the agent answers questions about, as its prompts describe it.
    The book's prompts hard-code movies; a domain passes its own scope
    instead (see ``graphrag.domains.movies.MOVIES_AGENT_SCOPE``).

    - ``source``: where answers come from ("a database of movies and the
      people who worked on them"), told to the query planners so they
      don't add details the data can't have;
    - ``topic``: what's on topic ("movies"), so the critic asks no
      follow-ups for small talk or unrelated questions;
    - ``capabilities``: what the ``respond`` tool reminds the user it can
      answer ("questions about movies and their cast").
    """

    source: str = "a database"
    topic: str = "the data in the database"
    capabilities: str = "questions about the data in the database"

    def fill(self, template: str) -> str:
        """Fills ``{source}`` and ``{topic}`` in a prompt. ``str.format``
        can't, since the prompts also hold literal JSON braces.
        """
        return template.replace("{source}", self.source).replace("{topic}", self.topic)
