from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime config, read once at the composition root (see examples/)."""

    openai_api_key: str
    kuzu_db_path: Path

    @classmethod
    def from_env(cls) -> Settings:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env (see .env.example).")
        db_path = os.environ.get("KUZU_DB_PATH", "data/kuzu_db")
        return cls(openai_api_key=api_key, kuzu_db_path=Path(db_path))

    def db_path_for(self, name: str) -> Path:
        """A database of its own for ``name``, next to ``kuzu_db_path``
        (``data/kuzu_db_movies``), so graphs built by different pipelines
        never overwrite each other.
        """
        return self.kuzu_db_path.with_name(f"{self.kuzu_db_path.name}_{name}")
