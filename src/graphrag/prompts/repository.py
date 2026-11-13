from __future__ import annotations

from pathlib import Path


class PromptRepository:
    """Loads prompt text from files rather than inline string constants, so
    prompts can be edited without touching code. Defaults to this package's
    own directory; constructor injection lets tests point it elsewhere.
    """

    def __init__(self, prompts_dir: Path | None = None) -> None:
        self._dir = prompts_dir or Path(__file__).parent

    def load(self, name: str) -> str:
        return (self._dir / name).read_text(encoding="utf-8").strip()
