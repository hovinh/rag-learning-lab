from __future__ import annotations

import json
from typing import Any

from graphrag.llm.errors import LLMResponseError


def load_json_field(response: str, field: str) -> Any:
    """Read ``field`` from a JSON-object response, e.g. one produced with
    ``LLMClient.complete(..., json_mode=True)``.

    Raises ``LLMResponseError`` if the response isn't a JSON object or has
    no such field.
    """
    try:
        data = json.loads(response)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"Response isn't valid JSON: {response}") from exc
    if not isinstance(data, dict) or field not in data:
        raise LLMResponseError(f"Response has no {field!r} field: {response}")
    return data[field]
