from __future__ import annotations

import json
import logging
import re
import textwrap
import time
from collections.abc import Mapping, Sequence
from typing import Any

from graphrag.observability._context import current_depth

TIME_FORMAT = "%H:%M:%S"
TIME_WIDTH = 8
GUTTER = "│ "
LABEL_WIDTH = 10
BANNER_WIDTH = 78

# Record attributes ``Trace`` sets through ``extra=``; records from other
# loggers (e.g. a library warning) have none and are indented to the
# section they were logged in.
DEPTH_ATTR = "trace_depth"
KIND_ATTR = "trace_kind"
LABEL_ATTR = "trace_label"
LABEL_WIDTH_ATTR = "trace_label_width"
TIME_ATTR = "trace_time"

WRAP_WIDTH = 100

KIND_OPEN = "open"
KIND_CLOSE = "close"
KIND_BANNER = "banner"

_LEVEL_MARKERS = {logging.WARNING: "⚠ ", logging.ERROR: "✗ ", logging.CRITICAL: "✗ "}

_RESET = "\x1b[0m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_CYAN = "\x1b[36m"
_YELLOW = "\x1b[33m"
_RED = "\x1b[31m"


class TraceFormatter(logging.Formatter):
    """Renders records as an indented trace::

        11:44:51 ┌ Question 1/2
        11:44:53 │ plan      Who is the actor in the movie Matrix?
        11:44:54 │ tool      movie_info_by_title(title="Matrix")
        11:44:57 └ 6.1s · 7 LLM calls · 3.2k tokens

    Only ``HH:MM:SS`` is shown (the run header has the date). Lines after
    the first of a multi-line message are aligned under its text, so
    pretty-printed JSON and answers stay inside their section's gutter.
    ``color`` adds ANSI colours, for a terminal only.
    """

    def __init__(self, *, color: bool = False) -> None:
        super().__init__()
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        lines = message.splitlines() or [""]
        kind = getattr(record, KIND_ATTR, None)
        if kind == KIND_BANNER:
            return "\n".join(self._paint(line, _BOLD) for line in lines)

        depth = getattr(record, DEPTH_ATTR, None)
        gutter = GUTTER * (current_depth() if depth is None else depth)
        created = getattr(record, TIME_ATTR, record.created)
        stamp = time.strftime(TIME_FORMAT, time.localtime(created))

        head = ""
        if kind is None:
            head = _LEVEL_MARKERS.get(record.levelno, "")
        label = getattr(record, LABEL_ATTR, None)
        if label is not None:
            width = max(getattr(record, LABEL_WIDTH_ATTR, LABEL_WIDTH), len(label) + 1)
            head += self._paint(label.ljust(width), _CYAN)
        hang = " " * _visible_len(head)

        style = self._style_for(record.levelno, kind)
        first = f"{self._paint(stamp, _DIM)} {self._paint(gutter, _DIM)}{head}"
        out = [first + self._paint(lines[0], style)]
        indent = f"{' ' * TIME_WIDTH} {self._paint(gutter, _DIM)}{hang}"
        out.extend(indent + self._paint(line, style) for line in lines[1:])
        return "\n".join(line.rstrip() for line in out)

    def _style_for(self, level: int, kind: str | None) -> str:
        if level >= logging.ERROR:
            return _RED
        if level >= logging.WARNING:
            return _YELLOW
        if kind == KIND_OPEN:
            return _BOLD
        if kind == KIND_CLOSE or level < logging.INFO:
            return _DIM
        return ""

    def _paint(self, text: str, style: str) -> str:
        if not self._color or not style or not text:
            return text
        return f"{style}{text}{_RESET}"


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(text: str) -> int:
    return len(_ANSI.sub("", text))


def to_text(value: Any) -> str:
    """``value`` as readable text: strings as they are, query rows (a list
    of dicts) as one JSON object per line, anything else as indented JSON.
    Non-ASCII characters are kept (``Miloš``, not ``Milo\\u0161``).
    """
    if isinstance(value, str):
        return value
    if _is_rows(value) and value:
        return "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in value)
    if isinstance(value, list) and not any(isinstance(v, list | dict) for v in value):
        return json.dumps(value, ensure_ascii=False, default=str)
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def wrap(text: str, width: int = WRAP_WIDTH) -> str:
    """Wraps each line of ``text`` longer than ``width`` at word
    boundaries; shorter lines (Cypher, tables) are left as they are.
    """
    return "\n".join(
        "\n".join(textwrap.wrap(line, width, break_long_words=False, break_on_hyphens=False))
        if len(line) > width
        else line
        for line in text.splitlines()
    )


def preview(value: Any, width: int = 100) -> str:
    """A one-line summary of ``value`` at most ``width`` characters, e.g.
    ``2 rows: The Matrix, 1999 | The Matrix Reloaded, 2003``. Query rows
    (a list of dicts) are summarized by their values; one row keeps its
    keys, since those name what was counted (``directorCount=10``).
    """
    if isinstance(value, str):
        text = " ".join(value.split())
    elif _is_rows(value):
        if len(value) == 1:
            text = ", ".join(f"{k}={_scalar(v)}" for k, v in value[0].items())
        else:
            rows = " | ".join(", ".join(_scalar(v) for v in row.values()) for row in value)
            text = f"{len(value)} rows: {rows}" if value else "0 rows"
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    return truncate(text, width)


def is_lossy_preview(value: Any, short: str) -> bool:
    """Whether ``short`` (a ``preview`` of ``value``) left anything out:
    it was cut, collapsed line breaks, or stood in for nested lists or
    objects (``[4 items]``).
    """
    if isinstance(value, str):
        return short != value
    if short.endswith("…"):
        return True
    if _is_rows(value):
        return any(isinstance(v, list | dict) for row in value for v in row.values())
    return False


def truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def render_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
    *,
    max_width: int = 48,
) -> str:
    """An aligned plain-text table; cells longer than ``max_width`` are
    cut with ``…``. Numbers are right-aligned.
    """
    if not rows:
        return "(no rows)"
    columns = list(columns) if columns is not None else list(rows[0].keys())
    cells = [[truncate(_scalar(row.get(c, "")), max_width) for c in columns] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(columns)]
    numeric = [all(_is_number(row.get(c)) for row in rows) for c in columns]

    def line(values: Sequence[str]) -> str:
        parts = [
            v.rjust(w) if num else v.ljust(w)
            for v, w, num in zip(values, widths, numeric, strict=True)
        ]
        return "  ".join(parts).rstrip()

    out = [line(columns), "  ".join("─" * w for w in widths)]
    out.extend(line(r) for r in cells)
    return "\n".join(out)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}m{rest:02d}s"


def _is_rows(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(row, dict) for row in value)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _scalar(value: Any) -> str:
    if isinstance(value, list):
        return f"[{len(value)} item]" if len(value) == 1 else f"[{len(value)} items]"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.3f}"
    return " ".join(str(value).split())
