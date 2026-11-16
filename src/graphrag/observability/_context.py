"""State shared by ``Trace`` (which opens sections), ``TraceFormatter``
(which indents by them) and ``RunLogging`` (which installs the run's usage
meter). Context variables rather than globals so nested sections unwind
correctly even when an exception escapes one.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field

from graphrag.observability.usage import Usage, UsageMeter


@dataclass
class SectionFrame:
    """An open ``Trace.section``. Its opening line is written lazily, just
    before the first line logged inside it, so a section nothing logs into
    collapses to a single line.
    """

    title: str
    logger: logging.Logger
    level: int
    usage_start: Usage | None
    start: float = field(default_factory=time.perf_counter)
    # Wall-clock start, so the lazily written opening line still shows
    # when the section began rather than when its first line was logged.
    started_at: float = field(default_factory=time.time)
    opened: bool = False


SECTIONS: ContextVar[tuple[SectionFrame, ...]] = ContextVar("graphrag_trace_sections", default=())
ACTIVE_METER: ContextVar[UsageMeter | None] = ContextVar("graphrag_usage_meter", default=None)


def current_depth() -> int:
    return len(SECTIONS.get())
