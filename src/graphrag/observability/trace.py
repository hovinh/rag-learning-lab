from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from graphrag.observability._context import ACTIVE_METER, SECTIONS, SectionFrame
from graphrag.observability.formatting import (
    DEPTH_ATTR,
    KIND_ATTR,
    KIND_CLOSE,
    KIND_OPEN,
    LABEL_ATTR,
    LABEL_WIDTH,
    LABEL_WIDTH_ATTR,
    TIME_ATTR,
    format_duration,
    is_lossy_preview,
    preview,
    render_table,
    to_text,
    wrap,
)


class Trace:
    """Structured logging on top of a stdlib logger: nested, timed
    sections and aligned ``label  value`` lines that ``TraceFormatter``
    renders as an indented tree.

    Library classes create one per module (``Trace(__name__)``) and never
    configure handlers -- that's left to the application (``RunLogging``
    in the examples), so using ``graphrag`` without it stays silent.

    Convention: INFO is the readable story (summaries, previews), DEBUG
    holds the full payloads behind them (whole query results, contexts).
    """

    def __init__(self, logger: logging.Logger | str) -> None:
        self._logger = logging.getLogger(logger) if isinstance(logger, str) else logger

    @contextmanager
    def section(self, title: str, *, level: int = logging.INFO) -> Iterator[None]:
        """Lines logged inside are indented under ``title``; on exit the
        section closes with its duration and, if a ``RunLogging`` run is
        active, the LLM/embedding calls made inside it.
        """
        meter = ACTIVE_METER.get()
        frame = SectionFrame(title, self._logger, level, meter.snapshot() if meter else None)
        frames = SECTIONS.get()
        token = SECTIONS.set((*frames, frame))
        try:
            yield
        except BaseException:
            self._close(frame, len(frames), failed=True)
            raise
        else:
            self._close(frame, len(frames), failed=False)
        finally:
            SECTIONS.reset(token)

    def info(self, message: str) -> None:
        self._emit(logging.INFO, message)

    def debug(self, message: str) -> None:
        self._emit(logging.DEBUG, message)

    def warning(self, message: str) -> None:
        self._emit(logging.WARNING, message)

    def exception(self, message: str) -> None:
        """An ERROR line with the traceback of the exception being handled."""
        self._emit(logging.ERROR, message, exc_info=True)

    def field(self, label: str, value: Any, *, level: int = logging.INFO) -> None:
        """``label  value``; a multi-line value is aligned under itself,
        and long lines of text are wrapped.
        """
        self._emit(level, wrap(to_text(value)), label=label)

    def fields(self, values: Mapping[str, Any], *, level: int = logging.INFO) -> None:
        """Several ``field`` lines with their values in one column."""
        width = max([LABEL_WIDTH, *(len(label) + 2 for label in values)])
        for label, value in values.items():
            self._emit(level, wrap(to_text(value)), label=label, label_width=width)

    def bullets(
        self,
        label: str,
        items: Sequence[str],
        *,
        empty: str = "none",
        level: int = logging.INFO,
    ) -> None:
        bullets = [wrap(item).replace("\n", "\n  ") for item in items]
        text = "\n".join(f"• {item}" for item in bullets) if items else empty
        self._emit(level, text, label=label)

    def block(self, label: str, content: Any, *, level: int = logging.INFO) -> None:
        """``label`` on its own line, then ``content`` (text, or anything
        JSON-serializable, pretty-printed) indented beneath it.
        """
        self._emit(level, "\n" + to_text(content), label=label)

    def summary(self, label: str, value: Any, *, width: int = 100) -> None:
        """A one-line ``preview`` of ``value`` at INFO, followed at DEBUG by
        the whole value when the preview had to leave something out.
        """
        short = preview(value, width)
        self._emit(logging.INFO, short, label=label)
        if is_lossy_preview(value, short):
            self._emit(logging.DEBUG, to_text(value), label="")

    def table(
        self,
        label: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        columns: Sequence[str] | None = None,
        max_width: int = 48,
        level: int = logging.INFO,
    ) -> None:
        self.block(label, render_table(rows, columns, max_width=max_width), level=level)

    def progress(self, label: str, total: int) -> Progress:
        return Progress(self, label, total)

    def _close(self, frame: SectionFrame, depth: int, *, failed: bool) -> None:
        parts = [format_duration(time.perf_counter() - frame.start)]
        meter = ACTIVE_METER.get()
        if meter is not None and frame.usage_start is not None:
            used = (meter.snapshot() - frame.usage_start).describe()
            if used:
                parts.append(used)
        stats = " · ".join(parts)
        level = logging.WARNING if failed else frame.level
        if frame.opened:
            text = f"└ failed after {stats}" if failed else f"└ {stats}"
        else:
            # Nothing was logged inside: one line instead of an empty box.
            self._flush(depth)
            text = f"• {frame.title} · " + (f"failed after {stats}" if failed else stats)
        frame.logger.log(level, text, extra={DEPTH_ATTR: depth, KIND_ATTR: KIND_CLOSE})

    def _flush(self, upto: int) -> None:
        """Writes the opening lines of the first ``upto`` sections not yet
        written.
        """
        for depth, frame in enumerate(SECTIONS.get()[:upto]):
            if not frame.opened:
                frame.opened = True
                frame.logger.log(
                    frame.level,
                    f"┌ {frame.title}",
                    extra={DEPTH_ATTR: depth, KIND_ATTR: KIND_OPEN, TIME_ATTR: frame.started_at},
                )

    def _emit(
        self,
        level: int,
        message: str,
        *,
        label: str | None = None,
        label_width: int = LABEL_WIDTH,
        exc_info: bool = False,
    ) -> None:
        if not self._logger.isEnabledFor(level):
            return
        frames = SECTIONS.get()
        self._flush(len(frames))
        extra: dict[str, Any] = {DEPTH_ATTR: len(frames)}
        if label is not None:
            extra[LABEL_ATTR] = label
            extra[LABEL_WIDTH_ATTR] = label_width
        self._logger.log(level, message, extra=extra, exc_info=exc_info, stacklevel=3)


class Progress:
    """Reports a loop of ``total`` items: every item at DEBUG, and at INFO
    only every quarter of the way through, so long loops (ch07's chunk
    extraction) stay short on the console but complete in the log file.
    """

    def __init__(self, trace: Trace, label: str, total: int) -> None:
        self._trace = trace
        self._label = label
        self._total = total
        self._done = 0

    def advance(self, detail: str = "") -> None:
        self._done += 1
        quarter = self._done * 4 // self._total if self._total else 4
        previous = (self._done - 1) * 4 // self._total if self._total else 4
        milestone = quarter != previous or self._done == self._total
        percent = 100 * self._done // self._total if self._total else 100
        text = f"{self._done}/{self._total} ({percent}%)" + (f"  {detail}" if detail else "")
        self._trace.field(self._label, text, level=logging.INFO if milestone else logging.DEBUG)
