from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from collections.abc import Mapping
from contextvars import Token
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self, TextIO

from graphrag.observability._context import ACTIVE_METER
from graphrag.observability.formatting import (
    BANNER_WIDTH,
    KIND_ATTR,
    KIND_BANNER,
    TraceFormatter,
    format_duration,
)
from graphrag.observability.usage import UsageMeter, format_count, plural

# The loggers a run turns up to DEBUG: the library's, and the script
# being run. Everything else (openai, httpx, pdfminer, ...) stays at
# WARNING, so their debug chatter never reaches the trace.
_OWN_LOGGERS = ("graphrag", "__main__")


class RunLogging:
    """Sets up logging for one run of a script, as a context manager::

        with RunLogging("ch05_agentic_rag_movies", LOG_DIR, title="...") as run:
            llm = OpenAILLMClient(OpenAI(), usage=run.usage)
            ...

    - the console shows INFO (DEBUG with ``verbose``), coloured when it's
      a terminal; ``log_dir/<process_name>.log`` always gets DEBUG, i.e.
      the full payloads too, and is overwritten on each run;
    - stdout/stderr are switched to UTF-8 (Windows defaults to cp1252,
      which can't encode names like "Miloš Forman");
    - a header names the run, and a footer gives its duration and the
      LLM/embedding usage recorded on ``usage``;
    - an exception escaping the run is logged with its traceback (so it
      lands in the log file too) and the process exits with status 1.
    """

    def __init__(
        self,
        process_name: str,
        log_dir: Path,
        *,
        title: str = "",
        details: Mapping[str, Any] | None = None,
        verbose: bool = False,
    ) -> None:
        self._process_name = process_name
        self._title = title
        self._details = dict(details or {})
        self._verbose = verbose
        self.log_path = log_dir / f"{process_name}.log"
        self.usage = UsageMeter()
        self._logger = logging.getLogger("graphrag.run")
        self._handlers: list[logging.Handler] = []
        self._saved_levels: dict[str, int] = {}
        self._meter_token: Token[UsageMeter | None] | None = None
        self._start = 0.0

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser, *, default_process_name: str) -> None:
        """The ``--process-name`` and ``--verbose`` flags a script passes
        on to ``RunLogging``.
        """
        parser.add_argument(
            "--process-name",
            default=default_process_name,
            help="Names the log file written to logs/<process_name>.log.",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help=(
                "Also show the full payloads (DEBUG lines) on the console; "
                "logs/<process_name>.log always has them."
            ),
        )

    def __enter__(self) -> Self:
        for stream in (sys.stdout, sys.stderr):
            if isinstance(stream, io.TextIOWrapper):
                stream.reconfigure(encoding="utf-8")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.DEBUG if self._verbose else logging.INFO)
        console.setFormatter(TraceFormatter(color=_supports_color(sys.stderr)))
        logfile = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        logfile.setLevel(logging.DEBUG)
        logfile.setFormatter(TraceFormatter())
        self._handlers = [console, logfile]

        root = logging.getLogger()
        for handler in self._handlers:
            root.addHandler(handler)
        for name in ("", *_OWN_LOGGERS):
            logger = logging.getLogger(name)
            self._saved_levels[name] = logger.level
        root.setLevel(logging.WARNING)
        for name in _OWN_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)

        self._meter_token = ACTIVE_METER.set(self.usage)
        self._start = time.perf_counter()
        self._banner(self._header_lines())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        failed = exc is not None and not isinstance(exc, KeyboardInterrupt | SystemExit)
        if failed:
            self._logger.error("The run failed", exc_info=(exc_type, exc, tb))  # type: ignore[arg-type]
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else None
        self._banner(self._footer_lines(failed, status))
        self._teardown()
        if failed:
            # Already logged above; exiting here instead of re-raising
            # keeps the traceback from being printed twice.
            raise SystemExit(1) from None

    def _teardown(self) -> None:
        root = logging.getLogger()
        for handler in self._handlers:
            root.removeHandler(handler)
            handler.close()
        for name, level in self._saved_levels.items():
            logging.getLogger(name).setLevel(level)
        if self._meter_token is not None:
            ACTIVE_METER.reset(self._meter_token)

    def _header_lines(self) -> list[str]:
        lines = [f" {self._title}"] if self._title else []
        details = {
            "run": f"{self._process_name} · {datetime.now().astimezone():%Y-%m-%d %H:%M:%S UTC%z}",
            **self._details,
            "log file": f"{self.log_path} (full detail)",
            "console": "DEBUG (--verbose)"
            if self._verbose
            else "INFO (--verbose shows full payloads)",
        }
        lines.extend(_detail_lines(details))
        return lines

    def _footer_lines(self, failed: bool, status: str | None) -> list[str]:
        elapsed = format_duration(time.perf_counter() - self._start)
        outcome = "✗ failed" if failed else (f"■ {status}" if status else "✓ finished")
        usage = self.usage.snapshot()
        details: dict[str, str] = {}
        if usage.llm_calls:
            details["LLM"] = (
                f"{plural(usage.llm_calls, 'call')} · {format_count(usage.llm_tokens)} tokens "
                f"({format_count(usage.prompt_tokens)} in, "
                f"{format_count(usage.completion_tokens)} out)"
            )
        if usage.embedding_calls:
            details["embeddings"] = (
                f"{plural(usage.embedding_calls, 'call')} · "
                f"{format_count(usage.embedding_tokens)} tokens"
            )
        details["log file"] = str(self.log_path)
        return [f" {outcome} in {elapsed}", *_detail_lines(details)]

    def _banner(self, lines: list[str]) -> None:
        rule = "═" * BANNER_WIDTH
        self._logger.info("\n".join([rule, *lines, rule]), extra={KIND_ATTR: KIND_BANNER})


def _detail_lines(details: Mapping[str, Any]) -> list[str]:
    width = max(len(label) for label in details) + 2
    return [f" {label.ljust(width)}{value}" for label, value in details.items()]


def _supports_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR") or not stream.isatty():
        return False
    if sys.platform != "win32":
        return True
    # Windows consoles only render ANSI colours once virtual terminal
    # processing is switched on for the handle.
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-12)  # STD_ERROR_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except (AttributeError, OSError):
        return False
