"""Benchmarks the text recipes against each other with the ch08
``RagEvaluation`` recipe: every variant indexes the same Odyssey book,
answers the same benchmark questions (hand-written answers, split into
"specific" and "broad"), and the same LLM judges score the answers for
answer correctness, context recall and faithfulness.

The variants are ``VectorRag`` (vector and hybrid search),
``ParentChildRag`` (with and without step-back), ``HypotheticalQuestionRag``
and ``GraphRag`` (local and global search). Each recipe's index persists in
its own Kuzu database, so only the first run (or one with --reindex)
builds it; GraphRag's is the expensive one (~100k tokens per book).
Answered samples are saved per variant, and --rescore re-runs only the
judges on them. See examples/README.md for the findings. Requires
OPENAI_API_KEY in .env (see .env.example). Run from the repo root:

    .venv/Scripts/python.exe examples/benchmark_text_recipes_odyssey.py \\
        --process-name benchmark_text_recipes_odyssey
"""

from __future__ import annotations

import argparse
import csv
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Protocol, TypeVar

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR, add_reindex_argument
from dotenv import load_dotenv

from graphrag.domains.odyssey import (
    ODYSSEY_BENCHMARK_PATH,
    ODYSSEY_ENTITY_TYPES,
    ODYSSEY_QUESTION_PURPOSE,
    OdysseyCorpus,
)
from graphrag.evaluation import (
    EvaluationReport,
    EvaluationSample,
    Responder,
    format_score,
    load_benchmark,
    load_samples,
    log_report,
    save_samples,
)
from graphrag.graph import KuzuCypherGraphStore, KuzuGraphStore, KuzuParentChildGraphStore
from graphrag.ingestion import TextChunker
from graphrag.observability import RunLogging, Trace, format_duration
from graphrag.observability.usage import format_count
from graphrag.recipes import (
    GraphRag,
    HypotheticalQuestionRag,
    ParentChildRag,
    RagEvaluation,
    VectorRag,
)
from graphrag.services import OpenAIServices
from graphrag.settings import Settings

ODYSSEY_PATH = Path("data/ch07-odyssey.txt")
BOOKS = 1
K = 4
QUESTIONS_PER_CHUNK = 8
VARIANTS = (
    "vector",
    "hybrid",
    "parent_child",
    "parent_child_step_back",
    "hypothetical_questions",
    "graphrag_local",
    "graphrag_global",
)
CATEGORIES = ("specific", "broad")
METRICS = ("answer_correctness", "context_recall", "faithfulness")

trace = Trace(__name__)

StoreT = TypeVar("StoreT", bound=AbstractContextManager)


class Indexable(Protocol):
    def is_indexed(self) -> bool: ...

    def index(self, source: OdysseyCorpus) -> None: ...


@dataclass
class VariantResult:
    variant: str
    report: EvaluationReport
    tokens_per_question: int | None


class RecipeBench:
    """Opens each recipe's store on first use, indexes it unless it already
    holds an index, and hands out a responder per variant.
    """

    def __init__(
        self, services: OpenAIServices, settings: Settings, stack: ExitStack, reindex: bool
    ) -> None:
        self._services = services
        self._settings = settings
        self._stack = stack
        self._reindex = reindex
        self._source = OdysseyCorpus(ODYSSEY_PATH, books=BOOKS)

    def responder(self, variant: str) -> Responder:
        if variant == "vector":
            return lambda question: self._vector_rag.query(question, search="vector")
        if variant == "hybrid":
            return lambda question: self._vector_rag.query(question, search="hybrid")
        if variant == "parent_child":
            return self._parent_child_rag(step_back=False).query
        if variant == "parent_child_step_back":
            return self._parent_child_rag(step_back=True).query
        if variant == "hypothetical_questions":
            return self._hypothetical_question_rag.query
        if variant == "graphrag_local":
            return self._graph_rag.local_search
        return self._graph_rag.global_search

    @cached_property
    def _vector_rag(self) -> VectorRag:
        store = self._open(KuzuGraphStore(self._db("bench_vector"), embedding_dim=self._dim))
        rag = VectorRag(
            store, self._services.embedder, self._services.llm, k=K, model=DEFAULT_ANSWER_MODEL
        )
        self._index_once("VectorRag", rag)
        return rag

    @cached_property
    def _parent_child_store(self) -> KuzuParentChildGraphStore:
        # Both ParentChildRag variants share one index: step-back only
        # changes how it is queried.
        store = self._open(
            KuzuParentChildGraphStore(self._db("bench_parent_child"), embedding_dim=self._dim)
        )
        self._index_once("ParentChildRag", self._parent_child_rag(store, step_back=False))
        return store

    def _parent_child_rag(
        self, store: KuzuParentChildGraphStore | None = None, *, step_back: bool
    ) -> ParentChildRag:
        return ParentChildRag(
            store or self._parent_child_store,
            self._services.embedder,
            self._services.llm,
            step_back=step_back,
            k=K,
            model=DEFAULT_ANSWER_MODEL,
        )

    @cached_property
    def _hypothetical_question_rag(self) -> HypotheticalQuestionRag:
        store = self._open(
            KuzuParentChildGraphStore(self._db("bench_questions"), embedding_dim=self._dim)
        )
        rag = HypotheticalQuestionRag(
            store,
            self._services.embedder,
            self._services.llm,
            purpose=ODYSSEY_QUESTION_PURPOSE,
            questions_per_chunk=QUESTIONS_PER_CHUNK,
            k=K,
            model=DEFAULT_ANSWER_MODEL,
        )
        self._index_once("HypotheticalQuestionRag", rag)
        return rag

    @cached_property
    def _graph_rag(self) -> GraphRag:
        store = self._open(KuzuCypherGraphStore(self._db("bench_graphrag")))
        rag = GraphRag(
            store,
            self._services.llm,
            self._services.embedder,
            entity_types=ODYSSEY_ENTITY_TYPES,
            chunker=TextChunker(1000, 40),
            model=DEFAULT_ANSWER_MODEL,
        )
        self._index_once("GraphRag", rag)
        return rag

    def _index_once(self, name: str, rag: Indexable) -> None:
        if self._reindex or not rag.is_indexed():
            with trace.section(f"Index {name}"):
                rag.index(self._source)
        else:
            trace.field(name, "reusing its index (--reindex rebuilds it)")

    def _open(self, store: StoreT) -> StoreT:
        return self._stack.enter_context(store)

    def _db(self, name: str) -> Path:
        return self._settings.db_path_for(name)

    @property
    def _dim(self) -> int:
        return self._services.embedder.dimensions


def main(
    process_name: str,
    variants: list[str],
    limit: int | None,
    rescore: bool,
    reindex: bool,
    verbose: bool,
) -> None:
    with RunLogging(
        process_name,
        LOG_DIR,
        title="Benchmark · the text recipes on Odyssey Book I",
        details={
            "model": DEFAULT_ANSWER_MODEL,
            "variants": ", ".join(variants),
            "questions": limit or "all",
            "mode": "re-score saved answers" if rescore else "answer, then score",
        },
        verbose=verbose,
    ) as run:
        load_dotenv()
        settings = Settings.from_env()
        services = OpenAIServices.create(settings, usage=run.usage)
        evaluation = RagEvaluation(services.llm, services.embedder, model=DEFAULT_ANSWER_MODEL)
        cases = load_benchmark(ODYSSEY_BENCHMARK_PATH)[:limit]
        categories = {case.question: case.category for case in cases}

        answered: dict[str, tuple[list[EvaluationSample], int | None]] = {}
        if rescore:
            for variant in variants:
                answered[variant] = (
                    load_samples(_samples_path(process_name, variant))[:limit],
                    None,
                )
        else:
            with ExitStack() as stack:
                bench = RecipeBench(services, settings, stack, reindex)
                for variant in variants:
                    respond = bench.responder(variant)
                    with trace.section(f"{variant}: answer {len(cases)} questions"):
                        before = run.usage.snapshot()
                        samples = evaluation.answer(cases, respond)
                        used = run.usage.snapshot() - before
                    save_samples(samples, _samples_path(process_name, variant))
                    answered[variant] = (samples, used.llm_tokens // max(len(samples), 1))

        results = []
        for variant, (samples, tokens) in answered.items():
            with trace.section(f"{variant}: score {len(samples)} answers (LLM judges)"):
                report = evaluation.score(samples)
                log_report(report)
                report.write_csv(LOG_DIR / f"{process_name}_{variant}_results.csv")
            results.append(VariantResult(variant, report, tokens))

        with trace.section("Comparison"):
            rows = [_comparison_row(result, categories) for result in results]
            trace.table("variants", rows, max_width=24)
            path = LOG_DIR / f"{process_name}_comparison.csv"
            _write_rows(rows, path)
            trace.field("comparison", f"written to {path}")


def _samples_path(process_name: str, variant: str) -> Path:
    return LOG_DIR / f"{process_name}_{variant}_samples.json"


def _comparison_row(result: VariantResult, categories: dict[str, str]) -> dict[str, str]:
    report = result.report
    by_category = {
        category: EvaluationReport(
            report.metric_names,
            [row for row in report.rows if categories.get(row.sample.question) == category],
        ).means()
        for category in CATEGORIES
    }
    means = report.means()
    latency = report.mean_latency()
    return {
        "variant": result.variant,
        "correct": format_score(means["answer_correctness"]),
        **{
            f"correct·{category}": format_score(by_category[category]["answer_correctness"])
            for category in CATEGORIES
        },
        "recall": format_score(means["context_recall"]),
        "faithful": format_score(means["faithfulness"]),
        "latency": "n/a" if latency is None else format_duration(latency),
        "tokens/q": "n/a"
        if result.tokens_per_question is None
        else format_count(result.tokens_per_question),
        "failed": str(len(report.failed())),
    }


def _write_rows(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    RunLogging.add_arguments(parser, default_process_name="benchmark_text_recipes_odyssey")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help="Which recipe variants to benchmark (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Benchmark only the first N questions (default: all 20).",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help=(
            "Skip answering and only re-score the samples an earlier run saved "
            "(logs/<process_name>_<variant>_samples.json)."
        ),
    )
    add_reindex_argument(parser)
    args = parser.parse_args()
    if args.rescore and args.reindex:
        parser.error("--reindex has no effect with --rescore")
    main(args.process_name, args.variants, args.limit, args.rescore, args.reindex, args.verbose)
