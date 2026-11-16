"""Re-implementation of ch08 (template/kg-rag/notebooks/ch08.ipynb):
evaluating ch05's agentic RAG over the movies graph against the book's
benchmark. Each benchmark question's Cypher gives its ground truth; the
agent answers it (timed); then three RAGAS-style LLM-judged metrics score
every answer: answer correctness, context recall, and faithfulness.

See examples/README.md for the metrics and where this departs from the
book. Requires OPENAI_API_KEY in .env (see .env.example). Every question
costs several agent and judge calls, so --limit runs only the first N.
Answered samples are saved, and --rescore re-runs only the judges on them.
--decompose splits each question into atomic questions first, as the
book's ch08 agent does.
Run from the repo root:

    .venv/Scripts/python.exe examples/evaluate_agent_movies.py \\
        --process-name ch08_evaluate_agent_movies
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR, add_verbose_argument
from dotenv import load_dotenv
from movie_agent import build_movie_agent
from openai import OpenAI

from graphrag.datasets import MOVIES_BENCHMARK_PATH, MovieGraphSeeder
from graphrag.evaluation import (
    AnswerCorrectness,
    BenchmarkRunner,
    ContextRecall,
    EvaluationReport,
    Evaluator,
    Faithfulness,
    format_score,
    load_benchmark,
    load_samples,
    save_samples,
)
from graphrag.graph import KuzuCypherGraphStore
from graphrag.llm import OpenAILLMClient
from graphrag.observability import RunLogging, Trace, format_duration, preview
from graphrag.retrieval import OpenAIEmbedder
from graphrag.settings import Settings

trace = Trace(__name__)


def main(
    process_name: str, limit: int | None, rescore: Path | None, decompose: bool, verbose: bool
) -> None:
    mode = f"re-score {rescore}" if rescore else ("decompose" if decompose else "update")
    with RunLogging(
        process_name,
        LOG_DIR,
        title="ch08 · Evaluating the ch05 agent on the movies benchmark",
        details={
            "model": DEFAULT_ANSWER_MODEL,
            "planner": mode,
            "questions": limit or "all",
        },
        verbose=verbose,
    ) as run:
        load_dotenv()
        settings = Settings.from_env()
        openai_client = OpenAI(api_key=settings.openai_api_key)
        llm = OpenAILLMClient(openai_client, usage=run.usage)

        if rescore is not None:
            # Judges only: score answers an earlier run saved, so a change to a
            # metric is measured on the same answers.
            samples = load_samples(rescore)[:limit]
            trace.field("samples", f"{len(samples)} loaded from {rescore}")
        else:
            cases = load_benchmark(MOVIES_BENCHMARK_PATH)[:limit]
            # Kept apart from the other chapters' databases: seeding drops and
            # recreates its tables on every run.
            db_path = settings.kuzu_db_path.with_name(f"{settings.kuzu_db_path.name}_ch08")
            with KuzuCypherGraphStore(db_path) as store:
                with trace.section("Seed the movies graph"):
                    MovieGraphSeeder(store).seed()
                agent = build_movie_agent(llm, store, decompose=decompose)
                with trace.section(f"Answer {len(cases)} benchmark questions"):
                    samples = BenchmarkRunner(store, agent.respond).run(cases)
            samples_path = LOG_DIR / f"{process_name}_samples.json"
            save_samples(samples, samples_path)
            trace.field("samples", f"written to {samples_path}")

        evaluator = Evaluator(
            [
                AnswerCorrectness(
                    llm,
                    OpenAIEmbedder(openai_client, usage=run.usage),
                    model=DEFAULT_ANSWER_MODEL,
                ),
                ContextRecall(llm, model=DEFAULT_ANSWER_MODEL),
                Faithfulness(llm, model=DEFAULT_ANSWER_MODEL),
            ]
        )
        with trace.section(f"Score {len(samples)} answers (LLM judges)"):
            report = evaluator.evaluate(samples)

        with trace.section("Report"):
            log_report(report)
            results_path = LOG_DIR / f"{process_name}_results.csv"
            write_results(report, results_path)
            trace.field("results", f"per question in {results_path}")


def log_report(report: EvaluationReport) -> None:
    trace.table(
        "scores",
        [
            {
                "#": index,
                "question": preview(row.sample.question, 45),
                "latency": format_duration(row.sample.latency),
                **{name: format_score(score) for name, score in row.scores.items()},
            }
            for index, row in enumerate(report.rows, start=1)
        ],
        max_width=45,
    )
    mean_latency = report.mean_latency()
    trace.fields(
        {
            **{f"mean {name}": format_score(mean) for name, mean in report.means().items()},
            "mean latency": "n/a" if mean_latency is None else format_duration(mean_latency),
            "failed": f"{len(report.failed())}/{len(report.rows)} questions",
        }
    )
    for sample in report.failed():
        trace.field("failed", f"{sample.question}: {sample.error}", level=logging.WARNING)


def write_results(report: EvaluationReport, path: Path) -> None:
    """One row per question, like the book's final ``test_data`` table."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["question", "ground_truth", "answer", "error", "latency", *report.metric_names]
        )
        for row in report.rows:
            sample = row.sample
            writer.writerow(
                [
                    sample.question,
                    sample.ground_truth,
                    sample.answer,
                    sample.error or "",
                    f"{sample.latency:.3f}",
                    *(format_score(row.scores[name]) for name in report.metric_names),
                ]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--process-name",
        default="ch08_evaluate_agent_movies",
        help="Names the log file written to logs/<process_name>.log, and the results CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N benchmark questions (default: all 17).",
    )
    parser.add_argument(
        "--rescore",
        type=Path,
        default=None,
        help=(
            "Skip the agent and only re-score the answers in a samples file an "
            "earlier run wrote (logs/<process_name>_samples.json)."
        ),
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help=(
            "Split each question into atomic questions, each routed to a tool on "
            "its own (the book's ch08 agent), instead of ch05's one rewritten question."
        ),
    )
    add_verbose_argument(parser)
    args = parser.parse_args()
    if args.rescore is not None and args.decompose:
        parser.error("--decompose has no effect with --rescore, which doesn't run the agent")
    main(args.process_name, args.limit, args.rescore, args.decompose, args.verbose)
