"""Re-implementation of ch08 (template/kg-rag/notebooks/ch08.ipynb):
evaluating ch05's agentic RAG over the movies graph against the book's
benchmark. Each benchmark question's Cypher gives its ground truth; the
agent answers it (timed); then three RAGAS-style LLM-judged metrics score
every answer: answer correctness, context recall, and faithfulness.

See examples/README.md for the metrics and where this departs from the
book. Requires OPENAI_API_KEY in .env (see .env.example). Every question
costs several agent and judge calls, so --limit runs only the first N.
Answered samples are saved, and --rescore re-runs only the judges on them.
Run from the repo root:

    .venv/Scripts/python.exe examples/evaluate_agent_movies.py \\
        --process-name ch08_evaluate_agent_movies
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import logzero
from constants import DEFAULT_ANSWER_MODEL, LOG_DIR
from dotenv import load_dotenv
from logzero import logger
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
    load_benchmark,
    load_samples,
    save_samples,
)
from graphrag.graph import KuzuCypherGraphStore
from graphrag.llm import OpenAILLMClient
from graphrag.retrieval import OpenAIEmbedder
from graphrag.settings import Settings


def main(process_name: str, limit: int | None, rescore: Path | None) -> None:
    # Windows defaults stdout/the log file to the cp1252 codepage, which
    # can't encode some names in the movies dataset (e.g. "Miloš Forman");
    # force UTF-8 so those log lines don't get silently dropped.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logzero.logfile(LOG_DIR / f"{process_name}.log", encoding="utf-8")

    load_dotenv()
    settings = Settings.from_env()
    openai_client = OpenAI(api_key=settings.openai_api_key)
    llm = OpenAILLMClient(openai_client)

    if rescore is not None:
        # Judges only: score answers an earlier run saved, so a change to a
        # metric is measured on the same answers.
        samples = load_samples(rescore)[:limit]
        logger.info(f"Re-scoring {len(samples)} samples from {rescore}")
    else:
        cases = load_benchmark(MOVIES_BENCHMARK_PATH)[:limit]
        # Kept apart from the other chapters' databases: seeding drops and
        # recreates its tables on every run.
        db_path = settings.kuzu_db_path.with_name(f"{settings.kuzu_db_path.name}_ch08")
        with KuzuCypherGraphStore(db_path) as store:
            MovieGraphSeeder(store).seed()
            agent = build_movie_agent(llm, store)
            samples = BenchmarkRunner(store, agent.respond).run(cases)
        samples_path = LOG_DIR / f"{process_name}_samples.json"
        save_samples(samples, samples_path)
        logger.info(f"Answered samples written to {samples_path}")

    evaluator = Evaluator(
        [
            AnswerCorrectness(llm, OpenAIEmbedder(openai_client), model=DEFAULT_ANSWER_MODEL),
            ContextRecall(llm, model=DEFAULT_ANSWER_MODEL),
            Faithfulness(llm, model=DEFAULT_ANSWER_MODEL),
        ]
    )
    report = evaluator.evaluate(samples)

    log_report(report)
    results_path = LOG_DIR / f"{process_name}_results.csv"
    write_results(report, results_path)
    logger.info(f"Per-question results written to {results_path}")


def format_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.3f}"


def log_report(report: EvaluationReport) -> None:
    for row in report.rows:
        scores = ", ".join(f"{name}={format_score(s)}" for name, s in row.scores.items())
        logger.info(f"--- {row.sample.question} ---")
        logger.info(f"Ground truth: {row.sample.ground_truth}")
        logger.info(f"Answer: {row.sample.error or row.sample.answer}")
        logger.info(f"Latency: {row.sample.latency:.1f}s; {scores}")

    logger.info("=== Summary ===")
    for name, mean in report.means().items():
        logger.info(f"{name}: {format_score(mean)}")
    logger.info(f"mean latency: {format_score(report.mean_latency())}s")
    logger.info(f"failed questions: {len(report.failed())}/{len(report.rows)}")


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
    args = parser.parse_args()
    main(args.process_name, args.limit, args.rescore)
