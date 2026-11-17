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
from pathlib import Path

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR
from dotenv import load_dotenv

from graphrag.domains.movies import MOVIES_BENCHMARK_PATH, MovieGraphSeeder, build_movie_agent
from graphrag.evaluation import (
    AnswerCorrectness,
    BenchmarkRunner,
    ContextRecall,
    Evaluator,
    Faithfulness,
    load_benchmark,
    load_samples,
    log_report,
    save_samples,
)
from graphrag.graph import KuzuCypherGraphStore
from graphrag.observability import RunLogging, Trace
from graphrag.services import OpenAIServices
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
        services = OpenAIServices.create(settings, usage=run.usage)
        llm = services.llm

        if rescore is not None:
            # Judges only: score answers an earlier run saved, so a change to a
            # metric is measured on the same answers.
            samples = load_samples(rescore)[:limit]
            trace.field("samples", f"{len(samples)} loaded from {rescore}")
        else:
            cases = load_benchmark(MOVIES_BENCHMARK_PATH)[:limit]
            # Kept apart from the other chapters' databases: seeding drops and
            # recreates its tables on every run.
            with KuzuCypherGraphStore(settings.db_path_for("ch08")) as store:
                with trace.section("Seed the movies graph"):
                    MovieGraphSeeder(store).seed()
                agent = build_movie_agent(
                    llm, store, decompose=decompose, model=DEFAULT_ANSWER_MODEL
                )
                with trace.section(f"Answer {len(cases)} benchmark questions"):
                    samples = BenchmarkRunner(store, agent.respond).run(cases)
            samples_path = LOG_DIR / f"{process_name}_samples.json"
            save_samples(samples, samples_path)
            trace.field("samples", f"written to {samples_path}")

        evaluator = Evaluator(
            [
                AnswerCorrectness(llm, services.embedder, model=DEFAULT_ANSWER_MODEL),
                ContextRecall(llm, model=DEFAULT_ANSWER_MODEL),
                Faithfulness(llm, model=DEFAULT_ANSWER_MODEL),
            ]
        )
        with trace.section(f"Score {len(samples)} answers (LLM judges)"):
            report = evaluator.evaluate(samples)

        with trace.section("Report"):
            log_report(report)
            results_path = LOG_DIR / f"{process_name}_results.csv"
            report.write_csv(results_path)
            trace.field("results", f"per question in {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    RunLogging.add_arguments(parser, default_process_name="ch08_evaluate_agent_movies")
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
    args = parser.parse_args()
    if args.rescore is not None and args.decompose:
        parser.error("--decompose has no effect with --rescore, which doesn't run the agent")
    main(args.process_name, args.limit, args.rescore, args.decompose, args.verbose)
