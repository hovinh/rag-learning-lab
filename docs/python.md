# Python Best Practices

This document sets the conventions for re-implementing the **Essential
GraphRAG** book (see [template/kg-rag](../template/kg-rag) for the reference
notebooks) as a reusable, object-oriented codebase. See [CLAUDE.md](../CLAUDE.md)
for the overall project intent and the Neo4j → Kuzu decision.

## Why OOP here

The book's notebooks are procedural: module-level globals (`neo4j_driver`,
`open_ai_client`) and free functions (`chat()`, `embed()`, `chunk_text()`)
that every chapter imports and mutates directly. That's fine for a notebook
walking through a chapter, but it doesn't compose, isn't testable in
isolation, and hides which pieces depend on which.

Each notebook concept becomes a class with a narrow, explicit interface:

| Book concept                        | Class (indicative)                          |
| ------------------------------------ | -------------------------------------------- |
| PDF ingestion, chunking, vector & hybrid search (ch02) | `PdfLoader`, `TextChunker`, `Embedder`, `GraphStore` |
| Step-back prompting, parent-child retrieval (ch03)     | `QueryRewriter`, `ParentChildGraphStore`, `TokenCounter` |
| text2cypher (ch04)                                     | `Text2Cypher`, `CypherGraphStore`, `MovieGraphSeeder` |
| Agentic RAG: tool routing, critique loop (ch05)        | `Agent`, `QueryPlanner` (`QueryUpdater`, ch08's `QueryDecomposer`), `ToolRouter`, `AnswerCritic`, `Tool` |
| Structured extraction into a KG (ch06)                 | `Contract` models, `StructuredExtractor`, `ContractGraphImporter` |
| LLM-driven KG construction, local/global search (ch07) | `EntityRelationshipExtractor`, `EntityGraphImporter`, `DescriptionSummarizer`, `KuzuLouvainCommunityDetector`, `CommunityReporter`, `EntityEmbeddingIndexer`, `GlobalSearch`, `LocalSearch` |
| RAG evaluation against a benchmark (ch08)              | `BenchmarkRunner`, `Evaluator`, `Metric` (`AnswerCorrectness`, `ContextRecall`, `Faithfulness`), `StatementExtractor` |

Guidelines:

- **Depend on interfaces, not concrete clients.** e.g. `GraphStore` is an
  ABC; `KuzuGraphStore` is the only implementation today, but callers never
  import `kuzu` directly. Same idea for `LLMClient` (wraps the OpenAI SDK).
- **Constructor-inject dependencies** (a `GraphStore`, an `LLMClient`)
  instead of reaching for module-level singletons. This is what makes
  classes mockable in tests.
- **One class, one responsibility.** Don't let `Agent` also know how to
  talk to the graph database — it should hold a `GraphStore`/tool and call
  it.
- **Prefer composition over inheritance.** Reserve inheritance for genuine
  is-a relationships (e.g. `KuzuGraphStore(GraphStore)`); build behavior by
  passing collaborators in, not by subclassing for code reuse.
- It's fine for a class to be a thin wrapper at first — the point is the
  seam it creates, not premature abstraction. Don't add configurability or
  extension points a chapter doesn't need yet.

## Project layout

```
src/
  graphrag/
    ingestion/      # PdfLoader, TextChunker, SectionSplitter
    llm/             # LLMClient (OpenAI wrapper, incl. structured outputs), QueryRewriter, TokenCounter
    graph/           # GraphStore/CypherGraphStore ABCs, Kuzu stores, Text2Cypher
    datasets/        # sample data (MovieGraphSeeder + its CSVs, ch06's contract text, ch07's OdysseyCorpus,
                     # ch08's movies benchmark)
    retrieval/       # Embedder, EntityEmbeddingIndexer, GlobalSearch, LocalSearch
    prompts/         # PromptRepository + prompt text files
    pipeline/        # Pipeline runner + steps shared across chapters
    agent/           # Agent, QueryPlanner/QueryUpdater/QueryDecomposer, ToolRouter, AnswerCritic, Tool subclasses
    extraction/      # Pydantic extraction models, StructuredExtractor, ContractGraphImporter,
                     # ch07's entity extraction, summarization, communities and reports
    evaluation/      # ch08: benchmark loading, BenchmarkRunner, LLM-judged Metrics, Evaluator
    observability/   # Trace (structured, sectioned logging), RunLogging, UsageMeter
tests/
  ...                # mirrors src/graphrag structure
docs/
  python.md          # this file
notebooks/            # optional: thin notebooks that exercise src/graphrag,
                       # not where logic lives
```

Notebooks (if kept) should import from `graphrag` and demonstrate usage —
they should not contain business logic that isn't otherwise unit-tested.

## Style

- **Formatting/linting**: [ruff](https://docs.astral.sh/ruff/) for both
  linting and formatting (`ruff check`, `ruff format`). One tool, one config,
  no debate.
- **Type hints everywhere.** Public methods and functions must be fully
  typed (parameters + return type). Use built-in generics (`list[str]`,
  `dict[str, Any]`) — Python 3.11, no need for `typing.List`/`typing.Dict`.
- **Type checking**: [mypy](https://mypy-lang.org/) run in CI/pre-commit;
  keep `src/` clean of `Any` leaking across module boundaries where a real
  type is known.
- **Docstrings** only where behavior isn't obvious from the signature —
  don't restate the method name in prose. No comments explaining *what*
  code does; only *why*, when it's non-obvious (a workaround, an invariant,
  a constraint from the book/Kuzu API).
- **Naming**: `PascalCase` for classes, `snake_case` for functions/methods/
  variables, `UPPER_SNAKE_CASE` for module-level constants (e.g. prompt
  templates that are truly constant).
- **Errors**: raise specific exceptions (define custom exception classes
  under each package, e.g. `GraphQueryError`) rather than letting a bare
  `Exception` propagate or silently swallowing errors (the book's
  notebooks do this — e.g. catching `json.JSONDecodeError` and just
  `print`-ing — don't carry that pattern forward).
- **Config/secrets**: read from environment variables (`.env`, loaded via
  `python-dotenv`) at the composition root (e.g. `main.py` / a `Settings`
  class), not scattered `os.environ.get(...)` calls inside business logic
  classes.

## Logging

- Library code logs through a module-level `Trace(__name__)`
  (`graphrag.observability`), never `print` and never by configuring
  handlers itself. Only the application sets logging up, with
  `RunLogging`.
- INFO tells the story in short lines (a summary, a preview, a count). The
  full payload behind it goes to DEBUG (`Trace.summary` does both), which
  the log file always records.
- Group work in `Trace.section(...)` so it's indented and timed. Pipeline
  steps get one automatically.

## Testing

- `pytest`, tests mirror the `src/graphrag` package structure.
- Unit tests mock the `GraphStore`/`LLMClient` interfaces — no real Kuzu
  database or OpenAI API calls in unit tests.
- Reserve real-Kuzu integration tests (using a temp on-disk or in-memory
  Kuzu database) for a separate marked test suite.

## Dependency management

- All dependencies (runtime and dev tooling) are declared in the single
  `requirements.in` and compiled with `pip-tools` into a pinned
  `requirements.txt`. Never hand-edit `requirements.txt` or `pip install`
  ad hoc into `.venv` without updating `requirements.in` and recompiling —
  the lockfile is the source of truth for what's installed.
- One virtualenv at `.venv/`, Python 3.11.
- Workflow: edit `requirements.in` → `pip-compile requirements.in` →
  `pip-sync requirements.txt`.
