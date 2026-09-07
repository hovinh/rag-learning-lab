# rag-learning-lab

## Purpose

This repo re-implements the material from the **Essential GraphRAG** book
(reference copy vendored at [template/kg-rag](template/kg-rag), notebooks
`ch02`–`ch08`) as a proper, reusable Python codebase instead of linear
notebook scripts.

Two goals drive every design decision here:

1. **Faithfully cover the book's concepts** — PDF ingestion & chunking,
   query understanding (step-back prompting), LLM-driven knowledge graph
   construction, text2cypher, structured extraction, embeddings/vector
   retrieval, and agentic RAG (query rewriting, tool routing, critique loop).
2. **Re-implement it with OOP for reusability** — the book's notebooks use
   module-level globals and free functions (e.g. a shared `neo4j_driver`,
   `chat()`/`embed()` helpers). This project instead wraps each concern in a
   class with a clear interface (graph store, chunker, embedder, LLM client,
   text2cypher generator, agent/tool router) so pieces can be swapped,
   mocked, and unit-tested independently.

`template/kg-rag` is **reference-only**. It's its own nested git repo
(vendored copy of the book's companion repo) — do not edit it, do not commit
into it, and do not add it as a real git submodule. Read it to understand
what a chapter is doing, then write the equivalent as a class under `src/`.

## Key architectural decision: Kuzu instead of Neo4j

The book uses Neo4j (a server you must run separately, `neo4j` Python
driver, Cypher via `driver.execute_query`). This project replaces Neo4j with
**[Kuzu](https://kuzudb.com/)**:

- Embedded, in-process, open-source (MIT) graph database — no server/Docker
  required, just `pip install kuzu`.
- Speaks Cypher, so the book's Cypher queries and the `text2cypher` approach
  port over with minimal changes (mainly: connection/session API differs,
  and Kuzu requires explicit node/relationship table schemas before you
  insert data — there's no implicit label creation like Neo4j).
- Good fit for a learning-lab repo: a graph lives in a local directory, is
  trivial to reset, and needs no credentials — compare the old
  `.env.example` (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`) with just
  a local Kuzu database path.

Design the graph-store layer behind an interface/abstract base class rather
than importing `kuzu` throughout the codebase, so the backend can be swapped
again later without touching callers.

## Dependency management

- Python 3.11, virtualenv at `.venv/`.
- All dependencies (runtime and dev tooling alike) are declared in the one
  `requirements.in` and compiled to a pinned `requirements.txt` with
  **pip-tools** (`pip-compile requirements.in`). Do not hand-edit
  `requirements.txt` — edit `requirements.in` and recompile.
- Use `pip-sync requirements.txt` (or `python -m piptools sync
  requirements.txt`) to make `.venv` match the lockfile exactly.

## Coding standards

See [docs/python.md](docs/python.md) for Python style and OOP conventions
used throughout `src/`.

## Secrets

Never commit `.env`. Only `OPENAI_API_KEY` should be required at runtime
once Neo4j is replaced by an embedded Kuzu database.
