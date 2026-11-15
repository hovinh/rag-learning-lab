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
  requirements.txt`) to make `.venv` match the lockfile exactly. `pip-sync`
  uninstalls the editable `graphrag` package (it isn't in the lockfile), so
  follow it with `pip install -e . --no-deps`.

## Coding standards

See [docs/python.md](docs/python.md) for Python style and OOP conventions
used throughout `src/`.

## Workflow

For every new change request, propose a plan and get explicit approval from
the user before making any code changes.

## Repo-feature deck: ALWAYS keep it in sync

[docs/repo-feature/build_deck.py](docs/repo-feature/build_deck.py) builds a
slide deck (`docs/repo-feature/repo-feature.pptx`) describing the whole repo:
purpose, layout, architecture, every public class, every chapter's flow
with a code example, and the dev workflow. It's the repo's reference
overview, so it must never fall behind the code.

- **Every new development updates the deck in the same change.** Adding,
  renaming or removing a public class/function, package, example script,
  CLI flag, dependency, or workflow step means editing `SLIDES` in
  `build_deck.py`, then rebuilding with
  `.venv/Scripts/python.exe docs/repo-feature/build_deck.py`.
- Slide content is data (`SLIDES`); `DeckBuilder` only renders it. Change
  the data, not the renderer, unless the layout itself needs to change.
- [tests/docs/test_repo_feature_deck.py](tests/docs/test_repo_feature_deck.py)
  enforces this: it fails when any public `graphrag` name or
  `examples/*.py` script is missing from the deck, or when the deck names a
  class or `.py` file that no longer exists. Fix the deck, never loosen
  the test.
- After changing the deck, check that the slides still render cleanly
  (text not overflowing its box).
- Only the script is committed. Don't commit the built `.pptx` until the
  user says the deck is finalized.

## Secrets

Never commit `.env`. Only `OPENAI_API_KEY` should be required at runtime
once Neo4j is replaced by an embedded Kuzu database.
