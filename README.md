# rag-learning-lab

The book *Essential GraphRAG* (Tomaž Bratanič and Oskar Hane), rebuilt as a
reusable Python package.

The book teaches GraphRAG in notebooks that share global helpers. Here each
concern is its own class: graph store, chunker, embedder, LLM client,
retriever, text2cypher and agent. Each chapter becomes a **recipe** composed
from those same pieces, and you can swap, mock or test each piece on its
own. Neo4j is replaced by [Kuzu](https://kuzudb.com/), an embedded graph
database, so there is no server to run.

🧱 For a 70-second visual tour, open
[docs/linkedin/explainer.html](docs/linkedin/explainer.html) in a browser.

## Quick start

You need Python 3.11+ and an OpenAI API key.

```bash
git clone https://github.com/hovinh/rag-learning-lab.git
cd rag-learning-lab

python -m venv .venv
.venv/Scripts/activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps

cp .env.example .env            # then set OPENAI_API_KEY in .env
```

Then run any example from the repo root, for example chapter 2:

```bash
python examples/vector_similarity_and_hybrid_search.py
```

The first run downloads its data into `data/` and builds an index in a
local Kuzu database (`data/kuzu_db_<name>`). Later runs reuse the index and
go straight to answering questions.

## The recipes

Each recipe lives in `graphrag.recipes` and has a runnable example.

| Chapter | Recipe | What it does | Example |
| --- | --- | --- | --- |
| ch02 | `VectorRag` | chunk, embed, answer with vector or hybrid (vector + keyword) search | [vector_similarity_and_hybrid_search.py](examples/vector_similarity_and_hybrid_search.py) |
| ch03 | `ParentChildRag` | match small chunks, answer from their larger parents, with optional step-back prompting | [parent_child_stepback_rag.py](examples/parent_child_stepback_rag.py) |
| ch04 | `Text2CypherQA` | turn a question into a Cypher query and run it on a movies graph | [text2cypher_movies.py](examples/text2cypher_movies.py) |
| ch05 | `AgenticRag` | an agent that rewrites the question, picks a tool, critiques its answer and fills gaps | [agentic_rag_movies.py](examples/agentic_rag_movies.py) |
| ch06 | `StructuredGraphExtraction` | extract a typed contract with structured outputs and import it as a graph | [structured_extraction_contracts.py](examples/structured_extraction_contracts.py) |
| ch07 | `GraphRag` | Microsoft-style GraphRAG over *The Odyssey*: entity graph, communities, global and local search | [graphrag_odyssey.py](examples/graphrag_odyssey.py) |
| ch08 | `RagEvaluation` | score a recipe for answer correctness, context recall and faithfulness | [evaluate_agent_movies.py](examples/evaluate_agent_movies.py) |
| extra | `HypotheticalQuestionRag` | embed the questions each chunk answers instead of the chunk | [benchmark_text_recipes_odyssey.py](examples/benchmark_text_recipes_odyssey.py) |

All the examples accept these flags:

- `--process-name NAME` names the log file, `logs/NAME.log`.
- `--verbose` prints full prompts and payloads to the console.
- `--reindex` rebuilds the index even if one already exists.

Some examples have extra flags, such as `--limit`, `--rescore` and
`--books`. Run a script with `--help` to see them.

> **Cost:** most examples make a handful of OpenAI calls. The exception is
> building the ch07 graph, which costs roughly 100k tokens per book of
> *The Odyssey*, and the full benchmark, which took about 1.2M tokens. Both
> keep their indexes, so you pay for them once.

[examples/README.md](examples/README.md) walks through every chapter: how
it works, how to read the logs, and where it departs from the book.

## Using it as a library

The recipes take their pieces as constructor arguments, so you assemble a
pipeline from your own store, documents and model:

```python
from pathlib import Path

from dotenv import load_dotenv

from graphrag.graph import KuzuGraphStore
from graphrag.ingestion import PdfSource
from graphrag.recipes import VectorRag
from graphrag.services import OpenAIServices
from graphrag.settings import Settings

load_dotenv()
settings = Settings.from_env()
services = OpenAIServices.create(settings)

with KuzuGraphStore(settings.db_path_for("mydocs"),
                    embedding_dim=services.embedder.dimensions) as store:
    rag = VectorRag(store, services.embedder, services.llm)
    if not rag.is_indexed():
        rag.index(PdfSource("https://arxiv.org/pdf/1709.00666.pdf", Path("data/paper.pdf")))

    result = rag.query("What is this paper about?", search="hybrid")
    print(result.answer)
```

To swap a piece, pass a different implementation of its interface:
`GraphStore`, `Embedder`, `LLMClient`, `Retriever` or `Tool`. To build a new
recipe, compose `graphrag.pipeline` steps in a `Pipeline`.

## Which recipe should I use?

[examples/benchmark_text_recipes_odyssey.py](examples/benchmark_text_recipes_odyssey.py)
scores seven text-recipe variants on the same 20 questions about Book I of
*The Odyssey*:

| Variant | Answer correctness |
| --- | --- |
| **ParentChildRag** | **0.785** |
| ParentChildRag + step-back | 0.749 |
| HypotheticalQuestionRag | 0.733 |
| VectorRag (vector) | 0.651 |
| VectorRag (hybrid) | 0.645 |
| GraphRag (local search) | 0.616 |
| GraphRag (global search) | 0.475 |

This is one run on one book, so gaps under about 0.05 are noise. GraphRAG
is built for corpora too large for a few chunks to cover, which a single
book isn't. Full results, caveats and a recipe guide are in
[examples/README.md](examples/README.md#benchmarking-the-text-recipes).

## Repo layout

```
src/graphrag/
  graph/          GraphStore interface, Kuzu stores, text2cypher
  ingestion/      PDF loading, section splitting, chunking, document sources
  llm/            LLMClient interface, OpenAI client, query rewriting
  retrieval/      Embedder, Retriever and their implementations, GraphRAG search
  extraction/     entity/relationship extraction, communities, reports
  agent/          Agent, tool router, critic, query planners, tools
  pipeline/       Pipeline and its reusable steps
  recipes/        one predefined pipeline per chapter
  evaluation/     benchmarks, LLM-judged metrics, runner
  domains/        domain packs: movies, contracts, odyssey
  observability/  run logging, tracing, token usage
  prompts/        every prompt as a .txt file
examples/         one runnable script per chapter, plus the benchmark
tests/            unit tests (no API key or network needed)
docs/             coding standards, repo overview deck, LinkedIn explainer
```

## Development

```bash
pytest          # unit tests
ruff check .    # lint
mypy            # type check
```

Dependencies are pinned with pip-tools. Edit `requirements.in`, then run
`pip-compile requirements.in` and `pip-sync requirements.txt`. `pip-sync`
removes the editable package, so run `pip install -e . --no-deps` again
afterwards.

Code style and OOP conventions are in [docs/python.md](docs/python.md).

## Credits

Based on *Essential GraphRAG* by Tomaž Bratanič and Oskar Hane (Manning),
and its companion notebooks at
[tomasonjo/kg-rag](https://github.com/tomasonjo/kg-rag). This repo is an
independent learning re-implementation, not affiliated with the authors.

## License

[MIT](LICENSE)
