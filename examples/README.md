# Examples

Runnable scripts that exercise `graphrag` (`src/graphrag`), one per book
chapter (see [template/kg-rag](../template/kg-rag) for the original
notebooks and [CLAUDE.md](../CLAUDE.md) for why this repo re-implements
them). Each section below documents what its script does and any
non-obvious mechanics worth understanding before you read the code.

## Chapter 2 — PDF ingestion, vector search, and hybrid search

Script: [vector_similarity_and_hybrid_search.py](vector_similarity_and_hybrid_search.py)
(re-implements `template/kg-rag/notebooks/ch02.ipynb`).

Run from the repo root (requires `OPENAI_API_KEY` in `.env`, see
`.env.example`):

```
.venv/Scripts/python.exe examples/vector_similarity_and_hybrid_search.py --process-name ch02_pdf_qa
```

Pipeline: download a PDF → extract text (`PdfLoader`) → chunk it
(`TextChunker`) → embed each chunk (`OpenAIEmbedder`) → store chunks +
embeddings in Kuzu (`KuzuGraphStore`) → answer a question two ways: plain
vector search, then hybrid (vector + keyword) search. `--process-name`
names the log file written to `logs/<process_name>.log` (see
[`graphrag.settings`](../src/graphrag/settings.py) and the script's use of
`logzero`).

### How Kuzu's vector search works (HNSW)

Set up in
[`KuzuGraphStore._create_indexes()`](../src/graphrag/graph/kuzu_store.py)
via `CREATE_VECTOR_INDEX('Chunk', ..., 'embedding', metric := 'cosine')`.

**Structure:** Kuzu builds an **HNSW graph** (Hierarchical Navigable Small
World) over the embedding vectors — not a tree or a hash, but a graph
where each vector is a node connected to a handful of its nearest
neighbors. It has two layers:

- a **lower layer** containing every vector, densely connected
- a sparser **upper layer** containing a random sample of vectors
  (Kuzu defaults to ~5% — see the `pu` parameter via `CALL SHOW_INDEXES()`),
  used as highway shortcuts across the graph

**Search:** a query starts at an entry point in the sparse upper layer and
greedily hops to whichever neighbor is closer to the query vector, layer
by layer, until it drops into the dense lower layer and refines the
answer there. This is why it's *fast* — instead of comparing the query
against every chunk (brute force), it only touches a small, graph-local
neighborhood — and *approximate* — the greedy walk can miss the true
nearest neighbor if it goes down a locally-suboptimal path. At this
corpus size (dozens of chunks) that trade-off is irrelevant; it matters
once you're indexing millions of vectors.

**Distance vs. score:** `QUERY_VECTOR_INDEX` returns **cosine distance**
(0 = identical, larger = further apart), not similarity — the opposite
direction from what keyword search returns. That's why
[`KuzuGraphStore.vector_search()`](../src/graphrag/graph/kuzu_store.py)
does `score = 1.0 - distance`: so "higher is better" is consistent across
both search methods and `hybrid_search` can compare them directly.

### How Kuzu's keyword search works (FTS / BM25)

Set up alongside via `CREATE_FTS_INDEX('Chunk', ..., ['text'])`.

**Structure:** this is the classic **inverted index** used by every text
search engine (Lucene, Postgres FTS, etc.). At build time, Kuzu tokenizes
each chunk's `text`, lowercases it, strips English stopwords, and stems
words to their root form (e.g. "companions" → "companion") by default. It
then stores, per term, which chunks contain it and how often.

**Search:** the query string goes through the same tokenize/stem
pipeline, then Kuzu scores every chunk that shares a term with the query
using **Okapi BM25**, which combines:

- **term frequency** — how often the term appears in that chunk (with
  diminishing returns, so repeating a word 10x doesn't score 10x)
- **inverse document frequency** — rare terms across the whole corpus
  count for more than common ones
- **length normalization** — so a short chunk matching a term isn't
  unfairly out- or under-scored relative to a long one

This is purely lexical: it matches (stemmed) words, not meaning. One
quirk worth knowing: because the default stopword list is fairly
aggressive, an unexpectedly common word can return zero hits if it
happens to be on that list.

### Why combine them (`hybrid_search`)

Vector search catches paraphrases and synonyms a keyword search would
miss (e.g. "loyal companions" matching a query about "dogs"); keyword
search catches exact terminology a vector embedding might blur past.
Since cosine similarity and BM25 scores live on completely different
numeric scales,
[`GraphStore._merge_normalized()`](../src/graphrag/graph/store.py)
divides each result set by its own max score before merging — the same
normalize-then-union trick the book's original Neo4j `hybrid_query` used
— then keeps the best normalized score per chunk index.
