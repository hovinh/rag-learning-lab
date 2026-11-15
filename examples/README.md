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

## Chapter 3 — Step-back prompting and parent-child retrieval

Script: [parent_child_stepback_rag.py](parent_child_stepback_rag.py)
(re-implements `template/kg-rag/notebooks/ch03.ipynb`).

Run from the repo root (requires `OPENAI_API_KEY` in `.env`, see
`.env.example`):

```
.venv/Scripts/python.exe examples/parent_child_stepback_rag.py --process-name ch03_parent_child_stepback_rag
```

Pipeline: download the same arxiv PDF as ch02 → split it into titled
sections (`SectionSplitter`) → chunk each section into large "parent"
chunks and each parent into small "child" chunks (`TextChunker`, run
twice with different sizes) → embed only the children (`OpenAIEmbedder`)
→ store the parent/child hierarchy in Kuzu (`KuzuParentChildGraphStore`)
→ rewrite the question with step-back prompting (`QueryRewriter`) →
retrieve at parent granularity (`parent_retrieval`) → answer with the
*original* question using the retrieved parent text as context.

### Why parent-child retrieval

A single 500-character chunk is often too small to carry the full context
an answer needs — but embedding a large 2000-character chunk directly
makes similarity search fuzzier (more topics blended into one vector).
Parent-child retrieval gets both: small, precise child chunks are what
gets embedded and searched, while the *parent* chunk they came from
(their surrounding section text) is what's returned to the LLM. See
[`KuzuParentChildGraphStore.parent_retrieval()`](../src/graphrag/graph/kuzu_parent_child_store.py):
it vector-searches `Child` nodes, walks each hit's `HAS_CHILD` edge back
to its `Parent`, and keeps the best (`min` distance) child score per
parent — mirroring the book's Neo4j `parent_retrieval` Cypher.

### Why step-back prompting

[`QueryRewriter.step_back()`](../src/graphrag/llm/query_rewriter.py) asks
the LLM to paraphrase a specific question into a more generic one before
embedding it for retrieval (e.g. "Who was Einstein's collaborator on
sound reproduction?" → "Who were Einstein's professional associates?").
The rewritten question is only used to *retrieve* context; the original
question is still what's sent to the LLM for the final answer, so the
answer stays scoped to what was actually asked.

## Chapter 4 — text2cypher

Script: [text2cypher_movies.py](text2cypher_movies.py)
(re-implements `template/kg-rag/notebooks/ch04.ipynb`).

Run from the repo root (requires `OPENAI_API_KEY` in `.env`, see
`.env.example`):

```
.venv/Scripts/python.exe examples/text2cypher_movies.py --process-name ch04_text2cypher_movies
```

Pipeline: seed a movies graph into Kuzu (`MovieGraphSeeder`) → read its
schema (`KuzuCypherGraphStore.get_schema()`) → have the LLM write a Cypher
query for the question (`Text2Cypher`) → run that query against the graph
and log the rows. The book's notebook stops after printing the Cypher;
running it is added here because it's what proves the query is valid, and
ch05's `text2cypher` tool needs it anyway.

Chapters 2 and 3 retrieve unstructured *text*. text2cypher answers from
*structured* data instead: questions like "Who directed the most movies?"
need counting and aggregation across many nodes, which no amount of chunk
similarity search can do, but which one Cypher query answers exactly.

### The movies dataset

The book assumes Neo4j's "Movies" demo graph is already loaded. Kuzu has
no such sample, so
[`graphrag/datasets/movies/`](../src/graphrag/datasets/movies/) holds a
small hand-curated graph in the same shape: 14 films, 48 people, and
`ACTED_IN` (with `roles`), `DIRECTED`, `PRODUCED`, and `WROTE`
relationships. The films are picked so the questions in
`template/kg-rag/data/benchmark_data.csv` (used by ch08) have answers, e.g.
Top Gun, The Matrix, and several Tom Hanks films. Cast, crew, years, and
birth years are real; **`votes` values are illustrative**, made up only so
the benchmark's "more than 5000 votes" question has something to filter.

[`MovieGraphSeeder.seed()`](../src/graphrag/datasets/movie_graph.py)
drops and recreates the tables and then bulk-loads one CSV per table with
`COPY ... FROM`. Unlike Neo4j, Kuzu needs every node and relationship table
declared (`CREATE NODE TABLE` / `CREATE REL TABLE`) before data goes in.

### How the schema is read from Kuzu

The book's `get_schema()` calls APOC's `apoc.meta.data()`, which samples
the Neo4j graph to infer which labels, properties, and relationship
patterns exist. Kuzu has no APOC, but it doesn't need to infer anything:
every table is declared up front, so
[`KuzuCypherGraphStore.get_schema()`](../src/graphrag/graph/kuzu_cypher_store.py)
reads it straight from the catalog:

- `CALL show_tables()`: every node and rel table
- `CALL table_info('<table>')`: each table's properties and types
- `CALL show_connection('<rel table>')`: which node tables a rel table
  connects, giving the `(:Person)-[:DIRECTED]->(:Movie)` patterns

[`GraphSchema.to_prompt()`](../src/graphrag/graph/models.py) formats this
in the book's layout. Property types are Kuzu's (`INT64`, `STRING[]`), not
Neo4j's, since those are the types the generated query runs against.

### How the prompt is assembled

[`Text2Cypher`](../src/graphrag/graph/text2cypher.py) builds the prompt
from the book's sections, each loaded from a `text2cypher_*.txt` file in
[`graphrag/prompts/`](../src/graphrag/prompts/):

1. **Instructions and format rules**: write only Cypher, with no
   explanation and no code fences, so the output can be executed as-is.
   One line is added to the book's wording: the target is Kuzu, so the LLM
   sticks to Cypher that Kuzu supports.
2. **Schema**: the labels, properties, and relationship patterns above,
   with an instruction to use nothing else. This is the main defence
   against invented relationship types or properties.
3. **Terminology mapping** (optional): links the user's words to the
   schema, e.g. "actor/director/producer" all mean a `Person` node. Without
   it, the LLM might look for a `Director` label.
4. **Few-shot examples** (optional): question/Cypher pairs showing
   patterns the LLM might not find on its own, like the book's
   "two people who acted together most" self-join.
5. **The question.**

The book's class sets sections through a mutable `set_prompt_section()`.
Here everything except the question is fixed in the constructor, so one
instance can answer many questions, which is how ch05 will reuse it as an
agent tool. Kuzu ran the book's few-shot query (including
`WHERE p1 <> p2` on nodes) unchanged, so no dialect workaround was needed.

## Chapter 5 — Agentic RAG

Script: [agentic_rag_movies.py](agentic_rag_movies.py)
(re-implements `template/kg-rag/notebooks/ch05.ipynb`).

Run from the repo root (requires `OPENAI_API_KEY` in `.env`, see
`.env.example`):

```
.venv/Scripts/python.exe examples/agentic_rag_movies.py --process-name ch05_agentic_rag_movies
```

Ch04 always answers in one way: generate Cypher, then run it. Ch05 puts an
**agent** in front of that. The LLM decides how to answer each question,
and checks whether the answer is complete before replying. The script
seeds the same movies graph as ch04 (its own `_ch05` Kuzu database) and
asks the book's two questions.

### The flow

[`Agent.run()`](../src/graphrag/agent/agent.py) takes one question through
four LLM-driven stages, keeping a **history** of the answers gathered so
far:

1. **Query update**
   ([`QueryUpdater`](../src/graphrag/agent/query_updater.py)): rewrite the
   question into one atomic, specific question, filling in references from
   the history ("that person" → "Keanu Reeves"). The response is a JSON
   object (`json_mode=True`).
2. **Tool routing** ([`ToolRouter`](../src/graphrag/agent/router.py)): the
   LLM is offered every tool's function-calling definition and picks one
   or more (`LLMClient.choose_tools()`). The router runs them. Results go
   into the history as an assistant message, *"For the question X, we have
   the answer Y"*. That's the book's wording, from `agent_tool_answer.txt`.
3. **Critique** ([`AnswerCritic`](../src/graphrag/agent/critic.py)): does
   the history fully answer the *original* question? If not, the critic
   returns follow-up questions. They're joined into one question and go
   through steps 1–2 **once** more, as in the book. This is a single
   critique pass, not a loop.
4. **Final answer**: the LLM answers the original question from the
   history alone. It's told to say when information is missing rather
   than make anything up.

### The tools

| Tool (name the LLM sees)                         | What it does                                                                 |
| ------------------------------------------------ | ---------------------------------------------------------------------------- |
| `movie_info_by_title` (`MovieInfoByTitleTool`)   | fixed Cypher: movies whose title contains the text, with cast and directors  |
| `movies_info_by_actor` (`MoviesInfoByActorTool`) | fixed Cypher: the movies an actor is in, with each one's cast and directors  |
| `text2cypher` (`Text2CypherTool`)                | fallback: ch04's `Text2Cypher` (same terminology and example) plus execution |
| `respond` (`AnswerGivenTool`)                    | the answer is already in the history, or the user is making small talk       |

The fixed lookups handle the question shapes known in advance. They're
fast and can't produce invalid Cypher. text2cypher covers everything else,
e.g. "How many directors and producers are in the database?". If its
generated Cypher fails, the error text is returned as the tool's result
instead of being raised, so it becomes part of the history the critic sees.

Every tool subclasses [`Tool`](../src/graphrag/agent/tools.py). One `name`
serves as both the function name the LLM sees and the router's lookup key.
To add a tool, subclass `Tool` and pass an instance to `ToolRouter`.

### Where this departs from the book

- **Tool name mismatch.** The book keys its tools dict `"answer_given"`,
  but the function the LLM sees is named `"respond"`. Whenever the LLM
  picks that tool, `handle_tool_calls` raises a `KeyError`. Here there's
  only one name. An unknown tool name or a missing required argument
  raises `ToolRoutingError`.
- **`movies_info_by_actor` filters by actor.** The book's query rebinds
  `a` in an `OPTIONAL MATCH` and attaches its `WHERE` to that optional
  match, so the actor filter has no effect.
- **No duplicated names.** Two `OPTIONAL MATCH`es multiply rows, so the
  book's plain `collect()` repeats every cast member once per director.
  `collect(DISTINCT ...)` fixes that. The queries also return properties
  (`title`, `released`, `tagline`) rather than whole nodes.
- **Kuzu dialect.** `toLower` becomes `lower`, and the `cast` column is
  backtick-escaped because `cast` is a reserved word in Kuzu.
- **Fresh history per question.** The book's
  `handle_user_input(input, answers=[])` has a mutable default argument,
  so the second question silently sees the first one's answers.
- **Bad JSON raises.** The book prints "Error decoding JSON" and returns
  `[]`, even where a question string is expected. Here unusable JSON
  raises `LLMResponseError`.
- **Final answer returned whole**, not streamed, so `Agent.run()` returns
  a plain string.

### What a run looks like

On the first question, the updater names Keanu Reeves, and the router
calls *both* `movie_info_by_title("The Matrix")` and
`movies_info_by_actor("Keanu Reeves")` in one turn. The critic still asks
about his other movies, so a second round runs. The curated dataset only
has him in *The Matrix* and *The Matrix Reloaded*, so the answer can be
thin: in test runs the model said his other movies were "missing" instead
of naming *Reloaded*. The tools returned the right rows; this is the final
LLM call reading them conservatively. The second question goes to
`text2cypher`, which returns 10 directors and 4 producers, matching the
seeded graph. The critic is satisfied, so there's no second round.

## Chapter 6 — Structured extraction into a knowledge graph

Script: [structured_extraction_contracts.py](structured_extraction_contracts.py)
(re-implements `template/kg-rag/notebooks/ch06.ipynb`).

Run from the repo root (requires `OPENAI_API_KEY` in `.env`, see
`.env.example`):

```
.venv/Scripts/python.exe examples/structured_extraction_contracts.py --process-name ch06_structured_extraction_contracts
```

Chapters 2–5 query text or a graph that already exists. Ch06 **builds** a
graph from a document. You decide up front which entities matter, write
them down as Pydantic models, and have the LLM fill them in from a legal
contract. Ch07 drops the predefined schema and lets the LLM decide.

Pipeline: load the contract
([`LICENSE_AGREEMENT_PATH`](../src/graphrag/datasets/sample_contracts.py),
a 1999 licensing and web site hosting agreement from an SEC filing) →
extract a `Contract`
([`StructuredExtractor`](../src/graphrag/extraction/extractor.py)) →
import it into its own `_ch06` Kuzu database
([`ContractGraphImporter`](../src/graphrag/extraction/contract_importer.py))
→ query the parties back. The database is reset on every run.

### Structured outputs

[`LLMClient.parse()`](../src/graphrag/llm/client.py) passes the Pydantic
model as `response_format`. The API turns it into a JSON schema and
**constrains decoding to it**: the response always has every field, with
the declared types, and `contract_type` is always one of the five listed
values. This is stronger than ch05's `json_mode`, which only guarantees
valid JSON, so the prompt had to describe the shape. Here the field
`description`s in [contract_models.py](../src/graphrag/extraction/contract_models.py)
are the instructions the LLM sees for each field.

Strict mode requires every field to be present. That's why the optional
fields are `str | None` with no default (required, but may be null)
rather than having a default value.

[`StructuredExtractor`](../src/graphrag/extraction/extractor.py) isn't tied
to contracts. It takes any Pydantic model plus a system message
(`contract_extraction_system.txt` here, the book's wording) and sends the
whole document in one call. The agreement is ~33k characters, well within
the context window, so there's no chunking.

### The graph model

```
(:Organization {name})-[:HAS_PARTY {role}]->(:Contract {id, contract_type, effective_date,
    term, contract_scope, end_date, total_amount, governing_law})
(:Organization)-[:LOCATED_AT]->(:Location {full_address, address, city, state, country})
```

Organizations are keyed by `name` and locations by `full_address`, and both
are merged. A second contract with the same party links to the existing
organization node, which is how a graph of many contracts connects up.
Every import creates a new `Contract`, with a `gen_random_uuid()` id.

### Where this departs from the book

- **Tables instead of constraints.** Kuzu needs node and rel tables
  declared before any data goes in (`create_schema()`). Each key the book
  makes unique with `CREATE CONSTRAINT` is the table's primary key here.
- **Null-safe concatenation.** The book builds `fullAddress` and
  `governing_law` with Cypher's `+`, and `'a' + null` is `null`. If the
  extracted location has no street address, which the schema allows,
  `fullAddress` becomes null and the `MERGE` on it fails. Here
  `Location.full_address()` joins only the parts that are present.
- **`CREATE`, not `MERGE`, for the contract.** The book merges on
  `{id: randomUUID()}`. That id is new every time, so the `MERGE` never
  matches anything and always creates.
- **Several small queries.** The book sends one `UNWIND $data.parties`
  query with the whole nested dict as a parameter. Here there's one query
  for the contract plus one per party, with flat parameters. That avoids
  depending on Kuzu's type inference for nested structs that contain
  nulls, and each query can be tested on its own.
- **`Literal` for the contract type.** The book passes `enum=` to
  `Field`, which Pydantic v2 deprecates. A `Literal` gives the same JSON
  schema `enum`, and Pydantic validates it too.
- **Typed result.** `parse()` returns the `Contract` instance rather than
  `json.loads` of the raw content. A refusal raises `LLMResponseError`.
- **Read back.** The book stops at the import. The script then queries the
  contract's parties, roles and locations to show what landed.

### What a run looks like

The LLM extracts a *Licensing Agreement* between *Mortgage Logic.com,
Inc.* (Client, Irvine CA) and *TrueLink, Inc.* (Provider, San Luis Obispo
CA). It's effective 1999-02-26 with a one-year, auto-renewing term, and
governed by California law. `end_date` and `total_amount` come back null:
the agreement has neither a fixed end date nor a single total value. The
final query returns both parties with their role and location.

## Chapter 7 — Microsoft-style GraphRAG

Script: [graphrag_odyssey.py](graphrag_odyssey.py)
(re-implements `template/kg-rag/notebooks/ch07.ipynb`).

Run from the repo root (requires `OPENAI_API_KEY` in `.env`, see
`.env.example`):

```
.venv/Scripts/python.exe examples/graphrag_odyssey.py --process-name ch07_graphrag_odyssey
```

Ch06 extracted into a schema written down in advance. Ch07 drops the
schema: the LLM is only given a list of entity *types* and decides which
entities and relationships a text contains. The graph is then summarized
at several levels, which lets it answer two kinds of question that chunk
retrieval handles badly: broad ones about the whole corpus, and ones about
a specific entity spread across many passages.

The corpus is Samuel Butler's translation of Homer's *Odyssey* from
Project Gutenberg
([`OdysseyCorpus`](../src/graphrag/datasets/odyssey.py), cached at
`data/ch07-odyssey.txt`), split into its 24 books. As in the book, only
the first is processed by default; `--books N` processes more, at roughly
100–150 LLM calls per book. The script uses its own `_ch07` Kuzu database,
reset on every run.

### Indexing: building the graph

1. **Extraction**
   ([`EntityRelationshipExtractor`](../src/graphrag/extraction/entity_extractor.py)).
   Each 1000-character chunk goes to the LLM with a few-shot prompt
   (`graph_extraction.txt`) and the types PERSON, ORGANIZATION, LOCATION,
   GOD, EVENT, CREATURE and WEAPON_OR_TOOL. The response is delimited
   tuple records, `("entity";NAME;TYPE;description)` and
   `("relationship";SOURCE;TARGET;description;strength)`, separated by
   `|`, which `parse()` turns into `ExtractedEntity` /
   `ExtractedRelationship`.
2. **Import**
   ([`EntityGraphImporter`](../src/graphrag/extraction/entity_graph_importer.py)).
   Entities are merged by name, so one that appears in many chunks is one
   node holding a *list* of descriptions, one per mention. Relationships
   are created, not merged: every mention is its own edge.
3. **Summarization**
   ([`DescriptionSummarizer`](../src/graphrag/extraction/description_summarizer.py)).
   An entity with several descriptions gets one LLM-written `summary`
   (`description_summary.txt`); one with a single description keeps it.
   Likewise, all the `RELATIONSHIP` edges between a pair collapse into one
   `SUMMARIZED_RELATIONSHIP`, whose `weight` is the number of edges it
   replaces.
4. **Communities**
   ([`KuzuLouvainCommunityDetector`](../src/graphrag/extraction/community_detector.py)).
   Louvain clusters the entity graph into densely connected groups and
   writes each entity's `community_id`.
5. **Community reports**
   ([`CommunityReporter`](../src/graphrag/extraction/community_reporter.py)).
   For each community of two or more entities, the LLM reads its entities
   and relationships and writes a JSON report (`community_report.txt`):
   title, summary, a 0–10 importance `rating` and an explanation. The
   report becomes a `Community` node that each member links to.
6. **Entity embeddings**
   ([`EntityEmbeddingIndexer`](../src/graphrag/retrieval/entity_index.py)).
   Each entity summary is embedded, and an HNSW vector index is built over
   them (see chapter 2 for how that index works).

The resulting graph:

```
(:Book)-[:HAS_CHUNK]->(:Chunk {text})-[:MENTIONS]->(:Entity {name, type, description[], summary, community_id, embedding})
(:Entity)-[:RELATIONSHIP {description, strength}]->(:Entity)
(:Entity)-[:SUMMARIZED_RELATIONSHIP {summary, weight}]->(:Entity)
(:Entity)-[:IN_COMMUNITY]->(:Community {title, summary, rating, rating_explanation})
```

### Querying: global and local search

**Global search** ([`GlobalSearch`](../src/graphrag/retrieval/global_search.py))
is for questions about the whole corpus, like "What is this story
about?". No single chunk or entity answers that, but the community reports
together cover the entire graph. It is a map-reduce:

- **map**: each report rated at least 5 is sent with the question
  (`global_map_system.txt`). The LLM returns JSON key points, each with a
  0–100 importance score.
- **reduce**: points scored 0 are dropped, the rest are sorted by score,
  and one final call (`global_reduce_system.txt`) merges them into the
  answer.

**Local search** ([`LocalSearch`](../src/graphrag/retrieval/local_search.py))
is for questions about particular entities, like "Who is Ulysses?". The
question is embedded and the 5 nearest entity summaries are found. Around
them it gathers the 3 chunks that mention the most of those entities, the
3 highest-rated reports of their communities, and the 3 heaviest
summarized relationships among them. That context
(`LocalContext.to_prompt()`) goes to the LLM with `local_search_system.txt`.

### How Louvain runs in Kuzu

The book projects an in-memory graph and runs Louvain with Neo4j's GDS
library (`gds.graph.project`, `gds.louvain.write`). Kuzu has the same two
steps in its `algo` extension, statically linked into 0.11.3, so no
`INSTALL`/`LOAD` is needed:

```
CALL project_graph('entity_graph', ['Entity'], ['RELATIONSHIP'])
CALL louvain('entity_graph') WITH node, louvain_id SET node.community_id = ...
```

Louvain treats the edges as undirected, as the book's
`undirectedRelationshipTypes: ['*']` does. Kuzu gives entities with no
relationships `louvain_id = -1`. They're stored with no community rather
than one of their own, which has the same effect as the book's filter on
communities of at least two entities. The projected graph is dropped
afterwards, so detection can run again.

### Where this departs from the book

- **Relationship summaries are actually stored.** The book calls
  `import_rels_summary(neo4j_driver, summaries)` with the *entity*
  summaries. None of the relationship summaries it generated is written,
  so its cell 14 prints a single raw description.
- **Local search finds chunks.** The book's `local_search_query` matches
  `HAS_ENTITY` edges, but its import creates `MENTIONS`, so the chunk part
  of its context is always empty.
- **Ordering by properties that exist.** The book orders reports and
  relationships by `rank` and `weight`, which nothing sets. Neo4j quietly
  ignores the missing properties; Kuzu rejects them. Reports are ordered
  by `rating` here, and relationships by `weight`, the number of mentions
  they summarize.
- **Ranked map points.** The reduce prompt says its reports are "ranked in
  the descending order of importance", but the book passes the raw map
  responses unsorted. Here score-0 ("I don't know") points are dropped
  and the rest sorted, as in Microsoft's GraphRAG.
- **Type as a property.** The book adds each entity's type as an extra
  Neo4j label (`SET n:$(row.entity_type)`). A Kuzu node belongs to exactly
  one table, so it's a `type` property, set from the first mention.
- **Chunk ids include the book.** The book merges chunks on their index
  within a book, so ids repeat across books. A Kuzu primary key must be
  unique, so the id is `"<book>-<chunk>"`.
- **No APOC or COLLECT subqueries.** A community's relationships come
  from a plain match on edges with both ends in it, instead of
  `apoc.path.subgraphAll`. Local search runs one query per context part
  instead of one query with `COLLECT {}` subqueries, which Kuzu lacks.
- **Direction for summarized relationships.** Kuzu's `MERGE` needs a
  direction, so `SUMMARIZED_RELATIONSHIP` always points from the
  alphabetically smaller name.
- **Sturdier parsing.** The parser splits on the delimiters the prompt
  specified rather than guessing them. A description that contains `;`
  is kept whole instead of being dropped, and names are upper-cased so
  "Ulysses" and "ULYSSES" merge. Malformed records are still skipped, as
  in the book.
- **JSON mode and errors.** Community reports and map responses use
  `json_mode=True` rather than stripping code fences. Unusable JSON raises
  `LLMResponseError`.
- **Unextracted entities aren't embedded.** A relationship can name an
  entity that wasn't extracted itself. That entity has no summary, and the
  book would call `embed(None)` on it; here it's skipped.
- **Resetting the schema.** Kuzu won't drop a table that a vector index
  still references, so `create_schema(reset=True)` drops the indexes on
  its tables first.

### What a run looks like

One book gives 22 chunks. A test run extracted 67 entities and 103
relationships (the book's run had 62 and 91). 24 entities and 14
relationship pairs needed an LLM summary. Louvain found 8 communities of
2 to 21 entities (the book's run found 9). Their reports were rated
between 3 ("Euryclea and Laertes Household") and 8.5 ("Ithaca and the
Odyssey of Telemachus", "Aegisthus, Agamemnon, and Orestes"), so 5 of the
8 passed the global-search threshold. Extraction isn't deterministic: an
earlier run of the same code found 73 entities and 11 communities.

Global search turned those 5 reports into 15 key points. Its answer
covers Ulysses' journey home, Telemachus and the suitors in Ithaca,
Neptune's anger, the Aegisthus/Agamemnon/Orestes subplot and Mentes and
the Taphians. Book 1 alone covers all of that, so the summary is accurate
to the part that was indexed. Local search's context started with the
opening chunk ("Tell me, O Muse, of that ingenious hero..."). Its answer
describes Ulysses as the hero of the Trojan War held by Calypso, kept
from home by Neptune, with suitors courting Penelope. It cites
`[Data: Entities (0); Relationships (0)]`-style references, but the
context has no record ids, so those numbers are the list positions the
LLM saw. The book's run has the same issue.
