"""Domain packs: everything specific to one dataset -- its data, schema,
prompts, models, tools and agent scope -- kept out of the core packages,
which stay domain-agnostic. Each pack is also a template for adding a new
domain: ``movies`` (Cypher over a structured graph, ch04/05/08),
``contracts`` (schema-first extraction, ch06) and ``odyssey`` (a corpus for
schema-free GraphRAG, ch07).
"""
