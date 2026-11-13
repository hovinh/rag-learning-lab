from graphrag.graph.models import GraphSchema, Property, RelationshipPattern


def test_to_prompt_matches_book_layout():
    schema = GraphSchema(
        node_props={
            "Movie": [Property("title", "STRING"), Property("released", "INT64")],
            "Person": [Property("name", "STRING")],
        },
        rel_props={"ACTED_IN": [Property("roles", "STRING[]")]},
        relationships=[RelationshipPattern("Person", "ACTED_IN", "Movie")],
    )

    assert schema.to_prompt() == (
        "Node properties:\n"
        "Movie {title: STRING, released: INT64}\n"
        "Person {name: STRING}\n"
        "Relationship properties:\n"
        "ACTED_IN {roles: STRING[]}\n"
        "The relationships:\n"
        "(:Person)-[:ACTED_IN]->(:Movie)"
    )
