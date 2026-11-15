from graphrag.extraction.community_detector import KuzuLouvainCommunityDetector
from graphrag.extraction.entity_models import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelationship,
)


def triangle(*names):
    a, b, c = names
    return [
        ExtractedRelationship(a, b, "r", 1.0),
        ExtractedRelationship(b, c, "r", 1.0),
        ExtractedRelationship(c, a, "r", 1.0),
    ]


def import_two_triangles_and_a_loner(importer):
    importer.import_chunk(
        0,
        0,
        "a",
        ChunkExtraction(
            [ExtractedEntity("LONER", "PERSON", "Related to no one")],
            [
                *triangle("A", "B", "C"),
                *triangle("X", "Y", "Z"),
                ExtractedRelationship("C", "X", "r", 1.0),
            ],
        ),
    )


def communities(store):
    rows = store.query(
        "MATCH (e:Entity) RETURN e.community_id AS community, collect(e.name) AS names"
    )
    return {row["community"]: sorted(row["names"]) for row in rows}


def test_detect_clusters_densely_connected_entities(entity_store, entity_importer):
    import_two_triangles_and_a_loner(entity_importer)

    sizes = KuzuLouvainCommunityDetector(entity_store).detect()

    clusters = communities(entity_store)
    assert sorted(names for community, names in clusters.items() if community is not None) == [
        ["A", "B", "C"],
        ["X", "Y", "Z"],
    ]
    assert sorted(sizes.values()) == [3, 3]


def test_detect_leaves_unconnected_entities_without_a_community(entity_store, entity_importer):
    import_two_triangles_and_a_loner(entity_importer)

    KuzuLouvainCommunityDetector(entity_store).detect()

    assert communities(entity_store)[None] == ["LONER"]


def test_detect_can_run_again(entity_store, entity_importer):
    import_two_triangles_and_a_loner(entity_importer)
    detector = KuzuLouvainCommunityDetector(entity_store)

    detector.detect()

    assert sorted(detector.detect().values()) == [3, 3]
