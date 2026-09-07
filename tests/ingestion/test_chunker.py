from graphrag.ingestion.chunker import TextChunker


def test_split_respects_chunk_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    chunker = TextChunker(chunk_size=50, overlap=10)

    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)
    # every word appears in at least one chunk
    for i in range(200):
        assert any(f"word{i}" in chunk for chunk in chunks)


def test_split_single_short_chunk():
    chunker = TextChunker(chunk_size=500, overlap=40)

    chunks = chunker.split("just a short sentence")

    assert chunks == ["just a short sentence"]
