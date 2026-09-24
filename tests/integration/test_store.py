import pytest

from app.corpus.models import Chunk
from app.corpus.store import ChunkStore


def make_chunk(chunk_id: str, body: str = "text", source: str = "REG1", **kwargs) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=f"header\n\n{body}",
        body=body,
        source=source,
        **kwargs,
    )


def test_schema_creation_is_repeatable(store: ChunkStore) -> None:
    store.create_schema()

    assert store.count() == 0


def test_chunks_are_stored(store: ChunkStore) -> None:
    stored = store.upsert([make_chunk("a"), make_chunk("b")], [[1, 0, 0, 0], [0, 1, 0, 0]])

    assert stored == 2
    assert store.count() == 2


def test_reingesting_the_same_source_does_not_duplicate(store: ChunkStore) -> None:
    store.upsert([make_chunk("a")], [[1, 0, 0, 0]])
    store.upsert([make_chunk("a")], [[1, 0, 0, 0]])

    assert store.count() == 1


def test_reingesting_updates_the_existing_row(store: ChunkStore) -> None:
    store.upsert([make_chunk("a", body="old")], [[1, 0, 0, 0]])
    store.upsert([make_chunk("a", body="new")], [[1, 0, 0, 0]])

    assert store.search([1, 0, 0, 0])[0].chunk.body == "new"


def test_mismatched_inputs_are_rejected(store: ChunkStore) -> None:
    with pytest.raises(ValueError):
        store.upsert([make_chunk("a")], [[1, 0, 0, 0], [0, 1, 0, 0]])


def test_empty_input_is_a_no_op(store: ChunkStore) -> None:
    assert store.upsert([], []) == 0


def test_search_ranks_the_nearest_vector_first(store: ChunkStore) -> None:
    store.upsert(
        [make_chunk("near"), make_chunk("far")],
        [[1, 0, 0, 0], [0, 0, 0, 1]],
    )

    results = store.search([1, 0, 0, 0], limit=2)

    assert [r.chunk.chunk_id for r in results] == ["near", "far"]
    assert results[0].score > results[1].score


def test_an_identical_vector_scores_one(store: ChunkStore) -> None:
    store.upsert([make_chunk("a")], [[1, 0, 0, 0]])

    assert store.search([1, 0, 0, 0])[0].score == pytest.approx(1.0, abs=1e-6)


def test_search_honours_the_limit(store: ChunkStore) -> None:
    chunks = [make_chunk(str(i)) for i in range(5)]
    store.upsert(chunks, [[1, 0, 0, 0]] * 5)

    assert len(store.search([1, 0, 0, 0], limit=2)) == 2


def test_search_can_be_scoped_to_one_source(store: ChunkStore) -> None:
    store.upsert(
        [make_chunk("a", source="REG1"), make_chunk("b", source="REG2")],
        [[1, 0, 0, 0], [1, 0, 0, 0]],
    )

    results = store.search([1, 0, 0, 0], source="REG2")

    assert [r.chunk.chunk_id for r in results] == ["b"]


def test_structural_metadata_survives_a_round_trip(store: ChunkStore) -> None:
    chunk = make_chunk(
        "a",
        chapter="III",
        section="1",
        article="6",
        article_title="Classification rules",
        paragraph="2",
    )
    store.upsert([chunk], [[1, 0, 0, 0]])

    retrieved = store.search([1, 0, 0, 0])[0].chunk

    assert retrieved.chapter == "III"
    assert retrieved.section == "1"
    assert retrieved.citation == "Article 6(2)"
